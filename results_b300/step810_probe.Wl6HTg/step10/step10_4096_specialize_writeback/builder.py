def hgemm_v10(M, N, K):
    if min(M, N, K) <= 0 or M % 512 or N % 256 or K % 64:
        raise ValueError("Step 10 requires positive M divisible by 512, N divisible by 256 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    CTA_GROUP = 2
    NUM_CONSUMER = 2
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 32  # Eight loads per consumer, with separate FP16 row buffers.
    WG_NUMBER = 3
    CLUSTER_COUNT = min(SM_COUNT // CTA_GROUP, (M // (MMA_M * NUM_CONSUMER)) * (N // MMA_N))
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        # fmt: off
        T.device_entry()
        bx = T.cta_id([CLUSTER_COUNT * CTA_GROUP])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld = TCGen05Bar(pool, NUM_CONSUMER)
        ld2mma = MBarrier(pool, NUM_CONSUMER)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)
        pool.commit()

        # These helpers elect CTA thread zero internally; keep them at CTA scope.
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        tmem = T.decl_buffer((128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // (MMA_M * NUM_CONSUMER), num_n_tiles=N // MMA_N,
            l2_group_size=8, num_clusters=CLUSTER_COUNT)
        tile_scheduler.init(bx // CTA_GROUP)
        m_st = T.meta_var(tile_scheduler.m_idx * MMA_M * NUM_CONSUMER + cbx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * MMA_N + cbx * BLK_N)
        n_out = T.meta_var(tile_scheduler.n_idx * MMA_N)
        tma2mma_cta0 = tma2mma.remote_view(0)
        tma_phase = PipelineState(PIPE_DEPTH)
        mma_phase = PipelineState(PIPE_DEPTH)
        ld_phase = PipelineState(1)
        wb_phase = PipelineState(1)
        tma_phase.init(1)
        mma_phase.init(0)
        ld_phase.init(1)
        wb_phase.init(0)

        @T.inline
        def tma_load(k_st):
            for consumer in T.unroll(NUM_CONSUMER):
                m_consumer = T.meta_var(m_st + consumer * MMA_M)
                Tx.copy_async(Asmem[tma_phase.stage, consumer, :, :],
                              A[m_consumer:m_consumer + BLK_M, k_st:k_st + BLK_K],
                              dispatch="tma_auto", cta_group=CTA_GROUP,
                              mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
            Tx.copy_async(Bsmem[tma_phase.stage, :, :],
                          B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=CTA_GROUP,
                          mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))

        if wg_id == 2:
            if warp_id == 3:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_phase.stage, tma_phase.phase)
                            tma_load(k * BLK_K)
                            # Both CTAs complete bytes against CTA 0's barrier.
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_phase.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_phase.advance()
                        tile_scheduler.next_tile()
            elif warp_id < NUM_CONSUMER:
                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_phase.phase)
                            ld_phase.advance()
                            for k in range(K_TILES):
                                tma2mma.wait(mma_phase.stage, mma_phase.phase)
                                T.ptx.tcgen05.fence.after_thread_sync()
                                Tx.gemm_async(tmem[:, warp_id * MMA_N:(warp_id + 1) * MMA_N],
                                    Asmem[mma_phase.stage, warp_id, :, :], Bsmem[mma_phase.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_phase.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_phase.advance()
                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()
        else:
            for wb_consumer in T.unroll(NUM_CONSUMER):
                if wg_id == wb_consumer:
                    m_out = T.meta_var(m_st + wb_consumer * MMA_M)
                    Dreg = T.alloc_local((TMEM_LD_N,), acc_type)
                    Dreg_f16 = T.alloc_local((MMA_N,), d_type)
                    Dreg_wg = Dreg.view(128, TMEM_LD_N,
                        layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
                    while tile_scheduler.valid():
                        mma2ld.wait(wb_consumer, wb_phase.phase)
                        wb_phase.advance()
                        T.ptx.tcgen05.fence.after_thread_sync()
                        for i in T.unroll(MMA_N // TMEM_LD_N):
                            col = T.meta_var(i * TMEM_LD_N)
                            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, wb_consumer * MMA_N + col:wb_consumer * MMA_N + col + TMEM_LD_N])
                            T.ptx.tcgen05.wait.ld()
                            Tx.cast(Dreg_f16[col:col + TMEM_LD_N], Dreg[:])
                        # All TMEM reads have finished. MMA can overlap the TMA epilogue.
                        T.ptx.tcgen05.fence.before_thread_sync()
                        ld2mma.arrive(wb_consumer, remote=0)
                        for i in T.unroll(MMA_N // EPI_N):
                            col = T.meta_var(i * EPI_N)
                            Tx.copy(Dsmem[wb_consumer, warp_id * 32 + lane_id, :], Dreg_f16[col:col + EPI_N])
                            T.ptx.fence.proxy_async("shared::cta")
                            T.cuda.warpgroup_sync(wb_consumer + 10)
                            if warp_id == 0:
                                if T.filter(lane_id, T.ptx.elect_sync()):
                                    Tx.copy_async(D[m_out:m_out + BLK_M, n_out + col:n_out + col + EPI_N],
                                                  Dsmem[wb_consumer, :, :], dispatch="tma_auto")
                                    T.ptx.cp_async.bulk.commit_group()
                                    T.ptx.cp_async.bulk.wait_group(0)
                            T.cuda.warpgroup_sync(wb_consumer + 10)
                        tile_scheduler.next_tile()

        T.cuda.cluster_sync()
        # Exactly the allocating warp releases the TMEM allocation.
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
