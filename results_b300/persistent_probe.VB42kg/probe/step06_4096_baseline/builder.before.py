def hgemm_v6(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("Step 6 requires positive M,N divisible by 128 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    PRE_NUM = min(PIPE_DEPTH, K_TILES)
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        # fmt: off
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                for s in T.unroll(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)
        # Phases live across output tiles: an odd number of arrivals must
        # not be forgotten when the next tile starts at pipeline slot zero.
        phase_tma = T.alloc_local((PIPE_DEPTH,), "int32")
        phase_mma: T.int32
        for s in T.unroll(PIPE_DEPTH):
            phase_tma[s] = 0
        phase_mma = 0

        @T.inline
        def tma_load(stage, k_st):
            Tx.copy_async(Asmem[stage, :, :], A[m_st:m_st + BLK_M, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=1, mbar=tma_bar.ptr_to([stage]))
            Tx.copy_async(Bsmem[stage, :, :], B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=1, mbar=tma_bar.ptr_to([stage]))
            T.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        while tile_scheduler.valid():
            # Prefetch both buffers. Refill a stage only after MMA releases it.
            if warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    for s in T.unroll(PRE_NUM):
                        tma_load(s, s * BLK_K)
                    for k in range(K_TILES):
                        stage = T.meta_var(k % PIPE_DEPTH)
                        T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma[stage])
                        T.ptx.tcgen05.fence.after_thread_sync()
                        mma(stage, k != 0)
                        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        T.ptx.tcgen05.fence.before_thread_sync()
                        phase_tma[stage] = phase_tma[stage] ^ 1
                        phase_mma = phase_mma ^ 1
                        if k + PIPE_DEPTH < K_TILES:
                            tma_load(stage, (k + PIPE_DEPTH) * BLK_K)

            # Publish the elected thread's completion to all writeback threads.
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            # Publish completed TMEM reads before CTA reuse or deallocation.
            T.ptx.tcgen05.fence.before_thread_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, :], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            if warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    Tx.copy_async(D[m_st:m_st + BLK_M, n_st:n_st + BLK_N],
                                  Dsmem[:, :], dispatch="tma_auto")
                    T.ptx.cp_async.bulk.commit_group()
                    T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.cta_sync()
            tile_scheduler.next_tile()

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
