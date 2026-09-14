def hgemm_v10(M, N, K):
    if min(M, N, K) <= 0 or M % 512 or N % 256 or K % 64:
        raise ValueError("Step 10 requires positive M divisible by 512, N divisible by 256 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    # B300 measurements favor narrow tiles up to 256 cluster output tiles.
    # Larger outputs retain N256 to avoid the measured 8192 regression.
    NARROW_N = M * N <= 4096 * 4096
    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64
    CTA_GROUP = 2
    NUM_CONSUMER = 2
    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 32 if NARROW_N else 64
    TMEM_LD_N = 32  # Four/eight loads per narrow/wide consumer.
    WG_NUMBER = 3
    # Preserve the maximum tiles per cluster while reducing the partial tail.
    # For 4096 on 148 SMs, 256 narrow tiles use 64 clusters with four each.
    TOTAL_TILES = (M // (MMA_M * NUM_CONSUMER)) * (N // MMA_N)
    MAX_CLUSTERS = SM_COUNT // CTA_GROUP
    # A single wave has no next tile to overlap. For persistent narrow tiles,
    # use both 128-column slots per consumer within the same 512-column TMEM.
    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1
    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0
    TILES_PER_CLUSTER = (TOTAL_TILES + MAX_CLUSTERS - 1) // MAX_CLUSTERS
    CLUSTER_COUNT = (TOTAL_TILES + TILES_PER_CLUSTER - 1) // TILES_PER_CLUSTER
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_SWIZZLE = SwizzleMode.SWIZZLE_64B_ATOM if NARROW_N else SwizzleMode.SWIZZLE_128B_ATOM
    D_layout = mma_shared_layout(d_type, D_SWIZZLE, (NUM_CONSUMER, BLK_M, EPI_N))

    PROFILE_SOURCE = '\n#ifndef TIRX_PROFILE_TIMERS_DEFINED\n#define TIRX_PROFILE_TIMERS_DEFINED\n__forceinline__ __device__ uint64_t tirx_profile_ns() {\n    uint64_t value;\n    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");\n    return value;\n}\n__forceinline__ __device__ uint64_t tirx_profile_cycles() {\n    uint64_t value;\n    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");\n    return value;\n}\n__forceinline__ __device__ void tirx_profile_start(uint64_t* s, bool leader) {\n    if (leader) {\n        s[0] = tirx_profile_ns();\n        s[1] = tirx_profile_cycles();\n        unsigned int sm;\n        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));\n        s[2] = sm;\n        s[3] = 0; s[4] = 0; s[5] = 0; s[6] = 0;\n    }\n}\n__forceinline__ __device__ void tirx_profile_mark(uint64_t* s, bool leader) {\n    if (leader) s[7] = tirx_profile_ns();\n}\n__forceinline__ __device__ void tirx_profile_add_wait(uint64_t* s, bool leader) {\n    if (leader) s[3] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_add_work(uint64_t* s, bool leader) {\n    if (leader) s[4] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_add_handoff(uint64_t* s, bool leader) {\n    if (leader) s[5] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_add_epilogue(uint64_t* s, bool leader) {\n    if (leader) s[6] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_finish(\n        uint64_t* s, int64_t* out, int m, int n, bool leader) {\n    if (leader) {\n        uint64_t end = tirx_profile_ns();\n        uint64_t cycles = tirx_profile_cycles();\n        out[0] = s[0]; out[1] = end;\n        out[2] = s[3]; out[3] = s[4]; out[4] = s[5]; out[5] = s[6];\n        out[6] = s[1]; out[7] = cycles; out[8] = s[2];\n        out[9] = m; out[10] = n;\n    }\n}\n#endif\n'

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
        Profile: T.Buffer((128, 4, 2, 11), "int64"),
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
        mma2ld = TCGen05Bar(pool, TMEM_BUFFERS * NUM_CONSUMER)
        ld2mma = MBarrier(pool, TMEM_BUFFERS * NUM_CONSUMER)
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
        # The allocation result is immutable after the cluster sync.
        mma_tmem_base: T.let = tmem_addr[0]
        tmem = T.decl_buffer((128, 512), acc_type, scope="tmem", allocated_addr=mma_tmem_base,
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
        ld_phase = PipelineState(TMEM_BUFFERS)
        wb_phase = PipelineState(TMEM_BUFFERS)
        tma_phase.init(1)
        mma_phase.init(0)
        ld_phase.init(1)
        wb_phase.init(0)

        @T.inline
        def tma_load(k_st):
            Tx.copy_async(Bsmem[tma_phase.stage, :, :],
                          B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=CTA_GROUP,
                          mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
            for consumer in T.unroll(NUM_CONSUMER):
                m_consumer = T.meta_var(m_st + consumer * MMA_M)
                Tx.copy_async(Asmem[tma_phase.stage, consumer, :, :],
                              A[m_consumer:m_consumer + BLK_M, k_st:k_st + BLK_K],
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
                        profile_state = T.alloc_local((8,), "uint64")
                        while tile_scheduler.valid():
                            T.cuda.func_call("tirx_profile_start", profile_state.data, True, source_code=PROFILE_SOURCE)
                            T.cuda.func_call("tirx_profile_mark", profile_state.data, True, source_code=PROFILE_SOURCE)
                            ld2mma.wait(ld_phase.stage * TMEM_SLOT_STRIDE + warp_id, ld_phase.phase)
                            T.cuda.func_call("tirx_profile_add_handoff", profile_state.data, True, source_code=PROFILE_SOURCE)
                            if TMEM_BUFFERS == 1:
                                ld_phase.advance()
                            for k in range(K_TILES):
                                T.cuda.func_call("tirx_profile_mark", profile_state.data, True, source_code=PROFILE_SOURCE)
                                tma2mma.wait(mma_phase.stage, mma_phase.phase)
                                T.cuda.func_call("tirx_profile_add_wait", profile_state.data, True, source_code=PROFILE_SOURCE)
                                T.cuda.func_call("tirx_profile_mark", profile_state.data, True, source_code=PROFILE_SOURCE)
                                T.ptx.tcgen05.fence.after_thread_sync()
                                Tx.gemm_async(tmem[:, (ld_phase.stage * TMEM_SLOT_STRIDE + warp_id) * MMA_N:(ld_phase.stage * TMEM_SLOT_STRIDE + warp_id + 1) * MMA_N],
                                    Asmem[mma_phase.stage, warp_id, :, :], Bsmem[mma_phase.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_phase.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_phase.advance()
                                T.cuda.func_call("tirx_profile_add_work", profile_state.data, True, source_code=PROFILE_SOURCE)
                            mma2ld.arrive(ld_phase.stage * TMEM_SLOT_STRIDE + warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            if TMEM_BUFFERS == 2:
                                ld_phase.advance()
                            T.cuda.func_call("tirx_profile_finish", profile_state.data, Profile.ptr_to([bx, tile_scheduler.tile_count, warp_id, 0]), tile_scheduler.m_idx, tile_scheduler.n_idx, True, source_code=PROFILE_SOURCE)
                            tile_scheduler.next_tile()
        elif wg_id < NUM_CONSUMER:
            m_out = T.meta_var(m_st + wg_id * MMA_M)
            Dreg = T.alloc_local((TMEM_LD_N,), acc_type)
            Dreg_f16 = T.alloc_local((MMA_N,), d_type)
            Dreg_wg = Dreg.view(128, TMEM_LD_N,
                layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
            while tile_scheduler.valid():
                mma2ld.wait(wb_phase.stage * TMEM_SLOT_STRIDE + wg_id, wb_phase.phase)
                if TMEM_BUFFERS == 1:
                    wb_phase.advance()
                T.ptx.tcgen05.fence.after_thread_sync()
                for i in T.unroll(MMA_N // TMEM_LD_N):
                    col = T.meta_var(i * TMEM_LD_N)
                    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, (wb_phase.stage * TMEM_SLOT_STRIDE + wg_id) * MMA_N + col:(wb_phase.stage * TMEM_SLOT_STRIDE + wg_id) * MMA_N + col + TMEM_LD_N])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(Dreg_f16[col:col + TMEM_LD_N], Dreg[:])
                # All TMEM reads have finished. MMA can overlap the TMA epilogue.
                T.ptx.tcgen05.fence.before_thread_sync()
                ld2mma.arrive(wb_phase.stage * TMEM_SLOT_STRIDE + wg_id, remote=0)
                if TMEM_BUFFERS == 2:
                    wb_phase.advance()
                for i in T.unroll(MMA_N // EPI_N):
                    col = T.meta_var(i * EPI_N)
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], Dreg_f16[col:col + EPI_N])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if T.filter(lane_id, T.ptx.elect_sync()):
                            Tx.copy_async(D[m_out:m_out + BLK_M, n_out + col:n_out + col + EPI_N],
                                          Dsmem[wg_id, :, :], dispatch="tma_auto")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)
                tile_scheduler.next_tile()

        T.cuda.cluster_sync()
        # Exactly the allocating warp releases the TMEM allocation.
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
