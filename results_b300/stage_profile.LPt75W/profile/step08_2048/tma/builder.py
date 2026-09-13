def hgemm_v8(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("Step 8 requires positive M,N divisible by 128 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 32  # Amortize TMEM load/wait overhead without a full-row FP32 buffer.
    WG_NUMBER = 2
    CTA_COUNT = min(SM_COUNT, (M // BLK_M) * (N // BLK_N))
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    # B300 step810_probe.Wl6HTg: shorten only the TMA data-ready suspension
    # hint. Keep the measured acquire/retry loop and all completion waits.
    TMA_WAIT_SOURCE = r"""
__forceinline__ __device__ void tirx_tma_wait_64ns(void* barrier, int phase) {
    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);
    unsigned int ticks = 64;
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory");
}
"""

    PROFILE_SOURCE = '\n#ifndef TIRX_PROFILE_TIMERS_DEFINED\n#define TIRX_PROFILE_TIMERS_DEFINED\n__forceinline__ __device__ uint64_t tirx_profile_ns() {\n    uint64_t value;\n    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");\n    return value;\n}\n__forceinline__ __device__ uint64_t tirx_profile_cycles() {\n    uint64_t value;\n    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");\n    return value;\n}\n__forceinline__ __device__ void tirx_profile_start(uint64_t* s, bool leader) {\n    if (leader) {\n        s[0] = tirx_profile_ns();\n        s[1] = tirx_profile_cycles();\n        unsigned int sm;\n        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));\n        s[2] = sm;\n        s[3] = 0; s[4] = 0; s[5] = 0; s[6] = 0;\n    }\n}\n__forceinline__ __device__ void tirx_profile_mark(uint64_t* s, bool leader) {\n    if (leader) s[7] = tirx_profile_ns();\n}\n__forceinline__ __device__ void tirx_profile_add_wait(uint64_t* s, bool leader) {\n    if (leader) s[3] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_add_work(uint64_t* s, bool leader) {\n    if (leader) s[4] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_add_handoff(uint64_t* s, bool leader) {\n    if (leader) s[5] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_add_epilogue(uint64_t* s, bool leader) {\n    if (leader) s[6] += tirx_profile_ns() - s[7];\n}\n__forceinline__ __device__ void tirx_profile_finish(\n        uint64_t* s, int64_t* out, int m, int n, bool leader) {\n    if (leader) {\n        uint64_t end = tirx_profile_ns();\n        uint64_t cycles = tirx_profile_cycles();\n        out[0] = s[0]; out[1] = end;\n        out[2] = s[3]; out[3] = s[4]; out[4] = s[5]; out[5] = s[6];\n        out[6] = s[1]; out[7] = cycles; out[8] = s[2];\n        out[9] = m; out[10] = n;\n    }\n}\n#endif\n'

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
        Profile: T.Buffer((148, 2, 1, 11), "int64"),
    ):
        # fmt: off
        T.device_entry()
        bx = T.cta_id([CTA_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld = TCGen05Bar(pool, 1)
        ld2mma = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, EPI_N), d_type, layout=D_layout)
        pool.commit()

        # These helpers elect CTA thread zero internally; keep them at CTA scope.
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=CTA_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)
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
            Tx.copy_async(Asmem[tma_phase.stage, :, :],
                          A[m_st:m_st + BLK_M, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=1,
                          mbar=tma2mma.ptr_to([tma_phase.stage]))
            Tx.copy_async(Bsmem[tma_phase.stage, :, :],
                          B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=1,
                          mbar=tma2mma.ptr_to([tma_phase.stage]))

        if wg_id == 1:
            if warp_id == 3:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    profile_state = T.alloc_local((8,), "uint64")
                    while tile_scheduler.valid():
                        T.cuda.func_call("tirx_profile_start", profile_state.data, True, source_code=PROFILE_SOURCE)
                        for k in range(K_TILES):
                            T.cuda.func_call("tirx_profile_mark", profile_state.data, True, source_code=PROFILE_SOURCE)
                            mma2tma.wait(tma_phase.stage, tma_phase.phase)
                            T.cuda.func_call("tirx_profile_add_wait", profile_state.data, True, source_code=PROFILE_SOURCE)
                            T.cuda.func_call("tirx_profile_mark", profile_state.data, True, source_code=PROFILE_SOURCE)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_phase.stage,
                                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_phase.advance()
                            T.cuda.func_call("tirx_profile_add_work", profile_state.data, True, source_code=PROFILE_SOURCE)
                        T.cuda.func_call("tirx_profile_finish", profile_state.data, Profile.ptr_to([bx, tile_scheduler.tile_count, 0, 0]), tile_scheduler.m_idx, tile_scheduler.n_idx, True, source_code=PROFILE_SOURCE)
                        tile_scheduler.next_tile()
            elif warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        ld2mma.wait(0, ld_phase.phase)
                        ld_phase.advance()
                        for k in range(K_TILES):
                            T.cuda.func_call("tirx_tma_wait_64ns", tma2mma.ptr_to([mma_phase.stage]),
                                             mma_phase.phase ^ 0, source_code=TMA_WAIT_SOURCE)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            Tx.gemm_async(tmem[:, :MMA_N],
                                Asmem[mma_phase.stage, :, :], Bsmem[mma_phase.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_phase.stage)
                            mma_phase.advance()
                        mma2ld.arrive(0)
                        tile_scheduler.next_tile()
        elif wg_id == 0:
            Dreg = T.alloc_local((TMEM_LD_N,), acc_type)
            Dreg_f16 = T.alloc_local((MMA_N,), d_type)
            Dreg_wg = Dreg.view(128, TMEM_LD_N,
                layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
            while tile_scheduler.valid():
                mma2ld.wait(0, wb_phase.phase)
                wb_phase.advance()
                T.ptx.tcgen05.fence.after_thread_sync()
                for i in T.unroll(MMA_N // TMEM_LD_N):
                    col = T.meta_var(i * TMEM_LD_N)
                    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, col:col + TMEM_LD_N])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(Dreg_f16[col:col + TMEM_LD_N], Dreg[:])
                # All TMEM reads have finished. MMA can overlap the TMA epilogue.
                T.ptx.tcgen05.fence.before_thread_sync()
                ld2mma.arrive(0)
                for i in T.unroll(MMA_N // EPI_N):
                    col = T.meta_var(i * EPI_N)
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], Dreg_f16[col:col + EPI_N])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if T.filter(lane_id, T.ptx.elect_sync()):
                            Tx.copy_async(D[m_st:m_st + BLK_M, n_st + col:n_st + col + EPI_N],
                                          Dsmem[:, :], dispatch="tma_auto")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)
                tile_scheduler.next_tile()

        T.cuda.cta_sync()
        # Exactly the allocating warp releases the TMEM allocation.
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
