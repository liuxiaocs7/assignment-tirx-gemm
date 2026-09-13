def hgemm_v4(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 128:
        raise ValueError("Step 4 requires positive M,N divisible by 128 and K divisible by 128")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 128
    K_TILES = K // BLK_K
    # Each CTA accumulates 128 columns; leave the rest of TMEM for peer CTAs.
    TMEM_COLS = BLK_N
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        # fmt: off
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=TMEM_COLS, cta_group=1)
            # This is the CTA's only allocation. Let peer CTAs allocate while
            # we compute; keep our TMEM live until writeback completes.
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, TMEM_COLS), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, TMEM_COLS) : (1@TLane, 1@TCol)]))
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32
        phase_mma: T.int32
        phase_tma = 0
        phase_mma = 0

        @T.inline
        def tma_load(k_st):
            Tx.copy_async(Asmem[:, :], A[m_st:m_st + BLK_M, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=1, mbar=tma_bar.ptr_to([0]))
            Tx.copy_async(Bsmem[:, :], B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=1, mbar=tma_bar.ptr_to([0]))
            T.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # One elected thread dispatches both async engines and tracks their phases.
        if warp_id == 0:
            if T.filter(lane_id, T.ptx.elect_sync()):
                for k in range(K_TILES):
                    tma_load(k * BLK_K)
                    T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
                    T.ptx.tcgen05.fence.after_thread_sync()
                    mma(k != 0)
                    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                    phase_tma = phase_tma ^ 1
                    phase_mma = phase_mma ^ 1
                # Only the final MMA completion is handed off to writeback threads.
                T.ptx.tcgen05.fence.after_thread_sync()
                T.ptx.tcgen05.fence.before_thread_sync()

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
        if warp_id == 0:
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=TMEM_COLS, cta_group=1)

    return kernel
