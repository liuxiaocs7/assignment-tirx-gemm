"""Blackwell GEMM kernels, adapted from Modern GPU Programming for MLSys.

Targets Apache TVM 0.26.0 and Blackwell SM100/SM103 GPUs.
Reference: https://mlc.ai/modern-gpu-programming-for-mlsys/
"""

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D
from tvm.backend.cuda.lang.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar

SM_COUNT = 148  # B200
F16_SIZE = 2

# ======================================================================
# Step 1: Single-tile synchronous GEMM
#   M=128, N=128, K=64 — exactly one tile, no loops.
#   All threads sync-load GMEM→SMEM, one MMA, sync writeback.
# ======================================================================

def hgemm_v1(M, N, K):
    if (M, N, K) != (128, 128, 64):
        raise ValueError("Step 1 requires M=N=128 and K=64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64

    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

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

        # --- Shared memory allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")    # Slot to store the TMEM base address returned by tcgen05.alloc
        mma_bar = pool.alloc((1,), "uint64", align=8)  # mbarrier for MMA completion signaling
        pool.move_base_to(1024)                   # Skip to offset 1024 so data buffers don't overlap with barriers
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()                             # Finalize all shared memory allocations

        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                # Init mbarrier with count=1 (one arrival expected). ptr_to([0]) gets pointer to the 0th element.
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            # Allocate 512 TMEM columns. address_of() passes the address where the HW writes the TMEM base.
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        # Flush shared memory writes, ensure mbarrier init is visible, then sync all threads
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        # Declare a logical view of the allocated TMEM (using the base returned by tcgen05.alloc)
        tmem = T.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
                              layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        m_st = T.meta_var(bx * BLK_M)           # Compile-time alias for tile row offset
        n_st = T.meta_var(by * BLK_N)           # Compile-time alias for tile col offset

        # TIR requires explicit type declaration for mutable variables
        phase_mma: T.int32
        phase_mma = 0

        # All CTA threads load the operands before the async MMA reads SMEM.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence.after_thread_sync()

        if warp_id == 0:
            if T.filter(lane_id, T.ptx.elect_sync()):
                Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                              accum=False, dispatch="tcgen05", cta_group=1)
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # Every writeback thread must observe MMA completion.
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
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
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st + BLK_N], Dreg_f16[:])

        # --- TMEM cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 2: K-loop — accumulate in TMEM
#   M=128, N=128, K=any multiple of 64.
#   Loop over K dimension with accumulation.
# ======================================================================

def hgemm_v2(M, N, K):
    if M != 128 or N != 128 or K <= 0 or K % 64:
        raise ValueError("Step 2 requires M=N=128 and a positive K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

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

        # --- Shared memory allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")    # Slot to store the TMEM base address returned by tcgen05.alloc
        mma_bar = pool.alloc((1,), "uint64", align=8)  # mbarrier for MMA completion signaling
        pool.move_base_to(1024)                   # Skip to offset 1024 so data buffers don't overlap with barriers
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()                             # Finalize all shared memory allocations

        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                # Init mbarrier with count=1 (one arrival expected). ptr_to([0]) gets pointer to the 0th element.
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            # Allocate 512 TMEM columns. address_of() passes the address where the HW writes the TMEM base.
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        # Flush shared memory writes, ensure mbarrier init is visible, then sync all threads
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        # Declare a logical view of the allocated TMEM (using the base returned by tcgen05.alloc)
        tmem = T.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
                              layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        m_st = T.meta_var(bx * BLK_M)           # Compile-time alias for tile row offset
        n_st = T.meta_var(by * BLK_N)           # Compile-time alias for tile col offset

        # TIR requires explicit type declaration for mutable variables
        phase_mma: T.int32
        phase_mma = 0

        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
            # All CTA threads load the operands before the async MMA reads SMEM.
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, k_st:k_st + BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, k_st:k_st + BLK_K])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()

            if warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(k != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Every writeback thread must observe MMA completion.
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            T.ptx.tcgen05.fence.after_thread_sync()
            phase_mma = phase_mma ^ 1

        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
            layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        # Publish completed TMEM reads before CTA reuse or deallocation.
        T.ptx.tcgen05.fence.before_thread_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st + BLK_N], Dreg_f16[:])

        # --- TMEM cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 3: Spatial tiling — multi-CTA
#   M, N any multiples of 128, K any multiple of 64.
#   Grid of (M/128)×(N/128) CTAs.
# ======================================================================

def hgemm_v3(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("Step 3 requires positive M,N divisible by 128 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

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

        # --- Shared memory allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")    # Slot to store the TMEM base address returned by tcgen05.alloc
        mma_bar = pool.alloc((1,), "uint64", align=8)  # mbarrier for MMA completion signaling
        pool.move_base_to(1024)                   # Skip to offset 1024 so data buffers don't overlap with barriers
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()                             # Finalize all shared memory allocations

        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                # Init mbarrier with count=1 (one arrival expected). ptr_to([0]) gets pointer to the 0th element.
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            # Allocate 512 TMEM columns. address_of() passes the address where the HW writes the TMEM base.
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        # Flush shared memory writes, ensure mbarrier init is visible, then sync all threads
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        # Declare a logical view of the allocated TMEM (using the base returned by tcgen05.alloc)
        tmem = T.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
                              layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        m_st = T.meta_var(bx * BLK_M)           # Compile-time alias for tile row offset
        n_st = T.meta_var(by * BLK_N)           # Compile-time alias for tile col offset

        # TIR requires explicit type declaration for mutable variables
        phase_mma: T.int32
        phase_mma = 0

        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
            # All CTA threads load the operands before the async MMA reads SMEM.
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, k_st:k_st + BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, k_st:k_st + BLK_K])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()

            if warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(k != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Every writeback thread must observe MMA completion.
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            T.ptx.tcgen05.fence.after_thread_sync()
            phase_mma = phase_mma ^ 1

        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
            layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        # Publish completed TMEM reads before CTA reuse or deallocation.
        T.ptx.tcgen05.fence.before_thread_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st + BLK_N], Dreg_f16[:])

        # --- TMEM cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 4: TMA async load
#   Replace sync load with TMA (single-thread dispatch, mbarrier sync).
#   Writeback uses TMA store: TMEM → RF → SMEM → TMA → GMEM.
# ======================================================================

def hgemm_v4(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("Step 4 requires positive M,N divisible by 128 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
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
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))
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
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 5: Software pipeline
#   PIPE_DEPTH=2 multi-buffered SMEM. Prefetch + overlap.
# ======================================================================

def hgemm_v5(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("Step 5 requires positive M,N divisible by 128 and K divisible by 64")
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
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
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
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32
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

        # Prefetch both buffers. Refill a stage only after MMA releases it.
        if warp_id == 0:
            if T.filter(lane_id, T.ptx.elect_sync()):
                for s in T.unroll(PRE_NUM):
                    tma_load(s, s * BLK_K)
                # Unroll one ring, keeping shared-memory stage offsets constant.
                # Each stage is used once per ring, so its phase is ring parity.
                for ring in range((K_TILES + PIPE_DEPTH - 1) // PIPE_DEPTH):
                    for stage in T.unroll(PIPE_DEPTH):
                        k = T.meta_var(ring * PIPE_DEPTH + stage)
                        if k < K_TILES:
                            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), ring % 2)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            mma(stage, k != 0)
                            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                            phase_mma = phase_mma ^ 1
                            if k + PIPE_DEPTH < K_TILES:
                                tma_load(stage, (k + PIPE_DEPTH) * BLK_K)
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
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 6: Persistent kernel + tile scheduler
#   Fixed SM_COUNT CTAs, loop over tiles with L2-friendly ordering.
# ======================================================================

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


# ======================================================================
# Step 7: Warp specialization (PIPE_DEPTH=2)
#   WG1: warp0 (MMA) + warp3 (TMA producer)
#   WG0: writeback (TMEM → RF → SMEM → GMEM)
#   4 barrier types: tma2mma, mma2tma, mma2ld, ld2mma
#   PIPE_DEPTH=2 (same as step 6, focus on warp spec structure)
# ======================================================================

def hgemm_v7(M, N, K):
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 64:
        raise ValueError("Step 7 requires positive M,N divisible by 128 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    EPI_N = 64
    TMEM_LD_N = 32  # Amortize TMEM load/wait overhead without a full-row FP32 buffer.
    WG_NUMBER = 2
    CTA_COUNT = min(SM_COUNT, (M // BLK_M) * (N // BLK_N))
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
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
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_phase.stage, tma_phase.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_phase.stage,
                                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_phase.advance()
                        tile_scheduler.next_tile()
            elif warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        ld2mma.wait(0, ld_phase.phase)
                        ld_phase.advance()
                        for k in range(K_TILES):
                            tma2mma.wait(mma_phase.stage, mma_phase.phase)
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


# ======================================================================
# Step 8: Deeper pipeline (PIPE_DEPTH=4)
#   Same warp-specialized structure as v7, but with 4-stage pipeline
#   to better hide TMA latency. Only changes: PIPE_DEPTH=2 → 4,
#   which affects barrier array sizes and Asmem/Bsmem stage dimensions.
# ======================================================================

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

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
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
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_phase.stage, tma_phase.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_phase.stage,
                                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_phase.advance()
                        tile_scheduler.next_tile()
            elif warp_id == 0:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        ld2mma.wait(0, ld_phase.phase)
                        ld_phase.advance()
                        for k in range(K_TILES):
                            tma2mma.wait(mma_phase.stage, mma_phase.phase)
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


# ======================================================================
# Step 9: Cluster — 2-CTA cooperation
#   CTA_GROUP=2, MMA_M=MMA_N=256, cross-CTA TMEM sharing.
# ======================================================================

def hgemm_v9(M, N, K):
    if min(M, N, K) <= 0 or M % 256 or N % 256 or K % 64:
        raise ValueError("Step 9 requires positive M,N divisible by 256 and K divisible by 64")
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    CTA_GROUP = 2
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    TMEM_LD_N = 32  # Eight loads cover each CTA's 256-column result.
    WG_NUMBER = 2
    CLUSTER_COUNT = min(SM_COUNT // CTA_GROUP, (M // MMA_M) * (N // MMA_N))
    A_layout = mma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = mma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, EPI_N))

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
            "ts", num_m_tiles=M // MMA_M, num_n_tiles=N // MMA_N,
            l2_group_size=8, num_clusters=CLUSTER_COUNT)
        tile_scheduler.init(bx // CTA_GROUP)
        m_st = T.meta_var(tile_scheduler.m_idx * MMA_M + cbx * BLK_M)
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
            Tx.copy_async(Asmem[tma_phase.stage, :, :],
                          A[m_st:m_st + BLK_M, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=CTA_GROUP,
                          mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
            Tx.copy_async(Bsmem[tma_phase.stage, :, :],
                          B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=CTA_GROUP,
                          mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))

        if wg_id == 1:
            if warp_id == 3:
                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_phase.stage, tma_phase.phase)
                            tma_load(k * BLK_K)
                            # Both CTAs complete bytes against CTA 0's barrier.
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_phase.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_phase.advance()
                        tile_scheduler.next_tile()
            elif warp_id == 0:
                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(0, ld_phase.phase)
                            ld_phase.advance()
                            for k in range(K_TILES):
                                tma2mma.wait(mma_phase.stage, mma_phase.phase)
                                T.ptx.tcgen05.fence.after_thread_sync()
                                Tx.gemm_async(tmem[:, :MMA_N],
                                    Asmem[mma_phase.stage, :, :], Bsmem[mma_phase.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_phase.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_phase.advance()
                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
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
                ld2mma.arrive(0, remote=0)
                for i in T.unroll(MMA_N // EPI_N):
                    col = T.meta_var(i * EPI_N)
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], Dreg_f16[col:col + EPI_N])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if T.filter(lane_id, T.ptx.elect_sync()):
                            Tx.copy_async(D[m_st:m_st + BLK_M, n_out + col:n_out + col + EPI_N],
                                          Dsmem[:, :], dispatch="tma_auto")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)
                tile_scheduler.next_tile()

        T.cuda.cluster_sync()
        # Exactly the allocating warp releases the TMEM allocation.
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel


# ======================================================================
# Step 10: 2-consumer warp specialization
#   NUM_CONSUMER=2, WG2 (TMA+MMA), WG0/WG1 (writeback).
#   This is the final optimized kernel.
# ======================================================================

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
        elif wg_id < NUM_CONSUMER:
            m_out = T.meta_var(m_st + wg_id * MMA_M)
            Dreg = T.alloc_local((TMEM_LD_N,), acc_type)
            Dreg_f16 = T.alloc_local((MMA_N,), d_type)
            Dreg_wg = Dreg.view(128, TMEM_LD_N,
                layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_phase.phase)
                wb_phase.advance()
                T.ptx.tcgen05.fence.after_thread_sync()
                for i in T.unroll(MMA_N // TMEM_LD_N):
                    col = T.meta_var(i * TMEM_LD_N)
                    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, wg_id * MMA_N + col:wg_id * MMA_N + col + TMEM_LD_N])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(Dreg_f16[col:col + TMEM_LD_N], Dreg[:])
                # All TMEM reads have finished. MMA can overlap the TMA epilogue.
                T.ptx.tcgen05.fence.before_thread_sync()
                ld2mma.arrive(wg_id, remote=0)
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
