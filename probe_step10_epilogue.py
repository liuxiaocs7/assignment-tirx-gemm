"""Shared-B Step 10 writeback candidates on the current N128/two-slot path.

Keep four input stages and the current grid/MMA/TMEM ownership. The narrow
path has room to batch output stores or double-buffer them without reducing
input depth, unlike the archived wide-N three-stage epilogue experiment.
Measured in step10_epilogue.xVWC2d: all three lose every pair to production.
Retained for reproducibility; none is adopted. See B300_VALIDATION.md.
"""

from probe_step45 import replace_once


EPILOGUE_CONFIGS = {
    "tmem_epi64": (64, 1),
    "tmem_epi128": (128, 1),
    "tmem_epi32_double": (32, 2),
}
EPILOGUE_CONTROLS = {variant: "baseline" for variant in EPILOGUE_CONFIGS}
EPILOGUE_VERIFY_SHAPES = (
    (4096, 3072, 64), (4096, 3072, 192), (4096, 3072, 256),
    (4096, 3072, 320), (4608, 2560, 320), (512, 9728, 192),
)


def epilogue_builder_source(source, variant):
    """Change only output staging on eligible shapes; fail on baseline drift."""
    required = (
        "    NARROW_N = M * N <= 4096 * 4096\n",
        "    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64\n",
        "    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)\n",
        "    NUM_CONSUMER = 2\n", "    PIPE_DEPTH = 4\n",
        "    EPI_N = 32 if NARROW_N else 64\n",
        "    D_SWIZZLE = SwizzleMode.SWIZZLE_64B_ATOM if NARROW_N else SwizzleMode.SWIZZLE_128B_ATOM\n",
        "    D_layout = mma_shared_layout(d_type, D_SWIZZLE, (NUM_CONSUMER, BLK_M, EPI_N))\n",
        "    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1\n",
        "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n",
        "            l2_group_size=8, num_clusters=CLUSTER_COUNT)\n",
        "        mma_tmem_base: T.let = tmem_addr[0]\n",
        "        def tma_load(k_st):\n            Tx.copy_async(Bsmem[",
        "                ld2mma.arrive(wb_phase.stage * TMEM_SLOT_STRIDE + wg_id, remote=0)\n",
        "                            T.ptx.cp_async.bulk.wait_group(0)\n",
    )
    if variant not in EPILOGUE_CONFIGS or any(source.count(s) != 1 for s in required):
        raise ValueError("epilogue probes require the adopted Step 10 narrow/TMEM baseline")
    width, buffers = EPILOGUE_CONFIGS[variant]
    marker = "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n"
    if buffers == 1:
        source = replace_once(source, marker, marker + f"    if TMEM_BUFFERS == 2:\n        EPI_N = {width}\n")
        return replace_once(source,
            "    D_SWIZZLE = SwizzleMode.SWIZZLE_64B_ATOM if NARROW_N else SwizzleMode.SWIZZLE_128B_ATOM\n",
            "    D_SWIZZLE = SwizzleMode.SWIZZLE_64B_ATOM if EPI_N == 32 else SwizzleMode.SWIZZLE_128B_ATOM\n")

    source = replace_once(source, marker, marker + "    EPI_BUFFERS = 2 if TMEM_BUFFERS == 2 else 1\n")
    shape = "(NUM_CONSUMER, BLK_M, EPI_N)"
    if source.count(shape) != 2:
        raise ValueError("expected exactly one output layout and one output allocation")
    source = source.replace(shape, "(NUM_CONSUMER, EPI_BUFFERS, BLK_M, EPI_N)")
    source = replace_once(source, "Dsmem[wg_id, warp_id * 32 + lane_id, :]",
                          "Dsmem[wg_id, i % EPI_BUFFERS, warp_id * 32 + lane_id, :]")
    source = replace_once(source, "Dsmem[wg_id, :, :]", "Dsmem[wg_id, i % EPI_BUFFERS, :, :]")
    # Bulk groups belong to the issuing thread. Use a fixed lane in the
    # pipelined branch; the fallback keeps the original election predicate.
    source = replace_once(source, "                        if T.filter(lane_id, T.ptx.elect_sync()):\n",
                          "                        if T.filter(lane_id, lane_id == 0 if EPI_BUFFERS == 2 else T.ptx.elect_sync()):\n")
    # After chunk 1, wait(1) releases buffer 0 before chunk 2 overwrites it;
    # after chunk 2 it releases buffer 1. The final wait(0) drains both.
    # The unchanged trailing WG barrier publishes completion to all writers.
    return replace_once(source, "                            T.ptx.cp_async.bulk.wait_group(0)\n",
        "                            if EPI_BUFFERS == 1 or i == MMA_N // EPI_N - 1:\n"
        "                                T.ptx.cp_async.bulk.wait_group(0)\n"
        "                            elif i > 0:\n"
        "                                T.ptx.cp_async.bulk.wait_group(1)\n")
