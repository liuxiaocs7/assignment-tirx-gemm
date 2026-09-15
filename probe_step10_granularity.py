"""Archived shared-B input-granularity experiments on the current Step 10.

Separate the required 64-byte swizzle from K32 staging, then vary capacity.
K64/depth4 and K32/depth8 hold the same input bytes. K32 doubles per-tile
TMA requests and barrier epochs; smaller transactions need not be faster.
Production dispatch, timing and grading are unchanged.
Measured in step10_granularity.nVKp6r: all three candidates lose every pair
to production by about 28% latency. None is adopted; retained for replay.
"""

from probe_step45 import replace_once


GRANULARITY_CONFIGS = {
    "tmem_k64_sw64": (64, 4),
    "tmem_k32_depth8": (32, 8),
    "tmem_k32_depth10": (32, 10),
}
GRANULARITY_CONTROLS = {
    "tmem_k64_sw64": "baseline",
    "tmem_k32_depth8": "tmem_k64_sw64",
    "tmem_k32_depth10": "tmem_k32_depth8",
}
# K32: short rings, exactly 8/10 stages, partial rings and two full rings.
# Three output tiles per cluster exercise TMEM reuse and cross-tile phases.
GRANULARITY_VERIFY_SHAPES = tuple((4096, 3072, k) for k in (64, 192, 256, 320, 384, 512)) + (
    (4608, 2560, 320), (512, 9728, 192),
)


def granularity_builder_source(source, variant):
    """Change input shape/layout only on the original narrow two-slot path."""
    marker = "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n"
    required = (
        "    NARROW_N = M * N <= 4096 * 4096\n",
        "    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64\n",
        "    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)\n",
        "    PIPE_DEPTH = 4\n", "    K_TILES = K // BLK_K\n",
        "    NUM_CONSUMER = 2\n", "    CTA_GROUP = 2\n",
        "    EPI_N = 32 if NARROW_N else 64\n",
        "    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1\n",
        marker, "            l2_group_size=8, num_clusters=CLUSTER_COUNT)\n",
        "        mma2tma.init(NUM_CONSUMER)\n",
        "        def tma_load(k_st):\n            Tx.copy_async(Bsmem[",
        "                                mma2tma.arrive(mma_phase.stage, cta_group=CTA_GROUP, cta_mask=3)\n",
    )
    if (variant not in GRANULARITY_CONFIGS or "INPUT_SWIZZLE" in source
            or any(source.count(s) != 1 for s in required)):
        raise ValueError("granularity probes require the adopted Step 10 narrow/TMEM baseline")
    blk_k, depth = GRANULARITY_CONFIGS[variant]
    source = replace_once(source, required[1], "    BLK_M, BLK_N = 128, (64 if NARROW_N else 128)\n")
    source = replace_once(source, required[3], "")
    source = replace_once(source, required[4], "")
    source = replace_once(source, marker, marker +
        f"    BLK_K = {blk_k} if TMEM_BUFFERS == 2 else 64\n"
        f"    PIPE_DEPTH = {depth} if TMEM_BUFFERS == 2 else 4\n"
        "    K_TILES = K // BLK_K\n"
        "    INPUT_SWIZZLE = SwizzleMode.SWIZZLE_64B_ATOM if TMEM_BUFFERS == 2 else SwizzleMode.SWIZZLE_128B_ATOM\n")
    for name in ("A", "B"):
        source = replace_once(source,
            f"    {name}_layout = mma_shared_layout({name.lower()}_type, SwizzleMode.SWIZZLE_128B_ATOM,",
            f"    {name}_layout = mma_shared_layout({name.lower()}_type, INPUT_SWIZZLE,")
    return source
