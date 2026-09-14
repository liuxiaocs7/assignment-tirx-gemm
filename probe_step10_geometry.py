"""Independent grid/tile experiments for the current shared-B Step 10 builder.

Only the production narrow, persistent two-slot branch is changed. B300 run
step10_geometry.PywrmG measured all three candidates slower than baseline;
they remain archived experiments, not production dispatch choices.
"""

from probe_step45 import replace_once


GEOMETRY_CONFIGS = {
    "tmem_max_clusters": (128, 4),
    "tmem_n64": (64, 4),
    "tmem_n64_depth5": (64, 5),
}
GEOMETRY_CONTROLS = {
    "tmem_max_clusters": "baseline",
    "tmem_n64": "tmem_max_clusters",
    "tmem_n64_depth5": "tmem_n64",
}
# Short/full/partial four- and five-stage input rings, repeated TMEM reuse,
# partial L2 groups, and the first shape beyond the production single wave.
GEOMETRY_VERIFY_SHAPES = tuple((4096, 3072, k) for k in (64, 128, 192, 256, 320, 384)) + (
    (4608, 2560, 320), (512, 9728, 192),
)


def geometry_builder_source(source, variant):
    """Vary grid count, then N width, then input depth against direct controls."""
    tiles = "    TOTAL_TILES = (M // (MMA_M * NUM_CONSUMER)) * (N // MMA_N)\n"
    waves = "    TILES_PER_CLUSTER = (TOTAL_TILES + MAX_CLUSTERS - 1) // MAX_CLUSTERS\n"
    clusters = "    CLUSTER_COUNT = (TOTAL_TILES + TILES_PER_CLUSTER - 1) // TILES_PER_CLUSTER\n"
    required = (
        "    NARROW_N = M * N <= 4096 * 4096\n",
        "    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64\n",
        "    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)\n",
        "    CTA_GROUP = 2\n", "    NUM_CONSUMER = 2\n", "    PIPE_DEPTH = 4\n",
        "    EPI_N = 32 if NARROW_N else 64\n", tiles, waves, clusters,
        "    MAX_CLUSTERS = SM_COUNT // CTA_GROUP\n",
        "    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1\n",
        "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n",
        '            "ts", num_m_tiles=M // (MMA_M * NUM_CONSUMER), num_n_tiles=N // MMA_N,\n',
        "            l2_group_size=8, num_clusters=CLUSTER_COUNT)\n",
        "        mma_tmem_base: T.let = tmem_addr[0]\n",
        "        def tma_load(k_st):\n            Tx.copy_async(Bsmem[",
    )
    if variant not in GEOMETRY_CONFIGS or any(source.count(s) != 1 for s in required):
        raise ValueError("geometry probes require the adopted Step 10 narrow/TMEM baseline")
    mma_n, depth = GEOMETRY_CONFIGS[variant]
    if mma_n == 64:
        # Decide eligibility using the original N128 tile count. Recomputing
        # that decision after shrinking N would also change single-wave paths.
        config = (
            "    # Probe only: finer shared-B tiles; keep the 512-column TMEM allocation.\n"
            "    if TMEM_BUFFERS == 2:\n"
            "        BLK_N = 32\n"
            "        MMA_N = 64\n"
            f"    {tiles}"
        )
        if depth == 5:
            config += "        PIPE_DEPTH = 5\n"
        source = replace_once(source, waves, config + waves)
    return replace_once(source, clusters,
        "    CLUSTER_COUNT = (min(MAX_CLUSTERS, TOTAL_TILES) if TMEM_BUFFERS == 2\n"
        "                     else (TOTAL_TILES + TILES_PER_CLUSTER - 1) // TILES_PER_CLUSTER)\n")
