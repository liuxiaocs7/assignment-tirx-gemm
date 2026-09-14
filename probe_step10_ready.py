"""Split input-ready notifications on the current shared-B Step 10 path.

Each MMA consumer needs B and its own A, not the other consumer's A. Keep
one shared input-free barrier, so neither A nor B is overwritten until both
consumers finish. This is an unmeasured candidate, not production dispatch.
"""

from probe_step45 import replace_once


READY_VARIANT = "tmem_split_ready"
READY_VERIFY_SHAPES = tuple((4096, 3072, k) for k in (64, 128, 192, 256, 320, 384)) + (
    (4608, 2560, 320), (512, 9728, 192),
)


def ready_builder_source(source):
    """Preserve input storage/work, splitting only A/B completion barriers."""
    required = (
        "    NARROW_N = M * N <= 4096 * 4096\n",
        "    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64\n",
        "    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)\n",
        "    PIPE_DEPTH = 4\n", "    NUM_CONSUMER = 2\n",
        "    EPI_N = 32 if NARROW_N else 64\n",
        "    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1\n",
        "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n",
        "        mma2tma.init(NUM_CONSUMER)\n",
        "            l2_group_size=8, num_clusters=CLUSTER_COUNT)\n",
        "        def tma_load(k_st):\n            Tx.copy_async(Bsmem[",
        "                            mma2tma.wait(tma_phase.stage, tma_phase.phase)\n",
        "                                mma2tma.arrive(mma_phase.stage, cta_group=CTA_GROUP, cta_mask=3)\n",
    )
    if "SPLIT_INPUT_READY" in source or any(source.count(s) != 1 for s in required):
        raise ValueError("split-ready probe requires the adopted Step 10 narrow/TMEM baseline")
    source = replace_once(source, required[7], required[7] +
                          "    SPLIT_INPUT_READY = TMEM_BUFFERS == 2\n")
    # Append to the reserved barrier area. All existing barriers, data SMEM,
    # and TMEM keep their original addresses; 25 uint64 slots fit below 1024.
    source = replace_once(source,
        "        ld2mma = MBarrier(pool, TMEM_BUFFERS * NUM_CONSUMER)\n",
        "        ld2mma = MBarrier(pool, TMEM_BUFFERS * NUM_CONSUMER)\n"
        "        if SPLIT_INPUT_READY:\n"
        "            tma2mma_a = TMABar(pool, PIPE_DEPTH * NUM_CONSUMER)\n")
    source = replace_once(source, "        tma2mma.init(1)\n",
        "        tma2mma.init(1)\n"
        "        if SPLIT_INPUT_READY:\n"
        "            tma2mma_a.init(1)\n")
    source = replace_once(source, "        tma2mma_cta0 = tma2mma.remote_view(0)\n",
        "        tma2mma_cta0 = tma2mma.remote_view(0)\n"
        "        if SPLIT_INPUT_READY:\n"
        "            tma2mma_a_cta0 = tma2mma_a.remote_view(0)\n")
    source = replace_once(source,
        "                m_consumer = T.meta_var(m_st + consumer * MMA_M)\n",
        "                m_consumer = T.meta_var(m_st + consumer * MMA_M)\n"
        "                a_ready = T.meta_var(tma2mma_a_cta0.ptr_to([tma_phase.stage * NUM_CONSUMER + consumer])\n"
        "                    if SPLIT_INPUT_READY else tma2mma_cta0.ptr_to([tma_phase.stage]))\n")
    source = replace_once(source,
        "                              mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))\n",
        "                              mbar=a_ready)\n")
    before = ("                                tma2mma_cta0.arrive(tma_phase.stage,\n"
              "                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)\n")
    after = ("                                if SPLIT_INPUT_READY:\n"
             "                                    # B is loaded once and acquired by both consumers.\n"
             "                                    tma2mma_cta0.arrive(tma_phase.stage, CTA_GROUP * BLK_N * BLK_K * F16_SIZE)\n"
             "                                    for consumer in T.unroll(NUM_CONSUMER):\n"
             "                                        tma2mma_a_cta0.arrive(tma_phase.stage * NUM_CONSUMER + consumer,\n"
             "                                            CTA_GROUP * BLK_M * BLK_K * F16_SIZE)\n"
             "                                else:\n" +
             "".join("    " + line for line in before.splitlines(keepends=True)))
    source = replace_once(source, before, after)
    wait = "                                tma2mma.wait(mma_phase.stage, mma_phase.phase)\n"
    return replace_once(source, wait, wait +
        "                                if SPLIT_INPUT_READY:\n"
        "                                    tma2mma_a.wait(mma_phase.stage * NUM_CONSUMER + warp_id, mma_phase.phase)\n")
