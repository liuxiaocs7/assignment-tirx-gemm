"""Independent B300 performance experiments for persistent GEMM kernels.

Step 10 has adopted measured narrow-N/TMEM choices by output workload.
The default validates the production baseline. Historical wide-N probes must
be replayed at their recorded commits, before the workload dispatch adoption.
Explicit tmem_input_* / tmem_k128_depth2 probes vary the input ring only for
the current double-buffered narrow path, retaining single-slot/wide fallbacks.
Explicit tmem_share_a_depth* probes place the two consumers along N to reuse A,
then compare five/six input stages on that layout.
Explicit tmem_l2_group* probes change only tile ordering on the current
double-buffered narrow path, retaining the production group-eight fallbacks.
Step 9 has adopted cluster_cache_tmem_base after AB/BA rechecks. Replay that
comparison at 3a9d486; use benchmark.py --steps 9 for the current production path.
Historical experiments are explicit; adopted transforms refuse reapplication.
GPU verification precedes every scored experiment.
No production kernel is edited by this tool.
All variants must verify before interleaved timing with the original CUDA-event
timer. SLOW is a measured result; numerical or compilation errors stop the run.
"""

import argparse
from contextlib import contextmanager
import csv
import difflib
import hashlib
import inspect
import json
import re
import statistics
from pathlib import Path

from benchmark_diagnostics import capture_compilation, run_metadata, write_json
from probe_step45 import replace_once, source_experiment, summarize, trial_order
from profile_persistent import dump_sass


STEP_VARIANTS = {
    6: ("baseline", "k_tile_128", "mma_wait_64ns", "final_fence"),
    7: ("baseline", "k_tile_128", "mma_wait_64ns", "epilogue_128"),
    8: ("baseline", "tma_wait_64ns", "epilogue_128", "cache_tmem_base", "reuse_wait_64ns"),
    9: ("baseline", "cluster_cache_tmem_base"),
    10: ("baseline", "mma_wait_64ns", "tmem_load_16", "tmem_load_64",
         "l2_group_4", "balanced_clusters", "tma_wait_64ns",
         "specialize_mma", "specialize_writeback", "unroll_ring",
         "pipe_depth_2", "k128_depth_2", "stream_epilogue",
         "cache_tmem_base", "reuse_wait_64ns", "ring_wait_64ns",
         "cache_unroll_ring", "cache_mma_no_unroll", "cache_balanced_clusters",
         "balanced_depth3", "balanced_depth3_epi128", "balanced_fused_a",
         "warp_release", "paired_tmem_loads", "epilogue_depth3", "epilogue_double_buffer",
         "role_registers", "tma_b_first", "n_tile_128", "n128_epi32", "n128_epi32_depth5",
         "epilogue_32", "split_tma", "mma_batch", "mma_batch_no_unroll",
         "mma_unroll4", "mma_batch_unroll4", "n128_tmem_double_buffer",
         "tmem_input_depth2", "tmem_k128_depth2", "tmem_input_depth5",
         "tmem_share_a_depth5", "tmem_share_a_depth6",
         "tmem_l2_group4", "tmem_l2_group2", "tmem_l2_group1"),
}
DEFAULT_STEP_VARIANTS = {6: ("baseline",), 7: ("baseline",),
                         8: ("baseline",),
                         9: ("baseline",),
                         10: ("baseline",)}
VARIANTS = tuple(dict.fromkeys(v for variants in STEP_VARIANTS.values() for v in variants))
# Each combination varies exactly one factor relative to cache_tmem_base.
CACHE_EXPERIMENTS = {
    "cache_unroll_ring": "unroll_ring",
    "cache_mma_no_unroll": None,  # CUDA-only pragma; the builder is cache-only.
    "cache_balanced_clusters": "balanced_clusters",
}

# Every new comparison names its direct control; selecting a leaf includes
# the whole chain so each added change remains separable.
EXPERIMENT_CONTROLS = {**{v: "cache_tmem_base" for v in CACHE_EXPERIMENTS},
                       "cluster_cache_tmem_base": "baseline",
                       "balanced_depth3": "cache_balanced_clusters",
                       "balanced_depth3_epi128": "balanced_depth3",
                       "balanced_fused_a": "cache_balanced_clusters",
                       "epilogue_double_buffer": "epilogue_depth3",
                       "n128_epi32": "n_tile_128",
                       "n128_epi32_depth5": "n128_epi32",
                       "mma_batch_no_unroll": "mma_batch",
                       "mma_batch_unroll4": "mma_unroll4",
                       "n128_tmem_double_buffer": "n128_epi32",
                       "tmem_input_depth2": "baseline",
                       "tmem_k128_depth2": "tmem_input_depth2",
                       "tmem_input_depth5": "baseline",
                       "tmem_share_a_depth5": "tmem_input_depth5",
                       "tmem_share_a_depth6": "tmem_share_a_depth5",
                       **{f"tmem_l2_group{g}": "baseline" for g in (4, 2, 1)}}
CURRENT_INPUT_VARIANTS = ("tmem_input_depth2", "tmem_k128_depth2", "tmem_input_depth5")
SHARE_A_VARIANTS = ("tmem_share_a_depth5", "tmem_share_a_depth6")
CURRENT_L2_VARIANTS = {f"tmem_l2_group{g}": g for g in (4, 2, 1)}
NARROW_N_VARIANTS = ("n_tile_128", "n128_epi32", "n128_epi32_depth5")
DEPTH3_VARIANTS = ("balanced_depth3", "balanced_depth3_epi128")
# K=1/3/5 stages covers shorter-than-ring, odd complete ring, and partial
# ring; 96 cluster tiles force reuse of both consumers on the balanced grid.
DEPTH3_VERIFY_SHAPES = ((4096, 3072, 64), (4096, 3072, 192), (4096, 3072, 320))
VERIFICATION_SHAPES = {
    # Single tile, cross-tile reuse with short/full/partial rings, and a
    # 9x9 grid that crosses both a partial L2 group and the 74-cluster limit.
    "cluster_cache_tmem_base": ((256, 256, 64), (4096, 3072, 64),
                                (4096, 3072, 192), (4096, 3072, 256),
                                (4096, 3072, 320), (2304, 2304, 320)),
    **{variant: DEPTH3_VERIFY_SHAPES for variant in DEPTH3_VARIANTS},
    "balanced_fused_a": ((4096, 3072, 64), (4096, 3072, 320)),
    "warp_release": ((4096, 3072, 64), (4096, 3072, 320)),
    "paired_tmem_loads": ((4096, 3072, 64), (4096, 3072, 320)),
    "epilogue_depth3": DEPTH3_VERIFY_SHAPES,
    "epilogue_double_buffer": DEPTH3_VERIFY_SHAPES,
    "role_registers": ((4096, 3072, 64), (4096, 3072, 320)),
    "tma_b_first": ((4096, 3072, 64), (4096, 3072, 320)),
    # One stage, exactly one five-stage ring, and a partial second ring.
    # 192 narrow output tiles on 64 clusters force three tiles per cluster.
    **{variant: ((4096, 3072, 64), (4096, 3072, 320), (4096, 3072, 384))
       for variant in NARROW_N_VARIANTS},
    "epilogue_32": ((4096, 3072, 64), (4096, 3072, 256), (4096, 3072, 320)),
    "split_tma": ((4096, 3072, 64), (4096, 3072, 256), (4096, 3072, 320)),
    "mma_batch": ((4096, 3072, 64), (4096, 3072, 256), (4096, 3072, 320)),
    "mma_batch_no_unroll": ((4096, 3072, 64), (4096, 3072, 256), (4096, 3072, 320)),
    "mma_unroll4": ((4096, 3072, 64), (4096, 3072, 192), (4096, 3072, 256), (4096, 3072, 320)),
    "mma_batch_unroll4": ((4096, 3072, 64), (4096, 3072, 192), (4096, 3072, 256), (4096, 3072, 320)),
    # 192 narrow tiles / 64 clusters = three tiles per cluster: slot 0 is
    # reused after slot 1. K=64/192/256/320 crosses different input-ring phases.
    "n128_tmem_double_buffer": ((4096, 3072, 64), (4096, 3072, 192),
                                 (4096, 3072, 256), (4096, 3072, 320)),
    # Three output tiles per cluster reuse TMEM slot zero after slot one.
    # K64/192/320 retain the K64 fallback; K128/256/384 exercise K128 and
    # odd/even two-stage rings. K320/384 also cross the five-stage ring.
    **{variant: tuple((4096, 3072, k) for k in (64, 128, 192, 256, 320, 384))
       for variant in CURRENT_INPUT_VARIANTS},
    # Same output area, now 256x256 per cluster. Three tiles force TMEM reuse;
    # K320/384/448 end before/on/after the six-stage input ring boundary.
    **{variant: tuple((4096, 3072, k) for k in (64, 320, 384, 448))
       + ((1536, 5376, 320),)  # 6x21 output tiles, partial L2 group and two waves.
       for variant in SHARE_A_VARIANTS},
    # Three persistent tiles per cluster with short/full/partial input rings;
    # nine M tiles exercise complete L2 groups followed by a one-row tail.
    # The last shape has only one M tile and just exceeds the single-wave limit.
    **{variant: tuple((4096, 3072, k) for k in (64, 192, 256, 320))
       + ((4608, 2560, 320), (512, 9728, 192))
       for variant in CURRENT_L2_VARIANTS},
}


def select_variants(step, requested=None):
    selected = ["baseline"]
    def add(variant):
        if variant not in STEP_VARIANTS[step]:
            raise ValueError(f"Step {step} does not support {variant}")
        if variant in EXPERIMENT_CONTROLS:
            add(EXPERIMENT_CONTROLS[variant])
        if variant not in selected:
            selected.append(variant)
    for variant in requested or DEFAULT_STEP_VARIANTS[step]:
        add(variant)
    return selected


def unroll_pipeline_ring(source):
    """Use fixed stage offsets for complete rings; preserve partial-ring code."""
    # A partial ring carries its stage into the next output tile. Keep that
    # path byte-for-byte; for aligned K, each tile starts at stage zero while
    # phase still lives across tiles (including an odd number of rings).
    source = replace_once(source, "    PIPE_DEPTH = 4\n",
                          "    PIPE_DEPTH = 4\n    UNROLL_RING = K_TILES % PIPE_DEPTH == 0\n")
    source = replace_once(source, "        def tma_load(k_st):\n",
                          "        def tma_load(k_st, stage):\n")
    start, end = "        def tma_load(k_st, stage):\n", "        if wg_id == 2:\n"
    prefix, tail = source.split(start)
    block, suffix = tail.split(end)
    source = prefix + start + block.replace("tma_phase.stage", "stage") + end + suffix
    source = replace_once(source, "                            tma_load(k * BLK_K)\n",
                          "                            tma_load(k * BLK_K, tma_phase.stage)\n")
    for phase, indent, sentinel in (
        ("mma_phase", 28, "                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)\n"),
        ("tma_phase", 24, "                        tile_scheduler.next_tile()\n"),
    ):
        pad = " " * indent
        start = "\n" + pad + "for k in range(K_TILES):\n"
        if source.count(start) != 1:
            raise ValueError("expected one K loop for the pipeline role")
        prefix, tail = source.split(start)
        sentinel = "\n" + sentinel
        if tail.count(sentinel) != 1:
            raise ValueError("expected one end of the pipeline K loop")
        block, suffix = tail.split(sentinel)
        block += "\n"
        original = start.lstrip("\n") + block
        static = replace_once(block, pad + f"    {phase}.advance()\n", "")
        static = static.replace(phase + ".stage", "stage")
        static = "".join("        " + line if line.strip() else line
                         for line in static.splitlines(keepends=True))
        replacement = (pad + "if UNROLL_RING:\n" +
                       pad + "    for ring in range(K_TILES // PIPE_DEPTH):\n" +
                       pad + "        for stage in T.unroll(PIPE_DEPTH):\n" +
                       pad + "            k = T.meta_var(ring * PIPE_DEPTH + stage)\n" +
                       static + pad + f"        {phase}.phase = {phase}.phase ^ 1\n" +
                       pad + "else:\n" +
                       "".join("    " + line if line.strip() else line
                               for line in original.splitlines(keepends=True)))
        source = prefix + "\n" + replacement + sentinel.lstrip("\n") + suffix
    return source


def fuse_consumer_a_loads(source):
    """One 3-D TMA box gathers the two A blocks without changing MMA layout."""
    source = replace_once(source, "        @T.inline\n        def tma_load(k_st):\n",
                          "        A_blocks = A.view(M // MMA_M, MMA_M, K)\n\n"
                          "        @T.inline\n        def tma_load(k_st):\n")
    before = '''            for consumer in T.unroll(NUM_CONSUMER):
                m_consumer = T.meta_var(m_st + consumer * MMA_M)
                Tx.copy_async(Asmem[tma_phase.stage, consumer, :, :],
                              A[m_consumer:m_consumer + BLK_M, k_st:k_st + BLK_K],
                              dispatch="tma_auto", cta_group=CTA_GROUP,
                              mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
'''
    after = '''            Tx.copy_async(Asmem[tma_phase.stage, :, :, :],
                          A_blocks[tile_scheduler.m_idx * NUM_CONSUMER:tile_scheduler.m_idx * NUM_CONSUMER + NUM_CONSUMER,
                                   cbx * BLK_M:cbx * BLK_M + BLK_M, k_st:k_st + BLK_K],
                          dispatch="tma_auto", cta_group=CTA_GROUP,
                          mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
'''
    # A[(tile*2 + consumer)*256 + rank*128 + row, k]. The view aliases
    # the original contiguous A buffer; no transpose, packing, or allocation.
    return replace_once(source, before, after)


def aggregate_tmem_release(source):
    """One arrival per warp after every lane finishes its TMEM reads."""
    before = ("                T.ptx.tcgen05.fence.before_thread_sync()\n"
              "                ld2mma.arrive(wg_id, remote=0)\n")
    after = ("                T.ptx.tcgen05.fence.before_thread_sync()\n"
             "                T.cuda.warp_sync()\n"
             "                if T.filter(lane_id, T.ptx.elect_sync()):\n"
             "                    ld2mma.arrive(wg_id, remote=0, count=32)\n")
    # Keep init(128 * CTA_GROUP): 4 warps * 2 CTAs * 32 arrivals per leader.
    # The full-warp sync orders the other lanes' completed reads before the
    # elected lane releases TMEM. Each consumer retains its own barrier slot.
    return replace_once(source, before, after)


def pair_tmem_loads(source):
    """Issue two x32 reads into disjoint registers, then wait and convert."""
    source = replace_once(source, "            Dreg = T.alloc_local((TMEM_LD_N,), acc_type)\n",
                          "            Dreg = T.alloc_local((2 * TMEM_LD_N,), acc_type)\n")
    source = replace_once(source, "            Dreg_wg = Dreg.view(128, TMEM_LD_N,\n"
                          "                layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))\n",
                          "            Dreg_wg = Dreg.view(128, 2 * TMEM_LD_N,\n"
                          "                layout=TileLayout(S[(128, 2 * TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))\n")
    before = '''                for i in T.unroll(MMA_N // TMEM_LD_N):
                    col = T.meta_var(i * TMEM_LD_N)
                    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, wg_id * MMA_N + col:wg_id * MMA_N + col + TMEM_LD_N])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(Dreg_f16[col:col + TMEM_LD_N], Dreg[:])
'''
    after = '''                for i in T.unroll(MMA_N // (2 * TMEM_LD_N)):
                    col = T.meta_var(i * 2 * TMEM_LD_N)
                    for part in T.unroll(2):
                        off = T.meta_var(part * TMEM_LD_N)
                        Tx.wg.copy_async(Dreg_wg[:, off:off + TMEM_LD_N],
                            tmem[:, wg_id * MMA_N + col + off:wg_id * MMA_N + col + off + TMEM_LD_N])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(Dreg_f16[col:col + 2 * TMEM_LD_N], Dreg[:])
'''
    return replace_once(source, before, after)


def double_buffer_tmem(source):
    """Two independently fenced accumulator slots per narrow-N consumer.

    Each slot owns a ready/free barrier. MMA may start tile t+1 in the other
    slot while writeback reads tile t, but tile t+2 must await that slot's
    readers. Input-ring reuse still waits for both MMA consumers as before.
    """
    if "    MMA_M, MMA_N = 256, 128\n" not in source or "    EPI_N = 32\n" not in source:
        raise ValueError("TMEM double buffering requires the n128_epi32 control")
    source = replace_once(source, "    NUM_CONSUMER = 2\n",
                          "    NUM_CONSUMER = 2\n    TMEM_BUFFERS = 2\n")
    for before, after in (
        ("mma2ld = TCGen05Bar(pool, NUM_CONSUMER)", "mma2ld = TCGen05Bar(pool, TMEM_BUFFERS * NUM_CONSUMER)"),
        ("ld2mma = MBarrier(pool, NUM_CONSUMER)", "ld2mma = MBarrier(pool, TMEM_BUFFERS * NUM_CONSUMER)"),
        ("ld_phase = PipelineState(1)", "ld_phase = PipelineState(TMEM_BUFFERS)"),
        ("wb_phase = PipelineState(1)", "wb_phase = PipelineState(TMEM_BUFFERS)"),
        ("ld2mma.wait(warp_id, ld_phase.phase)",
         "ld2mma.wait(ld_phase.stage * NUM_CONSUMER + warp_id, ld_phase.phase)"),
        ("tmem[:, warp_id * MMA_N:(warp_id + 1) * MMA_N]",
         "tmem[:, (ld_phase.stage * NUM_CONSUMER + warp_id) * MMA_N:(ld_phase.stage * NUM_CONSUMER + warp_id + 1) * MMA_N]"),
        ("mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)",
         "mma2ld.arrive(ld_phase.stage * NUM_CONSUMER + warp_id, cta_group=CTA_GROUP, cta_mask=3)"),
        ("mma2ld.wait(wg_id, wb_phase.phase)",
         "mma2ld.wait(wb_phase.stage * NUM_CONSUMER + wg_id, wb_phase.phase)"),
        ("tmem[:, wg_id * MMA_N + col:wg_id * MMA_N + col + TMEM_LD_N]",
         "tmem[:, (wb_phase.stage * NUM_CONSUMER + wg_id) * MMA_N + col:(wb_phase.stage * NUM_CONSUMER + wg_id) * MMA_N + col + TMEM_LD_N]"),
        ("ld2mma.arrive(wg_id, remote=0)",
         "ld2mma.arrive(wb_phase.stage * NUM_CONSUMER + wg_id, remote=0)"),
    ):
        source = replace_once(source, before, after)
    # Keep the current slot unchanged through wait, TMEM use and commit/release.
    # Advance phase only after both slots, not after every output tile.
    source = replace_once(source, "                            ld_phase.advance()\n", "")
    source = replace_once(source,
        "                            mma2ld.arrive(ld_phase.stage * NUM_CONSUMER + warp_id, cta_group=CTA_GROUP, cta_mask=3)\n",
        "                            mma2ld.arrive(ld_phase.stage * NUM_CONSUMER + warp_id, cta_group=CTA_GROUP, cta_mask=3)\n"
        "                            ld_phase.advance()\n")
    source = replace_once(source, "                wb_phase.advance()\n", "")
    return replace_once(source,
        "                ld2mma.arrive(wb_phase.stage * NUM_CONSUMER + wg_id, remote=0)\n",
        "                ld2mma.arrive(wb_phase.stage * NUM_CONSUMER + wg_id, remote=0)\n"
        "                wb_phase.advance()\n")


def double_buffer_epilogue(source):
    """Overlap stores using separate SMEM buffers; drain before each tile ends."""
    # Four input stages plus two output buffers exceed B300's SMEM budget.
    # Compare only with the explicit three-stage control, not a hidden change.
    if "    PIPE_DEPTH = 3\n" not in source:
        raise ValueError("double-buffered epilogue requires the three-stage control")
    for before, after in (
        ("(NUM_CONSUMER, BLK_M, EPI_N)", "(NUM_CONSUMER, 2, BLK_M, EPI_N)"),
        ("Dsmem[wg_id, warp_id * 32 + lane_id, :]",
         "Dsmem[wg_id, i % 2, warp_id * 32 + lane_id, :]"),
        ("Dsmem[wg_id, :, :]", "Dsmem[wg_id, i % 2, :, :]"),
        ("                        if T.filter(lane_id, T.ptx.elect_sync()):\n",
         "                        if T.filter(lane_id, lane_id == 0):\n"),
    ):
        # Shape occurs once in the layout and once in the pool allocation.
        expected = 2 if before.startswith("(NUM_CONSUMER") else 1
        if source.count(before) != expected:
            raise ValueError(f"unexpected epilogue layout: {before}")
        source = source.replace(before, after)
    before = "                            T.ptx.cp_async.bulk.wait_group(0)\n"
    after = '''                            if i == MMA_N // EPI_N - 1:
                                T.ptx.cp_async.bulk.wait_group(0)
                            elif i > 0:
                                T.ptx.cp_async.bulk.wait_group(1)
'''
    # After stores 1/2, wait(1) frees the buffer used two chunks earlier;
    # the unchanged trailing WG barrier publishes this to all writers.
    # Store 3 drains both groups before any next tile or cluster teardown.
    # Bulk groups are per thread: a fixed lane owns all commits and waits.
    return replace_once(source, before, after)


def repartition_role_registers(source):
    """Trade unused producer registers for more writeback scheduling room."""
    if "T.ptx.setmaxnreg(" in source:
        raise ValueError("register budgets are already applied")
    # All 128 threads in a WG participate, before election or role divergence.
    # WG2 can yield while the other WGs wait; no CTA barrier may precede it.
    # 128 * (64 + 208 + 208) = 61,440 registers across the three WGs.
    marker = "        pool.commit()\n"
    return replace_once(source, marker, marker + '''
        if wg_id == 2:
            T.ptx.setmaxnreg(False, 64)
        else:
            T.ptx.setmaxnreg(True, 208)
''')


def load_shared_b_first(source):
    """Reorder independent TMA requests; keep the same byte-count barrier."""
    start = "        def tma_load(k_st):\n"
    end = "\n        if wg_id == 2:\n"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("expected one TMA loader and role boundary")
    prefix, rest = source.split(start)
    block, suffix = rest.split(end)
    split = "            Tx.copy_async(Bsmem[tma_phase.stage, :, :],\n"
    if block.startswith(split):
        raise ValueError("Step 10 has adopted tma_b_first; validate the production kernel "
                         "with benchmark.py --steps 10 and tests/test_step10.py")
    if block.count(split) != 1 or not block.startswith("            for consumer in T.unroll(NUM_CONSUMER):\n"):
        raise ValueError("expected A-first TMA loader")
    a_loads, b_tail = block.split(split)
    b_load = split + b_tail
    if b_load.count("Tx.copy_async(") != 1 or a_loads.count("Tx.copy_async(") != 1:
        raise ValueError("unexpected TMA operand requests")
    return prefix + start + b_load + a_loads + end + suffix


def split_tma_producers(source):
    """WG2 warp 2 loads B; warp 3 loads A0/A1 and announces total bytes."""
    start = "        def tma_load(k_st):\n"
    end = "\n        if wg_id == 2:\n"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("expected one TMA loader and role boundary")
    prefix, rest = source.split(start)
    block, suffix = rest.split(end)
    a_start = "            for consumer in T.unroll(NUM_CONSUMER):\n"
    if not block.startswith("            Tx.copy_async(Bsmem[") or block.count(a_start) != 1:
        raise ValueError("split_tma requires the adopted B-first loader")
    b_load, a_tail = block.split(a_start)
    if b_load.count("Tx.copy_async(") != 1 or a_tail.count("Tx.copy_async(") != 1:
        raise ValueError("unexpected A/B TMA requests")
    indent = lambda text: "".join("    " + line for line in text.splitlines(keepends=True))
    block = ("            if warp_id == 2:\n" + indent(b_load)
             + "            else:\n" + indent(a_start + a_tail))
    source = prefix + start + block + end + suffix
    source = replace_once(source, "            if warp_id == 3:\n",
                          "            if warp_id >= 2:\n")
    # Both producers wait the same empty stage. Only A's elected lane in CTA
    # zero arrives; all six requests still complete bytes on that full slot.
    # Thread-local scheduler and phase state advance identically in both warps.
    source = replace_once(source, "                            if cbx == 0:\n",
                          "                            if (cbx == 0) & (warp_id == 3):\n")
    return source


@contextmanager
def check_role_register_budget(variant, directory):
    """Reject insufficient capacity or explicitly ignored hints before launch."""
    if variant != "role_registers":
        yield
        return
    import tvm_ffi

    name = "tvm_callback_cuda_compile"
    original = tvm_ffi.get_global_func(name)

    def compile_cuda(code):
        binary = original(code)  # Outer capture writes this cubin's resources.
        report = directory / "module_01.resources.txt"
        text = report.read_text() if report.exists() else ""
        matches = re.findall(r"\bREG:(\d+)\b", text)
        if len(matches) != 1:
            raise RuntimeError("role_registers needs cuobjdump resource usage before launch")
        initial = int(matches[0])
        # The three 128-thread WGs request 64/208/208 registers. Inc may
        # block until WG2 releases, but cannot succeed if the CTA pool is too
        # small. Also enforce the PTX inc/dec direction preconditions.
        required = 128 * (64 + 208 + 208)
        capacity_valid = 64 <= initial <= 208 and initial * 384 >= required
        # Capacity alone does not show the hint survived compilation. The
        # uploaded B300 run had REG167 but ptxas C7508 discarded setmaxnreg.
        ignored = any(re.search(r"\bsetmaxnreg\b[^\n]*\bignored\b", log.read_text(), re.I)
                      for log in directory.glob("nvrtc_*.log"))
        valid = capacity_valid and not ignored
        write_json(directory / "register_budget.json", dict(
            initial_registers=initial, threads=384, requested_registers=required,
            budgets=[208, 208, 64], capacity_valid=capacity_valid,
            setmaxnreg_ignored=ignored, valid=valid))
        if ignored:
            raise RuntimeError("role_registers setmaxnreg was ignored by ptxas; not launched")
        if not capacity_valid:
            raise RuntimeError(f"role_registers initial REG:{initial} cannot support 64/208/208; not launched")
        return binary

    tvm_ffi.register_global_func(name, compile_cuda, override=True)
    try:
        yield
    finally:
        tvm_ffi.register_global_func(name, original, override=True)


def stream_epilogue(source):
    """Retain only EPI_N FP16 values, releasing TMEM after the final chunk."""
    source = replace_once(source, "            Dreg_f16 = T.alloc_local((MMA_N,), d_type)\n",
                          "            Dreg_f16 = T.alloc_local((EPI_N,), d_type)\n")
    start = "                for i in T.unroll(MMA_N // TMEM_LD_N):\n"
    end = "                    T.ptx.fence.proxy_async(\"shared::cta\")\n"
    if source.count(start) != 1:
        raise ValueError("expected one TMEM read loop")
    prefix, tail = source.split(start)
    if tail.count(end) != 1:
        raise ValueError("expected one epilogue proxy fence")
    _, suffix = tail.split(end)
    replacement = '''                for i in T.unroll(MMA_N // EPI_N):
                    col = T.meta_var(i * EPI_N)
                    for j in T.unroll(EPI_N // TMEM_LD_N):
                        offset = T.meta_var(j * TMEM_LD_N)
                        Tx.wg.copy_async(Dreg_wg[:, :],
                            tmem[:, wg_id * MMA_N + col + offset:wg_id * MMA_N + col + offset + TMEM_LD_N])
                        T.ptx.tcgen05.wait.ld()
                        Tx.cast(Dreg_f16[offset:offset + TMEM_LD_N], Dreg[:])
                    if i == MMA_N // EPI_N - 1:
                        T.ptx.tcgen05.fence.before_thread_sync()
                        ld2mma.arrive(wg_id, remote=0)
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], Dreg_f16[:])
'''
    return prefix + replacement + end + suffix


def specialize_role(source, role):
    """Unroll the two role branches, keeping each warp/group on its own slot."""
    if role == "mma":
        start = "            elif warp_id < NUM_CONSUMER:\n"
        end = "        elif wg_id < NUM_CONSUMER:\n"
        index, consumer = "warp_id", "mma_consumer"
        replacement = ("            else:\n"
                       "                for mma_consumer in T.unroll(NUM_CONSUMER):\n"
                       "                    if warp_id == mma_consumer:\n")
    elif role == "writeback":
        start = "        elif wg_id < NUM_CONSUMER:\n"
        end = "\n        T.cuda.cluster_sync()\n"
        index, consumer = "wg_id", "wb_consumer"
        replacement = ("        else:\n"
                       "            for wb_consumer in T.unroll(NUM_CONSUMER):\n"
                       "                if wg_id == wb_consumer:\n")
    else:
        raise ValueError(f"unsupported consumer role: {role}")
    if source.count(start) != 1:
        raise ValueError(f"expected exactly one {role} role branch")
    prefix, tail = source.split(start)
    if tail.count(end) != 1:
        raise ValueError(f"expected exactly one end of {role} role branch")
    block, suffix = tail.split(end)
    # Whole identifiers only: warp_id inside the writeback remains a lane role.
    block = re.sub(rf"\b{index}\b", consumer, block)
    block = "".join("        " + line if line.strip() else line
                    for line in block.splitlines(keepends=True))
    return prefix + replacement + block + end + suffix


def current_l2_group(source, variant):
    """Change work-to-tile ordering only on the adopted two-slot narrow path."""
    required = (
        "    NARROW_N = M * N <= 4096 * 4096\n",
        "    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64\n",
        "    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)\n",
        "    NUM_CONSUMER = 2\n",
        "    PIPE_DEPTH = 4\n",
        "    EPI_N = 32 if NARROW_N else 64\n",
        "    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1\n",
        "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n",
        '            "ts", num_m_tiles=M // (MMA_M * NUM_CONSUMER), num_n_tiles=N // MMA_N,\n',
        "            l2_group_size=8, num_clusters=CLUSTER_COUNT)\n",
        "        def tma_load(k_st):\n            Tx.copy_async(Bsmem[",
    )
    if variant not in CURRENT_L2_VARIANTS or any(source.count(s) != 1 for s in required):
        raise ValueError("current L2 probes require the adopted Step 10 narrow/TMEM baseline")
    group = CURRENT_L2_VARIANTS[variant]
    return replace_once(source, required[9],
                        f"            l2_group_size={group} if TMEM_BUFFERS == 2 else 8, "
                        "num_clusters=CLUSTER_COUNT)\n")


def current_input_ring(source, variant):
    """Isolate input depth/K width on the adopted narrow, two-TMEM-slot path."""
    required = (
        "    NARROW_N = M * N <= 4096 * 4096\n",
        "    BLK_M, BLK_N, BLK_K = 128, (64 if NARROW_N else 128), 64\n",
        "    MMA_M, MMA_N = 256, (128 if NARROW_N else 256)\n",
        "    PIPE_DEPTH = 4\n",
        "    K_TILES = K // BLK_K\n",
        "    EPI_N = 32 if NARROW_N else 64\n",
        "    TMEM_BUFFERS = 2 if NARROW_N and TOTAL_TILES > MAX_CLUSTERS else 1\n",
        "    TMEM_SLOT_STRIDE = NUM_CONSUMER if TMEM_BUFFERS == 2 else 0\n",
        "        mma_tmem_base: T.let = tmem_addr[0]\n",
        "        def tma_load(k_st):\n            Tx.copy_async(Bsmem[",
    )
    if variant not in CURRENT_INPUT_VARIANTS or any(source.count(s) != 1 for s in required):
        raise ValueError("current input probes require the adopted Step 10 narrow/TMEM baseline")
    # Derive the experiment from the production slot decision. Other workloads
    # retain the measured K64/depth4 path, including large N256 matrices.
    source = replace_once(source, "    PIPE_DEPTH = 4\n", "")
    depth = 5 if variant == "tmem_input_depth5" else 2
    config = f"    PIPE_DEPTH = {depth} if TMEM_BUFFERS == 2 else 4\n"
    if variant == "tmem_k128_depth2":
        source = replace_once(source, required[1],
                              "    BLK_M, BLK_N = 128, (64 if NARROW_N else 128)\n")
        source = replace_once(source, "    K_TILES = K // BLK_K\n", "")
        config += ("    BLK_K = 128 if TMEM_BUFFERS == 2 and K % 128 == 0 else 64\n"
                   "    K_TILES = K // BLK_K\n")
    return replace_once(source, required[7], config + required[7])


def share_a_consumers(source, variant):
    """Transpose the consumer grid, retaining each 256x128 MMA and TMEM slot."""
    if variant not in SHARE_A_VARIANTS:
        raise ValueError(f"unsupported shared-A variant: {variant}")
    # This validates the production baseline and reproduces the measured
    # depth-five control before changing operand sharing.
    source = current_input_ring(source, "tmem_input_depth5")
    depth = 6 if variant == "tmem_share_a_depth6" else 5
    config = "    PIPE_DEPTH = 5 if TMEM_BUFFERS == 2 else 4\n"
    source = replace_once(source, config,
        "    SHARE_A = TMEM_BUFFERS == 2\n"
        "    A_CONSUMERS = 1 if SHARE_A else NUM_CONSUMER\n"
        "    B_CONSUMERS = NUM_CONSUMER if SHARE_A else 1\n"
        f"    PIPE_DEPTH = {depth} if SHARE_A else 4\n")
    # M and N alignment guarantees that transposing the consumer grid leaves
    # TOTAL_TILES and CLUSTER_COUNT unchanged, so both variants do equal work.
    replacements = (
        ("(PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K)",
         "(PIPE_DEPTH, A_CONSUMERS, BLK_M, BLK_K)"),
        ("(PIPE_DEPTH, BLK_N, BLK_K)",
         "(PIPE_DEPTH, B_CONSUMERS, BLK_N, BLK_K)"),
    )
    for before, after in replacements:
        if source.count(before) != 2:  # layout and allocation
            raise ValueError(f"shared-A probe expected layout and allocation: {before}")
        source = source.replace(before, after)
    source = replace_once(source,
        '"ts", num_m_tiles=M // (MMA_M * NUM_CONSUMER), num_n_tiles=N // MMA_N,',
        '"ts", num_m_tiles=M // (MMA_M * A_CONSUMERS), num_n_tiles=N // (MMA_N * B_CONSUMERS),')
    source = replace_once(source,
        "m_st = T.meta_var(tile_scheduler.m_idx * MMA_M * NUM_CONSUMER + cbx * BLK_M)",
        "m_st = T.meta_var(tile_scheduler.m_idx * MMA_M * A_CONSUMERS + cbx * BLK_M)")
    source = replace_once(source,
        "n_st = T.meta_var(tile_scheduler.n_idx * MMA_N + cbx * BLK_N)",
        "n_st = T.meta_var(tile_scheduler.n_idx * MMA_N * B_CONSUMERS + cbx * BLK_N)")
    source = replace_once(source, "n_out = T.meta_var(tile_scheduler.n_idx * MMA_N)",
                          "n_out = T.meta_var(tile_scheduler.n_idx * MMA_N * B_CONSUMERS)")
    old_load = ("            Tx.copy_async(Bsmem[tma_phase.stage, :, :],\n"
                "                          B[n_st:n_st + BLK_N, k_st:k_st + BLK_K],\n"
                '                          dispatch="tma_auto", cta_group=CTA_GROUP,\n'
                "                          mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))\n"
                "            for consumer in T.unroll(NUM_CONSUMER):\n")
    new_load = ("            for consumer in T.unroll(B_CONSUMERS):\n"
                "                n_consumer = T.meta_var(n_st + consumer * MMA_N)\n"
                "                Tx.copy_async(Bsmem[tma_phase.stage, consumer, :, :],\n"
                "                              B[n_consumer:n_consumer + BLK_N, k_st:k_st + BLK_K],\n"
                '                              dispatch="tma_auto", cta_group=CTA_GROUP,\n'
                "                              mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))\n"
                "            for consumer in T.unroll(A_CONSUMERS):\n")
    source = replace_once(source, old_load, new_load)
    source = replace_once(source,
        "CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE",
        "CTA_GROUP * (A_CONSUMERS * BLK_M * BLK_K + B_CONSUMERS * BLK_N * BLK_K) * F16_SIZE")
    source = replace_once(source,
        "Asmem[mma_phase.stage, warp_id, :, :], Bsmem[mma_phase.stage, :, :],",
        "Asmem[mma_phase.stage, 0 if SHARE_A else warp_id, :, :], "
        "Bsmem[mma_phase.stage, warp_id if SHARE_A else 0, :, :],")
    source = replace_once(source,
        "            m_out = T.meta_var(m_st + wg_id * MMA_M)\n",
        "            m_out = T.meta_var(m_st + (0 if SHARE_A else wg_id) * MMA_M)\n"
        "            n_consumer_out = T.meta_var(n_out + (wg_id if SHARE_A else 0) * MMA_N)\n")
    source = replace_once(source,
        "D[m_out:m_out + BLK_M, n_out + col:n_out + col + EPI_N]",
        "D[m_out:m_out + BLK_M, n_consumer_out + col:n_consumer_out + col + EPI_N]")
    return source


def variant_builder_source(source, step, variant):
    """Keep each variant independent; refuse an unexpected production builder."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
    if variant == "cluster_cache_tmem_base":
        if "mma_tmem_base: T.let" not in source:
            required = ("    CTA_GROUP = 2\n", "    PIPE_DEPTH = 4\n",
                        "        T.cuda.cta_sync()\n        T.cuda.cluster_sync()\n"
                        "        tmem = T.decl_buffer(")
            if any(source.count(part) != 1 for part in required):
                raise ValueError("Step 9 TMEM cache requires the four-stage cluster baseline "
                                 "with the allocation published before the snapshot")
        # Reuse the measured cache transform; the unique Step 9 name keeps
        # boundary shapes and controls distinct from historical Step 8/10.
        variant = "cache_tmem_base"
    if step == 10 and variant in CURRENT_INPUT_VARIANTS:
        return current_input_ring(source, variant)
    if step == 10 and variant in SHARE_A_VARIANTS:
        return share_a_consumers(source, variant)
    if step == 10 and variant in CURRENT_L2_VARIANTS:
        return current_l2_group(source, variant)
    if step == 10 and variant != "baseline" and "    NARROW_N =" in source:
        raise ValueError("Step 10 has adopted workload-based narrow N and TMEM buffering; "
                         "validate with benchmark.py --steps 10 and tests/test_step10.py. "
                         "Replay historical probes at their recorded commit (0f23484 for TMEM comparisons).")
    if variant == "n128_tmem_double_buffer":
        return double_buffer_tmem(variant_builder_source(source, step, "n128_epi32"))
    if variant in ("mma_batch", "mma_batch_no_unroll", "mma_unroll4", "mma_batch_unroll4"):
        for required in ("mma_tmem_base: T.let", "TILES_PER_CLUSTER =",
                         "    BLK_M, BLK_N, BLK_K = 128, 128, 64\n",
                         "    MMA_M, MMA_N = 256, 256\n",
                         "    PIPE_DEPTH = 4\n", "    EPI_N = 64\n",
                         "        def tma_load(k_st):\n            Tx.copy_async(Bsmem["):
            if required not in source:
                raise ValueError("MMA batch experiments require the adopted Step 10 K64 baseline")
        return source  # Only CUDA emission changes; TIR and the builder stay identical.
    if variant in ("epilogue_32", "split_tma"):
        loader = "        def tma_load(k_st):\n            Tx.copy_async(Bsmem["
        if ("mma_tmem_base: T.let" not in source or "TILES_PER_CLUSTER =" not in source
                or loader not in source or "    MMA_M, MMA_N = 256, 256\n" not in source):
            raise ValueError("wide N experiments require the adopted Step 10 B-first baseline")
        if variant == "split_tma":
            return split_tma_producers(source)
        source = replace_once(source, "    EPI_N = 64\n", "    EPI_N = 32\n")
        return replace_once(source,
            "    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))\n",
            "    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_64B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))\n")
    if variant in NARROW_N_VARIANTS:
        loader = "        def tma_load(k_st):\n            Tx.copy_async(Bsmem["
        if ("mma_tmem_base: T.let" not in source or "TILES_PER_CLUSTER =" not in source
                or loader not in source):
            raise ValueError("narrow N experiments require the adopted Step 10 B-first baseline")
        # Each CTA loads half of the cluster's N dimension. Retain two MMA
        # consumers, TMEM allocation, and all barrier/phase protocols.
        source = replace_once(source, "    BLK_M, BLK_N, BLK_K = 128, 128, 64\n",
                              "    BLK_M, BLK_N, BLK_K = 128, 64, 64\n")
        source = replace_once(source, "    MMA_M, MMA_N = 256, 256\n",
                              "    MMA_M, MMA_N = 256, 128\n")
        if variant != "n_tile_128":
            source = replace_once(source, "    EPI_N = 64\n", "    EPI_N = 32\n")
            # A 32-half output row needs a 64-byte swizzle atom; the old
            # 128-byte atom requires at least 64 halves. Input layouts stay.
            source = replace_once(source,
                "    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))\n",
                "    D_layout = mma_shared_layout(d_type, SwizzleMode.SWIZZLE_64B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))\n")
        if variant == "n128_epi32_depth5":
            source = replace_once(source, "    PIPE_DEPTH = 4\n", "    PIPE_DEPTH = 5\n")
        return source
    if variant in ("role_registers", "tma_b_first"):
        if "mma_tmem_base: T.let" not in source or "TILES_PER_CLUSTER =" not in source:
            raise ValueError("role experiments require the adopted Step 10 cache/grid baseline")
        transform = repartition_role_registers if variant == "role_registers" else load_shared_b_first
        return transform(source)
    if variant in ("epilogue_depth3", "epilogue_double_buffer"):
        if "mma_tmem_base: T.let" not in source or "TILES_PER_CLUSTER =" not in source:
            raise ValueError("epilogue experiments require the adopted Step 10 cache/grid baseline")
        source = replace_once(source, "    PIPE_DEPTH = 4\n", "    PIPE_DEPTH = 3\n")
        return double_buffer_epilogue(source) if variant == "epilogue_double_buffer" else source
    if variant in ("warp_release", "paired_tmem_loads"):
        if "mma_tmem_base: T.let" not in source or "TILES_PER_CLUSTER =" not in source:
            raise ValueError("writeback experiments require the adopted Step 10 cache/grid baseline")
        transform = aggregate_tmem_release if variant == "warp_release" else pair_tmem_loads
        return transform(source)
    if variant == "balanced_fused_a":
        source = variant_builder_source(source, step, "cache_balanced_clusters")
        return fuse_consumer_a_loads(source)
    if variant in DEPTH3_VARIANTS:
        source = variant_builder_source(source, step, "cache_balanced_clusters")
        source = replace_once(source, "    PIPE_DEPTH = 4\n", "    PIPE_DEPTH = 3\n")
        if variant == "balanced_depth3_epi128":
            source = replace_once(source, "    EPI_N = 64\n", "    EPI_N = 128\n")
        return source
    if variant in CACHE_EXPERIMENTS:
        cached = variant_builder_source(source, step, "cache_tmem_base")
        extra = CACHE_EXPERIMENTS[variant]
        return variant_builder_source(cached, step, extra) if extra else cached
    if variant == "tma_wait_64ns" and "tirx_tma_wait_64ns" in source:
        raise ValueError(f"Step {step} has adopted tma_wait_64ns; validate the production kernel "
                         f"with benchmark.py --steps {step} and tests/test_step{step:02d}.py")
    if variant in ("baseline", "mma_wait_64ns", "tma_wait_64ns", "reuse_wait_64ns", "ring_wait_64ns"):
        return source
    if variant == "cache_tmem_base":
        if "mma_tmem_base: T.let" in source:
            raise ValueError(f"Step {step} has adopted cache_tmem_base; validate the production kernel "
                             f"with benchmark.py --steps {step} and tests/test_step{step:02d}.py")
        # The allocation result is published by the existing CTA/cluster sync.
        # No role changes it until final deallocation. A let snapshots it once,
        # so compiler memory clobbers need not reload SMEM in every MMA stage.
        source = replace_once(source, "        tmem = T.decl_buffer(",
                              "        mma_tmem_base: T.let = tmem_addr[0]\n        tmem = T.decl_buffer(")
        return replace_once(source, "allocated_addr=tmem_addr[0]", "allocated_addr=mma_tmem_base")
    if variant == "unroll_ring":
        return unroll_pipeline_ring(source)
    if variant == "stream_epilogue":
        return stream_epilogue(source)
    if variant in ("pipe_depth_2", "k128_depth_2"):
        source = replace_once(source, "    PIPE_DEPTH = 4\n", "    PIPE_DEPTH = 2\n")
        if variant == "k128_depth_2":
            source = replace_once(source, "BLK_M, BLK_N, BLK_K = 128, 128, 64",
                                  "BLK_M, BLK_N, BLK_K = 128, 128, (128 if K % 128 == 0 else 64)")
        return source
    if variant in ("specialize_mma", "specialize_writeback"):
        return specialize_role(source, variant.removeprefix("specialize_"))
    if variant == "k_tile_128":
        if "BLK_K = 128 if K % 128 == 0 else 64" in source:
            raise ValueError(f"Step {step} has adopted k_tile_128; validate the production kernel "
                             f"with benchmark.py --steps {step} and tests/test_step{step:02d}.py")
        return replace_once(source, "BLK_M, BLK_N, BLK_K = 128, 128, 64",
                            "BLK_M, BLK_N, BLK_K = 128, 128, (128 if K % 128 == 0 else 64)")
    if variant == "epilogue_128":
        return replace_once(source, "    EPI_N = 64\n", "    EPI_N = 128\n")
    if variant == "tmem_load_16":
        return replace_once(source, "    TMEM_LD_N = 32", "    TMEM_LD_N = 16")
    if variant == "tmem_load_64":
        return replace_once(source, "    TMEM_LD_N = 32", "    TMEM_LD_N = 64")
    if variant == "l2_group_4":
        return replace_once(source, "l2_group_size=8, num_clusters=CLUSTER_COUNT",
                            "l2_group_size=4, num_clusters=CLUSTER_COUNT")
    if variant == "balanced_clusters":
        if "TILES_PER_CLUSTER =" in source:
            raise ValueError(f"Step {step} has adopted balanced_clusters; validate the production kernel "
                             f"with benchmark.py --steps {step} and tests/test_step{step:02d}.py")
        # Keep the original maximum number of tiles per cluster while using
        # the smallest grid that can cover them. At 4096: 128 tiles / 64
        # clusters = two each, versus 54 clusters with two and 20 with one.
        before = "    CLUSTER_COUNT = min(SM_COUNT // CTA_GROUP, (M // (MMA_M * NUM_CONSUMER)) * (N // MMA_N))"
        after = ("    TOTAL_TILES = (M // (MMA_M * NUM_CONSUMER)) * (N // MMA_N)\n"
                 "    MAX_CLUSTERS = SM_COUNT // CTA_GROUP\n"
                 "    TILES_PER_CLUSTER = (TOTAL_TILES + MAX_CLUSTERS - 1) // MAX_CLUSTERS\n"
                 "    CLUSTER_COUNT = (TOTAL_TILES + TILES_PER_CLUSTER - 1) // TILES_PER_CLUSTER")
        return replace_once(source, before, after)
    # Step 6's elected thread uses one MMA pipeline; the writeback handoff is
    # after the K loop. Keep per-tile completion and TMA acquire fences, moving
    # only this after/before pair to the handoff (as in measured Steps 4/5).
    pair = ("                        T.ptx.tcgen05.fence.after_thread_sync()\n"
            "                        T.ptx.tcgen05.fence.before_thread_sync()\n")
    source = replace_once(source, pair, "")
    anchor = "                            tma_load(stage, (k + PIPE_DEPTH) * BLK_K)\n"
    return replace_once(source, anchor, anchor + pair.replace("                        ", "                    "))


def build_variant(step, shape, variant, directory):
    import gemm_kernels

    builder = getattr(gemm_kernels, f"hgemm_v{step}")
    before = inspect.getsource(builder)
    after = variant_builder_source(before, step, variant)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "builder.before.py").write_text(before)
    path = directory / "builder.py"
    path.write_text(after)
    (directory / "builder.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="baseline.py", tofile=f"{variant}.py",
    )))
    write_json(directory / "builder.json", dict(
        before_sha256=hashlib.sha256(before.encode()).hexdigest(),
        compiled_sha256=hashlib.sha256(after.encode()).hexdigest(),
    ))
    if before != after:
        # A separate namespace and real source file support TVMScript inspection
        # while leaving every production builder and TVM's globals unchanged.
        namespace = dict(vars(gemm_kernels))
        exec(compile(after, str(path.resolve()), "exec"), namespace)
        builder = namespace[f"hgemm_v{step}"]
    return builder(*shape)


def batch_mma_stage(source):
    """Reuse two SMEM descriptors in one PTX block, retaining four K16 MMAs."""
    helper_name = "tvm_probe_mma_batch_k64"
    marker = 'extern "C" __global__'
    if helper_name in source:
        raise ValueError("MMA batch experiment is already applied")
    if marker not in source:
        raise ValueError("CUDA kernel declaration was not found")
    header, body = source.split(marker, 1)
    op = "ptx_tcgen05_mma_cta_2_kind_f16_SS"
    original_helper = re.search(rf"^__forceinline__ __device__ void {op}\([^\n]+\) \{{\n.*?^\}}\n",
                                header, re.M | re.S)
    expected_ptx = ('{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n'
                    'tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, '
                    '{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n}\n')
    if (original_helper is None or 'asm volatile(' not in original_helper[0]
            or ''.join(json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"',
                       original_helper[0].split('\n        :', 1)[0])) != expected_ptx):
        raise ValueError("unexpected original MMA PTX helper")
    calls = list(re.finditer(rf"^( +){op}\((.*)\);$", body, re.M))
    if len(calls) != 4:
        raise ValueError("MMA batch requires exactly four K16 instructions")
    # These generated arguments contain nested parentheses but no comma
    # operators or quoted strings. Split only at the top call level.
    def args(text):
        result, depth, start = [], 0, 0
        for i, char in enumerate(text):
            depth += (char == '(') - (char == ')')
            if char == ',' and depth == 0:
                result.append(text[start:i].strip())
                start = i + 1
            if depth < 0:
                raise ValueError("unbalanced MMA arguments")
        if depth:
            raise ValueError("unbalanced MMA arguments")
        return result + [text[start:].strip()]
    operands = [args(c[2]) for c in calls]
    first = operands[0]
    if (len(first) != 13 or first[3] != "(uint)272629776"
            or first[4] not in ("(0 < k_1)", "(bool)0") or first[5:] != ["0"] * 8):
        raise ValueError("unexpected MMA descriptor, accumulation, or masks")
    offsets = ("((mma_phase_stage_ptr[0] * 2048) + ((warp_id_in_cta & 3) * 1024))",
               "(mma_phase_stage_ptr[0] * 1024)")
    for i, actual in enumerate(operands):
        expected = first.copy()
        expected[4] = first[4] if i == 0 else "(bool)1"
        for axis, offset in enumerate(offsets):
            if i:
                offset = f"({offset} + {2 * i})"
            expected[axis + 1] = f"tvm_builtin_smem_desc_add_16B_offset(desc{'AB'[axis]}_ptr[0], {offset})"
        if actual != expected:
            raise ValueError("MMA batch requires unchanged operands and offsets 0/2/4/6")
    between = body[calls[0].start():calls[-1].end()]
    if between != "\n".join(c[0] for c in calls):
        raise ValueError("MMA batch cannot move intervening instructions")
    # Match TVM's descriptor helper: add to the low 32 bits without carrying
    # into the upper descriptor fields. A/B advance by 32 bytes per K16.
    ptx = ["{", ".reg .b64 a, b;", ".reg .b32 alo, ahi, blo, bhi, z;",
           ".reg .pred p;", "mov.b64 {alo, ahi}, %1;", "mov.b64 {blo, bhi}, %2;",
           "mov.b64 a, %1;", "mov.b64 b, %2;", "mov.u32 z, 0;",
           "setp.ne.b32 p, %4, 0;"]
    for i in range(4):
        if i:
            ptx.extend(["add.u32 alo, alo, 2;", "add.u32 blo, blo, 2;",
                        "mov.b64 a, {alo, ahi};", "mov.b64 b, {blo, bhi};"])
        ptx.append("tcgen05.mma.cta_group::2.kind::f16 [%0], a, b, %3, {z, z, z, z, z, z, z, z}, p;")
        if i == 0:
            ptx.append("setp.ne.b32 p, 1, 0;")
    ptx.append("}")
    helper = (f"__forceinline__ __device__ void {helper_name}(uint32_t d, uint64_t a, uint64_t b, "
              "uint32_t desc, uint32_t accum) {\n    asm volatile(\n"
              + "".join(f'        "{line}\\n"\n' for line in ptx)
              + '        :\n        : "r"(d), "l"(a), "l"(b), "r"(desc), "r"(accum)\n    );\n}\n\n')
    replacement = calls[0][1] + helper_name + "(" + ", ".join(first[:5]) + ");"
    body = body[:calls[0].start()] + replacement + body[calls[-1].end():]
    return header + helper + marker + body


def variant_source(source, step, variant):
    """Apply isolated CUDA wait, MMA emission, or loop-pragma experiments."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
    if variant in ("mma_unroll4", "mma_batch_unroll4"):
        batched = batch_mma_stage(source)  # Validate the same K64 operands in both controls.
        if variant == "mma_batch_unroll4":
            source = batched
        # Reuse the existing guarded MMA-only pragma insertion. Keeping the
        # original and batch controls at the same factor isolates emission.
        loop = re.search(r"for \(int k_1 =", source)
        if loop is None:
            # The K=64 loop was removed by TVM; only the known single-stage
            # accumulation pattern may treat the pragma as a no-op.
            if not re.search(r"tvm_probe_mma_batch_k64\([^\n]+, \(bool\)0\);", batched):
                raise ValueError("missing MMA K loop without a single-stage fallback")
            return source
        changed = variant_source(source, step, "cache_mma_no_unroll")
        return re.sub(r"#pragma unroll 1\n(?= +for \(int k_1 =)",
                      "#pragma unroll 4\n", changed)
    if variant in ("mma_batch", "mma_batch_no_unroll"):
        source = batch_mma_stage(source)
        if variant == "mma_batch":
            return source
        # With K=64, TVM removes the one-iteration loop entirely.
        if not re.search(r"for \(int k_1 =", source):
            if not re.search(r"tvm_probe_mma_batch_k64\([^\n]+, \(bool\)0\);", source):
                raise ValueError("missing MMA K loop without a single-stage fallback")
            return source
        return variant_source(source, step, "cache_mma_no_unroll")
    if variant == "cache_mma_no_unroll":
        # The measured cache-only cubin unrolled the MMA K loop four times.
        # Isolate that compiler choice; leave the producer loop and every
        # barrier/descriptor/MMA operand unchanged. Fail on source drift.
        if source.count("uint mma_tmem_base = ((uint*)pool_buf_ptr)[0];") != 1:
            raise ValueError("MMA unroll experiment requires the cached TMEM base")
        loop = re.compile(r"^( +)(for \(int k_1 = 0; k_1 < \d+; \+\+k_1\) \{)$", re.M)
        matches = list(loop.finditer(source))
        if len(matches) != 1:
            raise ValueError("expected exactly one MMA K loop; use a larger K")
        match = matches[0]
        if re.search(r"#pragma unroll(?: \d+)?$", source[:match.start()].rstrip()):
            raise ValueError("MMA unroll experiment is already applied")
        if not source[match.end():].lstrip().startswith("tvm_builtin_ptx_mbarrier_try_wait("):
            raise ValueError("unexpected MMA K loop entry")
        return loop.sub(lambda m: f"{m[1]}#pragma unroll 1\n{m[0]}", source)
    if variant not in ("mma_wait_64ns", "tma_wait_64ns", "reuse_wait_64ns", "ring_wait_64ns"):
        return source
    marker = 'extern "C" __global__'
    if marker not in source:
        raise ValueError("CUDA kernel declaration was not found")
    header, body = source.split(marker, 1)
    name = "tvm_builtin_ptx_mbarrier_try_wait"
    short_name = f"tvm_probe_{variant}"
    if short_name in source:
        raise ValueError("wait experiment is already applied")
    helpers = re.findall(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}\n",
                         header, re.M | re.S)
    if len(helpers) != 1:
        raise ValueError("expected exactly one default wait helper")
    helper = helpers[0]
    required = ("unsigned int ticks = 0x989680;",
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;",
                ':: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory"')
    if any(helper.count(part) != 1 for part in required):
        raise ValueError("unexpected default wait implementation")
    # Exact barrier/phase arguments from the saved production CUDA. The paired
    # ring experiment builds on the isolated reuse/data-ready controls; it
    # leaves accumulator-free and writeback-completion waits unchanged.
    if variant in ("reuse_wait_64ns", "ring_wait_64ns"):
        arguments = ["(&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 5)])), (tma_phase_phase_ptr[0] ^ 0)"]
        if variant == "ring_wait_64ns":
            arguments.append("(&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 1)])), (mma_phase_phase_ptr[0] ^ 0)")
    elif variant == "tma_wait_64ns":
        # Both four-stage kernels use slots 1..4 for tma2mma. This wait is
        # executed by the MMA warp(s); completion/accumulator-free waits stay.
        arguments = ["(&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 1)])), (mma_phase_phase_ptr[0] ^ 0)"]
    else:
        arguments = {
            6: ["(&(((uint64_t*)pool_buf_ptr)[3])), phase_mma_ptr[0]"],
            7: ["(&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 3)])), (tma_phase_phase_ptr[0] ^ 0)",
                "(&(((uint64_t*)pool_buf_ptr)[5])), (wb_phase_phase_ptr[0] ^ 0)"],
            10: ["(&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 5)])), (tma_phase_phase_ptr[0] ^ 0)",
                 "(&(((uint64_t*)pool_buf_ptr)[((warp_id_in_cta >> 2) + 9)])), (wb_phase_phase_ptr[0] ^ 0)"],
        }[step]
    expected = 2 if step == 6 else 3 if step == 8 and "tirx_tma_wait_64ns(" in body else 4
    if body.count(name + "(") != expected:
        raise ValueError("unexpected TMA/MMA wait count")
    for args in arguments:
        body = replace_once(body, f"{name}({args});", f"{short_name}({args});")
    shorter = helper.replace(name, short_name).replace(required[0], "unsigned int ticks = 64;")
    return header + shorter + "\n" + marker + body


def summarize_with_cache_control(cases, reference_times, tolerance):
    """Keep original grading and add paired comparisons with the cache control."""
    rows = summarize(cases, reference_times, tolerance)
    controls = {(c["step"], c["size"]): c["samples_ms"] for c in cases
                if c["variant"] == "cache_tmem_base"}
    by_variant = {(c["step"], c["size"], c["variant"]): c["samples_ms"] for c in cases}
    for case, row in zip(cases, rows):
        control = controls.get((case["step"], case["size"]))
        if control is not None:
            if len(control) != len(case["samples_ms"]):
                raise ValueError("cache control and variant need matching trials")
            row["paired_cache_speedup"] = statistics.median(
                base / sample for base, sample in zip(control, case["samples_ms"]))
        else:
            row["paired_cache_speedup"] = None
        control_name = EXPERIMENT_CONTROLS.get(case["variant"], "baseline")
        direct = by_variant.get((case["step"], case["size"], control_name))
        if direct is None or len(direct) != len(case["samples_ms"]):
            raise ValueError(f"missing or incomplete comparison control: {control_name}")
        row["comparison_control"] = control_name
        row["paired_control_speedup"] = statistics.median(
            base / sample for base, sample in zip(direct, case["samples_ms"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory, must not exist")
    parser.add_argument("--steps", type=int, nargs="+", choices=STEP_VARIANTS, default=[10])
    parser.add_argument("--size", type=int, choices=(1024, 2048, 4096, 8192), default=4096)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        help="optional subset, must apply to every selected step; baseline always included")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.trials < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("trials/repeat must be positive; warmup must be nonnegative")
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    selected = {}
    for step in dict.fromkeys(args.steps):
        try:
            selected[step] = select_variants(step, args.variants)
        except ValueError as error:
            parser.error(str(error))

    import torch
    import tvm
    import gemm_kernels
    from utils import REFERENCE_TIMES, TIMING_TOLERANCE, blackwell_target, prepare_data, time_cuda_call, verify

    if not torch.cuda.is_available():
        parser.error("a Blackwell GPU and CUDA-enabled PyTorch are required")
    target = blackwell_target()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    cluster_steps = sorted(set(selected) & {9, 10})
    if cluster_steps and device.multi_processor_count % 2:
        parser.error(f"Step {cluster_steps[0]} requires an even SM count for its two-CTA clusters")
    gemm_kernels.SM_COUNT = device.multi_processor_count
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True)
    metadata = dict(run_metadata(), target=str(target), gpu=device.name,
                    sm_count=device.multi_processor_count, tvm=tvm.__version__,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    probe_step45_sha256=hashlib.sha256(Path(__file__).with_name("probe_step45.py").read_bytes()).hexdigest(),
                    profile_persistent_sha256=hashlib.sha256(Path(__file__).with_name("profile_persistent.py").read_bytes()).hexdigest(),
                    steps=list(selected), size=args.size, variants=selected,
                    verification_shapes={v: VERIFICATION_SHAPES[v] for variants in selected.values()
                                         for v in variants if v in VERIFICATION_SHAPES},
                    comparison_controls={str(step): {v: EXPERIMENT_CONTROLS[v] for v in variants
                                                    if v in EXPERIMENT_CONTROLS}
                                         for step, variants in selected.items()},
                    trials=args.trials, warmup=args.warmup, repeat=args.repeat, seed=args.seed)
    write_json(args.output / "run.json", metadata)
    print(f"Code: {metadata['git_revision']}; gemm_sha256={metadata['gemm_kernels_sha256']}", flush=True)
    print(f"GPU: {device.name}; SMs: {device.multi_processor_count}; {target}", flush=True)
    print("Separate builds; each experiment records its direct comparison control. "
          "Verify all outputs before interleaved CUDA-event timing.", flush=True)
    A, B, _ = prepare_data(args.size, args.size, args.size)
    cases, executables, outputs = [], [], []
    for step, variants in selected.items():
        for variant in variants:
            directory = args.output / f"step{step:02d}_{args.size}_{variant}"
            kernel = build_variant(step, (args.size,) * 3, variant, directory)
            output = torch.full((args.size, args.size), float("nan"), dtype=A.dtype, device=A.device)
            with target:
                with capture_compilation(directory), check_role_register_budget(variant, directory), source_experiment(
                    step, variant, directory, transform=variant_source
                ) as calls:
                    executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
                    executable.mod(A, B, output)
                if len(calls) != 1:
                    raise RuntimeError(f"{directory}: expected one CUDA compilation, got {len(calls)}")
                verify(output, A, B)
            # SASS permits checking whether a source-level hoist survived
            # NVRTC. Disassembly is outside all timed launches.
            dump_sass(directory)
            print(f"Verified step {step} / {variant}", flush=True)
            cases.append(dict(step=step, size=args.size, variant=variant, samples_ms=[]))
            executables.append(executable)
            outputs.append(output)

    # Pipeline, TMA-layout and writeback experiments need boundary/reuse validation.
    # Run it before any scored timing with the same numerical tolerances.
    for step, variants in selected.items():
        for variant in variants:
            if variant not in VERIFICATION_SHAPES:
                continue
            for shape in VERIFICATION_SHAPES[variant]:
                directory = args.output / "verification" / f"{variant}_{'_'.join(map(str, shape))}"
                kernel = build_variant(step, shape, variant, directory)
                va, vb, _ = prepare_data(*shape)
                output = torch.full((shape[0], shape[1]), float("nan"), dtype=va.dtype, device=va.device)
                with target:
                    with capture_compilation(directory), check_role_register_budget(variant, directory), source_experiment(
                        step, variant, directory, transform=variant_source
                    ) as calls:
                        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
                        executable.mod(va, vb, output)
                    if len(calls) != 1:
                        raise RuntimeError(f"{directory}: expected one CUDA compilation, got {len(calls)}")
                    verify(output, va, vb)
                    output.fill_(float("nan"))
                    executable.mod(va, vb, output)
                    verify(output, va, vb)
                write_json(directory / "verification.json", dict(step=step, variant=variant,
                           shape=shape, launches=2, verified=True, timed=False))
                print(f"Verified boundary Step {step} / {variant} / {shape} twice (untimed)", flush=True)

    orders = []
    for trial in range(args.trials):
        order = trial_order(len(cases), trial)
        orders.append(order)
        for index in order:
            with target:
                elapsed = time_cuda_call(lambda: executables[index].mod(A, B, outputs[index]),
                                         args.warmup, args.repeat)
            cases[index]["samples_ms"].append(elapsed)
            verify(outputs[index], A, B)
            print(f"Trial {trial + 1}: step {cases[index]['step']} / "
                  f"{cases[index]['variant']}: {elapsed:.6f} ms", flush=True)
        write_json(args.output / "samples.json", dict(orders=orders, cases=cases))
    rows = summarize_with_cache_control(cases, REFERENCE_TIMES, TIMING_TOLERANCE)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("step variant         median_ms    min_ms    max_ms paired_speedup vs_cache control                   vs_control limit_ms status")
    for row in rows:
        cache_ratio = row["paired_cache_speedup"]
        cache_text = "n/a" if cache_ratio is None else f"{cache_ratio:.3f}"
        print(f"{row['step']:>4} {row['variant']:<15} {row['median_ms']:9.6f} "
              f"{row['min_ms']:9.6f} {row['max_ms']:9.6f} {row['paired_speedup']:14.3f} "
              f"{cache_text:>8} {row['comparison_control']:<25} {row['paired_control_speedup']:10.3f} "
              f"{row['limit_ms']:8.6f} {row['status']}")
    print(f"Saved {args.output / 'summary.csv'}; SLOW is a measured result, not a tool error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
