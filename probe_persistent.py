"""Independent B300 performance experiments for persistent GEMM kernels.

Step 10 production is correct but still marginal at 4096. The default compares
its measured cache/grid baseline with two independent writeback experiments:
warp-aggregated TMEM release and paired x32 TMEM reads. Adopted transforms
refuse reapplication. GPU verification precedes every scored experiment.
No production kernel is edited by this tool.
All variants must verify before interleaved timing with the original CUDA-event
timer. SLOW is a measured result; numerical or compilation errors stop the run.
"""

import argparse
import csv
import difflib
import hashlib
import inspect
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
    10: ("baseline", "mma_wait_64ns", "tmem_load_16", "tmem_load_64",
         "l2_group_4", "balanced_clusters", "tma_wait_64ns",
         "specialize_mma", "specialize_writeback", "unroll_ring",
         "pipe_depth_2", "k128_depth_2", "stream_epilogue",
         "cache_tmem_base", "reuse_wait_64ns", "ring_wait_64ns",
         "cache_unroll_ring", "cache_mma_no_unroll", "cache_balanced_clusters",
         "balanced_depth3", "balanced_depth3_epi128", "balanced_fused_a",
         "warp_release", "paired_tmem_loads"),
}
DEFAULT_STEP_VARIANTS = {6: ("baseline",), 7: ("baseline",),
                         8: ("baseline",),
                         10: ("baseline", "warp_release", "paired_tmem_loads")}
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
                       "balanced_depth3": "cache_balanced_clusters",
                       "balanced_depth3_epi128": "balanced_depth3",
                       "balanced_fused_a": "cache_balanced_clusters"}
DEPTH3_VARIANTS = ("balanced_depth3", "balanced_depth3_epi128")
# K=1/3/5 stages covers shorter-than-ring, odd complete ring, and partial
# ring; 96 cluster tiles force reuse of both consumers on the balanced grid.
DEPTH3_VERIFY_SHAPES = ((4096, 3072, 64), (4096, 3072, 192), (4096, 3072, 320))
VERIFICATION_SHAPES = {
    **{variant: DEPTH3_VERIFY_SHAPES for variant in DEPTH3_VARIANTS},
    "balanced_fused_a": ((4096, 3072, 64), (4096, 3072, 320)),
    "warp_release": ((4096, 3072, 64), (4096, 3072, 320)),
    "paired_tmem_loads": ((4096, 3072, 64), (4096, 3072, 320)),
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


def variant_builder_source(source, step, variant):
    """Keep each variant independent; refuse an unexpected production builder."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
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


def variant_source(source, step, variant):
    """Apply isolated CUDA wait hints or the cache-control MMA loop pragma."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
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
        if source[:match.start()].rstrip().endswith("#pragma unroll 1"):
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
    if 10 in selected and device.multi_processor_count % 2:
        parser.error("Step 10 requires an even SM count for its two-CTA clusters")
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
                with capture_compilation(directory), source_experiment(
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
                    with capture_compilation(directory), source_experiment(
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
