"""Independent B300 performance experiments for persistent GEMM kernels.

Defaults to Step 10 at 4096: compare stage addressing, K-stage granularity,
and streamed writeback. The two-stage K64 control isolates the wider K tile.
Steps 6/7 have passed validation; Step 8 has adopted its measured TMA wait.
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
from pathlib import Path

from benchmark_diagnostics import capture_compilation, run_metadata, write_json
from probe_step45 import replace_once, source_experiment, summarize, trial_order


STEP_VARIANTS = {
    6: ("baseline", "k_tile_128", "mma_wait_64ns", "final_fence"),
    7: ("baseline", "k_tile_128", "mma_wait_64ns", "epilogue_128"),
    8: ("baseline", "tma_wait_64ns", "epilogue_128"),
    10: ("baseline", "mma_wait_64ns", "tmem_load_16", "tmem_load_64",
         "l2_group_4", "balanced_clusters", "tma_wait_64ns",
         "specialize_mma", "specialize_writeback", "unroll_ring",
         "pipe_depth_2", "k128_depth_2", "stream_epilogue"),
}
DEFAULT_STEP_VARIANTS = {6: ("baseline",), 7: ("baseline",),
                         8: ("baseline",),
                         10: ("baseline", "unroll_ring", "pipe_depth_2",
                              "k128_depth_2", "stream_epilogue")}
VARIANTS = tuple(dict.fromkeys(v for variants in STEP_VARIANTS.values() for v in variants))


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
    if variant == "tma_wait_64ns" and "tirx_tma_wait_64ns" in source:
        raise ValueError(f"Step {step} has adopted tma_wait_64ns; validate the production kernel "
                         f"with benchmark.py --steps {step} and tests/test_step{step:02d}.py")
    if variant in ("baseline", "mma_wait_64ns", "tma_wait_64ns"):
        return source
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
    """Change selected barrier waits only, retaining acquire, parity, and retry."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
    if variant not in ("mma_wait_64ns", "tma_wait_64ns"):
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
    # Exact barrier/phase arguments from the saved production CUDA. Change
    # data-ready OR MMA-completion waits, never both in the same experiment.
    if variant == "tma_wait_64ns":
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
    if body.count(name + "(") != (2 if step == 6 else 4):
        raise ValueError("unexpected TMA/MMA wait count")
    for args in arguments:
        body = replace_once(body, f"{name}({args});", f"{short_name}({args});")
    shorter = helper.replace(name, short_name).replace(required[0], "unsigned int ticks = 64;")
    return header + shorter + "\n" + marker + body


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
        variants = list(dict.fromkeys(("baseline", *(args.variants or DEFAULT_STEP_VARIANTS[step]))))
        if any(v not in STEP_VARIANTS[step] for v in variants):
            parser.error(f"Step {step} supports: {', '.join(STEP_VARIANTS[step])}")
        selected[step] = variants

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
                    steps=list(selected), size=args.size, variants=selected,
                    trials=args.trials, warmup=args.warmup, repeat=args.repeat, seed=args.seed)
    write_json(args.output / "run.json", metadata)
    print(f"Code: {metadata['git_revision']}; gemm_sha256={metadata['gemm_kernels_sha256']}", flush=True)
    print(f"GPU: {device.name}; SMs: {device.multi_processor_count}; {target}", flush=True)
    print("Independent variants; verify all outputs before interleaved CUDA-event timing.", flush=True)
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
            print(f"Verified step {step} / {variant}", flush=True)
            cases.append(dict(step=step, size=args.size, variant=variant, samples_ms=[]))
            executables.append(executable)
            outputs.append(output)

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
    rows = summarize(cases, REFERENCE_TIMES, TIMING_TOLERANCE)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("step variant         median_ms    min_ms    max_ms paired_speedup limit_ms status")
    for row in rows:
        print(f"{row['step']:>4} {row['variant']:<15} {row['median_ms']:9.6f} "
              f"{row['min_ms']:9.6f} {row['max_ms']:9.6f} {row['paired_speedup']:14.3f} "
              f"{row['limit_ms']:8.6f} {row['status']}")
    print(f"Saved {args.output / 'summary.csv'}; SLOW is a measured result, not a tool error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
