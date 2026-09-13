"""Independent B300 performance experiments for Steps 6, 7, and 10.

Defaults to Step 10 at 4096: compare wider TMEM loads, L2 grouping, and a
balanced cluster grid with production. Steps 6/7 have adopted k_tile_128;
use their production tests for validation. No production kernel is edited.
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
    10: ("baseline", "mma_wait_64ns", "tmem_load_16", "tmem_load_64",
         "l2_group_4", "balanced_clusters"),
}
DEFAULT_STEP_VARIANTS = {6: ("baseline",), 7: ("baseline",),
                         10: ("baseline", "tmem_load_64", "l2_group_4", "balanced_clusters")}
VARIANTS = tuple(dict.fromkeys(v for variants in STEP_VARIANTS.values() for v in variants))


def variant_builder_source(source, step, variant):
    """Keep each variant independent; refuse an unexpected production builder."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
    if variant in ("baseline", "mma_wait_64ns"):
        return source
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
    """Change only waits for MMA-signaled barriers, retaining acquire and retry."""
    if step not in STEP_VARIANTS or variant not in STEP_VARIANTS[step]:
        raise ValueError(f"unsupported Step {step} variant: {variant}")
    if variant != "mma_wait_64ns":
        return source
    marker = 'extern "C" __global__'
    if marker not in source:
        raise ValueError("CUDA kernel declaration was not found")
    header, body = source.split(marker, 1)
    name = "tvm_builtin_ptx_mbarrier_try_wait"
    short_name = "tvm_probe_mma_wait_64ns"
    if short_name in source:
        raise ValueError("MMA wait experiment is already applied")
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
    # Exact barrier/phase arguments from the saved 4096 production CUDA.
    # Step 6: mma_bar; Step 7/10: mma2tma and mma2ld. Do not alter tma2mma
    # (data ready) or ld2mma (accumulator free), nor remote barrier arrivals.
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
