"""Controlled Step 4/5 CUDA-code experiments; not an assignment implementation.

Each variant changes one thing after TVM lowering. The original gemm_kernels.py,
compiler options, verification tolerances, and CUDA-event timer are unchanged.
Use a fresh --output directory. All actual compiler inputs/binaries are saved.
"""

import argparse
from contextlib import contextmanager
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics

from benchmark_diagnostics import capture_compilation, run_metadata, write_json


VARIANTS = ("baseline", "early_release", "wait_64ns", "no_k_unroll")


def variant_source(source, step, variant):
    """Fail closed if a non-baseline experiment no longer matches TVM's output."""
    if step not in (4, 5) or variant not in VARIANTS:
        raise ValueError("only Step 4/5 and the declared variants are supported")
    if variant == "baseline":
        return source
    marker = 'extern "C" __global__'
    if marker not in source:
        raise ValueError("CUDA kernel declaration was not found")
    header, body = source.split(marker, 1)
    if variant == "early_release":
        # Same warp, after its only allocation; deallocation stays at the end.
        release = "    tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_1();\n"
        alloc = re.compile(r"^    tvm_builtin_ptx_tcgen05_alloc_cta_group_1\([^\n]+, 128\);\n", re.M)
        if body.count(release) != 1 or len(alloc.findall(body)) != 1:
            raise ValueError("expected exactly one 128-column allocation and release")
        body = body.replace(release, "")
        body = alloc.sub(lambda match: match.group() + release, body)
    elif variant == "wait_64ns":
        # This is a suspension hint, not a timeout that permits stale operands.
        # The existing retry-until-complete loop and acquire semantics remain.
        ticks = "unsigned int ticks = 0x989680;"
        if header.count(ticks) != 1:
            raise ValueError("expected exactly one default mbarrier wait helper")
        header = header.replace(ticks, "unsigned int ticks = 64;")
    elif variant == "no_k_unroll":
        name = "k" if step == 4 else "ring"
        loop = re.compile(rf"^( +)(for \(int {name} = 0; {name} < \d+; \+\+{name}\) \{{)$", re.M)
        if len(loop.findall(body)) != 1:
            raise ValueError(f"expected exactly one {name} loop; use a larger K")
        body = loop.sub(lambda match: f"{match[1]}#pragma unroll 1\n{match[0]}", body)
    return header + marker + body


@contextmanager
def source_experiment(step, variant, directory):
    """Install inside capture_compilation so it records the transformed source."""
    import tvm_ffi

    name = "tvm_callback_cuda_compile"
    original = tvm_ffi.get_global_func(name)
    calls = []

    def compile_cuda(code):
        before = str(code)
        after = variant_source(before, step, variant)
        stem = f"source_{len(calls) + 1:02d}"
        (directory / f"{stem}.before.cu").write_text(before)
        (directory / f"{stem}.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="baseline.cu", tofile=f"{variant}.cu",
        )))
        calls.append(dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                          compiled_sha256=hashlib.sha256(after.encode()).hexdigest()))
        return original(after)

    tvm_ffi.register_global_func(name, compile_cuda, override=True)
    try:
        yield calls
    finally:
        tvm_ffi.register_global_func(name, original, override=True)
        write_json(directory / "experiment.json", dict(step=step, variant=variant, sources=calls))


def trial_order(count, trial):
    """Rotate and reverse measurement order to expose ordering/clock drift."""
    order = list(range(count))
    offset = trial % count
    order = order[offset:] + order[:offset]
    return order if trial % 2 == 0 else order[::-1]


def summarize(cases, reference_times, tolerance):
    baselines = {case["step"]: case["samples_ms"] for case in cases if case["variant"] == "baseline"}
    rows = []
    for case in cases:
        samples = case["samples_ms"]
        elapsed = statistics.median(samples)
        size = case["size"]
        limit = reference_times[(case["step"], size, size, size)] * tolerance
        ratios = [base / value for base, value in zip(baselines[case["step"]], samples)]
        rows.append(dict(step=case["step"], size=size, variant=case["variant"],
                         median_ms=elapsed, min_ms=min(samples), max_ms=max(samples),
                         paired_speedup=statistics.median(ratios),
                         tflops=2 * size**3 / (elapsed * 1e9), limit_ms=limit,
                         status="PASS" if elapsed <= limit else "SLOW",
                         samples_ms=json.dumps(samples)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory, must not exist")
    parser.add_argument("--steps", type=int, nargs="+", choices=(4, 5), default=[4, 5])
    parser.add_argument("--size", type=int, choices=(512, 1024, 2048), default=2048)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.trials < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("trials/repeat must be positive; warmup must be nonnegative")
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    variants = list(dict.fromkeys(("baseline", *args.variants)))

    import torch
    import tvm
    import gemm_kernels
    from utils import REFERENCE_TIMES, TIMING_TOLERANCE, blackwell_target, prepare_data, time_cuda_call, verify

    if not torch.cuda.is_available():
        parser.error("a Blackwell GPU and CUDA-enabled PyTorch are required")
    target = blackwell_target()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True)
    metadata = dict(run_metadata(), target=str(target), gpu=device.name,
                    sm_count=device.multi_processor_count, tvm=tvm.__version__,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    steps=args.steps, size=args.size, variants=variants, trials=args.trials,
                    warmup=args.warmup, repeat=args.repeat, seed=args.seed)
    write_json(args.output / "run.json", metadata)
    print(f"Code: {metadata['git_revision']}; gemm_sha256={metadata['gemm_kernels_sha256']}", flush=True)
    print(f"GPU: {device.name}; {target}", flush=True)
    print("Diagnostic variants only; verify all outputs before interleaved CUDA-event timing.", flush=True)
    A, B, _ = prepare_data(args.size, args.size, args.size)
    cases, executables, outputs = [], [], []
    for step in dict.fromkeys(args.steps):
        for variant in variants:
            directory = args.output / f"step{step:02d}_{args.size}_{variant}"
            kernel = getattr(gemm_kernels, f"hgemm_v{step}")(args.size, args.size, args.size)
            # NaNs make any uninitialized output fail verification.
            output = torch.full((args.size, args.size), float("nan"), dtype=A.dtype, device=A.device)
            with target:
                with capture_compilation(directory), source_experiment(step, variant, directory) as calls:
                    executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
                    executable.mod(A, B, output)
                if len(calls) != 1:
                    raise RuntimeError(f"{directory}: expected one actual CUDA compilation, got {len(calls)}")
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
            # Verify after reuse as well, outside the timing interval.
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
