"""Verify GEMM kernels and report CUDA-event timings (see RUNNING.md)."""

import argparse
import csv
import statistics
import sys
from pathlib import Path

from benchmark_diagnostics import capture_compilation, run_metadata, write_json


def trial_order(count, trial):
    """Pair each order with its reverse, rotating by two between pairs.

    A cycle takes count trials for even counts and 2*count for odd counts.
    Each cycle balances every position and both orders of every pair. With
    two cases this is AB/BA; an incomplete cycle can still be unbalanced.
    """
    order = list(range(count))
    offset = (2 * (trial // 2)) % count
    order = order[offset:] + order[:offset]
    return order if trial % 2 == 0 else order[::-1]


def parse_steps(value):
    if value == "all":
        return list(range(1, 11))
    try:
        steps = list(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use 1-10, a comma-separated list, or all") from error
    if not steps or any(step < 1 or step > 10 for step in steps):
        raise argparse.ArgumentTypeError("steps must be between 1 and 10")
    return steps


def select_shapes(step, sizes, reference_times):
    """Default to the graded shapes; custom sizes keep each step's contract."""
    if sizes is None:
        return [key[1:] for key in reference_times if key[0] == step]
    shapes = []
    for size in dict.fromkeys(sizes):
        if step == 1:
            if size != 64:
                raise ValueError("step 1 only supports --sizes 64 (M=N=128, K=64)")
            shape = (128, 128, 64)
        elif step == 2:
            if size <= 0 or size % 64:
                raise ValueError("step 2 sizes specify K and must be positive multiples of 64")
            shape = (128, 128, size)
        else:
            multiple = 512 if step == 10 else 256 if step == 9 else 128
            if size <= 0 or size % multiple:
                raise ValueError(f"step {step} square sizes must be positive multiples of {multiple}")
            shape = (size, size, size)
        shapes.append(shape)
    return shapes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=parse_steps, default=[10], help="1-10, comma-separated list, or all")
    parser.add_argument("--sizes", type=int, nargs="+", help="square dimensions; for steps 1-2, specifies K")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3, help="report median of this many timing trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="write timing results to this file")
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="save run metadata, CUDA/binary, and NVRTC compiler logs before timing")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1 or args.trials < 1:
        parser.error("warmup must be nonnegative; repeat and trials must be positive")
    if args.sizes is not None:
        try:
            for step in args.steps:
                select_shapes(step, args.sizes, {})
        except ValueError as error:
            parser.error(str(error))

    # Keep --help and argument validation usable without the GPU dependencies.
    import torch
    import tvm
    import gemm_kernels
    from utils import (REFERENCE_TIMES, TIMING_TOLERANCE, blackwell_target,
                       prepare_data, time_cuda_call, verify)

    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build and a Blackwell GPU are required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        parser.error("these kernels require SM100/SM103 (B200/B100/B300)")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    # Match persistent CTA count to the actual device (148 on B200).
    gemm_kernels.SM_COUNT = device.multi_processor_count
    if gemm_kernels.SM_COUNT % 2:
        parser.error("cluster kernels require an even persistent CTA count")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    target = blackwell_target()
    metadata = run_metadata()
    metadata["target"] = str(target)
    metadata["comparison_order"] = "paired_reverse_rotation_v1"
    print(f"Code: {metadata['git_revision']}; dirty={metadata['git_dirty']}; "
          f"gemm_sha256={metadata['gemm_kernels_sha256']}")
    print(f"Compiler: {metadata['compiler']}; ptxas register-usage-level={metadata['ptxas_reg_level']}; "
          f"target={target}")
    for key, value in metadata.items():
        if key.isupper() and value:
            print(f"  {key}={value}")
    print(f"GPU: {device.name}; SMs: {device.multi_processor_count}")
    print(f"TVM: {tvm.__version__}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"CUDA events: warmup={args.warmup}, repeat={args.repeat}, trials={args.trials}, seed={args.seed}")
    print("cuBLAS comparison: alternate kernel/cuBLAS and cuBLAS/kernel per trial; save paired samples.")
    if args.diagnostics_dir:
        # A fresh directory prevents stale compiler logs from a previous run.
        args.diagnostics_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.diagnostics_dir / "run.json", dict(
            metadata, gpu=device.name, sm_count=device.multi_processor_count,
            tvm=tvm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
            steps=args.steps, sizes=args.sizes, warmup=args.warmup, repeat=args.repeat,
            trials=args.trials, seed=args.seed,
        ))
        print(f"Compiler artifacts: {args.diagnostics_dir} (hooks removed before timing)")
    print("step       M       N       K   median_ms   TFLOP/s   cuBLAS_ms   vs_cuBLAS  paired_vs   limit_ms   status")

    rows = []
    failed = False
    for step in args.steps:
        for M, N, K in select_shapes(step, args.sizes, REFERENCE_TIMES):
            kernel = getattr(gemm_kernels, f"hgemm_v{step}")(M, N, K)
            A, B, output = prepare_data(M, N, K)
            dump_dir = (args.diagnostics_dir / f"step{step:02d}_{M}_{N}_{K}"
                        if args.diagnostics_dir else None)
            with target:
                with capture_compilation(dump_dir):
                    executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
                    executable.mod(A, B, output)
                verify(output, A, B)
                # Prepare and verify the exact cuBLAS call before either timing.
                # Both calls use the same inputs/current stream and separate outputs.
                reference = torch.empty_like(output)
                calls = (lambda: executable.mod(A, B, output),
                         lambda: torch.mm(A, B.T, out=reference))
                calls[1]()
                torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-2)
                samples, reference_samples, orders = [], [], []
                sample_lists = (samples, reference_samples)
                names = ("kernel", "cublas")
                for trial in range(args.trials):
                    order = trial_order(2, trial)
                    orders.append([names[index] for index in order])
                    for index in order:
                        sample_lists[index].append(time_cuda_call(calls[index], args.warmup, args.repeat))
                    torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-2)
            paired_ratios = [ref / value for value, ref in zip(samples, reference_samples)]
            paired_speedup = statistics.median(paired_ratios)
            elapsed = statistics.median(samples)
            cublas_ms = statistics.median(reference_samples)
            tflops = 2 * M * N * K / (elapsed * 1e-3) / 1e12
            ref_ms = REFERENCE_TIMES.get((step, M, N, K))
            limit_ms = ref_ms * TIMING_TOLERANCE if ref_ms is not None else None
            status = "UNSCORED" if limit_ms is None else "PASS" if elapsed <= limit_ms else "SLOW"
            failed |= status == "SLOW"
            limit = "-" if limit_ms is None else f"{limit_ms:.6f}"
            print(f"{step:>4} {M:>7} {N:>7} {K:>7} {elapsed:>11.6f} {tflops:>9.2f} "
                  f"{cublas_ms:>11.6f} {cublas_ms / elapsed:>10.2f} {paired_speedup:>10.3f} "
                  f"{limit:>10}   {status}", flush=True)
            rows.append(dict(step=step, M=M, N=N, K=K, median_ms=elapsed, min_ms=min(samples),
                             max_ms=max(samples), tflops=tflops, cublas_ms=cublas_ms,
                             speedup_vs_cublas=cublas_ms / elapsed, reference_ms=ref_ms,
                             limit_ms=limit_ms, status=status, seed=args.seed,
                             warmup=args.warmup, repeat=args.repeat, trials=args.trials,
                             gpu=device.name, sm_count=device.multi_processor_count,
                             tvm=tvm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                             samples_ms=samples, cublas_samples_ms=reference_samples,
                             trial_orders=orders, cublas_speedup_samples=paired_ratios,
                             paired_speedup_vs_cublas=paired_speedup, **metadata))
            if dump_dir:
                write_json(dump_dir / "timing.json", rows[-1])
            if args.csv:
                args.csv.parent.mkdir(parents=True, exist_ok=True)
                with args.csv.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
