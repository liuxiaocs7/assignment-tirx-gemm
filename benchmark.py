"""Verify GEMM kernels and report CUDA-event timings (see RUNNING.md)."""

import argparse
import csv
import statistics
import sys
from pathlib import Path


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
    from utils import REFERENCE_TIMES, TIMING_TOLERANCE, prepare_data, time_cuda_call, verify

    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build and a Blackwell GPU are required")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        parser.error("these kernels require SM100/SM103 (B200/B100/B300)")
    device = torch.cuda.get_device_properties(0)
    # Match persistent CTA count to the actual device (148 on B200).
    gemm_kernels.SM_COUNT = device.multi_processor_count
    if gemm_kernels.SM_COUNT % 2:
        parser.error("cluster kernels require an even persistent CTA count")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"GPU: {device.name}; SMs: {device.multi_processor_count}")
    print(f"TVM: {tvm.__version__}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"CUDA events: warmup={args.warmup}, repeat={args.repeat}, trials={args.trials}, seed={args.seed}")
    print("step       M       N       K   median_ms   TFLOP/s   cuBLAS_ms   vs_cuBLAS   limit_ms   status")

    rows = []
    failed = False
    target = tvm.target.Target("cuda")
    for step in args.steps:
        for M, N, K in select_shapes(step, args.sizes, REFERENCE_TIMES):
            kernel = getattr(gemm_kernels, f"hgemm_v{step}")(M, N, K)
            A, B, output = prepare_data(M, N, K)
            with target:
                executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
                executable(A, B, output)
                verify(output, A, B)
                samples = [time_cuda_call(lambda: executable(A, B, output), args.warmup, args.repeat)
                           for _ in range(args.trials)]
            # Time cuBLAS with preallocated output on the same data and stream.
            reference = torch.empty_like(output)
            reference_samples = [time_cuda_call(lambda: torch.mm(A, B.T, out=reference), args.warmup, args.repeat)
                                 for _ in range(args.trials)]
            torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-2)
            elapsed = statistics.median(samples)
            cublas_ms = statistics.median(reference_samples)
            tflops = 2 * M * N * K / (elapsed * 1e-3) / 1e12
            ref_ms = REFERENCE_TIMES.get((step, M, N, K))
            limit_ms = ref_ms * TIMING_TOLERANCE if ref_ms is not None else None
            status = "UNSCORED" if limit_ms is None else "PASS" if elapsed <= limit_ms else "SLOW"
            failed |= status == "SLOW"
            limit = "-" if limit_ms is None else f"{limit_ms:.6f}"
            print(f"{step:>4} {M:>7} {N:>7} {K:>7} {elapsed:>11.6f} {tflops:>9.2f} "
                  f"{cublas_ms:>11.6f} {cublas_ms / elapsed:>10.2f} {limit:>10}   {status}", flush=True)
            rows.append(dict(step=step, M=M, N=N, K=K, median_ms=elapsed, min_ms=min(samples),
                             max_ms=max(samples), tflops=tflops, cublas_ms=cublas_ms,
                             speedup_vs_cublas=cublas_ms / elapsed, reference_ms=ref_ms,
                             limit_ms=limit_ms, status=status, seed=args.seed,
                             warmup=args.warmup, repeat=args.repeat, trials=args.trials,
                             gpu=device.name, sm_count=device.multi_processor_count,
                             tvm=tvm.__version__, torch=torch.__version__, cuda=torch.version.cuda))
            if args.csv:
                args.csv.parent.mkdir(parents=True, exist_ok=True)
                with args.csv.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
