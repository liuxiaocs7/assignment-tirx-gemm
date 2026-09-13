"""Controlled Step 4/5 experiments; not an assignment implementation.

Each variant changes one thing in a saved builder or after TVM lowering.
The original gemm_kernels.py, compiler options, verification tolerances,
and CUDA-event timer are unchanged.
Use a fresh --output directory. All actual compiler inputs/binaries are saved.
Defaults measure only the production baseline: Step 4 has adopted k_tile_128
and Step 5 has adopted mma_wait_64ns. Use benchmark.py for production validation.
Completed experiments remain explicit opt-ins where applicable to the builder.
"""

import argparse
from contextlib import contextmanager
import csv
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics

from benchmark_diagnostics import capture_compilation, run_metadata, write_json


WAIT_VARIANTS = ("wait_64ns", "wait_poll", "tma_wait_64ns", "mma_wait_64ns")
BUILDER_VARIANTS = ("k_tile_128", "tmem_load_64")
STEP4_VARIANTS = (*BUILDER_VARIANTS, "unroll_k")
DEFAULT_VARIANTS = ("baseline",)
VARIANTS = (*DEFAULT_VARIANTS, *STEP4_VARIANTS, *WAIT_VARIANTS, "early_release", "no_k_unroll")


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("experiment no longer matches the Step 4 builder")
    return source.replace(before, after)


def variant_builder_source(source, variant):
    """Change one Step 4 parameter/path; the saved diff makes it reviewable."""
    if variant == "k_tile_128":
        if "BLK_K = 128 if K % 128 == 0 else 64" in source:
            raise ValueError("Step 4 has adopted k_tile_128; validate the production kernel "
                             "with benchmark.py --steps 4 and tests/test_step04.py")
        source = replace_once(source, "BLK_M, BLK_N, BLK_K = 128, 128, 64",
                              "BLK_M, BLK_N, BLK_K = 128, 128, 128")
        source = replace_once(source, "K % 64:", "K % 128:")
        return replace_once(source, "K divisible by 64", "K divisible by 128")
    if variant == "tmem_load_64":
        source = replace_once(source, "    TMEM_COLS = BLK_N\n",
                              "    TMEM_COLS = BLK_N\n    TMEM_LD_N = 64\n")
        before = '''        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
            layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        # Publish completed TMEM reads before CTA reuse or deallocation.
        T.ptx.tcgen05.fence.before_thread_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
'''
        after = '''        Dreg = T.alloc_local((TMEM_LD_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, TMEM_LD_N,
            layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
        for chunk in T.unroll(BLK_N // TMEM_LD_N):
            col = T.meta_var(chunk * TMEM_LD_N)
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, col:col + TMEM_LD_N])
            T.ptx.tcgen05.wait.ld()
            Tx.cast(Dreg_f16[col:col + TMEM_LD_N], Dreg[:])
        # Publish completed TMEM reads before CTA reuse or deallocation.
        T.ptx.tcgen05.fence.before_thread_sync()
'''
        return replace_once(source, before, after)
    raise ValueError(f"unsupported builder variant: {variant}")


def build_variant(step, size, variant, directory):
    import gemm_kernels

    builder = getattr(gemm_kernels, f"hgemm_v{step}")
    if variant in BUILDER_VARIANTS:
        if step != 4:
            raise ValueError(f"{variant} is a Step 4 experiment")
        before = inspect.getsource(builder)
        after = variant_builder_source(before, variant)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "builder.before.py").write_text(before)
        (directory / "builder.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="baseline.py", tofile=f"{variant}.py",
        )))
        # A real file lets TVMScript inspect the nested prim_func. The new
        # namespace inherits imports, without replacing production builders.
        path = directory / "builder.py"
        path.write_text(after)
        write_json(directory / "builder.json", dict(
            before_sha256=hashlib.sha256(before.encode()).hexdigest(),
            compiled_sha256=hashlib.sha256(after.encode()).hexdigest(),
        ))
        namespace = dict(vars(gemm_kernels))
        exec(compile(after, str(path.resolve()), "exec"), namespace)
        builder = namespace[f"hgemm_v{step}"]
    return builder(size, size, size)


def change_wait(header, body, step, variant):
    """Change wait scheduling while retaining parity, acquire, and retry logic."""
    if "tirx_mma_wait_64ns" in header or "tirx_mma_wait_64ns" in body:
        raise ValueError("Step 5 has adopted mma_wait_64ns; validate the production kernel "
                         "with benchmark.py --steps 5 and tests/test_step05.py")
    name = "tvm_builtin_ptx_mbarrier_try_wait"
    helper = re.compile(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}\n",
                        re.M | re.S)
    matches = list(helper.finditer(header))
    if len(matches) != 1:
        raise ValueError("expected exactly one default mbarrier wait helper")
    original = matches[0].group()
    ticks = "    unsigned int ticks = 0x989680;\n"
    instruction = "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;"
    operands = ':: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory"'
    if any(original.count(part) != 1 for part in (ticks, instruction, operands)):
        raise ValueError("unexpected default mbarrier wait implementation")
    if variant == "wait_poll":
        # test_wait is non-blocking; the existing loop still retries until ready.
        # No relaxed qualifier: default acquire.cta semantics are preserved.
        changed = original.replace(ticks, "").replace(
            instruction, "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;"
        ).replace(operands, ':: "r"(barrier_addr_int), "r"(phase) : "memory"')
    else:
        changed = original.replace(ticks, "    unsigned int ticks = 64;\n")
    if variant in ("wait_64ns", "wait_poll"):
        return header.replace(original, changed), body

    # The production layouts put TMA barriers at uint64 offsets 1[/2], with
    # MMA at 2 (Step 4) or 3 (Step 5). Refuse a changed layout/call structure.
    indices = (1,) if step == 4 else (1, 2)
    if variant == "mma_wait_64ns":
        indices = (2,) if step == 4 else (3,)
    expected = 1 if step == 4 else 2
    probe_name = "tvm_probe_ptx_mbarrier_wait_64ns"
    if probe_name in header or probe_name in body:
        raise ValueError("wait probe is already applied")
    if body.count(name + "(") != 2 * expected:
        raise ValueError("unexpected number of TMA/MMA wait calls")
    calls = [f"{name}((&(((uint64_t*)pool_buf_ptr)[{index}]))" for index in indices]
    if sum(body.count(call) for call in calls) != expected:
        raise ValueError("unexpected TMA/MMA barrier offsets")
    for call in calls:
        body = body.replace(call, call.replace(name, probe_name))
    return header + changed.replace(name, probe_name) + "\n", body


def variant_source(source, step, variant):
    """Fail closed if a non-baseline experiment no longer matches TVM's output."""
    if step not in (4, 5) or variant not in VARIANTS:
        raise ValueError("only Step 4/5 and the declared variants are supported")
    if variant in STEP4_VARIANTS and step != 4:
        raise ValueError(f"{variant} is a Step 4 experiment")
    if variant == "baseline":
        return source
    marker = 'extern "C" __global__'
    if marker not in source:
        raise ValueError("CUDA kernel declaration was not found")
    header, body = source.split(marker, 1)
    if variant in BUILDER_VARIANTS:
        # The builder diff is saved separately. Capture the emitted CUDA as-is.
        return source
    if variant == "early_release":
        # Same warp, after its only allocation; deallocation stays at the end.
        release = "    tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_1();\n"
        alloc = re.compile(r"^    tvm_builtin_ptx_tcgen05_alloc_cta_group_1\([^\n]+, 128\);\n", re.M)
        if body.count(release) != 1 or len(alloc.findall(body)) != 1:
            raise ValueError("expected exactly one 128-column allocation and release")
        if body[alloc.search(body).end():].startswith(release):
            raise ValueError("early_release is already applied; validate the production kernel "
                             "with benchmark.py --steps 4,5 and tests/test_step04.py/test_step05.py")
        body = body.replace(release, "")
        body = alloc.sub(lambda match: match.group() + release, body)
    elif variant in WAIT_VARIANTS:
        header, body = change_wait(header, body, step, variant)
    elif variant in ("no_k_unroll", "unroll_k"):
        name = "k" if step == 4 else "ring"
        loop = re.compile(rf"^( +)(for \(int {name} = 0; {name} < \d+; \+\+{name}\) \{{)$", re.M)
        if len(loop.findall(body)) != 1:
            raise ValueError(f"expected exactly one {name} loop; use a larger K")
        pragma = "unroll 1" if variant == "no_k_unroll" else "unroll"
        body = loop.sub(lambda match: f"{match[1]}#pragma {pragma}\n{match[0]}", body)
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
    parser.add_argument("--steps", type=int, nargs="+", choices=(4, 5), default=[4])
    parser.add_argument("--size", type=int, choices=(512, 1024, 2048), default=1024)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
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
    if 5 in args.steps and any(variant in STEP4_VARIANTS for variant in variants):
        parser.error("new variants target Step 4; use benchmark.py --steps 5 for production Step 5")
    if 5 in args.steps and any(variant in WAIT_VARIANTS for variant in variants):
        parser.error("Step 5 has adopted mma_wait_64ns; use benchmark.py --steps 5")

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
            kernel = build_variant(step, args.size, variant, directory)
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
