"""Locate Step 8/10 stalls with separate, instrumented role kernels.

No production kernel or grading is changed. Baseline uses the original CUDA
event timer. TMA, MMA, and writeback are instrumented in separate copies; their
timings are diagnostic overhead measurements, never PASS/SLOW results.
Each trace describes the last launch of a trial, not the entire timing batch.
"""

import argparse
import csv
import difflib
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from benchmark_diagnostics import capture_compilation, run_metadata, write_json
from probe_step45 import replace_once, trial_order


ROLES = ("tma", "mma", "writeback")
FIELDS = ("start_ns", "end_ns", "wait_ns", "work_ns", "handoff_ns", "epilogue_ns",
          "start_cycles", "end_cycles", "sm_id", "tile_m", "tile_n")

# Local accumulators avoid a global store at every K-stage boundary. Only the
# original elected lane, or lane 0 of writeback warp 0, records observations.
# The memory clobber orders the inline timer reads with memory/async calls;
# register-only compiler scheduling and the added instructions can still alter
# the measured kernel. Compare against baseline, not against grading limits.
TIMER_SOURCE = r'''
#ifndef TIRX_PROFILE_TIMERS_DEFINED
#define TIRX_PROFILE_TIMERS_DEFINED
__forceinline__ __device__ uint64_t tirx_profile_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}
__forceinline__ __device__ uint64_t tirx_profile_cycles() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
    return value;
}
__forceinline__ __device__ void tirx_profile_start(uint64_t* s, bool leader) {
    if (leader) {
        s[0] = tirx_profile_ns();
        s[1] = tirx_profile_cycles();
        unsigned int sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        s[2] = sm;
        s[3] = 0; s[4] = 0; s[5] = 0; s[6] = 0;
    }
}
__forceinline__ __device__ void tirx_profile_mark(uint64_t* s, bool leader) {
    if (leader) s[7] = tirx_profile_ns();
}
__forceinline__ __device__ void tirx_profile_add_wait(uint64_t* s, bool leader) {
    if (leader) s[3] += tirx_profile_ns() - s[7];
}
__forceinline__ __device__ void tirx_profile_add_work(uint64_t* s, bool leader) {
    if (leader) s[4] += tirx_profile_ns() - s[7];
}
__forceinline__ __device__ void tirx_profile_add_handoff(uint64_t* s, bool leader) {
    if (leader) s[5] += tirx_profile_ns() - s[7];
}
__forceinline__ __device__ void tirx_profile_add_epilogue(uint64_t* s, bool leader) {
    if (leader) s[6] += tirx_profile_ns() - s[7];
}
__forceinline__ __device__ void tirx_profile_finish(
        uint64_t* s, int64_t* out, int m, int n, bool leader) {
    if (leader) {
        uint64_t end = tirx_profile_ns();
        uint64_t cycles = tirx_profile_cycles();
        out[0] = s[0]; out[1] = end;
        out[2] = s[3]; out[3] = s[4]; out[4] = s[5]; out[5] = s[6];
        out[6] = s[1]; out[7] = cycles; out[8] = s[2];
        out[9] = m; out[10] = n;
    }
}
#endif
'''


def layout(step, shape, sm_count):
    """Match the production launch grid; reserve one record per real tile."""
    M, N, K = shape
    if step not in (8, 10) or min(M, N, K, sm_count) <= 0 or K % 64:
        raise ValueError("profiling supports Steps 8/10 and positive aligned shapes")
    group, consumers, tile_m, tile_n = (1, 1, 128, 128) if step == 8 else (2, 2, 512, 256)
    if M % tile_m or N % tile_n or sm_count % group:
        raise ValueError("shape or SM count does not match the production grid")
    total = (M // tile_m) * (N // tile_n)
    clusters = min(sm_count // group, total)
    if step == 10:
        tiles_per_cluster = (total + clusters - 1) // clusters
        clusters = (total + tiles_per_cluster - 1) // tiles_per_cluster
    return dict(ctas=clusters * group, clusters=clusters, group=group,
                consumers=consumers, total_tiles=total,
                max_tiles=(total + clusters - 1) // clusters,
                m_tiles=M // tile_m, n_tiles=N // tile_n)


def instrument_builder(source, step, role, info):
    """Wrap one role's original operations, retaining its barriers and work."""
    if role not in ROLES or step not in (8, 10):
        raise ValueError("unsupported profiling role or step")
    if "tirx_profile_" in source:
        raise ValueError("builder is already instrumented")
    shape = (info["ctas"], info["max_tiles"], info["consumers"], len(FIELDS))
    source = replace_once(source, "    @T.prim_func\n",
                          f"    PROFILE_SOURCE = {TIMER_SOURCE!r}\n\n    @T.prim_func\n")
    source = replace_once(source, "        D: T.Buffer((M, N), d_type),\n",
                          f"        D: T.Buffer((M, N), d_type),\n        Profile: T.Buffer({shape}, \"int64\"),\n")
    mma_branch = "            elif warp_id == 0:\n" if step == 8 else "            elif warp_id < NUM_CONSUMER:\n"
    wb_branch = "        elif wg_id == 0:\n" if step == 8 else "        elif wg_id < NUM_CONSUMER:\n"
    if role == "tma":
        start, end = "            if warp_id == 3:\n", mma_branch
    elif role == "mma":
        start, end = mma_branch, wb_branch
    else:
        start = wb_branch
        end = "\n        T.cuda.cta_sync()\n" if step == 8 else "\n        T.cuda.cluster_sync()\n"
    if source.count(start) != 1:
        raise ValueError("expected a unique role branch")
    prefix, tail = source.split(start)
    if tail.count(end) != 1:
        raise ValueError("expected a unique role boundary")
    block, suffix = tail.split(end)
    leader = "(warp_id == 0) & (lane_id == 0)" if role == "writeback" else "True"
    consumer = "0" if step == 8 or role == "tma" else "warp_id" if role == "mma" else "wg_id"

    def call(action, indent, extra=""):
        return (" " * indent + f'T.cuda.func_call("tirx_profile_{action}", profile_state.data, '
                + extra + leader + ", source_code=PROFILE_SOURCE)\n")

    loop = next(line for line in block.splitlines(True) if "while tile_scheduler.valid():" in line)
    indent = len(loop) - len(loop.lstrip())
    block = replace_once(block, loop, " " * indent + 'profile_state = T.alloc_local((8,), "uint64")\n' +
                         loop + call("start", indent + 4))

    def wrap(before, after, metric):
        nonlocal block
        first = next(line for line in block.splitlines(True) if before in line)
        last = next(line for line in block.splitlines(True) if after in line)
        begin_indent = len(first) - len(first.lstrip())
        end_indent = begin_indent  # End of a multiline call keeps its statement indentation.
        block = replace_once(block, first, call("mark", begin_indent) + first)
        block = replace_once(block, last, last + call("add_" + metric, end_indent))

    if role == "tma":
        wrap("mma2tma.wait(", "mma2tma.wait(", "wait")
        wrap("tma_load(k * BLK_K)", "tma_phase.advance()", "work")
    elif role == "mma":
        wrap("ld2mma.wait(", "ld2mma.wait(", "handoff")
        if step == 8:
            # Step 8's local wait call spans two Python lines.
            wrap('T.cuda.func_call("tirx_tma_wait_64ns"', "source_code=TMA_WAIT_SOURCE)", "wait")
        else:
            wrap("tma2mma.wait(", "tma2mma.wait(", "wait")
        wrap("T.ptx.tcgen05.fence.after_thread_sync()", "mma_phase.advance()", "work")
    else:
        wrap("mma2ld.wait(", "mma2ld.wait(", "wait")
        wrap("T.ptx.tcgen05.fence.after_thread_sync()", "ld2mma.arrive(", "work")
        anchor = next(line for line in block.splitlines(True) if "for i in T.unroll(MMA_N // EPI_N):" in line)
        block = replace_once(block, anchor, call("mark", indent + 4) + anchor)

    advance = " " * (indent + 4) + "tile_scheduler.next_tile()\n"
    finish = call("add_epilogue", indent + 4) if role == "writeback" else ""
    finish += call("finish", indent + 4,
                   f"Profile.ptr_to([bx, tile_scheduler.tile_count, {consumer}, 0]), "
                   "tile_scheduler.m_idx, tile_scheduler.n_idx, ")
    block = replace_once(block, advance, finish + advance)
    return prefix + start + block + end + suffix


def build_profile(step, shape, role, directory, sm_count):
    import gemm_kernels

    builder = getattr(gemm_kernels, f"hgemm_v{step}")
    before = inspect.getsource(builder)
    info = layout(step, shape, sm_count)
    after = instrument_builder(before, step, role, info)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "builder.py"
    path.write_text(after)
    (directory / "builder.before.py").write_text(before)
    (directory / "builder.patch").write_text("".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="baseline.py", tofile=f"{role}.py")))
    write_json(directory / "builder.json", dict(role=role, layout=info, fields=FIELDS,
        before_sha256=hashlib.sha256(before.encode()).hexdigest(),
        compiled_sha256=hashlib.sha256(after.encode()).hexdigest()))
    namespace = dict(vars(gemm_kernels))
    namespace["SM_COUNT"] = sm_count
    exec(compile(after, str(path.resolve()), "exec"), namespace)
    return namespace[f"hgemm_v{step}"](*shape), info


def trace_records(values, info, role):
    """Reject inactive slots, missing/duplicate tiles, and malformed records."""
    rows = []
    for cta in range(info["ctas"]):
        cluster, rank = divmod(cta, info["group"])
        tiles = (info["total_tiles"] - 1 - cluster) // info["clusters"] + 1
        for tile in range(info["max_tiles"]):
            for consumer in range(info["consumers"]):
                record = values[cta][tile][consumer]
                active = tile < tiles and (role != "tma" or consumer == 0) and (role != "mma" or rank == 0)
                if not active:
                    if any(x != -1 for x in record):
                        raise ValueError("trace contains a write from an inactive role/tile")
                    continue
                if len(record) != len(FIELDS) or min(record) < 0:
                    raise ValueError("missing or incomplete trace record")
                row = dict(zip(FIELDS, record))
                elapsed = row["end_ns"] - row["start_ns"]
                measured = sum(row[key] for key in ("wait_ns", "work_ns", "handoff_ns", "epilogue_ns"))
                if elapsed <= 0 or measured > elapsed or row["end_cycles"] < row["start_cycles"]:
                    raise ValueError("invalid timestamp ordering or overlapping trace intervals")
                if row["tile_m"] >= info["m_tiles"] or row["tile_n"] >= info["n_tiles"]:
                    raise ValueError("trace tile coordinate is out of range")
                row.update(cta=cta, cluster=cluster, cta_rank=rank, tile_ordinal=tile,
                           consumer=consumer, elapsed_ns=elapsed, unmeasured_ns=elapsed-measured)
                rows.append(row)
    for rank in range(info["group"]):
        for consumer in range(info["consumers"]):
            if (role == "mma" and rank != 0) or (role == "tma" and consumer != 0):
                continue
            coords = [(row["tile_m"], row["tile_n"]) for row in rows
                      if row["cta_rank"] == rank and row["consumer"] == consumer]
            if len(set(coords)) != info["total_tiles"]:
                raise ValueError("trace has duplicate or missing output tile coordinates")
    return rows


def summarize_traces(rows):
    """Keep consumers, cluster ranks, and initial/reused tiles separate."""
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ("step", "size", "role", "cta_rank", "consumer", "tile_ordinal"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        total = sum(row["elapsed_ns"] for row in group)
        summary = dict(zip(("step", "size", "role", "cta_rank", "consumer", "tile_ordinal"), key))
        summary.update(records=len(group), median_us=statistics.median(row["elapsed_ns"] for row in group)/1000,
                       max_us=max(row["elapsed_ns"] for row in group)/1000)
        for metric in ("wait", "work", "handoff", "epilogue", "unmeasured"):
            summary[metric + "_pct"] = 100 * sum(row[metric + "_ns"] for row in group) / total
        summary["cycles_per_ns"] = statistics.median(
            (row["end_cycles"] - row["start_cycles"]) / row["elapsed_ns"] for row in group)
        result.append(summary)
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump_sass(directory):
    """Read actual binaries, never recompile to obtain an assembly report."""
    executable = shutil.which("cuobjdump")
    if executable is None:
        candidate = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin/cuobjdump"
        if candidate.is_file():
            executable = str(candidate)
    binaries = sorted([*directory.glob("module_*.cubin"), *directory.glob("module_*.fatbin")])
    for path in binaries:
        if executable is None:
            path.with_suffix(".sass.txt").write_text("SASS unavailable: cuobjdump was not found.\n")
            continue
        try:
            result = subprocess.run([executable, "--dump-sass", str(path)],
                                    capture_output=True, text=True, timeout=30)
            message = f"cuobjdump exit code: {result.returncode}\n" + result.stdout + result.stderr
        except (OSError, subprocess.SubprocessError) as error:
            message = f"SASS unavailable: {error}\n"
        path.with_suffix(".sass.txt").write_text(message)


def gpu_snapshot():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return "nvidia-smi not available\n"
    try:
        result = subprocess.run([executable, "--query-gpu=uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu",
                                 "--format=csv,nounits"], capture_output=True, text=True, timeout=10)
        return result.stdout + result.stderr
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi unavailable: {error}\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory")
    parser.add_argument("--steps", type=int, nargs="+", choices=(8, 10), default=[8, 10])
    parser.add_argument("--size", type=int, choices=(1024, 2048, 4096, 8192),
                        help="otherwise Step 8 uses 2048 and Step 10 uses 4096")
    parser.add_argument("--roles", choices=ROLES, nargs="+", default=list(ROLES))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args(argv)
    if args.trials < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("trials/repeat must be positive; warmup must be nonnegative")
    if args.output.exists():
        parser.error("--output must be a fresh directory")

    import torch
    import tvm
    import gemm_kernels
    from utils import blackwell_target, prepare_data, verify, time_cuda_call, REFERENCE_TIMES, TIMING_TOLERANCE

    if not torch.cuda.is_available():
        parser.error("a Blackwell GPU and CUDA-enabled PyTorch are required")
    target = blackwell_target()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    gemm_kernels.SM_COUNT = device.multi_processor_count
    args.output.mkdir(parents=True)
    metadata = dict(run_metadata(), profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    probe_step45_sha256=hashlib.sha256(Path(__file__).with_name("probe_step45.py").read_bytes()).hexdigest(),
                    target=str(target), gpu=device.name, sm_count=device.multi_processor_count,
                    tvm=tvm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                    steps=args.steps, size=args.size, roles=args.roles, seed=0,
                    trials=args.trials, warmup=args.warmup, repeat=args.repeat,
                    trace_scope="last timed launch of each trial; per-role kernels; uncorrected overhead")
    write_json(args.output / "run.json", metadata)
    print(f"Code: {metadata['git_revision']}; gemm_sha256={metadata['gemm_kernels_sha256']}", flush=True)
    print(f"GPU: {device.name}; SMs: {device.multi_processor_count}; {target}", flush=True)
    print("Instrumented roles are diagnostics, not performance candidates. Only baseline is scored.", flush=True)
    timings, traces = [], []
    for step in dict.fromkeys(args.steps):
        size = args.size or (2048 if step == 8 else 4096)
        info = layout(step, (size,) * 3, device.multi_processor_count)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        A, B, _ = prepare_data(size, size, size)
        cases = []
        root = args.output / f"step{step:02}_{size}"
        root.mkdir()
        for role in ("baseline", *dict.fromkeys(args.roles)):
            directory = root / role
            if role == "baseline":
                kernel = getattr(gemm_kernels, f"hgemm_v{step}")(size, size, size)
                trace = None
            else:
                kernel, _ = build_profile(step, (size,) * 3, role, directory, device.multi_processor_count)
                trace = torch.full((info["ctas"], info["max_tiles"], info["consumers"], len(FIELDS)),
                                   -1, dtype=torch.int64, device=A.device)
            output = torch.full_like(A, float("nan"))
            with target, capture_compilation(directory):
                executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
                call_args = (A, B, output) if trace is None else (A, B, output, trace)
                executable.mod(*call_args)
            verify(output, A, B)
            if trace is not None:
                trace_records(trace.cpu().tolist(), info, role)
            dump_sass(directory)
            cases.append((role, executable, call_args, trace, output))
            print(f"Verified step {step} / {role}", flush=True)
        for trial in range(args.trials):
            (root / f"gpu_trial{trial+1:02}_before.csv").write_text(gpu_snapshot())
            for index in trial_order(len(cases), trial):
                role, executable, call_args, trace, output = cases[index]
                output.fill_(float("nan"))
                if trace is not None:
                    trace.fill_(-1)
                with target:
                    elapsed = time_cuda_call(lambda: executable.mod(*call_args), args.warmup, args.repeat)
                verify(output, A, B)
                timings.append(dict(step=step, size=size, trial=trial+1, role=role, elapsed_ms=elapsed))
                if trace is not None:
                    records = trace_records(trace.cpu().tolist(), info, role)
                    traces.extend(dict(step=step, size=size, trial=trial+1, role=role, **row) for row in records)
                    write_json(root / f"{role}_trial{trial+1:02}.json", records)
                print(f"Trial {trial+1}: step {step} / {role}: {elapsed:.6f} ms" +
                      (" (instrumented)" if trace is not None else ""), flush=True)
                write_json(args.output / "timings.json", timings)
            (root / f"gpu_trial{trial+1:02}_after.csv").write_text(gpu_snapshot())
        baseline = [r["elapsed_ms"] for r in timings if r["step"] == step and r["role"] == "baseline"]
        limit = REFERENCE_TIMES[(step, size, size, size)] * TIMING_TOLERANCE
        print(f"Step {step} baseline: median {statistics.median(baseline):.6f} ms; "
              f"{sum(t <= limit for t in baseline)}/{len(baseline)} samples within {limit:.6f} ms", flush=True)
        for role in dict.fromkeys(args.roles):
            values = [r["elapsed_ms"] for r in timings if r["step"] == step and r["role"] == role]
            ratio = statistics.median(t / b for t, b in zip(values, baseline))
            print(f"  {role}: instrumented/baseline {ratio:.3f}x (perturbed; do not grade)", flush=True)
    write_csv(args.output / "trace.csv", traces)
    summaries = summarize_traces(traces)
    write_csv(args.output / "stages.csv", summaries)
    print("Instrumented stages: work is issue time for TMA/MMA; roles overlap and must not be added.")
    print("step role       rank consumer tile  median_us   wait%   work% handoff% epilogue%")
    for row in summaries:
        print(f"{row['step']:4} {row['role']:<10} {row['cta_rank']:4} {row['consumer']:8} {row['tile_ordinal']:4} "
              f"{row['median_us']:10.3f} {row['wait_pct']:7.1f} {row['work_pct']:7.1f} "
              f"{row['handoff_pct']:8.1f} {row['epilogue_pct']:9.1f}")
    print(f"Saved {args.output}/trace.csv and compiler artifacts (SASS when cuobjdump is available).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
