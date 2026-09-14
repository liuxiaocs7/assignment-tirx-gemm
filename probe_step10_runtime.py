"""Single-GPU Step 10 diagnosis: ordinary launches, CUDA Graph, then load telemetry.

Graph and sustained-load timings are diagnostic only. The kernel, official
timer, pytest and assignment limits are not changed. No clock or GPU mapping
settings are changed. Run from the repository with uv run python -u this_file.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
import traceback
import uuid

from benchmark import trial_order
from benchmark_diagnostics import capture_compilation, run_metadata, write_json

ROOT = Path(__file__).resolve().parent
WARMUP, REPEAT, TRIALS = 10, 30, 8
MODES = ("eager", "graph")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def nvidia_uuid(value):
    """Address the allocated physical GPU by UUID, never by logical index."""
    return "GPU-" + str(uuid.UUID(str(value).removeprefix("GPU-")))


def verify_output(torch, output, reference):
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-2)


def capture_graph(torch, ffi, call, output, reference):
    # TVM and Torch must share the non-default capture stream. Compilation,
    # lazy initialization and allocation have already happened before capture.
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with ffi.use_torch_stream(torch.cuda.stream(stream)):
        for _ in range(WARMUP):
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with ffi.use_torch_stream(torch.cuda.graph(graph, stream=stream)):
        for _ in range(REPEAT):
            call()
    # A stale correct output would hide an empty/wrong-stream capture. Poison
    # the output before each replay, and check repeated reuse, outside timing.
    for _ in range(2):
        output.fill_(float("nan"))
        graph.replay()
        verify_output(torch, output, reference)
    return graph


def measure(mode, call, graph, torch, timer):
    if mode == "eager":
        return timer(call, warmup=WARMUP, repeat=REPEAT)
    if mode != "graph":
        raise ValueError(f"Unknown submission mode: {mode}")
    # Match the ordinary path's ten warmup kernels, then submit one graph
    # containing exactly thirty calls. Do not warm up with ten *graphs*.
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    return timer(graph.replay, warmup=0, repeat=1) / REPEAT


def paired_trials(torch, call, graph, output, reference, timer, directory, phase, limit):
    rows = []
    for trial in range(TRIALS):
        for position, index in enumerate(trial_order(2, trial)):
            mode = MODES[index]
            output.fill_(float("nan"))
            # Warmup can write the sentinel; independent poisoned replay checks
            # in capture_graph establish that the graph itself actually writes.
            value = measure(mode, call, graph, torch, timer)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {mode} timing: {value}")
            row = dict(phase=phase, trial=trial + 1, position=position + 1,
                       mode=mode, ms=value, verified=False)
            rows.append(row)
            write_json(directory / f"{phase}.json", rows)
            verify_output(torch, output, reference)
            row["verified"] = True
            write_json(directory / f"{phase}.json", rows)
            label = ("WITHIN_LIMIT" if value <= limit else "SLOW") if mode == "eager" else "DIAGNOSTIC"
            print(f"{phase} trial {trial+1} / {mode}: {value:.6f} ms {label}", flush=True)
    return rows


def summary(rows, limit):
    values = {mode: [r["ms"] for r in rows if r["mode"] == mode] for mode in MODES}
    if any(len(v) != TRIALS for v in values.values()) or not all(r["verified"] for r in rows):
        raise ValueError("Incomplete or unverified diagnostic samples")
    return dict(
        eager_median_ms=statistics.median(values["eager"]),
        eager_min_ms=min(values["eager"]), eager_max_ms=max(values["eager"]),
        eager_within_limit=sum(v <= limit for v in values["eager"]),
        graph_median_ms=statistics.median(values["graph"]),
        paired_eager_over_graph=statistics.median(
            a / b for a, b in zip(values["eager"], values["graph"])),
        trials=TRIALS, limit_ms=limit,
    )


def snapshot(directory, label, device_uuid):
    with (directory / f"{label}.txt").open("w") as stream:
        subprocess.run(["nvidia-smi", "-i", device_uuid, "-q"], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=15)


def stop_monitor(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def load_telemetry(torch, call, graph, output, reference, directory, device_uuid, seconds):
    # Keep polling completely outside paired timing. These longer load blocks
    # let a 100 ms sampler observe active clocks; they are not grading runs.
    fields = ("timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,"
              "power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory")
    command = ["nvidia-smi", "-i", device_uuid, f"--query-gpu={fields}",
               "--format=csv", "--loop-ms=100"]
    write_json(directory / "telemetry_command.json", command)
    intervals = []
    snapshot(directory, "gpu_before_load", device_uuid)
    with (directory / "telemetry.csv").open("w") as out, (directory / "telemetry.stderr").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        try:
            for index in (0, 1, 1, 0):
                mode = MODES[index]
                print(f"Load telemetry / {mode}: {seconds:g} seconds (diagnostic only)", flush=True)
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                started = utc_now()
                before = time.monotonic()
                batches = 0
                while time.monotonic() - before < seconds:
                    if mode == "graph":
                        graph.replay()
                    else:
                        for _ in range(REPEAT):
                            call()
                    torch.cuda.synchronize()
                    batches += 1
                intervals.append(dict(mode=mode, started=started, finished=utc_now(),
                                      wall_seconds=time.monotonic()-before, batches=batches,
                                      kernels_per_batch=REPEAT))
                write_json(directory / "load_intervals.json", intervals)
                verify_output(torch, output, reference)
                if process.poll() is not None:
                    raise RuntimeError("GPU monitor exited early; see telemetry.stderr")
        finally:
            stop_monitor(process)
    if (directory / "telemetry.csv").stat().st_size == 0:
        raise RuntimeError("GPU monitor produced no samples")
    snapshot(directory, "gpu_after_load", device_uuid)


def run(directory, seconds):
    import torch
    import tvm
    import tvm_ffi
    import gemm_kernels
    from utils import (REFERENCE_TIMES, TIMING_TOLERANCE, blackwell_target,
                       prepare_data, time_cuda_call)

    target = blackwell_target()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_uuid = nvidia_uuid(device.uuid)
    gemm_kernels.SM_COUNT = device.multi_processor_count
    if gemm_kernels.SM_COUNT % 2:
        raise ValueError("Step 10 requires an even SM count")
    meta = dict(run_metadata(), started=utc_now(), host=socket.gethostname(),
                uuid=device_uuid, gpu=device.name, sm_count=gemm_kernels.SM_COUNT,
                target=str(target), tvm=tvm.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, cpu_affinity=sorted(os.sched_getaffinity(0)),
                load_average=os.getloadavg(), warmup=WARMUP, repeat=REPEAT, trials=TRIALS,
                load_seconds=seconds, pid=os.getpid(),
                probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                slurm={key: os.environ.get(key, "") for key in
                       ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_CPUS_PER_NODE",
                        "SLURM_CPUS_PER_TASK", "SLURM_STEP_GPUS")})
    write_json(directory / "run.json", meta)
    print(f"GPU: {device.name}; UUID: {device_uuid}; CPU affinity: {meta['cpu_affinity']}")
    print(f"Code: {meta['git_revision']}; gemm_sha256={meta['gemm_kernels_sha256']}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    shape = (4096, 4096, 4096)
    limit = REFERENCE_TIMES[(10, *shape)] * TIMING_TOLERANCE
    A, B, output = prepare_data(*shape)
    with target:
        with capture_compilation(directory / "compiler"):
            executable = tvm.compile(tvm.IRModule({"main": gemm_kernels.hgemm_v10(*shape)}),
                                     target=target, tir_pipeline="tirx")
            executable.mod(A, B, output)
        reference = torch.mm(A, B.T)
        verify_output(torch, output, reference)
        call = lambda: executable.mod(A, B, output)
        graph = capture_graph(torch, tvm_ffi, call, output, reference)
        print("Verified ordinary output and two poisoned graph replays.", flush=True)
        results = {}
        for phase in ("before_load", "after_load"):
            if phase == "after_load":
                load_telemetry(torch, call, graph, output, reference, directory, device_uuid, seconds)
            rows = paired_trials(torch, call, graph, output, reference, time_cuda_call,
                                 directory, phase, limit)
            results[phase] = summary(rows, limit)
            write_json(directory / "summary.json", results)
            s = results[phase]
            print(f"{phase}: eager median={s['eager_median_ms']:.6f} ms, "
                  f"{s['eager_within_limit']}/{TRIALS} within {limit:.6f}; "
                  f"graph={s['graph_median_ms']:.6f} ms, "
                  f"paired eager/graph={s['paired_eager_over_graph']:.3f}", flush=True)
    print("Graph and load blocks are diagnostic only; this is not full pytest acceptance.")
    return int(any(s["eager_within_limit"] != TRIALS for s in results.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh result directory; generated by default")
    parser.add_argument("--load-seconds", type=float, default=2.0,
                        help="seconds per diagnostic load block (four blocks, default 2)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.load_seconds) or not 1 <= args.load_seconds <= 10:
        parser.error("--load-seconds must be between 1 and 10")
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi and the currently allocated Blackwell GPU are required")
    if args.output is None:
        root = ROOT / "results_b300"
        root.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="step10_runtime.", dir=root))
    else:
        directory = args.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    print(f"结果目录：{directory}", flush=True)
    code = 2
    try:
        code = run(directory, args.load_seconds)
    except Exception:
        message = traceback.format_exc()
        (directory / "error.txt").write_text(message)
        print(message, flush=True)
    finally:
        (directory / "exitcode.txt").write_text(f"{code}\n")
    print(f"结果目录：{directory}；退出码：{code}（0=采集完整且普通样本全部达标，"
          "1=采集完整但存在慢样本，2=采集/校验失败）", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
