"""Collect Nsight Compute counters for production Step 10 / 4096.

This is a diagnostic run, not a performance test. Compilation, verification
and ten warmups precede a single cudaProfilerStart/Stop-delimited GEMM launch.
NCU kernel replay can perturb execution even with clock/cache control disabled.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from benchmark_diagnostics import capture_compilation, run_metadata, write_json


SECTIONS = ("SpeedOfLight", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis",
            "SchedulerStats", "WarpStateStats", "LaunchStats", "Occupancy")
ROOT = Path(__file__).resolve().parent


def metadata():
    return dict(run_metadata(ROOT), step=10, shape=[4096, 4096, 4096], seed=0,
                warmup=10, diagnostic_only=True,
                profile_hardware_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def find_ncu(requested=None):
    candidates = ([requested] if requested else
                  ["ncu", str(Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin/ncu")])
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return str(Path(executable).absolute())
    raise RuntimeError("Nsight Compute ncu was not found; use --ncu /path/to/ncu "
                       "or load the site's Nsight Compute module. No counters collected.")


def select_sections(listing):
    available = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", listing))
    selected = [section for section in SECTIONS if section in available]
    if not {"SpeedOfLight", "LaunchStats"}.issubset(selected):
        raise RuntimeError("ncu lacks SpeedOfLight/LaunchStats; see ncu_sections.txt. "
                           "Use a Nsight Compute version supporting this GPU.")
    return selected


def collection_command(ncu, directory, sections, help_text):
    command = [ncu, "--profile-from-start", "off", "--launch-count", "1",
               "--kernel-name-base", "function", "--kernel-name", "kernel_kernel",
               "--replay-mode", "kernel", "--clock-control", "none", "--cache-control", "none"]
    # Newer NCU versions otherwise request stable Tensor Core boosting. Older
    # versions lack this switch; retain their behavior and record their help.
    if "--pipeline-boost-state" in help_text:
        command += ["--pipeline-boost-state", "dynamic"]
    for section in sections:
        command += ["--section", section]
    return command + ["--export", str(directory / "step10_4096"), sys.executable,
                      "-u", str(Path(__file__).resolve()), "--_worker", "--output", str(directory)]


def profile_once(call, cuda):
    """No setup, verification, allocation or other kernel inside this range."""
    cuda.synchronize()
    cuda.profiler.start()
    try:
        call()
        cuda.synchronize()
    finally:
        cuda.profiler.stop()


def worker(directory):
    import torch
    import tvm
    import gemm_kernels
    from profile_persistent import dump_sass
    from utils import blackwell_target, prepare_data, verify

    info = metadata()
    info["status"] = "preparing"
    write_json(directory / "worker.json", info)
    if not torch.cuda.is_available():
        raise RuntimeError("a Blackwell GPU and CUDA-enabled PyTorch are required")
    target = blackwell_target()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    gemm_kernels.SM_COUNT = device.multi_processor_count
    info.update(target=str(target), gpu=device.name, sm_count=device.multi_processor_count,
                tvm=tvm.__version__, torch=torch.__version__, cuda=torch.version.cuda)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    A, B, output = prepare_data(4096, 4096, 4096)
    output.fill_(float("nan"))
    kernel = gemm_kernels.hgemm_v10(4096, 4096, 4096)
    compiler_dir = directory / "compiler"
    with target, capture_compilation(compiler_dir):
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        executable.mod(A, B, output)
    verify(output, A, B)
    dump_sass(compiler_dir)
    info["status"] = "verified_before_profile"
    write_json(directory / "worker.json", info)
    print(f"Verified production Step 10 / 4096; GPU: {device.name}; SMs: {device.multi_processor_count}",
          flush=True)
    call = lambda: executable.mod(A, B, output)
    for _ in range(10):
        call()
    # A post-profile verification must not accept output left by the warmups.
    # This fill and its synchronization both remain outside the capture range.
    output.fill_(float("nan"))
    profile_once(call, torch.cuda)
    verify(output, A, B)
    torch.cuda.synchronize()
    info["status"] = "verified_after_profile"
    write_json(directory / "worker.json", info)
    print("Verified profiled output. Profiler durations are not grading measurements.", flush=True)


def check_raw_report(raw):
    """Fail on an empty report, a wrong kernel, or unavailable hardware metrics."""
    lines = raw.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith(('"ID",', 'ID,'))), None)
    if start is None:
        raise RuntimeError("NCU raw export has no kernel metrics header")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    launches = {(row.get("Process ID"), row.get("ID"), row.get("Kernel Name")) for row in rows}
    if len(launches) != 1 or next(iter(launches))[2] != "kernel_kernel":
        raise RuntimeError("expected exactly one kernel_kernel result in NCU raw export")
    counters = [row for row in rows
                if re.match(r"(?:sm|smsp|dram|lts|tpc|tc)__", row.get("Metric Name", ""))
                and re.fullmatch(r"[-+0-9.,eE]+", row.get("Metric Value", ""))]
    if not counters:
        raise RuntimeError("NCU report has no numeric hardware counters; inspect ncu.log/raw.stderr.txt")
    return dict(kernel="kernel_kernel", launches=1, numeric_counter_rows=len(counters))


def collect(directory, requested_ncu=None):
    directory.mkdir(parents=True, exist_ok=False)
    info = metadata()
    info.update(status="collecting", replay_mode="kernel", clock_control="none", cache_control="none",
                replay_caveat="No cache flush: metric values may differ between replay passes. "
                              "Profiling perturbs execution; no PASS/SLOW scoring.", commands=[])
    write_json(directory / "run.json", info)

    def run(command, name, stream=False):
        record = dict(argv=command)
        info["commands"].append(record)
        write_json(directory / "run.json", info)
        if stream:
            with (directory / name).open("w") as log, subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            output = (directory / name).read_text()
        else:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            code, output = result.returncode, result.stdout
            (directory / name).write_text(output)
            (directory / f"{Path(name).stem}.stderr.txt").write_text(result.stderr)
        record["returncode"] = code
        write_json(directory / "run.json", info)
        if code:
            raise RuntimeError(f"ncu command failed (exit {code}); see {directory / name}")
        return output

    try:
        ncu = find_ncu(requested_ncu)
        info["ncu"] = ncu
        info["ncu_version"] = run([ncu, "--version"], "ncu_version.txt").strip()
        help_text = run([ncu, "--help"], "ncu_help.txt")
        sections = select_sections(run([ncu, "--list-sections"], "ncu_sections.txt"))
        info.update(sections=sections, missing_sections=[s for s in SECTIONS if s not in sections],
                    pipeline_boost_state="dynamic" if "--pipeline-boost-state" in help_text else "tool default")
        print(f"{info['ncu_version']}\nSections: {', '.join(sections)}", flush=True)
        if info["missing_sections"]:
            print(f"Unavailable sections: {', '.join(info['missing_sections'])}", flush=True)
        print("Collecting production Step 10 / 4096. Diagnostic counters only; no timing score.", flush=True)
        run(collection_command(ncu, directory, sections, help_text), "ncu.log", stream=True)
        child = json.loads((directory / "worker.json").read_text())
        if child["status"] != "verified_after_profile":
            raise RuntimeError("profile worker did not finish output verification")
        for key in ("gemm_kernels_sha256", "utils_sha256", "benchmark_diagnostics_sha256", "profile_hardware_sha256"):
            if child[key] != info[key]:
                raise RuntimeError(f"source changed during collection: {key}")
        reports = [p for p in (directory / "step10_4096.ncu-rep", directory / "step10_4096.nsight-cuprof",
                               directory / "step10_4096") if p.is_file() and p.stat().st_size]
        if len(reports) != 1:
            raise RuntimeError("expected one nonempty NCU report; see ncu.log (no kernels or unsupported GPU)")
        report = reports[0]
        info["report"] = report.name
        raw = run([ncu, "--import", str(report), "--page", "raw", "--csv",
                   "--print-kernel-base", "function"], "raw.csv")
        info["report_check"] = check_raw_report(raw)
        detail_flag = "--print-details" if "--print-details" in help_text else "--details-all"
        detail_args = [detail_flag, "all"] if detail_flag == "--print-details" else [detail_flag]
        run([ncu, "--import", str(report), "--page", "details", *detail_args], "details.txt")
        info["status"] = "collected"
        write_json(directory / "run.json", info)
        print(f"Saved {report}, raw.csv, details.txt and compiler artifacts. No performance verdict.", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        info.update(status="failed", error=str(error))
        write_json(directory / "run.json", info)
        print(f"Hardware profiling failed: {error}\nArtifacts retained at {directory}.", file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh directory; must not exist")
    parser.add_argument("--ncu", help="Nsight Compute executable; default: PATH, then CUDA_PATH/bin/ncu")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    directory = args.output.resolve()
    if args._worker:
        worker(directory)
        return 0
    if directory.exists():
        parser.error("--output must be a fresh directory")
    return collect(directory, args.ncu)


if __name__ == "__main__":
    raise SystemExit(main())
