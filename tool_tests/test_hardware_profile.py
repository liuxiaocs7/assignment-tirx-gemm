"""Exercise the profiler boundary and failure handling without a CUDA device."""

from contextlib import contextmanager, nullcontext
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import profile_hardware as hardware


RECORDED = hardware.ROOT / "results_b300/step10_hardware.Mosdpx/profile"


def test_actual_ncu_2025_raw_export_has_one_launch_and_preserves_units():
    raw = (RECORDED / "raw.csv").read_text()
    checked = hardware.check_raw_report(raw)
    assert checked["launches"] == 1
    assert checked["kernel"] == "kernel_kernel"
    assert checked["numeric_counter_rows"] == 371
    parsed = hardware.parse_raw_report(raw)
    assert parsed["format"] == "wide"
    assert parsed["launch"]["Process ID"] == "2139037"
    assert parsed["launch"]["Grid Size"] == "(128, 1, 1)"
    assert parsed["metrics"]["sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active"] == {
        "value": "91.850102", "unit": "%"}
    assert parsed["metrics"]["gpu__time_duration.avg"] == {"value": "144.800000", "unit": "us"}
    assert parsed["metrics"]["launch__shared_mem_per_block_dynamic"] == {
        "value": "230.400000", "unit": "Kbyte/block"}


def raw_csv(kernel="kernel_kernel", ids=(0,), value="72.5"):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Process ID", "Kernel Name", "Metric Name", "Metric Value"])
    for ident in ids:
        writer.writerow([ident, "123", kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed", value])
    return output.getvalue()


def wide_csv(kernel="kernel_kernel", ids=(0,), value="72.5"):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Process ID", "Kernel Name", "sm__throughput.avg.pct_of_peak_sustained_elapsed"])
    writer.writerow(["", "", "", "%"])
    for ident in ids:
        writer.writerow([ident, "123", kernel, value])
    return output.getvalue()


@pytest.mark.parametrize("raw,match", [
    (wide_csv(kernel="cublas_kernel"), "exactly one"),
    (wide_csv(ids=(0, 1)), "exactly one"),
    (wide_csv(ids=(0, 0)), "one wide launch row"),
    (wide_csv(ids=()), "exactly one"),
    (wide_csv(value="no data"), "no numeric"),
    (wide_csv(value="nan"), "no numeric"),
    (wide_csv().replace(",,,%\r\n", ""), "missing its units"),
    (wide_csv() + "malformed,row\n", "row width"),
    (wide_csv().replace("0,123", ",123"), "invalid launch identity"),
])
def test_wide_units_support_does_not_hide_bad_data(raw, match):
    with pytest.raises(RuntimeError, match=match):
        hardware.parse_raw_report(raw)


def test_long_metric_tables_still_normalize_units_and_reject_conflicts():
    raw = raw_csv(value="1,234.50")
    parsed = hardware.parse_raw_report(raw)
    assert parsed["format"] == "long"
    assert parsed["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"]["value"] == "1,234.50"
    conflicting = raw + raw_csv(value="99").splitlines(True)[1]
    with pytest.raises(RuntimeError, match="conflicting metric"):
        hardware.parse_raw_report(conflicting)


def test_offline_recovery_of_real_report_without_dependencies_or_mutation(tmp_path):
    before = {p.name: p.read_bytes() for p in RECORDED.iterdir() if p.is_file()}
    result = subprocess.run([sys.executable, "-S", str(hardware.ROOT / "profile_hardware.py"),
                             "--analyze", str(RECORDED), "--output", str(tmp_path / "recovered")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    info = json.loads((tmp_path / "recovered/analysis.json").read_text())
    assert info["status"] == "analyzed" and info["report_check"]["launches"] == 1
    assert info["collected"]["status"] == "failed"  # preserve original failure record
    assert info["worker"]["status"] == "verified_after_profile"
    with (tmp_path / "recovered/metrics.csv").open() as handle:
        metrics = {row["metric"]: row for row in csv.DictReader(handle)}
    assert len(metrics) == 795
    assert metrics["gpu__time_duration.avg"]["unit"] == "us"
    assert metrics["sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active"]["value"] == "91.850102"
    assert {p.name: p.read_bytes() for p in RECORDED.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("change", ["unverified", "source_mismatch", "bad_csv"])
def test_offline_recovery_rejects_invalid_evidence(tmp_path, change):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("run.json", "worker.json", "raw.csv"):
        (source / name).write_bytes((RECORDED / name).read_bytes())
    worker = json.loads((source / "worker.json").read_text())
    if change == "unverified":
        worker["status"] = "verified_before_profile"
    elif change == "source_mismatch":
        worker["gemm_kernels_sha256"] = "different"
    else:
        (source / "raw.csv").write_text(wide_csv(ids=(0, 1)))
    (source / "worker.json").write_text(json.dumps(worker))
    assert hardware.main(["--analyze", str(source), "--output", str(tmp_path / "failed")]) == 1
    assert json.loads((tmp_path / "failed/analysis.json").read_text())["status"] == "failed"
    assert not (tmp_path / "failed/metrics.csv").exists()


@pytest.mark.parametrize("failure", [None, "launch", "synchronize"])
def test_worker_profiles_only_warmed_production_launch_and_stops_on_failure(tmp_path, monkeypatch, failure):
    events = []
    state = dict(active=False, launches=0, captured=False)
    A, B = object(), object()
    output = SimpleNamespace(fill_=lambda value: events.append("fill"))

    def launch(*args):
        assert args == (A, B, output)
        state["launches"] += 1
        events.append("profiled_launch" if state["active"] else "launch")
        if state["active"]:
            assert not state["captured"]
            assert state["launches"] == 12  # first compile/verify + ten warmups
            if failure == "launch":
                raise RuntimeError("fixture launch failed")

    def start():
        assert not state["active"] and not state["captured"]
        state["active"] = True
        events.append("start")

    def stop():
        assert state["active"]
        state["active"] = False
        events.append("stop")

    def sync():
        events.append("sync")
        if state["active"] and failure == "synchronize":
            raise RuntimeError("fixture synchronize failed")

    def verify(*args):
        assert args == (output, A, B)
        assert not state["active"] and not state["captured"]
        events.append("verify")

    device = SimpleNamespace(name="B300 fixture", multi_processor_count=148)
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                           get_device_properties=lambda index: device, synchronize=sync,
                           manual_seed_all=lambda seed: None, profiler=SimpleNamespace(start=start, stop=stop))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, manual_seed=lambda seed: None, __version__="fixture", version=SimpleNamespace(cuda="13.0")))
    kernel = object()
    gemm = SimpleNamespace(SM_COUNT=0)

    def builder(*shape):
        assert shape == (4096,) * 3 and gemm.SM_COUNT == 148
        return kernel

    gemm.hgemm_v10 = builder
    monkeypatch.setitem(sys.modules, "gemm_kernels", gemm)

    def compile_kernel(mod, target, tir_pipeline):
        assert mod == {"main": kernel} and tir_pipeline == "tirx" and state["captured"]
        return SimpleNamespace(mod=launch)

    monkeypatch.setitem(sys.modules, "tvm", SimpleNamespace(
        compile=compile_kernel, IRModule=lambda value: value, __version__="fixture"))
    monkeypatch.setitem(sys.modules, "utils", SimpleNamespace(
        blackwell_target=nullcontext, prepare_data=lambda *shape: (A, B, output), verify=verify))
    monkeypatch.setitem(sys.modules, "profile_persistent", SimpleNamespace(dump_sass=lambda path: None))

    @contextmanager
    def capture(path):
        state["captured"] = True
        try:
            yield
        finally:
            state["captured"] = False

    monkeypatch.setattr(hardware, "capture_compilation", capture)
    if failure:
        with pytest.raises(RuntimeError, match=f"fixture {failure} failed"):
            hardware.worker(tmp_path)
        assert events[-1] == "stop"
        assert json.loads((tmp_path / "worker.json").read_text())["status"] != "verified_after_profile"
    else:
        hardware.worker(tmp_path)
        assert events[events.index("start"):events.index("stop") + 1] == ["start", "profiled_launch", "sync", "stop"]
        assert events[events.index("start") - 2:events.index("start")] == ["fill", "sync"]
        assert events.count("fill") == 2
        assert events.count("verify") == 2
        assert json.loads((tmp_path / "worker.json").read_text())["status"] == "verified_after_profile"
    assert not state["active"]


@pytest.fixture
def fake_ncu(tmp_path):
    """A process fixture tests real argv, report paths, and stdout/stderr handling."""
    executable = tmp_path / "ncu fixture with spaces"
    executable.write_text(f"#!{sys.executable}\n" + '''
import csv, io, json, os, pathlib, sys
args = sys.argv[1:]
mode = os.environ.get("TIRX_TEST_NCU_MODE", "modern")
if args == ["--version"]:
    print("Nsight Compute fixture")
elif args == ["--help"]:
    print("--print-details --pipeline-boost-state" if mode != "legacy" else "--details-all")
elif args == ["--list-sections"]:
    print("SpeedOfLight ComputeWorkloadAnalysis SchedulerStats LaunchStats")
elif "--import" in args:
    assert pathlib.Path(args[args.index("--import") + 1]).is_file()
    if "raw" in args:
        writer = csv.writer(sys.stdout)
        if mode == "wide":
            writer.writerow(["ID", "Process ID", "Kernel Name", "sm__throughput.avg.pct_of_peak_sustained_elapsed"])
            writer.writerow(["", "", "", "%"])
            writer.writerow(["0", "123", "kernel_kernel", "72.5"])
        else:
            writer.writerow(["ID", "Process ID", "Kernel Name", "Metric Name", "Metric Value"])
        if mode not in ("empty", "wide"):
            writer.writerow(["0", "123", "kernel_kernel", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "72.5"])
    else:
        assert ("--details-all" if mode == "legacy" else "--print-details") in args
        print("fixture details")
else:
    for key, expected in (("--profile-from-start", "off"), ("--launch-count", "1"),
                          ("--clock-control", "none"), ("--cache-control", "none"),
                          ("--replay-mode", "kernel"), ("--kernel-name", "kernel_kernel")):
        assert args[args.index(key) + 1] == expected
    assert "--force-overwrite" not in args
    if mode == "permission":
        print("ERR_NVGPUCTRPERM: access to GPU Performance Counters denied", file=sys.stderr)
        sys.exit(1)
    directory = pathlib.Path(args[args.index("--output") + 1])
    meta = json.loads((directory / "run.json").read_text())
    meta["status"] = "verified_after_profile" if mode != "unverified" else "preparing"
    if mode == "changed_source":
        meta["gemm_kernels_sha256"] = "changed"
    (directory / "worker.json").write_text(json.dumps(meta))
    report = pathlib.Path(args[args.index("--export") + 1])
    if mode != "no_report":
        report.with_suffix(".ncu-rep" if mode == "legacy" else ".nsight-cuprof").write_bytes(b"fixture report")
    print("fixture collection complete")
''')
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize("mode", ["modern", "legacy", "wide"])
def test_collection_preserves_report_and_exports_with_version_appropriate_flags(tmp_path, fake_ncu, monkeypatch, mode):
    monkeypatch.setenv("TIRX_TEST_NCU_MODE", mode)
    directory = tmp_path / "result with spaces"
    assert hardware.main(["--output", str(directory), "--ncu", str(fake_ncu)]) == 0
    info = json.loads((directory / "run.json").read_text())
    assert info["status"] == "collected" and info["diagnostic_only"]
    assert info["report_check"]["launches"] == 1
    assert info["missing_sections"] == ["MemoryWorkloadAnalysis", "WarpStateStats", "Occupancy"]
    assert info["pipeline_boost_state"] == ("tool default" if mode == "legacy" else "dynamic")
    assert info["report_format"] == ("wide" if mode == "wide" else "long")
    assert "72.5" in (directory / "metrics.csv").read_text()
    assert (directory / info["report"]).read_bytes() == b"fixture report"
    assert (directory / "details.txt").read_text().strip() == "fixture details"
    assert len(info["commands"]) == 6 and all(c["returncode"] == 0 for c in info["commands"])
    # Reusing a directory must preserve its evidence, even if ncu could overwrite.
    with pytest.raises(SystemExit):
        hardware.main(["--output", str(directory), "--ncu", str(fake_ncu)])
    assert json.loads((directory / "run.json").read_text()) == info


@pytest.mark.parametrize("mode,expected", [("permission", "exit 1"), ("empty", "exactly one"),
    ("no_report", "nonempty NCU report"), ("unverified", "verification"), ("changed_source", "source changed")])
def test_incomplete_collection_is_failure_with_evidence(tmp_path, fake_ncu, monkeypatch, mode, expected):
    monkeypatch.setenv("TIRX_TEST_NCU_MODE", mode)
    directory = tmp_path / "failed"
    assert hardware.main(["--output", str(directory), "--ncu", str(fake_ncu)]) == 1
    info = json.loads((directory / "run.json").read_text())
    assert info["status"] == "failed" and expected in info["error"]
    if mode == "permission":
        assert "ERR_NVGPUCTRPERM" in (directory / "ncu.log").read_text()


def test_missing_ncu_fails_before_importing_gpu_dependencies(tmp_path):
    # Launch via Python -S to ensure this path doesn't require torch/TVM installed.
    directory = tmp_path / "missing"
    result = subprocess.run([sys.executable, "-S", str(hardware.ROOT / "profile_hardware.py"),
                             "--output", str(directory), "--ncu", str(tmp_path / "absent")],
                            capture_output=True, text=True)
    assert result.returncode == 1 and "ncu was not found" in result.stderr
    assert json.loads((directory / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("raw", ["", raw_csv(kernel="cublas_kernel"), raw_csv(ids=(0, 1)),
                                 raw_csv(value="n/a"), raw_csv(value="nan")])
def test_report_rejects_missing_wrong_or_unavailable_counters(raw):
    with pytest.raises(RuntimeError):
        hardware.check_raw_report(raw)


def test_section_selection_never_silently_collects_an_empty_set():
    with pytest.raises(RuntimeError, match="lacks"):
        hardware.select_sections("UnknownSection LaunchStats")
    assert hardware.select_sections('"SpeedOfLight","GPU Speed of Light"\n"LaunchStats","Launch Statistics"') == [
        "SpeedOfLight", "LaunchStats"]
