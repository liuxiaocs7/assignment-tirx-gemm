"""Hardware comparison provenance and real CLI failure paths, without a GPU."""

import csv
import json
import subprocess
import sys

import pytest

import profile_hardware as hardware
import profile_step10_layout as layout
from test_hardware_profile import fake_ncu  # noqa: F401
from test_step8_adoption import body


def test_metric_discovery_requires_full_exact_names():
    listing = ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed\n"
               '"sm__pipe_tc_cycles_active.avg","cycles"\n'
               "smsp__inst_executed.sum\n")
    assert hardware.select_layout_metrics(listing) == [
        "sm__pipe_tc_cycles_active.avg", "smsp__inst_executed.sum"]


@pytest.mark.parametrize("variant", hardware.VARIANTS)
def test_profile_compiles_same_kernel_as_measured_experiment(tmp_path, variant):
    tvm = pytest.importorskip("tvm")
    import gemm_kernels
    assert gemm_kernels.SM_COUNT == 148
    before = gemm_kernels.hgemm_v10
    kernel = hardware.profile_kernel(variant, tmp_path)
    with tvm.target.Target({"kind": "cuda", "arch": "sm_103a"}) as target:
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    actual = executable.mod.imports[0].inspect_source()
    recorded = (hardware.ROOT / "results_b300/step10_granularity.nVKp6r/step10_4096" /
                f"step10_4096_{variant}/module_01.cu").read_text()
    assert body(actual) == body(recorded)
    assert gemm_kernels.hgemm_v10 is before
    with pytest.raises(ValueError, match="unsupported"):
        hardware.profile_kernel("tmem_k32_depth8", tmp_path)


@pytest.fixture
def collected(tmp_path, fake_ncu, monkeypatch):
    monkeypatch.setenv("TIRX_TEST_NCU_MODE", "layout")
    output = tmp_path / "profile result"
    # -S demonstrates the collection coordinator doesn't import TVM/PyTorch.
    result = subprocess.run([sys.executable, "-S", str(hardware.ROOT / "profile_step10_layout.py"),
                             "--output", str(output), "--ncu", str(fake_ncu)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_real_cli_abba_collection_export_and_offline_recovery(collected, tmp_path):
    info = json.loads((collected / "comparison.json").read_text())
    assert info["status"] == "compared" and info["order"] == list(layout.ORDER)
    assert [p["variant"] for p in info["collections"]] == list(layout.ORDER)
    assert all(p["exitcode"] == 0 for p in info["collections"])
    with (collected / "comparison.csv").open() as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    row = rows["sm__pipe_tc_cycles_active.avg"]
    assert float(row["sw64_over_baseline_pair1"]) == float(row["sw64_over_baseline_pair2"]) == 1.28
    assert float(row["baseline_last_over_first"]) == float(row["sw64_second_over_first"]) == 1
    assert rows["dram__bytes.sum"]["status"] == "unavailable_or_unit_mismatch"
    assert rows["dram__bytes.sum"]["sw64_over_baseline_pair1"] == ""
    first = json.loads((collected / "01_baseline/run.json").read_text())
    assert "dram__bytes.sum" in first["missing_layout_metrics"]
    assert "smsp__inst_executed.sum" in first["unavailable_layout_metrics"]
    before = {p.relative_to(collected): p.read_bytes() for p in collected.rglob("*") if p.is_file()}
    assert layout.main(["--analyze", str(collected), "--output", str(tmp_path / "analysis")]) == 0
    assert (tmp_path / "analysis/comparison.csv").read_bytes() == (collected / "comparison.csv").read_bytes()
    with pytest.raises(SystemExit):
        layout.main(["--analyze", str(collected), "--output", str(collected)])
    assert before == {p.relative_to(collected): p.read_bytes() for p in collected.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change,match", [
    ("uuid", "gpu_uuid"), ("source", "collection/worker mismatch"),
    ("both_sources", "gemm_kernels_sha256"), ("unverified", "verification"),
    ("variant", "unexpected variant"), ("options", "compiler settings"),
    ("binary", "compiler artifacts differ"), ("missing_report", "missing NCU report"),
    ("metrics", "exactly one"),
])
def test_refuse_mixed_or_invalid_evidence(collected, tmp_path, change, match):
    directory = collected / "04_baseline"
    worker_path = directory / "worker.json"
    worker = json.loads(worker_path.read_text())
    if change == "uuid":
        worker["gpu_uuid"] = "different GPU"
    elif change in ("source", "both_sources"):
        worker["gemm_kernels_sha256"] = "different source"
        if change == "both_sources":
            path = directory / "run.json"
            run = json.loads(path.read_text())
            run["gemm_kernels_sha256"] = worker["gemm_kernels_sha256"]
            path.write_text(json.dumps(run))
    elif change == "unverified":
        worker["status"] = "verified_before_profile"
    elif change == "variant":
        worker["variant"] = "tmem_k64_sw64"
    elif change == "options":
        (directory / "compiler/nvrtc_01.options.json").write_text('["different"]')
    elif change == "binary":
        (directory / "compiler/module_01.cubin").write_bytes(b"changed")
    elif change == "missing_report":
        (directory / "step10_4096.nsight-cuprof").unlink()
    else:
        path = directory / "raw.csv"
        path.write_text(path.read_text().replace("kernel_kernel", "another_kernel"))
    worker_path.write_text(json.dumps(worker))
    output = tmp_path / "failed"
    assert layout.main(["--analyze", str(collected), "--output", str(output)]) == 1
    info = json.loads((output / "comparison.json").read_text())
    assert info["status"] == "failed" and match in info["error"]
    assert not (output / "comparison.csv").exists()


def test_missing_units_zero_and_nan_never_produce_misleading_ratios():
    def row(values, units=("cycle",) * 4):
        profiles = [dict(parsed=dict(metrics={"m": dict(value=v, unit=u)})) for v, u in zip(values, units)]
        return layout.metric_row("m", profiles)
    assert row(["0", "10", "10", "0"])["sw64_over_baseline_pair1"] == ""
    assert row(["100", "nan", "128", "100"])["status"] == "unavailable_or_unit_mismatch"
    assert row(["100", "128", "128", "100"], ("cycle", "Kcycle", "cycle", "cycle"))[
        "sw64_over_baseline_pair2"] == ""


def test_permission_error_stops_without_skipping_failed_profile(tmp_path, fake_ncu, monkeypatch):
    monkeypatch.setenv("TIRX_TEST_NCU_MODE", "permission")
    directory = tmp_path / "denied"
    assert layout.main(["--output", str(directory), "--ncu", str(fake_ncu)]) == 1
    info = json.loads((directory / "comparison.json").read_text())
    assert info["status"] == "failed" and len(info["collections"]) == 1
    assert "ERR_NVGPUCTRPERM" in (directory / "01_baseline/ncu.log").read_text()
    assert not (directory / "comparison.csv").exists()

