"""Validate profile-guided experiments against actual B300 compiler inputs."""

import csv
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import build_variant, variant_source
from profile_persistent import layout, trace_records, summarize_traces

PROFILE = ROOT / "results_b300/stage_profile.LPt75W/profile"


def body(source):
    return source.split('extern "C" __global__', 1)[1]


def generate(kernel):
    tvm = pytest.importorskip("tvm")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        result = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    return result.mod.imports[0].inspect_source()


@pytest.mark.parametrize("step,shape", [(8, (2048,) * 3), (8, (1024, 3072, 64)),
                                       (8, (1024, 3072, 320)), (10, (4096,) * 3),
                                       (10, (4096, 3072, 64)), (10, (4096, 3072, 320))])
def test_cached_tmem_base_is_published_before_snapshot_and_keeps_all_operations(step, shape, tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels

    # Replay the pre-adoption Step 8 builder; Step 10 is still experimental.
    if step == 8:
        path = ROOT / "results_b300/profile_guided.6KUfDZ/step08/step08_2048_baseline/builder.py"
        namespace = dict(vars(gemm_kernels))
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        monkeypatch.setattr(gemm_kernels, "hgemm_v8", namespace["hgemm_v8"])
    baseline = body(generate(getattr(gemm_kernels, f"hgemm_v{step}")(*shape)))
    actual = body(generate(build_variant(step, shape, "cache_tmem_base", tmp_path)))
    load = "uint mma_tmem_base = ((uint*)pool_buf_ptr)[0];"
    assert actual.count(load) == 1
    sync = "tvm_builtin_cuda_cluster_sync();" if step == 10 else "tvm_builtin_cuda_cta_sync();"
    assert actual.index(sync) < actual.index(load) < actual.index("while (1)")
    calls = [line.strip() for line in actual.splitlines() if re.match(r"\s*ptx_tcgen05_mma_", line)]
    assert len(calls) == 4
    assert all("mma_tmem_base" in line and "((uint*)pool_buf_ptr)[0]" not in line for line in calls)
    # Re-substitute the immutable value: every barrier, MMA operand, TMEM
    # read, TMA address and final deallocation must match production exactly.
    def operations(code):
        return [re.sub(r"\bcse_v\d+\b", "cse_v", line.strip()) for line in code.splitlines()
                if re.match(r"\s*(?:tvm_builtin_|ptx_|tirx_tma_wait_)\w*\(", line)]
    assert operations(actual.replace("mma_tmem_base", "((uint*)pool_buf_ptr)[0]")) == operations(baseline)


@pytest.mark.parametrize("step,variant,count", [(8, "reuse_wait_64ns", 1),
                                               (10, "reuse_wait_64ns", 1), (10, "ring_wait_64ns", 2)])
def test_wait_controls_replay_measured_cuda_and_leave_writeback_waits_intact(step, variant, count):
    size = 2048 if step == 8 else 4096
    source = (PROFILE / f"step{step:02}_{size}/baseline/module_01.cu").read_text()
    changed = variant_source(source, step, variant)
    old, new = "tvm_builtin_ptx_mbarrier_try_wait", f"tvm_probe_{variant}"
    assert body(changed).replace(new, old) == body(source)
    assert body(changed).count(new + "(") == count
    assert f"{new}((&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 5)]))" in changed
    if variant == "ring_wait_64ns":
        assert f"{new}((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 1)]))" in changed
    def helper(code, name):
        return re.search(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}\n",
                         code, re.M | re.S).group()
    assert helper(changed, old) == helper(source, old)
    assert helper(changed, new).replace(new, old).replace(
        "unsigned int ticks = 64;", "unsigned int ticks = 0x989680;") == helper(source, old)
    if step == 8:
        assert helper(changed, "tirx_tma_wait_64ns") == helper(source, "tirx_tma_wait_64ns")
    with pytest.raises(ValueError, match="already applied"):
        variant_source(changed, step, variant)
    with pytest.raises(ValueError):
        variant_source(source.replace("tma_phase_stage_ptr", "changed_stage"), step, variant)


@pytest.mark.parametrize("step,size", [(8, 2048), (10, 4096)])
@pytest.mark.parametrize("role", ["tma", "mma", "writeback"])
def test_uploaded_trace_reconstructs_complete_tile_coverage_and_published_summary(step, size, role):
    info = layout(step, (size,) * 3, 148)
    rows = []
    fields = ("start_ns", "end_ns", "wait_ns", "work_ns", "handoff_ns", "epilogue_ns",
              "start_cycles", "end_cycles", "sm_id", "tile_m", "tile_n")
    for trial in range(1, 6):
        saved = json.loads((PROFILE / f"step{step:02}_{size}/{role}_trial{trial:02}.json").read_text())
        values = [[[[-1] * len(fields) for _ in range(info["consumers"])]
                   for _ in range(info["max_tiles"])] for _ in range(info["ctas"])]
        for row in saved:
            slot = values[row["cta"]][row["tile_ordinal"]][row["consumer"]]
            assert slot == [-1] * len(fields)
            slot[:] = [row[f] for f in fields]
        restored = trace_records(values, info, role)
        assert restored == saved
        rows.extend(dict(step=step, size=size, role=role, **r) for r in restored)
    expected = [r for r in csv.DictReader((PROFILE / "stages.csv").open())
                if r["step"] == str(step) and r["role"] == role]
    assert len(expected) == len(summarize_traces(rows))
    for actual, expected_row in zip(summarize_traces(rows), expected):
        for key, value in actual.items():
            assert value == expected_row[key] if isinstance(value, str) else value == pytest.approx(float(expected_row[key]))
