"""Check diagnostic scope, trace integrity, and generated role instrumentation.

These tests do not validate NVIDIA machine code or GPU profiling overhead.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from profile_persistent import (FIELDS, ROLES, TIMER_SOURCE, build_profile, instrument_builder,
                                layout, main, summarize_traces, trace_records)


def generate(kernel):
    tvm = pytest.importorskip("tvm")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    return ex.mod.imports[0].inspect_source()


def body(source):
    return source.split('extern "C" __global__', 1)[1]


@pytest.mark.parametrize("step,shape", [(8, (2048,) * 3), (10, (4096,) * 3),
                                       (8, (1024, 3072, 320)), (10, (4096, 3072, 64))])
@pytest.mark.parametrize("role", ROLES)
def test_profile_keeps_production_operations_and_unique_output_slot(step, shape, role, tmp_path):
    pytest.importorskip("tvm")
    import gemm_kernels

    builder = getattr(gemm_kernels, f"hgemm_v{step}")
    kernel, info = build_profile(step, shape, role, tmp_path / role, 148)
    generated = generate(kernel)
    actual, original = body(generated), body(generate(builder(*shape)))
    # No change to barriers, async operations, tensor descriptors, or addresses.
    # Local CSE temporary identifiers may be renumbered by adding the trace.
    def hardware_calls(source):
        return [re.sub(r'\bcse_v\d+\b', 'cse_v', line.strip()) for line in source.splitlines()
                if re.match(r'\s*(?:tvm_builtin_|ptx_|tirx_tma_wait_)\w*\(', line)]
    assert hardware_calls(actual) == hardware_calls(original)
    for name in ("start", "finish", "add_wait", "add_work"):
        assert actual.count(f"tirx_profile_{name}(") == 1
    assert actual.count("tirx_profile_add_handoff(") == (role == "mma")
    assert actual.count("tirx_profile_add_epilogue(") == (role == "writeback")
    assert actual.count("int64_t* __restrict__ Profile_ptr") == 2  # prototype and definition
    line = next(s for s in actual.splitlines() if "tirx_profile_finish(" in s)
    assert "tile_scheduler_tile_count_ptr[0]" in line
    if role == "writeback":
        assert "threadIdx.x) % 32) == 0" in line
        if step == 10:
            assert "warp_id_in_cta % 4) == 0" in line
            assert "((warp_id_in_cta >> 2) * 11)" in line
        else:
            assert "warp_id_in_cta == 0" in line
    elif role == "mma" and step == 10:
        assert "((warp_id_in_cta & 3) * 11)" in line
    assert getattr(gemm_kernels, f"hgemm_v{step}") is builder
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_builder((tmp_path / role / "builder.py").read_text(), step, role, info)


def test_timer_helper_is_safe_when_tvm_emits_it_for_each_function(tmp_path):
    clang = shutil.which("clang++")
    if clang is None:
        pytest.skip("C++ preprocessor not available")
    path = tmp_path / "duplicate.cpp"
    path.write_text(TIMER_SOURCE * 7)
    result = subprocess.run([clang, "-E", "-P", str(path)], check=True, capture_output=True, text=True)
    # TVM keys injected sources by called function name. Each group needs a
    # header guard or NVRTC will see multiple definitions in the same file.
    assert result.stdout.count("void tirx_profile_finish(") == 1
    assert result.stdout.count("uint64_t tirx_profile_ns()") == 1
    assert '%%globaltimer' in result.stdout and '%%clock64' in result.stdout


def fixture_trace(info, role):
    data = [[[[-1] * len(FIELDS) for _ in range(info["consumers"])]
             for _ in range(info["max_tiles"])] for _ in range(info["ctas"])]
    for cta in range(info["ctas"]):
        cluster, rank = divmod(cta, info["group"])
        for ordinal in range(info["max_tiles"]):
            index = cluster + ordinal * info["clusters"]
            if index >= info["total_tiles"] or (role == "mma" and rank != 0):
                continue
            for consumer in range(1 if role == "tma" else info["consumers"]):
                m, n = divmod(index, info["n_tiles"])
                data[cta][ordinal][consumer] = [1000, 2000, 200, 300, 100, 200, 1000, 2500, cta, m, n]
    return data


@pytest.mark.parametrize("step,size", [(8, 2048), (10, 4096)])
@pytest.mark.parametrize("role", ROLES)
def test_trace_covers_each_active_cluster_rank_consumer_and_persistent_tile(step, size, role):
    info = layout(step, (size,) * 3, 148)
    rows = trace_records(fixture_trace(info, role), info, role)
    expected = info["total_tiles"] * (1 if role == "mma" else info["group"]) * (1 if role == "tma" else info["consumers"])
    assert len(rows) == expected
    assert all(row["unmeasured_ns"] == 200 for row in rows)
    summaries = summarize_traces([dict(step=step, size=size, role=role, **r) for r in rows])
    assert sum(r["records"] for r in summaries) == expected
    assert all(r["wait_pct"] == 20 and r["work_pct"] == 30 and r["cycles_per_ns"] == 1.5 for r in summaries)
    assert {r["tile_ordinal"] for r in summaries} == {0, 1}


@pytest.mark.parametrize("failure", ["missing", "inactive", "backwards", "overlap", "coords", "duplicate"])
def test_invalid_trace_is_not_reported_as_valid_timing(failure):
    info = layout(10, (4096,) * 3, 148)
    data = fixture_trace(info, "mma")
    if failure == "missing":
        data[0][0][0] = [-1] * len(FIELDS)
    elif failure == "inactive":
        data[1][0][0] = data[0][0][0].copy()
    elif failure == "backwards":
        data[0][0][0][1] = 900
    elif failure == "overlap":
        data[0][0][0][2] = 1100
    elif failure == "coords":
        data[0][0][0][9] = 999
    else:
        data[2][0][0][9:11] = data[0][0][0][9:11]
    with pytest.raises(ValueError):
        trace_records(data, info, "mma")


@pytest.mark.parametrize("args", [["--steps", "7"], ["--size", "512"], ["--repeat", "0"],
                                  ["--trials", "0"], ["--warmup", "-1"]])
def test_invalid_cli_stops_before_importing_gpu_libraries(args, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "new"), *args])
    assert error.value.code == 2
