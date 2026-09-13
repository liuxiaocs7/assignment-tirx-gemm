"""Check that the production kernel emits the CUDA that passed the B300 probe.

This replays a measured compiler input, not a replacement for GPU timing tests.
Helper declaration order varies across TVM processes, so compare kernel bodies.
"""

import importlib
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def generated_source(step, size):
    tvm = pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        kernel = getattr(kernels, f"hgemm_v{step}")(size, size, size)
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    return executable.mod.imports[0].inspect_source()


def helper(source, name):
    return re.search(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}",
                     source, re.M | re.S).group()


def test_step4_production_matches_measured_early_release():
    actual = generated_source(4, 2048)
    recorded = (ROOT / "results_b300/step45_probe.PS9CFi/probe" /
                "step04_2048_early_release/module_01.cu").read_text()
    marker = 'extern "C" __global__'
    assert actual.split(marker, 1)[1] == recorded.split(marker, 1)[1]


def test_step5_production_matches_measured_mma_wait():
    actual = generated_source(5, 1024)
    recorded = (ROOT / "results_b300/wait1024.UP24Tv/probe" /
                "step05_1024_mma_wait_64ns/module_01.cu").read_text()
    recorded = recorded.replace("tvm_probe_ptx_mbarrier_wait_64ns", "tirx_mma_wait_64ns")
    marker = 'extern "C" __global__'
    assert actual.split(marker, 1)[1] == recorded.split(marker, 1)[1]
    # Declaration order varies, but both retry/acquire helper bodies must match.
    for name in ("tirx_mma_wait_64ns", "tvm_builtin_ptx_mbarrier_try_wait"):
        assert helper(actual, name) == helper(recorded, name)
