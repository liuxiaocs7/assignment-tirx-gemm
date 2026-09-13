"""Check that the production kernel emits the CUDA that passed the B300 probe.

This replays a measured compiler input, not a replacement for GPU timing tests.
Helper declaration order varies across TVM processes, so compare kernel bodies.
"""

import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("step", [4, 5])
def test_production_matches_measured_early_release(step):
    tvm = pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        kernel = getattr(kernels, f"hgemm_v{step}")(2048, 2048, 2048)
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    actual = executable.mod.imports[0].inspect_source()
    recorded = (ROOT / "results_b300/step45_probe.PS9CFi/probe" /
                f"step{step:02d}_2048_early_release/module_01.cu").read_text()
    marker = 'extern "C" __global__'
    assert actual.split(marker, 1)[1] == recorded.split(marker, 1)[1]
