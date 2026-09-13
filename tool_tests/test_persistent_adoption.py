"""Compare production kernels to the measured B300 K-tile experiment.

CUDA source equivalence catches adoption errors; GPU tests still validate other
shapes and persistent barrier phases on the target device.
"""

import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
PROBE = ROOT / "results_b300/persistent_probe.VB42kg/probe"


def source_body(builder, shape, arch="sm_103a"):
    tvm = pytest.importorskip("tvm")
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": builder(*shape)}),
                                 target=target, tir_pipeline="tirx")
    return executable.mod.imports[0].inspect_source().split('extern "C" __global__', 1)[1]


@pytest.mark.parametrize("step", [6])
def test_production_matches_measured_wider_k_tile(step):
    pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    recorded = (PROBE / f"step{step:02}_4096_k_tile_128/module_01.cu").read_text()
    assert source_body(getattr(kernels, f"hgemm_v{step}"), (4096,) * 3) == recorded.split(
        'extern "C" __global__', 1)[1]


@pytest.mark.parametrize("step", [6])
@pytest.mark.parametrize("K", [64, 128, 192, 384])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_short_k_matches_recorded_builder(step, K, arch, tmp_path):
    pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    # Reuse the measured builder for the 128 path and the old production
    # builder for the 64 path. Do not derive the expected code from production.
    name = "builder.before.py" if K % 128 else "builder.py"
    recorded = (PROBE / f"step{step:02}_4096_k_tile_128" / name).read_text()
    path = tmp_path / "recorded.py"
    path.write_text(recorded)
    namespace = dict(vars(kernels))
    exec(compile(recorded, str(path), "exec"), namespace)
    shape = (1024, 3072, K)  # 192 output tiles, more than 148 persistent CTAs.
    actual = source_body(getattr(kernels, f"hgemm_v{step}"), shape, arch)
    assert actual == source_body(namespace[f"hgemm_v{step}"], shape, arch)
