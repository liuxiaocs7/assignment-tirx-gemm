"""Replay the measured Step 8 TMEM cache without claiming GPU validation."""

import importlib
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
PROBE = ROOT / "results_b300/profile_guided.6KUfDZ/step08/step08_2048_cache_tmem_base"


def generate(builder, shape, arch="sm_103a"):
    tvm = pytest.importorskip("tvm")
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": builder(*shape)}),
                                 target=target, tir_pipeline="tirx")
    return executable.mod.imports[0].inspect_source()


def body(source):
    return source.split('extern "C" __global__', 1)[1]


def helper(source, name):
    return re.search(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}",
                     source, re.M | re.S).group()


def test_step8_matches_measured_tmem_cache():
    pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    actual = generate(kernels.hgemm_v8, (2048,) * 3)
    recorded = (PROBE / "module_01.cu").read_text()
    assert body(actual) == body(recorded)
    for name in ("tirx_tma_wait_64ns", "tvm_builtin_ptx_mbarrier_try_wait"):
        assert helper(actual, name) == helper(recorded, name)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape", [(1024,) * 3, (4096,) * 3, (8192,) * 3,
                                   (1024, 3072, 64), (1024, 3072, 320)])
def test_step8_replays_measured_cache_across_shapes(shape, arch):
    pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    path = PROBE / "builder.py"
    namespace = dict(vars(kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    recorded = generate(namespace["hgemm_v8"], shape, arch)
    actual = generate(kernels.hgemm_v8, shape, arch)
    assert body(actual).count("tirx_tma_wait_64ns(") == 1
    assert body(actual).count("tvm_builtin_ptx_mbarrier_try_wait(") == 3
    assert body(actual) == body(recorded)
    assert helper(actual, "tirx_tma_wait_64ns") == helper(recorded, "tirx_tma_wait_64ns")
