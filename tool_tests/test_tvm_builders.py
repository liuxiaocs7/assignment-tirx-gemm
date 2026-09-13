"""Catch TIRx API/lowering/codegen errors without launching CUDA kernels.

This stops at CUDA source generation; NVRTC/PTX assembly and GPU execution
are exercised by tests/test_step*.py on Blackwell.
"""

import importlib
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[1]))
tvm = pytest.importorskip("tvm")


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("step", range(1, 11))
def test_kernel_builds_and_lowers_on_tvm026(step, arch):
    kernels = importlib.import_module("gemm_kernels")
    shape = (128, 128, 64) if step == 1 else (128, 128, 192) if step == 2 else (1024, 1024, 320)
    kernel = getattr(kernels, f"hgemm_v{step}")(*shape)
    assert isinstance(kernel, tvm.tirx.PrimFunc)
    assert_cuda_source(kernel, arch)


@pytest.mark.parametrize("step,M,N", [(5, 256, 384), (6, 1024, 3072), (7, 1024, 3072),
                                     (8, 1024, 3072), (9, 2048, 3072), (10, 4096, 3072)])
@pytest.mark.parametrize("K", [64, 320])
def test_short_and_partial_pipeline_builds(step, M, N, K):
    kernels = importlib.import_module("gemm_kernels")
    kernel = getattr(kernels, f"hgemm_v{step}")(M, N, K)
    assert_cuda_source(kernel, "sm_103a")


def assert_cuda_source(kernel, arch):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    source = executable.mod.imports[0].inspect_source()
    assert "__global__" in source
    assert "tcgen05" in source
