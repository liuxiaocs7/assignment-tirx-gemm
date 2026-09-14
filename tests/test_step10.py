import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import prepare_data, compile_and_run, verify, check_timing
from gemm_kernels import hgemm_v10


@pytest.mark.parametrize("size", [1024, 2048, 4096, 8192])
def test_multi_consumer(size):
    M, N, K = size, size, size
    kernel = hgemm_v10(M, N, K)
    A, B, C = prepare_data(M, N, K)
    C_tir = compile_and_run(kernel, A, B, C)
    verify(C_tir, A, B)
    check_timing(kernel, step=10, M=M, N=N, K=K)


@pytest.mark.parametrize("K", [64, 320])
def test_multi_consumer_rectangular_reuse(K):
    # 192 narrow tiles / 64 clusters on 148 SMs exercise persistent slot reuse.
    M, N = 4096, 3072
    kernel = hgemm_v10(M, N, K)
    A, B, C = prepare_data(M, N, K)
    verify(compile_and_run(kernel, A, B, C), A, B)


@pytest.mark.parametrize("M,N,K", [
    (512, 9472, 192),   # 74 narrow tiles: one TMEM slot on 148 SMs.
    (512, 9728, 192),   # 76 narrow tiles: two TMEM slots, partial L2 group.
    (4096, 4352, 64),   # Just beyond the narrow-area cutoff; short K, wide N.
    (4096, 4352, 320),  # Wide N with a partial input ring across output tiles.
    (4608, 2560, 320),  # 9 M tiles: partial L2 group plus persistent reuse.
])
def test_multi_consumer_dispatch_boundaries(M, N, K):
    """Numerical coverage for dispatch branches; no new performance thresholds."""
    import tvm
    from utils import blackwell_target

    kernel = hgemm_v10(M, N, K)
    A, B, C = prepare_data(M, N, K)
    target = blackwell_target()
    with target:
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        for _ in range(2):
            C.fill_(float("nan"))
            executable.mod(A, B, C)
            verify(C, A, B)
