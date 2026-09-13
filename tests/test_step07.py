import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import prepare_data, compile_and_run, verify, check_timing
from gemm_kernels import hgemm_v7


@pytest.mark.parametrize("size", [1024, 2048, 4096, 8192])
def test_warp_spec(size):
    M, N, K = size, size, size
    kernel = hgemm_v7(M, N, K)
    A, B, C = prepare_data(M, N, K)
    C_tir = compile_and_run(kernel, A, B, C)
    verify(C_tir, A, B)
    check_timing(kernel, step=7, M=M, N=N, K=K)


@pytest.mark.parametrize("K", [128, 192, 384])
def test_warp_spec_odd_k_across_tiles(K):
    """Persist one/three 128-wide stages and the odd 64-wide fallback across tiles."""
    M, N = 1024, 3072
    kernel = hgemm_v7(M, N, K)
    A, B, C = prepare_data(M, N, K)
    verify(compile_and_run(kernel, A, B, C), A, B)
