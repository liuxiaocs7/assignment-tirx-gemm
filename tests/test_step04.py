import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import prepare_data, compile_and_run, verify, check_timing
from gemm_kernels import hgemm_v4


@pytest.mark.parametrize("size", [256, 512, 1024, 2048])
def test_tma_async(size):
    M, N, K = size, size, size
    kernel = hgemm_v4(M, N, K)
    A, B, C = prepare_data(M, N, K)
    C_tir = compile_and_run(kernel, A, B, C)
    verify(C_tir, A, B)
    check_timing(kernel, step=4, M=M, N=N, K=K)


@pytest.mark.parametrize("K", [64, 128, 192, 384])
def test_tma_async_short_and_odd_k(K):
    """One/three barrier rounds for both K tile widths, with rectangular output."""
    M, N = 256, 384
    A, B, C = prepare_data(M, N, K)
    kernel = hgemm_v4(M, N, K)
    verify(compile_and_run(kernel, A, B, C), A, B)
