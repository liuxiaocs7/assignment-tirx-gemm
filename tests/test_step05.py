import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import prepare_data, compile_and_run, verify, check_timing
from gemm_kernels import hgemm_v5


@pytest.mark.parametrize("size", [512, 1024, 2048, 4096])
def test_pipeline(size):
    M, N, K = size, size, size
    kernel = hgemm_v5(M, N, K)
    A, B, C = prepare_data(M, N, K)
    C_tir = compile_and_run(kernel, A, B, C)
    verify(C_tir, A, B)
    check_timing(kernel, step=5, M=M, N=N, K=K)


@pytest.mark.parametrize("K", [64, 192, 320])
def test_pipeline_short_and_odd_k(K):
    """Exercise prefetch shorter than the pipeline and incomplete ring traversals."""
    M, N = 256, 384
    kernel = hgemm_v5(M, N, K)
    A, B, C = prepare_data(M, N, K)
    verify(compile_and_run(kernel, A, B, C), A, B)
