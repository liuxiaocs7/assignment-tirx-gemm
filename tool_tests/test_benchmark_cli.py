"""CPU-only checks for the benchmark CLI's shape selection and arguments."""

import argparse
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("gemm_benchmark", Path(__file__).parents[1] / "benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_step_selection_preserves_order_and_deduplicates():
    assert benchmark.parse_steps("9,7,9,10") == [9, 7, 10]
    assert benchmark.parse_steps("all") == list(range(1, 11))


@pytest.mark.parametrize("value", ["", "0", "11", "1,x", "1,"])
def test_invalid_step_selection(value):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_steps(value)


def test_default_shapes_follow_reference_table():
    references = {(1, 128, 128, 64): 0.1, (2, 128, 128, 512): 0.8, (2, 128, 128, 64): 0.1}
    assert benchmark.select_shapes(2, None, references) == [(128, 128, 512), (128, 128, 64)]


def test_small_steps_keep_single_tile_dimensions():
    assert benchmark.select_shapes(1, [64], {}) == [(128, 128, 64)]
    assert benchmark.select_shapes(2, [64, 512, 64], {}) == [(128, 128, 64), (128, 128, 512)]
    with pytest.raises(ValueError):
        benchmark.select_shapes(1, [4096], {})


@pytest.mark.parametrize("step,size", [(2, 65), (3, 64), (9, 128), (10, 256), (10, -512)])
def test_unsupported_shapes_fail_before_gpu_imports(step, size):
    with pytest.raises(ValueError):
        benchmark.select_shapes(step, [size], {})


def test_cluster_shapes_are_supported():
    assert benchmark.select_shapes(9, [256, 1024], {}) == [(256, 256, 256), (1024, 1024, 1024)]
    assert benchmark.select_shapes(10, [512], {}) == [(512, 512, 512)]


@pytest.mark.parametrize("argv", [["--repeat", "0"], ["--trials", "0"], ["--warmup", "-1"],
                                  ["--steps", "all", "--sizes", "4096"]])
def test_invalid_cli_fails_without_torch_or_tvm(argv):
    with pytest.raises(SystemExit) as error:
        benchmark.main(argv)
    assert error.value.code == 2
