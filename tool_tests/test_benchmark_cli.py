"""CPU-only checks for the benchmark CLI's shape selection and arguments."""

import argparse
import ast
from contextlib import contextmanager, nullcontext
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

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


@pytest.mark.parametrize("trials,reference_ms,status", [
    (4, 10, "PASS"), (3, 2, "SLOW"), (1, None, "UNSCORED"),
])
def test_benchmark_interleaves_cublas_and_saves_paired_trials(
    tmp_path, monkeypatch, trials, reference_ms, status,
):
    """Exercise the CLI with a deterministic timer, without GPU dependencies."""
    a, b, output, reference = object(), SimpleNamespace(T=object()), object(), object()
    events, timed, checks = [], [], []
    capturing = False
    kernel_samples = [2.0, 4.0, 8.0, 16.0][:trials]
    cublas_samples = [4.0, 2.0, 16.0, 8.0][:trials]
    returned = {"kernel": [], "cublas": []}

    def launch_kernel(*args):
        assert args == (a, b, output)
        events.append("kernel")

    def launch_cublas(left, right, *, out):
        assert (left, right, out) == (a, b.T, reference)
        events.append("cublas")

    def check(actual, expected, **kwargs):
        assert (actual, expected) == (output, reference)
        assert kwargs == dict(rtol=1e-3, atol=1e-2)
        checks.append(len(timed))

    def time_call(call, warmup, repeat):
        assert not capturing
        assert (warmup, repeat) == (10, 30)
        assert "verified" in events and "reference_allocated" in events
        call()
        kind = events[-1]
        timed.append(kind)
        values = kernel_samples if kind == "kernel" else cublas_samples
        value = values[len(returned[kind])]
        returned[kind].append(value)
        return value

    @contextmanager
    def capture(directory):
        nonlocal capturing
        directory.mkdir()
        capturing = True
        yield
        capturing = False

    def compile_kernel(*args, **kwargs):
        assert capturing and not timed
        events.append("compiled")
        return SimpleNamespace(mod=launch_kernel)

    def allocate_like(tensor):
        assert tensor is output and not timed
        events.append("reference_allocated")
        return reference

    device = SimpleNamespace(name="test B300", multi_processor_count=148)
    torch = SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (10, 3),
                             current_device=lambda: 0, get_device_properties=lambda _: device,
                             manual_seed_all=lambda _: None),
        manual_seed=lambda _: None, empty_like=allocate_like, mm=launch_cublas,
        testing=SimpleNamespace(assert_close=check),
    )
    utils = SimpleNamespace(
        REFERENCE_TIMES=({(10, 4096, 4096, 4096): reference_ms}
                         if reference_ms is not None else {}), TIMING_TOLERANCE=1.3,
        blackwell_target=nullcontext, prepare_data=lambda *args: (a, b, output),
        time_cuda_call=time_call, verify=lambda *args: events.append("verified"),
    )
    for name, module in {
        "torch": torch,
        "tvm": SimpleNamespace(__version__="test", IRModule=lambda value: value, compile=compile_kernel),
        "gemm_kernels": SimpleNamespace(hgemm_v10=lambda *args: object()),
        "utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(benchmark, "capture_compilation", capture)
    monkeypatch.setattr(benchmark, "run_metadata", lambda: dict(
        git_revision="test", git_dirty=False, gemm_kernels_sha256="test",
        compiler="nvrtc", ptxas_reg_level="10"))
    path = tmp_path / "timing.csv"
    diagnostics = tmp_path / "compiler"
    assert benchmark.main(["--steps", "10", "--sizes", "4096", "--trials", str(trials),
                           "--csv", str(path), "--diagnostics-dir", str(diagnostics)]) == (status == "SLOW")
    assert events.count("compiled") == 1
    assert timed == (["kernel", "cublas", "cublas", "kernel"] * 2)[:2 * trials]
    assert checks[0] == 0 and checks[-1] == 2 * trials
    with path.open() as stream:
        row, = csv.DictReader(stream)
    assert ast.literal_eval(row["samples_ms"]) == kernel_samples
    assert ast.literal_eval(row["cublas_samples_ms"]) == cublas_samples
    orders = ([["kernel", "cublas"], ["cublas", "kernel"]] * 2)[:trials]
    assert ast.literal_eval(row["trial_orders"]) == orders
    ratios = [2.0, 0.5, 2.0, 0.5][:trials]
    assert ast.literal_eval(row["cublas_speedup_samples"]) == ratios
    assert float(row["paired_speedup_vs_cublas"]) == statistics.median(ratios)
    assert float(row["speedup_vs_cublas"]) == statistics.median(cublas_samples) / statistics.median(kernel_samples)
    assert float(row["median_ms"]) == statistics.median(kernel_samples)
    assert row["status"] == status
    saved = json.loads((diagnostics / "step10_4096_4096_4096/timing.json").read_text())
    assert saved["trial_orders"] == orders
    assert saved["samples_ms"] == kernel_samples
    assert saved["cublas_speedup_samples"] == ratios
