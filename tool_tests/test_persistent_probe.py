"""Replay recorded compiler inputs and check the new persistent experiments.

These checks cover instrumentation and TVM source generation, not GPU timing.
"""

import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import DEFAULT_STEP_VARIANTS, STEP_VARIANTS, build_variant, main, variant_builder_source, variant_source
from probe_step45 import source_experiment


def body(source):
    return source.split('extern "C" __global__', 1)[1]


def recorded_source(step, size=4096):
    return (ROOT / "results_b300/b300_diag.RA0gpm/compiler" /
            f"step{step:02}_{size}_{size}_{size}/module_01.cu").read_text()


def generate(kernel):
    tvm = pytest.importorskip("tvm")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    return executable.mod.imports[0].inspect_source()


def install_recorded_builder(step, monkeypatch):
    """Replay pre-adoption experiments using the exact measured baseline."""
    import gemm_kernels

    path = (ROOT / "results_b300/persistent_probe.VB42kg/probe" /
            f"step{step:02}_4096_baseline/builder.py")
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    name = f"hgemm_v{step}"
    monkeypatch.setattr(gemm_kernels, name, namespace[name])


@pytest.mark.parametrize("step", STEP_VARIANTS)
@pytest.mark.parametrize("size", [1024, 2048, 4096, 8192])
def test_recorded_mma_wait_changes_only_selected_calls(step, size):
    source = recorded_source(step, size)
    changed = variant_source(source, step, "mma_wait_64ns")
    old_name, new_name = "tvm_builtin_ptx_mbarrier_try_wait", "tvm_probe_mma_wait_64ns"
    assert body(changed).replace(new_name, old_name) == body(source)
    assert body(changed).count(new_name + "(") == (1 if step == 6 else 2)
    assert body(changed).count(old_name + "(") == (1 if step == 6 else 2)
    def helper(code, name):
        return re.search(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}\n",
                         code, re.M | re.S).group()
    assert helper(changed, old_name) == helper(source, old_name)
    restored = helper(changed, new_name).replace(new_name, old_name).replace(
        "unsigned int ticks = 64;", "unsigned int ticks = 0x989680;")
    assert restored == helper(source, old_name)
    with pytest.raises(ValueError, match="already applied"):
        variant_source(changed, step, "mma_wait_64ns")


@pytest.mark.parametrize("step", STEP_VARIANTS)
def test_changed_wait_layout_fails_closed(step):
    source = recorded_source(step).replace("pool_buf_ptr", "moved_pool_buf_ptr")
    with pytest.raises(ValueError):
        variant_source(source, step, "mma_wait_64ns")


@pytest.mark.parametrize("step,variant", [(s, v) for s, vv in STEP_VARIANTS.items() for v in vv])
def test_experiment_lowers_and_preserves_protocol(step, variant, tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels

    install_recorded_builder(step, monkeypatch)
    original = getattr(gemm_kernels, f"hgemm_v{step}")
    directory = tmp_path / variant
    source = generate(build_variant(step, (4096,) * 3, variant, directory))
    source = variant_source(source, step, variant)
    baseline = body(recorded_source(step))
    actual = body(source)
    assert getattr(gemm_kernels, f"hgemm_v{step}") is original
    info = json.loads((directory / "builder.json").read_text())
    assert info["before_sha256"] == hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    assert info["compiled_sha256"] == hashlib.sha256((directory / "builder.py").read_bytes()).hexdigest()

    # Every variant retains allocation lifetime, persistence, and MMA commits.
    for name in ("tcgen05_alloc_cta_group_", "tcgen05_dealloc_cta_group_",
                 "tcgen05_relinquish_alloc_permit_cta_group_", "ptx_tcgen05_commit_"):
        assert actual.count(name) == baseline.count(name)
    assert actual.count("while (") == baseline.count("while (")
    if variant == "baseline":
        assert actual == baseline
    elif variant == "mma_wait_64ns":
        assert actual.replace("tvm_probe_mma_wait_64ns", "tvm_builtin_ptx_mbarrier_try_wait") == baseline
    elif variant == "k_tile_128":
        assert actual.count("k < 32") == 1 and "k < 64" in baseline
        if step == 7:
            assert "k_1 < 32" in actual and "k_1 < 64" in baseline
        assert actual.count("ptx_tcgen05_mma_cta_1_kind_f16_SS(") == 8
        assert baseline.count("ptx_tcgen05_mma_cta_1_kind_f16_SS(") == 4
        assert "65536);" in actual and "32768);" in baseline
        assert actual.count("tvm_builtin_ptx_mbarrier_try_wait(") == baseline.count("tvm_builtin_ptx_mbarrier_try_wait(")
    elif variant == "final_fence":
        before = ("          tvm_builtin_ptx_tcgen05_fence_after_thread_sync();\n"
                  "          tvm_builtin_ptx_tcgen05_fence_before_thread_sync();\n")
        after = before.replace("          ", "        ")
        assert baseline.count(before) == actual.count(after) == 1
        assert actual.replace(after, "") == baseline.replace(before, "")
        # The handoff fence stays before the writeback CTA barrier.
        assert after + "      }\n    }\n    tvm_builtin_cuda_cta_sync();" in actual
    elif variant == "epilogue_128":
        assert baseline.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 2
        assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 1
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 4
        # Read-TMEM completion still signals the free accumulator before TMA.
        assert actual.index("tvm_builtin_ptx_tcgen05_wait_ld();") < actual.index(
            "tvm_builtin_ptx_mbarrier_arrive_shared(") < actual.index("ptx_cp_async_bulk_tensor_shared_to_global_")
    else:
        # Step 10 trades more loads for a smaller FP32 temporary, independently.
        assert "float Dreg_ptr[16]" in actual and "half Dreg_f16_ptr[256]" in actual
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 16
        assert baseline.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 8
        assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 4
        marker = "    alignas(64) float Dreg_ptr["
        # More epilogue loads renumber TVM's CSE temporaries, including ones
        # in the producer. Preserve distinct identifiers and all expressions.
        def canonicalize_cse(code):
            names = {}
            return re.sub(r"\bcse_v\d+\b", lambda m: names.setdefault(
                m[0], f"cse_v{len(names)}"), code)
        assert canonicalize_cse(actual.split(marker)[0]) == canonicalize_cse(baseline.split(marker)[0])


@pytest.mark.parametrize("step", [6, 7])
@pytest.mark.parametrize("K", [64, 192, 384])
def test_k_tile_variant_retains_short_and_odd_phase_support(step, K, tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels

    install_recorded_builder(step, monkeypatch)
    shape = (1024, 3072, K)  # 192 output tiles force persistent CTA reuse.
    actual = generate(build_variant(step, shape, "k_tile_128", tmp_path))
    if K % 128:
        assert body(actual) == body(generate(getattr(gemm_kernels, f"hgemm_v{step}")(*shape)))
    else:
        assert "65536);" in body(actual)
        assert "k < 3" in body(actual)


@pytest.mark.parametrize("step", [6])
def test_adopted_k_tile_is_not_applied_again(step, tmp_path):
    pytest.importorskip("tvm")
    assert DEFAULT_STEP_VARIANTS[step] == ("baseline",)
    with pytest.raises(ValueError, match=f"Step {step} has adopted k_tile_128"):
        build_variant(step, (4096,) * 3, "k_tile_128", tmp_path)


@pytest.mark.parametrize("fail", [False, True])
def test_custom_transform_callback_is_captured_and_restored(tmp_path, fail):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    pytest.importorskip("tvm.support.nvcc")
    key = "tvm_callback_cuda_compile"
    original = tvm_ffi.get_global_func(key)
    source = recorded_source(7)
    seen = []
    def compile_cuda(code):
        seen.append(str(code))
        if fail:
            raise RuntimeError("fixture compiler failure")
        return bytearray(b"ELF fixture")
    tvm_ffi.register_global_func(key, compile_cuda, override=True)
    try:
        if fail:
            with pytest.raises(Exception, match="fixture compiler failure"):
                with source_experiment(7, "mma_wait_64ns", tmp_path, transform=variant_source):
                    tvm_ffi.get_global_func(key)(source)
        else:
            with source_experiment(7, "mma_wait_64ns", tmp_path, transform=variant_source):
                tvm_ffi.get_global_func(key)(source)
        assert seen == [variant_source(source, 7, "mma_wait_64ns")]
        assert (tmp_path / "source_01.before.cu").read_text() == source
        assert json.loads((tmp_path / "experiment.json").read_text())["sources"][0]["compiled_sha256"] == hashlib.sha256(seen[0].encode()).hexdigest()
        if fail:
            with pytest.raises(Exception):
                tvm_ffi.get_global_func(key)(source)
        else:
            tvm_ffi.get_global_func(key)(source)
        assert seen[-1] == source
    finally:
        tvm_ffi.register_global_func(key, original, override=True)


@pytest.mark.parametrize("argv", [["--trials", "0"], ["--repeat", "0"], ["--warmup", "-1"],
                                  ["--size", "123"], ["--steps", "4"],
                                  ["--steps", "10", "--variants", "k_tile_128"]])
def test_invalid_arguments_fail_before_gpu_imports(argv, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "new"), *argv])
    assert error.value.code == 2
