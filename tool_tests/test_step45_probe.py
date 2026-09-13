"""Verify the historical probe against its captured TVM-generated baseline."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from probe_step45 import (DEFAULT_VARIANTS, VARIANTS, WAIT_VARIANTS, main, source_experiment,
                          summarize, trial_order, variant_source)


@pytest.fixture(scope="module", params=[4, 5])
def generated(request):
    path = (Path(__file__).parents[1] / "results_b300/step45_probe.PS9CFi/probe" /
            f"step{request.param:02d}_2048_baseline/module_01.cu")
    return request.param, path.read_text()


def test_variants_change_only_the_declared_operation(generated):
    step, source = generated
    assert variant_source(source, step, "baseline") == source

    moved = variant_source(source, step, "early_release")
    release = "    tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_1();\n"
    assert moved.replace(release, "") == source.replace(release, "")
    body = moved.split('extern "C" __global__', 1)[1]
    assert body.index("tvm_builtin_ptx_tcgen05_alloc_cta_group_1(") < body.index(release)
    assert body.index(release) < body.index("tvm_builtin_cuda_cta_sync();")

    shorter = variant_source(source, step, "wait_64ns")
    assert shorter.replace("unsigned int ticks = 64;", "unsigned int ticks = 0x989680;") == source
    rolled = variant_source(source, step, "no_k_unroll")
    assert rolled.replace("      #pragma unroll 1\n", "") == source


@pytest.mark.parametrize("fail", [False, True])
def test_capture_receives_variant_and_restores_compiler(generated, tmp_path, monkeypatch, fail):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    pytest.importorskip("tvm.support.nvcc")
    from benchmark_diagnostics import capture_compilation

    step, source = generated
    name = "tvm_callback_cuda_compile"
    original = tvm_ffi.get_global_func(name)
    seen = []
    binary = bytearray(b"ELF fixture")

    def compile_cuda(code):
        seen.append(str(code))
        if fail:
            raise RuntimeError("probe compile failed")
        return binary

    tvm_ffi.register_global_func(name, compile_cuda, override=True)
    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvcc")
    monkeypatch.setattr("benchmark_diagnostics.dump_binary_resources", lambda path: None)
    try:
        with capture_compilation(tmp_path):
            if fail:
                with pytest.raises(Exception, match="probe compile failed"):
                    with source_experiment(step, "early_release", tmp_path):
                        tvm_ffi.get_global_func(name)(source)
            else:
                with source_experiment(step, "early_release", tmp_path):
                    assert bytes(tvm_ffi.get_global_func(name)(source)) == binary
            assert (tmp_path / "module_01.cu").read_text() == seen[0]
        assert seen == [variant_source(source, step, "early_release")]
        assert (tmp_path / "source_01.before.cu").read_text() == source
        info = json.loads((tmp_path / "experiment.json").read_text())
        assert info["sources"][0]["before_sha256"] != info["sources"][0]["compiled_sha256"]
        # Neither experimental transformation nor capture applies after exit.
        if fail:
            with pytest.raises(Exception, match="probe compile failed"):
                tvm_ffi.get_global_func(name)(source)
        else:
            tvm_ffi.get_global_func(name)(source)
        assert seen[-1] == source
        assert not (tmp_path / "module_02.cu").exists()
    finally:
        tvm_ffi.register_global_func(name, original, override=True)


def test_adopted_variant_is_not_silently_benchmarked_as_a_new_change(generated):
    step, source = generated
    adopted = variant_source(source, step, "early_release")
    with pytest.raises(ValueError, match="early_release is already applied"):
        variant_source(adopted, step, "early_release")


@pytest.mark.parametrize("step", [4, 5])
@pytest.mark.parametrize("variant", WAIT_VARIANTS)
def test_current_1024_wait_probe_preserves_kernel_protocol(step, variant):
    tvm = pytest.importorskip("tvm")
    import gemm_kernels

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        kernel = getattr(gemm_kernels, f"hgemm_v{step}")(1024, 1024, 1024)
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    source = executable.mod.imports[0].inspect_source()
    modified = variant_source(source, step, variant)
    marker = 'extern "C" __global__'
    header, body = modified.split(marker, 1)
    original_header, original_body = source.split(marker, 1)
    if variant in ("wait_64ns", "wait_poll"):
        assert body == original_body  # Every dispatch, phase, fence and CTA sync is unchanged.
    else:
        probe_name = "tvm_probe_ptx_mbarrier_wait_64ns"
        assert body.replace(probe_name, "tvm_builtin_ptx_mbarrier_try_wait") == original_body
        selected = (1,) if step == 4 else (1, 2)
        if variant == "mma_wait_64ns":
            selected = (2,) if step == 4 else (3,)
        expected = 1 if step == 4 else 2
        assert body.count(probe_name + "(") == expected
        for index in selected:
            assert f"{probe_name}((&(((uint64_t*)pool_buf_ptr)[{index}]))" in body
        assert header.startswith(original_header)
    if variant == "wait_poll":
        restored = header.replace(
            "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;",
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;",
        ).replace(':: "r"(barrier_addr_int), "r"(phase) : "memory"',
                  ':: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory"')
        # Reinsert exactly the one removed hint declaration.
        anchor = "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        assert restored.count(anchor) == 1
        restored = restored.replace(anchor, anchor + "    unsigned int ticks = 0x989680;\n")
        assert restored == original_header
        assert '"bra.uni                   LAB_WAIT;' in header
    elif variant == "wait_64ns":
        assert header.replace("unsigned int ticks = 64;", "unsigned int ticks = 0x989680;") == original_header


def test_defaults_target_waits_on_the_current_kernel():
    assert DEFAULT_VARIANTS == ("baseline", *WAIT_VARIANTS)
    assert "early_release" not in DEFAULT_VARIANTS


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_changed_compiler_format_fails_closed(variant):
    with pytest.raises(ValueError):
        variant_source('extern "C" __global__ void different() {}', 4, variant)


def test_interleaved_results_use_same_trial_baseline():
    cases = [dict(step=4, size=2048, variant="baseline", samples_ms=[2, 4, 8]),
             dict(step=4, size=2048, variant="early_release", samples_ms=[1, 8, 4])]
    rows = summarize(cases, {(4, 2048, 2048, 2048): 3}, 1.3)
    # Median of paired speedups [2, 0.5, 2], not ratio of medians (4/4).
    assert rows[1]["paired_speedup"] == 2
    assert rows[1]["median_ms"] == 4 and rows[1]["status"] == "SLOW"
    assert json.loads(rows[1]["samples_ms"]) == [1, 8, 4]
    orders = [trial_order(8, trial) for trial in range(5)]
    assert all(sorted(order) == list(range(8)) for order in orders)
    assert len({tuple(order) for order in orders}) == 5


@pytest.mark.parametrize("argv", [["--trials", "0"], ["--repeat", "0"], ["--warmup", "-1"],
                                  ["--size", "123"], ["--steps", "6"]])
def test_invalid_arguments_fail_before_gpu_imports(tmp_path, argv):
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "new"), *argv])
    assert error.value.code == 2
