"""Validate the cluster cache candidate before numerical/timing checks on B300."""

import inspect
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (build_variant, select_variants, summarize_with_cache_control,
                              variant_builder_source, variant_source)

VARIANT = "cluster_cache_tmem_base"
RECORDED = ROOT / "results_b300/step9_cache.1WDeei"
BOUNDARIES = ((256, 256, 64), (4096, 3072, 64), (4096, 3072, 192),
              (4096, 3072, 256), (4096, 3072, 320), (2304, 2304, 320))


@pytest.fixture
def pre_cache_step9(monkeypatch):
    """Historical transforms keep the original measured builder after adoption."""
    pytest.importorskip("tvm")
    import gemm_kernels

    path = RECORDED / "step9_4096/step09_4096_baseline/builder.py"
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    monkeypatch.setattr(gemm_kernels, "hgemm_v9", namespace["hgemm_v9"])


def test_step9_cache_keeps_baseline_and_separate_direct_control():
    from probe_persistent import VERIFICATION_SHAPES

    assert select_variants(9) == ["baseline"]
    assert select_variants(9, [VARIANT]) == ["baseline", VARIANT]
    assert VERIFICATION_SHAPES[VARIANT] == BOUNDARIES
    cases = [dict(step=9, size=4096, variant=name, samples_ms=samples)
             for name, samples in [("baseline", [0.10, 0.12, 0.11]),
                                   (VARIANT, [0.095, 0.11, 0.10])]]
    rows = summarize_with_cache_control(cases, {(9, 4096, 4096, 4096): 0.116}, 1.3)
    assert rows[1]["comparison_control"] == "baseline"
    assert rows[1]["paired_control_speedup"] == pytest.approx(0.12 / 0.11)
    for step in (6, 7, 8, 10):
        with pytest.raises(ValueError, match="does not support"):
            select_variants(step, [VARIANT])


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape", [(size,) * 3 for size in (1024, 2048, 4096, 8192)]
                                 + list(BOUNDARIES))
def test_cache_hoists_base_after_cluster_sync_and_preserves_protocol(
    arch, shape, tmp_path, monkeypatch, pre_cache_step9,
):
    tvm = pytest.importorskip("tvm")
    import gemm_kernels
    from tvm.backend.cuda.tile_primitive.copy_async import tma

    original = gemm_kernels.hgemm_v9
    production_source = (ROOT / "gemm_kernels.py").read_bytes()
    plans = []
    emit = tma._emit_plan

    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)

    monkeypatch.setattr(tma, "_emit_plan", record)

    def generate(variant):
        directory = tmp_path / variant
        kernel = build_variant(9, shape, variant, directory)
        script = kernel.script()
        plans.clear()
        with tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
            executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        source = executable.mod.imports[0].inspect_source()
        assert variant_source(source, 9, variant) == source
        fields = ("global_dims", "global_strides", "box_dims", "element_strides",
                  "swizzle", "payload_bits", "transaction_bits", "smem_base_offset")
        descriptors = {p.spec.descriptor_name: tuple(str(getattr(p.spec, f)) for f in fields)
                       for p in plans}
        return script, source.split('extern "C" __global__', 1)[1], descriptors

    base_script, baseline, base_maps = generate("baseline")
    script, actual, maps = generate(VARIANT)
    assert maps == base_maps
    assert set(maps) == {"A", "B", "D"}
    for text in (script, base_script):
        assert '"tirx.dyn_smem_bytes": T.int64(148480)' in text
        clusters = min(74, (shape[0] // 256) * (shape[1] // 256))
        assert f"T.cta_id([{clusters * 2}])" in text

    load = "uint mma_tmem_base = ((uint*)pool_buf_ptr)[0];"
    assert actual.count(load) == 1
    assert (actual.index("tcgen05_alloc_cta_group_2(")
            < actual.index("tvm_builtin_cuda_cta_sync();")
            < actual.index("tvm_builtin_cuda_cluster_sync();")
            < actual.index(load) < actual.index("while (1)"))
    calls = lambda code: [line.strip() for line in code.splitlines()
                          if re.match(r"\s*ptx_tcgen05_mma_", line)]
    assert len(calls(baseline)) == len(calls(actual)) == 4
    assert all("((uint*)pool_buf_ptr)[0]" in line for line in calls(baseline))
    assert all("mma_tmem_base" in line and "((uint*)pool_buf_ptr)[0]" not in line
               for line in calls(actual))

    # Substitute back the immutable snapshot. The complete CUDA body must
    # match, including coordinates, phase transitions and all hardware calls.
    restored = re.sub(r"^ +" + re.escape(load) + r"\n", "", actual, flags=re.M)
    restored = restored.replace("mma_tmem_base", "((uint*)pool_buf_ptr)[0]")
    assert restored == baseline
    assert actual.rindex("tvm_builtin_cuda_cluster_sync();") < actual.index("tcgen05_dealloc_cta_group_2(")
    assert gemm_kernels.hgemm_v9 is original
    assert (ROOT / "gemm_kernels.py").read_bytes() == production_source


def test_step9_cache_refuses_reapplication_and_unpublished_allocation(pre_cache_step9):
    pytest.importorskip("tvm")
    import gemm_kernels

    source = inspect.getsource(gemm_kernels.hgemm_v9)
    changed = variant_builder_source(source, 9, VARIANT)
    with pytest.raises(ValueError, match="adopted"):
        variant_builder_source(changed, 9, VARIANT)
    for before, after in [("        T.cuda.cluster_sync()\n        tmem =", "        tmem ="),
                          ("    CTA_GROUP = 2\n", "    CTA_GROUP = 1\n"),
                          ("    PIPE_DEPTH = 4\n", "    PIPE_DEPTH = 2\n")]:
        assert source.count(before) == 1
        with pytest.raises(ValueError, match="Step 9 TMEM cache requires"):
            variant_builder_source(source.replace(before, after), 9, VARIANT)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape", [(size,) * 3 for size in (1024, 2048, 4096, 8192)]
                                 + list(BOUNDARIES))
def test_production_replays_measured_step9_cache(arch, shape, monkeypatch):
    """Compare production with independent recorded GPU compiler inputs/maps."""
    tvm = pytest.importorskip("tvm")
    import gemm_kernels
    from tvm.backend.cuda.tile_primitive.copy_async import tma

    if len(set(shape)) == 1:
        size = shape[0]
        directory = RECORDED / f"step9_{size}/step09_{size}_{VARIANT}"
    else:
        directory = RECORDED / "step9_4096/verification" / f"{VARIANT}_{'_'.join(map(str, shape))}"
    path = directory / "builder.py"
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    fields = ("global_dims", "global_strides", "box_dims", "element_strides",
              "swizzle", "payload_bits", "transaction_bits", "smem_base_offset")
    plans = []
    emit = tma._emit_plan

    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)

    monkeypatch.setattr(tma, "_emit_plan", record)

    def generate(builder):
        plans.clear()
        kernel = builder(*shape)
        assert '"tirx.dyn_smem_bytes": T.int64(148480)' in kernel.script()
        with tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
            executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        source = executable.mod.imports[0].inspect_source().split('extern "C" __global__', 1)[1]
        maps = {p.spec.descriptor_name: tuple(str(getattr(p.spec, f)) for f in fields)
                for p in plans}
        return source, maps

    actual, maps = generate(gemm_kernels.hgemm_v9)
    expected, expected_maps = generate(namespace["hgemm_v9"])
    assert actual == expected
    assert maps == expected_maps and set(maps) == {"A", "B", "D"}
    if arch == "sm_103a":
        assert actual == (directory / "module_01.cu").read_text().split('extern "C" __global__', 1)[1]


def test_production_step9_refuses_reapplying_adopted_cache(tmp_path):
    pytest.importorskip("tvm")
    assert select_variants(9) == ["baseline"]
    with pytest.raises(ValueError, match="Step 9 has adopted.*cache"):
        build_variant(9, (4096,) * 3, VARIANT, tmp_path)
