"""Production dispatch must reproduce the B300-measured compiler inputs."""

from pathlib import Path

import pytest

from test_step8_adoption import body, generate

ROOT = Path(__file__).parents[1]
RECORDED = ROOT / "results_b300/step10_tmem_sizes.I9nGIJ"


@pytest.mark.parametrize("size,variant", [(1024, "n128_epi32"), (2048, "n128_epi32"),
                                        (4096, "n128_tmem_double_buffer"), (8192, "baseline")])
def test_production_replays_each_measured_winner(size, variant, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels
    from tvm.backend.cuda.tile_primitive.copy_async import tma

    plans = []
    emit = tma._emit_plan
    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)
    monkeypatch.setattr(tma, "_emit_plan", record)
    actual = generate(gemm_kernels.hgemm_v10, (size,) * 3)
    directory = RECORDED / f"step10_{size}/step10_{size}_{variant}"
    assert body(actual) == body((directory / "module_01.cu").read_text())
    # Host-side tensor maps do not appear in the CUDA body. Compare them too.
    fields = ("global_dims", "global_strides", "box_dims", "element_strides",
              "swizzle", "payload_bits", "transaction_bits", "smem_base_offset")
    def descriptors():
        return {p.spec.descriptor_name: tuple(str(getattr(p.spec, f)) for f in fields)
                for p in plans}
    actual_descriptors = descriptors()
    plans.clear()
    path = directory / "builder.py"
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    generate(namespace["hgemm_v10"], (size,) * 3)
    assert set(actual_descriptors) == {"A", "B", "D"}
    assert actual_descriptors == descriptors()


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape,variant", [
    ((512, 256, 64), "n128_epi32"),
    ((1024, 3072, 320), "n128_epi32"),
    # Just below/above a single wave on 148 SMs, then the narrow-area cutoff.
    ((512, 9472, 192), "n128_epi32"),
    ((512, 9728, 192), "n128_tmem_double_buffer"),
    ((4096, 4096, 320), "n128_tmem_double_buffer"),
    ((4096, 4352, 64), "baseline"),
    ((4096, 4352, 320), "baseline"),
    ((4608, 2560, 320), "n128_tmem_double_buffer"),
    *[((4096, 3072, k), "n128_tmem_double_buffer") for k in (64, 192, 256, 320)],
])
def test_dispatch_preserves_measured_protocol_at_boundaries(arch, shape, variant):
    pytest.importorskip("tvm")
    import gemm_kernels

    path = RECORDED / f"step10_4096/step10_4096_{variant}/builder.py"
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    actual = generate(gemm_kernels.hgemm_v10, shape, arch)
    expected = generate(namespace["hgemm_v10"], shape, arch)
    assert body(actual) == body(expected)


@pytest.mark.parametrize("variant", ["n128_epi32", "n128_tmem_double_buffer", "mma_unroll4"])
def test_adoption_requires_formal_validation_not_reapplying_historical_probe(variant, tmp_path):
    pytest.importorskip("tvm")
    from probe_persistent import build_variant, select_variants
    assert select_variants(10) == ["baseline"]
    with pytest.raises(ValueError, match="Step 10 has adopted.*narrow N"):
        build_variant(10, (4096,) * 3, variant, tmp_path)
