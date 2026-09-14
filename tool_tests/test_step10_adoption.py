"""Replay the measured Step 10 B-first winner; GPU timing stays external."""

import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from test_step8_adoption import body, generate

RECORDED = ROOT / "results_b300/step10_roles.rx5lNL/step10/step10_4096_tma_b_first"


def test_step10_matches_winning_b300_compiler_input():
    pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    actual = generate(kernels.hgemm_v10, (4096,) * 3)
    assert body(actual) == body((RECORDED / "module_01.cu").read_text())


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape", [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                   (4096, 3072, 64), (4096, 3072, 320),
                                   (512, 256, 64)])
def test_step10_replays_measured_builder_for_other_shapes(arch, shape):
    pytest.importorskip("tvm")
    kernels = importlib.import_module("gemm_kernels")
    path = RECORDED / "builder.py"
    namespace = dict(vars(kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    recorded = generate(namespace["hgemm_v10"], shape, arch)
    actual = generate(kernels.hgemm_v10, shape, arch)
    assert body(actual) == body(recorded)


@pytest.mark.parametrize("sm_count", [132, 148])
@pytest.mark.parametrize("shape", [(512, 256, 64), (4096, 3072, 320),
                                   (4096,) * 3, (8192,) * 3])
def test_profiler_grid_matches_adopted_kernel_and_covers_all_tiles(sm_count, shape, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels
    from profile_persistent import layout
    monkeypatch.setattr(gemm_kernels, "SM_COUNT", sm_count)
    info = layout(10, shape, sm_count)
    script = gemm_kernels.hgemm_v10(*shape).script()
    assert f'T.cta_id([{info["ctas"]}])' in script
    tiles = [tile for cluster in range(info["clusters"])
             for tile in range(cluster, info["total_tiles"], info["clusters"])]
    assert sorted(tiles) == list(range(info["total_tiles"]))
    old_max = (info["total_tiles"] + sm_count // 2 - 1) // (sm_count // 2)
    assert info["max_tiles"] == old_max
    assert info["clusters"] <= sm_count // 2


@pytest.mark.parametrize("variant", ["cache_tmem_base", "balanced_clusters",
                                     "cache_balanced_clusters", "balanced_fused_a",
                                     "tma_b_first"])
def test_adopted_step10_experiments_require_production_validation(variant, tmp_path):
    pytest.importorskip("tvm")
    from probe_persistent import build_variant, select_variants
    assert variant not in select_variants(10)
    with pytest.raises(ValueError, match="Step 10 has adopted"):
        build_variant(10, (4096,) * 3, variant, tmp_path)
