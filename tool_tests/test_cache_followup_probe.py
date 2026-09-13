"""Verify cached Step 10 controls against measured CUDA and existing protocols."""

from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import build_variant, variant_source, summarize_with_cache_control
from test_persistent_probe import body, generate, canonicalize_codegen_locals, pre_adoption_step10

RECORDED = ROOT / "results_b300/profile_guided.6KUfDZ/step10/step10_4096_cache_tmem_base"


def expand_cse(code):
    # TVM emits different CSE temporaries after hoisting the base. Expand each
    # one at its uses so differences in operands cannot hide behind numbering.
    declarations = re.findall(r"^ +(?:int|uint|uint64_t) (cse_v\d+) = ([^;]+);\n", code, re.M)
    for name, expression in declarations:
        code = re.sub(rf"^ +(?:int|uint|uint64_t) {name} = [^;]+;\n", "", code, flags=re.M)
        code = re.sub(rf"\b{name}\b", lambda _: expression, code)
    return canonicalize_codegen_locals(code)


def test_cache_control_replays_actual_b300_compiler_input(tmp_path, pre_adoption_step10):
    pytest.importorskip("tvm")
    actual = generate(build_variant(10, (4096,) * 3, "cache_tmem_base", tmp_path))
    assert body(actual) == body((RECORDED / "module_01.cu").read_text())


@pytest.mark.parametrize("variant,control", [("cache_unroll_ring", "unroll_ring"),
                                            ("cache_balanced_clusters", "balanced_clusters")])
@pytest.mark.parametrize("K", [64, 320, 768, 4096])
def test_cached_combinations_keep_existing_protocol_and_partial_ring(variant, control, K, tmp_path, pre_adoption_step10):
    pytest.importorskip("tvm")
    shape = (4096, 3072, K)
    original = body(generate(build_variant(10, shape, control, tmp_path / "control")))
    actual = body(generate(build_variant(10, shape, variant, tmp_path / "combined")))
    snapshot = "  uint mma_tmem_base = ((uint*)pool_buf_ptr)[0];\n"
    assert actual.count(snapshot) == 1
    assert actual.index("tvm_builtin_cuda_cluster_sync();") < actual.index(snapshot) < actual.index("while (1)")
    # Substitute only the cached immutable load, then compare the entire
    # generated body, including phase updates, coordinates and launch stride.
    restored = actual.replace(snapshot, "").replace("mma_tmem_base", "((uint*)pool_buf_ptr)[0]")
    assert expand_cse(restored) == expand_cse(original)
    if variant == "cache_unroll_ring" and K % 256:
        cached = body(generate(build_variant(10, shape, "cache_tmem_base", tmp_path / "cache")))
        assert actual == cached


def test_mma_no_unroll_changes_only_measured_mma_loop():
    source = (RECORDED / "module_01.cu").read_text()
    changed = variant_source(source, 10, "cache_mma_no_unroll")
    original = "              for (int k_1 = 0; k_1 < 64; ++k_1) {"
    patched = "              #pragma unroll 1\n" + original
    assert changed.count(patched) == 1
    assert changed.replace(patched, original) == source
    with pytest.raises(ValueError, match="already applied"):
        variant_source(changed, 10, "cache_mma_no_unroll")
    for bad in (source.replace("int k_1 =", "int renamed ="),
                source.replace("uint mma_tmem_base =", "uint renamed_base =")):
        with pytest.raises(ValueError):
            variant_source(bad, 10, "cache_mma_no_unroll")


@pytest.mark.parametrize("size", [1024, 4096, 8192])
def test_mma_no_unroll_builds_cached_input_and_preserves_all_operands(size, tmp_path, pre_adoption_step10):
    pytest.importorskip("tvm")
    original = generate(build_variant(10, (size,) * 3, "cache_tmem_base", tmp_path / "control"))
    source = generate(build_variant(10, (size,) * 3, "cache_mma_no_unroll", tmp_path / "experiment"))
    assert source == original
    actual = variant_source(source, 10, "cache_mma_no_unroll")
    restored = re.sub(r"^ +#pragma unroll 1\n(?= +for \(int k_1 =)", "", actual, flags=re.M)
    assert restored == original


def test_paired_cache_summary_uses_same_trial_and_keeps_original_grading():
    cases = [dict(step=10, size=4096, variant=name, samples_ms=samples) for name, samples in
             [("baseline", [10, 30, 60]), ("cache_tmem_base", [8, 10, 40]),
              ("cache_unroll_ring", [4, 8, 10])]]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 5}, 1.3)
    assert rows[2]["paired_speedup"] == 3.75
    assert rows[2]["paired_cache_speedup"] == 2  # Not ratio of medians: 10 / 8.
    assert rows[1]["paired_cache_speedup"] == 1
    assert all(r["status"] == "SLOW" and r["limit_ms"] == 6.5 for r in rows)
    without_cache = summarize_with_cache_control(cases[:1], {(10, 4096, 4096, 4096): 5}, 1.3)
    assert without_cache[0]["paired_cache_speedup"] is None
    cases[-1]["samples_ms"].pop()
    with pytest.raises(ValueError, match="matching trials"):
        summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 5}, 1.3)
