"""Compare original/batched MMA emission at the same requested unroll factor."""

from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (VERIFICATION_SHAPES, build_variant, select_variants,
                              summarize_with_cache_control, variant_source)
from test_mma_batch_probe import batch_helper, BATCH, OP
from test_persistent_probe import body

RECORDED = ROOT / 'results_b300/step10_mma_batch.13Q2qL/step10'


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES['mma_unroll4'], (512, 256, 64)])
def test_mma_unroll_control_changes_only_pragma_and_four_calls(arch, shape, tmp_path):
    tvm = pytest.importorskip('tvm')
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    kernels = [build_variant(10, shape, v, tmp_path / v)
               for v in ('baseline', 'mma_unroll4', 'mma_batch_unroll4')]
    assert kernels[0].script() == kernels[1].script() == kernels[2].script()
    with target:
        ex = tvm.compile(tvm.IRModule({'main': kernels[0]}), target=target, tir_pipeline='tirx')
    source = ex.mod.imports[0].inspect_source()
    # Replay all three actual B300 compiler inputs before checking the new
    # requested factor. This ensures the comparison starts from the same code.
    if shape == (4096,) * 3:
        for v in ('baseline', 'mma_batch', 'mma_batch_no_unroll'):
            assert body(variant_source(source, 10, v)) == body(
                (RECORDED / f'step10_4096_{v}/module_01.cu').read_text())
    original = variant_source(source, 10, 'mma_unroll4')
    batch = variant_source(source, 10, 'mma_batch_unroll4')
    if shape[2] == 64:
        assert original == source
        assert batch == variant_source(source, 10, 'mma_batch')
    else:
        hint = re.compile(r'^ +#pragma unroll 4\n(?= +for \(int k_1 =)', re.M)
        assert len(hint.findall(original)) == len(hint.findall(batch)) == 1
        assert hint.sub('', original) == source
        assert hint.sub('', batch) == variant_source(source, 10, 'mma_batch')
        for v in ('mma_unroll4', 'mma_batch_unroll4'):
            with pytest.raises(ValueError, match='already applied'):
                variant_source(original, 10, v)
    old_calls = re.findall(rf'^ +{OP}\([^\n]+\);$', body(original), re.M)
    new_calls = re.findall(rf'^ +{BATCH}\([^\n]+\);$', body(batch), re.M)
    assert len(old_calls) == 4 and len(new_calls) == 1
    assert batch.replace(batch_helper(batch), '').replace(new_calls[0], '\n'.join(old_calls)) == original


def test_fixed_factor_uses_original_emission_as_direct_control():
    variants = ['baseline', 'mma_unroll4', 'mma_batch_unroll4']
    assert select_variants(10) == ['baseline']
    assert select_variants(10, ['mma_batch_unroll4']) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'mma_unroll4']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2]
    assert all(r['status'] == 'SLOW' for r in rows)
