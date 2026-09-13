"""Check the three-stage/epilogue tradeoff before asking B300 to time it."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (build_variant, DEPTH3_VERIFY_SHAPES, select_variants,
                              summarize_with_cache_control)
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body, pre_adoption_step10

RECORDED = ROOT / 'results_b300/cache_step8_step10.FwqbDd/step10'


def generate(kernel, arch):
    import tvm
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    with target:
        result = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
    return body(result.mod.imports[0].inspect_source())


def test_balanced_control_replays_uploaded_compiler_input(tmp_path, pre_adoption_step10):
    pytest.importorskip('tvm')
    actual = generate(build_variant(10, (4096,) * 3, 'cache_balanced_clusters', tmp_path), 'sm_103a')
    expected = body((RECORDED / 'step10_4096_cache_balanced_clusters/module_01.cu').read_text())
    assert actual == expected


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [*DEPTH3_VERIFY_SHAPES, (4096,) * 3])
def test_depth3_preserves_ring_handoff_and_wider_epilogue_preserves_producers(arch, shape, tmp_path, pre_adoption_step10):
    pytest.importorskip('tvm')
    sources = []
    for variant, epi, dynamic_bytes, stores in [('balanced_depth3', 64, 181248, 4),
                                              ('balanced_depth3_epi128', 128, 214016, 2)]:
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        assert 'Asmem = T.decl_buffer((3, 2, 128, 64)' in script
        assert 'Bsmem = T.decl_buffer((3, 128, 64)' in script
        assert f'Dsmem = T.decl_buffer((2, 128, {epi})' in script
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic_bytes})' in script
        actual = generate(kernel, arch)
        sources.append(actual)
        assert actual.count('ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(') == 3
        assert actual.count('ptx_tcgen05_mma_cta_2_kind_f16_SS(') == 4
        assert '98304, 0, actual_pred_ptr[0]);' in actual
        # Three full and three empty slots. Both consumers must finish before
        # overwriting each stage; every stage is primed before the CTA sync.
        for slot, arrivals in [(1, 1), (2, 1), (3, 1), (4, 2), (5, 2), (6, 2),
                               (7, 1), (8, 1), (9, 256), (10, 256)]:
            call = f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), {arrivals});'
            assert actual.count(call) == 1
            assert actual.index(call) < actual.index('tvm_builtin_cuda_cta_sync();')
        for phase in ('tma', 'mma'):
            stage = f'{phase}_phase_stage_ptr[0]'
            parity = f'{phase}_phase_phase_ptr[0]'
            assert actual.count(f'{stage} = 0;') == 2  # Init + wrap, no tile reset.
            assert actual.count(f'{stage} == 3') == 1
            assert actual.count(f'{parity} = ({parity} ^ 1);') == 1
            assert actual.index(f'{stage} = 0;') < actual.index('while (1)')
        for slot, phase in [('(tma_phase_stage_ptr[0] + 4)', 'tma'),
                            ('(mma_phase_stage_ptr[0] + 1)', 'mma'),
                            ('((warp_id_in_cta & 3) + 9)', 'ld'),
                            ('((warp_id_in_cta >> 2) + 7)', 'wb')]:
            assert (f'tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[{slot}])), '
                    f'({phase}_phase_phase_ptr[0] ^ 0));') in actual
        assert ('ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)'
                '[(mma_phase_stage_ptr[0] + 4)])), 3);') in actual
        assert actual.count('tvm_builtin_ptx_tcgen05_wait_ld();') == 8
        store_op = f'ptx_cp_async_bulk_tensor_shared_to_global_{3 if epi == 128 else 2}d('
        assert actual.count(store_op) == stores
        assert actual.count('ptx_cp_async_bulk_wait_group_read_0();') == stores
        assert actual.count('tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));') == stores * 2
        assert actual.rindex('tvm_builtin_ptx_tcgen05_wait_ld();') < actual.index(
            'tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(') < actual.index(
            store_op)
        clusters = 48 if shape[1] == 3072 else 64
        assert f'T.cta_id([{clusters * 2}])' in script
        assert actual.count(f'_ptr_2[0] = (_ptr_2[0] + {clusters});') == 3
    # Widening the epilogue leaves every TMA load, MMA operand, phase and
    # scheduler operation in both producers identical to the depth3 control.
    marker = '    alignas(64) float Dreg_ptr['
    assert expand_cse(sources[0].split(marker)[0]) == expand_cse(sources[1].split(marker)[0])
    # Each output chunk advances exactly EPI_N columns while keeping the same
    # consumer row and cluster rank; total output is still 128 x 256 per CTA/group.
    for actual, epi in zip(sources, (64, 128)):
        store_lines = [line for line in actual.splitlines()
                       if 'ptx_cp_async_bulk_tensor_shared_to_global_' in line]
        for i, line in enumerate(store_lines):
            if epi == 64:
                column = '(_ptr_1[0] * 256)' if i == 0 else f'((_ptr_1[0] * 256) + {i * epi})'
                assert f', {column}, (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256))' in line
            else:
                # 128 columns span two 64-column swizzle atoms. The TMA map
                # represents this as (64, M, N/64), not as a 2-D box.
                column = '(_ptr_1[0] * 4)' if i == 0 else '((_ptr_1[0] * 4) + 2)'
                assert f', 0, (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256))' in line
                assert line.endswith(f', {column});')


def test_selecting_wider_epilogue_includes_each_direct_control():
    expected = ['baseline', 'cache_tmem_base', 'cache_balanced_clusters',
                'balanced_depth3', 'balanced_depth3_epi128']
    assert select_variants(10, ['balanced_depth3_epi128']) == expected
    assert select_variants(10, ['balanced_depth3_epi128', 'balanced_depth3']) == expected
    assert select_variants(10, ['cache_mma_no_unroll']) == ['baseline', 'cache_tmem_base', 'cache_mma_no_unroll']
    with pytest.raises(ValueError):
        select_variants(8, ['balanced_depth3'])


def test_direct_control_summary_does_not_credit_epilogue_for_depth_change():
    variants = select_variants(10, ["balanced_depth3_epi128"])
    samples = [[12, 24, 48], [10, 20, 40], [8, 10, 40], [4, 8, 10], [2, 2, 5]]
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in zip(variants, samples)]
    refs = {(10, 4096, 4096, 4096): 1}
    rows = summarize_with_cache_control(cases, refs, 1.3)
    assert rows[-2]['comparison_control'] == 'cache_balanced_clusters'
    assert rows[-2]['paired_control_speedup'] == 2
    assert rows[-1]['comparison_control'] == 'balanced_depth3'
    assert rows[-1]['paired_control_speedup'] == 2  # median(2, 4, 2)
    assert rows[-1]['paired_cache_speedup'] == 8  # median(5, 10, 8)
    assert all(r['status'] == 'SLOW' for r in rows)
    with pytest.raises(ValueError, match='comparison control'):
        summarize_with_cache_control(cases[:3] + cases[-1:], refs, 1.3)
