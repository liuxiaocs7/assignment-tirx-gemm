"""Check independent wide-N epilogue and split-producer protocols before B300."""

from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (VERIFICATION_SHAPES, build_variant, select_variants,
                              summarize_with_cache_control, variant_builder_source)
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body


def normalized_lines(source):
    return [line.strip() for line in expand_cse(source).splitlines() if line.strip()]


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES['split_tma'], (512, 256, 64)])
def test_wide_n_experiments_preserve_work_and_stage_ownership(arch, shape, tmp_path, monkeypatch):
    tvm = pytest.importorskip('tvm')
    from tvm.backend.cuda.tile_primitive.copy_async import tma
    emit, plans = tma._emit_plan, []
    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)
    monkeypatch.setattr(tma, '_emit_plan', record)
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    sources, specs = {}, {}
    M, N, K = shape
    for variant, epi, dynamic in [('baseline', 64, 230400), ('epilogue_32', 32, 214016),
                                   ('split_tma', 64, 230400)]:
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert 'Asmem = T.decl_buffer((4, 2, 128, 64)' in script
        assert 'Bsmem = T.decl_buffer((4, 128, 64)' in script
        assert f'Dsmem = T.decl_buffer((2, 128, {epi})' in script
        plans.clear()
        with target:
            ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        sources[variant] = body(ex.mod.imports[0].inspect_source())
        specs[variant] = {p.spec.descriptor_name: p.spec for p in plans}
    baseline, epi32, split = sources.values()
    if shape == (4096,) * 3:
        recorded = ROOT / 'results_b300/step10_n128.eVKFm2/step10/step10_4096_baseline/module_01.cu'
        assert baseline == body(recorded.read_text())

    # Smaller epilogue: all TMA/MMA, full TMEM read and handoff instructions
    # precede the first SMEM write and must remain identical to production.
    start = '        alignas(64) int s_off_ptr[1];'
    assert expand_cse(baseline.split(start)[0]) == expand_cse(epi32.split(start)[0])
    dspec = specs['epilogue_32']['D']
    assert tuple(map(int, dspec.global_dims)) == (N, M)
    assert tuple(map(int, dspec.global_strides)) == (N * 2,)
    assert tuple(map(int, dspec.box_dims)) == (32, 128)
    assert int(dspec.swizzle) == 2
    assert int(dspec.payload_bits) == int(dspec.transaction_bits) == 65536
    stores = [line for line in epi32.splitlines() if 'ptx_cp_async_bulk_tensor_shared_to_global_2d(' in line]
    assert len(stores) == 8
    for chunk, line in enumerate(stores):
        col = '(_ptr_1[0] * 256)' if chunk == 0 else f'((_ptr_1[0] * 256) + {chunk * 32})'
        assert line.endswith(f', {col}, (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256))'
                             ' + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));')
    assert epi32.count('ptx_cp_async_bulk_wait_group_read_0();') == 8
    assert epi32.count('tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));') == 16
    offsets = re.findall(r'^ +s_off_ptr(?:_\d+)?\[0\] = ([^;]+);$', epi32, re.M)
    assert len(offsets) == 8
    for consumer in range(2):
        for expr in offsets:
            for warp in range(4):
                for lane in (0, 7, 31):
                    linear = eval(expr.replace('((int)threadIdx.x)', str(lane)),
                                  {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4 + warp})
                    assert linear == (consumer * 128 + warp * 32 + lane) * 32
                    for col in (0, 31):
                        address = int(dspec.smem_buffer.layout.apply(linear + col)['m'])
                        assert address == (linear + col) ^ (((linear + col) >> 3) & 24)

    # Split producer: every hardware operation and every buffer descriptor
    # stays the same. Only the guards around the TMA work may differ.
    calls = lambda code: [line for line in normalized_lines(code)
                          if re.match(r'(?:tvm_builtin_|ptx_)\w+\(', line)]
    assert calls(split) == calls(baseline)
    for name in ('A', 'B', 'D'):
        for field in ('global_dims', 'global_strides', 'box_dims', 'coordinates',
                      'smem_base_offset', 'swizzle', 'transaction_bits'):
            assert str(getattr(specs['split_tma'][name], field)) == str(getattr(specs['baseline'][name], field))
    for phase in ('tma', 'mma', 'ld', 'wb'):
        phase_lines = lambda code: [line for line in normalized_lines(code) if phase + '_phase_' in line]
        assert phase_lines(split) == phase_lines(baseline)
    writeback = '      alignas(64) float Dreg_ptr['
    assert normalized_lines(split.split(writeback)[1]) == normalized_lines(baseline.split(writeback)[1])

    load_op = 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr('
    loads = [line.strip() for line in baseline.splitlines() if load_op in line]
    assert len(loads) == 3
    lines = normalized_lines(split)
    begin = lines.index('if ((warp_id_in_cta & 3) == 2) {')
    assert lines[begin:begin + 7] == ['if ((warp_id_in_cta & 3) == 2) {', loads[0],
                                    '} else {', loads[1], loads[2], '}',
                                    'if ((((int)tvm_builtin_cluster_ctaid_x()) == 0) & ((warp_id_in_cta & 3) == 3)) {']
    assert lines.count('if (2 <= (warp_id_in_cta & 3)) {') == 1
    assert split.count('98304, 0, actual_pred_ptr[0]);') == 1
    # The existing empty-stage wait encloses BOTH producer branches. Neither
    # warp can lap the other: reuse requires both consumers, whose full wait
    # requires all six TMA completions plus the single expect_tx arrival.
    empty_wait = 'tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[(tma_phase_stage_ptr[0] + 5)])), (tma_phase_phase_ptr[0] ^ 0));'
    assert lines.index('if (2 <= (warp_id_in_cta & 3)) {') < lines.index(empty_wait) < begin
    assert split.count(empty_wait) == 1
    for phase in ('tma', 'mma'):
        assert split.count(f'{phase}_phase_stage_ptr[0] = 0;') == 2
        assert split.count(f'{phase}_phase_stage_ptr[0] == 4') == 1
    assert split.count('tvm_builtin_ptx_tcgen05_wait_ld();') == 8
    assert split.count('ptx_tcgen05_mma_cta_2_kind_f16_SS(') == 4


@pytest.mark.parametrize('variant', ['epilogue_32', 'split_tma'])
def test_wide_n_probe_refuses_reapplication_and_wrong_baseline(variant, tmp_path):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, variant, tmp_path)
    with pytest.raises(ValueError):
        variant_builder_source((tmp_path / 'builder.py').read_text(), 10, variant)
    for directory in ['results_b300/step10_roles.rx5lNL/step10/step10_4096_baseline',
                      'results_b300/step10_n128.eVKFm2/step10/step10_4096_n128_epi32_depth5']:
        with pytest.raises(ValueError, match='B-first baseline'):
            variant_builder_source((ROOT / directory / 'builder.py').read_text(), 10, variant)


def test_wide_n_experiments_use_independent_controls():
    variants = ['baseline', 'epilogue_32', 'split_tma']
    assert select_variants(10, variants) == variants
    assert select_variants(10, ['split_tma']) == ['baseline', 'split_tma']
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline'] * 3
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2]
    assert all(r['status'] == 'SLOW' for r in rows)
