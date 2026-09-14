"""Check TMEM read completion and counted remote arrivals before GPU probes."""

from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (build_variant, select_variants, variant_builder_source,
                              VERIFICATION_SHAPES, summarize_with_cache_control)
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES['warp_release']])
def test_writeback_experiments_preserve_read_coverage_and_producer_protocol(arch, shape, tmp_path, pre_tmem_step10):
    tvm = pytest.importorskip('tvm')
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    sources = {}
    for variant in ('baseline', 'warp_release', 'paired_tmem_loads'):
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        assert '"tirx.dyn_smem_bytes": T.int64(230400)' in script
        assert 'Asmem = T.decl_buffer((4, 2, 128, 64)' in script
        with target:
            ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        sources[variant] = ex.mod.imports[0].inspect_source()
    baseline, aggregated, paired = map(body, sources.values())
    # Both consumer slots continue to require all 128 lanes from both CTAs.
    for code in (baseline, aggregated, paired):
        for slot in (11, 12):
            assert f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), 256);' in code
    arrive = 'tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred'
    counted = 'tvm_builtin_ptx_mbarrier_arrive_shared_cluster_count_remote_pred'
    block = '''        tvm_builtin_cuda_warp_sync();
        if (tvm_builtin_elect_one_sync_op() != (uint)0) {
          alignas(64) bool actual_pred_ptr_1[1];
          actual_pred_ptr_1[0] = (bool)1;
          tvm_builtin_ptx_mbarrier_arrive_shared_cluster_count_remote_pred((&(((uint64_t*)pool_buf_ptr)[((warp_id_in_cta >> 2) + 11)])), 32, 0, actual_pred_ptr_1[0]);
        }
'''
    original = '''        alignas(64) bool actual_pred_ptr_1[1];
        actual_pred_ptr_1[0] = (bool)1;
        tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred((&(((uint64_t*)pool_buf_ptr)[((warp_id_in_cta >> 2) + 11)])), 0, actual_pred_ptr_1[0]);
'''
    assert aggregated.count(block) == 1
    assert aggregated.replace(block, original) == baseline
    # No leader can release other lanes' reads before they complete.
    assert aggregated.rindex('tvm_builtin_ptx_tcgen05_wait_ld();') < aggregated.index(
        'tvm_builtin_ptx_tcgen05_fence_before_thread_sync();') < aggregated.index(block)
    assert '__syncwarp();' in sources['warp_release']
    assert re.search(r'mbarrier\.arrive\.shared::cluster\.b64\s+_, \[remAddr32\], %1;', sources['warp_release'])
    # Only the register read/cast region changes in the paired-load experiment.
    marker = '      alignas(64) float Dreg_ptr['
    assert expand_cse(paired.split(marker)[0]) == expand_cse(baseline.split(marker)[0])
    fence = '        tvm_builtin_ptx_tcgen05_fence_before_thread_sync();'
    assert expand_cse(paired.split(fence)[1]) == expand_cse(baseline.split(fence)[1])
    assert 'float Dreg_ptr[64];' in paired and 'half Dreg_f16_ptr[256];' in paired
    region = paired.split(marker)[1].split(fence)[0]
    load_op = 'tvm_builtin_ptx_tcgen05_ld_32x32b_x32'
    before_loads = [line.strip() for line in baseline.splitlines() if load_op + '(' in line]
    loads = [line.strip() for line in region.splitlines() if load_op + '(' in line]
    assert len(loads) == len(before_loads) == 8
    for i, (actual, before) in enumerate(zip(loads, before_loads)):
        regs = list(map(int, re.findall(r'Dreg_ptr\[(\d+)\]', actual)))
        assert regs == list(range((i % 2) * 32, (i % 2 + 1) * 32))
        # Every thread reads the same TMEM source columns as the baseline.
        normalized = re.sub(r'Dreg_ptr\[(\d+)\]',
                            lambda m: f'Dreg_ptr[{int(m[1]) % 32}]', actual)
        assert normalized == before
    # Each pair finishes before a cast reads registers or the next pair reuses them.
    events = []
    for line in region.splitlines():
        if load_op + '(' in line:
            events.append('load')
        elif 'tvm_builtin_ptx_tcgen05_wait_ld();' in line:
            events.append('wait')
        elif 'tvm_builtin_cast_float32x2_float16x2(' in line:
            events.append('cast')
    assert events == ['load', 'load', 'wait', 'cast'] * 4
    loops = re.findall(r'for \(int (f(?:_\d+)?) = 0; \1 < 32; \+\+\1\)', region)
    casts = [line.strip() for line in region.splitlines() if 'tvm_builtin_cast_float32x2_float16x2(' in line]
    assert len(loops) == len(casts) == 4
    for i, (var, line) in enumerate(zip(loops, casts)):
        dst = f'({var} * 2)' if i == 0 else f'(({var} * 2) + {i * 64})'
        assert line == f'tvm_builtin_cast_float32x2_float16x2((&(Dreg_f16_ptr[{dst}])), (&(Dreg_ptr[({var} * 2)])));'
    assert paired.count(arrive + '(') == 1 and counted not in paired


@pytest.mark.parametrize('variant', ['warp_release', 'paired_tmem_loads'])
def test_writeback_transform_rejects_reapplication_and_pre_adoption_source(variant, tmp_path, pre_tmem_step10):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, variant, tmp_path)
    with pytest.raises(ValueError):
        variant_builder_source((tmp_path / 'builder.py').read_text(), 10, variant)
    before = (ROOT / 'results_b300/step10_fused_a.1VXYz2/step10/step10_4096_baseline/builder.py').read_text()
    with pytest.raises(ValueError, match='adopted Step 10'):
        variant_builder_source(before, 10, variant)


def test_writeback_variants_compare_independently_to_production():
    assert select_variants(10, ['warp_release', 'paired_tmem_loads']) == [
        'baseline', 'warp_release', 'paired_tmem_loads']
    cases = [dict(step=10, size=4096, variant=name, samples_ms=samples) for name, samples in
             [('baseline', [10, 20, 40]), ('warp_release', [5, 10, 20]), ('paired_tmem_loads', [8, 10, 10])]]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['paired_control_speedup'] for r in rows] == [1, 2, 2]
    assert all(r['comparison_control'] == 'baseline' and r['status'] == 'SLOW' for r in rows)
