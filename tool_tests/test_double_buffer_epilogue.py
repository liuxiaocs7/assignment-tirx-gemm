"""Check generated buffer ownership and TMA-store drain before GPU timing."""

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
                                  *VERIFICATION_SHAPES['epilogue_double_buffer']])
def test_double_buffer_owns_distinct_storage_and_drains_before_reuse(arch, shape, tmp_path, pre_bfirst_step10):
    tvm = pytest.importorskip('tvm')
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    sources = {}
    for variant, size, allocation in [('epilogue_depth3', 181248, '(2, 128, 64)'),
                                      ('epilogue_double_buffer', 214016, '(2, 2, 128, 64)')]:
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        assert f'"tirx.dyn_smem_bytes": T.int64({size})' in script
        assert f'Dsmem = T.decl_buffer({allocation}' in script
        assert 'Asmem = T.decl_buffer((3, 2, 128, 64)' in script
        assert 'Bsmem = T.decl_buffer((3, 128, 64)' in script
        with target:
            ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        sources[variant] = ex.mod.imports[0].inspect_source()
    control, actual = map(body, sources.values())
    # Existing three-stage control, not a new unmeasured combination.
    if shape == (4096,) * 3:
        recorded = ROOT / 'results_b300/step10_depth3.1zWPNg/step10/step10_4096_balanced_depth3/module_01.cu'
        assert control == body(recorded.read_text())
    marker = '      alignas(64) float Dreg_ptr['
    assert expand_cse(actual.split(marker)[0]) == expand_cse(control.split(marker)[0])
    start = '        alignas(64) int s_off_ptr[1];'
    # Includes all eight TMEM waits and the original counted handoff.
    assert expand_cse(actual.split(start)[0]) == expand_cse(control.split(start)[0])
    end = '        tile_scheduler_tile_count_ptr[0] ='
    assert expand_cse(actual.rsplit(end, 1)[1]) == expand_cse(control.rsplit(end, 1)[1])
    region = actual.split(start)[1].split(end)[0]
    assert 'cp.async.bulk.wait_group.read 1;' in sources['epilogue_double_buffer']
    assert 'cp.async.bulk.wait_group.read 0;' in sources['epilogue_double_buffer']
    # Every commit and wait belongs to a fixed lane (groups are per thread).
    assert 'tvm_builtin_elect_one_sync_op' not in region
    assert region.count('if ((((int)threadIdx.x) % 32) == 0)') == 4
    assert region.count('if ((warp_id_in_cta % 4) == 0)') == 4
    store_op = 'ptx_cp_async_bulk_tensor_shared_to_global_2d'
    stores = [line.strip() for line in region.splitlines() if store_op + '(' in line]
    assert len(stores) == 4
    for i, line in enumerate(stores):
        # Each consumer owns two 128x64 half buffers: stride 16384 halves,
        # alternate buffer offset 8192 halves, with original output coords.
        buffer = 74240 + (i % 2) * 8192
        assert f'(((warp_id_in_cta >> 2) * 16384) + {buffer})' in line
        column = '(_ptr_1[0] * 256)' if i == 0 else f'((_ptr_1[0] * 256) + {i * 64})'
        assert line.endswith(f', {column}, (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256))'
                             ' + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));')
    # Check the actual generated STS offsets cover each TMA source exactly.
    offsets = re.findall(r'^ +s_off_ptr(?:_\d+)?\[0\] = ([^;]+);$', region, re.M)
    assert len(offsets) == 4
    for consumer in range(2):
        for i, expr in enumerate(offsets):
            addresses = set()
            for warp in range(4):
                for lane in range(32):
                    offset = eval(expr.replace('((int)threadIdx.x)', str(lane)),
                                  {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4 + warp})
                    # SWIZZLE_128B_ATOM: a 64-half row permutes eight-half vectors.
                    for column in range(64):
                        linear = offset + column
                        addresses.add(linear ^ ((linear >> 3) & 56))
            lo = consumer * 16384 + (i % 2) * 8192
            assert addresses == set(range(lo, lo + 8192))
    # Simulate the worst case: stores read as late as the waits allow.
    # Derive event order from CUDA, so moving a wait after reuse fails.
    events = []
    for line in region.splitlines():
        if re.search(r's_off_ptr(?:_\d+)?\[0\] =', line):
            events.append('write')
        elif 'tvm_builtin_ptx_fence_proxy_async_shared_cta();' in line:
            events.append('fence')
        elif 'tvm_builtin_cuda_warpgroup_sync(' in line:
            events.append('sync')
        elif store_op + '(' in line:
            events.append('store')
        elif 'ptx_cp_async_bulk_tensor_commit_group();' in line:
            events.append('commit')
        elif match := re.search(r'ptx_cp_async_bulk_wait_group_read_(\d+)\(\);', line):
            events.append(int(match[1]))
    pending, released = [], []
    chunk = -1
    for index, event in enumerate(events):
        if event == 'write':
            chunk += 1
            assert chunk % 2 not in pending + released
        elif event == 'store':
            assert events[index - 2:index] == ['fence', 'sync']
            pending.append(chunk % 2)
        elif event == 'commit':
            assert events[index - 1] == 'store'
        elif isinstance(event, int):
            released += pending[:len(pending) - event]
            pending = pending[len(pending) - event:] if event else []
        elif event == 'sync':
            released.clear()
    assert chunk == 3 and not pending and not released
    assert events[-2:] == [0, 'sync']  # Drain before next tile / cluster teardown.


@pytest.mark.parametrize('variant', ['epilogue_depth3', 'epilogue_double_buffer'])
def test_epilogue_transform_refuses_reapplication_or_old_baseline(variant, tmp_path):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, variant, tmp_path)
    with pytest.raises(ValueError):
        variant_builder_source((tmp_path / 'builder.py').read_text(), 10, variant)
    before = (ROOT / 'results_b300/step10_fused_a.1VXYz2/step10/step10_4096_baseline/builder.py').read_text()
    with pytest.raises(ValueError, match='adopted Step 10'):
        variant_builder_source(before, 10, variant)


def test_epilogue_selection_includes_depth_control_and_scores_against_it():
    variants = ['baseline', 'epilogue_depth3', 'epilogue_double_buffer']
    assert select_variants(10, ['epilogue_double_buffer']) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20, 40], [8, 10, 20], [4, 5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert rows[-1]['comparison_control'] == 'epilogue_depth3'
    assert rows[-1]['paired_control_speedup'] == 2
    assert rows[-1]['paired_speedup'] == 4
    with pytest.raises(ValueError, match='comparison control'):
        summarize_with_cache_control(cases[::2], {(10, 4096, 4096, 4096): 1}, 1.3)
