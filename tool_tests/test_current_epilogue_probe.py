"""Validate current-path output layouts and store-buffer lifetimes on CPU.

Check emitted CUDA/TMA contracts, not a replacement implementation. GPU
numerical verification and balanced original-timer measurements remain required.
"""

from collections import Counter
import inspect
import re

import pytest

from probe_persistent import select_variants, summarize_with_cache_control, trial_order, variant_builder_source
from probe_step10_epilogue import EPILOGUE_CONFIGS, EPILOGUE_VERIFY_SHAPES
from test_current_input_probe import compile_probe, descriptors  # noqa: F401
from test_cache_followup_probe import expand_cse
from test_current_l2_probe import emitted_schedules, schedule_states
from test_share_a_probe import arguments, evaluate_coordinate
from test_tmem_double_buffer_probe import simulate_handoffs


def store_events(region):
    """Read the generated write/fence/store/wait order for one output tile."""
    result = []
    for line in region.splitlines():
        if re.search(r's_off_ptr(?:_\d+)?\[0\] =', line):
            result.append('write')
        elif 'tvm_builtin_ptx_fence_proxy_async_shared_cta();' in line:
            result.append('fence')
        elif 'tvm_builtin_cuda_warpgroup_sync(' in line:
            result.append('sync')
        elif 'ptx_cp_async_bulk_tensor_shared_to_global_' in line:
            result.append('store')
        elif 'ptx_cp_async_bulk_tensor_commit_group();' in line:
            result.append('commit')
        elif match := re.search(r'ptx_cp_async_bulk_wait_group_read_(\d+)\(\);', line):
            result.append(int(match[1]))
    return result


def assert_no_premature_reuse(events, buffers, chunks):
    # A store reads SMEM as late as the wait permits; then all WG threads
    # must observe that completion before they can overwrite its source.
    pending, released = [], []
    chunk = -1
    for index, event in enumerate(events):
        if event == 'write':
            chunk += 1
            assert chunk % buffers not in pending + released
        elif event == 'store':
            assert events[index-2:index] == ['fence', 'sync']
            pending.append(chunk % buffers)
        elif event == 'commit':
            assert events[index-1] == 'store'
        elif isinstance(event, int):
            count = max(0, len(pending) - event)
            released += pending[:count]
            pending = pending[count:]
        elif event == 'sync':
            released.clear()
    assert chunk == chunks - 1 and not pending and not released
    assert events[-2:] == [0, 'sync']


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(4096,) * 3, *EPILOGUE_VERIFY_SHAPES])
def test_output_maps_coverage_and_store_lifetimes(arch, shape, compile_probe):
    M, N, K = shape
    _, baseline, baseline_maps = compile_probe(shape, 'baseline', arch)
    marker = '        alignas(64) int s_off_ptr[1];'
    total = M // 512 * (N // 128)
    waves = (total + 73) // 74
    clusters = (total + waves - 1) // waves
    for variant, (width, buffers) in EPILOGUE_CONFIGS.items():
        script, cuda, maps = compile_probe(shape, variant, arch)
        dynamic = 1024 + 4 * (2 * 128 + 64) * 64 * 2 + 2 * buffers * 128 * width * 2
        assert dynamic <= 232448
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert f'T.cta_id([{clusters * 2}])' in script
        assert 'Asmem = T.decl_buffer((4, 2, 128, 64)' in script
        assert 'Bsmem = T.decl_buffer((4, 64, 64)' in script
        assert descriptors({k: maps[k] for k in ('A', 'B')}) == descriptors(
            {k: baseline_maps[k] for k in ('A', 'B')})
        # Everything before SMEM output staging includes input producers,
        # MMA consumers, four TMEM loads, fences, release and phase advance.
        assert expand_cse(cuda.split(marker)[0]) == expand_cse(baseline.split(marker)[0])
        assert all(simulate_handoffs(cuda, 9, seed) for seed in range(3))

        spec = maps['D']
        wide = width == 128
        assert tuple(map(int, spec.global_dims)) == ((64, M, N // 64) if wide else (N, M))
        assert tuple(map(int, spec.global_strides)) == ((N * 2, 128) if wide else (N * 2,))
        assert tuple(map(int, spec.box_dims)) == ((64, 128, 2) if wide else (width, 128))
        assert tuple(map(int, spec.element_strides)) == (1,) * (3 if wide else 2)
        assert int(spec.swizzle) == (2 if width == 32 else 3)
        assert int(spec.payload_bits) == int(spec.transaction_bits) == width * 128 * 16

        region = marker + cuda.split(marker)[1].split('        tile_scheduler_tile_count_ptr[0] =')[0]
        stores = [arguments(line) for line in region.splitlines()
                  if 'ptx_cp_async_bulk_tensor_shared_to_global_' in line]
        assert len(stores) == 128 // width
        boxes = []
        for _, rank, (m, n) in schedule_states(emitted_schedules(cuda), clusters, total):
            for consumer in range(2):
                for i, call in enumerate(stores):
                    coords = [evaluate_coordinate(x, m, n, rank, consumer) for x in call[3:]]
                    if wide:
                        x, row, plane = coords
                        col = plane * 64 + x
                    else:
                        col, row = coords
                    assert row == m * 512 + consumer * 256 + rank * 128
                    assert col == n * 128 + i * width
                    boxes.append((row, col))
        assert len(boxes) == len(set(boxes)) == M * N // (128 * width)
        assert set(boxes) == {(m, n) for m in range(0, M, 128) for n in range(0, N, width)}

        offsets = re.findall(r'^ +s_off_ptr(?:_\d+)?\[0\] = ([^;]+);$', region, re.M)
        assert len(offsets) == len(stores)
        for i, (expression, store) in enumerate(zip(offsets, stores)):
            # Evaluate emitted row offsets and the output layout. For the
            # 128-column case CUDA/TMA both use two 64-column planes.
            if wide:
                assert 'ds_ptr[0] = (((f_4 >> 3) * 8192) + ((f_4 & 7) * 8));' in region
            for consumer in range(2):
                lo = consumer * buffers * 128 * width + (i % buffers) * 128 * width
                addresses = set()
                for warp in range(4):
                    for lane in range(32):
                        off = eval(expression.replace('((int)threadIdx.x)', str(lane)),
                                   {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4 + warp})
                        for col in range(width):
                            linear = off + (col % 64 + (col // 64) * 8192 if wide else col)
                            addresses.add(linear ^ ((linear >> 3) & (24 if width == 32 else 56)))
                assert addresses == set(range(lo, lo + 128 * width))
                base = re.search(r'pool_buf_ptr\)\[(.*)\]\)\)', store[0])
                assert base is not None, store[0]
                assert eval(base[1], {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4}) == 82432 + lo

        events = store_events(region)
        assert_no_premature_reuse(events, buffers, len(stores))
        if buffers == 2:
            assert [e for e in events if isinstance(e, int)] == [1, 1, 0]
            assert region.count('if ((((int)threadIdx.x) % 32) == 0)') == 4
            assert 'tvm_builtin_elect_one_sync_op' not in region
        else:
            assert [e for e in events if isinstance(e, int)] == [0] * len(stores)
        assert cuda.rindex('tvm_builtin_cuda_cluster_sync();') < cuda.index('tcgen05_dealloc_cta_group_2(')


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                 (512, 9472, 192), (4096, 4352, 64), (4096, 4352, 320)])
def test_fallback_cuda_and_tensor_maps_are_unchanged(arch, shape, compile_probe):
    _, baseline, baseline_maps = compile_probe(shape, 'baseline', arch)
    for variant in EPILOGUE_CONFIGS:
        _, actual, maps = compile_probe(shape, variant, arch)
        assert actual == baseline
        assert descriptors(maps) == descriptors(baseline_maps)


@pytest.mark.parametrize('variant', EPILOGUE_CONFIGS)
def test_transform_fails_closed_on_reapplication_and_protocol_drift(variant):
    pytest.importorskip('tvm')
    import gemm_kernels
    source = inspect.getsource(gemm_kernels.hgemm_v10)
    changed = variant_builder_source(source, 10, variant)
    for invalid in (changed, source.replace('PIPE_DEPTH = 4', 'PIPE_DEPTH = 5'),
                    source.replace('NUM_CONSUMER = 2', 'NUM_CONSUMER = 1'),
                    source.replace('l2_group_size=8', 'l2_group_size=4'),
                    source.replace('EPI_N = 32', 'EPI_N = 64')):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(invalid, 10, variant)


def test_wait_simulator_rejects_missing_wait_before_buffer_reuse():
    invalid = ['write', 'fence', 'sync', 'store', 'commit', 'sync'] * 3
    with pytest.raises(AssertionError):
        assert_no_premature_reuse(invalid, 2, 3)


def test_epilogue_comparisons_balance_positions_and_use_baseline():
    variants = ['baseline', *EPILOGUE_CONFIGS]
    assert select_variants(10, EPILOGUE_CONFIGS) == variants
    orders = [trial_order(4, i) for i in range(8)]
    for a in range(4):
        assert Counter(o.index(a) for o in orders) == {i: 2 for i in range(4)}
        for b in range(a + 1, 4):
            assert sum(o.index(a) < o.index(b) for o in orders) == 4
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s)
             for v, s in zip(variants, [[10, 20], [8, 16], [5, 10], [4, 8]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): .107}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline'] * 4
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, 2.5]
    assert all(r['status'] == 'SLOW' for r in rows)
