"""Validate emitted tile coverage, TMA/MMA layouts and two-consumer ownership.

CPU source-generation checks cannot establish GPU numerical correctness or speed.
The B300 probe verifies every candidate and boundary before collecting timings.
"""

from collections import Counter
import inspect
import re

import pytest

from probe_persistent import (select_variants, summarize_with_cache_control,
                              trial_order, variant_builder_source)
from probe_step10_geometry import GEOMETRY_CONFIGS, GEOMETRY_VERIFY_SHAPES
from test_current_input_probe import compile_probe, descriptors  # noqa: F401
from test_current_l2_probe import emitted_schedules, schedule_states
from test_share_a_probe import arguments, evaluate_coordinate
from test_tmem_double_buffer_probe import simulate_handoffs


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(4096,) * 3, *GEOMETRY_VERIFY_SHAPES])
def test_geometry_matches_tma_mma_output_coverage_and_handoffs(arch, shape, compile_probe):
    import gemm_kernels

    M, N, K = shape
    for variant, (mma_n, depth) in GEOMETRY_CONFIGS.items():
        rows, cols = M // 512, N // mma_n
        total = rows * cols
        clusters = min(gemm_kernels.SM_COUNT // 2, total)
        script, cuda, maps = compile_probe(shape, variant, arch)
        dynamic = 1024 + depth * (2 * 128 + mma_n // 2) * 64 * 2 + 2 * 128 * 32 * 2
        assert dynamic <= 232448
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert f'T.cta_id([{clusters * 2}])' in script
        assert f'Asmem = T.decl_buffer(({depth}, 2, 128, 64)' in script
        assert f'Bsmem = T.decl_buffer(({depth}, {mma_n // 2}, 64)' in script
        assert 'Dsmem = T.decl_buffer((2, 128, 32)' in script
        assert set(maps) == {'A', 'B', 'D'}
        for name, dims, box, swizzle in [('A', (K, M), (64, 128), 3),
                                         ('B', (K, N), (64, mma_n // 2), 3),
                                         ('D', (N, M), (32, 128), 2)]:
            spec = maps[name]
            assert tuple(map(int, spec.global_dims)) == dims
            assert tuple(map(int, spec.global_strides)) == (dims[0] * 2,)
            assert tuple(map(int, spec.box_dims)) == box
            assert tuple(map(int, spec.element_strides)) == (1, 1)
            assert int(spec.swizzle) == swizzle
            assert int(spec.payload_bits) == int(spec.transaction_bits) == box[0] * box[1] * 16
        byte_count = 2 * (2 * int(maps['A'].transaction_bits) + int(maps['B'].transaction_bits)) // 8
        assert byte_count == (73728 if mma_n == 64 else 81920)
        assert f'{byte_count}, 0, actual_pred_ptr[0]);' in cuda

        # Decode actual MMA descriptors, including offsets at every input
        # stage and consumer, against the compiler's swizzled SMEM layouts.
        mma = [line for line in cuda.splitlines() if 'ptx_tcgen05_mma_cta_2_kind_f16_SS(' in line]
        assert len(mma) == 4  # four K16 instructions for each K64 input stage
        for index in ('k', 'k_1'):
            assert (f'for (int {index} = 0; {index} < {K // 64}; ++{index})' in cuda) == (K > 64)
        assert ('(bool)0' if K == 64 else '(0 < k_1)') in mma[0]
        assert all('(bool)1' in line for line in mma[1:])
        for i, line in enumerate(mma):
            desc = int(re.search(r', \(uint\)(\d+),', line)[1])
            assert ((desc >> 17) & 63) * 8 == mma_n
            operands = re.findall(r'tvm_builtin_smem_desc_add_16B_offset\(desc([AB])_ptr\[0\], (.*?)\),', line)
            assert [name for name, _ in operands] == ['A', 'B']
            for name, expression in operands:
                for stage in range(depth):
                    for consumer in range(2):
                        expr = expression.replace('mma_phase_stage_ptr[0]', str(stage))
                        actual = eval(expr, {'__builtins__': {}}, {'warp_id_in_cta': 8 + consumer}) * 8
                        linear = ((stage * 2 + consumer) * 128 * 64 if name == 'A'
                                  else stage * (mma_n // 2) * 64) + i * 16
                        assert actual == int(maps[name].smem_buffer.layout.apply(linear)['m'])

        # Execute the emitted scheduling arithmetic for each role and CTA;
        # independently enumerate every expected 128x32 output rectangle.
        expected = [(m, n) for start in range(0, rows, 8)
                    for n in range(cols) for m in range(start, min(start + 8, rows))]
        loads = [arguments(line) for line in cuda.splitlines()
                 if 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(' in line]
        stores = [arguments(line) for line in cuda.splitlines()
                  if 'ptx_cp_async_bulk_tensor_shared_to_global_2d(' in line]
        assert len(loads) == 3 and len(stores) == mma_n // 32
        assert ['B_tensormap' in call[2] for call in loads] == [True, False, False]
        assert cuda.count(f'_ptr_2[0] = (_ptr_2[0] + {clusters});') == 3
        assert cuda.count(f'if (!((_ptr_2[0] < {total})))') == 3
        boxes = []
        for work, rank, (m, n) in schedule_states(emitted_schedules(cuda), clusters, total):
            assert (m, n) == expected[work]
            for consumer in range(2):
                a_row = evaluate_coordinate(loads[consumer + 1][-1], m, n, rank, consumer)
                b_row = evaluate_coordinate(loads[0][-1], m, n, rank, consumer)
                assert a_row == m * 512 + consumer * 256 + rank * 128
                assert b_row == n * mma_n + rank * (mma_n // 2)
                for stage in range(K // 64):
                    assert all(evaluate_coordinate(call[-2], m, n, rank, consumer, stage) == stage * 64
                               for call in loads)
                for i, call in enumerate(stores):
                    row = evaluate_coordinate(call[-1], m, n, rank, consumer)
                    col = evaluate_coordinate(call[-2], m, n, rank, consumer)
                    assert row == a_row and col == n * mma_n + i * 32
                    boxes.append((row, col))
        assert len(boxes) == len(set(boxes)) == M * N // (128 * 32)
        assert set(boxes) == {(m, n) for m in range(0, M, 128) for n in range(0, N, 32)}

        for slot, arrivals in ([(i, 1) for i in range(1, depth + 1)]
                               + [(i, 2) for i in range(depth + 1, 2 * depth + 1)]
                               + [(i, 1) for i in range(2 * depth + 1, 2 * depth + 5)]
                               + [(i, 256) for i in range(2 * depth + 5, 2 * depth + 9)]):
            init = f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), {arrivals});'
            assert cuda.count(init) == 1
            assert cuda.index(init) < cuda.index('tvm_builtin_cuda_cta_sync();')
        for role, ring, initial in [('tma', depth, 1), ('mma', depth, 0), ('ld', 2, 1), ('wb', 2, 0)]:
            prefix = role + '_phase'
            assert cuda.index(f'{prefix}_phase_ptr[0] = {initial};') < cuda.index('while (')
            assert cuda.count(f'{prefix}_stage_ptr[0] = 0;') == 2
            assert cuda.count(f'if ({prefix}_stage_ptr[0] == {ring})') == 1
            assert cuda.count(f'{prefix}_phase_ptr[0] = ({prefix}_phase_ptr[0] ^ 1);') == 1
        commit = f'ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + {depth + 1})])), 3);'
        assert cuda.count(commit) == 1 and cuda.index(mma[-1]) < cuda.index(commit)
        assert cuda.index('ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(((ld_phase') < cuda.index(
            'ld_phase_stage_ptr[0] = (ld_phase_stage_ptr[0] + 1);')
        last_read = cuda.rindex('tvm_builtin_ptx_tcgen05_wait_ld();')
        fence = cuda.index('tvm_builtin_ptx_tcgen05_fence_before_thread_sync();', last_read)
        release = cuda.index('tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(', fence)
        advance = cuda.index('wb_phase_stage_ptr[0] = (wb_phase_stage_ptr[0] + 1);', release)
        store = cuda.index('ptx_cp_async_bulk_tensor_shared_to_global_2d(', advance)
        assert last_read < fence < release < advance < store
        assert cuda.count('ptx_cp_async_bulk_wait_group_read_0();') == mma_n // 32
        assert 'tcgen05_alloc_cta_group_2((&(((uint*)pool_buf_ptr)[0])), 512);' in cuda
        assert 'tcgen05_dealloc_cta_group_2(((uint*)pool_buf_ptr)[0], 512);' in cuda
        assert cuda.rindex('tvm_builtin_cuda_cluster_sync();') < cuda.index('tcgen05_dealloc_cta_group_2(')
        assert all(simulate_handoffs(cuda, 9, seed) for seed in range(3))


@pytest.mark.parametrize('shape', [(4096,) * 3, (4608, 2560, 320), (512, 9728, 192)])
def test_max_clusters_changes_only_grid_and_scheduler(shape, compile_probe):
    _, control, maps = compile_probe(shape, 'baseline', 'sm_103a')
    _, actual, actual_maps = compile_probe(shape, 'tmem_max_clusters', 'sm_103a')
    operations = lambda cuda: [line.strip() for line in cuda.splitlines()
                               if re.match(r'\s*(?:tvm_builtin_|ptx_)', line)]
    assert actual != control
    assert descriptors(actual_maps) == descriptors(maps)
    assert operations(actual) == operations(control)


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                 (512, 9472, 192), (4096, 4352, 320)])
def test_geometry_preserves_single_slot_and_wide_fallbacks(arch, shape, compile_probe):
    _, control, maps = compile_probe(shape, 'baseline', arch)
    for variant in GEOMETRY_CONFIGS:
        _, actual, actual_maps = compile_probe(shape, variant, arch)
        assert actual == control
        assert descriptors(actual_maps) == descriptors(maps)


@pytest.mark.parametrize('variant', GEOMETRY_CONFIGS)
def test_geometry_fails_closed_on_reapplication_or_source_drift(variant):
    pytest.importorskip('tvm')
    import gemm_kernels

    source = inspect.getsource(gemm_kernels.hgemm_v10)
    actual = variant_builder_source(source, 10, variant)
    for invalid in (actual, source.replace('PIPE_DEPTH = 4', 'PIPE_DEPTH = 5'),
                    source.replace('NUM_CONSUMER = 2', 'NUM_CONSUMER = 1'),
                    source.replace('l2_group_size=8', 'l2_group_size=4'),
                    source.replace('NARROW_N =', 'OLD_NARROW_N =')):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(invalid, 10, variant)
    with pytest.raises(ValueError, match='unsupported Step 9 variant'):
        variant_builder_source(source, 9, variant)


def test_geometry_includes_direct_controls_and_balances_eight_trials():
    assert select_variants(10) == ['baseline']
    variants = ['baseline', *GEOMETRY_CONFIGS]
    assert select_variants(10, ['tmem_n64_depth5']) == variants
    orders = [trial_order(4, trial) for trial in range(8)]
    for a in range(4):
        assert Counter(order.index(a) for order in orders) == {i: 2 for i in range(4)}
        for b in range(a + 1, 4):
            assert sum(order.index(a) < order.index(b) for order in orders) == 4
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'tmem_max_clusters', 'tmem_n64']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, .8]
