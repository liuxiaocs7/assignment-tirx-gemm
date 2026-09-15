"""Check TMA/MMA layouts, complete work and ring ownership before B300 timing."""

from collections import Counter
import inspect
import re

import pytest

from probe_persistent import select_variants, summarize_with_cache_control, trial_order, variant_builder_source
from probe_step10_granularity import GRANULARITY_CONFIGS, GRANULARITY_VERIFY_SHAPES
from test_current_input_probe import compile_probe, descriptors  # noqa: F401
from test_current_ready_probe import calls, simulate_input_ring, EXPECT, COMMIT
from test_current_l2_probe import emitted_schedules, schedule_states
from test_share_a_probe import evaluate_coordinate, host_maps
from test_tmem_double_buffer_probe import simulate_handoffs


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(4096,) * 3, *GRANULARITY_VERIFY_SHAPES])
def test_input_mapping_work_coverage_and_ring_ownership(arch, shape, compile_probe):
    M, N, K = shape
    _, baseline, baseline_maps = compile_probe(shape, 'baseline', arch)
    total = M // 512 * (N // 128)
    waves = (total + 73) // 74
    clusters = (total + waves - 1) // waves
    for variant, (blk_k, depth) in GRANULARITY_CONFIGS.items():
        script, cuda, maps = compile_probe(shape, variant, arch)
        dynamic = 1024 + depth * 320 * blk_k * 2 + 16384
        assert dynamic in (181248, 222208) and dynamic <= 232448
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert f'T.cta_id([{clusters * 2}])' in script
        assert f'Asmem = T.decl_buffer(({depth}, 2, 128, {blk_k})' in script
        assert f'Bsmem = T.decl_buffer(({depth}, 64, {blk_k})' in script
        assert 'Dsmem = T.decl_buffer((2, 128, 32)' in script
        # Depth10 moves D's SMEM base after the larger input region; its
        # host tensor map, layout and output coordinates remain unchanged.
        assert host_maps({'D': maps['D']}) == host_maps({'D': baseline_maps['D']})

        # K64/SW64 is split into two 32-element planes in the tensor map.
        # Both the host encoding and device coordinates must agree on this.
        for name, rows, height in [('A', M, 128), ('B', N, 64)]:
            spec = maps[name]
            assert tuple(map(int, spec.global_dims)) == ((32, rows, K // 32) if blk_k == 64 else (K, rows))
            assert tuple(map(int, spec.global_strides)) == ((K * 2, 64) if blk_k == 64 else (K * 2,))
            assert tuple(map(int, spec.box_dims)) == ((32, height, 2) if blk_k == 64 else (32, height))
            assert tuple(map(int, spec.element_strides)) == (1,) * (3 if blk_k == 64 else 2)
            assert int(spec.swizzle) == 2
            assert int(spec.payload_bits) == int(spec.transaction_bits) == height * blk_k * 16
        copies = calls(cuda, f'ptx_cp_async_bulk_tensor_g2s_cluster_tile_{3 if blk_k == 64 else 2}d_mbar_addr(')
        assert len(copies) == 3
        assert ['B_tensormap' in a[2] for a in copies] == [True, False, False]
        for copy, name, consumer in zip(copies, ('B', 'A', 'A'), (0, 0, 1)):
            expr = copy[0].split('pool_buf_ptr)[', 1)[1].rsplit(']', 1)[0]
            base = 512 + (depth * 256 * blk_k if name == 'B' else 0)
            for stage in range(depth):
                actual = eval(expr.replace('tma_phase_stage_ptr[0]', str(stage)), {'__builtins__': {}})
                assert actual == base + stage * (64 if name == 'B' else 256) * blk_k + consumer * 128 * blk_k
            assert copy[1].endswith('remote_mbar_ptr[tma_phase_stage_ptr[0]])))')

        # Every role traverses the same complete schedule; inputs match the
        # output rows, including both cluster ranks and the partial L2 group.
        boxes = []
        stores = calls(cuda, 'ptx_cp_async_bulk_tensor_shared_to_global_2d(')
        assert len(stores) == 4
        for store in stores:
            expr = store[0].split('pool_buf_ptr)[', 1)[1].rsplit(']', 1)[0]
            for consumer in range(2):
                assert eval(expr, {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4}) == 512 + depth * 320 * blk_k + consumer * 4096
        for _, rank, (m, n) in schedule_states(emitted_schedules(cuda), clusters, total):
            input_rows = [evaluate_coordinate(c[-2 if blk_k == 64 else -1], m, n, rank, 0) for c in copies]
            assert input_rows == [n * 128 + rank * 64, m * 512 + rank * 128, m * 512 + rank * 128 + 256]
            for consumer in range(2):
                for i, c in enumerate(stores):
                    col, row = [evaluate_coordinate(x, m, n, rank, consumer) for x in c[-2:]]
                    assert (row, col) == (input_rows[consumer + 1], n * 128 + i * 32)
                    boxes.append((row, col))
        assert len(boxes) == len(set(boxes)) == M * N // (128 * 32)
        for copy in copies:
            starts = []
            for k in range(K // blk_k):
                coords = [evaluate_coordinate(x, 0, 0, 0, 0, k) for x in copy[5:]]
                starts.append(coords[0] + coords[2] * 32 if blk_k == 64 else coords[0])
            assert starts == list(range(0, K, blk_k))

        mma = [line for line in cuda.splitlines() if 'ptx_tcgen05_mma_cta_2_kind_f16_SS(' in line]
        assert len(mma) == blk_k // 16
        assert len(mma) * 16 * (K // blk_k) == K
        for index in ('k', 'k_1'):
            loop = f'for (int {index} = 0; {index} < {K // blk_k}; ++{index})'
            assert (loop in cuda) == (K > blk_k)
        assert ('(bool)0' if K == blk_k else '(0 < k_1)') in mma[0]
        assert all('(bool)1' in line for line in mma[1:])
        encodings = re.findall(r'encode_matrix_descriptor\(\(&\(desc([AB])_ptr\[0\]\)\), .*?, (\d+), (\d+), (\d+)\);', cuda)
        assert {n: tuple(map(int, (l, s, w))) for n, l, s, w in encodings} == {
            'A': (512 if blk_k == 64 else 0, 32, 2), 'B': (256 if blk_k == 64 else 0, 32, 2)}
        for i, line in enumerate(mma):
            assert '(uint)270532624' in line  # unchanged M256/N128 instruction
            operands = re.findall(r'tvm_builtin_smem_desc_add_16B_offset\(desc([AB])_ptr\[0\], (.*?)\),', line)
            assert [name for name, _ in operands] == ['A', 'B']
            for name, expr in operands:
                for stage in range(depth):
                    for consumer in range(2):
                        offset = eval(expr.replace('mma_phase_stage_ptr[0]', str(stage)),
                                      {'__builtins__': {}}, {'warp_id_in_cta': 8 + consumer}) * 8
                        linear = ((stage * 2 + consumer) * 128 * blk_k if name == 'A'
                                  else stage * 64 * blk_k) + i * 16
                        assert offset == int(maps[name].smem_buffer.layout.apply(linear)['m'])

        sizes = [int(maps[n].transaction_bits) // 8 for n in ('B', 'A', 'A')]
        assert [int(c[1]) for c in calls(cuda, EXPECT)] == [2 * sum(sizes)]
        assert 2 * sum(sizes) * (K // blk_k) == 2 * 320 * K * 2
        for slot, count in ([(i, 1) for i in range(1, depth + 1)]
                            + [(i, 2) for i in range(depth + 1, 2 * depth + 1)]
                            + [(i, 1) for i in range(2 * depth + 1, 2 * depth + 5)]
                            + [(i, 256) for i in range(2 * depth + 5, 2 * depth + 9)]):
            init = f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), {count});'
            assert cuda.count(init) == 1
            assert cuda.index(init) < cuda.index('tvm_builtin_cuda_cta_sync();')
        for role, ring, initial in [('tma', depth, 1), ('mma', depth, 0), ('ld', 2, 1), ('wb', 2, 0)]:
            prefix = role + '_phase'
            assert cuda.index(f'{prefix}_phase_ptr[0] = {initial};') < cuda.index('while (')
            assert cuda.count(f'{prefix}_stage_ptr[0] = 0;') == 2
            assert cuda.count(f'if ({prefix}_stage_ptr[0] == {ring})') == 1
            assert cuda.count(f'{prefix}_phase_ptr[0] = ({prefix}_phase_ptr[0] ^ 1);') == 1
        free_commit, = [c for c in calls(cuda, COMMIT) if 'mma_phase' in c[0]]
        assert f'(mma_phase_stage_ptr[0] + {depth + 1})' in free_commit[0] and free_commit[1] == '3'
        acquire = 'tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 1)])), (mma_phase_phase_ptr[0] ^ 0));'
        assert cuda.count(acquire) == 1
        assert cuda.index(acquire) < cuda.index('tvm_builtin_ptx_tcgen05_fence_after_thread_sync();', cuda.index(acquire)) < cuda.index(mma[0])
        assert cuda.index(mma[-1]) < cuda.index(COMMIT + free_commit[0])
        last_read = cuda.rindex('tvm_builtin_ptx_tcgen05_wait_ld();')
        fence = cuda.index('tvm_builtin_ptx_tcgen05_fence_before_thread_sync();', last_read)
        release = cuda.index('tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(', fence)
        assert last_read < fence < release < cuda.index('ptx_cp_async_bulk_tensor_shared_to_global_2d(', release)
        assert cuda.count('ptx_cp_async_bulk_wait_group_read_0();') == 4
        assert cuda.rindex('tvm_builtin_cuda_cluster_sync();') < cuda.index('tcgen05_dealloc_cta_group_2(')
        assert all(simulate_handoffs(cuda, 7, seed) for seed in range(3))
        if K <= 512:
            for seed in range(6):
                simulate_input_ring(cuda, K // blk_k, seed, transfer_sizes=sizes)


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                 (512, 9472, 192), (4096, 4352, 64), (4096, 4352, 320)])
def test_inactive_dispatch_cuda_and_maps_identical(arch, shape, compile_probe):
    _, baseline, baseline_maps = compile_probe(shape, 'baseline', arch)
    for variant in GRANULARITY_CONFIGS:
        _, actual, maps = compile_probe(shape, variant, arch)
        assert actual == baseline and descriptors(maps) == descriptors(baseline_maps)


def test_ring_simulator_rejects_wrong_bytes_or_early_release(compile_probe):
    _, cuda, _ = compile_probe((4096, 3072, 320), 'tmem_k32_depth10', 'sm_103a')
    early_free = re.sub(r'(pool_buf_ptr\)\[(?:1[1-9]|20)\]\)\)), 2\);', r'\1, 1);', cuda)
    bad_bytes = cuda.replace('40960, 0, actual_pred', '20480, 0, actual_pred')
    for broken in (early_free, bad_bytes):
        assert broken != cuda
        with pytest.raises(AssertionError):
            for seed in range(20):
                simulate_input_ring(broken, 10, seed, transfer_sizes=[4096, 8192, 8192])


@pytest.mark.parametrize('variant', GRANULARITY_CONFIGS)
def test_transform_refuses_drift_and_reapplication(variant):
    pytest.importorskip('tvm')
    import gemm_kernels
    source = inspect.getsource(gemm_kernels.hgemm_v10)
    changed = variant_builder_source(source, 10, variant)
    for invalid in (changed, source.replace('PIPE_DEPTH = 4', 'PIPE_DEPTH = 5'),
                    source.replace('mma2tma.init(NUM_CONSUMER)', 'mma2tma.init(1)'),
                    source.replace('NUM_CONSUMER = 2', 'NUM_CONSUMER = 1'),
                    source.replace('l2_group_size=8', 'l2_group_size=4')):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(invalid, 10, variant)


def test_controls_balance_and_default_is_production():
    variants = ['baseline', *GRANULARITY_CONFIGS]
    assert select_variants(10) == ['baseline']
    assert select_variants(10, ['tmem_k32_depth10']) == variants
    orders = [trial_order(4, i) for i in range(8)]
    for a in range(4):
        assert Counter(o.index(a) for o in orders) == {i: 2 for i in range(4)}
        for b in range(a + 1, 4):
            assert sum(o.index(a) < o.index(b) for o in orders) == 4
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s)
             for v, s in zip(variants, [[10, 20], [8, 16], [4, 8], [2, 4]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): .107}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', *list(GRANULARITY_CONFIGS)[:2]]
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, 2]
