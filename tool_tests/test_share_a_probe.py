"""Check transposed consumer ownership and reduced TMA work before B300 timing."""

import inspect
from pathlib import Path
import re

import pytest

from probe_persistent import (SHARE_A_VARIANTS, VERIFICATION_SHAPES, select_variants,
                              summarize_with_cache_control, variant_builder_source)
from test_current_input_probe import compile_probe  # noqa: F401 - shared pytest fixture
from test_tmem_double_buffer_probe import simulate_handoffs


def host_maps(specs):
    # SMEM offsets belong to the device copy, not the host tensor-map encoding.
    fields = ('global_dims', 'global_strides', 'box_dims', 'element_strides',
              'swizzle', 'payload_bits', 'transaction_bits')
    return {name: tuple(str(getattr(spec, f)) for f in fields) for name, spec in specs.items()}


def arguments(line):
    """Split generated CUDA call operands while retaining nested expressions."""
    text = line[line.index('(') + 1:line.rindex(')')]
    depth, start, result = 0, 0, []
    for i, char in enumerate(text):
        depth += (char == '(') - (char == ')')
        if char == ',' and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    assert depth == 0
    return result + [text[start:].strip()]


def evaluate_coordinate(expression, m, n, rank, consumer, k=0):
    expression = expression.replace('_ptr[0]', str(m)).replace('_ptr_1[0]', str(n))
    expression = expression.replace('((int)tvm_builtin_cluster_ctaid_x())', str(rank))
    return eval(expression, {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4, 'k': k})


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(4096,) * 3, *VERIFICATION_SHAPES['tmem_share_a_depth6'],
                                 (512, 9728, 64),  # just above the single-wave cutoff
                                 (2560, 5376, 448)])  # full L2 groups plus a tail
def test_shared_a_coordinates_and_protocol(arch, shape, compile_probe, monkeypatch):
    import gemm_kernels

    original = gemm_kernels.ClusterPersistentScheduler2D
    schedulers = []

    def record(*args, **kwargs):
        schedulers.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(gemm_kernels, 'ClusterPersistentScheduler2D', record)
    M, N, K = shape
    total = (M // 256) * (N // 256)
    max_clusters = gemm_kernels.SM_COUNT // 2
    waves = (total + max_clusters - 1) // max_clusters
    clusters = (total + waves - 1) // waves
    for variant, depth in zip(SHARE_A_VARIANTS, (5, 6)):
        script, cuda, specs = compile_probe(shape, variant, arch)
        assert schedulers[-1] == dict(num_m_tiles=M // 256, num_n_tiles=N // 256,
                                     l2_group_size=8, num_clusters=clusters)
        assert f'T.cta_id([{2 * clusters}])' in script
        assert cuda.count(f'_ptr_2[0] = (_ptr_2[0] + {clusters});') == 3
        assert cuda.count(f'if (!((_ptr_2[0] < {total})))') == 3
        dynamic = 1024 + depth * 32768 + 16384
        assert dynamic <= 232448
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert f'Asmem = T.decl_buffer(({depth}, 1, 128, 64)' in script
        assert f'Bsmem = T.decl_buffer(({depth}, 2, 64, 64)' in script

        for name, dims, box, swizzle in [('A', (K, M), (64, 128), 3),
                                         ('B', (K, N), (64, 64), 3),
                                         ('D', (N, M), (32, 128), 2)]:
            spec = specs[name]
            assert tuple(map(int, spec.global_dims)) == dims
            assert tuple(map(int, spec.global_strides)) == (dims[0] * 2,)
            assert tuple(map(int, spec.box_dims)) == box
            assert tuple(map(int, spec.element_strides)) == (1, 1)
            assert int(spec.swizzle) == swizzle
            assert int(spec.payload_bits) == int(spec.transaction_bits) == box[0] * box[1] * 16

        loads = [arguments(line) for line in cuda.splitlines()
                 if 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(' in line]
        stores = [arguments(line) for line in cuda.splitlines()
                  if 'ptx_cp_async_bulk_tensor_shared_to_global_2d(' in line]
        assert len(loads) == 3 and len(stores) == 4
        assert ['B_tensormap' in a[2] for a in loads] == [True, True, False]
        assert 2 * (int(specs['A'].transaction_bits) + 2 * int(specs['B'].transaction_bits)) // 8 == 65536
        assert cuda.count('65536, 0, actual_pred_ptr[0]);') == 1
        # Actual TMA destinations must populate the same stage/consumer planes
        # used by the MMA descriptors below, relative to their allocated bases.
        for call, name, consumer in zip(loads, ('B', 'B', 'A'), (0, 1, 0)):
            expression = call[0].split('pool_buf_ptr)[', 1)[1].rsplit(']', 1)[0]
            base = 512 + (depth * 8192 if name == 'B' else 0)
            for stage in range(depth):
                offset = eval(expression.replace('tma_phase_stage_ptr[0]', str(stage)), {'__builtins__': {}})
                assert offset == base + stage * 8192 + consumer * 4096

        # Enumerate scheduled cluster tiles and evaluate actual CUDA operands.
        # Every 128x32 output box must correspond to exactly those A/B rows.
        boxes = []
        for cluster in range(clusters):
            for work in range(cluster, total, clusters):
                group, within = divmod(work, 8 * (N // 256))
                height = min(8, M // 256 - group * 8)
                m, n = group * 8 + within % height, within // height
                for rank in range(2):
                    for consumer in range(2):
                        a_row = evaluate_coordinate(loads[2][-1], m, n, rank, consumer)
                        b_row = evaluate_coordinate(loads[consumer][-1], m, n, rank, consumer)
                        assert a_row == m * 256 + rank * 128
                        assert b_row == n * 256 + consumer * 128 + rank * 64
                        # CTA-group-2 combines B's two 64-row halves for N128.
                        for i, call in enumerate(stores):
                            col = evaluate_coordinate(call[-2], m, n, rank, consumer)
                            row = evaluate_coordinate(call[-1], m, n, rank, consumer)
                            assert row == a_row
                            assert col == b_row - rank * 64 + i * 32
                            assert 0 <= row <= M - 128 and 0 <= col <= N - 32
                            boxes.append((row, col))
        assert len(boxes) == len(set(boxes)) == M * N // (128 * 32)
        for call in loads:
            offsets = [evaluate_coordinate(call[-2], 0, 0, 0, 0, k) for k in range(K // 64)]
            assert offsets == list(range(0, K, 64))

        mma = [line for line in cuda.splitlines() if 'ptx_tcgen05_mma_cta_2_kind_f16_SS(' in line]
        assert len(mma) == 4
        for index in ('k', 'k_1'):
            loop = f'for (int {index} = 0; {index} < {K // 64}; ++{index})'
            assert (loop in cuda) == (K > 64)
        assert ('(bool)0' if K == 64 else '(0 < k_1)') in mma[0]
        assert all('(bool)1' in line for line in mma[1:])
        for i, line in enumerate(mma):
            assert '(uint)270532624' in line  # original 256x128 MMA descriptor
            operands = re.findall(r'tvm_builtin_smem_desc_add_16B_offset\(desc([AB])_ptr\[0\], (.*?)\),', line)
            assert [name for name, _ in operands] == ['A', 'B']
            for name, expression in operands:
                for stage in range(depth):
                    for consumer in range(2):
                        expr = expression.replace('mma_phase_stage_ptr[0]', str(stage))
                        address = eval(expr, {'__builtins__': {}}, {'warp_id_in_cta': 8 + consumer}) * 8
                        linear = stage * 8192 + (consumer * 4096 if name == 'B' else 0) + i * 16
                        assert address == int(specs[name].smem_buffer.layout.apply(linear)['m'])

        # Same two-consumer free barrier, four ready/free accumulator slots.
        for slot, count in ([(i, 1) for i in range(1, depth + 1)]
                             + [(i, 2) for i in range(depth + 1, 2 * depth + 1)]
                             + [(i, 1) for i in range(2 * depth + 1, 2 * depth + 5)]
                             + [(i, 256) for i in range(2 * depth + 5, 2 * depth + 9)]):
            init = f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), {count});'
            assert cuda.count(init) == 1
            assert cuda.index(init) < cuda.index('tvm_builtin_cuda_cluster_sync();')
        for role, ring, phase in [('tma', depth, 1), ('mma', depth, 0), ('ld', 2, 1), ('wb', 2, 0)]:
            prefix = role + '_phase'
            assert cuda.index(f'{prefix}_phase_ptr[0] = {phase};') < cuda.index('while (')
            assert cuda.count(f'{prefix}_stage_ptr[0] = 0;') == 2
            assert cuda.count(f'if ({prefix}_stage_ptr[0] == {ring})') == 1
            assert cuda.count(f'{prefix}_phase_ptr[0] = ({prefix}_phase_ptr[0] ^ 1);') == 1
        assert f'(tma_phase_stage_ptr[0] + {depth + 1})' in cuda
        assert cuda.count(f'ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + {depth + 1})])), 3);') == 1
        assert all([simulate_handoffs(cuda, 7, seed) for seed in range(3)])
        last_read = cuda.rindex('tvm_builtin_ptx_tcgen05_wait_ld();')
        fence = cuda.index('tvm_builtin_ptx_tcgen05_fence_before_thread_sync();', last_read)
        release = cuda.index('tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(', fence)
        assert last_read < fence < release < cuda.index('ptx_cp_async_bulk_tensor_shared_to_global_2d(', release)
        assert cuda.count('ptx_cp_async_bulk_wait_group_read_0();') == 4
        assert cuda.count('tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));') == 8
        assert cuda.rindex('tvm_builtin_cuda_cluster_sync();') < cuda.index('tcgen05_dealloc_cta_group_2(')


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                  (512, 9472, 192), (4096, 4352, 320)])
def test_inactive_dispatch_keeps_production_cuda_and_host_maps(arch, shape, compile_probe):
    _, control, control_maps = compile_probe(shape, 'baseline', arch)
    for variant in SHARE_A_VARIANTS:
        _, actual, maps = compile_probe(shape, variant, arch)
        assert actual == control
        assert host_maps(maps) == host_maps(control_maps)


@pytest.mark.parametrize('variant', SHARE_A_VARIANTS)
def test_share_a_requires_current_builder_and_refuses_reapplication(variant):
    pytest.importorskip('tvm')
    import gemm_kernels

    source = inspect.getsource(gemm_kernels.hgemm_v10)
    actual = variant_builder_source(source, 10, variant)
    old = Path(__file__).parents[1] / 'results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/step10_4096_baseline/builder.py'
    for invalid in (actual, old.read_text()):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(invalid, 10, variant)


def test_share_a_comparison_chain_keeps_measured_depth5_control():
    assert select_variants(10) == ['baseline']
    variants = ['baseline', 'tmem_input_depth5', *SHARE_A_VARIANTS]
    assert select_variants(10, ['tmem_share_a_depth6']) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'tmem_input_depth5', 'tmem_share_a_depth5']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, .8]
