"""Validate the narrow-N work, TMA layouts, and ring protocol before B300."""

from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (NARROW_N_VARIANTS, VERIFICATION_SHAPES, build_variant,
                              select_variants, summarize_with_cache_control,
                              variant_builder_source)
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES['n128_epi32_depth5'], (512, 256, 64)])
def test_narrow_n_work_and_phase_survive_codegen(arch, shape, tmp_path, monkeypatch):
    tvm = pytest.importorskip('tvm')
    import gemm_kernels
    from tvm.backend.cuda.tile_primitive.copy_async import tma
    plans = []
    emit = tma._emit_plan
    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)
    monkeypatch.setattr(tma, '_emit_plan', record)
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    M, N, K = shape
    total = M // 512 * (N // 128)
    waves = (total + gemm_kernels.SM_COUNT // 2 - 1) // (gemm_kernels.SM_COUNT // 2)
    clusters = (total + waves - 1) // waves
    sources = []
    for variant, depth, epi in [('n_tile_128', 4, 64), ('n128_epi32', 4, 32),
                                 ('n128_epi32_depth5', 5, 32)]:
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        dynamic = 1024 + depth * (2 * 128 + 64) * 64 * 2 + 2 * 128 * epi * 2
        assert dynamic <= 232448
        assert f'T.cta_id([{clusters * 2}])' in script
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert f'Asmem = T.decl_buffer(({depth}, 2, 128, 64)' in script
        assert f'Bsmem = T.decl_buffer(({depth}, 64, 64)' in script
        d_line = next(line for line in script.splitlines() if 'Dsmem = T.decl_buffer' in line)
        assert f'((2, 128, {epi})' in d_line
        assert f'T.ComposeLayout(3, {3 if epi == 64 else 2}, 3,' in d_line
        plans.clear()
        with target:
            ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        cuda = body(ex.mod.imports[0].inspect_source())
        sources.append(cuda)
        # Inspect actual TMA maps, including the EPI32 swizzle conversion.
        specs = {p.spec.descriptor_name: p.spec for p in plans}
        assert set(specs) == {'A', 'B', 'D'}
        for name, dims, box, swizzle in [('A', (K, M), (64, 128), 3),
                                         ('B', (K, N), (64, 64), 3),
                                         ('D', (N, M), (epi, 128), 3 if epi == 64 else 2)]:
            spec = specs[name]
            assert tuple(map(int, spec.global_dims)) == dims
            assert tuple(map(int, spec.global_strides)) == (dims[0] * 2,)
            assert tuple(map(int, spec.box_dims)) == box
            assert tuple(map(int, spec.element_strides)) == (1, 1)
            assert int(spec.swizzle) == swizzle
            assert int(spec.payload_bits) == int(spec.transaction_bits) == box[0] * box[1] * 16
        assert 2 * (2 * int(specs['A'].transaction_bits) + int(specs['B'].transaction_bits)) // 8 == 81920

        # Source STS addressing and the TMA descriptor must see the same
        # swizzled rows. Derive the linear row offsets from generated CUDA.
        offsets = re.findall(r'^ +s_off_ptr(?:_\d+)?\[0\] = ([^;]+);$', cuda, re.M)
        assert len(offsets) == 128 // epi
        spec = specs['D']
        for consumer in range(2):
            for expression in offsets:
                for warp in range(4):
                    for lane in (0, 7, 31):
                        expr = expression.replace('((int)threadIdx.x)', str(lane))
                        linear = eval(expr, {'__builtins__': {}}, {'warp_id_in_cta': consumer * 4 + warp})
                        assert linear == (consumer * 128 + warp * 32 + lane) * epi
                        for col in (0, epi - 1):
                            expected = (linear + col) ^ (((linear + col) >> 3) & (56 if epi == 64 else 24))
                            actual = spec.smem_buffer.layout.apply(linear + col)
                            assert int(actual['m']) == expected

        # Every barrier slot has the original arrivals; only depth shifts offsets.
        for slot, arrivals in ([(i, 1) for i in range(1, depth + 1)]
                               + [(i, 2) for i in range(depth + 1, 2 * depth + 1)]
                               + [(2 * depth + 1, 1), (2 * depth + 2, 1),
                                  (2 * depth + 3, 256), (2 * depth + 4, 256)]):
            init = f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), {arrivals});'
            assert cuda.count(init) == 1
            assert cuda.index(init) < cuda.index('tvm_builtin_cuda_cta_sync();')
        for phase in ('tma', 'mma'):
            assert cuda.count(f'{phase}_phase_stage_ptr[0] = 0;') == 2
            assert cuda.count(f'{phase}_phase_stage_ptr[0] == {depth}') == 1
            assert cuda.count(f'{phase}_phase_phase_ptr[0] = ({phase}_phase_phase_ptr[0] ^ 1);') == 1
        assert cuda.count(f'_ptr_2[0] = (_ptr_2[0] + {clusters});') == 3

        # Each full barrier still covers both CTAs' A0/A1/B; B is half-width.
        assert '81920, 0, actual_pred_ptr[0]);' in cuda
        load = 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr'
        loads = [line.strip() for line in cuda.splitlines() if line.strip().startswith(load + '(')]
        assert len(loads) == 3
        assert ['B_tensormap' in line for line in loads] == [True, False, False]
        k_offset = '0' if K == 64 else '(k * 64)'
        assert loads[0].endswith(f', {k_offset}, ((_ptr_1[0] * 128) + (((int)tvm_builtin_cluster_ctaid_x()) * 64)));')
        assert loads[1].endswith(f', {k_offset}, ((_ptr[0] * 512) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));')
        assert loads[2].endswith(f', {k_offset}, (((_ptr[0] * 512) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)) + 256));')

        # Four K16 MMA instructions per K64 stage. Decode N from the emitted
        # instruction descriptor; consumer TMEM slices must be disjoint.
        mma = [line for line in cuda.splitlines() if 'ptx_tcgen05_mma_cta_2_kind_f16_SS(' in line]
        assert len(mma) == 4
        for line in mma:
            desc = int(re.search(r', \(uint\)(\d+),', line)[1])
            assert ((desc >> 17) & 63) * 8 == 128
            assert 'mma_tmem_base + ((uint)((warp_id_in_cta & 3) * 128))' in line
        assert ('(bool)0' if K == 64 else '(0 < k_1)') in mma[0]
        assert all('(bool)1' in line for line in mma[1:])
        assert cuda.count('tvm_builtin_ptx_tcgen05_wait_ld();') == 4
        assert 'half Dreg_f16_ptr[128];' in cuda
        read_op = 'tvm_builtin_ptx_tcgen05_ld_32x32b_x32('
        reads = [line for line in cuda.splitlines() if read_op in line]
        for i, line in enumerate(reads):
            offset = '((warp_id_in_cta >> 2) * 128)' if i == 0 else f'(((warp_id_in_cta >> 2) * 128) + {32 * i})'
            assert line.endswith(f', mma_tmem_base, 0, {offset});')
        assert len(reads) == 4

        store_op = 'ptx_cp_async_bulk_tensor_shared_to_global_2d('
        stores = [line for line in cuda.splitlines() if store_op in line]
        assert len(stores) == 128 // epi
        for i, line in enumerate(stores):
            col = '(_ptr_1[0] * 128)' if i == 0 else f'((_ptr_1[0] * 128) + {i * epi})'
            assert line.endswith(f', {col}, (((_ptr[0] * 512) + ((warp_id_in_cta >> 2) * 256))'
                                 ' + (((int)tvm_builtin_cluster_ctaid_x()) * 128)));')
        assert cuda.count('ptx_cp_async_bulk_wait_group_read_0();') == len(stores)
        assert cuda.count('tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));') == 2 * len(stores)
        assert cuda.rindex('tvm_builtin_ptx_tcgen05_wait_ld();') < cuda.index(
            'tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(') < cuda.index(store_op)

    # Reducing EPI_N changes only writeback storage/layout, not producer work.
    marker = '      alignas(64) float Dreg_ptr['
    assert expand_cse(sources[0].split(marker)[0]) == expand_cse(sources[1].split(marker)[0])
    # Output coordinates above form nonoverlapping 128xEPI boxes that cover D.
    # Check the tile schedule covers every 512x128 rectangle exactly once.
    work = [tile for cluster in range(clusters) for tile in range(cluster, total, clusters)]
    assert sorted(work) == list(range(total))
    assert sum(2 * 2 * 128 * 128 for _ in work) == M * N


@pytest.mark.parametrize('variant', NARROW_N_VARIANTS)
def test_narrow_n_transform_refuses_reapplication_or_old_control(variant, tmp_path):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, variant, tmp_path)
    with pytest.raises(ValueError):
        variant_builder_source((tmp_path / 'builder.py').read_text(), 10, variant)
    before = ROOT / 'results_b300/step10_roles.rx5lNL/step10/step10_4096_baseline/builder.py'
    with pytest.raises(ValueError, match='B-first baseline'):
        variant_builder_source(before.read_text(), 10, variant)


def test_narrow_n_comparison_chain_scores_each_change_against_its_control():
    variants = ['baseline', *NARROW_N_VARIANTS]
    assert select_variants(10, ['n128_epi32_depth5']) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'n_tile_128', 'n128_epi32']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, .8]
    assert all(r['status'] == 'SLOW' for r in rows)
