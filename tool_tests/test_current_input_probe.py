"""Compiler contracts for input-ring experiments on the adopted TMEM path.

These checks validate layouts, work coverage and ownership, not GPU correctness
or performance. The probe still checks numerical outputs before timing on B300.
"""

import inspect
from pathlib import Path
import re

import pytest

from probe_persistent import (CURRENT_INPUT_VARIANTS, VERIFICATION_SHAPES,
                              build_variant, select_variants,
                              summarize_with_cache_control, variant_builder_source)
from test_step8_adoption import body
from test_tmem_double_buffer_probe import simulate_handoffs

ROOT = Path(__file__).parents[1]


@pytest.fixture
def compile_probe(tmp_path, monkeypatch):
    tvm = pytest.importorskip("tvm")
    from tvm.backend.cuda.tile_primitive.copy_async import tma

    plans = []
    emit = tma._emit_plan

    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)

    monkeypatch.setattr(tma, "_emit_plan", record)

    def compile_variant(shape, variant, arch):
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        plans.clear()
        with tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
            ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        return script, body(ex.mod.imports[0].inspect_source()), {
            p.spec.descriptor_name: p.spec for p in plans}

    return compile_variant


def descriptors(specs):
    fields = ("global_dims", "global_strides", "box_dims", "element_strides",
              "swizzle", "payload_bits", "transaction_bits", "smem_base_offset")
    return {name: tuple(str(getattr(spec, f)) for f in fields) for name, spec in specs.items()}


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape", [(4096,) * 3, *VERIFICATION_SHAPES["tmem_k128_depth2"]])
def test_input_tiles_match_tma_mma_and_barrier_ownership(arch, shape, compile_probe):
    import gemm_kernels

    M, N, K = shape
    total = M // 512 * (N // 128)
    max_clusters = gemm_kernels.SM_COUNT // 2
    waves = (total + max_clusters - 1) // max_clusters
    clusters = (total + waves - 1) // waves
    results = {}
    for variant in CURRENT_INPUT_VARIANTS:
        depth = 5 if variant == "tmem_input_depth5" else 2
        blk_k = 128 if variant == "tmem_k128_depth2" and K % 128 == 0 else 64
        stages = K // blk_k
        script, cuda, specs = compile_probe(shape, variant, arch)
        results[variant] = (cuda, descriptors(specs))
        dynamic = 1024 + depth * (2 * 128 + 64) * blk_k * 2 + 2 * 128 * 32 * 2
        assert dynamic <= 232448
        assert f'"tirx.dyn_smem_bytes": T.int64({dynamic})' in script
        assert f'T.cta_id([{clusters * 2}])' in script
        assert f'Asmem = T.decl_buffer(({depth}, 2, 128, {blk_k})' in script
        assert f'Bsmem = T.decl_buffer(({depth}, 64, {blk_k})' in script
        assert 'Dsmem = T.decl_buffer((2, 128, 32)' in script
        assert set(specs) == {"A", "B", "D"}

        # K128 uses two 64-element planes, not a contiguous 2D tensor map.
        # Check host descriptors as well as the generated device coordinates.
        for name, rows, height in (("A", M, 128), ("B", N, 64), ("D", M, 128)):
            spec = specs[name]
            wide_k = name != "D" and blk_k == 128
            dims = (64, rows, K // 64) if wide_k else ((N, M) if name == "D" else (K, rows))
            box = (64, height, 2) if wide_k else ((32, 128) if name == "D" else (64, height))
            strides = (K * 2, 128) if wide_k else (dims[0] * 2,)
            assert tuple(map(int, spec.global_dims)) == dims
            assert tuple(map(int, spec.global_strides)) == strides
            assert tuple(map(int, spec.box_dims)) == box
            assert tuple(map(int, spec.element_strides)) == (1,) * len(dims)
            assert int(spec.swizzle) == (2 if name == "D" else 3)
            bits = height * (32 if name == "D" else blk_k) * 16
            assert int(spec.transaction_bits) == int(spec.payload_bits) == bits
        expected_bytes = 2 * (2 * int(specs['A'].transaction_bits) + int(specs['B'].transaction_bits)) // 8
        assert f'{expected_bytes}, 0, actual_pred_ptr[0]);' in cuda
        load_op = f'ptx_cp_async_bulk_tensor_g2s_cluster_tile_{3 if blk_k == 128 else 2}d_mbar_addr('
        loads = [line.strip() for line in cuda.splitlines() if line.strip().startswith(load_op)]
        assert len(loads) == 3
        assert ['B_tensormap' in line for line in loads] == [True, False, False]
        rows = ['((_ptr_1[0] * 128) + (((int)tvm_builtin_cluster_ctaid_x()) * 64))',
                '((_ptr[0] * 512) + (((int)tvm_builtin_cluster_ctaid_x()) * 128))',
                '(((_ptr[0] * 512) + (((int)tvm_builtin_cluster_ctaid_x()) * 128)) + 256)']
        for line, row in zip(loads, rows):
            k = '0' if stages == 1 else f'(k * {2 if blk_k == 128 else 64})'
            suffix = f', 0, {row}, {k});' if blk_k == 128 else f', {k}, {row});'
            assert line.endswith(suffix)

        mma = [line for line in cuda.splitlines() if 'ptx_tcgen05_mma_cta_2_kind_f16_SS(' in line]
        assert len(mma) * 16 * stages == K
        for index in ('k', 'k_1'):
            loop = f'for (int {index} = 0; {index} < {stages}; ++{index})'
            assert (loop in cuda) == (stages > 1)
        assert ('(bool)0' if stages == 1 else '(0 < k_1)') in mma[0]
        assert all('(bool)1' in line for line in mma[1:])
        encodings = re.findall(r'encode_matrix_descriptor\(\(&\(desc([AB])_ptr\[0\]\)\), .*?, (\d+), (\d+), (\d+)\);', cuda)
        assert {name: (int(lead), int(stride), int(swizzle)) for name, lead, stride, swizzle in encodings} == {
            'A': (1024 if blk_k == 128 else 0, 64, 3),
            'B': (512 if blk_k == 128 else 0, 64, 3)}
        for i, line in enumerate(mma):
            desc = int(re.search(r', \(uint\)(\d+),', line)[1])
            assert ((desc >> 17) & 63) * 8 == 128
            assert '(ld_phase_stage_ptr[0] * 256)' in line
            # Decode the actual 16-byte descriptor offsets and compare them to
            # the TMA SMEM layout at each K16 block, stage and A consumer.
            operands = re.findall(r'tvm_builtin_smem_desc_add_16B_offset\(desc([AB])_ptr\[0\], (.*?)\),', line)
            assert [name for name, _ in operands] == ['A', 'B']
            for name, expression in operands:
                spec = specs[name]
                for stage in range(depth):
                    for consumer in range(2):
                        expr = expression.replace('mma_phase_stage_ptr[0]', str(stage))
                        actual = eval(expr, {'__builtins__': {}}, {'warp_id_in_cta': 8 + consumer}) * 8
                        linear = ((stage * 2 + consumer) * 128 * blk_k if name == 'A'
                                  else stage * 64 * blk_k) + i * 16
                        assert actual == int(spec.smem_buffer.layout.apply(linear)['m'])

        # Input-ready/free slots remain distinct from both TMEM generations.
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
            assert cuda.count(f'{prefix}_stage_ptr[0] = 0;') == 2  # init and wrap only
            assert cuda.count(f'if ({prefix}_stage_ptr[0] == {ring})') == 1
            assert cuda.count(f'{prefix}_phase_ptr[0] = ({prefix}_phase_ptr[0] ^ 1);') == 1
        assert f'(tma_phase_stage_ptr[0] + {depth + 1})' in cuda
        commit = f'ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + {depth + 1})])), 3);'
        assert cuda.count(commit) == 1
        assert cuda.index(mma[-1]) < cuda.index(commit)
        last_read = cuda.rindex('tvm_builtin_ptx_tcgen05_wait_ld();')
        fence = cuda.index('tvm_builtin_ptx_tcgen05_fence_before_thread_sync();', last_read)
        release = cuda.index('tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(', fence)
        store = cuda.index('ptx_cp_async_bulk_tensor_shared_to_global_2d(', release)
        assert last_read < fence < release < store
        assert cuda.count('ptx_cp_async_bulk_tensor_shared_to_global_2d(') == 4
        assert cuda.count('ptx_cp_async_bulk_wait_group_read_0();') == 4
        assert all([simulate_handoffs(cuda, 7, seed) for seed in range(3)])
        assert cuda.count(f'_ptr_2[0] = (_ptr_2[0] + {clusters});') == 3
        work = [tile for c in range(clusters) for tile in range(c, total, clusters)]
        assert sorted(work) == list(range(total))
        assert len(work) * 512 * 128 == M * N

    if K % 128:
        assert results['tmem_k128_depth2'] == results['tmem_input_depth2']


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
@pytest.mark.parametrize("shape", [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                  (512, 9472, 192), (4096, 4352, 320)])
def test_single_slot_and_wide_paths_keep_production_cuda_and_maps(arch, shape, compile_probe):
    _, expected, maps = compile_probe(shape, "baseline", arch)
    for variant in CURRENT_INPUT_VARIANTS:
        _, actual, actual_maps = compile_probe(shape, variant, arch)
        assert actual == expected
        assert descriptors(actual_maps) == descriptors(maps)


@pytest.mark.parametrize("variant", CURRENT_INPUT_VARIANTS)
def test_input_probe_requires_current_baseline_and_rejects_reapplication(variant):
    pytest.importorskip("tvm")
    import gemm_kernels

    current = inspect.getsource(gemm_kernels.hgemm_v10)
    transformed = variant_builder_source(current, 10, variant)
    historical = ROOT / 'results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/step10_4096_baseline/builder.py'
    for source in (transformed, historical.read_text()):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(source, 10, variant)


def test_input_comparison_includes_depth_control_without_changing_defaults():
    assert select_variants(10) == ['baseline']
    variants = ['baseline', *CURRENT_INPUT_VARIANTS]
    assert select_variants(10, ['tmem_k128_depth2', 'tmem_input_depth5']) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'tmem_input_depth2', 'baseline']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, 2]
