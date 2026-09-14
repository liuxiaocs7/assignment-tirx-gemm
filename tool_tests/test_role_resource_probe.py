"""Check role-wide register operations and reordered TMA requests before B300."""

from pathlib import Path
import json
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (build_variant, select_variants, variant_builder_source,
                              VERIFICATION_SHAPES, summarize_with_cache_control,
                              check_role_register_budget)
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES['role_registers'], (512, 256, 64)])
def test_role_experiments_preserve_all_work_and_synchronization(arch, shape, tmp_path):
    tvm = pytest.importorskip('tvm')
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    sources = {}
    for variant in ('baseline', 'role_registers', 'tma_b_first'):
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        assert '"tirx.dyn_smem_bytes": T.int64(230400)' in script
        assert 'Asmem = T.decl_buffer((4, 2, 128, 64)' in script
        assert 'Dsmem = T.decl_buffer((2, 128, 64)' in script
        with target:
            ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        sources[variant] = ex.mod.imports[0].inspect_source()
    baseline, registers, b_first = map(body, sources.values())
    if shape == (4096,) * 3:
        recorded = ROOT / 'results_b300/step10_epilogue.Y2EFaW/step10/step10_4096_baseline/module_01.cu'
        assert baseline == body(recorded.read_text())
    # Register allocation must be uniform over an entire warpgroup. No
    # elected-lane predicate or CTA barrier may precede a blocked acquisition.
    block = '''  if ((warp_id_in_cta >> 2) == 2) {
    tvm_builtin_ptx_setmaxnreg_dec_64();
  } else {
    tvm_builtin_ptx_setmaxnreg_inc_208();
  }
'''
    assert registers.count(block) == 1
    assert registers.replace(block, '') == baseline
    allocation = registers.index(block)
    for operation in ('tvm_builtin_ptx_mbarrier_init(', 'tvm_builtin_cuda_cta_sync();',
                      'tvm_builtin_cuda_cluster_sync();', 'tvm_builtin_elect_one_sync_op()'):
        assert allocation < registers.index(operation)
    assert 'setmaxnreg.dec.sync.aligned.u32 64;' in sources['role_registers']
    assert 'setmaxnreg.inc.sync.aligned.u32 208;' in sources['role_registers']
    # Each warpgroup's full 128 lanes uses the same action. Requested total
    # fits below the uploaded baseline's 167 registers * 384 threads, so the
    # two increases can both finish after WG2 returns its unused registers.
    budgets = [64 if (thread // 32 >> 2) == 2 else 208 for thread in range(384)]
    assert sum(budgets) <= 167 * 384
    assert all(len(set(budgets[wg * 128:(wg + 1) * 128])) == 1 for wg in range(3))

    # TMA sources, destinations and completion barriers are unchanged; only
    # the order of the B/A0/A1 requests differs. Check all other generated
    # operations verbatim, including expect_tx after all three requests.
    op = 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr'
    pattern = re.compile(r'^ +'+op+r'\([^\n]+\);\n', re.M)
    old = pattern.findall(baseline)
    new = pattern.findall(b_first)
    assert len(old) == len(new) == 3
    assert ['B_tensormap' in line for line in new] == [True, False, False]
    assert new == [old[2], old[0], old[1]]
    assert expand_cse(pattern.sub('', b_first)) == expand_cse(pattern.sub('', baseline))
    assert b_first.index(new[-1]) < b_first.index('98304, 0, actual_pred_ptr[0]);')
    assert b_first.count('tvm_builtin_ptx_tcgen05_wait_ld();') == 8
    assert b_first.count('ptx_cp_async_bulk_wait_group_read_0();') == 4


@pytest.mark.parametrize('variant', ['role_registers', 'tma_b_first'])
def test_role_probe_rejects_reapplication_and_unadopted_source(variant, tmp_path):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, variant, tmp_path)
    with pytest.raises(ValueError):
        variant_builder_source((tmp_path / 'builder.py').read_text(), 10, variant)
    before = (ROOT / 'results_b300/step10_fused_a.1VXYz2/step10/step10_4096_baseline/builder.py').read_text()
    with pytest.raises(ValueError, match='adopted Step 10'):
        variant_builder_source(before, 10, variant)


def test_role_variants_use_independent_production_controls():
    variants = ['baseline', 'role_registers', 'tma_b_first']
    assert select_variants(10) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20, 40], [8, 10, 20], [5, 10, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert all(r['comparison_control'] == 'baseline' and r['status'] == 'SLOW' for r in rows)
    assert [r['paired_control_speedup'] for r in rows] == [1, 2, 2]


@pytest.mark.parametrize('reported,valid', [(160, True), (167, True), (168, True),
                                          (208, True), (152, False), (216, False), (None, False)])
def test_register_pool_checked_before_binary_is_returned_and_hook_restored(reported, valid, tmp_path):
    pytest.importorskip('tvm')
    import tvm_ffi
    from tvm.support import nvcc  # Ensure the real callback is registered.

    name = 'tvm_callback_cuda_compile'
    saved = tvm_ffi.get_global_func(name)
    calls = []
    def capture(code):
        calls.append(str(code))
        report = f'Function kernel_kernel:\n REG:{reported} STACK:0\n' if reported else 'Unavailable\n'
        (tmp_path / 'module_01.resources.txt').write_text(report)
        return bytearray(b'cubin')
    tvm_ffi.register_global_func(name, capture, override=True)
    try:
        if valid:
            with check_role_register_budget('role_registers', tmp_path):
                result = tvm_ffi.get_global_func(name)('compiled source')
                assert bytes(result) == b'cubin'
        else:
            with pytest.raises(RuntimeError, match='role_registers'):
                with check_role_register_budget('role_registers', tmp_path):
                    tvm_ffi.get_global_func(name)('compiled source')
        assert calls == ['compiled source']
        if reported:
            metadata = json.loads((tmp_path / 'register_budget.json').read_text())
            assert metadata['valid'] is valid and metadata['initial_registers'] == reported
        # A failed check must not leak a compiler wrapper into later runs.
        tvm_ffi.get_global_func(name)('after context')
        assert calls[-1] == 'after context'
    finally:
        tvm_ffi.register_global_func(name, saved, override=True)
