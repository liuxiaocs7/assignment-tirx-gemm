"""Validate accumulator ownership and independent ready/free phases before GPU timing."""

from pathlib import Path
import random
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (VERIFICATION_SHAPES, build_variant, double_buffer_tmem,
                              select_variants, summarize_with_cache_control, variant_builder_source)
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body

VARIANT = "n128_tmem_double_buffer"


def simulate_handoffs(cuda, tiles, seed):
    """Interleave MMA issue/completion and independently delayed CTA readers.

    Evaluate barrier slot and TMEM address expressions from the generated CUDA.
    A slot cannot be overwritten until both CTAs' 128 readers have released it;
    a reader must see its own tile, even when the producer is two tiles ahead.
    This models ownership, not Tensor Core execution or numerical accuracy.
    """
    wait_op = 'tvm_builtin_ptx_mbarrier_try_wait('
    commit_op = 'ptx_tcgen05_commit_cta_group_2_multicast('
    release_op = 'tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred('
    mma = next(line for line in cuda.splitlines() if 'ptx_tcgen05_mma_cta_2_kind_f16_SS(' in line)
    mma_expr = re.search(r'\(mma_tmem_base \+ \(\(uint\)(.*?)\)\),', mma)[1]
    read = next(line for line in cuda.splitlines() if 'tvm_builtin_ptx_tcgen05_ld_32x32b_x32(' in line)
    read_expr = read.split(', mma_tmem_base, 0, ')[1].removesuffix(');')

    def expression(op, role):
        line = next(line for line in cuda.splitlines() if op in line and role + '_stage_ptr' in line)
        return line.split('pool_buf_ptr)[', 1)[1].split('])),', 1)[0]

    free_wait = expression(wait_op, 'ld_phase')
    full_commit = expression(commit_op, 'ld_phase')
    full_wait = expression(wait_op, 'wb_phase')
    free_release = expression(release_op, 'wb_phase')
    initial = {role: int(re.search(rf'{role}_phase_ptr\[0\] = (\d);', cuda)[1])
               for role in ('ld_phase', 'wb_phase')}
    depth = {role: int(re.search(rf'if \({role}_stage_ptr\[0\] == (\d+)\)', cuda)[1])
             for role in initial}

    def evaluate(expr, role, tile, consumer):
        expr = expr.replace(f'{role}_stage_ptr[0]', str(tile % depth[role]))
        warp = 8 + consumer if role == 'ld_phase' else consumer * 4
        return eval(expr, {'__builtins__': {}}, {'warp_id_in_cta': warp})

    def phase(role, tile):
        return initial[role] ^ ((tile // depth[role]) % 2)

    # All mbarriers initialize with phase zero. Model each rank's local full
    # barrier separately, and the single remote CTA0 free barrier per slot.
    full, free, arrivals, contents = {}, {}, {}, {}
    issued, completed = [0, 0], [0, 0]
    consumed = [[0, 0], [0, 0]]
    pending = [[], []]
    rng = random.Random(seed)
    overlap = False
    for _ in range(tiles * 2 * 4):
        actions = []
        for consumer in range(2):
            tile = issued[consumer]
            slot = evaluate(free_wait, 'ld_phase', tile, consumer)
            if tile < tiles and free.get(slot, 0) != phase('ld_phase', tile):
                actions.append(('issue', consumer, 0))
            if pending[consumer]:
                actions.append(('complete', consumer, 0))
            for rank in range(2):
                tile = consumed[consumer][rank]
                slot = evaluate(full_wait, 'wb_phase', tile, consumer)
                if tile < tiles and full.get((rank, slot), 0) != phase('wb_phase', tile):
                    actions.append(('read', consumer, rank))
        assert actions, 'barrier deadlock before all tiles complete'
        action, consumer, rank = rng.choice(actions)
        if action == 'issue':
            tile = issued[consumer]
            address = evaluate(mma_expr, 'ld_phase', tile, consumer)
            assert contents.get(address) is None, 'MMA overwrote an unread accumulator'
            assert 0 <= address <= 384
            contents[address] = tile
            pending[consumer].append(tile)
            issued[consumer] += 1
            overlap |= issued[consumer] - min(consumed[consumer]) >= 2
        elif action == 'complete':
            tile = pending[consumer].pop(0)
            assert tile == completed[consumer]
            slot = evaluate(full_commit, 'ld_phase', tile, consumer)
            for r in range(2):
                full[r, slot] = full.get((r, slot), 0) ^ 1
            completed[consumer] += 1
        else:
            tile = consumed[consumer][rank]
            address = evaluate(read_expr, 'wb_phase', tile, consumer)
            assert contents[address] == tile, 'writeback read a stale/different output tile'
            slot = evaluate(free_release, 'wb_phase', tile, consumer)
            arrivals[slot] = arrivals.get(slot, 0) + 128
            if arrivals[slot] == 256:
                free[slot] = free.get(slot, 0) ^ 1
                arrivals[slot] = 0
                contents[address] = None
            consumed[consumer][rank] += 1
    assert issued == completed == [tiles] * 2
    assert consumed == [[tiles, tiles], [tiles, tiles]]
    assert not any(arrivals.values()) and all(v is None for v in contents.values())
    return overlap


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES[VARIANT], (512, 256, 64)])
def test_tmem_double_buffer_preserves_work_and_guards_each_slot(arch, shape, tmp_path, pre_tmem_step10):
    tvm = pytest.importorskip('tvm')
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    sources = []
    for variant in ('n128_epi32', VARIANT):
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        script = kernel.script()
        assert '"tirx.dyn_smem_bytes": T.int64(181248)' in script
        assert 'Asmem = T.decl_buffer((4, 2, 128, 64)' in script
        assert 'Bsmem = T.decl_buffer((4, 64, 64)' in script
        assert 'Dsmem = T.decl_buffer((2, 128, 32)' in script
        with target:
            ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        sources.append(body(ex.mod.imports[0].inspect_source()))
    control, actual = sources
    if shape == (4096,) * 3:
        saved = ROOT / 'results_b300/step10_n128.eVKFm2/step10/step10_4096_n128_epi32/module_01.cu'
        assert control == body(saved.read_text())

    # Exact accumulator address/phase deltas; everything else must match the
    # measured narrow control after removing unused CSE declarations.
    normalized = actual
    for slot in (11, 12):
        line = f'    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), 1);\n'
        assert normalized.count(line) == 1
        normalized = normalized.replace(line, '')
    for slot in (15, 16):
        line = f'    tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{slot}])), 256);\n'
        assert normalized.count(line) == 1
        normalized = normalized.replace(line, '')
    for slot in (13, 14):
        normalized = normalized.replace(f'pool_buf_ptr)[{slot}])), 256);', f'pool_buf_ptr)[{slot - 2}])), 256);')
    for role, warp in [('ld_phase', '(warp_id_in_cta & 3)'), ('wb_phase', '(warp_id_in_cta >> 2)')]:
        normalized = normalized.replace(f'(({role}_stage_ptr[0] * 256) + ({warp} * 128))', f'({warp} * 128)')
        for offset in (9, 13):
            normalized = normalized.replace(f'((({role}_stage_ptr[0] * 2) + {warp}) + {offset})',
                                              f'({warp} + {9 if offset == 9 else 11})')
        pad = '              ' if role == 'ld_phase' else '        '
        advance = (f'{pad}{role}_stage_ptr[0] = ({role}_stage_ptr[0] + 1);\n'
                   f'{pad}if ({role}_stage_ptr[0] == 2) {{\n'
                   f'{pad}  {role}_stage_ptr[0] = 0;\n'
                   f'{pad}  {role}_phase_ptr[0] = ({role}_phase_ptr[0] ^ 1);\n{pad}}}\n')
        assert normalized.count(advance) == 1
        normalized = normalized.replace(advance, '')
        wait = next(line for line in normalized.splitlines(True)
                    if 'tvm_builtin_ptx_mbarrier_try_wait(' in line and role + '_phase_ptr' in line)
        normalized = normalized.replace(wait, wait + f'{pad}{role}_phase_ptr[0] = ({role}_phase_ptr[0] ^ 1);\n')
    assert expand_cse(normalized) == expand_cse(control)

    # The slot stays stable until ready commit / read completion and release.
    assert actual.index('ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[(((ld_phase') < actual.index(
        'ld_phase_stage_ptr[0] = (ld_phase_stage_ptr[0] + 1);')
    last_read = actual.rindex('tvm_builtin_ptx_tcgen05_wait_ld();')
    fence = actual.index('tvm_builtin_ptx_tcgen05_fence_before_thread_sync();', last_read)
    release = actual.index('tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(', fence)
    advance = actual.index('wb_phase_stage_ptr[0] = (wb_phase_stage_ptr[0] + 1);', release)
    store = actual.index('ptx_cp_async_bulk_tensor_shared_to_global_2d(', advance)
    assert last_read < fence < release < advance < store
    assert actual.count('tcgen05_alloc_cta_group_2(') == actual.count('tcgen05_dealloc_cta_group_2(') == 1
    assert 'tcgen05_dealloc_cta_group_2(((uint*)pool_buf_ptr)[0], 512);' in actual
    # Same role owns the entire allocated 512-column region through teardown.
    assert actual.rindex('tvm_builtin_cuda_cluster_sync();') < actual.index('tcgen05_dealloc_cta_group_2(')

    # Exercise odd/even slot generations under delayed, independent CTA reads.
    # Source equivalence above preserves input-ring phases across all short K.
    assert any([simulate_handoffs(actual, 7, seed) for seed in range(12)])


def test_tmem_double_buffer_requires_its_control_and_refuses_reapplication(tmp_path, pre_tmem_step10):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, VARIANT, tmp_path)
    builder = (tmp_path / 'builder.py').read_text()
    with pytest.raises(ValueError):
        double_buffer_tmem(builder)
    with pytest.raises(ValueError):
        variant_builder_source(builder, 10, VARIANT)
    import inspect
    import gemm_kernels
    with pytest.raises(ValueError, match='n128_epi32'):
        double_buffer_tmem(inspect.getsource(gemm_kernels.hgemm_v10))


def test_explicit_historical_comparison_includes_measured_controls():
    variants = ['baseline', 'n_tile_128', 'n128_epi32', VARIANT]
    assert select_variants(10) == ['baseline']
    assert select_variants(10, [VARIANT]) == variants
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'n_tile_128', 'n128_epi32']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2, .8]
