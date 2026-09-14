"""Check emitted schedules and unchanged protocols; GPU timing remains external."""

from collections import Counter
import inspect
import re

import pytest

from probe_persistent import (CURRENT_L2_VARIANTS, VERIFICATION_SHAPES,
                              select_variants, summarize_with_cache_control,
                              trial_order, variant_builder_source)
from test_current_input_probe import compile_probe, descriptors  # noqa: F401
from test_share_a_probe import arguments, evaluate_coordinate
from test_tmem_double_buffer_probe import simulate_handoffs


def emitted_schedules(cuda):
    """Extract initialization and the three roles' actual next-tile CUDA."""
    init = re.search(r'^  _ptr_2\[0\] = .*?(?=^  uint64_t\* remote_mbar_ptr)',
                     cuda, re.M | re.S)
    assert init is not None
    updates = []
    for match in re.finditer(r'^( +)_ptr_2\[0\] = \(_ptr_2\[0\] \+ \d+\);', cuda, re.M):
        lines = []
        for line in cuda[match.start():].splitlines():
            if line.strip() and len(line) - len(line.lstrip()) < len(match[1]):
                break
            lines.append(line)
        updates.append('\n'.join(lines))
    assert len(updates) == 3
    return [compile_schedule(block) for block in [init[0], *updates]]


def compile_schedule(block):
    """Translate only emitted integer assignments/if blocks, not a scheduler model.

    All evaluated work indices are nonnegative, so integer division/remainder
    match CUDA. Unexpected emission fails rather than silently skipping a line.
    """
    def expression(text):
        text = re.sub(r'\((?:int|bool)\)', '', text)
        return text.replace('blockIdx.x', 'block').replace('/', '//')

    lines, depth = [], 0
    for line in block.splitlines():
        line = line.strip()
        if line == '} else {':
            depth -= 1
            lines.append('    ' * depth + 'else:')
            depth += 1
        elif line == '}':
            depth -= 1
        elif line.startswith('if (') and line.endswith(') {'):
            lines.append('    ' * depth + 'if ' + expression(line[4:-3]) + ':')
            depth += 1
        else:
            assignment = re.fullmatch(r'(?:int )?(\w+(?:\[0\])?) = (.*);', line)
            assert assignment is not None, line
            lines.append('    ' * depth + assignment[1] + ' = ' + expression(assignment[2]))
        assert depth >= 0
    assert depth == 0
    return compile('\n'.join(lines), '<emitted CUDA scheduler>', 'exec')


def schedule_states(programs, clusters, total):
    for cluster in range(clusters):
        for rank in range(2):
            # Run each role independently, with its own persistent state.
            roles = [dict(block=cluster * 2 + rank, _ptr=[0], _ptr_1=[0], _ptr_2=[0],
                          tile_scheduler_tile_count_ptr=[0]) for _ in range(3)]
            for state in roles:
                exec(programs[0], {'__builtins__': {}}, state)
            for visit, work in enumerate(range(cluster, total, clusters)):
                assert all(s['_ptr_2'][0] == work for s in roles)
                assert all(s['tile_scheduler_tile_count_ptr'][0] == visit for s in roles)
                positions = [(s['_ptr'][0], s['_ptr_1'][0]) for s in roles]
                assert positions.count(positions[0]) == 3
                yield work, rank, positions[0]
                for program, state in zip(programs[1:], roles):
                    exec(program, {'__builtins__': {}}, state)
            assert all(s['_ptr_2'][0] >= total for s in roles)


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(4096,) * 3, *VERIFICATION_SHAPES['tmem_l2_group4']])
def test_l2_schedule_covers_outputs_and_preserves_data_path(arch, shape, compile_probe):
    import gemm_kernels

    M, N, K = shape
    rows, cols = M // 512, N // 128
    total = rows * cols
    waves = (total + gemm_kernels.SM_COUNT // 2 - 1) // (gemm_kernels.SM_COUNT // 2)
    clusters = (total + waves - 1) // waves
    _, control, control_maps = compile_probe(shape, 'baseline', arch)
    # Compare actual hardware calls/operands, including all waits, barriers,
    # descriptors, MMA, TMEM reads, TMA loads/stores, and allocation/free.
    def operations(cuda):
        return [line.strip() for line in cuda.splitlines()
                if re.match(r'\s*(?:tvm_builtin_|ptx_)', line)]

    for variant, group in CURRENT_L2_VARIANTS.items():
        script, cuda, maps = compile_probe(shape, variant, arch)
        assert f'T.cta_id([{clusters * 2}])' in script
        assert '"tirx.dyn_smem_bytes": T.int64(181248)' in script
        assert descriptors(maps) == descriptors(control_maps)
        assert operations(cuda) == operations(control)
        assert cuda.count(f'if (!((_ptr_2[0] < {total})))') == 3
        assert all(simulate_handoffs(cuda, 7, seed) for seed in range(3))
        # Enumerate groups independently of the compiler's div/mod mapping.
        expected = [(m, n) for start in range(0, rows, group)
                    for n in range(cols) for m in range(start, min(start + group, rows))]
        loads = [arguments(line) for line in cuda.splitlines()
                 if 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(' in line]
        stores = [arguments(line) for line in cuda.splitlines()
                  if 'ptx_cp_async_bulk_tensor_shared_to_global_2d(' in line]
        assert len(loads) == 3 and len(stores) == 4
        assert ['B_tensormap' in call[2] for call in loads] == [True, False, False]
        boxes = []
        for work, rank, (m, n) in schedule_states(emitted_schedules(cuda), clusters, total):
            assert (m, n) == expected[work]
            for consumer in range(2):
                a_row = evaluate_coordinate(loads[consumer + 1][-1], m, n, rank, consumer)
                b_row = evaluate_coordinate(loads[0][-1], m, n, rank, consumer)
                assert a_row == m * 512 + consumer * 256 + rank * 128
                assert b_row == n * 128 + rank * 64
                for i, call in enumerate(stores):
                    row = evaluate_coordinate(call[-1], m, n, rank, consumer)
                    col = evaluate_coordinate(call[-2], m, n, rank, consumer)
                    assert row == a_row and col == n * 128 + i * 32
                    boxes.append((row, col))
        assert len(boxes) == len(set(boxes)) == M * N // (128 * 32)
        assert set(boxes) == {(m, n) for m in range(0, M, 128) for n in range(0, N, 32)}


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                 (512, 9472, 192), (4096, 4352, 320)])
def test_l2_keeps_single_slot_and_wide_fallbacks(arch, shape, compile_probe):
    _, control, maps = compile_probe(shape, 'baseline', arch)
    for variant in CURRENT_L2_VARIANTS:
        _, actual, actual_maps = compile_probe(shape, variant, arch)
        assert actual == control
        assert descriptors(actual_maps) == descriptors(maps)


@pytest.mark.parametrize('variant', CURRENT_L2_VARIANTS)
def test_l2_changes_one_setting_and_rejects_unexpected_builders(variant):
    pytest.importorskip('tvm')
    import gemm_kernels

    source = inspect.getsource(gemm_kernels.hgemm_v10)
    actual = variant_builder_source(source, 10, variant)
    replacement = f'l2_group_size={CURRENT_L2_VARIANTS[variant]} if TMEM_BUFFERS == 2 else 8'
    assert actual.replace(replacement, 'l2_group_size=8') == source
    for invalid in (actual, source.replace('PIPE_DEPTH = 4', 'PIPE_DEPTH = 5'),
                    source.replace('NUM_CONSUMER = 2', 'NUM_CONSUMER = 1'),
                    source.replace('l2_group_size=8', 'l2_group_size=4'),
                    source.replace('NARROW_N =', 'OLD_NARROW_N =')):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(invalid, 10, variant)
    with pytest.raises(ValueError, match='unsupported Step 9 variant'):
        variant_builder_source(source, 9, variant)


def test_l2_direct_controls_and_eight_trial_balance():
    assert select_variants(10) == ['baseline']
    variants = ['baseline', *CURRENT_L2_VARIANTS]
    assert select_variants(10, CURRENT_L2_VARIANTS) == variants
    orders = [trial_order(4, trial) for trial in range(8)]
    for a in range(4):
        assert Counter(order.index(a) for order in orders) == {i: 2 for i in range(4)}
        for b in range(a + 1, 4):
            assert sum(order.index(a) < order.index(b) for order in orders) == 4
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8], [5, 10]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline'] * 4
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2.5, 2]
