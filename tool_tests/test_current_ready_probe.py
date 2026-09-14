"""Check emitted split-ready barriers, data identity and asynchronous ownership."""

from collections import Counter
import inspect
import random
import re

import pytest

from probe_persistent import select_variants, trial_order, variant_builder_source
from probe_step10_ready import READY_VARIANT, READY_VERIFY_SHAPES
from test_current_input_probe import compile_probe, descriptors  # noqa: F401
from test_cache_followup_probe import expand_cse
from test_share_a_probe import arguments
from test_tmem_double_buffer_probe import simulate_handoffs


WAIT = 'tvm_builtin_ptx_mbarrier_try_wait('
EXPECT = 'tvm_builtin_ptx_mbarrier_arrive_expect_tx_shared_cluster_remote_pred('
COPY = 'ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr('
COMMIT = 'ptx_tcgen05_commit_cta_group_2_multicast('


def calls(cuda, op):
    return [arguments(line) for line in cuda.splitlines() if line.strip().startswith(op)]


def slot(pointer, role, stage, consumer=0):
    expression = pointer.split('pool_buf_ptr)[', 1)[1].split(']))', 1)[0]
    expression = expression.replace(f'{role}_phase_stage_ptr[0]', str(stage))
    return eval(expression, {'__builtins__': {}}, {'warp_id_in_cta': 8 + consumer})


def simulate_input_ring(cuda, k_stages, seed):
    """Delay each CTA's B/A transfers and each consumer independently.

    Barrier addresses, expected transaction bytes, and free arrival counts
    come from emitted CUDA. Each transfer may complete before expect_tx is
    posted, and the input phase persists across output tile boundaries.
    """
    depth, total = 4, k_stages * 7
    rng = random.Random(seed)
    loads, expects = calls(cuda, COPY), calls(cuda, EXPECT)
    waits = [a for a in calls(cuda, WAIT) if 'mma_phase_stage' in a[0]]
    commits = [a for a in calls(cuda, COMMIT) if 'mma_phase_stage' in a[0]]
    frees = [a for a in calls(cuda, WAIT) if 'tma_phase_stage' in a[0]]
    inits = {int(re.search(r'pool_buf_ptr\)\[(\d+)\]', a[0])[1]): int(a[1])
             for a in calls(cuda, 'tvm_builtin_ptx_mbarrier_init(')}
    remote_bases = {name: int(base) for name, base in re.findall(
        r'uint64_t\* (remote_mbar_ptr(?:_\d+)?) = .*?pool_buf_ptr\)\[(\d+)\]', cuda)}

    def destination(call, stage):
        name, expr = re.search(r'\(&\((remote_mbar_ptr(?:_\d+)?)\[(.*?)\]\)\)', call[1]).groups()
        return remote_bases[name] + eval(expr.replace('tma_phase_stage_ptr[0]', str(stage)),
                                        {'__builtins__': {}}, {})

    # Each rank issues three input copies with identical storage/shape to
    # production. Their completion credits must match exactly one ready bar.
    sizes = [8192, 16384, 16384]
    produced, issued, finished = [0, 0], [0, 0], [0, 0]
    free_phase = {(rank, s): 0 for rank in range(2) for s in range(depth)}
    ready_phase, ready_generation = {}, {}
    pending, computing, transfers = [], [[], []], {}
    credits, expected, arrivals = {}, {}, {}
    overwritten = {}
    max_actions = total * 20
    for _ in range(max_actions):
        if finished == [total, total]:
            break
        actions = []
        for rank in range(2):
            q = produced[rank]
            if q < total and free_phase[rank, q % depth] != (1 ^ ((q // depth) & 1)):
                actions.append(('load', rank))
        actions += [('transfer', i) for i in range(len(pending))]
        for consumer in range(2):
            q = issued[consumer]
            if q < total and all(ready_phase.get(slot(w[0], 'mma', q % depth, consumer), 0)
                                 != ((q // depth) & 1) for w in waits):
                actions.append(('mma', consumer))
            if computing[consumer]:
                actions.append(('finish', consumer))
        assert actions, 'input-ring deadlock'
        action, who = rng.choice(actions)
        if action == 'load':
            q = produced[who]
            stage = q % depth
            prior = overwritten.get((who, stage))
            assert prior is None or all(n > prior for n in finished), 'input overwritten before both consumers finished'
            overwritten[who, stage] = q
            for operand, (call, count) in enumerate(zip(loads, sizes)):
                pending.append((q, who, operand, destination(call, stage), count))
            if who == 0:
                for call in expects:
                    bar = slot(call[0], 'tma', stage)
                    assert inits[bar] == 1
                    expected[q, bar] = int(call[1])
            produced[who] += 1
        elif action == 'transfer':
            q, rank, operand, bar, count = pending.pop(who)
            transfers[q, rank, operand] = True
            credits[q, bar] = credits.get((q, bar), 0) + count
        elif action == 'mma':
            q = issued[who]
            for operand in (0, who + 1):
                assert all(transfers.get((q, rank, operand)) for rank in range(2)), 'MMA used unfinished input'
            assert all(ready_generation[slot(w[0], 'mma', q % depth, who)] == q for w in waits)
            computing[who].append(q)
            issued[who] += 1
        else:
            q = computing[who].pop(0)
            assert q == finished[who]
            bar = slot(commits[0][0], 'mma', q % depth, who)
            assert bar == slot(frees[0][0], 'tma', q % depth)
            arrivals[q, bar] = arrivals.get((q, bar), 0) + 1
            if arrivals[q, bar] == inits[bar]:
                for rank in range(2):
                    free_phase[rank, q % depth] ^= 1
            finished[who] += 1
        for (q, bar), amount in expected.items():
            if ready_generation.get(bar, -1) < q and credits.get((q, bar), 0) == amount:
                ready_phase[bar] = ready_phase.get(bar, 0) ^ 1
                ready_generation[bar] = q
    assert produced == issued == finished == [total, total]
    assert not pending and not any(computing)


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(4096,) * 3, *READY_VERIFY_SHAPES])
def test_split_ready_preserves_work_and_waits_for_both_ctas(arch, shape, compile_probe):
    script, cuda, maps = compile_probe(shape, READY_VARIANT, arch)
    _, baseline, baseline_maps = compile_probe(shape, 'baseline', arch)
    assert descriptors(maps) == descriptors(baseline_maps)
    assert '"tirx.dyn_smem_bytes": T.int64(181248)' in script
    assert calls(cuda, COMMIT) == calls(baseline, COMMIT)
    # The extra wait follows the shared B acquire and precedes every MMA.
    waits = calls(cuda, WAIT)
    input_waits = [a for a in waits if 'mma_phase_stage' in a[0]]
    assert len(input_waits) == 2
    for stage in range(4):
        for consumer in range(2):
            assert [slot(w[0], 'mma', stage, consumer) for w in input_waits] == [1 + stage, 17 + stage * 2 + consumer]
            assert all(w[1] == '(mma_phase_phase_ptr[0] ^ 0)' for w in input_waits)
    expects = calls(cuda, EXPECT)
    assert [int(a[1]) for a in expects] == [16384, 32768, 32768]
    for stage in range(4):
        assert [slot(a[0], 'tma', stage) for a in expects] == [1 + stage, 17 + 2 * stage, 18 + 2 * stage]
        assert sum(int(a[1]) for a in expects) == 81920
    loads, old_loads = calls(cuda, COPY), calls(baseline, COPY)
    assert len(loads) == 3
    for current, old in zip(loads, old_loads):
        assert current[:1] + current[2:] == old[:1] + old[2:]
    assert loads[0] == old_loads[0]  # B still loaded once per rank.
    assert 'remote_mbar_ptr_1[(tma_phase_stage_ptr[0] * 2)]' in loads[1][1]
    assert 'remote_mbar_ptr_1[((tma_phase_stage_ptr[0] * 2) + 1)]' in loads[2][1]
    assert 'pool_buf_ptr)[17])), 0));' in cuda
    assert 'if (((int)tvm_builtin_cluster_ctaid_x()) == 0)' in cuda
    for bar in range(17, 25):
        init = f'tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[{bar}])), 1);'
        assert cuda.count(init) == 1
        assert cuda.index(init) < cuda.index('tvm_builtin_ptx_fence_mbarrier_init();')

    # All computation, accumulator ownership, writeback, scheduling and phase
    # advancement remain identical after removing the one extra ready wait.
    marker = '              tvm_builtin_ptx_mbarrier_try_wait('
    suffix = cuda[cuda.index(marker):]
    extra, = [line for line in suffix.splitlines(keepends=True)
              if WAIT in line and 'mma_phase_stage' in line and '+ 17)' in line]
    def normalize(text):
        names = {}
        return re.sub(r'\bactual_pred_ptr(?:_\d+)?\b', lambda m: names.setdefault(m[0], f'pred_{len(names)}'),
                      expand_cse(text))
    assert normalize(suffix.replace(extra, '')) == normalize(baseline[baseline.index(marker):])
    assert all(simulate_handoffs(cuda, 7, seed) for seed in range(3))
    for seed in range(8):
        simulate_input_ring(cuda, shape[2] // 64, seed)


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (8192,) * 3,
                                 (512, 9472, 192), (4096, 4352, 64), (4096, 4352, 320)])
def test_fallback_is_identical(arch, shape, compile_probe):
    _, before, old_maps = compile_probe(shape, 'baseline', arch)
    _, after, new_maps = compile_probe(shape, READY_VARIANT, arch)
    assert after == before and descriptors(new_maps) == descriptors(old_maps)


def test_simulator_detects_missing_acquire_and_early_free(compile_probe):
    _, cuda, _ = compile_probe((4096, 3072, 320), READY_VARIANT, 'sm_103a')
    lines = cuda.splitlines(keepends=True)
    bad_wait = ''.join(l for l in lines if not (WAIT in l and 'mma_phase_stage' in l and '+ 17)' in l))
    bad_free = re.sub(r'(pool_buf_ptr\)\[[5-8]\]\)\)), 2\);', r'\1, 1);', cuda)
    assert bad_free != cuda
    for broken in (bad_wait, bad_free, cuda.replace('32768, 0, actual_pred', '16384, 0, actual_pred')):
        with pytest.raises(AssertionError):
            for seed in range(20):
                simulate_input_ring(broken, 5, seed)


def test_transform_refuses_drift_and_default_stays_production():
    pytest.importorskip('tvm')
    import gemm_kernels
    source = inspect.getsource(gemm_kernels.hgemm_v10)
    changed = variant_builder_source(source, 10, READY_VARIANT)
    for invalid in (changed, source.replace('PIPE_DEPTH = 4', 'PIPE_DEPTH = 5'),
                    source.replace('mma2tma.init(NUM_CONSUMER)', 'mma2tma.init(1)'),
                    source.replace('NUM_CONSUMER = 2', 'NUM_CONSUMER = 1')):
        with pytest.raises(ValueError, match='adopted Step 10 narrow/TMEM baseline'):
            variant_builder_source(invalid, 10, READY_VARIANT)
    assert select_variants(10) == ['baseline']
    assert select_variants(10, [READY_VARIANT]) == ['baseline', READY_VARIANT]
    orders = [trial_order(2, i) for i in range(8)]
    assert Counter(tuple(o) for o in orders) == {(0, 1): 4, (1, 0): 4}
