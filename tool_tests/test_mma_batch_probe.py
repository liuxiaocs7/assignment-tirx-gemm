"""Check emitted PTX operands and surrounding protocol for MMA stage batching."""

import json
from pathlib import Path
import random
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import (VERIFICATION_SHAPES, batch_mma_stage, build_variant,
                              select_variants, summarize_with_cache_control,
                              variant_builder_source, variant_source)
from test_persistent_probe import body

RECORDED = ROOT / 'results_b300/step10_wide_tma.FBbwDr/step10/step10_4096_baseline/module_01.cu'
OP = 'ptx_tcgen05_mma_cta_2_kind_f16_SS'
BATCH = 'tvm_probe_mma_batch_k64'


def batch_helper(source):
    return re.search(rf'^__forceinline__ __device__ void {BATCH}\([^\n]+\) \{{\n.*?^\}}\n\n',
                     source, re.M | re.S)[0]


def interpret_batch(ptx, inputs):
    """Evaluate integer/predicate PTX, recording each asynchronous MMA issue."""
    registers = {f'%{i}': value for i, value in enumerate(inputs)}
    issues = []
    def val(operand):
        return registers[operand] if operand in registers else int(operand)
    for line in ptx.splitlines():
        if line in ('{', '}') or line.startswith('.reg '):
            continue
        m = re.fullmatch(r'mov.b64 \{(\w+), (\w+)\}, (\S+);', line)
        if m:
            value = val(m[3]); registers[m[1]] = value & 0xffffffff; registers[m[2]] = value >> 32
            continue
        m = re.fullmatch(r'mov.b64 (\w+), \{(\w+), (\w+)\};', line)
        if m:
            registers[m[1]] = val(m[2]) | val(m[3]) << 32
            continue
        m = re.fullmatch(r'mov.(b64|u32) (\w+), (\S+);', line)
        if m:
            registers[m[2]] = val(m[3]) & ((1 << (64 if m[1] == 'b64' else 32)) - 1)
            continue
        m = re.fullmatch(r'(add.u32|setp.ne.b32) (\w+), (\S+), (\S+);', line)
        if m:
            registers[m[2]] = ((val(m[3]) + val(m[4])) & 0xffffffff
                               if m[1] == 'add.u32' else val(m[3]) != val(m[4]))
            continue
        m = re.fullmatch(r'tcgen05.mma.cta_group::2.kind::f16 \[(\S+)\], (\S+), (\S+), (\S+), \{([^}]+)\}, (\S+);', line)
        if m:
            issues.append((*[val(m[i]) for i in range(1, 5)],
                           tuple(val(x.strip()) for x in m[5].split(',')), val(m[6])))
            continue
        raise AssertionError(f'unsupported instruction: {line}')
    return issues


def test_emitted_ptx_preserves_four_k16_addresses_masks_and_accumulation():
    helper = batch_helper(batch_mma_stage(RECORDED.read_text()))
    ptx = ''.join(json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"',
                                                  helper.split('\n        :', 1)[0]))
    rng = random.Random(0)
    # Distinct arbitrary high fields and low-word wrap cases catch unwanted
    # carry into swizzle/stride fields. Production offsets also fit these rules.
    bases = [rng.getrandbits(64) for _ in range(16)] + [0x40004040fffffffe]
    for base in bases:
        for stage in range(4):
            for consumer in range(2):
                for accum in (0, 1):
                    bbase = base ^ 0x1234000076
                    def desc(value, offset):
                        return (value & 0xffffffff00000000) | ((value + offset) & 0xffffffff)
                    aoff, boff = stage * 2048 + consumer * 1024, stage * 1024
                    inputs = [64 + consumer * 256, desc(base, aoff), desc(bbase, boff), 272629776, accum]
                    expected = [(inputs[0], desc(base, aoff + i * 2), desc(bbase, boff + i * 2),
                                 inputs[3], (0,) * 8, bool(accum) if i == 0 else True)
                                for i in range(4)]
                    assert interpret_batch(ptx, inputs) == expected


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3,
                                  *VERIFICATION_SHAPES['mma_batch'], (512, 256, 64)])
def test_codegen_batches_only_one_stage_without_moving_sync(arch, shape, tmp_path):
    tvm = pytest.importorskip('tvm')
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    kernel = build_variant(10, shape, 'mma_batch', tmp_path)
    assert (tmp_path / 'builder.py').read_bytes() == (tmp_path / 'builder.before.py').read_bytes()
    assert '"tirx.dyn_smem_bytes": T.int64(230400)' in kernel.script()
    with target:
        ex = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
    source = ex.mod.imports[0].inspect_source()
    if shape == (4096,) * 3:
        assert body(source) == body(RECORDED.read_text())
    changed = variant_source(source, 10, 'mma_batch')
    old_calls = re.findall(rf'^ +{OP}\([^\n]+\);$', body(source), re.M)
    new_calls = re.findall(rf'^ +{BATCH}\([^\n]+\);$', body(changed), re.M)
    assert len(old_calls) == 4 and len(new_calls) == 1
    # Round-trip equality verifies the full producer, consumer, epilogue,
    # scheduler, fences and phase updates are byte-for-byte preserved.
    restored = changed.replace(batch_helper(changed), '').replace(new_calls[0], '\n'.join(old_calls))
    assert restored == source
    no_unroll = variant_source(source, 10, 'mma_batch_no_unroll')
    if shape[2] == 64:
        assert no_unroll == changed
    else:
        assert no_unroll.count('#pragma unroll 1') == changed.count('#pragma unroll 1') + 1
        assert re.sub(r'^ +#pragma unroll 1\n(?= +for \(int k_1 =)', '', no_unroll, flags=re.M) == changed
    assert all(x in new_calls[0] for x in ('descA_ptr[0]', 'descB_ptr[0]', '(uint)272629776'))
    for variant in ('mma_batch', 'mma_batch_no_unroll'):
        with pytest.raises(ValueError, match='already applied'):
            variant_source(changed, 10, variant)


@pytest.mark.parametrize('before,after', [
    ('+ 6)), (uint)272629776', '+ 8)), (uint)272629776'),
    ('(uint)272629776', '(uint)272629777'),
    (', 0, 0, 0, 0, 0, 0, 0, 0);', ', 1, 0, 0, 0, 0, 0, 0, 0);'),
    ('(bool)1, 0, 0, 0, 0, 0, 0, 0, 0);', '(bool)0, 0, 0, 0, 0, 0, 0, 0, 0);'),
    ('setp.ne.b32 p, %4, 0;', 'setp.eq.b32 p, %4, 0;'),
    ('\n                ' + OP, '\n                tvm_builtin_ptx_tcgen05_fence_after_thread_sync();\n                ' + OP),
])
def test_mma_batch_rejects_changed_operands_or_intervening_work(before, after):
    source = RECORDED.read_text()
    assert before in source
    with pytest.raises(ValueError):
        batch_mma_stage(source.replace(before, after))


def test_mma_batch_controls_and_baseline_guard(tmp_path):
    pytest.importorskip('tvm')
    variants = ['baseline', 'mma_batch', 'mma_batch_no_unroll']
    assert select_variants(10) == select_variants(10, ['mma_batch_no_unroll']) == variants
    build_variant(10, (4096,) * 3, 'mma_batch', tmp_path)
    source = (tmp_path / 'builder.py').read_text()
    for variant in variants[1:]:
        assert variant_builder_source(source, 10, variant) == source
        with pytest.raises(ValueError, match='K64 baseline'):
            variant_builder_source(source.replace('    PIPE_DEPTH = 4\n', '    PIPE_DEPTH = 3\n'), 10, variant)
    cases = [dict(step=10, size=4096, variant=v, samples_ms=s) for v, s in
             zip(variants, [[10, 20], [8, 16], [4, 8]])]
    rows = summarize_with_cache_control(cases, {(10, 4096, 4096, 4096): 1}, 1.3)
    assert [r['comparison_control'] for r in rows] == ['baseline', 'baseline', 'mma_batch']
    assert [r['paired_control_speedup'] for r in rows] == [1, 1.25, 2]
    assert all(r['status'] == 'SLOW' for r in rows)
