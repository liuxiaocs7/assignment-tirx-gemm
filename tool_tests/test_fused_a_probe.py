"""Verify both consumers' TMA addresses, transaction bytes and unchanged MMA."""

import importlib
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import build_variant, select_variants, VERIFICATION_SHAPES, fuse_consumer_a_loads
from test_cache_followup_probe import expand_cse
from test_persistent_probe import body, pre_adoption_step10


@pytest.mark.parametrize('arch', ['sm_100a', 'sm_103a'])
@pytest.mark.parametrize('shape', [*VERIFICATION_SHAPES['balanced_fused_a'],
                                   (1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3])
def test_fused_a_descriptor_addresses_and_consumer_protocol(arch, shape, tmp_path, monkeypatch, pre_adoption_step10):
    tvm = pytest.importorskip('tvm')
    tma = importlib.import_module('tvm.backend.cuda.tile_primitive.copy_async.tma')
    emit = tma._emit_plan
    plans = []
    def record(plan, *args):
        plans.append(plan)
        return emit(plan, *args)
    monkeypatch.setattr(tma, '_emit_plan', record)
    target = tvm.target.Target({'kind': 'cuda', 'arch': arch})
    M, N, K = shape
    sources, recorded = {}, {}
    for variant in ('cache_balanced_clusters', 'balanced_fused_a'):
        kernel = build_variant(10, shape, variant, tmp_path / variant)
        assert '"tirx.dyn_smem_bytes": T.int64(230400)' in kernel.script()
        assert 'Asmem = T.decl_buffer((4, 2, 128, 64)' in kernel.script()
        plans.clear()
        with target:
            exe = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        sources[variant] = body(exe.mod.imports[0].inspect_source())
        recorded[variant] = list(plans)
    original, actual = sources.values()
    aplan = next(p for p in recorded['balanced_fused_a'] if p.spec.descriptor_name == 'A_blocks')
    spec = aplan.spec
    original_a = next(p.spec for p in recorded['cache_balanced_clusters'] if p.spec.descriptor_name == 'A')
    assert str(spec.smem_buffer.layout) == str(original_a.smem_buffer.layout)
    assert tuple(map(int, spec.global_dims)) == (K, 256, M // 256)
    assert tuple(map(int, spec.global_strides)) == (K * 2, 256 * K * 2)
    assert tuple(map(int, spec.box_dims)) == (64, 128, 2)
    assert tuple(map(int, spec.element_strides)) == (1, 1, 1)
    assert int(spec.inner_stride) == 1 and int(spec.base_byte_offset) == 0
    assert int(spec.swizzle) == 3  # Same 128-byte swizzle as the MMA shared layout.
    assert int(aplan.issue_extent()) == 1
    assert int(spec.payload_bits) == int(spec.transaction_bits) == 2 * 128 * 64 * 16
    dplan = next(p for p in recorded['balanced_fused_a'] if p.spec.descriptor_name == 'D')
    scheduler_rows = []
    tvm.tirx.stmt_functor.post_order_visit(
        dplan.spec.coordinates[1],
        lambda node: scheduler_rows.append(node) if isinstance(node, tvm.tirx.BufferLoad) else None)
    assert len(scheduler_rows) == 1
    scheduler_row = scheduler_rows[0]
    # Check the planner's actual coordinates, including its scheduler scalar
    # BufferLoad, rather than assuming that all indices are plain TIR Vars.
    coords = spec.coordinates
    analyzer = tvm.arith.Analyzer()
    stage_load = spec.smem_start[0]
    assert isinstance(stage_load, tvm.tirx.BufferLoad)
    assert str(stage_load) == 'tma_phase_stage'
    assert tuple(map(int, spec.smem_start[1:])) == (0, 0, 0)
    for stage in range(4):
        def resolve_stage(node):
            if isinstance(node, tvm.tirx.BufferLoad):
                assert node.buffer.same_as(stage_load.buffer)
                return tvm.tirx.IntImm('int32', stage)
        start = analyzer.simplify(tvm.tirx.stmt_functor.ir_transform(
            tvm.tirx.Evaluate(spec.smem_base_offset), resolve_stage, None).value)
        assert int(start) == 512 + stage * 16384
        for consumer in (0, 1):
            offset = spec.smem_buffer.offset_of([stage, consumer, 0, 0])
            assert tuple(map(int, offset)) == (int(start) + consumer * 8192,)
    for tile in (0, M // 512 - 1):
        for rank in (0, 1):
            for ktile in (0, K // 64 - 1):
                def resolve(node):
                    if isinstance(node, tvm.tirx.Var):
                        assert node.name in {'k', 'cbx'}
                        return tvm.tirx.IntImm('int32', ktile if node.name == 'k' else rank)
                    if isinstance(node, tvm.tirx.BufferLoad):
                        # The anonymous scalar must be the same scheduler M
                        # coordinate used for D, not its separate N coordinate.
                        assert node.buffer.same_as(scheduler_row.buffer)
                        assert tuple(map(int, node.indices)) == (0,)
                        assert tuple(map(int, node.buffer.shape)) == (1,)
                        assert node.buffer.scope() == 'local'
                        return tvm.tirx.IntImm('int32', tile)
                c = [int(analyzer.simplify(tvm.tirx.stmt_functor.ir_transform(
                    tvm.tirx.Evaluate(x), resolve, None).value)) for x in coords]
                assert c == [ktile * 64, rank * 128, tile * 2]
                for consumer in (0, 1):
                    for row in (0, 127):
                        for col in (0, 63):
                            byte = (c[0] + col) * 2 + (c[1] + row) * K * 2 + (c[2] + consumer) * 256 * K * 2
                            expected = ((tile * 512 + consumer * 256 + rank * 128 + row) * K + ktile * 64 + col) * 2
                            assert byte == expected and 0 <= byte < M * K * 2
    bplan = next(p for p in recorded['balanced_fused_a'] if p.spec.descriptor_name == 'B')
    expected_bytes = 2 * (int(spec.transaction_bits) + int(bplan.spec.transaction_bits)) // 8
    assert expected_bytes == 98304
    assert f'{expected_bytes}, 0, actual_pred_ptr[0]);' in actual
    assert actual.count('ptx_cp_async_bulk_tensor_g2s_cluster_tile_3d_mbar_addr(') == 1
    assert actual.count('ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(') == 1
    assert original.count('ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(') == 3
    # Every non-load hardware operation retains exact operands. Remove TVM's
    # CSE declarations by substituting their full expressions first.
    def calls(code):
        return [line.strip() for line in expand_cse(code).splitlines()
                if re.match(r'\s*(?:tvm_builtin_|ptx_)\w+\(', line)
                and 'ptx_cp_async_bulk_tensor_g2s_cluster_' not in line]
    assert calls(actual) == calls(original)
    assert actual.count('ptx_tcgen05_mma_cta_2_kind_f16_SS(') == 4
    assert actual.count('ptx_cp_async_bulk_tensor_shared_to_global_2d(') == 4
    assert actual.count('tvm_builtin_ptx_tcgen05_wait_ld();') == 8
    for name in ('mma_phase', 'tma_phase', 'ld_phase', 'wb_phase'):
        lines = lambda text: [line.strip() for line in text.splitlines()
                              if name in line and 'ptx_cp_async_bulk_tensor_g2s_' not in line]
        assert lines(actual) == lines(original)
    for name in ('B', 'D'):
        before = next(p.spec for p in recorded['cache_balanced_clusters'] if p.spec.descriptor_name == name)
        after = next(p.spec for p in recorded['balanced_fused_a'] if p.spec.descriptor_name == name)
        for field in ('global_dims', 'global_strides', 'box_dims', 'coordinates', 'smem_base_offset'):
            assert str(getattr(before, field)) == str(getattr(after, field))


def test_fused_a_requires_exact_producer_and_keeps_control_chain(tmp_path, pre_adoption_step10):
    pytest.importorskip('tvm')
    build_variant(10, (4096,) * 3, 'balanced_fused_a', tmp_path)
    changed = (tmp_path / 'builder.py').read_text()
    with pytest.raises(ValueError):
        fuse_consumer_a_loads(changed)
    expected = ['baseline', 'cache_tmem_base', 'cache_balanced_clusters', 'balanced_fused_a']
    assert select_variants(10, ['balanced_fused_a']) == expected
