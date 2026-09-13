"""Replay recorded compiler inputs and check the new persistent experiments.

These checks cover instrumentation and TVM source generation, not GPU timing.
"""

import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent import DEFAULT_STEP_VARIANTS, STEP_VARIANTS, build_variant, main, variant_builder_source, variant_source
from probe_step45 import source_experiment

RECORDED_PROBE = ROOT / "results_b300/persistent_probe.VB42kg/probe"
RECORDED_VARIANTS = json.loads((RECORDED_PROBE / "run.json").read_text())["variants"]

def body(source):
    return source.split('extern "C" __global__', 1)[1]


def canonicalize_codegen_locals(code):
    """Ignore CSE/cast-loop numbering, retaining variable identity and expressions."""
    names = {}
    return re.sub(r"\b(cse_v|f_)\d+\b", lambda m: names.setdefault(
        m[0], f"{m[1]}{len(names)}"), code)


def recorded_source(step, size=4096):
    return (ROOT / "results_b300/b300_diag.RA0gpm/compiler" /
            f"step{step:02}_{size}_{size}_{size}/module_01.cu").read_text()


def generate(kernel):
    tvm = pytest.importorskip("tvm")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_103a"})
    with target:
        executable = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    return executable.mod.imports[0].inspect_source()


def install_recorded_builder(step, monkeypatch):
    """Replay pre-adoption experiments using the exact measured baseline."""
    import gemm_kernels

    path = RECORDED_PROBE / f"step{step:02}_4096_baseline/builder.py"
    if step == 8:
        path = ROOT / "results_b300/step810_probe.Wl6HTg/step08/step08_2048_baseline/builder.py"
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    name = f"hgemm_v{step}"
    monkeypatch.setattr(gemm_kernels, name, namespace[name])


@pytest.fixture
def pre_adoption_step10(monkeypatch):
    """Historical probes must keep their recorded baseline after adoption."""
    pytest.importorskip("tvm")
    install_recorded_builder(10, monkeypatch)


@pytest.mark.parametrize("step", [6, 7, 10])
@pytest.mark.parametrize("size", [1024, 2048, 4096, 8192])
def test_recorded_mma_wait_changes_only_selected_calls(step, size):
    source = recorded_source(step, size)
    changed = variant_source(source, step, "mma_wait_64ns")
    old_name, new_name = "tvm_builtin_ptx_mbarrier_try_wait", "tvm_probe_mma_wait_64ns"
    assert body(changed).replace(new_name, old_name) == body(source)
    assert body(changed).count(new_name + "(") == (1 if step == 6 else 2)
    assert body(changed).count(old_name + "(") == (1 if step == 6 else 2)
    def helper(code, name):
        return re.search(rf"^__forceinline__ __device__ void {name}\([^\n]+\) \{{\n.*?^\}}\n",
                         code, re.M | re.S).group()
    assert helper(changed, old_name) == helper(source, old_name)
    restored = helper(changed, new_name).replace(new_name, old_name).replace(
        "unsigned int ticks = 64;", "unsigned int ticks = 0x989680;")
    assert restored == helper(source, old_name)
    with pytest.raises(ValueError, match="already applied"):
        variant_source(changed, step, "mma_wait_64ns")


@pytest.mark.parametrize("step", [6, 7, 10])
def test_changed_wait_layout_fails_closed(step):
    source = recorded_source(step).replace("pool_buf_ptr", "moved_pool_buf_ptr")
    with pytest.raises(ValueError):
        variant_source(source, step, "mma_wait_64ns")


@pytest.mark.parametrize("step,variant", [(int(s), v) for s, vv in RECORDED_VARIANTS.items() for v in vv])
def test_experiment_lowers_and_preserves_protocol(step, variant, tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels

    install_recorded_builder(step, monkeypatch)
    original = getattr(gemm_kernels, f"hgemm_v{step}")
    directory = tmp_path / variant
    source = generate(build_variant(step, (4096,) * 3, variant, directory))
    source = variant_source(source, step, variant)
    baseline = body(recorded_source(step))
    actual = body(source)
    assert getattr(gemm_kernels, f"hgemm_v{step}") is original
    info = json.loads((directory / "builder.json").read_text())
    assert info["before_sha256"] == hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    assert info["compiled_sha256"] == hashlib.sha256((directory / "builder.py").read_bytes()).hexdigest()

    # Every variant retains allocation lifetime, persistence, and MMA commits.
    for name in ("tcgen05_alloc_cta_group_", "tcgen05_dealloc_cta_group_",
                 "tcgen05_relinquish_alloc_permit_cta_group_", "ptx_tcgen05_commit_"):
        assert actual.count(name) == baseline.count(name)
    assert actual.count("while (") == baseline.count("while (")
    if variant == "baseline":
        assert actual == baseline
    elif variant == "mma_wait_64ns":
        assert actual.replace("tvm_probe_mma_wait_64ns", "tvm_builtin_ptx_mbarrier_try_wait") == baseline
    elif variant == "k_tile_128":
        assert actual.count("k < 32") == 1 and "k < 64" in baseline
        if step == 7:
            assert "k_1 < 32" in actual and "k_1 < 64" in baseline
        assert actual.count("ptx_tcgen05_mma_cta_1_kind_f16_SS(") == 8
        assert baseline.count("ptx_tcgen05_mma_cta_1_kind_f16_SS(") == 4
        assert "65536);" in actual and "32768);" in baseline
        assert actual.count("tvm_builtin_ptx_mbarrier_try_wait(") == baseline.count("tvm_builtin_ptx_mbarrier_try_wait(")
    elif variant == "final_fence":
        before = ("          tvm_builtin_ptx_tcgen05_fence_after_thread_sync();\n"
                  "          tvm_builtin_ptx_tcgen05_fence_before_thread_sync();\n")
        after = before.replace("          ", "        ")
        assert baseline.count(before) == actual.count(after) == 1
        assert actual.replace(after, "") == baseline.replace(before, "")
        # The handoff fence stays before the writeback CTA barrier.
        assert after + "      }\n    }\n    tvm_builtin_cuda_cta_sync();" in actual
    elif variant == "epilogue_128":
        assert baseline.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 2
        assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 1
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 4
        # Read-TMEM completion still signals the free accumulator before TMA.
        assert actual.index("tvm_builtin_ptx_tcgen05_wait_ld();") < actual.index(
            "tvm_builtin_ptx_mbarrier_arrive_shared(") < actual.index("ptx_cp_async_bulk_tensor_shared_to_global_")
    else:
        # Step 10 trades more loads for a smaller FP32 temporary, independently.
        assert "float Dreg_ptr[16]" in actual and "half Dreg_f16_ptr[256]" in actual
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 16
        assert baseline.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 8
        assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 4
        marker = "    alignas(64) float Dreg_ptr["
        # More epilogue loads renumber TVM's CSE temporaries, including ones
        # in the producer. Preserve distinct identifiers and all expressions.
        assert canonicalize_codegen_locals(actual.split(marker)[0]) == canonicalize_codegen_locals(baseline.split(marker)[0])


@pytest.mark.parametrize("step", [6, 7])
@pytest.mark.parametrize("K", [64, 192, 384])
def test_k_tile_variant_retains_short_and_odd_phase_support(step, K, tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels

    install_recorded_builder(step, monkeypatch)
    shape = (1024, 3072, K)  # 192 output tiles force persistent CTA reuse.
    actual = generate(build_variant(step, shape, "k_tile_128", tmp_path))
    if K % 128:
        assert body(actual) == body(generate(getattr(gemm_kernels, f"hgemm_v{step}")(*shape)))
    else:
        assert "65536);" in body(actual)
        assert "k < 3" in body(actual)


@pytest.mark.parametrize("step", [6, 7])
def test_adopted_k_tile_is_not_applied_again(step, tmp_path):
    pytest.importorskip("tvm")
    assert DEFAULT_STEP_VARIANTS[step] == ("baseline",)
    with pytest.raises(ValueError, match=f"Step {step} has adopted k_tile_128"):
        build_variant(step, (4096,) * 3, "k_tile_128", tmp_path)


@pytest.mark.parametrize("variant", ["tma_wait_64ns", "cache_tmem_base"])
def test_adopted_step8_experiment_is_not_applied_again(variant, tmp_path):
    pytest.importorskip("tvm")
    assert variant not in DEFAULT_STEP_VARIANTS[8]
    with pytest.raises(ValueError, match=f"Step 8 has adopted {variant}"):
        build_variant(8, (2048,) * 3, variant, tmp_path)


@pytest.mark.parametrize("variant", ["tmem_load_64", "l2_group_4", "balanced_clusters"])
def test_step10_new_experiments_preserve_data_and_barrier_protocol(variant, tmp_path, pre_adoption_step10):
    pytest.importorskip("tvm")
    import gemm_kernels

    baseline = body(recorded_source(10))
    kernel = build_variant(10, (4096,) * 3, variant, tmp_path)
    actual = body(generate(kernel))
    if variant == "tmem_load_64":
        assert "float Dreg_ptr[64]" in actual and "half Dreg_f16_ptr[256]" in actual
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 4
        assert actual.count("tvm_builtin_ptx_tcgen05_ld_32x32b_x64(") == 4
        marker = "    alignas(64) float Dreg_ptr["
        assert canonicalize_codegen_locals(actual.split(marker)[0]) == canonicalize_codegen_locals(baseline.split(marker)[0])
        after_reads = "tvm_builtin_ptx_tcgen05_fence_before_thread_sync();"
        assert canonicalize_codegen_locals(actual.split(after_reads)[1]) == canonicalize_codegen_locals(baseline.split(after_reads)[1])
    elif variant == "balanced_clusters":
        # All three roles advance by the same stride; no data/barrier change.
        old, new = "_ptr_2[0] = (_ptr_2[0] + 74);", "_ptr_2[0] = (_ptr_2[0] + 64);"
        assert actual.count(new) == baseline.count(old) == 3
        assert actual.replace(new, old) == baseline
    else:
        # Tile coordinates change, but every async operation and fence remains.
        calls = lambda s: [line.strip() for line in s.splitlines()
                           if re.match(r"\s*(tvm_builtin_|ptx_)\w+\(", line)]
        assert calls(actual) == calls(baseline)
        assert "(group_id_1 * 4) + (within_group_1 & 3)" in actual
        assert "(group_id_1 * 8) + (within_group_1 & 7)" in baseline
    assert getattr(gemm_kernels, "hgemm_v10").__module__ == "gemm_kernels"


@pytest.mark.parametrize("M,N,clusters", [(1024, 1024, 8), (2048, 2048, 32),
                                         (4096, 4096, 64), (8192, 8192, 74),
                                         (4096, 3072, 48)])
def test_balanced_grid_launch_matches_persistent_stride(M, N, clusters, tmp_path, monkeypatch, pre_adoption_step10):
    pytest.importorskip("tvm")
    import gemm_kernels

    monkeypatch.setattr(gemm_kernels, "SM_COUNT", 148)
    kernel = build_variant(10, (M, N, 320), "balanced_clusters", tmp_path)
    # A two-CTA launch per cluster and identical TMA/MMA/writeback strides
    # must cover all tiles, including a rectangular grid and partial K ring.
    assert f"T.cta_id([{clusters * 2}])" in kernel.script()
    source = body(generate(kernel))
    assert source.count(f"_ptr_2[0] = (_ptr_2[0] + {clusters});") == 3


@pytest.mark.parametrize("step", [8, 10])
@pytest.mark.parametrize("size", [1024, 2048, 4096, 8192])
def test_tma_wait_only_changes_data_ready_barrier(step, size):
    original = recorded_source(step, size)
    changed = variant_source(original, step, "tma_wait_64ns")
    old, new = "tvm_builtin_ptx_mbarrier_try_wait", "tvm_probe_tma_wait_64ns"
    assert body(changed).replace(new, old) == body(original)
    expected = (f"{new}((&(((uint64_t*)pool_buf_ptr)[(mma_phase_stage_ptr[0] + 1)])), "
                "(mma_phase_phase_ptr[0] ^ 0));")
    assert body(changed).count(expected) == body(changed).count(new + "(") == 1
    assert body(changed).count(old + "(") == 3
    pattern = r"^__forceinline__ __device__ void {}\([^\n]+\) \{{\n.*?^\}}\n"
    old_helper = re.search(pattern.format(old), original, re.M | re.S).group()
    new_helper = re.search(pattern.format(new), changed, re.M | re.S).group()
    assert new_helper.replace(new, old).replace("ticks = 64;", "ticks = 0x989680;") == old_helper
    with pytest.raises(ValueError, match="already applied"):
        variant_source(changed, step, "tma_wait_64ns")
    with pytest.raises(ValueError):
        variant_source(original.replace("mma_phase_stage_ptr", "moved_stage_ptr"), step, "tma_wait_64ns")


@pytest.mark.parametrize("step,size", [(8, 2048), (10, 4096)])
def test_recorded_tma_experiment_lowers_to_recorded_protocol(step, size, tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    install_recorded_builder(step, monkeypatch)
    actual = generate(build_variant(step, (size,) * 3, "tma_wait_64ns", tmp_path))
    assert body(actual) == body(recorded_source(step, size))
    assert body(variant_source(actual, step, "tma_wait_64ns")) == body(
        variant_source(recorded_source(step, size), step, "tma_wait_64ns"))


def test_step8_epilogue_keeps_four_stages_and_releases_tmem_after_reads(tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    install_recorded_builder(8, monkeypatch)
    kernel = build_variant(8, (2048,) * 3, "epilogue_128", tmp_path)
    actual, baseline = body(generate(kernel)), body(recorded_source(8, 2048))
    assert "Asmem = T.decl_buffer((4, 128, 64)" in kernel.script()
    assert "Bsmem = T.decl_buffer((4, 128, 64)" in kernel.script()
    assert "Dsmem = T.decl_buffer((128, 128)" in kernel.script()
    assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 1
    assert baseline.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 2
    assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 4
    assert actual.rindex("tvm_builtin_ptx_tcgen05_wait_ld();") < actual.index(
        "tvm_builtin_ptx_mbarrier_arrive_shared(") < actual.index("ptx_cp_async_bulk_tensor_shared_to_global_")
    # The two producer roles' barrier addresses, offsets, and MMA calls stay.
    waits = lambda s: [line.strip() for line in s.splitlines()
                       if "tvm_builtin_ptx_mbarrier_try_wait(" in line or "ptx_tcgen05_mma_cta_1_kind_f16_SS(" in line]
    assert waits(actual) == waits(baseline)


@pytest.mark.parametrize("variant", ["specialize_mma", "specialize_writeback"])
@pytest.mark.parametrize("shape", [(4096, 4096, 4096), (4096, 3072, 64), (4096, 3072, 320)])
def test_static_consumer_roles_keep_barrier_slots_and_cluster_protocol(variant, shape, tmp_path):
    pytest.importorskip("tvm")
    kernel = build_variant(10, shape, variant, tmp_path)
    actual = body(generate(kernel))
    # Both consumers use CTA-group 2; fixed slot indices do not collapse them.
    assert actual.count("tvm_builtin_ptx_tcgen05_alloc_cta_group_2(") == 1
    assert actual.count("tvm_builtin_ptx_tcgen05_dealloc_cta_group_2(") == 1
    assert actual.count("while (") == 4
    assert "Asmem = T.decl_buffer((4, 2, 128, 64)" in kernel.script()
    assert "Bsmem = T.decl_buffer((4, 128, 64)" in kernel.script()
    if variant == "specialize_mma":
        assert actual.count("ptx_tcgen05_mma_cta_2_kind_f16_SS(") == 8
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 8
        assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 4
        for slot in (9, 10):
            assert f"ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[{slot}])), 3);" in actual
        for slot in (11, 12):
            assert f"tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[{slot}])), (ld_phase_phase_ptr[0] ^ 0));" in actual
        assert actual.count("if (((int)tvm_builtin_cluster_ctaid_x()) == 0)") == 3
        assert "((warp_id_in_cta & 3) * 256)" not in actual
    else:
        assert actual.count("ptx_tcgen05_mma_cta_2_kind_f16_SS(") == 4
        assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 16
        assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_") == 8
        for consumer in range(2):
            assert f"tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[{9 + consumer}])), (wb_phase_phase_ptr[0] ^ 0));" in actual
            assert f"tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred((&(((uint64_t*)pool_buf_ptr)[{11 + consumer}])), 0," in actual
            assert actual.count(f"tvm_builtin_cuda_warpgroup_sync({10 + consumer});") == 8
        assert "((warp_id_in_cta >> 2) * 256)" not in actual
    # Distinct branch sentinels reject reapplying specialization.
    changed_builder = (tmp_path / "builder.py").read_text()
    with pytest.raises(ValueError):
        variant_builder_source(changed_builder, 10, variant)


@pytest.mark.parametrize("K", [64, 320])
def test_unrolled_ring_preserves_partial_ring_cuda(K, tmp_path):
    pytest.importorskip("tvm")
    import gemm_kernels

    shape = (4096, 3072, K)
    actual = body(generate(build_variant(10, shape, "unroll_ring", tmp_path)))
    assert actual == body(generate(gemm_kernels.hgemm_v10(*shape)))


@pytest.mark.parametrize("K", [256, 768, 4096])
def test_unrolled_ring_keeps_slot_order_and_phase_across_tiles(K, tmp_path):
    pytest.importorskip("tvm")
    kernel = build_variant(10, (4096, 3072, K), "unroll_ring", tmp_path)
    actual = body(generate(kernel))
    # TVM keeps unused declarations/initializers, but no dynamic stage use
    # may remain inside a persistent role's loop.
    assert "mma_phase_stage_ptr" not in actual.split("while (1)", 1)[1]
    assert "tma_phase_stage_ptr" not in actual.split("while (1)", 1)[1]
    for role in ("tma", "mma"):
        # Initialize once before all tile loops, then flip only after a ring.
        init = f"{role}_phase_phase_ptr[0] = {1 if role == 'tma' else 0};"
        flip = f"{role}_phase_phase_ptr[0] = ({role}_phase_phase_ptr[0] ^ 1);"
        assert actual.count(init) == actual.count(flip) == 1
        assert actual.index(init) < actual.index("while (1)")
    if K > 256:
        assert len(re.findall(rf"for \(int ring(?:_\d+)? = 0; ring(?:_\d+)? < {K // 256};", actual)) == 2
    assert actual.count("ptx_cp_async_bulk_tensor_g2s_cluster_tile_2d_mbar_addr(") == 12
    assert actual.count("ptx_tcgen05_mma_cta_2_kind_f16_SS(") == 16
    assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 8
    assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_2d(") == 4
    for stage in range(4):
        for slot, role in ((stage + 5, "tma"), (stage + 1, "mma")):
            call = (f"tvm_builtin_ptx_mbarrier_try_wait((&(((uint64_t*)pool_buf_ptr)[{slot}])), "
                    f"({role}_phase_phase_ptr[0] ^ 0));")
            assert actual.count(call) == 1
        commit = f"ptx_tcgen05_commit_cta_group_2_multicast((&(((uint64_t*)pool_buf_ptr)[{stage + 5}])), 3);"
        assert actual.count(commit) == 1
    assert actual.index("pool_buf_ptr)[8])), 3);") < actual.index("mma_phase_phase_ptr[0] = (mma_phase_phase_ptr[0] ^ 1);")


@pytest.mark.parametrize("variant", ["pipe_depth_2", "k128_depth_2"])
@pytest.mark.parametrize("K", [64, 128, 320, 384, 4096])
def test_two_stage_probe_preserves_cluster_work_and_transaction_bytes(variant, K, tmp_path):
    pytest.importorskip("tvm")
    kernel = build_variant(10, (4096, 3072, K), variant, tmp_path)
    actual = body(generate(kernel))
    width = 128 if variant == "k128_depth_2" and K % 128 == 0 else 64
    assert f"Asmem = T.decl_buffer((2, 2, 128, {width})" in kernel.script()
    assert f"Bsmem = T.decl_buffer((2, 128, {width})" in kernel.script()
    assert "Dsmem = T.decl_buffer((2, 128, 64)" in kernel.script()
    assert actual.count("ptx_tcgen05_mma_cta_2_kind_f16_SS(") == width // 16
    # A 128-wide K tile is represented as two 128-byte swizzle atoms using
    # a three-dimensional TMA box. It still issues one copy per A/B buffer.
    dimensions = 3 if width == 128 else 2
    assert actual.count(f"ptx_cp_async_bulk_tensor_g2s_cluster_tile_{dimensions}d_mbar_addr(") == 3
    assert f"{2 * 3 * 128 * width * 2}, 0, actual_pred_ptr[0]);" in actual
    assert "(tma_phase_stage_ptr[0] + 3)" in actual
    assert "(mma_phase_stage_ptr[0] + 1)" in actual
    assert "((warp_id_in_cta >> 2) + 7)" in actual
    assert actual.count("tvm_builtin_ptx_tcgen05_wait_ld();") == 8
    assert actual.count("ptx_cp_async_bulk_tensor_shared_to_global_2d(") == 4
    # Both consumers must commit before the producer may overwrite a stage.
    assert "tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[3])), 2);" in actual
    assert "tvm_builtin_ptx_mbarrier_init((&(((uint64_t*)pool_buf_ptr)[4])), 2);" in actual
    assert actual.count("tvm_builtin_ptx_tcgen05_alloc_cta_group_2(") == 1
    assert actual.count("tvm_builtin_ptx_tcgen05_dealloc_cta_group_2(") == 1


@pytest.mark.parametrize("K", [64, 320, 4096])
def test_stream_epilogue_keeps_producers_and_releases_after_all_tmem_reads(K, tmp_path):
    pytest.importorskip("tvm")
    import gemm_kernels

    shape = (4096, 3072, K)
    actual = body(generate(build_variant(10, shape, "stream_epilogue", tmp_path)))
    baseline = body(generate(gemm_kernels.hgemm_v10(*shape)))
    marker = "    alignas(64) float Dreg_ptr["
    assert canonicalize_codegen_locals(actual.split(marker)[0]) == canonicalize_codegen_locals(baseline.split(marker)[0])
    assert "half Dreg_f16_ptr[64]" in actual
    # Exactly two TMEM loads precede each 64-column TMA store. Only the final
    # chunk signals reusable TMEM, after all eight loads have completed.
    events = []
    for line in actual.splitlines():
        if "tvm_builtin_ptx_tcgen05_wait_ld();" in line:
            events.append("read_done")
        elif "tvm_builtin_ptx_mbarrier_arrive_shared_cluster_remote_pred(" in line:
            events.append("release")
        elif "ptx_cp_async_bulk_tensor_shared_to_global_2d(" in line:
            events.append("store")
        elif "ptx_cp_async_bulk_wait_group_read_0();" in line:
            events.append("store_read_done")
    assert events == (["read_done", "read_done", "store", "store_read_done"] * 3 +
                      ["read_done", "read_done", "release", "store", "store_read_done"])
    assert actual.count("tvm_builtin_cuda_warpgroup_sync(((warp_id_in_cta >> 2) + 10));") == 8


@pytest.mark.parametrize("fail", [False, True])
def test_custom_transform_callback_is_captured_and_restored(tmp_path, fail):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    pytest.importorskip("tvm.support.nvcc")
    key = "tvm_callback_cuda_compile"
    original = tvm_ffi.get_global_func(key)
    source = recorded_source(7)
    seen = []
    def compile_cuda(code):
        seen.append(str(code))
        if fail:
            raise RuntimeError("fixture compiler failure")
        return bytearray(b"ELF fixture")
    tvm_ffi.register_global_func(key, compile_cuda, override=True)
    try:
        if fail:
            with pytest.raises(Exception, match="fixture compiler failure"):
                with source_experiment(7, "mma_wait_64ns", tmp_path, transform=variant_source):
                    tvm_ffi.get_global_func(key)(source)
        else:
            with source_experiment(7, "mma_wait_64ns", tmp_path, transform=variant_source):
                tvm_ffi.get_global_func(key)(source)
        assert seen == [variant_source(source, 7, "mma_wait_64ns")]
        assert (tmp_path / "source_01.before.cu").read_text() == source
        assert json.loads((tmp_path / "experiment.json").read_text())["sources"][0]["compiled_sha256"] == hashlib.sha256(seen[0].encode()).hexdigest()
        if fail:
            with pytest.raises(Exception):
                tvm_ffi.get_global_func(key)(source)
        else:
            tvm_ffi.get_global_func(key)(source)
        assert seen[-1] == source
    finally:
        tvm_ffi.register_global_func(key, original, override=True)


@pytest.mark.parametrize("argv", [["--trials", "0"], ["--repeat", "0"], ["--warmup", "-1"],
                                  ["--size", "123"], ["--steps", "4"],
                                  ["--steps", "10", "--variants", "k_tile_128"]])
def test_invalid_arguments_fail_before_gpu_imports(argv, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--output", str(tmp_path / "new"), *argv])
    assert error.value.code == 2
