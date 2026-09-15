"""Compare production/SW64 hardware counters in ABBA order, without scoring NCU time.

Both are already numerically verified experiments; this diagnoses the layout
regression, not a new candidate. --analyze reads saved evidence without a GPU.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from benchmark_diagnostics import write_json
import profile_hardware as hardware


ORDER = ("baseline", "tmem_k64_sw64", "tmem_k64_sw64", "baseline")
FOCUS = (
    "gpu__time_duration.avg", *hardware.LAYOUT_METRICS,
    "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct", "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic",
)
COMMON = (*hardware.SOURCE_KEYS, "profile_sources", "step", "shape", "seed", "warmup",
          "compiler", "ptxas_reg_level", "TVM_CUDA_PTXAS_EXTRA_OPTS", "TVM_CUDA_NVRTC_EXTRA_OPTS",
          "TVM_CUDA_NVCC_NO_FAST_MATH", "TVM_KERNEL_DEBUG", "TVM_KERNEL_DUMP",
          "CUDA_LAUNCH_BLOCKING", "CUDA_VISIBLE_DEVICES")
COLLECTION = ("replay_mode", "clock_control", "cache_control", "pipeline_boost_state",
              "ncu_version", "sections", "layout_metrics")
DEVICE = ("gpu_uuid", "gpu", "sm_count", "target", "tvm", "torch", "cuda", "cpu_affinity")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_files(directory):
    """A comparison must refer to the actual, single compiled kernel."""
    captured = json.loads((directory / "compiler/capture.json").read_text())
    if captured["modules"] != 1 or captured["compiler"] != "nvrtc":
        raise RuntimeError("layout comparison requires one NVRTC module per profile")
    names = ("module_01.cu", "module_01.cubin", "nvrtc_01.options.json", "nvrtc_version.json")
    paths = [directory / "compiler" / name for name in names]
    if not all(p.is_file() and p.stat().st_size for p in paths):
        raise RuntimeError(f"missing compiler evidence: {directory}")
    return {p.name: sha(p) for p in paths}


def read_profiles(source):
    profiles = []
    for index, variant in enumerate(ORDER):
        directory = source / f"{index + 1:02d}_{variant}"
        run = json.loads((directory / "run.json").read_text())
        worker = json.loads((directory / "worker.json").read_text())
        if run.get("status") != "collected" or worker.get("status") != "verified_after_profile":
            raise RuntimeError(f"incomplete collection or output verification: {directory}")
        if run.get("variant") != variant or worker.get("variant") != variant:
            raise RuntimeError(f"unexpected variant/order: {directory}")
        for key in COMMON:
            if key not in run or key not in worker or run[key] != worker[key]:
                raise RuntimeError(f"collection/worker mismatch: {key} in {directory}")
        if not worker.get("gpu_uuid"):
            raise RuntimeError(f"missing GPU UUID: {directory}")
        for key in DEVICE:
            if key not in worker:
                raise RuntimeError(f"missing device evidence: {key} in {directory}")
        for key in COLLECTION:
            if key not in run:
                raise RuntimeError(f"missing collection evidence: {key} in {directory}")
        if (run["compiler"] != "nvrtc" or run["step"] != 10 or run["shape"] != [4096] * 3
                or run["seed"] != 0 or run["warmup"] != 10):
            raise RuntimeError("expected NVRTC Step 10 / 4096 layout profiles")
        report = directory / run["report"]
        if not report.is_file() or not report.stat().st_size:
            raise RuntimeError(f"missing NCU report: {directory}")
        parsed = hardware.parse_raw_report((directory / "raw.csv").read_text())
        compiled = compiler_files(directory)
        if profiles:
            first = profiles[0]
            for key in (*COMMON, *COLLECTION):
                if run[key] != first["run"][key]:
                    raise RuntimeError(f"profiles differ in {key}; comparison refused")
            for key in DEVICE:
                if worker[key] != first["worker"][key]:
                    raise RuntimeError(f"profiles differ in {key}; comparison refused")
            for key in ("nvrtc_01.options.json", "nvrtc_version.json"):
                if compiled[key] != first["compiler"][key]:
                    raise RuntimeError(f"compiler settings differ in {key}; comparison refused")
            previous = next((p for p in profiles if p["variant"] == variant), None)
            if previous and compiled != previous["compiler"]:
                raise RuntimeError(f"repeated {variant} compiler artifacts differ; comparison refused")
        profiles.append(dict(variant=variant, directory=directory.name, run=run, worker=worker,
                             compiler=compiled, parsed=parsed,
                             raw_sha256=sha(directory / "raw.csv"), report_sha256=sha(report)))
    if profiles[0]["compiler"]["module_01.cu"] == profiles[1]["compiler"]["module_01.cu"]:
        raise RuntimeError("baseline and SW64 compiled CUDA are identical; wrong comparison")
    return profiles


def metric_row(name, profiles):
    metrics = [p["parsed"]["metrics"].get(name, {}) for p in profiles]
    values = [m.get("value", "unavailable") for m in metrics]
    units = [m.get("unit", "") for m in metrics]
    # Never divide unlike units or pretend unsupported metrics are zero.
    valid = all(hardware.numeric_value(v) for v in values) and len(set(units)) == 1
    ratios = [""] * 4
    if valid:
        nums = [float(v.replace(",", "")) for v in values]
        for i, (numerator, denominator) in enumerate(((1, 0), (2, 3), (3, 0), (2, 1))):
            if nums[denominator] > 0:
                ratios[i] = nums[numerator] / nums[denominator]
    return dict(metric=name, unit=units[0] if len(set(units)) == 1 else " / ".join(units),
                baseline_first=values[0], sw64_first=values[1], sw64_second=values[2],
                baseline_last=values[3], sw64_over_baseline_pair1=ratios[0],
                sw64_over_baseline_pair2=ratios[1], baseline_last_over_first=ratios[2],
                sw64_second_over_first=ratios[3], status="numeric" if valid else "unavailable_or_unit_mismatch")


def compare(source, output):
    profiles = read_profiles(source)
    names = sorted(set(FOCUS).union(*(p["parsed"]["metrics"] for p in profiles)))
    rows = {name: metric_row(name, profiles) for name in names}
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    lines = ["# Step 10 SW128 / SW64 硬件对照（诊断，不评分）", "",
             "顺序：正式版 → SW64 → SW64 → 正式版。每项保存四个原值；两组比值均为 SW64 / 正式版。",
             "大于 1 表示该计数器数值更大，不统一代表更慢或更快。比值分别使用相邻 AB、BA 配对。",
             "缺失计数器和单位不一致明确标出，不用 0 补齐。comparison.csv 另列同版本首尾比值供检查漂移。", "",
             "| 指标 | 单位 | 正式1 | SW64 1 | SW64 2 | 正式2 | 比值1 | 比值2 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in FOCUS:
        row = rows[name]
        ratios = [f"{row[k]:.4f}" if row[k] != "" else "—"
                  for k in ("sw64_over_baseline_pair1", "sw64_over_baseline_pair2")]
        line = " | ".join([name, row["unit"], row["baseline_first"], row["sw64_first"],
                           row["sw64_second"], row["baseline_last"], *ratios])
        lines.append("| " + line + " |")
    lines += ["", "判读顺序：", "",
              "1. 检查两次正式版及两次 SW64 的计数器和时钟是否一致，再看配对差异；四次 profile 不能建立统计置信区间。",
              "2. 固定数学工作量下，计算管线绝对活跃周期增加是取数/执行路径的线索；活跃百分比高不能直接证明计算已达峰值。",
              "3. 检查 TMA 请求、L2 sectors、DRAM bytes 与计算周期变化。请求数不是搬运延迟，不能单凭它定位等待。",
              "4. sm__mem_tensor 指 TMEM 活动；普通 LSU 指标也不能直接当作异步 MMA 的 SMEM 冲突证据。",
              "5. NCU replay 会改变执行与缓存状态；报告中的耗时不能替代原 CUDA-event 计时，不能用于 PASS/SLOW 或宣称提速。",
              "", "若这些计数器仍不能区分 TMA 和计算取数，应对两版本补同口径角色插桩或单独的取数微基准，不能硬归因。"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("Counters: metric | unit | baseline1 | SW64 1 | SW64 2 | baseline2 | SW64/base pair1 | pair2",
          flush=True)
    for line in lines[8:8 + len(FOCUS)]:
        print(line, flush=True)
    evidence = [{k: v for k, v in p.items() if k != "parsed"} for p in profiles]
    print(f"ABBA comparison saved: {output / 'comparison.md'}\n"
          "Counter ratios are diagnostic only. No performance or acceptance verdict.", flush=True)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh directory; default: unique results_b300 directory")
    parser.add_argument("--ncu", help="Nsight Compute executable")
    parser.add_argument("--analyze", type=Path, help="read an existing four-profile directory without GPU work")
    args = parser.parse_args(argv)
    if args.analyze and args.ncu:
        parser.error("--analyze cannot be combined with --ncu")
    if args.output:
        output = args.output.resolve()
        if output.exists():
            parser.error("--output must be a fresh directory")
        output.mkdir(parents=True)
    else:
        base = hardware.ROOT / "results_b300"
        base.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="step10_layout_profile.", dir=base))
    info = dict(status="preparing", diagnostic_only=True, order=list(ORDER),
                script_sha256=sha(Path(__file__)), source=str(args.analyze or output), collections=[])
    write_json(output / "comparison.json", info)
    print(f"结果目录：{output}", flush=True)
    try:
        if not args.analyze:
            for i, variant in enumerate(ORDER):
                directory = output / f"{i + 1:02d}_{variant}"
                print(f"Profile {i + 1}/4: {variant}", flush=True)
                code = hardware.collect(directory, args.ncu, variant, layout_counters=True)
                info["collections"].append(dict(directory=directory.name, variant=variant, exitcode=code))
                write_json(output / "comparison.json", info)
                if code:
                    raise RuntimeError(f"collection failed: {directory}; all partial evidence retained")
        info["profiles"] = compare((args.analyze or output).resolve(), output)
        info["status"] = "compared"
        code = 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        info.update(status="failed", error=str(error))
        print(f"Layout profile failed: {error}", file=sys.stderr)
        code = 1
    write_json(output / "comparison.json", info)
    print(f"结果目录：{output}；退出码：{code}（只表示诊断采集/核对是否完整）", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
