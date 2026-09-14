"""Recompute the beginner guide's tables from archived B300 measurements.

No GPU execution, source transformation or benchmark configuration changes.
Usage: python tools/build_optimization_data.py
Plots: uv run --isolated --no-project --with matplotlib python tools/build_optimization_data.py --plots
"""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/optimization_data"
REVISION = "9e3b989cc50a8143718048351bcd79a1e1343d47"
KERNEL_SHA = "727a44403b6c1ea0a0499c8212e5796d0fa2b31944929e9ecd50d60be4b961f2"

# Id, archive directory, step, direct control, candidate, current decision.
# Direct controls are explicit even in old CSVs that did not record the chain.
EXPERIMENTS = [
    ("s4_permit", "step45_probe.PS9CFi/probe", 4, "baseline", "early_release", "adopted"),
    ("s4_k128", "mma64_step4.dh9ZC8/probe", 4, "baseline", "k_tile_128", "adopted"),
    ("s5_permit", "step45_probe.PS9CFi/probe", 5, "baseline", "early_release", "adopted"),
    ("s5_mma_wait", "wait1024.UP24Tv/probe", 5, "baseline", "mma_wait_64ns", "adopted"),
    ("s5_poll", "wait1024.UP24Tv/probe", 5, "baseline", "wait_poll", "not_adopted"),
    ("s6_k128", "persistent_probe.VB42kg/probe", 6, "baseline", "k_tile_128", "adopted"),
    ("s6_fence", "persistent_probe.VB42kg/probe", 6, "baseline", "final_fence", "not_adopted"),
    ("s7_k128", "persistent_probe.VB42kg/probe", 7, "baseline", "k_tile_128", "adopted"),
    ("s7_epi128", "persistent_probe.VB42kg/probe", 7, "baseline", "epilogue_128", "not_adopted"),
    ("s8_tma_wait", "step810_probe.Wl6HTg/step08", 8, "baseline", "tma_wait_64ns", "adopted"),
    ("s8_cache", "profile_guided.6KUfDZ/step08", 8, "baseline", "cache_tmem_base", "adopted"),
    ("s9_cache4096", "step9_cache.EWrE9G/step9_4096", 9, "baseline", "cluster_cache_tmem_base", "adopted"),
    ("s9_cache8192", "step9_cache.BBJEQB/step9_8192", 9, "baseline", "cluster_cache_tmem_base", "adopted"),
    ("s10_cache", "profile_guided.6KUfDZ/step10", 10, "baseline", "cache_tmem_base", "adopted"),
    ("s10_balance", "step10_fused_a.1VXYz2/step10", 10, "cache_tmem_base", "cache_balanced_clusters", "adopted"),
    ("s10_fused_a", "step10_fused_a.1VXYz2/step10", 10, "cache_balanced_clusters", "balanced_fused_a", "not_adopted"),
    ("s10_b_first", "step10_roles.rx5lNL/step10", 10, "baseline", "tma_b_first", "adopted"),
    ("s10_n128_1024", "step10_tmem_sizes.I9nGIJ/step10_1024", 10, "baseline", "n128_epi32", "adopted_for_shape"),
    ("s10_n128_2048", "step10_tmem_sizes.I9nGIJ/step10_2048", 10, "baseline", "n128_epi32", "adopted_for_shape"),
    ("s10_n128_only4096", "step10_tmem_sizes.I9nGIJ/step10_4096", 10, "baseline", "n_tile_128", "not_adopted_alone"),
    ("s10_epi32_4096", "step10_tmem_sizes.I9nGIJ/step10_4096", 10, "n_tile_128", "n128_epi32", "adopted_in_combination"),
    ("s10_double4096", "step10_tmem_sizes.I9nGIJ/step10_4096", 10, "n128_epi32", "n128_tmem_double_buffer", "adopted_for_shape"),
    ("s10_combined4096", "step10_tmem_sizes.I9nGIJ/step10_4096", 10, "baseline", "n128_tmem_double_buffer", "adopted_for_shape"),
    ("s10_combined8192", "step10_tmem_sizes.I9nGIJ/step10_8192", 10, "baseline", "n128_tmem_double_buffer", "not_adopted_for_shape"),
    ("s10_k64_depth2", "step10_input_ring.9SsKZM/step10", 10, "baseline", "tmem_input_depth2", "not_adopted"),
    ("s10_k128_depth2", "step10_input_ring.9SsKZM/step10", 10, "baseline", "tmem_k128_depth2", "not_adopted"),
    ("s10_depth5", "step10_depth5_balanced.Fm3CcJ/step10_4096", 10, "baseline", "tmem_input_depth5", "candidate_not_adopted"),
    *[(f"s10_l2_{g}", "step10_l2.nqNoLW/step10_4096", 10, "baseline", f"tmem_l2_group{g}", "not_adopted") for g in (4, 2, 1)],
    ("s10_maxclusters", "step10_geometry.PywrmG/step10_4096", 10, "baseline", "tmem_max_clusters", "not_adopted"),
    ("s10_n64", "step10_geometry.PywrmG/step10_4096", 10, "tmem_max_clusters", "tmem_n64", "not_adopted"),
    ("s10_n64depth5", "step10_geometry.PywrmG/step10_4096", 10, "tmem_n64", "tmem_n64_depth5", "not_adopted"),
    ("s10_n64depth5_total", "step10_geometry.PywrmG/step10_4096", 10, "baseline", "tmem_n64_depth5", "not_adopted"),
    ("s10_batch_fixed", "step10_mma_unroll4.iR07fS/step10", 10, "mma_unroll4", "mma_batch_unroll4", "not_adopted"),
    ("s10_shared_a", "step10_share_a_state.fGOjDy/step10", 10, "tmem_input_depth5", "tmem_share_a_depth5", "candidate_not_adopted"),
]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reference_times():
    values = {}
    for node in ast.parse((ROOT / "utils.py").read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("REFERENCE_TIMES", "TIMING_TOLERANCE"):
                    values[target.id] = ast.literal_eval(node.value)
    assert values["TIMING_TOLERANCE"] == 1.30
    return values["REFERENCE_TIMES"]


def current_steps(inputs):
    inputs.add(ROOT / "utils.py")
    refs = reference_times()
    runs = []
    for run in range(1, 6):
        path = ROOT / f"results_b300/all_steps_9e3b989_{run}.csv"
        inputs.add(path)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(refs) == 37
        by_shape = {tuple(int(row[k]) for k in ("step", "M", "N", "K")): row for row in rows}
        assert by_shape.keys() == refs.keys()
        for key, row in by_shape.items():
            assert row["git_revision"] == REVISION and row["gemm_kernels_sha256"] == KERNEL_SHA
            assert row["status"] == "PASS" and row["trials"] == "1"
            assert row["warmup"] == "10" and row["repeat"] == "30"
            assert float(row["reference_ms"]) == refs[key]
            assert ast.literal_eval(row["samples_ms"]) == [float(row["median_ms"])]
            assert float(row["median_ms"]) <= 1.3 * refs[key]
        runs.append(by_shape)
    output = []
    for key in sorted(refs):
        step, m, n, k = key
        samples = [float(run[key]["median_ms"]) for run in runs]
        median = stats.median(samples)
        output.append(dict(step=step, M=m, N=n, K=k, runs=5, median_ms=median,
                           min_ms=min(samples), max_ms=max(samples),
                           tflops_at_median=2*m*n*k/(median*1e9), limit_ms=refs[key]*1.3,
                           slowest_margin_pct=100*(1-max(samples)/(refs[key]*1.3)),
                           samples_ms=json.dumps(samples)))
    return output


def paired_experiments(inputs):
    output = []
    for ident, directory, step, control, candidate, decision in EXPERIMENTS:
        path = ROOT / "results_b300" / directory / "samples.json"
        inputs.add(path)
        record = json.loads(path.read_text())
        cases, orders = record["cases"], record["orders"]
        a, = [i for i,c in enumerate(cases) if c["step"] == step and c["variant"] == control]
        b, = [i for i,c in enumerate(cases) if c["step"] == step and c["variant"] == candidate]
        before, after = cases[a]["samples_ms"], cases[b]["samples_ms"]
        assert len(before) == len(after) == len(orders) > 0
        assert all(math.isfinite(value) and value > 0 for value in before + after)
        assert all(sorted(order) == list(range(len(cases))) for order in orders)
        assert cases[a]["size"] == cases[b]["size"]
        before_count = sum(order.index(b) < order.index(a) for order in orders)
        positions_a = [sum(o.index(a) == pos for o in orders) for pos in range(len(cases))]
        positions_b = [sum(o.index(b) == pos for o in orders) for pos in range(len(cases))]
        balanced = (before_count * 2 == len(orders)
                    and len(set(positions_a)) == len(set(positions_b)) == 1)
        output.append(dict(id=ident, step=step, size=cases[b]["size"], control=control,
                           candidate=candidate, trials=len(orders),
                           control_median_ms=stats.median(before), candidate_median_ms=stats.median(after),
                           paired_speedup=stats.median(x/y for x,y in zip(before, after)),
                           paired_latency_reduction_pct=stats.median(100*(1-y/x) for x,y in zip(before, after)),
                           paired_saving_us=stats.median(1000*(x-y) for x,y in zip(before, after)),
                           faster_pairs=sum(y < x for x,y in zip(before, after)),
                           candidate_first_count=before_count, positions_balanced=balanced,
                           control_positions=json.dumps(positions_a), candidate_positions=json.dumps(positions_b),
                           decision=decision, source=str(path.relative_to(ROOT))))
    return output


def tables(current, experiments):
    lines = ["# 自动重算的数据表", "", "由 `tools/build_optimization_data.py` 从原始 CSV/JSON 生成。", "",
             "## 当前版本：五次完整运行的中位数", "",
             "同一形状才可比较；跨 Step 是完整配置观察，不是单因素实验。单位 ms。", "",
             "| Step | 1024³ | 2048³ | 4096³ | 8192³ |", "|---|---:|---:|---:|---:|"]
    index = {(r['step'],r['M'],r['N'],r['K']):r for r in current}
    for step in range(3, 11):
        cells = [f"{index[step,n,n,n]['median_ms']:.6f}" if (step,n,n,n) in index else "—" for n in (1024,2048,4096,8192)]
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    lines += ["", "## 已归档独立实验", "", "耗时下降为逐对 `(1 - 候选/对照)` 的中位数，负数表示变慢。", "",
              "| id | 对照→候选中位 ms | 配对加速 | 耗时下降 | 更快轮数 | 候选先测 | 位置完全平衡 | 原始数据 |",
              "|---|---|---:|---:|---:|---:|---|---|"]
    for r in experiments:
        lines.append(f"| {r['id']} | {r['control_median_ms']:.6f} → {r['candidate_median_ms']:.6f} | "
                     f"{r['paired_speedup']:.6f}× | {r['paired_latency_reduction_pct']:+.3f}% | "
                     f"{r['faster_pairs']}/{r['trials']} | {r['candidate_first_count']}/{r['trials']} | "
                     f"{'是' if r['positions_balanced'] else '否'} | [JSON](../../{r['source']}) |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")


def plots(current):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    dest = ROOT / 'docs/figures'
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2,2,figsize=(12,8), layout='constrained')
    for ax,n in zip(axes.flat,(1024,2048,4096,8192)):
        rows = [r for r in current if (r['M'],r['N'],r['K']) == (n,n,n)]
        x = [1000*r['median_ms'] for r in rows]
        errors = [[1000*(r['median_ms']-r['min_ms']) for r in rows],
                  [1000*(r['max_ms']-r['median_ms']) for r in rows]]
        ax.barh([f"Step {r['step']}" for r in rows],x,xerr=errors,capsize=3,
                color=['#15806e' if r['step']==10 else '#557fa3' for r in rows])
        for i,value in enumerate(x):
            ax.text(value+max(x)*.025,i,f'{value:.2f}',va='center',fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,max(x)*1.19)
        ax.set_title(f'M = N = K = {n}');ax.set_xlabel('Latency (microseconds; lower is better)')
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Current configuration by step: five complete B300 runs at 9e3b989\nBars: median; whiskers: observed min/max (not confidence intervals)',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(dest/f'current_steps.{ext}',dpi=170)
    plt.close(fig)

    p = ROOT/'results_b300/step10_geometry.PywrmG/step10_4096/samples.json'
    cases=json.loads(p.read_text())['cases']
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    labels=['Baseline: N128 / 64 clusters / depth4','N128 / 74 clusters / depth4',
            'N64 / 74 clusters / depth4','N64 / 74 clusters / depth5']
    for c,label,color in zip(cases,labels,('#15806e','#4178b0','#b85340','#ac832b')):
        ax.plot(range(1,9),[x*1000 for x in c['samples_ms']],marker='o',label=label,color=color)
    ax.axhline(139.1,linestyle='--',color='#555555',label='Assignment limit: 139.1 us')
    ax.set(xlabel='Trial (each includes all four variants in balanced order)',ylabel='Latency (microseconds)',xticks=range(1,9))
    ax.set_ylim(85,158);ax.grid(alpha=.18)
    ax.set_title('Geometry probe: every candidate is slower than its same-trial baseline\nRaw samples; timing state changes within the run')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=9)
    for ext in ('png','pdf'):fig.savefig(dest/f'geometry_trials.{ext}',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=set()
    current=current_steps(inputs)
    experiments=paired_experiments(inputs)
    write_csv('current_steps.csv',current)
    write_csv('paired_experiments.csv',experiments)
    tables(current,experiments)
    (OUT/'sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted(inputs)},indent=2)+'\n')
    if args.plots:plots(current)
    print(f'Recomputed {len(current)} shapes, {len(experiments)} comparisons from {len(inputs)} input files.')


if __name__ == '__main__':
    main()
