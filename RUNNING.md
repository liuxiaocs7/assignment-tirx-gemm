# Blackwell GEMM：安装、测试与性能测量

## 1. 当前实现与验证范围

`gemm_kernels.py` 已实现 Step 1–10，每个 Step 独立提交。实现参考
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/)
的 GEMM 基础、异步优化和高级优化章节，当前使用 **Apache TVM 0.26.0 API**。
参考教程的本地版本为 `61415b0`。

| 作业步骤 | 实现 | 与教程的对应关系 |
|---|---|---|
| 1 | 单 tile，同步加载，FP32 累加和 FP16 写回 | 教程 Step 1 |
| 2 | 沿 K 分块累加，复用 MMA barrier | 教程 Step 2 |
| 3 | 二维 CTA 网格 | 教程 Step 3 |
| 4 | TMA 异步加载与写回 | 教程 Step 4 |
| 5 | 两级预取流水线 | 教程 Step 5 |
| 6 | 常驻 CTA、L2 友好调度、跨 tile 保留 phase | 教程 Step 6 |
| 7 | TMA / MMA / 写回分工 | 教程 Step 7 |
| 8 | K64 四级流水线、分块写回、等待提示与 TMEM 基址缓存 | 教程在后续 cluster 中使用四级流水线 |
| 9 | 双 CTA 协作，cluster 输出 256×256 | 教程 Step 8 |
| 10 | 两个 MMA consumer 共享 B，按输出量选择 512×128/256 及 TMEM 单/双缓冲 | 教程 Step 9 |

**当前正式版本在本次 B300 分配下验收通过。** 最终结果
[step10_final.vM3h3I](results_b300/step10_final.vM3h3I/) 在 `6e5f5f4`、Slurm
job 27503 / step 0 执行：全量 pytest **57 passed，76.59 s**；正式 Step 10
四个尺寸全部 PASS，原始计时样本 **28/28 达标**，两个退出码均为 0。
4096 最慢为 **0.094208 ms**，相对 0.139100 ms 门槛有 **32.27%** 余量。

正式内核仍为 `d283549` 采用的配置，四尺寸 CUDA/cubin 与首次验收逐字节一致。
历史 `ea69b2c` 五轮全步骤 benchmark 中 4096 的两次 SLOW 仍保留；本次验收
限定于记录的 GPU 与运行条件。共享 A 五级在前次 201 轮诊断中有约 0.44%
直接配对信号，但旧测量顺序存在位置偏置，尚未确认归因或采用。
后续验证属于可选优化，当前验收不依赖它。
完整证据与历史见 [B300_VALIDATION.md](B300_VALIDATION.md)，各步原理与
可选改进见 [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)。

本机已用 TVM 0.26.0 完成全部 10 个 step 的 TIR 构建、lowering 和 CUDA 源码生成，
覆盖 SM100a / SM103a，以及短 K、矩形和不完整流水线；本机没有 NVIDIA GPU，
这些检查与服务器的正式 GPU 验收分开记录。

不依赖 GPU 的工具测试可单独运行：`uv run python -m pytest tool_tests/ -q`
（覆盖构建、实测 CUDA/trace 重放、fallback、实验隔离、编译回调、角色插桩及硬件采集入口，
review 报告在 `c788f3a` 完整运行 **636 passed**；本轮覆盖 **311 个不同用例**：
相关回归 230 passed，候选生成检查 79 passed，随后扩充 CLI 检查并重跑
21 passed（其中 19 项与前述重叠）。未重跑完整工具集。
依赖 TVM 的用例在没有 TVM 时跳过，跳过不能作为源码生成通过）。

主文件保持自包含。原作业提交说明只收 `gemm_kernels.py`，但当前实现使用
TVM 0.26，不能据此推断兼容旧 `mlc-ai-tirx-cu130==0.0.1b2` grader；若要
提交原课程平台，需先核对实际 grader 依赖与调用接口。新增测试覆盖短 K、
奇数个 K tile、矩形输出和常驻 CTA 的跨 tile 重用；原有 37 个评分用例不变。
review 后新增五个 Step 10 分支边界用例，当前全量为 62 项，其中新增五项
尚待 B300 验证。可单独运行：

```bash
uv run python -m pytest tests/test_step10.py -k dispatch_boundaries -vs --tb=short
```

### 当前采用的配置与最新验收成绩

历史验收内核提交为 `d2835492913d94532bcd160ea79891a8f0e9d836`，SHA256 为
`5515a04dfc3018bff2fe06e7f1f00681db4ee4ce8b376f4098f2348d85989b6c`。
review 后仅增加/更正内核注释，当前 SHA256 为
`1a06fa5715eda5567a5635eb285480ef8c0f2dc67e85da65cf38e6fc57052f85`，
Python AST 不变，生成代码通过现有已测版本重放检查。
以下是最新正式 Step 10 benchmark 汇总，四个尺寸均为 7/7 样本达标。
首次验收与中间超线结果保留在 [验证历史](B300_VALIDATION.md) 中：

| 尺寸 | 正式选择 | 七轮中位数 ms | 门槛 ms | 结果 |
|---|---|---:|---:|---|
| 1024 | N128 / EPI32 / TMEM 单缓冲 | 0.010793 | 0.032500 | PASS |
| 2048 | N128 / EPI32 / TMEM 单缓冲 | 0.016595 | 0.045500 | PASS |
| 4096 | N128 / EPI32 / TMEM 双缓冲 | 0.094018 | 0.139100 | PASS |
| 8192 | N256 / EPI64 / TMEM 单缓冲 | 0.799646 | 0.946400 | PASS |

选择在编译时完成：输出面积不超过 4096² 时用窄 N，窄 tile 数超过最大 cluster
数时才启用双槽；其余用宽 N。接口对齐要求、两 consumer 结构、四级 K64、计时器
和原评分门槛均不变。该规则的性能证据覆盖以上四方阵，不能外推任意矩形和 K。

下面命令用于未来内核改动后验收或独立复现；当前正式验收已完成，无需重复。
保证 Slurm 分配的剩余时间覆盖编译、benchmark 和全量测试（已展示的每轮
全量约 76 s），各 GPU 任务顺序执行。每次命令记录源码指纹、作业 ID、日志及退出码；
`pipefail` 防止 `tee` 掩盖测试失败。保留所有轮次，包括失败结果。

```bash
cd ~/assignment-tirx-gemm
bash <<'SH'
set -uo pipefail
mkdir -p results_b300
tirx_run=$(mktemp -d results_b300/step10_validate.XXXXXX) || exit 1
tirx_failed=0

run_logged() {
  local tirx_label=$1
  shift
  {
    git rev-parse HEAD
    git status --short
    sha256sum gemm_kernels.py
    printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unset}"
  } > "$tirx_run/${tirx_label}_metadata.txt"
  "$@" 2>&1 | tee "$tirx_run/$tirx_label.log"
  local tirx_exit=$?
  printf '%s\n' "$tirx_exit" > "$tirx_run/${tirx_label}_exitcode.txt"
  if [ "$tirx_exit" -ne 0 ]; then
    tirx_failed=1
  fi
  return "$tirx_exit"
}

run_logged pytest_step10 uv run python -m pytest tests/test_step10.py -vs --tb=short
run_logged benchmark_step10 uv run python -u benchmark.py --steps 10 --trials 7 \
  --csv "$tirx_run/step10.csv" --diagnostics-dir "$tirx_run/compiler_step10"

for tirx_trial in 1 2; do
  run_logged "pytest_all_$tirx_trial" uv run python -m pytest tests/ -vs --tb=short
done
printf '结果目录：%s\n' "$tirx_run"
exit "$tirx_failed"
SH
```

汇总 PASS 只表示各尺寸的中位数达标；判断每轮及最慢成绩需读取保存的原始样本。
退出码文件表示包含 `tee` 的管道状态；整个脚本在任一命令失败时返回非零。
作业被外部取消时可能来不及写退出码，需连同 Slurm 状态判断。

benchmark 保存正式 CUDA/cubin、编译参数和七轮原始样本，可核对采用后是否仍与
实测候选一致。这里是正式评分路径，不再使用 `--variants n128_tmem_double_buffer`。
probe 默认只运行新的生产 baseline；历史变体依赖旧 N256 builder，会明确拒绝重复
应用。要重放四尺寸历史实验需使用其记录的 `0f23484`，不能把新旧 baseline 混用。

### 可选优化：Step 9 缓存 TMEM 基址

当前正式验收已通过。可选独立候选
`cluster_cache_tmem_base`：在初始化的 CTA/cluster 同步之后读取一次 TMEM
基址，用于后续 MMA 和读回，保留 Step 9 的四级输入、单 consumer、网格与
同步协议。首轮 [step9_cache.1WDeei](results_b300/step9_cache.1WDeei/) 已完成
四尺寸和边界验证，56/56 计时样本达标。4096/8192 配对加速约
1.0146×/1.0201×，但旧顺序七轮都先测 baseline，暂不采用。

同步修复后的代码，在 B300 仓库目录仅复核这两个尺寸：

```bash
bash run_step9_cache.sh --size 4096
bash run_step9_cache.sh --size 8192
```

不带参数时脚本测试全部四尺寸。每个尺寸包含正式 baseline 与一个缓存
候选，各测七轮，warmup=10、repeat=30；修复后为 AB/BA 交替，即七轮
4 次 baseline 先测、3 次候选先测，顺序保存在 `samples.json`。
自动记录版本、CUDA 设备 UUID、
前后 GPU 快照、日志及 probe/tee 退出码；结果放在 `results_b300/step9_cache.*`。
每个尺寸计时前，还校验单 tile、矩形 K64/192/256/320，以及 9×9 cluster
tile 网格的部分 L2 分组，每组执行两次。已有四尺寸首轮数据，本次先确认
大尺寸收益是否在两种先后顺序下都出现，再确定候选采用范围。

底层调用是 `probe_persistent.py --steps 9 --variants cluster_cache_tmem_base`。
baseline 是直接对照，默认不启用候选；工具不修改 `gemm_kernels.py`。SLOW
保留为有效测量，数值、编译或日志写入错误会停止脚本。重点比较配对耗时、
全部原始样本和 cubin/SASS 资源；若没有可重复收益或出现其他尺寸退化，就保留
正式实现。候选采用后再运行全量 pytest 与正式 Step 9 benchmark。

本地 TVM 0.26 检查覆盖 SM100a/SM103a 的十组形状，确认缓存值代回后整个
CUDA 函数体和 TMA 描述符与 baseline 一致，动态 SMEM 仍为 148480 bytes。
新增测试 **22 passed**，相关旧实验回归 **102 passed**；脚本通过语法检查和
八种模拟调用/故障场景。它们没有执行 NVIDIA GPU 数值或性能测量。

### 共享 A 五级采用前验证

**可选后续优化：当前正式版本已验收通过，以下流程尚未执行，也不是当前验收要求。**

`step10_share_a_state.fGOjDy` 已完成 201 轮状态采集，五级约 0.44% 的直接配对
信号跨时间段和旧顺序出现，但位置偏置仍需复核，六级没有额外收益。
若以后决定继续优化，可只测试五级，在同一 Slurm 分配/同一 GPU 上完成无监控
四尺寸对照；请使用修复 `trial_order()` 后的代码，旧 `f029ed7` 只用于历史重放：

```bash
bash <<'SH'
set -uo pipefail
mkdir -p results_b300
tirx_run=$(mktemp -d results_b300/step10_share_a_validate.XXXXXX) || exit 1
{
  date -Iseconds
  hostname
  git log -1 --oneline
  printf 'job=%s step=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
} > "$tirx_run/session.txt"
nvidia-smi -q > "$tirx_run/gpu_before.txt" 2>&1
tirx_exit=0
for tirx_size in 1024 2048 4096 8192; do
  uv run python -u probe_persistent.py --steps 10 --size "$tirx_size" --trials 7 \
    --variants tmem_share_a_depth5 \
    --output "$tirx_run/step10_$tirx_size" 2>&1 | tee "$tirx_run/step10_$tirx_size.log"
  tirx_status=("${PIPESTATUS[@]}")
  printf '%s\n' "${tirx_status[0]}" > "$tirx_run/probe_$tirx_size.exitcode.txt"
  printf 'probe=%s tee=%s\n' "${tirx_status[@]}" > "$tirx_run/pipeline_$tirx_size.txt"
  tirx_exit=${tirx_status[0]}
  if (( tirx_exit == 0 )); then tirx_exit=${tirx_status[1]}; fi
  if (( tirx_exit != 0 )); then break; fi
done
nvidia-smi -q > "$tirx_run/gpu_after.txt" 2>&1
printf '结果目录：%s；退出码：%s\n' "$tirx_run" "$tirx_exit"
exit "$tirx_exit"
SH
```

每个尺寸自动包含 `baseline`、`tmem_input_depth5`、`tmem_share_a_depth5` 三项。
1024/2048/8192 保留正式 fallback，4096 执行共享 A 新布局；同样进行矩形短 K
边界校验。这里仅保存运行前后快照，不启动连续监控；保持 warmup=10、repeat=30。
比较全部样本和直接对照，不因 SLOW 重跑直到通过。若数值/CUDA/日志错误则
保留现场并停止后续尺寸；SLOW 作为有效测量保存，probe 仍返回 0。

这一步只确认采用候选及回退路径的行为和无监控收益；有可重复收益后再决定
是否正式改内核，改动后重新执行全量 pytest 与正式 benchmark。无需继续重复
201 轮状态采集。历史另一 GPU 上的贴线失败仍保留，不把本轮余量外推给所有 GPU。

### 共享 A 结果漂移时的状态采集复测

以下诊断流程已经完成，结果为 `step10_share_a_state.fGOjDy`；之后的正式验收
也已完成。此处保留供历史重放或未来出现新的状态变化时使用。

`step10_share_a_state.X3xsC4` 已完成七轮状态采集。慢段的 SM 采样降到
1507 MHz，前后 SW Power Capping 计数增加 231.687 ms，没有热降频计数增长。
本次与此前角色诊断的物理 GPU UUID 不同，说明同一主机上的 Slurm 分配可能
使用不同 GPU。此次计时窗口约 131 ms，仅覆盖一个 200 ms 状态采样点；
不再重复原七轮窗口，改为延长原四版本序列，观察连续运行后的状态与收益。
保持当前 Slurm 分配，同步包含可选 `--trials` 的
[run_step10_state.sh](run_step10_state.sh) 后，在 B300 终端运行：

```bash
bash run_step10_state.sh --trials 201
```

脚本只运行一次原四版本 probe，将轮数改为 201，保留每轮 warmup=10、repeat=30，
预计计时阶段持续数秒；默认不带参数时仍为 7 轮。无需先跑全量 pytest。
它不更改实验实现或正式内核，也不设置 GPU 时钟和功耗。
`nvidia-smi -q` 保存驱动、UUID、功耗限制和时钟事件等可用字段；每 200 ms
采集 GPU 状态，每 1 s 采集可见计算进程。CUDA 设备 UUID 用来对应物理 GPU，
不能仅靠逻辑序号 0
判断两次是否使用同一张卡。若采样命令不受支持，错误保存在 `gpu_samples.err`，
进程查询失败保存在 `compute_processes.err`，不能把采集缺失解释成没有竞争。
`probe_exitcode.txt` 保存 probe 自身退出码，`pipeline_exitcodes.txt` 另外记录
时间戳进程和 tee 的状态；无论 probe 成功或失败，退出时都清理本次监控进程。

这轮带监控的运行用于诊断；200 ms 采样和毫秒级日志接收时间不足以覆盖每个
kernel，
日志和监控也可能扰动进程间隔。保留所有样本，查看时间区间与状态是否共同变化，
不要据单个快照直接断言降频。若没有捕获到状态变化但耗时仍漂移，再检查宿主
发射间隔、缓存和调度等因素。确定可比较条件后，再用下方原命令无监控复测候选。

全部 201 轮保留，不能事后只挑快段或删除超线结果。这是延长观测的诊断，
不能拿新的长序列中位数替换原始七轮或正式 pytest 的成绩。
结果目录会自动打印。分析时保留整个目录，至少需要 `step10/samples.json`、
`step10/run.json`、`step10.log`、`gpu_samples.csv`、`compute_processes.csv`、
各 `.err`、`gpu_before.txt`、`gpu_after.txt`、`cuda_device.txt`、`session.txt`
和退出码文件；只看 summary 无法对应耗时与状态。

包装脚本已通过 Bash 语法检查及 14 种本地模拟：默认/201 轮、probe/tee/
时间戳失败、监控查询失败、CUDA 设备检查失败，以及七种非法参数。核对了
退出码、轮数传递和监控清理；这些模拟未执行 GPU 命令，不属于 GPU 性能验证。

### 共享 A 与六级输入缓冲实验

首轮已在 `f029ed7` 执行，结果见本文开头；以下保留原实验命令。
用于独立比较输入共享方式，不需要先重跑全量 pytest：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_share_a.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 --trials 7 \
  --variants tmem_share_a_depth6 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
tirx_probe_exit=$?
printf '%s\n' "$tirx_probe_exit" > "$tirx_run/probe_exitcode.txt"
printf '结果目录：%s；退出码：%s\n' "$tirx_run" "$tirx_probe_exit"
```

工具自动补齐四个版本：`baseline` → `tmem_input_depth5` →
`tmem_share_a_depth5` → `tmem_share_a_depth6`。后两项各自相对前一项只推进
输入共享方式或输入深度，前两项保留正式及实测对照。共享 A 的每级 TMA 请求
减少 20%，但不保证相同比例的提速；寄存器与实际延迟需看 B300 编译产物和计时。

新候选保持 TMEM 双槽和原正确性容差，在计时前检查评分形状及五个矩形短 K
组合，各边界执行两次。输入环 K64/320/384/448 覆盖六级前、恰好一圈及超出
一圈；`(1536,5376,320)` 检查非完整 L2 分组。原五级对照的六个边界也会验算。
全部通过后按原 warmup=10、repeat=30 交错计时，保存样本、源码、cubin 和 SASS。

重点比较共享 A 五级相对原五级、共享 A 六级相对共享 A 五级的配对收益，以及
相对正式 baseline 的最慢样本余量。出现数值/CUDA 错误先处理；SLOW 是有效
测量，不能丢弃。明确有益后再做独立复测、适用形状验证和正式采用后的全量验收。

### 当前路径输入环对照实验

四版本实验已保存在 `step10_input_ring.9SsKZM`：两级 K64/K128 明显退化，
K64 五级仅小幅改善。随后两版本独立复测已保存在 `step10_depth5_recheck.sMMqkr`：
五级中位数 0.138745 ms、最大值 0.139429 ms，4/7 达标，稳定性仍未解决。
不再要求重复这组测试；以下命令仅供重放。**已有 `dd90c57` 就能运行**：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_depth5_recheck.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 --trials 7 \
  --variants tmem_input_depth5 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
tirx_probe_exit=$?
printf '%s\n' "$tirx_probe_exit" > "$tirx_run/probe_exitcode.txt"
printf '结果目录：%s；退出码：%s\n' "$tirx_run" "$tirx_probe_exit"
```

工具自动补齐正式 `baseline`，共两个版本。保持原 warmup=10、repeat=30，
每轮交错顺序。先校验两者的评分形状，再将五级候选的 `(4096,3072,K)`、
K=64/128/192/256/320/384 各验算两次，覆盖部分输入环和第三个输出 tile
对 TMEM 槽零的复用。

五级候选仅改当前窄 N 双槽路径的输入深度，小网格单槽/大面积宽 N 保留正式
配置。两次短 K 和 4096 数值均通过，REG=112、STACK/LOCAL=0；两次编译产物
完全一致，小幅相对收益重复出现，但第二次有三个样本超线。正式内核、门槛和
计时器不变。若需要重放完整四版本实验，用新的目录并把参数改为
`--variants tmem_k128_depth2 tmem_input_depth5`。

请保留整个目录，尤其 `summary.csv`、原始样本和编译产物。除了中位数，看
候选相对 baseline、直接对照的配对收益，以及最大值/门槛内样本数。出现
`SLOW` 是有效测量，不能丢弃；CUDA/编译错误则需要先处理再继续。明确有益后
验证受影响形状，最后走上面的正式验收流程。当前五级只保留为小收益候选和
后续实验的直接对照，不能因中位数 PASS 就宣称最终稳定通过。

### Step 10／4096 不稳定时：采集当前路径

以下流程已在 `cede7ec` 执行并回传 `step10_current_profile.3cjntP`，保留供之后
内核变化时重采。旧 N256 单槽报告不能直接解释新内核；两条命令采集当前
生产 builder，无需更改评分设置。

在已分配 B300 的终端顺序运行，确保 Slurm 余时足够。角色工具会验算 baseline
与三个独立插桩副本，保存每轮原始 timing、逐 tile trace、编译产物和轮前后 GPU
快照；这些快照不等同于精确覆盖每个 kernel 的时钟遥测。

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_current_profile.XXXXXX)
uv run python -u profile_persistent.py --steps 10 --size 4096 --trials 7 \
  --output "$tirx_run/roles" 2>&1 | tee "$tirx_run/roles.log"
tirx_roles_exit=$?
printf '%s\n' "$tirx_roles_exit" > "$tirx_run/roles_exitcode.txt"
printf '角色诊断目录：%s；退出码：%s\n' "$tirx_run" "$tirx_roles_exit"
```

重点看 MMA 的 `handoff`（等待累加器可复用）、`wait`（输入就绪），以及 writeback
的 `work`（读回）和 `epilogue`；TMA/MMA `work` 是发射时间，不是异步硬件执行
时长。不同角色并行，不能把百分比相加，也不能把插桩副本的耗时当作优化成绩。
只对未经插桩的 baseline 统计门槛内样本数；采集成功不等于 baseline 全部达标。

随后在同一 shell 中采集当前生产内核 NCU（原生报告与角色 trace 分开保存）：

```bash
uv run python -u profile_hardware.py \
  --output "$tirx_run/hardware" 2>&1 | tee "$tirx_run/hardware.log"
tirx_hardware_exit=$?
printf '%s\n' "$tirx_hardware_exit" > "$tirx_run/hardware_exitcode.txt"
printf '完整结果目录：%s；退出码：%s\n' "$tirx_run" "$tirx_hardware_exit"
```

如果角色工具出现 CUDA 数值或运行错误，先检查日志再运行后续诊断；每个工具
使用独立进程。NVRTC/ncu 失败会保留已有文件，不能用空报告作结论。
本机能复核源码生成和导出解析，无法代替服务器执行这些 GPU 命令。

若后续要单独重现“全步骤 benchmark 的运行上下文”，沿用原命令保存失败现场：

```bash
uv run python -u benchmark.py --steps all --trials 1 \
  --csv "$tirx_run/all_steps.csv" --diagnostics-dir "$tirx_run/compiler_all" \
  2>&1 | tee "$tirx_run/all_steps.log"
tirx_benchmark_exit=$?
printf '%s\n' "$tirx_benchmark_exit" > "$tirx_run/all_steps_exitcode.txt"
```

这条命令已有真实 SLOW，不需要为了“证明失败”反复跑到出现相同结果。新增
`--diagnostics-dir` 用于补齐当次二进制；计时前移除编译 hook，保留原 timer。
独立单尺寸 benchmark 可帮助比较上下文，但在它通过时不能推翻全步骤的 SLOW。

### b033434：离线恢复已有硬件报告

`results_b300/step10_hardware.Mosdpx/analysis/metrics.csv` 已恢复全部 795 个指标及
其原始单位。原 `profile/` 和失败的 `run.json` 均未覆盖；修复后的采集入口可直接
处理这种宽表，也继续支持长表。新版在采集后额外保存便于阅读的 `metrics.csv`。
如果要在服务器上自行恢复已有 CSV，可运行以下离线命令，无需 ncu、Torch 或 GPU：

```bash
tirx_run=$(mktemp -d results_b300/step10_hardware_recover.XXXXXX)
uv run python -u profile_hardware.py \
  --analyze results_b300/step10_hardware.Mosdpx/profile \
  --output "$tirx_run/analysis"
printf '结果目录：%s\n' "$tirx_run"
```

离线模式会检查原 worker 的验算完成状态、原父子进程源码指纹是否一致、内核数量
及数值指标，并保存 `analysis.json` 和 `metrics.csv` 到新目录。它不重跑内核、
不重新导出原生报告，也不将原 `run.json` 中的失败状态改为成功。
若需要 ncu 自带的 rules/details 页面，可从已有报告导出，不用 GPU：

```bash
/usr/local/cuda/bin/ncu \
  --import results_b300/step10_hardware.Mosdpx/profile/step10_4096.ncu-rep \
  --page details --print-details all > "$tirx_run/details.txt"
```

### 硬件采集命令（仅需新报告时使用）

需要新报告时运行以下命令。需要支持 B300 的 Nsight Compute `ncu`；
沿用当前 Slurm GPU 分配和 Python 环境，无需改编译选项。

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_hardware.XXXXXX)
uv run python -u profile_hardware.py \
  --output "$tirx_run/profile" 2>&1 | tee "$tirx_run/profile.log"
printf '结果目录：%s\n' "$tirx_run"
```

工具先检查 `ncu --version` 和可用 section；从 PATH、`CUDA_PATH/bin/ncu`
（默认 `/usr/local/cuda/bin/ncu`）查找，也可添加 `--ncu /实际路径/ncu`。
编译并首次运行正式 `hgemm_v10`，移除编译 hook，检查结果并预热十次，再只采集一个 GEMM。
采集前将输出置为 NaN，采集后重新验算，防止误用预热的输出。cuBLAS 验证位于
profiler 区间外；过滤器限定 `kernel_kernel`，原始报告也会检查只有一个内核结果。

`profile/` 保存 `run.json`、`worker.json`、编译 CUDA/cubin/资源/SASS、`ncu.log`、
`raw.csv`、`metrics.csv`、`details.txt` 及 `.ncu-rep` 或 `.nsight-cuprof` 报告。采集项包括
SpeedOfLight、Compute/MemoryWorkloadAnalysis、SchedulerStats、WarpStateStats、
LaunchStats、Occupancy；缺少的 section 会明确记录。使用 kernel replay，显式设置
`--clock-control none --cache-control none`；版本支持时设置动态 Tensor Core boost。
缓存不刷新时，多 pass 的指标可能受之前的工作影响；profiling 本身也会扰动执行，
**报告耗时不能当作评分或稳定通过的证据**。

缺少 `ncu`、GPU 不受支持、`ERR_NVGPUCTRPERM`、空报告或验证失败时返回非零，
保存已有日志，不修改驱动权限。缺少 ncu 时可加载平台提供的 Nsight Compute module
或指定已有安装路径；计数器权限错误需平台管理员开放访问，保留目录供判断。
当前 probe 默认运行正式版本，也可显式指定 `--variants baseline`。
历史展开和 TMEM 对照需在各自记录的提交重放。
详见 [最新复测与诊断依据](B300_VALIDATION.md)。

已采用的 Step 6/7 K128、Step 8 等待提示/基址缓存、Step 10 基址缓存/均衡网格/B 优先
拒绝重复应用；依赖旧基线的历史组合需在对应历史提交重放。profiling 使用当前
生产网格，历史 trace 应使用其保存的 layout 解读。

### 已有 B300 + Torch + TVM 0.26 + uv：直接运行

先把此次修复同步到服务器（见下一节），在仓库根目录运行。若 Slurm 已分配 GPU，保留它设置的
`CUDA_VISIBLE_DEVICES`；无需重装你现有的 Torch、TVM 或 FFI。

```bash
mkdir -p results
set -o pipefail

# 先确认运行的是新代码和正确的 Python 环境
uv run python -c "import sys, tvm, gemm_kernels; print(sys.executable); print(tvm.__version__); print(gemm_kernels.__file__)"

# 快速获得所有 step 的 37 组正确性检查、性能、cuBLAS 对照和 CSV
uv run python -u benchmark.py --steps all --trials 1 --csv results/all_steps.csv 2>&1 | tee results/benchmark.log

# 完整验收：62 个 GPU 用例，包含 25 个额外边界用例
uv run python -m pytest tests/ -v -s --tb=short 2>&1 | tee results/pytest.log
```

`--trials 1` 保留默认的 10 次预热和 30 次计时。首次执行包含 JIT 编译等待，报告的耗时不含编译。
两条 GPU 命令按顺序执行，避免同时抢占 GPU；如果 benchmark 出现数值错误或 CUDA 异常，
先定位该错误。有 `SLOW` 时 benchmark 会继续收集其他形状，并返回退出码 1。
pytest 不加 `-x`，可汇总所有性能失败；发生 CUDA 异常后应结束进程并单步重试。

## 2. 把本地提交带到服务器

如果提交已经推送到你使用的远端，可在服务器 clone / pull 后查看 `git log -10 --oneline`。
本次操作只创建本地 commit，不自动推送。10 个步骤各有一个 commit；运行工具和文档另有一个 commit。

也可以从 Mac 直接打一个包含完整历史的 Git bundle，不依赖推送：

```bash
# 在 Mac 的 assignment-tirx-gemm 目录执行；替换你的 SSH 别名
git bundle create /tmp/assignment-tirx-gemm.bundle main
scp /tmp/assignment-tirx-gemm.bundle YOUR_GPU_HOST:/tmp/

# SSH 登录服务器，在希望放项目的位置执行
git clone -b main /tmp/assignment-tirx-gemm.bundle assignment-tirx-gemm
cd assignment-tirx-gemm
git log -10 --format='%h %an <%ae> %s'
```

## 3. 硬件和环境准备

使用 **Linux x86_64 + NVIDIA B200（SM100）/ B300（SM103）+ CUDA 13.x**。
CUDA 13.0 通常需要 **580 系列或更新的 NVIDIA 驱动**；如果平台提供经过配置的
CUDA compatibility 环境，以平台说明为准。Toolkit 中需要 `nvcc` / `ptxas`。
B100 也属于 SM100，但本仓库性能门槛取自 B200，不能保证 B100 达到同样分数。
B300（SM103）的测试入口使用实际 SM 数和 `sm_103a` 编译目标；B200 使用 `sm_100a`。
性能门槛仍使用 B200 参考值；本文顶部记录当前 B300 / TVM 0.26 的实测结果，
其他设备或软件版本仍需单独测量。
A100、H100、RTX 4090/5090 和 Mac GPU 不适用这些 SM100 内核。

```bash
nvidia-smi
nvcc --version
command -v ptxas
python3 --version
```

现有环境已满足依赖时，跳过安装。新环境推荐 Python 3.12：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "torch==2.9.1+cu130" --index-url https://download.pytorch.org/whl/cu130
uv pip install "apache-tvm==0.26.0" "apache-tvm-ffi==0.1.13.post3" cuda-bindings pytest numpy
uv pip check
```

上面的 Torch 固定版本用于可复现的新环境；已有可用的 CUDA 版 Torch（包括 2.14）无需降级。
当前代码不再支持作业最初的 `mlc-ai-tirx-cu130==0.0.1b2`，不要将两种 TVM 包混装。
原来的 `No module named 'tvm.tirx.op_schedule'` 是旧源码与 TVM 0.26 的 API 不匹配，
已通过迁移全部内核修复；只替换 import 路径不足以解决。

TVM 0.26 默认使用 NVRTC 延迟编译 CUDA。若平台只有完整 Toolkit、NVRTC 加载失败，
可以设置 `export TVM_CUDA_COMPILE_MODE=nvcc` 后重试，并确保 `nvcc --version` 为 CUDA 13.x。

显式选择空闲 GPU，再检查依赖：

```bash
# 仅在未由 Slurm 分配 GPU 时按需设置：export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch
import tvm
import gemm_kernels
from utils import blackwell_target

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}
device = torch.cuda.get_device_properties(0)
print('TVM:', tvm.__version__)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', device.name, 'SM count:', device.multi_processor_count)
print('Target:', blackwell_target())
print('TIRX imports OK')
PY
```

pytest 和 `benchmark.py` 会按实际 GPU 调整常驻 CTA 数量；直接使用
`hgemm_v6`–`hgemm_v10` 时默认 `SM_COUNT=148`，换设备时需在构建前设置该值。
Step 7–10 的网格还会按输出 tile 数量限制 CTA / cluster 数，避免小矩阵启动空闲任务。

## 4. 按步骤验收

所有命令在项目根目录、激活虚拟环境后运行。先做最小 smoke test：

```bash
python -m pytest tests/test_step01.py -xvs
python -m pytest tests/test_step02.py -xvs
python -m pytest tests/test_step03.py -xvs
```

然后逐步执行全部用例，分别保存日志。每步使用新 Python 进程，有利于定位编译错误或 GPU 异常：

```bash
# 以下循环使用 bash
bash <<'SH'
set -euo pipefail
mkdir -p results
for step in 01 02 03 04 05 06 07 08 09 10; do
  python -m pytest "tests/test_step${step}.py" -xvs 2>&1 | tee "results/step${step}.log"
done
SH
```

全部通过后跑一次完整套件：

```bash
python -m pytest tests/ -xvs
```

当前共有 **62 个 GPU 用例**。其中原有 37 个用例依次执行：

1. 编译并运行 TIRX 内核。
2. 与 `torch.matmul(A, B.T)` 比较，要求 `rtol=1e-3, atol=1e-2`。
3. 预热 10 次，CUDA event 测量 30 次，要求平均耗时不超过参考值的 `1.30` 倍。

其余 25 个边界用例只检查正确性，没有任意新增性能门槛。
历史验收覆盖其中 20 项；review 后新增五项尚待 B300 运行。
默认随机种子是 0；通过后可换种子检查稳定性：

```bash
GEMM_TEST_SEED=1 python -m pytest tests/test_step09.py tests/test_step10.py -xvs
```

需要区分“正确但慢”与“计算不正确”：`Submission too slow` 是性能断言，
`Tensor-likes are not close` 才是数值检查失败。不要为了通过用例而放宽误差容限或改写参考时间。

## 5. 测量性能、比较 cuBLAS、导出 CSV

`benchmark.py` 默认先检查正确性，每个形状只编译一次，再重复计时。
每轮交替 kernel/cuBLAS 与 cuBLAS/kernel，计时前分配输出并校验。输出中位
耗时、TFLOP/s、同形状 cuBLAS 耗时、中位数比值、配对速度比和评分门槛。

```bash
# 最终 Step 10：默认形状 1024 / 2048 / 4096 / 8192
python benchmark.py --steps 10 --csv results/step10.csv

# 相同形状比较后四个优化阶段
python benchmark.py --steps 7,8,9,10 --sizes 4096 8192 --csv results/optimized.csv

# 严格按各自原有测试形状测量所有步骤
python benchmark.py --steps all --csv results/all_steps.csv

# 增加预热和重复次数观察稳定性；这是额外测量，不替代原测试的评分
python benchmark.py --steps 10 --sizes 4096 8192 --warmup 30 --repeat 100 --trials 5 --csv results/step10_stable.csv
```

Step 1 固定 `M=N=128,K=64`，Step 2 固定 `M=N=128`。因此早期步骤不能用
`--steps all --sizes 4096` 做统一方阵比较；对 Step 2，`--sizes` 指 K：

```bash
python benchmark.py --steps 1 --sizes 64
python benchmark.py --steps 2 --sizes 64 512 1024 4096
```

输出示意（具体数值取决于实测）：

```text
step       M       N       K   median_ms   TFLOP/s   cuBLAS_ms   vs_cuBLAS  paired_vs   limit_ms   status
```

- `vs_cuBLAS = cuBLAS_ms / median_ms`，大于 1 表示本内核在当前测量下更快。
- `paired_vs` 是同轮 `cuBLAS_ms / kernel_ms` 比值的中位数，CSV 为
  `paired_speedup_vs_cublas`。原 `speedup_vs_cublas` 含义不变。
- `PASS`：正确性通过且耗时在参考门槛内；`SLOW`：正确但慢；`UNSCORED`：自定义形状没有参考门槛。
- 有 `SLOW` 时脚本退出码为 1，CSV 仍保留。数值验证失败则立即报错。
- CSV 包含环境、种子、计时配置、原始耗时、`trial_orders` 和逐轮
  `cublas_speedup_samples`；启用 diagnostics 时每个形状还保存 `timing.json`。
- 日志和 CSV 记录 commit、工作区是否有修改、内核/工具源码 SHA256、目标架构和编译参数。

计算公式为 `TFLOP/s = 2*M*N*K / (time_ms*1e-3) / 1e12`。
默认计时方式与原测试保持一致：当前 CUDA stream 上的 event 包围重复 kernel launch，
不包含编译和输入创建。Python 发射间隙仍可能影响小矩阵结果；它不是 CUDA Graph 或峰值吞吐测试。
`benchmark.py` 报告多轮中位数，pytest 使用单轮平均值，最终作业验收以 pytest 为准。
旧 CSV 是先测完 kernel 再测 cuBLAS，不能事后视为交替测量。新顺序减少
位置偏置，但不能消除状态漂移；七轮的先后次数为 4/3，偶数轮可以完全平衡。
不要只凭一次约 1% 的比值认定稳定领先 cuBLAS。

Step 10 的 B200 参考门槛如下，仅用于比较：

| 方阵尺寸 | 参考耗时（ms） | 最大允许耗时（ms） |
|---|---:|---:|
| 1024 | 0.025 | 0.0325 |
| 2048 | 0.035 | 0.0455 |
| 4096 | 0.107 | 0.1391 |
| 8192 | 0.728 | 0.9464 |

计时期间保持 GPU 空闲，不启用 `CUDA_LAUNCH_BLOCKING=1`，不要同时开多个 benchmark。
记录环境以便复现：

```bash
mkdir -p results
git rev-parse HEAD > results/commit.txt
nvidia-smi > results/gpu.txt
uv pip freeze > results/packages.txt
```

后续若出现性能退化，先采集原有编译过程的资源信息；下例以 Step 10 为例，
其他步骤按实际问题修改 `--steps`：

```bash
mkdir -p results
set -o pipefail
tirx_run=$(mktemp -d results/b300_diag.XXXXXX)
nvidia-smi > "$tirx_run/gpu_before.txt"
uv run python -u benchmark.py --steps 10 --trials 3 \
  --diagnostics-dir "$tirx_run/compiler" --csv "$tirx_run/focus.csv" \
  2>&1 | tee "$tirx_run/benchmark.log"
nvidia-smi > "$tirx_run/gpu_after.txt"
printf '结果目录：%s\n' "$tirx_run"
```

`--diagnostics-dir` 必须是新目录，避免混入上轮日志。每个形状保存 CUDA 源码、编译二进制，
默认 NVRTC 路径还保存实际参数、编译器版本和编译日志。部分 NVRTC 版本不返回 ptxas 资源统计；
系统有 `cuobjdump` 时另存 `module_01.resources.txt`，直接读取实际二进制的 REG/STACK/LOCAL 信息，
无需重新编译。工具缺失时写明原因，仍可继续计时。
采集复用 TVM 原编译回调，**不更改编译参数**，在正确性检查和预热计时前移除钩子。
未指定该选项时不安装钩子。若使用 NVCC，则保存 `.fatbin`，不采集 NVRTC 日志。
目录内 `capture.json` 的 `modules: 0` 表示没有捕获到编译回调，不能据此判断没有寄存器溢出。
详细产物说明及可选 Nsight Compute 命令见 [B300_VALIDATION.md](B300_VALIDATION.md)。

## 6. 编译、死锁与数值错误排查

查看生成的 CUDA：

```bash
python inspect_cuda.py 1 > results/step01.cu
python inspect_cuda.py 7 1024 > results/step07.cu
python inspect_cuda.py 10 4096 > results/step10.cu
```

`inspect_cuda.py` 的 Step 1 自动使用 `128×128×64`，Step 2 使用 `128×128×size`，
Step 3–10 使用 `size³`。

先把问题缩小到一个用例：

```bash
python -m pytest 'tests/test_step10.py::test_multi_consumer[1024]' -xvs
python -m pytest tests/test_step10.py -k rectangular -xvs
```

怀疑异步错误时，在**新进程**里打开同步定位；此时性能结果无意义：

```bash
CUDA_LAUNCH_BLOCKING=1 python -m pytest 'tests/test_step10.py::test_multi_consumer[1024]' -xvs
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest 'tests/test_step10.py::test_multi_consumer[1024]' -xvs
compute-sanitizer --tool synccheck --error-exitcode 1 python -m pytest 'tests/test_step10.py::test_multi_consumer[1024]' -xvs
```

如果没有 `compute-sanitizer`，使用 CUDA 13 Toolkit 附带的版本。
sanitizer 会扰动耗时，即使无内存/同步报告，也可能触发 pytest 的性能断言。
一旦出现 illegal memory access / XID / launch failure，退出该 Python 进程后再试，
不要继续使用已出错的 CUDA context。

优先排查：

- ImportError：确认同步了新源码、`tvm.__version__ == '0.26.0'`，没有混装旧版 `mlc-ai-tirx-cu130`。
- 编译失败：保留完整堆栈、失败 step/shape、`pip freeze`、`nvcc --version`。
- 小尺寸通过、大尺寸卡住：检查跨 tile phase、TMA/MMA 迭代次数、consumer barrier 槽位。
- 部分行错误：检查 TMEM fence、warpgroup 的 128 线程参与、TMA store 完成等待；Step 10 的两个写回组分别使用 barrier 10 和 11。
- 仅性能失败：检查 GPU 占用、功耗/时钟、设备型号、是否开启同步调试或 sanitizer，再重复测量。

## 7. 可选：Modal 云端测试

没有自己的服务器时可使用仓库原有的 B200 runner：

```bash
python -m pip install modal
modal setup
modal run run_modal.py --step 1
modal run run_modal.py --step 7,8,9,10
modal run run_modal.py
modal run run_modal.py --inspect 10 --size 4096 > results/step10_modal.cu
```

Modal runner 已固定 CUDA 13.0、TVM `0.26.0`、PyTorch `2.9.1+cu130` 和 FFI `0.1.13.post3`。
本次未运行 Modal 云端验证。
它运行 pytest 并输出每个评分用例的耗时与 TFLOP/s；CSV 性能脚本按上面的服务器方式运行。

## 8. 作业打包

在所有 GPU 测试与性能检查完成后：

```bash
tar cvf handin.tar gemm_kernels.py
tar tvf handin.tar
```

归档应只包含 `gemm_kernels.py`。
