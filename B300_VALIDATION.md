# B300 验证记录与性能诊断

## 最新结果：k128_step67.TdkZy5 与完整套件摘要

数据：[pytest_step67.log](results_b300/k128_step67.TdkZy5/pytest_step67.log)、
[step67.csv](results_b300/k128_step67.TdkZy5/step67.csv)、
[probe/summary.csv](results_b300/k128_step67.TdkZy5/probe/summary.csv)、
[probe/run.json](results_b300/k128_step67.TdkZy5/probe/run.json)。

**Step 6、7 正式 16 项测试通过；40 个 benchmark 样本全部达标。**
用户随后回传完整套件摘要：**55 passed / 2 failed，57 项，74.85 s**。
仅 Step 8 / 2048、Step 10 / 4096 的性能断言失败。该摘要目前来自会话；
上传目录尚不含此次完整 pytest 日志，不能为 Step 8 补造具体耗时或运行指纹。

上传的独立测试和 probe 版本为 `ade5040`，B300 / 148 SM / `sm_103a` /
TVM 0.26.0 / PyTorch 2.14.0+cu130 / NVRTC 13.0。
内核 SHA256 为 `11a94f0b64a8f5a1034f432eea11865c7d0fe4ef7bb53c26e35ca615be397aad`。
虽然记录为 dirty，六个 Python 文件指纹与该提交一致；四个 Step 10 变体的
builder/source SHA256 核对通过，编译选项完全相同，初始和逐轮数值验证均通过。

### Step 6、7 正式验收已完成

| Step | 方阵尺寸 | benchmark 中位数 ms | 最慢 ms | 允许 ms | 五轮结果 |
|---|---:|---:|---:|---:|---|
| 6 | 1024 | 0.014475 | 0.014927 | 0.016900 | 全部 PASS |
| 6 | 2048 | 0.041347 | 0.041385 | 0.045500 | 全部 PASS |
| 6 | 4096 | 0.248435 | 0.249065 | 0.310700 | 全部 PASS |
| 6 | 8192 | 1.970222 | 1.973202 | 2.785900 | 全部 PASS |
| 7 | 1024 | 0.014468 | 0.014529 | 0.016900 | 全部 PASS |
| 7 | 2048 | 0.034962 | 0.035127 | 0.040300 | 全部 PASS |
| 7 | 4096 | 0.219531 | 0.219938 | 0.299000 | 全部 PASS |
| 7 | 8192 | 1.867681 | 1.872214 | 2.697500 | 全部 PASS |

评分尺寸和两条 K 宽度路径的边界用例全部通过，确认 `fb7e38b` / `eeb0ae0` 有效。
最窄的 benchmark 余量是 Step 6 / 2048，最慢一轮仍低于门槛约 9.04%。
最新完整套件摘要也未再报告这两步失败。

### Step 8、10 剩余情况

Step 8 / 2048 上一次全量为 0.029792 ms，仅比门槛 0.029900 ms 快约 0.36%；
近期历史记录有通过也有失败。本次具体耗时尚缺，无法量化新差距。期间未改动 Step 8，
不能把这次失败归因为 Step 6、7 的内核改动；仍需用相同环境的配对实验寻找足够余量。

| Step 10 / 4096 变体 | 中位数 ms | 最慢 ms | 配对加速比 | 五轮结果 |
|---|---:|---:|---:|---|
| baseline | 0.142295 | 0.142660 | 1.000× | 全部 SLOW |
| tmem_load_64 | 0.142640 | 0.142770 | 0.998× | 全部 SLOW |
| l2_group_4 | 0.142675 | 0.143065 | 0.997× | 全部 SLOW |
| balanced_clusters | 0.141646 | 0.142030 | 1.006× | 全部 SLOW |

cluster 调整只有约 0.6% 配对收益，最慢一轮仍超 0.139100 ms 门槛，暂不采用。
四个变体均为 REG:168；x64 的 STACK 为 56，其余 32，SMEM 参数未改变。
头文件弃用和未使用符号警告不能解释性能差距，stack 也不是动态 spill 流量。
最新 pytest 摘要给出 Step 10 为 963.33 TFLOP/s，按该舍入值推算约 0.142671 ms，
超门槛约 2.57%。所需吞吐为参考吞吐 / 1.30，约 988.06 TFLOP/s，而非报错中的 1284.48。

### 下一轮：Step 8 与 Step 10 独立对照

本轮生产内核不改，`probe_persistent.py` 只增加以下独立实验，原评分、容差和
CUDA event 方法保留。Step 8 默认三个版本，Step 10 默认四个版本，都包含正式 baseline。

1. **TMA 数据就绪等待（Step 8、10）**：`tma_wait_64ns` 只缩短 `tma2mma` 等待的
   挂起提示。如果生产者/消费者交接因等待响应受阻，应降低耗时。此前 Step 10 测的是
   `mma2tma` / `mma2ld`，没有测这条数据就绪路径。保持 acquire、parity 和 retry；
   64 ns 不是放行超时，仍须反复检查到数据就绪，其他三个等待点保留原实现。
2. **Step 8 写回同步**：`epilogue_128` 将两个 64 列 TMA store 合为一个 128 列。
   若写回同步限制 2048 的性能，应留出更多余量。四级流水线和 x32 TMEM 读取保留；
   动态 SMEM 从 148480 增到 164864 字节。该实验在 Step 7 / 4096 无收益，
   不据此推定不同流水线深度和尺寸的 Step 8 结果。
3. **Step 10 动态 consumer 索引**：`specialize_mma` 与 `specialize_writeback`
   分别把 MMA 或写回的两条角色路径展开，用 0/1 常量索引各自 TMEM、SMEM 和 barrier。
   预测减少动态地址运算，代价是代码体积增大。仍由原来的两条 MMA warp / 两组写回执行，
   不合并 consumer，不改变每个输出 tile 的计算量，两个实验不叠加。

**198 项本地工具及源码生成检查通过**：新增八份实测 CUDA 的数据就绪等待重放，
确认只改变一个等待点；生成并检查 Step 8 的完整四级流水线，以及 Step 10 两个独立
consumer 槽位、10/11 写回同步 ID、cluster 到达和短 K / 矩形重用路径。
本机没有 NVIDIA GPU，新变体尚未验证 NVRTC 编译、数值结果和性能。

同步本地提交后只需执行：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step810_probe.XXXXXX)
git log -2 --oneline

uv run python -u probe_persistent.py --steps 8 --size 2048 \
  --output "$tirx_run/step08" 2>&1 | tee "$tirx_run/probe_step08.log"

uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/probe_step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

两个进程顺序执行，各自五轮交错计时，每轮预热 10 次、计时 30 次，计时前后验算。
回传整个目录即可；`SLOW` 正常结束采集，数值或编译失败则停止。暂不重复跑全量套件。

## 前轮实验：persistent_probe.VB42kg

数据：[probe.log](results_b300/persistent_probe.VB42kg/probe.log)、
[summary.csv](results_b300/persistent_probe.VB42kg/probe/summary.csv)、
[samples.json](results_b300/persistent_probe.VB42kg/probe/samples.json)、
[run.json](results_b300/persistent_probe.VB42kg/probe/run.json)。

**Step 6、7 / 4096 的 K tile 128 变体五轮全部达标；Step 10 的两个变体无收益。**
运行版本 `f414130`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
虽然记录为 dirty，六个 Python 文件指纹与运行提交一致；11 个变体的 builder/source
指纹核对通过，编译选项相同。全部版本通过初始验算和每轮计时后的输出验证。

| Step | 变体 | 中位数 ms | 最慢 ms | 配对加速比 | 五轮结果 |
|---|---|---:|---:|---:|---|
| 6 | baseline | 0.316145 | 0.316791 | 1.000× | 全部 SLOW |
| 6 | **k_tile_128** | **0.249217** | **0.249423** | **1.268×** | **全部 PASS** |
| 6 | mma_wait_64ns | 0.311706 | 0.312639 | 1.014× | 全部 SLOW |
| 6 | final_fence | 0.314058 | 0.314529 | 1.007× | 全部 SLOW |
| 7 | baseline | 0.309405 | 0.310277 | 1.000× | 全部 SLOW |
| 7 | **k_tile_128** | **0.219035** | **0.219488** | **1.413×** | **全部 PASS** |
| 7 | mma_wait_64ns | 0.303867 | 0.305982 | 1.017× | 全部 SLOW |
| 7 | epilogue_128 | 0.310886 | 0.311665 | 0.995× | 全部 SLOW |
| 10 | baseline | 0.142082 | 0.142192 | 1.000× | 全部 SLOW |
| 10 | mma_wait_64ns | 0.142221 | 0.142559 | 0.998× | 全部 SLOW |
| 10 | tmem_load_16 | 0.143234 | 0.143278 | 0.992× | 全部 SLOW |

Step 6、7 的中位耗时分别减少 21.17%、29.21%；最慢一轮分别低于各自门槛
0.310700 / 0.299000 ms 的 19.72%、26.59%。这支持加宽 K tile、减少流水线轮数的方向；
布局和共享内存用量也随之变化，不能只归因于单个硬件延迟。
其他变体的收益不足或回退，均不落到正式内核。Step 10 baseline 超过 0.139100 ms
门槛约 2.14%，本轮未解决。

资源报告中，Step 6 K128 的 REG/STACK 从 150/0 变为 165/112，但耗时明显下降；
Step 7 K128 从 128/8 变为 128/0。Step 10 x16 从 168/32 变为 168/16，耗时反而增加。
stack 大小不能直接代表 spill 流量或性能瓶颈。报告的 SHARED:1024 只是静态部分；
K128 的 A/B 动态 SMEM 用量翻倍。编译日志为头文件弃用和未使用符号警告。

### Step 6、7 已采用实测改动

- `fb7e38b`：Step 6 使用 `BLK_K = 128 if K % 128 == 0 else 64`。
- `eeb0ae0`：Step 7 使用相同选择规则，仍保持两级流水线和 warp 分工。

两者均保留已有 barrier、fence、分配生命周期和跨 tile phase 状态。分别新增
K=128、384 的矩形跨 tile GPU 用例，覆盖宽 K 路径的一轮与三轮流水线。
本地先让实测 CUDA 对照失败，再采用改动；正式 4096 CUDA 主体与相应实测变体逐字一致。
SM100a/SM103a 下，K=64、192 与旧 builder 一致，K=128、384 与实测宽 K builder 一致。

当时 **181 项本地工具及源码生成检查通过**，实验仅测了 4096。正式版本的其他评分尺寸
与新边界用例已由后续 `k128_step67.TdkZy5` 完成验收。全套增至 **57 项**，最新还有
Step 8 / 2048、Step 10 / 4096 两项性能失败。

### 当时的 Step 6、7 验收与 Step 10 对照（现已完成）

Step 10 的两个结果排除了“仅缩短 MMA 等待”和“将 TMEM x32 缩为 x16”这两项单独改动。
以下三项独立实验保持正式内核、四级流水线、两 consumer 和原计时/评分方法：

1. **TMEM load/wait 次数**：`tmem_load_64` 从八次 x32 改为四次 x64。
   若 load/wait 开销较大，应更快；代价是 FP32 临时寄存器增多，是否抵消收益由实测判断。
   256 列 FP16 暂存和四次 TMA store 保留。
2. **L2 顺序**：`l2_group_4` 只把调度器分组行数从 8 改为 4。
   每个 cluster tile 为 512×256；若跨 cluster 的数据复用受当前顺序限制，应降低耗时。
3. **cluster 任务分配**：`balanced_clusters` 保持原先每个 cluster 的最大任务数，
   用能覆盖任务的最小 cluster 数。4096 有 128 个输出 tile：原先 74 个 cluster 中
   54 个处理两块、20 个处理一块；变为 64 个各处理两块。预测减少不同工作量造成的
   尾部开销，但少用 20 个 SM 也可能抵消收益。该公式不特判矩阵尺寸：1024/2048/8192
   在 148 SM 上的 cluster 数仍为 8/32/74。

当时默认只测 Step 10 / 4096 的 baseline 与上述三项，共四个版本、五轮交错计时。
每个版本计时前后验算，并保存 builder/CUDA 差异、编译选项、资源和样本。
三个变体的 GPU 结果现已回传为 `k128_step67.TdkZy5`，见本文开头。
不继续放大 Step 10 的 K 或输出 SMEM：现有每 CTA 动态 SMEM 已为 230400 字节。

以下为 `ade5040` 的验收命令，无需重复：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/k128_step67.XXXXXX)
git log -3 --oneline

# Step 6/7：共 16 项，含两种 K 宽度的跨 tile 检查
uv run python -m pytest tests/test_step06.py tests/test_step07.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_step67.log"

uv run python -u benchmark.py --steps 6,7 --trials 5 \
  --csv "$tirx_run/step67.csv" --diagnostics-dir "$tirx_run/compiler_step67" \
  2>&1 | tee "$tirx_run/benchmark_step67.log"

# Step 10：四个独立版本，只测 4096
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/probe" 2>&1 | tee "$tirx_run/probe.log"
printf '结果目录：%s\n' "$tirx_run"
```

该目录及后续完整套件摘要已回传。probe 的 `SLOW` 会正常完成采集，
数值或编译失败则停止；`--output` 必须是新目录。Step 6、7 的已采用 K tile 实验
会拒绝重复应用，正式验证请用 pytest/benchmark。

## 前轮完整套件：step4_k128.6GZwy2

数据：[pytest_all.log](results_b300/step4_k128.6GZwy2/pytest_all.log)、
[pytest_step04.log](results_b300/step4_k128.6GZwy2/pytest_step04.log)、
[step04.csv](results_b300/step4_k128.6GZwy2/step04.csv)、
[compiler_step04/run.json](results_b300/step4_k128.6GZwy2/compiler_step04/run.json)。

**完整套件 46 passed / 7 failed，共 53 项；全部数值检查通过，7 项均为性能断言。**
Step 1–5、8、9 全部通过。运行版本 `c6435e4`，B300 / 148 SM / `sm_103a` /
TVM 0.26.0 / PyTorch 2.14.0+cu130 / NVRTC 13.0。内核 SHA256 为
`107641c0d559e52fb86b43aeab033d242e7f363d9c5875ee0784596ec0091984`。
记录虽为 dirty，内核、utils、benchmark 和 diagnostics 四个文件指纹与当前版本一致。
以下剩余项来自完整 pytest，未把单次结果当作多轮统计。

| Step | 方阵尺寸 | 实测 ms | 允许 ms | 超过门槛 |
|---|---:|---:|---:|---:|
| 6 | 2048 | 0.047647 | 0.045500 | 4.72% |
| 6 | 4096 | 0.315329 | 0.310700 | 1.49% |
| 6 | 8192 | 2.799590 | 2.785900 | 0.49% |
| 7 | 2048 | 0.045373 | 0.040300 | 12.59% |
| 7 | 4096 | 0.314241 | 0.299000 | 5.10% |
| 7 | 8192 | 2.777617 | 2.697500 | 2.97% |
| 10 | 4096 | 0.143087 | 0.139100 | 2.87% |

断言报错中的 `reference TFLOP/S` 是参考值，实际最低吞吐为参考值 / 1.30；
等价的时间上限是参考时间 × 1.30。表中已包含该容差。接近门槛的项仍是失败，
不能用浮动解释为已经通过，也不能把 Step 4、5 单独通过理解为全部十步完成性能验收。

### Step 4 正式验收已完成

| 方阵尺寸 | 单独 pytest ms | benchmark 中位数 ms | benchmark 最慢 ms | 允许 ms | 五轮结果 |
|---|---:|---:|---:|---:|---|
| 256 | 0.009523 | 0.009453 | 0.009598 | 0.018200 | 全部 PASS |
| 512 | 0.012595 | 0.012451 | 0.012548 | 0.019500 | 全部 PASS |
| 1024 | 0.018733 | 0.018606 | 0.018700 | 0.022100 | 全部 PASS |
| 2048 | 0.035138 | 0.034378 | 0.035036 | 0.067600 | 全部 PASS |

K=64、128、192、384 的矩形用例也通过，正式 `9db8b87` 的两条 K 分块路径得到验证。
随后全量套件再次通过全部 Step 4 与 Step 5 用例。

### 当时的独立对照：Step 6、7、10（现已完成）

当时生产内核的 Step 6、7、10 主体与 `RA0gpm` 诊断版本相同；本地生成的 4096 CUDA
主体与当时记录一致。选三步共同失败的 4096 作为反馈点，用 `probe_persistent.py`
逐项检验以下预测，不叠加变体：

1. **K 分块同步开销（Step 6、7）**：若每轮同步是主要开销，`k_tile_128` 将 K tile
   从 64 加宽至 128、轮数从 64 减至 32，应减少耗时。仍保持两级流水线，A/B SMEM 翻倍；
   K 不被 128 整除时保留 64 路径，phase 状态仍跨输出 tile 保留。
2. **MMA 完成等待（Step 6、7、10）**：若等待挂起提示拖慢进度，`mma_wait_64ns`
   应更快。仅改 MMA 发信号的 barrier：Step 6 的 `mma_bar`，Step 7/10 的
   `mma2tma`、`mma2ld`。保留 acquire、parity 和 retry，直到完成才继续；
   `tma2mma` 数据就绪与 `ld2mma` 累加器空闲等待不改。Step 5 的收益只是线索，
   不能直接推定三个常驻内核也会加速。
3. **Step 6 重复 fence 开销**：`final_fence` 将每个 K tile 的 MMA 完成后
   after/before fence 对移至 K 循环结束、交接写回线程之前。每轮 TMA acquire fence、
   MMA 完成等待和 SMEM 重用条件保留；phase 更新与下一次加载不变。
4. **写回开销**：Step 7 的 `epilogue_128` 将两个 64 列 TMA store 合成一个 128 列
   store，预测降低写回同步成本，代价是 Dsmem 增大。Step 10 的 `tmem_load_16`
   将 FP32 暂存从 32 降为 16，预测减轻寄存器压力，代价是 TMEM load/wait 次数翻倍；
   256 列 FP16 暂存和四次 TMA store 保留。旧 cubin 资源信息中的 stack 不代表
   已测到 spill 流量，是否有收益需结合新资源报告和耗时。

当时默认 Step 6、7 各四个版本（包含 baseline），Step 10 三个版本，总计 **11 个版本**。
全部版本先验算，再做五轮交错 CUDA event 计时，每轮预热 10 次、计时 30 次，计时后再次验算。
编译 hook 在计时前移除；原评分、容差和计时方法不变。每个版本保存 builder/source 差异及
SHA256、实际 CUDA、编译选项、cubin 和资源报告，用于确认实际测了什么。

当时没有修改生产内核。**153 项本地工具与源码生成检查通过**，包括所有 11 个变体的
TVM 0.26 CUDA 生成、12 份已有 CUDA 的选择性等待重放，以及短/奇数 K 的分块路径检查。
这些变体现已由 `persistent_probe.VB42kg` 回传 GPU 结果，见本文开头。

以下为 `f414130` 时使用的命令，无需重复：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/persistent_probe.XXXXXX)
git log -2 --oneline
uv run python -u probe_persistent.py --output "$tirx_run/probe" \
  2>&1 | tee "$tirx_run/probe.log"
printf '结果目录：%s\n' "$tirx_run"
```

回传 `probe.log`、`probe/summary.csv` 和整个目录即可分析。`SLOW` 是测量结果，
脚本会继续收集其他版本并正常结束；编译或数值错误则停止。无需先重跑全部 53 项。
可选 `--steps 6 7`、`--size 2048` 缩小或切换范围；`--variants mma_wait_64ns`
只测该实验和 baseline。`--output` 每次必须是尚不存在的新目录。
选出有效改动后按 step 分别提交，再用全部评分尺寸与边界用例验收。

## 前轮结果：mma64_step4.dh9ZC8

数据：[pytest_step05.log](results_b300/mma64_step4.dh9ZC8/pytest_step05.log)、
[step05.csv](results_b300/mma64_step4.dh9ZC8/step05.csv)、
[probe/summary.csv](results_b300/mma64_step4.dh9ZC8/probe/summary.csv)、
[probe/run.json](results_b300/mma64_step4.dh9ZC8/probe/run.json)。

**Step 5 正式验收 7 项全过；Step 4 / 1024 的 K tile 变体五轮全部达标。**
运行版本 `7ef4938`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
内核 SHA256 为 `69734d3c4beaac5a9797fe6875dfe4cc3d9fd19c0cb13c9094f3ca884396346a`。
虽然记录为 dirty，五个 Python 文件指纹均与该提交一致；各实验实际 CUDA 及 builder
指纹也与记录一致，八次编译选项相同。

### Step 5：正式实现已通过本轮验收

| 方阵尺寸 | pytest ms | benchmark 中位数 ms | benchmark 最慢 ms | 允许 ms | 五轮结果 |
|---|---|---|---|---|---|
| 512 | 0.010601 | 0.010418 | 0.010558 | 0.016900 | 全部 PASS |
| 1024 | 0.014682 | 0.014597 | 0.014805 | 0.015600 | 全部 PASS |
| 2048 | 0.032284 | 0.031540 | 0.031948 | 0.042900 | 全部 PASS |
| 4096 | 0.208406 | 0.208441 | 0.208742 | 0.353600 | 全部 PASS |

K=64、192、320 的三项边界检查也通过。1024 的最慢一轮仍低于门槛 **5.09%**；
2048 / 4096 未观察到性能回退。该结果验证了 `9cfd9f3` 的正式 MMA 等待实现。
Step 5 / 4096 为 659.37 TFLOP/s、cuBLAS 吞吐的约 61%，通过作业门槛不代表超过 cuBLAS。

### Step 4：K tile 有效，另外两项无收益

| 变体 | 中位数 ms | 最慢 ms | 配对加速比 | 寄存器数 | 五轮结果 |
|---|---|---|---|---|---|
| baseline | 0.022709 | 0.022956 | 1.000× | 164 | 全部 SLOW |
| **k_tile_128** | **0.018613** | **0.018631** | **1.220×** | 164 | **全部 PASS** |
| tmem_load_64 | 0.022721 | 0.022766 | 1.000× | 126 | 全部 SLOW |
| unroll_k | 0.022711 | 0.022850 | 1.000× | 164 | 全部 SLOW |

四个版本均通过初始数值检查及每轮输出重用检查。`k_tile_128` 耗时减少 **18.04%**，
最慢一轮低于 0.022100 ms 门槛 **15.69%**。增大 K tile 将串行加载/计算的同步轮数
从 16 降到 8，同时增加每轮数据量并重新生成 TMA 映射；结果支持减少这部分开销的方向，
不能仅凭这组对照拆分各个硬件延迟的贡献。

`tmem_load_64` 将寄存器数从 164 降到 126，耗时却没有改善；显式展开也无收益，因此均不采用。
所有资源报告的 STACK/LOCAL 均为 0。`SHARED:1024` 仅是静态部分，K tile 加宽会增加
动态 SMEM；不能将其理解为共享内存用量不变。日志只有 CUDA 头文件弃用和未使用符号警告。

### 当时的 Step 4 采用与验收命令（现已完成）

`9db8b87` 将 Step 4 的 `BLK_K` 改为：K 能被 128 整除时取 128，否则取 64。
继续支持全部正的 K%64=0 形状。单组 A/B SMEM、完成等待、phase 和 FP16 写回路径保留。
1024 生成的 CUDA 主体与实测有效变体逐字一致；K=64、192 的生成主体与原实现一致。
新增 K=64、128、192、384 的矩形 GPU 用例，覆盖两条路径各一轮/三轮 barrier phase。

当时 **113 项本地工具和源码生成检查通过**，其中包含新路径的 SM100a/SM103a lowering。
Step 4 正式版本的其他评分尺寸和新增边界用例，已由后续 `step4_k128.6GZwy2` 全部验证。
旧 probe 默认只测 baseline，并拒绝重复应用已采用的 K tile 改动。

以下为当时的验收命令，结果现已回传：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step4_k128.XXXXXX)
git log -2 --oneline

# Step 4：4 个评分尺寸 + 4 个边界用例
uv run python -m pytest tests/test_step04.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_step04.log"

uv run python -u benchmark.py --steps 4 --trials 5 \
  --csv "$tirx_run/step04.csv" --diagnostics-dir "$tirx_run/compiler_step04" \
  2>&1 | tee "$tirx_run/benchmark_step04.log"
printf '结果目录：%s\n' "$tirx_run"
```

随后执行了完整套件，更新 Step 6–10 的剩余性能项：

```bash
# 当前为 53 项；比此前的 49 项新增 4 个 Step 4 边界用例
uv run python -m pytest tests/ -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_all.log"
```

本节当时只有 Step 5 验收和 Step 4 单个形状的实验结果；最新完整验收结论见本文开头。

## 前轮等待对照：wait1024.UP24Tv

数据：[summary.csv](results_b300/wait1024.UP24Tv/probe/summary.csv)、
[samples.json](results_b300/wait1024.UP24Tv/probe/samples.json)、
[run.json](results_b300/wait1024.UP24Tv/probe/run.json)。

**没有整体回退：Step 5 找到了稳定有效的 MMA 等待调整，Step 4 仍未达标。**
运行版本 `c853534`，内核 SHA256 与前一轮 `early_release.dz3roD` 相同。
记录虽为 dirty，内核和工具文件的指纹均与该提交一致。
10 个独立版本全部通过初始数值验证及计时后的输出重用验证。

| Step / 1024 | 变体 | 中位数 ms | 配对加速比 | 五轮结果 |
|---|---|---|---|---|
| 4 | baseline | 0.022713 | 1.000× | 全部 SLOW |
| 4 | 两类等待 64 ns | 0.022605 | 1.005× | 全部 SLOW |
| 4 | 仅 TMA 64 ns | 0.022688 | 1.002× | 全部 SLOW |
| 4 | 仅 MMA 64 ns | 0.022699 | 1.001× | 全部 SLOW |
| 4 | 持续轮询 | 0.023085 | 0.991× | 全部 SLOW |
| 5 | baseline | 0.016515 | 1.000× | 全部 SLOW |
| 5 | 两类等待 64 ns | 0.014947 | 1.105× | 全部 PASS |
| 5 | 仅 TMA 64 ns | 0.016495 | 1.000× | 全部 SLOW |
| 5 | **仅 MMA 64 ns** | **0.014513** | **1.137×** | **全部 PASS** |
| 5 | 持续轮询 | 0.017083 | 0.967× | 全部 SLOW |

Step 5 的耗时中位数减少 **12.12%**，最慢一轮 0.014854 ms，仍低于 0.015600 ms 门槛
约 4.78%。这将收益定位到 MMA completion 等待路径；不能据此断言具体的硬件 stall 原因。
所有版本的编译选项相同，cubin 均为 `REG:164 STACK:0 SHARED:1024 LOCAL:0`。
Step 4 的四种等待实验都未解决问题；其中轮询变慢。实验之间独立，不会逐项叠加到正式内核。

### Step 5 已采用实测改动

`9cfd9f3` 只调整 Step 5 的 MMA 等待。通过 `T.cuda.func_call` 嵌入局部 helper，
保留测过的 PTX retry loop、parity 和默认 acquire.cta 语义；TMA 等待保持原样。
64 ns 是挂起时间提示，等待仍须反复检查直到 MMA 完成，不能超时后继续使用未完成结果。
文件保持自包含，不覆盖 TVM 全局 codegen，也不依赖 probe 的编译补丁。

源码对照检查在修改前失败、修改后通过：生成的 1024 CUDA 主体和两个 wait helper
与实测版本一致（仅新 helper 名不同）。该提交的 **98 项本地检查通过**。
当时仍需正式 GPU 复测；后续 `mma64_step4.dh9ZC8` 已完成 Step 5 全部评分尺寸及短 K 验收。

### 当时的 Step 5 验收与 Step 4 独立对照（现已完成）

Step 4 / 1024 baseline 超过 0.022100 ms 门槛约 **0.613 µs / 2.77%**。
等待实验的收益不足，下一轮按以下可区分的预测分别测试，不组合变体：

1. 若每个 K tile 的串行 TMA/MMA 同步限制速度，`k_tile_128` 将 BLK_K 从 64 增到 128，
   把 16 轮降到 8 轮，应减少耗时。仍只有一组 A/B SMEM，加载和计算不重叠。
   代价是 A/B SMEM 用量翻倍；TMA 映射由 TVM 重新生成，byte count 同步从 32768 改为 65536。
2. 若写回时的寄存器存活量/调度限制速度，`tmem_load_64` 将 x128 TMEM 读取拆为两次 x64，
   分段转为 FP16，应减少耗时。仍使用完整 128 列 Dsmem 和一次 TMA store；多一次 TMEM wait
   也可能抵消收益。以实际 cubin 资源报告和耗时检验，不将无 spill 等同于无寄存器影响。
3. 若 K 循环控制或动态条件限制速度，`unroll_k` 仅增加显式展开提示，应优于 baseline。
   原来的 `no_k_unroll` 在 2048 更慢，但这不足以证明显式展开在 1024 有效。

新 probe 默认仅测 Step 4 的 baseline 和这三个独立变体，仍为 5 轮交错计时，
每轮预热 10 次、计时 30 次。builder 改动单独保存 `.py`、diff 和 SHA256；
所有版本都保存实际 CUDA、编译选项、cubin、资源报告和逐轮耗时。
Step 4 正式内核保持原样，原评分门槛和计时方法保持不变。
当时本地 **106 项工具及源码检查通过**；实测现已回传为 `mma64_step4.dh9ZC8`，见本文开头。

以下为 `7ef4938` 当时使用的命令，无需重复执行该 probe：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/mma64_step4.XXXXXX)
git log -3 --oneline

# Step 5：4 个评分尺寸及 K=64/192/320，共 7 项原有 GPU 用例
uv run python -m pytest tests/test_step05.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_step05.log"

# Step 5：确认各尺寸是否稳定，并保留正式编译产物
uv run python -u benchmark.py --steps 5 --trials 5 \
  --csv "$tirx_run/step05.csv" --diagnostics-dir "$tirx_run/compiler_step05" \
  2>&1 | tee "$tirx_run/benchmark_step05.log"

# Step 4：只测尚未解决的 1024，4 个版本
uv run python -u probe_step45.py --steps 4 --size 1024 \
  --output "$tirx_run/probe" 2>&1 | tee "$tirx_run/probe.log"
printf '结果目录：%s\n' "$tirx_run"
```

这轮只验证 Step 5 和定位 Step 4，不能据此认定当时的全部 49 项已达标。
旧等待 probe 保留供历史复现，但会拒绝对已采用该改动的 Step 5 重复应用。

## 上一轮正式复测：early_release.dz3roD

数据：[pytest.log](results_b300/early_release.dz3roD/pytest.log)、
[focus.csv](results_b300/early_release.dz3roD/focus.csv)、
[run.json](results_b300/early_release.dz3roD/compiler/run.json)。

**正式 Step 4、5 的 11 项测试为 9 passed / 2 failed，剩余失败都是 1024 的性能断言。**
11 项数值检查全部通过，包含 Step 5 的 K=64、192、320。
运行版本 `98636d3`，四个 Python 文件的指纹均与该版本一致；内核 SHA256 为
`bfa00feb11ee890da2b0e93d47532f8ef2ea7ddf1fcbc822b5343066626fca9c`。

| Step / 方阵尺寸 | pytest ms | benchmark 中位数 ms | 允许 ms | 五轮 benchmark |
|---|---|---|---|---|
| 4 / 1024 | 0.022789 | 0.022704 | 0.022100 | 全部 SLOW，超标 2.73% |
| 4 / 2048 | 0.043391 | 0.042995 | 0.067600 | 全部 PASS |
| 5 / 1024 | 0.016196 | 0.016495 | 0.015600 | 全部 SLOW，超标 5.74% |
| 5 / 2048 | 0.033616 | 0.033203 | 0.042900 | 全部 PASS |
| 5 / 4096 | 0.211488 | 0.212002 | 0.353600 | 全部 PASS |

其余 Step 4 / 256、512 及 Step 5 / 512 也全部通过。Step 5 / 4096 相对
`tmem128.OZkJOD` 的 0.362154 ms 降低 41.5%，说明提前释放分配许可也改善了更大的网格。
这轮 benchmark 共 6 PASS / 2 SLOW，不等同于全仓库 49 项验收。

### 为什么 1024 仍慢

128×128 输出 tile 在 1024 方阵上只产生 **64 个 CTA**，少于该 B300 的 **148 个 SM**；
2048 / 4096 则分别产生 256 / 1024 个 CTA。小网格从释放额外分配许可中能获得的并发收益有限，
与实测“1024 几乎未改善、大网格显著提速”一致。此处是网格规模分析，不是 profiler 的实际调度记录。
Step 4 / 1024 五轮最低 0.022646 ms，Step 5 / 1024 最低 0.016463 ms，都仍高于门槛，
不能靠取最小值或重跑认定通过。

两个 1024 cubin 均为 `REG:164 STACK:0 SHARED:1024 LOCAL:0`，仍无 spill 证据。
目前需要分别减少约 **0.604 µs / 0.895 µs** 的中位耗时。
优先验证等待路径，依据是上一轮 `wait_64ns` 在 2048 已带来可重复的 4.8%–6.7% 配对加速。
它当时没有与 early_release 组合，也没有测过 1024，不能直接断言在此处有效。

按以下顺序检验可区分的预测：

1. 若 MMA completion 的等待/恢复开销占主要差距，仅缩短 MMA 等待提示应接近两类等待都缩短的收益。
2. 若 TMA 数据就绪的等待/恢复开销更重要，仅缩短 TMA 等待提示应获得主要收益。
3. 若潜在挂起和恢复本身限制短循环，保持 acquire 语义的非阻塞 `test_wait` 轮询应优于当前 `try_wait`。
   若没有改善或变慢，该方向不采纳，再定位其他路径。

### 当时的 1024 等待实验（现已完成）

`c853534` 的 [probe_step45.py](probe_step45.py) 当时用正式内核作 baseline，
保留 early_release，并比较以下独立变体：

| 变体 | 相对当前内核的唯一改动 |
|---|---|
| `wait_64ns` | TMA、MMA 的 try_wait 挂起时间提示都改为 64 ns |
| `tma_wait_64ns` | 仅 TMA 的提示改为 64 ns |
| `mma_wait_64ns` | 仅 MMA 的提示改为 64 ns |
| `wait_poll` | 两类等待改为 test_wait 持续轮询到完成 |

所有变体保留 phase、完成条件、fence、原有 acquire.cta 语义及评分标准。
[CUDA 13.0 PTX 规范](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
规定 test_wait 为非阻塞检查，try_wait 可以挂起；未写 `.sem` / `.scope` 时默认 `.acquire.cta`。
缩短时间提示不会允许提前使用未完成的数据。

以下是当时的命令，结果已回传为 `wait1024.UP24Tv`，无需重复执行：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/wait1024.XXXXXX)
uv run python -u probe_step45.py --size 1024 --output "$tirx_run/probe" \
  2>&1 | tee "$tirx_run/probe.log"
printf '结果目录：%s\n' "$tirx_run"
```

默认两个 step 各 5 个版本，共 10 次编译，5 轮交错计时，每轮仍预热 10 次、计时 30 次。
各版本先验算，复用输出后再次验算；保存源码差异、实际 cubin、资源报告及逐轮耗时。
回传整个目录，重点看 `probe.log` 与 `probe/summary.csv`。
这轮实验已完成，Step 5 的 MMA 等待改动已采用，详见本文开头。

## 已完成对照：step45_probe.PS9CFi

数据：[probe.log](results_b300/step45_probe.PS9CFi/probe.log)、
[summary.csv](results_b300/step45_probe.PS9CFi/probe/summary.csv)、
[samples.json](results_b300/step45_probe.PS9CFi/probe/samples.json)、
[run.json](results_b300/step45_probe.PS9CFi/probe/run.json)。

**提前释放 TMEM 分配许可的独立对照显著提速，Step 4、5 的 2048 形状五轮全部达标。**
运行版本为 `683da59`，B300 / NVRTC 13.0 / TVM 0.26.0 / `sm_103a`。
内核、benchmark、诊断工具及 probe 的源码指纹都与该版本一致。8 个编译版本均先通过数值验证，
再完成 5 轮交错计时，每轮预热 10 次、计时 30 次，复用输出后再次验证。

| Step / 2048 | baseline ms | early_release ms | 配对加速比 | 允许 ms | 五轮结果 |
|---|---|---|---|---|---|
| 4 | 0.076641 | 0.043150 | 1.776× | 0.067600 | 全部 PASS |
| 5 | 0.045228 | 0.033455 | 1.352× | 0.042900 | 全部 PASS |

按耗时中位数计算，Step 4 减少 43.7%，Step 5 减少 26.0%。最慢一轮分别为
0.043157 / 0.033551 ms，仍比允许耗时低 36.2% / 21.8%；不是临界 PASS。
吞吐分别为 398.14 / 513.52 TFLOP/s。这里的加速比相对同轮 baseline，**不是 cuBLAS 对比**。

### 证据与结论

捕获的 `source_01.patch` 确认 early_release 只移动一次 `relinquish_alloc_permit`：
从 kernel 末尾移到唯一一次 `alloc` 之后，执行者仍为整个 warp 0，TMEM 的 `dealloc`
仍位于写回完成之后。两者含义不同：放弃后续分配权利不释放当前累加器。
编译参数相同，baseline 与 early_release 的资源报告均为
`REG:164 STACK:0 SHARED:1024 LOCAL:0`。

这验证了 **分配许可的释放时机是这两个慢项的重要性能因素**。提前释放允许其他 CTA
申请剩余 TMEM，与该收益一致；本轮没有 profiler，尚未直接测量 CTA 并发数或分配等待周期。
前一轮只把列数从 512 减至 128 而不改变释放时机，耗时几乎不变。

另两个独立实验没有解决慢项：

| 变体 | Step 4 ms / 配对加速比 | Step 5 ms / 配对加速比 | 决策 |
|---|---|---|---|
| 等待提示改为 64 ns | 0.071823 / 1.067× | 0.043155 / 1.048× | 均 SLOW，暂不采用 |
| 禁止外层 K/ring 展开 | 0.086410 / 0.886× | 0.047627 / 0.950× | 均变慢，不采用 |

禁止展开后 Step 4 的寄存器数从 164 降到 159，但性能变差，进一步说明不能仅按静态资源
大小判断优化效果。本轮没有测量组合变体，不能把各项收益相加。

### 正式实现与复测

- `142c601`：Step 4 在 alloc 后立即 relinquish。
- `43124ff`：Step 5 在 alloc 后立即 relinquish，双缓冲预取和 phase 逻辑保持原样。

两个提交均在 `gemm_kernels.py` 中直接实现，无需 probe 编译补丁。
新增的实测源码对照在修改前失败、修改后通过：2048 形状生成的 CUDA kernel 主体
与回传的 early_release 源码逐字一致。全部 **82 项本地工具及源码生成检查通过**，
包括 SM100/SM103 lowering、边界形状构建和编译回调恢复。
本机没有 NVIDIA GPU；该版本后续的评分尺寸与短 K 实测见 `early_release.dz3roD`，当时剩两个 1024 慢项。

以下为 `early_release.dz3roD` 已完成的正式内核复测命令：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/early_release.XXXXXX)
git log -4 --oneline

uv run python -m pytest tests/test_step04.py tests/test_step05.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest.log"

uv run python -u benchmark.py --steps 4,5 --trials 5 \
  --csv "$tirx_run/focus.csv" --diagnostics-dir "$tirx_run/compiler" \
  2>&1 | tee "$tirx_run/benchmark.log"
printf '结果目录：%s\n' "$tirx_run"
```

pytest 包含 Step 5 的 K=64、192、320，benchmark 覆盖 Step 4 的
256–2048 及 Step 5 的 512–4096。后续结果见本文开头，不能据此宣布全仓库 49 项通过。
Step 6–10 本轮未改动，后续仍需解决已记录的性能失败。

**不再重跑原 early_release 对照**：正式内核已包含该改动。工具会拒绝把相同变换重复应用，
避免把相同代码误当作一次新的对照。原实验如需复现，应使用 `683da59` 的完整版本；
最新版本的 Step 4 验收命令见本文开头。

## 前轮诊断：tmem128.OZkJOD

数据：[pytest.log](results_b300/tmem128.OZkJOD/pytest.log)、
[focus.csv](results_b300/tmem128.OZkJOD/focus.csv)、
[run.json](results_b300/tmem128.OZkJOD/compiler/run.json)。

**128 列 TMEM 分配已生效，但没有解决 Step 4、5 的主要性能失败。**
运行提交为 `e7d40d6`，包含 `eebcd07` / `2afab29`；虽记录 `dirty=True`，
四个 Python 文件的 SHA256 均与本地对应版本一致。内核 SHA256 为
`3fd407fc54aa98b6c1c2b9f38893fde231324e5ecb12b65a01f4266f62c12194`。
捕获的 CUDA 主体确认 alloc/dealloc 参数都是 128，不能把这轮结果归因于未同步代码。

11 项 pytest 中 **6 passed / 5 failed**，失败均来自性能断言；数值检查全部通过，
包括 Step 5 的 K=64、192、320。独立 benchmark 的 8 组为 4 PASS / 4 SLOW：

| Step / 方阵尺寸 | 调整前 RA0gpm ms | 调整后 ms | 允许 ms | 说明 |
|---|---|---|---|---|
| 4 / 1024 | 0.022671 | 0.022695 | 0.022100 | SLOW，超标 2.69% |
| 4 / 2048 | 0.076598 | 0.076465 | 0.067600 | SLOW，超标 13.11% |
| 5 / 1024 | 0.016368 | 0.015593 | 0.015600 | 临界 PASS；pytest 为 0.016660，仍失败 |
| 5 / 2048 | 0.045258 | 0.045344 | 0.042900 | SLOW，超标 5.70% |
| 5 / 4096 | 0.363007 | 0.362154 | 0.353600 | SLOW，超标 2.42% |

除 Step 5 / 1024 外，上述主要慢项前后变化只有约 0.2%。Step 5 / 1024 三轮为
0.015593、0.016407、0.015429 ms，最大/最小相差 6.34%，不能认定为稳定提速。

### 已排除与仍需验证的方向

八份实际 cubin 的 `cuobjdump` 结果均为 `REG:164 STACK:0 SHARED:1024 LOCAL:0`。
没有栈帧或 local memory 的证据，不应继续把 Step 4、5 归因为寄存器 spill。
`SHARED:1024` 只表示静态共享内存。相同源码经 TVM 0.26 lowering 后，host launch
另传入动态共享内存 **66,560 字节（Step 4，65 KiB）** 和
**99,328 字节（Step 5，97 KiB）**。这些数值不是实测 occupancy，不能由它们断言实际并发 CTA 数。

当时按以下顺序安排独立对照，每次只改变一个变量；结果见本文开头：

1. **TMEM 分配许可的持有时间**：当前直到退出前才 relinquish。如果它让其他 CTA 在分配时等待，
   把 relinquish 移到唯一一次 alloc 后应降低多 CTA 形状的耗时。保留原来的 dealloc 位置。
   [CUDA 13.0 PTX 规范](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit)
   要求 relinquish 后不得再次 alloc；本例满足。但规范本身不能证明此处存在跨 CTA 阻塞。
2. **barrier 等待方式**：TVM 默认 `try_wait` 使用 10,000,000 ns 的 suspension hint。
   若唤醒/调度成本影响短循环，把 hint 改为 64 ns 应改善耗时。它不是固定睡眠时间，
   也不是跳过等待的超时；实验仍循环到 barrier 完成为止，保留原有 acquire 语义。
3. **K 循环自动展开**：给 Step 4 的 `k`、Step 5 的外层 `ring` 加 `#pragma unroll 1`。
   如果 CUDA 编译器展开导致指令开销或取指压力，收紧代码应改善耗时。
   Step 5 两个 stage 的显式展开不变；单凭 cubin 大小不能确认指令缓存瓶颈。

### 已完成的最小对照（683da59）

新增 [probe_step45.py](probe_step45.py) 自动测试当前版及以上三个独立变体，默认只测两个
稳定慢项 Step 4 / 2048 与 Step 5 / 2048。它在编译回调中修改生成的 CUDA，
**不会修改 `gemm_kernels.py`、编译器选项、计时函数、数值容限或评分标准**。
匹配不到预期代码结构时直接报错，不会静默测试无效变体。

以下为 `683da59` 上已执行的历史命令；新版本请用上面的正式内核复测命令：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step45_probe.XXXXXX)
uv run python -u probe_step45.py --output "$tirx_run/probe" \
  2>&1 | tee "$tirx_run/probe.log"
printf '结果目录：%s\n' "$tirx_run"
```

默认 8 次编译，每个版本先验证数值，再按变化顺序交错进行 5 轮计时，每轮仍预热 10 次、
计时 30 次。每轮复用输出后再次验证。`paired_speedup` 为各轮相对同轮 baseline 加速比的
中位数，大于 1 表示更快。`summary.csv` 保留原性能门槛；诊断脚本成功完成返回 0，
即使某个版本仍为 SLOW。数值错误、编译错误或 CUDA 异常会终止。

结果目录的重点文件为 `probe.log`、`probe/summary.csv`、`probe/samples.json`。
各版本目录包含 CUDA、cubin、NVRTC 日志、资源报告及 `source_01.patch`，可核实确切改动。
所有编译完成后才开始计时，编译钩子在计时前恢复。只验证该形状不能代替全套验收。

当时本地仅检查了实际 TVM 生成代码的变换、编译回调恢复和统计逻辑。
现已收到 `step45_probe.PS9CFi` 的 B300 结果并采用 early_release，详见本文开头。

## 前轮诊断：b300_diag.RA0gpm

完整数据：[benchmark.log](results_b300/b300_diag.RA0gpm/benchmark.log)、
[focus.csv](results_b300/b300_diag.RA0gpm/focus.csv)、
[run.json](results_b300/b300_diag.RA0gpm/compiler/run.json)。

该轮运行 `a8eabc6`，工作区干净。内核 SHA256 为
`ddada98d73674172edc97f14fd596eeab7181242b0b8885802174570f2d2f7b8`，
与前轮 `b300_diag.xP7WOw` 一致；两份共有的 13 个 cubin 也逐字节相同。
因此这是同版内核的重复测量，**尚未包含 `eebcd07` / `2afab29` 的 TMEM 分配调整**。

24 组数值检查通过，性能 **12 PASS / 12 SLOW**。这不是完整 49 项 pytest 验收。
Step 8 四组全过；Step 10 的 8192 达到 1259.36 TFLOP/s，为同轮 cuBLAS 吞吐的 97.60%。

| Step | 未达标尺寸：相对允许耗时超标 |
|---|---|
| 4 | 1024：2.58%；2048：13.31% |
| 5 | 1024：4.92%；2048：5.50%；4096：2.66% |
| 6 | 2048：4.04%；4096：2.67%；8192：0.62% |
| 7 | 2048：12.16%；4096：4.77%；8192：3.13% |
| 8 | 无 |
| 10 | 4096：1.80% |

12 个慢项每轮都慢，三轮间最大/最小耗时差均小于 1%。Step 4 / 2048 为 0.55%，
Step 7 / 2048 为 0.76%，显著低于它们 12–13% 的性能缺口。
Step 6 / 2048 稳定在 0.047338 ms，确认此前 0.0616 ms 回退已恢复。

### 编译资源证据

24 份 CUDA 源码、cubin、NVRTC 日志及参数均齐全，编译器是 NVRTC 13.0、目标 `sm_103a`。
参数包含 `--ptxas-options=-v`，但这版 NVRTC 日志没有寄存器/spill 统计，仅有 CUDA 头文件弃用、
生成代码未使用变量/函数的警告。这些警告本身不是性能瓶颈的证据。
此前文档将日志描述为一定包含 ptxas 资源统计不准确，已修正。

从 cubin 的 ELF `.nv.info` 元数据提取结果如下；逐形状数据见
[cubin_metadata.csv](results_b300/b300_diag.RA0gpm/cubin_metadata.csv)。
提取字段为 `EIATTR_REGCOUNT`（0x2f04）、`EIATTR_FRAME_SIZE`（0x1104）和
`EIATTR_MIN_STACK_SIZE`（0x1204），按符号表映射到 `kernel_kernel`。

| Step / 尺寸 | 每线程寄存器数 | 栈帧字节数 |
|---|---|---|
| 4、5 全部 | 164 | 0 |
| 6 / 1024 | 164 | 0 |
| 6 / 2048、4096、8192 | 150 | 0 |
| 7 / 1024 | 106 | 0 |
| 7 / 2048、4096、8192 | 128 | 8 |
| 8 全部 | 106 | 0 |
| 10 / 1024、2048 | 168 | 96 |
| 10 / 4096、8192 | 168 | 32 |

Step 7、10 存在非零栈帧，值得继续检查 local memory 访问和寄存器分配。
**栈帧大小不等于动态 spill 流量，不能单凭它认定瓶颈**。还需 SASS 或 profiler 区分编译器溢出、
局部数组和其他栈用途。后续诊断在 `cuobjdump` 可用时自动从实际二进制生成 `module_01.resources.txt`。

### 当时的候选：128 列 TMEM

Step 4、5 为空间网格，每 CTA 只使用 128 列累加器，却分配了整块 512 列 TMEM。
PTX 规定 `tcgen05.alloc` 在资源不足时阻塞；超额分配限制同一 SM 上其他 CTA 的并行执行。
`eebcd07`（Step 4）和 `2afab29`（Step 5）分别将分配、逻辑视图和释放改为 128 列，
保持各 step 的流水线深度、MMA、写回及同步顺序。
生成 kernel 主体对比确认仅 alloc/dealloc 两个参数不同；最新 `tmem128.OZkJOD` 已完成复测，
数值通过但主要耗时未改善，详见本文开头。
Step 6–10 继续保留当前版本，等待进一步运行时证据。

以下是该轮已执行的命令记录，无需再次重复这一轮：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results
set -o pipefail
tirx_run=$(mktemp -d results/tmem128.XXXXXX)
git log -5 --oneline
uv run python -m pytest tests/test_step04.py tests/test_step05.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest.log"
uv run python -u benchmark.py --steps 4,5 --trials 3 \
  --csv "$tirx_run/focus.csv" --diagnostics-dir "$tirx_run/compiler" \
  2>&1 | tee "$tirx_run/benchmark.log"
printf '结果目录：%s\n' "$tirx_run"
```

已有 Step 7、10 二进制无需重新运行 benchmark，即可在 CUDA Toolkit 机器上导出资源与 SASS：

```bash
tirx_previous=results/b300_diag.RA0gpm/compiler
cuobjdump --dump-resource-usage "$tirx_previous/step07_2048_2048_2048/module_01.cubin"
cuobjdump --dump-sass "$tirx_previous/step07_2048_2048_2048/module_01.cubin" \
  > "$tirx_previous/step07_2048_2048_2048/module_01.sass.txt"
cuobjdump --dump-resource-usage "$tirx_previous/step10_4096_4096_4096/module_01.cubin"
cuobjdump --dump-sass "$tirx_previous/step10_4096_4096_4096/module_01.cubin" \
  > "$tirx_previous/step10_4096_4096_4096/module_01.sass.txt"
```

如果工具不在 PATH，可使用 `/usr/local/cuda/bin/cuobjdump`。仓库中保存的路径前缀是
`results_b300/`，服务器原始结果路径前缀是 `results/`，按实际目录调整。

## 完整 pytest 实测：dfc9065

以下来自用户回传日志 `pytest3_1.log`，不是本机 GPU 测量。
用户确认服务器提交为 `dfc9065d02614bfde96d47742b7d57466db7fa7c`。

- GPU：NVIDIA B300 SXM6 AC，148 SM。
- Python：3.12.13；TVM：0.26.0；PyTorch：2.14.0+cu130；CUDA：13.0。
- pytest：**36 passed、13 failed**，共 49 项，70.04 秒。
- 49 项数值检查全部通过；13 个失败均为 `Submission too slow`。
- 12 个额外边界用例全部通过，包含短 K、奇数 K tile、矩形与常驻 CTA 重用。
- Step 1–3、9 全部通过。Step 8 / 4096 略超门槛，Step 10 仅 4096 未达标。

百分比按 `实测耗时 / 允许耗时 - 1` 计算。允许耗时已包含原作业 30% 容差；
不能直接用报错中的完整 reference TFLOP/s 当作最低及格线。

| Step | 方阵尺寸 | 实测 ms | 允许 ms | 超标幅度 |
|---|---|---|---|---|
| 4 | 1024 | 0.022846 | 0.022100 | 3.38% |
| 4 | 2048 | 0.076671 | 0.067600 | 13.42% |
| 5 | 1024 | 0.016352 | 0.015600 | 4.82% |
| 5 | 2048 | 0.045422 | 0.042900 | 5.88% |
| 5 | 4096 | 0.362940 | 0.353600 | 2.64% |
| 6 | 2048 | 0.047595 | 0.045500 | 4.60% |
| 6 | 4096 | 0.322978 | 0.310700 | 3.95% |
| 6 | 8192 | 2.802924 | 2.785900 | 0.61% |
| 7 | 2048 | 0.045601 | 0.040300 | 13.15% |
| 7 | 4096 | 0.313424 | 0.299000 | 4.82% |
| 7 | 8192 | 2.783132 | 2.697500 | 3.17% |
| 8 | 4096 | 0.172100 | 0.171600 | 0.29% |
| 10 | 4096 | 0.141857 | 0.139100 | 1.98% |

Step 10 / 8192 为 **0.872906 ms，1259.60 TFLOP/s**，通过原评分门槛。
此前相近版本 benchmark 为 0.874757 ms，对照同次 cuBLAS 0.852076 ms，约为其 97% 吞吐。
这不代表所有步骤已达标，也不等同于硬件峰值利用率。

## 回退结果与迭代历史

| 版本 | 用户回传结果 | 说明 |
|---|---|---|
| TVM 0.26 迁移版（本地 `488b211`） | 31 passed / 18 failed | 日志未记录服务器 hash；Step 6 / 2048 为 0.047766 ms。 |
| 第一轮性能改动（本地到 `7f4ea10`） | 37 passed / 12 failed | 日志未记录服务器 hash；Step 8、9 全过，但 Step 6 / 2048 退到 0.061570 ms。 |
| 用户确认 `dfc9065` | 36 passed / 13 failed | Step 6 完整回退到迁移版函数；2048 恢复为 0.047595 ms。 |

Step 6 的回退让 2048 耗时减少 **22.7%**，回到原始基线范围。这验证了上轮
`b7236ab` 改动引入回退；尚不能据此确定是寄存器分配、指令调度还是其他编译效应。
失败数 12→13 也不能表示全部性能变差：Step 6 / 8192 和 Step 8 / 4096 本来就在门槛附近，
本轮变为失败，而 Step 10 / 2048 变为通过。

内核改动仍按每个 step 一个 commit。没有修改数值容限、参考时间、性能容差或原评分计时方式。

| Step | Commit | 改动及观测 |
|---|---|---|
| 4 | `a675a11` | 写回交接 fence 移出 K 循环；实测无明显改善，仍需诊断。 |
| 5 | `c02b55b` | 两个 stage 展开，TMA phase 按 ring 奇偶计算；2048 有改善，仍未达标。 |
| 6 | `b7236ab` → `dfc9065` | 撤销引起 2048 回退的 stage 展开和 fence 移动，恢复完整基线函数。 |
| 7 | `d2eef01` | TMEM 每次读取 32 列；限制空闲 CTA。仍有性能失败。 |
| 8 | `4d7834c` | 与 Step 7 一致的读取/网格调整，保留四级流水线。4096 接近门槛。 |
| 9 | `ccd7b6d` | 每 CTA 的 TMEM load/wait 从 32 次降至 8 次；限制空闲 cluster。最新全过。 |
| 10 | `7f4ea10` | 每个 consumer 的 TMEM load/wait 从 32 次降至 8 次；限制空闲 cluster。4096 仍慢约 2%。 |

## 通用流程：同时拿到耗时与实际编译产物

同步工具提交 `f543ce9`（或其后续版本）即可，不需要重装依赖。一次仅运行一个 GPU benchmark，保留 Slurm
设置的 `CUDA_VISIBLE_DEVICES`。下一轮重点是区分编译资源问题与流水线等待，继续修改内核前先取得证据。

```bash
cd ~/assignment-tirx-gemm
mkdir -p results
set -o pipefail
tirx_run=$(mktemp -d results/b300_diag.XXXXXX)
nvidia-smi > "$tirx_run/gpu_before.txt"

uv run python -u benchmark.py --steps 4,5,6,7,8,10 --trials 3 \
  --diagnostics-dir "$tirx_run/compiler" --csv "$tirx_run/focus.csv" \
  2>&1 | tee "$tirx_run/benchmark.log"

nvidia-smi > "$tirx_run/gpu_after.txt"
printf '结果目录：%s\n' "$tirx_run"
```

默认使用这些 step 各自的评分形状，共 24 组；每组仍使用 10 次预热、30 次计时。
`--trials 3` 报告三轮中位数，并把每轮结果保存到 CSV。`SLOW` 仍会让 benchmark 返回 1；
数值错误或 CUDA 异常会立即终止，保留已完成的 CSV 行和编译产物。

日志自动打印 commit、dirty 状态、内核 SHA256、目标架构、编译模式和相关环境参数。
`compiler/run.json` 记录完整运行配置。每个形状的目录中：

- `module_01.cu`：交给 TVM 原编译回调的 CUDA 源码。
- `module_01.cubin`：默认 NVRTC 编译实际返回的二进制；NVCC 模式为 `.fatbin`。
- `nvrtc_01.options.json`：传入 NVRTC 的实际参数。
- `nvrtc_01.log`：NVRTC 编译日志；部分版本仅返回前端警告，不保证有 ptxas 资源统计。
- `module_01.resources.txt`：`cuobjdump` 从实际二进制读取的资源用量；工具缺失/失败时写明原因。
- `nvrtc_version.json`：NVRTC 编译器版本；它不必等于 PyTorch 显示的 CUDA 版本。
- `capture.json`：编译回调次数。若 `modules` 为 0，则本轮未捕获编译，不能判断无溢出。

TVM 0.26 原本不打印成功的 NVRTC 编译日志。采集钩子只在编译和第一次调用期间启用，
复用原编译回调与全部参数，在正确性验证、预热、计时前恢复。它不会换编译器或改变寄存器预算。
工具覆盖调用透明性、异常恢复、源码指纹、资源导出和 TIR/CUDA 源码构建检查（共 63 项），
本机无 NVIDIA GPU，实际 NVRTC/运行采集仍需服务器验证。

先回传 `benchmark.log` 与这些资源日志即可：

```bash
grep -RniE --include='*.resources.txt' 'REG:|STACK:|LOCAL:|SHARED:|unavailable' "$tirx_run/compiler"
```

若没有 `rg`，可直接读取单个文件，例如：

```bash
cat "$tirx_run/compiler/step07_2048_2048_2048/nvrtc_01.log"
cat "$tirx_run/compiler/step10_4096_4096_4096/nvrtc_01.log"
```

## 可选：已有 Nsight Compute 时定位 Step 7

如果服务器已装支持 B300 的 `ncu`，以下仅抓一个 Step 7 内核 launch。
该命令单独运行；profiling 会扰动 CUDA event 时间，输出的 PASS/SLOW 不用于评分。

```bash
ncu --target-processes all --kernel-name 'regex:kernel_kernel' \
  --launch-skip 1 --launch-count 1 --set full \
  --export "$tirx_run/ncu_step07_2048" \
  uv run python -u benchmark.py --steps 7 --sizes 2048 \
    --warmup 0 --repeat 1 --trials 1
```

查看 memory workload、warp stall、occupancy、Tensor Core 利用率，再决定调整等待、调度或寄存器布局。
若计数器权限不足，先使用上面的 NVRTC 资源日志；本轮诊断不要求安装 profiler 或修改机器权限。

## 最终验收

后续内核修改完成后，仍须完整运行原测试：

```bash
uv run python -m pytest tests/ -vs --tb=short 2>&1 | tee results/pytest_next.log
uv run python -u benchmark.py --steps all --trials 3 \
  --csv results/all_steps_next.csv 2>&1 | tee results/benchmark_next.log
```

只有完整 GPU 测试实际通过，才能称为全部达标；源码生成通过不代表性能通过。
