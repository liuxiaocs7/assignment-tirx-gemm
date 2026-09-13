# B300 验证记录与性能诊断

## 最新诊断：step45_probe.PS9CFi

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
本机没有 NVIDIA GPU，正式版本在其他评分尺寸和短 K 上仍需上机验证。

同步以上提交后，直接跑正式内核的 11 项测试及 8 组评分形状：

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

回传整个新目录。pytest 包含 Step 5 的 K=64、192、320，benchmark 覆盖 Step 4 的
256–2048 及 Step 5 的 512–4096。这里只宣称 2048 对照已达标，不能据此宣布全仓库 49 项通过。
Step 6–10 本轮未改动，后续仍需解决已记录的性能失败。

**不再重跑原 probe**：正式内核已包含 early_release。工具会拒绝把相同变换重复应用，
避免把相同代码误当作一次新的对照。原实验如需复现，应使用 `683da59` 的完整版本。

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
所有编译完成后才开始计时，编译钩子在计时前恢复。只验证该形状不能代替最终 49 项验收。

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
