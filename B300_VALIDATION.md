# B300 验证记录与性能诊断

## 下一轮实验：两个 consumer 沿 N 排列，共享 A

五级输入环的独立复测仍有 3/7 超线（实测见下节）。下一轮优先改变输入复用
方式，在相同计算量下减少 TMA 请求字节，再比较节省空间后的更深预取。
这两项已实现为独立 probe，**尚未在 B300 测量，不修改正式内核**。

当前两个 consumer 各做一个 256×128 MMA，沿 M 排列，共享 B，cluster 输出
512×128。新候选沿 N 排列、共享 A，cluster 输出 256×256：

| 配置 | 每 CTA 的每级输入 | 两 CTA 每级 TMA 请求 | 动态 SMEM | 直接对照 |
|---|---|---:|---:|---|
| 正式 baseline，K64 四级 | A0 + A1 + B | 81920 B | 181248 B | — |
| `tmem_input_depth5` | A0 + A1 + B | 81920 B | 222208 B | baseline |
| `tmem_share_a_depth5` | A + B0 + B1 | **65536 B** | **181248 B** | tmem_input_depth5 |
| `tmem_share_a_depth6` | A + B0 + B1 | **65536 B** | **214016 B** | tmem_share_a_depth5 |

每个 A 分块为 128×64 FP16，每个 B 分块为 64×64 FP16。把两份 A、一份 B
改成一份 A、两份 B，每级请求字节减少 **20%**；三条 TMA 指令的条数仍相同。
这是输入请求量变化，**不代表 DRAM 流量或耗时一定下降 20%**。输出 tile
由长方形变为正方形，也会改变 L2 访问关系和调度坐标，须用实测判断净收益。

候选保留每个 consumer 的 256×128 MMA、四条 K16 指令、K64 输入宽度、两槽
TMEM、EPI32 写回、384 线程和两 CTA 协作。在 4096 下仍有 256 个输出 tile，
64 个 cluster 各处理四个；tile 网格由 8×32 改成 16×16，L2 分组参数仍为 8。
输入 free barrier 仍等两个 consumer，累加器 free barrier 仍等两 CTA 共
256 个读回线程。共享 A 要等两个 consumer 用完后才能覆盖，不能提前释放。

实验仅作用于生产规则选出的窄 N 双 TMEM 槽工作量。单波次小网格和大面积宽 N
保留正式生成代码。新五级/六级在当前布局下都低于 232448 B opt-in 上限；
此前“六级超限”指的是 A0+A1+B 布局，减少输入存储后才可测试六级。

可检验的假设按优先级为：

1. 若输入搬运/槽位循环仍限制执行，共享 A 五级应比已测共享 B 五级有明确收益。
2. 若在途输入阶段仍不足，共享 A 六级应继续胜过共享 A 五级；否则保持五级。
3. 若均无收益，再检查 tile 顺序与全程利用率损失。现有 NCU 不足以证明其中
   任何一项为根因，不根据峰值百分比承诺速度。

本地编译检查覆盖 SM100a/SM103a、评分尺寸和布局选择边界，并检查实际 CUDA
输入/输出坐标、TMA/MMA 共享内存地址、每个 128×32 输出块的唯一覆盖、两套
独立 phase 和延迟读回下的 TMEM 所有权。GPU probe 在计时前校验 4096 和
`(4096,3072,K)` 的 K64/320/384/448；另外验证 `(1536,5376,320)` 的非完整
L2 分组。边界各两次，全部通过后才计时。数值和速度尚需 B300 验证。

本轮新增检查 **29 passed，29.28 s**；随后使用 TVM 0.26.0 运行完整
`tool_tests/`，结果 **614 passed，358.02 s**。正式 baseline 和原五级对照
生成的 CUDA 与此前 B300 实测版本一致；小网格/宽 N 的候选 fallback 与正式
CUDA 完全一致。这些是本地源码生成与协议检查，不替代 GPU 数值和计时验证。

运行命令见 [RUNNING.md](RUNNING.md#共享-a-与六级输入缓冲实验)。选择六级叶子
会自动加入正式 baseline、已测五级和共享 A 五级，总共四个版本。
只有在独立复测仍有足够收益、受影响形状验证通过后才采用。约 ≤0.135 ms
仍是希望获得余量的实验目标，原门槛保持 0.139100 ms。

## 最新复测：step10_depth5_recheck.sMMqkr，五级收益重复出现，但 3/7 超线

`4a05516` 补齐了在 `633497f` 上执行的独立两版本复测：
[summary.csv](results_b300/step10_depth5_recheck.sMMqkr/step10/summary.csv)、
[原始样本](results_b300/step10_depth5_recheck.sMMqkr/step10/samples.json)、
[运行元数据](results_b300/step10_depth5_recheck.sMMqkr/step10/run.json)、
[日志](results_b300/step10_depth5_recheck.sMMqkr/step10.log) 和
[退出码](results_b300/step10_depth5_recheck.sMMqkr/probe_exitcode.txt)。
仍使用原 warmup=10、repeat=30、trials=7、seed=0。两个评分构建及五级六种
短 K 边界均通过数值检查，六个边界各执行两次。

**五级的相对收益得到复测支持，但稳定达标的目标没有完成。**
汇总 PASS 只表示七轮中位数达标，不能解释为所有样本通过。

| 配置 | 中位数 ms | 最大值 ms | 达标样本 | 对 baseline 配对加速 |
|---|---:|---:|---:|---:|
| baseline，K64 四级 | 0.139428 | 0.139993 | **1/7** | 1.000000× |
| `tmem_input_depth5` | **0.138745** | **0.139429** | **4/7** | **1.002884×** |

五级第 1、2、5 轮分别为 0.139429、0.139207、0.139167 ms，超过 0.139100 ms
门槛约 **0.329 / 0.107 / 0.067 μs**。最慢样本超出门槛 **0.237%**。
每轮都比同轮 baseline 快，配对耗时差中位数约 **0.402 μs**；第 3 轮相差
3.503 μs，但不同版本顺序测量，这个较大差值不能作为常态收益。

两次输入环实验的五级累计 **14/14 个同轮样本数值上更快**，各次配对加速
约 1.004399× 和 1.002884×，达标样本累计 **11/14**；baseline 累计 **2/14**。
其中前次一轮只快 0.002 μs，不能把所有正差值都视为显著收益。
这些计数描述已记录样本，不代表总体通过率，也不把不同运行的绝对耗时合成
一个新评分结果。退出码 0 表示 probe 正常完成，超线样本仍需保留。

### 已核对产物与后续决策

七个运行源码哈希匹配 `633497f`。本次两个评分构建和六个边界构建的
builder、CUDA、cubin、NVRTC 参数及版本文件，全部与前次对应产物逐字节一致。
baseline 与五级仍都是 REG=112、STACK=0、LOCAL=0；实验并未编译回其他路径。
本次数据没有逐 kernel 时钟/功耗遥测，不能据运行波动断言降频或其他环境原因。

独立复测已经回答了原问题：**五级可以保留为后续实验的直接对照，但不能作为
稳定性修复进入最终验收。** 不再要求反复运行同一组 4096 测试来证明小收益；
正式内核仍保持原版本。下一项结构实验应同时与正式 baseline、五级比较，
重点扩大最慢样本余量，并验证受影响形状。若只决定采用这项小收益，也仍须
完成适用形状验证，并明确记录稳定性问题未解决。

两个两级方案已有明确退化，不继续推进；当前布局六级动态 SMEM 超上限，
不能简单加深。进一步方向需检查输入布局、tile 调度或全程利用率损失，
现有数据尚不能指定其中某一项为根因，也没有新的候选 GPU 成绩。
约 ≤0.135 ms 仍是实验目标，原评分门槛和计时规则不变。

本轮只更新结果文档；离线重建 summary 和八组编译产物核对均通过，未修改
正式内核、实验实现或测试。完整本地工具回归仍为此前的 585 passed，
不把文档核对算作新的 GPU 测试或性能修复。

## 前次实验：step10_input_ring.9SsKZM，五级略快，两级明显退化

`633497f` 补齐了 `dd90c57` 的输入环实验：
[summary.csv](results_b300/step10_input_ring.9SsKZM/step10/summary.csv)、
[原始样本](results_b300/step10_input_ring.9SsKZM/step10/samples.json)、
[运行元数据](results_b300/step10_input_ring.9SsKZM/step10/run.json)、
[完整日志](results_b300/step10_input_ring.9SsKZM/step10.log) 和
[退出码](results_b300/step10_input_ring.9SsKZM/probe_exitcode.txt)。
四个版本均完成 4096 数值检查；三个候选的六种矩形短 K 边界各验算两次，
共 18 个边界形状/候选组合、36 次 untimed launch，全部通过。

**五级是本轮唯一有收益的候选，但尚未达到稳定修复的程度，暂不采用。**
下表中配对加速是各轮 baseline/候选耗时比的中位数；门槛仍为 0.139100 ms。

| 变体 | 中位数 ms | 最大值 ms | 达标样本 | 对 baseline 配对加速 | REG / STACK / LOCAL | 动态 SMEM |
|---|---:|---:|---:|---:|---|---:|
| baseline，K64 四级 | 0.139238 | 0.139392 | **1/7** | 1.000000× | 112 / 0 / 0 | 181248 B |
| `tmem_input_depth2` | 0.224105 | 0.224755 | 0/7 | 0.620542× | 129 / 0 / 0 | 99328 B |
| `tmem_k128_depth2` | 0.166755 | 0.167441 | 0/7 | 0.834347× | 112 / 0 / 0 | 181248 B |
| `tmem_input_depth5` | **0.138660** | **0.138849** | **7/7** | **1.004399×** | 112 / 0 / 0 | 222208 B |

### 本轮能决定什么

- **停止推进这两个两级配置。** K64 两级耗时中位数比 baseline 增加约 61%，
  K128 两级增加约 20%。K128 对两级 K64 的直接配对加速为 1.343421×，但它
  仍明显慢于正式四级；不能把这个直接对照收益当作胜过 baseline。
- **保留五级做独立复测。** 七轮数值上都快于同轮 baseline，配对收益约 0.44%，
  耗时差中位数约 0.610 μs。但第 2 轮只差 **0.002 μs**，几乎持平；第 1 轮
  差 2.868 μs，后五轮差 0.320–0.719 μs，不应只拿首轮放大收益。
- **7/7 PASS 仍然余量不足。** 五级最慢离门槛仅 **0.251 μs / 0.180%**，
  中位数余量约 0.316%。这批 baseline 六轮超线，说明现有正式路径仍不稳定；
  五级在同轮状态下稍有改善，尚无独立复测或全步骤 benchmark 的稳定性证据。
- **退出码 0 表示实验完整运行。** 工具将 SLOW 作为测量结果保留；它不表示
  四个版本都达标。数值和编译错误才会中止这条 probe。

本轮否定了“减小输入深度即可改善”的候选，也说明 K64 四级与 K128 两级即使
输入容量相同，表现也不同；阶段数量、每级事务粒度和发射行为都可能参与。
没有候选角色/NCU 数据，不能把退化精确归因于寄存器、barrier 或显存。
五级 REG 与 baseline 相同，STACK/LOCAL 都为零，但二进制确实不同。
在当前布局下 K64 六级需要 **263168 B 动态 SMEM**，已经超过报告中的
232448 B 单 block opt-in 上限，不能直接继续加到六级。

### 产物核对与下一步

七个源码哈希匹配 `dd90c57`。baseline 的 CUDA/cubin、NVRTC 参数和版本文件
与首次正式验收逐字节一致，排除了本轮误跑旧内核；四个变体的 cubin 各不相同。
22 个评分/边界构建均使用同一正式 builder，builder/CUDA 哈希与记录一致，
没有额外 CUDA 文本变换。按原评分常量重建的四行 summary 与保存文件完全一致。

本地重放四个评分构建和五级六种边界，**10 份 SM103a CUDA 主体全部匹配实测
产物**。这是源码重放，不是本机 GPU 测量。本轮只更新文档，完整工具回归仍是
前轮的 585 passed；正式内核、实验实现、计时器和评分规则均未改动。

当时安排只测 **baseline 与 `tmem_input_depth5`**，保留同轮对照，不再把两项明显
退化的配置放进测量序列。沿用 `dd90c57` 即可运行，无需新实验代码；命令见
[RUNNING.md](RUNNING.md#当前路径输入环对照实验)，结果已回传并记录在本文开头。如果独立复测仍获益，再验证
受影响形状并进入正式采用/全步骤验收；若仍贴线或无稳定收益，继续寻找更大
收益的结构改动，不放宽门槛。约 ≤0.135 ms 仍是实验目标，尚未达到。

## 前次诊断：step10_current_profile.3cjntP，采集成功，余量问题仍未解决

`d9c1aa6` 补齐了在 `cede7ec` 上运行的当前 N128 / TMEM 双槽报告：
[角色元数据](results_b300/step10_current_profile.3cjntP/roles/run.json)、
[原始计时](results_b300/step10_current_profile.3cjntP/roles/timings.json)、
[逐 tile trace](results_b300/step10_current_profile.3cjntP/roles/trace.csv)、
[阶段汇总](results_b300/step10_current_profile.3cjntP/roles/stages.csv)、
[NCU 元数据](results_b300/step10_current_profile.3cjntP/hardware/run.json)、
[硬件指标](results_b300/step10_current_profile.3cjntP/hardware/metrics.csv) 和
[NCU 详情](results_b300/step10_current_profile.3cjntP/hardware/details.txt)。
NCU 的采集、导出、解析均成功，worker 在 profile 后再次验算输出。

源码指纹与 `cede7ec` 一致；角色 baseline 和 NCU 的 CUDA/cubin、NVRTC 参数及
版本文件，都与首次正式验收的 4096 产物逐字节一致。当前正式内核仍为
`d283549`，SHA256 为 `5515a04dfc3018bff2fe06e7f1f00681db4ee4ce8b376f4098f2348d85989b6c`。
离线核对了 21 份逐轮 trace，14,336 条记录均通过坐标、时间顺序和活动槽检查；
重建的 32 行阶段汇总与保存文件一致，NCU raw 重新解析也与 metrics 一致。

### 评分与诊断分开看

未经插桩的 baseline 七轮 **7/7 达标**，中位数 **0.138113 ms**，最大值
**0.138620 ms**；最慢样本离 0.139100 ms 仅 **0.480 μs / 0.345%**。
这次通过不能覆盖前次五轮全步骤 benchmark 中的两个 SLOW。三个插桩版本与
baseline 的同轮耗时比约 1.000–1.003，属于测量扰动，不能作为优化收益。

| 当前路径观测 | 结果 | 能支持的判断 |
|---|---:|---|
| MMA 等输入 / 等累加器复用 | 71.66% / **0.175%** | TMEM 交接等待很小，暂不优先增加累加器缓冲 |
| TMA 等输入槽释放 / 发射 | 81.44% / 9.50% | producer 自己也在等待 MMA 消费完成，不能只归因于显存搬运慢 |
| 写回等 MMA / 读回 / epilogue | 86.19% / 1.09% / 12.27% | 大部分时间在等结果，不支持把读回发射当主要瓶颈 |
| TC 活跃周期占 SM 活跃期 / 全时段 | **93.27% / 75.21%** | 活跃期利用率高，全程仍有利用率损失；不是精确的启动/尾部耗时分解 |
| L2 / DRAM 峰值吞吐占比 | **36.13% / 9.85%** | 本报告没有显示全局带宽饱和；不能排除延迟或局部资源限制 |
| L2 命中率 | 86.19% | 与多数数据由缓存服务一致 |
| 寄存器 / 栈 / LOCAL；动态 SMEM | 112 / 0 / 0；181248 B | NCU 的寄存器和 SMEM occupancy limit 均为 1 block/SM |

角色百分比按全部 trace 的阶段时长合计后除以角色总时长，不等同于逐行百分比
的简单平均。TMA/MMA 的 work 是发射时间；角色并行，百分比不能相加。
MMA handoff 中位数仅 64 ns、最大 96 ns，接近计时粒度与插桩开销。

256 个输出 tile 分给 64 个 cluster，每个四 tile；首 tile 起点偏差仅约
0.096–0.288 μs，没有明显证据支持直接回到 74 个 cluster。NCU 用 20 次 replay，
未固定 cache/clock；其中 152.608 μs 的 duration 不参加评分。14 份轮前后快照
均记录 SM 1095 MHz，但离散快照不能证明内核执行中频率恒定或排除降频。

### 当时的输入环实验设计（结果已回传，见本文开头）

当前数据支持优先检查输入环与 MMA 的耦合等待，尚未证明某个 barrier 是根因。
当时在 `probe_persistent.py` 准备以下独立候选，未改正式内核；GPU 结果现见本文开头：

| 变体 | 输入配置 | 直接对照 | 4096 动态 SMEM | 实验要回答的问题 |
|---|---|---|---:|---|
| `baseline` | K64 / 深度 4 | — | 181248 B | 当前正式路径 |
| `tmem_input_depth2` | K64 / 深度 2 | baseline | 99328 B | 减少在途输入是否反而改善执行，还是暴露更多延迟 |
| `tmem_k128_depth2` | K128 / 深度 2 | tmem_input_depth2 | 181248 B | 每 tile 同步迭代从 64 减为 32 是否获益 |
| `tmem_input_depth5` | K64 / 深度 5 | baseline | 222208 B | 增加预取容量能否减少气泡 |

K128/深度 2 与 baseline 的输入容量相同，但 K128 同时改变每级布局、TMA
事务粒度和同步频率，不能将收益简单归于某一因素。K128 在 K 不能被 128 整除时
回退 K64/深度 2。三个候选仅作用于正式规则选出的窄 N 双槽工作量，小网格单槽
和大面积宽 N 均保留当前代码。

本地检查覆盖 SM100a/SM103a 的 K64/128/192/256/320/384、4096 方阵、两级/五级
输入环及跨 tile 双槽复用。特别核对 K128 的三维 TMA 映射与 MMA 的共享内存
描述符地址一致，且小尺寸/宽 N 的 CUDA 与宿主 TMA 描述符逐字一致。
GPU probe 会先验算每个候选的评分形状，再将六个矩形短 K 边界各验算两次，
全部通过后交错计时。
源码生成与协议检查不能代替 GPU 数值和性能验证。

新增输入环检查 **28 passed**；完整本地工具/源码生成回归
**585 passed，338.64 s**。新运行命令已检查 bash/zsh 语法和管道失败退出码保存，
正式内核、计时器、数值容差及评分测试均未改变。

运行命令见 [RUNNING.md](RUNNING.md#当前路径输入环对照实验)。只有独立复测收益
明确、最慢样本有更大余量且受影响形状验证通过后才采用；≤0.135 ms 仍只是实验目标。

## 前次复测：ea69b2c，pytest 五轮全过，Step 10／4096 benchmark 仍有 2/5 超线

`ea69b2ce68052ade16eb7ba7f5f7370d98988c26` 只补充结果文件，内核仍是
`d283549` 的窄 N / TMEM 双缓冲实现。已逐文件读取五轮 pytest、五份全步骤
benchmark CSV，以及此前 `step10_tmem_adopt.OQ0WHS` 的原始样本和编译产物。
**当前结论是数值校验和五轮全量 pytest 均通过，但 Step 10／4096 的性能余量
仍不足以覆盖这批 benchmark 的波动，不能称为所有测量都稳定达标。**

### 五轮完整结果

五份 CSV 各含 37 个评分形状、每形状 `trials=1`；全部使用 warmup=10、
repeat=30、seed=0。即每个数值是一批 30 次调用的平均值，不是单次 kernel
时长，也不是七轮中位数。185 条记录中 183 PASS、2 SLOW，两个 SLOW 都在
Step 10／4096。其余 36 个形状五轮全部通过，所有已记录数值检查通过。

| 轮次 | 全量 pytest | pytest 的 Step 10／4096 ms | 全步骤 benchmark ms | 相对 0.139100 ms 门槛 | 同次 cuBLAS ms |
|---|---|---:|---:|---|---:|
| 1 | [57 passed，76.50 s](results_b300/pytest_d283549_1.log) | 0.137679 | [0.138265](results_b300/all_steps_d283549_1.csv) | PASS，余量 0.835 μs | 0.128082 |
| 2 | [57 passed，76.38 s](results_b300/pytest_d283549_2.log) | 0.138149 | [0.138953](results_b300/all_steps_d283549_2.csv) | PASS，余量 0.147 μs | 0.128479 |
| 3 | [57 passed，76.54 s](results_b300/pytest_d283549_3.log) | 0.137782 | [**0.139564**](results_b300/all_steps_d283549_3.csv) | **SLOW，超出 0.464 μs / 0.333%** | 0.129295 |
| 4 | [57 passed，77.59 s](results_b300/pytest_d283549_4.log) | 0.137723 | [0.138464](results_b300/all_steps_d283549_4.csv) | PASS，余量 0.636 μs | 0.128791 |
| 5 | [57 passed，77.22 s](results_b300/pytest_d283549_5.log) | 0.138217 | [**0.139290**](results_b300/all_steps_d283549_5.csv) | **SLOW，超出 0.190 μs / 0.136%** | 0.129090 |

这确认了“五轮 pytest 全过”的反馈，也确认 benchmark 仍会超线。2/5 是这批
样本的结果，不能外推为总体失败概率。pytest 和 benchmark 是不同进程/运行，
同编号不表示同时测量；不能把两者差值归为固定的工具开销。

### 已排除什么，尚不能判断什么

五份 CSV 全部 185 行的 `gemm_kernels.py`、`utils.py`、`benchmark.py` 和
`benchmark_diagnostics.py` SHA256 均与 `d283549` 一致，记录的目标架构、
GPU/SM 数、软件版本、NVRTC 和 ptxas 参数也相同：B300 / 148 SM / `sm_103a` /
TVM 0.26.0 / PyTorch 2.14.0+cu130 / CUDA 13.0 / register-usage-level=10。
`dirty=True` 没有表现为这四个关键源码的变化。

首次正式验收的四尺寸 CUDA、cubin、NVRTC 参数和版本文件，与历史四尺寸
probe 中各自被采用的版本逐字节一致。4096 是 N128 双槽，REG=112、STACK=0、
LOCAL=0；没有证据表明采用时编译回了旧宽 N 路径。**五次新全步骤 benchmark
没有保存各自的 cubin，不能把首次产物一致扩大为五次二进制均已核对。**

Step 10／4096 的五次 benchmark 极差约为最小值的 **0.940%**，同次 cuBLAS
极差约 **0.947%**；本内核相对 cuBLAS 的比值保持在 **0.9246–0.9301**。
这与运行状态共同变化的解释相容，但 cuBLAS 是顺序测量，且缺少同步的时钟、
功耗、温度和竞争任务记录，不能据此断言热降频。也不能仅凭耗时表认定某个
barrier 是瓶颈。

已有失败信号是原 `benchmark.py --steps all --trials 1` 路径的 SLOW；本机
通过读取真实 CSV 复核了门槛判断，没有 B300，未在本机重新运行 GPU 内核或
复现硬件状态。本轮不据离线分析修改同步协议或宣称性能已修复。

### 当时的诊断计划（当前报告已回传，见本文开头）

1. 对当前生产 **N128 / 双 TMEM 槽**采集一次角色诊断，观察四个 persistent
   tile 的输入等待、累加器交接和写回；已有 `profile_persistent.py` 支持当前
   builder，保留原 baseline 与诊断副本。源码操作保持检查与硬件采集入口测试
   已在本轮本地复核，GPU 计数器仍需在服务器采集。
2. 再采当前生产路径的 NCU 报告，核对 SM 活跃期和全程 Tensor Core 利用率、
   内存流量及资源限制。旧 `step10_hardware.Mosdpx` 属于宽 N 单槽，不能替代
   新路径报告。角色插桩和 NCU 的时长都不参与评分。
3. 若证据显示输入等待主导，才比较 K 宽度/输入深度；若累加器交接或 epilogue
   主导，才调整读回/写回重叠；若尾部工作不均衡主导，才修改 tile/cluster 调度。
   保留同轮对照和当前双槽结构，四尺寸验证后再采用。

目标仍是将 4096 的常态耗时推至约 **≤0.135 ms** 并检查最慢样本，给 0.139100 ms
门槛留下约 3% 余量；这是后续实验目标，不是已达到的成绩。增加 trials 或挑中位数
可以描述分布，不能抹去单轮 SLOW，更不能放宽原计时或评分标准。具体命令见
[RUNNING.md](RUNNING.md#step-104096-不稳定时采集当前路径)。

## 首次正式验收：step10_tmem_adopt.OQ0WHS，原始产物已核对

此前文档只能依据贴出的摘要；本次提交补齐了
[Step 10 pytest](results_b300/step10_tmem_adopt.OQ0WHS/pytest_step10.log)、
[全量第 1 轮](results_b300/step10_tmem_adopt.OQ0WHS/pytest_all_1.log)、
[全量第 2 轮](results_b300/step10_tmem_adopt.OQ0WHS/pytest_all_2.log)、
[七轮 CSV](results_b300/step10_tmem_adopt.OQ0WHS/step10.csv) 和
[编译元数据](results_b300/step10_tmem_adopt.OQ0WHS/compiler_step10/run.json)。
这两份首次全量日志与上述五轮复测日志不同，分别保留，不再仅用用户反馈描述。

| 尺寸 | 当前配置 | 单独 pytest ms | 首次全量第 1 轮 ms | 首次全量第 2 轮 ms | 七轮中位数 ms | 七轮最大值 ms |
|---|---|---:|---:|---:|---:|---:|
| 1024 | N128 / EPI32 / 单槽 | 0.018695 | 0.018734 | 0.018687 | 0.018547 | 0.018729 |
| 2048 | N128 / EPI32 / 单槽 | 0.027318 | 0.027557 | 0.027907 | 0.027124 | 0.027373 |
| 4096 | N128 / EPI32 / 双槽 | 0.138047 | 0.137666 | 0.137785 | 0.137610 | **0.138281** |
| 8192 | N256 / EPI64 / 单槽 | 0.869177 | 0.868961 | 0.866174 | 0.868908 | 0.870297 |

Step 10 **6 passed，13.19 s**；两次全量分别 **57 passed，76.33 / 76.43 s**。
七轮 CSV 的 28 个原始样本现在已核对，全部在各自门槛内。4096 最慢
0.138280535 ms，余量 **0.819 μs / 0.589%**，比仅看中位数的 1.071% 更小。
这解释了为什么首次通过不足以证明后续所有运行都稳定达标。

内核 SHA256 为
`5515a04dfc3018bff2fe06e7f1f00681db4ee4ce8b376f4098f2348d85989b6c`。
原第二轮全量含 Slurm time-limit 消息，但日志仍以完整 `57 passed` 结束；
没有单独的作业退出码。最新五轮日志均显示完整的通过摘要。

本机此前完整工具/源码生成检查为 **557 passed，299.28 s**；本轮对当前诊断
入口运行 `tool_tests/test_persistent_profile.py` 和 `tool_tests/test_hardware_profile.py`，
**63 passed，16.27 s**，覆盖角色源码生成、硬件采集流程及真实 NCU CSV 解析。
这不代表已执行新的 GPU profile。本轮保留内核、数值容差、门槛和 CUDA-event
计时。下文的“下一步”“待验收”保留历史上下文，
当前稳定性结论以本文顶部为准。

## 历史采用依据：step10_tmem_sizes.I9nGIJ

数据：[1024](results_b300/step10_tmem_sizes.I9nGIJ/step10_1024/summary.csv)、
[2048](results_b300/step10_tmem_sizes.I9nGIJ/step10_2048/summary.csv)、
[4096](results_b300/step10_tmem_sizes.I9nGIJ/step10_4096/summary.csv)、
[8192](results_b300/step10_tmem_sizes.I9nGIJ/step10_8192/summary.csv)。
测量版本 `0f23484`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
四个进程各自的七项源码指纹与该提交一致；56 份 builder/CUDA 编译前后指纹、
40 份边界验算记录（80 次非计时运行）核对通过。四版本×四尺寸×七轮，共 112 个
计时样本，数值检查全部通过。4096 四版本的 CUDA/cubin/编译参数与上次独立运行
`step10_tmem_double.DYMgCt` 逐字节相同。

| 尺寸 | baseline 中位数 ms | N128/EPI32 单缓冲 ms | N128 双缓冲 ms | 本次正式选择 |
|---|---:|---:|---:|---|
| 1024 | 0.027389 | **0.018566** | 0.018570 | N128/EPI32 单缓冲 |
| 2048 | 0.043305 | **0.027029** | 0.027035 | N128/EPI32 单缓冲 |
| 4096 | 0.138129 | 0.138861 | **0.137422** | N128/EPI32 双缓冲 |
| 8192 | **0.868543** | 0.909238 | 0.902522 | 原 N256/EPI64 单缓冲 |

1024/2048 单缓冲对 baseline 的同轮加速中位数为 **1.475045× / 1.606506×**，
中位耗时减少约 32.2% / 37.6%。这两个尺寸的窄网格分别只有 16 / 64 个输出 tile，
都能在 74 个 cluster 容量内一次分配，每个 cluster 没有后续 tile 可重叠，双缓冲
相对单缓冲仅快 1/7、3/7 轮，故使用单槽。收益主要来自窄 N，不归功于双槽。

4096 双缓冲本次同轮对 baseline 为 **1.004499×**，对直接单缓冲为 **1.009652×**，
再次七轮全部更快；两次独立运行累计 **14/14** 都更快。最慢 **0.137803 ms**，
低于 0.139100 ms 门槛约 **1.297 微秒 / 0.933%**。它改善了余量，但尚未达到
≤0.135 ms 的理想目标，正式全量中的稳定性仍需验证。

8192 双缓冲虽七轮都快于窄 N 单缓冲，仍七轮都慢于正式宽 N；中位耗时增加
**3.912%**，因此保留 N256。不能因为四尺寸双缓冲都 PASS 就统一替换全部尺寸。

### 采用范围与协议

`hgemm_v10` 保持一个自包含 builder、两个 MMA consumer 和 WG0/WG1 读回结构，
只在编译时按输出工作量选择配置：

- `M*N <= 4096²` 使用 N128/EPI32；更大的输出保留 N256/EPI64。
- 窄 N 的输出 tile 数超过 `SM_COUNT//2` 时用两个 TMEM 槽，否则用一个。
- 单槽沿用原 phase 推进位置；双槽在当前槽 ready commit / 完整读回释放之后推进，
  两个槽轮转后才翻转 phase。每槽仍等待两个 CTA 共 256 个读回线程完成。

该规则覆盖矩形输出，不按测试函数或精确评分 shape 分支。跨面积边界和波次边界
的形状也进行源码对照。B300 性能证据覆盖四个评分尺寸；任意其他矩形、K 长度或
SM 数的性能不由这些测量保证。本机 SM100a/SM103a 检查验证协议和生成行为。
输入四级 K64、B-first、均衡网格、512 列 TMEM 分配及最终 cluster sync 保留，
接口仍要求 M%512、N%256、K%64。Step 1–9、评分阈值、数值容差及 CUDA-event
计时未变。正式源码 SHA256 为
`5515a04dfc3018bff2fe06e7f1f00681db4ee4ce8b376f4098f2348d85989b6c`。

新增回归先在旧生产实现上复现与所选实测 CUDA 不同，再验证采用后的四尺寸 CUDA
主体逐字节一致；宿主侧 TMA 描述符也与实测 builder 生成值一致。边界覆盖短 K、
不完整输入 ring、首次/重复 TMEM 槽使用、单波次上沿及窄 N 面积上沿。profiling
的 tile 数和网格改为匹配正式选择；历史 probe 测试使用保存的旧 builder 重放，
避免新 baseline 混入旧实验。probe 默认只运行正式 baseline，并拒绝向新生产
builder 重复应用历史变体。
新增采用检查 **27 passed**；完整本地工具/源码生成回归 **557 passed，299.28 s**。

采用时仍需正式 Step 10、七轮 benchmark 及全量 pytest 验收；当时最新全量为
旧生产版本的 **56 passed / 1 failed**。后续 `d283549` 正式验收已完成，结果
见本文顶部。probe 验证和源码重放是采用依据，正式测试是后续独立证据。

## 前轮结果：step10_tmem_double.DYMgCt，双缓冲七轮均有收益，进入四尺寸验证

数据：[summary.csv](results_b300/step10_tmem_double.DYMgCt/step10/summary.csv)、
[samples.json](results_b300/step10_tmem_double.DYMgCt/step10/samples.json)、
[run.json](results_b300/step10_tmem_double.DYMgCt/step10/run.json)。
版本 `88b0b39`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹与该提交及当前文件一致；14 份 builder/CUDA 编译前后指纹全部匹配。
baseline 的 CUDA、cubin、NVRTC 参数与正式 B-first 产物完全一致；两个窄 N
单缓冲对照的这三项产物也与 `step10_n128.eVKFm2` 完全一致。
十份边界记录、合计 20 次非计时验算通过，七轮交错计时后的数值检查也全部通过。

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 同轮相对 baseline | 同轮相对直接对照 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.138114 | 0.138799 | 7/7 | 1.000000× | 1.000000× |
| n_tile_128 | 0.140251 | 0.140985 | 0/7 | 0.984498× | 0.984498× |
| n128_epi32 | 0.138810 | 0.139249 | 6/7 | 0.996069× | 1.011041×（对 n_tile_128） |
| n128_tmem_double_buffer | **0.137312** | **0.137650** | **7/7** | **1.005996×** | **1.011616×（对 n128_epi32）** |

双缓冲在 **全部七轮** 同时快于 baseline 和直接单缓冲对照，同轮加速中位数分别为
**0.600% / 1.162%**。单独缩 N、改 EPI32 仍七轮都慢于 baseline，因此有价值的
是完整双槽组合。最慢样本比 0.139100 ms 门槛低 **1.450 微秒 / 1.042%**；
本轮 baseline 最慢样本余量仅 0.301 微秒 / 0.216%。收益方向比前几轮一致，
但没有达到此前 ≤0.135 ms、约 3% 余量的理想目标，也尚无跨运行稳定性结论。

| 版本 | 寄存器 / thread | STACK / LOCAL | 动态 SMEM / CTA |
|---|---:|---:|---:|
| baseline | 167 | 0 / 0 | 230400 bytes |
| n_tile_128 | 105 | 0 / 0 | 197632 bytes |
| n128_epi32 | 112 | 0 / 0 | 181248 bytes |
| n128_tmem_double_buffer | 112 | 0 / 0 | 181248 bytes |

四份 SASS 均无 LDL/STL；双缓冲 ptxas 日志明确为 0 spill stores / loads。
新增四个 ready/free barrier 已出现在二进制初始化中，CUDA 使用随槽位改变的
TMEM 地址并在两槽轮转后翻转 phase。与直接对照相比，双槽未增加寄存器或 SMEM；
不能把 baseline 的 167→112 寄存器变化归为双缓冲本身的收益。
这些证据支持继续验证 accumulator 双槽，尚不能把具体等待认定为唯一瓶颈。

**下一步固定此候选，检查 1024、2048、4096、8192 四个评分尺寸。** 沿用 `88b0b39`
已有的 `--size` / `--variants` 即可，无需新实验代码或重采 NCU。四个进程顺序运行，
各七轮且保留完整对照链；其中 4096 同时构成一次独立复测。命令见
[RUNNING.md](RUNNING.md)。检查每个尺寸的正确性、原门槛、同轮收益及最慢样本，
尤其不能用 4096 的改善推断 8192 也会改善。

四尺寸证据支持后再将候选纳入正式内核，随后运行原 Step 10 / 全量 pytest 和
benchmark；当前直接运行 pytest 仍使用旧生产版本。本次只更新实测记录和验证步骤，
生产 SHA256 仍为 `3dfec9f00bda86d46f1664ca17ffe7af5f6095e78884cdc54bec956990c7bcf7`，
最新正式全量状态仍是 **56 passed / 1 failed**，不能称为已稳定全过。

## 前轮结果：step10_hardware.Mosdpx，采集成功，CSV 解析错误已修复

原始数据：[raw.csv](results_b300/step10_hardware.Mosdpx/profile/raw.csv)、
[worker.json](results_b300/step10_hardware.Mosdpx/profile/worker.json)、
[run.json](results_b300/step10_hardware.Mosdpx/profile/run.json)、
[ncu.log](results_b300/step10_hardware.Mosdpx/profile/ncu.log)。
恢复结果：[metrics.csv](results_b300/step10_hardware.Mosdpx/analysis/metrics.csv)、
[analysis.json](results_b300/step10_hardware.Mosdpx/analysis/analysis.json)。
采集版本 `880a361`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0，
Nsight Compute **2025.3.1.0**。五个源码指纹均匹配记录的提交；正式内核 CUDA、
cubin 和 NVRTC 参数与 B-first 验收产物逐字节一致。

### 错误发生在导出后的解析，已有采集结果有效

ncu 对 `kernel_kernel` 完成 **20 passes**，退出码为 0，worker 在采集后验算通过，
`.ncu-rep` 和 `raw.csv` 已保存。原 parser 误把 raw CSV 当作“每行一个 metric”的
长表。实际文件是 **806 列、3 行**：表头、单位行、单次内核结果。单位行的 ID、
Process ID、Kernel Name 均为空，被误算为第二个内核；因此触发错误的数量断言。
此前进程模拟测试只构造长表，遗漏了实际 NCU raw 页面格式。

用该真实文件新增回归测试，先复现相同 RuntimeError，再修改 parser：识别宽表
单位行并分离 launch 信息和指标，同时保留长表支持。异常行宽、无 ID 的数据行、
错误内核、多次 launch、空报告和无数字硬件指标仍拒绝通过，不是简单删掉断言。
解析保留原值及单位，如 `144.800000 us`、`230.400000 Kbyte/block`，不能因为
`--csv` 文档提及 base units 就假定这份导出没有单位缩放。

已在本机离线恢复 **795 个指标**，其中 371 个符合检查用 SM/SMSP/DRAM/LTS 等
硬件前缀并有有限数值。`--analyze` 只读取原 collection/worker/CSV，校验原验算
状态和父子源码指纹，在新目录保存结果。原失败日志和原生报告保持不变，
无需重新跑 GPU。ncu 的 rules/details 页面尚未导出，离线步骤不伪造该页面。

新增 16 项回归覆盖真实宽表、单位保留、缺少单位行、多次 launch、畸形行、长表
兼容、完整宽表导出流程，以及无 Torch/TVM/ncu 的离线恢复和原证据不被修改。
采集工具定向检查 **33 passed**；完整本地工具/源码生成回归 **510 passed，259.58 s**。

### 计数器支持的结论与限制

以下均来自本次 profile，不是 CUDA-event 成绩：

| 指标 | 值 | 含义 |
|---|---:|---|
| TC pipeline active cycles / SM active | 91.850102% | SM 活跃时 TC 长时间忙碌；不等同于达到该比例的峰值 FLOP/s |
| TC pipeline active cycles / elapsed | 75.687654% | 按整体执行时段计，利用率明显低于活跃期 |
| L2 throughput / peak elapsed | 22.610307% | 未显示 L2 带宽接近饱和 |
| DRAM throughput / peak elapsed | 10.734518% | 未显示 DRAM 带宽接近饱和，不能排除访存延迟问题 |
| L2 sector hit rate | 72.256016% | 需结合未清缓存、多 pass 的条件解读 |
| Active / eligible warps per scheduler | 2.998267 / 0.037752 | 发射候选很少，但异步 MMA 和角色等待不能直接按普通指令核解释 |
| Issue active / peak active | 3.342205% | 不代表 TC 只有 3% 忙碌，TC 活跃指标见上 |
| Grid / CTA / cluster | 128 CTA / 384 threads / 2 CTA | 当前为 64 个 cluster，GPU 共有 148 SM |
| Registers / dynamic shared memory | 167/thread / 230400 bytes/CTA | 与编译产物一致，寄存器和 SMEM 均限制每 SM 一个 CTA |

这些数据支持优先检查计算之外的整体空档，与“削减 MMA 发射指令却不提速”的
前轮结果相符。当前 128 个输出任务由 64 个 cluster 各做两次，避免了 74 cluster
网格的尾部不均衡；128 CTA 最多同时覆盖 128 个 SM，其余至少 20 个不能同时参与，
这是这项既有选择的一部分，不能只看 SM 数就把已经测过较慢的 74 cluster 调度
换回来。后续调度或 tile 交接假设还需直接对照。

WarpStateStats 的 `long_scoreboard` 为 **55.787340**、`barrier` 为 **19.038570**，
对应字段是 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`，
不是百分比，也没有按角色或 PC 分解。不能把这些数值直接说成 “56% 时间在读
显存” 或某个 barrier 是根因。定位具体等待点需源码/PC 相关采样；本次原始表
尚不提供这种归因。

profiling 时长 **144.8 us**，GPC 平均时钟约 **1.040150 GHz**；本次明确关闭
clock/cache control，ncu 已提示多 pass 指标可能不一致。不能拿它与原门槛
139.1 us 直接评分，或单凭它宣称热降频。正式全量结果仍为 **56 passed / 1 failed**，
本轮只修复诊断工具，生产内核、计时器和 GPU 验收标准均未改变。

### TMEM 双缓冲实验设计（4096 首轮实测见本文开头）

`b033434` 是报告修复，不是性能修复。下一步用一个新变体检验 tile 间的读回等待，
依据是 TC 在 SM 活跃期间较忙、整体利用率较低，以及发射端指令削减没有收益。
这些指标只支持安排实验，尚不能证明 accumulator 重用是主要瓶颈。

可区分的预测：

1. 若单个 accumulator 槽让下一 tile 的 MMA 等待前一 tile 读回，增加第二槽应使
   `n128_tmem_double_buffer` 稳定快于相同 N128/EPI32 的单槽版本。
2. 若这段等待已被输入供数或其他工作隐藏，双槽不会带来净收益，可能因为地址
   和 phase 状态增多变慢；不采用，也不把 5/5 或 7/7 PASS 单独当成改进证据。
3. 缩窄 N 和 EPI32 已测过，整体没有稳健净收益。新版本必须同时胜过直接对照和
   正式 N256 baseline，才能说明隐藏的等待足以补偿窄 N 的额外任务数。

默认及显式命令均保留四项对照链：

| 版本 | N / EPI_N | 每 consumer 的 accumulator 槽 | 直接对照 |
|---|---:|---:|---|
| baseline | 256 / 64 | 1 | baseline |
| n_tile_128 | 128 / 64 | 1 | baseline |
| n128_epi32 | 128 / 32 | 1 | n_tile_128 |
| n128_tmem_double_buffer | 128 / 32 | 2 | n128_epi32 |

原 N256 两个 consumer 占满 512 列 TMEM，直接翻倍超出硬件容量。新版本沿用
既有 N128/EPI32：每槽 128 列，地址为 `(slot * 2 + consumer) * 128`，四个区域
覆盖 `[0,512)`。保留原 512 列分配/释放，只增加此前窄 N 版本未用的第二半区域。
不增加线程、输入 ring、输出 SMEM buffer 或数学运算；与直接对照均为四级 K64、
181248 字节动态 SMEM、384 线程，在 4096 下 64 cluster 各做四个 512×128 输出任务。

`mma2ld` 和 `ld2mma` 各由两个槽增为四个，新增 barrier 仍在保留的 1024 字节
头部内，故 SMEM 总量不变。每槽的 free 等待初始 phase=1、ready 等待初始 phase=0；
两个槽轮转后才翻转 phase。MMA 在当前槽 ready commit 后推进，读回在所有 TMEM
load wait、before-thread-sync fence 和 256 个远端线程到达后才释放当前槽。
下一 tile 可以计算到另一个槽，隔一个 tile 重用时仍必须等两个 CTA 读完。
原输入 `mma2tma.init(2)`、B-first TMA、输出写回与最终 cluster sync 不变。

新变体计时前在 `(4096,3072,K)` 的 K=64/192/256/320 各验算两次；192 个窄输出
任务使用 64 cluster，每个 cluster 做三个 tile，强制验证槽 0 的重用。原两个窄 N
对照也保留其 K=64/320/384 边界验证。四个主尺寸和单 tile 同时覆盖本地生成检查。
已核对 4096 的单槽对照 CUDA 与 `step10_n128.eVKFm2` 记录一致；新旧 CUDA 除
槽地址、配套 barrier 和 phase 推进外逐字比较一致（展开并移除冗余 CSE 定义后）。
随机交错模型从实际生成的 CUDA 提取槽位/地址表达式，模拟两个 CTA 的延迟读回，
检验七个 tile 中的多轮 phase、不同 consumer 独立推进、无提前覆盖/死锁。
这是协议检查，不能替代 GPU 数值检查和 SASS/资源检查。
新增 20 项检查通过，完整本地工具/源码生成回归 **530 passed，284.63 s**。

该实验已在 `88b0b39` 完成 4096 七轮实测，结果见本文开头。
原理想目标为最慢样本 ≤0.135 ms，给原门槛约 3% 余量；它是选优目标，
**不改变原评分标准**。首测有一致收益，下一步按 [RUNNING.md](RUNNING.md)
完成四尺寸对照，再决定正式采用；生产内核和正式全量状态尚未改变。

## 前轮结果：step10_mma_unroll4.iR07fS，固定展开后没有性能收益

数据：[summary](results_b300/step10_mma_unroll4.iR07fS/step10/summary.csv)、
[samples](results_b300/step10_mma_unroll4.iR07fS/step10/samples.json)、
[run.json](results_b300/step10_mma_unroll4.iR07fS/step10/run.json)。
版本 `f2963b7`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、十一份 builder/编译 CUDA 指纹和八份边界验算记录核对通过。
矩形 K=64/192/256/320 的两次验算和逐轮数值检查均通过。
baseline 的 CUDA、cubin 和 NVRTC 参数与正式 B-first 验收产物一致。

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 同轮相对 baseline | 同轮相对直接对照 | 最慢样本余量 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.138396 | 0.138846 | 5/5 | 1.000000× | 1.000000× | 0.183% |
| mma_unroll4 | 0.138352 | 0.138522 | 5/5 | 1.000486× | 1.000486× | 0.416% |
| mma_batch_unroll4 | 0.138343 | 0.138985 | 5/5 | 0.999204× | 0.998207×（对 mma_unroll4） | 0.083% |

**`mma_unroll4` 的 cubin 和 SASS 与 baseline 逐字节相同**，其计时差异属于波动。
batch 对直接对照五轮只快两轮，同轮加速中位数为 **0.998207×**，并未加速。
虽然表中三个中位数都通过，batch 最慢样本距门槛仅 **0.115 微秒**。
本轮两项均不采用。最新完整 GPU 套件仍是 **56 passed / 1 failed**，
不能将这次 probe 的 PASS 当作正式稳定验收。

三个版本均 **REG167 / STACK0**，没有 LDL/STL，MMA 主循环同为四级 K64：

| 版本 | 主循环范围 | UTCHMMA | R2UR | 循环内指令数 |
|---|---|---:|---:|---:|
| baseline / mma_unroll4 | 0x1620–0x21c0 | 16 | 63 | 187 |
| mma_batch_unroll4 | 0x1580–0x1e50 | 16 | 22 | 142 |

统计包含主循环地址范围内的指令，不含跳出该范围的等待分支；这是静态代码计数。
batch 减少寄存器搬运和指令数，却没有加速，当前证据不支持继续沿描述符复用和
展开方向微调。所有版本的输入、数学运算和写回相同，生产文件 SHA256 仍为
`3dfec9f00bda86d46f1664ca17ffe7af5f6095e78884cdc54bec956990c7bcf7`。

### 已完成采集：正式内核的 Nsight Compute 计数器

按以下预测区分下一步方向：

1. 若 Tensor Core 执行吞吐接近上限，减少发射端指令不应显著提速；查看实际 TC
   pipeline 利用率和 ComputeWorkloadAnalysis。Blackwell 的 TC 与旧 Tensor
   pipeline 不等价，不能只套用旧架构 metric 名称。
2. 若输入供数受限，MemoryWorkloadAnalysis、L2/DRAM 吞吐和命中率应提供证据；
   结合 ComputeWorkloadAnalysis 判断异步等待是否伴随计算空闲。
3. 若供数和计算均未充分利用，结合 SchedulerStats、WarpStateStats 和 cluster
   LaunchStats/Occupancy 检查可发射 warp 与等待。单个 stall 百分比不能独立定因。

`profile_hardware.py` 只编译生产 Step 10 / 4096。第一次调用后移除编译 hook，
验算并预热十次，输出置 NaN，再用 `cudaProfilerStart/Stop` 包围一次 GEMM 和完成同步。
区间内没有数值验证、cuBLAS 或其他显式内核。区间结束后验算输出，检查报告中
只有一个 `kernel_kernel` 结果且含数字形式的硬件计数器。工具保存编译产物、版本、
命令参数、可用/缺失 section、原始 CSV、details 与原生报告，不给 PASS/SLOW 分数。

使用 kernel replay，因为该内核不依赖 launch 之后的 host 响应；不为每个采集 pass
重做 Python/TVM 编译。显式禁用 clock/cache control，支持时使用 dynamic pipeline
boost；这不消除 profiler 扰动。缓存不刷新的多 pass 指标可能不完全一致，应结合
报告警告解读，不能用 profiler 耗时解释原 CUDA-event 的零点几微秒差值。
依据：[NVIDIA CLI 选项](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)、
[Kernel Replay 与 pipeline 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

缺少工具、计数器权限、空报告或 worker 验算失败会返回非零，并保留日志。
当时本机无 NVIDIA GPU，仅完成工具进程和区间测试；B300 采集结果及 parser
格式缺陷见本文最新记录。
新增 17 项检查覆盖正式 builder 调用、采集前后验证与输出清空、区间内单次调用、
异常时关闭 profiler、新旧 ncu 报告导出、权限失败、缺少工具、空报告和源码变化。
完整本地工具/源码生成回归 **494 passed，258.27 s**；生产内核、计时器和 GPU
用例均未修改。
以下是该轮的历史采集命令，本次无需重复：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_hardware.XXXXXX)
uv run python -u profile_hardware.py \
  --output "$tirx_run/profile" 2>&1 | tee "$tirx_run/profile.log"
printf '结果目录：%s\n' "$tirx_run"
```

若 ncu 不在 PATH，添加 `--ncu /实际路径/ncu`。输出目录必须是新目录。
完整操作及权限错误处理见 [RUNNING.md](RUNNING.md)。

## 前轮结果：step10_mma_batch.13Q2qL，小幅改善伴随自动展开变化

数据：[summary](results_b300/step10_mma_batch.13Q2qL/step10/summary.csv)、
[samples](results_b300/step10_mma_batch.13Q2qL/step10/samples.json)、
[run.json](results_b300/step10_mma_batch.13Q2qL/step10/run.json)。
版本 `56301a2`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、九份 builder/编译 CUDA 指纹及六份边界验算记录核对通过。
矩形 K=64/256/320 的两次验算和逐轮数值检查均通过。
baseline 的 CUDA、cubin 和 NVRTC 参数与正式 B-first 验收产物完全一致。

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 同轮相对 baseline | 同轮相对直接对照 | 最慢样本余量 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.138100 | 0.139365 | 4/5 | 1.000000× | 1.000000× | -0.191% |
| mma_batch | 0.137809 | 0.138735 | 5/5 | 1.004509× | 1.004509× | 0.262% |
| mma_batch_no_unroll | 0.138135 | 0.138566 | 5/5 | 1.001556× | 1.000062×（对 mma_batch） | 0.384% |

`mma_batch` 五轮中四轮更快，同轮加速中位数 **0.451%**，第五轮却慢于 baseline
约 0.625 微秒。关闭展开相对 `mma_batch` 的同轮加速中位数仅 **0.006%**，且整体
中位耗时更高。两个候选虽 5/5 达标，最慢样本仍只低于门槛 0.365/0.534 微秒。
本轮保留候选但不采用，不能据此宣布稳定性已解决；最新完整 GPU 套件仍是
**56 passed / 1 failed**。

### SASS：自动展开从四级变成八级，原假设未得到直接支持

三个版本资源数都是 **REG167 / STACK0**，没有 LDL/STL。输入和输出静态指令数
相同：12 处 UTMALDG、四处 UTMASTG、八处 LDTM。MMA 主循环则不同：

| 版本 | 每次主循环的 K64 stage 数 | 循环内 UTCHMMA | 循环内 R2UR | R2UR / stage |
|---|---:|---:|---:|---:|
| baseline | 4 | 16 | 63 | 15.75 |
| mma_batch | 8 | 32 | 134 | 16.75 |
| mma_batch_no_unroll | 1 | 4 | 5 | 5.00 |

循环范围分别为 `0x1620–0x21c0`、`0x1580–0x2900`、`0x15a0–0x1860`，按包含
MMA 的主循环回跳统计，不含跳出循环范围的等待分支。上述是代码结构计数，不是
硬件 profiler 的动态指令数；不能把静态 MMA 数增减解读为总数学运算变化。

默认编译的 batch 没有按预期减少每 stage 的 R2UR，同时自动展开由四级变为八级。
无展开版减少了搬运但没有明显额外加速。因此上一轮的约 0.45% 收益不能单独归因
于描述符复用，也不足以证明任何具体硬件瓶颈。

### 已完成实验：双方固定展开四次，消除自动展开差异

按以下可区分的预测补齐对照：

1. 若固定展开后 batch 的发射方式仍有独立收益，`mma_batch_unroll4` 应快于
   `mma_unroll4`，且 SASS 需确认两者每轮处理相同数量的 K stage。
2. 若之前的收益主要伴随自动展开八次或测量波动，固定四次后这点优势可能消失；
   此时不继续把它归因于描述符复用，也不直接采用上一轮的 batch。
3. 若显式 pragma 本身改变编译器决策，`mma_unroll4` 也可能不同于 baseline；
   保留生产 baseline 以发现这一点，不能假设“原先自动四次”等于“强制四次”。

当时默认比较 `baseline`、`mma_unroll4`、`mma_batch_unroll4`。第一个新版本只给原
MMA K 循环加 `#pragma unroll 4`；第二个在相同 pragma 下使用已经验算过的四条
K16 batch helper，直接对照 `mma_unroll4`。三个版本的 builder/TIR 相同。
K=64 的循环已被 TVM 消去，原版保持原源码，batch 保持已测的单级发射块。
不改变 producer 的展开、不移动 fence/wait/commit，也不改变所有 stage 的运算。

每项实验计时前在 `(4096,3072,K)` 的 **K=64/192/256/320** 下各验算两次，
覆盖单级、短于展开因子、整组四级和一组加尾部，以及 persistent tile 重用。
随后使用原 10 warmup / 30 repeat / 五轮交错 CUDA-event 计时并保存编译产物。
19 项新增检查覆盖两个架构和九种形状，重放本轮实际编译输入，逐字比较除 pragma
与四次 MMA 调用之外的源码；完整工具/源码生成回归 **477 项通过，249.46 s**。
GPU 实际结果见本文最新记录，两项均不采用。以下为历史命令，现需显式选变体。
**生产内核和评分规则不变。**

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_mma_unroll4.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --variants mma_batch_unroll4 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_wide_tma.FBbwDr，写回与加载分工仍无足够收益

数据：[summary](results_b300/step10_wide_tma.FBbwDr/step10/summary.csv)、
[samples](results_b300/step10_wide_tma.FBbwDr/step10/samples.json)、
[run.json](results_b300/step10_wide_tma.FBbwDr/step10/run.json)。
版本 `2ce0615`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、九份 builder/编译 CUDA 指纹及六份边界验算记录核对通过。
两个实验在矩形 K=64/256/320 下各完成两次验算，逐轮校验也通过。
baseline 的 CUDA、cubin 和 NVRTC 参数与正式 B-first 验收产物完全一致。

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 快于 baseline 的轮次 | 同轮加速比 | REG / STACK |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.138144 | 0.138831 | 5/5 | — | 1.000000× | 167 / 0 |
| epilogue_32 | 0.138205 | 0.139390 | 4/5 | 1/5 | 0.998264× | 168 / 8 |
| split_tma | 0.137762 | 0.138817 | 5/5 | 4/5 | 1.001471× | 167 / 0 |

`epilogue_32` 中位数稍慢，且仍有样本超过 **0.139100 ms** 门槛；窄 N 下的
EPI32 收益未能在原 N256 上重现。SASS 中 UTMASTG 从四处增至八处，新增
**三处 LDL、三处 STL 和 8 字节栈帧**。这些是实际额外开销，但本轮不能分离
写回次数和栈访问各自对性能的影响。

`split_tma` 五轮有四轮更快，但同轮加速中位数只有 **0.147%**。最慢样本余量
**0.203%**，baseline 同轮余量也是 **0.193%**，两者最慢值仅差 **0.014 微秒**。
它没有 LDL/STL，资源数与 baseline 相同；二者均有 16 处 UTCHMMA、12 处
UTMALDG、四处 UTMASTG、八处 LDTM。本轮没有足够证据把拆分 producer 作为
稳定性修复，两项均不采用。最新完整 GPU 套件仍是 **56 passed / 1 failed**。

### 已完成实验：MMA 描述符复用与编译器展开

本轮 baseline SASS 的 MMA 循环在每组四条 UTCHMMA 周围仍有反复的 R2UR、
描述符准备及 uniform 寄存器搬运。此前等待提示、写回、输入深度和 producer 分工
没有取得足够余量。按以下可区分的预测安排对照：

1. 若四个独立 inline-PTX 调用的描述符准备限制 MMA 发射，将同一 K64 stage 的
   四条 K16 MMA 放入一个 PTX 块，复用 A/B 描述符并在块内递增，应减少搬运并提速。
2. 若编译器对 K 循环的展开增加了描述符活跃值和寄存器搬运，关闭展开应在上述
   改动上继续受益。历史 `cache_mma_no_unroll` 没有明显收益，本轮只检验它与
   新发射块的关系，不将历史实验重新算作新发现。
3. 若实际指令减少仍未加速，则本轮不支持描述符准备是主要限制，应继续区分
   异步数据供给与硬件执行等待。若 SASS 不变，则该源码改动未影响实际发射。

当时默认比较 `baseline`、`mma_batch`、`mma_batch_no_unroll`。`mma_batch` 直接对照
baseline；`mma_batch_no_unroll` 直接对照 `mma_batch`，只增加 MMA K 循环的
`#pragma unroll 1`。K=64 时 TVM 已移除单次循环，两项实验生成相同代码。

两个实验的 builder/TIR 均保持生产版本；工具只替换 CUDA 中连续的四次 MMA 调用，
不移动其前后的 wait、fence、commit、phase 或写回。新块保留 **四条** K16 MMA，
保留 M256/N256、相同 TMEM 目标、八个零 mask 和累加顺序：每个输出 tile 的
第一次 MMA 清零，之后全部累加。A/B descriptor 的低 32 位依次加 2（32 字节），
高 32 位保持不变，与原 `smem_desc_add_16B_offset` 一致。不能把它误读为
减少数学运算或把四条硬件 MMA 变成一条。

输入四级 K64、230,400 字节动态 SMEM、两个 consumer、两个写回 warpgroup、
B 优先加载和调度网格不变。GPU 计时前，两项实验分别对矩形 K=64/256/320 各验算
两次，随后使用原 10 warmup / 30 repeat / 五轮交错 CUDA-event 计时。工具保存
实际编译 CUDA、cubin、SASS 和资源数，下一轮需核对 R2UR、UTCHMMA 与栈访问。
完整本地工具/源码生成回归 **458 项通过，232.84 s**。其中 24 项新增检查
覆盖两个架构、四个评分尺寸、三个矩形边界和单 tile；
包含对实际生成 PTX 的整数/谓词解释，核对 descriptor 高低位、mask 和累加行为。
GPU 结果见本文最新记录，两项暂不采用。以下为该轮历史命令，现需显式选变体：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_mma_batch.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --variants mma_batch_no_unroll \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_n128.eVKFm2，窄 N 没有取得稳定收益

数据：[summary](results_b300/step10_n128.eVKFm2/step10/summary.csv)、
[samples](results_b300/step10_n128.eVKFm2/step10/samples.json)、
[run.json](results_b300/step10_n128.eVKFm2/step10/run.json)。
版本 `bcd2214`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、十三份 builder/编译 CUDA 指纹和九份边界验算记录核对通过。
三个实验在矩形 K=64/320/384 下均完成两次数值验算，逐轮校验也通过。
baseline 的 CUDA、cubin 和 NVRTC 参数与正式 B-first 验收产物一致。

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 直接对照 | 同轮加速比 | REG / STACK |
|---|---:|---:|---:|---|---:|---:|
| baseline | 0.138363 | 0.139395 | 4/5 | baseline | 1.000000× | 167 / 0 |
| n_tile_128 | 0.140717 | 0.141091 | 0/5 | baseline | 0.983270× | 105 / 0 |
| n128_epi32 | 0.138884 | 0.139219 | 4/5 | n_tile_128 | 1.014371× | 112 / 0 |
| n128_epi32_depth5 | 0.138657 | 0.138816 | 5/5 | n128_epi32 | 1.002654× | 112 / 0 |

门槛仍为 **0.139100 ms**。单独缩小 N 五轮都更慢；EPI32 相对窄 N 的 EPI64
五轮都更快，但相对生产 baseline 只快了一轮。五级版本虽 5/5 达标，中位耗时仍
高于 baseline，相对 baseline 的同轮加速中位数仅 **0.103%**，最慢样本余量
**0.204%**。本轮不采用任何候选，不能据五个 PASS 宣称已解决性能稳定性。
最新完整 GPU 套件仍是 **56 passed / 1 failed**。

窄 N 的寄存器数明显减少，整体却没有加速；四个版本都没有 LDL/STL，不能仅凭
寄存器数判断瓶颈。SASS 静态计数中，四者都是 16 处 UTCHMMA、12 处 UTMALDG；
UTMASTG 分别为 4/2/4/4，LDTM 为 8/4/4/4。EPI32 在窄 N 对照下的收益支持
单独检验原 N256 的小写回分块，但尚不能外推为宽 N 下的性能收益。

### 已完成实验：保留 N256，独立检验写回分块与 A/B 加载分工

当时默认比较以下三个版本，两项实验都直接对照生产 baseline，不叠加：

| 版本 | 改动 | 动态 SMEM 字节 |
|---|---|---:|
| baseline | 已采用的 N256、K64、四级输入、EPI64、单个 TMA producer warp | 230400 |
| epilogue_32 | 仅缩小 EPI_N 至 32，并使用匹配 64 字节行的 D swizzle | 214016 |
| split_tma | WG2 warp 2 加载 B，warp 3 加载 A0/A1 | 230400 |

`epilogue_32` 保留原输入、两个 MMA consumer、两个写回 warpgroup、256 列寄存器
暂存和 TMEM 释放点。检验小写回分块的收益能否保留到宽 N；每个写回 warpgroup
处理一个 tile 时的 TMA store/wait 由四次增加到八次，可能抵消更小分块和 SMEM
占用的收益。

`split_tma` 检验单个 elected lane 串行准备、发射 B/A0/A1 请求是否限制供数。
两个 producer warp 都先等待原 `mma2tma` 空槽，再分别发 B 与 A0/A1；各自的线程
私有调度器和 phase 同步推进。只有 **CTA 0 的 warp 3** 执行一次 `arrive.expect_tx`，
仍宣告 **98,304 字节**；`tma2mma.init(1)` 不变，两个 CTA 的六个 TMA 请求共同
完成同一 full barrier。必须等全部输入就绪、两个 consumer 都完成 MMA，才可重用
该槽，因此快的 producer 无法越过慢的 producer 重用其输入。写回、内存分配和总
传输量不变；两个 warp 的发射先后不保证 B 优先，额外调度和等待也可能抵消收益。

计时前，每项实验在 `(4096,3072,K)` 的 K=64/256/320 各验算两次，共六个边界
构建、十二次 launch，覆盖短 ring、完整四级 ring、不完整 ring 和跨 tile 重用。
之后按原 10 warmup / 30 repeat / 五轮交错 CUDA-event 计时，保存源码、编译资源与
SASS。完整本地工具/源码生成回归 **434 项通过，234.22 s**。其中 19 项新增检查
覆盖两个架构、四个评分尺寸、三个矩形边界及单 tile，核对 TMA 描述符、写回覆盖
范围、实际生成的 barrier 到达归属和 phase。
GPU 结果见本文最新记录，两项均未采用。以下为该轮的历史命令，现需显式选变体：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_wide_tma.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --variants epilogue_32 split_tma \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：B-first 一次全过，复测仍有性能失败

正式验收数据：[pytest](results_b300/step10_bfirst.A0Iwp0/pytest_all.log)、
[Step 10 benchmark](results_b300/step10_bfirst.A0Iwp0/step10.csv)、
[编译记录](results_b300/step10_bfirst.A0Iwp0/compiler_step10/run.json)。
版本 `cb26383`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
四个记录的源码指纹核对通过，4096 的 CUDA、cubin 和 NVRTC 参数与实测
`step10_roles.rx5lNL` 中的 B-first 胜出版本完全一致。

| 运行 | 全量结果 | Step 10 / 4096 ms | 相对 0.139100 ms 门槛 |
|---|---|---:|---|
| `step10_bfirst.A0Iwp0` | **57 passed，74.54 s** | 0.139047 | 低 0.053 微秒 / 0.038% |
| 用户回传 `pytest_1e37e7_1.log` | **56 passed / 1 failed，74.17 s** | 0.139284 | 高 0.184 微秒 / 0.132% |

第二行的[完整日志](results_b300/pytest_1e37e7_1.log)现已入库，与用户回传的终端
输出一致；该日志不包含编译产物，不能额外核对该次运行的 cubin。
仓库 `1e37e76` 相对 `cb26383` 的生产内核、评分代码和 GPU 测试未变。
两次的所有数值检查都通过，失败只在同一个性能门槛。
**上次“本次全部通过”属实，但不表示已取得稳定通过的余量。**

正式 Step 10 benchmark 的四个尺寸共 20 个样本均达标：

| 大小 | 中位数 ms | 最大值 ms | 门槛 ms | 最慢样本余量 |
|---|---:|---:|---:|---:|
| 1024 | 0.027481 | 0.027627 | 0.032500 | 14.99% |
| 2048 | 0.043169 | 0.043280 | 0.045500 | 4.88% |
| 4096 | 0.138409 | 0.138994 | 0.139100 | **0.076%** |
| 8192 | 0.869102 | 0.869911 | 0.946400 | 8.08% |

这些结果与极小余量下的跨运行波动相符，不能据此确定波动来自频率、负载或其他具体
原因。继续原计时和评分规则，优化目标是增大余量，而非重跑挑选通过的一次。

### 已完成实验：缩小 N 分块与输入供给

按可区分的预测安排三个实验，每一步有直接对照：

1. 若 256 列的 accumulator 写回暂存/调度限制性能，将 MMA_N 从 256 降至 128，
   每个 CTA 的 B 行数从 128 降至 64，应降低写回暂存和输入 SMEM。反面代价是
   输出 tile 数翻倍、A 的重复读取与每矩阵 barrier 次数增加，收益必须实测。
2. 若缩小 epilogue 能释放更多 SMEM，EPI64 改为 EPI32 后应降低占用；其 store/wait
   次数翻倍可能抵消收益，因此将这一步单独计时，不能把代价隐藏在流水线深度实验中。
3. 若供数延迟仍限制窄 N 的 MMA，利用腾出的 SMEM 将输入由四级加到五级应加速；
   若没有加速，则更深的 ring 和额外 phase 计算没有净收益。

| 版本 | 直接对照 | MMA_N / 每 CTA 的 B 行数 | EPI_N | 输入深度 | 动态 SMEM 字节 |
|---|---|---:|---:|---:|---:|
| baseline | baseline | 256 / 128 | 64 | 4 | 230400 |
| n_tile_128 | baseline | 128 / 64 | 64 | 4 | 197632 |
| n128_epi32 | n_tile_128 | 128 / 64 | 32 | 4 | 181248 |
| n128_epi32_depth5 | n128_epi32 | 128 / 64 | 32 | 5 | 222208 |

所有实验保留两个 MMA consumer、两个写回 warpgroup、B 优先加载、缓存 TMEM 基址、
均衡网格公式、512 列 TMEM 分配和现有同步协议。4096 输出从 128 个 512×256 tile
变为 256 个 512×128 tile，仍由 64 个 cluster 覆盖，但每个 cluster 由两个 tile
变为四个。两个 consumer 分别读写 TMEM 的 `[0,128)` 和 `[128,256)`。
每级 TMA transaction bytes 从 98,304 变为 81,920。
EPI32 使用与 64 字节行匹配的 `SWIZZLE_64B_ATOM`；A/B 的 128 字节 swizzle 保留。
五级输入和 EPI64 会占 238,592 字节，超过本工具使用的 232,448 字节动态预算，
因此以独立 EPI32 对照作为五级实验的基础。

计时前，每个新变体在 `(4096,3072,K)` 的 K=64/320/384 各验算两次，覆盖短于 ring、
完整五级 ring 和不完整 ring，以及三个输出 tile 的重用。数值验证失败会停止。
之后四个版本使用原 10 warmup / 30 repeat / 五轮交错 CUDA-event 计时，保存直接
对照、源码指纹、编译资源与 SASS。GPU 结果见本文最新记录，三个实验均未采用。
该轮本地完整工具/源码生成回归 **415 项通过，205.25 s**。新增 20 项覆盖两个架构、
评分与边界形状，检查实际 TMA box/stride/swizzle、SMEM 范围、MMA N 描述符、
consumer TMEM 偏移、写回坐标、barrier/phase、比较链与重复应用拒绝。
这些检查不替代 B300 上的编译、数值验算和性能测量。

以下为当时的历史命令；现需显式选择窄 N 变体，默认已改为宽 N 的两个独立实验：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_n128.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --variants n128_epi32_depth5 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_roles.rx5lNL

数据：[summary](results_b300/step10_roles.rx5lNL/step10/summary.csv)、
[samples](results_b300/step10_roles.rx5lNL/step10/samples.json)、
[run.json](results_b300/step10_roles.rx5lNL/step10/run.json)。
版本 `c69e612`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、七份 builder/编译 CUDA 指纹核对通过。三个版本数值校验通过，
两个实验的矩形 K=64/320 各完成两次不计时验算。
baseline 的 CUDA、cubin 和 NVRTC 参数与 `step10_adopt.UFT7xo` 相同。

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 同轮加速比 | REG / STACK |
|---|---:|---:|---:|---:|---:|
| baseline | 0.138356 | 0.139254 | 4/5 | 1.000000× | 167 / 0 |
| role_registers | 0.138396 | 0.138643 | 5/5 | 1.001733× | 167 / 0 |
| tma_b_first | **0.137891** | **0.138605** | **5/5** | **1.004687×** | 167 / 0 |

### 采用 B 优先加载

`tma_b_first` 五轮都快于同轮 baseline，同轮加速中位数约 **0.469%**。
最慢样本距 0.139100 ms 门槛约 **0.495 微秒 / 0.356%**，收益一致但余量仍窄。
CUDA 和 SASS 确认每级请求由 A0/A1/B 改为 B/A0/A1；请求数、地址、字节数、
barrier、MMA 和写回协议保持一致。三个版本都无 LDL/STL，静态指令数量均为
12 处 UTMALDG、16 处 UTCHMMA、8 处 LDTM.x32。
这支持采用请求顺序这一小改动，尚不能证明它解决了全量测试中的波动。

正式 `hgemm_v10` 采用实测 B-first builder，保留四级 K64 输入、两个 consumer、
EPI64、缓存 TMEM 基址和均衡网格。4096 生成的 CUDA kernel 与实测输入一致，
其他评分尺寸、矩形短 K 和单 tile 在 SM100a / SM103a 上重放实测 builder。
生产文件的新 SHA256：
`3dfec9f00bda86d46f1664ca17ffe7af5f6095e78884cdc54bec956990c7bcf7`。
实测 B-first cubin SHA256：
`afadd2f4f7f171021362e419883ab200ac3fa1e7f504c58e9a762eb471725e3d`。
随后的正式 cubin 一致性已核对，验收和复测结果见本文最新记录。

### 寄存器提示被编译器忽略

`role_registers` 的 [NVRTC 日志](results_b300/step10_roles.rx5lNL/step10/step10_4096_role_registers/nvrtc_01.log)
报告：`(C7508) Potential Performance Loss: 'setmaxnreg' ignored; unable to determine register count at entry.`
SASS 中没有对应的寄存器重分配指令。因此这次实验**没有实际检验寄存器重分配的性能**，
不能根据 PASS 或小幅计时变化认为该策略有效，也不采用它。
历史 `register_budget.json` 的 `valid: true` 仅说明初始 REG167 足以容纳所请求总量，
不能证明指令生效。工具现在分别记录 `capacity_valid` 和 `setmaxnreg_ignored`，
遇到该忽略诊断会在返回 cubin、启动 kernel 之前停止，并恢复编译回调。
没有此诊断也不等于提示必然生效，今后的寄存器实验仍需检查实际 SASS。

当时默认 probe 只运行正式 baseline；已采用的 `tma_b_first` 拒绝重复应用，
历史角色和双缓冲实验的本地回归使用记录的旧 builder，保留原对照含义。
完整本地工具/源码生成回归 **395 项通过，188.26 s**，包含实测 CUDA 重放、
跨架构边界检查和忽略提示时停止/恢复回调的回归。原评分、容差、计时规则和 GPU
测试未改。该次采用时最新全量为 **56/57**；以下正式验收已完成，见本文最新记录：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_bfirst.XXXXXX)
uv run python -m pytest tests/ -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_all.log"
uv run python -u benchmark.py --steps 10 --trials 5 \
  --csv "$tirx_run/step10.csv" \
  --diagnostics-dir "$tirx_run/compiler_step10" \
  2>&1 | tee "$tirx_run/benchmark_step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_epilogue.Y2EFaW

数据：[summary](results_b300/step10_epilogue.Y2EFaW/step10/summary.csv)、
[samples](results_b300/step10_epilogue.Y2EFaW/step10/samples.json)、
[run.json](results_b300/step10_epilogue.Y2EFaW/step10/run.json)。
版本 `a6e820d`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、九份 builder/编译 CUDA 指纹及六份边界验算记录核对通过。
所有版本通过数值校验，两个实验的矩形 K=64/192/320 均完成两次不计时验算。

### 双缓冲增加了栈开销，未带来性能收益

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 直接对照 | 同轮加速比 | REG / STACK |
|---|---:|---:|---:|---|---:|---:|
| baseline | 0.138462 | 0.139356 | 3/5 | baseline | 1.000000× | 167 / 0 |
| epilogue_depth3 | 0.138224 | 0.138671 | 5/5 | baseline | 1.000694× | 167 / 0 |
| epilogue_double_buffer | 0.139356 | 0.140256 | 2/5 | epilogue_depth3 | 0.991879× | 168 / 32 |

双缓冲五轮都慢于直接对照，同轮耗时增加的中位数约 **0.82%**，不采用。
SASS 的 `DEPBAR.LE` 静态位置由 6 减至 5，但新增 **8 处 STL / 8 处 LDL** 和
**32 字节栈帧**。三个版本都有八处 `LDTM.x32`、四处 `UTMASTG.2D` 和 16 处
`UTCHMMA.2CTA`。减少等待的同时增加了寄存器压力；这些结果说明本次双缓冲实现
没有净收益，不能据此断定所有写回重叠方案都无效。

三级对照本轮虽然 5/5 达标，最慢样本余量 **0.31%**，但只在三轮快于 baseline，
同轮加速中位数仅 **0.069%**。它的 CUDA、cubin 和 NVRTC 参数与此前
`step10_depth3.1zWPNg` 的 `balanced_depth3` 完全一致，后者五轮均超限。
baseline 的编译产物也与正式采用时相同。因此不以本轮 PASS 宣称三级输入解决了
稳定性问题。生产继续保持四级输入、单写回缓冲区；最新全量仍为 **56/57**。

### 已完成实验：角色寄存器预算与 TMA 发射顺序

三个可区分的假设按优先级为：

1. 若统一寄存器分配限制写回指令调度，TMA/MMA warpgroup 释放多余配额、两个
   写回 warpgroup 获得更多配额后，耗时应下降；若 64 个寄存器不足以容纳 producer，
   可能出现新的 spill，需检查实际 cubin 和计时。
2. 若两个 consumer 共享的 B 请求完成较晚，改为先发 B 再发 A0/A1 应缩短同一
   full barrier 的就绪延迟。请求数、字节数和 barrier 不变，没有收益即停止该方向。
3. 若以上微调仍无明显收益，输入/MMA tile 的形状或供给粒度更可能限制吞吐；下一步
   再测 tile 结构变化，本轮不将它与前两项叠加。

当时默认比较 `baseline`、`role_registers`、`tma_b_first`，两项实验均直接对照正式 baseline：

- `role_registers`：在角色分歧、线程选举和初始化同步之前，整个 WG2 执行
  `setmaxnreg.dec 64`，WG0/WG1 各执行 `setmaxnreg.inc 208`。请求总量为
  `128 × (64 + 208 + 208) = 61,440`；该预算不代表改变 occupancy。
  两个 inc 可以等到 WG2 释放配额；释放路径前无 CTA barrier，避免相互等待。
  编译回调在首次启动前读取实际 cubin 的 REG，要求初始配额满足 PTX 方向限制及
  CTA 总量（当前预算要求 160–208）；不能读取或不满足则停止，并保存判断记录。
- `tma_b_first`：每级从 A0/A1/B 改为 B/A0/A1；TMA 地址、字节数、expect_tx 的位置、
  MMA、写回、所有 phase 与同步均保持不变。

寄存器语义依据 [NVIDIA PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)：
配额池属于 CTA，inc 可能阻塞，同一 warpgroup 的所有 warp 必须执行同一指令。
本实验使用 TVM 0.26 的 intrinsic；不改变全局编译选项。SASS 和资源报告用于确认
提示实际生效及是否引入 spill，源码生成不能替代该验证。

两个实验保留四级 K64 流水线、230,400 字节动态 SMEM、缓存基址和均衡网格；
计时前对矩形 K=64/320 各两次验算，再用原 10 warmup / 30 repeat / 五轮交错计时。
完整本地工具/源码生成回归 **393 项通过，181.16 s**，新增检查覆盖两个架构、四个评分
尺寸、矩形边界、单 tile，以及寄存器配额不足/资源报告缺失时拦截和恢复编译回调。
该轮工具提交未改生产内核；随后 GPU 结果和采用决策见本文最新记录。

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_roles.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_writeback.2bP3jK

数据：[summary](results_b300/step10_writeback.2bP3jK/step10/summary.csv)、
[samples](results_b300/step10_writeback.2bP3jK/step10/samples.json)、
[run.json](results_b300/step10_writeback.2bP3jK/step10/run.json)。
版本 `50b0987`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个源码指纹、七份 builder/编译 CUDA 指纹及四份边界验算记录核对通过。
baseline 的 CUDA、cubin、NVRTC 参数与 `step10_adopt.UFT7xo` 完全一致。
三个版本的初始、逐轮数值校验通过；两个新版本各自在矩形 K=64/320 下两次验算通过。

### 中位数达标，仍无足够余量

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 同轮加速比 | REG / STACK |
|---|---:|---:|---:|---:|---:|
| baseline | 0.138267 | 0.139415 | 4/5 | 1.000000× | 167 / 0 |
| warp_release | 0.138192 | 0.139214 | 4/5 | 1.000540× | 168 / 0 |
| paired_tmem_loads | 0.138149 | 0.139172 | 4/5 | 1.001747× | 168 / 0 |

门槛仍为 **0.139100 ms**。三个版本均在第一轮超限；summary 的 PASS 只代表中位数。
warp 聚合四轮快、一轮慢，同轮加速中位数 **0.054%**；成对读取五轮都快，但同轮
加速中位数仅 **0.175%**，最慢样本仍超限 **0.072 微秒**。没有足够证据把任一版本
作为解决性能余量的正式改动，也不叠加两项微小收益。最新全量仍为 **56/57**。

SASS 确认聚合到达变成带 count 的 `SYNCS.ARRIVE.TRANS64.RED.ART0`；成对读取仍是
八条 `LDTM.x32`，CUDA 中 `wait.ld` 从八次减至四次。三个版本均有四处 `UTMASTG.2D`、
16 处 `UTCHMMA.2CTA`，没有 `LDL`/`STL`。这次结果不支持栈溢出解释，也表明这两处
调整尚不足以解决剩余失败。

### 已完成实验：双缓冲 TMA 写回

按当前证据保留三个可区分的假设：

1. **写回串行化**：若每块输出都等待同一个 Dsmem 的 TMA 读取完成限制尾部延迟，
   两个交替缓冲区应允许下一块 SMEM 写入与前一块 TMA store 重叠，缩短写回耗时。
2. **输入流水线深度**：双缓冲额外需要 32 KiB；保留四级输入会达到 263,168 字节，
   超出共享内存预算。减为三级后总量为 214,016 字节，但可能影响 TMA/MMA 隐藏延迟，
   因此必须保留独立三级对照，不能把深度变化算作写回收益。
3. **主要限制在输入/MMA**：若双缓冲相对三级对照仍无明显收益，写回交接和读取的
   优化都不足以解释剩余耗时，应回到输入/MMA 路径；暂不继续堆叠写回小改动。

默认三个构建：

- `baseline`：当前正式四级输入、单写回缓冲区。
- `epilogue_depth3`：只将输入改为三级，动态 SMEM 181,248 字节；4096 CUDA 与此前
  实测 `balanced_depth3` 完全一致。作为新接口支持当前已采用缓存/网格的生产源码。
- `epilogue_double_buffer`：直接对照是 `epilogue_depth3`；保持 EPI_N=64，两个
  consumer 各自拥有两个 128×64 的 FP16 缓冲区。四块输出交替使用 0/1/0/1。
  第一块提交后不等待，第二、三块提交后 `wait_group(1)`，第四块 `wait_group(0)`
  排空读取。每次原 warpgroup 同步保留，确保等待完成后其他 lane 才复用缓冲区。

TMA bulk group 属于发射线程，双缓冲版本固定由各 consumer 的 warp 0 / lane 0
执行所有 store/commit/wait，避免不同次选举的线程拥有不同提交组。`wait_group`
沿用 TVM 的 `.read` 语义，等待 TMA 完成共享内存读取；在每个 tile 末尾排空，
不会把未完成的 SMEM 读取带入下一 tile 或最终 cluster 同步。TMEM 读取、释放点、
producer 协议、FP16 转换、输出坐标和原性能门槛保持不变。

本地新增 17 项检查已通过，包括两个架构、四个评分尺寸、矩形 K=64/192/320，
并从实际 CUDA 检查缓冲区覆盖、固定发射线程、最迟完成条件下的复用和最终排空顺序。
完整本地工具/源码生成回归 **369 项通过，158.24 s**。
随后 GPU 结果见本文最新记录；双缓冲未采用，生产内核不变。
工具在计时前对三级对照和双缓冲各做三种矩形形状、每种两次验算；之后五轮交错计时。

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_epilogue.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_adopt.UFT7xo

数据：[全量 pytest](results_b300/step10_adopt.UFT7xo/pytest_all.log)、
[Step 10 benchmark](results_b300/step10_adopt.UFT7xo/step10.csv)、
[编译记录](results_b300/step10_adopt.UFT7xo/compiler_step10/run.json)。
benchmark 版本 `70287d4`；B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
四个记录的 Python 源文件指纹与提交一致；生产 `gemm_kernels.py` SHA256 仍为
`2d2a0df6c80ecc809ece3f509d00ea65d6cb3a7af079e4c6fe9b6481143d4ba1`。
4096 的 CUDA、cubin、NVRTC 参数与 `step10_fused_a.1VXYz2` 胜出的
`cache_balanced_clusters` 全部完全一致，确认采用进入了实际编译内核。

### 全量 56/57，剩余性能余量不足

全量 **56 passed / 1 failed，75.28 s**。唯一失败是 Step 10 / 4096 的性能断言：
**0.139278 ms，门槛 0.139100 ms，超出约 0.178 微秒 / 0.13%**。
所有用例的数值校验均通过，Step 8 全部通过，Step 10 其余评分形状及矩形 K=64/320
也通过。此前的 55/57 现在可更新为 56/57，但尚未全过。

独立 benchmark 四个评分形状共 **20 个样本全部达标**：

| 大小 | 中位数 ms | 最大值 ms | 门槛 ms | 最慢样本余量 | REG / STACK |
|---|---:|---:|---:|---:|---:|
| 1024 | 0.027510 | 0.027860 | 0.032500 | 14.28% | 168 / 48 |
| 2048 | 0.043172 | 0.043295 | 0.045500 | 4.85% | 168 / 48 |
| 4096 | **0.138315** | **0.138570** | **0.139100** | **0.38%** | 167 / 0 |
| 8192 | 0.869437 | 0.870303 | 0.946400 | 8.04% | 167 / 0 |

benchmark 与 pytest 是不同的计时运行。4096 仅有很小余量，和已有跨运行变化相符；
不应通过反复重跑挑选 PASS、改变容差或改动计时规则来处理。保留已验证有效的缓存和
均衡网格，继续检验具体开销，争取足以覆盖波动的收益。

### 已完成实验：写回交接与 TMEM 成对读取

已排除或未测到额外收益的方向包括等待提示、循环展开、三级流水线、宽 TMA 写回和
合并 A。剩下三个可区分的假设按优先级为：

1. 若每 lane 向远端 `ld2mma` 报到限制写回交接，按 warp 聚合应缩短交接；额外
   warp 同步也可能抵消收益，必须看实测和 SASS。
2. 若八次串行 TMEM 读/等待造成延迟，两条独立 x32 读取后合并等待应比原版更快；
   代价是 FP32 暂存 32→64，可能增加寄存器或栈压力。这与之前直接使用 x64 指令不同。
3. 若主要开销仍在共享写回/TMA store 同步，前两项可能没有收益。暂不改这条路径，
   保留原重叠机会和明确对照。

默认只测三个独立构建：生产 `baseline`，以及各自仅改变一个因素的两个实验：

- **`warp_release`**：每个 lane 仍完成全部 TMEM 等待和 before-thread-sync fence，
  然后全 warp 同步，再由 elected lane 执行一次 `ld2mma.arrive(..., count=32)`。
  每 consumer 的计数仍为 `4 warps × 2 CTAs × 32 = 256`，原 init、consumer 槽位、
  acquire wait 和 phase 不变。不能仅减少到达线程而省略 warp 同步。
- **`paired_tmem_loads`**：保留八条 x32 TMEM load，每两条写入不重叠的 FP32 寄存器
  范围，再执行一次 `wait.ld`，完成后转换对应 64 列。等待由八次变为四次；
  八条源地址、整行 FP16 暂存、最终 TMEM 释放、TMA 写回与同步保持原协议。

两者均以当前生产 baseline 为直接对照，不互相叠加。四级流水线、230,400 字节动态
SMEM、网格、TMA 加载、MMA、原容差和 10 warmup / 30 repeat / 五轮 CUDA-event
计时不变。计时前自动对每个新版本做 `(4096,3072,64)` 和 `(4096,3072,320)` 各两次
验算，覆盖短 K、不完整 ring 和跨 tile 的两个 consumer 重用；数值失败立即停止。

本地已在 SM100a/SM103a 对四个评分形状及两个矩形形状检查实际生成代码：
到达指令的 count/remote、warp 同步顺序、源列覆盖、成对寄存器范围、等待/转换顺序，
并比对未改动的全部 producer 和 TMA 写回路径。完整本地工具/源码生成回归
**352 项通过，143.52 s**。随后 GPU 结果见本文最新记录。
当时未修改生产内核，以下为已执行的历史命令：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_writeback.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step10_fused_a.1VXYz2

数据：[summary](results_b300/step10_fused_a.1VXYz2/step10/summary.csv)、
[samples](results_b300/step10_fused_a.1VXYz2/step10/samples.json)、
[run.json](results_b300/step10_fused_a.1VXYz2/step10/run.json)。
运行版本 `b434e45`；B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个 Python 源文件指纹与运行提交一致；六个构建（四个计时、两个边界）的 builder、
变换前/实际编译 CUDA 指纹全部核对通过，NVRTC 参数一致。四个版本均通过初始和逐轮
数值校验；合并 A 版本的两个矩形边界形状均完成两次不计时验算。

### 达标的是已有的缓存与均衡网格组合

| 版本 | 中位数 ms | 最大值 ms | 达标轮次 | 直接对照 | 同轮加速比 | REG / STACK |
|---|---:|---:|---:|---|---:|---:|
| baseline | 0.141246 | 0.142004 | 0/5 | baseline | 1.000× | 168 / 32 |
| cache_tmem_base | 0.139632 | 0.140222 | 0/5 | baseline | 1.013× | 167 / 0 |
| cache_balanced_clusters | **0.137918** | **0.138302** | **5/5** | cache_tmem_base | **1.012×** | 167 / 0 |
| balanced_fused_a | 0.138321 | 0.138784 | 5/5 | cache_balanced_clusters | **0.996×** | 167 / 0 |

门槛为 **0.139100 ms**。均衡网格版最慢样本有 **0.57%** 余量；合并 A 版只有
**0.23%**。合并 A 的五轮均慢于直接对照，因此不采用。SASS 确认变换实际生效：
`UTMALDG` 静态位置由 12 减至 8，出现 3D A 加载；两者均无 `LDL`/`STL`，
`UTCHMMA` 静态位置都是 16。发射数减少并未转化为本次实测收益，停止沿此方向叠加。

三个对照的 cubin 与 `cache_step8_step10.FwqbDd`、`step10_depth3.1zWPNg` 完全相同。
相比紧邻前轮，本轮 baseline 中位数快 **0.83%**，cache-only 快 **1.08%**，
cache+balanced 快 **1.65%**。这说明存在跨运行性能变化，不能把这次 PASS 归因于
新增的合并 A 代码，也不能据五个样本宣称稳定过线。均衡网格相对 cache-only 的收益
在这三轮各五次配对中均存在，但前两轮仍有超限样本。

### 已完成采用：正式验收结果见最新记录

`48d743a` 正式 **Step 10 采用 `cache_balanced_clusters` 的两个改动**：

- 在原 cluster 同步之后缓存一次不可变 TMEM 分配基址，供 MMA 和写回使用。
- 保持原最大每 cluster tile 数，启动足够覆盖所有 tile 的最小网格；4096 时由
  74 个 cluster 改为 64 个，每个处理两个 tile。不是按测试大小硬编码分支。

保留四级 K64 流水线、独立 A 加载、EPI_N64 写回和全部同步、数值运算、评分条件。
4096 生成的 CUDA kernel 与本轮胜出 probe 完全一致；其余评分形状、短 K/矩形复用、
单 tile 在 SM100a/SM103a 重放胜出 builder。profiling 的网格同步更新，历史 trace
从已保存的 `builder.json` 读取原网格，避免把此前 74-cluster 数据当成新网格解读。
probe 默认只测生产 baseline，已采用的缓存和网格变换会拒绝重复应用；依赖旧基线的
组合也应通过对应历史提交重放。

完整本地工具/源码生成回归 **337 项通过，130.41 s**；新生产文件 SHA256 为
`2d2a0df6c80ecc809ece3f509d00ea65d6cb3a7af079e4c6fe9b6481143d4ba1`。
这些检查不替代 GPU 数值与性能验收。

该轮先采用生产改动，再进行 GPU 正式验收；结果见本文最新记录。
当时全量摘要为 55/57，以下为已经执行的历史命令：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_adopt.XXXXXX)
uv run python -m pytest tests/ -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_all.log"
uv run python -u benchmark.py --steps 10 --trials 5 \
  --csv "$tirx_run/step10.csv" --diagnostics-dir "$tirx_run/compiler_step10" \
  2>&1 | tee "$tirx_run/benchmark_step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

全量 pytest 验证 57 个 GPU 用例，benchmark 检查 Step 10 四个评分形状各五轮的余量
并保存正式编译产物。若还有超限，以上两份日志可以区分生产代码是否与胜出版本一致，
以及是否仍受此前观察到的跨运行变化影响。

## 前轮结果：step10_depth3.1zWPNg

数据：[summary](results_b300/step10_depth3.1zWPNg/step10/summary.csv)、
[samples](results_b300/step10_depth3.1zWPNg/step10/samples.json)、
[run.json](results_b300/step10_depth3.1zWPNg/step10/run.json)。
运行版本 `6301533`；B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
七个 Python 文件指纹与运行提交一致，全部 11 份构建记录（五个计时版本、六个边界版本）
的 builder、变换前及实际编译 CUDA 指纹核对通过，各版本编译参数一致。
三个对照的 cubin 均与前轮 `cache_step8_step10.FwqbDd` 完全相同。
所有版本通过初始、逐轮数值校验；六个矩形边界记录均为两次验算通过、不计时。

### 三级与宽写回均未解决剩余失败

| 版本 | 中位数 ms | 最大值 ms | 直接对照 | 同轮加速比 | REG / STACK |
|---|---:|---:|---|---:|---:|
| baseline | 0.142430 | 0.143562 | baseline | 1.000× | 168 / 32 |
| cache_tmem_base | 0.141151 | 0.141371 | baseline | 1.010× | 167 / 0 |
| cache_balanced_clusters | 0.140233 | 0.140822 | cache_tmem_base | 1.007× | 167 / 0 |
| balanced_depth3 | 0.140003 | 0.140631 | cache_balanced_clusters | **1.001×** | 167 / 0 |
| balanced_depth3_epi128 | 0.141946 | 0.142506 | balanced_depth3 | **0.986×** | 168 / 40 |

门槛为 **0.139100 ms，25 个计时样本全部超限**。三级版本中位数仍超门槛 0.65%，
相对均衡网格的同轮收益只有约 0.14%；不能把它相对生产 baseline 的收益全部归给三级。
宽写回比三级对照更慢，SASS 出现 40 字节栈帧、10 处 `LDL` 和 10 处 `STL`；
三级对照无栈帧，两者均有 16 个静态 `UTCHMMA` 位置。因此停止叠加这个宽写回版本，
三级也没有足够收益支持正式采用。生产 Step 10 保持原实现。

Step 8 沿用前轮已验证的缓存基址版本：六项 pytest、20 个 benchmark 样本均通过。
本轮没有新的完整套件结果，最新全量摘要仍为此前 55/57；剩余优化集中在 Step 10 / 4096。

### 已完成实验：合并两个 consumer 的 A 加载

旧角色 trace 的 TMA 区间主要在等待空槽，MMA 也有较长等待；这些重叠区间不能相加，
不能单凭等待比例确定瓶颈。等待提示、循环展开、三级和宽写回实验均没有补上剩余差距。
下一项检验具体的 TMA 发射开销：原实现每 CTA 每级分别加载两个 A 块和一个 B 块，
若两次 A 发射及地址准备限制 stage 周转，合并它们应比直接对照更快；若主要受实际搬运
或其他环节限制，则可能没有收益。3D 描述符也可能改变编译器调度，须核对实际 SASS。

新增独立版本 **`balanced_fused_a`**，以四级 `cache_balanced_clusters` 为直接对照：

- 用 `A.view(M // 256, 256, K)` 表示原连续 A 的别名，不分配、转置或打包输入。
- 两个 consumer 通过一次 3D TMA box `(64,128,2)` 加载；CUDA 维度顺序为
  K、CTA 内行、consumer 块。坐标为 `(k*64, rank*128, tile_m*2)`，对应原地址
  `A[tile_m*512 + consumer*256 + rank*128 + row, k*64 + col]`。
- 每 CTA 每级由三个加载发射变为两个（一次 A、一次 B），A 仍搬运 32,768 字节，
  B 仍搬运 16,384 字节。双 CTA 的 barrier 预期事务总量仍为 98,304 字节。
- 四级输入流水线、230,400 字节动态 SMEM、128 字节 swizzle、MMA、写回、barrier
  和跨 tile phase 均保持直接对照的协议。

该轮默认四个版本为 `baseline`、`cache_tmem_base`、`cache_balanced_clusters`、
`balanced_fused_a`；`vs_control` 指向各自的直接对照，不把缓存或网格的收益归给合并加载。
工具在计时前自动为新版本验算 `(4096,3072,K)`、K=64/320，每个形状运行两次且重填
NaN 输出；覆盖单级、跨四级 ring 的五级 K 循环，以及两个 consumer 的矩形 tile 复用。
仍用原数值容差、10 次 warmup / 30 次 repeat / 五轮交错 CUDA-event 计时。

本地已在 SM100a/SM103a 检查四个评分形状及两个边界形状的 TMA 维度、字节数、
两种 rank/consumer 的首尾地址、共享偏移，并比较所有非加载硬件操作的操作数及 phase。
完整本地工具回归 **312 项通过，137.03 s**。这些是源码生成与工具检查，
GPU 结果见本文最新记录；以下为该轮已完成的历史命令：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_fused_a.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：cache_step8_step10.FwqbDd

数据：[Step 8 pytest](results_b300/cache_step8_step10.FwqbDd/pytest_step08.log)、
[Step 8 benchmark](results_b300/cache_step8_step10.FwqbDd/step08.csv)、
[Step 10 summary](results_b300/cache_step8_step10.FwqbDd/step10/summary.csv)、
[Step 10 samples](results_b300/cache_step8_step10.FwqbDd/step10/samples.json)。
运行版本 `a7b18d8`；B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
benchmark 记录的四个、probe 记录的七个 Python 文件指纹与提交一致，五个 probe
版本的 builder、变换前/实际编译 CUDA 指纹全部核对通过；每组编译参数一致。
所有 probe 通过初始和逐轮数值校验。

### Step 8 正式验证通过

六项 pytest 全过（12.42 s），包括 K=64、320 的矩形不完整 ring 用例。
四个评分形状各五轮 benchmark，**20 个样本全部达标**：

| 大小 | 中位数 ms | 最大值 ms | 门槛 ms | 最慢样本余量 |
|---|---:|---:|---:|---:|
| 1024 | 0.012410 | 0.012585 | 0.018200 | 30.85% |
| 2048 | 0.028816 | 0.028910 | 0.029900 | **3.31%** |
| 4096 | 0.164668 | 0.164718 | 0.171600 | 4.01% |
| 8192 | 1.384891 | 1.385043 | 1.441700 | 3.93% |

2048 正式内核的 cubin 与 `profile_guided.6KUfDZ` 成功缓存 probe 完全相同，
确认改动进入实际运行内核。Step 8 在本轮覆盖的全部用例中通过，可继续保留。
本轮没有新的全量测试；最新完整摘要仍为此前的 55/57，不能直接改写成 56/57。
按已有分步结果，剩余优化集中在 Step 10 / 4096。

### Step 10 均衡网格有收益，仍未稳定达标

| 版本 | 中位数 ms | 最大值 ms | 相对 baseline | 相对 cache-only | 达标轮次 | REG / STACK |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.142347 | 0.143387 | 1.000× | 0.990× | 0/5 | 168 / 32 |
| cache_tmem_base | 0.140948 | 0.141383 | 1.010× | 1.000× | 0/5 | 167 / 0 |
| cache_unroll_ring | 0.141274 | 0.141568 | 1.008× | 0.998× | 0/5 | 168 / 8 |
| cache_mma_no_unroll | 0.140930 | 0.141770 | 1.011× | 0.999× | 0/5 | 167 / 0 |
| cache_balanced_clusters | **0.139947** | **0.140229** | **1.017×** | **1.006×** | **1/5** | 167 / 0 |

比值均为同轮配对比值的中位数。均衡网格五轮都比 cache-only 快，说明这个组合有
重复收益；但中位数仍超 0.139100 ms 门槛 **0.61%**，最慢一轮超 0.81%。
不能挑最快的 0.137492 ms 样本宣称已解决，也不将该组合直接加入正式 Step 10。

实际 SASS 给出两个排除结果：

- `cache_mma_no_unroll` 的静态 `UTCHMMA` 位置从 16 减到 4、`R2UR` 从 103 减到
  45，机器码确实改变；但只有 2/5 轮快于 cache-only，未测到增益。静态指令计数不等于
  动态执行工作量，MMA 总工作量保持一致。
- `cache_unroll_ring` 被编译器继续展开成 256 个静态 `UTCHMMA` 位置，SASS 地址范围
  约为 cache-only 的 3.3 倍，出现 8 字节栈帧和 `LDL`/`STL`；五轮均比 cache-only 慢。
  因而停止沿这两个循环展开版本继续叠加。

### 已完成实验：三级流水线与写回宽度的取舍

目前只差不到 1 微秒，但旧阶段记录显示写回自身仍占角色区间约 8–9%，存在具体
可测的尾部开销。它不证明写回是整个 kernel 的唯一瓶颈。原 Step 10 四级输入流水线
加 64 列写回占 230,400 字节动态 SMEM；直接扩大到 128 列需要 263,168 字节，
不能作为可运行的独立对照。因此用以下分层对照区分容量与写回收益：

1. **`balanced_depth3`**：在已测 `cache_balanced_clusters` 上仅把输入流水线 4→3，
   保持 K64 与 EPI_N64。动态 SMEM 降为 **181,248 字节**。预测：若三层仍足以维持
   stage 周转，延迟损失应较小；若隐藏 TMA 延迟依赖第四层，则会明显变慢。
2. **`balanced_depth3_epi128`**：在上一个版本上仅把写回宽度 64→128，动态 SMEM
   为 **214,016 字节**。预测：若每块两次 warpgroup 同步与 TMA 提交/等待造成尾部
   开销，每个 consumer 的四次写回降为两次后，应快于三级 EPI_N64，并争取越过原门槛。
   保留整行 FP16 暂存，八次 TMEM 读取全部结束后仍先通知 MMA，可保持原重叠机会。

该轮默认五个版本依次为 `baseline`、`cache_tmem_base`、`cache_balanced_clusters`、
`balanced_depth3`、`balanced_depth3_epi128`。显式选择末级版本也会自动带上各级对照。
`comparison_control` / `vs_control` 记录直接对照及同轮比值，避免把前一级的收益
归给后一级；原 baseline、cache-only 比值和原始 PASS/SLOW 规则仍保留。

K4096 共有 64 个 stage，不能整除三层 ring，阶段与 phase 必须跨输出 tile 持续推进。
工具在计时前自动为两个新版本各验算 `(4096,3072,K)`，K=64/192/320，分别覆盖
比 ring 短、一个完整奇数 ring、不完整 ring；每个形状运行两次且重新填 NaN 输出。
矩形共有 96 个 cluster 输出 tile，在均衡的 48-cluster 网格中每个 cluster 复用两次。
这些检查不计时，结果保存于 `verification/`；编译或数值失败会停止运行。

生产 `gemm_kernels.py`、原评分、容差、10 次 warmup / 30 次 repeat / 五轮交错
CUDA-event 计时均不改变。新版本已通过本地 SM100a/SM103a 源码生成和协议检查，
GPU 结果见本文最新记录；以下为该轮已完成的历史命令：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step10_depth3.XXXXXX)
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

本地完整回归：**299 项工具/源码生成检查通过，118.86 s**；不替代 GPU 验证。

## 前轮结果：profile_guided.6KUfDZ 与全量回归

用户最新粘贴的全量结果仍是 **55 passed / 2 failed，74.41 s**，均为性能断言：
Step 8 / 2048 为 572.34 TFLOP/s（约 0.030017 ms，超门槛 0.39%），
Step 10 / 4096 为 957.41 TFLOP/s（约 0.143553 ms，超门槛 3.20%）。
这次摘要未附运行提交与源文件指纹，不把它绑定到某个提交。
本地 `dba5316` 的标题虽为 “finish step 8”，但它只增加日志；直到本轮采用前，
生产内核仍未包含 `cache_tmem_base`。probe 的成功不会自动改变 pytest 测量的代码。

数据：[Step 8 summary](results_b300/profile_guided.6KUfDZ/step08/summary.csv)、
[Step 8 samples](results_b300/profile_guided.6KUfDZ/step08/samples.json)、
[Step 10 summary](results_b300/profile_guided.6KUfDZ/step10/summary.csv)、
[Step 10 samples](results_b300/profile_guided.6KUfDZ/step10/samples.json)。
运行版本 `570b680`，B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
两份 run.json 中各七个 Python 源文件指纹与运行提交一致，全部八个版本的
builder、变换前 CUDA、实际编译 CUDA 指纹核对通过，各 step 内编译选项相同。
全部版本均通过初始和逐轮数值校验。

| Step / 大小 | 版本 | 中位数 ms | 同轮加速比 | 达标轮次 | REG / STACK |
|---|---|---:|---:|---:|---:|
| 8 / 2048 | baseline | 0.029491 | 1.000× | 4/5 | 106 / 0 |
| 8 / 2048 | cache_tmem_base | **0.028849** | **1.024×** | **5/5** | 128 / 0 |
| 8 / 2048 | reuse_wait_64ns | 0.029706 | 0.988× | 4/5 | 106 / 0 |
| 10 / 4096 | baseline | 0.142305 | 1.000× | 0/5 | 168 / 32 |
| 10 / 4096 | cache_tmem_base | **0.140656** | **1.010×** | **0/5** | 167 / 0 |
| 10 / 4096 | reuse_wait_64ns | 0.142170 | 1.001× | 0/5 | 168 / 32 |
| 10 / 4096 | tma_wait_64ns | 0.142100 | 1.002× | 0/5 | 168 / 32 |
| 10 / 4096 | ring_wait_64ns | 0.142229 | 1.000× | 0/5 | 168 / 32 |

门槛分别为 0.029900 / 0.139100 ms。Step 8 缓存赢得 4/5 轮配对比较，
最慢样本 0.029453 ms 仍有约 1.50% 余量。Step 10 缓存赢得全部五轮，但中位数
仍超门槛约 1.12%；等待变体没有实质收益，暂不继续调等待提示。

### 本轮采用与验证边界

`3bb51c9` **仅采用 Step 8 的缓存基址**：在原 CTA 初始化同步之后用 `T.let`
读取一次 TMEM 分配结果，供 MMA 和写回使用；原 64 ns 数据就绪等待、分配释放、
全部 barrier 和数值运算保留。2048 生成 CUDA 与实测缓存版本完全一致，另重放
SM100a / SM103a 的其余评分形状及短 K、矩形形状。Step 8 的 REG 虽增加到 128，
实测仍更快；不能用寄存器数本身代替性能测量。**该生产改动还需要六项 GPU pytest
及四个评分形状 benchmark，不能宣称 Step 8 已稳定全过。**

Step 10 生产内核尚未修改。缓存版本的 SASS 在 MMA 主循环内消除了 TMEM 基址
`LDS`，STACK 32→0，整个 kernel 无 `LDL`/`STL`。初始化和清理处仍有共享读取。
编译器同时把 MMA K 循环展开为四个 stage（16 个静态 `UTCHMMA` 发射位置），
循环内仍有描述符准备和 `R2UR`。这些是下一轮可改变的具体代码路径，尚不能证明
它们占据了剩余的全部耗时。

### 已完成实验：Step 8 正式验证与 Step 10 缓存对照

Step 10 默认五个独立构建：原版 `baseline`、已测 `cache_tmem_base`，及以下三个
分别在缓存基础上只增加一个因素的版本。三者不互相叠加。

1. **`cache_unroll_ring`**：完整 K-ring 使用固定 stage 地址。预测：若动态 stage
   地址和描述符准备仍限制缓存后的循环，固定地址应比 cache-only 更快。未缓存时
   这一变体只有约 0.5% 收益；本轮检验移除基址读取之后是否改变收益。短 K 和不完整
   ring 保持缓存版原路径，phase 仍跨输出 tile 保留。
2. **`cache_mma_no_unroll`**：只给生成 CUDA 的 MMA K 循环加 `#pragma unroll 1`，
   不改变 TMA 循环。预测：若自动四次展开造成描述符同时存活和调度开销，限制展开
   应减少这些指令并缩短耗时；若延迟隐藏依赖展开，则会变慢。须同时检查 SASS
   是否实际改变，不能以源码 pragma 推断编译器结果。
3. **`cache_balanced_clusters`**：保留每个 cluster 的最大输出 tile 数，缩小至足够
   覆盖它们的网格；4096 下为 64 个 cluster，每个两 tile。预测：若缓存后仍受
   74-cluster 网格的部分空闲尾部影响，应改善整次 launch；也可能因并发减少而变慢。
   未缓存时只有约 0.6% 收益，本轮单独检验与缓存的组合。

工具自动为这些组合保留 cache-only 对照，`vs_cache` / `paired_cache_speedup`
报告逐轮相对缓存版的比值中位数；`paired_speedup` 仍相对生产 baseline。
PASS/SLOW 仍只用原始时延门槛。原数值校验、10 次 warmup、30 次 repeat、五轮交错
CUDA-event 计时保持不变；编译输入、diff、资源和可用的 SASS 均保存。
整套本地工具与 CUDA 源码生成检查 **288 passed，106.91 s**；其中新增 14 项覆盖
缓存组合的操作数、phase 与短 K 路径、MMA pragma 重放和配对汇总。本机没有 NVIDIA GPU，
这些检查不替代 GPU 数值和性能验证。详见 [RUNNING.md](RUNNING.md)。

以下为已完成的历史命令，无需重复：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/cache_step8_step10.XXXXXX)
uv run python -m pytest tests/test_step08.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_step08.log"
uv run python -u benchmark.py --steps 8 --trials 5 \
  --csv "$tirx_run/step08.csv" --diagnostics-dir "$tirx_run/compiler_step08" \
  2>&1 | tee "$tirx_run/benchmark_step08.log"
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：stage_profile.LPt75W

数据：[profile.log](results_b300/stage_profile.LPt75W/profile.log)、
[timings.json](results_b300/stage_profile.LPt75W/profile/timings.json)、
[stages.csv](results_b300/stage_profile.LPt75W/profile/stages.csv)、
[trace.csv](results_b300/stage_profile.LPt75W/profile/trace.csv)。运行版本 `4d053c1`；
B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。六个记录的 Python 文件
指纹均与运行提交一致，六个插桩 builder 的指纹核对通过，每组编译选项相同。
八个版本通过初始和逐轮数值校验；**8,960 条记录的完整性和汇总重放通过**。
两个 baseline 的 cubin 均与前轮 VnSWDg 完全相同。

### 性能状态与插桩影响

| 形状 | baseline 中位数 ms | 门槛 ms | 达标样本 | TMA / baseline | MMA / baseline | 写回 / baseline |
|---|---:|---:|---:|---:|---:|---:|
| Step 8 / 2048 | 0.030157 | 0.029900 | 2/5 | 1.010× | 0.996× | 1.021× |
| Step 10 / 4096 | 0.142354 | 0.139100 | 0/5 | 1.000× | 1.017× | 1.023× |

比值为各轮与同轮 baseline 比值的中位数。0.996× 不代表一次优化成功：插桩改变了
调度，且不同版本不是同时测量。Step 8 插桩的 REG 分别为 106 / 128 / 125，
baseline 为 106，STACK 均为 0。Step 10 全部 REG:168；写回插桩的 STACK 从
32 增至 128，其余仍为 32，写回 trace 的精确比例需考虑这一扰动。

所有轮次前后的 `nvidia-smi` 快照均为 P0 / SM 1095 MHz / 显存 3996 MHz；
温度 34–38℃，采样功耗约 199–266 W。这些快照在计时批次外，不能据此排除运行中
的瞬时频率变化，也不能推断整个 kernel 的利用率。cycles/ns 只是另一个区间观察值。
本轮没有新的完整 pytest；全量仍以此前 **55/57** 摘要为准。

### 阶段数据支持什么、尚不能证明什么

以下百分比是相应角色区间内按总时长加权的占比；前后数字分别对应首次 / 第二个
输出 tile。它们不是全 GPU 的 stall 百分比，各角色耗时不能相加。

| 观测 | Step 8 / 2048 | Step 10 / 4096 |
|---|---:|---:|
| TMA 等待 stage 复用 | 78.4% / 80.2% | 89.5–89.9% / 90.1–90.4% |
| MMA 等待输入就绪 | 29.4% / 26.6% | 56.2–56.8% / 54.2–54.8% |
| MMA 等待 accumulator 释放 | 0.4% / 3.2% | 0.1% / 4.5% |
| 写回等待 MMA 完成 | 82.6% / 81.7% | 88.1–88.4% / 88.7–88.8% |
| 写回 epilogue | 13.8% / 14.5% | 9.2–9.4% / 8.2% |
| MMA 未单独计量区间 | 37.7% / 36.6% | 27.1–27.8% / 25.1–25.6% |

Step 10 同一 launch 内两个 consumer 的完成时间差绝对值中位数为 **0 ns**，
95 分位为 128 ns，最大 352 ns（按本轮计时器记录精度）；没有明显的 consumer 失衡证据。它们的完成时间
本来就可能因共用 stage barrier 被耦合，不能由此断言两者每个 stage 都没有差异。

TMA 和 MMA 同时有高等待占比，说明应先检查 K-ring 的交接和发射路径；这不是
“HBM 带宽不足”或“Tensor Core 已满载”的证明。写回自身不是最大的角色区间，
但仍可能贡献尾部开销。MMA 还有约 25–38% 未单独计量，包含循环、地址计算、
计时累加和调度间隙，不能把这部分全部归因于任何一种指令。

进一步检查 [Step 10 baseline SASS](results_b300/stage_profile.LPt75W/profile/step10_4096/baseline/module_01.sass.txt)
发现 MMA K 循环内会重新读取不变的 TMEM 基址：例如 `0x1870` 的 `LDS R2, [R27]`
之后经地址处理，`0x19d0` 的 `R2UR UR5, R2` 将目标地址交给 `0x19f0` 起的四条
`UTCHMMA.2CTA`。后续循环体也有同类读取。生成 CUDA 每条 MMA 的目标均引用
`((uint*)pool_buf_ptr)[0]`，而该分配结果从初始化同步后直到最后 dealloc 都不改变。
这给出一个具体、可验证的缓存机会，但静态指令存在不等于已证明其性能占比。

### 已完成实验：缓存 TMEM 基址与 K-ring 等待对照

本轮生产 `gemm_kernels.py` 和评分不变，`probe_persistent.py` 增加以下对照：

1. **`cache_tmem_base`（Step 8、10）**：在原 CTA/cluster 初始化同步之后用
   `T.let` 读取一次 TMEM 基址，TMEM buffer 引用该标量。若重复 SMEM 读取及地址
   传递限制循环，应减少 K 循环中的 `LDS` 并加速。它也可能增加寄存器活跃期；
   需用新 SASS 确认 NVRTC 是否保留缓存。分配、barrier、MMA 工作量、写回和
   最终 dealloc 均保持原有语义。
2. **`reuse_wait_64ns`（Step 8、10）**：只缩短 TMA warp 在 `mma2tma` 上的
   等待挂起提示。Step 8 已采用 64 ns 数据就绪等待，尚未测试这个复用等待点。
   Step 10 旧 `mma_wait_64ns` 同时改过复用和写回完成等待；本次只改复用，
   两个输出交接等待保持原样。若唤醒响应限制 stage 周转，应带来收益。
3. **`ring_wait_64ns`（仅 Step 10）**：同时缩短 `mma2tma` 和 `tma2mma`，
   用同轮 baseline、reuse-only 和 data-ready-only 作为控制，检验两侧唤醒是否
   存在相互影响。所有等待仍保留 acquire、parity 和 retry，不设置跳过完成的超时。

每个变体从正式内核独立生成；缓存不与等待变体叠加。Step 8 默认三个版本，
Step 10 默认五个版本（含同轮 `tma_wait_64ns` 控制）。不重复无收益的写回块实验。
工具现也保存可用的 SASS，便于检查缓存是否真正消除了循环内读取。

整套 **273 项本地检查通过**，新增 15 项覆盖缓存发布顺序、全部硬件操作及地址等价、短 K/矩形路径、
实测等待代码重放，以及上传 trace/summary 重建。当时 GPU 性能结论待回传；实测结果现见本文开头。

以下为已完成的历史命令，无需重复：

```bash
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/profile_guided.XXXXXX)
uv run python -u probe_persistent.py --steps 8 --size 2048 \
  --output "$tirx_run/step08" 2>&1 | tee "$tirx_run/step08.log"
uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step8_wait_step10_pipeline.VnSWDg

数据：[Step 8 pytest](results_b300/step8_wait_step10_pipeline.VnSWDg/pytest_step08.log)、
[Step 8 benchmark](results_b300/step8_wait_step10_pipeline.VnSWDg/step08.csv)、
[Step 10 summary](results_b300/step8_wait_step10_pipeline.VnSWDg/step10/summary.csv)、
[Step 10 samples](results_b300/step8_wait_step10_pipeline.VnSWDg/step10/samples.json)。
运行版本 `59464cf`；B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
benchmark 的四个、probe 的六个 Python 文件指纹均与运行提交一致；五个 probe
版本的 builder/CUDA 指纹与实际文件一致，编译选项相同。内核 SHA256 为
`20b44b79295eb45d13a97c2e0812690f2a5e20b737c76438359bbf3be782cd2c`。

### Step 8 正式用例通过，但性能余量仍不足

正式 Step 8 **6 项 pytest 全过**，包含 K64 / K320 的不完整流水线用例。
独立 benchmark 的 2048 仍有三轮超时：

| 方阵尺寸 | 中位数 ms | 最慢 ms | 门槛 ms | 达标样本 |
|---|---:|---:|---:|---:|
| 1024 | 0.012432 | 0.012669 | 0.018200 | 5/5 |
| 2048 | 0.029904 | 0.030345 | 0.029900 | **2/5** |
| 4096 | 0.171245 | 0.171543 | 0.171600 | 5/5 |
| 8192 | 1.427212 | 1.427942 | 1.441700 | 5/5 |

2048 的五轮原始耗时依次为 0.029904、0.03034453、0.03005440、0.02944747、
0.02917973 ms。中位数只超出门槛约 4 ns，但最慢样本超出约 1.49%；
4096 最慢样本的余量也仅约 0.033%。不能据单次 pytest 通过认定性能已稳定。

本轮正式 2048 的 [cubin](results_b300/step8_wait_step10_pipeline.VnSWDg/compiler_step08/step08_2048_2048_2048/module_01.cubin)
与前轮成功的 [tma_wait_64ns cubin](results_b300/step810_probe.Wl6HTg/step08/step08_2048_tma_wait_64ns/module_01.cubin)
**逐字节一致**，SHA256 均为
`af1e41be2bba58b520a79840fe14624a4b787d224c9109069ef199f0ca46c998`。
生成 CUDA 主体和等待 helper 也一致（规范化 helper 名称后）。采用改动时没有丢失
已测量的机器码；同一内核跨轮次仍有波动，具体环境原因尚无证据。

### Step 10 流水线及写回实验全部未达标

| Step 10 / 4096 | 中位数 ms | 最快 ms | 最慢 ms | 配对加速比 | REG / STACK |
|---|---:|---:|---:|---:|---:|
| baseline | 0.142486 | 0.140492 | 0.143290 | 1.000× | 168 / 32 |
| unroll_ring | 0.141756 | 0.141548 | 0.142789 | 1.005× | 168 / 24 |
| pipe_depth_2 | 0.173078 | 0.172806 | 0.173213 | 0.823× | 168 / 24 |
| k128_depth_2 | 0.142629 | 0.142249 | 0.143230 | 1.000× | 168 / 24 |
| stream_epilogue | 0.143861 | 0.141907 | 0.144143 | 0.989× | 77 / 0 |

五个版本的 **25 个样本全部超过 0.139100 ms**，初始及逐轮数值验证通过。
展开流水线仅约 0.5% 收益；两级 K64 明显回退，扩大 K 到 128 只恢复至 baseline
附近。分块写回将寄存器从 168 降到 77、栈帧降到 0，却仍变慢，不能把寄存器数或
栈帧当作已定位的主要瓶颈。这里的 REG / STACK 来自实际 cubin 资源报告，
不等于动态 spill 流量。没有采用这些 Step 10 变体。

本轮没有新的完整套件结果；最近一次全量仍是 **55/57 通过**，发生在 Step 8
采用等待改动之前。本轮独立结果不能组合成一次新的全量通过数。

### 当时的阶段诊断设计（已回传 stage_profile.LPt75W）

当时改动只增加 [profile_persistent.py](profile_persistent.py) 和诊断说明，生产内核、
评分门槛、数值容差、原 CUDA-event 计时方法均未修改。已完成的性能变体保留为
`probe_persistent.py --variants ...` 显式选项；当时默认只测正式 baseline。

优先区分以下预测；等待占比只能缩小范围，仍需后续无插桩实验验证：

1. **数据供应限制**：MMA 的 `tma2mma` 等待应占显著时间，而 TMA 等待可复用
   stage 的占比相对较低。
2. **MMA / stage 复用限制**：TMA 的 `mma2tma` 等待应占显著时间；对比两个
   consumer 的 MMA 数据等待和发射区间，检查是否有不平衡。
3. **写回交接限制**：后续输出 tile 的 MMA `ld2mma` 等待应显著，或写回的
   TMEM 读取、转换和 epilogue 区间较长。

工具默认依次采集 Step 8 / 2048 和 Step 10 / 4096。每个形状编译原始 baseline，
以及分别只对 TMA、MMA、写回插桩的三个副本；计时前及每轮后验算。
五轮交错计时、每轮预热 10 次、计时 30 次。原始 baseline 报告达标样本数，
插桩副本只报告与同轮 baseline 的耗时比，不给 PASS/SLOW。

| 角色 | wait | work | handoff / epilogue |
|---|---|---|---|
| TMA | 等待 `mma2tma`，SMEM stage 可复用 | 发射 TMA、登记事务字节、推进 phase | 无 |
| MMA | 等待 `tma2mma`，输入数据就绪 | fence、MMA 发射、commit、推进 phase | handoff：等待 `ld2mma`，TMEM 可复用 |
| 写回 | 等待 `mma2ld`，MMA 完成 | TMEM 读取、转换、fence、释放 accumulator | epilogue：写 SMEM、TMA store 及同步 |

记录按 CTA、consumer、输出 tile 序号分开，初次与复用 tile 分开汇总。
`trace.csv` 和每轮 JSON 保留原始 `%globaltimer` / `%clock64`、SM ID 和 tile 坐标；
`stages.csv` 给出区间中位数/最大值和按区间总时长加权的占比。
每轮 buffer 重置，**trace 对应最后一次计时 launch**；CUDA event 则统计 30 次
launch 的平均耗时，二者不可直接相等。缺失、非法槽位或不完整 tile 覆盖会报错停止。

插桩在原 elected lane 或写回 warp 0 / lane 0 读取计时器，每个输出 tile 结束时
才写入全局记录。额外指令和寄存器、写回 leader 的分支都会扰动调度，应先看插桩/
baseline 比值和实际编译资源。各角色在不同副本中测量、且执行本来存在重叠，
**不能把各角色耗时相加，也不能跨副本对齐时间线**。TMA/MMA 的 work 是指令发射
区间，不是异步引擎完成工作的时长；角色区间不包含初始分配、tile 调度间隙、
末尾全局记录写入和最终 cluster 清理。cycles/ns 仅为观察值，不能替代 GPU 时钟遥测。

编译回调在计时前移除，保留实际 CUDA、编译选项、cubin/fatbin、资源报告和
SASS（若可用 `cuobjdump`）；另保存每轮前后 `nvidia-smi` 时钟、功耗、温度快照。
快照只帮助比较轮次，不能单凭它确定运行中瞬时降频。工具不设置 GPU 时钟。

**258 项本地工具及源码生成检查通过**，其中新增 30 项覆盖插桩保留原硬件操作和地址、
两个 consumer 的独立记录、跨 tile 覆盖、异常 trace 拒绝与重复 helper 定义防护。
本机没有 NVIDIA GPU；工具的 B300 编译、数值和实际扰动已由 LPt75W 回传，见本文开头。

以下为 `4d053c1` 的历史命令，无需重复：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/stage_profile.XXXXXX)
uv run python -u profile_persistent.py --output "$tirx_run/profile" \
  2>&1 | tee "$tirx_run/profile.log"
printf '结果目录：%s\n' "$tirx_run"
```

## 前轮结果：step810_probe.Wl6HTg

数据：[Step 8 summary](results_b300/step810_probe.Wl6HTg/step08/summary.csv)、
[Step 8 samples](results_b300/step810_probe.Wl6HTg/step08/samples.json)、
[Step 10 summary](results_b300/step810_probe.Wl6HTg/step10/summary.csv)、
[Step 10 samples](results_b300/step810_probe.Wl6HTg/step10/samples.json)。
版本 `adf93eb`；B300 / 148 SM / `sm_103a` / TVM 0.26.0 / NVRTC 13.0。
两份 run.json 的六个 Python 文件指纹均与运行提交一致；七个版本的 builder/CUDA
指纹核对通过，每组编译选项相同。所有版本均通过初始和每轮计时后验算。

### Step 8 采用已测量的 TMA 数据就绪等待

| Step 8 / 2048 | 中位数 ms | 最慢 ms | 配对加速比 | 达标样本 |
|---|---:|---:|---:|---:|
| baseline | 0.029859 | 0.030447 | 1.000× | 3/5 |
| **tma_wait_64ns** | **0.029245** | **0.029321** | **1.018×** | **5/5** |
| epilogue_128 | 0.029830 | 0.030849 | 0.998× | 3/5 |

门槛为 0.029900 ms。summary 的 PASS 只表示中位数达标，不能掩盖 baseline 和
epilogue_128 各两次超时。TMA 等待变体五轮都达标，最慢一轮仍有约 **1.94%** 余量；
因此只采用该变体，不放大 epilogue。REG:106 / STACK:0 / 动态 SMEM:148480 字节
与 baseline 相同；epilogue_128 则为 REG:113 / STACK:0 / 动态 SMEM:164864 字节。

正式 Step 8 仅将 `tma2mma` 等待接入局部 `tirx_tma_wait_64ns`，沿用实测的 acquire、
parity 和 retry；其余三个等待点及四级流水线保留。64 ns 为挂起提示，必须等到 barrier
完成才能执行 MMA。先以实测 CUDA 建立失败对照，再采用改动；正式 2048 的 CUDA 主体和
等待 helper 与实测版本一致（只规范化 helper 名称），其余评分尺寸及短 K / 矩形路径
也只改变这一等待点。六个正式用例已由后续 VnSWDg 验收，性能余量仍不足，见本文开头。

### Step 10 三个变体仍不足以解决性能失败

| Step 10 / 4096 | 中位数 ms | 最快 ms | 最慢 ms | 配对加速比 | 达标样本 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.142630 | 0.142403 | 0.143263 | 1.000× | 0/5 |
| tma_wait_64ns | 0.142341 | 0.141940 | 0.142759 | 1.003× | 0/5 |
| specialize_mma | 0.141881 | 0.139570 | 0.142368 | 1.004× | 0/5 |
| specialize_writeback | 0.142587 | 0.140279 | 0.142994 | 1.002× | 0/5 |

门槛 0.139100 ms，最好的单次样本也未达标。MMA 特化将 REG:168 / STACK:32
变成 REG:167 / STACK:0，却只有约 0.4% 配对收益，不能将栈帧视为已定位的主要瓶颈。
写回特化为 REG:168 / STACK:24，数据等待变体为 REG:168 / STACK:32。
这三项均不进入生产 Step 10；最新完整套件状态仍为 **55/57 通过**。

### 当时的 Step 8 验收与 Step 10 流水线对照（现已完成）

当时检验三个方向，Step 10 默认五个版本，包含 baseline 和四项变体：

1. **K-stage 交接频率**：`pipe_depth_2` 仅把 K64 的四级流水线改为两级，作为控制组；
   `k128_depth_2` 在同样两级下扩大 K 到 128，K 不整除 128 时仍用 64。若交接限制性能，
   后者应受益于每输出 tile 64→32 次 K 轮，且胜过两级 K64 控制组。两级 K128 的
   动态 SMEM 为 230400 字节，与正式 K64 四级相同；两级 K64 为 132096 字节。
   K128 同时改变 TMA box 的 swizzle 分解，所以收益也可能来自数据搬运方式，需要实测。
2. **动态流水线索引**：`unroll_ring` 只展开一圈四个 stage，保持 K64、四级、
   230400 字节 SMEM 和两条 MMA warp。phase 每圈翻转并跨输出 tile 保存；
   K 不是 256 的倍数时保留原来的动态流水线路径。若地址计算是瓶颈，应优于 baseline。
3. **写回寄存器占用**：`stream_epilogue` 每读取并转换 64 列就写回，将 FP16 暂存
   从 256 个元素缩至 64。仍是八次 x32 TMEM load、四次 TMA store，SMEM 不变，
   最后一次 TMEM 读取完成后才释放 accumulator。它以推迟下一 tile 的 MMA 为代价
   减少寄存器活跃量；若该代价更大，测量会变慢。它与其他变体不叠加。

当时 **228 项本地工具及源码生成检查通过**，覆盖实测源码重放、两级流水线的完整事务字节、
两 consumer 的 barrier 计数、展开后的 phase 传递和分块写回的释放顺序。
这些 Step 10 变体已由 VnSWDg 完成 B300 编译、数值和性能验证，全部未达标。

以下为 `59464cf` 的历史命令，无需重复：

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/step8_wait_step10_pipeline.XXXXXX)
git log -3 --oneline

uv run python -m pytest tests/test_step08.py -vs --tb=short \
  2>&1 | tee "$tirx_run/pytest_step08.log"

uv run python -u benchmark.py --steps 8 --trials 5 \
  --csv "$tirx_run/step08.csv" --diagnostics-dir "$tirx_run/compiler_step08" \
  2>&1 | tee "$tirx_run/benchmark_step08.log"

uv run python -u probe_persistent.py --steps 10 --size 4096 \
  --output "$tirx_run/step10" 2>&1 | tee "$tirx_run/probe_step10.log"
printf '结果目录：%s\n' "$tirx_run"
```

Step 10 各版本五轮交错计时、每轮预热 10 次、计时 30 次，计时前后验算。
`SLOW` 会完成采集，数值或编译失败则停止。Step 8 已采用的 `tma_wait_64ns`
会拒绝重复应用，应以正式 pytest / benchmark 验收。评分、容差和计时方法不变。

## 前轮结果：k128_step67.TdkZy5 与完整套件摘要

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

### 当时的 Step 8 与 Step 10 独立对照（现已完成）

当时生产内核未改，`probe_persistent.py` 增加以下独立实验，原评分、容差和
CUDA event 方法保留。Step 8 三个版本，Step 10 四个版本，都包含正式 baseline。

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

当时 **198 项本地工具及源码生成检查通过**：新增八份实测 CUDA 的数据就绪等待重放，
确认只改变一个等待点；生成并检查 Step 8 的完整四级流水线，以及 Step 10 两个独立
consumer 槽位、10/11 写回同步 ID、cluster 到达和短 K / 矩形重用路径。
上述七个版本的 GPU 结果已回传为 `step810_probe.Wl6HTg`，见本文开头。

以下为 `adf93eb` 的对照命令，无需重复：

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
该目录已经回传；`SLOW` 正常结束采集，数值或编译失败则停止。

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
