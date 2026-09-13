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
| 8 | 将 Step 7 流水线扩为四级 | 教程在后续 cluster 中使用四级流水线 |
| 9 | 双 CTA 协作，cluster 输出 256×256 | 教程 Step 8 |
| 10 | 两个 MMA consumer 共享 B，cluster 输出 512×256 | 教程 Step 9 |

本机已用 TVM 0.26.0 完成全部 10 个 step 的 TIR 构建、lowering 和 CUDA 源码生成，
覆盖 SM100a / SM103a，以及短 K、矩形和不完整流水线。
最新用户回传的完整套件摘要为 **55 passed / 2 failed，共 57 项**，失败均为性能断言：
Step 8 / 2048、Step 10 / 4096。Step 1–7、9 全部通过，Step 10 的其他用例通过。
最新 `step8_wait_step10_pipeline.VnSWDg`（`59464cf`）中，Step 8 正式 **6 项 pytest 全过**；
但 2048 的 benchmark 五轮只有两轮达标，中位数 0.029904 ms，门槛 0.029900 ms。
4096 五轮全过，最慢样本余量仅约 0.033%。正式 2048 与前轮成功等待变体的 cubin
完全相同，确认改动已采用，但性能还不稳定。Step 10 的流水线和分块写回实验共
25 个样本全部超时，未采用。最新全量摘要在 Step 8 改动之前，不能据独立结果更新通过数。

`k128_step67.TdkZy5`（`ade5040`）已确认 Step 6、7 正式 **16 项全过**，8 个评分形状
各五轮 benchmark、合计 40 个样本全部通过，64/128 两种 K 宽度的边界用例也通过。
Step 10 / 4096 的 x64 TMEM、L2 分组、cluster 数实验均未解决性能失败。
详见 [最新验证和对照记录](B300_VALIDATION.md)。

不依赖 GPU 的工具测试可单独运行：`uv run python -m pytest tool_tests/ -q`
（覆盖构建、实测 CUDA 重放、fallback、实验隔离、编译回调与角色插桩，共 **258 项本地通过**；
依赖 TVM 的用例在没有 TVM 时跳过）。

主文件保持自包含，作业提交仍只需要 `gemm_kernels.py`。新增测试专门覆盖短 K、
奇数个 K tile、矩形输出和常驻 CTA 的跨 tile 重用；原有 37 个正确性与性能用例没有降低标准。

### 当前进度：测量 Step 8、10 各角色的等待时间

将新提交同步到服务器后，运行下面一个工具，依次测量 Step 8 / 2048、Step 10 / 4096。
每个形状包含正式 baseline 和分别对 TMA、MMA、写回插桩的三个副本；五轮交错计时，
计时前后验算。阶段数据用于区分数据供应、MMA/stage 复用和写回交接瓶颈。

```bash
cd ~/assignment-tirx-gemm
mkdir -p results_b300
set -o pipefail
tirx_run=$(mktemp -d results_b300/stage_profile.XXXXXX)
uv run python -u profile_persistent.py --output "$tirx_run/profile" \
  2>&1 | tee "$tirx_run/profile.log"
printf '结果目录：%s\n' "$tirx_run"
```

`trace.csv` 保存各 CTA、consumer、输出 tile 的原始时间戳；`stages.csv` 汇总等待、
发射/读取、交接及 epilogue 区间。trace 来自每轮**最后一次计时 launch**，CUDA event
则是 30 次 launch 的平均。工具保留 baseline 比值、builder diff、CUDA、实际二进制、
编译资源、可用的 SASS，以及每轮前后的 GPU 时钟/功耗快照。

插桩会改变指令、寄存器和调度；各角色在独立副本中测量，不能叠加其耗时或对齐其时间线。
TMA/MMA 的 work 是发射区间，不是异步引擎完成时长；初始分配、tile 调度间隙和末尾清理
不在角色区间内。插桩副本不参与性能评分，只有 baseline 报告达标样本数；采集成功返回 0，
数值、编译或 trace 校验失败返回非零。本地已验证源码生成，NVRTC 与 GPU 执行待回传。
完整解释见 [阶段诊断说明](B300_VALIDATION.md#下一轮按角色测量等待与执行区间)。

`probe_persistent.py` 默认只测 baseline，已完成的变体通过 `--variants` 显式选择。
已采用的 Step 6/7 `k_tile_128` 和 Step 8 `tma_wait_64ns` 会拒绝重复应用。

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

# 完整验收：57 个 GPU 用例，包含 20 个额外边界用例
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
性能门槛仍使用 B200 参考值，B300 和新版 TVM 的实际性能需要上机测量。
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
`hgemm_v6`–`hgemm_v10` 时默认 `SM_COUNT=148`（B200）。
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

当前共有 **57 个 GPU 用例**。其中原有 37 个用例依次执行：

1. 编译并运行 TIRX 内核。
2. 与 `torch.matmul(A, B.T)` 比较，要求 `rtol=1e-3, atol=1e-2`。
3. 预热 10 次，CUDA event 测量 30 次，要求平均耗时不超过参考值的 `1.30` 倍。

其余 20 个新增边界用例只检查正确性，没有任意新增性能门槛。
默认随机种子是 0；通过后可换种子检查稳定性：

```bash
GEMM_TEST_SEED=1 python -m pytest tests/test_step09.py tests/test_step10.py -xvs
```

需要区分“正确但慢”与“计算不正确”：`Submission too slow` 是性能断言，
`Tensor-likes are not close` 才是数值检查失败。不要为了通过用例而放宽误差容限或改写参考时间。

## 5. 测量性能、比较 cuBLAS、导出 CSV

`benchmark.py` 默认先检查正确性，每个形状只编译一次，再重复计时。
输出每次测量的中位耗时、TFLOP/s、同形状 cuBLAS 耗时、相对 cuBLAS 的速度比和评分门槛。

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
step       M       N       K   median_ms   TFLOP/s   cuBLAS_ms   vs_cuBLAS   limit_ms   status
```

- `vs_cuBLAS = cuBLAS_ms / median_ms`，大于 1 表示本内核在当前测量下更快。
- `PASS`：正确性通过且耗时在参考门槛内；`SLOW`：正确但慢；`UNSCORED`：自定义形状没有参考门槛。
- 有 `SLOW` 时脚本退出码为 1，CSV 仍保留。数值验证失败则立即报错。
- CSV 包含 GPU / SM 数、TVM / PyTorch / CUDA 版本、种子、计时配置、各 trial 的原始耗时及最小/最大值。
- 日志和 CSV 记录 commit、工作区是否有修改、内核/工具源码 SHA256、目标架构和编译参数。

计算公式为 `TFLOP/s = 2*M*N*K / (time_ms*1e-3) / 1e12`。
默认计时方式与原测试保持一致：当前 CUDA stream 上的 event 包围重复 kernel launch，
不包含编译和输入创建。Python 发射间隙仍可能影响小矩阵结果；它不是 CUDA Graph 或峰值吞吐测试。
`benchmark.py` 报告多轮中位数，pytest 使用单轮平均值，最终作业验收以 pytest 为准。

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

持续偏慢时，先采集原有编译过程的资源信息；下例只测当前仍有失败的步骤：

```bash
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
- 仅性能失败：检查 GPU 占用、功耗/时钟、是否 B200、是否开启同步调试或 sanitizer，再重复测量。

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
