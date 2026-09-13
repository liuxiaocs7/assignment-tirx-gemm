# Blackwell GEMM：安装、测试与性能测量

## 1. 当前实现与验证范围

`gemm_kernels.py` 已实现 Step 1–10，每个 Step 独立提交。实现参考
[Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/)
的 GEMM 基础、异步优化和高级优化章节，并适配本作业使用的旧版 TIRX API。
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

当前只完成本机静态检查及不依赖 GPU 的工具测试；**未实测 GPU 编译、数值正确性或性能**。
以下命令是上机验收流程，不能把文档中的参考值视为本实现的实测成绩。

不依赖 GPU 的性能 CLI 测试可单独运行：`python -m pytest tool_tests/ -q`（18 个用例）。

主文件保持自包含，作业提交仍只需要 `gemm_kernels.py`。新增测试专门覆盖短 K、
奇数个 K tile、矩形输出和常驻 CTA 的跨 tile 重用；原有 37 个正确性与性能用例没有降低标准。

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

建议使用 **Linux x86_64 + NVIDIA B200（SM100）+ CUDA Toolkit 13.0**。
CUDA 13.0 通常需要 **580 系列或更新的 NVIDIA 驱动**；如果平台提供经过配置的
CUDA compatibility 环境，以平台说明为准。Toolkit 中需要 `nvcc` / `ptxas`。
B100 也属于 SM100，但本仓库性能门槛取自 B200，不能保证 B100 达到同样分数。
B300（SM103）也已放行测试入口，会使用实际 SM 数；尚未在 B300 上验证编译与运行，性能门槛仍使用 B200 参考值。
A100、H100、RTX 4090/5090 和 Mac GPU 不适用这些 SM100 内核。

```bash
nvidia-smi
nvcc --version
command -v ptxas
python3 --version
```

使用 Python 3.10–3.12，推荐 3.12，在项目根目录安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --pre -U -f https://mlc.ai/wheels "mlc-ai-tirx-cu130==0.0.1b2"
python -m pip install "torch==2.9.1+cu130" --index-url https://download.pytorch.org/whl/cu130
python -m pip install pytest numpy
python -m pip install --force-reinstall "apache-tvm-ffi==0.1.9"
python -m pip check
```

如果 `mlc.ai/wheels` 无法访问，固定版本 wheel 的同源 GitHub Release 地址是：

```bash
python -m pip install "https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_tirx_cu130-0.0.1b2-py3-none-manylinux_2_28_x86_64.whl"
# 然后继续安装上面的 PyTorch / pytest / numpy，并最后固定 apache-tvm-ffi。
```

不要在同一个环境里安装教程最新版的 `apache-tvm==0.26.0`：它与作业的
`Tx.kernel()`、`Tx.PoolAllocator()`、`tvm.tirx.pipeline` 等 API 不兼容。

显式选择空闲 GPU，再检查依赖：

```bash
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch
import tvm
from tvm.script import tirx as Tx
from tvm.tirx.pipeline import PipelineState
from tvm.tirx.op_schedule.cuda.common import tma_shared_layout

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) in {(10, 0), (10, 3)}
device = torch.cuda.get_device_properties(0)
print('TVM:', tvm.__version__)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', device.name, 'SM count:', device.multi_processor_count)
print('TIRX imports OK')
PY
```

pytest 和 `benchmark.py` 会按实际 GPU 调整常驻 CTA 数量；直接使用
`hgemm_v6`–`hgemm_v10` 时默认 `SM_COUNT=148`（B200）。

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

当前共有 **49 个 GPU 用例**。其中原有 37 个用例依次执行：

1. 编译并运行 TIRX 内核。
2. 与 `torch.matmul(A, B.T)` 比较，要求 `rtol=1e-3, atol=1e-2`。
3. 预热 10 次，CUDA event 测量 30 次，要求平均耗时不超过参考值的 `1.30` 倍。

其余 12 个新增边界用例只检查正确性，没有任意新增性能门槛。
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
- CSV 包含 GPU / SM 数、TVM / PyTorch / CUDA 版本、种子、计时配置、各 trial 的最小/最大耗时。

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
python -m pip freeze > results/packages.txt
```

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

- ImportError：确认使用 `0.0.1b2` 和 `apache-tvm-ffi==0.1.9`，没有混装新版 TVM。
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

Modal runner 已固定 CUDA 13.0、TIRX `0.0.1b2`、PyTorch `2.9.1+cu130` 和 FFI `0.1.9`。
它运行 pytest 并输出每个评分用例的耗时与 TFLOP/s；CSV 性能脚本按上面的服务器方式运行。

## 8. 作业打包

在所有 GPU 测试与性能检查完成后：

```bash
tar cvf handin.tar gemm_kernels.py
tar tvf handin.tar
```

归档应只包含 `gemm_kernels.py`。
