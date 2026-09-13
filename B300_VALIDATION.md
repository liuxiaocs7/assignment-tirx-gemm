# B300 验证记录与性能诊断

## 最新实测：dfc9065

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

## 下一轮：同时拿到耗时与实际编译日志

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
- `nvrtc_01.log`：包括成功编译时的 ptxas 资源日志，可查看寄存器数量、stack frame、spill load/store。
- `nvrtc_version.json`：NVRTC 编译器版本；它不必等于 PyTorch 显示的 CUDA 版本。
- `capture.json`：编译回调次数。若 `modules` 为 0，则本轮未捕获编译，不能判断无溢出。

TVM 0.26 原本不打印成功的 NVRTC 编译日志。采集钩子只在编译和第一次调用期间启用，
复用原编译回调与全部参数，在正确性验证、预热、计时前恢复。它不会换编译器或改变寄存器预算。
工具在本机通过调用透明性、异常恢复、源码指纹和全部 TIR/CUDA 源码构建检查（共 60 项），
本机无 NVIDIA GPU，实际 NVRTC/运行采集仍需服务器验证。

先回传 `benchmark.log` 与这些资源日志即可：

```bash
rg -n 'registers|spill|stack frame|smem|warning' "$tirx_run/compiler" -g '*.log'
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
