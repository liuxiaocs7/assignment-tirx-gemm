#!/usr/bin/env bash
# Focused reproduction on the current allocation. Never retry until success.
# Formal kernel, timer, verification and performance thresholds are unchanged.
set -uo pipefail

if (( $# != 0 )); then
  printf 'Usage: bash run_step10_diagnose.sh\n' >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
for tirx_tool in uv nvidia-smi python3; do
  if ! command -v "$tirx_tool" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$tirx_tool" >&2
    exit 1
  fi
done
mkdir -p results_b300 || exit 1
tirx_run=$(mktemp -d results_b300/step10_diagnose.XXXXXX) || exit 1
printf '结果目录：%s\n' "$tirx_run"
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  hostname
  git rev-parse HEAD
  git status --short
  printf 'job=%s step=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
} > "$tirx_run/session.txt" 2>&1

# CUDA logical device 0 can differ from nvidia-smi physical index 0.
# Record UUID and relevant compiler flags without dumping the environment.
uv run python -c 'import hashlib, json, pathlib, torch, tvm
from benchmark_diagnostics import run_metadata
p = torch.cuda.get_device_properties(torch.cuda.current_device())
data = run_metadata()
data.update(logical_device=torch.cuda.current_device(), uuid=str(getattr(p, "uuid", "unavailable")), gpu=p.name, SMs=p.multi_processor_count, torch=torch.__version__, tvm=tvm.__version__, cuda=torch.version.cuda)
for name in ("tests/test_step10.py", "run_step10_diagnose.sh", "tools/summarize_benchmark_stability.py"):
    data[name + "_sha256"] = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
print(json.dumps(data, indent=2))' > "$tirx_run/device_and_source.json" 2>&1
tirx_device_exit=$?
printf '%s\n' "$tirx_device_exit" > "$tirx_run/device_exitcode.txt"
if (( tirx_device_exit != 0 )); then
  cat "$tirx_run/device_and_source.json" >&2
  exit "$tirx_device_exit"
fi

tirx_failed=0
trap 'exit 130' INT
trap 'exit 143' TERM
run_logged() {
  local tirx_label=$1
  shift
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$tirx_run/${tirx_label}_started.txt"
  "$@" 2>&1 | tee "$tirx_run/$tirx_label.log"
  local tirx_status=("${PIPESTATUS[@]}")
  printf 'command=%s tee=%s\n' "${tirx_status[@]}" > "$tirx_run/${tirx_label}_pipeline.txt"
  local tirx_code=${tirx_status[0]}
  if (( tirx_code == 0 )); then tirx_code=${tirx_status[1]}; fi
  printf '%s\n' "$tirx_code" > "$tirx_run/${tirx_label}_exitcode.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$tirx_run/${tirx_label}_finished.txt"
  if (( tirx_code != 0 )); then tirx_failed=1; fi
  return "$tirx_code"
}

# Snapshots are outside timing; they do not establish in-kernel clock stability.
# No continuous nvidia-smi polling, clock changes, profiler or candidate kernel.
snapshot() {
  local tirx_label=$1
  nvidia-smi -q > "$tirx_run/$tirx_label.txt" 2>&1
  local tirx_code=$?
  printf '%s\n' "$tirx_code" > "$tirx_run/${tirx_label}_exitcode.txt"
  if (( tirx_code != 0 )); then
    printf '设备快照失败：%s/%s.txt\n' "$tirx_run" "$tirx_label" >&2
    tirx_failed=1
  fi
}
snapshot gpu_before
for tirx_trial in 1 2 3; do
  run_logged "pytest_$tirx_trial" uv run python -m pytest \
    'tests/test_step10.py::test_multi_consumer[4096]' -vs --tb=short
  snapshot "gpu_after_pytest_$tirx_trial"
done
run_logged benchmark uv run python -u benchmark.py --steps 10 --sizes 4096 --trials 8 \
  --csv "$tirx_run/step10.csv" --diagnostics-dir "$tirx_run/compiler_step10"
snapshot gpu_after
run_logged summary python3 tools/summarize_benchmark_stability.py \
  "$tirx_run/step10.csv" --expected-trials 8
printf '%s\n' "$tirx_failed" > "$tirx_run/overall_exitcode.txt"
printf '结果目录：%s；诊断综合退出码：%s\n' "$tirx_run" "$tirx_failed"
printf '非零表示测试、采集管道或逐样本检查失败；请保留全部结果，勿只选通过轮次。\n'
exit "$tirx_failed"
