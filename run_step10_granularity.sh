#!/usr/bin/env bash
# Shared-B input granularity and capacity with explicit controls; eight balanced trials.
set -uo pipefail

if (( $# != 0 )); then
  printf 'Usage: bash run_step10_granularity.sh\n' >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
for tirx_tool in uv nvidia-smi; do
  if ! command -v "$tirx_tool" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$tirx_tool" >&2
    exit 1
  fi
done
mkdir -p results_b300 || exit 1
tirx_run=$(mktemp -d results_b300/step10_granularity.XXXXXX) || exit 1
printf '结果目录：%s\n' "$tirx_run"
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  hostname
  git rev-parse HEAD
  git status --short
  printf 'job=%s step=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
  printf 'size=4096 variants=baseline,tmem_k64_sw64,tmem_k32_depth8,tmem_k32_depth10 trials=8 warmup=10 repeat=30 seed=0\n'
} > "$tirx_run/session.txt" 2>&1
uv run python -c 'import hashlib, json, os, pathlib, torch
p = torch.cuda.get_device_properties(torch.cuda.current_device())
print(json.dumps(dict(uuid=str(p.uuid), gpu=p.name, SMs=p.multi_processor_count,
    cpu_affinity=sorted(os.sched_getaffinity(0)),
    gemm_sha256=hashlib.sha256(pathlib.Path("gemm_kernels.py").read_bytes()).hexdigest()), indent=2))' \
  > "$tirx_run/device.json" 2>&1
tirx_device_exit=$?
printf '%s\n' "$tirx_device_exit" > "$tirx_run/device_exitcode.txt"
if (( tirx_device_exit != 0 )); then
  cat "$tirx_run/device.json" >&2
  exit "$tirx_device_exit"
fi
tirx_failed=0
snapshot() {
  local tirx_label=$1
  nvidia-smi -q > "$tirx_run/$tirx_label.txt" 2>&1
  local tirx_code=$?
  printf '%s\n' "$tirx_code" > "$tirx_run/${tirx_label}_exitcode.txt"
  if (( tirx_code != 0 )); then tirx_failed=1; fi
}
snapshot gpu_before
uv run python -u probe_persistent.py --steps 10 --size 4096 --trials 8 \
  --warmup 10 --repeat 30 --seed 0 \
  --variants tmem_k32_depth10 \
  --output "$tirx_run/step10_4096" 2>&1 | tee "$tirx_run/step10_4096.log"
tirx_status=("${PIPESTATUS[@]}")
printf '%s\n' "${tirx_status[0]}" > "$tirx_run/probe_exitcode.txt"
printf 'probe=%s tee=%s\n' "${tirx_status[@]}" > "$tirx_run/pipeline.txt"
if (( tirx_status[0] != 0 || tirx_status[1] != 0 )); then tirx_failed=1; fi
snapshot gpu_after
printf '%s\n' "$tirx_failed" > "$tirx_run/overall_exitcode.txt"
printf '结果目录：%s；综合退出码：%s\n' "$tirx_run" "$tirx_failed"
printf '退出码 0 表示实验采集完成；是否提速和达标请看全部原始样本与 summary.csv。\n'
exit "$tirx_failed"
