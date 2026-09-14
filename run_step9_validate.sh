#!/usr/bin/env bash
# Validate the adopted Step 9 cache with all GPU tests and the formal benchmark.
set -uo pipefail

if (( $# != 0 )); then
  printf 'Usage: bash run_step9_validate.sh\n' >&2
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
tirx_run=$(mktemp -d results_b300/step9_adopt.XXXXXX) || exit 1
printf '结果目录：%s\n' "$tirx_run"
{
  date -Iseconds
  hostname
  git log -1 --oneline
  git status --short
  printf 'job=%s step=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
} > "$tirx_run/session.txt" 2>&1
uv run python -c 'import hashlib, pathlib, torch; p = torch.cuda.get_device_properties(torch.cuda.current_device()); print("gemm_sha256:", hashlib.sha256(pathlib.Path("gemm_kernels.py").read_bytes()).hexdigest()); print("uuid:", getattr(p, "uuid", "unavailable"), "name:", p.name, "SMs:", p.multi_processor_count)' \
  > "$tirx_run/cuda_device.txt" 2>&1
tirx_exit=$?
printf '%s\n' "$tirx_exit" > "$tirx_run/device_exitcode.txt"
if (( tirx_exit != 0 )); then
  cat "$tirx_run/cuda_device.txt" >&2
  exit "$tirx_exit"
fi
nvidia-smi -q > "$tirx_run/gpu_before.txt" 2>&1
tirx_failed=0
run_logged() {
  local tirx_label=$1
  shift
  "$@" 2>&1 | tee "$tirx_run/$tirx_label.log"
  local tirx_status=("${PIPESTATUS[@]}")
  printf 'command=%s tee=%s\n' "${tirx_status[@]}" > "$tirx_run/${tirx_label}_pipeline.txt"
  local tirx_code=${tirx_status[0]}
  if (( tirx_code == 0 )); then tirx_code=${tirx_status[1]}; fi
  printf '%s\n' "$tirx_code" > "$tirx_run/${tirx_label}_exitcode.txt"
  if (( tirx_code != 0 )); then tirx_failed=1; fi
  return "$tirx_code"
}

run_logged pytest_all uv run python -m pytest tests/ -vs --tb=short
run_logged benchmark_step9 uv run python -u benchmark.py --steps 9 --trials 7 \
  --csv "$tirx_run/step9.csv" --diagnostics-dir "$tirx_run/compiler_step9"
nvidia-smi -q > "$tirx_run/gpu_after.txt" 2>&1
printf '结果目录：%s；综合退出码：%s\n' "$tirx_run" "$tirx_failed"
exit "$tirx_failed"
