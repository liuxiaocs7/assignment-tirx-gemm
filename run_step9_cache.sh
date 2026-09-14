#!/usr/bin/env bash
# Historical comparison: replay at 3a9d486, before Step 9 cache adoption.
# Current production validation: bash run_step9_validate.sh.
set -uo pipefail

tirx_sizes=(1024 2048 4096 8192)
if (( $# == 2 )) && [[ "$1" == --size && "$2" =~ ^(1024|2048|4096|8192)$ ]]; then
  tirx_sizes=("$2")
elif (( $# != 0 )); then
  printf 'Usage: bash run_step9_cache.sh [--size 1024|2048|4096|8192]\n' >&2
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
tirx_run=$(mktemp -d results_b300/step9_cache.XXXXXX) || exit 1
printf '结果目录：%s\n' "$tirx_run"
{
  date -Iseconds
  hostname
  git log -1 --oneline
  git status --short
  printf 'job=%s step=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
  printf 'sizes=%s trials=7 warmup=10 repeat=30\n' "${tirx_sizes[*]}"
} > "$tirx_run/session.txt" 2>&1
uv run python -c 'import torch; i = torch.cuda.current_device(); p = torch.cuda.get_device_properties(i); print("logical_device:", i, "uuid:", getattr(p, "uuid", "unavailable"), "name:", p.name, "SMs:", p.multi_processor_count)' \
  > "$tirx_run/cuda_device.txt" 2>&1
tirx_exit=$?
printf '%s\n' "$tirx_exit" > "$tirx_run/device_exitcode.txt"
if (( tirx_exit != 0 )); then
  cat "$tirx_run/cuda_device.txt" >&2
  exit "$tirx_exit"
fi
nvidia-smi -q > "$tirx_run/gpu_before.txt" 2>&1

for tirx_size in "${tirx_sizes[@]}"; do
  uv run python -u probe_persistent.py --steps 9 --size "$tirx_size" --trials 7 \
    --variants cluster_cache_tmem_base --output "$tirx_run/step9_$tirx_size" \
    2>&1 | tee "$tirx_run/step9_$tirx_size.log"
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
