#!/usr/bin/env bash
# Compare grid count, then finer shared-B tiles, then input depth (eight balanced trials).
set -uo pipefail

if (( $# != 0 )); then
  printf 'Usage: bash run_step10_geometry.sh\n' >&2
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
tirx_run=$(mktemp -d results_b300/step10_geometry.XXXXXX) || exit 1
printf '结果目录：%s\n' "$tirx_run"
{
  date -Iseconds
  hostname
  git log -1 --oneline
  git status --short
  printf 'job=%s step=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
  printf 'size=4096 variants=baseline,tmem_max_clusters,tmem_n64,tmem_n64_depth5 l2_group=8 trials=8 warmup=10 repeat=30 seed=0\n'
} > "$tirx_run/session.txt" 2>&1
uv run python -c 'import hashlib, pathlib, torch; i = torch.cuda.current_device(); p = torch.cuda.get_device_properties(i); print("gemm_sha256:", hashlib.sha256(pathlib.Path("gemm_kernels.py").read_bytes()).hexdigest()); print("logical_device:", i, "uuid:", getattr(p, "uuid", "unavailable"), "name:", p.name, "SMs:", p.multi_processor_count)' \
  > "$tirx_run/cuda_device.txt" 2>&1
tirx_exit=$?
printf '%s\n' "$tirx_exit" > "$tirx_run/device_exitcode.txt"
if (( tirx_exit != 0 )); then
  cat "$tirx_run/cuda_device.txt" >&2
  exit "$tirx_exit"
fi
nvidia-smi -q > "$tirx_run/gpu_before.txt" 2>&1
uv run python -u probe_persistent.py --steps 10 --size 4096 --trials 8 \
  --warmup 10 --repeat 30 --seed 0 \
  --variants tmem_n64_depth5 \
  --output "$tirx_run/step10_4096" 2>&1 | tee "$tirx_run/step10_4096.log"
tirx_status=("${PIPESTATUS[@]}")
printf '%s\n' "${tirx_status[0]}" > "$tirx_run/probe_exitcode.txt"
printf 'probe=%s tee=%s\n' "${tirx_status[@]}" > "$tirx_run/pipeline.txt"
tirx_exit=${tirx_status[0]}
if (( tirx_exit == 0 )); then tirx_exit=${tirx_status[1]}; fi
nvidia-smi -q > "$tirx_run/gpu_after.txt" 2>&1
printf '结果目录：%s；综合退出码：%s\n' "$tirx_run" "$tirx_exit"
exit "$tirx_exit"
