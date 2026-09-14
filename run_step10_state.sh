#!/usr/bin/env bash
# Diagnostic wrapper for the existing four-way probe; no timing/kernel changes.
set -uo pipefail

if (( $# != 0 )); then
  printf 'Usage: bash run_step10_state.sh\n' >&2
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
tirx_run=$(mktemp -d results_b300/step10_share_a_state.XXXXXX) || exit 1
printf '结果目录：%s\n' "$tirx_run"
{
  date -Iseconds
  hostname
  git log -1 --oneline
  printf 'job=%s step=%s job_gpus=%s step_gpus=%s visible=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${SLURM_JOB_GPUS:-}" \
    "${SLURM_STEP_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"
} > "$tirx_run/session.txt" 2>&1

# CUDA logical index 0 need not be nvidia-smi physical index 0 under Slurm.
uv run python -c 'import torch; i = torch.cuda.current_device(); p = torch.cuda.get_device_properties(i); print("logical_device:", i, "uuid:", getattr(p, "uuid", "unavailable"), "name:", p.name, "SMs:", p.multi_processor_count)' \
  > "$tirx_run/cuda_device.txt" 2>&1
tirx_device_exit=$?
if (( tirx_device_exit != 0 )); then
  cat "$tirx_run/cuda_device.txt" >&2
  printf '%s\n' "$tirx_device_exit" > "$tirx_run/device_exitcode.txt"
  exit "$tirx_device_exit"
fi

nvidia-smi -q > "$tirx_run/gpu_before.txt" 2>&1
tirx_gpu_pid=''
tirx_process_pid=''
cleanup() {
  local tirx_pid
  for tirx_pid in "$tirx_gpu_pid" "$tirx_process_pid"; do
    if [[ -n "$tirx_pid" ]]; then
      kill "$tirx_pid" 2>/dev/null || true
      wait "$tirx_pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Match nvidia-smi UUIDs to cuda_device.txt, not logical CUDA indices.
nvidia-smi \
  --query-gpu=timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu \
  --format=csv,nounits --loop-ms=200 \
  > "$tirx_run/gpu_samples.csv" 2> "$tirx_run/gpu_samples.err" &
tirx_gpu_pid=$!
nvidia-smi \
  --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,nounits --loop-ms=1000 \
  > "$tirx_run/compute_processes.csv" 2> "$tirx_run/compute_processes.err" &
tirx_process_pid=$!

# Log receipt times are coarse context, not kernel start/end timestamps.
uv run python -u probe_persistent.py --steps 10 --size 4096 --trials 7 \
  --variants tmem_share_a_depth6 --output "$tirx_run/step10" 2>&1 \
  | python3 -u -c 'import datetime, sys
for line in sys.stdin:
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    print("[" + stamp + "] " + line, end="", flush=True)' \
  | tee "$tirx_run/step10.log"
tirx_pipe_status=("${PIPESTATUS[@]}")
tirx_probe_exit=${tirx_pipe_status[0]}
printf '%s\n' "$tirx_probe_exit" > "$tirx_run/probe_exitcode.txt"
printf 'probe=%s timestamp=%s tee=%s\n' "${tirx_pipe_status[@]}" \
  > "$tirx_run/pipeline_exitcodes.txt"
nvidia-smi -q > "$tirx_run/gpu_after.txt" 2>&1

for tirx_capture in gpu_samples compute_processes; do
  if [[ ! -s "$tirx_run/$tirx_capture.csv" || -s "$tirx_run/$tirx_capture.err" ]]; then
    printf '状态采集可能不完整：请检查 %s/%s.csv 和 .err\n' "$tirx_run" "$tirx_capture" >&2
  fi
done
printf '结果目录：%s；probe 退出码：%s\n' "$tirx_run" "$tirx_probe_exit"
tirx_exit=$tirx_probe_exit
if (( tirx_exit == 0 )); then
  for tirx_code in "${tirx_pipe_status[@]:1}"; do
    if (( tirx_code != 0 )); then
      tirx_exit=$tirx_code
      break
    fi
  done
fi
exit "$tirx_exit"
