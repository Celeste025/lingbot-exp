#!/usr/bin/env bash
set -euo pipefail

# Official clean-only 8k run: smoke, train, and archive concurrently.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
export CLEAN_LIST="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
export NORM_STATS="${NORM_STATS:-$asset_root/norm_stats/robotwin_official_clean_v21.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$asset_root/outputs/competition_official_clean_8k}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29600}"

bash "$repo_root/scripts/download_robotwin_clean_data.sh"

available_bytes="$(df --output=avail -B1 "$asset_root" | tail -1 | tr -d ' ')"
minimum_bytes=$((140 * 1024 * 1024 * 1024))
if (( available_bytes < minimum_bytes )); then
  echo "Insufficient disk for rolling checkpoints: available=$available_bytes required=$minimum_bytes" >&2
  exit 1
fi

# Other users may claim the selected GPUs while data is being prepared.
# Start smoke only when all four are actually free, and recheck before training.
wait_for_gpus() {
  local mem_csv busy
  while true; do
    mem_csv="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
    busy="$(awk -F, -v selected="$CUDA_VISIBLE_DEVICES" 'BEGIN { n=0; split(selected, ids, ","); for (i in ids) wanted[ids[i]]=1 } { gsub(/ /, "", $1); gsub(/ /, "", $2); if (($1 in wanted) && $2>1024) n++ } END {print n}' <<< "$mem_csv")"
    [[ "$busy" == 0 ]] && return 0
    echo "[official-clean] waiting for GPUs $CUDA_VISIBLE_DEVICES to be free ($busy busy)"
    sleep 60
  done
}
wait_for_gpus
bash "$repo_root/scripts/compute_competition_clean_norm_4gpu.sh"
wait_for_gpus
smoke_dir="$asset_root/outputs/competition_official_clean_smoke_$(date +%Y%m%d_%H%M%S)"
echo "[official-clean] running 2-step smoke: $smoke_dir"
bash "$repo_root/scripts/run_competition_clean_4gpu.sh" \
  --train.output_dir "$smoke_dir" \
  --train.micro_batch_size 32 \
  --train.gradient_accumulation_steps 2 \
  --train.global_batch_size 256 \
  --train.max_steps 2 \
  --train.save_steps 0 \
  --train.save_epochs 0 \
  --train.save_final_checkpoint false
echo "[official-clean] smoke passed"
wait_for_gpus

checkpoint_root="$OUTPUT_DIR/checkpoints"
mkdir -p "$checkpoint_root" "$OUTPUT_DIR/archive_logs"
bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$checkpoint_root" --watch --poll-seconds 30 --rolling-two \
  >"$OUTPUT_DIR/archive_logs/watch.log" 2>&1 &
archive_pid=$!
cleanup() {
  kill "$archive_pid" 2>/dev/null || true
  wait "$archive_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "[official-clean] training 8k steps; archive watcher PID=$archive_pid"
bash "$repo_root/scripts/run_competition_clean_4gpu.sh" \
  --train.micro_batch_size 32 \
  --train.gradient_accumulation_steps 2 \
  --train.global_batch_size 256 \
  --train.max_steps 8000 \
  --train.save_steps 2000 \
  --train.full_state_save_steps 4000 \
  --train.save_epochs 0 \
  --train.rolling_checkpoint_retention true

# The watcher may still be verifying the final checkpoint when training exits.
while [[ ! -f "$OUTPUT_DIR/archive_logs/verified_full_step8000.sha256" ]]; do
  kill -0 "$archive_pid" 2>/dev/null || {
    echo "Archive watcher exited before verifying step 8000" >&2
    exit 1
  }
  sleep 30
done
bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$checkpoint_root" --rolling-two
[[ -d "$checkpoint_root/global_step_6000/hf_ckpt" &&
   ! -d "$checkpoint_root/global_step_6000/optimizer" &&
   -d "$checkpoint_root/global_step_8000/optimizer" &&
   -f "$OUTPUT_DIR/archive_logs/verified_inference_step6000.sha256" ]] || {
  echo "Final rolling checkpoints are incomplete" >&2
  exit 1
}
echo "[official-clean] 8k training and remote archive complete"
