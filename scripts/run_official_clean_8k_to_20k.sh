#!/usr/bin/env bash
set -euo pipefail

# Continue the verified official-clean run from its latest full-state checkpoint.
# The absolute LR segment starts at step 8000: 8e-5 -> 4e-5 at step 20000.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
export CLEAN_LIST="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
export NORM_STATS="${NORM_STATS:-$asset_root/norm_stats/robotwin_official_clean_v21.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$asset_root/outputs/competition_official_clean_8k}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29600}"
checkpoint_root="$OUTPUT_DIR/checkpoints"
check_only=false
if [[ "${1:-}" == "--check" && $# -eq 1 ]]; then
  check_only=true
elif (( $# != 0 )); then
  echo "Usage: $0 [--check]" >&2
  exit 2
fi

[[ -s "$CLEAN_LIST" && -s "$NORM_STATS" ]] || { echo "Clean data list or norm stats missing" >&2; exit 1; }
schema_digest="$({
  sha256sum "$CLEAN_LIST" "$NORM_STATS" "$repo_root/configs/robot_configs/robotwin.yaml"
  while read -r robot data_path; do
    sha256sum "$data_path/meta/info.json"
  done < "$CLEAN_LIST"
} | sha256sum | cut -d' ' -f1)"
[[ -f "$OUTPUT_DIR/archive_logs/verified_full_step8000.sha256" ]] || {
  echo "Verified remote full-state step 8000 is required" >&2; exit 1;
}
[[ -s "$OUTPUT_DIR/robotwin_data.sha256" ]] || { echo "Training data provenance missing" >&2; exit 1; }
[[ "$(cat "$OUTPUT_DIR/robotwin_data.sha256")" == "$schema_digest" ]] || {
  echo "Clean data, robot mapping, or normalization stats changed since step 8000" >&2; exit 1;
}

mapfile -t full_checkpoints < <(
  find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -print |
    while IFS= read -r path; do
      [[ -d "$path/optimizer" && -d "$path/extra_state" && -d "$path/model" ]] || continue
      step="${path##*global_step_}"
      [[ "$step" =~ ^[0-9]+$ ]] && printf '%012d %s\n' "$step" "$path"
    done | sort -nr
)
(( ${#full_checkpoints[@]} > 0 )) || { echo "No local full-state checkpoint to resume" >&2; exit 1; }
resume_step="$((10#${full_checkpoints[0]%% *}))"
resume_checkpoint="${full_checkpoints[0]#* }"
(( resume_step >= 8000 && resume_step <= 20000 )) || {
  echo "Unexpected resume step: $resume_step" >&2; exit 1;
}
if (( resume_step == 20000 )); then
  echo "Step 20000 is already available: $resume_checkpoint"
  exit 0
fi

available_bytes="$(df --output=avail -B1 "$asset_root" | tail -1 | tr -d ' ')"
minimum_bytes=$((90 * 1024 * 1024 * 1024))
(( available_bytes >= minimum_bytes )) || {
  echo "Insufficient free disk for the next rolling checkpoint: $available_bytes bytes" >&2; exit 1;
}
IFS=, read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || { echo "Specify exactly four GPU IDs" >&2; exit 1; }

echo "[continue] checkpoint=$resume_checkpoint; GPUs=$CUDA_VISIBLE_DEVICES; free_bytes=$available_bytes"
echo "[continue] LR: 8e-5 at step 8000, cosine to 4e-5 at step 20000"
if "$check_only"; then
  exit 0
fi

wait_for_gpus() {
  local busy
  while true; do
    busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v selected="$CUDA_VISIBLE_DEVICES" 'BEGIN {split(selected, ids, ","); for (i in ids) wanted[ids[i]]=1} {gsub(/ /,"",$1); gsub(/ /,"",$2); if (($1 in wanted) && $2>1024) n++} END {print n+0}')"
    [[ "$busy" == 0 ]] && return 0
    echo "[continue] waiting for GPUs $CUDA_VISIBLE_DEVICES ($busy busy)"
    sleep 60
  done
}
wait_for_gpus

mkdir -p "$OUTPUT_DIR/archive_logs"
bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$checkpoint_root" --watch --poll-seconds 30 --rolling-two \
  >"$OUTPUT_DIR/archive_logs/watch_8k_to_20k.log" 2>&1 &
archive_pid=$!
cleanup() {
  kill "$archive_pid" 2>/dev/null || true
  wait "$archive_pid" 2>/dev/null || true
}
trap cleanup EXIT

bash "$repo_root/scripts/run_competition_clean_4gpu.sh" \
  --train.load_checkpoint_path "$resume_checkpoint" \
  --train.micro_batch_size 32 \
  --train.gradient_accumulation_steps 2 \
  --train.global_batch_size 256 \
  --train.max_steps 20000 \
  --train.lr 1e-4 \
  --train.lr_min 4e-5 \
  --train.lr_decay_style cosine_resume \
  --train.lr_anneal_start_step 8000 \
  --train.lr_anneal_start 8e-5 \
  --train.save_steps 2000 \
  --train.full_state_save_steps 4000 \
  --train.save_epochs 0 \
  --train.rolling_checkpoint_retention true

while [[ ! -f "$OUTPUT_DIR/archive_logs/verified_full_step20000.sha256" ]]; do
  kill -0 "$archive_pid" 2>/dev/null || { echo "Archive watcher exited before verifying step 20000" >&2; exit 1; }
  sleep 30
done
bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$checkpoint_root" --rolling-two
[[ -d "$checkpoint_root/global_step_18000/hf_ckpt" &&
   ! -d "$checkpoint_root/global_step_18000/optimizer" &&
   -d "$checkpoint_root/global_step_20000/optimizer" &&
   -f "$OUTPUT_DIR/archive_logs/verified_inference_step18000.sha256" ]] || {
  echo "Final rolling checkpoints are incomplete" >&2
  exit 1
}
echo "[continue] step 20000 training and remote archive complete"
