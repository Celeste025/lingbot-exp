#!/usr/bin/env bash
set -euo pipefail

# Start the 16,000-step clean-only submission training after smoke validation.
# The smoke selection file records a capacity-tested per-GPU micro batch.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
selection_file="${SMOKE_SELECTION_FILE:?Set SMOKE_SELECTION_FILE to the successful smoke .selection.env file}"
output_dir="${OUTPUT_DIR:-$asset_root/outputs/competition_clean_4gpu}"

[[ -f "$selection_file" ]] || { echo "Missing smoke selection: $selection_file" >&2; exit 1; }
# shellcheck disable=SC1090
source "$selection_file"
[[ "${GLOBAL_BATCH_SIZE:-}" == "256" ]] || { echo "Smoke selection has unexpected global batch" >&2; exit 1; }
[[ "${MICRO_BATCH_SIZE:-}" =~ ^(1|2|4|8|16|32)$ ]] || { echo "Invalid selected micro batch" >&2; exit 1; }
[[ "${GRADIENT_ACCUMULATION_STEPS:-}" =~ ^[0-9]+$ ]] || { echo "Invalid selected accumulation" >&2; exit 1; }
(( MICRO_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS == 256 )) || {
  echo "Smoke selection does not preserve global batch 256" >&2
  exit 1
}

# Every 2k checkpoint is model-only; every 4k checkpoint additionally stores
# optimizer state. Older completed checkpoints are archived and verified on
# ModelScope by a background monitor before their local copies are removed.
smoke_dir="${selection_file%.selection.env}"
smoke_checkpoint="$smoke_dir/checkpoints/global_step_20"
[[ -d "$smoke_checkpoint" ]] || { echo "Smoke checkpoint missing: $smoke_checkpoint" >&2; exit 1; }
available_bytes="$(df --output=avail -B1 "$asset_root" | tail -1 | tr -d ' ')"
minimum_bytes=$((100 * 1024 * 1024 * 1024))
if (( available_bytes < minimum_bytes )); then
  echo "Insufficient disk for model-only saves plus upload staging: available=$available_bytes required=$minimum_bytes" >&2
  exit 1
fi

echo "[formal] micro_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=256"
if [[ -d "$output_dir/checkpoints" ]] &&
   compgen -G "$output_dir/checkpoints/global_step_*" >/dev/null &&
   [[ ! -f "$output_dir/robotwin_data.sha256" ]]; then
  echo "Existing checkpoints have no data provenance: $output_dir. Use a fresh OUTPUT_DIR for joint-space training." >&2
  exit 1
fi
if [[ -f "$output_dir/robotwin_data.sha256" ]]; then
  data_list="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
  norm_stats="${NORM_STATS:-$asset_root/norm_stats/robotwin_official_clean_v21.json}"
  current_digest="$({
    sha256sum "$data_list" "$norm_stats" "$repo_root/configs/robot_configs/robotwin.yaml"
    while read -r robot data_path; do
      sha256sum "$data_path/meta/info.json"
    done < "$data_list"
  } | sha256sum | cut -d' ' -f1)"
  if [[ "$(cat "$output_dir/robotwin_data.sha256")" != "$current_digest" ]]; then
    echo "Dataset, robot config, or norm stats differ from existing run: $output_dir. Use a fresh OUTPUT_DIR." >&2
    exit 1
  fi
fi
mkdir -p "$output_dir/checkpoints"
bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$output_dir/checkpoints" --watch --poll-seconds 120 &
archive_pid=$!
cleanup_archive_monitor() {
  kill "$archive_pid" 2>/dev/null || true
  wait "$archive_pid" 2>/dev/null || true
}
trap cleanup_archive_monitor EXIT INT TERM

set +e
bash "$repo_root/scripts/run_competition_clean_4gpu.sh" \
  --train.output_dir "$output_dir" \
  --train.micro_batch_size "$MICRO_BATCH_SIZE" \
  --train.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --train.global_batch_size 256 \
  --train.max_steps 16000 \
  --train.save_steps 2000 \
  --train.full_state_save_steps 4000 \
  --train.save_epochs 0 \
  "$@"
train_status=$?
set -e

# The trainer waits for its asynchronous HF export before returning. Give the
# archiver one final synchronous pass so all recent checkpoints are safe in
# ModelScope. Retain the final full-state checkpoint locally for immediate
# evaluation while also preserving it remotely.
bash "$repo_root/scripts/archive_checkpoints_modelscope.sh" \
  --checkpoint-root "$output_dir/checkpoints" \
  --include-latest-full --keep-latest-full
exit "$train_status"
