#!/usr/bin/env bash
set -euo pipefail

# Capacity probe followed by the requested 20-step end-to-end smoke test.
# Every candidate preserves global batch 256 on four GPUs.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
smoke_base="${SMOKE_OUTPUT_BASE:-$asset_root/outputs/smoke_clean_4gpu}"
run_id="$(date +%Y%m%d_%H%M%S)"
probe_steps="${PROBE_STEPS:-2}"
candidates=(32 16 8 4 2 1)

mkdir -p "$smoke_base"
selected_micro_batch=""
for micro_batch in "${candidates[@]}"; do
  accumulation=$((256 / (4 * micro_batch)))
  probe_dir="$smoke_base/${run_id}_capacity_mb${micro_batch}"
  echo "[probe] micro_batch=$micro_batch accumulation=$accumulation global_batch=256"
  set +e
  bash "$repo_root/scripts/run_competition_clean_4gpu.sh" \
    --train.output_dir "$probe_dir" \
    --train.micro_batch_size "$micro_batch" \
    --train.gradient_accumulation_steps "$accumulation" \
    --train.global_batch_size 256 \
    --train.max_steps "$probe_steps" \
    --train.save_steps 0 \
    --train.save_epochs 0 \
    --train.save_final_checkpoint false \
    >"$probe_dir.launcher.log" 2>&1
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    selected_micro_batch="$micro_batch"
    break
  fi
  echo "[probe] micro_batch=$micro_batch failed (exit=$status); trying next candidate. Log: $probe_dir.launcher.log" >&2
done

if [[ -z "$selected_micro_batch" ]]; then
  echo "No candidate micro batch completed the capacity probe." >&2
  exit 1
fi

selected_accumulation=$((256 / (4 * selected_micro_batch)))
final_dir="$smoke_base/${run_id}_mb${selected_micro_batch}_20step"
mkdir -p "$final_dir"
printf 'MICRO_BATCH_SIZE=%s\nGRADIENT_ACCUMULATION_STEPS=%s\nGLOBAL_BATCH_SIZE=256\n' \
  "$selected_micro_batch" "$selected_accumulation" > "$final_dir.selection.env"
echo "[smoke] selected micro_batch=$selected_micro_batch accumulation=$selected_accumulation"

exec bash "$repo_root/scripts/run_competition_clean_4gpu.sh" \
  --train.output_dir "$final_dir" \
  --train.micro_batch_size "$selected_micro_batch" \
  --train.gradient_accumulation_steps "$selected_accumulation" \
  --train.global_batch_size 256 \
  --train.max_steps 20 \
  --train.save_steps 20 \
  --train.save_epochs 0 \
  "$@"
