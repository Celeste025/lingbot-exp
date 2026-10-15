#!/usr/bin/env bash
set -euo pipefail

# End-to-end guarded pipeline. It never starts formal training until clean
# assets, normalization, four idle GPUs, and the 20-step smoke artifacts pass.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
pipeline_log_dir="$asset_root/pipeline_logs"
smoke_base="$asset_root/outputs/smoke_clean_4gpu"
formal_dir="$asset_root/outputs/competition_clean_4gpu"
mkdir -p "$pipeline_log_dir"

export PATH="$env_prefix/bin:$PATH"
export HF_HUB_DISABLE_XET=1
export HF_HOME="$asset_root/hf_cache"
export TMPDIR="$asset_root/tmp"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

wait_for_existing_base_download() {
  while pgrep -f 'bash scripts/download_lingbot_base_modelscope.sh' >/dev/null; do
    echo "[pipeline] waiting for active LingBot base download"
    sleep 30
  done
}

download_moge() {
  local hf_bin="$env_prefix/bin/hf"
  local target="$asset_root/models/moge-2-vitb-normal"
  for attempt in $(seq 1 8); do
    if "$hf_bin" download Ruicheng/moge-2-vitb-normal --local-dir "$target" --max-workers 1; then
      return
    fi
    echo "[pipeline] MoGe download retry $attempt/8" >&2
    sleep 10
  done
  return 1
}

validate_clean_list() {
  local list="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
  python3 "$repo_root/scripts/validate_robotwin_joint_data.py" "$list"
  local robot dataset
  read -r robot dataset < "$list"
  [[ "$robot" == robotwin && -d "$dataset" ]] || return 1
  python3 "$repo_root/scripts/validate_robotwin_official_clean.py" "$dataset"
}

select_four_idle_gpus() {
  local ids
  while true; do
    ids="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2 + 0 < 1024 {gsub(/ /, "", $1); print $1}' | head -4 | paste -sd, -)"
    if [[ "$(awk -F, '{print NF}' <<<"$ids")" -eq 4 ]]; then
      echo "$ids"
      return
    fi
    echo "[pipeline] fewer than four idle GPUs; waiting 60 seconds" >&2
    sleep 60
  done
}

verify_smoke() {
  local smoke_dir="$1"
  [[ -s "$smoke_dir/checkpoints/loss.jsonl" ]] || { echo "Smoke loss log missing" >&2; return 1; }
  find "$smoke_dir" -type f -name 'events.out.tfevents.*' -print -quit | grep -q . || {
    echo "Smoke TensorBoard event missing" >&2; return 1;
  }
  find "$smoke_dir" -type f \( -name '.metadata' -o -name '*.distcp' -o -name '*.safetensors' \) -print -quit | grep -q . || {
    echo "Smoke checkpoint artifact missing" >&2; return 1;
  }
}

cd "$repo_root"
wait_for_existing_base_download
bash scripts/download_lingbot_base_modelscope.sh
bash scripts/download_qwen_modelscope.sh
bash scripts/download_robotwin_clean_data.sh
download_moge
validate_clean_list

gpu_ids="$(select_four_idle_gpus)"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
echo "[pipeline] selected GPUs: $CUDA_VISIBLE_DEVICES"
bash scripts/compute_competition_clean_norm_4gpu.sh |& tee "$pipeline_log_dir/norm_$(date +%Y%m%d_%H%M%S).log"

gpu_ids="$(select_four_idle_gpus)"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
echo "[pipeline] smoke GPUs: $CUDA_VISIBLE_DEVICES"
bash scripts/run_competition_smoke_4gpu.sh |& tee "$pipeline_log_dir/smoke_$(date +%Y%m%d_%H%M%S).log"

selection_file="$(find "$smoke_base" -maxdepth 1 -type f -name '*.selection.env' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$selection_file" && -f "$selection_file" ]] || { echo "Smoke selection file missing" >&2; exit 1; }
smoke_dir="${selection_file%.selection.env}"
verify_smoke "$smoke_dir"

gpu_ids="$(select_four_idle_gpus)"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
echo "[pipeline] formal GPUs: $CUDA_VISIBLE_DEVICES"
SMOKE_SELECTION_FILE="$selection_file" OUTPUT_DIR="$formal_dir" \
  bash scripts/run_competition_formal_4gpu.sh |& tee "$pipeline_log_dir/formal_$(date +%Y%m%d_%H%M%S).log"
