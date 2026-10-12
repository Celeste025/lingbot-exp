#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
output_dir="${OUTPUT_DIR:-$asset_root/outputs/competition_clean_4gpu}"
config="$repo_root/configs/vla/robotwin/competition_clean_4gpu.yaml"
data_list="$repo_root/assets/training_data/robotwin_clean_50.txt"
norm_stats="${NORM_STATS:-$asset_root/norm_stats/robotwin_clean_50.json}"

for path in \
  "$data_list" \
  "$norm_stats" \
  "$asset_root/models/lingbot-vla-v2-6b" \
  "$asset_root/models/Qwen3-VL-4B-Instruct" \
  "$asset_root/models/moge-2-vitb-normal/model.pt" \
  "$asset_root/models/lingbot-vla-v2-6b/depth/model.pt" \
  "$asset_root/models/lingbot-vla-v2-6b/dino_video/config.yaml" \
  "$asset_root/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -x "$env_prefix/bin/python" ]] || { echo "Missing training environment: $env_prefix" >&2; exit 1; }

mkdir -p "$output_dir"
export PATH="$env_prefix/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export HF_HOME="${HF_HOME:-$asset_root/hf_cache}"

cd "$repo_root"
exec bash train.sh tasks/vla/train_lingbotvla.py "$config" \
  --data.train_path "$data_list" \
  --data.norm_stats_file "$norm_stats" \
  --train.output_dir "$output_dir" \
  "$@"
