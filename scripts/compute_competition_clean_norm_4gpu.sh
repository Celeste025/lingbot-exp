#!/usr/bin/env bash
set -euo pipefail

# Recompute normalization from the permitted 50 x 50 clean trajectories only.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
data_list="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
norm_path="${NORM_STATS:-$asset_root/norm_stats/robotwin_official_clean_v21.json}"
work_dir="${NORM_WORK_DIR:-$asset_root/outputs/norm_clean_4gpu}"

[[ -f "$data_list" ]] || { echo "Missing data list: $data_list" >&2; exit 1; }
[[ -x "$env_prefix/bin/python" ]] || { echo "Missing training environment: $env_prefix" >&2; exit 1; }
while read -r robot data_path; do
  [[ "$robot" == "robotwin" && -d "$data_path" ]] || {
    echo "Missing or invalid clean dataset entry: $robot $data_path" >&2
    exit 1
  }
done < "$data_list"
python3 "$repo_root/scripts/validate_robotwin_joint_data.py" "$data_list"

mkdir -p "$(dirname "$norm_path")" "$work_dir"
export PATH="$env_prefix/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29601}"
export HF_HOME="${HF_HOME:-$asset_root/hf_cache}"

cd "$repo_root"
exec bash train.sh scripts/compute_norm_stats.py configs/vla/norm_compute/post_data.yaml \
  --data.robot_name robotwin \
  --data.train_path "$data_list" \
  --data.norm_path "$norm_path" \
  --data.num_workers "${NORM_WORKERS:-4}" \
  --train.micro_batch_size "${NORM_MICRO_BATCH_SIZE:-16}" \
  --train.output_dir "$work_dir"
