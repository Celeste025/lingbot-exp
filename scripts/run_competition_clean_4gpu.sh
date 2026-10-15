#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
output_dir="${OUTPUT_DIR:-$asset_root/outputs/competition_clean_4gpu}"
config="$repo_root/configs/vla/robotwin/competition_clean_4gpu.yaml"
data_list="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
norm_stats="${NORM_STATS:-$asset_root/norm_stats/robotwin_official_clean_v21.json}"

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
python3 "$repo_root/scripts/validate_robotwin_joint_data.py" "$data_list"

# A checkpoint can load even when its actions were learned from a different
# schema. Bind each run to its dataset metadata, joint mapping, and norm stats.
schema_digest="$({
  sha256sum "$data_list" "$norm_stats" "$repo_root/configs/robot_configs/robotwin.yaml"
  while read -r robot data_path; do
    sha256sum "$data_path/meta/info.json"
  done < "$data_list"
} | sha256sum | cut -d' ' -f1)"
provenance_file="$output_dir/robotwin_data.sha256"
if [[ -d "$output_dir/checkpoints" ]] &&
   compgen -G "$output_dir/checkpoints/global_step_*" >/dev/null &&
   [[ ! -f "$provenance_file" ]]; then
  echo "Existing checkpoints have no data provenance: $output_dir. Use a fresh OUTPUT_DIR for joint-space training." >&2
  exit 1
fi
if [[ -f "$provenance_file" && "$(cat "$provenance_file")" != "$schema_digest" ]]; then
  echo "Dataset, robot config, or norm stats differ from existing run: $output_dir. Use a fresh OUTPUT_DIR." >&2
  exit 1
fi

mkdir -p "$output_dir"
printf '%s\n' "$schema_digest" > "$provenance_file"
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
