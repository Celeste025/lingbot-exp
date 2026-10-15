#!/usr/bin/env bash
set -euo pipefail

# Official Tianchi/RoboTwin 2.0 Aloha-AgileX clean LeRobot v2.1 export.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
target="$asset_root/data/RoboTwin2.0-official"
archive="$target/lerobot_dataset/RoboTwin_lerobot_v21.zip"
dataset="$target/RoboTwin_lerobot_v21"
clean_list="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_official_clean_v21.txt}"
revision="3dc3b798668feb99ac61cc9086d84cbcc3d79186"
expected_sha="98a1d4bfb51d6e90469c9ef8d55e004a2cba01d20374adb68b554e48e9fe2cd8"
extract_marker="$target/.v21_extracted.sha256"

[[ -x "$env_prefix/bin/hf" ]] || { echo "Missing Hugging Face CLI: $env_prefix/bin/hf" >&2; exit 1; }
mkdir -p "$target" "$asset_root/hf_cache" "$asset_root/tmp" "$(dirname "$clean_list")"
export HF_HOME="${HF_HOME:-$asset_root/hf_cache}"
export TMPDIR="${TMPDIR:-$asset_root/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"

if [[ ! -f "$extract_marker" || "$(cat "$extract_marker")" != "$expected_sha" ||
      -z "$(find "$dataset" -type f -path "*/meta/info.json" -print -quit 2>/dev/null)" ]]; then
  if [[ ! -f "$archive" ]] ||
     [[ "$(sha256sum "$archive" | cut -d' ' -f1)" != "$expected_sha" ]]; then
    for attempt in $(seq 1 12); do
      echo "[official-clean] download attempt $attempt/12"
      if "$env_prefix/bin/hf" download TianxingChen/RoboTwin2.0 \
        lerobot_dataset/RoboTwin_lerobot_v21.zip --repo-type dataset \
        --revision "$revision" --local-dir "$target" --max-workers 1; then
        break
      fi
      (( attempt < 12 )) || { echo "Official clean data download failed" >&2; exit 1; }
      sleep 10
    done
  fi
  echo "$expected_sha  $archive" | sha256sum --check --status || {
    echo "Official data archive SHA256 mismatch" >&2
    exit 1
  }
  unzip -tq "$archive" >/dev/null
  unzip -q -o "$archive" -d "$target"
  [[ -n "$(find "$dataset" -type f -path "*/meta/info.json" -print -quit 2>/dev/null)" ]] || { echo "Expected dataset metadata missing: $dataset" >&2; exit 1; }
  printf '%s\n' "$expected_sha" > "$extract_marker"
fi

python3 "$repo_root/scripts/validate_robotwin_official_clean.py" "$dataset" --write-list "$clean_list"
python3 "$repo_root/scripts/validate_robotwin_joint_data.py" "$clean_list"
echo "[official-clean] validated official 2500 clean episodes: $dataset"
