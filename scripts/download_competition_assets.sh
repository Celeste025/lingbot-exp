#!/usr/bin/env bash
set -euo pipefail

# Download exactly the public assets needed for the Tianchi clean-only run.
# Set HF_ENDPOINT to a mirror before invoking if direct Hugging Face is slow.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
hf_bin="${HF_BIN:-$asset_root/envs/lingbotvla/bin/hf}"

[[ -x "$hf_bin" ]] || { echo "Missing Hugging Face CLI: $hf_bin" >&2; exit 1; }
mkdir -p "$asset_root/models" "$asset_root/data" "$asset_root/hf_cache" "$asset_root/tmp"
export HF_HOME="${HF_HOME:-$asset_root/hf_cache}"
export TMPDIR="${TMPDIR:-$asset_root/tmp}"
# The server's Xet route was measured at an unusably low throughput here;
# regular CDN ranged requests are materially faster and support resuming.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

download() {
  echo "[download] $1"
  shift
  "$hf_bin" download "$@" --local-dir "$1" --max-workers "${HF_MAX_WORKERS:-4}"
}

echo "[download] LingBot-VLA2 base model"
"$hf_bin" download robbyant/lingbot-vla-v2-6b \
  --local-dir "$asset_root/models/lingbot-vla-v2-6b" \
  --max-workers "${HF_MAX_WORKERS:-4}"

echo "[download] Qwen3-VL tokenizer/backbone"
"$hf_bin" download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$asset_root/models/Qwen3-VL-4B-Instruct" \
  --max-workers "${HF_MAX_WORKERS:-4}"

echo "[download] MoGe depth estimator"
"$hf_bin" download Ruicheng/moge-2-vitb-normal \
  --local-dir "$asset_root/models/moge-2-vitb-normal" \
  --max-workers "${HF_MAX_WORKERS:-4}"

echo "[download] official RoboTwin 2.0 LeRobot clean-only dataset"
bash "$repo_root/scripts/download_robotwin_clean_data.sh"

echo "[download] complete"
