#!/usr/bin/env bash
set -euo pipefail

# HF-only public dataset: retry snapshot_download because individual video
# transfers occasionally reset on this server. The CLI resumes partial files.
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
hf_bin="${HF_BIN:-$asset_root/envs/lingbotvla/bin/hf}"
target="$asset_root/data/RoboTwin-LeRobot"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
clean_list="${CLEAN_LIST:-$repo_root/assets/training_data/robotwin_clean_50.txt}"
mkdir -p "$target" "$asset_root/hf_cache" "$asset_root/tmp"
export HF_HOME="${HF_HOME:-$asset_root/hf_cache}"
export TMPDIR="${TMPDIR:-$asset_root/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# hf-mirror's redirect response is incompatible with the installed hub client's
# mandatory HEAD metadata validation.  Use the official endpoint with longer
# timeouts and one worker; partial files are resumed on the next attempt.
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

[[ -f "$clean_list" ]] || { echo "Clean dataset manifest not found: $clean_list" >&2; exit 1; }
[[ "$(wc -l < "$clean_list")" -eq 50 ]] || { echo "Expected exactly 50 clean dataset entries" >&2; exit 1; }

# `hf download` can return zero after a connection reset while only a prefix of
# the snapshot is materialized.  Require the metadata, parquet trajectory, and
# every camera stream for every manifest entry before accepting an attempt.
clean_data_complete() {
  local robot dataset missing=0
  while read -r robot dataset; do
    if [[ "$robot" != robotwin ]] ||
       [[ ! -s "$dataset/meta/info.json" ]] ||
       [[ ! -s "$dataset/data/chunk-000/file-000.parquet" ]] ||
       [[ ! -s "$dataset/videos/observation.images.cam_high/chunk-000/file-000.mp4" ]] ||
       [[ ! -s "$dataset/videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4" ]] ||
       [[ ! -s "$dataset/videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4" ]]; then
      echo "[clean-data] incomplete: $dataset" >&2
      missing=$((missing + 1))
    fi
  done < "$clean_list"
  if (( missing > 0 )); then
    echo "[clean-data] $missing/50 tasks remain incomplete" >&2
    return 1
  fi
}

for attempt in $(seq 1 12); do
  echo "[clean-data] attempt $attempt/12"
  if "$hf_bin" download SJTU-DENG-Lab/RoboTwin-LeRobot --repo-type dataset \
    --include '*/aloha-agilex_clean_50/**' \
    --local-dir "$target" --max-workers "${HF_MAX_WORKERS:-1}" \
    && clean_data_complete; then
    echo "[clean-data] complete"
    exit 0
  fi
  echo "[clean-data] transient failure; retrying resumably in 10 seconds" >&2
  sleep 10
done
echo "[clean-data] failed after 12 resumable attempts" >&2
exit 1
