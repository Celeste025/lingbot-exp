#!/usr/bin/env bash
set -euo pipefail

# Resumable Qwen dependency download from the verified ModelScope mirror.
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
target="$asset_root/models/Qwen3-VL-4B-Instruct"
base_url="https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct/resolve/master"

download() {
  local relative="$1" expected_sha="${2:-}" target_file="$target/$1"
  mkdir -p "$(dirname "$target_file")"
  if [[ -n "$expected_sha" && -f "$target_file" ]]; then
    [[ "$(sha256sum "$target_file" | awk '{print $1}')" == "$expected_sha" ]] && return
    echo "Incomplete or mismatched $relative; resuming from its local byte offset." >&2
  fi
  echo "[modelscope] $relative"
  curl --fail --location --continue-at - --retry 12 --retry-all-errors \
    --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    --output "$target_file" "$base_url/$relative"
  if [[ -n "$expected_sha" ]]; then
    [[ "$(sha256sum "$target_file" | awk '{print $1}')" == "$expected_sha" ]] || {
      echo "SHA256 verification failed for $relative" >&2
      exit 1
    }
  fi
}

for file in config.json generation_config.json chat_template.json merges.txt \
  model.safetensors.index.json preprocessor_config.json tokenizer.json \
  tokenizer_config.json video_preprocessor_config.json vocab.json; do
  [[ -f "$target/$file" ]] || download "$file"
done
download model-00001-of-00002.safetensors 30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9
download model-00002-of-00002.safetensors 046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6
echo "[modelscope] Qwen dependency is complete and checksum-verified."
