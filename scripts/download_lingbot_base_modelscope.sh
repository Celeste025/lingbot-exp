#!/usr/bin/env bash
set -euo pipefail

# Resumable recovery path for LingBot's 26.3 GiB base checkpoint. ModelScope
# mirrors the exact upstream shard hashes and was benchmarked faster/stabler
# than the interrupted HF direct transfer on this server.
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
target="$asset_root/models/lingbot-vla-v2-6b"
base_url="https://modelscope.cn/models/robbyant/lingbot-vla-v2-6b/resolve/master"

download() {
  local relative="$1"
  local expected_sha="${2:-}"
  local target_file="$target/$relative"
  mkdir -p "$(dirname "$target_file")"
  if [[ -n "$expected_sha" && -f "$target_file" ]]; then
    local actual_sha
    actual_sha="$(sha256sum "$target_file" | awk '{print $1}')"
    [[ "$actual_sha" == "$expected_sha" ]] && return
    echo "Incomplete or mismatched $relative; resuming from its local byte offset." >&2
  fi
  echo "[modelscope] $relative"
  curl --fail --location --continue-at - --retry 12 --retry-all-errors \
    --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    --output "$target_file" "$base_url/$relative"
  if [[ -n "$expected_sha" ]]; then
    local actual_sha
    actual_sha="$(sha256sum "$target_file" | awk '{print $1}')"
    [[ "$actual_sha" == "$expected_sha" ]] || {
      echo "SHA256 verification failed for $relative" >&2
      exit 1
    }
  fi
}

for file in \
  config.json model.safetensors.index.json preprocessor_config.json \
  special_tokens_map.json tokenizer.json tokenizer_config.json \
  video_preprocessor_config.json vocab.json depth/model.pt \
  dino_video/config.yaml dino_video/teacher_step_10000.pth; do
  [[ -f "$target/$file" ]] || download "$file"
done

download model-00001-of-00006.safetensors 4afb52b06a13df8b738a156ae5c8196d3bfe6b3ca931cecbd701e44cb9674e45
download model-00002-of-00006.safetensors ec131afa26a340db94c0dba8ec00e990be5f3d842ce6532070f0c8e26a067501
download model-00003-of-00006.safetensors 7dccb068ca66c11fa514476d64661eeead56e2e80e1c0572c9ea82aa0d9ecf27
download model-00004-of-00006.safetensors 1c2cb78066b69ae11255db851df95071e2cd69a3a4bc5020b3fd3b13b17819fb
download model-00005-of-00006.safetensors 8fe36bf1f4f617869954bdfa1ad12e16abd8b0235a5fa99c88d571d4cddf4a17
download model-00006-of-00006.safetensors 3cf613d592dad64b1e2a1b1bb34a6f556a7fa05eacdb9972d73e0ea4882555a0

echo "[modelscope] LingBot base model is complete and checksum-verified."
