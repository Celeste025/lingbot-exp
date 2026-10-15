#!/usr/bin/env bash
# Sequential RobotWin evaluation sweep.  Exactly one remotely archived model is
# kept locally at a time, which keeps the 100+ GB evaluation volume bounded.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
sim_root="${ROBOTWIN_ROOT:-/home/wjq/workspace/lingbo/RoboTwin-legacy}"
env_prefix="${LINGBOT_ENV_PREFIX:-$asset_root/envs/lingbotvla}"
sim_env="${ROBOTWIN_ENV_PREFIX:-$asset_root/envs/robotwin}"
repo_id="${MODELSCOPE_CHECKPOINT_REPO:-Celeste025/lingbot-vla2-robotwin-clean50-checkpoints}"
output_base="${EVAL_OUTPUT_BASE:-$asset_root/evaluations}"
cache_root="${EVAL_MODEL_CACHE:-$output_base/model_cache}"
gpu_ids="${EVAL_GPU_IDS:-4,5,6,7}"
wait_pid="${WAIT_FOR_PID:-}"
qwen_path="${QWEN3VL_PATH:-$asset_root/models/Qwen3-VL-4B-Instruct}"
ms="$env_prefix/bin/modelscope"

[[ -x "$ms" ]] || { echo "missing ModelScope CLI: $ms" >&2; exit 1; }
[[ -d "$qwen_path" ]] || { echo "missing Qwen backbone: $qwen_path" >&2; exit 1; }
mkdir -p "$cache_root"
cd "$repo_root"

# ``download_model`` is used in command substitution; operational messages
# must stay off stdout so the captured value is exactly one filesystem path.
log() { printf '[%s] %s\n' "$(date -Is)" "$*" >&2; }

wait_for_pid() {
  [[ -n "$wait_pid" ]] || return 0
  while kill -0 "$wait_pid" 2>/dev/null; do
    log "waiting for pre-existing evaluation pid=$wait_pid"
    sleep 60
  done
}

wait_for_backgrounds() {
  local path="$sim_root/assets/background_texture"
  until find "$path" -type f -name '*.png' -print -quit 2>/dev/null | grep -q .; do
    log "waiting for random background package at $path"
    sleep 60
  done
}

run_eval() {
  local label="$1" model_path="$2" task_config="$3"
  log "starting ${label} ${task_config}: ${model_path}"
  QWEN3VL_PATH="$qwen_path" bash "$repo_root/experiment/robotwin/start_robotwin_infer_and_eval.sh" \
    --model_path "$model_path" --eval_workdir "$sim_root" --output_base "$output_base" \
    --conda_sh /opt/anaconda3/etc/profile.d/conda.sh --inference_env "$env_prefix" --sim_env "$sim_env" \
    --task_config "$task_config" --num_tasks 50 --episodes_per_task 10 \
    --num_gpus 4 --gpu_ids "$gpu_ids" --num_per_gpu 1 --sim_per_server 2 \
    --use_compile False --no_video
}

download_model() {
  local step="$1" remote_dir="$2" local_root="$cache_root/global_step_${step}" model_path config_dir config_source attempt
  model_path="$local_root/$remote_dir"
  config_source="$asset_root/outputs/competition_clean_4gpu/lingbotvla_cli.yaml"
  [[ -f "$config_source" ]] || { echo "missing training config: $config_source" >&2; return 1; }
  mkdir -p "$local_root"

  # ModelScope may exit zero after individual HTTP timeouts.  A checkpoint is
  # usable only when every payload named by its checksum manifest verifies;
  # never let a partial '.incomplete' shard reach the inference launcher.
  attempt=0
  until [[ -f "$model_path/CHECKSUMS.sha256" ]] && \
        ! find "$model_path" -maxdepth 1 -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q . && \
        (cd "$model_path" && sha256sum -c CHECKSUMS.sha256 >/dev/null 2>&1); do
    attempt=$((attempt + 1))
    log "downloading/resuming step=${step} from ${remote_dir} (attempt ${attempt}; checksum not yet valid)"
    # Two workers are deliberately used here: six parallel large shard
    # requests repeatedly timed out on this server's ModelScope connection.
    "$ms" download "$repo_id" --repo-type model --local-dir "$local_root" \
      --include "${remote_dir}/**" --max-workers 2 >&2 || true
    if (( attempt >= 30 )); then
      log "step=${step} did not pass integrity verification after ${attempt} download attempts"
      return 1
    fi
    [[ -f "$model_path/CHECKSUMS.sha256" ]] && log "step=${step} manifest present; retrying incomplete or failed files"
    sleep 30
  done
  log "step=${step} checkpoint checksum verified"

  # The inference loader resolves lingbotvla_cli.yaml three parents above the
  # model directory.  The uploaded 12k and inference layouts have different
  # depths, so stage the immutable training config at both possible roots.
  config_dir="$(dirname "$(dirname "$(dirname "$model_path")")")"
  mkdir -p "$config_dir"
  cp -f -- "$config_source" "$config_dir/lingbotvla_cli.yaml"
  cp -f -- "$config_source" "$local_root/lingbotvla_cli.yaml"
  printf '%s\n' "$model_path"
}

wait_for_pid
# Clean evaluations do not depend on the background texture package. Run them
# first so the four GPUs remain productive while the large random asset is
# still resuming over the network.  The remote model is deleted immediately.
for step in 14000 12000 10000 6000 2000; do
  if [[ "$step" == 12000 ]]; then remote="full_state/global_step_${step}/hf_ckpt"; else remote="inference/global_step_${step}"; fi
  model_path="$(download_model "$step" "$remote")"
  run_eval "step${step}" "$model_path" demo_clean
  rm -rf -- "$cache_root/global_step_${step}"
  log "deleted local cache for step=${step} after clean evaluation"
done

wait_for_backgrounds

# 16k is local from formal training; its clean half has already completed.
run_eval step16000 "$asset_root/outputs/competition_clean_4gpu/checkpoints/global_step_16000/hf_ckpt" demo_randomized

for step in 14000 12000 10000 6000 2000; do
  if [[ "$step" == 12000 ]]; then remote="full_state/global_step_${step}/hf_ckpt"; else remote="inference/global_step_${step}"; fi
  model_path="$(download_model "$step" "$remote")"
  run_eval "step${step}" "$model_path" demo_randomized
  rm -rf -- "$cache_root/global_step_${step}"
  log "deleted local cache for step=${step} after random evaluation"
done
