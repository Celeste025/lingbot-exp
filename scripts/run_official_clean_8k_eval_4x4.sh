#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${ASSET_ROOT:-/data1/models/lingbot-vla2}"
model_path="$asset_root/outputs/competition_official_clean_8k/checkpoints/global_step_8000/hf_ckpt"
output_base="$asset_root/evaluations"
sim_root="${ROBOTWIN_ROOT:-/home/wjq/workspace/lingbo/RoboTwin-legacy}"
run_log_dir="$output_base/official_clean_8k_4x4_logs"
mkdir -p "$run_log_dir"

[[ -s "$model_path/model.safetensors.index.json" ]] || { echo "8k HF model export missing" >&2; exit 1; }
[[ -d "$(dirname "$model_path")/optimizer" ]] || { echo "8k full-state optimizer missing" >&2; exit 1; }
[[ -s "$asset_root/outputs/competition_official_clean_8k/lingbotvla_cli.yaml" ]] || { echo "Training config missing" >&2; exit 1; }
for shard in "$model_path"/model-*.safetensors; do
  [[ -s "$shard" ]] || { echo "Empty HF model shard: $shard" >&2; exit 1; }
done

cd "$repo_root"
for task_config in demo_clean demo_randomized; do
  echo "[$(date -Is)] starting $task_config 50 tasks x 10 episodes, 4 GPUs x 1 model x 4 workers"
  bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path "$model_path" \
    --eval_workdir "$sim_root" \
    --output_base "$output_base" \
    --conda_sh /opt/anaconda3/etc/profile.d/conda.sh \
    --inference_env "$asset_root/envs/lingbotvla" \
    --sim_env "$asset_root/envs/robotwin" \
    --task_config "$task_config" \
    --num_tasks 50 --episodes_per_task 10 \
    --num_gpus 4 --gpu_ids 0,1,2,3 \
    --num_per_gpu 1 --sim_per_server 4 \
    --use_fp32 True --use_bf16 False --use_compile False --no_video \
    --start_port 9430 \
    > "$run_log_dir/${task_config}.log" 2>&1
  run_dir="$(sed -n 's/.*Run directory: \([^[:cntrl:]]*\).*/\1/p' "$run_log_dir/${task_config}.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -1)"
  [[ -n "$run_dir" && -s "$run_dir/stats.txt" ]] || { echo "No stats file for $task_config" >&2; exit 1; }
  grep -q 'All tasks fully executed 10 episodes' "$run_dir/stats.txt" || {
    echo "Incomplete 50x10 evaluation: $run_dir/stats.txt" >&2
    exit 1
  }
  grep -qE 'Summary: total [0-9]+s, success [0-9]+/500,' "$run_dir/stats.txt" || {
    echo "Expected 500 episodes missing from $run_dir/stats.txt" >&2
    exit 1
  }
  echo "[$(date -Is)] completed $task_config: $run_dir/stats.txt"
done
