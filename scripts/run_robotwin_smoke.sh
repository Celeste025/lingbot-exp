#!/usr/bin/env bash
# One-task / one-rollout integration smoke test for the LingBot-VLA2 RoboTwin
# evaluation stack.  This is deliberately separate from screening (10/task)
# and official submission evaluation (100/task).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${1:-/data1/models/lingbot-vla2/outputs/competition_clean_4gpu/checkpoints/global_step_12000/hf_ckpt}"
gpu_id="${GPU_ID:-4}"

export QWEN3VL_PATH="${QWEN3VL_PATH:-/data1/models/lingbot-vla2/models/Qwen3-VL-4B-Instruct}"

cd "$repo_root"
exec bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path "$model_path" \
  --eval_workdir /home/wjq/workspace/lingbo/RoboTwin-legacy \
  --output_base /data1/models/lingbot-vla2/evaluations \
  --conda_sh /opt/anaconda3/etc/profile.d/conda.sh \
  --inference_env /data1/models/lingbot-vla2/envs/lingbotvla \
  --sim_env /data1/models/lingbot-vla2/envs/robotwin \
  --task_config demo_clean \
  --num_tasks 1 \
  --episodes_per_task 1 \
  --num_gpus 1 \
  --gpu_ids "$gpu_id" \
  --num_per_gpu 1 \
  --use_compile False \
  --no_video
