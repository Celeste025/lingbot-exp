---
base_model:
  - robbyant/lingbot-vla-v2-6b
frameworks:
  - PyTorch
license: Apache-2.0
tags:
  - robotwin
  - lingbot-vla-v2
---

# LingBot-VLA 2.0 · RoboTwin 2.0 clean-only checkpoints

This repository archives the Aloha-AgileX clean-only post-training run for the
Tianchi competition. Training uses the official
[`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
LeRobot v2.1 clean export at revision
`3dc3b798668feb99ac61cc9086d84cbcc3d79186`. No randomized trajectories
are used for training. Evaluation on randomized settings is separate.

The training and simulator action interface uses 14 joint-space values: six arm
joints and one gripper per side. The previous checkpoints in this repository
were trained from a different 16-dimensional end-effector dataset under that
14-dimensional mapping. Those incompatible checkpoints were removed on
2026-10-13. They should not be used or resumed.

The replacement run uses four GPUs, global batch size 256, and 8,000 optimizer
steps. Its checkpoints are organized as follows:

| Repository path | Contents | Intended use |
| --- | --- | --- |
| `inference/global_step_2000/` | Model weights and inference assets | Evaluation at step 2,000 |
| `full_state/global_step_4000/` | Distributed model, optimizer, scheduler, RNG and inference export | Resume training or evaluate step 4,000 |
| `inference/global_step_6000/` | Model weights and inference assets | Evaluation at step 6,000 |
| `full_state/global_step_8000/` | Distributed model, optimizer, scheduler, RNG and inference export | Resume training or evaluate step 8,000 |

Each uploaded checkpoint includes `CHECKSUMS.sha256`. Upload and verification
run concurrently with training. After remote verification, local storage retains
the newest full checkpoint and the newest model-only checkpoint. The paths above
appear when training reaches and archives each checkpoint.

## Prepared continuation: step 8,000 to 20,000

The local launcher `scripts/run_official_clean_8k_to_20k.sh` resumes the latest
complete optimizer checkpoint from this run. It keeps the same official clean
dataset, normalization statistics, four GPU setup, and global batch size 256.
The new cosine segment raises the base learning rate from the old endpoint
`5e-5` to `8e-5` at step 8,000, then lowers it to `4e-5` at step 20,000.
Muon expert parameter groups retain their original relative LR scaling.

The launcher saves inference weights at steps 10k, 14k, and 18k and full
resumable checkpoints at 12k, 16k, and 20k. It verifies uploads to this same
ModelScope repository before older local checkpoints are removed. The
continuation started on 2026-10-15 from the verified step-8,000 full checkpoint;
later checkpoint rows appear as training reaches and archives each step.
