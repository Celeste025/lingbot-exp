# LingBot-VLA2 RoboTwin official-clean experiment

This repository contains code and small reproducibility artifacts for the Tianchi RoboTwin experiment. Model weights, raw training data, simulator assets, environments, and credentials are stored outside Git.

## Current valid run

- Data: official RoboTwin2.0 LeRobot v2.1 **clean** trajectories, using the 14-dimensional Aloha joint-space action mapping. No randomized trajectories enter training.
- Training: four GPUs, global batch size 256, 2,144 optimizer steps per epoch.
- Step 8,000: full-state checkpoint archived in `Celeste025/lingbot-vla2-robotwin-clean50-checkpoints` on ModelScope. The 50-task × 10-episode evaluation completed with **153/500 (30.6%) clean** and **90/500 (18.0%) randomized** success.
- Step 8,000 to 20,000: continuation is running from the full checkpoint. The base learning rate starts at `8e-5` and cosine-anneals to `4e-5` at step 20,000. The continuation launcher is `scripts/run_official_clean_8k_to_20k.sh`.
- Checkpoints: inference weights every 2,000 steps; full optimizer/resume state every 4,000 steps. Upload and checksum verification run concurrently with training. Older local checkpoints are removed only after their ModelScope copies are verified.

`experiment_artifacts/training/official_clean_8k_loss.jsonl` is the fixed 8,000-step loss record. The two `experiment_artifacts/evaluation/official_clean_8k_*/stats.txt` files are the complete evaluation summaries. `reports/official_clean_8k_loss.png` plots the training curve.

## Superseded artifacts

The older files under `experiment_artifacts/training/` with generic names, including `loss.jsonl` and `resolved_train_config.yaml`, describe a **different 16-dimensional end-effector run**. Its checkpoints were invalid for the 14-dimensional competition setup and were removed from ModelScope. Those files are retained only as debugging history; do not use them to resume or report the current model.

## Security

Do not commit access tokens, raw datasets, checkpoint tensors, or downloaded RoboTwin assets.
