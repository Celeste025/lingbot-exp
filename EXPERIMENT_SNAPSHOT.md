# LingBot-VLA2 RoboTwin clean-50 experiment snapshot

This repository is a code-and-metadata snapshot for the Tianchi RoboTwin preliminary experiment. It intentionally excludes model weights, training data, simulator assets, Python/Conda environments, and access tokens.

## Training record

- Base model: LingBot-VLA2.
- Training data: competition clean 50 tasks × 50 demonstrations only.
- Hardware: 4 GPUs.
- Batch configuration: global batch size 256, microbatch size 32 per GPU, gradient accumulation 2.
- Training length: 16,000 steps.
- Final recorded loss: `0.01129150390625` at step 16,000.
- Full-state checkpoint policy: every 4,000 steps; inference checkpoint policy: every 2,000 steps.
- Remote checkpoint archive: ModelScope repository `Celeste025/lingbot-vla2-robotwin-clean50-checkpoints`.

## Contents retained for reporting

- `experiment_artifacts/training/loss.jsonl`: per-step loss and optimizer metrics through step 16,000.
- `experiment_artifacts/training/resolved_train_config.yaml`: resolved training invocation.
- `experiment_artifacts/training/robotwin_clean_50_norm_stats.json`: normalization statistics.
- `experiment_artifacts/training/*future_metrics.jsonl`: auxiliary future-video metrics at 5k, 10k, and 15k.
- `experiment_artifacts/checkpoints/global_step_16000.CHECKSUMS.sha256`: checksum manifest for the final recoverable checkpoint.
- `experiment_artifacts/evaluation/smoke_12k_clean/`: initial smoke logs. This smoke is expected to be non-passing: it records the missing official RoboTwin Objaverse asset discovered during environment bring-up, before downloading the asset package.

## Evaluation convention

The launcher supports `--episodes_per_task`. Use 1 for integration smoke tests, 10 for checkpoint screening (50 clean + 50 random tasks = 1,000 rollouts), and 100 for the final official-style evaluation.

## Recovery pointers

Model weights and full optimizer state are deliberately not in Git. Retrieve archived checkpoints from the ModelScope repository above; verify their checksum against the retained manifest before use.

## Security

Do not commit GitHub PATs, ModelScope tokens, raw datasets, model weights, or downloaded RoboTwin assets.
