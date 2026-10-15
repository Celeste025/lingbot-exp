#!/usr/bin/env python3
"""Measure how closely a checkpoint reproduces action chunks from training data.

This is intentionally an offline diagnostic: it loads no RobotWin simulator and
does not produce benchmark success rates.  It compares the model's unnormalized
action chunk against the demonstrated chunk and against the best constant-action
baseline over the sampled chunks.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from open_loop_eval import (
    LEROBOT_DATASET_API,
    LeRobotDataset,
    LeRobotDatasetMetadata,
    load_policy_server,
    prepare_eval_observation,
    to_numpy,
)


def dataset_for(root: Path, action_features: list[str], chunk_size: int):
    metadata = LeRobotDatasetMetadata(root.name, root=root)
    timestamps = {
        feature: [step / metadata.fps for step in range(chunk_size)]
        for feature in action_features
    }
    return LeRobotDataset(root.name, root=root, delta_timestamps=timestamps)


def episode_start(dataset, episode: int) -> int:
    if LEROBOT_DATASET_API == "v2":
        return int(dataset.episode_data_index["from"][episode])
    return int(dataset.meta.episodes[episode]["dataset_from_index"])


def sample_chunk(policy, dataset_root: Path, episode: int, chunk_size: int):
    action_features = policy.vla.feature_transform.org_features["actions"]
    dataset = dataset_for(dataset_root, action_features, chunk_size)
    item = dataset[episode_start(dataset, episode)]
    observation, round_tripped = prepare_eval_observation(policy, item)
    # Keep the model-space action output as well as the RobotWin-space action.
    # Comparing both distinguishes a bad policy from a bad de-normalization path.
    transformed = policy._prepare_model_input(observation)
    prediction = policy.infer(observation, return_normalized=True)
    gt = np.concatenate(
        [to_numpy(round_tripped[key])[:chunk_size] for key in action_features], axis=-1
    )
    pred = np.concatenate(
        [to_numpy(prediction[key])[:chunk_size] for key in action_features], axis=-1
    )
    if gt.shape != pred.shape:
        raise RuntimeError(f"shape mismatch: gt={gt.shape}, prediction={pred.shape}")
    valid = to_numpy(transformed["action_joint_mask"]).astype(bool)
    gt_normalized = to_numpy(transformed["actions"])[:chunk_size, valid]
    pred_normalized = to_numpy(prediction["_normalized_actions"])[:chunk_size, valid]
    if gt_normalized.shape != pred_normalized.shape:
        raise RuntimeError(
            f"normalized shape mismatch: gt={gt_normalized.shape}, prediction={pred_normalized.shape}"
        )
    # Re-evaluate the exact flow-matching objective on the same transformed
    # input.  This is deliberately distinct from the decoded action error
    # above: a low flow loss only matters if it is also consistent across the
    # sampling trajectory used at inference.
    model_inputs = {}
    for key in ("images", "img_masks", "state", "lang_tokens", "lang_masks", "actions"):
        value = transformed[key]
        model_inputs[key] = value.unsqueeze(0).to("cuda")
    if "image_grid_thw" in transformed:
        model_inputs["image_grid_thw"] = transformed["image_grid_thw"].unsqueeze(0).to("cuda")
    joint_mask = transformed["joint_mask"].unsqueeze(0).to("cuda")
    flow_l1_by_time = {}
    with torch.no_grad():
        for time_value in (0.1, 0.5, 0.9):
            noise = torch.randn_like(model_inputs["actions"])
            time = torch.full((1,), time_value, device="cuda", dtype=model_inputs["actions"].dtype)
            # The policy returns the total loss first; with the saved config it
            # includes alignment terms, so record the VLA term separately.
            _, loss_vla, *_ = policy.vla(
                **model_inputs,
                joint_mask=joint_mask,
                noise=noise,
                time=time,
            )
            flow_l1_by_time[str(time_value)] = float(loss_vla.cpu())
    return (
        gt.astype(np.float64),
        pred.astype(np.float64),
        gt_normalized.astype(np.float64),
        pred_normalized.astype(np.float64),
        flow_l1_by_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", action="append", required=True,
                        help="Absolute LeRobot dataset root; repeat for each sample.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy_cls = load_policy_server("qwen3vl", args.model_path)
    policy = policy_cls(
        path_to_pi_model=args.model_path,
        use_length=args.chunk_size,
        chunk_ret=True,
        use_bf16=False,
        use_fp32=True,
        use_compile=False,
    )
    policy.reset("robotwin")

    samples = []
    gts, predictions, normalized_gts, normalized_predictions = [], [], [], []
    for dataset_arg in args.dataset:
        root = Path(dataset_arg)
        policy.reset("robotwin")
        gt, prediction, gt_normalized, prediction_normalized, flow_l1_by_time = sample_chunk(
            policy, root, args.episode, args.chunk_size
        )
        constant_for_sample = np.broadcast_to(np.mean(gt, axis=0, keepdims=True), gt.shape)
        samples.append({
            "dataset": str(root),
            "episode": args.episode,
            "model_mae": float(np.mean(np.abs(prediction - gt))),
            "model_mse": float(np.mean((prediction - gt) ** 2)),
            "gt_std": float(np.std(gt)),
            "prediction_std": float(np.std(prediction)),
            "constant_chunk_mae": float(np.mean(np.abs(constant_for_sample - gt))),
            "constant_chunk_mse": float(np.mean((constant_for_sample - gt) ** 2)),
            "normalized_model_mse": float(np.mean((prediction_normalized - gt_normalized) ** 2)),
            "flow_l1_by_time": flow_l1_by_time,
        })
        gts.append(gt)
        predictions.append(prediction)
        normalized_gts.append(gt_normalized)
        normalized_predictions.append(prediction_normalized)

    gt_all = np.concatenate(gts, axis=0)
    pred_all = np.concatenate(predictions, axis=0)
    gt_normalized_all = np.concatenate(normalized_gts, axis=0)
    pred_normalized_all = np.concatenate(normalized_predictions, axis=0)
    constant = np.broadcast_to(np.mean(gt_all, axis=0, keepdims=True), gt_all.shape)
    centered_gt = gt_all.ravel() - gt_all.mean()
    centered_pred = pred_all.ravel() - pred_all.mean()
    correlation = float(np.dot(centered_gt, centered_pred) / (
        np.linalg.norm(centered_gt) * np.linalg.norm(centered_pred) + 1e-12
    ))
    report = {
        "model_path": args.model_path,
        "chunk_size": args.chunk_size,
        "sample_count": len(samples),
        "samples": samples,
        "aggregate": {
            "model_mae": float(np.mean(np.abs(pred_all - gt_all))),
            "model_mse": float(np.mean((pred_all - gt_all) ** 2)),
            "constant_mean_mae": float(np.mean(np.abs(constant - gt_all))),
            "constant_mean_mse": float(np.mean((constant - gt_all) ** 2)),
            "mse_ratio_to_constant_mean": float(np.mean((pred_all - gt_all) ** 2) / (np.mean((constant - gt_all) ** 2) + 1e-12)),
            "gt_std": float(np.std(gt_all)),
            "prediction_std": float(np.std(pred_all)),
            "flattened_correlation": correlation,
            "normalized_model_mse": float(np.mean((pred_normalized_all - gt_normalized_all) ** 2)),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
