#!/usr/bin/env python3
"""Reject RoboTwin datasets whose actions are not Aloha joint targets.

The robotwin.yaml mapping and the simulator's qpos action interface both use
six arm joints plus one gripper per side. Some public RoboTwin LeRobot datasets
instead store seven end-effector pose values plus a gripper per side. Slicing
the latter as 14 joint values silently trains an unusable policy.
"""

import argparse
import json
from pathlib import Path


EXPECTED_NAMED_JOINTS = (
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
)


EXPECTED_INDEXED_JOINTS = tuple(
    f"{side}_joint_{index}" for side in ("left", "right") for index in range(7)
)
EXPECTED_NAME_SETS = {EXPECTED_NAMED_JOINTS, EXPECTED_INDEXED_JOINTS}


def validate_dataset_source(data_name: str, train_path: str | Path) -> list[str]:
    """Validate local RoboTwin input before model or distributed initialization."""
    if data_name not in ("robotwin", "multi"):
        return []
    if not train_path:
        return [f"{data_name}: training data path is empty"]
    source = Path(train_path)
    if data_name == "multi":
        try:
            entries = [line.strip() for line in source.read_text().splitlines() if line.strip()]
        except OSError as exc:
            return [f"{source}: cannot read dataset list: {exc}"]
        if not entries:
            return [f"{source}: dataset list is empty"]
        datasets = []
        for entry in entries:
            parts = entry.split(maxsplit=1)
            if len(parts) != 2:
                return [f"{source}: invalid dataset entry {entry!r}"]
            if parts[0] == "robotwin":
                datasets.append(Path(parts[1]))
    else:
        datasets = [source]

    errors = []
    for root in datasets:
        try:
            features = json.loads((root / "meta/info.json").read_text())["features"]
            for key in ("observation.state", "action"):
                feature = features[key]
                shape = feature["shape"]
                names = feature["names"]
                if shape != [14] or not isinstance(names, list) or len(names) != 1 or tuple(names[0]) not in EXPECTED_NAME_SETS:
                    errors.append(
                        f"{root}: {key} has shape={shape}, names={names}; "
                        "expected 14 Aloha joint/gripper values"
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{root}: cannot read feature schema: {exc}")
    return errors


def validate(data_list: Path) -> list[str]:
    return validate_dataset_source("multi", data_list)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_list", type=Path)
    args = parser.parse_args()
    errors = validate(args.data_list)
    if errors:
        for error in errors[:4]:
            print(f"ERROR: {error}")
        if len(errors) > 4:
            print(f"... {len(errors) - 4} further schema errors")
        print(
            "This training setup requires joint-space LeRobot data. "
            "The 16-D left_x/left_q1/... format is end-effector pose data; "
            "do not use it with configs/robot_configs/robotwin.yaml."
        )
        return 1
    print(f"Validated joint-space schema for {len(args.data_list.read_text().splitlines())} RoboTwin datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
