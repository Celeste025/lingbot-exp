#!/usr/bin/env python3
"""Validate the pinned official RoboTwin v2.1 clean LeRobot export."""

import argparse
import json
from pathlib import Path

from validate_robotwin_joint_data import validate_dataset_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write-list", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v2.1":
        raise SystemExit(f"Unexpected LeRobot version: {info.get('codebase_version')}")
    if info.get("total_episodes") != 2500 or info.get("total_videos") != 7500:
        raise SystemExit("Expected 2500 clean episodes and 7500 camera videos")
    errors = validate_dataset_source("robotwin", root)
    if errors:
        raise SystemExit("\n".join(errors))
    expected_cameras = {"observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"}
    if not expected_cameras.issubset(info["features"]):
        raise SystemExit("Official camera fields do not match robotwin.yaml mapping")
    episodes_file = root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_file.read_text().splitlines() if line.strip()]
    if len(episodes) != 2500 or {e["episode_index"] for e in episodes} != set(range(2500)):
        raise SystemExit("Episode metadata is not a complete set of 2500 clean episodes")
    tasks_file = root / "meta/tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_file.read_text().splitlines() if line.strip()]
    if len(tasks) != info.get("total_tasks"):
        raise SystemExit("Natural language task metadata count does not match info.json")
    if args.write_list:
        args.write_list.parent.mkdir(parents=True, exist_ok=True)
        args.write_list.write_text(f"robotwin {root}\n")
    print(f"Validated official 14D joint dataset: 2500 episodes, {len(tasks)} language instructions, 3 cameras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
