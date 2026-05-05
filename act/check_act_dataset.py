#!/usr/bin/env python3
from __future__ import annotations

import argparse

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a LeRobot dataset is ready for ACT.")
    parser.add_argument("--repo-id", default="carmensc/record-test-screwdriver")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--video-backend", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    policy_features = dataset_to_policy_features(meta.features)

    cfg = ACTConfig(device="cpu")
    cfg.input_features = {
        key: feature for key, feature in policy_features.items() if feature.type is not FeatureType.ACTION
    }
    cfg.output_features = {
        key: feature for key, feature in policy_features.items() if feature.type is FeatureType.ACTION
    }
    cfg.validate_features()

    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        revision=args.revision,
        episodes=[args.episode],
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=True,
    )
    sample = dataset[0]

    print(f"Dataset: {args.repo_id}")
    print(f"Robot: {meta.robot_type}")
    print(f"FPS: {meta.fps}")
    print(f"Episodes: {meta.total_episodes}")
    print(f"Frames: {meta.total_frames}")
    print("Policy input features:")
    for key, feature in cfg.input_features.items():
        print(f"  {key}: {feature.type.name} {tuple(feature.shape)}")
    print("Policy output features:")
    for key, feature in cfg.output_features.items():
        print(f"  {key}: {feature.type.name} {tuple(feature.shape)}")
    print("ACT sample:")
    for key in [*cfg.input_features, ACTION]:
        value = sample[key]
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")


if __name__ == "__main__":
    main()
