#!/usr/bin/env python3
"""Build ACT-ready LeRobot datasets for the banana bowl runs.

The default output is position-conditioned: for each original episode it
creates three color renderings where the target-position bowl is blue, red,
and green. The language names the target position, not the target color.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

from data_processing.bowl_color_swap import (
    COLORS,
    apply_photo_aug,
    apply_swap,
    banana_mask_for_frame,
    color_masks_for_frame,
    compute_bowl_means,
    filter_static_masks,
    stabilize_banana_color,
    static_masks_from_frames,
)


IMAGE_KEY = "observation.images.front"
SOURCE_REPOS = ("rslxcvg/banana_blue", "rslxcvg/banana_red1", "rslxcvg/banana_green")
TARGET_BY_REPO = {
    "rslxcvg/banana_blue": "blue",
    "rslxcvg/banana_red1": "red",
    "rslxcvg/banana_green": "green",
}
DISTRACTOR_SWAP_BY_TARGET = {
    "blue": {"red": "green", "green": "red"},
    "red": {"green": "blue", "blue": "green"},
    "green": {"red": "blue", "blue": "red"},
}
TASK_BY_TARGET = {
    "blue": "put the banana in the blue bowl on the left",
    "red": "put the banana in the red bowl in the center",
    "green": "put the banana in the green bowl on the right",
}
POSITION_TASK_BY_TARGET = {
    "blue": "put the banana in the left bowl",
    "red": "put the banana in the center bowl",
    "green": "put the banana in the right bowl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-repo-id", default="rslxcvg/banana_act_position_targetcolor_v1")
    parser.add_argument(
        "--mode",
        choices=("position_target_colors", "labelsafe_colorpos"),
        default="position_target_colors",
        help=(
            "position_target_colors creates three renderings per episode, "
            "with position-only task text. labelsafe_colorpos keeps the "
            "target color fixed and swaps only distractors."
        ),
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--source-repo", action="append", default=None)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--sat-boost", type=float, default=1.0)
    parser.add_argument("--arm-v", type=int, default=40)
    parser.add_argument("--photo-seed", type=int, default=20260512)
    args = parser.parse_args()
    args.source_repo = args.source_repo or list(SOURCE_REPOS)
    return args


def root_from_repo(repo_id: str) -> Path:
    return Path("outputs/lerobot") / repo_id.split("/")[-1]


def tensor_rgb_to_bgr(image) -> np.ndarray:
    rgb = image.permute(1, 2, 0).cpu().numpy()
    assert rgb.dtype == np.uint8
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def sample_photo_aug(seed: int, label: str) -> dict:
    rng = np.random.default_rng(seed + sum(ord(c) for c in label))
    return {
        "brightness": float(rng.uniform(-0.08, 0.08)),
        "contrast": float(rng.uniform(0.90, 1.10)),
        "saturation": float(rng.uniform(0.90, 1.12)),
        "hue": 0,
        "noise_std": 0.0,
        "rng": rng,
    }


def swap_to_make_target_color(target: str, desired_color: str) -> dict[str, str] | None:
    if target == desired_color:
        return None
    return {target: desired_color, desired_color: target}


def episode_variants(target: str, mode: str) -> list[tuple[str, dict[str, str] | None, str]]:
    if mode == "labelsafe_colorpos":
        return [
            ("orig", None, TASK_BY_TARGET[target]),
            ("distractors", DISTRACTOR_SWAP_BY_TARGET[target], TASK_BY_TARGET[target]),
        ]
    return [
        (
            f"target_{desired_color}",
            swap_to_make_target_color(target, desired_color),
            POSITION_TASK_BY_TARGET[target],
        )
        for desired_color in COLORS
    ]


def add_episode(
    out: LeRobotDataset,
    src: LeRobotDataset,
    start: int,
    end: int,
    swap: dict[str, str] | None,
    label: str,
    task: str,
    args: argparse.Namespace,
) -> None:
    first = src[start]
    ref_bgr = tensor_rgb_to_bgr(first[IMAGE_KEY])
    static_masks = None
    means = None
    photo_aug = None

    if swap:
        sample_offsets = sorted(set([0, (end - start) // 3, (2 * (end - start)) // 3, end - start - 1]))
        sample_frames = [
            tensor_rgb_to_bgr(src[start + offset][IMAGE_KEY])
            for offset in sample_offsets
            if start + offset < end
        ]
        static_masks = static_masks_from_frames(sample_frames, args.sat_boost)
        kernel = np.ones((3, 3), np.uint8)
        means = compute_bowl_means([
            (frame, color_masks_for_frame(frame, kernel, args.sat_boost))
            for frame in sample_frames
        ])
        photo_aug = sample_photo_aug(args.photo_seed, label)
        shown = {k: v for k, v in photo_aug.items() if k != "rng"}
        print(f"{label}: swap={swap} photo_aug={shown}")

    for idx in tqdm(range(start, end), desc=label, leave=False):
        item = src[idx]
        frame_bgr = tensor_rgb_to_bgr(item[IMAGE_KEY])
        if swap:
            source_bgr = frame_bgr
            banana_mask = banana_mask_for_frame(source_bgr, dilation=3)
            masks = filter_static_masks(
                frame_bgr,
                static_masks,
                args.arm_v,
                args.sat_boost,
                banana_mask=banana_mask,
            )
            if banana_mask.any():
                masks = {name: mask & ~banana_mask for name, mask in masks.items()}
            frame_bgr = apply_swap(frame_bgr, masks, swap, means)
            frame_bgr = apply_photo_aug(
                frame_bgr,
                photo_aug,
                preserve_mask=banana_mask,
                preserve_source=stabilize_banana_color(source_bgr, banana_mask),
            )

        out.add_frame(
            {
                IMAGE_KEY: bgr_to_rgb(frame_bgr),
                "observation.state": item["observation.state"].cpu().numpy(),
                "action": item["action"].cpu().numpy(),
                "task": task,
            }
        )

    out.save_episode()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else root_from_repo(args.output_repo_id)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    first = LeRobotDataset(args.source_repo[0], video_backend=args.video_backend, return_uint8=True)
    user_features = copy.deepcopy({k: v for k, v in first.features.items() if k not in DEFAULT_FEATURES})
    for ft in user_features.values():
        if ft["dtype"] == "video":
            ft.pop("info", None)
    assert IMAGE_KEY in user_features

    out = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=root,
        fps=first.fps,
        features=user_features,
        robot_type=first.meta.robot_type,
        use_videos=True,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
        streaming_encoding=True,
        encoder_queue_maxsize=120,
        encoder_threads=args.encoder_threads,
        metadata_buffer_size=20,
    )

    try:
        for repo_id in args.source_repo:
            target = TARGET_BY_REPO[repo_id]
            src = first if repo_id == args.source_repo[0] else LeRobotDataset(
                repo_id, video_backend=args.video_backend, return_uint8=True
            )
            n_eps = src.num_episodes
            if args.max_episodes_per_repo is not None:
                n_eps = min(n_eps, args.max_episodes_per_repo)

            print(f"{repo_id}: {n_eps} episode(s), target={target}")
            for ep in range(n_eps):
                meta = src.meta.episodes[ep]
                start = int(meta["dataset_from_index"])
                end = int(meta["dataset_to_index"])
                stem = repo_id.split("/")[-1]
                for variant, swap, task in episode_variants(target, args.mode):
                    label = f"{stem}_ep{ep:03d}_{variant}"
                    add_episode(out, src, start, end, swap, label, task, args)
    finally:
        out.finalize()

    print(f"Wrote {out.repo_id} to {root}")
    if args.push_to_hub:
        out.push_to_hub(private=args.private, upload_large_folder=True)
        print(f"Pushed {out.repo_id} to Hugging Face")


if __name__ == "__main__":
    main()
