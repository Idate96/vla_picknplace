#!/usr/bin/env python3
"""Build ACT-ready LeRobot datasets for the banana bowl runs."""

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
POSITIONS = ("left", "center", "right")
ORIGINAL_COLORS_BY_POSITION = ("blue", "red", "green")
PROMPT_BUCKETS = ("direct_color", "position_ordinal", "relative_color", "exclusion")
TARGET_BY_REPO = {
    "rslxcvg/banana_blue": "blue",
    "rslxcvg/banana_red1": "red",
    "rslxcvg/banana_green": "green",
}
TARGET_POSITION_BY_REPO_TARGET = {
    "blue": "left",
    "red": "center",
    "green": "right",
}
DISTRACTOR_SWAP_BY_TARGET = {
    "blue": {"red": "green", "green": "red"},
    "red": {"green": "blue", "blue": "green"},
    "green": {"red": "blue", "blue": "red"},
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
            "making the demonstrated target-position bowl red, green, and blue. "
            "labelsafe_colorpos keeps the target color fixed and swaps only "
            "distractors."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("position", "eval_sampled", "eval_mixed", "all_eval"),
        default="eval_mixed",
        help=(
            "position keeps one legacy position-only prompt per rendering. "
            "eval_sampled uses one deterministic prompt from the eval buckets. "
            "eval_mixed duplicates each rendering once per eval bucket. "
            "all_eval duplicates for every template."
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


def stable_index(label: str, n: int) -> int:
    if n <= 0:
        raise ValueError("stable_index needs a positive length")
    return sum(ord(c) for c in label) % n


def swap_to_make_target_color(target: str, desired_color: str) -> dict[str, str] | None:
    if target == desired_color:
        return None
    return {target: desired_color, desired_color: target}


def colors_by_position_for_swap(swap: dict[str, str] | None) -> tuple[str, str, str]:
    swap = swap or {}
    return tuple(swap.get(color, color) for color in ORIGINAL_COLORS_BY_POSITION)


def rendered_target_position(target: str) -> str:
    return TARGET_POSITION_BY_REPO_TARGET[target]


def prompt_buckets_for_rendering(
    colors_by_position: tuple[str, str, str],
    target_position: str,
) -> dict[str, list[str]]:
    target_idx = POSITIONS.index(target_position)
    target_color = colors_by_position[target_idx]
    other_colors = [color for color in COLORS if color != target_color]

    direct_color = [
        f"Put the banana in the {target_color} colored bowl.",
        f"Put the banana in the {target_color} bowl.",
        f"Place the banana in the {target_color} bowl.",
        f"Move the banana to the {target_color} colored bowl.",
    ]

    position_ordinal = [
        f"Put the banana into the {target_position} bowl from the robot perspective.",
    ]
    if target_position == "left":
        position_ordinal.extend([
            "Put the banana into the 1st bowl from the left from the robot perspective.",
            "Put the banana into the leftmost bowl from the robot perspective.",
        ])
    elif target_position == "center":
        position_ordinal.extend([
            "Put the banana into the 2nd bowl from the left from the robot perspective.",
            "Put the banana into the middle bowl from the robot perspective.",
        ])
    elif target_position == "right":
        position_ordinal.extend([
            "Put the banana into the 3rd bowl from the left from the robot perspective.",
            "Put the banana into the rightmost bowl from the robot perspective.",
        ])

    relative_color = []
    if target_idx > 0:
        ref_color = colors_by_position[target_idx - 1]
        relative_color.append(
            f"Put the banana into the bowl on the right of the {ref_color} bowl "
            "from the robot perspective."
        )
    if target_idx < len(POSITIONS) - 1:
        ref_color = colors_by_position[target_idx + 1]
        relative_color.append(
            f"Put the banana into the bowl on the left of the {ref_color} bowl "
            "from the robot perspective."
        )
    if target_position == "center":
        left_color = colors_by_position[0]
        right_color = colors_by_position[2]
        relative_color.append(
            f"Put the banana into the bowl between the {left_color} bowl and the "
            f"{right_color} bowl."
        )

    exclusion = [
        f"Put the banana into the bowl that is not {other_colors[0]} and not {other_colors[1]}."
    ]

    return {
        "direct_color": direct_color,
        "position_ordinal": position_ordinal,
        "relative_color": relative_color,
        "exclusion": exclusion,
    }


def prompts_for_rendering(
    colors_by_position: tuple[str, str, str],
    target_position: str,
    prompt_mode: str,
    label: str,
) -> list[tuple[str, str]]:
    if prompt_mode == "position":
        return [("position", f"put the banana in the {target_position} bowl")]

    buckets = prompt_buckets_for_rendering(colors_by_position, target_position)
    if prompt_mode == "all_eval":
        return [
            (bucket, prompt)
            for bucket in PROMPT_BUCKETS
            for prompt in buckets[bucket]
        ]
    if prompt_mode == "eval_mixed":
        return [
            (bucket, buckets[bucket][stable_index(f"{label}:{bucket}", len(buckets[bucket]))])
            for bucket in PROMPT_BUCKETS
        ]
    if prompt_mode == "eval_sampled":
        bucket = PROMPT_BUCKETS[stable_index(label, len(PROMPT_BUCKETS))]
        prompt = buckets[bucket][stable_index(f"{label}:{bucket}", len(buckets[bucket]))]
        return [(bucket, prompt)]
    raise ValueError(f"unknown prompt mode: {prompt_mode}")


def episode_variants(
    target: str,
    mode: str,
) -> list[tuple[str, dict[str, str] | None, tuple[str, str, str]]]:
    if mode == "labelsafe_colorpos":
        distractor_swap = DISTRACTOR_SWAP_BY_TARGET[target]
        return [
            ("orig", None, colors_by_position_for_swap(None)),
            ("distractors", distractor_swap, colors_by_position_for_swap(distractor_swap)),
        ]
    return [
        (
            f"target_{desired_color}",
            swap_to_make_target_color(target, desired_color),
            colors_by_position_for_swap(swap_to_make_target_color(target, desired_color)),
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
                target_position = rendered_target_position(target)
                for variant, swap, colors_by_position in episode_variants(target, args.mode):
                    label = f"{stem}_ep{ep:03d}_{variant}"
                    task_prompts = prompts_for_rendering(
                        colors_by_position,
                        target_position,
                        args.prompt_mode,
                        label,
                    )
                    for prompt_idx, (prompt_bucket, task) in enumerate(task_prompts):
                        prompt_label = f"{label}_{prompt_bucket}_{prompt_idx:02d}"
                        add_episode(out, src, start, end, swap, prompt_label, task, args)
    finally:
        out.finalize()

    print(f"Wrote {out.repo_id} to {root}")
    if args.push_to_hub:
        out.push_to_hub(private=args.private, upload_large_folder=True)
        print(f"Pushed {out.repo_id} to Hugging Face")


if __name__ == "__main__":
    main()
