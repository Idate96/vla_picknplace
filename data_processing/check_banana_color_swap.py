#!/usr/bin/env python3
"""Regression checks for banana bowl color-swap augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.bowl_color_swap import (
    apply_photo_aug,
    apply_swap,
    banana_mask_for_frame,
    color_masks_for_frame,
    compute_bowl_means,
    filter_static_masks,
    stabilize_banana_color,
    static_masks_from_frame,
    static_masks_from_frames,
)


COLOR_HUES = {
    "red": 2,
    "green": 55,
    "blue": 105,
}
BOWL_CENTERS = {
    "red": (280, 90),
    "green": (390, 220),
    "blue": (170, 220),
}


def hsv_bgr(hue: int, saturation: int, value: int) -> np.ndarray:
    hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def draw_split_bowl(frame: np.ndarray, center: tuple[int, int], axes: tuple[int, int],
                    color: str) -> np.ndarray:
    full = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(full, center, axes, 0, 0, 360, 255, -1)

    yy, xx = np.indices(full.shape)
    high_sat = (full > 0) & (xx <= center[0])
    low_sat = (full > 0) & (xx > center[0])

    hue = COLOR_HUES[color]
    frame[high_sat] = hsv_bgr(hue, 210, 230)
    frame[low_sat] = hsv_bgr(hue, 45, 238)
    return full.astype(bool)


def make_synthetic_scene(
    banana_on: str = "blue",
    banana_blue_cast: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    frame = np.full((360, 560, 3), 255, dtype=np.uint8)
    bowl_masks = {
        "red": draw_split_bowl(frame, BOWL_CENTERS["red"], (62, 42), "red"),
        "green": draw_split_bowl(frame, BOWL_CENTERS["green"], (72, 52), "green"),
        "blue": draw_split_bowl(frame, BOWL_CENTERS["blue"], (72, 52), "blue"),
    }

    bx, by = BOWL_CENTERS[banana_on]
    banana = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(banana, (bx + 35, by), (32, 12), -15, 0, 360, 255, -1)
    frame[banana > 0] = hsv_bgr(25, 220, 230)
    if banana_blue_cast:
        stripe = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(stripe, (bx + 38, by - 2), (22, 4), 25, 0, 360, 255, -1)
        stripe = (stripe > 0) & (banana > 0)
        frame[stripe] = hsv_bgr(100, 70, 220)
    return frame, bowl_masks, banana.astype(bool)


def hue_mask(frame_bgr: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    colored = (hsv[..., 1] >= 20) & (hsv[..., 2] >= 35)
    if color == "red":
        return colored & ((hue <= 15) | (hue >= 165))
    if color == "green":
        return colored & (hue >= 35) & (hue <= 90)
    if color == "blue":
        return colored & (hue >= 95) & (hue <= 135)
    raise ValueError(color)


def assert_bowl_color(frame_bgr: np.ndarray, region: np.ndarray, expected: str,
                      forbidden: str, label: str, min_expected_frac: float = 0.90,
                      max_forbidden_frac: float = 0.06) -> None:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    colored = region & (hsv[..., 1] >= 20) & (hsv[..., 2] >= 35)
    if int(colored.sum()) < 500:
        raise AssertionError(f"{label}: too few visible colored pixels: {int(colored.sum())}")

    expected_count = int((hue_mask(frame_bgr, expected) & colored).sum())
    forbidden_count = int((hue_mask(frame_bgr, forbidden) & colored).sum())
    total = int(colored.sum())
    expected_frac = expected_count / total
    forbidden_frac = forbidden_count / total
    if expected_frac < min_expected_frac or forbidden_frac > max_forbidden_frac:
        raise AssertionError(
            f"{label}: mixed bowl colors after swap; "
            f"expected {expected} frac={expected_frac:.3f}, "
            f"forbidden {forbidden} frac={forbidden_frac:.3f}, total={total}"
        )


def run_swap(
    frame_bgr: np.ndarray,
    swap: dict[str, str],
    static_frames: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    static_masks = (
        static_masks_from_frames(static_frames)
        if static_frames is not None
        else static_masks_from_frame(frame_bgr)
    )
    ref_masks = color_masks_for_frame(frame_bgr, np.ones((3, 3), np.uint8))
    means = compute_bowl_means([(frame_bgr, ref_masks)])
    banana_mask = banana_mask_for_frame(frame_bgr, dilation=3)
    masks = filter_static_masks(frame_bgr, static_masks, 40, banana_mask=banana_mask)
    masks = {name: mask & ~banana_mask for name, mask in masks.items()}
    swapped = apply_swap(frame_bgr, masks, swap, means)

    aug = {
        "brightness": 0.04,
        "contrast": 1.05,
        "saturation": 1.08,
        "hue": 0,
        "noise_std": 0.0,
        "rng": np.random.default_rng(0),
    }
    swapped = apply_photo_aug(
        swapped,
        aug,
        preserve_mask=banana_mask,
        preserve_source=stabilize_banana_color(frame_bgr, banana_mask),
    )
    return swapped, banana_mask


def check_no_half_swapped_bowls() -> None:
    frame, bowl_masks, _banana_gt = make_synthetic_scene()
    cases = [
        ("blue_to_green", {"blue": "green", "green": "blue"}),
        ("blue_to_red", {"blue": "red", "red": "blue"}),
    ]
    for case_name, swap in cases:
        swapped, banana_mask = run_swap(frame, swap)
        for source, expected in swap.items():
            visible_bowl = bowl_masks[source] & ~banana_mask
            assert_bowl_color(
                swapped,
                visible_bowl,
                expected=expected,
                forbidden=source,
                label=f"{case_name}:{source}->{expected}",
            )


def check_banana_is_preserved() -> None:
    frame, _bowl_masks, banana_gt = make_synthetic_scene()
    swapped, banana_mask = run_swap(frame, {"blue": "green", "green": "blue"})
    protected = banana_gt & banana_mask
    if int(protected.sum()) < 100:
        raise AssertionError("banana preservation mask did not cover synthetic banana")
    diff = np.abs(swapped[protected].astype(np.int16) - frame[protected].astype(np.int16))
    if int(diff.max()) > 2:
        raise AssertionError(f"banana pixels changed; max diff={int(diff.max())}")


def check_no_source_color_halo_around_banana() -> None:
    frame, bowl_masks, banana_gt = make_synthetic_scene(banana_on="red")
    swapped, _banana_mask = run_swap(frame, {"red": "green", "green": "red"})

    tight = cv2.dilate(banana_gt.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    halo = cv2.dilate(banana_gt.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    halo = halo & bowl_masks["red"] & ~tight
    if int(halo.sum()) < 100:
        raise AssertionError(f"halo test region too small: {int(halo.sum())}")

    assert_bowl_color(
        swapped,
        halo,
        expected="green",
        forbidden="red",
        label="red_to_green:banana_halo",
        min_expected_frac=0.95,
        max_forbidden_frac=0.01,
    )


def check_banana_blue_cast_is_neutralized() -> None:
    frame, _bowl_masks, banana_gt = make_synthetic_scene(
        banana_on="blue",
        banana_blue_cast=True,
    )
    swapped, _banana_mask = run_swap(frame, {"blue": "red", "red": "blue"})
    banana_pixels = banana_gt
    b, g, r = cv2.split(swapped)
    blueish = (
        banana_pixels
        & (b.astype(np.int16) >= r.astype(np.int16) + 10)
        & (b.astype(np.int16) >= g.astype(np.int16) - 10)
        & (np.maximum.reduce([b, g, r]) >= 80)
    )
    blueish_count = int(blueish.sum())
    if blueish_count > 25:
        raise AssertionError(f"banana retained blue cast: {blueish_count} px")


def check_static_masks_cover_initially_occluded_bowl() -> None:
    current, bowl_masks, _banana_gt = make_synthetic_scene(banana_on="blue")
    ref = current.copy()
    green_bowl = bowl_masks["green"]
    yy, xx = np.indices(green_bowl.shape)
    left_half = green_bowl & (xx < BOWL_CENTERS["green"][0])
    ref[left_half] = (8, 8, 8)

    swapped, banana_mask = run_swap(
        current,
        {"green": "blue", "blue": "green"},
        static_frames=[ref, current],
    )
    visible_bowl = green_bowl & ~banana_mask
    assert_bowl_color(
        swapped,
        visible_bowl,
        expected="blue",
        forbidden="green",
        label="initially_occluded_green_bowl_to_blue",
        min_expected_frac=0.90,
        max_forbidden_frac=0.03,
    )


def check_real_green_source_right_target_blue() -> None:
    from lerobot.datasets import LeRobotDataset
    from data_processing.augment_banana_lerobot import sample_photo_aug, tensor_rgb_to_bgr

    src = LeRobotDataset("rslxcvg/banana_green", video_backend="pyav", return_uint8=True)
    start = int(src.meta.episodes[0]["dataset_from_index"])
    end = int(src.meta.episodes[0]["dataset_to_index"])
    length = end - start
    offsets = sorted(set([0, length // 3, (2 * length) // 3, length - 1]))
    sample_frames = [
        tensor_rgb_to_bgr(src[start + offset]["observation.images.front"])
        for offset in offsets
    ]
    static_masks = static_masks_from_frames(sample_frames)
    means = compute_bowl_means([
        (frame, color_masks_for_frame(frame, np.ones((3, 3), np.uint8)))
        for frame in sample_frames
    ])

    frame = tensor_rgb_to_bgr(src[start + 30]["observation.images.front"])
    banana_mask = banana_mask_for_frame(frame, dilation=3)
    masks = filter_static_masks(frame, static_masks, 40, banana_mask=banana_mask)
    masks = {name: mask & ~banana_mask for name, mask in masks.items()}
    swapped = apply_swap(frame, masks, {"green": "blue", "blue": "green"}, means)
    swapped = apply_photo_aug(
        swapped,
        sample_photo_aug(20260512, "banana_green_ep000_target_blue"),
        preserve_mask=banana_mask,
        preserve_source=stabilize_banana_color(frame, banana_mask),
    )

    crop = swapped[190:291, 500:580]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[..., 0] * 2.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    colored = (s >= 0.25) & (v >= 0.15)
    green = colored & (h >= 70) & (h <= 150)
    blue = colored & (h >= 180) & (h <= 260)
    total = int((green | blue).sum())
    green_frac = float(green.sum() / max(1, total))
    if total < 1000 or green_frac >= 0.03:
        raise AssertionError(
            "real green_source_right_target_blue frame 30 failed: "
            f"total={total}, green_px={int(green.sum())}, "
            f"blue_px={int(blue.sum())}, green_frac={green_frac:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Also run cached/Hugging Face LeRobot source-frame regressions.",
    )
    args = parser.parse_args()

    check_no_half_swapped_bowls()
    check_banana_is_preserved()
    check_no_source_color_halo_around_banana()
    check_banana_blue_cast_is_neutralized()
    check_static_masks_cover_initially_occluded_bowl()
    if args.real_data:
        check_real_green_source_right_target_blue()
    print("banana color-swap checks passed")


if __name__ == "__main__":
    main()
