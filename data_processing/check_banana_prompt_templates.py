#!/usr/bin/env python3
"""Regression checks for banana task prompt generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_processing.augment_banana_lerobot import (
    COLORS,
    POSITIONS,
    PROMPT_BUCKETS,
    colors_by_position_for_swap,
    prompt_buckets_for_rendering,
    prompts_for_rendering,
    rendered_target_position,
    swap_to_make_target_color,
)


def assert_contains(prompt: str, expected: str, label: str) -> None:
    if expected not in prompt:
        raise AssertionError(f"{label}: expected {expected!r} in {prompt!r}")


def check_eval_prompt_buckets_use_rendered_colors() -> None:
    for original_target in COLORS:
        target_position = rendered_target_position(original_target)
        target_idx = POSITIONS.index(target_position)
        for rendered_target_color in COLORS:
            label = f"{original_target}_as_{rendered_target_color}"
            swap = swap_to_make_target_color(original_target, rendered_target_color)
            colors_by_position = colors_by_position_for_swap(swap)
            target_color = colors_by_position[target_idx]
            if target_color != rendered_target_color:
                raise AssertionError(
                    f"{label}: target color not taken from rendered layout: "
                    f"{colors_by_position}"
                )

            buckets = prompt_buckets_for_rendering(colors_by_position, target_position)
            if set(buckets) != set(PROMPT_BUCKETS):
                raise AssertionError(f"{label}: missing prompt buckets: {buckets.keys()}")

            for prompt in buckets["direct_color"]:
                assert_contains(prompt, target_color, f"{label}:direct_color")
                assert_contains(prompt, "bowl", f"{label}:direct_color")

            for prompt in buckets["position_ordinal"]:
                position_terms = [
                    target_position,
                    "1st",
                    "2nd",
                    "3rd",
                    "middle",
                    "leftmost",
                    "rightmost",
                ]
                if not any(term in prompt for term in position_terms):
                    raise AssertionError(f"{label}: bad position prompt: {prompt!r}")

            other_colors = [color for color in COLORS if color != target_color]
            exclusion_prompt = buckets["exclusion"][0]
            assert_contains(exclusion_prompt, f"not {other_colors[0]}", f"{label}:exclusion")
            assert_contains(exclusion_prompt, f"not {other_colors[1]}", f"{label}:exclusion")
            if f"not {target_color}" in exclusion_prompt:
                raise AssertionError(f"{label}: exclusion rejects target color: {exclusion_prompt!r}")


def check_relative_prompts_are_layout_consistent() -> None:
    rendered = ("green", "blue", "red")
    left_prompts = prompt_buckets_for_rendering(rendered, "left")["relative_color"]
    center_prompts = prompt_buckets_for_rendering(rendered, "center")["relative_color"]
    right_prompts = prompt_buckets_for_rendering(rendered, "right")["relative_color"]

    if not any("left of the blue bowl" in prompt for prompt in left_prompts):
        raise AssertionError(f"left target relative prompt wrong: {left_prompts}")
    if not any("right of the green bowl" in prompt for prompt in center_prompts):
        raise AssertionError(f"center target missing right-of-left-color: {center_prompts}")
    if not any("left of the red bowl" in prompt for prompt in center_prompts):
        raise AssertionError(f"center target missing left-of-right-color: {center_prompts}")
    if not any("between the green bowl and the red bowl" in prompt for prompt in center_prompts):
        raise AssertionError(f"center target missing between prompt: {center_prompts}")
    if not any("right of the blue bowl" in prompt for prompt in right_prompts):
        raise AssertionError(f"right target relative prompt wrong: {right_prompts}")


def check_prompt_modes() -> None:
    colors_by_position = ("red", "blue", "green")
    target_position = "left"
    label = "banana_blue_ep000_target_red"

    mixed = prompts_for_rendering(colors_by_position, target_position, "eval_mixed", label)
    if [bucket for bucket, _prompt in mixed] != list(PROMPT_BUCKETS):
        raise AssertionError(f"eval_mixed did not emit one prompt per bucket: {mixed}")

    sampled = prompts_for_rendering(colors_by_position, target_position, "eval_sampled", label)
    if len(sampled) != 1 or sampled[0][0] not in PROMPT_BUCKETS:
        raise AssertionError(f"eval_sampled should emit one bucket prompt: {sampled}")

    all_eval = prompts_for_rendering(colors_by_position, target_position, "all_eval", label)
    if len(all_eval) <= len(mixed):
        raise AssertionError("all_eval should include every template, not just one per bucket")

    position = prompts_for_rendering(colors_by_position, target_position, "position", label)
    if position != [("position", "put the banana in the left bowl")]:
        raise AssertionError(f"legacy position prompt changed: {position}")


def check_eval_mixed_rotates_templates() -> None:
    seen = defaultdict(set)
    expected = {}

    for original_target in COLORS:
        target_position = rendered_target_position(original_target)
        for rendered_target_color in COLORS:
            swap = swap_to_make_target_color(original_target, rendered_target_color)
            colors_by_position = colors_by_position_for_swap(swap)
            buckets = prompt_buckets_for_rendering(colors_by_position, target_position)
            for bucket in PROMPT_BUCKETS:
                key = (original_target, rendered_target_color, bucket)
                expected[key] = set(buckets[bucket])
            for ep in range(60):
                label = f"banana_{original_target}_ep{ep:03d}_target_{rendered_target_color}"
                for bucket, prompt in prompts_for_rendering(
                    colors_by_position,
                    target_position,
                    "eval_mixed",
                    label,
                ):
                    seen[(original_target, rendered_target_color, bucket)].add(prompt)

    for key, expected_prompts in expected.items():
        if seen[key] != expected_prompts:
            raise AssertionError(
                f"eval_mixed did not rotate through all templates for {key}: "
                f"seen={sorted(seen[key])}, expected={sorted(expected_prompts)}"
            )


def main() -> None:
    check_eval_prompt_buckets_use_rendered_colors()
    check_relative_prompts_are_layout_consistent()
    check_prompt_modes()
    check_eval_mixed_rotates_templates()
    print("banana prompt template checks passed")


if __name__ == "__main__":
    main()
