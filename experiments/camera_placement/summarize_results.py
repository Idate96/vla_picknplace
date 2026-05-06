"""Summarize the HW3 camera-placement ablation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CAMERAS = ("top_wrist", "angle")
MODES = ("train", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def eval_name(mode: str, checkpoint_step: str | None) -> str:
    if checkpoint_step:
        return f"eval_{mode}_seed1042_{checkpoint_step}.json"
    return f"eval_{mode}_seed1042.json"


def summarize_camera(
    camera: str,
    dataset_root: Path,
    run_root: Path,
    checkpoint_step: str | None,
) -> dict:
    dataset = load_json(dataset_root / camera / "export_summary.json")
    output = run_root / camera
    evals = {
        mode: load_json(output / eval_name(mode, checkpoint_step))
        for mode in MODES
    }
    rates = [
        item["success_rate"]
        for item in evals.values()
        if item is not None and "success_rate" in item
    ]
    return {
        "dataset": {
            "saved_episodes": (dataset or {}).get("saved_episodes"),
            "attempts": (dataset or {}).get("attempts"),
            "discarded_failures": (dataset or {}).get("discarded_failures"),
            "attempt_success_rate": (dataset or {}).get("attempt_success_rate"),
            "cube_pos_std": (dataset or {}).get("cube_pos_std"),
            "goal_pos_std": (dataset or {}).get("goal_pos_std"),
            "skipped_stationary_frames": (dataset or {}).get(
                "skipped_stationary_frames"
            ),
        },
        "eval": {
            mode: {
                "successes": (evals[mode] or {}).get("successes"),
                "episodes": (evals[mode] or {}).get("episodes"),
                "success_rate": (evals[mode] or {}).get("success_rate"),
            }
            for mode in MODES
        },
        "mean_success_rate": sum(rates) / len(rates) if rates else None,
    }


def main() -> None:
    args = parse_args()
    result = {
        "checkpoint_step": args.checkpoint_step,
        "dataset_root": str(args.dataset_root),
        "run_root": str(args.run_root),
        "cameras": {
            camera: summarize_camera(
                camera,
                args.dataset_root,
                args.run_root,
                args.checkpoint_step,
            )
            for camera in CAMERAS
        },
    }
    complete = all(
        result["cameras"][camera]["eval"][mode]["success_rate"] is not None
        for camera in CAMERAS
        for mode in MODES
    )
    result["complete"] = complete
    if complete:
        result["recommendation"] = max(
            CAMERAS,
            key=lambda camera: result["cameras"][camera]["mean_success_rate"],
        )
    else:
        result["recommendation"] = None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
