#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from check_finetune_readiness import (
    EXPECTED_JOINTS,
    FRONT_IMAGE_KEY,
    MODEL_ID,
    NORM_TAG,
    check_dataset_metadata,
    check_model_norm,
    compare_ranges,
    load_frame_table,
)


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight a newly collected SO100/SO101 LeRobot dataset for MolmoAct2.",
    )
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument(
        "--max-frame-check-rows",
        type=int,
        default=2000,
        help="Rows to inspect for finite 6D state/action values; 0 checks all rows.",
    )
    parser.add_argument("--skip-ranges", action="store_true")
    parser.add_argument("--skip-frame-check", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--max-image-check-frames", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def convert_check(check) -> Check:
    return Check(check.name, check.status, check.detail)


def check_camera_contract(features: dict) -> Check:
    cameras = sorted(key for key in features if key.startswith("observation.images."))
    if FRONT_IMAGE_KEY not in cameras:
        return Check("camera contract", "BLOCKED", f"missing {FRONT_IMAGE_KEY}; found {cameras}")
    extras = [key for key in cameras if key != FRONT_IMAGE_KEY]
    if extras:
        return Check("camera contract", "WARN", f"{FRONT_IMAGE_KEY} is present; extra cameras will be ignored: {extras}")
    return Check("camera contract", "OK", f"single RGB camera key is {FRONT_IMAGE_KEY}")


def first_present_key(keys: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return None


def check_vector_values(table, key: str, rows_to_check: int) -> tuple[bool, str]:
    if rows_to_check == 0 or rows_to_check >= len(table):
        values = table[key]
    else:
        values = table.select(range(rows_to_check))[key]
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 6:
        return False, f"{key} sampled shape is {list(array.shape)}, expected [N, 6]"
    if not np.isfinite(array).all():
        bad = int(np.size(array) - np.isfinite(array).sum())
        return False, f"{key} has {bad} non-finite sampled values"
    mins = array.min(axis=0)
    maxs = array.max(axis=0)
    ranges = ", ".join(
        f"{name}=[{mins[i]:.2f},{maxs[i]:.2f}]" for i, name in enumerate(EXPECTED_JOINTS)
    )
    return True, f"{key} sampled {len(array)} rows; {ranges}"


def check_frame_table(
    repo_id: str,
    revision: str,
    root: Path | None,
    min_frames: int,
    min_episodes: int,
    max_rows: int,
) -> list[Check]:
    try:
        table = load_frame_table(repo_id, revision, root)
    except Exception as exc:
        return [Check("frame table", "BLOCKED", f"could not load frame rows: {exc}")]

    checks: list[Check] = []
    keys = set(table.column_names)
    frame_count = len(table)
    if frame_count < min_frames:
        checks.append(Check("frame count", "BLOCKED", f"{frame_count} frames, expected at least {min_frames}"))
    else:
        checks.append(Check("frame count", "OK", f"{frame_count} frames"))

    episode_key = first_present_key(keys, ("episode_index", "episode.index", "episode_id"))
    if episode_key is None:
        checks.append(Check("episode count", "WARN", "no episode index column found in frame table"))
    else:
        episode_count = len(set(table[episode_key]))
        status = "OK" if episode_count >= min_episodes else "BLOCKED"
        checks.append(
            Check(
                "episode count",
                status,
                f"{episode_count} episodes from {episode_key}, expected at least {min_episodes}",
            )
        )

    rows_to_check = 0 if max_rows == 0 else min(max_rows, frame_count)
    for key in ("observation.state", "action"):
        if key not in keys:
            checks.append(Check(f"{key} values", "BLOCKED", "missing from frame table"))
            continue
        ok, detail = check_vector_values(table, key, rows_to_check)
        checks.append(Check(f"{key} values", "OK" if ok else "BLOCKED", detail))
    return checks


def image_to_array(image) -> np.ndarray:
    if hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected rank-3 RGB image, got shape {list(array.shape)}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] < 3:
        raise ValueError(f"expected RGB image with at least 3 channels, got shape {list(array.shape)}")
    array = array[..., :3].astype(np.float32)
    if array.max(initial=0) <= 1.0:
        array = array * 255.0
    return array


def check_image_frames(
    repo_id: str,
    revision: str,
    root: Path | None,
    max_frames: int,
) -> Check:
    if max_frames <= 0:
        return Check("image frames", "WARN", "image frame check disabled by max-image-check-frames=0")
    try:
        from lerobot.datasets import LeRobotDataset

        dataset_kwargs = {"revision": revision}
        if root is not None:
            dataset_kwargs["root"] = root
        dataset = LeRobotDataset(repo_id, **dataset_kwargs)
        if FRONT_IMAGE_KEY not in dataset.meta.camera_keys:
            return Check("image frames", "BLOCKED", f"missing {FRONT_IMAGE_KEY}; found {dataset.meta.camera_keys}")
        if len(dataset) == 0:
            return Check("image frames", "BLOCKED", "dataset has no frames")
        indices = np.unique(np.linspace(0, len(dataset) - 1, min(max_frames, len(dataset)), dtype=int))
        stats = []
        shapes = []
        for index in indices:
            image = dataset[int(index)][FRONT_IMAGE_KEY]
            array = image_to_array(image)
            if not np.isfinite(array).all():
                return Check("image frames", "BLOCKED", f"{FRONT_IMAGE_KEY}[{int(index)}] has non-finite pixels")
            stats.append(float(array.std()))
            shapes.append(list(array.shape))
    except Exception as exc:
        return Check("image frames", "BLOCKED", f"could not load sampled RGB frames: {exc}")

    min_std = min(stats) if stats else 0.0
    if min_std <= 1.0:
        return Check(
            "image frames",
            "BLOCKED",
            f"{FRONT_IMAGE_KEY} sampled frame is nearly blank; stds={[round(item, 2) for item in stats]}",
        )
    return Check(
        "image frames",
        "OK",
        f"loaded {len(stats)} sampled RGB frames; shapes={shapes}; stds={[round(item, 2) for item in stats]}",
    )


def print_checks(checks: list[Check]) -> int:
    blockers = [check for check in checks if check.status == "BLOCKED"]
    for check in checks:
        print(f"{check.status:7} {check.name}: {check.detail}")
    print()
    if blockers:
        print("Dataset is not ready for MolmoAct2 collection handoff.")
        print("Blockers:")
        for check in blockers:
            print(f"  - {check.name}: {check.detail}")
        return 1
    print("Dataset passes the MolmoAct2 collection preflight.")
    return 0


def write_json(path: Path, checks: list[Check]) -> None:
    blockers = [check for check in checks if check.status == "BLOCKED"]
    warnings = [check for check in checks if check.status == "WARN"]
    report = {
        "ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "model": MODEL_ID,
        "norm_tag": NORM_TAG,
        "image_key": FRONT_IMAGE_KEY,
        "joint_order": EXPECTED_JOINTS,
        "blockers": [check.__dict__ for check in blockers],
        "warnings": [check.__dict__ for check in warnings],
        "checks": [check.__dict__ for check in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    checks = [convert_check(check_model_norm())]
    dataset_check, meta = check_dataset_metadata(args.dataset_repo_id, args.dataset_revision, args.dataset_root)
    checks.append(convert_check(dataset_check))
    if meta is not None:
        checks.append(check_camera_contract(meta.features))
    if dataset_check.status != "BLOCKED" and not args.skip_frame_check:
        checks.extend(
            check_frame_table(
                args.dataset_repo_id,
                args.dataset_revision,
                args.dataset_root,
                args.min_frames,
                args.min_episodes,
                args.max_frame_check_rows,
            )
        )
    if dataset_check.status != "BLOCKED" and not args.skip_image_check:
        checks.append(
            check_image_frames(
                args.dataset_repo_id,
                args.dataset_revision,
                args.dataset_root,
                args.max_image_check_frames,
            )
        )
    if dataset_check.status != "BLOCKED" and not args.skip_ranges:
        checks.append(convert_check(compare_ranges(args.dataset_repo_id, args.dataset_revision, args.dataset_root)))
    if args.output_json is not None:
        write_json(args.output_json, checks)
    raise SystemExit(print_checks(checks))


if __name__ == "__main__":
    main()
