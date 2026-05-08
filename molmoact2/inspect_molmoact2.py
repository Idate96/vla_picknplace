#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw

from datasets import load_dataset
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
FRONT_IMAGE_KEY = "observation.images.front"
OUT_DIR = Path("outputs/molmoact2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MolmoAct2 inputs and compare with our dataset.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def save_labeled_sheet(images: list[tuple[str, Image.Image]], path: Path) -> None:
    tile_w, tile_h = 360, 310
    sheet = Image.new("RGB", (tile_w * len(images), tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(images):
        preview = image.copy()
        preview.thumbnail((tile_w - 20, tile_h - 55))
        x0 = i * tile_w + (tile_w - preview.width) // 2
        y0 = 12
        sheet.paste(preview, (x0, y0))
        draw.text((i * tile_w + 12, tile_h - 32), label, fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def tensor_to_pil(image) -> Image.Image:
    array = image.detach().cpu()
    if str(array.dtype) != "torch.uint8":
        array = (array.float().clamp(0, 1) * 255).to(torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def load_norm_stats() -> dict:
    path = hf_hub_download(MODEL_ID, "norm_stats.json")
    return json.loads(Path(path).read_text())["metadata_by_tag"][NORM_TAG]


def load_frame_table(repo_id: str, revision: str, root: Path | None):
    if root is None:
        return load_dataset(repo_id, revision=revision, split="train")
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no parquet files found under {root / 'data'}")
    return load_dataset("parquet", data_files=[str(path) for path in data_files], split="train")


def compare_ranges(name: str, values: np.ndarray, stats: dict) -> list[str]:
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    names = stats["names"]
    lines = []
    for i, joint_name in enumerate(names):
        outside = mins[i] < q01[i] or maxs[i] > q99[i]
        marker = "WARN" if outside else "ok"
        lines.append(
            f"{marker:4} {name}.{joint_name:13} dataset=[{mins[i]:8.2f}, {maxs[i]:8.2f}] "
            f"molmo_q01_q99=[{q01[i]:8.2f}, {q99[i]:8.2f}]"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(MODEL_ID, files_metadata=True)
    norm = load_norm_stats()
    meta = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.dataset_revision,
    )
    if FRONT_IMAGE_KEY not in meta.camera_keys:
        raise SystemExit(f"Dataset must contain {FRONT_IMAGE_KEY}; found {meta.camera_keys}")

    top = Image.open(hf_hub_download(MODEL_ID, "assets/sample_realsense_top_rgb.png")).convert("RGB")
    side = Image.open(hf_hub_download(MODEL_ID, "assets/sample_realsense_side_rgb.png")).convert("RGB")

    dataset_kwargs = {"revision": args.dataset_revision, "episodes": [args.episode]}
    if args.dataset_root is not None:
        dataset_kwargs["root"] = args.dataset_root
    ds = LeRobotDataset(args.dataset_repo_id, **dataset_kwargs)
    first = ds[0]
    our_image = tensor_to_pil(first[FRONT_IMAGE_KEY])

    save_labeled_sheet(
        [
            ("MolmoAct2 sample: realsense_top_rgb", top),
            ("MolmoAct2 sample: realsense_side_rgb", side),
            (f"Our dataset: {FRONT_IMAGE_KEY}", our_image),
        ],
        args.out_dir / "image_inputs.jpg",
    )

    table = load_frame_table(args.dataset_repo_id, args.dataset_revision, args.dataset_root)
    action = np.asarray(table["action"], dtype=np.float32)
    state = np.asarray(table["observation.state"], dtype=np.float32)

    print(f"Model: {MODEL_ID}")
    print(f"Model sha: {info.sha}")
    print(f"Norm tag: {NORM_TAG}")
    print(f"Action horizon: {norm['action_horizon']}")
    print(f"n_action_steps: {norm['n_action_steps']}")
    print(f"Model norm camera_keys: {norm['camera_keys']} (empty means not fixed by metadata)")
    print("Model sample images:")
    print(f"  sample_realsense_top_rgb.png: {top.size} RGB")
    print(f"  sample_realsense_side_rgb.png: {side.size} RGB")
    print("Our dataset:")
    print(f"  repo: {args.dataset_repo_id}")
    if args.dataset_root is not None:
        print(f"  root: {args.dataset_root}")
    print(f"  robot_type: {meta.robot_type}")
    print(f"  camera_keys: {meta.camera_keys}")
    print(f"  selected camera: {FRONT_IMAGE_KEY}")
    print(f"  first image: {our_image.size} RGB")
    print(f"  frames: {meta.total_frames}, episodes: {meta.total_episodes}, fps: {meta.fps}")
    print("\nRange comparison against MolmoAct2 q01/q99:")
    for line in compare_ranges("state", state, norm["state_stats"]):
        print(" ", line)
    for line in compare_ranges("action", action, norm["action_stats"]):
        print(" ", line)
    print(f"\nWrote image sheet: {args.out_dir / 'image_inputs.jpg'}")


if __name__ == "__main__":
    main()
