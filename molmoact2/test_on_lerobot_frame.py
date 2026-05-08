#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

from lerobot.datasets import LeRobotDataset


MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
FRONT_IMAGE_KEY = "observation.images.front"
TASK = "pickup screwdriver"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or run MolmoAct2 on one LeRobot frame.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Capture CUDA graphs during action prediction. Faster after warmup, but uses more memory.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu()
    if array.dtype != torch.uint8:
        array = (array.float().clamp(0, 1) * 255).to(torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def norm_metadata() -> dict:
    path = hf_hub_download(MODEL_ID, "norm_stats.json")
    return json.loads(Path(path).read_text())["metadata_by_tag"][NORM_TAG]


def model_snapshot_path() -> str:
    return snapshot_download(MODEL_ID)


def autocast_context(device: str, dtype: torch.dtype):
    if device.startswith("cuda") and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def state_warnings(state: np.ndarray, stats: dict) -> list[str]:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    warnings = []
    for i, name in enumerate(stats["names"]):
        if state[i] < q01[i] or state[i] > q99[i]:
            warnings.append(
                f"{name}: state={state[i]:.2f} outside Molmo q01/q99 [{q01[i]:.2f}, {q99[i]:.2f}]"
            )
    return warnings


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.run_model = False

    dataset_kwargs = {"revision": args.dataset_revision, "episodes": [args.episode]}
    if args.dataset_root is not None:
        dataset_kwargs["root"] = args.dataset_root
    dataset = LeRobotDataset(args.dataset_repo_id, **dataset_kwargs)
    if FRONT_IMAGE_KEY not in dataset.meta.camera_keys:
        raise SystemExit(f"Dataset must contain {FRONT_IMAGE_KEY}; found {dataset.meta.camera_keys}")
    item = dataset[args.frame]
    image = tensor_to_pil(item[FRONT_IMAGE_KEY])
    state = item["observation.state"].detach().cpu().numpy().astype(np.float32)
    stats = norm_metadata()

    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {args.dataset_repo_id}")
    if args.dataset_root is not None:
        print(f"Dataset root: {args.dataset_root}")
    print(f"Episode/frame: {args.episode}/{args.frame}")
    print(f"Task: {args.task!r}")
    print(f"Image input: [{FRONT_IMAGE_KEY}] -> one RGB image {image.size}")
    print(f"State: {np.array2string(state, precision=2, suppress_small=True)}")
    print(f"Norm tag: {NORM_TAG}")
    print(f"Expected action shape: ({stats['n_action_steps']}, {len(stats['action_stats']['names'])})")

    warnings = state_warnings(state, stats["state_stats"])
    if warnings:
        print("State range warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("State range warnings: none")

    summary = {
        "model": MODEL_ID,
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "episode": args.episode,
        "frame": args.frame,
        "task": args.task,
        "camera_key": FRONT_IMAGE_KEY,
        "image_size": list(image.size),
        "state": state.astype(float).tolist(),
        "state_warnings": warnings,
        "norm_tag": NORM_TAG,
        "run_model": bool(args.run_model),
    }

    if not args.run_model:
        print("Dry run only. Re-run with --run-model on a GPU machine to load the 5B checkpoint.")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2))
            print(f"Wrote summary: {args.output}")
        return

    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_by_name = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    model_path = model_snapshot_path()
    torch_dtype = dtype_by_name[args.dtype]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(args.device).eval()

    with torch.inference_mode(), autocast_context(args.device, torch_dtype):
        out = model.predict_action(
            processor=processor,
            images=[image],
            task=args.task,
            state=state,
            norm_tag=NORM_TAG,
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=10,
            normalize_language=True,
            enable_cuda_graph=args.enable_cuda_graph,
        )

    actions = out.actions.detach().cpu().numpy()
    action_horizon = actions[0] if actions.ndim == 3 and actions.shape[0] == 1 else actions
    print(f"Actions shape: {actions.shape}")
    print("First target:")
    print(np.array2string(action_horizon[0], precision=3, suppress_small=True))
    summary["actions_shape"] = list(actions.shape)
    summary["action_horizon"] = action_horizon.astype(float).tolist()
    summary["first_target"] = action_horizon[0].astype(float).tolist()
    summary["first_action"] = action_horizon.astype(float).tolist()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary: {args.output}")


if __name__ == "__main__":
    main()
