#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


DEFAULT_REPO_ID = "carmensc/record-test-screwdriver"
DEFAULT_OUTPUT_DIR = Path("outputs/playground/record-test-screwdriver")

PLOT_COLORS_BGR = [
    (36, 75, 220),
    (60, 160, 60),
    (180, 90, 30),
    (150, 70, 170),
    (40, 150, 190),
    (110, 110, 110),
    (220, 120, 40),
    (40, 120, 220),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a LeRobot dataset episode and check ACT compatibility."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--contact-frames", type=int, default=12)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-contact-sheet", action="store_true")
    return parser.parse_args()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def episode_record(meta: LeRobotDatasetMetadata, episode_index: int) -> dict[str, Any]:
    for row in meta.episodes:
        if int(row["episode_index"]) == episode_index:
            return dict(row)
    raise ValueError(f"Episode {episode_index} not found. Dataset has {meta.total_episodes} episodes.")


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    array = image.detach().cpu()
    if array.dtype != torch.uint8:
        array = (array.float().clamp(0, 1) * 255).to(torch.uint8)
    array = array.permute(1, 2, 0).numpy()
    return np.ascontiguousarray(array)


def short_feature_names(names: list[str] | None, width: int) -> list[str]:
    if not names:
        return [f"d{i}" for i in range(width)]
    cleaned = []
    for i, name in enumerate(names[:width]):
        text = str(name)
        if text.endswith(".pos"):
            text = text[:-4]
        cleaned.append(text)
    return cleaned


def compatibility_report(meta: LeRobotDatasetMetadata) -> tuple[dict[str, Any], list[str]]:
    policy_features = dataset_to_policy_features(meta.features)
    visual_features = {
        key: value for key, value in policy_features.items() if value.type is FeatureType.VISUAL
    }
    action_feature = policy_features.get(ACTION)
    state_feature = policy_features.get(OBS_STATE)

    messages = []
    ok = True

    if not visual_features:
        ok = False
        messages.append("missing visual observation; ACT needs at least one image or environment state")
    else:
        messages.append(f"visual inputs: {', '.join(visual_features)}")

    if action_feature is None:
        ok = False
        messages.append("missing action output")
    else:
        messages.append(f"action shape: {tuple(action_feature.shape)}")

    if state_feature is None:
        messages.append("observation.state is absent; ACT can run with images only, but proprioception is useful")
    else:
        messages.append(f"state shape: {tuple(state_feature.shape)}")

    episode_lengths = [int(row["length"]) for row in meta.episodes]
    min_episode_length = min(episode_lengths) if episode_lengths else 0
    default_act_chunk_size = 100
    if min_episode_length < default_act_chunk_size:
        messages.append(
            f"shortest episode has {min_episode_length} frames; lower ACT chunk_size below "
            f"{default_act_chunk_size}"
        )
    else:
        messages.append(
            f"shortest episode has {min_episode_length} frames; default ACT chunk_size="
            f"{default_act_chunk_size} fits"
        )

    report = {
        "act_compatible": ok,
        "policy_input_features": {
            key: {"type": value.type.name, "shape": list(value.shape)}
            for key, value in policy_features.items()
            if value.type is not FeatureType.ACTION
        },
        "policy_output_features": {
            key: {"type": value.type.name, "shape": list(value.shape)}
            for key, value in policy_features.items()
            if value.type is FeatureType.ACTION
        },
        "min_episode_length": min_episode_length,
    }
    return report, messages


def collect_series(dataset: LeRobotDataset) -> dict[str, np.ndarray]:
    actions = []
    states = []
    timestamps = []
    for i in range(len(dataset)):
        item = dataset[i]
        actions.append(item[ACTION].detach().cpu().numpy())
        states.append(item[OBS_STATE].detach().cpu().numpy())
        timestamps.append(float(item["timestamp"]))
    return {
        ACTION: np.stack(actions),
        OBS_STATE: np.stack(states),
        "timestamp": np.asarray(timestamps, dtype=np.float32),
    }


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_plot(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    values: np.ndarray,
    frame_index: int,
    title: str,
    names: list[str],
) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (248, 248, 248), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (190, 190, 190), 1)
    draw_text(canvas, title, (x, y - 10), scale=0.55, color=(20, 20, 20), thickness=1)

    if len(values) < 2:
        return

    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    xs = np.linspace(x + 8, x + w - 8, len(values)).astype(np.int32)

    for dim in range(values.shape[1]):
        normalized = (values[:, dim] - mins[dim]) / spans[dim]
        ys = (y + h - 12 - normalized * (h - 24)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape((-1, 1, 2))
        color = PLOT_COLORS_BGR[dim % len(PLOT_COLORS_BGR)]
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

    current_x = int(xs[min(frame_index, len(xs) - 1)])
    cv2.line(canvas, (current_x, y + 1), (current_x, y + h - 1), (25, 25, 25), 1)

    legend_y = y + h + 22
    for dim, name in enumerate(names[: values.shape[1]]):
        lx = x + (dim % 3) * 155
        ly = legend_y + (dim // 3) * 20
        color = PLOT_COLORS_BGR[dim % len(PLOT_COLORS_BGR)]
        cv2.rectangle(canvas, (lx, ly - 10), (lx + 10, ly), color, -1)
        draw_text(canvas, name[:16], (lx + 16, ly), scale=0.42, color=(50, 50, 50), thickness=1)


def render_frame(
    image_rgb: np.ndarray,
    series: dict[str, np.ndarray],
    frame_index: int,
    metadata: dict[str, Any],
    action_names: list[str],
    state_names: list[str],
) -> np.ndarray:
    canvas = np.full((720, 1280, 3), 238, dtype=np.uint8)
    canvas[:, :760] = (231, 233, 235)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image_bgr, (720, 540), interpolation=cv2.INTER_AREA)
    canvas[92:632, 20:740] = image_resized

    draw_text(canvas, metadata["title"], (20, 38), scale=0.72, color=(20, 20, 20), thickness=2)
    draw_text(canvas, metadata["subtitle"], (20, 68), scale=0.52, color=(70, 70, 70), thickness=1)
    draw_text(
        canvas,
        f"frame {frame_index + 1}/{len(series[ACTION])}  t={series['timestamp'][frame_index]:.2f}s",
        (790, 42),
        scale=0.62,
        color=(20, 20, 20),
        thickness=2,
    )

    draw_plot(canvas, (790, 90, 450, 190), series[ACTION], frame_index, "action (per-dim normalized)", action_names)
    draw_plot(
        canvas,
        (790, 390, 450, 190),
        series[OBS_STATE],
        frame_index,
        "observation.state (per-dim normalized)",
        state_names,
    )

    action_now = series[ACTION][frame_index]
    state_now = series[OBS_STATE][frame_index]
    draw_text(canvas, "current action", (790, 635), scale=0.48, color=(20, 20, 20), thickness=1)
    draw_text(canvas, np.array2string(action_now, precision=1, suppress_small=True), (790, 660), scale=0.43)
    draw_text(canvas, "current state", (790, 690), scale=0.48, color=(20, 20, 20), thickness=1)
    draw_text(canvas, np.array2string(state_now, precision=1, suppress_small=True), (920, 690), scale=0.43)
    return canvas


def render_video(
    dataset: LeRobotDataset,
    series: dict[str, np.ndarray],
    camera_key: str,
    output_path: Path,
    stride: int,
    max_frames: int | None,
    metadata: dict[str, Any],
    action_names: list[str],
    state_names: list[str],
) -> None:
    frame_ids = list(range(0, len(dataset), stride))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    output_fps = max(float(dataset.fps) / stride, 1.0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for frame_id in frame_ids:
        item = dataset[frame_id]
        image_rgb = tensor_to_rgb_uint8(item[camera_key])
        canvas = render_frame(image_rgb, series, frame_id, metadata, action_names, state_names)
        writer.write(canvas)

    writer.release()


def render_contact_sheet(
    dataset: LeRobotDataset,
    camera_key: str,
    output_path: Path,
    count: int,
) -> None:
    if count <= 0:
        return
    frame_ids = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    thumbs = []
    for frame_id in frame_ids:
        item = dataset[int(frame_id)]
        image = Image.fromarray(tensor_to_rgb_uint8(item[camera_key]))
        image.thumbnail((240, 180))
        tile = Image.new("RGB", (260, 220), "white")
        tile.paste(image, ((260 - image.width) // 2, 12))
        draw = ImageDraw.Draw(tile)
        draw.text((12, 194), f"frame {int(frame_id)}  t={float(item['timestamp']):.2f}s", fill=(20, 20, 20))
        thumbs.append(tile)

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 260, rows * 220), (236, 236, 236))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 260, (i // cols) * 220))
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    episode = episode_record(meta, args.episode)
    report, messages = compatibility_report(meta)

    dataset = LeRobotDataset(
        args.repo_id,
        revision=args.revision,
        episodes=[args.episode],
        video_backend=args.video_backend,
        download_videos=True,
    )
    series = collect_series(dataset)

    camera_key = dataset.meta.camera_keys[0]
    action_names = short_feature_names(meta.features[ACTION].get("names"), series[ACTION].shape[1])
    state_names = short_feature_names(meta.features[OBS_STATE].get("names"), series[OBS_STATE].shape[1])

    safe_repo = args.repo_id.replace("/", "_")
    prefix = args.out_dir / f"{safe_repo}_episode_{args.episode:03d}"
    video_path = prefix.with_suffix(".mp4")
    sheet_path = prefix.with_name(prefix.name + "_contact_sheet.jpg")
    summary_path = prefix.with_name(prefix.name + "_summary.json")

    render_meta = {
        "title": f"{args.repo_id} / episode {args.episode}",
        "subtitle": f"task: {', '.join(episode['tasks'])} | fps: {meta.fps} | camera: {camera_key}",
    }

    if not args.skip_video:
        render_video(
            dataset,
            series,
            camera_key,
            video_path,
            args.stride,
            args.max_frames,
            render_meta,
            action_names,
            state_names,
        )

    if not args.skip_contact_sheet:
        render_contact_sheet(dataset, camera_key, sheet_path, args.contact_frames)

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "resolved_root": str(dataset.root),
        "robot_type": meta.robot_type,
        "fps": meta.fps,
        "total_episodes": meta.total_episodes,
        "total_frames": meta.total_frames,
        "episode": as_jsonable(episode),
        "camera_keys": meta.camera_keys,
        "features": as_jsonable(meta.features),
        "compatibility": report,
        "outputs": {
            "video": str(video_path) if not args.skip_video else None,
            "contact_sheet": str(sheet_path) if not args.skip_contact_sheet else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Dataset: {args.repo_id}")
    print(f"Robot: {meta.robot_type} | fps: {meta.fps} | episodes: {meta.total_episodes} | frames: {meta.total_frames}")
    print(f"Episode {args.episode}: {episode['length']} frames | task: {', '.join(episode['tasks'])}")
    print("ACT check:")
    for message in messages:
        print(f"  - {message}")
    print(f"  - compatible: {report['act_compatible']}")
    if not args.skip_video:
        print(f"Wrote video: {video_path}")
    if not args.skip_contact_sheet:
        print(f"Wrote contact sheet: {sheet_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
