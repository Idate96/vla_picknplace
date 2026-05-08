#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from simulate_mujoco_so101 import (
    JOINT_NAMES,
    ROBOTSTUDIO_REF,
    actuator_and_joint_ids,
    ensure_assets,
    lerobot_to_mujoco,
    mujoco_to_lerobot,
)


MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
TASK = "pickup screwdriver"
DEFAULT_INITIAL_STATE = [-5.0, 45.0, 35.0, 90.0, -65.0, 34.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop MolmoAct2 rollout in a simple public SO101 MuJoCo scene. "
            "The scene has arm physics and a screwdriver proxy, but is not a "
            "validated task-success benchmark."
        )
    )
    parser.add_argument("--asset-dir", type=Path, default=Path("outputs/molmoact2/so101_mujoco_assets"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--dry-run", action="store_true", help="Render and step without loading MolmoAct2.")
    parser.add_argument("--rollout-steps", type=int, default=1, help="Number of model calls.")
    parser.add_argument("--actions-per-inference", type=int, default=1, help="Targets to execute from each horizon.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--steps-per-command", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--initial-state", nargs=6, type=float, default=DEFAULT_INITIAL_STATE)
    parser.add_argument("--num-steps", type=int, default=10, help="MolmoAct2 diffusion/action sampling steps.")
    parser.add_argument("--no-download", action="store_true", help="Use existing RobotStudio assets only.")
    return parser.parse_args()


def dtype_by_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def autocast_context(device: str, dtype: torch.dtype):
    if device.startswith("cuda") and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def model_snapshot_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID)


def load_policy(device: str, dtype_name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = dtype_by_name(dtype_name)
    model_path = model_snapshot_path()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device).eval()
    return processor, model, torch_dtype


def write_task_scene(asset_dir: Path) -> Path:
    path = asset_dir / "molmoact2_screwdriver_scene.xml"
    path.write_text(
        """<mujoco model="molmoact2_screwdriver_scene">
    <include file="so101_new_calib.xml"/>

    <visual>
        <headlight diffuse="0.65 0.65 0.65" ambient="0.35 0.35 0.35" specular="0 0 0"/>
        <global azimuth="150" elevation="-25"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.45 0.56 0.66" rgb2="0.08 0.09 0.10"
            width="512" height="3072"/>
        <texture type="2d" name="table_grid" builtin="checker" mark="edge" rgb1="0.72 0.72 0.68"
            rgb2="0.62 0.62 0.58" markrgb="0.35 0.35 0.35" width="256" height="256"/>
        <material name="table_grid" texture="table_grid" texuniform="true" texrepeat="5 5"/>
        <material name="target_green" rgba="0.1 0.7 0.28 0.6"/>
        <material name="handle_red" rgba="0.8 0.08 0.04 1"/>
        <material name="shaft_metal" rgba="0.72 0.72 0.76 1"/>
    </asset>

    <worldbody>
        <light pos="0 -0.5 1.2" dir="0 0 -1" directional="true"/>
        <geom name="table" type="box" pos="0.02 0 0.005" size="0.38 0.34 0.005"
            material="table_grid"/>
        <geom name="target_area" type="box" pos="0.11 0.15 0.014" size="0.08 0.045 0.003"
            material="target_green" contype="0" conaffinity="0"/>
        <body name="screwdriver" pos="-0.13 -0.11 0.035" euler="0 0 0.35">
            <freejoint/>
            <geom name="screwdriver_handle" type="capsule" fromto="-0.075 0 0 0.015 0 0"
                size="0.014" mass="0.045" material="handle_red" friction="1.0 0.02 0.001"/>
            <geom name="screwdriver_shaft" type="capsule" fromto="0.015 0 0 0.12 0 0"
                size="0.004" mass="0.02" material="shaft_metal" friction="1.0 0.02 0.001"/>
        </body>
    </worldbody>
</mujoco>
"""
    )
    return path


def make_camera(mujoco_module) -> object:
    camera = mujoco_module.MjvCamera()
    camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 132.0
    camera.elevation = -23.0
    camera.distance = 0.62
    camera.lookat[:] = [-0.04, 0.01, 0.12]
    return camera


def render_image(renderer, data, camera) -> Image.Image:
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()
    return Image.fromarray(rgb)


def predict_action_horizon(processor, model, image: Image.Image, state: np.ndarray, args, torch_dtype) -> np.ndarray:
    with torch.inference_mode(), autocast_context(args.device, torch_dtype):
        out = model.predict_action(
            processor=processor,
            images=[image],
            task=args.task,
            state=state.astype(np.float32),
            norm_tag=NORM_TAG,
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=args.num_steps,
            normalize_language=True,
            enable_cuda_graph=False,
        )
    actions = out.actions.detach().cpu().numpy()
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
        raise RuntimeError(f"unexpected action shape from MolmoAct2: {actions.shape}")
    return actions.astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be positive")
    if args.actions_per_inference <= 0:
        raise SystemExit("--actions-per-inference must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("install mujoco first: uv pip install --python .venv/bin/python mujoco==3.2.5") from exc

    ensure_assets(args.asset_dir, allow_download=not args.no_download)
    scene_path = write_task_scene(args.asset_dir)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    actuator_ids, qpos_ids, joint_ranges = actuator_and_joint_ids(model, mujoco)
    ctrl_ranges = model.actuator_ctrlrange[actuator_ids].astype(np.float32)
    gripper_range = ctrl_ranges[5]

    initial_state = np.asarray(args.initial_state, dtype=np.float32)
    initial_qpos = lerobot_to_mujoco(initial_state, gripper_range)
    clipped_initial_qpos = np.clip(initial_qpos, joint_ranges[:, 0], joint_ranges[:, 1])
    data.qpos[qpos_ids] = clipped_initial_qpos
    data.ctrl[actuator_ids] = np.clip(clipped_initial_qpos, ctrl_ranges[:, 0], ctrl_ranges[:, 1])
    mujoco.mj_forward(model, data)

    steps_per_command = (
        args.steps_per_command
        if args.steps_per_command > 0
        else max(1, round((1.0 / args.fps) / model.opt.timestep))
    )
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = make_camera(mujoco)

    processor = policy = torch_dtype = None
    if not args.dry_run:
        processor, policy, torch_dtype = load_policy(args.device, args.dtype)

    if args.frames_dir is not None:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    records = []
    clipped_controls = 0
    for rollout_index in range(args.rollout_steps):
        state = mujoco_to_lerobot(data.qpos[qpos_ids], gripper_range).astype(np.float32)
        image = render_image(renderer, data, camera)
        if args.frames_dir is not None:
            image.save(args.frames_dir / f"frame_{rollout_index:03d}.jpg", quality=92)

        if args.dry_run:
            horizon = np.repeat(state.reshape(1, -1), args.actions_per_inference, axis=0)
        else:
            assert processor is not None and policy is not None and torch_dtype is not None
            horizon = predict_action_horizon(processor, policy, image, state, args, torch_dtype)

        executed = []
        for target in horizon[: args.actions_per_inference]:
            ctrl = lerobot_to_mujoco(target, gripper_range)
            clipped_ctrl = np.clip(ctrl, ctrl_ranges[:, 0], ctrl_ranges[:, 1])
            if np.any(np.abs(ctrl - clipped_ctrl) > 1e-6):
                clipped_controls += 1
            data.ctrl[actuator_ids] = clipped_ctrl
            for _ in range(steps_per_command):
                mujoco.mj_step(model, data)
            executed.append(mujoco_to_lerobot(clipped_ctrl, gripper_range).astype(float).tolist())

        next_state = mujoco_to_lerobot(data.qpos[qpos_ids], gripper_range).astype(np.float32)
        records.append(
            {
                "rollout_index": rollout_index,
                "state_before": state.astype(float).tolist(),
                "horizon_shape": list(horizon.shape),
                "first_target": horizon[0].astype(float).tolist(),
                "executed_targets": executed,
                "state_after": next_state.astype(float).tolist(),
            }
        )

    final_image = render_image(renderer, data, camera)
    if args.frames_dir is not None:
        final_image.save(args.frames_dir / f"frame_{len(records):03d}_final.jpg", quality=92)

    final_state = mujoco_to_lerobot(data.qpos[qpos_ids], gripper_range).astype(np.float32)
    screwdriver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screwdriver")
    screwdriver_pos = data.xpos[screwdriver_id].copy() if screwdriver_id >= 0 else np.full(3, np.nan)
    target_pos = np.asarray([0.11, 0.15, 0.014], dtype=np.float32)
    screwdriver_target_xy_dist = float(np.linalg.norm(screwdriver_pos[:2] - target_pos[:2]))

    summary = {
        "simulator": {
            "type": "molmoact2_closed_loop_robotstudio_so101_mujoco",
            "note": (
                "Closed-loop model-control smoke with simple screwdriver proxy. "
                "Not a validated task-success benchmark."
            ),
            "robotstudio_repo": "https://github.com/TheRobotStudio/SO-ARM100",
            "robotstudio_ref": ROBOTSTUDIO_REF,
            "scene_xml": str(scene_path),
            "fps": args.fps,
            "mujoco_timestep": float(model.opt.timestep),
            "steps_per_command": int(steps_per_command),
            "image_size": [args.width, args.height],
        },
        "model": MODEL_ID,
        "norm_tag": NORM_TAG,
        "task": args.task,
        "dry_run": bool(args.dry_run),
        "device": args.device,
        "dtype": args.dtype,
        "rollout_steps": args.rollout_steps,
        "actions_per_inference": args.actions_per_inference,
        "joint_names": JOINT_NAMES,
        "initial_state_lerobot": initial_state.astype(float).tolist(),
        "initial_qpos_was_clipped": bool(np.any(np.abs(initial_qpos - clipped_initial_qpos) > 1e-6)),
        "final_state_lerobot": final_state.astype(float).tolist(),
        "screwdriver_pos_xyz": screwdriver_pos.astype(float).tolist(),
        "screwdriver_target_xy_dist": screwdriver_target_xy_dist,
        "clipped_control_count": int(clipped_controls),
        "records": records,
    }

    print("Simulator: closed-loop RobotStudio SO101 MuJoCo smoke")
    print(f"MolmoAct2 loaded: {not args.dry_run}")
    print(f"Rollout steps: {args.rollout_steps}, actions per inference: {args.actions_per_inference}")
    print(f"Final state: {np.array2string(final_state, precision=2, suppress_small=True)}")
    print(f"Screwdriver xy distance to target: {screwdriver_target_xy_dist:.3f} m")
    print(f"Clipped control count: {clipped_controls}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary: {args.output}")


if __name__ == "__main__":
    main()
