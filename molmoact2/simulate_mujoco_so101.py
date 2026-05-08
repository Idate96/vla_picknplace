#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from simulate_joint_control import JOINT_NAMES, as_array, load_action_horizon


ROBOTSTUDIO_REF = "fda892cba81032c46c40976a48c9ceadbf40a9ca"
ROBOTSTUDIO_RAW = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/"
    f"{ROBOTSTUDIO_REF}/Simulation/SO101"
)
SCENE_XML = "scene.xml"
ROBOT_XML = "so101_new_calib.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a MolmoAct2 6D action horizon into the public RobotStudio "
            "SO101 MuJoCo model. This is a robot physics smoke, not a screwdriver "
            "task environment."
        )
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=Path("outputs/molmoact2/so101_mujoco_assets"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--steps-per-command",
        type=int,
        default=0,
        help="MuJoCo steps per 30 Hz command. Default computes this from model timestep.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use existing assets only; fail if the RobotStudio files are missing.",
    )
    return parser.parse_args()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vla-picknplace-mujoco-smoke"})
    with urllib.request.urlopen(req, timeout=30) as response:
        path.write_bytes(response.read())


def ensure_assets(asset_dir: Path, allow_download: bool) -> Path:
    scene_path = asset_dir / SCENE_XML
    robot_path = asset_dir / ROBOT_XML
    required = [scene_path, robot_path]

    if allow_download:
        for name in (SCENE_XML, ROBOT_XML):
            path = asset_dir / name
            if not path.exists():
                download(f"{ROBOTSTUDIO_RAW}/{name}", path)

        robot_xml = ET.fromstring(robot_path.read_text())
        mesh_files = sorted(
            {
                mesh.attrib["file"]
                for mesh in robot_xml.findall(".//mesh")
                if "file" in mesh.attrib and mesh.attrib["file"].endswith(".stl")
            }
        )
        for mesh_file in mesh_files:
            path = asset_dir / "assets" / mesh_file
            if not path.exists():
                download(f"{ROBOTSTUDIO_RAW}/assets/{mesh_file}", path)

    missing = [str(path) for path in required if not path.exists()]
    if not missing and robot_path.exists():
        mesh_files = re.findall(r'<mesh file="([^"]+\.stl)"', robot_path.read_text())
        missing.extend(
            str(asset_dir / "assets" / mesh_file)
            for mesh_file in mesh_files
            if not (asset_dir / "assets" / mesh_file).exists()
        )
    if missing:
        raise SystemExit("missing MuJoCo assets: " + ", ".join(missing[:5]))
    return scene_path


def actuator_and_joint_ids(model, mujoco_module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actuator_ids = []
    qpos_ids = []
    joint_ranges = []
    for name in JOINT_NAMES:
        actuator_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_ACTUATOR, name)
        joint_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, name)
        if actuator_id < 0 or joint_id < 0:
            raise SystemExit(f"MuJoCo model is missing joint/actuator {name!r}")
        actuator_ids.append(actuator_id)
        qpos_ids.append(model.jnt_qposadr[joint_id])
        joint_ranges.append(model.jnt_range[joint_id])
    return (
        np.asarray(actuator_ids, dtype=np.int32),
        np.asarray(qpos_ids, dtype=np.int32),
        np.asarray(joint_ranges, dtype=np.float32),
    )


def lerobot_to_mujoco(values: np.ndarray, gripper_range: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values, dtype=np.float32)
    out[..., :5] = np.deg2rad(values[..., :5])
    out[..., 5] = np.interp(values[..., 5], [0.0, 100.0], gripper_range)
    return out


def mujoco_to_lerobot(values: np.ndarray, gripper_range: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values, dtype=np.float32)
    out[..., :5] = np.rad2deg(values[..., :5])
    out[..., 5] = np.interp(values[..., 5], gripper_range, [0.0, 100.0])
    return out


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit("install mujoco first: uv pip install --python .venv/bin/python mujoco==3.2.5") from exc

    scene_path = ensure_assets(args.asset_dir, allow_download=not args.no_download)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    actuator_ids, qpos_ids, joint_ranges = actuator_and_joint_ids(model, mujoco)
    ctrl_ranges = model.actuator_ctrlrange[actuator_ids].astype(np.float32)
    gripper_range = ctrl_ranges[5]

    source = json.loads(args.model_output.read_text())
    initial_state = as_array(source.get("state"), "state")
    if initial_state.shape != (len(JOINT_NAMES),):
        raise SystemExit(f"expected state shape [6], got {list(initial_state.shape)}")
    action_horizon = load_action_horizon(source)

    initial_qpos = lerobot_to_mujoco(initial_state, gripper_range)
    clipped_initial_qpos = np.clip(initial_qpos, joint_ranges[:, 0], joint_ranges[:, 1])
    data.qpos[qpos_ids] = clipped_initial_qpos
    data.ctrl[actuator_ids] = np.clip(clipped_initial_qpos, ctrl_ranges[:, 0], ctrl_ranges[:, 1])
    mujoco.mj_forward(model, data)

    if args.steps_per_command > 0:
        steps_per_command = args.steps_per_command
    else:
        steps_per_command = max(1, round((1.0 / args.fps) / model.opt.timestep))

    trajectory_qpos = [data.qpos[qpos_ids].copy()]
    controls = []
    clipped_control_steps = []
    for target in action_horizon:
        ctrl = lerobot_to_mujoco(target, gripper_range)
        clipped_ctrl = np.clip(ctrl, ctrl_ranges[:, 0], ctrl_ranges[:, 1])
        data.ctrl[actuator_ids] = clipped_ctrl
        for _ in range(steps_per_command):
            mujoco.mj_step(model, data)
        trajectory_qpos.append(data.qpos[qpos_ids].copy())
        controls.append(clipped_ctrl.copy())
        clipped_control_steps.append(bool(np.any(np.abs(ctrl - clipped_ctrl) > 1e-6)))

    trajectory_qpos = np.asarray(trajectory_qpos, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    clipped_control_steps = np.asarray(clipped_control_steps, dtype=bool)
    trajectory_lerobot = mujoco_to_lerobot(trajectory_qpos, gripper_range)
    controls_lerobot = mujoco_to_lerobot(controls, gripper_range)
    joint_ranges_lerobot = np.column_stack(
        [
            mujoco_to_lerobot(joint_ranges[:, 0], gripper_range),
            mujoco_to_lerobot(joint_ranges[:, 1], gripper_range),
        ]
    )

    summary = {
        "simulator": {
            "type": "robotstudio_so101_mujoco",
            "note": (
                "Robot physics smoke only; no screwdriver object, task contacts, "
                "camera feedback, or task-success metric."
            ),
            "robotstudio_repo": "https://github.com/TheRobotStudio/SO-ARM100",
            "robotstudio_ref": ROBOTSTUDIO_REF,
            "scene_xml": str(scene_path),
            "fps": args.fps,
            "mujoco_timestep": float(model.opt.timestep),
            "steps_per_command": int(steps_per_command),
            "gripper_mapping_note": (
                "RobotStudio README says LeRobot gripper 0..100 is not reflected "
                "in the MJCF yet; this script maps it linearly onto actuator ctrlrange."
            ),
        },
        "model": source.get("model"),
        "norm_tag": source.get("norm_tag"),
        "source_model_output": str(args.model_output),
        "dataset_repo_id": source.get("dataset_repo_id"),
        "episode": source.get("episode"),
        "frame": source.get("frame"),
        "task": source.get("task"),
        "joint_names": JOINT_NAMES,
        "horizon_steps": int(action_horizon.shape[0]),
        "duration_s": float(action_horizon.shape[0] / args.fps),
        "initial_state_lerobot": initial_state.astype(float).tolist(),
        "initial_qpos_was_clipped": bool(np.any(np.abs(initial_qpos - clipped_initial_qpos) > 1e-6)),
        "joint_ranges_lerobot": joint_ranges_lerobot.astype(float).tolist(),
        "ctrl_ranges_radian": ctrl_ranges.astype(float).tolist(),
        "clipped_control_steps": int(clipped_control_steps.sum()),
        "final_state_lerobot": trajectory_lerobot[-1].astype(float).tolist(),
        "final_qpos_radian": trajectory_qpos[-1].astype(float).tolist(),
        "controls_lerobot": controls_lerobot.astype(float).tolist(),
        "trajectory_lerobot": trajectory_lerobot.astype(float).tolist(),
    }

    print("Simulator: RobotStudio SO101 MuJoCo physics smoke")
    print("Task success: not modeled")
    print(f"Scene: {scene_path}")
    print(f"Horizon: {summary['horizon_steps']} commands; {steps_per_command} MuJoCo steps/command")
    print(f"Initial qpos clipped: {summary['initial_qpos_was_clipped']}")
    print(f"Control-clipped steps: {summary['clipped_control_steps']}/{summary['horizon_steps']}")
    print(
        "Final LeRobot-unit state: "
        + np.array2string(trajectory_lerobot[-1], precision=2, suppress_small=True)
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary: {args.output}")


if __name__ == "__main__":
    main()
