#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a public SO100/SO101 joint-space control smoke from a MolmoAct2 "
            "one-frame inference JSON. This is not a physics/task-success sim."
        )
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-body-step",
        type=float,
        default=6.0,
        help="Max simulated body-joint target change per 30 Hz command, in degrees.",
    )
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=10.0,
        help="Max simulated gripper target change per 30 Hz command, in LeRobot 0..100 units.",
    )
    parser.add_argument(
        "--tracking-gain",
        type=float,
        default=1.0,
        help="Fraction of the clipped target reached each simulated step.",
    )
    parser.add_argument(
        "--skip-model-bounds",
        action="store_true",
        help="Skip Hugging Face norm-stat download and q01/q99 warnings.",
    )
    parser.add_argument(
        "--strict-model-bounds",
        action="store_true",
        help="Exit nonzero when initial state or targets sit outside MolmoAct2 q01/q99.",
    )
    return parser.parse_args()


def as_array(value: object, name: str) -> np.ndarray:
    if value is None:
        raise SystemExit(f"missing {name}")
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise SystemExit(f"{name} contains NaN or inf")
    return array


def load_action_horizon(data: dict) -> np.ndarray:
    if "action_horizon" in data:
        actions = as_array(data["action_horizon"], "action_horizon")
    elif "actions" in data:
        actions = as_array(data["actions"], "actions")
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
    elif "first_action" in data:
        actions = as_array(data["first_action"], "first_action")
    else:
        raise SystemExit("model output must contain action_horizon, actions, or first_action")

    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
        raise SystemExit(f"expected action horizon shape [T, 6], got {list(actions.shape)}")
    return actions


def load_norm_stats() -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, "norm_stats.json")
    root = json.loads(Path(path).read_text())
    return root["metadata_by_tag"][NORM_TAG]


def range_warnings(values: np.ndarray, stats: dict, label: str) -> list[str]:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    warnings = []
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    for index, name in enumerate(JOINT_NAMES):
        if mins[index] < q01[index] or maxs[index] > q99[index]:
            warnings.append(
                f"{label}.{name}=[{mins[index]:.2f},{maxs[index]:.2f}] "
                f"outside Molmo q01/q99 [{q01[index]:.2f},{q99[index]:.2f}]"
            )
    return warnings


def simulate(
    initial_state: np.ndarray,
    action_horizon: np.ndarray,
    max_body_step: float,
    max_gripper_step: float,
    tracking_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < tracking_gain <= 1.0:
        raise SystemExit("--tracking-gain must be in (0, 1]")
    max_step = np.asarray([max_body_step] * 5 + [max_gripper_step], dtype=np.float32)
    if (max_step <= 0).any():
        raise SystemExit("--max-body-step and --max-gripper-step must be positive")

    state = initial_state.astype(np.float32).copy()
    sent_targets = []
    simulated_states = [state.copy()]
    was_clipped = []
    for target in action_horizon:
        delta = target - state
        clipped_delta = np.clip(delta, -max_step, max_step)
        sent = state + clipped_delta
        state = state + tracking_gain * (sent - state)
        sent_targets.append(sent.copy())
        simulated_states.append(state.copy())
        was_clipped.append(bool(np.any(np.abs(delta) > max_step + 1e-6)))
    return (
        np.asarray(sent_targets, dtype=np.float32),
        np.asarray(simulated_states, dtype=np.float32),
        np.asarray(was_clipped, dtype=bool),
    )


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    data = json.loads(args.model_output.read_text())
    initial_state = as_array(data.get("state"), "state")
    if initial_state.shape != (len(JOINT_NAMES),):
        raise SystemExit(f"expected state shape [6], got {list(initial_state.shape)}")
    action_horizon = load_action_horizon(data)

    sent_targets, simulated_states, was_clipped = simulate(
        initial_state=initial_state,
        action_horizon=action_horizon,
        max_body_step=args.max_body_step,
        max_gripper_step=args.max_gripper_step,
        tracking_gain=args.tracking_gain,
    )

    initial_warnings: list[str] = []
    target_warnings: list[str] = []
    if not args.skip_model_bounds:
        norm = load_norm_stats()
        initial_warnings = range_warnings(
            initial_state.reshape(1, -1),
            norm["state_stats"],
            "initial_state",
        )
        target_warnings = range_warnings(action_horizon, norm["action_stats"], "target")

    raw_target_delta = action_horizon - simulated_states[:-1]
    sent_delta = sent_targets - simulated_states[:-1]
    summary = {
        "simulator": {
            "type": "joint_space_absolute_target",
            "note": "SO100/SO101 command-path smoke only; no physics, contacts, objects, or visual feedback.",
            "fps": args.fps,
            "max_body_step": args.max_body_step,
            "max_gripper_step": args.max_gripper_step,
            "tracking_gain": args.tracking_gain,
        },
        "model": data.get("model", MODEL_ID),
        "norm_tag": data.get("norm_tag", NORM_TAG),
        "source_model_output": str(args.model_output),
        "dataset_repo_id": data.get("dataset_repo_id"),
        "episode": data.get("episode"),
        "frame": data.get("frame"),
        "task": data.get("task"),
        "joint_names": JOINT_NAMES,
        "horizon_steps": int(action_horizon.shape[0]),
        "duration_s": float(action_horizon.shape[0] / args.fps),
        "initial_state": initial_state.astype(float).tolist(),
        "final_state": simulated_states[-1].astype(float).tolist(),
        "target_min": action_horizon.min(axis=0).astype(float).tolist(),
        "target_max": action_horizon.max(axis=0).astype(float).tolist(),
        "max_abs_raw_target_delta": np.abs(raw_target_delta).max(axis=0).astype(float).tolist(),
        "max_abs_sent_delta": np.abs(sent_delta).max(axis=0).astype(float).tolist(),
        "clipped_steps": int(was_clipped.sum()),
        "initial_state_warnings": initial_warnings,
        "target_warnings": target_warnings,
        "sent_targets": sent_targets.astype(float).tolist(),
        "simulated_states": simulated_states.astype(float).tolist(),
    }

    print("Simulator: joint-space absolute-target command smoke")
    print("Physics/task success: not modeled")
    print(f"Horizon: {summary['horizon_steps']} steps ({summary['duration_s']:.2f}s at {args.fps:g} Hz)")
    print(f"Clipped steps: {summary['clipped_steps']}/{summary['horizon_steps']}")
    print(f"Initial state: {np.array2string(initial_state, precision=2, suppress_small=True)}")
    print(f"Final state:   {np.array2string(simulated_states[-1], precision=2, suppress_small=True)}")
    for warning in initial_warnings + target_warnings:
        print(f"Warning: {warning}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary: {args.output}")

    if args.strict_model_bounds and (initial_warnings or target_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
