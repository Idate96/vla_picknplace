#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from lerobot.datasets import LeRobotDatasetMetadata


MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
FRONT_IMAGE_KEY = "observation.images.front"
UPSTREAM_README_URL = "https://raw.githubusercontent.com/allenai/molmoact2/main/README.md"
LEROBOT_CONFIG_URL = (
    "https://raw.githubusercontent.com/allenai/lerobot/molmoact2-hf-inference/"
    "src/lerobot/policies/molmoact2/configuration_molmoact2.py"
)
LEROBOT_MODEL_URL = (
    "https://raw.githubusercontent.com/allenai/lerobot/molmoact2-hf-inference/"
    "src/lerobot/policies/molmoact2/modeling_molmoact2.py"
)
LEROBOT_TRAIN_URL = (
    "https://raw.githubusercontent.com/allenai/lerobot/molmoact2-hf-inference/"
    "src/lerobot/scripts/lerobot_train.py"
)
EXPECTED_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check readiness for MolmoAct2 SO100/SO101 fine-tuning.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--skip-ranges",
        action="store_true",
        help="Skip full dataset loading and only inspect LeRobot metadata.",
    )
    return parser.parse_args()


def fetch_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "vla-picknplace-readiness"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def try_fetch_text(url: str, timeout: float = 10.0) -> tuple[str | None, Exception | None]:
    try:
        return fetch_text(url, timeout=timeout), None
    except Exception as exc:
        return None, exc


def git_ls_remote(repo: str, ref: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "ls-remote", repo, ref],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    return line[0].split()[0]


def strip_pos(name: str) -> str:
    return name.removeprefix("main_").removesuffix(".pos")


def check_model_norm() -> Check:
    path = hf_hub_download(MODEL_ID, "norm_stats.json")
    root = json.loads(Path(path).read_text())
    tags = root.get("metadata_by_tag", {})
    if NORM_TAG not in tags:
        return Check("model norm", "BLOCKED", f"missing norm tag {NORM_TAG}")
    meta = tags[NORM_TAG]
    names = meta["state_stats"]["names"]
    control = meta["control_mode"]
    if names != EXPECTED_JOINTS or control != "absolute joint pose":
        return Check("model norm", "BLOCKED", f"unexpected names={names} control={control!r}")
    return Check(
        "model norm",
        "OK",
        f"{MODEL_ID} uses {NORM_TAG}, 6D absolute joint-pose actions, horizon={meta['action_horizon']}",
    )


def check_dataset_metadata(
    repo_id: str,
    revision: str,
    root: Path | None,
) -> tuple[Check, LeRobotDatasetMetadata | None]:
    try:
        meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    except Exception as exc:
        return Check("dataset metadata", "BLOCKED", f"could not load metadata: {exc}"), None

    features = meta.features
    missing = [key for key in ("observation.state", "action") if key not in features]
    camera_keys = [key for key in features if key.startswith("observation.images.")]
    if missing:
        return Check("dataset metadata", "BLOCKED", f"missing required features: {missing}"), meta
    if not camera_keys:
        return Check("dataset metadata", "BLOCKED", "missing observation.images.* RGB stream"), meta
    if FRONT_IMAGE_KEY not in camera_keys:
        return Check("dataset metadata", "BLOCKED", f"missing required camera {FRONT_IMAGE_KEY}"), meta

    state = features["observation.state"]
    action = features["action"]
    state_names = [strip_pos(name) for name in state.get("names") or []]
    action_names = [strip_pos(name) for name in action.get("names") or []]
    problems = []
    if tuple(state.get("shape", ())) != (6,):
        problems.append(f"state shape={state.get('shape')}")
    if tuple(action.get("shape", ())) != (6,):
        problems.append(f"action shape={action.get('shape')}")
    if state_names and state_names != EXPECTED_JOINTS:
        problems.append(f"state names={state_names}")
    if action_names and action_names != EXPECTED_JOINTS:
        problems.append(f"action names={action_names}")
    if meta.fps != 30:
        problems.append(f"fps={meta.fps}")
    if problems:
        return Check("dataset metadata", "BLOCKED", "; ".join(problems)), meta

    return (
        Check(
            "dataset metadata",
            "OK",
            f"{repo_id}{f' at {root}' if root else ''} has 30 Hz 6D state/action and {FRONT_IMAGE_KEY}",
        ),
        meta,
    )


def load_frame_table(repo_id: str, revision: str, root: Path | None):
    if root is None:
        return load_dataset(repo_id, revision=revision, split="train")
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no parquet files found under {root / 'data'}")
    return load_dataset("parquet", data_files=[str(path) for path in data_files], split="train")


def compare_ranges(repo_id: str, revision: str, root: Path | None) -> Check:
    norm_path = hf_hub_download(MODEL_ID, "norm_stats.json")
    norm = json.loads(Path(norm_path).read_text())["metadata_by_tag"][NORM_TAG]
    try:
        table = load_frame_table(repo_id, revision, root)
    except Exception as exc:
        return Check("dataset ranges", "BLOCKED", f"could not load frames for range check: {exc}")

    warnings = []
    for key, stats_key in (("observation.state", "state_stats"), ("action", "action_stats")):
        values = np.asarray(table[key], dtype=np.float32)
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        q01 = np.asarray(norm[stats_key]["q01"], dtype=np.float32)
        q99 = np.asarray(norm[stats_key]["q99"], dtype=np.float32)
        for i, name in enumerate(EXPECTED_JOINTS):
            if mins[i] < q01[i] or maxs[i] > q99[i]:
                warnings.append(
                    f"{key}.{name} dataset=[{mins[i]:.2f},{maxs[i]:.2f}] "
                    f"molmo_q01_q99=[{q01[i]:.2f},{q99[i]:.2f}]"
                )

    if warnings:
        shown = "; ".join(warnings[:4])
        suffix = f"; +{len(warnings) - 4} more" if len(warnings) > 4 else ""
        return Check("dataset ranges", "BLOCKED", shown + suffix)
    return Check("dataset ranges", "OK", "state/action ranges are inside MolmoAct2 q01/q99")


def check_brev() -> Check:
    brev = shutil.which("brev")
    if brev is None:
        return Check("brev", "BLOCKED", "brev CLI not found")

    try:
        proc = subprocess.run(
            [brev, "ls"],
            input="n\n",
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Check("brev", "BLOCKED", "brev CLI found, but auth check timed out")

    output = (proc.stdout + "\n" + proc.stderr).strip()
    if "logged out" in output.lower() or "would you like to log in" in output.lower() or "EOF" in output:
        return Check("brev", "BLOCKED", "brev CLI found, but this machine is logged out")
    if proc.returncode != 0:
        return Check("brev", "WARN", f"brev CLI found, but `brev ls` returned {proc.returncode}: {output[:240]}")
    return Check("brev", "OK", "brev CLI is installed and authenticated enough to list instances")


def check_upstream_finetune_code() -> Check:
    readme, readme_exc = try_fetch_text(UPSTREAM_README_URL)
    config, config_exc = try_fetch_text(LEROBOT_CONFIG_URL)
    model, model_exc = try_fetch_text(LEROBOT_MODEL_URL)
    train_script, train_exc = try_fetch_text(LEROBOT_TRAIN_URL)

    molmo_ref = git_ls_remote("https://github.com/allenai/molmoact2.git", "HEAD")
    lerobot_ref = git_ls_remote(
        "https://github.com/allenai/lerobot.git",
        "refs/heads/molmoact2-hf-inference",
    )
    ref_note = []
    if molmo_ref:
        ref_note.append(f"allenai/molmoact2 HEAD={molmo_ref}")
    if lerobot_ref:
        ref_note.append(f"allenai/lerobot molmoact2-hf-inference={lerobot_ref}")

    missing = [
        f"README ({readme_exc})" if readme is None else "",
        f"MolmoAct2 config ({config_exc})" if config is None else "",
        f"MolmoAct2 model ({model_exc})" if model is None else "",
    ]
    missing = [item for item in missing if item]
    if missing:
        return Check(
            "upstream fine-tune code",
            "BLOCKED",
            "could not inspect required upstream raw files, so trainability is unverified: "
            + "; ".join(missing)
            + (f"; {'; '.join(ref_note)}" if ref_note else ""),
        )

    details = []
    if "coming soon" in readme.lower():
        details.append("top-level README says full training/fine-tuning/deployment/evaluation code is coming soon")
    if ref_note:
        details.append("; ".join(ref_note))

    has_train_script = train_script is not None
    if train_script is None and train_exc is not None:
        details.append(f"generic lerobot_train.py could not be inspected: {train_exc}")

    inference_only = (
        "inference-only" in config.lower()
        or "inference-only" in model.lower()
        or "def forward" in model
        and "NotImplementedError" in model
    )
    if inference_only:
        train_script_note = "generic lerobot_train.py exists" if has_train_script else "generic train script not found"
        details.append(
            "allenai/lerobot molmoact2-hf-inference has a MolmoAct2 policy wrapper, "
            f"but it is inference-only ({train_script_note}; no trainable MolmoAct2 forward/optimizer)"
        )
        return Check("upstream fine-tune code", "BLOCKED", "; ".join(details))

    has_forward = "def forward" in model
    has_optimizer_contract = "get_optim_params" in model or "get_optimizer_preset" in config
    if has_train_script and has_forward and has_optimizer_contract:
        return Check(
            "upstream fine-tune code",
            "OK",
            "raw upstream files expose a train script plus MolmoAct2 forward/optimizer hooks; inspect official recipe before launch",
        )

    missing_trainable = []
    if not has_train_script:
        missing_trainable.append("generic lerobot_train.py")
    if not has_forward:
        missing_trainable.append("MolmoAct2 forward")
    if not has_optimizer_contract:
        missing_trainable.append("MolmoAct2 optimizer hooks")
    details.append("trainability is not positively verified; missing " + ", ".join(missing_trainable))
    return Check("upstream fine-tune code", "BLOCKED", "; ".join(details))


def print_checks(checks: list[Check]) -> int:
    blockers = [check for check in checks if check.status == "BLOCKED"]
    for check in checks:
        print(f"{check.status:7} {check.name}: {check.detail}")
    print()
    if blockers:
        print("Not ready for Brev fine-tuning.")
        print("Blockers:")
        for check in blockers:
            print(f"  - {check.name}: {check.detail}")
        return 1
    print("Ready to attempt Brev fine-tuning.")
    return 0


def main() -> None:
    args = parse_args()
    checks: list[Check] = []
    checks.append(check_model_norm())
    dataset_check, _ = check_dataset_metadata(args.dataset_repo_id, args.dataset_revision, args.dataset_root)
    checks.append(dataset_check)
    if not args.skip_ranges and dataset_check.status != "BLOCKED":
        checks.append(compare_ranges(args.dataset_repo_id, args.dataset_revision, args.dataset_root))
    checks.append(check_brev())
    checks.append(check_upstream_finetune_code())
    raise SystemExit(print_checks(checks))


if __name__ == "__main__":
    main()
