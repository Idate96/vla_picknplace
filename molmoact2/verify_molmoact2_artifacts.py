#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    ok: bool
    detail: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def exists(path: Path) -> Check:
    return Check(str(path), path.exists(), "exists" if path.exists() else "missing")


def check_requirements() -> Check:
    path = ROOT / "requirements.txt"
    if not path.exists():
        return Check(str(path), False, "missing")
    deps = set()
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        name = raw.split("[", 1)[0]
        for marker in ("==", ">=", "<=", "~=", " @ "):
            name = name.split(marker, 1)[0]
        deps.add(name.strip())
    required = {
        "datasets",
        "huggingface_hub",
        "lerobot",
        "mujoco",
        "numpy",
        "pillow",
        "torch",
        "transformers",
    }
    missing = sorted(required - deps)
    return Check(
        str(path),
        not missing,
        "contains MolmoAct2 script deps" if not missing else f"missing {missing}",
    )


def check_brev_manifest() -> Check:
    path = ROOT / "molmoact2/brev_finetune_manifest.json"
    if not path.exists():
        return Check(str(path), False, "missing")
    try:
        data = load_json(path)
    except Exception as exc:
        return Check(str(path), False, f"invalid json: {exc}")

    model = data.get("model", {})
    contract = data.get("target_dataset_contract", {})
    brev = data.get("brev", {})
    commands = data.get("commands", {})
    upstream = data.get("upstream", {})
    diagnostic = data.get("diagnostic_readiness", {})
    artificial = data.get("artificial_act_dataset", {})
    old_artificial_schema = artificial.get("old_schema", {})
    required_artificial_schema = artificial.get("required_molmoact2_schema", {})
    ok = (
        data.get("status") == "blocked"
        and data.get("last_checked") == "2026-05-08"
        and model.get("repo_id") == "allenai/MolmoAct2-SO100_101"
        and model.get("norm_tag") == "so100_so101_molmoact2"
        and model.get("joint_names") == EXPECTED_JOINTS
        and contract.get("fps") == 30
        and contract.get("image_key") == "observation.images.front"
        and contract.get("state_shape") == [6]
        and contract.get("action_shape") == [6]
        and brev.get("instance_name") == "mw-newton-dev"
        and brev.get("token_env") == "BREV_TOKEN"
        and brev.get("workflow") == "ssh_rsync_uv"
        and brev.get("setup_script") == "cluster/brev/setup_brev_env.sh"
        and brev.get("submit_script") == "cluster/brev/submit_finetune_brev.sh"
        and upstream.get("molmoact2_head") == "c45fcbca4501339bc0b12e30a273c15bf4d56cf0"
        and upstream.get("lerobot_ref") == "c123084cf840c00af5c0833832fc58e590412851"
        and "inference-only" in upstream.get("trainability", "")
        and diagnostic.get("dataset_repo_id") == "carmensc/record-test-screwdriver"
        and diagnostic.get("ready") is False
        and diagnostic.get("blockers") == ["dataset ranges", "upstream fine-tune code"]
        and artificial.get("status") == "not_usable_for_molmoact2_finetuning_or_rollout"
        and artificial.get("decision_doc") == "docs/molmoact2_artificial_dataset_compatibility.md"
        and "loader/model diagnostics" in artificial.get("safe_use", "")
        and old_artificial_schema.get("camera_views") == "two camera positions"
        and old_artificial_schema.get("fps") == 10
        and old_artificial_schema.get("state") == "4D end-effector state"
        and old_artificial_schema.get("action") == "4D end-effector delta action"
        and required_artificial_schema.get("image_key") == "observation.images.front"
        and required_artificial_schema.get("camera") == "one fixed front RGB camera"
        and required_artificial_schema.get("fps") == 30
        and required_artificial_schema.get("state") == "6D calibrated current joint state"
        and required_artificial_schema.get("action") == "6D absolute calibrated joint target"
        and "recomputed statistics" in artificial.get("conversion_rule", "")
        and "cluster/brev/setup_brev_env.sh" in commands.get("setup_brev", "")
        and "cluster/brev/submit_finetune_brev.sh" in commands.get("launch_when_unblocked", "")
        and "--readiness-report" in commands.get("launch_when_unblocked", "")
        and "check_finetune_readiness.py" in commands.get("readiness_json", "")
        and "summarize_readiness.py" in commands.get("readiness_summary", "")
        and "--strict-exit-code" in commands.get("readiness_summary", "")
        and "check_collection_dataset.py" in commands.get("collection_preflight", "")
        and "--allow-blocked-dry-run" in commands.get("diagnostic_blocked_dry_run", "")
        and "--train-command" in commands.get("launch_when_unblocked", "")
    )
    return Check(
        str(path),
        ok,
        "machine-readable blocked Brev fine-tune handoff and artificial ACT dataset decision"
        if ok
        else "missing required model/dataset/Brev command fields",
    )


def git_ls_remote(repo: str, ref: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "ls-remote", repo, ref],
            cwd=ROOT,
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


def check_manifest_upstream_refs_current() -> Check:
    path = ROOT / "molmoact2/brev_finetune_manifest.json"
    if not path.exists():
        return Check("manifest upstream refs", False, "missing manifest")
    try:
        upstream = load_json(path).get("upstream", {})
    except Exception as exc:
        return Check("manifest upstream refs", False, f"invalid manifest json: {exc}")

    expected_molmo = upstream.get("molmoact2_head")
    expected_lerobot = upstream.get("lerobot_ref")
    actual_molmo = git_ls_remote("https://github.com/allenai/molmoact2.git", "HEAD")
    actual_lerobot = git_ls_remote(
        "https://github.com/allenai/lerobot.git",
        "refs/heads/molmoact2-hf-inference",
    )
    ok = (
        actual_molmo is not None
        and actual_lerobot is not None
        and actual_molmo == expected_molmo
        and actual_lerobot == expected_lerobot
    )
    return Check(
        "manifest upstream refs",
        ok,
        f"current refs match manifest ({actual_molmo}, {actual_lerobot})"
        if ok
        else (
            "manifest refs are stale or upstream could not be inspected: "
            f"manifest=({expected_molmo}, {expected_lerobot}) "
            f"live=({actual_molmo}, {actual_lerobot})"
        ),
    )


def check_py_compile(paths: list[Path]) -> Check:
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", *[str(path) for path in paths]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return Check(
        "py_compile MolmoAct2 scripts",
        proc.returncode == 0,
        "passed" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-500:],
    )


def check_bash_syntax(paths: list[Path]) -> Check:
    proc = subprocess.run(
        ["bash", "-n", *[str(path) for path in paths]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return Check(
        "bash -n Brev scripts",
        proc.returncode == 0,
        "passed" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-500:],
    )


def check_python_imports() -> Check:
    modules = ["datasets", "huggingface_hub", "lerobot.datasets", "mujoco", "torch", "transformers"]
    script = "import importlib\n" + "\n".join(f"importlib.import_module({module!r})" for module in modules)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return Check(
        "Python dependency imports",
        proc.returncode == 0,
        "passed" if proc.returncode == 0 else (proc.stderr or proc.stdout)[-500:],
    )


def check_brev_env_template() -> Check:
    path = ROOT / "cluster/brev/.env.brev.template"
    if not path.exists():
        return Check(str(path), False, "missing")
    text = path.read_text()
    required = [
        "CLUSTER_TYPE=brev",
        "BREV_INSTANCE_NAME",
        "BREV_CODE_DIR",
        "BREV_LOGS_DIR",
        "BREV_DATA_DIR",
    ]
    missing = [item for item in required if item not in text]
    return Check(
        str(path),
        not missing,
        "contains Brev SSH/rsync env defaults" if not missing else f"missing {missing}",
    )


def check_brev_video_decode_setup() -> Check:
    setup_path = ROOT / "cluster/brev/setup_brev_env.sh"
    readme_path = ROOT / "cluster/brev/README.md"
    runbook_path = ROOT / "docs/molmoact2_brev_finetuning.md"
    if not setup_path.exists():
        return Check("Brev video decode setup", False, "missing setup_brev_env.sh")
    setup_text = setup_path.read_text()
    required_setup = [
        "libavutil\\.so",
        "sudo -n true",
        "apt-get install -y ffmpeg",
    ]
    missing = [item for item in required_setup if item not in setup_text]

    for path in (readme_path, runbook_path):
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        text = path.read_text()
        if "FFmpeg shared libraries" not in text or "LeRobot" not in text or "video decoding" not in text:
            missing.append(str(path.relative_to(ROOT)))

    return Check(
        "Brev video decode setup",
        not missing,
        "setup installs FFmpeg shared libraries needed for LeRobot video decoding"
        if not missing
        else f"missing {missing}",
    )


def check_artificial_act_decision() -> Check:
    path = ROOT / "docs/molmoact2_artificial_dataset_compatibility.md"
    if not path.exists():
        return Check("artificial ACT compatibility decision", False, "missing compatibility doc")
    text = path.read_text()
    required = [
        "Do not use it as the MolmoAct2 fine-tuning or rollout dataset.",
        "two camera positions",
        "10 Hz",
        "4D end-effector state",
        "4D end-effector delta action",
        "one fixed front RGB camera",
        "30 Hz",
        "6D joint state",
        "6D absolute joint target action",
        "Safe uses:",
        "Unsafe uses:",
        "not a simple camera remap",
        "No HW3 simulator scripts are required or expected.",
    ]
    missing = [item for item in required if item not in text]
    return Check(
        "artificial ACT compatibility decision",
        not missing,
        "old two-camera ACT sim data is documented as historical only, not a MolmoAct2 fine-tune/control dataset"
        if not missing
        else f"missing {missing}",
    )


def check_no_external_course_paths() -> Check:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        paths = [ROOT / line for line in proc.stdout.splitlines()]
    else:
        skip_dirs = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "outputs"}
        paths = [
            path
            for path in ROOT.rglob("*")
            if not any(part in skip_dirs for part in path.relative_to(ROOT).parts)
        ]

    forbidden_fragments = (
        "ethz" + "-course-2026",
        "hw3" + "_imitation_learning",
        "hw3" + "_camera_ablation",
    )
    suffixes = {".md", ".py", ".json", ".sh", ".template", ".sbatch", ".args"}
    matches: list[str] = []
    for path in paths:
        if not path.is_file() or path.suffix not in suffixes:
            continue
        text = path.read_text(errors="ignore")
        if any(fragment in text for fragment in forbidden_fragments):
            matches.append(str(path.relative_to(ROOT)))
    return Check(
        "repo-local command paths",
        not matches,
        "no tracked/fallback-scanned commands point at the old course repo"
        if not matches
        else f"old paths in {matches}",
    )


def template_env(name: str) -> str | None:
    path = ROOT / "cluster/brev/.env.brev.template"
    if not path.exists():
        return None
    prefix = f"export {name}="
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        return value.strip("\"'")
    return None


def check_readiness_blocked() -> Check:
    env = os.environ.copy()
    env.pop("BREV_INSTANCE_NAME", None)
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "readiness.json"
        proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/check_finetune_readiness.py",
                "--dataset-repo-id",
                "carmensc/record-test-screwdriver",
                "--skip-ranges",
                "--output-json",
                str(report_path),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = {}
    output = proc.stdout + "\n" + proc.stderr
    upstream_accounted_for = (
        "BLOCKED upstream fine-tune code" in output
        and ("inference-only" in output or "trainability is unverified" in output)
    )
    brev_accounted_for = "OK      brev:" in output or "BLOCKED brev:" in output
    brev_uses_ssh_default = "brev CLI found" not in output and "logged out" not in output
    expected = (
        proc.returncode != 0
        and "Not ready for Brev fine-tuning." in output
        and upstream_accounted_for
        and brev_accounted_for
        and brev_uses_ssh_default
        and report.get("ready") is False
        and report.get("status") == "blocked"
        and any(item.get("name") == "upstream fine-tune code" for item in report.get("blockers", []))
    )
    if expected and "OK      brev:" in output:
        detail = "uses repo Brev SSH default without Brev CLI auth; blocks on upstream fine-tune support"
    elif expected:
        detail = "repo Brev SSH default is not reachable from this shell; upstream fine-tune support is still blocked"
    else:
        detail = f"unexpected readiness output: {output[-500:]}"
    return Check(
        "check_finetune_readiness.py --skip-ranges",
        expected,
        detail,
    )


def check_joint_control_smoke() -> Check:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        model_output = tmpdir / "molmo_one_frame.json"
        sim_output = tmpdir / "joint_sim.json"
        model_output.write_text(
            json.dumps(
                {
                    "model": "allenai/MolmoAct2-SO100_101",
                    "norm_tag": "so100_so101_molmoact2",
                    "dataset_repo_id": "synthetic/smoke",
                    "episode": 0,
                    "frame": 0,
                    "task": "pickup screwdriver",
                    "state": [0, 50, 50, 20, 0, 10],
                    "action_horizon": [
                        [1, 55, 52, 21, -1, 20],
                        [2, 60, 54, 22, -2, 30],
                    ],
                }
            )
        )
        proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/simulate_joint_control.py",
                "--model-output",
                str(model_output),
                "--skip-model-bounds",
                "--output",
                str(sim_output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return Check("joint-space sim control smoke", False, (proc.stderr or proc.stdout)[-500:])
        try:
            result = json.loads(sim_output.read_text())
        except Exception as exc:
            return Check("joint-space sim control smoke", False, f"invalid output json: {exc}")
        ok = (
            result.get("simulator", {}).get("type") == "joint_space_absolute_target"
            and result.get("horizon_steps") == 2
            and len(result.get("sent_targets", [])) == 2
            and len(result.get("simulated_states", [])) == 3
        )
        return Check(
            "joint-space sim control smoke",
            ok,
            "simulates absolute 6D joint-target commands"
            if ok
            else f"unexpected output: {result}",
        )


def check_readiness_summary_smoke() -> Check:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "readiness.json"
        manifest_path = Path(tmp) / "manifest.json"
        report_path.write_text(
            json.dumps(
                {
                    "ready": False,
                    "status": "blocked",
                    "blockers": [
                        {
                            "name": "dataset ranges",
                            "detail": "shoulder_lift dataset range is outside MolmoAct2 q01/q99",
                        },
                        {
                            "name": "upstream fine-tune code",
                            "detail": "MolmoAct2 wrapper is inference-only",
                        },
                    ],
                    "checks": [
                        {"name": "model norm", "status": "OK"},
                        {"name": "dataset ranges", "status": "BLOCKED"},
                        {"name": "upstream fine-tune code", "status": "BLOCKED"},
                    ],
                }
            )
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "blocked_reasons": [
                        "old Carmen diagnostic dataset has joint range/calibration mismatches",
                        "upstream MolmoAct2 LeRobot wrapper is inference-only",
                    ],
                    "diagnostic_readiness": {
                        "blockers": [
                            "dataset ranges",
                            "upstream fine-tune code",
                        ],
                    },
                }
            )
        )
        readiness_proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/summarize_readiness.py",
                str(report_path),
                "--strict-exit-code",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        manifest_proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/summarize_readiness.py",
                str(manifest_path),
                "--strict-exit-code",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    output = readiness_proc.stdout + "\n" + readiness_proc.stderr
    manifest_output = manifest_proc.stdout + "\n" + manifest_proc.stderr
    ok = (
        readiness_proc.returncode == 1
        and "Status: BLOCKED" in output
        and "Brev launch: NO" in output
        and "Recollect or prove a calibrated offline conversion" in output
        and "Wait for Ai2 trainable MolmoAct2 code" in output
        and manifest_proc.returncode == 1
        and "old Carmen diagnostic dataset has joint range/calibration mismatches" in manifest_output
        and "upstream MolmoAct2 LeRobot wrapper is inference-only" in manifest_output
        and "Brev launch: NO" in manifest_output
    )
    return Check(
        "readiness summary smoke",
        ok,
        "summarizes readiness and manifest blockers into a no-launch decision with next actions"
        if ok
        else f"unexpected output: {(output + manifest_output)[-500:]}",
    )


def check_mujoco_rollout_dry_run() -> Check:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        output = tmpdir / "rollout.json"
        proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/rollout_mujoco_so101.py",
                "--dry-run",
                "--rollout-steps",
                "1",
                "--actions-per-inference",
                "1",
                "--width",
                "160",
                "--height",
                "120",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if proc.returncode != 0:
            return Check("MuJoCo closed-loop dry-run smoke", False, (proc.stderr or proc.stdout)[-500:])
        try:
            result = json.loads(output.read_text())
        except Exception as exc:
            return Check("MuJoCo closed-loop dry-run smoke", False, f"invalid output json: {exc}")
        ok = (
            result.get("simulator", {}).get("type") == "molmoact2_closed_loop_robotstudio_so101_mujoco"
            and result.get("dry_run") is True
            and result.get("model_loaded") is False
            and result.get("rollout_steps") == 1
            and result.get("records", [{}])[0].get("horizon_shape") == [1, 6]
            and result.get("records", [{}])[0].get("horizon_source") == "dry_run_current_state"
            and result.get("records", [{}])[0].get("image_stats", {}).get("std", 0) > 0
            and result.get("final_image_stats", {}).get("std", 0) > 0
            and len(result.get("final_state_lerobot", [])) == 6
        )
        return Check(
            "MuJoCo closed-loop dry-run smoke",
            ok,
            "renders nonblank camera frames and steps the public SO101 MuJoCo scene without loading MolmoAct2"
            if ok
            else f"unexpected output: {result}",
        )


def check_blocked_brev_dry_run_guard() -> Check:
    submit = ROOT / "cluster/brev/submit_finetune_brev.sh"
    brev_doc = ROOT / "docs/molmoact2_brev_finetuning.md"
    if not submit.exists() or not brev_doc.exists():
        return Check("blocked Brev dry-run guard", False, "missing submit script or runbook")
    submit_text = submit.read_text()
    doc_text = brev_doc.read_text()
    required = [
        "--allow-blocked-dry-run",
        "--readiness-report",
        "ALLOW_BLOCKED_DRY_RUN",
        "READINESS_REPORT",
        "--output-json",
        "summarize_readiness.py",
        "DRY_RUN",
        "Readiness gate blocked; continuing only because --allow-blocked-dry-run was set.",
        "Dry run only; readiness blocked; not syncing or launching.",
    ]
    missing = [item for item in required if item not in submit_text]
    if (
        "--allow-blocked-dry-run" not in doc_text
        or "--readiness-report" not in doc_text
        or "does not sync, SSH launch, or start" not in doc_text
    ):
        missing.append("runbook diagnostic dry-run note")
    if missing:
        return Check("blocked Brev dry-run guard", False, f"missing {missing}")

    env_file = ROOT / "cluster/brev/.env.brev"
    if not env_file.exists():
        return Check(
            "blocked Brev dry-run guard",
            True,
            "dry-run-only bypass is documented, writes readiness JSON, and cannot launch remotely",
        )

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "blocked_brev_readiness.json"
        proc = subprocess.run(
            [
                str(submit),
                "--dataset-repo-id",
                "carmensc/record-test-screwdriver",
                "--dry-run",
                "--allow-blocked-dry-run",
                "--readiness-report",
                str(report_path),
                "--train-command",
                "echo would train",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = {}
    output = proc.stdout + "\n" + proc.stderr
    ok = (
        proc.returncode == 0
        and report.get("ready") is False
        and "Brev launch: NO" in output
        and "Dry run only; readiness blocked; not syncing or launching." in output
        and "Readiness gate blocked; continuing only because --allow-blocked-dry-run was set." in output
        and "Synced to Brev." not in output
        and "Training started" not in output
    )
    return Check(
        "blocked Brev dry-run guard",
        ok,
        "blocked dry-run prints readiness summary and exits before sync or launch"
        if ok
        else f"unexpected dry-run output: {output[-800:]}",
    )


def check_collection_preflight_metadata_only() -> Check:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "collection_preflight.json"
        proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/check_collection_dataset.py",
                "--dataset-repo-id",
                "carmensc/record-test-screwdriver",
                "--skip-ranges",
                "--skip-frame-check",
                "--skip-image-check",
                "--output-json",
                str(report_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = {}
    output = proc.stdout + "\n" + proc.stderr
    ok = (
        proc.returncode == 0
        and report.get("ready") is True
        and report.get("image_key") == "observation.images.front"
        and report.get("joint_order") == EXPECTED_JOINTS
        and "Dataset passes the MolmoAct2 collection preflight." in output
    )
    return Check(
        "collection dataset preflight",
        ok,
        "metadata-only preflight accepts a LeRobot-shaped screwdriver dataset"
        if ok
        else f"unexpected output: {output[-500:]}",
    )


def check_old_carmen_dataset_range_blocked() -> Check:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "old_carmen_full_preflight.json"
        proc = subprocess.run(
            [
                sys.executable,
                "molmoact2/check_collection_dataset.py",
                "--dataset-repo-id",
                "carmensc/record-test-screwdriver",
                "--output-json",
                str(report_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = {}
    output = proc.stdout + "\n" + proc.stderr
    checks_by_name = {item.get("name"): item for item in report.get("checks", [])}
    blockers = report.get("blockers", [])
    range_blocker = next((item for item in blockers if item.get("name") == "dataset ranges"), {})
    detail = range_blocker.get("detail", "")
    image_check = checks_by_name.get("image frames", {})
    ok = (
        proc.returncode != 0
        and report.get("ready") is False
        and report.get("status") == "blocked"
        and image_check.get("status") == "OK"
        and range_blocker
        and "shoulder_lift" in detail
        and "MolmoAct2 collection handoff" in output
    )
    return Check(
        "old Carmen full preflight blocks on ranges",
        ok,
        "full preflight loads front RGB frames, then rejects the old screwdriver dataset on joint range/calibration mismatch"
        if ok
        else f"unexpected output: {output[-500:]}",
    )


def main() -> None:
    checks: list[Check] = []
    script_paths = [
        ROOT / "molmoact2/inspect_molmoact2.py",
        ROOT / "molmoact2/test_on_lerobot_frame.py",
        ROOT / "molmoact2/simulate_joint_control.py",
        ROOT / "molmoact2/simulate_mujoco_so101.py",
        ROOT / "molmoact2/rollout_mujoco_so101.py",
        ROOT / "molmoact2/summarize_readiness.py",
        ROOT / "molmoact2/verify_molmoact2_artifacts.py",
        ROOT / "molmoact2/check_finetune_readiness.py",
        ROOT / "molmoact2/check_collection_dataset.py",
    ]
    bash_paths = [
        ROOT / "cluster/brev/sync_code_brev.sh",
        ROOT / "cluster/brev/setup_brev_env.sh",
        ROOT / "cluster/brev/submit_finetune_brev.sh",
    ]
    checks.append(check_brev_manifest())
    checks.append(check_manifest_upstream_refs_current())
    checks.append(check_brev_env_template())
    checks.append(check_brev_video_decode_setup())
    checks.append(check_artificial_act_decision())
    checks.append(check_no_external_course_paths())
    for rel in [
        "cluster/brev/README.md",
        "cluster/brev/sync_code_brev.sh",
        "cluster/brev/setup_brev_env.sh",
        "cluster/brev/submit_finetune_brev.sh",
        "docs/molmoact2_investigation_plan.md",
        "docs/molmoact2_command_sheet.md",
        "docs/molmoact2_artificial_dataset_compatibility.md",
        "docs/molmoact2_data_collection.md",
        "docs/molmoact2_brev_finetuning.md",
        "docs/molmoact2_completion_audit.md",
        "docs/so100_vla_data_collection_guide.md",
    ]:
        checks.append(exists(ROOT / rel))
    checks.append(check_requirements())
    checks.extend(exists(path) for path in script_paths)
    checks.extend(exists(path) for path in bash_paths)
    checks.append(check_py_compile(script_paths))
    checks.append(check_bash_syntax(bash_paths))
    checks.append(check_python_imports())

    checks.extend(
        [
            check_readiness_blocked(),
            check_readiness_summary_smoke(),
            check_joint_control_smoke(),
            check_mujoco_rollout_dry_run(),
            check_blocked_brev_dry_run_guard(),
            check_collection_preflight_metadata_only(),
            check_old_carmen_dataset_range_blocked(),
        ]
    )

    failed = [check for check in checks if not check.ok]
    for check in checks:
        print(f"{'OK' if check.ok else 'FAIL':5} {check.name}: {check.detail}")
    if failed:
        raise SystemExit(1)
    print("\nAll MolmoAct2 local artifacts verified.")


if __name__ == "__main__":
    main()
