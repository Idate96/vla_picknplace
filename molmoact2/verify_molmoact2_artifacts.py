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
    ok = (
        data.get("status") == "blocked"
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
        and "cluster/brev/setup_brev_env.sh" in commands.get("setup_brev", "")
        and "cluster/brev/submit_finetune_brev.sh" in commands.get("launch_when_unblocked", "")
        and "--train-command" in commands.get("launch_when_unblocked", "")
    )
    return Check(
        str(path),
        ok,
        "machine-readable blocked Brev fine-tune handoff"
        if ok
        else "missing required model/dataset/Brev command fields",
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


def check_no_external_course_paths() -> Check:
    scan_roots = [ROOT / "README.md", ROOT / "docs", ROOT / "molmoact2", ROOT / "cluster/brev"]
    forbidden_fragments = ("ethz" + "-course-2026", "hw3" + "_imitation_learning/.venv")
    matches: list[str] = []
    for scan_root in scan_roots:
        paths = [scan_root] if scan_root.is_file() else sorted(scan_root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in {".md", ".py", ".json", ".sh", ".template"}:
                continue
            text = path.read_text(errors="ignore")
            if any(fragment in text for fragment in forbidden_fragments):
                matches.append(str(path.relative_to(ROOT)))
    return Check(
        "repo-local command paths",
        not matches,
        "no operational commands point at the old course repo" if not matches else f"old paths in {matches}",
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
    env.setdefault("BREV_INSTANCE_NAME", template_env("BREV_INSTANCE_NAME") or "")
    proc = subprocess.run(
        [
            sys.executable,
            "molmoact2/check_finetune_readiness.py",
            "--dataset-repo-id",
            "carmensc/record-test-screwdriver",
            "--skip-ranges",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + "\n" + proc.stderr
    upstream_accounted_for = (
        "BLOCKED upstream fine-tune code" in output
        and ("inference-only" in output or "trainability is unverified" in output)
    )
    brev_accounted_for = "OK      brev:" in output or "BLOCKED brev:" in output
    expected = (
        proc.returncode != 0
        and "Not ready for Brev fine-tuning." in output
        and upstream_accounted_for
        and brev_accounted_for
    )
    if expected and "OK      brev:" in output:
        detail = "uses configured Newton Brev instance; blocks on upstream fine-tune support"
    elif expected:
        detail = "Brev SSH is not reachable from this shell; upstream fine-tune support is still blocked"
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


def main() -> None:
    checks: list[Check] = []
    script_paths = [
        ROOT / "molmoact2/inspect_molmoact2.py",
        ROOT / "molmoact2/test_on_lerobot_frame.py",
        ROOT / "molmoact2/simulate_joint_control.py",
        ROOT / "molmoact2/simulate_mujoco_so101.py",
        ROOT / "molmoact2/verify_molmoact2_artifacts.py",
        ROOT / "molmoact2/check_finetune_readiness.py",
    ]
    bash_paths = [
        ROOT / "cluster/brev/sync_code_brev.sh",
        ROOT / "cluster/brev/setup_brev_env.sh",
        ROOT / "cluster/brev/submit_finetune_brev.sh",
    ]
    checks.append(check_brev_manifest())
    checks.append(check_brev_env_template())
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
            check_joint_control_smoke(),
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
