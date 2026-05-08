#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MolmoAct2 collection and fine-tune readiness gate.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/molmoact2/dataset_gate"))
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> int:
    print(f"\n== {name} ==")
    print(shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, check=False)
    print(f"== {name} exit: {proc.returncode} ==")
    return proc.returncode


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_status(path: Path) -> str:
    try:
        report = json.loads(path.read_text())
    except Exception:
        return "missing"
    return str(report.get("status", "unknown"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_json = output_dir / "collection_preflight.json"
    readiness_json = output_dir / "readiness.json"

    base_dataset_args = [
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--dataset-revision",
        args.dataset_revision,
    ]
    if args.dataset_root is not None:
        base_dataset_args.extend(["--dataset-root", str(args.dataset_root)])

    collection_cmd = [
        sys.executable,
        "molmoact2/check_collection_dataset.py",
        *base_dataset_args,
        "--output-json",
        str(collection_json),
    ]
    readiness_cmd = [
        sys.executable,
        "molmoact2/check_finetune_readiness.py",
        *base_dataset_args,
        "--output-json",
        str(readiness_json),
    ]

    collection_status = run_step("collection preflight", collection_cmd)
    run_step(
        "collection summary",
        [
            sys.executable,
            "molmoact2/summarize_readiness.py",
            str(collection_json),
        ],
    )

    readiness_status = run_step("fine-tune readiness", readiness_cmd)
    run_step(
        "fine-tune summary",
        [
            sys.executable,
            "molmoact2/summarize_readiness.py",
            str(readiness_json),
        ],
    )

    summary = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_revision": args.dataset_revision,
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
        "collection_preflight": {
            "path": display_path(collection_json),
            "status": load_status(collection_json),
            "exit_code": collection_status,
        },
        "fine_tune_readiness": {
            "path": display_path(readiness_json),
            "status": load_status(readiness_json),
            "exit_code": readiness_status,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n== dataset gate summary ==")
    print(display_path(summary_path))
    print(json.dumps(summary, indent=2))

    if collection_status != 0 or readiness_status != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
