#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NEXT_ACTIONS = {
    "model norm": "Use allenai/MolmoAct2-SO100_101 with norm_tag so100_so101_molmoact2.",
    "dataset metadata": "Fix the LeRobot schema: 30 Hz, observation.images.front, 6D observation.state, and 6D action.",
    "dataset ranges": "Recollect or prove a calibrated offline conversion; do not fine-tune on mismatched joint ranges.",
    "frame samples": "Inspect sampled state/action rows for NaN, wrong shape, or corrupt parquet data.",
    "image frames": "Fix the front RGB video path/codec/camera; the model needs loadable nonblank observation.images.front frames.",
    "brev": "Use the mw-newton-dev SSH alias or repair it with Brev CLI only if SSH is stale.",
    "upstream fine-tune code": "Wait for Ai2 trainable MolmoAct2 code or approve a public local recipe before launching Brev training.",
    "upstream MolmoAct2 LeRobot wrapper is inference-only": "Wait for Ai2 trainable MolmoAct2 code or approve a public local recipe before launching Brev training.",
    "old Carmen diagnostic dataset has joint range/calibration mismatches": "Recollect the Carmen dataset or prove a calibrated offline conversion before fine-tuning.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a MolmoAct2 readiness JSON report.")
    parser.add_argument("report", type=Path, help="JSON from check_finetune_readiness.py or check_collection_dataset.py")
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Exit 1 when the report is blocked.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"could not read readiness report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise SystemExit(f"readiness report must be a JSON object: {path}")
    return report


def normalized_blockers(report: dict[str, Any]) -> list[dict[str, str]]:
    blockers = report.get("blockers", [])
    if not blockers and isinstance(report.get("blocked_reasons"), list):
        blockers = report["blocked_reasons"]
    if not blockers and isinstance(report.get("diagnostic_readiness"), dict):
        diagnostic_blockers = report["diagnostic_readiness"].get("blockers", [])
        if isinstance(diagnostic_blockers, list):
            blockers = diagnostic_blockers
    if not isinstance(blockers, list):
        return [{"name": "report", "detail": "blockers field is not a list"}]

    normalized = []
    for item in blockers:
        if isinstance(item, dict):
            name = str(item.get("name", "unknown"))
            detail = str(item.get("detail", "")).strip()
        else:
            name = str(item)
            detail = ""
        normalized.append({"name": name, "detail": detail})
    return normalized


def normalized_checks(report: dict[str, Any]) -> list[dict[str, str]]:
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return []
    normalized = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": str(item.get("name", "unknown")),
                "status": str(item.get("status", "unknown")),
            }
        )
    return normalized


def next_action(name: str) -> str:
    return NEXT_ACTIONS.get(name, "Inspect the blocker detail and update the MolmoAct2 handoff docs if this is a new failure mode.")


def infer_ready(report: dict[str, Any], blockers: list[dict[str, str]]) -> bool:
    if "ready" in report:
        return bool(report["ready"]) and not blockers
    if "status" in report:
        return str(report["status"]).lower() == "ready" and not blockers
    return not blockers


def main() -> None:
    args = parse_args()
    report = load_report(args.report)
    blockers = normalized_blockers(report)
    ready = infer_ready(report, blockers)
    status = "READY" if ready else "BLOCKED"

    print(f"Report: {args.report}")
    print(f"Status: {status}")
    print(f"Brev launch: {'YES' if ready else 'NO'}")

    if blockers:
        print("\nBlockers:")
        for blocker in blockers:
            if blocker["detail"]:
                print(f"- {blocker['name']}: {blocker['detail']}")
            else:
                print(f"- {blocker['name']}")
            print(f"  next: {next_action(blocker['name'])}")
    else:
        print("\nNo blockers reported. Re-check the exact train command before launching a real Brev job.")

    checks = normalized_checks(report)
    if checks:
        print("\nChecks:")
        for check in checks:
            print(f"- {check['status']} {check['name']}")

    if args.strict_exit_code and not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
