"""Fail-closed descriptive-only analysis for the frozen Dry Bean block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from method_registry import load_method_registry
from scientific_evidence import validate_scientific_run_structure


REAL_SEEDS = (11, 23, 37, 53, 71, 89, 101, 131, 157, 181)
SUPPORT_COLUMNS = (
    "candidate_average_precision",
    "support_precision",
    "support_recall",
    "support_f1",
    "false_inclusions",
    "false_discovery_proportion",
    "group_tp",
    "group_precision",
    "group_recall",
    "group_f1",
    "group_false_inclusions",
    "group_false_discovery_proportion",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_closure(root: Path) -> None:
    terminal = json.loads((root / "terminal_status.json").read_text(encoding="utf-8"))
    closure = json.loads((root / "terminal_closure.json").read_text(encoding="utf-8"))
    output_hashes = json.loads(
        (root / "output_hashes.json").read_text(encoding="utf-8")
    )
    for relative, expected in output_hashes.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected).upper():
            raise AssertionError(f"Dry Bean output hash mismatch: {relative}")
    notifications = sorted((root / "notifications").glob("invocation-*.jsonl"))
    notification_bindings = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in notifications
    ]
    if (
        terminal.get("status") != "completed"
        or terminal.get("exit_code") != 0
        or closure.get("status") != "closed_completed"
        or closure.get("terminal_status_sha256")
        != sha256_file(root / "terminal_status.json")
        or closure.get("terminal_status_history_sha256")
        != sha256_file(root / "terminal_status_history.jsonl")
        or closure.get("output_hashes_sha256")
        != sha256_file(root / "output_hashes.json")
        or closure.get("run_manifest_sha256") != sha256_file(root / "RUN_MANIFEST.json")
        or not notifications
        or closure.get("notification_outboxes") != notification_bindings
    ):
        raise AssertionError("Dry Bean terminal closure mismatch")


def validate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_scientific_run_structure(
        root,
        expected_run_id="EXP-SHIL-TRACK-A-FULL-REAL-DESCRIPTIVE-full-real",
        expected_role="scientific_full_real_descriptive",
        expected_mode="full-real",
        expected_unit_count=10,
        expected_scenarios=("dry_bean_uci",),
        expected_split_seeds=REAL_SEEDS,
    )
    _validate_closure(root)
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config_resolved.json").read_text(encoding="utf-8"))
    if (
        summary.get("mode") != "full-real"
        or summary.get("block_role") != "scientific_full_real_descriptive"
        or summary.get("dataset_runs") != 10
        or summary.get("reporting_label") != "DESCRIPTIVE_NO_INFERENCE"
        or config.get("real_split_seeds") != list(REAL_SEEDS)
        or config.get("structured_comparator_candidate", {}).get("execution_enabled")
        is not False
    ):
        raise AssertionError("Dry Bean frozen block identity mismatch")
    registry = load_method_registry(root / "method_registry_resolved.json")
    expected_methods = set(
        registry.expected_methods_for(
            {"cell_type": "real", "candidate_orders": config["candidate_orders"]}
        )
    )
    metrics = pd.read_csv(root / "metrics.csv")
    partitions = pd.read_csv(root / "partition_manifest.csv")
    if len(metrics) != 10 * len(expected_methods) or len(partitions) != 10:
        raise AssertionError("Dry Bean row count mismatch")
    for seed, frame in metrics.groupby("split_seed"):
        if int(seed) not in REAL_SEEDS or set(frame["method_id"]) != expected_methods:
            raise AssertionError("Dry Bean method/seed coverage mismatch")
    if set(metrics["status"]) != {"OK"}:
        raise AssertionError("Dry Bean contains non-OK primary method rows")
    if set(metrics["reporting_label"]) != {"DESCRIPTIVE_NO_INFERENCE"}:
        raise AssertionError("Dry Bean metrics lack descriptive-only label")
    if any(not metrics[column].isna().all() for column in SUPPORT_COLUMNS):
        raise AssertionError("Dry Bean unknown-truth support fields are not all NaN")
    sensitivity = pd.read_csv(root / "structured_sensitivity_status.csv")
    if len(sensitivity) != 10 or set(sensitivity["status"]) != {"N/A"}:
        raise AssertionError("Dry Bean structured comparator must be exact N/A")
    if set(sensitivity["reporting_label"]) != {"DESCRIPTIVE_NO_INFERENCE"}:
        raise AssertionError("Dry Bean sensitivity status lacks descriptive label")
    return metrics, partitions, summary


def analyze(root: Path, output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics, partitions, summary = validate(root)
    predictive = (
        metrics.groupby(["method_id", "ranker_id", "policy_id"], dropna=False)
        .agg(
            runs=("method_id", "size"),
            log_loss_mean=("log_loss", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            roc_auc_ovr_mean=("roc_auc_ovr", "mean"),
        )
        .reset_index()
    )
    predictive["analysis_role"] = "DESCRIPTIVE_NO_INFERENCE"
    predictive.to_csv(output / "dry_bean_predictive_summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "status": "ANALYZED_DESCRIPTIVE_ONLY",
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
        "inferential_statistics_computed": False,
        "support_superiority_claim_allowed": False,
        "run_id": json.loads(
            (root / "terminal_status.json").read_text(encoding="utf-8")
        )["run_id"],
        "validated_split_count": len(partitions),
        "validated_method_rows": len(metrics),
        "input_terminal_closure_sha256": sha256_file(root / "terminal_closure.json"),
    }
    (output / "dry_bean_analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "analysis_hashes.json").write_text(
        json.dumps(
            {path.name: sha256_file(path) for path in files}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return analyze(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
