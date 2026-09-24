"""Descriptive-only analyzer for the isolated frozen S13 comparator block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scientific_evidence import validate_scientific_run_structure


SEEDS = (1103, 1129, 1151, 1171, 1181, 1201, 1213, 1231, 1249, 1277)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_closure(root: Path) -> None:
    closure = json.loads((root / "terminal_closure.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal_status.json").read_text(encoding="utf-8"))
    hashes = json.loads((root / "output_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected).upper():
            raise AssertionError(f"S13 output hash mismatch: {relative}")
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
        raise AssertionError("S13 terminal closure mismatch")


def analyze(root: Path, primary_root: Path, output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    validate_scientific_run_structure(
        root,
        expected_run_id=("EXP-SHIL-TRACK-A-FULL-STRUCTURED-S13-full-structured-s13"),
        expected_role="scientific_full_structured_s13",
        expected_mode="full-structured-s13",
        expected_unit_count=10,
        expected_scenarios=("S13",),
        expected_split_seeds=SEEDS,
    )
    validate_scientific_run_structure(
        primary_root,
        expected_run_id=("EXP-SHIL-TRACK-A-FULL-SYNTHETIC-PRIMARY-full-synthetic"),
        expected_role="scientific_full_synthetic_primary",
        expected_mode="full-synthetic",
        expected_unit_count=140,
        expected_scenarios=(f"S{index:02d}" for index in range(14)),
        expected_split_seeds=SEEDS,
    )
    _validate_closure(root)
    _validate_closure(primary_root)
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("mode") != "full-structured-s13"
        or summary.get("block_role") != "scientific_full_structured_s13"
        or summary.get("dataset_runs") != 10
        or summary.get("primary_method_execution_performed") is not False
        or summary.get("method_rows") != 0
        or summary.get("ranking_metric_rows") != 0
        or summary.get("reporting_label") != "DESCRIPTIVE_NO_INFERENCE"
    ):
        raise AssertionError("S13 comparator-only block identity mismatch")
    status = pd.read_csv(root / "structured_sensitivity_status.csv")
    metrics = pd.read_csv(root / "structured_sensitivity_metrics.csv")
    raw = pd.read_csv(root / "structured_sensitivity_raw.csv")
    partitions = pd.read_csv(root / "partition_manifest.csv")
    if not (
        len(status) == len(metrics) == len(raw) == len(partitions) == 10
        and set(status["scenario"]) == {"S13"}
        and set(status["split_seed"].astype(int)) == set(SEEDS)
        and set(status["status"]) == {"COMPLETED"}
    ):
        raise AssertionError("S13 comparator completeness mismatch")
    for frame in (status, metrics, raw):
        if (
            set(frame["analysis_role"]) != {"sensitivity_only"}
            or set(frame["primary_estimand"].astype(str).str.lower()) != {"false"}
            or set(frame["reporting_label"]) != {"DESCRIPTIVE_NO_INFERENCE"}
            or {"method_id", "score_family_id"} & set(frame.columns)
        ):
            raise AssertionError("S13 comparator role contamination")
    for name in (
        "metrics.csv",
        "selected_edges.csv",
        "ranking_metrics.csv",
        "ranking_scores.csv",
    ):
        path = root / name
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if not frame.empty:
            raise AssertionError(f"Comparator-only block produced primary rows: {name}")
    primary = pd.read_csv(primary_root / "partition_manifest.csv")
    primary = primary.loc[primary["scenario"] == "S13"].copy()
    join_columns = [
        "scenario",
        "data_seed",
        "split_seed",
        "dataset_fingerprint_sha256",
        "outer_split_fingerprint_sha256",
    ]
    if len(primary) != 10 or set(primary["split_seed"].astype(int)) != set(SEEDS):
        raise AssertionError("Primary S13 identity coverage mismatch")
    left = (
        partitions[join_columns]
        .sort_values(["scenario", "split_seed"])
        .reset_index(drop=True)
    )
    right = (
        primary[join_columns]
        .sort_values(["scenario", "split_seed"])
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise AssertionError("S13 comparator dataset/split fingerprint mismatch")
    output.mkdir(parents=True, exist_ok=True)
    descriptive = metrics.copy()
    descriptive["analysis_role"] = "DESCRIPTIVE_NO_INFERENCE"
    descriptive.to_csv(output / "s13_structured_metrics_descriptive.csv", index=False)
    payload = {
        "schema_version": 1,
        "status": "ANALYZED_DESCRIPTIVE_ONLY",
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
        "inferential_statistics_computed": False,
        "primary_estimand_contaminated": False,
        "validated_unit_count": 10,
        "primary_identity_match": True,
        "input_terminal_closure_sha256": sha256_file(root / "terminal_closure.json"),
        "primary_terminal_closure_sha256": sha256_file(
            primary_root / "terminal_closure.json"
        ),
    }
    (output / "s13_structured_analysis_summary.json").write_text(
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
    parser.add_argument("--primary-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return analyze(args.input, args.primary_input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
