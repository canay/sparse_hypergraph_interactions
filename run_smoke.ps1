$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$requiredFiles = @(
  "requirements.txt",
  "code/shil_run_experiments.py",
  "code/shil_q1_extension.py",
  "code/shil_scale_stress.py",
  "code/q1_action_remediation_analysis.py",
  "code/q1_bounded_sensitivity.py",
  "replication_package/manifest.csv",
  "replication_package/calibrated_runs/README.md",
  "replication_package/calibrated_runs/manifest_sha256.csv",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/README.md",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/src/sc_shil_experiment.py",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/src/run_partitioned_full_real.py",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/src/analyze_results.py",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/tests/test_sc_shil.py",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/tests/test_partitioned_full_real.py",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/protocol/method_extension_protocol_20260717.md",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/synthetic/outputs/metrics.csv",
  "replication_package/calibrated_runs/EXP-SHIL-SC-001/dry_bean/outputs/full-real-combined/metrics.csv",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/README.md",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/src/fixed_c_experiment.py",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/src/run_partitioned_fixed_c.py",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/src/analyze_fixed_c.py",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/protocol/fixed_c_cost_sensitivity_protocol_20260727.md",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/dry_bean/outputs/full-real-combined/metrics.csv",
  "replication_package/calibrated_runs/EXP-SHIL-FIXED-C-001/dry_bean/analysis/analysis_summary.json",
  "replication_package/results/frozen_metrics_summary.csv",
  "replication_package/results/EXP-SHIL-Q1-003_q1_action_20260623_0740_l1_top8_summary.csv",
  "replication_package/results/EXP-SHIL-Q1-004_bounded_sensitivity_20260623_2234_summary.csv",
  "LICENSE",
  "LICENSE-CONTENT.md",
  "CITATION.cff"
)

foreach ($path in $requiredFiles) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Missing required package file: $path"
  }
}

python -c "import numpy, pandas, sklearn, scipy, psutil, matplotlib, PIL; import xgboost, lightgbm, catboost; print('dependency import check ok')"
if ($LASTEXITCODE -ne 0) {
  throw "Dependency import check failed. Install the pinned requirements before rerunning the smoke check."
}

$integrityCheck = @'
import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

required = {
    "replication_package/results/frozen_metrics_summary.csv": {"dataset", "model", "accuracy_mean"},
    "replication_package/results/EXP-SHIL-Q1-003_q1_action_20260623_0740_l1_top8_summary.csv": {"dataset", "model", "support_precision_mean", "support_recall_mean"},
    "replication_package/results/EXP-SHIL-Q1-004_bounded_sensitivity_20260623_2234_summary.csv": {"dataset", "model", "accuracy_mean", "roc_auc_ovr_mean"},
}

for relative_path, expected_columns in required.items():
    frame = pd.read_csv(Path(relative_path))
    if frame.empty:
        raise SystemExit(f"Evidence table is empty: {relative_path}")
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise SystemExit(f"Evidence table is missing columns {sorted(missing)}: {relative_path}")

calibrated = Path("replication_package/calibrated_runs")
manifest_path = calibrated / "manifest_sha256.csv"
with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
    manifest_rows = list(csv.DictReader(handle))

if not manifest_rows:
    raise SystemExit("Calibrated artifact manifest is empty")

for row in manifest_rows:
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"Unsafe manifest path: {relative}")
    artifact = calibrated / relative
    if not artifact.is_file():
        raise SystemExit(f"Manifest artifact is missing: {relative}")
    if artifact.stat().st_size != int(row["bytes"]):
        raise SystemExit(f"Manifest byte count differs: {relative}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest().upper()
    if digest != row["sha256"].upper():
        raise SystemExit(f"Manifest SHA-256 differs: {relative}")

sc_root = calibrated / "EXP-SHIL-SC-001"
synthetic = pd.read_csv(sc_root / "synthetic/outputs/metrics.csv")
dry_bean = pd.read_csv(sc_root / "dry_bean/outputs/full-real-combined/metrics.csv")

if len(synthetic) != 720 or synthetic[["scenario", "data_seed"]].drop_duplicates().shape[0] != 120:
    raise SystemExit("EXP-SHIL-SC-001 synthetic row count differs from the locked full matrix")
if len(dry_bean) != 60 or dry_bean["split_seed"].nunique() != 10:
    raise SystemExit("EXP-SHIL-SC-001 Dry Bean row or seed count differs")

sc_statuses = sorted(
    (sc_root / "dry_bean/outputs/full-real-partitions").glob("seed-*/attempt-01/partition_status.json")
)
if len(sc_statuses) != 10 or any(json.loads(path.read_text(encoding="utf-8"))["exit_code"] != 0 for path in sc_statuses):
    raise SystemExit("EXP-SHIL-SC-001 per-seed completion check failed")

sc_synthetic_summary = json.loads(
    (sc_root / "synthetic/analysis/analysis_summary.json").read_text(encoding="utf-8")
)
if sc_synthetic_summary["decision"] != "PROTOCOL_ONLY":
    raise SystemExit("EXP-SHIL-SC-001 decision differs from the locked analysis")

fixed_root = calibrated / "EXP-SHIL-FIXED-C-001"
fixed = pd.read_csv(fixed_root / "dry_bean/outputs/full-real-combined/metrics.csv")
if len(fixed) != 20 or fixed["split_seed"].nunique() != 10:
    raise SystemExit("EXP-SHIL-FIXED-C-001 row or seed count differs")

fixed_statuses = sorted(
    (fixed_root / "dry_bean/outputs/full-real-partitions").glob("seed-*/attempt-01/partition_status.json")
)
if len(fixed_statuses) != 10 or any(json.loads(path.read_text(encoding="utf-8"))["exit_code"] != 0 for path in fixed_statuses):
    raise SystemExit("EXP-SHIL-FIXED-C-001 per-seed completion check failed")

fixed_summary = json.loads(
    (fixed_root / "dry_bean/analysis/analysis_summary.json").read_text(encoding="utf-8")
)
if fixed_summary["decision"] != "RESIDUAL_FIXED_C_L1_SLOWER":
    raise SystemExit("EXP-SHIL-FIXED-C-001 decision differs from the locked analysis")
if not math.isclose(fixed_summary["fixed_c_l1_over_sc_shil_mean"], 14.694430159604599, rel_tol=0, abs_tol=1e-12):
    raise SystemExit("EXP-SHIL-FIXED-C-001 mean timing ratio differs")

print(f"package-local evidence and calibrated artifact checks passed ({len(manifest_rows)} SHA-256 entries)")
'@

$integrityCheck | python -
if ($LASTEXITCODE -ne 0) {
  throw "Package-local evidence schema check failed."
}

Write-Host "artifact and dependency smoke check passed; no experiment executed"
