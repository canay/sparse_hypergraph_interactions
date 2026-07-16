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

python -c "import numpy, pandas, sklearn, matplotlib, PIL; import xgboost, lightgbm, catboost; print('dependency import check ok')"
if ($LASTEXITCODE -ne 0) {
  throw "Dependency import check failed. Install the pinned requirements before rerunning the smoke check."
}

$integrityCheck = @'
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

print("package-local evidence schema check ok")
'@

$integrityCheck | python -
if ($LASTEXITCODE -ne 0) {
  throw "Package-local evidence schema check failed."
}

Write-Host "artifact and dependency smoke check passed; no experiment executed"
