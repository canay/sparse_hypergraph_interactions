from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

import run_partitioned_full_real as partitioned  # noqa: E402


class PartitionedExecutionTests(unittest.TestCase):
    def test_derived_config_changes_only_execution_partition_and_seed_subset(self) -> None:
        base = json.loads((RUN_ROOT / "config" / "full_config.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            partitioned.derive_seed_config(base, 37, path)
            derived = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(derived["real_split_seeds"], [37])
        self.assertFalse(derived["execution_partition"]["scientific_protocol_changed"])
        self.assertEqual(derived["protocol_sha256"], base["protocol_sha256"])
        for key in base:
            if key != "real_split_seeds":
                self.assertEqual(derived[key], base[key])

    def test_completed_attempt_requires_all_six_methods_and_seed_provenance(self) -> None:
        base = json.loads((RUN_ROOT / "config" / "full_config.json").read_text(encoding="utf-8"))
        protocol = base["protocol_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt-01"
            attempt.mkdir()
            rows = []
            for model in partitioned.METHODS:
                rows.append(
                    {
                        "dataset": "dry_bean_uci",
                        "scenario": "dry_bean_uci",
                        "data_seed": 11,
                        "split_seed": 11,
                        "model": model,
                    }
                )
            pd.DataFrame(rows).to_csv(attempt / "metrics.csv", index=False)
            for name in partitioned.COLLECTIONS[1:]:
                pd.DataFrame([{"split_seed": 11}]).to_csv(attempt / f"{name}.csv", index=False)
            pd.DataFrame([{"split_seed": 11}]).to_csv(attempt / "dataset_manifest.csv", index=False)
            (attempt / "run_summary.json").write_text(
                json.dumps({"status": "completed", "dataset_runs": 1}), encoding="utf-8"
            )
            (attempt / "config_resolved.json").write_text(
                json.dumps({"real_split_seeds": [11], "protocol_sha256": protocol}),
                encoding="utf-8",
            )
            (attempt / "run_identity.json").write_text(
                json.dumps({"protocol_sha256": protocol}), encoding="utf-8"
            )
            (attempt / "output_hashes.json").write_text("{}", encoding="utf-8")
            result = partitioned.validate_completed_attempt(attempt, 11, protocol)
        self.assertEqual(result["metrics_rows"], 6)


if __name__ == "__main__":
    unittest.main()
