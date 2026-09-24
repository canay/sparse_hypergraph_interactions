"""Shared fail-closed structural validation for full scientific evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import durability


def _sha(path: Path) -> str:
    return durability.sha256_file(path)


def validate_scientific_run_structure(
    run_root: Path,
    *,
    expected_run_id: str,
    expected_role: str,
    expected_mode: str,
    expected_unit_count: int,
    expected_scenarios: Iterable[str],
    expected_split_seeds: Iterable[int],
) -> dict[str, Any]:
    """Validate live manifest/freeze/checkpoints without opening science tables."""

    root = Path(run_root).resolve()
    candidate_root = Path(__file__).resolve().parents[1]
    project_root = candidate_root.parents[1]
    manifest_path = root / "RUN_MANIFEST.json"
    summary_path = root / "run_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != expected_run_id
        or manifest.get("run_class") != expected_role
        or manifest.get("status") != "scientific_run_frozen"
        or manifest.get("planned_unit_count") != int(expected_unit_count)
        or len(manifest.get("planned_units", [])) != int(expected_unit_count)
        or summary.get("status") != "completed"
        or summary.get("mode") != expected_mode
        or summary.get("block_role") != expected_role
        or summary.get("dataset_runs") != int(expected_unit_count)
    ):
        raise AssertionError("Full scientific run identity/count mismatch")

    bindings = durability.validate_run_manifest(
        manifest,
        candidate_root=candidate_root,
        project_root=project_root,
        require_execution_disabled=False,
    )
    config_binding = manifest["bindings"]["config"]
    config_path = candidate_root / str(config_binding["path"])
    resolved_config = json.loads(
        (root / "config_resolved.json").read_text(encoding="utf-8")
    )
    live_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        _sha(config_path) != config_binding["sha256"]
        or resolved_config != live_config
        or live_config.get("execution_class") != expected_role
    ):
        raise AssertionError("Full scientific live config binding mismatch")

    durability_block = manifest.get("durability", {})
    freeze_binding = durability_block.get("scientific_run_freeze")
    if not isinstance(freeze_binding, dict) or set(freeze_binding) != {
        "path",
        "sha256",
    }:
        raise AssertionError("Full scientific freeze binding missing")
    freeze_path = candidate_root / str(freeze_binding["path"])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        _sha(freeze_path) != freeze_binding["sha256"]
        or freeze.get("status") != "SCIENTIFIC_RUN_FROZEN"
        or freeze.get("results_seen") is not False
        or freeze.get("authorized_run_id") != expected_run_id
        or freeze.get("block_role") != expected_role
        or freeze.get("authorized_mode") != expected_mode
        or freeze.get("exact_units") != manifest["planned_units"]
    ):
        raise AssertionError("Full scientific live freeze binding mismatch")

    planned = manifest["planned_units"]
    observed_scenarios = [str(row["atomic_unit"]["scenario"]) for row in planned]
    observed_seeds = [int(row["atomic_unit"]["split_seed"]) for row in planned]
    expected_scenario_set = set(expected_scenarios)
    expected_seed_set = {int(value) for value in expected_split_seeds}
    if (
        set(observed_scenarios) != expected_scenario_set
        or set(observed_seeds) != expected_seed_set
        or any(row["atomic_unit"]["mode"] != expected_mode for row in planned)
    ):
        raise AssertionError("Full scientific exact unit identity mismatch")

    validated_paths: list[str] = []
    for unit in planned:
        unit_id = str(unit["unit_id"])
        paths = sorted(
            (root / "units" / unit_id / "attempts").glob("*/checkpoint.json")
        )
        valid: list[Path] = []
        for path in paths:
            try:
                durability.validate_checkpoint(
                    path,
                    run_root=root,
                    expected_run_id=expected_run_id,
                    expected_unit_id=unit_id,
                    expected_atomic_unit=dict(unit["atomic_unit"]),
                    expected_bindings=bindings,
                )
            except durability.DurabilityError:
                continue
            valid.append(path)
        if len(valid) != 1:
            raise AssertionError(
                f"Full scientific unit lacks one canonical valid checkpoint: {unit_id}"
            )
        first_completed = next(
            (
                path
                for path in paths
                if json.loads(path.read_text(encoding="utf-8")).get("status")
                == "completed"
            ),
            None,
        )
        if first_completed != valid[0]:
            raise AssertionError(f"First-valid checkpoint rule mismatch: {unit_id}")
        validated_paths.append(valid[0].relative_to(root).as_posix())

    return {
        "run_id": expected_run_id,
        "block_role": expected_role,
        "mode": expected_mode,
        "planned_unit_count": expected_unit_count,
        "validated_checkpoint_count": len(validated_paths),
        "freeze_sha256": freeze_binding["sha256"],
        "manifest_sha256": _sha(manifest_path),
        "scientific_tables_read": False,
    }
