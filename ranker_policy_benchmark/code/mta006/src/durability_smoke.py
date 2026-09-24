"""Controlled failure-injection harness for PR4 durability verification.

The payloads are deterministic engineering tokens, not scientific results.
The harness completes two units, records an injected failure for a third,
builds a validated resume plan, and completes only the missing unit under a new
attempt.  It emits an evidence JSON that can later be referenced by a real run
manifest after the scaffold is wired to the scientific entrypoint.
"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import durability


def _unit_payload(unit_id: str) -> bytes:
    return durability.canonical_json_bytes(
        {
            "evidence_class": "engineering_smoke_only",
            "unit_id": unit_id,
            "deterministic_token": durability.sha256_bytes(unit_id.encode("utf-8")),
        }
    )


def _attempt_paths(root: Path, unit_id: str, attempt_id: str) -> tuple[Path, Path]:
    attempt_root = root / "units" / unit_id / "attempts" / attempt_id
    return attempt_root / "raw" / "token.json", attempt_root / "checkpoint.json"


def _write_completed_attempt(
    root: Path,
    *,
    run_id: str,
    unit: Mapping[str, Any],
    attempt_id: str,
    bindings: durability.BindingSet,
) -> Path:
    started = durability.utc_now()
    output_path, checkpoint_path = _attempt_paths(
        root, str(unit["unit_id"]), attempt_id
    )
    durability.atomic_write_bytes(output_path, _unit_payload(str(unit["unit_id"])))
    relative_output = output_path.relative_to(root).as_posix()
    artifact = durability.artifact_record(root, relative_output, "engineering_token")
    checkpoint = durability.make_checkpoint(
        run_id=run_id,
        unit_id=str(unit["unit_id"]),
        attempt_id=attempt_id,
        status="completed",
        atomic_unit=unit["atomic_unit"],
        bindings=bindings,
        started_at_utc=started,
        ended_at_utc=durability.utc_now(),
        duration_seconds=0.0,
        pid=os.getpid(),
        worker_id="controlled-interruption-smoke",
        exit_code=0,
        signal=None,
        artifacts=[artifact],
    )
    durability.atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint_path


def _write_failed_attempt(
    root: Path,
    *,
    run_id: str,
    unit: Mapping[str, Any],
    attempt_id: str,
    bindings: durability.BindingSet,
) -> Path:
    _, checkpoint_path = _attempt_paths(root, str(unit["unit_id"]), attempt_id)
    checkpoint = durability.make_checkpoint(
        run_id=run_id,
        unit_id=str(unit["unit_id"]),
        attempt_id=attempt_id,
        status="failed",
        atomic_unit=unit["atomic_unit"],
        bindings=bindings,
        started_at_utc=durability.utc_now(),
        ended_at_utc=durability.utc_now(),
        duration_seconds=0.0,
        pid=os.getpid(),
        worker_id="controlled-interruption-smoke",
        exit_code=97,
        signal="FAILURE_INJECTION",
        artifacts=[],
        error="intentional controlled failure injection",
    )
    durability.atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint_path


def run_controlled_interruption_smoke(
    smoke_root: str | Path,
    *,
    bindings: durability.BindingSet,
    unit_ids: Sequence[str] = ("smoke-a", "smoke-b", "smoke-c"),
) -> Mapping[str, Any]:
    """Run and persist the deterministic interruption/resume evidence."""

    if len(unit_ids) < 3 or len(unit_ids) != len(set(unit_ids)):
        raise durability.DurabilityError("smoke requires at least three unique units")
    for unit_id in unit_ids:
        durability.validate_identifier(unit_id, "smoke unit_id")
    root = Path(smoke_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = "PR4-CONTROLLED-INTERRUPTION-SMOKE"
    units = [
        {"unit_id": unit_id, "atomic_unit": {"engineering_unit": unit_id}}
        for unit_id in unit_ids
    ]
    checkpoint_paths: dict[str, list[Path]] = {unit_id: [] for unit_id in unit_ids}
    initial_completed: list[str] = []
    for unit in units[:2]:
        path = _write_completed_attempt(
            root,
            run_id=run_id,
            unit=unit,
            attempt_id="attempt-001",
            bindings=bindings,
        )
        checkpoint_paths[str(unit["unit_id"])].append(path)
        initial_completed.append(str(unit["unit_id"]))
    failed_path = _write_failed_attempt(
        root,
        run_id=run_id,
        unit=units[2],
        attempt_id="attempt-001",
        bindings=bindings,
    )
    checkpoint_paths[str(units[2]["unit_id"])].append(failed_path)

    first_plan = durability.build_resume_plan(
        run_root=root,
        run_id=run_id,
        planned_units=units,
        checkpoint_paths_by_unit=checkpoint_paths,
        bindings=bindings,
    )
    resumed_units: list[str] = []
    for unit, decision in zip(units, first_plan, strict=True):
        if decision.action == "run_new_attempt":
            path = _write_completed_attempt(
                root,
                run_id=run_id,
                unit=unit,
                attempt_id=decision.next_attempt_id,
                bindings=bindings,
            )
            checkpoint_paths[decision.unit_id].append(path)
            resumed_units.append(decision.unit_id)

    final_plan = durability.build_resume_plan(
        run_root=root,
        run_id=run_id,
        planned_units=units,
        checkpoint_paths_by_unit=checkpoint_paths,
        bindings=bindings,
    )
    if any(decision.action != "skipped_validated" for decision in final_plan):
        raise durability.DurabilityError("smoke resume did not validate every unit")
    if set(initial_completed) & set(resumed_units):
        raise durability.DurabilityError("resume recomputed a validated unit")

    reference_hashes = {
        unit_id: durability.sha256_bytes(_unit_payload(unit_id)) for unit_id in unit_ids
    }
    resumed_hashes: dict[str, str] = {}
    for unit_id, paths in checkpoint_paths.items():
        valid = next(
            durability.validate_checkpoint(
                path,
                run_root=root,
                expected_run_id=run_id,
                expected_unit_id=unit_id,
                expected_atomic_unit={"engineering_unit": unit_id},
                expected_bindings=bindings,
                require_completed=True,
            )
            for path in paths
            if __checkpoint_status(path) == "completed"
        )
        resumed_hashes[unit_id] = str(valid["artifacts"][0]["sha256"])
    if resumed_hashes != reference_hashes:
        raise durability.DurabilityError("resumed outputs differ from reference tokens")

    evidence = {
        "schema_version": durability.SCHEMA_VERSION,
        "evidence_class": "engineering_durability_smoke_not_scientific_result",
        "run_id": run_id,
        "completed_before_interruption": initial_completed,
        "injected_failure": {
            "unit_id": str(units[2]["unit_id"]),
            "attempt_id": "attempt-001",
            "exit_code": 97,
        },
        "first_resume_plan": [asdict(decision) for decision in first_plan],
        "resumed_units": resumed_units,
        "validated_units_after_resume": [decision.unit_id for decision in final_plan],
        "validated_units_recomputed": [],
        "reference_hashes": reference_hashes,
        "resumed_hashes": resumed_hashes,
        "content_equivalent_to_uninterrupted_reference": True,
        "created_at_utc": durability.utc_now(),
    }
    durability.atomic_write_json(root / "interruption_smoke_evidence.json", evidence)
    return evidence


def __checkpoint_status(path: Path) -> str | None:
    try:
        import json

        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document.get("status") if isinstance(document, Mapping) else None
