"""Read-only, result-blind structural validator for Track-A Pilot Block A.

The validator reads a sealed pilot run but emits only a fixed allowlist of
engineering predicates, non-scientific counts, hashes, statuses, and fixed
failure codes.  It never imports or invokes the scientific analysis module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import durability
import freeze_bindings
from method_registry import load_method_registry
from structured_comparator_registry import structured_registry_document_sha256


SCHEMA_VERSION = 1
VALIDATOR_ID = "track-a-blind-pilot-structural-validator-v3"
PILOT_SCENARIOS = ("S00", "S11", "S13")
PILOT_SEEDS = (901, 907)
EXPECTED_UNIT_KEYS = frozenset(itertools.product(PILOT_SCENARIOS, PILOT_SEEDS))
EXPECTED_METHOD_ROWS = 6 * 14
EXPECTED_SCORE_ROWS = 6 * 4
EXPECTED_STRUCTURED_COMPLETIONS = 2
HASH_FIELDS = frozenset(
    {
        "freeze_sha256",
        "manifest_sha256",
        "terminal_sha256",
        "config_sha256",
        "protocol_sha256",
        "primary_registry_file_sha256",
        "structured_registry_file_sha256",
        "source_bundle_sha256",
        "preflight_sha256",
        "output_hash_manifest_sha256",
        "notification_lifecycle_sha256",
    }
)


class BlindPilotValidationError(ValueError):
    """Internal validation failure carrying only a fixed non-scientific code."""


def _fail(code: str) -> None:
    raise BlindPilotValidationError(code)


def _read_json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _read_jsonl(path: Path, code: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _fail(code)
    if not values or any(not isinstance(value, Mapping) for value in values):
        _fail(code)
    return values


def _read_csv(path: Path, code: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        _fail(code)


def _json_list(raw: Any, code: str) -> list[Any]:
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        _fail(code)
    if not isinstance(value, list):
        _fail(code)
    return value


def _finite_number(value: Any, code: str) -> float:
    if isinstance(value, bool):
        _fail(code)
    try:
        result = float(value)
    except (TypeError, ValueError):
        _fail(code)
    if not math.isfinite(result):
        _fail(code)
    return result


def _integer(value: Any, code: str) -> int:
    number = _finite_number(value, code)
    if not number.is_integer():
        _fail(code)
    return int(number)


def _bool(value: Any, code: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    _fail(code)


def _relative_child(root: Path, raw: Any, code: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        _fail(code)
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        _fail(code)
    return target


def _resolve_bound_path(
    record: Mapping[str, Any], candidate_root: Path, project_root: Path, code: str
) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        _fail(code)
    for root in (candidate_root, project_root):
        target = _relative_child(root, raw, code)
        if target.is_file():
            return target
    _fail(code)


def _require_file_hash(path: Path, expected: Any, code: str) -> str:
    if not path.is_file():
        _fail(code)
    observed = durability.sha256_file(path)
    if observed != str(expected).upper():
        _fail(code)
    return observed


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(durability.canonical_json_bytes(value)).hexdigest().upper()


@dataclass
class ValidationState:
    pilot_root: Path
    freeze_path: Path
    expected_freeze_sha256: str
    candidate_root: Path
    project_root: Path
    preflight_path: Path
    freeze: Mapping[str, Any] | None = None
    manifest: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    config: Mapping[str, Any] | None = None
    primary_registry: Any = None
    structured_registry_document: Mapping[str, Any] | None = None
    bindings: durability.BindingSet | None = None
    canonical_checkpoints: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    partitions: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    completed_units: int = 0


def _freeze_exact(state: ValidationState) -> None:
    observed = _require_file_hash(
        state.freeze_path, state.expected_freeze_sha256, "FREEZE_HASH_MISMATCH"
    )
    freeze = _read_json(state.freeze_path, "FREEZE_UNREADABLE")
    if freeze.get("schema_version") != SCHEMA_VERSION:
        _fail("FREEZE_SCHEMA_MISMATCH")
    if (
        freeze.get("freeze_id") != "track-a-scientific-pilot-a2-schema-repair-20260826"
        or freeze.get("authorized_run_id") != "EXP-SHIL-PILOT-A2-SCHEMA-REPAIR-pilot"
    ):
        _fail("FREEZE_REPAIR_IDENTITY_MISMATCH")
    if freeze.get("run_class") != "pilot":
        _fail("FREEZE_RUN_CLASS_MISMATCH")
    if freeze.get("results_seen") is not False:
        _fail("FREEZE_NOT_RESULT_UNSEEN")
    if freeze.get("status") != "SCIENTIFIC_RUN_FROZEN":
        _fail("FREEZE_SCIENTIFIC_RUN_NOT_FROZEN")
    if freeze.get("execution_enabled") is not True:
        _fail("FREEZE_EXECUTION_GATE_MISMATCH")
    if not isinstance(freeze.get("bindings"), Mapping):
        _fail("FREEZE_BINDINGS_MISSING")
    exact_units = freeze.get("exact_units")
    observed_units = (
        {
            (
                str(record.get("scenario_id")),
                _integer(record.get("seed"), "FREEZE_UNIT_SEED_INVALID"),
            )
            for record in exact_units
        }
        if isinstance(exact_units, list)
        and all(isinstance(record, Mapping) for record in exact_units)
        else set()
    )
    workload = freeze.get("workload")
    blindness = freeze.get("blindness")
    durability_contract = freeze.get("durability")
    authorization = freeze.get("authorization")
    repair = freeze.get("repair")
    if (
        not isinstance(exact_units, list)
        or observed_units != EXPECTED_UNIT_KEYS
        or len(exact_units) != 6
    ):
        _fail("FREEZE_UNIT_GRID_MISMATCH")
    if (
        not isinstance(workload, Mapping)
        or workload.get("n_pairs") != 4
        or workload.get("ranker_call_budget_4B_plus_2") != 18
    ):
        _fail("FREEZE_WORKLOAD_MISMATCH")
    if (
        not isinstance(blindness, Mapping)
        or blindness.get("mode") != "structural_only"
        or blindness.get("scientific_metrics_visible_during_pilot") is not False
        or blindness.get("scientific_magnitudes_visible_during_pilot") is not False
    ):
        _fail("FREEZE_BLINDNESS_MISMATCH")
    if (
        not isinstance(durability_contract, Mapping)
        or durability_contract.get("minimum_advancing_heartbeat_samples") != 2
        or durability_contract.get("max_attempts_per_unit") != 2
        or durability_contract.get("reentry_rule")
        != "same_atomic_unit_seed_only_no_replacement_seed"
        or durability_contract.get("authorized_reentry_seeds") != []
    ):
        _fail("FREEZE_DURABILITY_MISMATCH")
    if (
        not isinstance(authorization, Mapping)
        or authorization.get("decision")
        != "AUTHORIZE_STRUCTURAL_ONLY_SIX_UNIT_PILOT_A2_SCHEMA_REPAIR"
        or authorization.get("pilot_execution_authorized") is not True
        or authorization.get("global_scientific_execution_authorized") is not False
        or authorization.get("analysis_or_magnitude_inspection_authorized") is not False
    ):
        _fail("FREEZE_AUTHORIZATION_MISMATCH")
    if (
        not isinstance(repair, Mapping)
        or repair.get("classification") != "SOURCE_ONLY_SCHEMA_WRITER_REPAIR"
        or repair.get("repair_number") != 1
        or repair.get("further_source_repair_allowed") is not False
        or repair.get("all_six_units_restart_required") is not True
        or repair.get("pilot_a1_artifact_reuse_allowed") is not False
    ):
        _fail("FREEZE_REPAIR_CONTRACT_MISMATCH")
    for label, expected_path in (
        (
            "pilot_a1_freeze",
            "freeze/SCIENTIFIC_RUN_FROZEN_PILOT_A1_IMMUTABLE.json",
        ),
        (
            "repair_classification",
            "evidence/pilot_a1_repair_classification_20260826/DUAL_REPAIR_CLASSIFICATION.json",
        ),
        (
            "repair_implementation_receipt",
            "evidence/pilot_a1_repair_classification_20260826/REPAIR_IMPLEMENTATION_RECEIPT.json",
        ),
        ("blind_validator_v3", "src/validate_blind_pilot.py"),
    ):
        record = freeze["bindings"].get(label)
        if not isinstance(record, Mapping) or record.get("path") != expected_path:
            _fail("FREEZE_REPAIR_BINDING_MISSING")
        bound_path = _resolve_bound_path(
            record,
            state.candidate_root,
            state.project_root,
            "FREEZE_REPAIR_BINDING_PATH_INVALID",
        )
        _require_file_hash(
            bound_path,
            record.get("sha256"),
            "FREEZE_REPAIR_BINDING_HASH_MISMATCH",
        )
    state.freeze = freeze
    state.hashes["freeze_sha256"] = observed


def _manifest_terminal_exact(state: ValidationState) -> None:
    manifest_path = state.pilot_root / "RUN_MANIFEST.json"
    terminal_path = state.pilot_root / "terminal_status.json"
    manifest = _read_json(manifest_path, "MANIFEST_UNREADABLE")
    terminal = _read_json(terminal_path, "TERMINAL_UNREADABLE")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "run_id"
    ) != terminal.get("run_id"):
        _fail("MANIFEST_TERMINAL_IDENTITY_MISMATCH")
    if (
        terminal.get("schema_version") != SCHEMA_VERSION
        or terminal.get("status") != "completed"
    ):
        _fail("TERMINAL_NOT_COMPLETED")
    if terminal.get("exit_code") != 0 or terminal.get("error") is not None:
        _fail("TERMINAL_EXIT_MISMATCH")
    history = _read_jsonl(
        state.pilot_root / "terminal_status_history.jsonl",
        "TERMINAL_HISTORY_UNREADABLE",
    )
    if history[-1] != terminal or history[0].get("status") != "running":
        _fail("TERMINAL_HISTORY_MISMATCH")
    summary = _read_json(
        state.pilot_root / "run_summary.json", "RUN_SUMMARY_UNREADABLE"
    )
    if (
        summary.get("mode") != "pilot"
        or summary.get("status") != "completed"
        or summary.get("dataset_runs") != 6
        or summary.get("method_rows") != EXPECTED_METHOD_ROWS
        or summary.get("ranking_metric_rows") != EXPECTED_SCORE_ROWS
        or summary.get("durability_bindings")
        != {
            "protocol_sha256": manifest.get("bindings", {})
            .get("protocol", {})
            .get("sha256"),
            "config_sha256": manifest.get("bindings", {})
            .get("config", {})
            .get("sha256"),
            "registry_sha256": manifest.get("bindings", {})
            .get("registry", {})
            .get("sha256"),
            "source_bundle_sha256": manifest.get("bindings", {})
            .get("source", {})
            .get("source_bundle_sha256"),
        }
    ):
        _fail("RUN_SUMMARY_STRUCTURAL_MISMATCH")
    state.manifest = manifest
    state.terminal = terminal
    state.hashes["manifest_sha256"] = durability.sha256_file(manifest_path)
    state.hashes["terminal_sha256"] = durability.sha256_file(terminal_path)


def _config_protocol_registry_source_exact(state: ValidationState) -> None:
    if state.freeze is None or state.manifest is None:
        _fail("BINDING_PREREQUISITE_MISSING")
    freeze_bindings_doc = state.freeze["bindings"]
    manifest_bindings = state.manifest.get("bindings")
    if not isinstance(manifest_bindings, Mapping):
        _fail("MANIFEST_BINDINGS_MISSING")
    config_record = freeze_bindings_doc.get("execution_config")
    protocol_record = freeze_bindings_doc.get("protocol")
    primary_record = freeze_bindings_doc.get("primary_registry")
    structured_record = freeze_bindings_doc.get("structured_registry")
    source_record = state.freeze.get("source_bundle")
    if not all(
        isinstance(record, Mapping)
        for record in (
            config_record,
            protocol_record,
            primary_record,
            structured_record,
            source_record,
        )
    ):
        _fail("FREEZE_REQUIRED_BINDING_MISSING")
    config_path = _resolve_bound_path(
        config_record, state.candidate_root, state.project_root, "CONFIG_PATH_INVALID"
    )
    protocol_path = _resolve_bound_path(
        protocol_record,
        state.candidate_root,
        state.project_root,
        "PROTOCOL_PATH_INVALID",
    )
    primary_path = _resolve_bound_path(
        primary_record,
        state.candidate_root,
        state.project_root,
        "PRIMARY_REGISTRY_PATH_INVALID",
    )
    structured_path = _resolve_bound_path(
        structured_record,
        state.candidate_root,
        state.project_root,
        "STRUCTURED_REGISTRY_PATH_INVALID",
    )
    config_sha = _require_file_hash(
        config_path, config_record.get("sha256"), "CONFIG_HASH_MISMATCH"
    )
    protocol_sha = _require_file_hash(
        protocol_path, protocol_record.get("sha256"), "PROTOCOL_HASH_MISMATCH"
    )
    primary_sha = _require_file_hash(
        primary_path,
        primary_record.get("file_sha256"),
        "PRIMARY_REGISTRY_HASH_MISMATCH",
    )
    structured_sha = _require_file_hash(
        structured_path,
        structured_record.get("file_sha256"),
        "STRUCTURED_REGISTRY_HASH_MISMATCH",
    )
    config = _read_json(config_path, "CONFIG_UNREADABLE")
    if config.get("execution_enabled") is not True:
        _fail("PILOT_CONFIG_EXECUTION_NOT_ENABLED")
    if config.get("protocol_sha256") != protocol_sha:
        _fail("CONFIG_PROTOCOL_BINDING_MISMATCH")
    if config.get("pilot_scenario_ids") != list(PILOT_SCENARIOS) or config.get(
        "pilot_seeds"
    ) != list(PILOT_SEEDS):
        _fail("PILOT_CONFIG_PLAN_MISMATCH")
    resolved_config = config.get("track_a_resolved_config")
    if not isinstance(resolved_config, Mapping):
        _fail("PILOT_CONFIG_RESOLVED_GRID_MISSING")
    primary = load_method_registry(
        primary_path, available_config_keys=resolved_config.keys()
    )
    if (
        primary.canonical_sha256 != primary_record.get("canonical_sha256")
        or len(primary.methods) != 14
    ):
        _fail("PRIMARY_REGISTRY_CANONICAL_MISMATCH")
    structured_document = _read_json(structured_path, "STRUCTURED_REGISTRY_UNREADABLE")
    if (
        structured_registry_document_sha256(structured_document)
        != structured_record.get("canonical_sha256")
        or structured_document.get("canonical_sha256")
        != structured_record.get("canonical_sha256")
        or structured_document.get("registry_active") is not True
    ):
        _fail("STRUCTURED_REGISTRY_CANONICAL_MISMATCH")
    if manifest_bindings.get("config", {}).get("sha256") != config_sha:
        _fail("MANIFEST_CONFIG_BINDING_MISMATCH")
    if manifest_bindings.get("protocol", {}).get("sha256") != protocol_sha:
        _fail("MANIFEST_PROTOCOL_BINDING_MISMATCH")
    if manifest_bindings.get("registry", {}).get("sha256") != primary_sha:
        _fail("MANIFEST_REGISTRY_BINDING_MISMATCH")
    try:
        bindings = durability.validate_run_manifest(
            state.manifest,
            candidate_root=state.candidate_root,
            project_root=state.project_root,
            require_execution_disabled=False,
        )
    except durability.DurabilityError:
        _fail("MANIFEST_LIVE_BINDING_VALIDATION_FAILED")
    if bindings.source_bundle_sha256 != source_record.get("sha256") or len(
        manifest_bindings.get("source", {}).get("files", [])
    ) != source_record.get("file_count"):
        _fail("SOURCE_FREEZE_BINDING_MISMATCH")
    resolved_config_path = state.pilot_root / "config_resolved.json"
    resolved_primary_path = state.pilot_root / "method_registry_resolved.json"
    resolved_structured_path = (
        state.pilot_root / "structured_comparator_registry_resolved.json"
    )
    resolved_config_document = _read_json(
        resolved_config_path, "RESOLVED_CONFIG_UNREADABLE"
    )
    if resolved_config_document != config:
        _fail("RESOLVED_CONFIG_CONTENT_MISMATCH")
    _require_file_hash(
        resolved_primary_path, primary_sha, "RESOLVED_PRIMARY_REGISTRY_HASH_MISMATCH"
    )
    _require_file_hash(
        resolved_structured_path,
        structured_sha,
        "RESOLVED_STRUCTURED_REGISTRY_HASH_MISMATCH",
    )
    state.config = config
    state.primary_registry = primary
    state.structured_registry_document = structured_document
    state.bindings = bindings
    state.hashes.update(
        {
            "config_sha256": config_sha,
            "protocol_sha256": protocol_sha,
            "primary_registry_file_sha256": primary_sha,
            "structured_registry_file_sha256": structured_sha,
            "source_bundle_sha256": bindings.source_bundle_sha256,
        }
    )


def _preflight_exact(state: ValidationState) -> None:
    if state.freeze is None:
        _fail("PREFLIGHT_PREREQUISITE_MISSING")
    record = state.freeze["bindings"].get("timing_resource_preflight")
    if not isinstance(record, Mapping):
        _fail("PREFLIGHT_FREEZE_BINDING_MISSING")
    observed = _require_file_hash(
        state.preflight_path, record.get("sha256"), "PREFLIGHT_HASH_MISMATCH"
    )
    document = _read_json(state.preflight_path, "PREFLIGHT_UNREADABLE")
    required = {
        "schema_version",
        "capture_kind",
        "captured_at_utc",
        "timing_evidence",
        "source_run_manifest",
        "timing_summary",
        "resources",
        "checks",
        "overall_status",
    }
    if set(document) != required or document.get("schema_version") != SCHEMA_VERSION:
        _fail("PREFLIGHT_SCHEMA_MISMATCH")
    timing_record = state.freeze["bindings"].get("timing_evidence")
    timing_manifest_record = state.freeze["bindings"].get("timing_run_manifest")
    if not isinstance(timing_record, Mapping) or not isinstance(
        timing_manifest_record, Mapping
    ):
        _fail("PREFLIGHT_UPSTREAM_BINDING_MISSING")
    timing_path = _resolve_bound_path(
        timing_record,
        state.candidate_root,
        state.project_root,
        "TIMING_EVIDENCE_PATH_INVALID",
    )
    timing_manifest_path = _resolve_bound_path(
        timing_manifest_record,
        state.candidate_root,
        state.project_root,
        "TIMING_MANIFEST_PATH_INVALID",
    )
    _require_file_hash(
        timing_path, timing_record.get("sha256"), "TIMING_EVIDENCE_HASH_MISMATCH"
    )
    _require_file_hash(
        timing_manifest_path,
        timing_manifest_record.get("sha256"),
        "TIMING_MANIFEST_HASH_MISMATCH",
    )
    if (
        document.get("overall_status") != "pass"
        or set(document.get("checks", {}).values()) != {"pass"}
        or document.get("timing_summary", {}).get("planned_unit_count") != 6
        or float(document.get("timing_summary", {}).get("safety_margin_fraction", -1))
        < 0.5
        or document.get("timing_evidence", {}).get("sha256")
        != timing_record.get("sha256")
        or document.get("source_run_manifest", {}).get("sha256")
        != timing_manifest_record.get("sha256")
    ):
        _fail("PREFLIGHT_SCOPE_OR_STATUS_MISMATCH")
    state.hashes["preflight_sha256"] = observed


def _unit_plan_exact(state: ValidationState) -> None:
    if state.manifest is None:
        _fail("UNIT_PLAN_PREREQUISITE_MISSING")
    if (
        state.manifest.get("planned_unit_count") != 6
        or state.manifest.get("execution_enabled") is not True
        or state.manifest.get("run_id") != state.freeze.get("authorized_run_id")
    ):
        _fail("UNIT_PLAN_COUNT_OR_GATE_MISMATCH")
    units = state.manifest.get("planned_units")
    if not isinstance(units, list) or len(units) != 6:
        _fail("UNIT_PLAN_SCHEMA_MISMATCH")
    observed: set[tuple[str, int]] = set()
    unit_ids: set[str] = set()
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != {"unit_id", "atomic_unit"}:
            _fail("UNIT_PLAN_SCHEMA_MISMATCH")
        atomic = unit["atomic_unit"]
        if not isinstance(atomic, Mapping) or atomic.get("mode") != "pilot":
            _fail("UNIT_PLAN_ATOMIC_SCOPE_MISMATCH")
        key = (
            str(atomic.get("scenario")),
            _integer(atomic.get("split_seed"), "UNIT_SEED_INVALID"),
        )
        if _integer(atomic.get("data_seed"), "UNIT_SEED_INVALID") != key[1]:
            _fail("UNIT_SEED_IDENTITY_MISMATCH")
        observed.add(key)
        unit_ids.add(str(unit["unit_id"]))
    if observed != EXPECTED_UNIT_KEYS or len(unit_ids) != 6:
        _fail("UNIT_PLAN_GRID_MISMATCH")


def _attempts_and_artifacts_exact(state: ValidationState) -> None:
    if state.manifest is None or state.bindings is None:
        _fail("ATTEMPT_PREREQUISITE_MISSING")
    for unit in state.manifest["planned_units"]:
        unit_id = str(unit["unit_id"])
        attempt_root = state.pilot_root / "units" / unit_id / "attempts"
        attempt_dirs = sorted(
            path for path in attempt_root.glob("attempt-*") if path.is_dir()
        )
        attempt_names = [path.name for path in attempt_dirs]
        checkpoints = [path / "checkpoint.json" for path in attempt_dirs]
        max_attempts = int(state.freeze["durability"]["max_attempts_per_unit"])
        if (
            not checkpoints
            or len(checkpoints) > max_attempts
            or any(not path.is_file() for path in checkpoints)
        ):
            _fail("ATTEMPT_LIMIT_VIOLATION")
        if attempt_names != [
            f"attempt-{index:03d}" for index in range(1, len(checkpoints) + 1)
        ]:
            _fail("ATTEMPT_SEQUENCE_VIOLATION")
        valid: list[tuple[Path, Mapping[str, Any]]] = []
        first_valid = False
        for index, path in enumerate(checkpoints):
            try:
                checkpoint = durability.validate_checkpoint(
                    path,
                    run_root=state.pilot_root,
                    expected_run_id=str(state.manifest["run_id"]),
                    expected_unit_id=unit_id,
                    expected_atomic_unit=unit["atomic_unit"],
                    expected_bindings=state.bindings,
                    require_completed=False,
                )
            except durability.DurabilityError:
                _fail("ATTEMPT_SCHEMA_OR_BINDING_INVALID")
            if checkpoint.get("attempt_id") != path.parent.name:
                _fail("ATTEMPT_DIRECTORY_IDENTITY_MISMATCH")
            if checkpoint.get("status") == "completed":
                valid.append((path, checkpoint))
                if index == 0:
                    first_valid = True
        if len(valid) != 1 or (first_valid and len(checkpoints) != 1):
            _fail("FIRST_VALID_ATTEMPT_CANONICAL_VIOLATION")
        path, checkpoint = valid[0]
        state.canonical_checkpoints[unit_id] = checkpoint
        by_kind = {
            str(record["kind"]): state.pilot_root / str(record["path"])
            for record in checkpoint["artifacts"]
        }
        required_kinds = {
            "partition_dataset_manifest",
            *{
                f"partition_collection:{name}"
                for name in (
                    "metrics",
                    "selected_edges",
                    "ranking_metrics",
                    "ranking_scores",
                    "fit_calls",
                    "calibration_path",
                    "structured_sensitivity_status",
                    "structured_sensitivity_metrics",
                    "structured_sensitivity_selected_edges",
                    "structured_sensitivity_raw",
                    "partition_manifest",
                )
            },
        }
        if set(by_kind) != required_kinds:
            _fail("PARTITION_ARTIFACT_SET_MISMATCH")
        frames = {
            kind.removeprefix("partition_collection:"): _read_csv(
                artifact_path, "PARTITION_CSV_UNREADABLE"
            )
            for kind, artifact_path in by_kind.items()
            if kind.startswith("partition_collection:")
        }
        frames["dataset_manifest"] = _read_csv(
            by_kind["partition_dataset_manifest"], "DATASET_MANIFEST_UNREADABLE"
        )
        state.partitions[unit_id] = frames
    state.completed_units = len(state.canonical_checkpoints)


def _primary_identity_exact(state: ValidationState) -> None:
    if state.primary_registry is None or not state.partitions:
        _fail("PRIMARY_IDENTITY_PREREQUISITE_MISSING")
    expected = {method.method_id for method in state.primary_registry.methods}
    if len(expected) != 14:
        _fail("PRIMARY_METHOD_EXPECTATION_MISMATCH")
    for frames in state.partitions.values():
        for name in ("metrics", "selected_edges"):
            frame = frames[name]
            if len(frame) != 14 or set(frame.get("method_id", ())) != expected:
                _fail("PRIMARY_METHOD_IDENTITY_MISMATCH")
            if frame["method_id"].duplicated().any():
                _fail("PRIMARY_METHOD_DUPLICATE")


def _partition_manifest_exact(state: ValidationState) -> None:
    if state.primary_registry is None or not state.partitions:
        _fail("PARTITION_MANIFEST_PREREQUISITE_MISSING")
    expected_methods = [method.method_id for method in state.primary_registry.methods]
    expected_families = list(
        dict.fromkeys(
            method.score_family_id for method in state.primary_registry.methods
        )
    )
    structured_sha = state.structured_registry_document.get("canonical_sha256")
    for unit_id, frames in state.partitions.items():
        manifest = frames["partition_manifest"]
        if len(manifest) != 1:
            _fail("PARTITION_MANIFEST_CARDINALITY_MISMATCH")
        row = manifest.iloc[0]
        atomic = state.canonical_checkpoints[unit_id]["atomic_unit"]
        if (
            str(row.get("scenario")) != str(atomic.get("scenario"))
            or _integer(row.get("split_seed"), "PARTITION_SEED_INVALID")
            != int(atomic.get("split_seed"))
            or _integer(row.get("data_seed"), "PARTITION_SEED_INVALID")
            != int(atomic.get("data_seed"))
            or _integer(row.get("n_pairs"), "PARTITION_WORKLOAD_INVALID") != 4
            or _json_list(
                row.get("expected_method_ids_json"),
                "PARTITION_METHOD_JSON_INVALID",
            )
            != expected_methods
            or _json_list(
                row.get("expected_score_family_ids_json"),
                "PARTITION_SCORE_JSON_INVALID",
            )
            != expected_families
            or str(row.get("registry_sha256"))
            != state.primary_registry.canonical_sha256
            or str(row.get("structured_registry_sha256")) != structured_sha
        ):
            _fail("PARTITION_MANIFEST_IDENTITY_MISMATCH")


def _score_family_identity_exact(state: ValidationState) -> None:
    if state.primary_registry is None or not state.partitions:
        _fail("SCORE_IDENTITY_PREREQUISITE_MISSING")
    expected = {method.score_family_id for method in state.primary_registry.methods}
    if len(expected) != 4:
        _fail("SCORE_FAMILY_EXPECTATION_MISMATCH")
    for frames in state.partitions.values():
        for name in ("ranking_metrics", "ranking_scores"):
            frame = frames[name]
            if len(frame) != 4 or set(frame.get("score_family_id", ())) != expected:
                _fail("SCORE_FAMILY_IDENTITY_MISMATCH")
            if frame["score_family_id"].duplicated().any():
                _fail("SCORE_FAMILY_DUPLICATE")


def _finite_range_schema_exact(state: ValidationState) -> None:
    proportion_fields = (
        "macro_f1",
        "accuracy",
        "support_precision",
        "support_recall",
        "support_f1",
        "false_discovery_proportion",
        "group_precision",
        "group_recall",
        "group_f1",
        "group_false_discovery_proportion",
    )
    count_fields = (
        "selected_edge_count",
        "false_inclusions",
        "group_tp",
        "group_false_inclusions",
    )
    for frames in state.partitions.values():
        metrics = frames["metrics"]
        required = {
            "method_id",
            "ranker_id",
            "policy_id",
            "score_family_id",
            "selected_edge_count",
            "candidate_average_precision",
            "log_loss",
            *proportion_fields,
            *count_fields,
        }
        if not required.issubset(metrics.columns):
            _fail("PRIMARY_METRIC_SCHEMA_MISMATCH")
        for row in metrics.to_dict(orient="records"):
            if str(row.get("status")) != "OK":
                _fail("PRIMARY_STATUS_NOT_OK")
            if _finite_number(row["log_loss"], "NUMERIC_FINITE_CHECK_FAILED") < 0:
                _fail("NUMERIC_RANGE_CHECK_FAILED")
            if str(row.get("scenario")) != "S00":
                average_precision = _finite_number(
                    row["candidate_average_precision"],
                    "NUMERIC_FINITE_CHECK_FAILED",
                )
                if average_precision < 0 or average_precision > 1:
                    _fail("NUMERIC_RANGE_CHECK_FAILED")
            for field in proportion_fields:
                value = _finite_number(row[field], "NUMERIC_FINITE_CHECK_FAILED")
                if value < 0 or value > 1:
                    _fail("NUMERIC_RANGE_CHECK_FAILED")
            for field in count_fields:
                if _integer(row[field], "INTEGER_SCHEMA_CHECK_FAILED") < 0:
                    _fail("INTEGER_RANGE_CHECK_FAILED")
        ranking = frames["ranking_metrics"]
        if not {
            "candidate_count",
            "nonzero_score_count",
            "candidate_average_precision",
        }.issubset(ranking.columns):
            _fail("RANKING_METRIC_SCHEMA_MISMATCH")
        for row in ranking.to_dict(orient="records"):
            if str(row.get("status")) != "OK":
                _fail("RANKING_STATUS_NOT_OK")
            candidate_count = _integer(row["candidate_count"], "RANKING_COUNT_INVALID")
            nonzero = _integer(row["nonzero_score_count"], "RANKING_COUNT_INVALID")
            if candidate_count < 1 or not 0 <= nonzero <= candidate_count:
                _fail("RANKING_COUNT_INVALID")
            if str(row.get("scenario")) != "S00":
                average_precision = _finite_number(
                    row["candidate_average_precision"],
                    "RANKING_NUMERIC_FINITE_CHECK_FAILED",
                )
                if average_precision < 0 or average_precision > 1:
                    _fail("RANKING_NUMERIC_RANGE_CHECK_FAILED")
        ranking_scores = frames["ranking_scores"]
        if any(
            str(row.get("status")) != "OK"
            for row in ranking_scores.to_dict(orient="records")
        ):
            _fail("RANKING_STATUS_NOT_OK")


def _selection_and_raw_structure_exact(state: ValidationState) -> None:
    if state.primary_registry is None:
        _fail("SELECTION_PREREQUISITE_MISSING")
    for frames in state.partitions.values():
        rankings = frames["ranking_scores"]
        raw_by_ranker: dict[
            str, tuple[list[Any], list[float], list[int], list[int]]
        ] = {}
        for row in rankings.to_dict(orient="records"):
            candidates = _json_list(
                row.get("candidate_edges_json"), "CANDIDATE_JSON_INVALID"
            )
            scores = [
                _finite_number(value, "SCORE_NONFINITE")
                for value in _json_list(row.get("scores_json"), "SCORE_JSON_INVALID")
            ]
            if len(candidates) != len(scores) or len(candidates) != len(
                {json.dumps(edge, sort_keys=True) for edge in candidates}
            ):
                _fail("CANDIDATE_SCORE_IDENTITY_MISMATCH")
            if str(row.get("score_source")) != "final_ranker_score":
                continue
            ranked = [
                _integer(value, "RANKED_INDEX_INVALID")
                for value in _json_list(
                    row.get("ranked_indices_json"), "RANKED_INDEX_JSON_INVALID"
                )
            ]
            active = [
                _integer(value, "ACTIVE_INDEX_INVALID")
                for value in _json_list(
                    row.get("active_indices_json"), "ACTIVE_INDEX_JSON_INVALID"
                )
            ]
            if (
                len(ranked) != len(set(ranked))
                or set(ranked) != set(active)
                or any(index < 0 or index >= len(candidates) for index in ranked)
            ):
                _fail("RAW_REFERENCE_ELIGIBILITY_INVALID")
            ranker = str(row.get("ranker_id"))
            raw_by_ranker[ranker] = (candidates, scores, ranked, active)
        if set(raw_by_ranker) != {"shil", "l1"}:
            _fail("RAW_REFERENCE_RANKER_SET_MISMATCH")
        metrics_by_method = frames["metrics"].set_index("method_id")
        for row in frames["selected_edges"].to_dict(orient="records"):
            method_id = str(row.get("method_id"))
            ranker = str(row.get("ranker_id"))
            if ranker not in raw_by_ranker:
                _fail("SELECTION_RANKER_INVALID")
            candidates, _, ranked, _ = raw_by_ranker[ranker]
            indices = [
                _integer(value, "SELECTION_INDEX_INVALID")
                for value in _json_list(
                    row.get("support_indices_json"), "SELECTION_INDEX_JSON_INVALID"
                )
            ]
            edges = _json_list(row.get("edges_json"), "SELECTION_EDGE_JSON_INVALID")
            count = _integer(row.get("selected_edge_count"), "SELECTION_COUNT_INVALID")
            if (
                count != len(indices)
                or count != len(edges)
                or len(indices) != len(set(indices))
                or count > len(ranked)
                or any(index < 0 or index >= len(candidates) for index in indices)
            ):
                _fail("SELECTION_CARDINALITY_OR_DUPLICATE_INVALID")
            if edges != [candidates[index] for index in indices]:
                _fail("SELECTION_CANDIDATE_IDENTITY_MISMATCH")
            method = state.primary_registry.method_by_id.get(method_id)
            if method is None:
                _fail("SELECTION_METHOD_UNKNOWN")
            metric_row = metrics_by_method.loc[method_id]
            if _integer(
                metric_row.get("selected_edge_count"),
                "SELECTION_METRIC_COUNT_INVALID",
            ) != count or _bool(
                metric_row.get("zero_padding_applied"),
                "ZERO_PADDING_SCHEMA_INVALID",
            ):
                _fail("SELECTION_METRIC_IDENTITY_OR_PADDING_INVALID")
            if method.policy_id == "fixed_k":
                requested = int(method.policy_parameters["k"])
                if (
                    count != requested
                    or _integer(
                        metric_row.get("budget_shortfall"),
                        "FIXED_BUDGET_SCHEMA_INVALID",
                    )
                    != 0
                ):
                    _fail("FIXED_BUDGET_OR_ZERO_PADDING_INVALID")


def _adaptive_raw_reference_exact(state: ValidationState) -> None:
    for frames in state.partitions.values():
        rankings = frames["ranking_scores"]
        raw_rows = {
            str(row.ranker_id): row
            for row in rankings.itertuples(index=False)
            if str(row.score_source) == "final_ranker_score"
        }
        selected = frames["selected_edges"].set_index("method_id")
        for ranker in ("shil", "l1"):
            ranked = _json_list(
                raw_rows[ranker].ranked_indices_json,
                "ADAPTIVE_RAW_RANKING_UNAVAILABLE",
            )
            for policy in ("validation_one_se", "cpss_one_se"):
                method_id = f"{ranker}.{policy}"
                row = selected.loc[method_id]
                count = _integer(row["selected_edge_count"], "ADAPTIVE_COUNT_INVALID")
                if count > len(ranked):
                    _fail("ADAPTIVE_RAW_REFERENCE_UNCONSTRUCTIBLE")
                if policy == "validation_one_se":
                    indices = _json_list(
                        row["support_indices_json"], "ADAPTIVE_INDEX_JSON_INVALID"
                    )
                    if indices != ranked[:count]:
                        _fail("VALIDATION_RAW_IDENTITY_MISMATCH")


def _null_schema_exact(state: ValidationState) -> None:
    for unit_id, frames in state.partitions.items():
        atomic = state.canonical_checkpoints[unit_id]["atomic_unit"]
        if atomic.get("scenario") != "S00":
            continue
        dataset = frames["dataset_manifest"]
        if (
            len(dataset) != 1
            or _json_list(
                dataset.iloc[0].get("true_edges_json"), "NULL_TRUTH_SCHEMA_INVALID"
            )
            != []
        ):
            _fail("NULL_TRUTH_SCHEMA_INVALID")
        for row in frames["metrics"].to_dict(orient="records"):
            count = _integer(row["selected_edge_count"], "NULL_COUNT_SCHEMA_INVALID")
            if (
                _integer(row["false_inclusions"], "NULL_COUNT_SCHEMA_INVALID") != count
                or _integer(row["group_false_inclusions"], "NULL_COUNT_SCHEMA_INVALID")
                != count
            ):
                _fail("NULL_FALSE_SELECTION_SCHEMA_MISMATCH")
            expected_ratio = 0.0 if count == 0 else 1.0
            if (
                float(row["false_discovery_proportion"]) != expected_ratio
                or float(row["group_false_discovery_proportion"]) != expected_ratio
            ):
                _fail("NULL_FALSE_SELECTION_RATIO_SCHEMA_MISMATCH")


def _group_schema_exact(state: ValidationState) -> None:
    for unit_id, frames in state.partitions.items():
        atomic = state.canonical_checkpoints[unit_id]["atomic_unit"]
        if atomic.get("scenario") != "S11":
            continue
        dataset = frames["dataset_manifest"]
        if len(dataset) != 1 or not _json_list(
            dataset.iloc[0].get("equivalence_groups_json"), "GROUP_SCHEMA_MISSING"
        ):
            _fail("GROUP_SCHEMA_MISSING")
        required = {
            "group_tp",
            "group_precision",
            "group_recall",
            "group_f1",
            "group_false_inclusions",
            "group_false_discovery_proportion",
        }
        if not required.issubset(frames["metrics"].columns):
            _fail("GROUP_METRIC_SCHEMA_MISSING")


def _structured_comparator_exact(state: ValidationState) -> None:
    structured_sha = None
    if state.structured_registry_document is not None:
        structured_sha = state.structured_registry_document.get("canonical_sha256")
    completed = 0
    for unit_id, frames in state.partitions.items():
        scenario = state.canonical_checkpoints[unit_id]["atomic_unit"].get("scenario")
        status = frames["structured_sensitivity_status"]
        if len(status) != 1:
            _fail("STRUCTURED_STATUS_CARDINALITY_MISMATCH")
        row = status.iloc[0]
        if (
            str(row.get("analysis_role")) != "sensitivity_only"
            or _bool(row.get("primary_estimand"), "STRUCTURED_PRIMARY_FLAG_INVALID")
            or str(row.get("structured_registry_sha256")) != structured_sha
        ):
            _fail("STRUCTURED_ROLE_OR_BINDING_MISMATCH")
        expected_status = "COMPLETED" if scenario == "S13" else "N/A"
        if str(row.get("status")) != expected_status:
            _fail("STRUCTURED_APPLICABILITY_STATUS_MISMATCH")
        metric_rows = frames["structured_sensitivity_metrics"]
        raw_rows = frames["structured_sensitivity_raw"]
        selection_rows = frames["structured_sensitivity_selected_edges"]
        if scenario == "S13":
            completed += 1
            if len(metric_rows) != 1 or len(raw_rows) != 1:
                _fail("STRUCTURED_COMPLETED_ARTIFACT_SET_MISMATCH")
            for frame in (metric_rows, raw_rows, selection_rows):
                if not frame.empty and (
                    set(frame["analysis_role"].astype(str)) != {"sensitivity_only"}
                    or set(frame["primary_estimand"].astype(str).str.lower())
                    != {"false"}
                ):
                    _fail("STRUCTURED_COMPLETED_ROLE_MISMATCH")
            metric = metric_rows.iloc[0]
            required_metric_fields = {
                "candidate_pair_count",
                "selected_edge_count",
                "candidate_average_precision",
                "log_loss",
                "macro_f1",
                "accuracy",
                "support_precision",
                "support_recall",
                "support_f1",
                "false_discovery_proportion",
                "group_precision",
                "group_recall",
                "group_f1",
                "group_false_discovery_proportion",
            }
            if not required_metric_fields.issubset(metric_rows.columns):
                _fail("STRUCTURED_METRIC_SCHEMA_MISMATCH")
            candidate_count = _integer(
                metric["candidate_pair_count"], "STRUCTURED_COUNT_INVALID"
            )
            selected_count = _integer(
                metric["selected_edge_count"], "STRUCTURED_COUNT_INVALID"
            )
            if candidate_count < 1 or not 0 <= selected_count <= candidate_count:
                _fail("STRUCTURED_COUNT_INVALID")
            for field in required_metric_fields - {
                "candidate_pair_count",
                "selected_edge_count",
                "log_loss",
            }:
                value = _finite_number(
                    metric[field], "STRUCTURED_NUMERIC_FINITE_CHECK_FAILED"
                )
                if value < 0 or value > 1:
                    _fail("STRUCTURED_NUMERIC_RANGE_CHECK_FAILED")
            if (
                _finite_number(
                    metric["log_loss"], "STRUCTURED_NUMERIC_FINITE_CHECK_FAILED"
                )
                < 0
            ):
                _fail("STRUCTURED_NUMERIC_RANGE_CHECK_FAILED")
            raw = raw_rows.iloc[0]
            scores = [
                _finite_number(value, "STRUCTURED_RAW_SCORE_NONFINITE")
                for value in _json_list(
                    raw.get("pair_scores_json"), "STRUCTURED_RAW_SCORE_JSON_INVALID"
                )
            ]
            probabilities = [
                _finite_number(value, "STRUCTURED_RAW_PROBABILITY_NONFINITE")
                for value in _json_list(
                    raw.get("probabilities_class_1_json"),
                    "STRUCTURED_RAW_PROBABILITY_JSON_INVALID",
                )
            ]
            if len(scores) != candidate_count or any(
                value < 0 or value > 1 for value in probabilities
            ):
                _fail("STRUCTURED_RAW_SCHEMA_OR_RANGE_INVALID")
        elif not metric_rows.empty or not raw_rows.empty or not selection_rows.empty:
            _fail("STRUCTURED_NA_HAS_SUBSTITUTE_ARTIFACT")
    if completed != EXPECTED_STRUCTURED_COMPLETIONS:
        _fail("STRUCTURED_COMPLETION_COUNT_MISMATCH")
    aggregate_status = _read_csv(
        state.pilot_root / "structured_sensitivity_status.csv",
        "STRUCTURED_AGGREGATE_STATUS_UNREADABLE",
    )
    aggregate_metrics = _read_csv(
        state.pilot_root / "structured_sensitivity_metrics.csv",
        "STRUCTURED_AGGREGATE_METRICS_UNREADABLE",
    )
    aggregate_raw = _read_csv(
        state.pilot_root / "structured_sensitivity_raw.csv",
        "STRUCTURED_AGGREGATE_RAW_UNREADABLE",
    )
    if (
        len(aggregate_status) != 6
        or int((aggregate_status["status"].astype(str) == "COMPLETED").sum())
        != EXPECTED_STRUCTURED_COMPLETIONS
        or int((aggregate_status["status"].astype(str) == "N/A").sum()) != 4
        or len(aggregate_metrics) != EXPECTED_STRUCTURED_COMPLETIONS
        or len(aggregate_raw) != EXPECTED_STRUCTURED_COMPLETIONS
    ):
        _fail("STRUCTURED_AGGREGATE_CARDINALITY_MISMATCH")


def _primary_aggregate_uncontaminated(state: ValidationState) -> None:
    expected_methods = {method.method_id for method in state.primary_registry.methods}
    expected_families = {
        method.score_family_id for method in state.primary_registry.methods
    }
    checks = (
        ("metrics.csv", "method_id", expected_methods, EXPECTED_METHOD_ROWS),
        ("selected_edges.csv", "method_id", expected_methods, EXPECTED_METHOD_ROWS),
        (
            "ranking_metrics.csv",
            "score_family_id",
            expected_families,
            EXPECTED_SCORE_ROWS,
        ),
        (
            "ranking_scores.csv",
            "score_family_id",
            expected_families,
            EXPECTED_SCORE_ROWS,
        ),
    )
    for name, identity, expected, expected_rows in checks:
        frame = _read_csv(state.pilot_root / name, "PRIMARY_AGGREGATE_UNREADABLE")
        if len(frame) != expected_rows or set(frame.get(identity, ())) != expected:
            _fail("PRIMARY_AGGREGATE_IDENTITY_MISMATCH")
        if "comparator_id" in frame.columns or (
            "analysis_role" in frame.columns
            and (frame["analysis_role"].astype(str) == "sensitivity_only").any()
        ):
            _fail("PRIMARY_AGGREGATE_SENSITIVITY_CONTAMINATION")


def _heartbeat_exact(state: ValidationState) -> None:
    required_samples = int(
        state.freeze["durability"]["minimum_advancing_heartbeat_samples"]
    )
    for unit in state.manifest["planned_units"]:
        unit_id = str(unit["unit_id"])
        for checkpoint_path in sorted(
            (state.pilot_root / "units" / unit_id / "attempts").glob(
                "attempt-*/checkpoint.json"
            )
        ):
            history_path = checkpoint_path.parent / "heartbeat_history.jsonl"
            rows = _read_jsonl(history_path, "HEARTBEAT_HISTORY_UNREADABLE")
            if len(rows) < required_samples:
                _fail("HEARTBEAT_SAMPLE_COUNT_BELOW_TWO")
            for row in rows:
                try:
                    durability.validate_heartbeat(row)
                except durability.DurabilityError:
                    _fail("HEARTBEAT_SCHEMA_INVALID")
            times = [durability.parse_utc(str(row["timestamp"])) for row in rows]
            elapsed = [float(row["unit_elapsed_seconds"]) for row in rows]
            if any(right <= left for left, right in zip(times, times[1:])) or any(
                right <= left for left, right in zip(elapsed, elapsed[1:])
            ):
                _fail("HEARTBEAT_NOT_ADVANCING")


def _notification_exact(state: ValidationState) -> None:
    paths = sorted((state.pilot_root / "notifications").glob("invocation-*.jsonl"))
    if not paths:
        _fail("NOTIFICATION_OUTBOX_MISSING")
    terminal_events: list[str] = []
    records: list[dict[str, str]] = []
    for path in paths:
        try:
            lifecycle = freeze_bindings.validate_notification_lifecycle(path)
        except freeze_bindings.FreezeBindingError:
            _fail("NOTIFICATION_LIFECYCLE_INVALID")
        terminal_events.append(str(lifecycle[-1]["event"]))
        records.append({"sha256": durability.sha256_file(path)})
    if terminal_events[-1] != "COMPLETED" or any(
        event not in {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}
        for event in terminal_events
    ):
        _fail("NOTIFICATION_TERMINAL_SEQUENCE_INVALID")
    state.hashes["notification_lifecycle_sha256"] = _sha256_json(records)


def _terminal_output_hashes_exact(state: ValidationState) -> None:
    path = state.pilot_root / "output_hashes.json"
    document = _read_json(path, "OUTPUT_HASH_MANIFEST_UNREADABLE")
    required_names = {
        "RUN_MANIFEST.json",
        "terminal_status.json",
        "run_summary.json",
        "config_resolved.json",
        "method_registry_resolved.json",
        "structured_comparator_registry_resolved.json",
        "metrics.csv",
        "selected_edges.csv",
        "ranking_metrics.csv",
        "ranking_scores.csv",
        "partition_manifest.csv",
        "dataset_manifest.csv",
    }
    if not required_names.issubset(document):
        _fail("OUTPUT_HASH_REQUIRED_SET_MISSING")
    for name, expected in document.items():
        target = _relative_child(state.pilot_root, name, "OUTPUT_HASH_PATH_INVALID")
        _require_file_hash(target, expected, "OUTPUT_ARTIFACT_HASH_MISMATCH")
    state.hashes["output_hash_manifest_sha256"] = durability.sha256_file(path)


PREDICATES: tuple[tuple[str, Callable[[ValidationState], None]], ...] = (
    ("BP01_FREEZE_EXACT", _freeze_exact),
    ("BP02_TERMINAL_MANIFEST_EXACT", _manifest_terminal_exact),
    (
        "BP03_CONFIG_PROTOCOL_REGISTRY_SOURCE_EXACT",
        _config_protocol_registry_source_exact,
    ),
    ("BP04_PREFLIGHT_EXACT", _preflight_exact),
    ("BP05_SIX_UNIT_PLAN_EXACT", _unit_plan_exact),
    ("BP06_FIRST_VALID_ATTEMPT_AND_RETRY_RULE", _attempts_and_artifacts_exact),
    ("BP07_PARTITION_MANIFEST_EXACT", _partition_manifest_exact),
    ("BP08_PRIMARY_METHOD_IDENTITY_14", _primary_identity_exact),
    ("BP09_SCORE_FAMILY_IDENTITY_4", _score_family_identity_exact),
    ("BP10_FINITE_RANGE_AND_SCHEMA", _finite_range_schema_exact),
    (
        "BP11_SELECTION_STRUCTURE_NO_PADDING_OR_DUPLICATES",
        _selection_and_raw_structure_exact,
    ),
    ("BP12_ADAPTIVE_RAW_REFERENCE_CONSTRUCTIBILITY", _adaptive_raw_reference_exact),
    ("BP13_NULL_REPORTING_SCHEMA", _null_schema_exact),
    ("BP14_GROUP_AWARE_REPORTING_SCHEMA", _group_schema_exact),
    ("BP15_S13_COMPARATOR_SEPARATE_AND_EXACT", _structured_comparator_exact),
    ("BP16_PRIMARY_AGGREGATE_UNCONTAMINATED", _primary_aggregate_uncontaminated),
    ("BP17_ADVANCING_HEARTBEATS", _heartbeat_exact),
    ("BP18_NOTIFICATION_LIFECYCLE", _notification_exact),
    ("BP19_TERMINAL_OUTPUT_HASHES", _terminal_output_hashes_exact),
)


def validate_blind_pilot(
    *,
    pilot_root: str | Path,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    candidate_root: str | Path,
    project_root: str | Path,
    preflight_path: str | Path,
) -> dict[str, Any]:
    """Validate without writing and return an allowlisted blind disposition."""

    state = ValidationState(
        pilot_root=Path(pilot_root).resolve(),
        freeze_path=Path(freeze_path).resolve(),
        expected_freeze_sha256=str(expected_freeze_sha256).upper(),
        candidate_root=Path(candidate_root).resolve(),
        project_root=Path(project_root).resolve(),
        preflight_path=Path(preflight_path).resolve(),
    )
    results: list[dict[str, Any]] = []
    for predicate_id, validator in PREDICATES:
        try:
            validator(state)
            result = "PASS"
            reasons: list[str] = []
        except BlindPilotValidationError as error:
            result = "FAIL"
            reasons = [str(error)]
        except Exception:
            result = "FAIL"
            reasons = ["INTERNAL_VALIDATION_ERROR"]
        results.append(
            {
                "predicate_id": predicate_id,
                "result": result,
                "failure_reasons": reasons,
            }
        )
    hashes = {
        key: state.hashes[key] for key in sorted(state.hashes) if key in HASH_FIELDS
    }
    overall = "PASS" if all(item["result"] == "PASS" for item in results) else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "validator_id": VALIDATOR_ID,
        "validation_mode": "READ_ONLY_BLIND_STRUCTURAL_NO_SCIENTIFIC_INFERENCE",
        "overall_result": overall,
        "predicates": results,
        "non_scientific_counts": {
            "planned_units": 6,
            "completed_units": int(state.completed_units),
            "expected_schema_rows": {
                "primary_method_rows": EXPECTED_METHOD_ROWS,
                "primary_selection_rows": EXPECTED_METHOD_ROWS,
                "ranking_metric_rows": EXPECTED_SCORE_ROWS,
                "ranking_score_rows": EXPECTED_SCORE_ROWS,
                "sensitivity_status_rows": 6,
                "structured_completed_metric_rows": EXPECTED_STRUCTURED_COMPLETIONS,
                "structured_completed_raw_rows": EXPECTED_STRUCTURED_COMPLETIONS,
            },
        },
        "binding_hashes": hashes,
        "statuses": {
            "freeze": "BOUND" if "freeze_sha256" in hashes else "INVALID",
            "terminal": (
                "COMPLETED"
                if state.terminal and state.terminal.get("status") == "completed"
                else "INVALID"
            ),
            "preflight": "PASS" if "preflight_sha256" in hashes else "INVALID",
            "source": "BOUND" if "source_bundle_sha256" in hashes else "INVALID",
            "notification_lifecycle": (
                "PASS" if "notification_lifecycle_sha256" in hashes else "INVALID"
            ),
            "scientific_inference": "PROHIBITED",
        },
    }


def write_blind_pilot_artifacts(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    sealed_pilot_root: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir).resolve()
    sealed = Path(sealed_pilot_root).resolve()
    try:
        output.relative_to(sealed)
    except ValueError:
        pass
    else:
        raise ValueError("Validator output must stay outside the sealed pilot output")
    output.mkdir(parents=True, exist_ok=True)
    predicate_path = output / "BLIND_PILOT_PREDICATES.json"
    receipt_path = output / "BLIND_PILOT_VALIDATOR_RECEIPT.json"
    if predicate_path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite blind-pilot validator artifacts")
    durability.atomic_write_json(predicate_path, dict(result))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "validator_id": VALIDATOR_ID,
        "validation_mode": "READ_ONLY_BLIND_STRUCTURAL_NO_SCIENTIFIC_INFERENCE",
        "overall_result": result["overall_result"],
        "predicate_artifact_sha256": durability.sha256_file(predicate_path),
        "validator_source_sha256": durability.sha256_file(Path(__file__)),
        "scientific_inference": "PROHIBITED",
    }
    durability.atomic_write_json(receipt_path, receipt)
    return predicate_path, receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot_root = args.pilot_output.resolve()
    output_dir = args.output_dir.resolve()
    result = validate_blind_pilot(
        pilot_root=pilot_root,
        freeze_path=args.freeze,
        expected_freeze_sha256=args.expected_freeze_sha256,
        candidate_root=args.candidate_root,
        project_root=args.project_root,
        preflight_path=args.preflight,
    )
    write_blind_pilot_artifacts(
        result, output_dir=output_dir, sealed_pilot_root=pilot_root
    )
    return 0 if result["overall_result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
