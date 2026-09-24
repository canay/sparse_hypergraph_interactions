"""Fail-closed engineering bindings required before a candidate freeze.

The module never runs a scientific model and never sends a remote message.  It
captures lead-supplied timing evidence plus a live resource snapshot, polls a
predecessor's durable terminal status, records a local notification outbox, and
creates hash-bound decision/SVG artifacts from already saved analysis files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import psutil

import durability


SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"completed", "failed", "timed_out", "cancelled"})
NOTIFICATION_EVENTS = frozenset(
    {"STARTED", "MILESTONE", "COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}
)
NOTIFICATION_TERMINAL_EVENTS = frozenset(
    {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}
)


class FreezeBindingError(ValueError):
    """Raised when freeze evidence is absent, inconsistent, or unbound."""


def _mapping(document: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise FreezeBindingError(f"{context} must be an object")
    return document


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), context)
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeBindingError(f"{context} is unreadable: {error}") from error


def _positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeBindingError(f"{context} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise FreezeBindingError(f"{context} must be finite and positive")
    return value


def _artifact_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeBindingError(f"bound artifact does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": durability.sha256_file(path),
    }


def capture_timing_resource_preflight(
    *,
    timing_evidence_path: str | Path,
    output_path: str | Path,
    resource_path: str | Path,
    planned_unit_count: int,
    required_disk_bytes: int,
    required_ram_bytes: int,
    safety_margin_fraction: float,
    max_parallel_units: int = 1,
    observed_disk_free_bytes: int | None = None,
    observed_ram_available_bytes: int | None = None,
    observed_cpu_count: int | None = None,
) -> Mapping[str, Any]:
    """Bind lead-measured unit timings and capture resources; run no science."""

    timing_path = Path(timing_evidence_path).resolve()
    timing = _read_json(timing_path, "timing evidence")
    required_timing = {
        "schema_version",
        "measurement_id",
        "measured_at_utc",
        "run_class",
        "unit_durations_seconds",
        "source_run_manifest_path",
        "source_run_manifest_sha256",
    }
    if set(timing) != required_timing or timing["schema_version"] != SCHEMA_VERSION:
        raise FreezeBindingError("timing evidence schema mismatch")
    durability.validate_identifier(timing["measurement_id"], "measurement_id")
    durability.parse_utc(str(timing["measured_at_utc"]))
    durations_raw = timing["unit_durations_seconds"]
    if not isinstance(durations_raw, list) or len(durations_raw) < 2:
        raise FreezeBindingError("timing evidence requires at least two measured units")
    durations = [_positive_number(value, "unit duration") for value in durations_raw]
    source_manifest_sha = str(timing["source_run_manifest_sha256"])
    if (
        len(source_manifest_sha) != 64
        or source_manifest_sha != source_manifest_sha.upper()
    ):
        raise FreezeBindingError("source run manifest SHA-256 must be uppercase")
    try:
        int(source_manifest_sha, 16)
    except ValueError as error:
        raise FreezeBindingError("source run manifest SHA-256 is invalid") from error
    source_manifest_path = Path(str(timing["source_run_manifest_path"]))
    if not source_manifest_path.is_absolute():
        source_manifest_path = timing_path.parent / source_manifest_path
    source_manifest_path = source_manifest_path.resolve()
    source_manifest_binding = _artifact_binding(source_manifest_path)
    if source_manifest_binding["sha256"] != source_manifest_sha:
        raise FreezeBindingError(
            "source run manifest hash differs from timing evidence"
        )
    if not isinstance(planned_unit_count, int) or planned_unit_count < 1:
        raise FreezeBindingError("planned_unit_count must be positive")
    if (
        isinstance(max_parallel_units, bool)
        or not isinstance(max_parallel_units, int)
        or not 1 <= max_parallel_units <= 3
    ):
        raise FreezeBindingError("max_parallel_units must be an integer in [1,3]")
    if not (0 <= float(safety_margin_fraction) <= 2):
        raise FreezeBindingError("safety margin fraction must be in [0, 2]")
    required_disk = int(_positive_number(required_disk_bytes, "required disk bytes"))
    required_ram = int(_positive_number(required_ram_bytes, "required RAM bytes"))
    resource_root = Path(resource_path).resolve()
    disk_free = (
        shutil.disk_usage(resource_root).free
        if observed_disk_free_bytes is None
        else int(observed_disk_free_bytes)
    )
    ram_available = (
        psutil.virtual_memory().available
        if observed_ram_available_bytes is None
        else int(observed_ram_available_bytes)
    )
    cpu_count = (
        os.cpu_count() if observed_cpu_count is None else int(observed_cpu_count)
    )
    if disk_free < 0 or ram_available < 0 or not cpu_count or cpu_count < 1:
        raise FreezeBindingError("observed resource values are invalid")
    ordered = sorted(durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    p95 = ordered[p95_index]
    margin = 1.0 + float(safety_margin_fraction)
    estimated_batch_count = math.ceil(planned_unit_count / max_parallel_units)
    timing_summary = {
        "measurement_id": timing["measurement_id"],
        "measured_at_utc": timing["measured_at_utc"],
        "run_class": timing["run_class"],
        "source_run_manifest_sha256": source_manifest_sha,
        "measured_unit_count": len(durations),
        "median_unit_seconds": statistics.median(durations),
        "p95_unit_seconds": p95,
        "safety_margin_fraction": float(safety_margin_fraction),
        "planned_unit_count": planned_unit_count,
        "estimated_whole_run_seconds": p95 * margin * estimated_batch_count,
    }
    capture_kind = "lead_measured_timing_plus_live_resource_snapshot"
    if max_parallel_units > 1:
        capture_kind = (
            "lead_measured_timing_plus_live_resource_snapshot_parallel_batches"
        )
        timing_summary.update(
            {
                "max_parallel_units": max_parallel_units,
                "estimated_batch_count": estimated_batch_count,
            }
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "capture_kind": capture_kind,
        "captured_at_utc": durability.utc_now(),
        "timing_evidence": _artifact_binding(timing_path),
        "source_run_manifest": source_manifest_binding,
        "timing_summary": timing_summary,
        "resources": {
            "resource_path": str(resource_root),
            "required_disk_bytes": required_disk,
            "observed_disk_free_bytes": disk_free,
            "required_ram_bytes": required_ram,
            "observed_ram_available_bytes": ram_available,
            "observed_cpu_count": cpu_count,
        },
        "checks": {
            "disk": "pass" if disk_free >= required_disk else "fail",
            "ram": "pass" if ram_available >= required_ram else "fail",
            "timing_sample": "pass",
        },
    }
    document["overall_status"] = (
        "pass" if set(document["checks"].values()) == {"pass"} else "fail"
    )
    durability.atomic_write_json(output_path, document)
    return document


def validate_timing_resource_preflight(
    path: str | Path, *, require_pass: bool = True
) -> Mapping[str, Any]:
    document = _read_json(Path(path), "timing/resource preflight")
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
    if set(document) != required or document["schema_version"] != SCHEMA_VERSION:
        raise FreezeBindingError("timing/resource preflight schema mismatch")
    durability.parse_utc(str(document["captured_at_utc"]))
    evidence = _mapping(document["timing_evidence"], "timing evidence binding")
    evidence_path = Path(str(evidence.get("path", "")))
    if _artifact_binding(evidence_path) != dict(evidence):
        raise FreezeBindingError("timing evidence binding differs from live file")
    source_manifest = _mapping(
        document["source_run_manifest"], "source run manifest binding"
    )
    if _artifact_binding(Path(str(source_manifest.get("path", "")))) != dict(
        source_manifest
    ):
        raise FreezeBindingError("source run manifest binding differs from live file")
    timing_document = _read_json(evidence_path, "bound timing evidence")
    if timing_document.get("source_run_manifest_sha256") != source_manifest.get(
        "sha256"
    ):
        raise FreezeBindingError("timing evidence and source manifest bindings differ")
    durations = [
        _positive_number(value, "bound unit duration")
        for value in timing_document.get("unit_durations_seconds", [])
    ]
    if len(durations) < 2:
        raise FreezeBindingError("bound timing sample is too small")
    timing_summary = _mapping(document["timing_summary"], "timing summary")
    summary_fields = {
        "measurement_id",
        "measured_at_utc",
        "run_class",
        "source_run_manifest_sha256",
        "measured_unit_count",
        "median_unit_seconds",
        "p95_unit_seconds",
        "safety_margin_fraction",
        "planned_unit_count",
        "estimated_whole_run_seconds",
    }
    parallel_capture = (
        document["capture_kind"]
        == "lead_measured_timing_plus_live_resource_snapshot_parallel_batches"
    )
    if parallel_capture:
        summary_fields |= {"max_parallel_units", "estimated_batch_count"}
    elif document["capture_kind"] != (
        "lead_measured_timing_plus_live_resource_snapshot"
    ):
        raise FreezeBindingError("unknown timing/resource capture kind")
    if set(timing_summary) != summary_fields:
        raise FreezeBindingError("timing summary schema mismatch")
    ordered = sorted(durations)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    max_parallel_units = (
        int(timing_summary["max_parallel_units"]) if parallel_capture else 1
    )
    if not 1 <= max_parallel_units <= 3:
        raise FreezeBindingError("timing summary parallel worker count is invalid")
    estimated_batch_count = math.ceil(
        int(timing_summary["planned_unit_count"]) / max_parallel_units
    )
    expected_summary = {
        "measurement_id": timing_document["measurement_id"],
        "measured_at_utc": timing_document["measured_at_utc"],
        "run_class": timing_document["run_class"],
        "source_run_manifest_sha256": timing_document["source_run_manifest_sha256"],
        "measured_unit_count": len(durations),
        "median_unit_seconds": statistics.median(durations),
        "p95_unit_seconds": p95,
        "safety_margin_fraction": timing_summary["safety_margin_fraction"],
        "planned_unit_count": timing_summary["planned_unit_count"],
        "estimated_whole_run_seconds": p95
        * (1.0 + float(timing_summary["safety_margin_fraction"]))
        * estimated_batch_count,
    }
    if parallel_capture:
        expected_summary.update(
            {
                "max_parallel_units": max_parallel_units,
                "estimated_batch_count": estimated_batch_count,
            }
        )
    if dict(timing_summary) != expected_summary:
        raise FreezeBindingError("timing summary differs from bound measurements")
    resources = _require_resource_snapshot(document["resources"])
    checks = _mapping(document["checks"], "resource checks")
    if set(checks) != {"disk", "ram", "timing_sample"} or any(
        value not in {"pass", "fail"} for value in checks.values()
    ):
        raise FreezeBindingError("resource check schema mismatch")
    expected_checks = {
        "disk": (
            "pass"
            if resources["observed_disk_free_bytes"] >= resources["required_disk_bytes"]
            else "fail"
        ),
        "ram": (
            "pass"
            if resources["observed_ram_available_bytes"]
            >= resources["required_ram_bytes"]
            else "fail"
        ),
        "timing_sample": "pass",
    }
    if dict(checks) != expected_checks:
        raise FreezeBindingError("resource checks differ from captured values")
    expected = "pass" if set(checks.values()) == {"pass"} else "fail"
    if document["overall_status"] != expected:
        raise FreezeBindingError("preflight overall status is inconsistent")
    if require_pass and expected != "pass":
        raise FreezeBindingError("timing/resource preflight did not pass")
    return document


def _require_resource_snapshot(value: Any) -> Mapping[str, Any]:
    resources = _mapping(value, "resource snapshot")
    fields = {
        "resource_path",
        "required_disk_bytes",
        "observed_disk_free_bytes",
        "required_ram_bytes",
        "observed_ram_available_bytes",
        "observed_cpu_count",
    }
    if set(resources) != fields:
        raise FreezeBindingError("resource snapshot schema mismatch")
    for field in fields - {"resource_path"}:
        value = resources[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FreezeBindingError(f"resource snapshot value is invalid: {field}")
    if resources["required_disk_bytes"] < 1 or resources["required_ram_bytes"] < 1:
        raise FreezeBindingError("required resource values must be positive")
    if resources["observed_cpu_count"] < 1:
        raise FreezeBindingError("observed CPU count must be positive")
    return resources


@dataclass(frozen=True)
class PredecessorOutcome:
    status: str
    terminal_document: Mapping[str, Any] | None
    polls: int


def poll_predecessor_terminal_status(
    path: str | Path,
    *,
    expected_run_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PredecessorOutcome:
    """Bounded poll: success only on completed; failures and unknown are distinct."""

    timeout = _positive_number(timeout_seconds, "predecessor timeout")
    interval = _positive_number(poll_interval_seconds, "predecessor poll interval")
    if interval > timeout:
        raise FreezeBindingError("poll interval cannot exceed timeout")
    terminal_path = Path(path)
    started = monotonic()
    polls = 0
    while True:
        polls += 1
        if terminal_path.is_file():
            document = _read_json(terminal_path, "predecessor terminal status")
            if document.get("schema_version") != SCHEMA_VERSION:
                raise FreezeBindingError("predecessor terminal schema mismatch")
            if document.get("run_id") != expected_run_id:
                raise FreezeBindingError("predecessor run identity mismatch")
            status = str(document.get("status"))
            if status == "completed":
                if document.get("exit_code") != 0:
                    raise FreezeBindingError("completed predecessor has nonzero exit")
                return PredecessorOutcome(status, document, polls)
            if status in TERMINAL_STATUSES:
                return PredecessorOutcome(status, document, polls)
            if status != "running":
                raise FreezeBindingError("predecessor terminal status is unknown")
        if monotonic() - started >= timeout:
            return PredecessorOutcome("unknown_timeout", None, polls)
        sleep(interval)


def require_predecessor_completed(*args, **kwargs) -> PredecessorOutcome:
    """Fail closed unless the bounded predecessor poll proves completion."""

    outcome = poll_predecessor_terminal_status(*args, **kwargs)
    if outcome.status != "completed":
        raise FreezeBindingError(
            f"predecessor did not complete successfully: {outcome.status}"
        )
    return outcome


def append_notification_event(
    outbox_path: str | Path,
    *,
    run_id: str,
    event: str,
    message: str,
    completed_units: int | None = None,
    planned_units: int | None = None,
) -> Mapping[str, Any]:
    """Persist a local delivery outbox record; this function performs no send."""

    durability.validate_identifier(run_id, "notification run_id")
    if event not in NOTIFICATION_EVENTS:
        raise FreezeBindingError("unknown notification lifecycle event")
    path = Path(outbox_path)
    existing = validate_notification_lifecycle(path) if path.exists() else []
    if not existing and event != "STARTED":
        raise FreezeBindingError("notification lifecycle must begin with STARTED")
    if existing and existing[-1]["event"] in NOTIFICATION_TERMINAL_EVENTS:
        raise FreezeBindingError("notification lifecycle is already terminal")
    if existing and any(row["run_id"] != run_id for row in existing):
        raise FreezeBindingError("notification outbox mixes run identities")
    if event == "MILESTONE":
        if completed_units is None or planned_units is None:
            raise FreezeBindingError("MILESTONE requires completed/planned units")
        if not (0 <= completed_units <= planned_units):
            raise FreezeBindingError("MILESTONE unit counts are inconsistent")
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": f"event-{len(existing) + 1:04d}",
        "run_id": run_id,
        "event": event,
        "timestamp_utc": durability.utc_now(),
        "message": str(message),
        "completed_units": completed_units,
        "planned_units": planned_units,
        "delivery_status": "pending_external_delivery",
    }
    durability.append_jsonl_durable(path, record)
    return record


def validate_notification_lifecycle(path: str | Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        try:
            record = _mapping(json.loads(line), f"notification record {index}")
        except json.JSONDecodeError as error:
            raise FreezeBindingError(
                "notification outbox contains invalid JSON"
            ) from error
        expected_fields = {
            "schema_version",
            "event_id",
            "run_id",
            "event",
            "timestamp_utc",
            "message",
            "completed_units",
            "planned_units",
            "delivery_status",
        }
        if set(record) != expected_fields or record["schema_version"] != SCHEMA_VERSION:
            raise FreezeBindingError("notification record schema mismatch")
        if record["event_id"] != f"event-{index + 1:04d}":
            raise FreezeBindingError("notification event sequence is not contiguous")
        durability.parse_utc(str(record["timestamp_utc"]))
        if record["event"] not in NOTIFICATION_EVENTS:
            raise FreezeBindingError("notification event is unknown")
        records.append(record)
    if records and records[0]["event"] != "STARTED":
        raise FreezeBindingError("notification lifecycle does not begin with STARTED")
    terminal_indices = [
        index
        for index, record in enumerate(records)
        if record["event"] in NOTIFICATION_TERMINAL_EVENTS
    ]
    if terminal_indices and terminal_indices != [len(records) - 1]:
        raise FreezeBindingError("notification lifecycle has events after terminal")
    return records


def write_decision_artifact(
    *,
    criteria: Sequence[Mapping[str, Any]],
    bound_inputs: Sequence[str | Path],
    output_path: str | Path,
    decision_id: str,
) -> Mapping[str, Any]:
    durability.validate_identifier(decision_id, "decision_id")
    if not criteria:
        raise FreezeBindingError("decision artifact requires criteria")
    normalized: list[dict[str, Any]] = []
    operators = {
        "ge": lambda x, y: x >= y,
        "gt": lambda x, y: x > y,
        "le": lambda x, y: x <= y,
        "lt": lambda x, y: x < y,
    }
    for raw in criteria:
        criterion = dict(raw)
        required = {
            "criterion_id",
            "metric",
            "value",
            "operator",
            "threshold",
            "result",
            "discriminator_statistics",
        }
        if set(criterion) != required:
            raise FreezeBindingError("decision criterion schema mismatch")
        durability.validate_identifier(criterion["criterion_id"], "criterion_id")
        value = float(criterion["value"])
        threshold = float(criterion["threshold"])
        operator = str(criterion["operator"])
        if operator not in operators or not all(map(math.isfinite, (value, threshold))):
            raise FreezeBindingError("decision criterion comparison is invalid")
        expected = "pass" if operators[operator](value, threshold) else "fail"
        if criterion["result"] != expected:
            raise FreezeBindingError("decision criterion result is inconsistent")
        if not isinstance(criterion["discriminator_statistics"], Mapping):
            raise FreezeBindingError("discriminator statistics must be an object")
        normalized.append(criterion)
    inputs = [_artifact_binding(Path(path).resolve()) for path in bound_inputs]
    if not inputs:
        raise FreezeBindingError("decision artifact requires bound inputs")
    document = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_id,
        "created_at_utc": durability.utc_now(),
        "inputs": inputs,
        "criteria": normalized,
        "overall_result": (
            "pass" if all(row["result"] == "pass" for row in normalized) else "fail"
        ),
        "evidence_class": "postprocessed_saved_outputs_no_model_execution",
    }
    durability.atomic_write_json(output_path, document)
    return document


def validate_decision_artifact(path: str | Path) -> Mapping[str, Any]:
    document = _read_json(Path(path), "decision artifact")
    required = {
        "schema_version",
        "decision_id",
        "created_at_utc",
        "inputs",
        "criteria",
        "overall_result",
        "evidence_class",
    }
    if set(document) != required or document["schema_version"] != SCHEMA_VERSION:
        raise FreezeBindingError("decision artifact schema mismatch")
    durability.validate_identifier(document["decision_id"], "decision_id")
    durability.parse_utc(str(document.get("created_at_utc")))
    inputs = document["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise FreezeBindingError("decision artifact requires bound inputs")
    for record in inputs:
        bound = _mapping(record, "decision input binding")
        if _artifact_binding(Path(str(bound.get("path", "")))) != dict(bound):
            raise FreezeBindingError("decision input binding differs from live file")
    criteria = document["criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise FreezeBindingError("decision artifact requires criteria")
    operators = {
        "ge": lambda x, y: x >= y,
        "gt": lambda x, y: x > y,
        "le": lambda x, y: x <= y,
        "lt": lambda x, y: x < y,
    }
    for criterion in criteria:
        row = _mapping(criterion, "decision criterion")
        if set(row) != {
            "criterion_id",
            "metric",
            "value",
            "operator",
            "threshold",
            "result",
            "discriminator_statistics",
        }:
            raise FreezeBindingError("decision criterion schema mismatch")
        value, threshold = float(row["value"]), float(row["threshold"])
        operator = str(row["operator"])
        if operator not in operators or not all(map(math.isfinite, (value, threshold))):
            raise FreezeBindingError("decision criterion comparison is invalid")
        expected = "pass" if operators[operator](value, threshold) else "fail"
        if row["result"] != expected:
            raise FreezeBindingError("decision criterion result is inconsistent")
        if not isinstance(row["discriminator_statistics"], Mapping):
            raise FreezeBindingError("discriminator statistics must be an object")
    expected_overall = (
        "pass" if all(row["result"] == "pass" for row in criteria) else "fail"
    )
    if document["overall_result"] != expected_overall:
        raise FreezeBindingError("decision overall result is inconsistent")
    return document


def write_svg_bar_plot(
    *,
    input_csv: str | Path,
    output_svg: str | Path,
    manifest_path: str | Path,
    label_column: str,
    value_column: str,
    title: str,
) -> Mapping[str, Any]:
    """Render a deterministic dependency-free SVG from one saved CSV."""

    input_path = Path(input_csv).resolve()
    frame = pd.read_csv(input_path)
    if label_column not in frame or value_column not in frame or frame.empty:
        raise FreezeBindingError("plot input columns are missing or empty")
    labels = frame[label_column].astype(str).tolist()
    values = frame[value_column].astype(float).tolist()
    if not all(math.isfinite(value) for value in values):
        raise FreezeBindingError("plot values must be finite")
    maximum = max([abs(value) for value in values] + [1.0])
    width, left, row_height = 900, 260, 34
    height = 80 + row_height * len(values)
    escape = (
        lambda value: str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="18">{escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 55 + index * row_height
        bar_width = int((width - left - 40) * abs(value) / maximum)
        lines.extend(
            [
                f'<text x="20" y="{y + 16}" font-family="sans-serif" font-size="12">{escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width}" height="18" fill="#3568a8"/>',
                f'<text x="{left + bar_width + 6}" y="{y + 15}" font-family="sans-serif" font-size="11">{value:.6g}</text>',
            ]
        )
    lines.append("</svg>")
    durability.atomic_write_bytes(output_svg, ("\n".join(lines) + "\n").encode("utf-8"))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": durability.utc_now(),
        "input": _artifact_binding(input_path),
        "output": _artifact_binding(Path(output_svg).resolve()),
        "render_contract": {
            "renderer": "freeze_bindings.write_svg_bar_plot",
            "label_column": label_column,
            "value_column": value_column,
            "title": title,
        },
        "evidence_class": "postprocessed_saved_csv_no_model_execution",
    }
    durability.atomic_write_json(manifest_path, manifest)
    return manifest


def validate_plot_artifact(path: str | Path) -> Mapping[str, Any]:
    document = _read_json(Path(path), "plot artifact manifest")
    required = {
        "schema_version",
        "created_at_utc",
        "input",
        "output",
        "render_contract",
        "evidence_class",
    }
    if set(document) != required or document["schema_version"] != SCHEMA_VERSION:
        raise FreezeBindingError("plot artifact manifest schema mismatch")
    durability.parse_utc(str(document["created_at_utc"]))
    for name in ("input", "output"):
        record = _mapping(document[name], f"plot {name} binding")
        if _artifact_binding(Path(str(record.get("path", "")))) != dict(record):
            raise FreezeBindingError(f"plot {name} binding differs from live file")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture or validate candidate freeze engineering artifacts"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("capture-preflight")
    preflight.add_argument("--timing-evidence", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--resource-path", type=Path, required=True)
    preflight.add_argument("--planned-units", type=int, required=True)
    preflight.add_argument("--max-parallel-units", type=int, default=1)
    preflight.add_argument("--required-disk-bytes", type=int, required=True)
    preflight.add_argument("--required-ram-bytes", type=int, required=True)
    preflight.add_argument("--safety-margin-fraction", type=float, required=True)
    validate_preflight = commands.add_parser("validate-preflight")
    validate_preflight.add_argument("--artifact", type=Path, required=True)
    predecessor = commands.add_parser("poll-predecessor")
    predecessor.add_argument("--status", type=Path, required=True)
    predecessor.add_argument("--run-id", required=True)
    predecessor.add_argument("--timeout-seconds", type=float, required=True)
    predecessor.add_argument("--poll-interval-seconds", type=float, required=True)
    decision = commands.add_parser("write-decision")
    decision.add_argument("--criteria", type=Path, required=True)
    decision.add_argument("--input", type=Path, action="append", required=True)
    decision.add_argument("--output", type=Path, required=True)
    decision.add_argument("--decision-id", required=True)
    validate_decision = commands.add_parser("validate-decision")
    validate_decision.add_argument("--artifact", type=Path, required=True)
    plot = commands.add_parser("write-plot")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--manifest", type=Path, required=True)
    plot.add_argument("--label-column", required=True)
    plot.add_argument("--value-column", required=True)
    plot.add_argument("--title", required=True)
    validate_plot = commands.add_parser("validate-plot")
    validate_plot.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture-preflight":
        result = capture_timing_resource_preflight(
            timing_evidence_path=args.timing_evidence,
            output_path=args.output,
            resource_path=args.resource_path,
            planned_unit_count=args.planned_units,
            max_parallel_units=args.max_parallel_units,
            required_disk_bytes=args.required_disk_bytes,
            required_ram_bytes=args.required_ram_bytes,
            safety_margin_fraction=args.safety_margin_fraction,
        )
    elif args.command == "validate-preflight":
        result = validate_timing_resource_preflight(args.artifact)
    elif args.command == "poll-predecessor":
        result = require_predecessor_completed(
            args.status,
            expected_run_id=args.run_id,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        ).terminal_document
    elif args.command == "write-decision":
        criteria = json.loads(args.criteria.read_text(encoding="utf-8"))
        if not isinstance(criteria, list):
            raise FreezeBindingError("criteria file must contain a JSON list")
        result = write_decision_artifact(
            criteria=criteria,
            bound_inputs=args.input,
            output_path=args.output,
            decision_id=args.decision_id,
        )
    elif args.command == "validate-decision":
        result = validate_decision_artifact(args.artifact)
    elif args.command == "write-plot":
        result = write_svg_bar_plot(
            input_csv=args.input,
            output_svg=args.output,
            manifest_path=args.manifest,
            label_column=args.label_column,
            value_column=args.value_column,
            title=args.title,
        )
    else:
        result = validate_plot_artifact(args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
