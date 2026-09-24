"""Fail-closed durability primitives for the Track-A candidate.

This module is engineering infrastructure only.  It does not import the
scientific runner and cannot authorize candidate execution.  The public
functions implement the PR4-DURABILITY contracts for atomic checkpoints,
validated resume, source/config/registry binding, machine-readable heartbeat
snapshots, process-tree observation, conservative stall classification, and
preflight manifest validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil


SCHEMA_VERSION = 1
ALLOWED_UNIT_STATUSES = frozenset(
    {
        "pending",
        "running",
        "completed",
        "failed",
        "timed_out",
        "cancelled",
        "skipped_validated",
    }
)
TERMINAL_UNIT_STATUSES = frozenset(
    {"completed", "failed", "timed_out", "cancelled", "skipped_validated"}
)


class DurabilityError(ValueError):
    """Raised when durability evidence is missing, inconsistent, or unsafe."""


def validate_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise DurabilityError(
            f"{context} must use only ASCII letters, digits, dot, underscore, or hyphen"
        )
    return value


def utc_now() -> str:
    """Return an RFC-3339 UTC timestamp with observable subsecond progress."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DurabilityError("timestamp must be an RFC-3339 UTC string ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise DurabilityError(f"invalid UTC timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DurabilityError("timestamp must resolve to UTC")
    return parsed


def canonical_json_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DurabilityError(f"{context} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise DurabilityError(f"{context} is not hexadecimal") from error
    if value != value.upper():
        raise DurabilityError(f"{context} must use uppercase hexadecimal")
    return value


def _relative_child(root: Path, relative_path: Any, context: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise DurabilityError(f"{context} must be a non-empty relative path")
    raw = Path(relative_path)
    if raw.is_absolute():
        raise DurabilityError(f"{context} must be relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise DurabilityError(f"{context} escapes its declared root") from error
    return resolved


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write bytes with flush/fsync and same-directory atomic replacement."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        # Directory fsync is not supported on every Windows/Python combination.
        # Replacement remains atomic on the same volume; file fsync is mandatory.
        if os.name != "nt":
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, document: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(document))


def append_jsonl_durable(path: str | Path, document: Any) -> None:
    """Append one canonical JSONL record and fsync it before returning."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(canonical_json_bytes(document))
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class BindingSet:
    protocol_sha256: str
    config_sha256: str
    registry_sha256: str
    source_bundle_sha256: str
    structured_registry_sha256: str | None = None
    structured_registry_canonical_sha256: str | None = None
    scientific_freeze_sha256: str | None = None
    run_role: str | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.protocol_sha256, "protocol_sha256")
        _validate_sha256(self.config_sha256, "config_sha256")
        _validate_sha256(self.registry_sha256, "registry_sha256")
        _validate_sha256(self.source_bundle_sha256, "source_bundle_sha256")
        for value, label in (
            (self.structured_registry_sha256, "structured_registry_sha256"),
            (
                self.structured_registry_canonical_sha256,
                "structured_registry_canonical_sha256",
            ),
            (self.scientific_freeze_sha256, "scientific_freeze_sha256"),
        ):
            if value is not None:
                _validate_sha256(value, label)
        if self.run_role is not None:
            _require_nonempty_string(self.run_role, "run_role")

    def as_dict(self) -> dict[str, str]:
        result = {
            "protocol_sha256": self.protocol_sha256,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
        }
        optional = {
            "structured_registry_sha256": self.structured_registry_sha256,
            "structured_registry_canonical_sha256": (
                self.structured_registry_canonical_sha256
            ),
            "scientific_freeze_sha256": self.scientific_freeze_sha256,
            "run_role": self.run_role,
        }
        result.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return result


def build_source_binding(
    candidate_root: str | Path, relative_paths: Iterable[str]
) -> dict[str, Any]:
    """Create a deterministic source inventory without writing it."""

    root = Path(candidate_root).resolve()
    paths = tuple(relative_paths)
    if not paths or len(paths) != len(set(paths)):
        raise DurabilityError("source binding requires unique non-empty paths")
    records: list[dict[str, Any]] = []
    for relative in sorted(paths):
        resolved = _relative_child(root, relative, "source path")
        if not resolved.is_file():
            raise DurabilityError(f"bound source does not exist: {relative}")
        records.append(
            {
                "path": Path(relative).as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "files": records,
        "source_bundle_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def validate_source_binding(
    candidate_root: str | Path, binding: Mapping[str, Any]
) -> str:
    if not isinstance(binding, Mapping):
        raise DurabilityError("source binding must be an object")
    files = binding.get("files")
    if not isinstance(files, list) or not files:
        raise DurabilityError("source binding must contain at least one file")
    paths: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise DurabilityError(f"source binding file {index} must be an object")
        if set(record) != {"path", "bytes", "sha256"}:
            raise DurabilityError(f"source binding file {index} has a field mismatch")
        paths.append(str(record["path"]))
    rebuilt = build_source_binding(candidate_root, paths)
    if files != rebuilt["files"]:
        raise DurabilityError("source inventory differs from live candidate files")
    declared = _validate_sha256(
        binding.get("source_bundle_sha256"), "source_bundle_sha256"
    )
    if declared != rebuilt["source_bundle_sha256"]:
        raise DurabilityError("source bundle SHA-256 does not match its inventory")
    return declared


def artifact_record(
    run_root: str | Path, relative_path: str, kind: str
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    resolved = _relative_child(root, relative_path, "artifact path")
    if not resolved.is_file():
        raise DurabilityError(f"artifact does not exist: {relative_path}")
    if not isinstance(kind, str) or not kind.strip():
        raise DurabilityError("artifact kind must be a non-empty string")
    return {
        "path": Path(relative_path).as_posix(),
        "kind": kind,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_artifact(run_root: Path, record: Mapping[str, Any]) -> None:
    expected_fields = {"path", "kind", "bytes", "sha256"}
    if not isinstance(record, Mapping) or set(record) != expected_fields:
        raise DurabilityError("artifact record field mismatch")
    resolved = _relative_child(run_root, record["path"], "artifact path")
    if not resolved.is_file():
        raise DurabilityError(f"checkpoint artifact is missing: {record['path']}")
    if isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int):
        raise DurabilityError("artifact bytes must be an integer")
    if resolved.stat().st_size != record["bytes"]:
        raise DurabilityError(f"artifact size mismatch: {record['path']}")
    declared = _validate_sha256(record["sha256"], "artifact sha256")
    if sha256_file(resolved) != declared:
        raise DurabilityError(f"artifact hash mismatch: {record['path']}")


def make_checkpoint(
    *,
    run_id: str,
    unit_id: str,
    attempt_id: str,
    status: str,
    atomic_unit: Mapping[str, Any],
    bindings: BindingSet,
    started_at_utc: str,
    ended_at_utc: str | None,
    duration_seconds: float | None,
    pid: int,
    worker_id: str,
    exit_code: int | None,
    signal: str | None,
    artifacts: Sequence[Mapping[str, Any]],
    error: str | None = None,
) -> dict[str, Any]:
    validate_identifier(run_id, "run_id")
    validate_identifier(unit_id, "unit_id")
    validate_identifier(attempt_id, "attempt_id")
    if status not in ALLOWED_UNIT_STATUSES:
        raise DurabilityError(f"unknown unit status: {status!r}")
    if status == "completed" and (exit_code != 0 or not artifacts):
        raise DurabilityError("completed checkpoint requires exit_code 0 and artifacts")
    if status in TERMINAL_UNIT_STATUSES and ended_at_utc is None:
        raise DurabilityError("terminal checkpoint requires ended_at_utc")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "unit_id": unit_id,
        "attempt_id": attempt_id,
        "status": status,
        "atomic_unit": dict(atomic_unit),
        "bindings": bindings.as_dict(),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_seconds": duration_seconds,
        "process": {
            "pid": int(pid),
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
        },
        "exit": {"code": exit_code, "signal": signal},
        "artifacts": [dict(record) for record in artifacts],
        "error": error,
    }


def validate_checkpoint(
    checkpoint_path: str | Path,
    *,
    run_root: str | Path,
    expected_run_id: str,
    expected_unit_id: str,
    expected_atomic_unit: Mapping[str, Any],
    expected_bindings: BindingSet,
    require_completed: bool = True,
) -> Mapping[str, Any]:
    path = Path(checkpoint_path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DurabilityError(f"checkpoint is unreadable: {error}") from error
    if not isinstance(document, Mapping):
        raise DurabilityError("checkpoint must be a JSON object")
    required = {
        "schema_version",
        "run_id",
        "unit_id",
        "attempt_id",
        "status",
        "atomic_unit",
        "bindings",
        "started_at_utc",
        "ended_at_utc",
        "duration_seconds",
        "process",
        "exit",
        "artifacts",
        "error",
    }
    if set(document) != required:
        raise DurabilityError("checkpoint field mismatch")
    if document["schema_version"] != SCHEMA_VERSION:
        raise DurabilityError("unsupported checkpoint schema version")
    if document["run_id"] != expected_run_id or document["unit_id"] != expected_unit_id:
        raise DurabilityError("checkpoint run/unit identity mismatch")
    if document["atomic_unit"] != dict(expected_atomic_unit):
        raise DurabilityError("checkpoint atomic-unit identity mismatch")
    if document["bindings"] != expected_bindings.as_dict():
        raise DurabilityError("checkpoint binding mismatch")
    if document["status"] not in ALLOWED_UNIT_STATUSES:
        raise DurabilityError("checkpoint carries an unknown status")
    if require_completed and document["status"] != "completed":
        raise DurabilityError("checkpoint is not completed")
    parse_utc(document["started_at_utc"])
    if document["ended_at_utc"] is not None:
        parse_utc(document["ended_at_utc"])
    if document["status"] == "completed":
        if document["exit"] != {"code": 0, "signal": None}:
            raise DurabilityError("completed checkpoint has non-success exit evidence")
        duration = document["duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise DurabilityError("completed checkpoint duration must be numeric")
        if not math.isfinite(float(duration)) or float(duration) < 0.0:
            raise DurabilityError("completed checkpoint duration is invalid")
        artifacts = document["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise DurabilityError("completed checkpoint must carry artifacts")
        for record in artifacts:
            _validate_artifact(Path(run_root).resolve(), record)
    return document


@dataclass(frozen=True)
class ResumeDecision:
    unit_id: str
    action: str
    next_attempt_id: str
    validated_checkpoint: str | None
    reason: str


def build_resume_plan(
    *,
    run_root: str | Path,
    run_id: str,
    planned_units: Sequence[Mapping[str, Any]],
    checkpoint_paths_by_unit: Mapping[str, Sequence[str | Path]],
    bindings: BindingSet,
) -> tuple[ResumeDecision, ...]:
    """Validate completed attempts and open a new attempt for every gap."""

    unit_ids = [str(record.get("unit_id")) for record in planned_units]
    if not unit_ids or len(unit_ids) != len(set(unit_ids)):
        raise DurabilityError("planned units must have unique unit_id values")
    for unit_id in unit_ids:
        validate_identifier(unit_id, "planned unit_id")
    decisions: list[ResumeDecision] = []
    for unit in planned_units:
        unit_id = str(unit["unit_id"])
        atomic_unit = unit.get("atomic_unit")
        if not isinstance(atomic_unit, Mapping) or not atomic_unit:
            raise DurabilityError(f"planned unit {unit_id} lacks atomic_unit fields")
        paths = tuple(checkpoint_paths_by_unit.get(unit_id, ()))
        valid: list[Path] = []
        attempt_ids: set[str] = set()
        for raw_path in paths:
            path = Path(raw_path).resolve()
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            attempt_id = (
                document.get("attempt_id") if isinstance(document, Mapping) else None
            )
            if isinstance(attempt_id, str):
                if attempt_id in attempt_ids:
                    raise DurabilityError(f"duplicate attempt_id for unit {unit_id}")
                attempt_ids.add(attempt_id)
            try:
                validate_checkpoint(
                    path,
                    run_root=run_root,
                    expected_run_id=run_id,
                    expected_unit_id=unit_id,
                    expected_atomic_unit=atomic_unit,
                    expected_bindings=bindings,
                    require_completed=True,
                )
            except DurabilityError:
                continue
            valid.append(path)
        if len(valid) > 1:
            raise DurabilityError(f"multiple validated completions for unit {unit_id}")
        numeric_attempts = [
            int(match.group(1))
            for attempt_id in attempt_ids
            if (match := re.fullmatch(r"attempt-([0-9]+)", attempt_id))
        ]
        next_number = max(numeric_attempts, default=0) + 1
        next_attempt = f"attempt-{next_number:03d}"
        while next_attempt in attempt_ids:
            next_number += 1
            next_attempt = f"attempt-{next_number:03d}"
        if valid:
            decisions.append(
                ResumeDecision(
                    unit_id=unit_id,
                    action="skipped_validated",
                    next_attempt_id=next_attempt,
                    validated_checkpoint=str(valid[0]),
                    reason="completed checkpoint, schema, artifacts, and hashes validated",
                )
            )
        else:
            decisions.append(
                ResumeDecision(
                    unit_id=unit_id,
                    action="run_new_attempt",
                    next_attempt_id=next_attempt,
                    validated_checkpoint=None,
                    reason=(
                        "no prior attempt"
                        if not paths
                        else "all prior attempts are incomplete or invalid and remain preserved"
                    ),
                )
            )
    return tuple(decisions)


@dataclass(frozen=True)
class ProcessTreeObservation:
    observed_at_utc: str
    wrapper_pid: int
    wrapper_cpu_seconds: float
    process_tree_cpu_seconds: float
    process_tree_rss_bytes: int
    process_tree_io_bytes: int | None
    sampled_pids: tuple[int, ...]
    disappeared_pids: tuple[int, ...]
    process_alive: bool


def observe_process_tree(
    wrapper_pid: int, *, previous_sampled_pids: Sequence[int] = ()
) -> ProcessTreeObservation:
    """Sample a wrapper and all currently live descendants with psutil."""

    try:
        wrapper = psutil.Process(int(wrapper_pid))
        processes = [wrapper, *wrapper.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return ProcessTreeObservation(
            observed_at_utc=utc_now(),
            wrapper_pid=int(wrapper_pid),
            wrapper_cpu_seconds=0.0,
            process_tree_cpu_seconds=0.0,
            process_tree_rss_bytes=0,
            process_tree_io_bytes=None,
            sampled_pids=(),
            disappeared_pids=tuple(sorted(set(int(x) for x in previous_sampled_pids))),
            process_alive=False,
        )
    wrapper_cpu = 0.0
    tree_cpu = 0.0
    rss = 0
    io_total = 0
    io_available = True
    sampled: list[int] = []
    for process in processes:
        try:
            cpu = process.cpu_times()
            cpu_seconds = float(cpu.user + cpu.system)
            memory = process.memory_info()
            io = process.io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        sampled.append(process.pid)
        tree_cpu += cpu_seconds
        rss += int(memory.rss)
        if process.pid == wrapper.pid:
            wrapper_cpu = cpu_seconds
        if io is None:
            io_available = False
        else:
            io_total += int(io.read_bytes + io.write_bytes)
    previous = set(int(value) for value in previous_sampled_pids)
    current = set(sampled)
    return ProcessTreeObservation(
        observed_at_utc=utc_now(),
        wrapper_pid=wrapper.pid,
        wrapper_cpu_seconds=wrapper_cpu,
        process_tree_cpu_seconds=tree_cpu,
        process_tree_rss_bytes=rss,
        process_tree_io_bytes=io_total if io_available else None,
        sampled_pids=tuple(sorted(current)),
        disappeared_pids=tuple(sorted(previous - current)),
        process_alive=wrapper.pid in current,
    )


def make_heartbeat(
    *,
    run_id: str,
    unit_id: str,
    attempt_id: str,
    phase: str,
    phase_started_at_utc: str,
    unit_elapsed_seconds: float,
    completed_atomic_units: int,
    planned_atomic_units: int,
    last_durable_checkpoint_at: str | None,
    process_observation: ProcessTreeObservation,
    inner_completed: int | None = None,
    inner_total: int | None = None,
) -> dict[str, Any]:
    if completed_atomic_units < 0 or planned_atomic_units < 1:
        raise DurabilityError("heartbeat unit counts are invalid")
    if completed_atomic_units > planned_atomic_units:
        raise DurabilityError("completed units exceed planned units")
    if (inner_completed is None) != (inner_total is None):
        raise DurabilityError("inner progress requires both completed and total")
    if inner_total is not None and not (0 <= int(inner_completed) <= int(inner_total)):
        raise DurabilityError("inner progress values are inconsistent")
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": process_observation.observed_at_utc,
        "run_id": run_id,
        "unit_id": unit_id,
        "attempt_id": attempt_id,
        "pid": process_observation.wrapper_pid,
        "phase": phase,
        "phase_started_at": phase_started_at_utc,
        "unit_elapsed_seconds": float(unit_elapsed_seconds),
        "completed_atomic_units": int(completed_atomic_units),
        "planned_atomic_units": int(planned_atomic_units),
        "last_durable_checkpoint_at": last_durable_checkpoint_at,
        "inner_completed": inner_completed,
        "inner_total": inner_total,
        "wrapper_cpu_seconds": process_observation.wrapper_cpu_seconds,
        "process_tree_cpu_seconds": process_observation.process_tree_cpu_seconds,
        "process_tree_rss_bytes": process_observation.process_tree_rss_bytes,
        "process_tree_io_bytes": process_observation.process_tree_io_bytes,
        "sampled_pids": list(process_observation.sampled_pids),
        "disappeared_pids": list(process_observation.disappeared_pids),
        "process_alive": process_observation.process_alive,
        "evidence_class": "liveness_only_not_scientific_completion",
    }


HEARTBEAT_FIELDS = frozenset(
    {
        "schema_version",
        "timestamp",
        "run_id",
        "unit_id",
        "attempt_id",
        "pid",
        "phase",
        "phase_started_at",
        "unit_elapsed_seconds",
        "completed_atomic_units",
        "planned_atomic_units",
        "last_durable_checkpoint_at",
        "inner_completed",
        "inner_total",
        "wrapper_cpu_seconds",
        "process_tree_cpu_seconds",
        "process_tree_rss_bytes",
        "process_tree_io_bytes",
        "sampled_pids",
        "disappeared_pids",
        "process_alive",
        "evidence_class",
    }
)


def validate_heartbeat(heartbeat: Mapping[str, Any]) -> None:
    if not isinstance(heartbeat, Mapping) or set(heartbeat) != HEARTBEAT_FIELDS:
        raise DurabilityError("heartbeat field mismatch")
    if heartbeat["schema_version"] != SCHEMA_VERSION:
        raise DurabilityError("unsupported heartbeat schema version")
    for field in ("run_id", "unit_id", "attempt_id"):
        validate_identifier(heartbeat[field], f"heartbeat {field}")
    parse_utc(heartbeat["timestamp"])
    parse_utc(heartbeat["phase_started_at"])
    if heartbeat["last_durable_checkpoint_at"] is not None:
        parse_utc(heartbeat["last_durable_checkpoint_at"])
    if heartbeat["evidence_class"] != "liveness_only_not_scientific_completion":
        raise DurabilityError("heartbeat evidence class cannot claim completion")
    if not isinstance(heartbeat["process_alive"], bool):
        raise DurabilityError("heartbeat process_alive must be Boolean")
    sampled = heartbeat["sampled_pids"]
    disappeared = heartbeat["disappeared_pids"]
    if not isinstance(sampled, list) or not isinstance(disappeared, list):
        raise DurabilityError("heartbeat PID inventories must be lists")
    if len(sampled) != len(set(sampled)) or len(disappeared) != len(set(disappeared)):
        raise DurabilityError("heartbeat PID inventories contain duplicates")
    if set(sampled) & set(disappeared):
        raise DurabilityError("a PID cannot be sampled and disappeared simultaneously")
    if heartbeat["process_alive"] and heartbeat["pid"] not in sampled:
        raise DurabilityError("live wrapper PID is absent from sampled_pids")
    if float(heartbeat["process_tree_cpu_seconds"]) < float(
        heartbeat["wrapper_cpu_seconds"]
    ):
        raise DurabilityError("process-tree CPU cannot be below wrapper CPU")
    if heartbeat["completed_atomic_units"] > heartbeat["planned_atomic_units"]:
        raise DurabilityError("heartbeat completion count exceeds plan")


def write_heartbeat(path: str | Path, heartbeat: Mapping[str, Any]) -> None:
    validate_heartbeat(heartbeat)
    atomic_write_json(path, heartbeat)


class HeartbeatSupervisor:
    """Emit atomic snapshots and fsynced history while one unit is active."""

    def __init__(
        self,
        *,
        snapshot_path: str | Path,
        history_path: str | Path,
        run_id: str,
        unit_id: str,
        attempt_id: str,
        phase: str,
        completed_atomic_units: int,
        planned_atomic_units: int,
        last_durable_checkpoint_at: str | None,
        interval_seconds: float,
        wrapper_pid: int | None = None,
    ) -> None:
        if not (0 < float(interval_seconds) <= 300):
            raise DurabilityError("heartbeat supervisor interval must be in (0, 300]")
        self.snapshot_path = Path(snapshot_path).resolve()
        self.history_path = Path(history_path).resolve()
        self.run_id = validate_identifier(run_id, "heartbeat run_id")
        self.unit_id = validate_identifier(unit_id, "heartbeat unit_id")
        self.attempt_id = validate_identifier(attempt_id, "heartbeat attempt_id")
        self.phase = _require_nonempty_string(phase, "heartbeat phase")
        self.completed_atomic_units = int(completed_atomic_units)
        self.planned_atomic_units = int(planned_atomic_units)
        self.last_durable_checkpoint_at = last_durable_checkpoint_at
        self.interval_seconds = float(interval_seconds)
        self.wrapper_pid = os.getpid() if wrapper_pid is None else int(wrapper_pid)
        self.phase_started_at_utc = utc_now()
        self._started_monotonic = time.monotonic()
        self._previous_pids: tuple[int, ...] = ()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.emission_count = 0

    def _emit(self) -> None:
        observation = observe_process_tree(
            self.wrapper_pid, previous_sampled_pids=self._previous_pids
        )
        self._previous_pids = observation.sampled_pids
        heartbeat = make_heartbeat(
            run_id=self.run_id,
            unit_id=self.unit_id,
            attempt_id=self.attempt_id,
            phase=self.phase,
            phase_started_at_utc=self.phase_started_at_utc,
            unit_elapsed_seconds=time.monotonic() - self._started_monotonic,
            completed_atomic_units=self.completed_atomic_units,
            planned_atomic_units=self.planned_atomic_units,
            last_durable_checkpoint_at=self.last_durable_checkpoint_at,
            process_observation=observation,
        )
        write_heartbeat(self.snapshot_path, heartbeat)
        append_jsonl_durable(self.history_path, heartbeat)
        self.emission_count += 1

    def __enter__(self) -> "HeartbeatSupervisor":
        self._emit()

        def poll() -> None:
            try:
                while not self._stop.wait(self.interval_seconds):
                    self._emit()
            except BaseException as error:  # surfaced to the owner thread
                self._error = error
                self._stop.set()

        self._thread = threading.Thread(
            target=poll,
            name=f"heartbeat-{self.unit_id}-{self.attempt_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_seconds))
            if self._thread.is_alive():
                raise DurabilityError("heartbeat supervisor did not stop cleanly")
        self._emit()
        if self._error is not None:
            raise DurabilityError(f"heartbeat supervisor failed: {self._error}")


@dataclass(frozen=True)
class MonitorObservation:
    observed_at_utc: str
    heartbeat_age_seconds: float
    process_tree_cpu_seconds: float
    process_tree_io_bytes: int | None
    last_durable_checkpoint_at: str | None
    process_alive: bool
    descendant_sampling_available: bool


@dataclass(frozen=True)
class StallDecision:
    status: str
    automatic_kill_authorized: bool
    reason: str


def assess_stall(
    observations: Sequence[MonitorObservation],
    *,
    stall_threshold_seconds: float,
    heartbeat_interval_seconds: float,
) -> StallDecision:
    """Classify liveness conservatively; this function never authorizes kill."""

    if stall_threshold_seconds <= 0 or not (0 < heartbeat_interval_seconds <= 300):
        raise DurabilityError("stall/heartbeat thresholds are invalid")
    if not observations:
        return StallDecision("insufficient_evidence", False, "no monitor samples")
    latest = observations[-1]
    if not latest.process_alive:
        return StallDecision(
            "interrupted_unknown",
            False,
            "wrapper/process tree is absent; terminal status must decide the outcome",
        )
    if latest.heartbeat_age_seconds <= stall_threshold_seconds:
        return StallDecision("active", False, "heartbeat is within the declared bound")
    if not all(item.descendant_sampling_available for item in observations):
        return StallDecision(
            "observability_gap",
            False,
            "heartbeat is stale and descendant process-tree sampling is incomplete",
        )
    if len(observations) < 3:
        return StallDecision(
            "insufficient_evidence",
            False,
            "stall requires at least three samples spanning two intervals",
        )
    elapsed = (
        parse_utc(observations[-1].observed_at_utc)
        - parse_utc(observations[-3].observed_at_utc)
    ).total_seconds()
    if elapsed < 2 * heartbeat_interval_seconds:
        return StallDecision(
            "insufficient_evidence",
            False,
            "monitor samples do not span two heartbeat intervals",
        )
    window = observations[-3:]
    cpu_progress = (
        window[-1].process_tree_cpu_seconds > window[0].process_tree_cpu_seconds
    )
    io_values = [item.process_tree_io_bytes for item in window]
    io_progress = all(value is not None for value in io_values) and int(
        io_values[-1]
    ) > int(io_values[0])
    checkpoint_progress = (
        window[-1].last_durable_checkpoint_at is not None
        and window[-1].last_durable_checkpoint_at
        != window[0].last_durable_checkpoint_at
    )
    if checkpoint_progress:
        return StallDecision(
            "active_checkpoint_progress",
            False,
            "a durable checkpoint advanced despite the stale heartbeat",
        )
    if cpu_progress or io_progress:
        return StallDecision(
            "heartbeat_stale_compute_active",
            False,
            "heartbeat is stale but descendant CPU or I/O advanced",
        )
    return StallDecision(
        "stalled",
        False,
        "heartbeat is stale and descendant CPU, I/O, and durable checkpoints did not advance across two intervals",
    )


REQUIRED_DURABILITY_FIELDS = frozenset(
    {
        "checkpoint_path_and_schema",
        "atomic_write_strategy",
        "resume_command",
        "resume_validation_rule",
        "interruption_smoke_evidence",
        "per_unit_timeout_seconds",
        "whole_run_watchdog_seconds",
        "eta_basis_and_margin",
        "max_workers_and_thread_limits",
        "disk_ram_resource_preflight",
        "progress_heartbeat_path_and_stall_threshold",
        "heartbeat_cadence_schema_writer_and_atomicity",
        "heartbeat_advancement_smoke_evidence",
        "opaque_phase_supervisor_sampling_rule",
        "raw_output_contract",
        "aggregate_script_and_inputs",
        "plot_script_and_inputs",
        "decision_artifact_path_and_criterion_schema",
        "decision_discriminator_statistics_persisted",
        "notification_lifecycle",
        "terminal_status_paths",
        "predecessor_terminal_status_poll_rule",
        "partial_result_promotion_policy",
    }
)


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DurabilityError(f"{context} must be a non-empty string")
    return value


def _require_mapping_fields(
    value: Any, fields: set[str], context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DurabilityError(f"{context} contract field mismatch")
    if any(item in (None, "", [], {}) for item in value.values()):
        raise DurabilityError(f"{context} contract contains an empty value")
    return value


def _validate_freeze_engineering_bindings(block: Mapping[str, Any]) -> None:
    eta = _require_mapping_fields(
        block["eta_basis_and_margin"],
        {"artifact_path", "capture", "validator", "timing_policy", "margin_policy"},
        "ETA/timing",
    )
    resource = _require_mapping_fields(
        block["disk_ram_resource_preflight"],
        {"artifact_path", "schema_version", "capture", "validator", "required_checks"},
        "resource preflight",
    )
    if eta["artifact_path"] != resource["artifact_path"]:
        raise DurabilityError("timing and resource contracts must bind one artifact")
    if (
        eta["capture"] != "freeze_bindings.capture_timing_resource_preflight"
        or eta["validator"] != "freeze_bindings.validate_timing_resource_preflight"
        or resource["capture"] != eta["capture"]
        or resource["validator"] != eta["validator"]
    ):
        raise DurabilityError("timing/resource functions are not freeze-bound")
    if resource["schema_version"] != SCHEMA_VERSION or set(
        resource["required_checks"]
    ) != {"disk", "ram", "timing_sample"}:
        raise DurabilityError("resource preflight schema/check set mismatch")
    notification = _require_mapping_fields(
        block["notification_lifecycle"],
        {"outbox_path", "writer", "validator", "events", "delivery_policy"},
        "notification lifecycle",
    )
    if notification["outbox_path"] != "notifications/invocation-{NNN}.jsonl":
        raise DurabilityError("notification outbox path contract drift")
    if set(notification["events"]) != {
        "STARTED",
        "MILESTONE",
        "COMPLETED",
        "FAILED",
        "TIMED_OUT",
        "CANCELLED",
    }:
        raise DurabilityError("notification lifecycle event set mismatch")
    if "external" not in str(notification["delivery_policy"]).lower():
        raise DurabilityError(
            "notification contract must distinguish external delivery"
        )
    if (
        notification["writer"] != "freeze_bindings.append_notification_event"
        or notification["validator"]
        != "freeze_bindings.validate_notification_lifecycle"
    ):
        raise DurabilityError("notification functions are not freeze-bound")
    predecessor = _require_mapping_fields(
        block["predecessor_terminal_status_poll_rule"],
        {
            "poller",
            "status_path",
            "success_status",
            "failure_statuses",
            "unknown_status",
            "bounded_timeout_required",
        },
        "predecessor poll",
    )
    if (
        predecessor["poller"] != "freeze_bindings.require_predecessor_completed"
        or predecessor["success_status"] != "completed"
        or set(predecessor["failure_statuses"]) != {"failed", "timed_out", "cancelled"}
        or predecessor["unknown_status"] != "unknown_timeout"
        or predecessor["bounded_timeout_required"] is not True
    ):
        raise DurabilityError("predecessor terminal-state contract is unsafe")
    plot = _require_mapping_fields(
        block["plot_script_and_inputs"],
        {
            "script",
            "writer",
            "validator",
            "input",
            "output",
            "manifest",
            "input_policy",
        },
        "plot artifact",
    )
    if not str(plot["output"]).endswith(".svg") or not str(plot["manifest"]).endswith(
        ".json"
    ):
        raise DurabilityError("plot output/manifest paths have invalid types")
    if (
        plot["writer"] != "freeze_bindings.write_svg_bar_plot"
        or plot["validator"] != "freeze_bindings.validate_plot_artifact"
    ):
        raise DurabilityError("plot functions are not freeze-bound")
    decision = _require_mapping_fields(
        block["decision_artifact_path_and_criterion_schema"],
        {"path", "writer", "validator", "required_criterion_fields", "input_policy"},
        "decision artifact",
    )
    if set(decision["required_criterion_fields"]) != {
        "criterion_id",
        "metric",
        "value",
        "operator",
        "threshold",
        "result",
        "discriminator_statistics",
    }:
        raise DurabilityError("decision criterion field set mismatch")
    if (
        decision["writer"] != "freeze_bindings.write_decision_artifact"
        or decision["validator"] != "freeze_bindings.validate_decision_artifact"
    ):
        raise DurabilityError("decision functions are not freeze-bound")


def validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    candidate_root: str | Path,
    project_root: str | Path | None = None,
    require_execution_disabled: bool = True,
) -> BindingSet:
    """Validate the PR4 preflight schema and every live hash binding."""

    if not isinstance(manifest, Mapping):
        raise DurabilityError("run manifest must be an object")
    required_top = {
        "schema_version",
        "run_id",
        "run_class",
        "status",
        "execution_enabled",
        "atomic_unit",
        "planned_unit_count",
        "planned_units",
        "bindings",
        "durability",
    }
    if set(manifest) != required_top:
        raise DurabilityError("run manifest top-level field mismatch")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise DurabilityError("unsupported run manifest schema version")
    validate_identifier(manifest["run_id"], "run_id")
    _require_nonempty_string(manifest["run_class"], "run_class")
    _require_nonempty_string(manifest["status"], "status")
    if not isinstance(manifest["execution_enabled"], bool):
        raise DurabilityError("execution_enabled must be Boolean")
    if require_execution_disabled and manifest["execution_enabled"]:
        raise DurabilityError(
            "candidate preflight must preserve execution_enabled=false"
        )
    atomic_unit = manifest["atomic_unit"]
    if not isinstance(atomic_unit, Mapping) or not atomic_unit.get("fields"):
        raise DurabilityError("atomic_unit.fields must be non-empty")
    units = manifest["planned_units"]
    count = manifest["planned_unit_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise DurabilityError("planned_unit_count must be a positive integer")
    if not isinstance(units, list) or len(units) != count:
        raise DurabilityError("planned unit list/count mismatch")
    unit_ids = [
        record.get("unit_id") for record in units if isinstance(record, Mapping)
    ]
    if (
        len(unit_ids) != count
        or len(set(unit_ids)) != count
        or any(not x for x in unit_ids)
    ):
        raise DurabilityError("planned unit IDs must be present and unique")
    for unit_id in unit_ids:
        validate_identifier(unit_id, "planned unit_id")

    root = Path(candidate_root).resolve()
    protocol_root = root if project_root is None else Path(project_root).resolve()
    bindings = manifest["bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "protocol",
        "config",
        "registry",
        "source",
    }:
        raise DurabilityError(
            "manifest bindings must name protocol, config, registry, and source"
        )
    live_hashes: dict[str, str] = {}
    for label in ("protocol", "config", "registry"):
        record = bindings[label]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise DurabilityError(f"{label} binding field mismatch")
        binding_root = protocol_root if label == "protocol" else root
        resolved = _relative_child(
            binding_root, record["path"], f"{label} binding path"
        )
        if not resolved.is_file():
            raise DurabilityError(f"bound {label} file does not exist")
        declared = _validate_sha256(record["sha256"], f"{label} sha256")
        if sha256_file(resolved) != declared:
            raise DurabilityError(f"live {label} hash differs from manifest")
        live_hashes[label] = declared
    source_sha = validate_source_binding(root, bindings["source"])
    config_path = _relative_child(root, bindings["config"]["path"], "config path")
    registry_path = _relative_child(root, bindings["registry"]["path"], "registry path")
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    registry_document = json.loads(registry_path.read_text(encoding="utf-8"))
    if config_document.get("execution_enabled") != manifest["execution_enabled"]:
        raise DurabilityError("manifest/config execution gates differ")
    if config_document.get("track_a_registry", {}).get(
        "canonical_sha256"
    ) != registry_document.get("canonical_sha256"):
        raise DurabilityError("config does not bind the live registry canonical hash")

    durability = manifest["durability"]
    if not isinstance(durability, Mapping):
        raise DurabilityError("durability block must be an object")
    missing = REQUIRED_DURABILITY_FIELDS - set(durability)
    if missing:
        raise DurabilityError(
            f"durability preflight fields are missing: {sorted(missing)}"
        )
    for field in REQUIRED_DURABILITY_FIELDS:
        if durability[field] in (None, "", [], {}):
            raise DurabilityError(f"durability field is empty: {field}")
    _validate_freeze_engineering_bindings(durability)
    cadence = durability["heartbeat_cadence_schema_writer_and_atomicity"]
    if not isinstance(cadence, Mapping):
        raise DurabilityError("heartbeat cadence contract must be an object")
    cadence_seconds = cadence.get("cadence_seconds")
    if (
        isinstance(cadence_seconds, bool)
        or not isinstance(cadence_seconds, (int, float))
        or not (0 < float(cadence_seconds) <= 300)
    ):
        raise DurabilityError("heartbeat cadence must be in (0, 300] seconds")
    sampling = str(durability["opaque_phase_supervisor_sampling_rule"]).lower()
    if "descendant" not in sampling or "process tree" not in sampling:
        raise DurabilityError(
            "opaque-phase sampling must name the descendant process tree"
        )
    if durability["partial_result_promotion_policy"] != "prohibited":
        raise DurabilityError("partial result promotion must be prohibited")
    worker_limits = durability["max_workers_and_thread_limits"]
    if not isinstance(worker_limits, Mapping) or "workers" not in worker_limits:
        raise DurabilityError("worker/thread limit contract is malformed")
    workers = worker_limits["workers"]
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 3
    ):
        raise DurabilityError("atomic worker count must be an integer in [1,3]")
    for label, value in worker_limits.items():
        if label == "workers":
            continue
        if label == "containment":
            if value != "inherited_watchdog_process_group_plus_linux_pdeathsig":
                raise DurabilityError("parallel worker containment contract drift")
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value != 1:
            raise DurabilityError("every numerical thread limit must equal one")
    timeout = durability["per_unit_timeout_seconds"]
    watchdog = durability["whole_run_watchdog_seconds"]
    if (
        isinstance(timeout, bool)
        or isinstance(watchdog, bool)
        or not isinstance(timeout, (int, float))
        or not isinstance(watchdog, (int, float))
        or float(timeout) <= 0
        or float(watchdog) <= float(timeout)
    ):
        raise DurabilityError(
            "whole-run watchdog must exceed a positive per-unit timeout"
        )
    scientific = durability.get("scientific_checkpoint_bindings")
    optional: dict[str, str | None] = {
        "structured_registry_sha256": None,
        "structured_registry_canonical_sha256": None,
        "scientific_freeze_sha256": None,
        "run_role": None,
    }
    if scientific is not None:
        expected_scientific_fields = set(optional)
        if (
            not isinstance(scientific, Mapping)
            or set(scientific) != expected_scientific_fields
        ):
            raise DurabilityError("scientific checkpoint binding field mismatch")
        structured_path = config_path.parent / "structured_comparator_registry.json"
        if not structured_path.is_file():
            raise DurabilityError("structured registry is missing")
        structured_document = json.loads(structured_path.read_text(encoding="utf-8"))
        structured_file_sha = _validate_sha256(
            scientific["structured_registry_sha256"],
            "structured_registry_sha256",
        )
        if sha256_file(structured_path) != structured_file_sha:
            raise DurabilityError("structured registry file hash drift")
        structured_canonical_sha = _validate_sha256(
            scientific["structured_registry_canonical_sha256"],
            "structured_registry_canonical_sha256",
        )
        if structured_document.get("canonical_sha256") != structured_canonical_sha:
            raise DurabilityError("structured registry canonical hash drift")
        optional = {
            "structured_registry_sha256": structured_file_sha,
            "structured_registry_canonical_sha256": structured_canonical_sha,
            "scientific_freeze_sha256": _validate_sha256(
                scientific["scientific_freeze_sha256"], "scientific_freeze_sha256"
            ),
            "run_role": _require_nonempty_string(scientific["run_role"], "run_role"),
        }
    return BindingSet(
        protocol_sha256=live_hashes["protocol"],
        config_sha256=live_hashes["config"],
        registry_sha256=live_hashes["registry"],
        source_bundle_sha256=source_sha,
        **optional,
    )
