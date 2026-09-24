"""Fail-closed binding between the Track-A config and method registry.

Loading this module does not authorize or launch computation.  It only proves
that the candidate config names one in-tree registry and supplies the exact
configuration blocks declared by that registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from method_registry import (
    MethodRegistry,
    RegistryValidationError,
    load_method_registry,
)
from structured_comparator_registry import (
    StructuredComparatorRegistry,
    StructuredComparatorRegistryError,
    load_structured_comparator_registry,
)


class TrackAConfigError(ValueError):
    """Raised when the config-to-registry binding is incomplete or ambiguous."""


@dataclass(frozen=True)
class TrackAConfigBinding:
    config_path: Path
    config_sha256: str
    execution_enabled: bool
    candidate_status: str
    registry_path: Path
    registry: MethodRegistry
    resolved_config: Mapping[str, Mapping[str, Any]]
    structured_registry_path: Path
    structured_registry_file_sha256: str
    structured_registry: StructuredComparatorRegistry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require_exact_fields(
    record: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if not isinstance(record, Mapping):
        raise TrackAConfigError(f"{context} must be an object")
    missing = expected - set(record)
    unknown = set(record) - expected
    if missing or unknown:
        raise TrackAConfigError(
            f"{context} field mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def load_track_a_config(config_path: str | Path) -> TrackAConfigBinding:
    path = Path(config_path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrackAConfigError(f"Cannot load Track-A config: {error}") from error
    if not isinstance(document, Mapping):
        raise TrackAConfigError("Track-A config must be a JSON object")

    registry_record = document.get("track_a_registry")
    _require_exact_fields(
        registry_record,
        {
            "path",
            "canonical_sha256",
            "expected_method_count",
            "expected_score_family_count",
        },
        "track_a_registry",
    )
    raw_registry_path = registry_record["path"]
    if not isinstance(raw_registry_path, str) or not raw_registry_path.strip():
        raise TrackAConfigError("track_a_registry.path must be a non-empty string")
    if Path(raw_registry_path).is_absolute():
        raise TrackAConfigError("track_a_registry.path must be relative to config/")
    registry_path = (path.parent / raw_registry_path).resolve()
    candidate_root = path.parent.parent.resolve()
    try:
        registry_path.relative_to(candidate_root)
    except ValueError as error:
        raise TrackAConfigError("method registry escapes the candidate root") from error

    raw_resolved = document.get("track_a_resolved_config")
    if not isinstance(raw_resolved, Mapping):
        raise TrackAConfigError("track_a_resolved_config must be an object")
    if any(not isinstance(value, Mapping) for value in raw_resolved.values()):
        raise TrackAConfigError("every Track-A resolved config block must be an object")
    resolved = {str(key): dict(value) for key, value in raw_resolved.items()}
    try:
        registry = load_method_registry(
            registry_path, available_config_keys=resolved.keys()
        )
    except RegistryValidationError as error:
        raise TrackAConfigError(str(error)) from error
    if set(resolved) != set(registry.config_keys):
        raise TrackAConfigError("resolved config contains unknown registry keys")
    if registry_record["canonical_sha256"] != registry.canonical_sha256:
        raise TrackAConfigError("config-bound registry SHA-256 does not match registry")
    if registry_record["expected_method_count"] != len(registry.methods):
        raise TrackAConfigError("expected method count does not match registry")
    score_family_count = len({method.score_family_id for method in registry.methods})
    if registry_record["expected_score_family_count"] != score_family_count:
        raise TrackAConfigError("expected score-family count does not match registry")

    structured_registry_path = path.parent / "structured_comparator_registry.json"
    try:
        structured_registry = load_structured_comparator_registry(
            structured_registry_path
        )
    except StructuredComparatorRegistryError as error:
        raise TrackAConfigError(str(error)) from error

    execution_enabled = document.get("execution_enabled")
    if not isinstance(execution_enabled, bool):
        raise TrackAConfigError("execution_enabled must be Boolean")
    candidate_status = document.get("candidate_status")
    if not isinstance(candidate_status, str) or not candidate_status.strip():
        raise TrackAConfigError("candidate_status must be a non-empty string")
    return TrackAConfigBinding(
        config_path=path,
        config_sha256=_sha256_file(path),
        execution_enabled=execution_enabled,
        candidate_status=candidate_status,
        registry_path=registry_path,
        registry=registry,
        resolved_config=resolved,
        structured_registry_path=structured_registry_path.resolve(),
        structured_registry_file_sha256=_sha256_file(structured_registry_path),
        structured_registry=structured_registry,
    )
