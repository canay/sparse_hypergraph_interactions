"""Fail-closed registry for applicability-restricted sensitivity comparators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from method_registry import load_method_registry
from ranker_protocol import (
    RankerContractError,
    canonical_sha256,
    require_canonical_sha256,
)


class StructuredComparatorRegistryError(ValueError):
    """Raised when the structured-comparator contract is unsafe or ambiguous."""


@dataclass(frozen=True)
class StructuredComparatorSpec:
    comparator_id: str
    adapter_id: str
    package: str
    version: str
    source_sha256: str
    registry_active: bool
    reporting_role: str
    eligible_scenario_ids: tuple[str, ...]
    required_cell_type: str
    required_candidate_orders: tuple[int, ...]
    required_heredity: str
    required_hierarchy: str
    excluded_from_primary_estimand: bool
    excluded_from_main_method_registry: bool


@dataclass(frozen=True)
class StructuredComparatorRegistry:
    schema_version: int
    canonical_sha256: str
    registry_id: str
    registry_active: bool
    registry_role: str
    allowed_cell_types: tuple[str, ...]
    allowed_candidate_orders: tuple[int, ...]
    allowed_heredity: tuple[str, ...]
    allowed_hierarchy: tuple[str, ...]
    allowed_analysis_roles: tuple[str, ...]
    comparators: tuple[StructuredComparatorSpec, ...]

    def eligible_comparators_for(
        self, cell_metadata: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Return applicability matches without implying runtime activation."""

        metadata = _validate_cell_metadata(self, cell_metadata)
        if metadata["primary_estimand"]:
            return ()
        matches: list[str] = []
        for comparator in self.comparators:
            if metadata["scenario_id"] not in comparator.eligible_scenario_ids:
                continue
            if metadata["cell_type"] != comparator.required_cell_type:
                continue
            if (
                tuple(metadata["candidate_orders"])
                != comparator.required_candidate_orders
            ):
                continue
            if metadata["heredity"] != comparator.required_heredity:
                continue
            if metadata["hierarchy"] != comparator.required_hierarchy:
                continue
            if metadata["analysis_role"] != comparator.reporting_role:
                continue
            matches.append(comparator.comparator_id)
        return tuple(matches)

    def active_comparators_for(
        self, cell_metadata: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Return runtime-active matches; the frozen candidate currently returns none."""

        eligible = set(self.eligible_comparators_for(cell_metadata))
        if not self.registry_active:
            return ()
        return tuple(
            comparator.comparator_id
            for comparator in self.comparators
            if comparator.registry_active and comparator.comparator_id in eligible
        )


def _require_exact_fields(
    record: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if not isinstance(record, Mapping):
        raise StructuredComparatorRegistryError(f"{context} must be an object")
    missing = expected - set(record)
    unknown = set(record) - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise StructuredComparatorRegistryError(
            f"Invalid fields in {context}: {'; '.join(details)}"
        )


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredComparatorRegistryError(f"{context} must be a non-empty string")
    return value


def _unique_string_list(value: Any, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise StructuredComparatorRegistryError(
            f"{context} must be a non-empty unique string list"
        )
    return tuple(value)


def _unique_integer_list(value: Any, context: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or len(value) != len(set(value))
    ):
        raise StructuredComparatorRegistryError(
            f"{context} must be a non-empty unique integer list"
        )
    return tuple(value)


def structured_registry_document_sha256(document: Mapping[str, Any]) -> str:
    """Hash a registry document without its self-referential field."""

    payload = dict(document)
    payload.pop("canonical_sha256", None)
    try:
        return canonical_sha256(payload)
    except RankerContractError as error:
        raise StructuredComparatorRegistryError(str(error)) from error


def _validate_cell_metadata(
    registry: StructuredComparatorRegistry, cell_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "scenario_id",
        "cell_type",
        "candidate_orders",
        "heredity",
        "hierarchy",
        "analysis_role",
        "primary_estimand",
    }
    _require_exact_fields(cell_metadata, expected, "cell_metadata")
    scenario_id = _nonempty_string(cell_metadata["scenario_id"], "scenario_id")
    cell_type = cell_metadata["cell_type"]
    heredity = cell_metadata["heredity"]
    hierarchy = cell_metadata["hierarchy"]
    analysis_role = cell_metadata["analysis_role"]
    primary_estimand = cell_metadata["primary_estimand"]
    if cell_type not in registry.allowed_cell_types:
        raise StructuredComparatorRegistryError(f"Unknown cell_type: {cell_type!r}")
    if heredity not in registry.allowed_heredity:
        raise StructuredComparatorRegistryError(f"Unknown heredity: {heredity!r}")
    if hierarchy not in registry.allowed_hierarchy:
        raise StructuredComparatorRegistryError(f"Unknown hierarchy: {hierarchy!r}")
    if analysis_role not in registry.allowed_analysis_roles:
        raise StructuredComparatorRegistryError(
            f"Unknown analysis_role: {analysis_role!r}"
        )
    if not isinstance(primary_estimand, bool):
        raise StructuredComparatorRegistryError("primary_estimand must be boolean")
    candidate_orders = _unique_integer_list(
        cell_metadata["candidate_orders"], "candidate_orders"
    )
    unknown_orders = set(candidate_orders) - set(registry.allowed_candidate_orders)
    if unknown_orders:
        raise StructuredComparatorRegistryError(
            f"Unknown candidate order(s): {sorted(unknown_orders)}"
        )
    return {
        "scenario_id": scenario_id,
        "cell_type": cell_type,
        "candidate_orders": candidate_orders,
        "heredity": heredity,
        "hierarchy": hierarchy,
        "analysis_role": analysis_role,
        "primary_estimand": primary_estimand,
    }


def _validate_smoke_evidence(
    candidate_root: Path, record: Mapping[str, Any], index: int
) -> None:
    _require_exact_fields(
        record,
        {"path", "sha256", "required_status"},
        f"comparators[{index}].smoke_evidence",
    )
    relative_path = Path(_nonempty_string(record["path"], "smoke_evidence.path"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise StructuredComparatorRegistryError(
            "smoke_evidence.path must stay inside the candidate root"
        )
    evidence_path = candidate_root / relative_path
    try:
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StructuredComparatorRegistryError(
            f"Cannot validate smoke evidence: {error}"
        ) from error
    expected_hash = record["sha256"]
    try:
        require_canonical_sha256(expected_hash, "smoke_evidence.sha256")
    except RankerContractError as error:
        raise StructuredComparatorRegistryError(str(error)) from error
    actual_hash = hashlib.sha256(evidence_bytes).hexdigest().upper()
    if actual_hash != expected_hash:
        raise StructuredComparatorRegistryError(
            f"Smoke-evidence SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if evidence.get("status") != record["required_status"]:
        raise StructuredComparatorRegistryError("Smoke-evidence status mismatch")
    if evidence.get("scientific_compute") is not False:
        raise StructuredComparatorRegistryError(
            "Smoke evidence must be explicitly non-scientific"
        )
    if evidence.get("registry_active") is not False:
        raise StructuredComparatorRegistryError(
            "Smoke evidence must preserve registry_active=false"
        )


def _read_hash_bound_json(
    root: Path, record: Mapping[str, Any], context: str
) -> tuple[Path, Mapping[str, Any]]:
    relative_path = Path(_nonempty_string(record["path"], f"{context}.path"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise StructuredComparatorRegistryError(
            f"{context}.path must stay inside its declared root"
        )
    path = root / relative_path
    try:
        payload = path.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StructuredComparatorRegistryError(
            f"Cannot validate {context}: {error}"
        ) from error
    expected_hash = record["sha256"]
    try:
        require_canonical_sha256(expected_hash, f"{context}.sha256")
    except RankerContractError as error:
        raise StructuredComparatorRegistryError(str(error)) from error
    actual_hash = hashlib.sha256(payload).hexdigest().upper()
    if actual_hash != expected_hash:
        raise StructuredComparatorRegistryError(
            f"{context} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if not isinstance(document, Mapping):
        raise StructuredComparatorRegistryError(f"{context} must contain an object")
    return path, document


def _validate_activation_evidence(
    candidate_root: Path,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _require_exact_fields(
        record,
        {"path", "sha256", "required_decision"},
        "activation_evidence",
    )
    _path, evidence = _read_hash_bound_json(
        candidate_root, record, "activation_evidence"
    )
    _require_exact_fields(
        evidence,
        {
            "schema_version",
            "evidence_id",
            "metadata",
            "decision",
            "scope",
            "protocol_binding",
            "final_design_review",
            "arm64_smoke",
            "prior_inactive_registry",
            "gates",
        },
        "activation evidence document",
    )
    if (
        evidence["schema_version"] != 1
        or evidence["decision"] != record["required_decision"]
    ):
        raise StructuredComparatorRegistryError(
            "Activation evidence decision or schema mismatch"
        )
    metadata = evidence["metadata"]
    _require_exact_fields(
        metadata,
        {"date_time", "tool", "model_if_known", "operation_id"},
        "activation evidence metadata",
    )
    if metadata["operation_id"] != "shil-dual-track-freeze-pilot-theory-20260826-1624":
        raise StructuredComparatorRegistryError("Activation operation ID mismatch")

    scope = evidence["scope"]
    _require_exact_fields(
        scope,
        {
            "comparator_id",
            "eligible_scenario_ids",
            "analysis_role",
            "excluded_from_primary_estimand",
            "excluded_from_main_method_registry",
            "primary_method_count_unchanged",
        },
        "activation scope",
    )
    if scope != {
        "comparator_id": "glinternet.strong_hierarchy.pairwise",
        "eligible_scenario_ids": ["S13"],
        "analysis_role": "sensitivity_only",
        "excluded_from_primary_estimand": True,
        "excluded_from_main_method_registry": True,
        "primary_method_count_unchanged": 14,
    }:
        raise StructuredComparatorRegistryError("Activation scope mismatch")

    protocol = evidence["protocol_binding"]
    _require_exact_fields(protocol, {"path", "sha256"}, "protocol_binding")
    if protocol != {
        "path": "MD/02_design/track_a_benchmark_protocol_20260826.md",
        "sha256": "2E233357300FC66091CD66D24AA57BFEEC6FD1DE58D6139A18D86599F61836B8",
    }:
        raise StructuredComparatorRegistryError("Activation protocol binding mismatch")
    project_root = Path(__file__).resolve().parents[3]
    protocol_path = project_root / protocol["path"]
    if (
        not protocol_path.is_file()
        or hashlib.sha256(protocol_path.read_bytes()).hexdigest().upper()
        != protocol["sha256"]
        or config.get("protocol_sha256") != protocol["sha256"]
    ):
        raise StructuredComparatorRegistryError(
            "Live protocol differs from activation evidence"
        )

    review_record = evidence["final_design_review"]
    _require_exact_fields(
        review_record,
        {"path", "sha256", "required_decision"},
        "final_design_review",
    )
    _review_path, review = _read_hash_bound_json(
        candidate_root, review_record, "final_design_review"
    )
    if (
        review_record["sha256"]
        != "B92C48FC38B14ACF4F178CFFE8CA8BCD59C878EA89DAE978E900C5A94E694C3B"
        or review_record["required_decision"] != "sound_for_prospective_freeze"
        or not isinstance(review.get("review"), Mapping)
        or review["review"].get("decision") != review_record["required_decision"]
        or review["review"].get("confidence") != "high"
        or review["review"].get("remaining_p0_p1_defects") != []
    ):
        raise StructuredComparatorRegistryError(
            "Final design-review activation predicate failed"
        )

    smoke_record = evidence["arm64_smoke"]
    _require_exact_fields(
        smoke_record,
        {"path", "sha256", "required_status", "historic_registry_active"},
        "arm64_smoke",
    )
    _smoke_path, smoke = _read_hash_bound_json(
        candidate_root, smoke_record, "arm64_smoke"
    )
    if (
        smoke_record["sha256"]
        != "D97A49CE35F215F2DD6616FE3D092FF2E568313E713292FF3263F1B9F521FAF7"
        or smoke_record["required_status"] != "REAL_ARM64_DOUBLE_SMOKE_PASS"
        or smoke_record["historic_registry_active"] is not False
        or smoke.get("status") != smoke_record["required_status"]
        or smoke.get("registry_active") is not False
        or smoke.get("scientific_compute") is not False
    ):
        raise StructuredComparatorRegistryError(
            "ARM64 smoke activation predicate failed"
        )

    prior = evidence["prior_inactive_registry"]
    _require_exact_fields(
        prior,
        {"file_sha256", "canonical_sha256"},
        "prior_inactive_registry",
    )
    if prior != {
        "file_sha256": "20A64F736026FF0EBF942E8D3E5AB42F8CA559533B306F63A70E5D04D843D7A0",
        "canonical_sha256": "20AD1D25F18C8C39E793B59476A474A055DDC9F216037777CC4524DF44FDC8F8",
    }:
        raise StructuredComparatorRegistryError(
            "Prior inactive-registry binding mismatch"
        )
    gates = evidence["gates"]
    _require_exact_fields(
        gates,
        {
            "registry_active",
            "execution_enabled",
            "scientific_compute_performed",
            "protocol_or_manuscript_modified_by_activation",
        },
        "activation gates",
    )
    if gates != {
        "registry_active": True,
        "execution_enabled": False,
        "scientific_compute_performed": False,
        "protocol_or_manuscript_modified_by_activation": False,
    }:
        raise StructuredComparatorRegistryError("Activation gate evidence mismatch")


def _parse_comparator(
    record: Mapping[str, Any], index: int, candidate_root: Path
) -> StructuredComparatorSpec:
    expected = {
        "comparator_id",
        "adapter_id",
        "package",
        "version",
        "source_sha256",
        "registry_active",
        "reporting_role",
        "eligible_scenario_ids",
        "required_cell_type",
        "required_candidate_orders",
        "required_heredity",
        "required_hierarchy",
        "excluded_from_primary_estimand",
        "excluded_from_main_method_registry",
        "smoke_evidence",
    }
    _require_exact_fields(record, expected, f"comparators[{index}]")
    _validate_smoke_evidence(candidate_root, record["smoke_evidence"], index)
    source_sha256 = record["source_sha256"]
    try:
        require_canonical_sha256(source_sha256, "source_sha256")
    except RankerContractError as error:
        raise StructuredComparatorRegistryError(str(error)) from error
    for field in (
        "registry_active",
        "excluded_from_primary_estimand",
        "excluded_from_main_method_registry",
    ):
        if not isinstance(record[field], bool):
            raise StructuredComparatorRegistryError(f"{field} must be boolean")
    return StructuredComparatorSpec(
        comparator_id=_nonempty_string(record["comparator_id"], "comparator_id"),
        adapter_id=_nonempty_string(record["adapter_id"], "adapter_id"),
        package=_nonempty_string(record["package"], "package"),
        version=_nonempty_string(record["version"], "version"),
        source_sha256=source_sha256,
        registry_active=record["registry_active"],
        reporting_role=_nonempty_string(record["reporting_role"], "reporting_role"),
        eligible_scenario_ids=_unique_string_list(
            record["eligible_scenario_ids"], "eligible_scenario_ids"
        ),
        required_cell_type=_nonempty_string(
            record["required_cell_type"], "required_cell_type"
        ),
        required_candidate_orders=_unique_integer_list(
            record["required_candidate_orders"], "required_candidate_orders"
        ),
        required_heredity=_nonempty_string(
            record["required_heredity"], "required_heredity"
        ),
        required_hierarchy=_nonempty_string(
            record["required_hierarchy"], "required_hierarchy"
        ),
        excluded_from_primary_estimand=record["excluded_from_primary_estimand"],
        excluded_from_main_method_registry=record["excluded_from_main_method_registry"],
    )


def _validate_binding(
    registry: StructuredComparatorRegistry,
    document: Mapping[str, Any],
    candidate_root: Path,
) -> None:
    if registry.schema_version != 1:
        raise StructuredComparatorRegistryError("Unsupported schema_version")
    if registry.registry_role != "sensitivity_only":
        raise StructuredComparatorRegistryError(
            "Registry role must be sensitivity_only"
        )
    if not registry.registry_active:
        raise StructuredComparatorRegistryError(
            "Structured comparator registry must be activation-evidence bound"
        )
    if len(registry.comparators) != 1:
        raise StructuredComparatorRegistryError(
            "This amendment requires exactly one structured comparator"
        )
    comparator = registry.comparators[0]
    if (
        comparator.comparator_id != "glinternet.strong_hierarchy.pairwise"
        or comparator.package != "glinternet"
        or comparator.reporting_role != "sensitivity_only"
        or comparator.eligible_scenario_ids != ("S13",)
        or comparator.required_cell_type != "structured_pairwise"
        or comparator.required_candidate_orders != (2,)
        or comparator.required_heredity != "respected"
        or comparator.required_hierarchy != "strong"
        or not comparator.excluded_from_primary_estimand
        or not comparator.excluded_from_main_method_registry
        or not comparator.registry_active
    ):
        raise StructuredComparatorRegistryError(
            "glinternet applicability must remain active S13-only sensitivity-only"
        )

    primary = document["primary_method_registry"]
    _require_exact_fields(
        primary, {"path", "expected_method_count"}, "primary_method_registry"
    )
    primary_path = (
        candidate_root
        / "config"
        / _nonempty_string(primary["path"], "primary_method_registry.path")
    )
    main_registry = load_method_registry(primary_path)
    if (
        len(main_registry.methods) != primary["expected_method_count"]
        or len(main_registry.methods) != 14
    ):
        raise StructuredComparatorRegistryError(
            "Primary method registry must remain the 14-method grid"
        )
    if comparator.comparator_id in main_registry.method_by_id or any(
        method.ranker_id == "glinternet" for method in main_registry.methods
    ):
        raise StructuredComparatorRegistryError(
            "glinternet must be excluded from the primary method registry"
        )

    try:
        config = json.loads(
            (candidate_root / "config" / "full_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise StructuredComparatorRegistryError(
            f"Cannot load full_config.json: {error}"
        ) from error
    candidate = config.get("structured_comparator_candidate")
    if not isinstance(candidate, Mapping):
        raise StructuredComparatorRegistryError(
            "full_config lacks structured_comparator_candidate"
        )
    if (
        candidate.get("package") != comparator.package
        or candidate.get("version") != comparator.version
        or candidate.get("source_sha256") != comparator.source_sha256
        or candidate.get("applicability") != "pairwise_strong_hierarchy_only"
        or candidate.get("status") != "ACTIVE_S13_SENSITIVITY_ONLY_EXECUTION_DISABLED"
        or candidate.get("registry_active") is not True
    ):
        raise StructuredComparatorRegistryError(
            "Structured comparator registry does not match full_config"
        )
    _validate_activation_evidence(
        candidate_root, document["activation_evidence"], config
    )
    scenarios = {item.get("id"): item for item in config.get("scenarios", [])}
    s13 = scenarios.get("S13")
    if not isinstance(s13, Mapping) or (
        s13.get("kind") != "pair" or s13.get("heredity") != "respected"
    ):
        raise StructuredComparatorRegistryError(
            "S13 must remain a pairwise heredity-respected scenario"
        )


def load_structured_comparator_registry(
    path: str | Path,
) -> StructuredComparatorRegistry:
    registry_path = Path(path)
    candidate_root = registry_path.resolve().parent.parent
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructuredComparatorRegistryError(
            f"Cannot load structured comparator registry: {error}"
        ) from error
    _require_exact_fields(
        document,
        {
            "schema_version",
            "canonical_sha256",
            "registry_id",
            "registry_active",
            "registry_role",
            "activation_evidence",
            "primary_method_registry",
            "cell_metadata_schema",
            "comparators",
        },
        "registry",
    )
    try:
        require_canonical_sha256(document["canonical_sha256"], "canonical_sha256")
    except RankerContractError as error:
        raise StructuredComparatorRegistryError(str(error)) from error
    computed_hash = structured_registry_document_sha256(document)
    if document["canonical_sha256"] != computed_hash:
        raise StructuredComparatorRegistryError(
            f"Registry canonical_sha256 mismatch: expected {computed_hash}"
        )
    if not isinstance(document["registry_active"], bool):
        raise StructuredComparatorRegistryError("registry_active must be boolean")
    metadata = document["cell_metadata_schema"]
    _require_exact_fields(
        metadata,
        {
            "allowed_cell_types",
            "allowed_candidate_orders",
            "allowed_heredity",
            "allowed_hierarchy",
            "allowed_analysis_roles",
        },
        "cell_metadata_schema",
    )
    if not isinstance(document["comparators"], list):
        raise StructuredComparatorRegistryError("comparators must be an array")
    registry = StructuredComparatorRegistry(
        schema_version=document["schema_version"],
        canonical_sha256=document["canonical_sha256"],
        registry_id=_nonempty_string(document["registry_id"], "registry_id"),
        registry_active=document["registry_active"],
        registry_role=_nonempty_string(document["registry_role"], "registry_role"),
        allowed_cell_types=_unique_string_list(
            metadata["allowed_cell_types"], "allowed_cell_types"
        ),
        allowed_candidate_orders=_unique_integer_list(
            metadata["allowed_candidate_orders"], "allowed_candidate_orders"
        ),
        allowed_heredity=_unique_string_list(
            metadata["allowed_heredity"], "allowed_heredity"
        ),
        allowed_hierarchy=_unique_string_list(
            metadata["allowed_hierarchy"], "allowed_hierarchy"
        ),
        allowed_analysis_roles=_unique_string_list(
            metadata["allowed_analysis_roles"], "allowed_analysis_roles"
        ),
        comparators=tuple(
            _parse_comparator(item, index, candidate_root)
            for index, item in enumerate(document["comparators"])
        ),
    )
    _validate_binding(registry, document, candidate_root)
    return registry
