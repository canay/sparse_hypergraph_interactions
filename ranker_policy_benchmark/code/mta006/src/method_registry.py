"""Canonical, fail-closed ranker/method registry for the Track-A benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Collection, Mapping

from ranker_protocol import (
    ALLOWED_ZERO_SCORE_SEMANTICS,
    RankerContractError,
    canonical_sha256,
    require_canonical_sha256,
)


ALLOWED_POLICIES = frozenset({"fixed_k", "validation_one_se", "cpss_one_se"})
ALLOWED_METHOD_SCORE_SOURCES = frozenset(
    {"final_ranker_score", "final_development_selection_frequency"}
)
ALLOWED_RANKER_SCORE_DIRECTION = "higher_is_better"


class RegistryValidationError(ValueError):
    """Raised when the registry is ambiguous, unknown, or internally inconsistent."""


@dataclass(frozen=True)
class RankerSpec:
    ranker_id: str
    adapter_id: str
    display_name: str
    config_key: str
    score_source: str
    score_direction: str
    zero_score_semantics: str
    seed_namespace: str
    supported_orders: tuple[int, ...]


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    config_key: str
    score_source: str
    required_parameters: tuple[str, ...]
    allowed_k: tuple[int, ...]


@dataclass(frozen=True)
class ApplicabilityRule:
    applicability_key: str
    allowed_cell_types: tuple[str, ...]
    required_candidate_orders: tuple[int, ...]


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    ranker_id: str
    policy_id: str
    config_key: str
    policy_parameters: Mapping[str, Any]
    score_source: str
    score_family_id: str
    applicability_key: str


@dataclass(frozen=True)
class MethodRegistry:
    schema_version: int
    canonical_sha256: str
    config_keys: tuple[str, ...]
    allowed_cell_types: tuple[str, ...]
    allowed_candidate_orders: tuple[int, ...]
    rankers: tuple[RankerSpec, ...]
    policies: tuple[PolicySpec, ...]
    applicability_rules: tuple[ApplicabilityRule, ...]
    methods: tuple[MethodSpec, ...]

    @property
    def ranker_by_id(self) -> dict[str, RankerSpec]:
        return {spec.ranker_id: spec for spec in self.rankers}

    @property
    def method_by_id(self) -> dict[str, MethodSpec]:
        return {spec.method_id: spec for spec in self.methods}

    def expected_methods_for(self, cell_metadata: Mapping[str, Any]) -> tuple[str, ...]:
        """Return registry-ordered methods applicable to an explicitly typed cell."""

        if not isinstance(cell_metadata, Mapping):
            raise RegistryValidationError("cell_metadata must be a mapping")
        missing = {"cell_type", "candidate_orders"} - set(cell_metadata)
        if missing:
            raise RegistryValidationError(
                "cell_metadata is missing required fields: "
                + ", ".join(sorted(missing))
            )
        cell_type = cell_metadata["cell_type"]
        if not isinstance(cell_type, str) or cell_type not in self.allowed_cell_types:
            raise RegistryValidationError(f"Unknown cell_type: {cell_type!r}")
        raw_orders = cell_metadata["candidate_orders"]
        if (
            not isinstance(raw_orders, (list, tuple))
            or not raw_orders
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in raw_orders
            )
        ):
            raise RegistryValidationError(
                "candidate_orders must be a non-empty list of integer orders"
            )
        candidate_orders = tuple(raw_orders)
        if len(set(candidate_orders)) != len(candidate_orders):
            raise RegistryValidationError("candidate_orders contains duplicates")
        unknown_orders = set(candidate_orders) - set(self.allowed_candidate_orders)
        if unknown_orders:
            raise RegistryValidationError(
                f"Unknown candidate order(s): {sorted(unknown_orders)}"
            )
        rules = {rule.applicability_key: rule for rule in self.applicability_rules}
        output: list[str] = []
        for method in self.methods:
            rule = rules.get(method.applicability_key)
            if rule is None:
                raise RegistryValidationError(
                    f"Unknown applicability key at runtime: {method.applicability_key}"
                )
            if cell_type not in rule.allowed_cell_types:
                continue
            if not set(rule.required_candidate_orders).issubset(candidate_orders):
                continue
            output.append(method.method_id)
        return tuple(output)


def _require_exact_fields(
    record: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if not isinstance(record, Mapping):
        raise RegistryValidationError(f"{context} must be an object")
    unknown = set(record) - expected
    missing = expected - set(record)
    if unknown or missing:
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            detail.append("unknown=" + ",".join(sorted(unknown)))
        raise RegistryValidationError(
            f"Invalid fields in {context}: {'; '.join(detail)}"
        )


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryValidationError(f"{context} must be a non-empty string")
    return value


def _unique(records: Collection[Any], attribute: str, context: str) -> None:
    values = [getattr(record, attribute) for record in records]
    if len(values) != len(set(values)):
        raise RegistryValidationError(f"Duplicate {context} {attribute}")


def registry_document_sha256(document: Mapping[str, Any]) -> str:
    """Hash a registry document, excluding its self-referential hash field."""

    payload = dict(document)
    payload.pop("canonical_sha256", None)
    try:
        return canonical_sha256(payload)
    except RankerContractError as error:
        raise RegistryValidationError(str(error)) from error


def _parse_ranker(record: Mapping[str, Any], index: int) -> RankerSpec:
    expected = {
        "ranker_id",
        "adapter_id",
        "display_name",
        "config_key",
        "score_source",
        "score_direction",
        "zero_score_semantics",
        "seed_namespace",
        "supported_orders",
    }
    _require_exact_fields(record, expected, f"rankers[{index}]")
    raw_orders = record["supported_orders"]
    if (
        not isinstance(raw_orders, list)
        or not raw_orders
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_orders
        )
        or len(raw_orders) != len(set(raw_orders))
    ):
        raise RegistryValidationError(f"rankers[{index}].supported_orders is invalid")
    if record["score_direction"] != ALLOWED_RANKER_SCORE_DIRECTION:
        raise RegistryValidationError("Ranker score direction must be higher_is_better")
    if record["zero_score_semantics"] not in ALLOWED_ZERO_SCORE_SEMANTICS:
        raise RegistryValidationError(
            f"Unknown zero_score_semantics: {record['zero_score_semantics']!r}"
        )
    return RankerSpec(
        ranker_id=_nonempty_string(record["ranker_id"], "ranker_id"),
        adapter_id=_nonempty_string(record["adapter_id"], "adapter_id"),
        display_name=_nonempty_string(record["display_name"], "ranker display_name"),
        config_key=_nonempty_string(record["config_key"], "ranker config_key"),
        score_source=_nonempty_string(record["score_source"], "ranker score_source"),
        score_direction=record["score_direction"],
        zero_score_semantics=record["zero_score_semantics"],
        seed_namespace=_nonempty_string(record["seed_namespace"], "seed_namespace"),
        supported_orders=tuple(raw_orders),
    )


def _parse_policy(record: Mapping[str, Any], index: int) -> PolicySpec:
    expected = {
        "policy_id",
        "config_key",
        "score_source",
        "required_parameters",
        "allowed_k",
    }
    _require_exact_fields(record, expected, f"policies[{index}]")
    policy_id = _nonempty_string(record["policy_id"], "policy_id")
    if policy_id not in ALLOWED_POLICIES:
        raise RegistryValidationError(f"Unknown policy_id: {policy_id!r}")
    if record["score_source"] not in ALLOWED_METHOD_SCORE_SOURCES:
        raise RegistryValidationError(
            f"Unknown policy score source: {record['score_source']!r}"
        )
    required = record["required_parameters"]
    allowed_k = record["allowed_k"]
    if not isinstance(required, list) or any(
        not isinstance(item, str) for item in required
    ):
        raise RegistryValidationError(
            f"policies[{index}].required_parameters is invalid"
        )
    if len(required) != len(set(required)):
        raise RegistryValidationError(
            f"policies[{index}] duplicates required parameters"
        )
    if not isinstance(allowed_k, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in allowed_k
    ):
        raise RegistryValidationError(f"policies[{index}].allowed_k is invalid")
    if policy_id == "fixed_k":
        if tuple(required) != ("k",) or not allowed_k:
            raise RegistryValidationError(
                "fixed_k must require k and declare allowed_k"
            )
        if record["score_source"] != "final_ranker_score":
            raise RegistryValidationError("fixed_k must report final_ranker_score")
    else:
        if required or allowed_k:
            raise RegistryValidationError(
                f"{policy_id} must not accept policy parameters"
            )
        expected_source = (
            "final_development_selection_frequency"
            if policy_id == "cpss_one_se"
            else "final_ranker_score"
        )
        if record["score_source"] != expected_source:
            raise RegistryValidationError(
                f"{policy_id} has an incompatible score source"
            )
    return PolicySpec(
        policy_id=policy_id,
        config_key=_nonempty_string(record["config_key"], "policy config_key"),
        score_source=record["score_source"],
        required_parameters=tuple(required),
        allowed_k=tuple(allowed_k),
    )


def _parse_applicability(record: Mapping[str, Any], index: int) -> ApplicabilityRule:
    expected = {"applicability_key", "allowed_cell_types", "required_candidate_orders"}
    _require_exact_fields(record, expected, f"applicability_rules[{index}]")
    cell_types = record["allowed_cell_types"]
    orders = record["required_candidate_orders"]
    if (
        not isinstance(cell_types, list)
        or not cell_types
        or any(not isinstance(item, str) or not item for item in cell_types)
        or len(cell_types) != len(set(cell_types))
    ):
        raise RegistryValidationError(
            f"applicability_rules[{index}].allowed_cell_types is invalid"
        )
    if (
        not isinstance(orders, list)
        or not orders
        or any(isinstance(item, bool) or not isinstance(item, int) for item in orders)
        or len(orders) != len(set(orders))
    ):
        raise RegistryValidationError(
            f"applicability_rules[{index}].required_candidate_orders is invalid"
        )
    return ApplicabilityRule(
        applicability_key=_nonempty_string(
            record["applicability_key"], "applicability_key"
        ),
        allowed_cell_types=tuple(cell_types),
        required_candidate_orders=tuple(orders),
    )


def _parse_method(record: Mapping[str, Any], index: int) -> MethodSpec:
    expected = {
        "method_id",
        "display_name",
        "ranker_id",
        "policy_id",
        "config_key",
        "policy_parameters",
        "score_source",
        "score_family_id",
        "applicability_key",
    }
    _require_exact_fields(record, expected, f"methods[{index}]")
    parameters = record["policy_parameters"]
    if not isinstance(parameters, Mapping):
        raise RegistryValidationError(
            f"methods[{index}].policy_parameters must be an object"
        )
    try:
        canonical_sha256(parameters)
    except RankerContractError as error:
        raise RegistryValidationError(str(error)) from error
    return MethodSpec(
        method_id=_nonempty_string(record["method_id"], "method_id"),
        display_name=_nonempty_string(record["display_name"], "method display_name"),
        ranker_id=_nonempty_string(record["ranker_id"], "method ranker_id"),
        policy_id=_nonempty_string(record["policy_id"], "method policy_id"),
        config_key=_nonempty_string(record["config_key"], "method config_key"),
        policy_parameters=dict(parameters),
        score_source=_nonempty_string(record["score_source"], "method score_source"),
        score_family_id=_nonempty_string(record["score_family_id"], "score_family_id"),
        applicability_key=_nonempty_string(
            record["applicability_key"], "applicability_key"
        ),
    )


def _validate_registry(registry: MethodRegistry) -> None:
    if registry.schema_version != 1:
        raise RegistryValidationError(
            f"Unsupported schema_version: {registry.schema_version}"
        )
    # MCH-SHIL-004 (shil-claude-mch004-v6-ranker-menu-20260903): the PR-2 lock
    # named exactly {shil, l1} and 14 methods.  v6 adds the tree and screen
    # families because two coefficient-magnitude rankers could not make the
    # adaptive policy move the selected set.  The lock is re-pointed, not
    # removed: the registry is still pinned to an exact declared menu and any
    # other combination fails closed.
    if len(registry.rankers) != 4 or {item.ranker_id for item in registry.rankers} != {
        "shil",
        "l1",
        "tree",
        "screen",
    }:
        raise RegistryValidationError(
            "MCH-SHIL-004 requires exactly the shil, l1, tree and screen rankers"
        )
    if len(registry.methods) != 28:
        raise RegistryValidationError(
            "MCH-SHIL-004 requires exactly 28 MethodSpec records"
        )
    for collection, attribute, label in (
        (registry.rankers, "ranker_id", "ranker"),
        (registry.rankers, "adapter_id", "ranker"),
        (registry.rankers, "seed_namespace", "ranker"),
        (registry.policies, "policy_id", "policy"),
        (registry.applicability_rules, "applicability_key", "applicability"),
        (registry.methods, "method_id", "method"),
        (registry.methods, "display_name", "method"),
    ):
        _unique(collection, attribute, label)
    if {item.policy_id for item in registry.policies} != ALLOWED_POLICIES:
        raise RegistryValidationError(
            "Registry must declare all and only the three supported policies"
        )

    config_keys = set(registry.config_keys)
    if len(config_keys) != len(registry.config_keys):
        raise RegistryValidationError("Duplicate config key")
    used_config_keys = {item.config_key for item in registry.rankers}
    used_config_keys.update(item.config_key for item in registry.policies)
    if used_config_keys != config_keys:
        unknown = used_config_keys - config_keys
        unused = config_keys - used_config_keys
        raise RegistryValidationError(
            f"Config-key mismatch: unknown={sorted(unknown)}, unused={sorted(unused)}"
        )
    allowed_orders = set(registry.allowed_candidate_orders)
    if not allowed_orders or len(allowed_orders) != len(
        registry.allowed_candidate_orders
    ):
        raise RegistryValidationError("allowed_candidate_orders is empty or duplicated")
    if not registry.allowed_cell_types or len(set(registry.allowed_cell_types)) != len(
        registry.allowed_cell_types
    ):
        raise RegistryValidationError("allowed_cell_types is empty or duplicated")
    for ranker in registry.rankers:
        if not set(ranker.supported_orders).issubset(allowed_orders):
            raise RegistryValidationError(
                f"Ranker {ranker.ranker_id} has an unknown supported order"
            )
        if set(ranker.supported_orders) != {2, 3}:
            raise RegistryValidationError(
                f"Ranker {ranker.ranker_id} must support pair+triple candidates"
            )
    for rule in registry.applicability_rules:
        if not set(rule.allowed_cell_types).issubset(registry.allowed_cell_types):
            raise RegistryValidationError(
                f"Applicability {rule.applicability_key} has unknown cell types"
            )
        if not set(rule.required_candidate_orders).issubset(allowed_orders):
            raise RegistryValidationError(
                f"Applicability {rule.applicability_key} has unknown orders"
            )

    rankers = registry.ranker_by_id
    policies = {item.policy_id: item for item in registry.policies}
    rules = {item.applicability_key: item for item in registry.applicability_rules}
    method_identities: set[tuple[Any, ...]] = set()
    family_to_signature: dict[str, tuple[str, str]] = {}
    signature_to_family: dict[tuple[str, str], str] = {}
    for method in registry.methods:
        if method.ranker_id not in rankers:
            raise RegistryValidationError(
                f"Method {method.method_id} references an unknown ranker"
            )
        if method.policy_id not in policies:
            raise RegistryValidationError(
                f"Method {method.method_id} references an unknown policy"
            )
        if method.applicability_key not in rules:
            raise RegistryValidationError(
                f"Method {method.method_id} references an unknown applicability key"
            )
        policy = policies[method.policy_id]
        if method.config_key != policy.config_key:
            raise RegistryValidationError(
                f"Method {method.method_id} has an incompatible config key"
            )
        if method.score_source != policy.score_source:
            raise RegistryValidationError(
                f"Method {method.method_id} has an incompatible score source"
            )
        parameter_names = tuple(sorted(method.policy_parameters))
        if parameter_names != tuple(sorted(policy.required_parameters)):
            raise RegistryValidationError(
                f"Method {method.method_id} has invalid policy parameters"
            )
        if method.policy_id == "fixed_k":
            k_value = method.policy_parameters["k"]
            if (
                isinstance(k_value, bool)
                or not isinstance(k_value, int)
                or k_value not in policy.allowed_k
            ):
                raise RegistryValidationError(
                    f"Method {method.method_id} has an invalid fixed k"
                )
        identity = (
            method.ranker_id,
            method.policy_id,
            canonical_sha256(method.policy_parameters),
            method.applicability_key,
        )
        if identity in method_identities:
            raise RegistryValidationError(
                f"Duplicate method identity for {method.method_id}"
            )
        method_identities.add(identity)

        signature = (method.ranker_id, method.score_source)
        existing_signature = family_to_signature.setdefault(
            method.score_family_id, signature
        )
        if existing_signature != signature:
            raise RegistryValidationError(
                f"score_family_id {method.score_family_id} crosses ranker or score-source boundaries"
            )
        existing_family = signature_to_family.setdefault(
            signature, method.score_family_id
        )
        if existing_family != method.score_family_id:
            raise RegistryValidationError(
                f"Ranker/score-source signature {signature} is split across score families"
            )

    expected_k = set(policies["fixed_k"].allowed_k)
    for ranker_id in rankers:
        ranker_methods = [
            item for item in registry.methods if item.ranker_id == ranker_id
        ]
        observed_policies = {item.policy_id for item in ranker_methods}
        observed_k = {
            item.policy_parameters["k"]
            for item in ranker_methods
            if item.policy_id == "fixed_k"
        }
        if (
            observed_policies != ALLOWED_POLICIES
            or observed_k != expected_k
            or len(ranker_methods) != 7
        ):
            raise RegistryValidationError(
                f"Ranker {ranker_id} must define five fixed budgets plus both one-SE policies"
            )


def load_method_registry(
    path: str | Path,
    *,
    available_config_keys: Collection[str] | None = None,
) -> MethodRegistry:
    registry_path = Path(path)
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryValidationError(
            f"Cannot load method registry: {error}"
        ) from error
    top_fields = {
        "schema_version",
        "canonical_sha256",
        "config_keys",
        "cell_metadata_schema",
        "rankers",
        "policies",
        "applicability_rules",
        "methods",
    }
    _require_exact_fields(document, top_fields, "registry")
    try:
        require_canonical_sha256(document["canonical_sha256"], "canonical_sha256")
    except RankerContractError as error:
        raise RegistryValidationError(str(error)) from error
    computed_hash = registry_document_sha256(document)
    if document["canonical_sha256"] != computed_hash:
        raise RegistryValidationError(
            f"Registry canonical_sha256 mismatch: expected {computed_hash}"
        )
    config_keys = document["config_keys"]
    if not isinstance(config_keys, list) or any(
        not isinstance(item, str) or not item for item in config_keys
    ):
        raise RegistryValidationError("config_keys must be a list of non-empty strings")
    metadata_schema = document["cell_metadata_schema"]
    _require_exact_fields(
        metadata_schema,
        {"allowed_cell_types", "allowed_candidate_orders"},
        "cell_metadata_schema",
    )
    cell_types = metadata_schema["allowed_cell_types"]
    orders = metadata_schema["allowed_candidate_orders"]
    if not isinstance(cell_types, list) or any(
        not isinstance(item, str) or not item for item in cell_types
    ):
        raise RegistryValidationError("allowed_cell_types must be a list of strings")
    if not isinstance(orders, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in orders
    ):
        raise RegistryValidationError(
            "allowed_candidate_orders must be a list of integers"
        )
    for collection_name in ("rankers", "policies", "applicability_rules", "methods"):
        if not isinstance(document[collection_name], list):
            raise RegistryValidationError(f"{collection_name} must be an array")

    registry = MethodRegistry(
        schema_version=document["schema_version"],
        canonical_sha256=document["canonical_sha256"],
        config_keys=tuple(config_keys),
        allowed_cell_types=tuple(cell_types),
        allowed_candidate_orders=tuple(orders),
        rankers=tuple(
            _parse_ranker(record, index)
            for index, record in enumerate(document["rankers"])
        ),
        policies=tuple(
            _parse_policy(record, index)
            for index, record in enumerate(document["policies"])
        ),
        applicability_rules=tuple(
            _parse_applicability(record, index)
            for index, record in enumerate(document["applicability_rules"])
        ),
        methods=tuple(
            _parse_method(record, index)
            for index, record in enumerate(document["methods"])
        ),
    )
    _validate_registry(registry)
    if available_config_keys is not None:
        provided = set(available_config_keys)
        missing = set(registry.config_keys) - provided
        if missing:
            raise RegistryValidationError(
                "Runtime config is missing registry keys: " + ", ".join(sorted(missing))
            )
    return registry


def expected_methods_for(
    registry: MethodRegistry, cell_metadata: Mapping[str, Any]
) -> tuple[str, ...]:
    """Functional facade used by runners and partition validators."""

    return registry.expected_methods_for(cell_metadata)
