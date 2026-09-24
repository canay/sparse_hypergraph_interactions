"""Fail-closed ranker interface and result/provenance validation.

This module is deliberately independent of the experiment runner.  A new ranker
implements :class:`RankerAdapter`; the runner supplies an immutable fit request
and validates the returned complete candidate score vector before any policy is
allowed to consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


EDGE = tuple[int, ...]
FitPhase = Literal[
    "training_rank",
    "final_development_rank",
    "cpss_tuning",
    "cpss_final",
]
ZeroScoreSemantics = Literal["dense", "inactive_is_zero"]

ALLOWED_FIT_PHASES = frozenset(
    {"training_rank", "final_development_rank", "cpss_tuning", "cpss_final"}
)
ALLOWED_ZERO_SCORE_SEMANTICS = frozenset({"dense", "inactive_is_zero"})
CANONICAL_SHA256_RE = re.compile(r"^[0-9A-F]{64}$")


class RankerContractError(ValueError):
    """Raised when a request, result, or provenance object violates the contract."""


def _json_compatible(value: Any, path: str = "$") -> Any:
    """Return a deterministic JSON-compatible copy or fail closed.

    Tuples are represented as arrays.  Mapping keys must be strings and all
    floating-point values must be finite, avoiding platform-dependent NaN/Inf
    encodings in fingerprints.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RankerContractError(f"Non-finite value at {path}")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise RankerContractError(f"Non-finite value at {path}")
        return number
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RankerContractError(f"Non-string mapping key at {path}")
            output[key] = _json_compatible(item, f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RankerContractError(
        f"Unsupported value of type {type(value).__name__} at {path}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode *value* using the registry's canonical JSON convention."""

    compatible = _json_compatible(value)
    return json.dumps(
        compatible,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return an uppercase SHA-256 over :func:`canonical_json_bytes`."""

    return sha256(canonical_json_bytes(value)).hexdigest().upper()


def require_canonical_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or CANONICAL_SHA256_RE.fullmatch(value) is None:
        raise RankerContractError(
            f"{field_name} must be a 64-character uppercase hexadecimal SHA-256"
        )


def normalize_candidate_edges(
    candidate_edges: Sequence[Sequence[int]],
) -> tuple[EDGE, ...]:
    output: list[EDGE] = []
    for index, raw_edge in enumerate(candidate_edges):
        if not isinstance(raw_edge, Sequence) or isinstance(raw_edge, (str, bytes)):
            raise RankerContractError(f"candidate_edges[{index}] must be a sequence")
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            for item in raw_edge
        ):
            raise RankerContractError(
                f"candidate_edges[{index}] must contain integer feature indices"
            )
        edge = tuple(int(item) for item in raw_edge)
        if len(edge) not in {2, 3}:
            raise RankerContractError(
                f"candidate_edges[{index}] must have order two or three"
            )
        if len(set(edge)) != len(edge) or tuple(sorted(edge)) != edge or edge[0] < 0:
            raise RankerContractError(
                f"candidate_edges[{index}] must contain distinct, sorted, non-negative indices"
            )
        output.append(edge)
    if not output:
        raise RankerContractError("candidate_edges must not be empty")
    if len(set(output)) != len(output):
        raise RankerContractError("candidate_edges contains duplicates")
    return tuple(output)


def candidate_universe_sha256(candidate_edges: Sequence[Sequence[int]]) -> str:
    edges = normalize_candidate_edges(candidate_edges)
    return canonical_sha256({"candidate_edges": edges})


def _index_array(
    values: Any, field_name: str, n_rows: int, *, allow_empty: bool
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise RankerContractError(
            f"{field_name} must be a one-dimensional integer array"
        )
    result = raw.astype(np.int64, copy=True)
    if not allow_empty and result.size == 0:
        raise RankerContractError(f"{field_name} must not be empty")
    if len(np.unique(result)) != result.size:
        raise RankerContractError(f"{field_name} contains duplicate indices")
    if result.size and (int(result.min()) < 0 or int(result.max()) >= n_rows):
        raise RankerContractError(f"{field_name} contains an out-of-range index")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RankerFitRequest:
    """One ranker fit with explicit data boundary, seed, phase, and hashes."""

    X: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    candidate_edges: tuple[EDGE, ...]
    fit_indices: np.ndarray
    validation_indices: np.ndarray
    model_seed: int
    phase: FitPhase
    score_family_id: str
    split_plan_sha256: str
    candidate_universe_sha256: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        Z = np.asarray(self.Z)
        y = np.asarray(self.y)
        if X.ndim != 2 or Z.ndim != 2 or y.ndim != 1:
            raise RankerContractError("X and Z must be matrices and y must be a vector")
        if X.shape[0] != Z.shape[0] or X.shape[0] != y.shape[0]:
            raise RankerContractError("X, Z, and y must have the same number of rows")
        edges = normalize_candidate_edges(self.candidate_edges)
        if Z.shape[1] != len(edges):
            raise RankerContractError(
                "Z column count must equal the ordered candidate-edge count"
            )
        if max(max(edge) for edge in edges) >= X.shape[1]:
            raise RankerContractError(
                "candidate_edges contains a feature index outside X"
            )
        fit_indices = _index_array(
            self.fit_indices, "fit_indices", X.shape[0], allow_empty=False
        )
        validation_indices = _index_array(
            self.validation_indices,
            "validation_indices",
            X.shape[0],
            allow_empty=False,
        )
        if np.intersect1d(fit_indices, validation_indices).size:
            raise RankerContractError(
                "fit_indices and validation_indices must be disjoint"
            )
        if isinstance(self.model_seed, bool) or not isinstance(
            self.model_seed, (int, np.integer)
        ):
            raise RankerContractError("model_seed must be a non-negative integer")
        if int(self.model_seed) < 0:
            raise RankerContractError("model_seed must be a non-negative integer")
        if self.phase not in ALLOWED_FIT_PHASES:
            raise RankerContractError(f"Unknown ranker fit phase: {self.phase!r}")
        if (
            not isinstance(self.score_family_id, str)
            or not self.score_family_id.strip()
        ):
            raise RankerContractError("score_family_id must be a non-empty string")
        require_canonical_sha256(self.split_plan_sha256, "split_plan_sha256")
        require_canonical_sha256(
            self.candidate_universe_sha256, "candidate_universe_sha256"
        )
        expected_universe_hash = candidate_universe_sha256(edges)
        if self.candidate_universe_sha256 != expected_universe_hash:
            raise RankerContractError(
                "candidate_universe_sha256 does not bind the ordered candidate_edges"
            )
        if not isinstance(self.config, Mapping):
            raise RankerContractError("config must be a mapping")
        canonical_json_bytes(self.config)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Z", Z)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "candidate_edges", edges)
        object.__setattr__(self, "fit_indices", fit_indices)
        object.__setattr__(self, "validation_indices", validation_indices)
        object.__setattr__(self, "model_seed", int(self.model_seed))
        object.__setattr__(self, "config", dict(self.config))

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.config)


@dataclass(frozen=True)
class ScoreProvenance:
    ranker_id: str
    adapter_id: str
    phase: FitPhase
    score_source: str
    score_direction: Literal["higher_is_better"]
    score_family_id: str
    candidate_universe_sha256: str
    split_plan_sha256: str
    config_sha256: str
    model_seed: int
    zero_score_semantics: ZeroScoreSemantics


@dataclass(frozen=True)
class RankingResult:
    """Ranker output before a support-selection policy is applied."""

    scores: np.ndarray
    ranked_indices: np.ndarray
    active_indices: np.ndarray
    selected_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    internal_fit_count: int = 1
    provenance: ScoreProvenance | None = None


@runtime_checkable
class RankerAdapter(Protocol):
    """The only interface through which a runner may invoke a ranker."""

    def fit(self, request: RankerFitRequest) -> RankingResult: ...


class RankerDescriptor(Protocol):
    ranker_id: str
    adapter_id: str
    score_source: str
    zero_score_semantics: str


def _validated_result_indices(
    values: Any, field_name: str, n_candidates: int
) -> np.ndarray:
    return _index_array(values, field_name, n_candidates, allow_empty=True)


def validate_ranking_result(
    result: RankingResult,
    request: RankerFitRequest,
    ranker: RankerDescriptor,
) -> RankingResult:
    """Validate the complete score vector, ordering, activity, and provenance.

    The function returns *result* only after all invariants pass, making it easy
    for runners to validate inline before handing scores to a policy.
    """

    if not isinstance(result, RankingResult):
        raise RankerContractError("RankerAdapter.fit must return RankingResult")
    n_candidates = len(request.candidate_edges)
    scores = np.asarray(result.scores)
    if scores.ndim != 1 or scores.shape[0] != n_candidates:
        raise RankerContractError(
            "scores must be a complete one-dimensional candidate-wide vector"
        )
    if not (
        np.issubdtype(scores.dtype, np.integer)
        or np.issubdtype(scores.dtype, np.floating)
    ):
        raise RankerContractError("scores must be real-valued numeric data")
    scores = scores.astype(float, copy=False)
    if not np.all(np.isfinite(scores)):
        raise RankerContractError("scores contains a non-finite value")
    if np.any(scores < 0.0):
        raise RankerContractError("ranker scores must be non-negative")

    ranked = _validated_result_indices(
        result.ranked_indices, "ranked_indices", n_candidates
    )
    active = _validated_result_indices(
        result.active_indices, "active_indices", n_candidates
    )
    if set(ranked.tolist()) != set(active.tolist()):
        raise RankerContractError(
            "ranked_indices must contain every active index exactly once"
        )
    if ranked.size > 1 and np.any(np.diff(scores[ranked]) > 0.0):
        raise RankerContractError(
            "ranked_indices is not ordered by non-increasing score"
        )

    semantics = getattr(ranker, "zero_score_semantics", None)
    if semantics not in ALLOWED_ZERO_SCORE_SEMANTICS:
        raise RankerContractError(f"Unknown zero-score semantics: {semantics!r}")
    active_mask = np.zeros(n_candidates, dtype=bool)
    active_mask[active] = True
    if semantics == "dense":
        if not np.all(active_mask):
            raise RankerContractError("dense rankers must mark every candidate active")
    else:
        if np.any(scores[~active_mask] != 0.0):
            raise RankerContractError(
                "inactive_is_zero rankers must score every inactive candidate zero"
            )
        if np.any(scores[active_mask] <= 0.0):
            raise RankerContractError(
                "active candidates must have strictly positive scores"
            )

    if not isinstance(result.selected_hyperparameters, Mapping):
        raise RankerContractError("selected_hyperparameters must be a mapping")
    canonical_json_bytes(result.selected_hyperparameters)
    if not isinstance(result.warnings, tuple) or not all(
        isinstance(item, str) for item in result.warnings
    ):
        raise RankerContractError("warnings must be a tuple of strings")
    if isinstance(result.elapsed_seconds, (bool, np.bool_)) or not isinstance(
        result.elapsed_seconds, (int, float, np.integer, np.floating)
    ):
        raise RankerContractError("elapsed_seconds must be numeric")
    if (
        not math.isfinite(float(result.elapsed_seconds))
        or float(result.elapsed_seconds) < 0.0
    ):
        raise RankerContractError("elapsed_seconds must be finite and non-negative")
    if (
        isinstance(result.internal_fit_count, bool)
        or not isinstance(result.internal_fit_count, (int, np.integer))
        or int(result.internal_fit_count) < 1
    ):
        raise RankerContractError("internal_fit_count must be a positive integer")

    provenance = result.provenance
    if not isinstance(provenance, ScoreProvenance):
        raise RankerContractError("RankingResult requires ScoreProvenance")
    for field_name in (
        "candidate_universe_sha256",
        "split_plan_sha256",
        "config_sha256",
    ):
        require_canonical_sha256(getattr(provenance, field_name), field_name)
    expected_pairs = {
        "ranker_id": getattr(ranker, "ranker_id", None),
        "adapter_id": getattr(ranker, "adapter_id", None),
        "phase": request.phase,
        "score_source": getattr(ranker, "score_source", None),
        "score_direction": "higher_is_better",
        "score_family_id": request.score_family_id,
        "candidate_universe_sha256": request.candidate_universe_sha256,
        "split_plan_sha256": request.split_plan_sha256,
        "config_sha256": request.config_sha256,
        "model_seed": request.model_seed,
        "zero_score_semantics": semantics,
    }
    mismatches = [
        name
        for name, expected in expected_pairs.items()
        if getattr(provenance, name) != expected
    ]
    if mismatches:
        raise RankerContractError(
            "ScoreProvenance mismatch for: " + ", ".join(sorted(mismatches))
        )
    return result
