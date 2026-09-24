"""Registry-driven Track-A cell runner.

This module deliberately exposes a function API only.  It does not bypass the
candidate execution gate, provide a CLI, aggregate partitions, or write result
files.  Its purpose is to make the PR-2 ranker/policy boundary executable and
unit-testable without modifying the legacy runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from method_registry import MethodRegistry, MethodSpec
from ranker_adapters import registry_ranker_adapter
from ranker_protocol import EDGE, RankerAdapter, RankingResult
import sc_shil_experiment as legacy
import support_metrics
import support_policies


class TrackARunnerError(ValueError):
    """Raised when a Track-A cell cannot be executed without ambiguity."""


AdapterFactory = Callable[[MethodRegistry, str], RankerAdapter]


@dataclass(frozen=True)
class SelectionOutcome:
    support_indices: tuple[int, ...]
    score_vector: np.ndarray
    metadata: Mapping[str, Any]
    calibration_rows: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class TrackACellResult:
    registry_sha256: str
    split_plan_sha256: str
    expected_method_ids: tuple[str, ...]
    expected_score_family_ids: tuple[str, ...]
    outer_test_indices: tuple[int, ...]
    metrics: tuple[Mapping[str, Any], ...]
    selected_edges: tuple[Mapping[str, Any], ...]
    ranking_metrics: tuple[Mapping[str, Any], ...]
    ranking_scores: tuple[Mapping[str, Any], ...]
    fit_calls: tuple[Mapping[str, Any], ...]
    calibration_rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        expected_methods = set(self.expected_method_ids)
        if len(expected_methods) != len(self.expected_method_ids):
            raise TrackARunnerError("expected_method_ids contains duplicates")
        for label, rows in (
            ("metrics", self.metrics),
            ("selected_edges", self.selected_edges),
        ):
            observed = [str(row["method_id"]) for row in rows]
            if len(observed) != len(set(observed)) or set(observed) != expected_methods:
                raise TrackARunnerError(
                    f"{label} does not carry the exact registry method set"
                )
        expected_families = set(self.expected_score_family_ids)
        if len(expected_families) != len(self.expected_score_family_ids):
            raise TrackARunnerError("expected_score_family_ids contains duplicates")
        for label, rows in (
            ("ranking_metrics", self.ranking_metrics),
            ("ranking_scores", self.ranking_scores),
        ):
            observed = [str(row["score_family_id"]) for row in rows]
            if (
                len(observed) != len(set(observed))
                or set(observed) != expected_families
            ):
                raise TrackARunnerError(
                    f"{label} must contain exactly one row per score family"
                )
        forbidden = set(self.outer_test_indices)
        for row in self.fit_calls:
            if forbidden & set(row["fit_indices"]):
                raise TrackARunnerError("outer test leaked into a ranker fit partition")
            if forbidden & set(row["validation_indices"]):
                raise TrackARunnerError(
                    "outer test leaked into a ranker validation partition"
                )
        for row in self.calibration_rows:
            if forbidden & set(row["fit_scope_indices"]):
                raise TrackARunnerError("outer test leaked into policy calibration fit")
            if forbidden & set(row["validation_scope_indices"]):
                raise TrackARunnerError(
                    "outer test leaked into policy calibration validation"
                )


def candidate_edges_for_orders(
    n_features: int, candidate_orders: Sequence[int]
) -> tuple[EDGE, ...]:
    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, int)
        or n_features < 3
    ):
        raise TrackARunnerError("n_features must be an integer of at least three")
    orders = tuple(candidate_orders)
    if not orders or len(set(orders)) != len(orders) or set(orders) - {2, 3}:
        raise TrackARunnerError(
            "candidate_orders must contain unique orders from {2, 3}"
        )
    return tuple(
        edge for order in orders for edge in combinations(range(n_features), int(order))
    )


def _validate_resolved_config(
    registry: MethodRegistry,
    resolved_config: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(resolved_config, Mapping):
        raise TrackARunnerError("resolved_config must be a mapping")
    expected = set(registry.config_keys)
    observed = set(resolved_config)
    if observed != expected:
        raise TrackARunnerError(
            f"resolved_config key mismatch: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    if any(not isinstance(value, Mapping) for value in resolved_config.values()):
        raise TrackARunnerError("every resolved config value must be a mapping")


def _outer_scaled_matrix(
    dataset: legacy.Dataset, outer_split_seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    split = legacy.outer_split_scale(dataset, int(outer_split_seed))
    X = np.empty_like(np.asarray(dataset.X), dtype=float)
    for indices_key, matrix_key in (
        ("train_idx", "X_train"),
        ("val_idx", "X_val"),
        ("test_idx", "X_test"),
    ):
        X[np.asarray(split[indices_key], dtype=int)] = np.asarray(
            split[matrix_key], dtype=float
        )
    return X, split


def _indices_from_ranking(result: RankingResult, k: int) -> tuple[int, ...]:
    ranked = tuple(int(value) for value in np.asarray(result.ranked_indices))
    return ranked[: min(int(k), len(ranked))]


def _selection_frequency(
    results: Sequence[RankingResult], q: int, n_candidates: int
) -> np.ndarray:
    if not results:
        raise TrackARunnerError("selection frequency requires at least one ranking")
    if isinstance(q, bool) or not isinstance(q, int) or q < 1:
        raise TrackARunnerError("q must be a positive integer")
    counts = np.zeros(n_candidates, dtype=float)
    for result in results:
        ranked = np.asarray(result.ranked_indices, dtype=int)
        selected = ranked[: min(q, len(ranked))]
        counts[selected] += 1.0
    return counts / len(results)


def _validation_loss(
    X: np.ndarray,
    y: np.ndarray,
    split_plan: support_policies.SplitPlan,
    candidate_edges: tuple[EDGE, ...],
    support_indices: tuple[int, ...],
    interaction_clip: float,
    seed: int,
) -> tuple[float, float]:
    support = tuple(candidate_edges[index] for index in support_indices)
    train = np.asarray(split_plan.outer_train_indices, dtype=int)
    validation = np.asarray(split_plan.outer_validation_indices, dtype=int)
    model = legacy.fit_l2_refit(
        X[train], y[train], support, interaction_clip, int(seed)
    )
    probability = model.predict_proba(
        legacy.refit_design(X[validation], support, interaction_clip)
    )
    losses = legacy.per_observation_log_loss(y[validation], probability, model.classes_)
    mean = float(np.mean(losses))
    standard_error = (
        float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
        if len(losses) > 1
        else 0.0
    )
    return mean, standard_error


def _choose_one_se(
    rows: Sequence[Mapping[str, Any]],
    *,
    best_tie_fields: tuple[str, ...],
    sparse_tie_fields: tuple[str, ...],
) -> dict[str, Any]:
    if not rows:
        raise TrackARunnerError("one-SE calibration path is empty")

    def key(row: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
        output: list[Any] = []
        for field in fields:
            if field.startswith("-"):
                output.append(-float(row[field[1:]]))
            else:
                output.append(row[field])
        return tuple(output)

    best = min(
        rows,
        key=lambda row: (
            float(row["validation_log_loss"]),
            *key(row, best_tie_fields),
        ),
    )
    ceiling = float(best["validation_log_loss"]) + float(best["validation_log_loss_se"])
    eligible = [
        row for row in rows if float(row["validation_log_loss"]) <= ceiling + 1e-12
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            int(row["support_size"]),
            *key(row, sparse_tie_fields),
            tuple(row["support_indices"]),
        ),
    )
    return {**chosen, "one_se_ceiling": ceiling}


def _validation_one_se_outcome(
    execution: support_policies.PolicyFamilyExecution,
    k_grid: tuple[int, ...],
    X: np.ndarray,
    y: np.ndarray,
    candidate_edges: tuple[EDGE, ...],
    interaction_clip: float,
    master_seed: int,
    ranker_id: str,
) -> SelectionOutcome:
    boundary = support_policies.ValidationOneSeHandler().consume(execution)
    if set(boundary.policy_validation_indices) != set(
        execution.call_plan.split_plan.outer_validation_indices
    ):
        raise TrackARunnerError(
            "validation-one-SE boundary is not outer-validation bound"
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for k in k_grid:
        support_indices = _indices_from_ranking(boundary.training_result, k)
        if support_indices in seen:
            continue
        seen.add(support_indices)
        loss, standard_error = _validation_loss(
            X,
            y,
            execution.call_plan.split_plan,
            candidate_edges,
            support_indices,
            interaction_clip,
            support_policies.derive_label_seed(
                master_seed, f"calibration:{ranker_id}:validation:k-{k}"
            ),
        )
        rows.append(
            {
                "ranker_id": ranker_id,
                "policy_id": "validation_one_se",
                "k": int(k),
                "support_size": len(support_indices),
                "support_indices": support_indices,
                "fit_scope_indices": execution.call_plan.split_plan.outer_train_indices,
                "validation_scope_indices": execution.call_plan.split_plan.outer_validation_indices,
                "validation_log_loss": loss,
                "validation_log_loss_se": standard_error,
            }
        )
    chosen = _choose_one_se(
        rows,
        best_tie_fields=("support_size", "k"),
        sparse_tie_fields=("k",),
    )
    chosen_k = int(chosen["k"])
    final_indices = _indices_from_ranking(boundary.final_result, chosen_k)
    marked_rows = tuple(
        {
            **row,
            "chosen": bool(row["support_indices"] == chosen["support_indices"]),
            "one_se_ceiling": float(chosen["one_se_ceiling"]),
        }
        for row in rows
    )
    return SelectionOutcome(
        support_indices=final_indices,
        score_vector=np.asarray(boundary.final_result.scores, dtype=float).copy(),
        metadata={
            "chosen_k": chosen_k,
            "budget_shortfall": max(0, chosen_k - len(final_indices)),
            "zero_padding_applied": False,
            "validation_log_loss": float(chosen["validation_log_loss"]),
            "validation_log_loss_se": float(chosen["validation_log_loss_se"]),
            "one_se_ceiling": float(chosen["one_se_ceiling"]),
        },
        calibration_rows=marked_rows,
    )


def _cpss_one_se_outcome(
    execution: support_policies.PolicyFamilyExecution,
    policy_config: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    candidate_edges: tuple[EDGE, ...],
    interaction_clip: float,
    master_seed: int,
    ranker_id: str,
) -> SelectionOutcome:
    boundary = support_policies.CpssOneSeHandler().consume(execution)
    raw_q_grid = policy_config.get("q_grid")
    raw_pi_grid = policy_config.get("pi_grid")
    if (
        not isinstance(raw_q_grid, (list, tuple))
        or not raw_q_grid
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in raw_q_grid
        )
    ):
        raise TrackARunnerError("CPSS q_grid must contain positive integers")
    if (
        not isinstance(raw_pi_grid, (list, tuple))
        or not raw_pi_grid
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
            for value in raw_pi_grid
        )
    ):
        raise TrackARunnerError("CPSS pi_grid must contain finite values in (0, 1]")
    q_grid = tuple(sorted(set(int(value) for value in raw_q_grid)))
    pi_grid = tuple(sorted(set(float(value) for value in raw_pi_grid), reverse=True))
    n_candidates = len(candidate_edges)
    tuning_frequencies = {
        q: _selection_frequency(boundary.tuning_results, q, n_candidates)
        for q in q_grid
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for q in q_grid:
        for pi in pi_grid:
            support_indices = tuple(
                int(value) for value in np.flatnonzero(tuning_frequencies[q] >= pi)
            )
            if support_indices in seen:
                continue
            seen.add(support_indices)
            loss, standard_error = _validation_loss(
                X,
                y,
                execution.call_plan.split_plan,
                candidate_edges,
                support_indices,
                interaction_clip,
                support_policies.derive_label_seed(
                    master_seed, f"calibration:{ranker_id}:cpss:q-{q}:pi-{pi:.12g}"
                ),
            )
            rows.append(
                {
                    "ranker_id": ranker_id,
                    "policy_id": "cpss_one_se",
                    "q": q,
                    "pi": pi,
                    "support_size": len(support_indices),
                    "support_indices": support_indices,
                    "fit_scope_indices": execution.call_plan.split_plan.outer_train_indices,
                    "validation_scope_indices": execution.call_plan.split_plan.outer_validation_indices,
                    "validation_log_loss": loss,
                    "validation_log_loss_se": standard_error,
                }
            )
    chosen = _choose_one_se(
        rows,
        best_tie_fields=("support_size", "-pi", "q"),
        sparse_tie_fields=("-pi", "q"),
    )
    chosen_q = int(chosen["q"])
    chosen_pi = float(chosen["pi"])
    final_frequency = _selection_frequency(
        boundary.final_results, chosen_q, n_candidates
    )
    final_indices = tuple(
        int(value) for value in np.flatnonzero(final_frequency >= chosen_pi)
    )
    marked_rows = tuple(
        {
            **row,
            "chosen": bool(
                row["q"] == chosen_q
                and row["pi"] == chosen_pi
                and row["support_indices"] == chosen["support_indices"]
            ),
            "one_se_ceiling": float(chosen["one_se_ceiling"]),
        }
        for row in rows
    )
    return SelectionOutcome(
        support_indices=final_indices,
        score_vector=final_frequency,
        metadata={
            "chosen_q": chosen_q,
            "chosen_pi": chosen_pi,
            "tuning_support_size": int(chosen["support_size"]),
            "final_support_size": len(final_indices),
            "zero_padding_applied": False,
            "validation_log_loss": float(chosen["validation_log_loss"]),
            "validation_log_loss_se": float(chosen["validation_log_loss_se"]),
            "one_se_ceiling": float(chosen["one_se_ceiling"]),
        },
        calibration_rows=marked_rows,
    )


def _evaluate_support(
    dataset: legacy.Dataset,
    method: MethodSpec,
    support_indices: tuple[int, ...],
    score_vector: np.ndarray,
    selection_metadata: Mapping[str, Any],
    candidate_edges: tuple[EDGE, ...],
    X: np.ndarray,
    y: np.ndarray,
    split_plan: support_policies.SplitPlan,
    interaction_clip: float,
    master_seed: int,
) -> dict[str, Any]:
    support = tuple(candidate_edges[index] for index in support_indices)
    development = np.asarray(split_plan.outer_development_indices, dtype=int)
    test = np.asarray(split_plan.outer_test_indices, dtype=int)
    model = legacy.fit_l2_refit(
        X[development],
        y[development],
        support,
        interaction_clip,
        support_policies.derive_label_seed(master_seed, f"refit:{method.method_id}"),
    )
    probability = model.predict_proba(
        legacy.refit_design(X[test], support, interaction_clip)
    )
    return {
        "status": "OK",
        "method_id": method.method_id,
        "ranker_id": method.ranker_id,
        "policy_id": method.policy_id,
        "score_family_id": method.score_family_id,
        "selected_edge_count": len(support),
        "candidate_average_precision": support_metrics.candidate_wide_average_precision(
            candidate_edges,
            score_vector,
            dataset.true_edges,
            score_direction="higher_is_better",
        ),
        **legacy.predictive_metrics(y[test], probability, model.classes_),
        **support_metrics.support_metrics(
            support, dataset.true_edges, dataset.equivalence_groups
        ),
        **dict(selection_metadata),
    }


def run_track_a_cell(
    dataset: legacy.Dataset,
    *,
    registry: MethodRegistry,
    resolved_config: Mapping[str, Mapping[str, Any]],
    cell_metadata: Mapping[str, Any],
    interaction_clip: float,
    outer_split_seed: int,
    master_seed: int,
    n_pairs: int,
    adapter_factory: AdapterFactory = registry_ranker_adapter,
) -> TrackACellResult:
    """Execute one in-memory Track-A cell without file or CLI side effects."""

    _validate_resolved_config(registry, resolved_config)
    if not math.isfinite(float(interaction_clip)) or float(interaction_clip) <= 0.0:
        raise TrackARunnerError("interaction_clip must be finite and positive")
    expected_method_ids = registry.expected_methods_for(cell_metadata)
    if len(expected_method_ids) != 14:
        raise TrackARunnerError(
            "Track-A full-grid cell must resolve to exactly 14 registry methods"
        )
    methods = tuple(
        registry.method_by_id[method_id] for method_id in expected_method_ids
    )
    candidate_orders = tuple(cell_metadata["candidate_orders"])
    candidate_edges = candidate_edges_for_orders(
        int(np.asarray(dataset.X).shape[1]), candidate_orders
    )
    X, split = _outer_scaled_matrix(dataset, int(outer_split_seed))
    y = np.asarray(dataset.y)
    split_plan = support_policies.build_split_plan(
        y,
        outer_train_indices=split["train_idx"],
        outer_validation_indices=split["val_idx"],
        outer_test_indices=split["test_idx"],
        n_pairs=int(n_pairs),
        master_seed=int(master_seed),
    )
    call_plan = support_policies.PolicyFamilyCallPlan.from_split_plan(split_plan)
    Z = legacy.interaction_matrix(X, candidate_edges, float(interaction_clip))

    fixed_policy = next(
        policy for policy in registry.policies if policy.policy_id == "fixed_k"
    )
    cpss_policy = next(
        policy for policy in registry.policies if policy.policy_id == "cpss_one_se"
    )
    k_grid = tuple(int(value) for value in fixed_policy.allowed_k)
    outcome_by_method: dict[str, SelectionOutcome] = {}
    family_scores: dict[str, np.ndarray] = {}
    raw_reference_by_ranker: dict[str, RankingResult] = {}
    fit_calls: list[Mapping[str, Any]] = []
    calibration_rows: list[Mapping[str, Any]] = []

    def register_family(family_id: str, values: np.ndarray) -> None:
        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1 or vector.shape[0] != len(candidate_edges):
            raise TrackARunnerError(f"score family {family_id} is not candidate-wide")
        existing = family_scores.get(family_id)
        if existing is not None and not np.array_equal(existing, vector):
            raise TrackARunnerError(
                f"score family {family_id} received inconsistent score vectors"
            )
        family_scores[family_id] = vector.copy()

    for ranker in registry.rankers:
        adapter = adapter_factory(registry, ranker.ranker_id)
        execution = support_policies.execute_policy_family(
            adapter,
            ranker,
            call_plan,
            support_policies.SeedPlan.for_ranker(
                int(master_seed), ranker.ranker_id, split_plan
            ),
            X=X,
            Z=Z,
            y=y,
            candidate_edges=candidate_edges,
            config=resolved_config[ranker.config_key],
        )
        if execution.call_count != 4 * int(n_pairs) + 2:
            raise TrackARunnerError("ranker execution did not use exactly 4B+2 calls")
        request_by_id = dict(execution.requests)
        for call_id, result in execution.results:
            request = request_by_id[call_id]
            fit_calls.append(
                {
                    "ranker_id": ranker.ranker_id,
                    "call_id": call_id,
                    "phase": request.phase,
                    "score_family_id": request.score_family_id,
                    "fit_indices": tuple(int(value) for value in request.fit_indices),
                    "validation_indices": tuple(
                        int(value) for value in request.validation_indices
                    ),
                    "model_seed": request.model_seed,
                    "internal_fit_count": result.internal_fit_count,
                }
            )

        ranker_methods = tuple(
            method for method in methods if method.ranker_id == ranker.ranker_id
        )
        final_raw = execution.result(call_plan.final_development_anchor_call.call_id)
        raw_reference_by_ranker[ranker.ranker_id] = final_raw
        validation_outcome = _validation_one_se_outcome(
            execution,
            k_grid,
            X,
            y,
            candidate_edges,
            float(interaction_clip),
            int(master_seed),
            ranker.ranker_id,
        )
        cpss_outcome = _cpss_one_se_outcome(
            execution,
            resolved_config[cpss_policy.config_key],
            X,
            y,
            candidate_edges,
            float(interaction_clip),
            int(master_seed),
            ranker.ranker_id,
        )
        calibration_rows.extend(validation_outcome.calibration_rows)
        calibration_rows.extend(cpss_outcome.calibration_rows)
        for method in ranker_methods:
            if method.policy_id == "fixed_k":
                fixed = support_policies.FixedKHandler(
                    int(method.policy_parameters["k"])
                ).consume(execution)
                outcome = SelectionOutcome(
                    support_indices=fixed.support_indices,
                    score_vector=np.asarray(final_raw.scores, dtype=float).copy(),
                    metadata={
                        "requested_k": fixed.requested_budget,
                        "budget_shortfall": fixed.budget_shortfall,
                        "zero_padding_applied": fixed.zero_padding_applied,
                    },
                )
            elif method.policy_id == "validation_one_se":
                outcome = validation_outcome
            elif method.policy_id == "cpss_one_se":
                outcome = cpss_outcome
            else:
                raise TrackARunnerError(f"Unknown policy_id: {method.policy_id!r}")
            outcome_by_method[method.method_id] = outcome
            register_family(method.score_family_id, outcome.score_vector)

    expected_family_ids = tuple(
        dict.fromkeys(method.score_family_id for method in methods)
    )
    if set(family_scores) != set(expected_family_ids):
        raise TrackARunnerError("produced score-family set differs from registry")

    metrics: list[Mapping[str, Any]] = []
    selected_edges: list[Mapping[str, Any]] = []
    for method in methods:
        outcome = outcome_by_method[method.method_id]
        metrics.append(
            _evaluate_support(
                dataset,
                method,
                outcome.support_indices,
                outcome.score_vector,
                outcome.metadata,
                candidate_edges,
                X,
                y,
                split_plan,
                float(interaction_clip),
                int(master_seed),
            )
        )
        selected_edges.append(
            {
                "method_id": method.method_id,
                "ranker_id": method.ranker_id,
                "policy_id": method.policy_id,
                "score_family_id": method.score_family_id,
                "support_indices": outcome.support_indices,
                "edges": tuple(
                    candidate_edges[index] for index in outcome.support_indices
                ),
            }
        )

    ranking_metrics: list[Mapping[str, Any]] = []
    ranking_scores: list[Mapping[str, Any]] = []
    family_signature = {
        method.score_family_id: (method.ranker_id, method.score_source)
        for method in methods
    }
    for family_id in expected_family_ids:
        ranker_id, score_source = family_signature[family_id]
        vector = family_scores[family_id]
        ranking_metrics.append(
            {
                "status": "OK",
                "score_family_id": family_id,
                "ranker_id": ranker_id,
                "score_source": score_source,
                "candidate_count": len(candidate_edges),
                "nonzero_score_count": int(np.count_nonzero(vector)),
                "candidate_average_precision": support_metrics.candidate_wide_average_precision(
                    candidate_edges,
                    vector,
                    dataset.true_edges,
                    score_direction="higher_is_better",
                ),
            }
        )
        ranking_scores.append(
            {
                "status": "OK",
                "score_family_id": family_id,
                "ranker_id": ranker_id,
                "score_source": score_source,
                "candidate_edges": candidate_edges,
                "scores": tuple(float(value) for value in vector),
                "score_direction": "higher_is_better",
                "ranked_indices": (
                    tuple(
                        int(value)
                        for value in raw_reference_by_ranker[ranker_id].ranked_indices
                    )
                    if score_source == "final_ranker_score"
                    else ()
                ),
                "active_indices": (
                    tuple(
                        int(value)
                        for value in raw_reference_by_ranker[ranker_id].active_indices
                    )
                    if score_source == "final_ranker_score"
                    else ()
                ),
            }
        )

    return TrackACellResult(
        registry_sha256=registry.canonical_sha256,
        split_plan_sha256=split_plan.split_plan_sha256,
        expected_method_ids=expected_method_ids,
        expected_score_family_ids=expected_family_ids,
        outer_test_indices=split_plan.outer_test_indices,
        metrics=tuple(metrics),
        selected_edges=tuple(selected_edges),
        ranking_metrics=tuple(ranking_metrics),
        ranking_scores=tuple(ranking_scores),
        fit_calls=tuple(fit_calls),
        calibration_rows=tuple(calibration_rows),
    )
