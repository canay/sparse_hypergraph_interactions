"""Production adapters around the unchanged legacy SHIL and L1 rankers.

The adapter layer owns protocol conversion and provenance only.  Ranking,
hyperparameter selection, warnings, and elapsed time remain outputs of the
existing functions in :mod:`sc_shil_experiment`.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Callable, Mapping

import numpy as np

from method_registry import MethodRegistry, RankerSpec
from ranker_protocol import (
    RankerAdapter,
    RankerContractError,
    RankerFitRequest,
    RankingResult,
    ScoreProvenance,
    validate_ranking_result,
)
import sc_shil_experiment as legacy


class RankerAdapterFactoryError(ValueError):
    """Raised when a registry ranker or adapter implementation is unknown."""


def _validate_spec(
    spec: RankerSpec,
    *,
    ranker_id: str,
    adapter_id: str,
    score_source: str,
    zero_score_semantics: str,
) -> None:
    expected = {
        "ranker_id": ranker_id,
        "adapter_id": adapter_id,
        "score_source": score_source,
        "score_direction": "higher_is_better",
        "zero_score_semantics": zero_score_semantics,
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(spec, field_name) != expected_value
    ]
    if mismatches:
        raise RankerAdapterFactoryError(
            f"RankerSpec is incompatible with {adapter_id}: "
            + ", ".join(sorted(mismatches))
        )


def _provenance(request: RankerFitRequest, spec: RankerSpec) -> ScoreProvenance:
    return ScoreProvenance(
        ranker_id=spec.ranker_id,
        adapter_id=spec.adapter_id,
        phase=request.phase,
        score_source=spec.score_source,
        score_direction="higher_is_better",
        score_family_id=request.score_family_id,
        candidate_universe_sha256=request.candidate_universe_sha256,
        split_plan_sha256=request.split_plan_sha256,
        config_sha256=request.config_sha256,
        model_seed=request.model_seed,
        zero_score_semantics=spec.zero_score_semantics,
    )


def _converted_result(
    raw: legacy.RankingResult,
    request: RankerFitRequest,
    spec: RankerSpec,
    *,
    selected_hyperparameters: Mapping[str, object],
    internal_fit_count: int,
) -> RankingResult:
    result = RankingResult(
        scores=np.asarray(raw.scores),
        ranked_indices=np.asarray(raw.ranked_indices),
        active_indices=np.asarray(raw.active_indices),
        selected_hyperparameters=dict(selected_hyperparameters),
        warnings=tuple(raw.warnings),
        elapsed_seconds=float(raw.elapsed_seconds),
        internal_fit_count=internal_fit_count,
        provenance=_provenance(request, spec),
    )
    return validate_ranking_result(result, request, spec)


class SHILRankerAdapter:
    """Protocol adapter for ``sc_shil_experiment.fit_shil_ranking``."""

    def __init__(self, spec: RankerSpec) -> None:
        _validate_spec(
            spec,
            ranker_id="shil",
            adapter_id="shil_v1",
            score_source="gate_probability_x_edge_coefficient_l2",
            zero_score_semantics="dense",
        )
        self.spec = spec

    def fit(self, request: RankerFitRequest) -> RankingResult:
        required = {"epochs", "learning_rate", "gate_l1", "l2", "patience"}
        missing = required - set(request.config)
        if missing:
            raise RankerContractError(
                "SHIL config is missing required fields: " + ", ".join(sorted(missing))
            )
        raw = legacy.fit_shil_ranking(
            request.X,
            request.Z,
            request.y,
            request.fit_indices,
            request.validation_indices,
            request.model_seed,
            dict(request.config),
        )
        if raw.selected_c is not None:
            raise RankerContractError("Legacy SHIL unexpectedly returned selected_c")
        return _converted_result(
            raw,
            request,
            self.spec,
            selected_hyperparameters={},
            internal_fit_count=1,
        )


class L1RankerAdapter:
    """Protocol adapter for ``sc_shil_experiment.fit_l1_ranking``."""

    def __init__(self, spec: RankerSpec) -> None:
        _validate_spec(
            spec,
            ranker_id="l1",
            adapter_id="l1_logistic_v1",
            score_source="edge_class_coefficient_l2",
            zero_score_semantics="inactive_is_zero",
        )
        self.spec = spec

    def fit(self, request: RankerFitRequest) -> RankingResult:
        if set(request.config) != {"c_grid"}:
            raise RankerContractError(
                "L1 adapter config must contain exactly the c_grid field"
            )
        raw_grid = request.config["c_grid"]
        if not isinstance(raw_grid, (list, tuple)) or not raw_grid:
            raise RankerContractError("L1 c_grid must be a non-empty list or tuple")
        c_grid: list[float] = []
        for index, value in enumerate(raw_grid):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise RankerContractError(
                    f"L1 c_grid[{index}] must be a finite positive number"
                )
            c_grid.append(float(value))
        if len(set(c_grid)) != len(c_grid):
            raise RankerContractError("L1 c_grid contains duplicate values")
        raw = legacy.fit_l1_ranking(
            request.X,
            request.Z,
            request.y,
            request.fit_indices,
            request.validation_indices,
            request.model_seed,
            c_grid,
        )
        if raw.selected_c is None:
            raise RankerContractError("Legacy L1 did not return selected_c")
        return _converted_result(
            raw,
            request,
            self.spec,
            selected_hyperparameters={"C": float(raw.selected_c)},
            internal_fit_count=len(c_grid) + 1,
        )


# --------------------------------------------------------------------------
# MCH-SHIL-004 v6 ranker menu extension.
#
# Operation ID: shil-claude-mch004-v6-ranker-menu-20260903
#
# The two adapters below add scoring principles that the v5 menu did not
# contain.  SHIL and L1 both rank coefficient magnitude on the same explicit
# product design, which is why CPSS reproduced the raw top-K support in 258 of
# 260 non-empty ranker cells: a resampling policy can only move the support by
# reordering, and a stable coefficient ordering leaves nothing to reorder.
#
# `tree` scores joint nonlinear importance; `screen` scores marginal
# association.  Neither is a coefficient magnitude.
#
# Determinism note, and it is a scientific point rather than an engineering
# one: `screen` uses the ANOVA F statistic, which is deterministic given the
# rows.  A stochastic screening score such as mutual information would make
# CPSS move for a reason that belongs to the random number generator instead of
# to the data, and the pilot would then measure its own noise.
# --------------------------------------------------------------------------


class TreeRankerAdapter:
    """Random-forest impurity importance over the shared [X, Z] design."""

    def __init__(self, spec: RankerSpec) -> None:
        _validate_spec(
            spec,
            ranker_id="tree",
            adapter_id="tree_forest_v1",
            score_source="random_forest_impurity_importance",
            zero_score_semantics="dense",
        )
        self.spec = spec

    def fit(self, request: RankerFitRequest) -> RankingResult:
        from sklearn.ensemble import RandomForestClassifier

        required = {"n_estimators", "max_features", "min_samples_leaf"}
        missing = required - set(request.config)
        if missing:
            raise RankerContractError(
                "tree config is missing required fields: " + ", ".join(sorted(missing))
            )
        started = time.perf_counter()
        X = np.asarray(request.X)
        Z = np.asarray(request.Z)
        design = np.c_[X, Z]
        n_main = X.shape[1]
        n_candidates = Z.shape[1]
        model = RandomForestClassifier(
            n_estimators=int(request.config["n_estimators"]),
            max_features=request.config["max_features"],
            min_samples_leaf=int(request.config["min_samples_leaf"]),
            random_state=int(request.model_seed),
            n_jobs=1,
        )
        model.fit(design[request.fit_indices], np.asarray(request.y)[request.fit_indices])
        importance = np.asarray(model.feature_importances_, dtype=float)
        scores = importance[n_main : n_main + n_candidates]
        # Impurity importance is non-negative by construction; clip only the
        # floating-point dust that would otherwise trip the contract.
        scores = np.where(scores < 0.0, 0.0, scores)
        if not np.all(np.isfinite(scores)):
            raise RankerContractError("tree produced a non-finite importance")
        # Frozen deterministic tie-break: descending score, then candidate index.
        ranked = np.argsort(-scores, kind="stable")
        active = np.arange(n_candidates)
        elapsed = time.perf_counter() - started
        result = RankingResult(
            scores=scores,
            ranked_indices=ranked,
            active_indices=active,
            selected_hyperparameters={},
            warnings=(),
            elapsed_seconds=float(elapsed),
            internal_fit_count=1,
            provenance=_provenance(request, self.spec),
        )
        return validate_ranking_result(result, request, self.spec)


class ScreenRankerAdapter:
    """Deterministic univariate ANOVA F screening of the candidate columns."""

    def __init__(self, spec: RankerSpec) -> None:
        _validate_spec(
            spec,
            ranker_id="screen",
            adapter_id="screen_anova_v1",
            score_source="anova_f_statistic",
            zero_score_semantics="dense",
        )
        self.spec = spec

    def fit(self, request: RankerFitRequest) -> RankingResult:
        from sklearn.feature_selection import f_classif

        if set(request.config):
            raise RankerContractError("screen adapter takes no configurable fields")
        started = time.perf_counter()
        Z = np.asarray(request.Z, dtype=float)
        y = np.asarray(request.y)
        n_candidates = Z.shape[1]
        with warnings.catch_warnings():
            # A constant candidate column yields a 0/0 F statistic.  That is a
            # real property of the column, not a failure, and it is scored zero
            # below rather than allowed to propagate as NaN.
            warnings.simplefilter("ignore")
            statistic, _ = f_classif(Z[request.fit_indices], y[request.fit_indices])
        scores = np.asarray(statistic, dtype=float)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.where(scores < 0.0, 0.0, scores)
        ranked = np.argsort(-scores, kind="stable")
        active = np.arange(n_candidates)
        elapsed = time.perf_counter() - started
        result = RankingResult(
            scores=scores,
            ranked_indices=ranked,
            active_indices=active,
            selected_hyperparameters={},
            warnings=(),
            elapsed_seconds=float(elapsed),
            internal_fit_count=1,
            provenance=_provenance(request, self.spec),
        )
        return validate_ranking_result(result, request, self.spec)


AdapterBuilder = Callable[[RankerSpec], RankerAdapter]
ADAPTER_BUILDERS: Mapping[str, AdapterBuilder] = {
    "shil_v1": SHILRankerAdapter,
    "l1_logistic_v1": L1RankerAdapter,
    "tree_forest_v1": TreeRankerAdapter,
    "screen_anova_v1": ScreenRankerAdapter,
}


def create_ranker_adapter(spec: RankerSpec) -> RankerAdapter:
    """Create the implementation named by a validated ``RankerSpec``."""

    builder = ADAPTER_BUILDERS.get(spec.adapter_id)
    if builder is None:
        raise RankerAdapterFactoryError(f"Unknown adapter_id: {spec.adapter_id!r}")
    adapter = builder(spec)
    if not isinstance(adapter, RankerAdapter):
        raise RankerAdapterFactoryError(
            f"Adapter {spec.adapter_id!r} does not satisfy RankerAdapter"
        )
    return adapter


def registry_ranker_adapter(registry: MethodRegistry, ranker_id: str) -> RankerAdapter:
    """Resolve a ranker ID through the registry; unknown IDs never default."""

    spec = registry.ranker_by_id.get(ranker_id)
    if spec is None:
        raise RankerAdapterFactoryError(f"Unknown ranker_id: {ranker_id!r}")
    return create_ranker_adapter(spec)
