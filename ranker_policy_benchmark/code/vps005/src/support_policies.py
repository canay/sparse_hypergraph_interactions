"""Ranker-independent split and shared support-policy call boundaries.

The complete support-policy family is evaluated from one shared call plan per
ranker. Fixed-K and validation-one-SE consume the two raw-score anchors;
CPSS consumes only its complementary-half fits. Policy handlers never fit a
ranker themselves, preventing accidental double counting of the 4B+2 budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ranker_protocol import (
    EDGE,
    FitPhase,
    RankerAdapter,
    RankerDescriptor,
    RankerFitRequest,
    RankingResult,
    candidate_universe_sha256,
    canonical_sha256,
    normalize_candidate_edges,
    validate_ranking_result,
)


INDEX = tuple[int, ...]
FIXED_K_GRID = (1, 2, 4, 8, 16)


def derive_label_seed(master_seed: int, label: str) -> int:
    """Derive a stable, namespaced uint32 seed with SHA-256."""

    if int(master_seed) < 0:
        raise ValueError("master_seed must be non-negative")
    if not str(label).strip():
        raise ValueError("seed label must be non-empty")
    payload = f"sha256-label-seed-v1\n{int(master_seed)}\n{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _canonical_indices(values: INDEX | list[int] | np.ndarray) -> INDEX:
    indices = tuple(sorted(int(value) for value in values))
    if any(value < 0 for value in indices):
        raise ValueError("indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    return indices


@dataclass(frozen=True)
class FitPartition:
    """One sampled row universe and its non-empty inner fit/validation split."""

    label: str
    sample_indices: INDEX
    fit_indices: INDEX
    validation_indices: INDEX

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("partition label must be non-empty")
        if self.sample_indices != _canonical_indices(self.sample_indices):
            raise ValueError("sample_indices must be canonical sorted indices")
        if self.fit_indices != _canonical_indices(self.fit_indices):
            raise ValueError("fit_indices must be canonical sorted indices")
        if self.validation_indices != _canonical_indices(self.validation_indices):
            raise ValueError("validation_indices must be canonical sorted indices")
        if (
            not self.sample_indices
            or not self.fit_indices
            or not self.validation_indices
        ):
            raise ValueError(
                "sample, fit, and validation indices must all be non-empty"
            )
        if set(self.fit_indices) & set(self.validation_indices):
            raise ValueError("fit and validation indices must be disjoint")
        if set(self.fit_indices) | set(self.validation_indices) != set(
            self.sample_indices
        ):
            raise ValueError("fit and validation indices must partition sample_indices")


def _partition_payload(partition: FitPartition) -> dict[str, Any]:
    return {
        "label": partition.label,
        "sample_indices": list(partition.sample_indices),
        "fit_indices": list(partition.fit_indices),
        "validation_indices": list(partition.validation_indices),
    }


@dataclass(frozen=True)
class SplitPlan:
    """Canonical row boundaries shared unchanged by every ranker."""

    split_seed: int
    n_pairs: int
    outer_train_indices: INDEX
    outer_validation_indices: INDEX
    outer_test_indices: INDEX
    training_anchor: FitPartition
    final_development_anchor: FitPartition
    cpss_tuning_partitions: tuple[FitPartition, ...]
    cpss_final_partitions: tuple[FitPartition, ...]

    def __post_init__(self) -> None:
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be at least one")
        train = set(self.outer_train_indices)
        validation = set(self.outer_validation_indices)
        test = set(self.outer_test_indices)
        if not train or not validation or not test:
            raise ValueError(
                "outer train, validation, and test splits must be non-empty"
            )
        if train & validation or train & test or validation & test:
            raise ValueError(
                "outer train, validation, and test splits must be disjoint"
            )
        for values, label in (
            (self.outer_train_indices, "outer_train_indices"),
            (self.outer_validation_indices, "outer_validation_indices"),
            (self.outer_test_indices, "outer_test_indices"),
        ):
            if values != _canonical_indices(values):
                raise ValueError(f"{label} must be canonical sorted indices")

        development = train | validation
        self._validate_anchor(self.training_anchor, train, test, "training anchor")
        self._validate_anchor(
            self.final_development_anchor,
            development,
            test,
            "final-development anchor",
        )
        if len(self.cpss_tuning_partitions) != 2 * self.n_pairs:
            raise ValueError("cpss_tuning_partitions must contain two halves per pair")
        if len(self.cpss_final_partitions) != 2 * self.n_pairs:
            raise ValueError("cpss_final_partitions must contain two halves per pair")
        self._validate_cpss_partitions(
            self.cpss_tuning_partitions, train, test, "CPSS tuning"
        )
        self._validate_cpss_partitions(
            self.cpss_final_partitions, development, test, "CPSS final"
        )

    @staticmethod
    def _validate_anchor(
        partition: FitPartition,
        allowed: set[int],
        outer_test: set[int],
        phase: str,
    ) -> None:
        sample = set(partition.sample_indices)
        if sample & outer_test:
            raise ValueError("outer-test leakage detected in policy partition")
        if sample != allowed:
            raise ValueError(f"{phase} must cover its full allowed split")

    @classmethod
    def _validate_cpss_partitions(
        cls,
        partitions: tuple[FitPartition, ...],
        allowed: set[int],
        outer_test: set[int],
        phase: str,
    ) -> None:
        labels = [partition.label for partition in partitions]
        if len(set(labels)) != len(labels):
            raise ValueError(f"{phase} partition labels must be unique")
        for partition in partitions:
            sample = set(partition.sample_indices)
            if sample & outer_test:
                raise ValueError("outer-test leakage detected in policy partition")
            if not sample < allowed:
                raise ValueError(f"{phase} sample must be a strict subset of its split")
        for pair_index in range(len(partitions) // 2):
            left = set(partitions[2 * pair_index].sample_indices)
            right = set(partitions[2 * pair_index + 1].sample_indices)
            if left & right or left | right != allowed:
                raise ValueError(
                    f"{phase} pair {pair_index} halves must be disjoint and complete"
                )

    @property
    def outer_development_indices(self) -> INDEX:
        return tuple(sorted(self.outer_train_indices + self.outer_validation_indices))

    @property
    def split_plan_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": "ranker-independent-split-plan-v2",
                "split_seed": int(self.split_seed),
                "n_pairs": int(self.n_pairs),
                "outer_train_indices": list(self.outer_train_indices),
                "outer_validation_indices": list(self.outer_validation_indices),
                "outer_test_indices": list(self.outer_test_indices),
                "training_anchor": _partition_payload(self.training_anchor),
                "final_development_anchor": _partition_payload(
                    self.final_development_anchor
                ),
                "cpss_tuning_partitions": [
                    _partition_payload(partition)
                    for partition in self.cpss_tuning_partitions
                ],
                "cpss_final_partitions": [
                    _partition_payload(partition)
                    for partition in self.cpss_final_partitions
                ],
            }
        )

    def assert_no_outer_test(self, *partitions: FitPartition) -> None:
        forbidden = set(self.outer_test_indices)
        for partition in partitions:
            if set(partition.sample_indices) & forbidden:
                raise ValueError("outer-test leakage detected in ranker call")


def _stratified_inner_partition(
    y: np.ndarray,
    universe: INDEX,
    *,
    split_seed: int,
    label: str,
) -> FitPartition:
    """Split one sampled universe without exposing any rows outside it."""

    universe_array = np.asarray(universe, dtype=int)
    if universe_array.size < 2:
        raise ValueError("an anchor split requires at least two rows")
    sample_labels = np.asarray(y)[universe_array]
    fit: list[int] = []
    validation: list[int] = []
    for class_value in np.unique(sample_labels):
        class_indices = universe_array[sample_labels == class_value].copy()
        rng = np.random.default_rng(
            derive_label_seed(split_seed, f"inner:{label}:class-{class_value!s}")
        )
        rng.shuffle(class_indices)
        if class_indices.size == 1:
            fit.extend(int(value) for value in class_indices)
            continue
        n_validation = min(
            class_indices.size - 1,
            max(1, int(round(class_indices.size * 0.25))),
        )
        validation.extend(int(value) for value in class_indices[:n_validation])
        fit.extend(int(value) for value in class_indices[n_validation:])
    if not validation:
        validation.append(fit.pop())
    if not fit:
        fit.append(validation.pop())
    return FitPartition(
        label=label,
        sample_indices=_canonical_indices(universe),
        fit_indices=_canonical_indices(fit),
        validation_indices=_canonical_indices(validation),
    )


def _complementary_partitions(
    y: np.ndarray,
    universe: INDEX,
    *,
    n_pairs: int,
    split_seed: int,
    phase: str,
) -> tuple[FitPartition, ...]:
    universe_array = np.asarray(universe, dtype=int)
    labels = np.asarray(y)[universe_array]
    partitions: list[FitPartition] = []
    for pair_index in range(int(n_pairs)):
        left: list[int] = []
        right: list[int] = []
        for class_value in np.unique(labels):
            class_indices = universe_array[labels == class_value].copy()
            rng = np.random.default_rng(
                derive_label_seed(
                    split_seed,
                    f"{phase}:pair-{pair_index}:class-{class_value!s}",
                )
            )
            rng.shuffle(class_indices)
            cut = (len(class_indices) + 1) // 2
            left.extend(int(value) for value in class_indices[:cut])
            right.extend(int(value) for value in class_indices[cut:])
        left_indices = _canonical_indices(left)
        right_indices = _canonical_indices(right)
        if not left_indices or not right_indices:
            raise ValueError("each complementary half must contain at least one row")
        partitions.extend(
            (
                _stratified_inner_partition(
                    y,
                    left_indices,
                    split_seed=split_seed,
                    label=f"{phase}:pair-{pair_index}:left",
                ),
                _stratified_inner_partition(
                    y,
                    right_indices,
                    split_seed=split_seed,
                    label=f"{phase}:pair-{pair_index}:right",
                ),
            )
        )
    return tuple(partitions)


def build_split_plan(
    y: np.ndarray,
    *,
    outer_train_indices: INDEX | list[int] | np.ndarray,
    outer_validation_indices: INDEX | list[int] | np.ndarray,
    outer_test_indices: INDEX | list[int] | np.ndarray,
    n_pairs: int,
    master_seed: int,
) -> SplitPlan:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    train = _canonical_indices(outer_train_indices)
    validation = _canonical_indices(outer_validation_indices)
    test = _canonical_indices(outer_test_indices)
    all_indices = train + validation + test
    if all_indices and max(all_indices) >= len(labels):
        raise ValueError("split index exceeds y length")
    split_seed = derive_label_seed(master_seed, "split-plan")
    development = tuple(sorted(train + validation))
    return SplitPlan(
        split_seed=split_seed,
        n_pairs=int(n_pairs),
        outer_train_indices=train,
        outer_validation_indices=validation,
        outer_test_indices=test,
        training_anchor=_stratified_inner_partition(
            labels,
            train,
            split_seed=split_seed,
            label="training-anchor",
        ),
        final_development_anchor=_stratified_inner_partition(
            labels,
            development,
            split_seed=split_seed,
            label="final-development-anchor",
        ),
        cpss_tuning_partitions=_complementary_partitions(
            labels,
            train,
            n_pairs=int(n_pairs),
            split_seed=split_seed,
            phase="cpss-tuning",
        ),
        cpss_final_partitions=_complementary_partitions(
            labels,
            development,
            n_pairs=int(n_pairs),
            split_seed=split_seed,
            phase="cpss-final",
        ),
    )


@dataclass(frozen=True)
class SeedPlan:
    master_seed: int
    ranker_id: str
    split_plan_sha256: str

    @classmethod
    def for_ranker(
        cls, master_seed: int, ranker_id: str, split_plan: SplitPlan
    ) -> "SeedPlan":
        if not ranker_id.strip():
            raise ValueError("ranker_id must be non-empty")
        return cls(
            master_seed=int(master_seed),
            ranker_id=ranker_id,
            split_plan_sha256=split_plan.split_plan_sha256,
        )

    def seed_for(self, call_label: str) -> int:
        return derive_label_seed(
            self.master_seed, f"model:{self.ranker_id}:{call_label}"
        )


@dataclass(frozen=True)
class FamilyCallSpec:
    call_id: str
    phase: FitPhase
    partition: FitPartition
    score_family_id: str

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.score_family_id.strip():
            raise ValueError("call_id and score_family_id must be non-empty")


@dataclass(frozen=True)
class PolicyFamilyCallPlan:
    """Exactly two raw anchors plus the 4B CPSS complementary-half fits."""

    split_plan: SplitPlan
    training_anchor_call: FamilyCallSpec
    final_development_anchor_call: FamilyCallSpec
    cpss_tuning_calls: tuple[FamilyCallSpec, ...]
    cpss_final_calls: tuple[FamilyCallSpec, ...]

    def __post_init__(self) -> None:
        if self.training_anchor_call.phase != "training_rank":
            raise ValueError("training anchor call must use training_rank phase")
        if self.final_development_anchor_call.phase != "final_development_rank":
            raise ValueError("final anchor call must use final_development_rank phase")
        if self.training_anchor_call.partition != self.split_plan.training_anchor:
            raise ValueError("training anchor call is not bound to SplitPlan")
        if (
            self.final_development_anchor_call.partition
            != self.split_plan.final_development_anchor
        ):
            raise ValueError("final anchor call is not bound to SplitPlan")
        if len(self.cpss_tuning_calls) != 2 * self.split_plan.n_pairs:
            raise ValueError("call plan requires exactly 2B CPSS tuning calls")
        if len(self.cpss_final_calls) != 2 * self.split_plan.n_pairs:
            raise ValueError("call plan requires exactly 2B CPSS final calls")
        if any(call.phase != "cpss_tuning" for call in self.cpss_tuning_calls):
            raise ValueError("CPSS tuning calls must use cpss_tuning phase")
        if any(call.phase != "cpss_final" for call in self.cpss_final_calls):
            raise ValueError("CPSS final calls must use cpss_final phase")
        if tuple(call.partition for call in self.cpss_tuning_calls) != (
            self.split_plan.cpss_tuning_partitions
        ):
            raise ValueError("CPSS tuning calls are not bound to SplitPlan")
        if tuple(call.partition for call in self.cpss_final_calls) != (
            self.split_plan.cpss_final_partitions
        ):
            raise ValueError("CPSS final calls are not bound to SplitPlan")
        calls = self.all_calls
        if len(calls) != 4 * self.split_plan.n_pairs + 2:
            raise ValueError("policy-family call plan must contain exactly 4B+2 calls")
        if len({call.call_id for call in calls}) != len(calls):
            raise ValueError("policy-family call IDs must be unique")

    @classmethod
    def from_split_plan(cls, split_plan: SplitPlan) -> "PolicyFamilyCallPlan":
        tuning_calls = tuple(
            FamilyCallSpec(
                call_id=f"cpss_tuning_{index:03d}",
                phase="cpss_tuning",
                partition=partition,
                score_family_id="cpss-tuning",
            )
            for index, partition in enumerate(split_plan.cpss_tuning_partitions)
        )
        final_calls = tuple(
            FamilyCallSpec(
                call_id=f"cpss_final_{index:03d}",
                phase="cpss_final",
                partition=partition,
                score_family_id="cpss-final",
            )
            for index, partition in enumerate(split_plan.cpss_final_partitions)
        )
        return cls(
            split_plan=split_plan,
            training_anchor_call=FamilyCallSpec(
                call_id="training_anchor",
                phase="training_rank",
                partition=split_plan.training_anchor,
                score_family_id="raw-training",
            ),
            final_development_anchor_call=FamilyCallSpec(
                call_id="final_development_anchor",
                phase="final_development_rank",
                partition=split_plan.final_development_anchor,
                score_family_id="raw-final",
            ),
            cpss_tuning_calls=tuning_calls,
            cpss_final_calls=final_calls,
        )

    @property
    def split_plan_sha256(self) -> str:
        return self.split_plan.split_plan_sha256

    @property
    def all_calls(self) -> tuple[FamilyCallSpec, ...]:
        return (
            self.training_anchor_call,
            self.final_development_anchor_call,
            *self.cpss_tuning_calls,
            *self.cpss_final_calls,
        )

    @property
    def call_count(self) -> int:
        return len(self.all_calls)


@dataclass(frozen=True)
class PolicyFamilyExecution:
    """Validated results from one execution of the shared call plan."""

    call_plan: PolicyFamilyCallPlan
    requests: tuple[tuple[str, RankerFitRequest], ...]
    results: tuple[tuple[str, RankingResult], ...]

    def __post_init__(self) -> None:
        expected = tuple(call.call_id for call in self.call_plan.all_calls)
        if tuple(call_id for call_id, _ in self.requests) != expected:
            raise ValueError("request order does not match PolicyFamilyCallPlan")
        if tuple(call_id for call_id, _ in self.results) != expected:
            raise ValueError("result order does not match PolicyFamilyCallPlan")

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def request(self, call_id: str) -> RankerFitRequest:
        for observed_id, request in self.requests:
            if observed_id == call_id:
                return request
        raise KeyError(call_id)

    def result(self, call_id: str) -> RankingResult:
        for observed_id, result in self.results:
            if observed_id == call_id:
                return result
        raise KeyError(call_id)


def execute_policy_family(
    adapter: RankerAdapter,
    ranker: RankerDescriptor,
    call_plan: PolicyFamilyCallPlan,
    seed_plan: SeedPlan,
    *,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    candidate_edges: Sequence[Sequence[int]],
    config: Mapping[str, Any],
) -> PolicyFamilyExecution:
    """Execute every unique family fit once and validate all ranker results."""

    ranker_id = getattr(ranker, "ranker_id", None)
    if ranker_id != seed_plan.ranker_id:
        raise ValueError("ranker and SeedPlan ranker_id mismatch")
    if call_plan.split_plan_sha256 != seed_plan.split_plan_sha256:
        raise ValueError("SeedPlan is bound to a different SplitPlan")
    edges: tuple[EDGE, ...] = normalize_candidate_edges(candidate_edges)
    universe_hash = candidate_universe_sha256(edges)
    requests: list[tuple[str, RankerFitRequest]] = []
    results: list[tuple[str, RankingResult]] = []
    for call in call_plan.all_calls:
        call_plan.split_plan.assert_no_outer_test(call.partition)
        request = RankerFitRequest(
            X=X,
            Z=Z,
            y=y,
            candidate_edges=edges,
            fit_indices=np.asarray(call.partition.fit_indices, dtype=np.int64),
            validation_indices=np.asarray(
                call.partition.validation_indices, dtype=np.int64
            ),
            model_seed=seed_plan.seed_for(call.call_id),
            phase=call.phase,
            score_family_id=f"{ranker_id}:{call.score_family_id}",
            split_plan_sha256=call_plan.split_plan_sha256,
            candidate_universe_sha256=universe_hash,
            config=config,
        )
        result = validate_ranking_result(adapter.fit(request), request, ranker)
        requests.append((call.call_id, request))
        results.append((call.call_id, result))
    return PolicyFamilyExecution(call_plan, tuple(requests), tuple(results))


@dataclass(frozen=True)
class FixedKOutcome:
    requested_budget: int
    support_indices: INDEX
    budget_shortfall: int
    source_call_id: str
    source_result: RankingResult
    zero_padding_applied: bool = False


@dataclass(frozen=True)
class ValidationOneSeBoundary:
    """Inputs needed to select k without exposing outer-test rows."""

    training_request: RankerFitRequest
    training_result: RankingResult
    policy_validation_indices: INDEX
    final_request: RankerFitRequest
    final_result: RankingResult


@dataclass(frozen=True)
class CpssOneSeBoundary:
    """The 4B complementary-half results; raw anchors are intentionally absent."""

    tuning_call_ids: tuple[str, ...]
    tuning_results: tuple[RankingResult, ...]
    final_call_ids: tuple[str, ...]
    final_results: tuple[RankingResult, ...]

    @property
    def call_count(self) -> int:
        return len(self.tuning_results) + len(self.final_results)


class SupportPolicyConsumer(Protocol):
    def consume(self, execution: PolicyFamilyExecution) -> object: ...


@dataclass(frozen=True)
class FixedKHandler:
    k: int

    def __post_init__(self) -> None:
        if self.k not in FIXED_K_GRID:
            raise ValueError(f"k must be one of {FIXED_K_GRID}")

    def consume(self, execution: PolicyFamilyExecution) -> FixedKOutcome:
        call_id = execution.call_plan.final_development_anchor_call.call_id
        result = execution.result(call_id)
        ranked = tuple(int(index) for index in np.asarray(result.ranked_indices))
        support = ranked[: self.k]
        return FixedKOutcome(
            requested_budget=int(self.k),
            support_indices=support,
            budget_shortfall=max(0, int(self.k) - len(support)),
            source_call_id=call_id,
            source_result=result,
            zero_padding_applied=False,
        )


@dataclass(frozen=True)
class ValidationOneSeHandler:
    def consume(self, execution: PolicyFamilyExecution) -> ValidationOneSeBoundary:
        training_id = execution.call_plan.training_anchor_call.call_id
        final_id = execution.call_plan.final_development_anchor_call.call_id
        return ValidationOneSeBoundary(
            training_request=execution.request(training_id),
            training_result=execution.result(training_id),
            policy_validation_indices=execution.call_plan.split_plan.outer_validation_indices,
            final_request=execution.request(final_id),
            final_result=execution.result(final_id),
        )


@dataclass(frozen=True)
class CpssOneSeHandler:
    def consume(self, execution: PolicyFamilyExecution) -> CpssOneSeBoundary:
        tuning_ids = tuple(
            call.call_id for call in execution.call_plan.cpss_tuning_calls
        )
        final_ids = tuple(call.call_id for call in execution.call_plan.cpss_final_calls)
        return CpssOneSeBoundary(
            tuning_call_ids=tuning_ids,
            tuning_results=tuple(execution.result(call_id) for call_id in tuning_ids),
            final_call_ids=final_ids,
            final_results=tuple(execution.result(call_id) for call_id in final_ids),
        )
