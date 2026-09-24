from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


EDGE = tuple[int, ...]


def _edge(edge: Sequence[int]) -> EDGE:
    normalized = tuple(sorted(int(value) for value in edge))
    if not normalized:
        raise ValueError("Edges must contain at least one feature index")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Edge contains repeated feature indices: {normalized}")
    return normalized


def _edge_set(edges: Sequence[EDGE]) -> set[EDGE]:
    return {_edge(edge) for edge in edges}


def _truth_groups(
    true_edges: set[EDGE], equivalence_groups: Sequence[set[EDGE]]
) -> tuple[frozenset[EDGE], ...]:
    exact_truth = _edge_set(tuple(true_edges))
    if not equivalence_groups:
        return tuple(frozenset({edge}) for edge in sorted(exact_truth))

    groups = tuple(
        frozenset(_edge(edge) for edge in group) for group in equivalence_groups
    )
    if any(not group for group in groups):
        raise ValueError("Truth groups must be non-empty")
    if len(set(groups)) != len(groups):
        raise ValueError("Duplicate truth groups are not allowed")
    if any(not (group & exact_truth) for group in groups):
        raise ValueError("Every declared truth group must contain an exact true edge")

    covered_truth = set().union(*groups)
    missing_truth = exact_truth - covered_truth
    if missing_truth:
        raise ValueError(
            f"Declared truth groups do not cover exact true edges: {sorted(missing_truth)}"
        )
    return groups


def _maximum_group_matches(
    selected: set[EDGE], groups: Sequence[frozenset[EDGE]]
) -> int:
    adjacency = {
        edge: tuple(index for index, group in enumerate(groups) if edge in group)
        for edge in sorted(selected)
    }
    group_to_edge: dict[int, EDGE] = {}

    def augment(edge: EDGE, visited: set[int]) -> bool:
        for group_index in adjacency[edge]:
            if group_index in visited:
                continue
            visited.add(group_index)
            incumbent = group_to_edge.get(group_index)
            if incumbent is None or augment(incumbent, visited):
                group_to_edge[group_index] = edge
                return True
        return False

    return sum(augment(edge, set()) for edge in sorted(selected))


def support_metrics(
    selected: Sequence[EDGE],
    true_edges: set[EDGE] | None,
    equivalence_groups: Sequence[set[EDGE]] = (),
) -> dict[str, float | int]:
    selected_set = _edge_set(selected)
    if true_edges is None:
        return {
            "support_precision": math.nan,
            "support_recall": math.nan,
            "support_f1": math.nan,
            "false_inclusions": math.nan,
            "false_discovery_proportion": math.nan,
            "group_tp": math.nan,
            "group_precision": math.nan,
            "group_recall": math.nan,
            "group_f1": math.nan,
            "group_false_inclusions": math.nan,
            "group_false_discovery_proportion": math.nan,
        }

    exact_truth = _edge_set(tuple(true_edges))
    exact_hits = len(selected_set & exact_truth)
    precision = (
        exact_hits / len(selected_set)
        if selected_set
        else (1.0 if not exact_truth else 0.0)
    )
    recall = exact_hits / len(exact_truth) if exact_truth else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_inclusions = len(selected_set) - exact_hits

    groups = _truth_groups(exact_truth, equivalence_groups)
    group_tp = _maximum_group_matches(selected_set, groups)
    group_precision = (
        group_tp / len(selected_set) if selected_set else (1.0 if not groups else 0.0)
    )
    group_recall = group_tp / len(groups) if groups else 1.0
    group_f1 = (
        2.0 * group_precision * group_recall / (group_precision + group_recall)
        if group_precision + group_recall
        else 0.0
    )
    group_false_inclusions = len(selected_set) - group_tp

    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "false_inclusions": int(false_inclusions),
        "false_discovery_proportion": float(
            false_inclusions / len(selected_set) if selected_set else 0.0
        ),
        "group_tp": int(group_tp),
        "group_precision": float(group_precision),
        "group_recall": float(group_recall),
        "group_f1": float(group_f1),
        "group_false_inclusions": int(group_false_inclusions),
        "group_false_discovery_proportion": float(
            group_false_inclusions / len(selected_set) if selected_set else 0.0
        ),
    }


def candidate_wide_average_precision(
    candidate_edges: Sequence[EDGE],
    scores: Sequence[float] | np.ndarray,
    true_edges: set[EDGE] | None,
    *,
    score_direction: str = "higher_is_better",
) -> float:
    if score_direction != "higher_is_better":
        raise ValueError("Candidate scores must declare higher_is_better direction")

    candidates = tuple(_edge(edge) for edge in candidate_edges)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate edges must be unique")

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.shape[0] != len(candidates):
        raise ValueError(
            "Candidate score vector must be one-dimensional and match candidate count"
        )
    if not np.isfinite(values).all():
        raise ValueError("Candidate scores must all be finite")

    if true_edges is None:
        return math.nan
    exact_truth = _edge_set(tuple(true_edges))
    missing_truth = exact_truth - set(candidates)
    if missing_truth:
        raise ValueError(
            f"Exact true edges are absent from candidate grid: {sorted(missing_truth)}"
        )
    if not exact_truth:
        return math.nan

    labels = np.fromiter(
        (1 if edge in exact_truth else 0 for edge in candidates),
        dtype=int,
        count=len(candidates),
    )
    # sklearn computes non-interpolated AP over score-threshold blocks. Equal
    # scores therefore receive no index-dependent ordering or artificial jitter.
    return float(average_precision_score(labels, values))
