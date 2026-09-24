"""Registry-driven descriptive analysis for Track-A candidate outputs.

The previous six-method promotion verdict belonged to the historical extension
experiment. This analysis validates registry-bound evidence and produces
ranker/policy/score-family summaries without importing that verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from method_registry import MethodRegistry, load_method_registry
from scientific_evidence import validate_scientific_run_structure
from structured_comparator_registry import (
    load_structured_comparator_registry,
    structured_registry_document_sha256,
)
from support_metrics import support_metrics as calculate_support_metrics


KEYS = ["dataset", "scenario", "data_seed", "split_seed"]
CONFIRMATORY_SCENARIOS = tuple(f"S{index:02d}" for index in range(1, 14))
CONFIRMATORY_SEEDS = (1103, 1129, 1151, 1171, 1181, 1201, 1213, 1231, 1249, 1277)
CONFIRMATORY_RANKERS = ("shil", "l1")
FIXED_K_VALUES = (1, 2, 4, 8, 16)
CONFIRMATORY_BOOTSTRAP_REPLICATES = 1_024
CONFIRMATORY_MIN_FINITE_T = 1_000
REQUIRED_OUTPUTS = (
    "metrics.csv",
    "selected_edges.csv",
    "ranking_metrics.csv",
    "ranking_scores.csv",
    "partition_manifest.csv",
    "dataset_manifest.csv",
    "run_summary.json",
    "config_resolved.json",
    "method_registry_resolved.json",
    "structured_comparator_registry_resolved.json",
    "structured_sensitivity_status.csv",
    "structured_sensitivity_metrics.csv",
    "structured_sensitivity_selected_edges.csv",
    "structured_sensitivity_raw.csv",
    "output_hashes.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_edges_json(text: str) -> set[tuple[int, ...]]:
    raw = json.loads(str(text))
    if not isinstance(raw, list):
        raise AssertionError("edges_json must decode to a list")
    edges = {tuple(int(value) for value in edge) for edge in raw}
    if len(edges) != len(raw):
        raise AssertionError("edges_json contains duplicate edges")
    return edges


def pairwise_jaccard(supports: Sequence[set[tuple[int, ...]]]) -> list[float]:
    values: list[float] = []
    for left, right in itertools.combinations(supports, 2):
        union = left | right
        values.append(1.0 if not union else len(left & right) / len(union))
    return values


def nogueira_stability(
    supports: Sequence[set[tuple[int, ...]]], universe: Sequence[tuple[int, ...]]
) -> float:
    if len(supports) < 2 or not universe:
        return math.nan
    lookup = {edge: index for index, edge in enumerate(universe)}
    matrix = np.zeros((len(supports), len(universe)), dtype=float)
    for row_index, support in enumerate(supports):
        unknown = support - set(lookup)
        if unknown:
            raise AssertionError(
                f"Support contains edges outside candidate grid: {unknown}"
            )
        for edge in support:
            matrix[row_index, lookup[edge]] = 1.0
    average_size = matrix.sum(axis=1).mean()
    p = matrix.shape[1]
    denominator = (average_size / p) * (1.0 - average_size / p)
    if denominator <= 0:
        return 1.0 if np.all(matrix == matrix[0]) else math.nan
    numerator = matrix.var(axis=0, ddof=1).mean()
    return float(1.0 - numerator / denominator)


def _exact_group_set(
    frame: pd.DataFrame, group_keys: list[str], identity_column: str
) -> dict[tuple[Any, ...], set[str]]:
    output: dict[tuple[Any, ...], set[str]] = {}
    for key, group in frame.groupby(group_keys, dropna=False):
        normalized = tuple(key) if isinstance(key, tuple) else (key,)
        output[normalized] = set(group[identity_column].astype(str))
    return output


def _validate_hashes(input_dir: Path) -> None:
    document = json.loads(
        (input_dir / "output_hashes.json").read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise AssertionError("output_hashes.json must be an object")
    for name, expected in document.items():
        path = input_dir / str(name)
        if not path.is_file():
            raise AssertionError(f"Hashed output is missing: {name}")
        if sha256_file(path) != str(expected).upper():
            raise AssertionError(f"Output hash mismatch: {name}")
    closure_path = input_dir / "terminal_closure.json"
    if not closure_path.exists():
        return
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    notification_files = sorted(
        (input_dir / "notifications").glob("invocation-*.jsonl")
    )
    notification_bindings = [
        {
            "path": path.relative_to(input_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in notification_files
    ]
    if (
        closure.get("status") != "closed_completed"
        or closure.get("terminal_status_sha256")
        != sha256_file(input_dir / "terminal_status.json")
        or closure.get("terminal_status_history_sha256")
        != sha256_file(input_dir / "terminal_status_history.jsonl")
        or closure.get("output_hashes_sha256")
        != sha256_file(input_dir / "output_hashes.json")
        or closure.get("run_manifest_sha256")
        != sha256_file(input_dir / "RUN_MANIFEST.json")
        or not notification_files
        or closure.get("notification_outboxes") != notification_bindings
    ):
        raise AssertionError("Terminal closure binding mismatch")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _validate_structured_sensitivity_evidence(
    input_dir: Path,
    partitions: pd.DataFrame,
    run_summary: dict[str, Any],
    cell_count: int,
) -> dict[str, Any]:
    registry_document = json.loads(
        (input_dir / "structured_comparator_registry_resolved.json").read_text(
            encoding="utf-8"
        )
    )
    computed = structured_registry_document_sha256(registry_document)
    if registry_document.get("canonical_sha256") != computed:
        raise AssertionError("Structured registry canonical hash mismatch")
    if run_summary.get("structured_registry_sha256") != computed:
        raise AssertionError("Run-summary structured registry hash mismatch")
    if registry_document.get("registry_active") is not True:
        raise AssertionError("Structured comparator registry is not active")
    if run_summary.get("structured_registry_active") is not True:
        raise AssertionError("Run-summary structured registry active flag mismatch")
    if set(partitions["structured_registry_sha256"].astype(str)) != {computed}:
        raise AssertionError("Partition structured registry hash mismatch")
    candidate_root = Path(__file__).resolve().parents[1]
    live_registry = load_structured_comparator_registry(
        candidate_root / "config" / "structured_comparator_registry.json"
    )
    if (
        not live_registry.registry_active
        or live_registry.canonical_sha256 != computed
        or live_registry.registry_role != "sensitivity_only"
    ):
        raise AssertionError(
            "Resolved structured registry differs from live activation evidence binding"
        )
    activation = registry_document.get("activation_evidence")
    if activation != {
        "path": (
            "evidence/structured_comparator_activation_20260826/"
            "ACTIVATION_EVIDENCE.json"
        ),
        "sha256": "C3DB98D49C1C79BFD5EE6AD5CF9B5C81268A8FCDB51A900198DE101387D940C9",
        "required_decision": (
            "ACTIVATE_S13_SENSITIVITY_REGISTRY_WITH_EXECUTION_DISABLED"
        ),
    }:
        raise AssertionError("Structured activation-evidence binding mismatch")
    comparators = registry_document.get("comparators")
    if not isinstance(comparators, list) or len(comparators) != 1:
        raise AssertionError("Structured active registry must contain one comparator")
    comparator = comparators[0]
    if (
        comparator.get("comparator_id") != "glinternet.strong_hierarchy.pairwise"
        or comparator.get("registry_active") is not True
        or comparator.get("reporting_role") != "sensitivity_only"
        or comparator.get("eligible_scenario_ids") != ["S13"]
        or comparator.get("required_cell_type") != "structured_pairwise"
        or comparator.get("required_candidate_orders") != [2]
        or comparator.get("required_heredity") != "respected"
        or comparator.get("required_hierarchy") != "strong"
        or comparator.get("excluded_from_primary_estimand") is not True
        or comparator.get("excluded_from_main_method_registry") is not True
    ):
        raise AssertionError("Structured active comparator applicability drift")

    frames = {
        name: _read_optional_csv(input_dir / f"structured_sensitivity_{name}.csv")
        for name in ("status", "metrics", "selected_edges", "raw")
    }
    status = frames["status"]
    if len(status) != cell_count or set(status["status"]) - {
        "N/A",
        "DISABLED",
        "COMPLETED",
    }:
        raise AssertionError("Structured sensitivity status coverage is incomplete")
    for name, frame in frames.items():
        if frame.empty:
            continue
        if set(frame["analysis_role"].astype(str)) != {"sensitivity_only"}:
            raise AssertionError(f"Structured sensitivity role mismatch: {name}")
        if set(frame["primary_estimand"].astype(str).str.lower()) != {"false"}:
            raise AssertionError(
                f"Structured sensitivity entered primary estimand: {name}"
            )
        if {"method_id", "score_family_id"} & set(frame.columns):
            raise AssertionError(
                f"Structured sensitivity reused a primary identity: {name}"
            )
        if set(frame["structured_registry_sha256"].astype(str)) != {computed}:
            raise AssertionError(
                f"Structured sensitivity registry binding mismatch: {name}"
            )
    completed = int((status["status"] == "COMPLETED").sum())
    if len(frames["metrics"]) != completed or len(frames["raw"]) != completed:
        raise AssertionError("Structured sensitivity completed-artifact count mismatch")
    if run_summary.get("block_role") == "scientific_full_synthetic_primary":
        counts = status["status"].value_counts().to_dict()
        if counts != {"N/A": 130, "DISABLED": 10}:
            raise AssertionError(
                "Primary synthetic comparator statuses must be 130 N/A + 10 DISABLED"
            )
        s13 = status.loc[status["scenario"] == "S13", "status"]
        non_s13 = status.loc[status["scenario"] != "S13", "status"]
        if set(s13) != {"DISABLED"} or set(non_s13) != {"N/A"}:
            raise AssertionError("Primary synthetic comparator isolation drift")
    return {
        "registry_sha256": computed,
        "registry_active": True,
        "status_counts": {
            str(key): int(value)
            for key, value in status["status"].value_counts().sort_index().items()
        },
        "applicability": "S13_PAIRWISE_STRONG_HIERARCHY_ONLY",
        "analysis_role": "sensitivity_only",
        "artifact_role": "DESCRIPTIVE_NO_INFERENCE",
        "excluded_from_primary_estimand": True,
        "excluded_from_main_method_registry": True,
    }


def _registry_expectations(
    registry: MethodRegistry, config: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    orders = [int(value) for value in config["candidate_orders"]]
    synthetic = registry.expected_methods_for(
        {"cell_type": "synthetic", "candidate_orders": orders}
    )
    real = registry.expected_methods_for(
        {"cell_type": "real", "candidate_orders": orders}
    )
    if synthetic != real:
        raise AssertionError("Synthetic and real Track-A cells resolve differently")
    families = tuple(
        dict.fromkeys(registry.method_by_id[item].score_family_id for item in synthetic)
    )
    return synthetic, families


def validate_evidence(
    input_dir: Path,
    *,
    require_full_scientific: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    MethodRegistry,
]:
    missing = [name for name in REQUIRED_OUTPUTS if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing run outputs: {missing}")
    _validate_hashes(input_dir)
    run_summary = json.loads(
        (input_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    resolved_preview = json.loads(
        (input_dir / "config_resolved.json").read_text(encoding="utf-8")
    )
    manifest_preview_path = input_dir / "RUN_MANIFEST.json"
    manifest_preview = (
        json.loads(manifest_preview_path.read_text(encoding="utf-8"))
        if manifest_preview_path.is_file()
        else {}
    )
    scientific_signals = {
        str(run_summary.get("block_role", "")),
        str(resolved_preview.get("execution_class", "")),
        str(manifest_preview.get("run_class", "")),
    }
    expected_role = "scientific_full_synthetic_primary"
    if require_full_scientific:
        validate_scientific_run_structure(
            input_dir,
            expected_run_id=("EXP-SHIL-TRACK-A-FULL-SYNTHETIC-PRIMARY-full-synthetic"),
            expected_role=expected_role,
            expected_mode="full-synthetic",
            expected_unit_count=140,
            expected_scenarios=(f"S{index:02d}" for index in range(14)),
            expected_split_seeds=CONFIRMATORY_SEEDS,
        )
    elif expected_role in scientific_signals:
        raise AssertionError(
            "Non-scientific analyzer mode cannot accept full scientific evidence"
        )
    metrics = pd.read_csv(input_dir / "metrics.csv")
    supports = pd.read_csv(input_dir / "selected_edges.csv")
    ranking_metrics = pd.read_csv(input_dir / "ranking_metrics.csv")
    ranking_scores = pd.read_csv(input_dir / "ranking_scores.csv")
    partitions = pd.read_csv(input_dir / "partition_manifest.csv")
    dataset_manifest = pd.read_csv(input_dir / "dataset_manifest.csv")
    if require_full_scientific and not (input_dir / "terminal_closure.json").is_file():
        raise FileNotFoundError(
            "Full synthetic scientific evidence requires terminal_closure.json"
        )
    config = json.loads(
        (input_dir / "config_resolved.json").read_text(encoding="utf-8")
    )
    resolved_config = config.get("track_a_resolved_config")
    if not isinstance(resolved_config, dict):
        raise AssertionError("Track-A resolved config is missing")
    registry = load_method_registry(
        input_dir / "method_registry_resolved.json",
        available_config_keys=resolved_config.keys(),
    )
    expected_methods, expected_families = _registry_expectations(registry, config)
    if str(run_summary.get("registry_sha256")) != registry.canonical_sha256:
        raise AssertionError("Run-summary registry hash mismatch")
    if list(run_summary.get("expected_method_ids", [])) != list(expected_methods):
        raise AssertionError("Run-summary method set mismatch")
    if list(run_summary.get("expected_score_family_ids", [])) != list(
        expected_families
    ):
        raise AssertionError("Run-summary score-family set mismatch")

    cell_count = int(run_summary["dataset_runs"])
    if require_full_scientific and cell_count != 140:
        raise AssertionError("Full synthetic analysis requires exactly 140 units")
    run_summary["_validated_structured_sensitivity"] = (
        _validate_structured_sensitivity_evidence(
            input_dir, partitions, run_summary, cell_count
        )
    )
    for label, frame in (
        ("metrics", metrics),
        ("selected_edges", supports),
        ("ranking_metrics", ranking_metrics),
        ("ranking_scores", ranking_scores),
    ):
        if "comparator_id" in frame.columns or (
            "analysis_role" in frame.columns
            and (frame["analysis_role"].astype(str) == "sensitivity_only").any()
        ):
            raise AssertionError(f"Structured sensitivity contaminated primary {label}")
    checks = (
        (metrics, "method_id", set(expected_methods), "metrics"),
        (supports, "method_id", set(expected_methods), "selected_edges"),
        (ranking_metrics, "score_family_id", set(expected_families), "ranking_metrics"),
        (ranking_scores, "score_family_id", set(expected_families), "ranking_scores"),
    )
    for frame, identity, expected, label in checks:
        if frame.duplicated(KEYS + [identity]).any():
            raise AssertionError(f"Duplicate {label} keys")
        grouped = _exact_group_set(frame, KEYS, identity)
        if len(grouped) != cell_count or any(
            values != expected for values in grouped.values()
        ):
            raise AssertionError(
                f"{label} does not carry the registry identity set per cell"
            )
    if partitions.duplicated(KEYS).any() or len(partitions) != cell_count:
        raise AssertionError("Partition manifest must contain one row per cell")
    for row in partitions.itertuples(index=False):
        if json.loads(row.expected_method_ids_json) != list(expected_methods):
            raise AssertionError("Partition method set mismatch")
        if json.loads(row.expected_score_family_ids_json) != list(expected_families):
            raise AssertionError("Partition score-family set mismatch")
        if str(row.registry_sha256) != registry.canonical_sha256:
            raise AssertionError("Partition registry hash mismatch")
    for row in supports.itertuples(index=False):
        edges = parse_edges_json(row.edges_json)
        if len(edges) != int(row.selected_edge_count):
            raise AssertionError("Selected-edge count does not match edges_json")
    for row in ranking_scores.itertuples(index=False):
        candidates = json.loads(row.candidate_edges_json)
        scores = json.loads(row.scores_json)
        if len(candidates) != len(scores):
            raise AssertionError(
                "Candidate-edge and score vectors have different lengths"
            )
        if not np.all(np.isfinite(np.asarray(scores, dtype=float))):
            raise AssertionError("Ranking score vector contains non-finite values")
    return (
        metrics,
        supports,
        ranking_metrics,
        ranking_scores,
        partitions,
        dataset_manifest,
        run_summary,
        config,
        registry,
    )


def add_run_mean_jaccard(
    metrics: pd.DataFrame, supports_frame: pd.DataFrame
) -> pd.DataFrame:
    supports = {
        tuple(getattr(row, key) for key in KEYS)
        + (row.method_id,): parse_edges_json(row.edges_json)
        for row in supports_frame.itertuples(index=False)
    }
    output = metrics.copy()
    run_values: list[float] = []
    for row in output.itertuples(index=False):
        key = tuple(getattr(row, column) for column in KEYS) + (row.method_id,)
        own = supports[key]
        peer_rows = output[
            (output["scenario"] == row.scenario)
            & (output["method_id"] == row.method_id)
        ]
        peers: list[float] = []
        for other in peer_rows.itertuples(index=False):
            other_key = tuple(getattr(other, column) for column in KEYS) + (
                other.method_id,
            )
            if other_key == key:
                continue
            peer = supports[other_key]
            union = own | peer
            peers.append(1.0 if not union else len(own & peer) / len(union))
        run_values.append(float(np.mean(peers)) if peers else math.nan)
    output["run_mean_jaccard"] = run_values
    return output


def stability_table(
    metrics: pd.DataFrame,
    supports_frame: pd.DataFrame,
    ranking_scores: pd.DataFrame,
) -> pd.DataFrame:
    supports = {
        tuple(getattr(row, key) for key in KEYS)
        + (row.method_id,): parse_edges_json(row.edges_json)
        for row in supports_frame.itertuples(index=False)
    }
    universe_by_cell: dict[tuple[Any, ...], list[tuple[int, ...]]] = {}
    for row in ranking_scores.itertuples(index=False):
        key = tuple(getattr(row, column) for column in KEYS)
        universe = [
            tuple(int(value) for value in edge)
            for edge in json.loads(row.candidate_edges_json)
        ]
        previous = universe_by_cell.setdefault(key, universe)
        if previous != universe:
            raise AssertionError("Score families disagree on the candidate universe")
    rows: list[dict[str, Any]] = []
    for (scenario, method_id), group in metrics.groupby(
        ["scenario", "method_id"], dropna=False
    ):
        scenario_supports: list[set[tuple[int, ...]]] = []
        universe: list[tuple[int, ...]] | None = None
        for item in group.itertuples(index=False):
            cell_key = tuple(getattr(item, column) for column in KEYS)
            scenario_supports.append(supports[cell_key + (item.method_id,)])
            candidate_universe = universe_by_cell[cell_key]
            if universe is None:
                universe = candidate_universe
            elif universe != candidate_universe:
                raise AssertionError("Scenario runs use different candidate universes")
        jaccard = pairwise_jaccard(scenario_supports)
        rows.append(
            {
                "scenario": scenario,
                "method_id": method_id,
                "ranker_id": group["ranker_id"].iloc[0],
                "policy_id": group["policy_id"].iloc[0],
                "runs": len(scenario_supports),
                "candidate_edges": 0 if universe is None else len(universe),
                "pairwise_jaccard_mean": (
                    float(np.mean(jaccard)) if jaccard else math.nan
                ),
                "pairwise_jaccard_sd": (
                    float(np.std(jaccard, ddof=1)) if len(jaccard) > 1 else math.nan
                ),
                "pairwise_jaccard_min": float(np.min(jaccard)) if jaccard else math.nan,
                "pairwise_jaccard_max": float(np.max(jaccard)) if jaccard else math.nan,
                "nogueira_stability": nogueira_stability(
                    scenario_supports, [] if universe is None else universe
                ),
            }
        )
    return pd.DataFrame(rows)


def _mean_summary(metrics: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    candidate_metrics = [
        "selected_edge_count",
        "candidate_average_precision",
        "support_f1",
        "group_f1",
        "false_inclusions",
        "group_false_inclusions",
        "false_discovery_proportion",
        "group_false_discovery_proportion",
        "log_loss",
        "macro_f1",
        "accuracy",
        "run_mean_jaccard",
    ]
    columns = [name for name in candidate_metrics if name in metrics.columns]
    aggregations = {name: ["mean", "std"] for name in columns}
    summary = metrics.groupby(groups, dropna=False).agg(aggregations)
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    return summary.reset_index()


def _policy_level(method: Any) -> str:
    if method.policy_id == "fixed_k":
        return f"fixed_k.k{int(method.policy_parameters['k']):02d}"
    return str(method.policy_id)


def _confirmatory_method_contract(
    registry: MethodRegistry,
) -> tuple[tuple[str, ...], dict[tuple[str, str], str], tuple[str, ...]]:
    methods = tuple(registry.methods)
    method_ids = tuple(method.method_id for method in methods)
    if len(method_ids) != 14 or len(set(method_ids)) != 14:
        raise AssertionError("Confirmatory registry must contain exactly 14 methods")
    lookup: dict[tuple[str, str], str] = {}
    levels_by_ranker: dict[str, list[str]] = {
        ranker: [] for ranker in CONFIRMATORY_RANKERS
    }
    for method in methods:
        if method.ranker_id not in levels_by_ranker:
            raise AssertionError("Confirmatory registry must contain only SHIL and L1")
        level = _policy_level(method)
        key = (method.ranker_id, level)
        if key in lookup:
            raise AssertionError("Confirmatory ranker-policy identity is duplicated")
        lookup[key] = method.method_id
        levels_by_ranker[method.ranker_id].append(level)
    shil_levels = tuple(levels_by_ranker["shil"])
    l1_levels = tuple(levels_by_ranker["l1"])
    required_levels = tuple(
        [f"fixed_k.k{value:02d}" for value in FIXED_K_VALUES]
        + ["validation_one_se", "cpss_one_se"]
    )
    if shil_levels != required_levels or l1_levels != required_levels:
        raise AssertionError(
            "Confirmatory registry does not carry the locked seven policies"
        )
    return method_ids, lookup, required_levels


def _cell_failure_reason(
    cell: pd.DataFrame,
    selected_cell: pd.DataFrame,
    partition: pd.DataFrame,
    expected_method_ids: tuple[str, ...],
) -> str | None:
    if len(partition) != 1:
        return "PARTITION_MISSING_OR_DUPLICATE"
    if "status" in partition and str(partition.iloc[0]["status"]) != "OK":
        return "PARTITION_NOT_OK"
    if len(cell) != 14:
        return "PRIMARY_METHOD_COUNT_NOT_14"
    if cell["method_id"].duplicated().any():
        return "DUPLICATE_PRIMARY_METHOD"
    if set(cell["method_id"].astype(str)) != set(expected_method_ids):
        return "PRIMARY_METHOD_SET_MISMATCH"
    if len(selected_cell) != 14 or set(selected_cell["method_id"].astype(str)) != set(
        expected_method_ids
    ):
        return "SELECTED_SUPPORT_METHOD_SET_INCOMPLETE"
    if selected_cell["method_id"].duplicated().any():
        return "SELECTED_SUPPORT_METHOD_DUPLICATE"
    if "status" in cell and set(cell["status"].astype(str)) != {"OK"}:
        return "PRIMARY_METHOD_NOT_OK"
    fixed = cell[cell["policy_id"].astype(str) == "fixed_k"]
    if "budget_shortfall" not in fixed:
        return "FIXED_K_SHORTFALL_STATUS_UNAVAILABLE"
    shortfalls = pd.to_numeric(fixed["budget_shortfall"], errors="coerce").to_numpy(
        dtype=float
    )
    if len(shortfalls) != 10 or not np.all(np.isfinite(shortfalls)):
        return "FIXED_K_SHORTFALL_STATUS_INVALID"
    if np.any(shortfalls != 0.0):
        return "FIXED_K_SHORTFALL"
    selected_by_method = selected_cell.set_index("method_id")
    for ranker in CONFIRMATORY_RANKERS:
        for k in FIXED_K_VALUES:
            method_id = f"{ranker}.fixed_k.k{k:02d}"
            if int(selected_by_method.loc[method_id, "selected_edge_count"]) != k:
                return "FIXED_K_CARDINALITY_MISMATCH"
    values = pd.to_numeric(cell["group_f1"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        return "GROUP_F1_NONFINITE_OR_OUT_OF_RANGE"
    if (
        cell["dataset"].nunique(dropna=False) != 1
        or cell["data_seed"].nunique(dropna=False) != 1
    ):
        return "CELL_PROVENANCE_NOT_UNIQUE"
    return None


def _edge_sequence(raw: Any, context: str) -> tuple[tuple[int, ...], ...]:
    try:
        decoded = json.loads(str(raw))
    except json.JSONDecodeError as error:
        raise AssertionError(f"{context} is not valid JSON") from error
    if not isinstance(decoded, list):
        raise AssertionError(f"{context} must contain a JSON list")
    edges = tuple(tuple(sorted(int(value) for value in edge)) for edge in decoded)
    if any(not edge or len(edge) != len(set(edge)) for edge in edges):
        raise AssertionError(f"{context} contains an invalid edge")
    if len(edges) != len(set(edges)):
        raise AssertionError(f"{context} contains duplicate edges")
    return edges


def _dataset_truth(
    dataset_rows: pd.DataFrame,
) -> tuple[set[tuple[int, ...]], list[set[tuple[int, ...]]]]:
    if len(dataset_rows) != 1:
        raise AssertionError("DATASET_TRUTH_MISSING_OR_DUPLICATE")
    row = dataset_rows.iloc[0]
    if "true_edges_json" not in dataset_rows or not isinstance(
        row["true_edges_json"], str
    ):
        raise AssertionError("DATASET_TRUE_EDGES_JSON_UNAVAILABLE")
    true_edges = set(_edge_sequence(row["true_edges_json"], "true_edges_json"))
    if "equivalence_groups_json" not in dataset_rows or not isinstance(
        row["equivalence_groups_json"], str
    ):
        raise AssertionError("DATASET_EQUIVALENCE_GROUPS_JSON_UNAVAILABLE")
    decoded_groups = json.loads(row["equivalence_groups_json"])
    if not isinstance(decoded_groups, list):
        raise AssertionError("DATASET_EQUIVALENCE_GROUPS_JSON_INVALID")
    equivalence_groups = [
        set(_edge_sequence(json.dumps(group), "equivalence group"))
        for group in decoded_groups
    ]
    return true_edges, equivalence_groups


def _raw_exact_k_support(
    ranking_row: pd.Series,
    *,
    k: int,
    ranker_id: str,
    registry: MethodRegistry,
) -> tuple[tuple[int, ...], ...]:
    candidates = _edge_sequence(
        ranking_row["candidate_edges_json"], "candidate_edges_json"
    )
    scores = np.asarray(json.loads(ranking_row["scores_json"]), dtype=float)
    ranked = tuple(
        int(value) for value in json.loads(ranking_row["ranked_indices_json"])
    )
    active = tuple(
        int(value) for value in json.loads(ranking_row["active_indices_json"])
    )
    candidate_count = len(candidates)
    if (
        scores.shape != (candidate_count,)
        or not np.all(np.isfinite(scores))
        or len(ranked) != len(set(ranked))
        or len(active) != len(set(active))
        or set(ranked) != set(active)
        or any(index < 0 or index >= candidate_count for index in (*ranked, *active))
    ):
        raise AssertionError("RAW_REFERENCE_RANKING_CONTRACT_INVALID")
    if len(ranked) > 1 and np.any(np.diff(scores[np.asarray(ranked, dtype=int)]) > 0):
        raise AssertionError("RAW_REFERENCE_RANKING_ORDER_INVALID")
    semantics = registry.ranker_by_id[ranker_id].zero_score_semantics
    active_set = set(active)
    if semantics == "dense" and active_set != set(range(candidate_count)):
        raise AssertionError("RAW_REFERENCE_DENSE_ELIGIBILITY_INVALID")
    if semantics == "inactive_is_zero" and np.any(
        scores[[index for index in range(candidate_count) if index not in active_set]]
        != 0.0
    ):
        raise AssertionError("RAW_REFERENCE_INACTIVE_ZERO_INVALID")
    if k < 0 or k > len(ranked):
        raise AssertionError("RAW_EXACT_K_UNAVAILABLE_NO_ZERO_PADDING")
    return tuple(candidates[index] for index in ranked[:k])


def _raw_exact_k_group_f1(
    ranking_row: pd.Series,
    *,
    k: int,
    ranker_id: str,
    registry: MethodRegistry,
    true_edges: set[tuple[int, ...]],
    equivalence_groups: list[set[tuple[int, ...]]],
) -> float:
    selected = _raw_exact_k_support(
        ranking_row, k=k, ranker_id=ranker_id, registry=registry
    )
    return float(
        calculate_support_metrics(selected, true_edges, equivalence_groups)["group_f1"]
    )


def _matched_cell_effects(
    cell: pd.DataFrame,
    selected_cell: pd.DataFrame,
    ranking_cell: pd.DataFrame,
    dataset_rows: pd.DataFrame,
    method_lookup: dict[tuple[str, str], str],
    registry: MethodRegistry,
) -> tuple[float, float, float, list[dict[str, Any]], float, dict[str, float]]:
    values = {
        str(row.method_id): float(row.group_f1) for row in cell.itertuples(index=False)
    }
    true_edges, equivalence_groups = _dataset_truth(dataset_rows)
    raw_rows_by_ranker: dict[str, pd.Series] = {}
    cpss_k: dict[str, int] = {}
    cpss_f1: dict[str, float] = {}
    for ranker in CONFIRMATORY_RANKERS:
        raw_rows = ranking_cell[
            (ranking_cell["ranker_id"].astype(str) == ranker)
            & (ranking_cell["score_source"].astype(str) == "final_ranker_score")
        ]
        if len(raw_rows) != 1:
            raise AssertionError("RAW_REFERENCE_SCORE_FAMILY_MISSING_OR_DUPLICATE")
        raw_rows_by_ranker[ranker] = raw_rows.iloc[0]
        method_id = method_lookup[(ranker, "cpss_one_se")]
        selected_rows = selected_cell[
            selected_cell["method_id"].astype(str) == method_id
        ]
        if len(selected_rows) != 1:
            raise AssertionError("CPSS_SELECTED_SUPPORT_MISSING_OR_DUPLICATE")
        selected_row = selected_rows.iloc[0]
        selected_support = _edge_sequence(selected_row["edges_json"], "CPSS edges_json")
        realized_k = int(selected_row["selected_edge_count"])
        if realized_k != len(selected_support):
            raise AssertionError("CPSS_REALIZED_K_MISMATCH")
        observed_f1 = values[method_id]
        recomputed_f1 = float(
            calculate_support_metrics(selected_support, true_edges, equivalence_groups)[
                "group_f1"
            ]
        )
        if not math.isclose(observed_f1, recomputed_f1, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("CPSS_GROUP_F1_DOES_NOT_MATCH_SAVED_SUPPORT")
        cpss_k[ranker] = realized_k
        cpss_f1[ranker] = observed_f1
    raw_reference_rows: list[dict[str, Any]] = []
    raw_f1: dict[tuple[str, str], float] = {}
    for anchor_ranker in CONFIRMATORY_RANKERS:
        anchor_k = cpss_k[anchor_ranker]
        for target_ranker in CONFIRMATORY_RANKERS:
            reference_f1 = _raw_exact_k_group_f1(
                raw_rows_by_ranker[target_ranker],
                k=anchor_k,
                ranker_id=target_ranker,
                registry=registry,
                true_edges=true_edges,
                equivalence_groups=equivalence_groups,
            )
            raw_f1[(target_ranker, anchor_ranker)] = reference_f1
            raw_reference_rows.append(
                {
                    "anchor_ranker_id": anchor_ranker,
                    "target_ranker_id": target_ranker,
                    "anchor_k": anchor_k,
                    "raw_exact_k_group_f1": reference_f1,
                    "cpss_group_f1": (
                        cpss_f1[target_ranker]
                        if target_ranker == anchor_ranker
                        else math.nan
                    ),
                    "own_anchor_absolute_cpss_residual": (
                        abs(cpss_f1[target_ranker] - reference_f1)
                        if target_ranker == anchor_ranker
                        else math.nan
                    ),
                    "own_anchor_signed_cpss_minus_raw": (
                        cpss_f1[target_ranker] - reference_f1
                        if target_ranker == anchor_ranker
                        else math.nan
                    ),
                    "signed_component_role": "DESCRIPTIVE_NO_INFERENCE",
                    "status": "OK",
                }
            )
    signed_ranker_gaps = {
        anchor: raw_f1[("shil", anchor)] - raw_f1[("l1", anchor)]
        for anchor in CONFIRMATORY_RANKERS
    }
    for row in raw_reference_rows:
        row["signed_raw_shil_minus_l1_at_anchor"] = signed_ranker_gaps[
            str(row["anchor_ranker_id"])
        ]
    ranker_effect = (
        math.fsum(
            abs(raw_f1[("shil", anchor)] - raw_f1[("l1", anchor)])
            for anchor in CONFIRMATORY_RANKERS
        )
        / 2.0
    )
    policy_effect = (
        math.fsum(
            abs(cpss_f1[ranker] - raw_f1[(ranker, ranker)])
            for ranker in CONFIRMATORY_RANKERS
        )
        / 2.0
    )
    all_policy_pairwise = [
        abs(
            values[method_lookup[(ranker, left)]]
            - values[method_lookup[(ranker, right)]]
        )
        for ranker in CONFIRMATORY_RANKERS
        for left, right in itertools.combinations(
            (
                "fixed_k.k01",
                "fixed_k.k02",
                "fixed_k.k04",
                "fixed_k.k08",
                "fixed_k.k16",
                "validation_one_se",
                "cpss_one_se",
            ),
            2,
        )
    ]
    descriptive_menu_spread = math.fsum(all_policy_pairwise) / 42.0
    return (
        ranker_effect,
        policy_effect,
        policy_effect - ranker_effect,
        raw_reference_rows,
        descriptive_menu_spread,
        {
            "signed_cpss_minus_raw_shil": cpss_f1["shil"] - raw_f1[("shil", "shil")],
            "signed_cpss_minus_raw_l1": cpss_f1["l1"] - raw_f1[("l1", "l1")],
            "signed_raw_shil_minus_l1_at_shil_anchor": signed_ranker_gaps["shil"],
            "signed_raw_shil_minus_l1_at_l1_anchor": signed_ranker_gaps["l1"],
        },
    )


def _wild_bootstrap_t(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape != (13, 10) or not np.all(np.isfinite(matrix)):
        raise AssertionError("Wild bootstrap requires the complete finite 13x10 matrix")
    seed_cluster_means = matrix.mean(axis=0)
    theta_hat = float(seed_cluster_means.mean())
    residuals = seed_cluster_means - theta_hat
    observed_se = float(seed_cluster_means.std(ddof=1) / math.sqrt(10.0))
    second_moment = float(np.mean(residuals**2))
    skewness = (
        float(np.mean(residuals**3) / second_moment**1.5)
        if second_moment > 0.0
        else None
    )
    _, residual_counts = np.unique(residuals, return_counts=True)
    base = {
        "theta_hat": theta_hat,
        "standard_error": observed_se,
        "bootstrap_replicates": CONFIRMATORY_BOOTSTRAP_REPLICATES,
        "bootstrap_kind": "exhaustive_seed_cluster_rademacher_studentized",
        "minimum_finite_studentized_replicates": CONFIRMATORY_MIN_FINITE_T,
        "seed_cluster_means": [float(value) for value in seed_cluster_means],
        "seed_cluster_mean_role": "DESCRIPTIVE_NO_INFERENCE",
        "seed_cluster_skewness": skewness,
        "zero_cluster_residual_count": int(np.sum(residuals == 0.0)),
        "tied_cluster_residual_count": int(
            np.sum(residual_counts[residual_counts > 1])
        ),
        "tied_cluster_residual_count_definition": (
            "number_of_cluster_residual_observations_participating_in_an_exact_tie"
        ),
        "symmetry_diagnostics_role": "DESCRIPTIVE_NO_INFERENCE_NON_GATING",
    }
    if not math.isfinite(observed_se) or observed_se <= 0.0:
        return {
            **base,
            "finite_studentized_replicates": 0,
            "discarded_studentized_replicates": CONFIRMATORY_BOOTSTRAP_REPLICATES,
            "small_bootstrap_se_replicates": 0,
            "ci_lower": None,
            "ci_upper": None,
            "verdict": "UNAVAILABLE",
            "reason": "OBSERVED_STANDARD_ERROR_ZERO_OR_NONFINITE",
        }
    pattern_ids = np.arange(CONFIRMATORY_BOOTSTRAP_REPLICATES, dtype=np.uint16)
    bit_positions = np.arange(10, dtype=np.uint16)
    multipliers = ((pattern_ids[:, None] >> bit_positions[None, :]) & 1).astype(
        float
    ) * 2.0 - 1.0
    bootstrap = theta_hat + residuals[None, :] * multipliers
    bootstrap_theta = bootstrap.mean(axis=1)
    bootstrap_se = bootstrap.std(axis=1, ddof=1) / math.sqrt(10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = (bootstrap_theta - theta_hat) / bootstrap_se
    finite = studentized[np.isfinite(studentized)]
    finite_count = int(len(finite))
    discarded_count = CONFIRMATORY_BOOTSTRAP_REPLICATES - finite_count
    small_se_count = int(np.sum(bootstrap_se < 0.1 * observed_se))
    if finite_count < CONFIRMATORY_MIN_FINITE_T:
        return {
            **base,
            "finite_studentized_replicates": finite_count,
            "discarded_studentized_replicates": discarded_count,
            "small_bootstrap_se_replicates": small_se_count,
            "ci_lower": None,
            "ci_upper": None,
            "verdict": "UNAVAILABLE",
            "reason": "FINITE_STUDENTIZED_REPLICATES_BELOW_1000",
        }
    lower_t, upper_t = np.quantile(finite, [0.025, 0.975], method="linear")
    ci_lower = float(theta_hat - upper_t * observed_se)
    ci_upper = float(theta_hat - lower_t * observed_se)
    return {
        **base,
        "finite_studentized_replicates": finite_count,
        "discarded_studentized_replicates": discarded_count,
        "small_bootstrap_se_replicates": small_se_count,
        "studentized_quantile_025": float(lower_t),
        "studentized_quantile_975": float(upper_t),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "verdict": (
            "DEMONSTRATED_ON_THE_FIXED_MENU"
            if ci_lower > 0.0
            else "NOT_DEMONSTRATED_ON_THE_FIXED_MENU"
        ),
        "reason": (
            "LOWER_CI_STRICTLY_ABOVE_ZERO"
            if ci_lower > 0.0
            else "LOWER_CI_NOT_STRICTLY_ABOVE_ZERO"
        ),
    }


def confirmatory_decomposition(
    metrics: pd.DataFrame,
    selected_edges: pd.DataFrame,
    ranking_scores: pd.DataFrame,
    dataset_manifest: pd.DataFrame,
    partitions: pd.DataFrame,
    registry: MethodRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the locked group-F1 matched decomposition without cell dropping."""

    required_columns = set(KEYS) | {"method_id", "ranker_id", "policy_id", "group_f1"}
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise AssertionError(
            f"Confirmatory metric input is missing columns: {sorted(missing_columns)}"
        )
    if not {"scenario", "split_seed"}.issubset(partitions.columns):
        raise AssertionError("Partition input lacks confirmatory cell keys")
    expected_methods, method_lookup, policy_levels = _confirmatory_method_contract(
        registry
    )
    confirmatory_metrics = metrics[
        metrics["scenario"].astype(str).isin(CONFIRMATORY_SCENARIOS)
    ].copy()
    confirmatory_partitions = partitions[
        partitions["scenario"].astype(str).isin(CONFIRMATORY_SCENARIOS)
    ].copy()
    confirmatory_selected = selected_edges[
        selected_edges["scenario"].astype(str).isin(CONFIRMATORY_SCENARIOS)
    ].copy()
    confirmatory_rankings = ranking_scores[
        ranking_scores["scenario"].astype(str).isin(CONFIRMATORY_SCENARIOS)
    ].copy()
    cell_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    adaptive_reference_rows: list[dict[str, Any]] = []
    validation_invariant_rows: list[dict[str, Any]] = []
    dataset_has_keys = {"scenario", "split_seed"}.issubset(dataset_manifest.columns)
    for scenario in CONFIRMATORY_SCENARIOS:
        for seed in CONFIRMATORY_SEEDS:
            cell = confirmatory_metrics[
                (confirmatory_metrics["scenario"].astype(str) == scenario)
                & (pd.to_numeric(confirmatory_metrics["split_seed"]) == seed)
            ]
            partition = confirmatory_partitions[
                (confirmatory_partitions["scenario"].astype(str) == scenario)
                & (pd.to_numeric(confirmatory_partitions["split_seed"]) == seed)
            ]
            selected_cell = confirmatory_selected[
                (confirmatory_selected["scenario"].astype(str) == scenario)
                & (pd.to_numeric(confirmatory_selected["split_seed"]) == seed)
            ]
            ranking_cell = confirmatory_rankings[
                (confirmatory_rankings["scenario"].astype(str) == scenario)
                & (pd.to_numeric(confirmatory_rankings["split_seed"]) == seed)
            ]
            dataset_rows = (
                dataset_manifest[
                    (dataset_manifest["scenario"].astype(str) == scenario)
                    & (
                        pd.to_numeric(dataset_manifest["split_seed"], errors="coerce")
                        == seed
                    )
                ]
                if dataset_has_keys
                else dataset_manifest.iloc[0:0]
            )
            reason = _cell_failure_reason(
                cell, selected_cell, partition, expected_methods
            )
            if reason is None:
                try:
                    (
                        ranker_effect,
                        policy_effect,
                        difference,
                        raw_reference_rows,
                        descriptive_menu_spread,
                        signed_components,
                    ) = _matched_cell_effects(
                        cell,
                        selected_cell,
                        ranking_cell,
                        dataset_rows,
                        method_lookup,
                        registry,
                    )
                    for reference_row in raw_reference_rows:
                        adaptive_reference_rows.append(
                            {"scenario": scenario, "split_seed": seed, **reference_row}
                        )
                    raw_by_ranker = {
                        ranker: ranking_cell[
                            (ranking_cell["ranker_id"].astype(str) == ranker)
                            & (
                                ranking_cell["score_source"].astype(str)
                                == "final_ranker_score"
                            )
                        ].iloc[0]
                        for ranker in CONFIRMATORY_RANKERS
                    }
                    selected_by_method = selected_cell.set_index("method_id")
                    for ranker in CONFIRMATORY_RANKERS:
                        try:
                            validation_method = method_lookup[
                                (ranker, "validation_one_se")
                            ]
                            validation_row = selected_by_method.loc[validation_method]
                            validation_support = _edge_sequence(
                                validation_row["edges_json"],
                                "validation-one-SE edges_json",
                            )
                            validation_k = int(validation_row["selected_edge_count"])
                            if validation_k != len(validation_support):
                                raise AssertionError("VALIDATION_REALIZED_K_MISMATCH")
                            raw_support = _raw_exact_k_support(
                                raw_by_ranker[ranker],
                                k=validation_k,
                                ranker_id=ranker,
                                registry=registry,
                            )
                            invariant_status = "OK"
                            invariant_matches = validation_support == raw_support
                        except (
                            AssertionError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as error:
                            validation_k = math.nan
                            invariant_status = str(error) or type(error).__name__
                            invariant_matches = False
                        validation_invariant_rows.append(
                            {
                                "scenario": scenario,
                                "split_seed": seed,
                                "ranker_id": ranker,
                                "validation_selected_k": validation_k,
                                "support_matches_raw_exact_k": invariant_matches,
                                "status": invariant_status,
                                "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
                            }
                        )
                    status = "OK"
                except (AssertionError, KeyError, TypeError, ValueError) as error:
                    status = str(error) or type(error).__name__
                    ranker_effect = policy_effect = difference = math.nan
                    descriptive_menu_spread = math.nan
                    signed_components = {}
            else:
                ranker_effect = policy_effect = difference = math.nan
                descriptive_menu_spread = math.nan
                signed_components = {}
                status = reason
            if status != "OK":
                for ranker in CONFIRMATORY_RANKERS:
                    for target_ranker in CONFIRMATORY_RANKERS:
                        adaptive_reference_rows.append(
                            {
                                "scenario": scenario,
                                "split_seed": seed,
                                "anchor_ranker_id": ranker,
                                "target_ranker_id": target_ranker,
                                "anchor_k": math.nan,
                                "raw_exact_k_group_f1": math.nan,
                                "cpss_group_f1": math.nan,
                                "own_anchor_absolute_cpss_residual": math.nan,
                                "own_anchor_signed_cpss_minus_raw": math.nan,
                                "signed_raw_shil_minus_l1_at_anchor": math.nan,
                                "signed_component_role": "DESCRIPTIVE_NO_INFERENCE",
                                "status": status,
                            }
                        )
                    validation_invariant_rows.append(
                        {
                            "scenario": scenario,
                            "split_seed": seed,
                            "ranker_id": ranker,
                            "validation_selected_k": math.nan,
                            "support_matches_raw_exact_k": False,
                            "status": status,
                            "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
                        }
                    )
            cell_rows.append(
                {
                    "scenario": scenario,
                    "split_seed": seed,
                    "status": status,
                    "confirmatory_metric": "group_f1",
                    "ranker_effect_R": ranker_effect,
                    "policy_effect_P": policy_effect,
                    "difference_D": difference,
                    "descriptive_all_policy_pairwise_spread": descriptive_menu_spread,
                    "descriptive_all_policy_pairwise_spread_role": (
                        "DESCRIPTIVE_NO_INFERENCE"
                    ),
                    **{
                        key: signed_components.get(key, math.nan)
                        for key in (
                            "signed_cpss_minus_raw_shil",
                            "signed_cpss_minus_raw_l1",
                            "signed_raw_shil_minus_l1_at_shil_anchor",
                            "signed_raw_shil_minus_l1_at_l1_anchor",
                        )
                    },
                    "signed_component_role": "DESCRIPTIVE_NO_INFERENCE",
                }
            )
            by_method = (
                {
                    str(row.method_id): float(row.group_f1)
                    for row in cell.itertuples(index=False)
                    if math.isfinite(float(row.group_f1))
                }
                if reason is None
                else {}
            )
            for k in FIXED_K_VALUES:
                level = f"fixed_k.k{k:02d}"
                shil_id = method_lookup[("shil", level)]
                l1_id = method_lookup[("l1", level)]
                fixed_status = (
                    "OK" if shil_id in by_method and l1_id in by_method else status
                )
                shil_value = by_method.get(shil_id, math.nan)
                l1_value = by_method.get(l1_id, math.nan)
                fixed_rows.append(
                    {
                        "scenario": scenario,
                        "split_seed": seed,
                        "k": k,
                        "status": fixed_status,
                        "shil_group_f1": shil_value,
                        "l1_group_f1": l1_value,
                        "shil_minus_l1": (
                            shil_value - l1_value if fixed_status == "OK" else math.nan
                        ),
                        "absolute_ranker_difference": (
                            abs(shil_value - l1_value)
                            if fixed_status == "OK"
                            else math.nan
                        ),
                        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
                    }
                )
    cells = pd.DataFrame(cell_rows)
    fixed = pd.DataFrame(fixed_rows)
    adaptive_references = pd.DataFrame(adaptive_reference_rows)
    validation_invariants = pd.DataFrame(validation_invariant_rows)
    failed = cells[cells["status"] != "OK"]
    ok_references = adaptive_references[adaptive_references["status"] == "OK"]
    zero_reference_count = int(
        np.sum(pd.to_numeric(ok_references["anchor_k"], errors="coerce") == 0)
    )
    ok_cells = cells[cells["status"] == "OK"]
    coincident_anchor_pair_count = 0
    for _, cell_references in ok_references.groupby(["scenario", "split_seed"]):
        anchors = cell_references.groupby("anchor_ranker_id")["anchor_k"].first()
        if set(anchors.index) == set(CONFIRMATORY_RANKERS) and int(
            anchors["shil"]
        ) == int(anchors["l1"]):
            coincident_anchor_pair_count += 1
    contract = {
        "metric": "group_f1",
        "scenarios": list(CONFIRMATORY_SCENARIOS),
        "seeds": list(CONFIRMATORY_SEEDS),
        "rankers": list(CONFIRMATORY_RANKERS),
        "confirmatory_policy": "cpss_one_se",
        "descriptive_policy_levels": list(policy_levels),
        "primary_method_ids": list(expected_methods),
        "raw_exact_k_reference_count_per_cell": 4,
        "ranker_contrast_count_per_cell": 2,
        "cpss_residual_count_per_cell": 2,
        "ranker_effect_definition": "mean SHIL-L1 raw group_f1 gap at the SHIL-CPSS and L1-CPSS realized-K anchors",
        "policy_effect_definition": "mean CPSS-vs-same-ranker-raw-exact-k group_f1 residual",
        "all_policy_pairwise_spread_role": "DESCRIPTIVE_NO_INFERENCE",
        "cell_weighting": "equal_13_scenarios_x_10_seeds",
        "no_imputation_drop_or_rank_renormalization": True,
        "k_zero_reference_count": zero_reference_count,
        "coincident_anchor_pair_count": coincident_anchor_pair_count,
        "exact_zero_D_count": int(np.sum(ok_cells["difference_D"] == 0.0)),
        "aggregate_count_scope": (
            "complete_13x10_grid" if failed.empty else "available_OK_cells_only"
        ),
        "signed_component_role": "DESCRIPTIVE_NO_INFERENCE",
        "confirmatory_label_semantics": {
            "DEMONSTRATED_ON_THE_FIXED_MENU": (
                "within the frozen menu, mean absolute CPSS-versus-own-raw discrepancy "
                "exceeds mean absolute raw-ranker discrepancy at the same two "
                "CPSS-realized anchor slots"
            ),
            "NOT_DEMONSTRATED_ON_THE_FIXED_MENU": (
                "the frozen one-sided interval criterion was not met"
            ),
        },
        "no_benefit_superiority_direction_quality_or_causal_inference": True,
        "descriptive_artifact_role": "DESCRIPTIVE_NO_INFERENCE",
    }
    if not failed.empty:
        summary = {
            **contract,
            "theta_hat": None,
            "standard_error": None,
            "ci_lower": None,
            "ci_upper": None,
            "bootstrap_replicates": CONFIRMATORY_BOOTSTRAP_REPLICATES,
            "bootstrap_kind": "exhaustive_seed_cluster_rademacher_studentized",
            "finite_studentized_replicates": 0,
            "discarded_studentized_replicates": CONFIRMATORY_BOOTSTRAP_REPLICATES,
            "small_bootstrap_se_replicates": 0,
            "verdict": "UNAVAILABLE",
            "reason": "INCOMPLETE_OR_INVALID_CONFIRMATORY_CELLS",
            "invalid_cell_count": int(len(failed)),
            "invalid_status_counts": {
                str(key): int(value)
                for key, value in failed["status"].value_counts().sort_index().items()
            },
        }
        return cells, fixed, adaptive_references, validation_invariants, summary
    matrix = (
        cells.pivot(index="scenario", columns="split_seed", values="difference_D")
        .reindex(index=CONFIRMATORY_SCENARIOS, columns=CONFIRMATORY_SEEDS)
        .to_numpy(dtype=float)
    )
    summary = {
        **contract,
        **_wild_bootstrap_t(matrix),
        "invalid_cell_count": 0,
        "invalid_status_counts": {},
    }
    return cells, fixed, adaptive_references, validation_invariants, summary


def analyze(
    input_dir: Path,
    output_dir: Path,
    *,
    require_full_scientific: bool = True,
) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty analysis directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        metrics,
        supports,
        ranking_metrics,
        ranking_scores,
        partitions,
        dataset_manifest,
        run_summary,
        _config,
        registry,
    ) = validate_evidence(input_dir, require_full_scientific=require_full_scientific)
    metrics = add_run_mean_jaccard(metrics, supports)
    method_summary = _mean_summary(
        metrics, ["scenario", "method_id", "ranker_id", "policy_id"]
    )
    (
        decomposition,
        fixed_k_matched,
        adaptive_raw_references,
        validation_one_se_invariants,
        confirmatory,
    ) = confirmatory_decomposition(
        metrics,
        supports,
        ranking_scores,
        dataset_manifest,
        partitions,
        registry,
    )
    stability = stability_table(metrics, supports, ranking_scores)
    ranking_summary = (
        ranking_metrics.groupby(
            ["scenario", "score_family_id", "ranker_id", "score_source"],
            dropna=False,
        )
        .agg(
            runs=("score_family_id", "size"),
            candidate_average_precision_mean=("candidate_average_precision", "mean"),
            candidate_average_precision_sd=("candidate_average_precision", "std"),
            nonzero_score_count_mean=("nonzero_score_count", "mean"),
        )
        .reset_index()
    )
    for descriptive_table in (
        metrics,
        method_summary,
        stability,
        ranking_summary,
    ):
        descriptive_table["analysis_role"] = "DESCRIPTIVE_NO_INFERENCE"
    metrics.to_csv(output_dir / "metrics_with_run_stability.csv", index=False)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    decomposition.to_csv(output_dir / "ranker_policy_decomposition.csv", index=False)
    fixed_k_matched.to_csv(output_dir / "fixed_k_matched_cardinality.csv", index=False)
    adaptive_raw_references.to_csv(
        output_dir / "cpss_anchor_raw_exact_k_references.csv", index=False
    )
    validation_one_se_invariants.to_csv(
        output_dir / "validation_one_se_raw_invariants.csv", index=False
    )
    structured_sensitivity_validation = run_summary["_validated_structured_sensitivity"]
    (output_dir / "structured_sensitivity_summary.json").write_text(
        json.dumps(structured_sensitivity_validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "confirmatory_decomposition.json").write_text(
        json.dumps(confirmatory, indent=2, sort_keys=True), encoding="utf-8"
    )
    stability.to_csv(output_dir / "support_stability.csv", index=False)
    ranking_summary.to_csv(output_dir / "ranking_family_summary.csv", index=False)
    payload = {
        "experiment_id": run_summary["experiment_id"],
        "mode": run_summary["mode"],
        "verification_status": "ANALYZED",
        "decision": confirmatory["verdict"],
        "decision_reason": confirmatory["reason"],
        "structured_sensitivity_status_counts": structured_sensitivity_validation[
            "status_counts"
        ],
        "structured_sensitivity_validation": structured_sensitivity_validation,
        "primary_artifact_contract": {
            "aggregate_source": "metrics.csv",
            "decision_source": "confirmatory_decomposition.json",
            "plot_source": "ranker_policy_decomposition.csv",
            "structured_sensitivity_excluded": True,
            "structured_sensitivity_artifact_role": "DESCRIPTIVE_NO_INFERENCE",
            "structured_sensitivity_summary_source": (
                "structured_sensitivity_summary.json"
            ),
            "confirmatory_metric": "group_f1",
            "confirmatory_scenarios": "S01-S13",
            "estimator": "cpss_own_raw_exact_k_residual_minus_cross_ranker_gap_at_two_cpss_k_anchors",
            "fixed_k_descriptive_source": "fixed_k_matched_cardinality.csv",
            "cpss_anchor_reference_source": "cpss_anchor_raw_exact_k_references.csv",
            "validation_one_se_descriptive_invariant_source": (
                "validation_one_se_raw_invariants.csv"
            ),
            "descriptive_artifact_role": "DESCRIPTIVE_NO_INFERENCE",
        },
        "confirmatory": confirmatory,
        "registry_sha256": registry.canonical_sha256,
        "validated_partition_count": len(partitions),
        "expected_method_count": len(registry.methods),
        "expected_score_family_count": len(
            {method.score_family_id for method in registry.methods}
        ),
        "input_method_rows": len(metrics),
        "input_support_rows": len(supports),
        "input_ranking_metric_rows": len(ranking_metrics),
        "all_registry_rows_present": True,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return analyze(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
