from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


KEYS = ["dataset", "scenario", "data_seed", "split_seed"]
METHODS = ["L1-full", "L1-top8", "L1-match", "SHIL-k8", "SC-L1", "SC-SHIL"]


def parse_edge(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in str(text).split("-"))


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
        for edge in support:
            if edge in lookup:
                matrix[row_index, lookup[edge]] = 1.0
    average_size = matrix.sum(axis=1).mean()
    p = matrix.shape[1]
    denominator = (average_size / p) * (1.0 - average_size / p)
    if denominator <= 0:
        return 1.0 if np.all(matrix == matrix[0]) else math.nan
    numerator = matrix.var(axis=0, ddof=1).mean()
    return float(1.0 - numerator / denominator)


def scenario_stratified_bootstrap(
    paired: pd.DataFrame,
    value_column: str,
    n_resamples: int,
    seed: int = 20260717,
) -> dict[str, float]:
    observed = float(
        paired.groupby("scenario", dropna=False)[value_column].mean().mean()
    )
    rng = np.random.default_rng(seed)
    groups = [group[value_column].to_numpy(dtype=float) for _, group in paired.groupby("scenario", dropna=False)]
    samples = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        scenario_means = [
            float(np.mean(rng.choice(values, size=len(values), replace=True)))
            for values in groups
        ]
        samples[i] = float(np.mean(scenario_means))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean": observed, "ci95_low": float(low), "ci95_high": float(high)}


def validate_evidence(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    required = [
        "metrics.csv",
        "selected_edges.csv",
        "dataset_manifest.csv",
        "run_summary.json",
        "config_resolved.json",
        "output_hashes.json",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing run outputs: {missing}")
    metrics = pd.read_csv(input_dir / "metrics.csv")
    edges = pd.read_csv(input_dir / "selected_edges.csv")
    summary = json.loads((input_dir / "run_summary.json").read_text(encoding="utf-8"))
    config = json.loads((input_dir / "config_resolved.json").read_text(encoding="utf-8"))
    if set(metrics["model"]) != set(METHODS):
        raise AssertionError(f"Method set mismatch: {sorted(metrics['model'].unique())}")
    if metrics.duplicated(KEYS + ["model"]).any():
        raise AssertionError("Duplicate metric keys")
    expected_rows = int(summary["dataset_runs"]) * len(METHODS)
    if len(metrics) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} metric rows, observed {len(metrics)}")
    edge_counts = edges.groupby(KEYS + ["model"], dropna=False).size().rename("edge_rows")
    observed = metrics.set_index(KEYS + ["model"])["selected_edges"]
    aligned = observed.to_frame().join(edge_counts, how="left").fillna({"edge_rows": 0})
    if not np.array_equal(aligned["selected_edges"].astype(int), aligned["edge_rows"].astype(int)):
        raise AssertionError("Selected-edge counts do not match edge rows")
    return metrics, edges, summary, config


def add_run_mean_jaccard(metrics: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    supports: dict[tuple[Any, ...], set[tuple[int, ...]]] = {}
    for key, group in edges.groupby(KEYS + ["model"], dropna=False):
        supports[key] = {parse_edge(value) for value in group["edge"]}
    output = metrics.copy()
    run_values: list[float] = []
    for row in output.itertuples(index=False):
        key = tuple(getattr(row, column) for column in KEYS) + (row.model,)
        own = supports.get(key, set())
        peers = []
        for other in output[(output["scenario"] == row.scenario) & (output["model"] == row.model)].itertuples(index=False):
            other_key = tuple(getattr(other, column) for column in KEYS) + (other.model,)
            if other_key == key:
                continue
            peer = supports.get(other_key, set())
            union = own | peer
            peers.append(1.0 if not union else len(own & peer) / len(union))
        run_values.append(float(np.mean(peers)) if peers else math.nan)
    output["run_mean_jaccard"] = run_values
    return output


def stability_table(metrics: pd.DataFrame, edges: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    supports: dict[tuple[Any, ...], set[tuple[int, ...]]] = {}
    for key, group in edges.groupby(KEYS + ["model"], dropna=False):
        supports[key] = {parse_edge(value) for value in group["edge"]}
    rows: list[dict[str, Any]] = []
    for (scenario, model), group in metrics.groupby(["scenario", "model"], dropna=False):
        scenario_supports = []
        for row in group.itertuples(index=False):
            key = tuple(getattr(row, column) for column in KEYS) + (row.model,)
            scenario_supports.append(supports.get(key, set()))
        d = int(group["candidate_edges"].iloc[0])
        feature_count = int(group["feature_count"].iloc[0])
        universe = []
        for order in config["candidate_orders"]:
            universe.extend(itertools.combinations(range(feature_count), int(order)))
        jaccard = pairwise_jaccard(scenario_supports)
        rows.append(
            {
                "scenario": scenario,
                "model": model,
                "runs": len(scenario_supports),
                "candidate_edges": d,
                "pairwise_jaccard_mean": float(np.mean(jaccard)) if jaccard else math.nan,
                "pairwise_jaccard_sd": float(np.std(jaccard, ddof=1)) if len(jaccard) > 1 else math.nan,
                "pairwise_jaccard_min": float(np.min(jaccard)) if jaccard else math.nan,
                "pairwise_jaccard_max": float(np.max(jaccard)) if jaccard else math.nan,
                "nogueira_stability": nogueira_stability(scenario_supports, universe),
            }
        )
    return pd.DataFrame(rows)


def paired_difference(metrics: pd.DataFrame, left: str, right: str, metric: str) -> pd.DataFrame:
    left_frame = metrics[metrics.model == left][KEYS + [metric]].rename(columns={metric: "left"})
    right_frame = metrics[metrics.model == right][KEYS + [metric]].rename(columns={metric: "right"})
    merged = left_frame.merge(right_frame, on=KEYS, validate="one_to_one")
    merged["difference"] = merged["left"] - merged["right"]
    return merged


def decide(metrics: pd.DataFrame, config: dict[str, Any], mode: str) -> tuple[str, pd.DataFrame]:
    if mode != "full-synthetic":
        return "NON_PROMOTED_OR_REAL_ONLY", pd.DataFrame()
    n_resamples = int(config["bootstrap_resamples"])
    rows: list[dict[str, Any]] = []
    for comparator in ["L1-match", "SC-L1"]:
        for metric in ["support_f1", "false_inclusions", "log_loss"]:
            paired = paired_difference(metrics, "SC-SHIL", comparator, metric)
            estimate = scenario_stratified_bootstrap(paired, "difference", n_resamples)
            rows.append({"left": "SC-SHIL", "right": comparator, "metric": metric, **estimate})
    stability_paired = paired_difference(metrics, "SC-SHIL", "SHIL-k8", "run_mean_jaccard").dropna(subset=["difference"])
    stability_estimate = scenario_stratified_bootstrap(stability_paired, "difference", n_resamples)
    rows.append({"left": "SC-SHIL", "right": "SHIL-k8", "metric": "run_mean_jaccard", **stability_estimate})
    decisions = pd.DataFrame(rows)

    def row(right: str, metric: str) -> pd.Series:
        return decisions[(decisions.right == right) & (decisions.metric == metric)].iloc[0]

    margin = float(config["log_loss_noninferiority_margin"])
    method_supported = all(
        [
            row("L1-match", "support_f1").ci95_low > 0,
            row("SC-L1", "support_f1").ci95_low > 0,
            row("L1-match", "false_inclusions").ci95_high < 0,
            row("SC-L1", "false_inclusions").ci95_high < 0,
            row("L1-match", "log_loss").ci95_high < margin,
            row("SC-L1", "log_loss").ci95_high < margin,
            row("SHIL-k8", "run_mean_jaccard").ci95_low > 0,
        ]
    )
    if method_supported:
        return "METHOD_EXTENSION_SUPPORTED", decisions

    sc_vs_fixed_f1 = scenario_stratified_bootstrap(
        paired_difference(metrics, "SC-SHIL", "SHIL-k8", "support_f1"),
        "difference",
        n_resamples,
    )
    l1_vs_fixed_f1 = scenario_stratified_bootstrap(
        paired_difference(metrics, "SC-L1", "L1-top8", "support_f1"),
        "difference",
        n_resamples,
    )
    protocol_signal = sc_vs_fixed_f1["ci95_low"] > 0 or l1_vs_fixed_f1["ci95_low"] > 0
    return ("PROTOCOL_ONLY" if protocol_signal else "NEGATIVE_EXTENSION"), decisions


def analyze(input_dir: Path, output_dir: Path) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty analysis directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, edges, run_summary, config = validate_evidence(input_dir)
    metrics = add_run_mean_jaccard(metrics, edges)
    summary = metrics.groupby(["scenario", "model"], dropna=False).agg(
        n=("model", "size"),
        support_f1_mean=("support_f1", "mean"),
        support_f1_sd=("support_f1", "std"),
        false_inclusions_mean=("false_inclusions", "mean"),
        support_size_mean=("selected_edges", "mean"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        runtime_mean=("selection_seconds", "mean"),
        peak_rss_mean=("peak_rss_bytes", "mean"),
    ).reset_index()
    stability = stability_table(metrics, edges, config)
    verdict, paired = decide(metrics, config, str(run_summary["mode"]))
    metrics.to_csv(output_dir / "metrics_with_run_stability.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    stability.to_csv(output_dir / "support_stability.csv", index=False)
    paired.to_csv(output_dir / "paired_decision_intervals.csv", index=False)
    payload = {
        "experiment_id": run_summary["experiment_id"],
        "mode": run_summary["mode"],
        "verification_status": "ANALYZED",
        "decision": verdict,
        "input_method_rows": len(metrics),
        "input_selected_edge_rows": len(edges),
        "all_preregistered_rows_present": True,
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
