from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Inconsistent values: penalty=l1.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import shil_q1_extension as ext  # noqa: E402
import shil_run_experiments as base  # noqa: E402

OP_ID = "shil-q1-action-remediation-20260623-0740"
STAMP = "20260623_0740"
DATE_TIME = "2026-06-23 07:40 +03:00"
SEEDS_10 = [11, 23, 37, 53, 71, 89, 101, 131, 157, 181]
SEEDS_5 = [11, 23, 37, 53, 71]
SEEDS_SENSITIVITY = [11, 23, 37]
L1_C_GRID = [0.02, 0.05, 0.25, 1.0]
OUT = ROOT / "experiments" / "q1_extension"
MD_OUT = ROOT / "MD" / "06_results"


def ci95(mean: float, sd: float, n: int) -> tuple[float, float]:
    if n <= 1 or not np.isfinite(sd):
        return (math.nan, math.nan)
    tcrit = {4: 2.776, 9: 2.262}.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return mean - half, mean + half


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for metric in metrics:
            values = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(values) == 0:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_sd"] = math.nan
                row[f"{metric}_ci95_low"] = math.nan
                row[f"{metric}_ci95_high"] = math.nan
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            low, high = ci95(mean, sd, len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_diffs(df: pd.DataFrame, left_model: str, right_model: str, metrics: list[str], label: str) -> pd.DataFrame:
    left = df[df["model"] == left_model].copy()
    right = df[df["model"] == right_model].copy()
    merged = left.merge(right, on=["dataset", "seed"], suffixes=("_left", "_right"))
    rows = []
    for dataset, g in merged.groupby("dataset"):
        row = {"comparison": label, "dataset": dataset, "n": int(len(g))}
        for metric in metrics:
            diff = pd.to_numeric(g[f"{metric}_left"], errors="coerce") - pd.to_numeric(
                g[f"{metric}_right"], errors="coerce"
            )
            diff = diff.dropna()
            mean = float(diff.mean()) if len(diff) else math.nan
            sd = float(diff.std(ddof=1)) if len(diff) > 1 else math.nan
            low, high = ci95(mean, sd, len(diff))
            row[f"{metric}_diff_mean"] = mean
            row[f"{metric}_diff_sd"] = sd
            row[f"{metric}_diff_ci95_low"] = low
            row[f"{metric}_diff_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def l1_params(c_value: float, seed: int) -> dict:
    # All synthetic follow-up tasks are binary; liblinear gives the same L1
    # support-extraction target much faster than SAGA for this budget check.
    return {
        "solver": "liblinear",
        "penalty": "l1",
        "C": c_value,
        "max_iter": 800,
        "tol": 1e-3,
        "random_state": seed,
    }


def l1_topk_budget_check() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    edge_rows = []
    datasets = [ds for ds in base.load_datasets() if ds.name.startswith("synthetic_")]
    for ds in datasets:
        for seed in SEEDS_10:
            Xtr, Xv, Xte, ytr, yv, yte = base.split_scale(ds, seed)
            edges = base.candidate_edges(Xtr.shape[1], (2, 3))
            Ztr = ext.interaction_design(Xtr, edges)
            Zv = ext.interaction_design(Xv, edges)
            labels = np.arange(len(np.unique(np.r_[ytr, yv, yte])))
            best = None
            for c_value in L1_C_GRID:
                clf = LogisticRegression(**l1_params(c_value, seed))
                clf.fit(Ztr, ytr)
                loss = log_loss(yv, clf.predict_proba(Zv), labels=labels)
                if best is None or loss < best[0]:
                    best = (loss, c_value, clf)
            assert best is not None
            _, c_value, clf = best
            edge_coef = np.asarray(clf.coef_)[:, Xtr.shape[1] :]
            norms = np.linalg.norm(edge_coef, axis=0)
            active = np.flatnonzero(norms > 1e-8)
            ranked = active[np.argsort(norms[active])[::-1]]
            selected = [edges[i] for i in ranked[:8]]
            selected_set = set(selected)
            true_edges = ds.true_edges or set()
            hits = len(selected_set & true_edges)
            precision = hits / len(selected) if selected else math.nan
            recall = hits / len(true_edges) if true_edges else math.nan
            rows.append(
                {
                    "dataset": ds.name,
                    "model": "l1_pair_triple_top8",
                    "seed": seed,
                    "selected_c": c_value,
                    "nonzero_edges": int(len(active)),
                    "selected_edges": int(len(selected)),
                    "top8_true_hits": int(hits),
                    "support_precision": precision,
                    "support_recall": recall,
                }
            )
            for rank, edge_index in enumerate(ranked[:20], start=1):
                edge = edges[int(edge_index)]
                edge_rows.append(
                    {
                        "dataset": ds.name,
                        "model": "l1_pair_triple_top8_ranked_coefficients",
                        "seed": seed,
                        "rank": rank,
                        "edge_indices": "-".join(map(str, edge)),
                        "order": len(edge),
                        "coefficient_norm": float(norms[int(edge_index)]),
                        "is_true_edge": bool(edge in true_edges),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize(raw, ["dataset", "model"], ["nonzero_edges", "selected_edges", "support_precision", "support_recall"])
    return raw, summary, pd.DataFrame(edge_rows)


def make_sensitivity_dataset(scenario: str, seed: int) -> base.Dataset:
    rng = np.random.default_rng(seed)
    n = 1200 if scenario == "small_n_pair" else 2800
    d = 16
    if scenario == "correlated_pair":
        idx = np.arange(d)
        cov = 0.55 ** np.abs(np.subtract.outer(idx, idx))
        X = rng.multivariate_normal(np.zeros(d), cov, size=n)
        noise = 0.65
        threshold_quantile = 0.5
    else:
        X = rng.normal(size=(n, d))
        noise = 1.10 if scenario == "low_snr_pair" else 0.65
        threshold_quantile = 0.70 if scenario == "imbalanced_pair" else 0.5
    true_edges = {(0, 1), (2, 3), (4, 5)}
    linear = 0.15 * X[:, 12] - 0.12 * X[:, 13]
    score = 1.8 * X[:, 0] * X[:, 1] - 1.6 * X[:, 2] * X[:, 3] + 1.4 * X[:, 4] * X[:, 5]
    score = score + linear + rng.normal(scale=noise, size=n)
    threshold = np.quantile(score, threshold_quantile)
    y = (score > threshold).astype(int)
    return base.Dataset(scenario, X, y, [f"x{i}" for i in range(d)], true_edges)


def run_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    edge_rows = []
    scenarios = ["correlated_pair", "low_snr_pair", "small_n_pair", "imbalanced_pair"]
    for scenario in scenarios:
        for seed in SEEDS_SENSITIVITY:
            ds = make_sensitivity_dataset(scenario, seed)
            Xtr, Xv, Xte, ytr, yv, yte = base.split_scale(ds, seed)
            edges = base.candidate_edges(Xtr.shape[1], (2, 3))
            Ztr = ext.interaction_design(Xtr, edges)
            Zv = ext.interaction_design(Xv, edges)
            Zte = ext.interaction_design(Xte, edges)
            labels = np.arange(len(np.unique(np.r_[ytr, yv, yte])))
            best = None
            start = time.perf_counter()
            for c_value in L1_C_GRID:
                clf = LogisticRegression(**l1_params(c_value, seed))
                clf.fit(Ztr, ytr)
                loss = log_loss(yv, clf.predict_proba(Zv), labels=labels)
                if best is None or loss < best[0]:
                    best = (loss, c_value, clf)
            assert best is not None
            _, c_value, clf = best
            train_s = time.perf_counter() - start
            prob = clf.predict_proba(Zte)
            edge_coef = np.asarray(clf.coef_)[:, Xtr.shape[1] :]
            active = np.flatnonzero(np.linalg.norm(edge_coef, axis=0) > 1e-8)
            selected_edges = [edges[i] for i in active]
            selected = set(selected_edges)
            precision = len(selected & ds.true_edges) / len(selected) if selected else math.nan
            recall = len(selected & ds.true_edges) / len(ds.true_edges)
            l1_row, _ = base.metric_row(
                ds.name,
                "l1_pair_triple_full",
                seed,
                yte,
                prob,
                train_s,
                0.0,
                clf.coef_.size + clf.intercept_.size,
                {
                    "orders": "2+3",
                    "selected_edges": len(selected_edges),
                    "selected_c": c_value,
                    "support_precision": precision,
                    "support_recall": recall,
                },
            )
            rows.append(l1_row)
            norms = np.linalg.norm(edge_coef, axis=0)
            ranked_active = active[np.argsort(norms[active])[::-1]]
            for rank, edge_index in enumerate(ranked_active[:100], start=1):
                edge = edges[int(edge_index)]
                edge_rows.append(
                    {
                        "dataset": ds.name,
                        "model": "l1_pair_triple_full",
                        "seed": seed,
                        "rank": rank,
                        "edge_indices": "-".join(map(str, edge)),
                        "edge_names": " x ".join(ds.feature_names[j] for j in edge),
                        "order": len(edge),
                        "is_true_edge": bool(edge in ds.true_edges),
                        "coefficient_norm": float(norms[int(edge_index)]),
                    }
                )
            sh = base.SparseHypergraphSelector(orders=(2, 3), k=8)
            start = time.perf_counter()
            sh.fit(Xtr, ytr, Xv, yv, seed)
            train_s = time.perf_counter() - start
            prob = sh.predict_proba(Xte)
            selected = set(sh.selected_edges)
            precision = len(selected & ds.true_edges) / len(selected)
            recall = len(selected & ds.true_edges) / len(ds.true_edges)
            row, _ = base.metric_row(
                ds.name,
                "shil_pair_triple_k8",
                seed,
                yte,
                prob,
                train_s,
                0.0,
                sh.n_params,
                {
                    "orders": "2+3",
                    "selected_edges": len(sh.selected_edges),
                    "support_precision": precision,
                    "support_recall": recall,
                    "selected_c": math.nan,
                },
            )
            rows.append(row)
            for rank, idx in enumerate(sh.selected_idx, start=1):
                edge = sh.edges[idx]
                edge_rows.append(
                    {
                        "dataset": ds.name,
                        "model": "shil_pair_triple_k8",
                        "seed": seed,
                        "rank": rank,
                        "edge_indices": "-".join(map(str, edge)),
                        "edge_names": " x ".join(ds.feature_names[j] for j in edge),
                        "order": len(edge),
                        "is_true_edge": bool(edge in ds.true_edges),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize(
        raw,
        ["dataset", "model"],
        ["accuracy", "roc_auc_ovr", "selected_edges", "support_precision", "support_recall"],
    )
    return raw, summary, pd.DataFrame(edge_rows)


def write_existing_summaries(run_id: str) -> dict[str, Path]:
    corrected = pd.read_csv(OUT / "EXP-SHIL-Q1-002_corrected_l1_20260620_230450_raw.csv")
    scale = pd.read_csv(OUT / "EXP-SHIL-Q1-002_scale_20260620_232347_raw.csv")
    stability = pd.read_csv(ROOT / "experiments" / "frozen_table_selection_stability.csv")
    synthetic = corrected[corrected["dataset"].str.startswith("synthetic_")].copy()
    real = corrected[~corrected["dataset"].str.startswith("synthetic_")].copy()
    paths = {}
    paths["synthetic_uncertainty"] = MD_OUT / f"{run_id}_synthetic_l1_uncertainty.csv"
    paths["real_uncertainty"] = MD_OUT / f"{run_id}_real_followup_uncertainty.csv"
    paths["scale_uncertainty"] = MD_OUT / f"{run_id}_scale_stress_uncertainty.csv"
    paths["paired_diffs"] = MD_OUT / f"{run_id}_paired_differences.csv"
    paths["stability"] = MD_OUT / f"{run_id}_stability_numeric.csv"
    summarize(
        synthetic[synthetic["model"].isin(["l1_interactions_2+3", "shil_pair_triple_k8_ext"])],
        ["dataset", "model"],
        ["accuracy", "roc_auc_ovr", "selected_edges", "support_precision", "support_recall"],
    ).to_csv(paths["synthetic_uncertainty"], index=False)
    summarize(
        real[real["model"].isin(["l1_interactions_2", "l1_interactions_2+3", "shil_pair_triple_k8_ext"])],
        ["dataset", "model"],
        ["accuracy", "roc_auc_ovr", "selected_edges"],
    ).to_csv(paths["real_uncertainty"], index=False)
    summarize(
        scale[scale["model"].isin(["l1_pair_triple_full", "shil_pair_triple_k8"])],
        ["dataset", "model"],
        ["accuracy", "roc_auc_ovr", "selected_edges", "support_precision", "support_recall"],
    ).to_csv(paths["scale_uncertainty"], index=False)
    pd.concat(
        [
            paired_diffs(
                synthetic,
                "l1_interactions_2+3",
                "shil_pair_triple_k8_ext",
                ["accuracy", "roc_auc_ovr", "selected_edges", "support_precision"],
                "l1_pair_triple_minus_shil_pair_triple",
            ),
            paired_diffs(
                scale,
                "l1_pair_triple_full",
                "shil_pair_triple_k8",
                ["accuracy", "roc_auc_ovr", "selected_edges", "support_precision"],
                "scale_l1_pair_triple_minus_shil_pair_triple",
            ),
        ],
        ignore_index=True,
    ).to_csv(paths["paired_diffs"], index=False)
    stability.drop(columns=["index"], errors="ignore").to_csv(paths["stability"], index=False)
    return paths


def write_markdown_report(run_id: str, paths: dict[str, Path], topk_summary: pd.DataFrame, sensitivity_summary: pd.DataFrame) -> Path:
    report = MD_OUT / f"{run_id}_numeric_update.md"
    topk_rows = topk_summary.to_markdown(index=False, floatfmt=".4f")
    if sensitivity_summary.empty:
        sens_rows = "Bounded sensitivity rerun was attempted but not retained because the local SHIL rerun exceeded the practical runtime/process budget. The remediation uses the pre-approved D-003 fallback: strengthen the claim boundary instead of adding incomplete sensitivity evidence."
    else:
        sens_rows = sensitivity_summary.to_markdown(index=False, floatfmt=".4f")
    report.write_text(
        "\n".join(
            [
                "# Q1 Action Register Numeric Update",
                "",
                f"Date/time: {DATE_TIME}",
                "Tool: Codex",
                "Model, if known: GPT-5 Codex",
                f"Operation ID: {OP_ID}",
                "",
                "This generated report supports `q1-audit/action_register.md` items A-004, A-005, A-006, and A-007.",
                "",
                "## Generated CSVs",
                "",
                *[f"- `{p.relative_to(ROOT)}`" for p in paths.values()],
                "",
                "## Budget-Matched L1 Top-8 Summary",
                "",
                topk_rows,
                "",
                "Interpretation: the top-8 L1 coefficient ranking often recovers the planted edges with the same recall and precision as SHIL in the synthetic follow-up. The manuscript should therefore describe SHIL's compactness advantage relative to the tested nonzero L1 support, not as superiority over budget-matched L1 ranking.",
                "",
                "## Bounded Sensitivity Summary",
                "",
                sens_rows,
                "",
                "Interpretation: a bounded sensitivity rerun is not used as manuscript evidence unless the generated CSVs are complete. If it is not complete, the manuscript must explicitly state that correlated predictors, lower signal-to-noise, smaller samples, and class imbalance remain untested.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    run_id = f"EXP-SHIL-Q1-003_q1_action_{STAMP}"
    paths = write_existing_summaries(run_id)
    topk_raw, topk_summary, topk_edges = l1_topk_budget_check()
    sens_raw = pd.DataFrame()
    sens_summary = pd.DataFrame()
    sens_edges = pd.DataFrame()
    topk_raw_path = OUT / f"{run_id}_l1_top8_raw.csv"
    topk_summary_path = OUT / f"{run_id}_l1_top8_summary.csv"
    topk_edges_path = OUT / f"{run_id}_l1_top8_edges.csv"
    sens_raw_path = OUT / f"{run_id}_sensitivity_not_used_raw.csv"
    sens_summary_path = OUT / f"{run_id}_sensitivity_not_used_summary.csv"
    sens_edges_path = OUT / f"{run_id}_sensitivity_not_used_edges.csv"
    manifest_path = OUT / f"{run_id}_manifest.json"
    topk_raw.to_csv(topk_raw_path, index=False)
    topk_summary.to_csv(topk_summary_path, index=False)
    topk_edges.to_csv(topk_edges_path, index=False)
    pd.DataFrame(
        [
            {
                "status": "not_used",
                "reason": "Initial local bounded sensitivity rerun exceeded practical runtime/process budget; D-003 fallback is claim-boundary strengthening.",
            }
        ]
    ).to_csv(sens_summary_path, index=False)
    sens_raw.to_csv(sens_raw_path, index=False)
    sens_edges.to_csv(sens_edges_path, index=False)
    paths.update(
        {
            "topk_raw": topk_raw_path,
            "topk_summary": topk_summary_path,
            "topk_edges": topk_edges_path,
            "sensitivity_raw": sens_raw_path,
            "sensitivity_summary": sens_summary_path,
            "sensitivity_edges": sens_edges_path,
        }
    )
    report_path = write_markdown_report(run_id, paths, topk_summary, sens_summary)
    paths["report"] = report_path
    manifest = {
        "run_id": run_id,
        "date_time": DATE_TIME,
        "operation_id": OP_ID,
        "purpose": "Q1 action register remediation analyses for uncertainty, L1 top-k budget check, bounded sensitivity, and stability numeric reporting.",
        "seeds_topk": SEEDS_10,
        "seeds_sensitivity": [],
        "l1_c_grid": L1_C_GRID,
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
        "sensitivity_scenarios": [],
        "sensitivity_note": "Initial local bounded sensitivity rerun exceeded practical runtime/process budget and is not used as evidence; D-003 fallback is claim-boundary strengthening.",
        "python": sys.version,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
