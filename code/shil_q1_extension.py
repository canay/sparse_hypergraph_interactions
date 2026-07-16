from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import shil_run_experiments as base  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = ROOT / "experiments" / "q1_extension"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS_10 = [11, 23, 37, 53, 71, 89, 101, 131, 157, 181]


def optional_model_builders(seed: int) -> list[tuple[str, object]]:
    builders: list[tuple[str, object]] = []
    try:
        from xgboost import XGBClassifier

        builders.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=3,
                    learning_rate=0.035,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=seed,
                    n_jobs=4,
                    verbosity=0,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL_SKIP xgboost {type(exc).__name__}: {exc}", flush=True)
    try:
        from lightgbm import LGBMClassifier

        builders.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=350,
                    num_leaves=15,
                    learning_rate=0.035,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=4,
                    verbosity=-1,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL_SKIP lightgbm {type(exc).__name__}: {exc}", flush=True)
    try:
        from catboost import CatBoostClassifier

        builders.append(
            (
                "catboost",
                CatBoostClassifier(
                    iterations=350,
                    depth=4,
                    learning_rate=0.035,
                    l2_leaf_reg=3.0,
                    loss_function="MultiClass",
                    random_seed=seed,
                    allow_writing_files=False,
                    verbose=False,
                    thread_count=4,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MODEL_SKIP catboost {type(exc).__name__}: {exc}", flush=True)
    return builders


def add_digits_dataset(datasets: list[base.Dataset]) -> list[base.Dataset]:
    digits = load_digits()
    names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
    return datasets + [base.Dataset("digits_8x8", digits.data.astype(float), digits.target.astype(int), names)]


def param_count(model: object) -> int:
    if hasattr(model, "coef_"):
        return int(model.coef_.size + getattr(model, "intercept_", np.array([])).size)
    if hasattr(model, "n_features_in_"):
        return int(getattr(model, "n_features_in_", 0))
    return 0


def fit_predict_model(model, Xtr, ytr, Xv, yv, Xte, seed: int):
    t0 = time.perf_counter()
    name = model.__class__.__name__.lower()
    if "xgb" in name:
        model.set_params(num_class=int(len(np.unique(np.r_[ytr, yv]))))
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    elif "lgbm" in name:
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)])
    else:
        model.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob = model.predict_proba(Xte)
    pred_s = time.perf_counter() - t0
    return prob, train_s, pred_s


def interaction_design(X: np.ndarray, edges: list[tuple[int, ...]]) -> np.ndarray:
    return np.c_[X, base.interaction_matrix(X, edges)]


def run_l1_interaction(ds, Xtr, Xv, Xte, ytr, yv, yte, seed: int, orders=(2, 3), max_edges=6000, c_grid=None):
    edges = base.candidate_edges(Xtr.shape[1], orders)
    if len(edges) > max_edges:
        return None, []
    Ztr = interaction_design(Xtr, edges)
    Zv = interaction_design(Xv, edges)
    Zte = interaction_design(Xte, edges)
    labels = np.arange(len(np.unique(np.r_[ytr, yv, yte])))
    best = None
    t0 = time.perf_counter()
    for c_value in (c_grid or [0.05, 0.25, 1.0]):
        lr_params = {
            "solver": "saga",
            "C": c_value,
            "max_iter": 1200,
            "tol": 1e-3,
            "random_state": seed,
        }
        penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
        if penalty_default == "deprecated":
            lr_params["l1_ratio"] = 1.0
        else:
            lr_params["penalty"] = "l1"
        clf = LogisticRegression(**lr_params)
        clf.fit(Ztr, ytr)
        loss = log_loss(yv, clf.predict_proba(Zv), labels=labels)
        if best is None or loss < best[0]:
            best = (loss, c_value, clf)
    assert best is not None
    _, c_value, clf = best
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob = clf.predict_proba(Zte)
    pred_s = time.perf_counter() - t0
    coef = np.asarray(clf.coef_)
    edge_coef = coef[:, Xtr.shape[1] :]
    active = np.flatnonzero(np.linalg.norm(edge_coef, axis=0) > 1e-8)
    selected_edges = [edges[i] for i in active]
    precision = recall = np.nan
    if ds.true_edges is not None and selected_edges:
        selected = set(selected_edges)
        precision = len(selected & ds.true_edges) / len(selected)
        recall = len(selected & ds.true_edges) / len(ds.true_edges)
    row, pred = base.metric_row(
        ds.name,
        f"l1_interactions_{'+'.join(map(str, orders))}",
        seed,
        yte,
        prob,
        train_s,
        pred_s,
        coef.size + clf.intercept_.size,
        {
            "orders": "+".join(map(str, orders)),
            "selected_edges": len(selected_edges),
            "selected_c": c_value,
            "support_precision": precision,
            "support_recall": recall,
        },
    )
    edge_rows = []
    for rank, edge in enumerate(selected_edges[:200], start=1):
        edge_rows.append(
            {
                "dataset": ds.name,
                "model": f"l1_interactions_{'+'.join(map(str, orders))}",
                "seed": seed,
                "rank": rank,
                "edge_indices": "-".join(map(str, edge)),
                "edge_names": " x ".join(ds.feature_names[j] for j in edge),
                "order": len(edge),
                "is_true_edge": bool(ds.true_edges and edge in ds.true_edges),
            }
        )
    return row, edge_rows


def run(args: argparse.Namespace) -> int:
    datasets = base.load_datasets()
    if args.include_digits:
        datasets = add_digits_dataset(datasets)
    if args.datasets:
        wanted = set(args.datasets)
        datasets = [ds for ds in datasets if ds.name in wanted]
    seeds = args.seeds or SEEDS_10
    c_grid = [float(x) for x in args.l1_c_grid.split(",") if x.strip()]
    rows = []
    edge_rows = []
    metadata = []
    for ds in datasets:
        metadata.append(
            {
                "dataset": ds.name,
                "n_samples": len(ds.y),
                "n_features": int(ds.X.shape[1]),
                "n_classes": int(len(np.unique(ds.y))),
                "true_edges": sorted([list(e) for e in ds.true_edges]) if ds.true_edges else None,
            }
        )
        for seed in seeds:
            Xtr, Xv, Xte, ytr, yv, yte = base.split_scale(ds, seed)
            labels = np.arange(len(np.unique(ds.y)))
            if not args.skip_boosted:
                for model_name, model in optional_model_builders(seed):
                    try:
                        prob, train_s, pred_s = fit_predict_model(model, Xtr, ytr, Xv, yv, Xte, seed)
                        row, _ = base.metric_row(ds.name, model_name, seed, yte, prob, train_s, pred_s, param_count(model))
                        rows.append(row)
                        print(f"completed extension {ds.name} seed={seed} model={model_name}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        rows.append(
                            {
                                "dataset": ds.name,
                                "model": model_name,
                                "seed": seed,
                                "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"failed extension {ds.name} seed={seed} model={model_name}: {type(exc).__name__}", flush=True)
            for orders in [(2,), (2, 3)]:
                row, edges = run_l1_interaction(
                    ds,
                    Xtr,
                    Xv,
                    Xte,
                    ytr,
                    yv,
                    yte,
                    seed,
                    orders=orders,
                    max_edges=args.max_edges,
                    c_grid=c_grid,
                )
                if row is not None:
                    rows.append(row)
                    edge_rows.extend(edges)
                    print(f"completed extension {ds.name} seed={seed} model={row['model']}", flush=True)
            if ds.X.shape[1] <= args.shil_max_features:
                for orders, k, name in [((2,), 12, "shil_pair_k12_ext"), ((2, 3), 8, "shil_pair_triple_k8_ext")]:
                    sh = base.SparseHypergraphSelector(orders=orders, k=k)
                    t0 = time.perf_counter()
                    sh.fit(Xtr, ytr, Xv, yv, seed)
                    train_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    prob = sh.predict_proba(Xte)
                    pred_s = time.perf_counter() - t0
                    precision = recall = np.nan
                    if ds.true_edges is not None:
                        selected = set(sh.selected_edges)
                        precision = len(selected & ds.true_edges) / len(selected)
                        recall = len(selected & ds.true_edges) / len(ds.true_edges)
                    row, _ = base.metric_row(
                        ds.name,
                        name,
                        seed,
                        yte,
                        prob,
                        train_s,
                        pred_s,
                        sh.n_params,
                        {
                            "orders": "+".join(map(str, orders)),
                            "selected_edges": len(sh.selected_edges),
                            "support_precision": precision,
                            "support_recall": recall,
                        },
                    )
                    rows.append(row)
                    for rank, idx in enumerate(sh.selected_idx, start=1):
                        edge = sh.edges[idx]
                        edge_rows.append(
                            {
                                "dataset": ds.name,
                                "model": name,
                                "seed": seed,
                                "rank": rank,
                                "edge_indices": "-".join(map(str, edge)),
                                "edge_names": " x ".join(ds.feature_names[j] for j in edge),
                                "order": len(edge),
                                "is_true_edge": bool(ds.true_edges and edge in ds.true_edges),
                            }
                        )
                    print(f"completed extension {ds.name} seed={seed} model={name}", flush=True)
            print(f"completed extension {ds.name} seed={seed}", flush=True)
    raw = pd.DataFrame(rows)
    edges = pd.DataFrame(edge_rows)
    run_id = time.strftime("EXP-SHIL-Q1-001_%Y%m%d_%H%M%S")
    raw_path = OUT / f"{run_id}_raw.csv"
    edge_path = OUT / f"{run_id}_edges.csv"
    summary_path = OUT / f"{run_id}_summary.csv"
    raw.to_csv(raw_path, index=False)
    edges.to_csv(edge_path, index=False)
    status = raw["status"].fillna("ok") if "status" in raw.columns else pd.Series("ok", index=raw.index)
    ok = raw[status != "failed"].copy()
    summary = (
        ok.groupby(["dataset", "model"], as_index=False)
        .agg(
            n=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            auc_mean=("roc_auc_ovr", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            parameter_count_mean=("parameter_count", "mean"),
            selected_edges_mean=("selected_edges", "mean"),
            support_precision_mean=("support_precision", "mean"),
            support_recall_mean=("support_recall", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    metadata_path = OUT / f"{run_id}_manifest.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seeds": seeds,
                "datasets": metadata,
                "raw": str(raw_path),
                "edges": str(edge_path),
                "summary": str(summary_path),
                "max_edges": args.max_edges,
                "shil_max_features": args.shil_max_features,
                "l1_c_grid": c_grid,
                "skip_boosted": args.skip_boosted,
                "python": sys.version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run_id": run_id, "raw": str(raw_path), "summary": str(summary_path)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--include-digits", action="store_true")
    parser.add_argument("--max-edges", type=int, default=6000)
    parser.add_argument("--shil-max-features", type=int, default=35)
    parser.add_argument("--l1-c-grid", default="0.05,0.25,1.0")
    parser.add_argument("--skip-boosted", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
