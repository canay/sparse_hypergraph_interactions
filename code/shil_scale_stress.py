from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import shil_run_experiments as base  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")

OUT = ROOT / "experiments" / "q1_extension"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [11, 23, 37, 53, 71]


def make_wide_sparse(seed: int, n: int, d: int, kind: str) -> base.Dataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    linear = 0.10 * X[:, min(d - 1, 12)] - 0.08 * X[:, min(d - 1, 13)]
    if kind == "pair":
        true_edges = {(0, 1), (2, 3), (4, 5)}
        score = 1.8 * X[:, 0] * X[:, 1] - 1.6 * X[:, 2] * X[:, 3] + 1.4 * X[:, 4] * X[:, 5]
    elif kind == "mixed":
        true_edges = {(0, 1), (2, 3), (4, 5, 6), (7, 8, 9)}
        score = (
            1.5 * X[:, 0] * X[:, 1]
            - 1.3 * X[:, 2] * X[:, 3]
            + 1.4 * X[:, 4] * X[:, 5] * X[:, 6]
            - 1.2 * X[:, 7] * X[:, 8] * X[:, 9]
        )
    else:
        raise ValueError(kind)
    score = score + linear + rng.normal(scale=0.70, size=n)
    y = (score > np.median(score)).astype(int)
    return base.Dataset(f"wide_{kind}_d{d}", X, y, [f"x{i}" for i in range(d)], true_edges)


def split_scale(ds: base.Dataset, seed: int):
    X_train, X_tmp, y_train, y_tmp = train_test_split(ds.X, ds.y, test_size=0.4, random_state=seed, stratify=ds.y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed + 1, stratify=y_tmp
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), y_train, y_val, y_test


def interaction_design(X: np.ndarray, edges: list[tuple[int, ...]]) -> np.ndarray:
    return np.c_[X, base.interaction_matrix(X, edges)]


def lr_l1_params(c_value: float, seed: int) -> dict:
    return {
        "solver": "liblinear",
        "penalty": "l1",
        "C": c_value,
        "max_iter": 600,
        "tol": 1e-3,
        "random_state": seed,
    }


def evaluate(ds_name: str, model_name: str, seed: int, y_true: np.ndarray, prob: np.ndarray, train_s: float, n_params: int, extra: dict):
    pred = prob.argmax(axis=1)
    labels = np.arange(prob.shape[1])
    try:
        auc = roc_auc_score(y_true, prob[:, 1]) if prob.shape[1] == 2 else roc_auc_score(y_true, prob, multi_class="ovr")
    except Exception:
        auc = np.nan
    row = {
        "dataset": ds_name,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "log_loss": log_loss(y_true, prob, labels=labels),
        "roc_auc_ovr": auc,
        "train_seconds": train_s,
        "parameter_count": n_params,
    }
    row.update(extra)
    return row


def support_metrics(selected_edges: list[tuple[int, ...]], true_edges: set[tuple[int, ...]]):
    selected = set(selected_edges)
    if not selected:
        return 0.0, 0.0, 0
    hits = len(selected & true_edges)
    return hits / len(selected), hits / len(true_edges), hits


def run_l1(ds: base.Dataset, Xtr, Xv, Xte, ytr, yv, yte, seed: int, c_grid: list[float], max_edges: int):
    edges = base.candidate_edges(Xtr.shape[1], (2, 3))
    candidate_count = len(edges)
    if candidate_count > max_edges:
        return {
            "dataset": ds.name,
            "model": "l1_pair_triple_full",
            "seed": seed,
            "status": "skipped_candidate_limit",
            "candidate_edges": candidate_count,
        }, []
    Ztr = interaction_design(Xtr, edges)
    Zv = interaction_design(Xv, edges)
    Zte = interaction_design(Xte, edges)
    labels = np.arange(len(np.unique(np.r_[ytr, yv, yte])))
    best = None
    t0 = time.perf_counter()
    for c_value in c_grid:
        clf = LogisticRegression(**lr_l1_params(c_value, seed))
        clf.fit(Ztr, ytr)
        loss = log_loss(yv, clf.predict_proba(Zv), labels=labels)
        if best is None or loss < best[0]:
            best = (loss, c_value, clf)
    assert best is not None
    _, c_value, clf = best
    train_s = time.perf_counter() - t0
    prob = clf.predict_proba(Zte)
    edge_coef = np.asarray(clf.coef_)[:, Xtr.shape[1] :]
    active = np.flatnonzero(np.linalg.norm(edge_coef, axis=0) > 1e-6)
    selected_edges = [edges[i] for i in active]
    precision, recall, hits = support_metrics(selected_edges, ds.true_edges or set())
    row = evaluate(
        ds.name,
        "l1_pair_triple_full",
        seed,
        yte,
        prob,
        train_s,
        clf.coef_.size + clf.intercept_.size,
        {
            "status": "completed",
            "candidate_edges": candidate_count,
            "selected_edges": len(selected_edges),
            "selected_c": c_value,
            "support_precision": precision,
            "support_recall": recall,
            "true_edge_hits": hits,
        },
    )
    edge_rows = [
        {
            "dataset": ds.name,
            "model": "l1_pair_triple_full",
            "seed": seed,
            "rank": rank,
            "edge_indices": "-".join(map(str, edge)),
            "order": len(edge),
            "is_true_edge": edge in (ds.true_edges or set()),
        }
        for rank, edge in enumerate(selected_edges[:500], start=1)
    ]
    return row, edge_rows


def run_shil(ds: base.Dataset, Xtr, Xv, Xte, ytr, yv, yte, seed: int, k: int, epochs: int):
    sh = base.SparseHypergraphSelector(orders=(2, 3), k=k, epochs=epochs, patience=max(12, epochs // 5))
    candidate_count = len(base.candidate_edges(Xtr.shape[1], (2, 3)))
    t0 = time.perf_counter()
    sh.fit(Xtr, ytr, Xv, yv, seed)
    train_s = time.perf_counter() - t0
    prob = sh.predict_proba(Xte)
    precision, recall, hits = support_metrics(sh.selected_edges, ds.true_edges or set())
    row = evaluate(
        ds.name,
        f"shil_pair_triple_k{k}",
        seed,
        yte,
        prob,
        train_s,
        sh.n_params,
        {
            "status": "completed",
            "candidate_edges": candidate_count,
            "selected_edges": len(sh.selected_edges),
            "support_precision": precision,
            "support_recall": recall,
            "true_edge_hits": hits,
        },
    )
    edge_rows = [
        {
            "dataset": ds.name,
            "model": f"shil_pair_triple_k{k}",
            "seed": seed,
            "rank": rank,
            "edge_indices": "-".join(map(str, edge)),
            "order": len(edge),
            "is_true_edge": edge in (ds.true_edges or set()),
        }
        for rank, edge in enumerate(sh.selected_edges, start=1)
    ]
    return row, edge_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=[24, 32, 40])
    parser.add_argument("--kinds", nargs="+", default=["pair", "mixed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--max-edges", type=int, default=15000)
    parser.add_argument("--shil-k", type=int, default=8)
    parser.add_argument("--shil-epochs", type=int, default=120)
    parser.add_argument("--l1-c-grid", default="0.01,0.02,0.05,0.1,0.25")
    args = parser.parse_args()

    c_grid = [float(x) for x in args.l1_c_grid.split(",") if x.strip()]
    rows = []
    edge_rows = []
    metadata = []
    for d in args.dims:
        for kind in args.kinds:
            for seed in args.seeds:
                ds = make_wide_sparse(seed=2026 + seed + d, n=args.n, d=d, kind=kind)
                Xtr, Xv, Xte, ytr, yv, yte = split_scale(ds, seed)
                metadata.append(
                    {
                        "dataset": ds.name,
                        "seed": seed,
                        "n_samples": args.n,
                        "n_features": d,
                        "candidate_edges": len(base.candidate_edges(d, (2, 3))),
                        "true_edges": sorted([list(e) for e in ds.true_edges or set()]),
                    }
                )
                for runner in [
                    lambda: run_l1(ds, Xtr, Xv, Xte, ytr, yv, yte, seed, c_grid, args.max_edges),
                    lambda: run_shil(ds, Xtr, Xv, Xte, ytr, yv, yte, seed, args.shil_k, args.shil_epochs),
                ]:
                    row, edges = runner()
                    rows.append(row)
                    edge_rows.extend(edges)
                    print(f"completed scale {row['dataset']} seed={seed} model={row['model']} status={row.get('status')}", flush=True)

    run_id = time.strftime("EXP-SHIL-Q1-002_scale_%Y%m%d_%H%M%S")
    raw = pd.DataFrame(rows)
    edges = pd.DataFrame(edge_rows)
    raw_path = OUT / f"{run_id}_raw.csv"
    edge_path = OUT / f"{run_id}_edges.csv"
    summary_path = OUT / f"{run_id}_summary.csv"
    raw.to_csv(raw_path, index=False)
    edges.to_csv(edge_path, index=False)
    ok = raw[raw.get("status", "completed") == "completed"].copy()
    summary = (
        ok.groupby(["dataset", "model"], as_index=False)
        .agg(
            n=("accuracy", "count"),
            candidate_edges_mean=("candidate_edges", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            auc_mean=("roc_auc_ovr", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            selected_edges_mean=("selected_edges", "mean"),
            support_precision_mean=("support_precision", "mean"),
            support_recall_mean=("support_recall", "mean"),
            true_edge_hits_mean=("true_edge_hits", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)
    manifest_path = OUT / f"{run_id}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "raw": str(raw_path),
                "edges": str(edge_path),
                "summary": str(summary_path),
                "args": vars(args),
                "c_grid": c_grid,
                "metadata": metadata,
                "python": sys.version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run_id": run_id, "raw": str(raw_path), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
