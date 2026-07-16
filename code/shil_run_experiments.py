from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 23, 37, 53, 71]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    true_edges: set[tuple[int, ...]] | None = None


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50, 50))
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def make_synthetic(kind: str, seed: int = 2026, n: int = 2800, d: int = 16) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    linear = 0.15 * X[:, 12] - 0.12 * X[:, 13]
    if kind == "pair":
        edges = {(0, 1), (2, 3), (4, 5)}
        score = 1.8 * X[:, 0] * X[:, 1] - 1.6 * X[:, 2] * X[:, 3] + 1.4 * X[:, 4] * X[:, 5]
    elif kind == "triple":
        edges = {(0, 1, 2), (3, 4, 5), (6, 7, 8)}
        score = (
            1.7 * X[:, 0] * X[:, 1] * X[:, 2]
            - 1.5 * X[:, 3] * X[:, 4] * X[:, 5]
            + 1.3 * X[:, 6] * X[:, 7] * X[:, 8]
        )
    elif kind == "mixed":
        edges = {(0, 1), (2, 3), (4, 5, 6), (7, 8, 9)}
        score = (
            1.5 * X[:, 0] * X[:, 1]
            - 1.3 * X[:, 2] * X[:, 3]
            + 1.4 * X[:, 4] * X[:, 5] * X[:, 6]
            - 1.2 * X[:, 7] * X[:, 8] * X[:, 9]
        )
    else:
        raise ValueError(kind)
    score = score + linear + rng.normal(scale=0.65, size=n)
    threshold = np.median(score)
    y = (score > threshold).astype(int)
    return Dataset(f"synthetic_{kind}", X, y, [f"x{i}" for i in range(d)], edges)


def load_datasets() -> list[Dataset]:
    bc = load_breast_cancer()
    wine = load_wine()
    datasets = [
        make_synthetic("pair"),
        make_synthetic("triple"),
        make_synthetic("mixed"),
        Dataset("breast_cancer", bc.data.astype(float), bc.target.astype(int), list(bc.feature_names)),
        Dataset("wine", wine.data.astype(float), wine.target.astype(int), list(wine.feature_names)),
    ]
    try:
        diabetes = fetch_openml(data_id=37, as_frame=True, data_home=RESULTS / "data", parser="auto")
        frame = diabetes.data.apply(pd.to_numeric, errors="coerce")
        frame = frame.fillna(frame.median(numeric_only=True))
        y = LabelEncoder().fit_transform(diabetes.target.astype(str))
        datasets.append(Dataset("diabetes_openml37", frame.to_numpy(float), y, list(frame.columns)))
    except Exception as exc:
        (RESULTS / "openml_error.txt").write_text(str(exc), encoding="utf-8")
    return datasets


def split_scale(ds: Dataset, seed: int):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        ds.X, ds.y, test_size=0.4, random_state=seed, stratify=ds.y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed + 1, stratify=y_tmp
    )
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        y_train,
        y_val,
        y_test,
    )


def candidate_edges(d: int, orders: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [edge for order in orders for edge in itertools.combinations(range(d), order)]


def interaction_matrix(X: np.ndarray, edges: list[tuple[int, ...]]) -> np.ndarray:
    Z = np.empty((len(X), len(edges)), dtype=np.float64)
    for j, edge in enumerate(edges):
        Z[:, j] = np.prod(X[:, edge], axis=1)
    return np.clip(Z, -12.0, 12.0)


class Adam:
    def __init__(self, params, lr=0.02):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g * g)
            mh = self.m[i] / (1 - 0.9**self.t)
            vh = self.v[i] / (1 - 0.999**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class FactorizationMachine:
    def __init__(self, rank=8, epochs=350, lr=0.025, l2=2e-4, patience=45):
        self.rank = rank
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.patience = patience

    def fit(self, X, y, X_val, y_val, seed):
        rng = np.random.default_rng(seed)
        n, d = X.shape
        c = int(max(y.max(), y_val.max()) + 1)
        self.b = np.zeros(c)
        self.W = rng.normal(scale=0.02, size=(d, c))
        self.V = rng.normal(scale=0.05, size=(c, d, self.rank))
        opt = Adam([self.b, self.W, self.V], self.lr)
        Y = one_hot(y, c)
        best, best_loss, wait = None, float("inf"), 0
        X2 = X * X
        for _ in range(self.epochs):
            p = self.predict_proba(X)
            r = (p - Y) / n
            gb = r.sum(axis=0)
            gW = X.T @ r + self.l2 * self.W
            gV = np.zeros_like(self.V)
            for cls in range(c):
                xv = X @ self.V[cls]
                gV[cls] = X.T @ (r[:, cls, None] * xv) - (
                    X2.T @ r[:, cls]
                )[:, None] * self.V[cls]
                gV[cls] += self.l2 * self.V[cls]
            opt.step([gb, gW, gV])
            val_loss = log_loss(y_val, self.predict_proba(X_val), labels=np.arange(c))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best = (self.b.copy(), self.W.copy(), self.V.copy())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.b, self.W, self.V = best
        return self

    def decision_function(self, X):
        out = self.b + X @ self.W
        X2 = X * X
        for cls in range(len(self.b)):
            xv = X @ self.V[cls]
            out[:, cls] += 0.5 * ((xv * xv).sum(axis=1) - X2 @ (self.V[cls] ** 2).sum(axis=1))
        return out

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    @property
    def n_params(self):
        return self.b.size + self.W.size + self.V.size


class SparseHypergraphSelector:
    def __init__(self, orders=(2, 3), k=8, epochs=180, lr=0.025, gate_l1=8e-4, l2=2e-4, patience=28, init_mode="moment"):
        self.orders = tuple(orders)
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.gate_l1 = gate_l1
        self.l2 = l2
        self.patience = patience
        self.init_mode = init_mode

    def fit(self, X, y, X_val, y_val, seed):
        rng = np.random.default_rng(seed)
        n, d = X.shape
        c = int(max(y.max(), y_val.max()) + 1)
        self.edges = candidate_edges(d, self.orders)
        Z = interaction_matrix(X, self.edges)
        Zv = interaction_matrix(X_val, self.edges)
        m = Z.shape[1]
        self.b = np.zeros(c)
        self.Wx = rng.normal(scale=0.01, size=(d, c))
        Y = one_hot(y, c)
        if self.init_mode == "moment":
            residual0 = Y - Y.mean(axis=0, keepdims=True)
            moment = Z.T @ residual0 / n
            strength = np.linalg.norm(moment, axis=1)
            scale = max(np.quantile(strength, 0.95), 1e-8)
            self.We = 0.15 * moment / (np.std(Z, axis=0, ddof=1)[:, None] + 1e-6)
            self.alpha = -2.5 + 5.0 * np.clip(strength / scale, 0.0, 1.0)
        else:
            self.We = rng.normal(scale=0.025, size=(m, c))
            self.alpha = np.full(m, -1.2) + rng.normal(scale=0.03, size=m)
        opt = Adam([self.b, self.Wx, self.We, self.alpha], self.lr)
        best, best_loss, wait = None, float("inf"), 0
        for _ in range(self.epochs):
            gates = sigmoid(self.alpha)
            logits = self.b + X @ self.Wx + (Z * gates) @ self.We
            p = softmax(logits)
            r = (p - Y) / n
            gb = r.sum(axis=0)
            gWx = X.T @ r + self.l2 * self.Wx
            gWe = (Z * gates).T @ r + self.l2 * self.We
            gate_signal = np.sum((Z.T @ r) * self.We, axis=1)
            gAlpha = (gate_signal + self.gate_l1) * gates * (1 - gates)
            opt.step([gb, gWx, gWe, gAlpha])
            val_logits = self.b + X_val @ self.Wx + (Zv * sigmoid(self.alpha)) @ self.We
            val_loss = log_loss(y_val, softmax(val_logits), labels=np.arange(c))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best = (self.b.copy(), self.Wx.copy(), self.We.copy(), self.alpha.copy())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.b, self.Wx, self.We, self.alpha = best
        gates = sigmoid(self.alpha)
        self.scores = gates * np.linalg.norm(self.We, axis=1)
        self.ranking = np.argsort(self.scores)[::-1]
        self.select_and_refit(X, y, X_val, y_val, self.k, seed)
        return self

    def select_and_refit(self, X, y, X_val, y_val, k, seed):
        self.k = k
        self.selected_idx = self.ranking[: min(k, len(self.edges))]
        self.selected_edges = [self.edges[i] for i in self.selected_idx]
        Zs = interaction_matrix(X, self.selected_edges)
        Zvs = interaction_matrix(X_val, self.selected_edges)
        self.refit = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        self.refit.fit(np.c_[X, Zs], y)
        self.val_loss = log_loss(y_val, self.refit.predict_proba(np.c_[X_val, Zvs]), labels=self.refit.classes_)
        return self

    def transform(self, X):
        return np.c_[X, interaction_matrix(X, self.selected_edges)]

    def predict_proba(self, X):
        return self.refit.predict_proba(self.transform(X))

    @property
    def n_params(self):
        return self.b.size + self.Wx.size + self.We.size + self.alpha.size


def metric_row(ds_name, model_name, seed, y_true, prob, train_s, pred_s, n_params, extra=None):
    pred = prob.argmax(axis=1)
    row = {
        "dataset": ds_name,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "log_loss": log_loss(y_true, prob, labels=np.arange(prob.shape[1])),
        "train_seconds": train_s,
        "predict_seconds": pred_s,
        "parameter_count": int(n_params),
    }
    try:
        row["roc_auc_ovr"] = roc_auc_score(
            y_true, prob[:, 1] if prob.shape[1] == 2 else prob, multi_class="ovr"
        )
    except ValueError:
        row["roc_auc_ovr"] = np.nan
    if extra:
        row.update(extra)
    return row, pred


def count_hist_nodes(model):
    total = 0
    for stage in getattr(model, "_predictors", []):
        for predictor in stage:
            total += len(predictor.nodes)
    return total


def run():
    datasets = load_datasets()
    rows, edge_rows, pred_rows = [], [], []
    metadata = []
    for ds in datasets:
        metadata.append(
            {
                "dataset": ds.name,
                "n_samples": len(ds.y),
                "n_features": ds.X.shape[1],
                "n_classes": int(len(np.unique(ds.y))),
                "true_edges": sorted([list(e) for e in ds.true_edges]) if ds.true_edges else None,
            }
        )
        for seed in SEEDS:
            Xtr, Xv, Xte, ytr, yv, yte = split_scale(ds, seed)
            n_classes = len(np.unique(ds.y))

            models = []
            models.append(("linear", LogisticRegression(C=1.0, max_iter=2000, random_state=seed)))
            models.append(
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=0.003,
                        max_iter=700,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=40,
                        random_state=seed,
                    ),
                )
            )
            models.append(
                (
                    "hist_gradient_boosting",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        early_stopping=True,
                        random_state=seed,
                    ),
                )
            )

            pair_edges = candidate_edges(Xtr.shape[1], (2,))
            Ztr_pair = interaction_matrix(Xtr, pair_edges)
            Zv_pair = interaction_matrix(Xv, pair_edges)
            Zte_pair = interaction_matrix(Xte, pair_edges)
            pair_model = LogisticRegression(C=0.5, max_iter=2500, random_state=seed)
            t0 = time.perf_counter()
            pair_model.fit(np.c_[Xtr, Ztr_pair], ytr)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            prob = pair_model.predict_proba(np.c_[Xte, Zte_pair])
            pred_s = time.perf_counter() - t0
            row, pred = metric_row(
                ds.name,
                "explicit_all_pairs",
                seed,
                yte,
                prob,
                train_s,
                pred_s,
                pair_model.coef_.size + pair_model.intercept_.size,
                {"selected_edges": len(pair_edges), "orders": "2"},
            )
            rows.append(row)
            for i, (yt, yp) in enumerate(zip(yte, pred)):
                pred_rows.append({"dataset": ds.name, "model": "explicit_all_pairs", "seed": seed, "row": i, "y_true": int(yt), "y_pred": int(yp)})

            for model_name, model in models:
                t0 = time.perf_counter()
                model.fit(Xtr, ytr)
                train_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                prob = model.predict_proba(Xte)
                pred_s = time.perf_counter() - t0
                if model_name == "linear":
                    params = model.coef_.size + model.intercept_.size
                elif model_name == "mlp":
                    params = sum(x.size for x in model.coefs_) + sum(x.size for x in model.intercepts_)
                else:
                    params = count_hist_nodes(model)
                row, pred = metric_row(ds.name, model_name, seed, yte, prob, train_s, pred_s, params)
                rows.append(row)
                for i, (yt, yp) in enumerate(zip(yte, pred)):
                    pred_rows.append({"dataset": ds.name, "model": model_name, "seed": seed, "row": i, "y_true": int(yt), "y_pred": int(yp)})

            fm = FactorizationMachine(rank=8)
            t0 = time.perf_counter()
            fm.fit(Xtr, ytr, Xv, yv, seed)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            prob = fm.predict_proba(Xte)
            pred_s = time.perf_counter() - t0
            row, pred = metric_row(ds.name, "factorization_machine", seed, yte, prob, train_s, pred_s, fm.n_params)
            rows.append(row)
            for i, (yt, yp) in enumerate(zip(yte, pred)):
                pred_rows.append({"dataset": ds.name, "model": "factorization_machine", "seed": seed, "row": i, "y_true": int(yt), "y_pred": int(yp)})

            spec_groups = [
                ((2,), [(3, "shil_pair_k3"), (6, "shil_pair_k6"), (12, "shil_pair_k12")]),
                ((2, 3), [(4, "shil_pair_triple_k4"), (8, "shil_pair_triple_k8"), (16, "shil_pair_triple_k16")]),
            ]
            for orders, budgets in spec_groups:
                sh = SparseHypergraphSelector(orders=orders, k=max(k for k, _ in budgets))
                t0 = time.perf_counter()
                sh.fit(Xtr, ytr, Xv, yv, seed)
                selection_train_s = time.perf_counter() - t0
                for k, model_name in budgets:
                    t_refit = time.perf_counter()
                    sh.select_and_refit(Xtr, ytr, Xv, yv, k, seed)
                    train_s = selection_train_s + (time.perf_counter() - t_refit)
                    t0 = time.perf_counter()
                    prob = sh.predict_proba(Xte)
                    pred_s = time.perf_counter() - t0
                    precision = recall = np.nan
                    if ds.true_edges is not None:
                        selected = set(sh.selected_edges)
                        true_hits = len(selected & ds.true_edges)
                        precision = true_hits / len(selected)
                        recall = true_hits / len(ds.true_edges)
                    row, pred = metric_row(
                        ds.name,
                        model_name,
                        seed,
                        yte,
                        prob,
                        train_s,
                        pred_s,
                        sh.n_params,
                        {
                            "selected_edges": len(sh.selected_edges),
                            "orders": "+".join(map(str, orders)),
                            "initialization": sh.init_mode,
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
                                "model": model_name,
                                "seed": seed,
                                "rank": rank,
                                "edge_indices": "-".join(map(str, edge)),
                                "edge_names": " × ".join(ds.feature_names[j] for j in edge),
                                "order": len(edge),
                                "score": float(sh.scores[idx]),
                                "gate": float(sigmoid(sh.alpha[idx])),
                                "coefficient_norm": float(np.linalg.norm(sh.We[idx])),
                                "is_true_edge": bool(ds.true_edges and edge in ds.true_edges),
                            }
                        )
                    for i, (yt, yp) in enumerate(zip(yte, pred)):
                        pred_rows.append({"dataset": ds.name, "model": model_name, "seed": seed, "row": i, "y_true": int(yt), "y_pred": int(yp)})

            print(f"completed {ds.name} seed={seed}", flush=True)

    metrics = pd.DataFrame(rows)
    edges = pd.DataFrame(edge_rows)
    predictions = pd.DataFrame(pred_rows)
    metrics.to_csv(RESULTS / "metrics_raw.csv", index=False)
    edges.to_csv(RESULTS / "selected_edges_raw.csv", index=False)
    predictions.to_csv(RESULTS / "predictions_raw.csv.gz", index=False, compression="gzip")
    (RESULTS / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config = {
        "seeds": SEEDS,
        "split": "60% train / 20% validation / 20% test, stratified independently per seed",
        "scaling": "StandardScaler fitted on training fold only",
        "shil": "global task-supervised pair/triple gates; top-k selection; logistic refit",
        "python": sys.version,
    }
    (RESULTS / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = (
        metrics.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            auc_mean=("roc_auc_ovr", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            parameter_count_mean=("parameter_count", "mean"),
            support_precision_mean=("support_precision", "mean"),
            support_recall_mean=("support_recall", "mean"),
        )
    )
    summary.to_csv(RESULTS / "metrics_summary.csv", index=False)
    compute_stability(edges)


def compute_stability(edges: pd.DataFrame):
    records = []
    for (dataset, model), group in edges.groupby(["dataset", "model"]):
        sets = {int(seed): set(g.edge_indices) for seed, g in group.groupby("seed")}
        pairwise = []
        for a, b in itertools.combinations(sorted(sets), 2):
            union = sets[a] | sets[b]
            jac = len(sets[a] & sets[b]) / len(union) if union else 1.0
            pairwise.append(jac)
            records.append({"dataset": dataset, "model": model, "seed_a": a, "seed_b": b, "jaccard": jac})
    pd.DataFrame(records).to_csv(RESULTS / "selection_stability_pairwise.csv", index=False)


if __name__ == "__main__":
    run()
