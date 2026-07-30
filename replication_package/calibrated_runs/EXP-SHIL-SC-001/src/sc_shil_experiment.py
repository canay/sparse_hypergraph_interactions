from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import socket
import sys
import threading
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
import sklearn
from scipy.io import arff
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


EDGE = tuple[int, ...]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    true_edges: set[EDGE] | None = None
    equivalence_groups: list[set[EDGE]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingResult:
    ranked_indices: np.ndarray
    active_indices: np.ndarray
    scores: np.ndarray
    selected_c: float | None
    warnings: list[str]
    elapsed_seconds: float


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self._event = threading.Event()
        self.peak_rss_bytes = psutil.Process().memory_info().rss
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSSMonitor":
        def poll() -> None:
            process = psutil.Process()
            while not self._event.is_set():
                try:
                    self.peak_rss_bytes = max(
                        self.peak_rss_bytes, process.memory_info().rss
                    )
                except psutil.Error:
                    pass
                self._event.wait(self.interval_seconds)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, psutil.Process().memory_info().rss
            )
        except psutil.Error:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / np.sum(ex, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def candidate_edges(d: int, orders: Sequence[int]) -> list[EDGE]:
    edges: list[EDGE] = []
    for order in orders:
        edges.extend(tuple(edge) for edge in combinations(range(d), int(order)))
    return edges


def interaction_matrix(
    X: np.ndarray, edges: Sequence[EDGE], clip_value: float = 12.0
) -> np.ndarray:
    if not edges:
        return np.empty((X.shape[0], 0), dtype=float)
    Z = np.empty((X.shape[0], len(edges)), dtype=float)
    for j, edge in enumerate(edges):
        Z[:, j] = np.prod(X[:, edge], axis=1)
    return np.clip(Z, -float(clip_value), float(clip_value))


def _base_edges(kind: str) -> tuple[list[EDGE], list[float]]:
    if kind == "pair":
        return [(0, 1), (2, 3), (4, 5)], [1.8, -1.6, 1.4]
    if kind == "triple":
        return [(0, 1, 2), (3, 4, 5), (6, 7, 8)], [1.7, -1.5, 1.3]
    if kind in {"mixed", "redundant_mixed"}:
        return [(0, 1), (2, 3), (4, 5, 6), (7, 8, 9)], [1.5, -1.3, 1.4, -1.2]
    if kind == "dense_mixed":
        return [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9, 10),
            (11, 12, 13),
            (14, 15, 16),
            (17, 18, 19),
        ], [1.8, -1.6, 1.4, -1.2, 1.5, -1.3, 1.1, -0.9]
    raise ValueError(f"Unknown scenario kind: {kind}")


def make_synthetic(spec: dict[str, Any], seed: int) -> Dataset:
    rng = np.random.default_rng(int(seed))
    n, d = int(spec["n"]), int(spec["d"])
    rho = float(spec.get("rho", 0.0))
    if rho:
        idx = np.arange(d)
        covariance = rho ** np.abs(np.subtract.outer(idx, idx))
        X = rng.multivariate_normal(np.zeros(d), covariance, size=n)
    else:
        X = rng.normal(size=(n, d))

    kind = str(spec["kind"])
    edges, coefficients = _base_edges(kind)
    if max(max(edge) for edge in edges) >= d:
        raise ValueError(f"Scenario {spec['id']} has too few features for planted edges")

    equivalence_groups: list[set[EDGE]] = []
    if kind == "redundant_mixed":
        proxy_corr = float(spec.get("proxy_correlation", 0.95))
        residual_scale = math.sqrt(max(1.0 - proxy_corr**2, 0.0))
        proxy_pairs = [(16, 0), (17, 1), (18, 4), (19, 5)]
        for proxy, source in proxy_pairs:
            X[:, proxy] = (
                proxy_corr * X[:, source]
                + residual_scale * rng.normal(size=n)
            )
        equivalence_groups = [
            {(0, 1), (16, 1), (0, 17), (16, 17)},
            {(2, 3)},
            {
                (4, 5, 6),
                tuple(sorted((18, 5, 6))),
                tuple(sorted((4, 19, 6))),
                tuple(sorted((18, 19, 6))),
            },
            {(7, 8, 9)},
        ]

    score = np.zeros(n, dtype=float)
    signal_scale = float(spec.get("signal_scale", 1.0))
    for edge, coefficient in zip(edges, coefficients):
        score += signal_scale * coefficient * np.prod(X[:, edge], axis=1)

    participating = sorted({j for edge in edges for j in edge})
    if spec.get("heredity") == "respected":
        signs = np.where(np.arange(len(participating)) % 2 == 0, 1.0, -1.0)
        score += 0.35 * (X[:, participating] @ signs)

    nuisance_candidates = [j for j in range(d - 1, -1, -1) if j not in participating]
    if len(nuisance_candidates) >= 2:
        score += 0.15 * X[:, nuisance_candidates[0]] - 0.12 * X[:, nuisance_candidates[1]]
    score += rng.normal(scale=float(spec["noise_sd"]), size=n)
    threshold = np.quantile(score, float(spec.get("threshold_quantile", 0.5)))
    y = (score > threshold).astype(int)
    true_edges = {tuple(sorted(edge)) for edge in edges}
    return Dataset(
        name=str(spec["id"]),
        X=X,
        y=y,
        feature_names=[f"x{i}" for i in range(d)],
        true_edges=true_edges,
        equivalence_groups=equivalence_groups,
        metadata={**spec, "generator_seed": int(seed)},
    )


def load_dry_bean(config: dict[str, Any]) -> Dataset:
    run_root = Path(__file__).resolve().parents[1]
    cache_dir = run_root / "cache" / f"uci_{int(config['uci_dataset_id'])}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "dry_bean_dataset.zip"
    if archive_path.exists():
        archive_bytes = archive_path.read_bytes()
    else:
        request = urllib.request.Request(
            str(config["source_url"]),
            headers={"User-Agent": "EXP-SHIL-SC-001 reproducibility loader"},
        )
        with urllib.request.urlopen(
            request, timeout=float(config["download_timeout_seconds"])
        ) as response:
            archive_bytes = response.read()
        observed_archive_hash = hashlib.sha256(archive_bytes).hexdigest().upper()
        if observed_archive_hash != str(config["archive_sha256"]).upper():
            raise ValueError(
                "Dry Bean archive hash mismatch before cache write: "
                f"{observed_archive_hash}"
            )
        archive_path.write_bytes(archive_bytes)

    observed_archive_hash = hashlib.sha256(archive_bytes).hexdigest().upper()
    if observed_archive_hash != str(config["archive_sha256"]).upper():
        raise ValueError(f"Dry Bean cached archive hash mismatch: {observed_archive_hash}")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        arff_bytes = archive.read(str(config["archive_member"]))
    observed_arff_hash = hashlib.sha256(arff_bytes).hexdigest().upper()
    if observed_arff_hash != str(config["arff_sha256"]).upper():
        raise ValueError(f"Dry Bean ARFF hash mismatch: {observed_arff_hash}")

    records, _ = arff.loadarff(io.StringIO(arff_bytes.decode("utf-8")))
    frame = pd.DataFrame(records)
    target = frame.pop(str(config["target_column"])).map(
        lambda value: value.decode("utf-8")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
    )
    if frame.columns.tolist() != list(config["expected_feature_names"]):
        raise ValueError(f"Unexpected Dry Bean feature columns: {frame.columns.tolist()}")
    X_frame = frame.apply(pd.to_numeric, errors="raise")
    if X_frame.isna().any().any():
        raise ValueError("Dry Bean contains missing values; locked no-imputation path violated")
    encoder = LabelEncoder()
    y = encoder.fit_transform(target)
    if X_frame.shape != (
        int(config["expected_rows"]),
        int(config["expected_features"]),
    ):
        raise ValueError(f"Unexpected Dry Bean shape: {X_frame.shape}")
    if len(encoder.classes_) != int(config["expected_classes"]):
        raise ValueError(f"Unexpected Dry Bean class count: {len(encoder.classes_)}")
    normalized = X_frame.copy()
    normalized["__target__"] = target.to_numpy()
    content_hash = hashlib.sha256(
        pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    ).hexdigest().upper()
    return Dataset(
        name=str(config["name"]),
        X=X_frame.to_numpy(dtype=float),
        y=y.astype(int),
        feature_names=[str(column) for column in X_frame.columns],
        metadata={
            **config,
            "class_labels": encoder.classes_.tolist(),
            "content_hash": content_hash,
            "observed_archive_sha256": observed_archive_hash,
            "observed_arff_sha256": observed_arff_hash,
            "local_cache": str(archive_path.relative_to(run_root)),
        },
    )


def outer_split_scale(ds: Dataset, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(ds.y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.4,
        stratify=ds.y,
        random_state=int(seed) + 10000,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=ds.y[temp_idx],
        random_state=int(seed) + 10001,
    )
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        raise AssertionError("Outer split overlap detected")
    if set(np.r_[train_idx, val_idx, test_idx]) != set(indices):
        raise AssertionError("Outer split does not cover all rows")
    scaler = StandardScaler().fit(ds.X[train_idx])
    return {
        "X_train": scaler.transform(ds.X[train_idx]),
        "X_val": scaler.transform(ds.X[val_idx]),
        "X_test": scaler.transform(ds.X[test_idx]),
        "y_train": ds.y[train_idx],
        "y_val": ds.y[val_idx],
        "y_test": ds.y[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def stratified_complementary_halves(
    y: np.ndarray, n_pairs: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    all_indices = np.arange(len(y))
    halves: list[np.ndarray] = []
    for _ in range(int(n_pairs)):
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for label in np.unique(y):
            class_indices = np.flatnonzero(y == label)
            permuted = rng.permutation(class_indices)
            cut = len(permuted) // 2
            if cut == 0:
                raise ValueError("A class is too small for complementary halves")
            left_parts.append(permuted[:cut])
            right_parts.append(permuted[cut:])
        left = np.sort(np.concatenate(left_parts))
        right = np.sort(np.concatenate(right_parts))
        if set(left) & set(right):
            raise AssertionError("Complementary halves overlap")
        if set(np.r_[left, right]) != set(all_indices):
            raise AssertionError("Complementary halves do not cover all observations")
        halves.extend([left, right])
    return halves


def inner_split(indices: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fit_local, val_local = train_test_split(
        np.arange(len(indices)),
        test_size=0.2,
        stratify=y[indices],
        random_state=int(seed),
    )
    return indices[fit_local], indices[val_local]


class Adam:
    def __init__(self, params: Sequence[np.ndarray], lr: float):
        self.params = params
        self.lr = float(lr)
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]
        self.t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * grad
            self.v[i] = 0.999 * self.v[i] + 0.001 * (grad * grad)
            mhat = self.m[i] / (1.0 - 0.9**self.t)
            vhat = self.v[i] / (1.0 - 0.999**self.t)
            param -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def fit_shil_ranking(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    config: dict[str, Any],
) -> RankingResult:
    started = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    X_fit, X_val = X[fit_idx], X[val_idx]
    Z_fit, Z_val = Z[fit_idx], Z[val_idx]
    y_fit, y_val = y[fit_idx], y[val_idx]
    n, d = X_fit.shape
    n_classes = int(max(y_fit.max(), y_val.max()) + 1)
    n_edges = Z_fit.shape[1]
    b = np.zeros(n_classes)
    Wx = rng.normal(scale=0.01, size=(d, n_classes))
    Y = one_hot(y_fit, n_classes)
    residual0 = Y - Y.mean(axis=0, keepdims=True)
    moment = Z_fit.T @ residual0 / n
    strength = np.linalg.norm(moment, axis=1)
    scale = max(float(np.quantile(strength, 0.95)), 1e-8)
    edge_sd = np.std(Z_fit, axis=0, ddof=1)
    We = 0.15 * moment / (edge_sd[:, None] + 1e-6)
    alpha = -2.5 + 5.0 * np.clip(strength / scale, 0.0, 1.0)
    optimizer = Adam([b, Wx, We, alpha], float(config["learning_rate"]))
    best: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    best_loss = float("inf")
    wait = 0
    for _ in range(int(config["epochs"])):
        gates = sigmoid(alpha)
        logits = b + X_fit @ Wx + (Z_fit * gates) @ We
        probabilities = softmax(logits)
        residual = (probabilities - Y) / n
        gb = residual.sum(axis=0)
        gWx = X_fit.T @ residual + float(config["l2"]) * Wx
        gWe = (Z_fit * gates).T @ residual + float(config["l2"]) * We
        gate_signal = np.sum((Z_fit.T @ residual) * We, axis=1)
        gAlpha = (gate_signal + float(config["gate_l1"])) * gates * (1.0 - gates)
        optimizer.step([gb, gWx, gWe, gAlpha])
        val_logits = b + X_val @ Wx + (Z_val * sigmoid(alpha)) @ We
        val_loss = log_loss(
            y_val, softmax(val_logits), labels=np.arange(n_classes)
        )
        if val_loss < best_loss - 1e-5:
            best_loss = float(val_loss)
            best = (b.copy(), Wx.copy(), We.copy(), alpha.copy())
            wait = 0
        else:
            wait += 1
            if wait >= int(config["patience"]):
                break
    if best is None:
        raise RuntimeError("SHIL ranking failed to record an optimization state")
    _, _, We, alpha = best
    scores = sigmoid(alpha) * np.linalg.norm(We, axis=1)
    ranking = np.argsort(scores, kind="stable")[::-1]
    return RankingResult(
        ranked_indices=ranking,
        active_indices=np.arange(n_edges, dtype=int),
        scores=scores,
        selected_c=None,
        warnings=[],
        elapsed_seconds=time.perf_counter() - started,
    )


def _l1_parameters(c_value: float, seed: int, n_classes: int) -> dict[str, Any]:
    if n_classes == 2:
        return {
            "solver": "liblinear",
            "l1_ratio": 1.0,
            "C": float(c_value),
            "max_iter": 1200,
            "tol": 1e-3,
            "random_state": int(seed),
        }
    return {
        "solver": "saga",
        "l1_ratio": 1.0,
        "C": float(c_value),
        "max_iter": 1600,
        "tol": 1e-3,
        "random_state": int(seed),
    }


def fit_l1_ranking(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    c_grid: Sequence[float],
) -> RankingResult:
    started = time.perf_counter()
    design = np.c_[X, Z]
    labels = np.arange(int(y.max()) + 1)
    n_classes = len(labels)
    best: tuple[float, float] | None = None
    warning_text: list[str] = []
    for c_value in c_grid:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(**_l1_parameters(c_value, seed, n_classes))
            model.fit(design[fit_idx], y[fit_idx])
        warning_text.extend(str(item.message) for item in caught)
        loss = log_loss(
            y[val_idx], model.predict_proba(design[val_idx]), labels=model.classes_
        )
        if best is None or loss < best[0] - 1e-12 or (
            abs(loss - best[0]) <= 1e-12 and c_value < best[1]
        ):
            best = (float(loss), float(c_value))
    if best is None:
        raise RuntimeError("L1 C-grid selection failed")
    selected_c = best[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model = LogisticRegression(
            **_l1_parameters(selected_c, seed, n_classes)
        )
        combined = np.sort(np.r_[fit_idx, val_idx])
        final_model.fit(design[combined], y[combined])
    warning_text.extend(str(item.message) for item in caught)
    edge_coef = np.asarray(final_model.coef_)[:, X.shape[1] :]
    scores = np.linalg.norm(edge_coef, axis=0)
    active = np.flatnonzero(scores > 1e-8)
    ranked = active[np.argsort(scores[active], kind="stable")[::-1]]
    return RankingResult(
        ranked_indices=ranked,
        active_indices=active,
        scores=scores,
        selected_c=selected_c,
        warnings=warning_text,
        elapsed_seconds=time.perf_counter() - started,
    )


def frequencies_from_rankings(
    rankings: Sequence[np.ndarray], q_grid: Sequence[int], n_edges: int
) -> dict[int, np.ndarray]:
    if not rankings:
        raise ValueError("At least one ranking is required")
    output: dict[int, np.ndarray] = {}
    for q_value in q_grid:
        counts = np.zeros(int(n_edges), dtype=float)
        for ranking in rankings:
            selected = np.asarray(ranking, dtype=int)[: min(int(q_value), len(ranking))]
            counts[selected] += 1.0
        output[int(q_value)] = counts / len(rankings)
    return output


def stability_rankings(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    edges: Sequence[EDGE],
    clip_value: float,
    q_grid: Sequence[int],
    n_pairs: int,
    seed: int,
    shil_config: dict[str, Any],
    l1_c_grid: Sequence[float],
    phase: str,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], float, int]:
    Z = interaction_matrix(X, edges, clip_value)
    halves = stratified_complementary_halves(y, n_pairs, seed)
    rankings: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        for half_index, half in enumerate(halves):
            fit_idx, val_idx = inner_split(
                half, y, seed + 1000 + half_index
            )
            if method == "shil":
                result = fit_shil_ranking(
                    X,
                    Z,
                    y,
                    fit_idx,
                    val_idx,
                    seed + 2000 + half_index,
                    shil_config,
                )
            elif method == "l1":
                result = fit_l1_ranking(
                    X,
                    Z,
                    y,
                    fit_idx,
                    val_idx,
                    seed + 3000 + half_index,
                    l1_c_grid,
                )
            else:
                raise ValueError(f"Unknown stability base method: {method}")
            rankings.append(result.ranked_indices)
            diagnostics.append(
                {
                    "phase": phase,
                    "base_method": method,
                    "half_index": half_index,
                    "half_rows": len(half),
                    "fit_rows": len(fit_idx),
                    "inner_val_rows": len(val_idx),
                    "active_edges": len(result.active_indices),
                    "selected_c": result.selected_c,
                    "fit_seconds": result.elapsed_seconds,
                    "warning_count": len(result.warnings),
                    "warnings": " | ".join(sorted(set(result.warnings))),
                }
            )
    return (
        frequencies_from_rankings(rankings, q_grid, len(edges)),
        diagnostics,
        time.perf_counter() - started,
        int(memory.peak_rss_bytes),
    )


def refit_design(X: np.ndarray, edges: Sequence[EDGE], clip_value: float) -> np.ndarray:
    return np.c_[X, interaction_matrix(X, edges, clip_value)]


def fit_l2_refit(
    X: np.ndarray, y: np.ndarray, edges: Sequence[EDGE], clip_value: float, seed: int
) -> LogisticRegression:
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=int(seed))
    model.fit(refit_design(X, edges, clip_value), y)
    return model


def per_observation_log_loss(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_column = {int(label): i for i, label in enumerate(classes)}
    columns = np.array([class_to_column[int(label)] for label in y], dtype=int)
    chosen = probability[np.arange(len(y)), columns]
    return -np.log(np.clip(chosen, 1e-15, 1.0))


def build_calibration_path(
    frequencies: dict[int, np.ndarray],
    pi_grid: Sequence[float],
    edges: Sequence[EDGE],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clip_value: float,
    seed: int,
) -> list[dict[str, Any]]:
    path: list[dict[str, Any]] = []
    seen: set[tuple[EDGE, ...]] = set()
    for q_value in sorted(frequencies):
        frequency = frequencies[q_value]
        for pi_value in sorted((float(value) for value in pi_grid), reverse=True):
            selected_indices = np.flatnonzero(frequency >= pi_value)
            selected = tuple(sorted(edges[int(i)] for i in selected_indices))
            if selected in seen:
                continue
            seen.add(selected)
            model = fit_l2_refit(X_train, y_train, selected, clip_value, seed)
            val_probability = model.predict_proba(
                refit_design(X_val, selected, clip_value)
            )
            losses = per_observation_log_loss(y_val, val_probability, model.classes_)
            path.append(
                {
                    "q": int(q_value),
                    "pi": float(pi_value),
                    "support_size": len(selected),
                    "support": selected,
                    "validation_log_loss": float(np.mean(losses)),
                    "validation_log_loss_se": float(
                        np.std(losses, ddof=1) / math.sqrt(len(losses))
                    ),
                }
            )
    if not path:
        raise RuntimeError("No calibration support candidates were generated")
    return path


def select_one_se(path: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = min(
        path,
        key=lambda row: (
            row["validation_log_loss"],
            row["support_size"],
            -row["pi"],
            row["q"],
        ),
    )
    ceiling = best["validation_log_loss"] + best["validation_log_loss_se"]
    eligible = [row for row in path if row["validation_log_loss"] <= ceiling + 1e-12]
    chosen = min(
        eligible,
        key=lambda row: (
            row["support_size"],
            -row["pi"],
            row["q"],
            tuple(row["support"]),
        ),
    )
    return {**chosen, "one_se_ceiling": float(ceiling)}


def stable_support(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    edges: Sequence[EDGE],
    config: dict[str, Any],
    n_pairs: int,
    seed: int,
) -> tuple[list[EDGE], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tuning_frequency, tuning_diag, tuning_seconds, tuning_peak = stability_rankings(
        method,
        X_train,
        y_train,
        edges,
        float(config["interaction_clip"]),
        config["q_grid"],
        n_pairs,
        seed + 100,
        config["shil"],
        config["l1_c_grid"],
        "tuning_train",
    )
    calibration = build_calibration_path(
        tuning_frequency,
        config["pi_grid"],
        edges,
        X_train,
        y_train,
        X_val,
        y_val,
        float(config["interaction_clip"]),
        seed + 200,
    )
    chosen = select_one_se(calibration)
    final_frequency, final_diag, final_seconds, final_peak = stability_rankings(
        method,
        X_dev,
        y_dev,
        edges,
        float(config["interaction_clip"]),
        [int(chosen["q"])],
        n_pairs,
        seed + 300,
        config["shil"],
        config["l1_c_grid"],
        "final_development",
    )
    final_q = int(chosen["q"])
    final_indices = np.flatnonzero(final_frequency[final_q] >= float(chosen["pi"]))
    final_edges = [edges[int(i)] for i in final_indices]
    frequency_rows: list[dict[str, Any]] = []
    for phase, frequency_map in [
        ("tuning_train", tuning_frequency),
        ("final_development", final_frequency),
    ]:
        for q_value, values in frequency_map.items():
            for edge_index, value in enumerate(values):
                frequency_rows.append(
                    {
                        "phase": phase,
                        "base_method": method,
                        "q": int(q_value),
                        "edge_index": int(edge_index),
                        "edge": edge_to_string(edges[edge_index]),
                        "frequency": float(value),
                    }
                )
    calibration_rows = [
        {
            **{key: value for key, value in row.items() if key != "support"},
            "support": ";".join(edge_to_string(edge) for edge in row["support"]),
            "base_method": method,
            "chosen": bool(
                row["q"] == chosen["q"]
                and row["pi"] == chosen["pi"]
                and tuple(row["support"]) == tuple(chosen["support"])
            ),
        }
        for row in calibration
    ]
    metadata = {
        "chosen_q": int(chosen["q"]),
        "chosen_pi": float(chosen["pi"]),
        "tuning_support_size": int(chosen["support_size"]),
        "final_support_size": len(final_edges),
        "validation_log_loss": float(chosen["validation_log_loss"]),
        "validation_log_loss_se": float(chosen["validation_log_loss_se"]),
        "one_se_ceiling": float(chosen["one_se_ceiling"]),
        "selection_seconds": float(tuning_seconds + final_seconds),
        "peak_rss_bytes": int(max(tuning_peak, final_peak)),
        "base_fit_count": int(4 * n_pairs),
    }
    return final_edges, metadata, tuning_diag + final_diag, frequency_rows, calibration_rows


def edge_to_string(edge: EDGE) -> str:
    return "-".join(str(value) for value in edge)


def support_metrics(
    selected: Sequence[EDGE],
    true_edges: set[EDGE] | None,
    equivalence_groups: Sequence[set[EDGE]] = (),
) -> dict[str, float | int]:
    selected_set = set(selected)
    if true_edges is None:
        return {
            "support_precision": math.nan,
            "support_recall": math.nan,
            "support_f1": math.nan,
            "false_inclusions": math.nan,
            "false_discovery_proportion": math.nan,
            "equivalence_precision": math.nan,
            "equivalence_recall": math.nan,
        }
    hits = len(selected_set & true_edges)
    precision = hits / len(selected_set) if selected_set else (1.0 if not true_edges else 0.0)
    recall = hits / len(true_edges) if true_edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_inclusions = len(selected_set - true_edges)
    fdp = false_inclusions / len(selected_set) if selected_set else 0.0
    eq_precision = math.nan
    eq_recall = math.nan
    if equivalence_groups:
        acceptable = set().union(*equivalence_groups)
        group_hits = sum(bool(selected_set & group) for group in equivalence_groups)
        eq_recall = group_hits / len(equivalence_groups)
        eq_precision = len(selected_set & acceptable) / len(selected_set) if selected_set else 0.0
    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "false_inclusions": int(false_inclusions),
        "false_discovery_proportion": float(fdp),
        "equivalence_precision": float(eq_precision),
        "equivalence_recall": float(eq_recall),
    }


def predictive_metrics(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    prediction = classes[np.argmax(probability, axis=1)]
    output = {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "log_loss": float(log_loss(y, probability, labels=classes)),
    }
    try:
        output["roc_auc_ovr"] = float(
            roc_auc_score(
                y,
                probability[:, 1] if probability.shape[1] == 2 else probability,
                multi_class="ovr",
            )
        )
    except ValueError:
        output["roc_auc_ovr"] = math.nan
    return output


def fit_outer_l1_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    edges: Sequence[EDGE],
    config: dict[str, Any],
    seed: int,
) -> RankingResult:
    X = np.r_[X_train, X_val]
    y = np.r_[y_train, y_val]
    Z = interaction_matrix(X, edges, float(config["interaction_clip"]))
    fit_idx = np.arange(len(X_train))
    val_idx = np.arange(len(X_train), len(X))
    return fit_l1_ranking(X, Z, y, fit_idx, val_idx, seed, config["l1_c_grid"])


def fit_outer_shil_ranking(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    edges: Sequence[EDGE],
    config: dict[str, Any],
    seed: int,
) -> RankingResult:
    X = np.r_[X_train, X_val]
    y = np.r_[y_train, y_val]
    Z = interaction_matrix(X, edges, float(config["interaction_clip"]))
    fit_idx = np.arange(len(X_train))
    val_idx = np.arange(len(X_train), len(X))
    return fit_shil_ranking(X, Z, y, fit_idx, val_idx, seed, config["shil"])


def evaluate_support(
    ds: Dataset,
    model_name: str,
    support: Sequence[EDGE],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    split_seed: int,
    refit_seed: int,
    selection_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_seed = int(ds.metadata.get("generator_seed", split_seed))
    with PeakRSSMonitor() as memory:
        fit_started = time.perf_counter()
        model = fit_l2_refit(
            X_dev,
            y_dev,
            support,
            float(config["interaction_clip"]),
            refit_seed,
        )
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        probability = model.predict_proba(
            refit_design(X_test, support, float(config["interaction_clip"]))
        )
        predict_seconds = time.perf_counter() - predict_started
    row: dict[str, Any] = {
        "dataset": ds.name,
        "scenario": ds.metadata.get("id", ds.name),
        "data_seed": data_seed,
        "split_seed": int(split_seed),
        "model": model_name,
        "feature_count": int(X_dev.shape[1]),
        "candidate_edges": len(candidate_edges(X_dev.shape[1], config["candidate_orders"])),
        "selected_edges": len(support),
        "refit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_rss_bytes": int(max(memory.peak_rss_bytes, selection_metadata.get("peak_rss_bytes", 0))),
        **predictive_metrics(y_test, probability, model.classes_),
        **support_metrics(support, ds.true_edges, ds.equivalence_groups),
        **selection_metadata,
    }
    edge_rows = [
        {
            "dataset": ds.name,
            "scenario": ds.metadata.get("id", ds.name),
            "data_seed": data_seed,
            "split_seed": int(split_seed),
            "model": model_name,
            "rank": rank,
            "edge": edge_to_string(edge),
            "order": len(edge),
            "is_true_edge": None if ds.true_edges is None else bool(edge in ds.true_edges),
        }
        for rank, edge in enumerate(support, start=1)
    ]
    return row, edge_rows


def run_dataset(
    ds: Dataset,
    split_seed: int,
    config: dict[str, Any],
    n_pairs: int,
) -> dict[str, list[dict[str, Any]]]:
    split = outer_split_scale(ds, split_seed)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    X_dev, y_dev = np.r_[X_train, X_val], np.r_[y_train, y_val]
    edges = candidate_edges(X_train.shape[1], config["candidate_orders"])

    sc_shil, sc_shil_meta, shil_diag, shil_freq, shil_path = stable_support(
        "shil",
        X_train,
        y_train,
        X_val,
        y_val,
        X_dev,
        y_dev,
        edges,
        config,
        n_pairs,
        split_seed + 10000,
    )
    sc_l1, sc_l1_meta, l1_diag, l1_freq, l1_path = stable_support(
        "l1",
        X_train,
        y_train,
        X_val,
        y_val,
        X_dev,
        y_dev,
        edges,
        config,
        n_pairs,
        split_seed + 20000,
    )
    outer_l1 = fit_outer_l1_ranking(
        X_train, y_train, X_val, y_val, edges, config, split_seed + 30000
    )
    outer_shil = fit_outer_shil_ranking(
        X_train, y_train, X_val, y_val, edges, config, split_seed + 40000
    )
    l1_full = [edges[int(i)] for i in outer_l1.ranked_indices]
    l1_top8 = l1_full[:8]
    l1_match = l1_full[: len(sc_shil)]
    shil_k8 = [edges[int(i)] for i in outer_shil.ranked_indices[:8]]

    supports = [
        (
            "L1-full",
            l1_full,
            {
                "selection_seconds": outer_l1.elapsed_seconds,
                "selected_c": outer_l1.selected_c,
                "chosen_q": None,
                "chosen_pi": None,
                "base_fit_count": len(config["l1_c_grid"]) + 1,
                "warning_count": len(outer_l1.warnings),
            },
        ),
        (
            "L1-top8",
            l1_top8,
            {
                "selection_seconds": outer_l1.elapsed_seconds,
                "selected_c": outer_l1.selected_c,
                "chosen_q": 8,
                "chosen_pi": None,
                "base_fit_count": len(config["l1_c_grid"]) + 1,
                "warning_count": len(outer_l1.warnings),
            },
        ),
        (
            "L1-match",
            l1_match,
            {
                "selection_seconds": outer_l1.elapsed_seconds,
                "selected_c": outer_l1.selected_c,
                "chosen_q": len(sc_shil),
                "chosen_pi": None,
                "base_fit_count": len(config["l1_c_grid"]) + 1,
                "warning_count": len(outer_l1.warnings),
            },
        ),
        (
            "SHIL-k8",
            shil_k8,
            {
                "selection_seconds": outer_shil.elapsed_seconds,
                "selected_c": None,
                "chosen_q": 8,
                "chosen_pi": None,
                "base_fit_count": 1,
                "warning_count": 0,
            },
        ),
        ("SC-L1", sc_l1, {**sc_l1_meta, "selected_c": None, "warning_count": sum(row["warning_count"] for row in l1_diag)}),
        ("SC-SHIL", sc_shil, {**sc_shil_meta, "selected_c": None, "warning_count": 0}),
    ]

    metrics: list[dict[str, Any]] = []
    selected_edges: list[dict[str, Any]] = []
    for model_name, support, metadata in supports:
        row, edge_rows = evaluate_support(
            ds,
            model_name,
            support,
            X_dev,
            y_dev,
            X_test,
            y_test,
            config,
            split_seed,
            split_seed + 50000,
            metadata,
        )
        metrics.append(row)
        selected_edges.extend(edge_rows)

    for collection in (shil_diag, l1_diag, shil_freq, l1_freq, shil_path, l1_path):
        for row in collection:
            row.update(
                {
                    "dataset": ds.name,
                    "scenario": ds.metadata.get("id", ds.name),
                    "data_seed": int(ds.metadata.get("generator_seed", split_seed)),
                    "split_seed": int(split_seed),
                }
            )
    return {
        "metrics": metrics,
        "selected_edges": selected_edges,
        "base_fit_diagnostics": shil_diag + l1_diag,
        "selection_frequencies": shil_freq + l1_freq,
        "calibration_path": shil_path + l1_path,
    }


def environment_payload() -> dict[str, Any]:
    process = psutil.Process()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_bytes": psutil.virtual_memory().total,
        "initial_rss_bytes": process.memory_info().rss,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "psutil": psutil.__version__,
    }


def resolve_run_plan(mode: str, config: dict[str, Any]) -> tuple[list[Dataset], list[int], int]:
    if mode == "smoke":
        spec = {**config["scenarios"][0], "n": 320, "d": 10}
        return [make_synthetic(spec, 809)], [809], int(config["n_pairs_smoke"])
    if mode == "pilot":
        selected_ids = {"S03", "S07", "S11"}
        datasets = [
            make_synthetic(spec, seed)
            for spec in config["scenarios"]
            if spec["id"] in selected_ids
            for seed in config["pilot_seeds"]
        ]
        return datasets, [int(ds.metadata["generator_seed"]) for ds in datasets], int(config["n_pairs_pilot"])
    if mode == "full-synthetic":
        datasets = [
            make_synthetic(spec, seed)
            for spec in config["scenarios"]
            for seed in config["outer_seeds"]
        ]
        return datasets, [int(ds.metadata["generator_seed"]) for ds in datasets], int(config["n_pairs_full"])
    if mode == "full-real":
        dataset = load_dry_bean(config["real_dataset"])
        return [dataset for _ in config["real_split_seeds"]], [int(seed) for seed in config["real_split_seeds"]], int(config["n_pairs_full"])
    raise ValueError(f"Unsupported mode: {mode}")


def run(config_path: Path, output: Path, mode: str) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_protocol = str(config["protocol_sha256"]).upper()
    project_root = Path(__file__).resolve().parents[3]
    protocol_path = project_root / "MD" / "02_design" / "method_extension_protocol_20260717.md"
    actual_protocol = sha256_file(protocol_path)
    if actual_protocol != expected_protocol:
        raise RuntimeError(
            f"Protocol hash mismatch: expected {expected_protocol}, observed {actual_protocol}"
        )
    write_json(output / "environment.json", environment_payload())
    write_json(output / "config_resolved.json", config)
    write_json(
        output / "run_identity.json",
        {
            "experiment_id": config["experiment_id"],
            "mode": mode,
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "protocol_path": str(protocol_path),
            "protocol_sha256": actual_protocol,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )

    datasets, split_seeds, n_pairs = resolve_run_plan(mode, config)
    collections: dict[str, list[dict[str, Any]]] = {
        "metrics": [],
        "selected_edges": [],
        "base_fit_diagnostics": [],
        "selection_frequencies": [],
        "calibration_path": [],
    }
    manifest_rows: list[dict[str, Any]] = []
    progress_path = output / "progress.jsonl"
    started = time.perf_counter()
    total = len(datasets)
    for index, (dataset, split_seed) in enumerate(zip(datasets, split_seeds), start=1):
        data_seed = int(dataset.metadata.get("generator_seed", split_seed))
        append_progress(
            progress_path,
            {
                "event": "dataset_started",
                "index": index,
                "total": total,
                "dataset": dataset.name,
                "data_seed": data_seed,
                "split_seed": int(split_seed),
                "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
        )
        result = run_dataset(dataset, int(split_seed), config, n_pairs)
        for key in collections:
            collections[key].extend(result[key])
        manifest_rows.append(
            {
                "dataset": dataset.name,
                "data_seed": data_seed,
                "split_seed": int(split_seed),
                "rows": len(dataset.y),
                "features": dataset.X.shape[1],
                "classes": len(np.unique(dataset.y)),
                "class_counts": json.dumps(
                    {int(k): int(v) for k, v in zip(*np.unique(dataset.y, return_counts=True))},
                    sort_keys=True,
                ),
                "true_edges": "" if dataset.true_edges is None else ";".join(sorted(edge_to_string(edge) for edge in dataset.true_edges)),
                "metadata": json.dumps(dataset.metadata, ensure_ascii=False, sort_keys=True),
            }
        )
        append_progress(
            progress_path,
            {
                "event": "dataset_completed",
                "index": index,
                "total": total,
                "dataset": dataset.name,
                "data_seed": data_seed,
                "split_seed": int(split_seed),
                "elapsed_seconds": time.perf_counter() - started,
                "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
        )

    for name, rows in collections.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output / "dataset_manifest.csv", index=False)
    summary = {
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "status": "completed",
        "dataset_runs": total,
        "method_rows": len(collections["metrics"]),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(output / "run_summary.json", summary)
    write_json(
        output / "output_hashes.json",
        {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "output_hashes.json"
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["smoke", "pilot", "full-synthetic", "full-real"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.config, args.output, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
