from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import io
import json
import math
import multiprocessing
import os
import platform
import shutil
import signal
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

from support_metrics import candidate_wide_average_precision, support_metrics
import durability
import freeze_bindings


EDGE = tuple[int, ...]
PILOT_A_FREEZE_RELATIVE_PATH = "freeze/SCIENTIFIC_RUN_FROZEN_PILOT_A.json"
PILOT_A_PREFLIGHT_RELATIVE_PATH = (
    "evidence/vps_masked_timing_b4_20260826/" "PILOT_TIMING_RESOURCE_PREFLIGHT.json"
)
PILOT_A_PREFLIGHT_SHA256 = (
    "F4B6A56B644843E1A15B7B2B6DA10EEACE0999008F190A27F86BAA33F57BF8EE"
)
PILOT_A_SCENARIOS = ["S00", "S11", "S13"]
PILOT_A_SEEDS = [901, 907]
FULL_SCIENTIFIC_RUN_SPECS = {
    "scientific_full_synthetic_primary": {
        "config_path": "config/full_synthetic_primary_config.json",
        "freeze_path": "freeze/SCIENTIFIC_RUN_FROZEN_FULL_SYNTHETIC_PRIMARY.json",
        "mode": "full-synthetic",
        "run_class": "full-synthetic",
        "structured_execution": False,
    },
    "scientific_full_real_descriptive": {
        "config_path": "config/full_real_descriptive_config.json",
        "freeze_path": "freeze/SCIENTIFIC_RUN_FROZEN_FULL_REAL_DESCRIPTIVE.json",
        "mode": "full-real",
        "run_class": "full-real",
        "structured_execution": False,
    },
    "scientific_full_structured_s13": {
        "config_path": "config/full_structured_s13_config.json",
        "freeze_path": "freeze/SCIENTIFIC_RUN_FROZEN_FULL_STRUCTURED_S13.json",
        "mode": "full-structured-s13",
        "run_class": "full-synthetic",
        "structured_execution": True,
    },
}
MAX_PARALLEL_ATOMIC_UNITS = 3
PARALLEL_WORKER_CONTAINMENT = "inherited_watchdog_process_group_plus_linux_pdeathsig"


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


@contextmanager
def _wall_clock_timeout(seconds: float, *, require_posix: bool = False):
    """Enforce a real per-unit wall timeout on POSIX production hosts."""

    timeout = float(seconds)
    if timeout <= 0:
        raise ValueError("Per-unit timeout must be positive")
    if os.name != "posix" or not hasattr(signal, "setitimer"):
        if require_posix:
            raise RuntimeError(
                "Scientific per-unit wall timeout requires POSIX setitimer support"
            )
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"Atomic unit exceeded {timeout:.3f} seconds")

    prior_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, prior_timer[0], prior_timer[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    durability.atomic_write_json(path, payload)


def append_progress(path: Path, payload: dict[str, Any]) -> None:
    durability.append_jsonl_durable(path, payload)


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
    if kind == "null":
        return [], []
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
    if edges and max(max(edge) for edge in edges) >= d:
        raise ValueError(
            f"Scenario {spec['id']} has too few features for planted edges"
        )

    equivalence_groups: list[set[EDGE]] = []
    if kind == "redundant_mixed":
        proxy_corr = float(spec.get("proxy_correlation", 0.95))
        residual_scale = math.sqrt(max(1.0 - proxy_corr**2, 0.0))
        proxy_pairs = [(16, 0), (17, 1), (18, 4), (19, 5)]
        for proxy, source in proxy_pairs:
            X[:, proxy] = proxy_corr * X[:, source] + residual_scale * rng.normal(
                size=n
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
        score += (
            0.15 * X[:, nuisance_candidates[0]] - 0.12 * X[:, nuisance_candidates[1]]
        )
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
        raise ValueError(
            f"Dry Bean cached archive hash mismatch: {observed_archive_hash}"
        )
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        arff_bytes = archive.read(str(config["archive_member"]))
    observed_arff_hash = hashlib.sha256(arff_bytes).hexdigest().upper()
    if observed_arff_hash != str(config["arff_sha256"]).upper():
        raise ValueError(f"Dry Bean ARFF hash mismatch: {observed_arff_hash}")

    records, _ = arff.loadarff(io.StringIO(arff_bytes.decode("utf-8")))
    frame = pd.DataFrame(records)
    target = frame.pop(str(config["target_column"])).map(
        lambda value: (
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
        )
    )
    if frame.columns.tolist() != list(config["expected_feature_names"]):
        raise ValueError(
            f"Unexpected Dry Bean feature columns: {frame.columns.tolist()}"
        )
    X_frame = frame.apply(pd.to_numeric, errors="raise")
    if X_frame.isna().any().any():
        raise ValueError(
            "Dry Bean contains missing values; locked no-imputation path violated"
        )
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
    content_hash = (
        hashlib.sha256(
            pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
        )
        .hexdigest()
        .upper()
    )
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
    if (
        set(train_idx) & set(val_idx)
        or set(train_idx) & set(test_idx)
        or set(val_idx) & set(test_idx)
    ):
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


def inner_split(
    indices: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
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
        val_loss = log_loss(y_val, softmax(val_logits), labels=np.arange(n_classes))
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
    from importlib.metadata import version
    minor = tuple(int(v) for v in version("scikit-learn").split(".")[:2])
    if minor not in ((1, 7), (1, 8)):
        raise RuntimeError("Unsupported sklearn version for audited L1 semantics")
    regularization = {"penalty": "l1"} if minor == (1, 7) else {"l1_ratio": 1.0}
    if n_classes == 2:
        return {
            "solver": "liblinear",
            **regularization,
            "C": float(c_value),
            "max_iter": 1200,
            "tol": 1e-3,
            "random_state": int(seed),
        }
    return {
        "solver": "saga",
        **regularization,
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
            warnings.simplefilter("always")
            model = LogisticRegression(**_l1_parameters(c_value, seed, n_classes))
            model.fit(design[fit_idx], y[fit_idx])
        warning_text.extend(str(item.message) for item in caught)
        loss = log_loss(
            y[val_idx], model.predict_proba(design[val_idx]), labels=model.classes_
        )
        if (
            best is None
            or loss < best[0] - 1e-12
            or (abs(loss - best[0]) <= 1e-12 and c_value < best[1])
        ):
            best = (float(loss), float(c_value))
    if best is None:
        raise RuntimeError("L1 C-grid selection failed")
    selected_c = best[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model = LogisticRegression(**_l1_parameters(selected_c, seed, n_classes))
        combined = np.sort(np.r_[fit_idx, val_idx])
        final_model.fit(design[combined], y[combined])
    warning_text.extend(str(item.message) for item in caught)
    edge_coef = np.asarray(final_model.coef_)[:, X.shape[1] :]
    scores = np.linalg.norm(edge_coef, axis=0)
    scores = np.where(scores > 1e-8, scores, 0.0)
    active = np.flatnonzero(scores > 0.0)
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
            fit_idx, val_idx = inner_split(half, y, seed + 1000 + half_index)
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


def per_observation_log_loss(
    y: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> np.ndarray:
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
) -> tuple[
    list[EDGE],
    np.ndarray,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
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
    return (
        final_edges,
        final_frequency[final_q].copy(),
        metadata,
        tuning_diag + final_diag,
        frequency_rows,
        calibration_rows,
    )


def edge_to_string(edge: EDGE) -> str:
    return "-".join(str(value) for value in edge)


def predictive_metrics(
    y: np.ndarray, probability: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
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
    candidate_universe: Sequence[EDGE],
    candidate_scores: Sequence[float] | np.ndarray,
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
        "candidate_edges": len(candidate_universe),
        "selected_edges": len(support),
        "candidate_average_precision": candidate_wide_average_precision(
            candidate_universe,
            candidate_scores,
            ds.true_edges,
            score_direction="higher_is_better",
        ),
        "candidate_score_direction": "higher_is_better",
        "refit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "peak_rss_bytes": int(
            max(memory.peak_rss_bytes, selection_metadata.get("peak_rss_bytes", 0))
        ),
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
            "is_true_edge": (
                None if ds.true_edges is None else bool(edge in ds.true_edges)
            ),
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
    """Run the historical six-method compatibility path.

    This function remains available only for regression tests that bind the
    unchanged legacy rankers.  The candidate CLI bypasses it and routes every
    cell through :func:`track_a_runner.run_track_a_cell`.
    """
    split = outer_split_scale(ds, split_seed)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    X_dev, y_dev = np.r_[X_train, X_val], np.r_[y_train, y_val]
    edges = candidate_edges(X_train.shape[1], config["candidate_orders"])

    sc_shil, sc_shil_scores, sc_shil_meta, shil_diag, shil_freq, shil_path = (
        stable_support(
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
    )
    sc_l1, sc_l1_scores, sc_l1_meta, l1_diag, l1_freq, l1_path = stable_support(
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
            outer_l1.scores,
            {
                "selection_seconds": outer_l1.elapsed_seconds,
                "selected_c": outer_l1.selected_c,
                "chosen_q": None,
                "chosen_pi": None,
                "base_fit_count": len(config["l1_c_grid"]) + 1,
                "warning_count": len(outer_l1.warnings),
                "candidate_score_source": "outer_l1_coefficient_norm",
            },
        ),
        (
            "L1-top8",
            l1_top8,
            outer_l1.scores,
            {
                "selection_seconds": outer_l1.elapsed_seconds,
                "selected_c": outer_l1.selected_c,
                "chosen_q": 8,
                "chosen_pi": None,
                "base_fit_count": len(config["l1_c_grid"]) + 1,
                "warning_count": len(outer_l1.warnings),
                "candidate_score_source": "outer_l1_coefficient_norm",
            },
        ),
        (
            "L1-match",
            l1_match,
            outer_l1.scores,
            {
                "selection_seconds": outer_l1.elapsed_seconds,
                "selected_c": outer_l1.selected_c,
                "chosen_q": len(sc_shil),
                "chosen_pi": None,
                "base_fit_count": len(config["l1_c_grid"]) + 1,
                "warning_count": len(outer_l1.warnings),
                "candidate_score_source": "outer_l1_coefficient_norm",
            },
        ),
        (
            "SHIL-k8",
            shil_k8,
            outer_shil.scores,
            {
                "selection_seconds": outer_shil.elapsed_seconds,
                "selected_c": None,
                "chosen_q": 8,
                "chosen_pi": None,
                "base_fit_count": 1,
                "warning_count": 0,
                "candidate_score_source": "outer_shil_gate_weight_norm",
            },
        ),
        (
            "SC-L1",
            sc_l1,
            sc_l1_scores,
            {
                **sc_l1_meta,
                "selected_c": None,
                "warning_count": sum(row["warning_count"] for row in l1_diag),
                "candidate_score_source": "final_development_selection_frequency",
            },
        ),
        (
            "SC-SHIL",
            sc_shil,
            sc_shil_scores,
            {
                **sc_shil_meta,
                "selected_c": None,
                "warning_count": 0,
                "candidate_score_source": "final_development_selection_frequency",
            },
        ),
    ]

    metrics: list[dict[str, Any]] = []
    selected_edges: list[dict[str, Any]] = []
    for model_name, support, candidate_scores, metadata in supports:
        row, edge_rows = evaluate_support(
            ds,
            model_name,
            support,
            edges,
            candidate_scores,
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


def resolve_run_plan(
    mode: str, config: dict[str, Any]
) -> tuple[list[Dataset], list[int], int]:
    if mode == "masked-timing":
        authorization = config.get("masked_timing_authorization")
        expected = {
            "scope": "masked_timing_s00_s11_s13",
            "engineering_only": True,
            "reserved_seed": 883,
            "scenario_package": ["S00", "S11", "S13"],
            "comparator_execution_scenarios": ["S13"],
            "sealed_output": True,
            "scientific_magnitudes_exported": False,
        }
        if (
            config.get("execution_enabled") is not True
            or config.get("execution_class") != "engineering_only_masked_timing"
            or authorization != expected
        ):
            raise ValueError("masked-timing mode requires the exact engineering gate")
        scenarios = {str(spec["id"]): spec for spec in config["scenarios"]}
        if set(expected["scenario_package"]) - set(scenarios):
            raise ValueError("masked-timing scenario package is incomplete")
        datasets = [
            make_synthetic(scenarios[scenario_id], expected["reserved_seed"])
            for scenario_id in expected["scenario_package"]
        ]
        return datasets, [expected["reserved_seed"]] * 3, int(config["n_pairs_pilot"])
    if mode == "smoke":
        spec = {**config["scenarios"][0], "n": 320, "d": 10}
        return [make_synthetic(spec, 809)], [809], int(config["n_pairs_smoke"])
    if mode == "pilot":
        if config.get("execution_class") == "scientific_pilot_a":
            expected_units = [
                {"scenario_id": scenario_id, "seed": seed}
                for scenario_id in PILOT_A_SCENARIOS
                for seed in PILOT_A_SEEDS
            ]
            expected_contract = {
                "scenario_ids": PILOT_A_SCENARIOS,
                "seeds": PILOT_A_SEEDS,
                "n_pairs": 4,
                "ranker_call_budget_4B_plus_2": 18,
                "planned_units": expected_units,
                "heartbeat_interval_seconds": 2.0,
                "per_unit_timeout_seconds": 300,
                "whole_run_watchdog_seconds": 3600,
                "max_attempts_per_unit": 2,
                "reentry_rule": "same_atomic_unit_seed_only_no_replacement_seed",
                "authorized_reentry_seeds": [],
                "blindness": {
                    "mode": "structural_only",
                    "scientific_metrics_visible_during_pilot": False,
                    "scientific_magnitudes_visible_during_pilot": False,
                    "allowed_live_fields": [
                        "status",
                        "unit_count",
                        "artifact_count",
                        "bytes",
                        "sha256",
                        "heartbeat_sequence",
                        "failure_reason",
                    ],
                },
                "timing_resource_preflight": {
                    "path": PILOT_A_PREFLIGHT_RELATIVE_PATH,
                    "sha256": PILOT_A_PREFLIGHT_SHA256,
                    "required_status": "pass",
                },
            }
            if (
                config.get("execution_enabled") is not True
                or config.get("scientific_run_authorization")
                != {
                    "scope": "separately_frozen_pilot_or_full",
                    "run_class": "pilot",
                    "freeze_path": PILOT_A_FREEZE_RELATIVE_PATH,
                }
                or config.get("pilot_a_contract") != expected_contract
                or config.get("pilot_scenario_ids") != PILOT_A_SCENARIOS
                or config.get("pilot_seeds") != PILOT_A_SEEDS
                or config.get("n_pairs_pilot") != 4
            ):
                raise ValueError("Pilot A config does not match its exact frozen plan")
        raw_selected_ids = config.get("pilot_scenario_ids")
        if (
            not isinstance(raw_selected_ids, list)
            or not raw_selected_ids
            or any(
                not isinstance(value, str) or not value for value in raw_selected_ids
            )
            or len(raw_selected_ids) != len(set(raw_selected_ids))
        ):
            raise ValueError("pilot_scenario_ids must contain unique non-empty strings")
        selected_ids = set(raw_selected_ids)
        available_ids = {str(spec["id"]) for spec in config["scenarios"]}
        if selected_ids - available_ids:
            raise ValueError(
                f"Unknown pilot scenario IDs: {sorted(selected_ids - available_ids)}"
            )
        datasets = [
            make_synthetic(spec, seed)
            for spec in config["scenarios"]
            if spec["id"] in selected_ids
            for seed in config["pilot_seeds"]
        ]
        return (
            datasets,
            [int(ds.metadata["generator_seed"]) for ds in datasets],
            int(config["n_pairs_pilot"]),
        )
    if mode == "full-synthetic":
        datasets = [
            make_synthetic(spec, seed)
            for spec in config["scenarios"]
            for seed in config["outer_seeds"]
        ]
        return (
            datasets,
            [int(ds.metadata["generator_seed"]) for ds in datasets],
            int(config["n_pairs_full"]),
        )
    if mode == "full-real":
        dataset = load_dry_bean(config["real_dataset"])
        return (
            [dataset for _ in config["real_split_seeds"]],
            [int(seed) for seed in config["real_split_seeds"]],
            int(config["n_pairs_full"]),
        )
    if mode == "full-structured-s13":
        specs = [spec for spec in config["scenarios"] if spec.get("id") == "S13"]
        if len(specs) != 1:
            raise ValueError("full-structured-s13 requires exactly one S13 spec")
        datasets = [make_synthetic(specs[0], seed) for seed in config["outer_seeds"]]
        return (
            datasets,
            [int(ds.metadata["generator_seed"]) for ds in datasets],
            0,
        )
    raise ValueError(f"Unsupported mode: {mode}")


TRACK_A_COLLECTIONS = (
    "metrics",
    "selected_edges",
    "ranking_metrics",
    "ranking_scores",
    "fit_calls",
    "calibration_path",
    "structured_sensitivity_status",
    "structured_sensitivity_metrics",
    "structured_sensitivity_selected_edges",
    "structured_sensitivity_raw",
    "partition_manifest",
)


def _resolve_protocol_path(project_root: Path, config: dict[str, Any]) -> Path:
    raw_path = config.get("protocol_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError("Candidate config must name a non-empty protocol_path")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise RuntimeError("protocol_path must be relative to the project root")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("protocol_path escapes the project root") from error
    if not resolved.is_file():
        raise RuntimeError(f"Protocol file is missing: {resolved}")
    return resolved


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _track_a_cell_metadata(ds: Dataset, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_type": "real" if ds.true_edges is None else "synthetic",
        "candidate_orders": [int(order) for order in config["candidate_orders"]],
    }


def _structured_sensitivity_metadata(
    ds: Dataset, config: dict[str, Any]
) -> dict[str, Any]:
    scenario_id = str(ds.metadata.get("id", ds.name))
    is_s13 = scenario_id == "S13"
    return {
        "scenario_id": scenario_id,
        "cell_type": (
            "structured_pairwise"
            if is_s13
            else ("real" if ds.true_edges is None else "synthetic")
        ),
        "candidate_orders": (
            [2] if is_s13 else [int(order) for order in config["candidate_orders"]]
        ),
        "heredity": str(ds.metadata.get("heredity", "not_applicable")),
        "hierarchy": "strong" if is_s13 else "not_applicable",
        "analysis_role": "sensitivity_only",
        "primary_estimand": False,
    }


def _structured_authorization_context(
    ds: Dataset,
    split_seed: int,
    config: dict[str, Any],
    binding: Any,
) -> dict[str, Any] | None:
    """Build the one eligible live-config authorization context, or no context."""

    cell_metadata = _structured_sensitivity_metadata(ds, config)
    if cell_metadata != {
        "scenario_id": "S13",
        "cell_type": "structured_pairwise",
        "candidate_orders": [2],
        "heredity": "respected",
        "hierarchy": "strong",
        "analysis_role": "sensitivity_only",
        "primary_estimand": False,
    }:
        return None
    timing = config.get("masked_timing_authorization")
    expected = {
        "scope": "masked_timing_s00_s11_s13",
        "engineering_only": True,
        "reserved_seed": 883,
        "scenario_package": ["S00", "S11", "S13"],
        "comparator_execution_scenarios": ["S13"],
        "sealed_output": True,
        "scientific_magnitudes_exported": False,
    }
    candidate_root = binding.config_path.parent.parent.resolve()
    try:
        relative_config = binding.config_path.resolve().relative_to(candidate_root)
    except ValueError as error:
        raise RuntimeError("Execution config escapes the candidate root") from error
    if timing is not None:
        if (
            timing != expected
            or binding.execution_enabled is not True
            or int(split_seed) != expected["reserved_seed"]
            or relative_config.as_posix() != "config/timing_config.json"
        ):
            raise RuntimeError("Masked timing authorization context is not exact")
        return {
            "scope": expected["scope"],
            "execution_config_binding": {
                "path": relative_config.as_posix(),
                "sha256": binding.config_sha256,
            },
            "cell_metadata": cell_metadata,
            "scope_binding": {
                "scenario_package": expected["scenario_package"],
                "reserved_seed": expected["reserved_seed"],
                "sealed_output": expected["sealed_output"],
                "scientific_magnitudes_exported": expected[
                    "scientific_magnitudes_exported"
                ],
            },
        }
    scientific = config.get("scientific_run_authorization")
    execution_class = str(config.get("execution_class", ""))
    full_spec = FULL_SCIENTIFIC_RUN_SPECS.get(execution_class)
    if full_spec is not None:
        expected_full = {
            "scope": "separately_frozen_pilot_or_full",
            "run_class": full_spec["run_class"],
            "freeze_path": full_spec["freeze_path"],
            "authorized_mode": full_spec["mode"],
        }
        if (
            scientific != expected_full
            or full_spec["structured_execution"] is not True
            or binding.execution_enabled is not True
            or int(split_seed) not in [int(seed) for seed in config["outer_seeds"]]
            or relative_config.as_posix() != full_spec["config_path"]
        ):
            raise RuntimeError("Full S13 authorization context is not exact")
        freeze_path = candidate_root / str(full_spec["freeze_path"])
        if not freeze_path.is_file():
            raise RuntimeError("Full S13 scientific run freeze is missing")
        return {
            "scope": expected_full["scope"],
            "execution_config_binding": {
                "path": relative_config.as_posix(),
                "sha256": binding.config_sha256,
            },
            "cell_metadata": cell_metadata,
            "scope_binding": {
                "run_class": expected_full["run_class"],
                "freeze_path": str(full_spec["freeze_path"]),
                "freeze_sha256": sha256_file(freeze_path),
            },
        }
    expected_scientific = {
        "scope": "separately_frozen_pilot_or_full",
        "run_class": "pilot",
        "freeze_path": PILOT_A_FREEZE_RELATIVE_PATH,
    }
    if scientific is None:
        return None
    if (
        scientific != expected_scientific
        or binding.execution_enabled is not True
        or int(split_seed) not in PILOT_A_SEEDS
        or relative_config.as_posix() != "config/pilot_config.json"
    ):
        raise RuntimeError("Pilot A authorization context is not exact")
    freeze_path = candidate_root / PILOT_A_FREEZE_RELATIVE_PATH
    if not freeze_path.is_file():
        raise RuntimeError("Pilot A scientific run freeze is missing")
    return {
        "scope": expected_scientific["scope"],
        "execution_config_binding": {
            "path": relative_config.as_posix(),
            "sha256": binding.config_sha256,
        },
        "cell_metadata": cell_metadata,
        "scope_binding": {
            "run_class": "pilot",
            "freeze_path": PILOT_A_FREEZE_RELATIVE_PATH,
            "freeze_sha256": sha256_file(freeze_path),
        },
    }


def _track_a_identity(ds: Dataset, split_seed: int) -> dict[str, Any]:
    return {
        "dataset": ds.name,
        "scenario": ds.metadata.get("id", ds.name),
        "data_seed": int(ds.metadata.get("generator_seed", split_seed)),
        "split_seed": int(split_seed),
    }


def _with_identity(
    rows: Sequence[dict[str, Any]], identity: dict[str, Any]
) -> list[dict[str, Any]]:
    return [{**identity, **dict(row)} for row in rows]


def _validate_track_a_partition(
    collections: dict[str, list[dict[str, Any]]],
    *,
    registry: Any,
    cell_metadata: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate one serialized partition against registry-derived identities."""

    if set(collections) != set(TRACK_A_COLLECTIONS):
        raise RuntimeError("Track-A partition collection set is incomplete")
    expected_methods = tuple(registry.expected_methods_for(cell_metadata))
    expected_families = tuple(
        dict.fromkeys(
            registry.method_by_id[method_id].score_family_id
            for method_id in expected_methods
        )
    )
    for label in ("metrics", "selected_edges"):
        observed = [str(row["method_id"]) for row in collections[label]]
        if len(observed) != len(set(observed)) or set(observed) != set(
            expected_methods
        ):
            raise RuntimeError(
                f"Track-A {label} rows do not match the registry method set"
            )
    for row in collections["metrics"]:
        if row.get("status") != "OK":
            raise RuntimeError("Track-A primary metric status is not OK")
        if row.get("zero_padding_applied") is not False:
            raise RuntimeError("Track-A zero-padding flag must be explicit false")
    for label in ("ranking_metrics", "ranking_scores"):
        observed = [str(row["score_family_id"]) for row in collections[label]]
        if len(observed) != len(set(observed)) or set(observed) != set(
            expected_families
        ):
            raise RuntimeError(
                f"Track-A {label} rows do not match the registry score-family set"
            )
        if any(row.get("status") != "OK" for row in collections[label]):
            raise RuntimeError(f"Track-A {label} status is not OK")
    manifest = collections["partition_manifest"]
    if len(manifest) != 1:
        raise RuntimeError("Track-A partition must contain exactly one manifest row")
    if json.loads(manifest[0]["expected_method_ids_json"]) != list(expected_methods):
        raise RuntimeError("Track-A partition manifest method set mismatch")
    if json.loads(manifest[0]["expected_score_family_ids_json"]) != list(
        expected_families
    ):
        raise RuntimeError("Track-A partition manifest score-family set mismatch")
    sensitivity_status = collections["structured_sensitivity_status"]
    if len(sensitivity_status) != 1:
        raise RuntimeError("Structured sensitivity requires exactly one status row")
    status = sensitivity_status[0].get("status")
    if status not in {"N/A", "DISABLED", "COMPLETED"}:
        raise RuntimeError("Structured sensitivity status is invalid")
    sensitivity_names = (
        "structured_sensitivity_metrics",
        "structured_sensitivity_selected_edges",
        "structured_sensitivity_raw",
    )
    for name in ("structured_sensitivity_status", *sensitivity_names):
        for row in collections[name]:
            if row.get("analysis_role") != "sensitivity_only":
                raise RuntimeError("Structured sensitivity role is invalid")
            if row.get("primary_estimand") is not False:
                raise RuntimeError(
                    "Structured sensitivity entered the primary estimand"
                )
            if "method_id" in row or "score_family_id" in row:
                raise RuntimeError(
                    "Structured sensitivity cannot use primary method identities"
                )
    if status in {"N/A", "DISABLED"} and any(
        collections[name] for name in sensitivity_names
    ):
        raise RuntimeError("N/A/disabled sensitivity cannot carry substitute outputs")
    if status == "COMPLETED" and (
        len(collections["structured_sensitivity_metrics"]) != 1
        or len(collections["structured_sensitivity_raw"]) != 1
    ):
        raise RuntimeError("Completed structured sensitivity artifacts are incomplete")
    if manifest[0].get("structured_registry_sha256") != sensitivity_status[0].get(
        "structured_registry_sha256"
    ):
        raise RuntimeError("Structured sensitivity registry binding mismatch")
    return expected_methods, expected_families


def run_track_a_partition(
    ds: Dataset,
    split_seed: int,
    config: dict[str, Any],
    binding: Any,
    n_pairs: int,
    *,
    cell_runner: Any = None,
    structured_runner: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run and serialize one registry-bound Track-A cell.

    Dependency injection is retained for contract tests.  Production calls use
    the function-only runner imported locally to avoid a legacy-module cycle.
    """

    if cell_runner is None:
        from track_a_runner import run_track_a_cell as cell_runner
    if structured_runner is None:
        from structured_comparator_runtime import (
            run_structured_sensitivity_cell as structured_runner,
        )

    cell_metadata = _track_a_cell_metadata(ds, config)
    identity = _track_a_identity(ds, split_seed)
    result = cell_runner(
        ds,
        registry=binding.registry,
        resolved_config=binding.resolved_config,
        cell_metadata=cell_metadata,
        interaction_clip=float(config["interaction_clip"]),
        outer_split_seed=int(split_seed),
        master_seed=int(split_seed),
        n_pairs=int(n_pairs),
    )
    if result.registry_sha256 != binding.registry.canonical_sha256:
        raise RuntimeError("Track-A cell returned an unexpected registry hash")

    metrics = _with_identity([dict(row) for row in result.metrics], identity)
    selected_edges = _with_identity(
        [
            {
                "method_id": row["method_id"],
                "ranker_id": row["ranker_id"],
                "policy_id": row["policy_id"],
                "score_family_id": row["score_family_id"],
                "selected_edge_count": len(row["edges"]),
                "support_indices_json": _json_compact(list(row["support_indices"])),
                "edges_json": _json_compact([list(edge) for edge in row["edges"]]),
            }
            for row in result.selected_edges
        ],
        identity,
    )
    ranking_metrics = _with_identity(
        [dict(row) for row in result.ranking_metrics], identity
    )
    ranking_scores = _with_identity(
        [
            {
                "status": row["status"],
                "score_family_id": row["score_family_id"],
                "ranker_id": row["ranker_id"],
                "score_source": row["score_source"],
                "score_direction": row["score_direction"],
                "candidate_edges_json": _json_compact(
                    [list(edge) for edge in row["candidate_edges"]]
                ),
                "scores_json": _json_compact(list(row["scores"])),
                "ranked_indices_json": _json_compact(
                    list(row.get("ranked_indices", ()))
                ),
                "active_indices_json": _json_compact(
                    list(row.get("active_indices", ()))
                ),
            }
            for row in result.ranking_scores
        ],
        identity,
    )
    fit_calls = _with_identity(
        [
            {
                **{
                    key: value
                    for key, value in dict(row).items()
                    if key not in {"fit_indices", "validation_indices"}
                },
                "fit_indices_json": _json_compact(list(row["fit_indices"])),
                "validation_indices_json": _json_compact(
                    list(row["validation_indices"])
                ),
            }
            for row in result.fit_calls
        ],
        identity,
    )
    calibration_path = _with_identity(
        [
            {
                **{
                    key: value
                    for key, value in dict(row).items()
                    if key not in {"fit_scope_indices", "validation_scope_indices"}
                },
                "fit_scope_indices_json": _json_compact(list(row["fit_scope_indices"])),
                "validation_scope_indices_json": _json_compact(
                    list(row["validation_scope_indices"])
                ),
            }
            for row in result.calibration_rows
        ],
        identity,
    )
    sensitivity_metadata = _structured_sensitivity_metadata(ds, config)
    comparator_config = config.get("structured_comparator_candidate", {})
    structured_execution_enabled = bool(
        comparator_config.get("execution_enabled", binding.execution_enabled)
    )
    authorization_context = (
        _structured_authorization_context(ds, split_seed, config, binding)
        if structured_execution_enabled
        else None
    )
    sensitivity_result = structured_runner(
        ds,
        registry=binding.structured_registry,
        cell_metadata=sensitivity_metadata,
        execution_enabled=structured_execution_enabled,
        split_seed=int(split_seed),
        candidate_root=binding.config_path.parent.parent,
        authorization_context=authorization_context,
    )
    if (
        sensitivity_result.registry_sha256
        != binding.structured_registry.canonical_sha256
    ):
        raise RuntimeError(
            "Structured sensitivity returned an unexpected registry hash"
        )
    sensitivity_identity = {
        **identity,
        "structured_registry_sha256": sensitivity_result.registry_sha256,
    }
    sensitivity_status = _with_identity(
        [dict(row) for row in sensitivity_result.status_rows], sensitivity_identity
    )
    sensitivity_metrics = _with_identity(
        [dict(row) for row in sensitivity_result.metric_rows], sensitivity_identity
    )
    sensitivity_selected = _with_identity(
        [dict(row) for row in sensitivity_result.selected_edge_rows],
        sensitivity_identity,
    )
    sensitivity_raw = _with_identity(
        [dict(row) for row in sensitivity_result.raw_rows], sensitivity_identity
    )
    fingerprint_split = outer_split_scale(ds, int(split_seed))
    partition_manifest = [
        {
            **identity,
            "cell_type": cell_metadata["cell_type"],
            "candidate_orders_json": _json_compact(cell_metadata["candidate_orders"]),
            "registry_sha256": result.registry_sha256,
            "structured_registry_sha256": sensitivity_result.registry_sha256,
            "split_plan_sha256": result.split_plan_sha256,
            "expected_method_ids_json": _json_compact(result.expected_method_ids),
            "expected_score_family_ids_json": _json_compact(
                result.expected_score_family_ids
            ),
            "outer_test_indices_json": _json_compact(result.outer_test_indices),
            "outer_split_fingerprint_sha256": _array_fingerprint(
                fingerprint_split["train_idx"],
                fingerprint_split["val_idx"],
                fingerprint_split["test_idx"],
            ),
            "dataset_fingerprint_sha256": _array_fingerprint(ds.X, ds.y),
            "n_pairs": int(n_pairs),
        }
    ]
    collections = {
        "metrics": metrics,
        "selected_edges": selected_edges,
        "ranking_metrics": ranking_metrics,
        "ranking_scores": ranking_scores,
        "fit_calls": fit_calls,
        "calibration_path": calibration_path,
        "structured_sensitivity_status": sensitivity_status,
        "structured_sensitivity_metrics": sensitivity_metrics,
        "structured_sensitivity_selected_edges": sensitivity_selected,
        "structured_sensitivity_raw": sensitivity_raw,
        "partition_manifest": partition_manifest,
    }
    _validate_track_a_partition(
        collections, registry=binding.registry, cell_metadata=cell_metadata
    )
    if config.get("execution_class") == "scientific_full_real_descriptive":
        for name, rows in collections.items():
            if name == "partition_manifest":
                continue
            for row in rows:
                row["reporting_label"] = "DESCRIPTIVE_NO_INFERENCE"
        collections["partition_manifest"][0][
            "reporting_label"
        ] = "DESCRIPTIVE_NO_INFERENCE"
    return collections


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_json_compact(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def run_structured_s13_partition(
    ds: Dataset,
    split_seed: int,
    config: dict[str, Any],
    binding: Any,
    *,
    structured_runner: Any = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run only the isolated S13 sensitivity comparator, never primary methods."""

    if config.get("execution_class") != "scientific_full_structured_s13":
        raise RuntimeError(
            "Comparator-only partition requires the exact full S13 class"
        )
    if structured_runner is None:
        from structured_comparator_runtime import (
            run_structured_sensitivity_cell as structured_runner,
        )
    metadata = _structured_sensitivity_metadata(ds, config)
    authorization_context = _structured_authorization_context(
        ds, split_seed, config, binding
    )
    result = structured_runner(
        ds,
        registry=binding.structured_registry,
        cell_metadata=metadata,
        execution_enabled=True,
        split_seed=int(split_seed),
        candidate_root=binding.config_path.parent.parent,
        authorization_context=authorization_context,
    )
    if result.registry_sha256 != binding.structured_registry.canonical_sha256:
        raise RuntimeError("Structured-only result registry hash mismatch")
    if result.status_rows[0].get("status") != "COMPLETED":
        raise RuntimeError("Structured-only S13 unit did not complete")
    identity = {
        **_track_a_identity(ds, split_seed),
        "structured_registry_sha256": result.registry_sha256,
        "reporting_label": "DESCRIPTIVE_NO_INFERENCE",
    }
    split = outer_split_scale(ds, int(split_seed))
    outer_split_fingerprint = _array_fingerprint(
        split["train_idx"], split["val_idx"], split["test_idx"]
    )
    dataset_fingerprint = _array_fingerprint(ds.X, ds.y)
    collections = {name: [] for name in TRACK_A_COLLECTIONS}
    collections["structured_sensitivity_status"] = _with_identity(
        [dict(row) for row in result.status_rows], identity
    )
    collections["structured_sensitivity_metrics"] = _with_identity(
        [dict(row) for row in result.metric_rows], identity
    )
    collections["structured_sensitivity_selected_edges"] = _with_identity(
        [dict(row) for row in result.selected_edge_rows], identity
    )
    collections["structured_sensitivity_raw"] = _with_identity(
        [dict(row) for row in result.raw_rows], identity
    )
    collections["partition_manifest"] = [
        {
            **_track_a_identity(ds, split_seed),
            "block_role": "structured_s13_sensitivity_only",
            "cell_type": "structured_pairwise",
            "candidate_orders_json": _json_compact([2]),
            "registry_sha256": binding.registry.canonical_sha256,
            "structured_registry_sha256": result.registry_sha256,
            "expected_method_ids_json": _json_compact([]),
            "expected_score_family_ids_json": _json_compact([]),
            "outer_test_indices_json": _json_compact(
                [int(value) for value in split["test_idx"]]
            ),
            "outer_split_fingerprint_sha256": outer_split_fingerprint,
            "dataset_fingerprint_sha256": dataset_fingerprint,
            "n_pairs": 0,
            "reporting_label": "DESCRIPTIVE_NO_INFERENCE",
        }
    ]
    for name in (
        "structured_sensitivity_status",
        "structured_sensitivity_metrics",
        "structured_sensitivity_selected_edges",
        "structured_sensitivity_raw",
    ):
        for row in collections[name]:
            if row.get("analysis_role") != "sensitivity_only":
                raise RuntimeError("Structured-only row role drift")
            if row.get("primary_estimand") is not False:
                raise RuntimeError("Structured-only row entered the primary estimand")
            if row.get("reporting_label") != "DESCRIPTIVE_NO_INFERENCE":
                raise RuntimeError("Structured-only row lacks descriptive label")
    if any(
        collections[name]
        for name in (
            "metrics",
            "selected_edges",
            "ranking_metrics",
            "ranking_scores",
            "fit_calls",
            "calibration_path",
        )
    ):
        raise RuntimeError("Structured-only unit produced primary artifacts")
    return collections


TRACK_A_DURABLE_SOURCE_PATHS = (
    "config/full_envelope_timing_config.json",
    "config/full_real_envelope_timing_config.json",
    "config/pilot_config.json",
    "config/timing_config.json",
    "config/structured_comparator_registry.json",
    "comparators/glinternet/Dockerfile",
    "comparators/glinternet/build_contract.json",
    "comparators/glinternet/python_adapter.py",
    "comparators/glinternet/r/glinternet_adapter.R",
    "comparators/glinternet/schema_contract.json",
    "evidence/claude_cli_cross_model_review_20260826/design_freeze_review_attempt_06.json",
    "evidence/glinternet_build_authorization_20260826/BUILD_AUTHORIZATION_EVIDENCE.json",
    "evidence/masked_timing_run_01_validity_20260826/TIMING_VALIDITY.json",
    "evidence/vps_masked_timing_b4_20260826/PILOT_TIMING_RESOURCE_PREFLIGHT.json",
    "evidence/vps_masked_timing_b4_20260826/RUN_MANIFEST.json",
    "evidence/vps_masked_timing_b4_20260826/TIMING_EVIDENCE.json",
    "evidence/structured_comparator_activation_20260826/ACTIVATION_EVIDENCE.json",
    "evidence/vps_arm64_glinternet_smoke_20260826/SMOKE_RESULT.json",
    "requirements_locked.txt",
    "src/analyze_real_descriptive.py",
    "src/analyze_results.py",
    "src/analyze_structured_s13.py",
    "src/durability.py",
    "src/freeze_bindings.py",
    "src/method_registry.py",
    "src/ranker_adapters.py",
    "src/ranker_protocol.py",
    "src/scientific_evidence.py",
    "src/sc_shil_experiment.py",
    "src/structured_comparator_registry.py",
    "src/structured_comparator_runtime.py",
    "src/support_metrics.py",
    "src/support_policies.py",
    "src/track_a_config.py",
    "src/track_a_runner.py",
)


def _unit_id(index: int) -> str:
    return f"unit-{int(index):04d}"


def _atomic_unit(
    index: int,
    total: int,
    dataset: Dataset,
    split_seed: int,
    mode: str,
    n_pairs: int,
) -> dict[str, Any]:
    return {
        "unit_index": int(index),
        "planned_unit_count": int(total),
        "mode": mode,
        "dataset": dataset.name,
        "scenario": dataset.metadata.get("id", dataset.name),
        "data_seed": int(dataset.metadata.get("generator_seed", split_seed)),
        "split_seed": int(split_seed),
        "n_pairs": int(n_pairs),
    }


def _dataset_manifest_row(dataset: Dataset, split_seed: int) -> dict[str, Any]:
    data_seed = int(dataset.metadata.get("generator_seed", split_seed))
    return {
        "dataset": dataset.name,
        "scenario": dataset.metadata.get("id", dataset.name),
        "data_seed": data_seed,
        "split_seed": int(split_seed),
        "rows": len(dataset.y),
        "features": dataset.X.shape[1],
        "classes": len(np.unique(dataset.y)),
        "class_counts": json.dumps(
            {
                int(key): int(value)
                for key, value in zip(*np.unique(dataset.y, return_counts=True))
            },
            sort_keys=True,
        ),
        "true_edges": (
            ""
            if dataset.true_edges is None
            else ";".join(sorted(edge_to_string(edge) for edge in dataset.true_edges))
        ),
        "true_edges_json": (
            ""
            if dataset.true_edges is None
            else _json_compact([list(edge) for edge in sorted(dataset.true_edges)])
        ),
        "equivalence_groups_json": _json_compact(
            [
                [list(edge) for edge in sorted(group)]
                for group in dataset.equivalence_groups
            ]
        ),
        "metadata": json.dumps(dataset.metadata, ensure_ascii=False, sort_keys=True),
    }


def _csv_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _validate_full_freeze_transaction(
    candidate_root: Path,
    source_binding: dict[str, Any],
    freeze_relative: str,
    freeze_hash: str,
) -> None:
    evidence_root = candidate_root / "evidence" / "full_scientific_preflight_20260826"
    transaction_path = evidence_root / "FREEZE_BUILD_TRANSACTION.json"
    summary_path = evidence_root / "FULL_FREEZE_BUILD_SUMMARY.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_freezes = {
        str(record["freeze_path"]) for record in FULL_SCIENTIFIC_RUN_SPECS.values()
    }
    blocks = summary.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(expected_freezes):
        raise RuntimeError("Full-freeze summary block set is incomplete")
    observed_freezes = {
        str(block.get("freeze")): str(block.get("freeze_sha256", "")).upper()
        for block in blocks
        if isinstance(block, dict)
    }
    if (
        transaction.get("schema_version") != 1
        or transaction.get("status") != "COMPLETE"
        or transaction.get("results_seen") is not False
        or transaction.get("source_bundle_sha256")
        != source_binding["source_bundle_sha256"]
        or transaction.get("summary_sha256") != sha256_file(summary_path)
        or summary.get("schema_version") != 1
        or summary.get("status") != "PASS"
        or summary.get("results_seen") is not False
        or summary.get("source_bundle_sha256") != source_binding["source_bundle_sha256"]
        or summary.get("source_file_count") != len(source_binding["files"])
        or set(observed_freezes) != expected_freezes
    ):
        raise RuntimeError("Full-freeze build transaction is incomplete")
    for relative, expected_hash in observed_freezes.items():
        live = candidate_root / relative
        if not live.is_file() or sha256_file(live) != expected_hash:
            raise RuntimeError(f"Full-freeze publication drift: {relative}")
    if observed_freezes.get(freeze_relative) != freeze_hash:
        raise RuntimeError("Authorized freeze is absent from completed transaction")


def _resolve_scientific_freeze(
    *,
    config: dict[str, Any],
    binding: Any,
    candidate_root: Path,
    project_root: Path,
    protocol_path: Path,
    mode: str,
    planned_units: list[dict[str, Any]],
    source_binding: dict[str, Any],
    freeze_override: tuple[dict[str, Any], str] | None = None,
) -> tuple[dict[str, str] | None, str | None, dict[str, Any] | None]:
    authorization = config.get("scientific_run_authorization")
    if authorization is None:
        return None, None, None
    freeze_relative = str(authorization.get("freeze_path", ""))
    freeze_path = (candidate_root / freeze_relative).resolve()
    try:
        freeze_path.relative_to(candidate_root)
    except ValueError as error:
        raise RuntimeError("Scientific freeze escapes the candidate root") from error
    if freeze_override is None:
        if not freeze_path.is_file():
            raise RuntimeError("Scientific run freeze is missing")
        freeze_hash = sha256_file(freeze_path)
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    else:
        freeze, freeze_hash = freeze_override
        if (
            not isinstance(freeze, dict)
            or not isinstance(freeze_hash, str)
            or len(freeze_hash) != 64
            or freeze_hash != freeze_hash.upper()
        ):
            raise RuntimeError("Scientific freeze override is malformed")
    run_id = freeze.get("authorized_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError("Scientific freeze lacks an authorized run identity")

    execution_class = str(config.get("execution_class", ""))
    if execution_class == "scientific_pilot_a":
        if freeze_relative != PILOT_A_FREEZE_RELATIVE_PATH:
            raise RuntimeError("Pilot A manifest freeze binding is not exact")
        return {"path": freeze_relative, "sha256": freeze_hash}, run_id, freeze

    spec = FULL_SCIENTIFIC_RUN_SPECS.get(execution_class)
    if spec is None:
        raise RuntimeError("Unknown scientific execution class")
    config_relative = binding.config_path.relative_to(candidate_root).as_posix()
    expected_authorization = {
        "scope": "separately_frozen_pilot_or_full",
        "run_class": spec["run_class"],
        "freeze_path": spec["freeze_path"],
        "authorized_mode": spec["mode"],
    }
    if (
        authorization != expected_authorization
        or config_relative != spec["config_path"]
        or freeze_relative != spec["freeze_path"]
        or mode != spec["mode"]
    ):
        raise RuntimeError("Full scientific authorization is not exact")
    execution_config = {
        "path": config_relative,
        "sha256": binding.config_sha256,
    }
    if freeze_override is None:
        _validate_full_freeze_transaction(
            candidate_root,
            source_binding,
            freeze_relative,
            freeze_hash,
        )
    expected_durability = {
        "heartbeat_interval_seconds": float(config["heartbeat_interval_seconds"]),
        "per_unit_timeout_seconds": int(config["per_unit_timeout_seconds"]),
        "whole_run_watchdog_seconds": int(config["whole_run_watchdog_seconds"]),
        "max_parallel_units": _max_parallel_units(config),
        "worker_containment": PARALLEL_WORKER_CONTAINMENT,
        "max_attempts_per_unit": int(config["max_attempts_per_unit"]),
        "reentry_rule": "same_atomic_unit_only_first_valid_attempt_canonical",
    }
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "SCIENTIFIC_RUN_FROZEN"
        or freeze.get("execution_enabled") is not True
        or freeze.get("results_seen") is not False
        or freeze.get("scientific_compute_executed_during_authorization") is not False
        or freeze.get("run_class") != spec["run_class"]
        or freeze.get("authorized_mode") != mode
        or freeze.get("block_role") != execution_class
        or freeze.get("execution_config") != execution_config
        or freeze.get("exact_units") != planned_units
        or freeze.get("source_bundle")
        != {
            "sha256": source_binding["source_bundle_sha256"],
            "file_count": len(source_binding["files"]),
        }
        or freeze.get("structured_comparator_execution_enabled")
        is not spec["structured_execution"]
        or freeze.get("durability") != expected_durability
    ):
        raise RuntimeError("Full scientific freeze predicate failed")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("Full scientific freeze bindings are missing")
    expected_protocol = {
        "path": protocol_path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(protocol_path),
    }
    expected_primary_registry = {
        "path": binding.registry_path.relative_to(candidate_root).as_posix(),
        "file_sha256": sha256_file(binding.registry_path),
        "canonical_sha256": binding.registry.canonical_sha256,
        "method_count": 14,
    }
    expected_structured_registry = {
        "path": binding.structured_registry_path.relative_to(candidate_root).as_posix(),
        "file_sha256": binding.structured_registry_file_sha256,
        "canonical_sha256": binding.structured_registry.canonical_sha256,
    }
    if (
        bindings.get("protocol") != expected_protocol
        or bindings.get("execution_config") != execution_config
        or bindings.get("primary_registry") != expected_primary_registry
        or bindings.get("structured_registry") != expected_structured_registry
    ):
        raise RuntimeError("Full scientific core binding mismatch")
    for label in (
        "timing_resource_preflight",
        "timing_envelope_table",
        "pilot_a2_disposition",
    ):
        record = bindings.get(label)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"Full scientific {label} binding is malformed")
        path = (candidate_root / str(record["path"])).resolve()
        try:
            path.relative_to(candidate_root)
        except ValueError as error:
            raise RuntimeError(
                f"Full scientific {label} escapes candidate root"
            ) from error
        if not path.is_file() or sha256_file(path) != str(record["sha256"]).upper():
            raise RuntimeError(f"Full scientific {label} hash mismatch")
    freeze_bindings.validate_timing_resource_preflight(
        candidate_root / bindings["timing_resource_preflight"]["path"]
    )
    return {"path": freeze_relative, "sha256": freeze_hash}, run_id, freeze


def _build_run_manifest(
    *,
    config: dict[str, Any],
    binding: Any,
    candidate_root: Path,
    project_root: Path,
    protocol_path: Path,
    mode: str,
    datasets: Sequence[Dataset],
    split_seeds: Sequence[int],
    n_pairs: int,
    heartbeat_interval_seconds: float,
    scientific_freeze_override: tuple[dict[str, Any], str] | None = None,
) -> dict[str, Any]:
    source_binding = durability.build_source_binding(
        candidate_root, TRACK_A_DURABLE_SOURCE_PATHS
    )
    total = len(datasets)
    max_parallel_units = _max_parallel_units(config)
    planned_units = [
        {
            "unit_id": _unit_id(index),
            "atomic_unit": _atomic_unit(
                index, total, dataset, int(split_seed), mode, int(n_pairs)
            ),
        }
        for index, (dataset, split_seed) in enumerate(
            zip(datasets, split_seeds), start=1
        )
    ]
    config_relative = binding.config_path.relative_to(candidate_root).as_posix()
    per_unit_timeout = int(config.get("per_unit_timeout_seconds", 14400))
    whole_run_watchdog = int(
        config.get(
            "whole_run_watchdog_seconds",
            max(28800, per_unit_timeout * total + 3600),
        )
    )
    scientific_freeze_binding, authorized_run_id, scientific_freeze = (
        _resolve_scientific_freeze(
            config=config,
            binding=binding,
            candidate_root=candidate_root,
            project_root=project_root,
            protocol_path=protocol_path,
            mode=mode,
            planned_units=planned_units,
            source_binding=source_binding,
            freeze_override=scientific_freeze_override,
        )
    )
    manifest = {
        "schema_version": durability.SCHEMA_VERSION,
        "run_id": authorized_run_id or f"{config['experiment_id']}-{mode}",
        "run_class": str(
            config.get("execution_class", "candidate_engineering_or_scientific_run")
        ),
        "status": (
            "scientific_run_frozen"
            if scientific_freeze_binding is not None
            else "candidate_unfrozen"
        ),
        "execution_enabled": bool(binding.execution_enabled),
        "atomic_unit": {
            "fields": [
                "unit_index",
                "mode",
                "dataset",
                "scenario",
                "data_seed",
                "split_seed",
                "n_pairs",
            ]
        },
        "planned_unit_count": total,
        "planned_units": planned_units,
        "bindings": {
            "protocol": {
                "path": protocol_path.relative_to(project_root).as_posix(),
                "sha256": durability.sha256_file(protocol_path),
            },
            "config": {
                "path": config_relative,
                "sha256": durability.sha256_file(binding.config_path),
            },
            "registry": {
                "path": binding.registry_path.relative_to(candidate_root).as_posix(),
                "sha256": durability.sha256_file(binding.registry_path),
            },
            "source": source_binding,
        },
        "durability": {
            "checkpoint_path_and_schema": "units/{unit_id}/attempts/{attempt_id}/checkpoint.json; schema_version=1",
            "atomic_write_strategy": "same-directory temporary file; flush; fsync; atomic os.replace",
            "resume_command": f"python src/sc_shil_experiment.py --config {config_relative} --mode {mode} --output <run-output> --resume",
            "resume_validation_rule": "completed status + exact atomic-unit and protocol/config/registry/source bindings + artifact size/hash",
            "interruption_smoke_evidence": "tests/test_track_a_durability_integration.py::test_controlled_interruption_resume_equals_uninterrupted_reference",
            "per_unit_timeout_seconds": per_unit_timeout,
            "whole_run_watchdog_seconds": whole_run_watchdog,
            "eta_basis_and_margin": {
                "artifact_path": "preflight/timing_resource_preflight.json",
                "capture": "freeze_bindings.capture_timing_resource_preflight",
                "validator": "freeze_bindings.validate_timing_resource_preflight",
                "timing_policy": "lead-supplied measured durations from at least two completed atomic units",
                "margin_policy": "explicit safety_margin_fraction in [0,2] applied to measured p95",
            },
            "max_workers_and_thread_limits": {
                "workers": max_parallel_units,
                "containment": PARALLEL_WORKER_CONTAINMENT,
                "omp_threads": 1,
                "openblas_threads": 1,
                "mkl_threads": 1,
                "numexpr_threads": 1,
            },
            "disk_ram_resource_preflight": {
                "artifact_path": "preflight/timing_resource_preflight.json",
                "schema_version": 1,
                "capture": "freeze_bindings.capture_timing_resource_preflight",
                "validator": "freeze_bindings.validate_timing_resource_preflight",
                "required_checks": ["disk", "ram", "timing_sample"],
            },
            "progress_heartbeat_path_and_stall_threshold": "units/{unit_id}/attempts/{attempt_id}/heartbeat.json; stall threshold requires stale heartbeat plus two intervals without descendant CPU/I/O/checkpoint progress",
            "heartbeat_cadence_schema_writer_and_atomicity": {
                "cadence_seconds": float(heartbeat_interval_seconds),
                "schema_version": durability.SCHEMA_VERSION,
                "writer": "durability.HeartbeatSupervisor",
                "snapshot_atomic": True,
                "history_fsync": True,
            },
            "heartbeat_advancement_smoke_evidence": "tests/test_track_a_durability_integration.py::test_controlled_interruption_resume_equals_uninterrupted_reference",
            "opaque_phase_supervisor_sampling_rule": "sample wrapper and complete descendant process tree CPU/RSS/I/O separately; stall uses the descendant process tree",
            "raw_output_contract": "one immutable CSV collection set and dataset manifest per unit attempt",
            "aggregate_script_and_inputs": "sc_shil_experiment._aggregate_validated_units reads only hash-validated completed checkpoints",
            "plot_script_and_inputs": {
                "script": "src/freeze_bindings.py",
                "writer": "freeze_bindings.write_svg_bar_plot",
                "validator": "freeze_bindings.validate_plot_artifact",
                "input": "analysis/method_summary.csv",
                "output": "analysis/figures/method_summary.svg",
                "manifest": "analysis/figures/method_summary.plot.json",
                "input_policy": "saved postprocessed CSV only; no model execution",
            },
            "decision_artifact_path_and_criterion_schema": {
                "path": "analysis/decision_artifact.json",
                "writer": "freeze_bindings.write_decision_artifact",
                "validator": "freeze_bindings.validate_decision_artifact",
                "required_criterion_fields": [
                    "criterion_id",
                    "metric",
                    "value",
                    "operator",
                    "threshold",
                    "result",
                    "discriminator_statistics",
                ],
                "input_policy": "hash-bound saved analysis artifacts only",
            },
            "decision_discriminator_statistics_persisted": True,
            "notification_lifecycle": {
                "outbox_path": "notifications/invocation-{NNN}.jsonl",
                "writer": "freeze_bindings.append_notification_event",
                "validator": "freeze_bindings.validate_notification_lifecycle",
                "events": [
                    "STARTED",
                    "MILESTONE",
                    "COMPLETED",
                    "FAILED",
                    "TIMED_OUT",
                    "CANCELLED",
                ],
                "delivery_policy": "local durable outbox only; external delivery is a separate authorized action",
            },
            "terminal_status_paths": "terminal_status.json snapshot plus terminal_status_history.jsonl",
            "predecessor_terminal_status_poll_rule": {
                "poller": "freeze_bindings.require_predecessor_completed",
                "status_path": "<predecessor-output>/terminal_status.json",
                "success_status": "completed",
                "failure_statuses": ["failed", "timed_out", "cancelled"],
                "unknown_status": "unknown_timeout",
                "bounded_timeout_required": True,
            },
            "partial_result_promotion_policy": "prohibited",
        },
    }
    if scientific_freeze_binding is not None:
        manifest["durability"]["scientific_run_freeze"] = scientific_freeze_binding
        manifest["durability"]["max_attempts_per_unit"] = int(
            config["max_attempts_per_unit"]
        )
        if config.get("execution_class") == "scientific_pilot_a":
            manifest["durability"]["timing_resource_preflight_binding"] = dict(
                config["pilot_a_contract"]["timing_resource_preflight"]
            )
            manifest["durability"]["reentry_rule"] = str(
                config["pilot_a_contract"]["reentry_rule"]
            )
            manifest["durability"]["blindness"] = dict(
                config["pilot_a_contract"]["blindness"]
            )
        else:
            if scientific_freeze is None:
                raise RuntimeError(
                    "Full scientific freeze vanished during manifest build"
                )
            freeze_bindings_block = scientific_freeze["bindings"]
            manifest["durability"]["timing_resource_preflight_binding"] = dict(
                freeze_bindings_block["timing_resource_preflight"]
            )
            manifest["durability"]["reentry_rule"] = scientific_freeze["durability"][
                "reentry_rule"
            ]
            manifest["durability"]["blindness"] = dict(scientific_freeze["blindness"])
        manifest["durability"]["scientific_checkpoint_bindings"] = {
            "structured_registry_sha256": binding.structured_registry_file_sha256,
            "structured_registry_canonical_sha256": (
                binding.structured_registry.canonical_sha256
            ),
            "scientific_freeze_sha256": scientific_freeze_binding["sha256"],
            "run_role": str(config["execution_class"]),
        }
    durability.validate_run_manifest(
        manifest,
        candidate_root=candidate_root,
        project_root=project_root,
        require_execution_disabled=False,
    )
    return manifest


def _checkpoint_paths(output: Path, unit_id: str) -> list[Path]:
    return sorted((output / "units" / unit_id / "attempts").glob("*/checkpoint.json"))


def _charged_attempt_count(checkpoint_paths: Sequence[Path]) -> int:
    charged = 0
    for path in checkpoint_paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            charged += 1
            continue
        if not isinstance(document, dict) or document.get("status") != "cancelled":
            charged += 1
    return charged


def _write_terminal_status(
    output: Path, *, run_id: str, status: str, exit_code: int | None, error: str | None
) -> None:
    document = {
        "schema_version": durability.SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "error": error,
        "timestamp_utc": durability.utc_now(),
    }
    durability.append_jsonl_durable(output / "terminal_status_history.jsonl", document)
    durability.atomic_write_json(output / "terminal_status.json", document)


def _next_notification_outbox(output: Path) -> Path:
    root = output / "notifications"
    existing = sorted(root.glob("invocation-*.jsonl"))
    return root / f"invocation-{len(existing) + 1:03d}.jsonl"


def _notification_outbox_bindings(run_root: Path) -> list[dict[str, str]]:
    files = sorted((run_root / "notifications").glob("invocation-*.jsonl"))
    return [
        {
            "path": path.relative_to(run_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def _validate_predecessor_terminal_closure(
    terminal_status_path: Path, *, expected_run_id: str
) -> dict[str, Any]:
    run_root = terminal_status_path.resolve().parent
    closure_path = run_root / "terminal_closure.json"
    output_hashes_path = run_root / "output_hashes.json"
    manifest_path = run_root / "RUN_MANIFEST.json"
    history_path = run_root / "terminal_status_history.jsonl"
    notification_files = sorted((run_root / "notifications").glob("invocation-*.jsonl"))
    if (
        not all(
            path.is_file()
            for path in (
                terminal_status_path,
                closure_path,
                output_hashes_path,
                manifest_path,
                history_path,
            )
        )
        or not notification_files
    ):
        raise RuntimeError("Predecessor terminal closure is incomplete")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if (
        closure.get("run_id") != expected_run_id
        or closure.get("status") != "closed_completed"
        or closure.get("terminal_status_sha256") != sha256_file(terminal_status_path)
        or closure.get("terminal_status_history_sha256") != sha256_file(history_path)
        or closure.get("output_hashes_sha256") != sha256_file(output_hashes_path)
        or closure.get("run_manifest_sha256") != sha256_file(manifest_path)
        or closure.get("notification_outboxes")
        != _notification_outbox_bindings(run_root)
    ):
        raise RuntimeError("Predecessor terminal closure binding mismatch")
    for path in notification_files:
        freeze_bindings.validate_notification_lifecycle(path)
    final_notifications = freeze_bindings.validate_notification_lifecycle(
        notification_files[-1]
    )
    if final_notifications[-1].get("event") != "COMPLETED":
        raise RuntimeError("Predecessor final notification is not COMPLETED")
    return closure


def _validate_scientific_watchdog_receipt(
    receipt_path: Path,
    *,
    candidate_root: Path,
    freeze: dict[str, Any],
    run_id: str,
    mode: str,
    output: Path,
    watchdog_seconds: int,
) -> dict[str, Any]:
    path = receipt_path.resolve()
    try:
        path.relative_to(candidate_root)
    except ValueError as error:
        raise RuntimeError(
            "Scientific watchdog receipt escapes candidate root"
        ) from error
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "run_id",
        "mode",
        "run_output",
        "watchdog_seconds",
        "kill_after_seconds",
        "created_at_utc",
        "nonce",
        "supervisor",
    }
    if (
        set(receipt) != required
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "WATCHDOG_ARMED"
        or receipt.get("run_id") != run_id
        or receipt.get("mode") != mode
        or Path(str(receipt.get("run_output", ""))).resolve() != output.resolve()
        or receipt.get("watchdog_seconds") != int(watchdog_seconds)
        or receipt.get("kill_after_seconds") != 120
        or not isinstance(receipt.get("nonce"), str)
        or len(receipt["nonce"]) != 32
    ):
        raise RuntimeError("Scientific watchdog receipt predicate failed")
    age_seconds = (
        time.time() - durability.parse_utc(receipt["created_at_utc"]).timestamp()
    )
    if age_seconds < -5 or age_seconds > 600:
        raise RuntimeError("Scientific watchdog receipt is stale")
    launch = freeze.get("launch")
    freeze_durability = freeze.get("durability")
    supervisor = receipt.get("supervisor")
    if (
        not isinstance(freeze_durability, dict)
        or freeze_durability.get("worker_containment") != PARALLEL_WORKER_CONTAINMENT
    ):
        raise RuntimeError("Scientific worker containment binding mismatch")
    if not isinstance(launch, dict) or supervisor != {
        "path": "ops/remote_full_pipeline_supervisor.sh",
        "sha256": launch.get("ops/remote_full_pipeline_supervisor.sh"),
    }:
        raise RuntimeError("Scientific watchdog supervisor binding mismatch")
    for relative, expected in launch.items():
        live = candidate_root / str(relative)
        if not live.is_file() or sha256_file(live) != str(expected).upper():
            raise RuntimeError(f"Scientific launcher binding drift: {relative}")
    required_timeout = f"{int(watchdog_seconds)}s"
    ancestry = []
    try:
        ancestry = psutil.Process().parents()
    except psutil.Error as error:
        raise RuntimeError("Cannot inspect scientific watchdog ancestry") from error
    active = False
    for process in ancestry:
        try:
            command = process.cmdline()
        except psutil.Error:
            continue
        if (
            command
            and Path(command[0]).name == "timeout"
            and required_timeout in command
            and "--kill-after=120s" in command
        ):
            active = True
            break
    if not active:
        raise RuntimeError(
            "Frozen whole-run watchdog is not active in process ancestry"
        )
    return receipt


def _execute_partition_attempt(
    *,
    output: Path,
    run_id: str,
    unit_id: str,
    attempt_id: str,
    atomic_unit: dict[str, Any],
    bindings: durability.BindingSet,
    dataset: Dataset,
    split_seed: int,
    config: dict[str, Any],
    binding: Any,
    n_pairs: int,
    completed_units: int,
    planned_units: int,
    heartbeat_interval_seconds: float,
    inject_failure: bool,
) -> Path:
    attempt_root = output / "units" / unit_id / "attempts" / attempt_id
    checkpoint_path = attempt_root / "checkpoint.json"
    started_at = durability.utc_now()
    started_perf = time.perf_counter()
    running = durability.make_checkpoint(
        run_id=run_id,
        unit_id=unit_id,
        attempt_id=attempt_id,
        status="running",
        atomic_unit=atomic_unit,
        bindings=bindings,
        started_at_utc=started_at,
        ended_at_utc=None,
        duration_seconds=None,
        pid=os.getpid(),
        worker_id=f"track-a-atomic-worker-{os.getpid()}",
        exit_code=None,
        signal=None,
        artifacts=[],
    )
    durability.atomic_write_json(checkpoint_path, running)
    try:
        scenario_timeouts = config.get("scenario_timeout_seconds", {})
        unit_timeout = float(
            scenario_timeouts.get(
                str(atomic_unit.get("scenario")),
                config.get("per_unit_timeout_seconds", 14400),
            )
        )
        scientific_full = str(config.get("execution_class", "")) in (
            FULL_SCIENTIFIC_RUN_SPECS
        )
        with _wall_clock_timeout(
            unit_timeout, require_posix=scientific_full
        ), durability.HeartbeatSupervisor(
            snapshot_path=attempt_root / "heartbeat.json",
            history_path=attempt_root / "heartbeat_history.jsonl",
            run_id=run_id,
            unit_id=unit_id,
            attempt_id=attempt_id,
            phase="track_a_partition",
            completed_atomic_units=completed_units,
            planned_atomic_units=planned_units,
            last_durable_checkpoint_at=None,
            interval_seconds=heartbeat_interval_seconds,
        ):
            if inject_failure:
                raise RuntimeError("CONTROLLED_FAILURE_INJECTION")
            if atomic_unit.get("mode") == "full-structured-s13":
                collections = run_structured_s13_partition(
                    dataset, int(split_seed), config, binding
                )
            else:
                collections = run_track_a_partition(
                    dataset, int(split_seed), config, binding, int(n_pairs)
                )
        partition_identity = {
            "run_id": run_id,
            "unit_id": unit_id,
            "attempt_id": attempt_id,
            "protocol_sha256": bindings.protocol_sha256,
            "config_sha256": bindings.config_sha256,
            "registry_file_sha256": bindings.registry_sha256,
            "source_bundle_sha256": bindings.source_bundle_sha256,
        }
        collections["partition_manifest"][0].update(partition_identity)
        raw_root = attempt_root / "raw"
        artifacts: list[dict[str, Any]] = []
        for name in TRACK_A_COLLECTIONS:
            target = raw_root / f"{name}.csv"
            durability.atomic_write_bytes(target, _csv_bytes(collections[name]))
            artifacts.append(
                durability.artifact_record(
                    output,
                    target.relative_to(output).as_posix(),
                    f"partition_collection:{name}",
                )
            )
        dataset_target = raw_root / "dataset_manifest.csv"
        durability.atomic_write_bytes(
            dataset_target, _csv_bytes([_dataset_manifest_row(dataset, split_seed)])
        )
        artifacts.append(
            durability.artifact_record(
                output,
                dataset_target.relative_to(output).as_posix(),
                "partition_dataset_manifest",
            )
        )
        completed = durability.make_checkpoint(
            run_id=run_id,
            unit_id=unit_id,
            attempt_id=attempt_id,
            status="completed",
            atomic_unit=atomic_unit,
            bindings=bindings,
            started_at_utc=started_at,
            ended_at_utc=durability.utc_now(),
            duration_seconds=time.perf_counter() - started_perf,
            pid=os.getpid(),
            worker_id=f"track-a-atomic-worker-{os.getpid()}",
            exit_code=0,
            signal=None,
            artifacts=artifacts,
        )
        durability.atomic_write_json(checkpoint_path, completed)
        return checkpoint_path
    except BaseException as error:
        failed = durability.make_checkpoint(
            run_id=run_id,
            unit_id=unit_id,
            attempt_id=attempt_id,
            status="failed",
            atomic_unit=atomic_unit,
            bindings=bindings,
            started_at_utc=started_at,
            ended_at_utc=durability.utc_now(),
            duration_seconds=time.perf_counter() - started_perf,
            pid=os.getpid(),
            worker_id=f"track-a-atomic-worker-{os.getpid()}",
            exit_code=(
                124
                if isinstance(error, TimeoutError)
                else 97 if "CONTROLLED_FAILURE_INJECTION" in str(error) else 1
            ),
            signal=(
                "TIMEOUT"
                if isinstance(error, TimeoutError)
                else (
                    "FAILURE_INJECTION"
                    if "CONTROLLED_FAILURE_INJECTION" in str(error)
                    else None
                )
            ),
            artifacts=[],
            error=f"{type(error).__name__}: {error}",
        )
        durability.atomic_write_json(checkpoint_path, failed)
        raise


def _max_parallel_units(
    config: dict[str, Any], *, require_runtime_contract: bool = False
) -> int:
    value = config.get("max_parallel_units", 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("max_parallel_units must be an integer")
    if not 1 <= value <= MAX_PARALLEL_ATOMIC_UNITS:
        raise RuntimeError(
            f"max_parallel_units must be in [1,{MAX_PARALLEL_ATOMIC_UNITS}]"
        )
    if value > 1 and require_runtime_contract and not sys.platform.startswith("linux"):
        raise RuntimeError(
            "parallel atomic execution requires Linux worker containment"
        )
    if require_runtime_contract and "max_parallel_units" in config:
        required_thread_limits = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        drift = {
            name: os.environ.get(name)
            for name in required_thread_limits
            if os.environ.get(name) != "1"
        }
        if drift:
            raise RuntimeError(
                "parallel atomic execution requires exact one-thread limits: "
                + json.dumps(drift, sort_keys=True)
            )
    return value


def _validate_live_resource_gate(
    preflight_document: dict[str, Any],
    *,
    candidate_root: Path,
    configured_workers: int,
) -> dict[str, int]:
    timing_summary = preflight_document.get("timing_summary")
    resources = preflight_document.get("resources")
    if not isinstance(timing_summary, dict) or not isinstance(resources, dict):
        raise RuntimeError("Full scientific resource preflight is malformed")
    preflight_workers = int(timing_summary.get("max_parallel_units", 1))
    if preflight_workers != configured_workers:
        raise RuntimeError("Full scientific preflight worker-count drift")
    live_ram_available = int(psutil.virtual_memory().available)
    live_disk_free = int(shutil.disk_usage(candidate_root.resolve()).free)
    required_ram = int(resources["required_ram_bytes"])
    required_disk = int(resources["required_disk_bytes"])
    observed_cpu_count = int(resources["observed_cpu_count"])
    if live_ram_available < required_ram:
        raise RuntimeError("Live available RAM is below the frozen requirement")
    if live_disk_free < required_disk:
        raise RuntimeError("Live disk is below the frozen requirement")
    if observed_cpu_count < configured_workers:
        raise RuntimeError("Frozen resource host has fewer CPUs than workers")
    return {
        "configured_workers": configured_workers,
        "live_ram_available_bytes": live_ram_available,
        "live_disk_free_bytes": live_disk_free,
        "required_ram_bytes": required_ram,
        "required_disk_bytes": required_disk,
        "observed_cpu_count": observed_cpu_count,
    }


def _partition_process_entry(kwargs: dict[str, Any]) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("parallel atomic worker requires Linux")
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "cannot arm Linux parent-death signal")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)
    _execute_partition_attempt(**kwargs)


def _stop_partition_processes(
    processes: Iterable[multiprocessing.Process],
) -> None:
    active = [process for process in processes if process.is_alive()]
    for process in active:
        try:
            process.terminate()
        except (OSError, ProcessLookupError, ValueError):
            pass
    deadline = time.monotonic() + 10.0
    for process in active:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in active:
        if not process.is_alive():
            continue
        try:
            process.kill()
        except (OSError, ProcessLookupError, ValueError):
            pass
        process.join(2.0)


def _mark_partition_cancelled(
    process: multiprocessing.Process,
    task: dict[str, Any],
    *,
    reason: str,
) -> None:
    checkpoint_path = Path(str(task["checkpoint_path"]))
    existing: dict[str, Any] = {}
    try:
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            existing = candidate
    except (OSError, json.JSONDecodeError):
        pass
    if existing.get("status") in durability.TERMINAL_UNIT_STATUSES:
        return
    kwargs = task["kwargs"]
    ended_at = durability.utc_now()
    started_at = existing.get("started_at_utc", ended_at)
    try:
        duration_seconds = max(
            0.0,
            (
                durability.parse_utc(ended_at) - durability.parse_utc(str(started_at))
            ).total_seconds(),
        )
    except (durability.DurabilityError, ValueError):
        started_at = ended_at
        duration_seconds = 0.0
    process_record = existing.get("process")
    if not isinstance(process_record, dict):
        process_record = {}
    pid = int(process_record.get("pid") or process.pid or 0)
    worker_id = str(process_record.get("worker_id") or f"track-a-atomic-worker-{pid}")
    cancelled = durability.make_checkpoint(
        run_id=str(kwargs["run_id"]),
        unit_id=str(task["unit_id"]),
        attempt_id=str(task["attempt_id"]),
        status="cancelled",
        atomic_unit=dict(kwargs["atomic_unit"]),
        bindings=kwargs["bindings"],
        started_at_utc=str(started_at),
        ended_at_utc=ended_at,
        duration_seconds=duration_seconds,
        pid=pid,
        worker_id=worker_id,
        exit_code=143,
        signal="FAIL_FAST_SIBLING_CANCELLED",
        artifacts=[],
        error=reason,
    )
    durability.atomic_write_json(checkpoint_path, cancelled)


def _cancel_active_partitions(
    active: Iterable[tuple[multiprocessing.Process, dict[str, Any]]],
    *,
    reason: str,
) -> None:
    records = list(active)
    _stop_partition_processes(process for process, _ in records)
    for process, task in records:
        _mark_partition_cancelled(process, task, reason=reason)


def _parallel_partition_error(task: dict[str, Any], exit_code: int) -> BaseException:
    checkpoint_path = Path(str(task["checkpoint_path"]))
    failure: dict[str, Any] = {}
    try:
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            failure = candidate
    except (OSError, json.JSONDecodeError):
        pass
    unit_id = str(task["unit_id"])
    exit_record = failure.get("exit")
    if not isinstance(exit_record, dict):
        exit_record = {}
    if exit_record.get("signal") == "TIMEOUT" or exit_record.get("code") == 124:
        return TimeoutError(f"Parallel atomic unit {unit_id} timed out")
    detail = str(failure.get("error", "worker checkpoint unavailable"))
    return RuntimeError(
        f"Parallel atomic unit {unit_id} failed with process exit "
        f"{exit_code}: {detail}"
    )


def _iter_sequential_partition_events(
    tasks: Sequence[dict[str, Any]], *, initially_completed_units: int
) -> Iterable[tuple[str, dict[str, Any], Path | None]]:
    completed_in_scheduler = 0
    for task in tasks:
        yield "started", task, None
        kwargs = dict(task["kwargs"])
        kwargs["completed_units"] = initially_completed_units + completed_in_scheduler
        checkpoint = _execute_partition_attempt(**kwargs)
        completed_in_scheduler += 1
        yield "completed", task, checkpoint


def _iter_parallel_partition_events(
    tasks: Sequence[dict[str, Any]],
    *,
    max_parallel_units: int,
    initially_completed_units: int,
) -> Iterable[tuple[str, dict[str, Any], Path | None]]:
    if os.name != "posix" or max_parallel_units <= 1:
        raise RuntimeError("parallel partition scheduler requires POSIX and >1 worker")
    context = multiprocessing.get_context("fork")
    pending = iter(tasks)
    active: dict[int, tuple[multiprocessing.Process, dict[str, Any]]] = {}
    exhausted = False
    completed_in_scheduler = 0
    try:
        while active or not exhausted:
            while len(active) < max_parallel_units and not exhausted:
                try:
                    task = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                kwargs = dict(task["kwargs"])
                kwargs["completed_units"] = (
                    initially_completed_units + completed_in_scheduler
                )
                process = context.Process(
                    target=_partition_process_entry,
                    args=(kwargs,),
                    name=f"track-a-{task['unit_id']}",
                )
                process.start()
                if process.pid is None:
                    _stop_partition_processes([process])
                    raise RuntimeError(
                        f"Parallel atomic unit {task['unit_id']} did not start"
                    )
                active[process.pid] = (process, task)
                yield "started", task, None
            if not active:
                continue
            finished_pid: int | None = None
            for pid, (process, _) in active.items():
                process.join(0)
                if process.exitcode is not None:
                    finished_pid = pid
                    break
            if finished_pid is None:
                time.sleep(0.2)
                continue
            process, task = active.pop(finished_pid)
            exit_code = int(process.exitcode or 0)
            if exit_code != 0:
                _cancel_active_partitions(
                    active.values(),
                    reason=(
                        f"Sibling {task['unit_id']} failed; active unit cancelled "
                        "without consuming its retry budget"
                    ),
                )
                active.clear()
                raise _parallel_partition_error(task, exit_code)
            checkpoint_path = Path(str(task["checkpoint_path"]))
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                _cancel_active_partitions(
                    active.values(),
                    reason=(
                        f"Sibling {task['unit_id']} exited without a readable "
                        "checkpoint"
                    ),
                )
                active.clear()
                raise RuntimeError(
                    f"Parallel atomic unit {task['unit_id']} exited zero without "
                    "a readable checkpoint"
                ) from error
            if checkpoint.get("status") != "completed":
                _cancel_active_partitions(
                    active.values(),
                    reason=(
                        f"Sibling {task['unit_id']} exited without a completed "
                        "checkpoint"
                    ),
                )
                active.clear()
                raise RuntimeError(
                    f"Parallel atomic unit {task['unit_id']} exited zero without "
                    "a completed checkpoint"
                )
            completed_in_scheduler += 1
            yield "completed", task, checkpoint_path
    finally:
        _cancel_active_partitions(
            active.values(),
            reason="Parent scheduler stopped before this atomic unit completed",
        )


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return []


def _load_validated_partition(
    output: Path,
    checkpoint_path: Path,
    *,
    run_id: str,
    unit_id: str,
    atomic_unit: dict[str, Any],
    bindings: durability.BindingSet,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    checkpoint = durability.validate_checkpoint(
        checkpoint_path,
        run_root=output,
        expected_run_id=run_id,
        expected_unit_id=unit_id,
        expected_atomic_unit=atomic_unit,
        expected_bindings=bindings,
    )
    by_kind = {
        record["kind"]: output / record["path"] for record in checkpoint["artifacts"]
    }
    collections = {
        name: _read_csv_rows(by_kind[f"partition_collection:{name}"])
        for name in TRACK_A_COLLECTIONS
    }
    dataset_rows = _read_csv_rows(by_kind["partition_dataset_manifest"])
    if len(dataset_rows) != 1:
        raise RuntimeError("validated partition must contain one dataset manifest row")
    return collections, dataset_rows[0]


def _aggregate_validated_units(
    output: Path,
    *,
    run_id: str,
    planned_units: Sequence[dict[str, Any]],
    validated_checkpoints: dict[str, Path],
    bindings: durability.BindingSet,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    collections = {name: [] for name in TRACK_A_COLLECTIONS}
    dataset_rows: list[dict[str, Any]] = []
    for unit in planned_units:
        unit_id = str(unit["unit_id"])
        partition, dataset_row = _load_validated_partition(
            output,
            validated_checkpoints[unit_id],
            run_id=run_id,
            unit_id=unit_id,
            atomic_unit=dict(unit["atomic_unit"]),
            bindings=bindings,
        )
        for name in TRACK_A_COLLECTIONS:
            collections[name].extend(partition[name])
        dataset_rows.append(dataset_row)
    for name, rows in collections.items():
        durability.atomic_write_bytes(output / f"{name}.csv", _csv_bytes(rows))
    durability.atomic_write_bytes(
        output / "dataset_manifest.csv", _csv_bytes(dataset_rows)
    )
    return collections, dataset_rows


def run(
    config_path: Path,
    output: Path,
    mode: str,
    *,
    resume: bool = False,
    heartbeat_interval_seconds: float | None = None,
    failure_inject_before_unit_index: int | None = None,
    preflight_path: Path | None = None,
    predecessor_status_path: Path | None = None,
    predecessor_run_id: str | None = None,
    predecessor_timeout_seconds: float = 3600.0,
    predecessor_poll_interval_seconds: float = 60.0,
    watchdog_receipt_path: Path | None = None,
) -> int:
    from track_a_config import load_track_a_config

    binding = load_track_a_config(config_path)
    config = json.loads(binding.config_path.read_text(encoding="utf-8"))
    if binding.execution_enabled is not True:
        raise RuntimeError(
            "Candidate execution is disabled until the remaining pilot gates are closed"
        )
    is_pilot_a = config.get("execution_class") == "scientific_pilot_a"
    full_spec = FULL_SCIENTIFIC_RUN_SPECS.get(str(config.get("execution_class", "")))
    if is_pilot_a:
        if mode != "pilot":
            raise RuntimeError("Pilot A config authorizes only --mode pilot")
        frozen_heartbeat = float(config.get("heartbeat_interval_seconds", -1))
        if frozen_heartbeat != 2.0:
            raise RuntimeError("Pilot A heartbeat interval must be frozen at 2 seconds")
        if heartbeat_interval_seconds is None:
            heartbeat_interval_seconds = frozen_heartbeat
        elif float(heartbeat_interval_seconds) != frozen_heartbeat:
            raise RuntimeError("CLI heartbeat interval differs from Pilot A freeze")
        expected_preflight = (
            binding.config_path.parent.parent / PILOT_A_PREFLIGHT_RELATIVE_PATH
        ).resolve()
        if preflight_path is None or preflight_path.resolve() != expected_preflight:
            raise RuntimeError("Pilot A requires the exact frozen timing preflight")
        if sha256_file(expected_preflight) != PILOT_A_PREFLIGHT_SHA256:
            raise RuntimeError("Pilot A timing preflight hash mismatch")
        freeze_bindings.validate_timing_resource_preflight(expected_preflight)
    elif full_spec is not None:
        if mode != full_spec["mode"]:
            raise RuntimeError("Full scientific config authorizes one exact mode")
        frozen_heartbeat = float(config.get("heartbeat_interval_seconds", -1))
        if frozen_heartbeat != 10.0:
            raise RuntimeError("Full scientific heartbeat interval must be 10 seconds")
        if heartbeat_interval_seconds is None:
            heartbeat_interval_seconds = frozen_heartbeat
        elif float(heartbeat_interval_seconds) != frozen_heartbeat:
            raise RuntimeError("CLI heartbeat interval differs from full freeze")
        authorization = config["scientific_run_authorization"]
        freeze_path = (
            binding.config_path.parent.parent / authorization["freeze_path"]
        ).resolve()
        if not freeze_path.is_file():
            raise RuntimeError("Full scientific freeze is missing")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        preflight_record = freeze.get("bindings", {}).get("timing_resource_preflight")
        if not isinstance(preflight_record, dict):
            raise RuntimeError("Full scientific preflight binding is missing")
        expected_preflight = (
            binding.config_path.parent.parent / str(preflight_record.get("path", ""))
        ).resolve()
        if preflight_path is None or preflight_path.resolve() != expected_preflight:
            raise RuntimeError("Full scientific run requires its exact preflight")
        if (
            sha256_file(expected_preflight)
            != str(preflight_record.get("sha256", "")).upper()
        ):
            raise RuntimeError("Full scientific timing preflight hash mismatch")
        preflight_document = freeze_bindings.validate_timing_resource_preflight(
            expected_preflight
        )
        configured_workers = _max_parallel_units(config)
        _validate_live_resource_gate(
            preflight_document,
            candidate_root=binding.config_path.parent.parent,
            configured_workers=configured_workers,
        )
        if watchdog_receipt_path is None:
            raise RuntimeError("Full scientific run requires a watchdog receipt")
        _validate_scientific_watchdog_receipt(
            watchdog_receipt_path,
            candidate_root=binding.config_path.parent.parent.resolve(),
            freeze=freeze,
            run_id=str(freeze["authorized_run_id"]),
            mode=mode,
            output=output,
            watchdog_seconds=int(config["whole_run_watchdog_seconds"]),
        )
    else:
        if heartbeat_interval_seconds is None:
            heartbeat_interval_seconds = 60.0
        if preflight_path is not None:
            freeze_bindings.validate_timing_resource_preflight(preflight_path)
    if (predecessor_status_path is None) != (predecessor_run_id is None):
        raise RuntimeError(
            "predecessor status path and predecessor run ID must be supplied together"
        )
    if predecessor_status_path is not None:
        freeze_bindings.require_predecessor_completed(
            predecessor_status_path,
            expected_run_id=str(predecessor_run_id),
            timeout_seconds=predecessor_timeout_seconds,
            poll_interval_seconds=predecessor_poll_interval_seconds,
        )
        if full_spec is not None:
            _validate_predecessor_terminal_closure(
                predecessor_status_path,
                expected_run_id=str(predecessor_run_id),
            )
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    if resume and not output.is_dir():
        raise FileNotFoundError("resume requires an existing run output directory")
    output.mkdir(parents=True, exist_ok=True)
    expected_protocol = str(config["protocol_sha256"]).upper()
    project_root = Path(__file__).resolve().parents[3]
    protocol_path = _resolve_protocol_path(project_root, config)
    actual_protocol = sha256_file(protocol_path)
    if actual_protocol != expected_protocol:
        raise RuntimeError(
            f"Protocol hash mismatch: expected {expected_protocol}, observed {actual_protocol}"
        )
    datasets, split_seeds, n_pairs = resolve_run_plan(mode, config)
    candidate_root = binding.config_path.parent.parent.resolve()
    manifest = _build_run_manifest(
        config=config,
        binding=binding,
        candidate_root=candidate_root,
        project_root=project_root,
        protocol_path=protocol_path,
        mode=mode,
        datasets=datasets,
        split_seeds=split_seeds,
        n_pairs=n_pairs,
        heartbeat_interval_seconds=float(heartbeat_interval_seconds),
    )
    manifest_path = output / "RUN_MANIFEST.json"
    if resume:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError("resume manifest differs from the live bound run plan")
    else:
        durability.atomic_write_json(manifest_path, manifest)
        write_json(output / "environment.json", environment_payload())
        write_json(output / "config_resolved.json", config)
        durability.atomic_write_bytes(
            output / "method_registry_resolved.json", binding.registry_path.read_bytes()
        )
        durability.atomic_write_bytes(
            output / "structured_comparator_registry_resolved.json",
            binding.structured_registry_path.read_bytes(),
        )
    bindings = durability.validate_run_manifest(
        manifest,
        candidate_root=candidate_root,
        project_root=project_root,
        require_execution_disabled=False,
    )
    run_id = str(manifest["run_id"])
    run_identity_path = output / "run_identity.json"
    if resume:
        existing_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
        expected_identity_fields = {
            "run_id": run_id,
            "experiment_id": config["experiment_id"],
            "mode": mode,
            "protocol_sha256": actual_protocol,
            **bindings.as_dict(),
            "registry_canonical_sha256": binding.registry.canonical_sha256,
            "structured_registry_file_sha256": binding.structured_registry_file_sha256,
            "structured_registry_canonical_sha256": binding.structured_registry.canonical_sha256,
            "scientific_run_freeze": manifest["durability"].get(
                "scientific_run_freeze"
            ),
            "timing_resource_preflight_binding": manifest["durability"].get(
                "timing_resource_preflight_binding"
            ),
            "blindness": manifest["durability"].get("blindness"),
        }
        for key, expected in expected_identity_fields.items():
            if existing_identity.get(key) != expected:
                raise RuntimeError(f"resume run identity mismatch for {key}")
    else:
        write_json(
            run_identity_path,
            {
                "run_id": run_id,
                "experiment_id": config["experiment_id"],
                "mode": mode,
                "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "protocol_path": str(protocol_path),
                "protocol_sha256": actual_protocol,
                "config_path": str(config_path.resolve()),
                **bindings.as_dict(),
                "source_inventory": manifest["bindings"]["source"],
                "registry_path": str(binding.registry_path),
                "registry_canonical_sha256": binding.registry.canonical_sha256,
                "structured_registry_path": str(binding.structured_registry_path),
                "structured_registry_file_sha256": binding.structured_registry_file_sha256,
                "structured_registry_canonical_sha256": binding.structured_registry.canonical_sha256,
                "runner": "track_a_runner.run_track_a_cell",
                "legacy_run_dataset_bypassed": True,
                "scientific_run_freeze": manifest["durability"].get(
                    "scientific_run_freeze"
                ),
                "timing_resource_preflight_binding": manifest["durability"].get(
                    "timing_resource_preflight_binding"
                ),
                "blindness": manifest["durability"].get("blindness"),
            },
        )
    progress_path = output / "progress.jsonl"
    started = time.perf_counter()
    total = len(datasets)
    planned_units = list(manifest["planned_units"])
    checkpoint_map = {
        str(unit["unit_id"]): _checkpoint_paths(output, str(unit["unit_id"]))
        for unit in planned_units
    }
    resume_plan = durability.build_resume_plan(
        run_root=output,
        run_id=run_id,
        planned_units=planned_units,
        checkpoint_paths_by_unit=checkpoint_map,
        bindings=bindings,
    )
    validated_checkpoints: dict[str, Path] = {}
    completed_count = 0
    _write_terminal_status(
        output, run_id=run_id, status="running", exit_code=None, error=None
    )
    notification_outbox = _next_notification_outbox(output)
    try:
        freeze_bindings.append_notification_event(
            notification_outbox,
            run_id=run_id,
            event="STARTED",
            message=f"Track-A {mode} invocation started",
        )
        pending_tasks: list[dict[str, Any]] = []
        max_parallel_units = _max_parallel_units(config, require_runtime_contract=True)
        for index, (dataset, split_seed, unit, decision) in enumerate(
            zip(datasets, split_seeds, planned_units, resume_plan), start=1
        ):
            unit_id = str(unit["unit_id"])
            if decision.action == "skipped_validated":
                validated_checkpoints[unit_id] = Path(
                    str(decision.validated_checkpoint)
                )
                completed_count += 1
                append_progress(
                    progress_path,
                    {
                        "event": "unit_skipped_validated",
                        "unit_id": unit_id,
                        "completed": completed_count,
                        "total": total,
                        "timestamp_utc": durability.utc_now(),
                    },
                )
                freeze_bindings.append_notification_event(
                    notification_outbox,
                    run_id=run_id,
                    event="MILESTONE",
                    message=f"Validated atomic unit {unit_id} reused",
                    completed_units=completed_count,
                    planned_units=total,
                )
                continue
            max_attempts = int(config.get("max_attempts_per_unit", 1000000))
            charged_attempts = _charged_attempt_count(checkpoint_map[unit_id])
            if charged_attempts >= max_attempts:
                raise RuntimeError(
                    f"Maximum attempt count exceeded for {unit_id}: "
                    f"charged_attempts={charged_attempts}, limit={max_attempts}"
                )
            attempt_root = (
                output / "units" / unit_id / "attempts" / str(decision.next_attempt_id)
            )
            pending_tasks.append(
                {
                    "index": index,
                    "unit_id": unit_id,
                    "attempt_id": str(decision.next_attempt_id),
                    "checkpoint_path": attempt_root / "checkpoint.json",
                    "kwargs": {
                        "output": output,
                        "run_id": run_id,
                        "unit_id": unit_id,
                        "attempt_id": str(decision.next_attempt_id),
                        "atomic_unit": dict(unit["atomic_unit"]),
                        "bindings": bindings,
                        "dataset": dataset,
                        "split_seed": int(split_seed),
                        "config": config,
                        "binding": binding,
                        "n_pairs": int(n_pairs),
                        "completed_units": completed_count,
                        "planned_units": total,
                        "heartbeat_interval_seconds": float(heartbeat_interval_seconds),
                        "inject_failure": (failure_inject_before_unit_index == index),
                    },
                }
            )

        if max_parallel_units == 1:
            events: Iterable[tuple[str, dict[str, Any], Path | None]] = (
                _iter_sequential_partition_events(
                    pending_tasks, initially_completed_units=completed_count
                )
            )
        else:
            events = _iter_parallel_partition_events(
                pending_tasks,
                max_parallel_units=max_parallel_units,
                initially_completed_units=completed_count,
            )
        for event, task, checkpoint in events:
            if event == "started":
                append_progress(
                    progress_path,
                    {
                        "event": "unit_started",
                        "unit_id": task["unit_id"],
                        "attempt_id": task["attempt_id"],
                        "index": task["index"],
                        "total": total,
                        "timestamp_utc": durability.utc_now(),
                    },
                )
                continue
            if event != "completed" or checkpoint is None:
                raise RuntimeError("Unknown parallel partition scheduler event")
            unit_id = str(task["unit_id"])
            validated_checkpoints[unit_id] = checkpoint
            completed_count += 1
            append_progress(
                progress_path,
                {
                    "event": "unit_completed",
                    "unit_id": unit_id,
                    "attempt_id": task["attempt_id"],
                    "completed": completed_count,
                    "total": total,
                    "timestamp_utc": durability.utc_now(),
                },
            )
            freeze_bindings.append_notification_event(
                notification_outbox,
                run_id=run_id,
                event="MILESTONE",
                message=f"Atomic unit {unit_id} completed",
                completed_units=completed_count,
                planned_units=total,
            )
        collections, _ = _aggregate_validated_units(
            output,
            run_id=run_id,
            planned_units=planned_units,
            validated_checkpoints=validated_checkpoints,
            bindings=bindings,
        )
    except BaseException as error:
        event = (
            "TIMED_OUT"
            if isinstance(error, TimeoutError)
            else "CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        )
        freeze_bindings.append_notification_event(
            notification_outbox,
            run_id=run_id,
            event=event,
            message=f"{type(error).__name__}: {error}",
        )
        terminal_status = {
            "TIMED_OUT": "timed_out",
            "CANCELLED": "cancelled",
            "FAILED": "failed",
        }[event]
        exit_code = (
            124
            if event == "TIMED_OUT"
            else (
                130
                if event == "CANCELLED"
                else 97 if "CONTROLLED_FAILURE_INJECTION" in str(error) else 1
            )
        )
        _write_terminal_status(
            output,
            run_id=run_id,
            status=terminal_status,
            exit_code=exit_code,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    summary_cell_metadata = {
        "cell_type": "real" if mode == "full-real" else "synthetic",
        "candidate_orders": [int(order) for order in config["candidate_orders"]],
    }
    if mode == "full-structured-s13":
        summary_method_ids = ()
        summary_score_family_ids = ()
    else:
        summary_method_ids = tuple(
            binding.registry.expected_methods_for(summary_cell_metadata)
        )
        summary_score_family_ids = tuple(
            dict.fromkeys(
                binding.registry.method_by_id[method_id].score_family_id
                for method_id in summary_method_ids
            )
        )
    summary = {
        "experiment_id": config["experiment_id"],
        "mode": mode,
        "status": "completed",
        "runner": "track_a_runner.run_track_a_cell",
        "registry_sha256": binding.registry.canonical_sha256,
        "structured_registry_sha256": binding.structured_registry.canonical_sha256,
        "structured_registry_active": binding.structured_registry.registry_active,
        "durability_bindings": bindings.as_dict(),
        "expected_method_ids": list(summary_method_ids),
        "expected_score_family_ids": list(summary_score_family_ids),
        "dataset_runs": total,
        "method_rows": len(collections["metrics"]),
        "ranking_metric_rows": len(collections["ranking_metrics"]),
        "collection_row_counts": {
            name: len(rows) for name, rows in collections.items()
        },
        "block_role": str(config.get("execution_class", "")),
        "primary_method_execution_performed": mode != "full-structured-s13",
        "reporting_label": (
            "DESCRIPTIVE_NO_INFERENCE"
            if mode in {"full-real", "full-structured-s13"}
            else "PRIMARY_CONFIRMATORY_INPUT_WITH_DESCRIPTIVE_SECONDARIES"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(output / "run_summary.json", summary)
    _write_terminal_status(
        output, run_id=run_id, status="completed", exit_code=0, error=None
    )
    freeze_bindings.append_notification_event(
        notification_outbox,
        run_id=run_id,
        event="COMPLETED",
        message=f"Track-A {mode} invocation completed",
    )
    write_json(
        output / "output_hashes.json",
        {
            path.name: sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "output_hashes.json"
        },
    )
    durability.atomic_write_json(
        output / "terminal_closure.json",
        {
            "schema_version": durability.SCHEMA_VERSION,
            "run_id": run_id,
            "status": "closed_completed",
            "terminal_status_sha256": sha256_file(output / "terminal_status.json"),
            "terminal_status_history_sha256": sha256_file(
                output / "terminal_status_history.jsonl"
            ),
            "output_hashes_sha256": sha256_file(output / "output_hashes.json"),
            "run_manifest_sha256": sha256_file(output / "RUN_MANIFEST.json"),
            "notification_outboxes": _notification_outbox_bindings(output),
            "timestamp_utc": durability.utc_now(),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "masked-timing",
            "smoke",
            "pilot",
            "full-synthetic",
            "full-real",
            "full-structured-s13",
        ],
        required=True,
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        help="validate a lead-captured timing/resource preflight before execution",
    )
    parser.add_argument("--predecessor-status", type=Path)
    parser.add_argument("--predecessor-run-id")
    parser.add_argument("--predecessor-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--predecessor-poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--watchdog-receipt", type=Path)
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=None,
        help="must equal the frozen Pilot A cadence (2 seconds)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from hash-validated completed atomic-unit checkpoints",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        args.config,
        args.output,
        args.mode,
        resume=args.resume,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        preflight_path=args.preflight,
        predecessor_status_path=args.predecessor_status,
        predecessor_run_id=args.predecessor_run_id,
        predecessor_timeout_seconds=args.predecessor_timeout_seconds,
        predecessor_poll_interval_seconds=args.predecessor_poll_interval_seconds,
        watchdog_receipt_path=args.watchdog_receipt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
