from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


COLLECTIONS = [
    "metrics",
    "selected_edges",
    "base_fit_diagnostics",
    "selection_frequencies",
    "calibration_path",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_predecessor() -> ModuleType:
    source = Path(__file__).with_name("predecessor_sc_shil_experiment.py")
    spec = importlib.util.spec_from_file_location("shil_predecessor", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load predecessor source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_predecessor()


def fit_fixed_c_ranking(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    half_idx: np.ndarray,
    seed: int,
    fixed_c: float,
) -> Any:
    started = time.perf_counter()
    design = np.c_[X, Z]
    labels = np.arange(int(y.max()) + 1)
    warning_text: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(
            **BASE._l1_parameters(float(fixed_c), int(seed), len(labels))
        )
        model.fit(design[half_idx], y[half_idx])
    warning_text.extend(str(item.message) for item in caught)
    edge_coef = np.asarray(model.coef_)[:, X.shape[1] :]
    scores = np.linalg.norm(edge_coef, axis=0)
    active = np.flatnonzero(scores > 1e-8)
    ranked = active[np.argsort(scores[active], kind="stable")[::-1]]
    return BASE.RankingResult(
        ranked_indices=ranked,
        active_indices=active,
        scores=scores,
        selected_c=float(fixed_c),
        warnings=warning_text,
        elapsed_seconds=time.perf_counter() - started,
    )


def stability_rankings(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    edges: Sequence[tuple[int, ...]],
    clip_value: float,
    q_grid: Sequence[int],
    n_pairs: int,
    seed: int,
    config: dict[str, Any],
    phase: str,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], float, int]:
    Z = BASE.interaction_matrix(X, edges, clip_value)
    halves = BASE.stratified_complementary_halves(y, n_pairs, seed)
    rankings: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    with BASE.PeakRSSMonitor() as memory:
        for half_index, half in enumerate(halves):
            half_idx = np.asarray(half, dtype=int)
            if method == "l1_fixed":
                result = fit_fixed_c_ranking(
                    X,
                    Z,
                    y,
                    half_idx,
                    seed + 3000 + half_index,
                    float(config["fixed_c"]),
                )
                fit_rows = len(half_idx)
                inner_val_rows = 0
            elif method == "shil":
                fit_idx, val_idx = BASE.inner_split(
                    half_idx, y, seed + 1000 + half_index
                )
                result = BASE.fit_shil_ranking(
                    X,
                    Z,
                    y,
                    fit_idx,
                    val_idx,
                    seed + 2000 + half_index,
                    config["shil"],
                )
                fit_rows = len(fit_idx)
                inner_val_rows = len(val_idx)
            else:
                raise ValueError(f"Unknown sensitivity method: {method}")
            rankings.append(result.ranked_indices)
            diagnostics.append(
                {
                    "phase": phase,
                    "base_method": method,
                    "half_index": half_index,
                    "half_rows": len(half_idx),
                    "fit_rows": fit_rows,
                    "inner_val_rows": inner_val_rows,
                    "active_edges": len(result.active_indices),
                    "selected_c": result.selected_c,
                    "fit_seconds": result.elapsed_seconds,
                    "warning_count": len(result.warnings),
                    "warnings": " | ".join(sorted(set(result.warnings))),
                }
            )
    return (
        BASE.frequencies_from_rankings(rankings, q_grid, len(edges)),
        diagnostics,
        time.perf_counter() - started,
        memory.peak_rss_bytes,
    )


def stable_support(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    edges: Sequence[tuple[int, ...]],
    config: dict[str, Any],
    n_pairs: int,
    seed: int,
) -> tuple[
    list[tuple[int, ...]],
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
        config,
        "tuning_train",
    )
    calibration = BASE.build_calibration_path(
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
    chosen = BASE.select_one_se(calibration)
    final_frequency, final_diag, final_seconds, final_peak = stability_rankings(
        method,
        X_dev,
        y_dev,
        edges,
        float(config["interaction_clip"]),
        [int(chosen["q"])],
        n_pairs,
        seed + 300,
        config,
        "final_development",
    )
    final_q = int(chosen["q"])
    final_indices = np.flatnonzero(
        final_frequency[final_q] >= float(chosen["pi"])
    )
    final_edges = [edges[int(index)] for index in final_indices]
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
                        "edge": BASE.edge_to_string(edges[edge_index]),
                        "frequency": float(value),
                    }
                )
    calibration_rows = [
        {
            **{key: value for key, value in row.items() if key != "support"},
            "support": ";".join(
                BASE.edge_to_string(edge) for edge in row["support"]
            ),
            "base_method": method,
            "chosen": bool(
                row["q"] == chosen["q"]
                and row["pi"] == chosen["pi"]
                and tuple(row["support"]) == tuple(chosen["support"])
            ),
        }
        for row in calibration
    ]
    diagnostics = tuning_diag + final_diag
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
        "base_fit_count": len(diagnostics),
    }
    return final_edges, metadata, diagnostics, frequency_rows, calibration_rows


def locked_method_order(config: dict[str, Any], split_seed: int) -> list[str]:
    seed_order = [
        int(value)
        for value in config.get(
            "base_seed_order",
            config.get("execution_partition", {}).get("base_seed_order", []),
        )
    ]
    if split_seed not in seed_order:
        return ["l1_fixed", "shil"]
    position = seed_order.index(split_seed)
    return ["l1_fixed", "shil"] if position % 2 == 0 else ["shil", "l1_fixed"]


def run_dataset(
    dataset: Any, split_seed: int, config: dict[str, Any], n_pairs: int
) -> dict[str, list[dict[str, Any]]]:
    split = BASE.outer_split_scale(dataset, split_seed)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    X_dev, y_dev = np.r_[X_train, X_val], np.r_[y_train, y_val]
    edges = BASE.candidate_edges(X_train.shape[1], config["candidate_orders"])
    outputs: dict[str, Any] = {}
    order = locked_method_order(config, split_seed)
    for method in order:
        seed_offset = 20000 if method == "l1_fixed" else 10000
        outputs[method] = stable_support(
            method,
            X_train,
            y_train,
            X_val,
            y_val,
            X_dev,
            y_dev,
            edges,
            config,
            n_pairs,
            split_seed + seed_offset,
        )

    model_names = {
        "l1_fixed": "SC-L1-fixed-C",
        "shil": "SC-SHIL-contemporary",
    }
    metrics: list[dict[str, Any]] = []
    selected_edges: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_frequencies: list[dict[str, Any]] = []
    all_calibration: list[dict[str, Any]] = []
    for method in ["l1_fixed", "shil"]:
        support, metadata, diagnostics, frequencies, calibration = outputs[method]
        warning_count = sum(int(row["warning_count"]) for row in diagnostics)
        selection_metadata = {
            **metadata,
            "selected_c": float(config["fixed_c"])
            if method == "l1_fixed"
            else None,
            "warning_count": warning_count,
            "execution_order": order.index(method) + 1,
            "fit_unit": "logistic_fit"
            if method == "l1_fixed"
            else "screening_optimization",
        }
        row, edge_rows = BASE.evaluate_support(
            dataset,
            model_names[method],
            support,
            X_dev,
            y_dev,
            X_test,
            y_test,
            config,
            split_seed,
            split_seed + 50000,
            selection_metadata,
        )
        metrics.append(row)
        selected_edges.extend(edge_rows)
        all_diagnostics.extend(diagnostics)
        all_frequencies.extend(frequencies)
        all_calibration.extend(calibration)

    for collection in (all_diagnostics, all_frequencies, all_calibration):
        for row in collection:
            row.update(
                {
                    "dataset": dataset.name,
                    "scenario": dataset.metadata.get("id", dataset.name),
                    "data_seed": int(
                        dataset.metadata.get("generator_seed", split_seed)
                    ),
                    "split_seed": int(split_seed),
                }
            )
    return {
        "metrics": metrics,
        "selected_edges": selected_edges,
        "base_fit_diagnostics": all_diagnostics,
        "selection_frequencies": all_frequencies,
        "calibration_path": all_calibration,
    }


def resolve_plan(
    mode: str, config: dict[str, Any]
) -> tuple[list[Any], list[int], int]:
    if mode == "smoke":
        spec = {**config["scenarios"][0], "n": 320, "d": 10}
        return (
            [BASE.make_synthetic(spec, 809)],
            [809],
            int(config["n_pairs_smoke"]),
        )
    if mode == "full-real":
        dataset = BASE.load_dry_bean(config["real_dataset"])
        seeds = [int(value) for value in config["real_split_seeds"]]
        return [dataset for _ in seeds], seeds, int(config["n_pairs_full"])
    raise ValueError(f"Unsupported mode: {mode}")


def validate_locked_inputs(config: dict[str, Any]) -> dict[str, str]:
    run_root = Path(__file__).resolve().parents[1]
    paths = {
        "protocol": run_root
        / "inputs"
        / "fixed_c_cost_sensitivity_protocol_20260727.md",
        "predecessor_source": Path(__file__).with_name(
            "predecessor_sc_shil_experiment.py"
        ),
        "historical_metrics": run_root / "inputs" / "historical_tuned_metrics.csv",
        "dry_bean_archive": run_root
        / "cache"
        / "uci_602"
        / "dry_bean_dataset.zip",
    }
    expected = {
        "protocol": str(config["protocol_sha256"]).upper(),
        "predecessor_source": str(config["predecessor_source_sha256"]).upper(),
        "historical_metrics": str(config["historical_metrics_sha256"]).upper(),
        "dry_bean_archive": str(config["real_dataset"]["archive_sha256"]).upper(),
    }
    observed = {key: sha256_file(path) for key, path in paths.items()}
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if mismatches:
        raise RuntimeError(f"Locked input hash mismatch: {mismatches}")
    if float(config["fixed_c"]) != 1.0:
        raise RuntimeError("Fixed-C protocol violation")
    return observed


def run(config_path: Path, output: Path, mode: str) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    locked_hashes = validate_locked_inputs(config)
    environment = {
        **BASE.environment_payload(),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "predecessor_source_sha256": locked_hashes["predecessor_source"],
    }
    write_json(output / "environment.json", environment)
    write_json(output / "config_resolved.json", config)
    write_json(
        output / "run_identity.json",
        {
            "experiment_id": config["experiment_id"],
            "methodology_change_id": "MCH-SHIL-002",
            "mode": mode,
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "protocol_sha256": locked_hashes["protocol"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "runner_source_sha256": sha256_file(Path(__file__)),
            "predecessor_source_sha256": locked_hashes["predecessor_source"],
            "historical_metrics_sha256": locked_hashes["historical_metrics"],
            "dry_bean_archive_sha256": locked_hashes["dry_bean_archive"],
        },
    )
    datasets, split_seeds, n_pairs = resolve_plan(mode, config)
    collections: dict[str, list[dict[str, Any]]] = {
        name: [] for name in COLLECTIONS
    }
    manifest_rows: list[dict[str, Any]] = []
    progress_path = output / "progress.jsonl"
    started = time.perf_counter()
    for index, (dataset, split_seed) in enumerate(
        zip(datasets, split_seeds), start=1
    ):
        BASE.append_progress(
            progress_path,
            {
                "event": "dataset_started",
                "index": index,
                "total": len(datasets),
                "dataset": dataset.name,
                "split_seed": int(split_seed),
                "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
        )
        result = run_dataset(dataset, int(split_seed), config, n_pairs)
        for name in COLLECTIONS:
            collections[name].extend(result[name])
        manifest_rows.append(
            {
                "dataset": dataset.name,
                "split_seed": int(split_seed),
                "rows": len(dataset.y),
                "features": dataset.X.shape[1],
                "classes": len(np.unique(dataset.y)),
                "metadata": json.dumps(
                    dataset.metadata, ensure_ascii=False, sort_keys=True
                ),
            }
        )
        BASE.append_progress(
            progress_path,
            {
                "event": "dataset_completed",
                "index": index,
                "total": len(datasets),
                "dataset": dataset.name,
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
        "dataset_runs": len(datasets),
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
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full-real"], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.config, args.output, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
