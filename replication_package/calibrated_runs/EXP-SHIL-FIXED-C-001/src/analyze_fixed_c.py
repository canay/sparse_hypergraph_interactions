from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_FIXED = "SC-L1-fixed-C"
METHOD_SHIL = "SC-SHIL-contemporary"
HISTORICAL_METHOD = "SC-L1"


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_and_validate(
    combined: Path, historical_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = json.loads((combined / "config_resolved.json").read_text("utf-8"))
    metrics = pd.read_csv(combined / "metrics.csv")
    historical = pd.read_csv(historical_path)
    seeds = [int(value) for value in config["base_seed_order"]]
    if len(metrics) != len(seeds) * 2:
        raise AssertionError("Unexpected contemporary metric-row count")
    if set(metrics["model"]) != {METHOD_FIXED, METHOD_SHIL}:
        raise AssertionError("Unexpected contemporary method set")
    if set(metrics["split_seed"].astype(int)) != set(seeds):
        raise AssertionError("Contemporary seed mismatch")
    if set(metrics["base_fit_count"].astype(int)) != {80}:
        raise AssertionError("Contemporary fit-count invariant failed")
    historical = historical[
        (historical["model"] == HISTORICAL_METHOD)
        & (historical["split_seed"].astype(int).isin(seeds))
    ].copy()
    if len(historical) != len(seeds):
        raise AssertionError("Historical tuned SC-L1 seed set is incomplete")
    if sha256_file(historical_path) != str(
        config["historical_metrics_sha256"]
    ).upper():
        raise AssertionError("Historical metric snapshot hash mismatch")
    return metrics, historical, config


def paired_frame(
    metrics: pd.DataFrame, historical: pd.DataFrame
) -> pd.DataFrame:
    values = metrics.pivot(
        index="split_seed",
        columns="model",
        values=[
            "selection_seconds",
            "selected_edges",
            "macro_f1",
            "log_loss",
            "accuracy",
            "chosen_q",
            "chosen_pi",
            "execution_order",
        ],
    )
    rows: list[dict[str, Any]] = []
    historical_by_seed = historical.set_index(
        historical["split_seed"].astype(int)
    )
    for seed in sorted(values.index.astype(int)):
        fixed_time = float(values.loc[seed, ("selection_seconds", METHOD_FIXED)])
        shil_time = float(values.loc[seed, ("selection_seconds", METHOD_SHIL)])
        tuned_time = float(historical_by_seed.loc[seed, "selection_seconds"])
        rows.append(
            {
                "split_seed": int(seed),
                "fixed_c_l1_seconds": fixed_time,
                "contemporary_sc_shil_seconds": shil_time,
                "fixed_c_l1_over_sc_shil": fixed_time / shil_time,
                "historical_tuned_l1_seconds": tuned_time,
                "historical_tuned_over_fixed_c_l1": tuned_time / fixed_time,
                "fixed_c_selected_edges": int(
                    values.loc[seed, ("selected_edges", METHOD_FIXED)]
                ),
                "sc_shil_selected_edges": int(
                    values.loc[seed, ("selected_edges", METHOD_SHIL)]
                ),
                "fixed_c_macro_f1": float(
                    values.loc[seed, ("macro_f1", METHOD_FIXED)]
                ),
                "sc_shil_macro_f1": float(
                    values.loc[seed, ("macro_f1", METHOD_SHIL)]
                ),
                "fixed_c_log_loss": float(
                    values.loc[seed, ("log_loss", METHOD_FIXED)]
                ),
                "sc_shil_log_loss": float(
                    values.loc[seed, ("log_loss", METHOD_SHIL)]
                ),
                "fixed_c_execution_order": int(
                    values.loc[seed, ("execution_order", METHOD_FIXED)]
                ),
                "sc_shil_execution_order": int(
                    values.loc[seed, ("execution_order", METHOD_SHIL)]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            n=("split_seed", "size"),
            selection_seconds_mean=("selection_seconds", "mean"),
            selection_seconds_sd=("selection_seconds", "std"),
            selected_edges_mean=("selected_edges", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_sd=("log_loss", "std"),
            warning_count_total=("warning_count", "sum"),
        )
        .sort_values("model")
    )
    ratios = paired["fixed_c_l1_over_sc_shil"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    indexes = rng.integers(
        0,
        len(ratios),
        size=(int(config["bootstrap_resamples"]), len(ratios)),
    )
    bootstrap_means = ratios[indexes].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    if ci_low > 1.0:
        decision = "RESIDUAL_FIXED_C_L1_SLOWER"
    elif ci_high < 1.0:
        decision = "FIXED_C_L1_FASTER"
    else:
        decision = "NO_RESIDUAL_SELECTION_TIME_SEPARATION"
    payload = {
        "experiment_id": config["experiment_id"],
        "status": "ANALYZED",
        "decision": decision,
        "protocol_only_method_verdict_unchanged": True,
        "paired_seed_count": len(ratios),
        "fixed_c": float(config["fixed_c"]),
        "bootstrap_resamples": int(config["bootstrap_resamples"]),
        "bootstrap_seed": int(config["bootstrap_seed"]),
        "bootstrap_statistic": "arithmetic mean of paired per-seed time ratios",
        "fixed_c_l1_over_sc_shil_mean": float(np.mean(ratios)),
        "fixed_c_l1_over_sc_shil_median": float(np.median(ratios)),
        "fixed_c_l1_over_sc_shil_ci95_low": float(ci_low),
        "fixed_c_l1_over_sc_shil_ci95_high": float(ci_high),
        "historical_tuned_over_fixed_c_l1_mean": float(
            paired["historical_tuned_over_fixed_c_l1"].mean()
        ),
        "macro_f1_mean_difference_fixed_minus_shil": float(
            paired["fixed_c_macro_f1"].mean()
            - paired["sc_shil_macro_f1"].mean()
        ),
        "log_loss_mean_difference_fixed_minus_shil": float(
            paired["fixed_c_log_loss"].mean()
            - paired["sc_shil_log_loss"].mean()
        ),
    }
    return summary, payload


def analyze(combined: Path, historical: Path, output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics, historical_metrics, config = load_and_validate(combined, historical)
    paired = paired_frame(metrics, historical_metrics)
    method_summary, analysis_summary = summarize(metrics, paired, config)
    paired.to_csv(output / "paired_timing_and_metrics.csv", index=False)
    method_summary.to_csv(output / "method_summary.csv", index=False)
    write_json(output / "analysis_summary.json", analysis_summary)
    write_json(
        output / "analysis_identity.json",
        {
            "combined_output": str(combined.resolve()),
            "combined_metrics_sha256": sha256_file(combined / "metrics.csv"),
            "historical_metrics": str(historical.resolve()),
            "historical_metrics_sha256": sha256_file(historical),
            "analysis_source_sha256": sha256_file(Path(__file__)),
            "protocol_sha256": str(config["protocol_sha256"]).upper(),
        },
    )
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
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return analyze(args.combined, args.historical, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
