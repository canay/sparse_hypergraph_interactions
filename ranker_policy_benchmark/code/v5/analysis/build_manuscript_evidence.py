"""Build manuscript-facing descriptive summaries from the frozen Track A v5 outputs.

Date/time: 2026-08-30 23:47:38 +03:00
Tool: Codex
Model, if known: GPT-5.6 Sol
Operation ID: shil-g09-r1-safe-revision-20260830-234612

This script performs no model fitting and no inferential calculation.  It reads
only the post-run freeze and emits descriptive, provenance-bearing artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


OPERATION_ID = "shil-g09-r1-safe-revision-20260830-234612"
RUN_TIMESTAMP = "2026-08-30 23:47:38 +03:00"
MODEL = "GPT-5.6 Sol"

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[3]
FREEZE = PROJECT / "experiments" / "2026-08-30_codex_local_track-a-v5-postrun-freeze"
PROCESSED = FREEZE / "processed_outputs"
OUT = SCRIPT.parents[1] / "outputs"

INPUTS = {
    "synthetic_analysis": PROCESSED / "synthetic" / "analysis_summary.json",
    "decomposition": PROCESSED / "synthetic" / "ranker_policy_decomposition.csv",
    "method_metrics": PROCESSED / "synthetic" / "metrics_with_run_stability.csv",
    "fixed_k": PROCESSED / "synthetic" / "fixed_k_matched_cardinality.csv",
    "validation_invariants": PROCESSED / "synthetic" / "validation_one_se_raw_invariants.csv",
    "cpss_references": PROCESSED / "synthetic" / "cpss_anchor_raw_exact_k_references.csv",
    "dry_bean": PROCESSED / "real_descriptive" / "dry_bean_predictive_summary.csv",
    "dry_bean_analysis": PROCESSED / "real_descriptive" / "dry_bean_analysis_summary.json",
    "s13": PROCESSED / "structured_s13" / "s13_structured_metrics_descriptive.csv",
    "s13_analysis": PROCESSED / "structured_s13" / "s13_structured_analysis_summary.json",
    "final_adjudication": FREEZE / "freeze" / "final_adjudication.json",
    "protocol": PROJECT / "MD" / "02_design" / "track_a_benchmark_protocol_20260826.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def mean_sd(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None, None, 0
    return fmean(clean), stdev(clean) if len(clean) > 1 else 0.0, len(clean)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def aggregate_method_rows(rows: list[dict[str, str]], scenarios: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["scenario"] in scenarios:
            grouped[row["method_id"]].append(row)

    output: list[dict[str, Any]] = []
    for method_id in sorted(grouped, key=lambda item: (item.split(".")[1], item.split(".")[0], item)):
        group = grouped[method_id]
        selected_mean, selected_sd, n = mean_sd(numeric(row["selected_edge_count"]) for row in group)
        f1_mean, f1_sd, _ = mean_sd(numeric(row["group_f1"]) for row in group)
        fi_mean, fi_sd, _ = mean_sd(numeric(row["group_false_inclusions"]) for row in group)
        loss_mean, loss_sd, _ = mean_sd(numeric(row["log_loss"]) for row in group)
        output.append(
            {
                "method_id": method_id,
                "ranker_id": group[0]["ranker_id"],
                "policy_id": group[0]["policy_id"],
                "runs": n,
                "selected_edge_count_mean": selected_mean,
                "selected_edge_count_sd": selected_sd,
                "group_f1_mean": f1_mean,
                "group_f1_sd": f1_sd,
                "group_false_inclusions_mean": fi_mean,
                "group_false_inclusions_sd": fi_sd,
                "log_loss_mean": loss_mean,
                "log_loss_sd": loss_sd,
                "budget_shortfall_rows": sum((numeric(row["budget_shortfall"]) or 0.0) > 0 for row in group),
                "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
            }
        )
    return output


def aggregate_null_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["scenario"] == "S00":
            grouped[row["method_id"]].append(row)

    output: list[dict[str, Any]] = []
    for method_id in sorted(grouped, key=lambda item: (item.split(".")[1], item.split(".")[0], item)):
        group = grouped[method_id]
        count_mean, count_sd, n = mean_sd(numeric(row["group_false_inclusions"]) for row in group)
        fdp_mean, fdp_sd, _ = mean_sd(numeric(row["group_false_discovery_proportion"]) for row in group)
        output.append(
            {
                "method_id": method_id,
                "ranker_id": group[0]["ranker_id"],
                "policy_id": group[0]["policy_id"],
                "runs": n,
                "false_interaction_count_mean": count_mean,
                "false_interaction_count_sd": count_sd,
                "false_discovery_proportion_mean": fdp_mean,
                "false_discovery_proportion_sd": fdp_sd,
                "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
            }
        )
    return output


def decomposition_summary(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    statuses = Counter(row["status"] for row in rows)
    eligible = [row for row in rows if row["status"] == "OK"]
    invalid = [row for row in rows if row["status"] != "OK"]
    r_values = [numeric(row["ranker_effect_R"]) for row in eligible]
    p_values = [numeric(row["policy_effect_P"]) for row in eligible]
    d_values = [numeric(row["difference_D"]) for row in eligible]
    r_mean, r_sd, _ = mean_sd(r_values)
    p_mean, p_sd, _ = mean_sd(p_values)
    d_mean, d_sd, _ = mean_sd(d_values)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in sorted({row["scenario"] for row in eligible}):
        group = [row for row in eligible if row["scenario"] == scenario]
        sr, _, n = mean_sd(numeric(row["ranker_effect_R"]) for row in group)
        sp, _, _ = mean_sd(numeric(row["policy_effect_P"]) for row in group)
        sd, _, _ = mean_sd(numeric(row["difference_D"]) for row in group)
        scenario_rows.append(
            {
                "scenario": scenario,
                "eligible_cells": n,
                "ranker_effect_R_mean": sr,
                "policy_effect_P_mean": sp,
                "difference_D_mean": sd,
                "D_exact_zero_cells": sum(numeric(row["difference_D"]) == 0.0 for row in group),
                "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
            }
        )

    return (
        {
            "planned_cells": len(rows),
            "eligible_cells": len(eligible),
            "invalid_cells": len(invalid),
            "status_counts": dict(statuses),
            "invalid_cell_inventory": [
                {"scenario": row["scenario"], "split_seed": int(row["split_seed"]), "status": row["status"]}
                for row in invalid
            ],
            "ranker_effect_R_mean": r_mean,
            "ranker_effect_R_sd": r_sd,
            "policy_effect_P_mean": p_mean,
            "policy_effect_P_sd": p_sd,
            "difference_D_mean": d_mean,
            "difference_D_sd": d_sd,
            "P_exact_zero_cells": sum(value == 0.0 for value in p_values),
            "D_exact_zero_cells": sum(value == 0.0 for value in d_values),
            "D_negative_cells": sum(value is not None and value < 0.0 for value in d_values),
            "D_positive_cells": sum(value is not None and value > 0.0 for value in d_values),
            "scenario_mean_D_exact_zero_count": sum(row["difference_D_mean"] == 0.0 for row in scenario_rows),
            "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
        },
        scenario_rows,
    )


def invariant_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["status"] == "OK"]
    invalid = [row for row in rows if row["status"] != "OK"]
    matches = [row for row in eligible if row["support_matches_raw_exact_k"].lower() == "true"]
    return {
        "ranker_cells": len(rows),
        "eligible_ranker_cells": len(eligible),
        "invalid_ranker_cells": len(invalid),
        "support_matches_raw_exact_k": len(matches),
        "all_eligible_match": len(matches) == len(eligible),
        "invalid_inventory": [
            {
                "scenario": row["scenario"],
                "split_seed": int(row["split_seed"]),
                "ranker_id": row["ranker_id"],
                "status": row["status"],
            }
            for row in invalid
        ],
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
    }


def cpss_reference_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    own = [row for row in rows if row["anchor_ranker_id"] == row["target_ranker_id"]]
    eligible = [row for row in own if row["status"] == "OK"]
    residuals = [numeric(row["own_anchor_absolute_cpss_residual"]) for row in eligible]
    return {
        "own_ranker_anchor_cells": len(own),
        "eligible_own_ranker_anchor_cells": len(eligible),
        "invalid_own_ranker_anchor_cells": len(own) - len(eligible),
        "zero_group_f1_residual_cells": sum(value == 0.0 for value in residuals),
        "nonzero_group_f1_residual_cells": sum(value is not None and value != 0.0 for value in residuals),
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
    }


def fixed_k_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["status"] == "OK"]
    differences = [numeric(row["shil_minus_l1"]) for row in eligible]
    abs_differences = [numeric(row["absolute_ranker_difference"]) for row in eligible]
    return {
        "grid_rows": len(rows),
        "eligible_grid_rows": len(eligible),
        "invalid_grid_rows": len(rows) - len(eligible),
        "exact_tie_rows": sum(value == 0.0 for value in differences),
        "nonzero_rows": sum(value is not None and value != 0.0 for value in differences),
        "maximum_absolute_group_f1_difference": max(value for value in abs_differences if value is not None),
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
    }


def dry_bean_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics = ("log_loss_mean", "macro_f1_mean", "accuracy_mean", "roc_auc_ovr_mean")
    spans: dict[str, Any] = {}
    for metric in metrics:
        ordered = sorted((numeric(row[metric]), row["method_id"]) for row in rows)
        spans[metric] = {
            "minimum": ordered[0][0],
            "minimum_method": ordered[0][1],
            "maximum": ordered[-1][0],
            "maximum_method": ordered[-1][1],
            "span": ordered[-1][0] - ordered[0][0],
        }
    return {
        "methods": len(rows),
        "runs_per_method": sorted({int(row["runs"]) for row in rows}),
        "metric_spans": spans,
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
    }


def s13_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    fields = (
        "selected_edge_count",
        "accuracy",
        "macro_f1",
        "log_loss",
        "support_precision",
        "support_recall",
        "support_f1",
        "false_inclusions",
    )
    summary: dict[str, Any] = {"runs": len(rows), "comparator_id": rows[0]["comparator_id"]}
    for field in fields:
        avg, sd, _ = mean_sd(numeric(row[field]) for row in rows)
        summary[f"{field}_mean"] = avg
        summary[f"{field}_sd"] = sd
    summary["analysis_role"] = "DESCRIPTIVE_NO_INFERENCE"
    summary["primary_estimand"] = False
    return summary


def build_tex_tables(policy_rows: list[dict[str, Any]], scenario_rows: list[dict[str, Any]], dry_rows: list[dict[str, str]], s13: dict[str, Any]) -> None:
    policy_lookup = {row["method_id"]: row for row in policy_rows}
    policies = [
        ("Fixed $K=1$", "fixed_k.k01"),
        ("Fixed $K=2$", "fixed_k.k02"),
        ("Fixed $K=4$", "fixed_k.k04"),
        ("Fixed $K=8$", "fixed_k.k08"),
        ("Fixed $K=16$", "fixed_k.k16"),
        ("Validation one-SE", "validation_one_se"),
        ("CPSS one-SE", "cpss_one_se"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Policy & \multicolumn{4}{c}{SHIL ranker} & \multicolumn{4}{c}{L1 ranker} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r" & Selected & Group F1 & False incl. & Log loss & Selected & Group F1 & False incl. & Log loss \\",
        r"\midrule",
    ]
    for label, policy in policies:
        shil = policy_lookup[f"shil.{policy}"]
        l1 = policy_lookup[f"l1.{policy}"]
        lines.append(
            f"{label} & {fmt(shil['selected_edge_count_mean'], 2)} & {fmt(shil['group_f1_mean'], 3)} & "
            f"{fmt(shil['group_false_inclusions_mean'], 2)} & {fmt(shil['log_loss_mean'], 3)} & "
            f"{fmt(l1['selected_edge_count_mean'], 2)} & {fmt(l1['group_f1_mean'], 3)} & "
            f"{fmt(l1['group_false_inclusions_mean'], 2)} & {fmt(l1['log_loss_mean'], 3)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_synthetic_policy_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Eligible & $\bar R$ & $\bar P$ & $\bar D$ \\",
        r"\midrule",
    ]
    for row in scenario_rows:
        lines.append(
            f"{row['scenario']} & {row['eligible_cells']} & {fmt(row['ranker_effect_R_mean'], 3)} & "
            f"{fmt(row['policy_effect_P_mean'], 3)} & {fmt(row['difference_D_mean'], 3)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_decomposition_scenarios.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dry_lookup = {row["method_id"]: row for row in dry_rows}
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Policy & \multicolumn{4}{c}{SHIL ranker} & \multicolumn{4}{c}{L1 ranker} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r" & Log loss & Macro-F1 & Accuracy & AUC & Log loss & Macro-F1 & Accuracy & AUC \\",
        r"\midrule",
    ]
    for label, policy in policies:
        shil = dry_lookup[f"shil.{policy}"]
        l1 = dry_lookup[f"l1.{policy}"]
        lines.append(
            f"{label} & {fmt(numeric(shil['log_loss_mean']), 4)} & {fmt(numeric(shil['macro_f1_mean']), 4)} & "
            f"{fmt(numeric(shil['accuracy_mean']), 4)} & {fmt(numeric(shil['roc_auc_ovr_mean']), 4)} & "
            f"{fmt(numeric(l1['log_loss_mean']), 4)} & {fmt(numeric(l1['macro_f1_mean']), 4)} & "
            f"{fmt(numeric(l1['accuracy_mean']), 4)} & {fmt(numeric(l1['roc_auc_ovr_mean']), 4)} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_dry_bean_adaptive_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparator & Selected & Support F1 & Precision & Recall \\",
        r"\midrule",
        f"glinternet strong hierarchy & {fmt(s13['selected_edge_count_mean'], 1)} & {fmt(s13['support_f1_mean'], 3)} & "
        f"{fmt(s13['support_precision_mean'], 3)} & {fmt(s13['support_recall_mean'], 3)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT / "table_s13_structured_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    synthetic_analysis = read_json(INPUTS["synthetic_analysis"])
    final_adjudication = read_json(INPUTS["final_adjudication"])
    decomposition_rows = read_csv(INPUTS["decomposition"])
    metric_rows = read_csv(INPUTS["method_metrics"])
    fixed_k_rows = read_csv(INPUTS["fixed_k"])
    invariant_rows = read_csv(INPUTS["validation_invariants"])
    cpss_rows = read_csv(INPUTS["cpss_references"])
    dry_rows = read_csv(INPUTS["dry_bean"])
    s13_rows = read_csv(INPUTS["s13"])

    nonnull_scenarios = {f"S{index:02d}" for index in range(1, 14)}
    policy_rows = aggregate_method_rows(metric_rows, nonnull_scenarios)
    null_rows = aggregate_null_rows(metric_rows)
    decomposition, scenario_rows = decomposition_summary(decomposition_rows)
    invariants = invariant_summary(invariant_rows)
    cpss_references = cpss_reference_summary(cpss_rows)
    fixed_k = fixed_k_summary(fixed_k_rows)
    dry = dry_bean_summary(dry_rows)
    structured = s13_summary(s13_rows)

    if synthetic_analysis["confirmatory"]["verdict"] != "UNAVAILABLE":
        raise RuntimeError("Frozen confirmatory verdict changed unexpectedly")
    if decomposition["planned_cells"] != 130 or decomposition["eligible_cells"] != 129:
        raise RuntimeError("Unexpected confirmatory completeness counts")
    if decomposition["D_exact_zero_cells"] != 119 or decomposition["P_exact_zero_cells"] != 128:
        raise RuntimeError("Unexpected decomposition degeneracy counts")
    if not invariants["all_eligible_match"] or invariants["support_matches_raw_exact_k"] != 258:
        raise RuntimeError("Validation-one-SE raw-support invariant changed")
    if final_adjudication["scientific_result"]["confirmatory_verdict"] != "UNAVAILABLE":
        raise RuntimeError("Post-run adjudication is not UNAVAILABLE")

    write_csv(OUT / "synthetic_policy_summary.csv", policy_rows)
    write_csv(OUT / "null_policy_summary.csv", null_rows)
    write_csv(OUT / "decomposition_scenario_summary.csv", scenario_rows)
    write_csv(OUT / "dry_bean_predictive_summary.csv", dry_rows)
    build_tex_tables(policy_rows, scenario_rows, dry_rows, structured)

    summary = {
        "schema_version": 1,
        "date_time": RUN_TIMESTAMP,
        "tool": "Codex",
        "model": MODEL,
        "operation_id": OPERATION_ID,
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
        "no_model_fitting": True,
        "no_inferential_statistics": True,
        "protocol_sha256": sha256(INPUTS["protocol"]),
        "terminal_execution": {"completed_units": 160, "planned_units": 160},
        "confirmatory": {
            "verdict": synthetic_analysis["confirmatory"]["verdict"],
            "reason": synthetic_analysis["confirmatory"]["reason"],
            "interval_computed": False,
            "theta_hat": synthetic_analysis["confirmatory"]["theta_hat"],
            "ci_lower": synthetic_analysis["confirmatory"]["ci_lower"],
            "ci_upper": synthetic_analysis["confirmatory"]["ci_upper"],
            "finite_studentized_replicates": synthetic_analysis["confirmatory"]["finite_studentized_replicates"],
            "discarded_studentized_replicates": synthetic_analysis["confirmatory"]["discarded_studentized_replicates"],
        },
        "decomposition": decomposition,
        "validation_one_se_invariant": invariants,
        "cpss_own_raw_references": cpss_references,
        "fixed_k_ranker_comparison": fixed_k,
        "dry_bean": dry,
        "structured_s13": structured,
        "source_analysis_status": {
            "synthetic": synthetic_analysis["verification_status"],
            "dry_bean": read_json(INPUTS["dry_bean_analysis"])["status"],
            "structured_s13": read_json(INPUTS["s13_analysis"])["status"],
        },
    }
    summary_path = OUT / "manuscript_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_paths = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "schema_version": 1,
        "date_time": RUN_TIMESTAMP,
        "tool": "Codex",
        "model": MODEL,
        "operation_id": OPERATION_ID,
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
        "script": {"path": str(SCRIPT.relative_to(PROJECT)), "sha256": sha256(SCRIPT)},
        "inputs": {
            key: {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
            for key, path in sorted(INPUTS.items())
        },
        "outputs": {
            path.name: {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}
            for path in output_paths
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "summary": str(summary_path),
        "confirmatory_verdict": summary["confirmatory"]["verdict"],
        "eligible_cells": decomposition["eligible_cells"],
        "invalid_cells": decomposition["invalid_cells"],
        "D_exact_zero_cells": decomposition["D_exact_zero_cells"],
        "P_exact_zero_cells": decomposition["P_exact_zero_cells"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
