"""MCH-SHIL-004 falsifier discrimination pilot.

Operation ID: shil-claude-mch004-v6-ranker-menu-20260903

Purpose, and it is deliberately narrow: measure how often the adaptive policy
(CPSS one-SE) selects a DIFFERENT edge set from the same ranker's raw top-K
support at the policy's own realized cardinality.  v5 measured that rate at
2/260 across two coefficient-magnitude rankers, which is why the predeclared
falsifier had almost nothing to discriminate.  This pilot asks whether two
structurally different scoring principles change that.

The science is not reimplemented here.  Each cell is executed by
`track_a_runner.run_track_a_cell`, the same function the frozen v5 pipeline
used, so the split plan, leakage boundary, call plan and policies are the
originals.  This file only drives cells and compares sets.

Internal control, run on every cell before any number is reported: each
`fixed_k.kNN` support must equal the raw top-N of the same score family.  A
single control failure aborts the whole report.  That control is what caught a
wrong score-family reading in the earlier post-hoc analysis, and it is the
reason the numbers below can be trusted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sc_shil_experiment as legacy  # noqa: E402
from method_registry import load_method_registry  # noqa: E402
from track_a_runner import run_track_a_cell  # noqa: E402

CPSS_POLICY = "cpss_one_se"
NON_NULL_KIND = "null"


def _resolved_config(cfg: dict) -> dict:
    resolved = dict(cfg["track_a_resolved_config"])
    resolved["rankers.tree"] = dict(cfg["tree"])
    resolved["rankers.screen"] = dict(cfg["screen"])
    return resolved


def _raw_rank_by_family(result) -> dict[str, list[int]]:
    """Ordered candidate indices per score family, active candidates only."""
    # In-memory rows carry tuples, not JSON strings.  Only the
    # `final_ranker_score` family populates ranked_indices; the CPSS family
    # carries an empty tuple, and reading that one as the raw anchor is exactly
    # the mistake that produced a wrong post-hoc result once already.
    ranks: dict[str, list[int]] = {}
    for row in result.ranking_scores:
        if row.get("status") != "OK":
            continue
        if str(row.get("score_source")) != "final_ranker_score":
            continue
        ranked = [int(i) for i in row["ranked_indices"]]
        active = row.get("active_indices") or ()
        if active:
            keep = {int(i) for i in active}
            ranked = [i for i in ranked if i in keep]
        ranks[str(row["score_family_id"])] = ranked
    return ranks


def _cell_rows(result, scenario_id: str, seed: int) -> tuple[list[dict], int, int]:
    """Return per-ranker comparison rows plus internal-control counters."""
    ranks = _raw_rank_by_family(result)
    control_ok = control_fail = 0
    cpss_support: dict[str, set[int]] = {}

    for row in result.selected_edges:
        method_id = str(row["method_id"])
        ranker_id = str(row["ranker_id"])
        support = {int(i) for i in row["support_indices"]}
        family = f"{ranker_id}.final_ranker_score"
        if ".fixed_k.k" in method_id:
            ranked = ranks.get(family)
            if ranked is not None:
                if set(ranked[: len(support)]) == support:
                    control_ok += 1
                else:
                    control_fail += 1
        elif str(row["policy_id"]) == CPSS_POLICY:
            cpss_support[ranker_id] = support

    rows: list[dict] = []
    for ranker_id, support in sorted(cpss_support.items()):
        ranked = ranks.get(f"{ranker_id}.final_ranker_score")
        if ranked is None or not support:
            # An empty CPSS support has no matched-size raw counterpart; it is
            # recorded, not silently dropped.
            rows.append({
                "scenario": scenario_id, "seed": seed, "ranker": ranker_id,
                "k": len(support), "comparable": False,
                "identical": None, "symdiff": None,
            })
            continue
        raw = set(ranked[: len(support)])
        rows.append({
            "scenario": scenario_id, "seed": seed, "ranker": ranker_id,
            "k": len(support), "comparable": True,
            "identical": support == raw, "symdiff": len(support ^ raw),
        })
    return rows, control_ok, control_fail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--scenarios", default="", help="comma list; default all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", default="main",
                        help="shard label; keeps parallel heartbeats separate")
    parser.add_argument("--summary-only", action="store_true",
                        help="aggregate existing per-cell checkpoints, compute nothing")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    registry = load_method_registry(ROOT / "config" / "method_registry.json")
    resolved = _resolved_config(cfg)
    out = args.output
    (out / "cells").mkdir(parents=True, exist_ok=True)

    scenarios = list(cfg["scenarios"])
    if args.scenarios:
        wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["id"] in wanted]
    seeds = list(cfg["outer_seeds"])[: args.seeds]

    all_rows: list[dict] = []
    control_ok = control_fail = 0
    started = time.perf_counter()
    planned = len(scenarios) * len(seeds)
    done = 0

    for spec in scenarios:
        for seed in seeds:
            cell_id = f"{spec['id']}-{seed}"
            checkpoint = out / "cells" / f"{cell_id}.json"
            if (args.resume or args.summary_only) and checkpoint.is_file():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                all_rows.extend(saved["rows"])
                control_ok += saved["control_ok"]
                control_fail += saved["control_fail"]
                done += 1
                continue

            if args.summary_only:
                print(f"  {cell_id:10s} MISSING checkpoint", flush=True)
                continue
            cell_started = time.perf_counter()
            dataset = legacy.make_synthetic(spec, int(seed))
            result = run_track_a_cell(
                dataset,
                registry=registry,
                resolved_config=resolved,
                cell_metadata={"cell_type": "synthetic", "candidate_orders": cfg["candidate_orders"]},
                interaction_clip=float(cfg["interaction_clip"]),
                outer_split_seed=int(seed),
                master_seed=int(seed),
                n_pairs=int(cfg["n_pairs_full"]),
            )
            rows, ok, fail = _cell_rows(result, str(spec["id"]), int(seed))
            elapsed = time.perf_counter() - cell_started
            checkpoint.write_text(
                json.dumps({
                    "cell_id": cell_id, "scenario": spec["id"], "seed": seed,
                    "kind": spec["kind"], "rows": rows,
                    "control_ok": ok, "control_fail": fail,
                    "elapsed_seconds": elapsed,
                }, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            all_rows.extend(rows)
            control_ok += ok
            control_fail += fail
            done += 1
            (out / f"HEARTBEAT_{args.tag}.json").write_text(
                json.dumps({
                    "completed_cells": done, "planned_cells": planned,
                    "last_cell": cell_id, "last_cell_seconds": round(elapsed, 2),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "unix_time": time.time(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  {cell_id:10s} {elapsed:7.1f}s  control {ok} ok / {fail} fail", flush=True)

    print()
    print(f"INTERNAL CONTROL fixed_k: {control_ok} matched / {control_fail} mismatched")
    if control_fail or control_ok == 0:
        print("CONTROL FAILED -- no discrimination number is reported.")
        return 2

    non_null = {str(s["id"]) for s in scenarios if str(s["kind"]) != NON_NULL_KIND}
    report: dict[str, dict] = {}
    for ranker in sorted({r["ranker"] for r in all_rows}):
        pool = [r for r in all_rows
                if r["ranker"] == ranker and r["scenario"] in non_null and r["comparable"]]
        moved = [r for r in pool if not r["identical"]]
        rate = (len(moved) / len(pool)) if pool else 0.0
        report[ranker] = {
            "comparable_cells": len(pool),
            "moved_cells": len(moved),
            "set_movement_rate": rate,
            "median_symdiff_when_moved": (
                float(np.median([r["symdiff"] for r in moved])) if moved else 0.0
            ),
        }

    summary = {
        "operation_id": "shil-claude-mch004-v6-ranker-menu-20260903",
        "change_id": "MCH-SHIL-004",
        "predeclared_threshold": 0.10,
        "v5_reference_movement_rate": 2 / 260,
        "internal_control": {"matched": control_ok, "mismatched": control_fail},
        "cells_executed": done,
        "n_pairs": int(cfg["n_pairs_full"]),
        "seeds": seeds,
        "null_scenarios_excluded_from_rate": sorted(
            {str(s["id"]) for s in scenarios if str(s["kind"]) == NON_NULL_KIND}
        ),
        "per_ranker": report,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (out / f"DISCRIMINATION_SUMMARY_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(f"{'ranker':8s} {'comparable':>10s} {'moved':>6s} {'rate':>8s}  gate(>=10%)")
    for ranker, item in report.items():
        gate = "PASS" if item["set_movement_rate"] >= 0.10 else "below"
        print(f"{ranker:8s} {item['comparable_cells']:10d} {item['moved_cells']:6d} "
              f"{item['set_movement_rate']:7.1%}  {gate}")
    new_families = [r for r in ("tree", "screen") if report.get(r, {}).get("set_movement_rate", 0.0) >= 0.10]
    print()
    print("VERDICT:", "proceed_to_confirmatory" if new_families else "negative_generalized",
          f"(new families passing: {new_families or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
