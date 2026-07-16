"""Run the bounded sensitivity check promoted after the Q1 quality rescan."""

from __future__ import annotations

import json
import time
from pathlib import Path

import q1_action_remediation_analysis as q1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "q1_extension"
MD_OUT = ROOT / "MD" / "06_results"

STAMP = "20260623_2234"
RUN_ID = f"EXP-SHIL-Q1-004_bounded_sensitivity_{STAMP}"
DATE_TIME = "2026-06-23 22:34 +03:00"
OP_ID = "shil-quality-fixes-20260623-2234"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MD_OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    raw, summary, edges = q1.run_sensitivity()
    elapsed = time.perf_counter() - start

    raw_path = OUT / f"{RUN_ID}_raw.csv"
    summary_path = OUT / f"{RUN_ID}_summary.csv"
    edges_path = OUT / f"{RUN_ID}_edges.csv"
    manifest_path = OUT / f"{RUN_ID}_manifest.json"
    report_path = MD_OUT / f"{RUN_ID}_report.md"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    edges.to_csv(edges_path, index=False)
    manifest = {
        "run_id": RUN_ID,
        "date_time": DATE_TIME,
        "operation_id": OP_ID,
        "tool": "Codex",
        "model": "GPT-5 Codex",
        "source_function": "q1_action_remediation_analysis.run_sensitivity",
        "scenarios": sorted(raw["dataset"].unique().tolist()),
        "seeds": q1.SEEDS_SENSITIVITY,
        "l1_c_grid": q1.L1_C_GRID,
        "elapsed_seconds": round(elapsed, 3),
        "outputs": {
            "raw": str(raw_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "edges": str(edges_path.relative_to(ROOT)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path.write_text(
        "\n".join(
            [
                "# Bounded Sensitivity Check",
                "",
                f"Date/time: {DATE_TIME}",
                "Tool: Codex",
                "Model, if known: GPT-5 Codex",
                f"Operation ID: {OP_ID}",
                "",
                "Purpose: address the post-audit concern that correlated predictors, lower signal-to-noise, smaller samples, and class imbalance were only listed as limitations.",
                "",
                "## Outputs",
                "",
                f"- Raw rows: `{raw_path.relative_to(ROOT)}`",
                f"- Summary rows: `{summary_path.relative_to(ROOT)}`",
                f"- Selected edges: `{edges_path.relative_to(ROOT)}`",
                f"- Manifest: `{manifest_path.relative_to(ROOT)}`",
                "",
                "## Summary",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Interpretation",
                "",
                "The bounded rerun is descriptive rather than a new broad generalization claim. It tests four stressors on the planted pair generator with three seeds each. The manuscript may cite it as limited sensitivity evidence only if all generated CSVs are present and the results remain framed as bounded.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {summary_path.relative_to(ROOT)}")
    print(f"elapsed_seconds={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
