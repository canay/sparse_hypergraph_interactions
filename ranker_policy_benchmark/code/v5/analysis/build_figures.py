"""Create publication figures for the Track A v5 manuscript reassembly.

Date/time: 2026-08-31 01:02:50 +03:00
Tool: Codex
Model, if known: GPT-5.6 Sol
Operation ID: shil-g09-r1-safe-revision-20260830-234612
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OPERATION_ID = "shil-g09-r1-safe-revision-20260830-234612"
RUN_TIMESTAMP = "2026-08-31 01:02:50 +03:00"
SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[3]
RUN_ROOT = SCRIPT.parents[1]
OUTPUTS = RUN_ROOT / "outputs"
MANUSCRIPT_FIGURES = PROJECT / "JOURNAL_SHORTNAME" / "manuscript-r1" / "figures"
SCENARIO_CSV = OUTPUTS / "decomposition_scenario_summary.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def format_half_up(value: float, places: int = 3) -> str:
    """Format displayed evidence values with an explicit half-up convention."""
    quantum = Decimal(1).scaleb(-places)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{places}f}"


def save_both(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    output_png = OUTPUTS / f"{stem}.png"
    output_pdf = OUTPUTS / f"{stem}.pdf"
    manuscript_png = MANUSCRIPT_FIGURES / output_png.name
    manuscript_pdf = MANUSCRIPT_FIGURES / output_pdf.name

    # Render once, then copy byte-for-byte so the reproducibility output and
    # manuscript asset have identical hashes (PDF metadata timestamps included).
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    shutil.copyfile(output_png, manuscript_png)
    shutil.copyfile(output_pdf, manuscript_pdf)
    return [output_png, output_pdf, manuscript_png, manuscript_pdf]


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str,
             face: str, edge: str = "#28323C", fontsize: float = 8.6) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#17212B", linespacing=1.2)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#52606D") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color=color, shrinkA=2, shrinkB=2))


def study_flow() -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#D7E8F5"
    teal = "#D8EFE9"
    amber = "#F8E9C8"
    red = "#F5D5D2"
    grey = "#E8ECEF"

    draw_box(ax, (0.02, 0.66), 0.15, 0.18, "Outer replicate\n60/20/20 split", blue)
    draw_box(ax, (0.21, 0.66), 0.17, 0.18, "Shared pair/triple\ncandidate universe", blue)
    draw_box(ax, (0.42, 0.72), 0.15, 0.14, "SHIL ranker", teal)
    draw_box(ax, (0.42, 0.52), 0.15, 0.14, "L1 ranker", teal)
    draw_box(ax, (0.61, 0.60), 0.18, 0.22,
             "Seven policies\nfixed K: 1, 2, 4, 8, 16\nvalidation one-SE\nCPSS one-SE", amber, fontsize=8.2)
    draw_box(ax, (0.83, 0.66), 0.15, 0.18, "Common L2 refit\nlocked test score", blue)

    arrow(ax, (0.17, 0.75), (0.21, 0.75))
    arrow(ax, (0.38, 0.75), (0.42, 0.79))
    arrow(ax, (0.38, 0.75), (0.42, 0.59))
    arrow(ax, (0.57, 0.79), (0.61, 0.74))
    arrow(ax, (0.57, 0.59), (0.61, 0.67))
    arrow(ax, (0.79, 0.71), (0.83, 0.75))

    draw_box(ax, (0.20, 0.17), 0.20, 0.18,
             "Completeness gate\n14 methods + 4 raw\nexact-K references", grey)
    draw_box(ax, (0.48, 0.20), 0.20, 0.14, "All 130 cells valid", teal)
    draw_box(ax, (0.48, 0.02), 0.20, 0.14, "Any invalid cell", red)
    draw_box(ax, (0.76, 0.20), 0.22, 0.14,
             "Compute predeclared\nstudentized interval", teal)
    draw_box(ax, (0.76, 0.02), 0.22, 0.14,
             "UNAVAILABLE\nreport descriptive map only", red)

    arrow(ax, (0.90, 0.66), (0.40, 0.35))
    arrow(ax, (0.40, 0.26), (0.48, 0.27))
    arrow(ax, (0.40, 0.22), (0.48, 0.09))
    arrow(ax, (0.68, 0.27), (0.76, 0.27))
    arrow(ax, (0.68, 0.09), (0.76, 0.09))

    ax.text(0.44, 0.31, "PASS", fontsize=8, color="#177245", ha="center")
    ax.text(0.44, 0.12, "FAIL", fontsize=8, color="#A33A32", ha="center")
    ax.text(0.02, 0.95, "Frozen ranker–policy benchmark and fail-closed adjudication",
            fontsize=13, weight="bold", color="#17212B")
    ax.text(0.02, 0.90,
            "Ranking, support policy, predictive refit, and confirmatory eligibility remain separate.",
            fontsize=9.5, color="#52606D")
    return save_both(fig, "fig0_v5_study_flow")


def decomposition_map() -> list[Path]:
    with SCENARIO_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    scenarios = [row["scenario"] for row in rows]
    r_values = [float(row["ranker_effect_R_mean"]) for row in rows]
    p_values = [float(row["policy_effect_P_mean"]) for row in rows]
    d_values = [float(row["difference_D_mean"]) for row in rows]
    eligible = [int(row["eligible_cells"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.4, 5.3))
    x = list(range(len(scenarios)))
    width = 0.24
    colors = {"R": "#3B6FB6", "P": "#2A9D8F", "D": "#D1495B"}
    ax.bar([value - width for value in x], r_values, width, label=r"Ranker contrast $R$", color=colors["R"])
    ax.bar(x, p_values, width, label=r"Policy residual $P$", color=colors["P"])
    ax.bar([value + width for value in x], d_values, width, label=r"Difference $D=P-R$", color=colors["D"])

    ax.axhline(0, color="#4A5560", linewidth=0.9)
    ax.set_xticks(x)
    labels = [f"{scenario}\n(n={count})" if count != 10 else scenario for scenario, count in zip(scenarios, eligible)]
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean absolute-contrast scale", fontsize=10)
    ax.set_ylim(-0.34, 0.34)
    fig.suptitle("Only the redundant-proxy regime departs from zero", x=0.08, y=0.985,
                 ha="left", fontsize=13, weight="bold")
    fig.text(0.08, 0.935, "Descriptive scenario means; no interval or directional inference",
             ha="left", fontsize=9.2, color="#52606D")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.8)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#89939D")
    ax.spines["bottom"].set_color("#89939D")

    s11 = scenarios.index("S11")
    ax.annotate(f"R={format_half_up(r_values[s11])}", (s11 - width, r_values[s11]),
                xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=colors["R"])
    ax.annotate(f"P={format_half_up(p_values[s11])}", (s11, p_values[s11]),
                xytext=(10, 7), textcoords="offset points", ha="left", fontsize=8, color=colors["P"])
    ax.annotate(f"D={format_half_up(d_values[s11])}", (s11 + width, d_values[s11]),
                xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color=colors["D"])

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return save_both(fig, "fig1_v5_decomposition_map")


def main() -> None:
    if not SCENARIO_CSV.is_file():
        raise FileNotFoundError(SCENARIO_CSV)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelcolor": "#28323C",
        "xtick.color": "#4A5560",
        "ytick.color": "#4A5560",
    })
    generated = study_flow() + decomposition_map()
    plt.close("all")
    manifest = {
        "schema_version": 1,
        "date_time": RUN_TIMESTAMP,
        "tool": "Codex",
        "model": "GPT-5.6 Sol",
        "operation_id": OPERATION_ID,
        "analysis_role": "DESCRIPTIVE_NO_INFERENCE",
        "source": {"path": str(SCENARIO_CSV.relative_to(PROJECT)), "sha256": sha256(SCENARIO_CSV)},
        "script": {"path": str(SCRIPT.relative_to(PROJECT)), "sha256": sha256(SCRIPT)},
        "generated": {
            str(path.relative_to(PROJECT)): sha256(path)
            for path in generated
        },
    }
    manifest_path = OUTPUTS / "FIGURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "generated_files": len(generated), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
