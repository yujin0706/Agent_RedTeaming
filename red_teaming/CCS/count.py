#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot first-success trial distribution per agent from aggregate CSV.

Input:
  aggregate_attempts_S5.csv with columns: Agent, N, 1, 2, 3, 4, 5, 6

Output:
  first_success_distribution_S5.png

Colors:
  Position 1     : green   (success signal — first-try success)
  Position 2..5  : coral   (success noise — needed retries)
  Position 6     : gray + hatch (X = censored, all 5 trials failed)

Usage:
  python plot_first_success_distribution.py
  python plot_first_success_distribution.py --csv path/to/aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_CSV = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
    r"\red_teaming\CCS\generated_scenarios\aggregate_attempts_S5.csv"
)

BAR_COLOR = "#6E8FB5"       # slate blue — first-success trials (1~5)
CENSORED_COLOR = "#B5B5B5"  # gray — X (censored)


def short_name(agent: str) -> str:
    """banking_cs_agent → banking_cs"""
    return agent[:-6] if agent.endswith("_agent") else agent


def load_rows(csv_path: Path) -> list:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "agent": r["Agent"],
                "N": int(r["N"]),
                "counts": [int(r[str(i)]) for i in range(1, 7)],
            })
    return rows


def plot(rows: list, out_path: Path, ncols: int = 5):
    n_agents = len(rows)
    nrows = (n_agents + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(2.4 * ncols, 2.8 * nrows),
                              squeeze=False)

    x_labels = ["1", "2", "3", "4", "5", "X"]
    x_positions = list(range(6))

    for idx, row in enumerate(rows):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        counts = row["counts"]
        total = sum(counts)

        bar_colors = [BAR_COLOR] * 5 + [CENSORED_COLOR]
        ax.bar(x_positions, counts, color=bar_colors, edgecolor="none")

        # Faint separator between trials (1~5) and X (censored)
        ax.axvline(x=4.5, color="gray", linestyle="--",
                   linewidth=0.6, alpha=0.5)

        ymax = max(counts) if counts else 1
        for xp, val in zip(x_positions, counts):
            ax.text(xp, val + ymax * 0.02, str(val),
                    ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{short_name(row['agent'])} (n={total})",
                     fontsize=9)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("First-success trial", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_ylim(0, ymax * 1.18 if ymax else 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    for idx in range(n_agents, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle("Number of Cases by First-Success Trial per Agent",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    plt.show()

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--output", default=None)
    ap.add_argument("--ncols", type=int, default=5)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError("No rows in CSV.")

    out_path = Path(args.output) if args.output \
        else csv_path.with_name("first_success_distribution_S5.png")

    plot(rows, out_path, ncols=args.ncols)


if __name__ == "__main__":
    main()