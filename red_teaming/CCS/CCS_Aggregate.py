#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate attempts_to_success_S5.csv across all agents.

Input:
  <scenarios-root>/<agent>/attempts_to_success_S5.csv

Output (CSV):
  Agent, N, 1, 2, 3, 4, 5, 6
  banking_cs_agent, 5, 117, 8, 7, 7, 1, 145
  ecommerce_operations_agent, 5, 209, 17, 6, 6, 2, 45
  ...

Usage:
  python aggregate_attempts_S5.py
  python aggregate_attempts_S5.py --scenarios-root C:\...\generated_scenarios
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_SCENARIOS_ROOT = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
    r"\red_teaming\CCS\generated_scenarios"
)
S_FOLDER = "S5"
N_VALUE = 5  # S5 = 5 scenarios per threat-context


def aggregate_one_agent(csv_path: Path) -> Counter:
    """Read attempts_to_success_S5.csv and count position 1..6."""
    counts = Counter()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pos = int(row["position"])
            except (ValueError, KeyError):
                continue
            if 1 <= pos <= 6:
                counts[pos] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-root", default=DEFAULT_SCENARIOS_ROOT)
    ap.add_argument("--output", default=None,
                    help="Output CSV path. "
                         "Default: <scenarios-root>/aggregate_attempts_S5.csv")
    args = ap.parse_args()

    root = Path(args.scenarios_root)
    if not root.exists():
        raise RuntimeError(f"scenarios-root not found: {root}")

    agents = sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    rows = []
    print(f"\n{'Agent':<35s} {'N':>3s} "
          f"{'1':>5s} {'2':>5s} {'3':>5s} {'4':>5s} {'5':>5s} {'6':>5s}")
    print("-" * 75)

    for agent in agents:
        csv_path = root / agent / f"attempts_to_success_{S_FOLDER}.csv"
        if not csv_path.exists():
            print(f"  [skip] {agent}: no {csv_path.name}")
            continue

        counts = aggregate_one_agent(csv_path)
        row = {
            "Agent": agent,
            "N": N_VALUE,
            "1": counts.get(1, 0),
            "2": counts.get(2, 0),
            "3": counts.get(3, 0),
            "4": counts.get(4, 0),
            "5": counts.get(5, 0),
            "6": counts.get(6, 0),
        }
        rows.append(row)
        print(f"{agent:<35s} {N_VALUE:>3d} "
              f"{row['1']:>5d} {row['2']:>5d} {row['3']:>5d} "
              f"{row['4']:>5d} {row['5']:>5d} {row['6']:>5d}")

    # Save
    out_path = Path(args.output) if args.output \
        else root / f"aggregate_attempts_{S_FOLDER}.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Agent", "N", "1", "2", "3", "4", "5", "6"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()