#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempts-to-first-success analysis.

For each scenario (T*-C*-S*) in a given agent/S-folder:
  - Load its 5 traces ordered by timestamp (the natural run order)
  - Find the 1-indexed position where the first 'O' verdict occurs
  - If all 5 are 'X' → the scenario is a FAIL (not counted in the distribution)
  - [API_ERROR] traces are skipped (treated as "did not happen"); remaining
    traces are renumbered, so a scenario with O in slot 3 but slot 1 was an
    API error would count as position 2

Outputs:
  - Console table: distribution of first-success position (1..5) + fails
  - JSON:  <scenarios-root>/<agent>/attempts_to_success_<S>.json
  - CSV:   <scenarios-root>/<agent>/attempts_to_success_<S>.csv

Usage:
  python attempts_to_success.py --agent banking_cs_agent --s-folder S3
  python attempts_to_success.py --agent medical_consultation_agent --s-folder S5 --date 2026-04-20
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_LOGS_ROOT = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
    r"\red_teaming\CCS\run\logs"
)
DEFAULT_SCENARIOS_ROOT = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
    r"\red_teaming\CCS\generated_scenarios"
)

TRACE_FILENAME_RE = re.compile(r"^attack-(T\d+-C\d+-S\d+)_(\d+)\.jsonl$")


def parse_log_file(log_file: str):
    """'.../attack-T5-C1-S4_1700000000.jsonl' -> ('T5-C1-S4', '1700000000')."""
    fname = Path(log_file).name
    m = TRACE_FILENAME_RE.match(fname)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def load_judge_results(judge_path: Path) -> list:
    data = json.loads(judge_path.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("results", []):
        sid, ts = parse_log_file(r.get("log_file", ""))
        if sid is None:
            continue
        reason = r.get("reason", "") or ""
        rows.append({
            "scenario_id": sid,
            "timestamp": ts,
            "verdict": r.get("judge", "X"),
            "is_api_error": reason.startswith("[API_ERROR]"),
            "reason": reason,
            "log_file": r.get("log_file", ""),
        })
    return rows


def analyze(rows: list) -> dict:
    """
    Group by scenario_id, sort each group by timestamp, find first-success pos.
    If no 'O' in 5 valid traces, position is recorded as 6 (failure bucket).

    Returns:
      {
        "per_scenario": {sid: {"position": int,
                                "trace_verdicts": [...],
                                "n_api_errors": int, "n_valid_traces": int}},
        "position_counts": Counter,   # 1..5 for success, 6 for all-X
        "n_scenarios": int,
        "n_api_errors_total": int,
      }
    """
    FAIL_POSITION = 6

    by_scen = defaultdict(list)
    for r in rows:
        by_scen[r["scenario_id"]].append(r)

    per_scenario = {}
    position_counts = Counter()
    n_api_errors_total = 0

    for sid, lst in by_scen.items():
        lst_sorted = sorted(lst, key=lambda r: r["timestamp"])
        n_api_errors_total += sum(1 for r in lst_sorted if r["is_api_error"])

        valid = [r for r in lst_sorted if not r["is_api_error"]]

        position = FAIL_POSITION
        for i, r in enumerate(valid, start=1):
            if r["verdict"] == "O":
                position = i
                break

        per_scenario[sid] = {
            "position": position,
            "trace_verdicts": [r["verdict"] for r in lst_sorted],
            "trace_verdicts_valid": [r["verdict"] for r in valid],
            "n_api_errors": len(lst_sorted) - len(valid),
            "n_valid_traces": len(valid),
        }
        position_counts[position] += 1

    return {
        "per_scenario": per_scenario,
        "position_counts": position_counts,
        "n_scenarios": len(by_scen),
        "n_api_errors_total": n_api_errors_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=None,
                    help="Agent name, e.g. banking_cs_agent. "
                         "If omitted, all agents under logs-root are processed.")
    ap.add_argument("--s-folder", default=None,
                    help="S-folder name, e.g. S3. "
                         "If omitted, all S-folders of the agent are processed.")
    ap.add_argument("--date", default="",
                    help="YYYY-MM-DD. Default: latest.")
    ap.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    ap.add_argument("--scenarios-root", default=DEFAULT_SCENARIOS_ROOT)
    ap.add_argument("--output-dir", default=None,
                    help="Override output directory. "
                         "Default: <scenarios-root>/<agent>/")
    args = ap.parse_args()

    logs_root = Path(args.logs_root)
    if not logs_root.exists():
        raise RuntimeError(f"logs-root not found: {logs_root}")

    # Determine which agents to process
    if args.agent:
        agents = [args.agent]
    else:
        agents = sorted([
            d.name for d in logs_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
               and (d / "attack").exists()
        ])
        if not agents:
            raise RuntimeError(f"No agents found under {logs_root}")
        print(f"[auto] Processing all agents: {agents}\n")

    for agent in agents:
        process_agent(args, agent)


def process_agent(args, agent: str):
    """Process one agent across its S-folders (all or a single one)."""
    agent_attack = Path(args.logs_root) / agent / "attack"
    if not agent_attack.exists():
        print(f"[skip] {agent}: no attack dir ({agent_attack})")
        return

    # Resolve log date
    if args.date:
        log_date = args.date
    else:
        dates = [d.name for d in sorted(agent_attack.iterdir())
                 if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)]
        if not dates:
            print(f"[skip] {agent}: no date folders")
            return
        log_date = dates[-1]

    date_dir = agent_attack / log_date

    # Determine which S-folders to process
    if args.s_folder:
        s_folders = [args.s_folder]
    else:
        s_folders = sorted(
            [d.name for d in date_dir.iterdir()
             if d.is_dir() and re.match(r"^S\d+$", d.name)],
            key=lambda s: int(s.lstrip("S")),
        )
        if not s_folders:
            print(f"[skip] {agent}: no S-folders in {date_dir}")
            return

    print(f"\n{'#' * 70}")
    print(f"# Agent: {agent}  (S-folders: {s_folders})")
    print(f"{'#' * 70}")

    for s_folder in s_folders:
        process_one(args, agent, log_date, s_folder)


def process_one(args, agent: str, log_date: str, s_folder: str):
    s_dir = Path(args.logs_root) / agent / "attack" / log_date / s_folder
    if not s_dir.exists():
        print(f"[skip] S-folder not found: {s_dir}")
        return

    judge_path = s_dir / "judge_results.json"
    if not judge_path.exists():
        print(f"[skip] judge_results.json not found: {judge_path}")
        return

    print(f"[config]")
    print(f"  agent:      {agent}")
    print(f"  s_folder:   {s_folder}")
    print(f"  log_date:   {log_date}")
    print(f"  judge_file: {judge_path}")
    print()

    rows = load_judge_results(judge_path)
    print(f"Loaded {len(rows)} judge entries.")

    result = analyze(rows)

    # ── Print summary ───────────────────────────────────────────────────────
    n_scen = result["n_scenarios"]
    counts = result["position_counts"]
    n_api = result["n_api_errors_total"]
    n_fail = counts.get(6, 0)
    n_success = sum(c for p, c in counts.items() if p <= 5)

    print(f"\n{'═' * 60}")
    print(f"ATTEMPTS-TO-FIRST-SUCCESS — {agent} / {s_folder}")
    print(f"{'═' * 60}")
    print(f"  Scenarios analyzed:  {n_scen}")
    if n_api:
        print(f"  API_ERROR traces skipped: {n_api}")
    print()
    print(f"  First-success position distribution:")
    print(f"    (position 6 = all 5 traces were X → fail)")
    print(f"    {'pos':>4}  {'count':>6}  {'% of scen':>10}")
    print(f"    {'-'*4}  {'-'*6}  {'-'*10}")
    for pos in range(1, 7):
        c = counts.get(pos, 0)
        scen_pct = c / n_scen * 100 if n_scen else 0.0
        label = str(pos) if pos <= 5 else "6(X)"
        print(f"    {label:>4}  {c:>6}  {scen_pct:>9.1f}%")

    if n_success:
        mean_pos = (sum(p * c for p, c in counts.items() if p <= 5)
                    / n_success)
        print(f"\n  Mean attempts-to-success (successes only): {mean_pos:.2f}")
    mean_incl_fail = (sum(p * c for p, c in counts.items()) / n_scen
                       if n_scen else 0.0)
    print(f"  Mean position (including fail=6):          {mean_incl_fail:.2f}")

    asr_at_1 = counts.get(1, 0) / n_scen * 100 if n_scen else 0.0
    asr_at_5 = n_success / n_scen * 100 if n_scen else 0.0

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir \
        else Path(args.scenarios_root) / agent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"attempts_to_success_{s_folder}.json"
    payload = {
        "agent": agent,
        "s_folder": s_folder,
        "log_date": log_date,
        "n_scenarios": n_scen,
        "n_api_errors_skipped": n_api,
        "fail_count": n_fail,
        "fail_rate": n_fail / n_scen if n_scen else 0.0,
        "position_counts": {str(k): v for k, v in counts.items()},
        "asr_at_1": asr_at_1 / 100,
        "asr_at_5": asr_at_5 / 100,
        "mean_attempts_success_only": (sum(p * c for p, c in counts.items()
                                           if p <= 5) / n_success)
                                       if n_success else None,
        "mean_attempts_including_fail": mean_incl_fail,
        "per_scenario": {
            sid: {
                "position": r["position"],
                "trace_verdicts": r["trace_verdicts"],
                "trace_verdicts_valid": r["trace_verdicts_valid"],
                "n_api_errors": r["n_api_errors"],
                "n_valid_traces": r["n_valid_traces"],
            } for sid, r in result["per_scenario"].items()
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved JSON → {json_path}")

    # CSV: one row per scenario
    csv_path = out_dir / f"attempts_to_success_{s_folder}.csv"
    csv_lines = ["scenario_id,position,verdict_sequence,"
                 "n_valid_traces,n_api_errors,is_fail"]
    for sid in sorted(result["per_scenario"].keys(),
                       key=lambda s: (
                           int(re.match(r"T(\d+)", s).group(1)),
                           int(re.search(r"C(\d+)", s).group(1)),
                           int(re.search(r"S(\d+)", s).group(1)),
                       )):
        r = result["per_scenario"][sid]
        seq = "".join(r["trace_verdicts_valid"]) or "-"
        is_fail = 1 if r["position"] == 6 else 0
        csv_lines.append(f"{sid},{r['position']},{seq},{r['n_valid_traces']},"
                         f"{r['n_api_errors']},{is_fail}")
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved CSV  → {csv_path}\n")


if __name__ == "__main__":
    main()