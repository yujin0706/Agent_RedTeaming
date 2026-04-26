#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace-level UMAP visualization for CCS attack traces.

For a given agent and S-folder:
  1. Loads verdicts from judge_results.json  (O / X / skip if [API_ERROR])
  2. Loads cached embeddings (trace_emb_<agent>.npz)
  3. Flags outliers per scenario (verdict-based):
       5:0        → none
       4:1 / 1:4  → minority is outlier (Case A)
       3:2 / 2:3  → 3-group: trace whose avg sim < group_mean - delta (Case C)
                    2-group: both traces if pair sim < threshold     (Case B)
  4. Plots UMAP. One label per (scenario, verdict) group:
       5:0  → single label "T3-C1-S3 5O"
       4:1  → two labels "T3-C1-S3 4O" and "T3-C1-S3 1X"
       3:2  → two labels "T3-C1-S3 3O" and "T3-C1-S3 2X"
     Each label is placed at the centroid of its verdict group's traces.

Usage:
    python CCS_Embedding_Visualizer.py --agent banking_cs_agent --s-folder S3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Outlier thresholds — MUST match CCS_Trace_Reproducibility.py
# ═══════════════════════════════════════════════════════════════════════════
OUTLIER_DELTA_3GRP = 0.0030
OUTLIER_THRESHOLD_2GRP = 0.9785
# ═══════════════════════════════════════════════════════════════════════════


DEFAULT_LOGS_ROOT = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
    r"\red_teaming\CCS\run\logs"
)
DEFAULT_SCENARIOS_ROOT = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
    r"\red_teaming\CCS\generated_scenarios"
)

SCENARIO_ID_RE = re.compile(r"^(T\d+)-(C\d+)-(S\d+)$")
CASE_MARKERS = {"C1": "o", "C2": "s", "C3": "^"}
CASE_MARKER_FALLBACK = "D"


# ───────────────────── helpers ──────────────────────────────────────────────

def parse_scenario_id(sid: str):
    m = SCENARIO_ID_RE.match(sid or "")
    return (m.group(1), m.group(2), m.group(3)) if m else ("T?", "C?", "S?")


def load_verdicts(judge_path: Path) -> dict:
    data = json.loads(judge_path.read_text(encoding="utf-8"))
    out = {}
    for r in data.get("results", []):
        fname = Path(r.get("log_file", "")).name
        reason = r.get("reason", "") or ""
        if reason.startswith("[API_ERROR]"):
            continue
        out[fname] = r.get("judge", "X")
    return out


def cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.clip(norm, 1e-12, None)
    return normed @ normed.T


def flag_outliers(idxs: list, verdicts_local: list,
                   sim: np.ndarray) -> set:
    n = len(idxs)
    if n < 2:
        return set()
    v_counter = Counter(verdicts_local)
    counts = sorted(v_counter.values(), reverse=True)
    groups = defaultdict(list)
    for li, v in enumerate(verdicts_local):
        groups[v].append(li)

    outlier = set()
    if len(v_counter) == 1:
        return outlier

    if len(counts) == 2 and counts[-1] == 1 and counts[0] == n - 1:
        minority = min(v_counter, key=v_counter.get)
        for li in groups[minority]:
            outlier.add(idxs[li])
        return outlier

    if counts == [3, 2]:
        for _v, locs in groups.items():
            if len(locs) == 3:
                per_mean = {}
                for li in locs:
                    others = [lj for lj in locs if lj != li]
                    per_mean[li] = float(np.mean(
                        [sim[idxs[li], idxs[lj]] for lj in others]))
                group_mean = float(np.mean(list(per_mean.values())))
                for li, m in per_mean.items():
                    if m < group_mean - OUTLIER_DELTA_3GRP:
                        outlier.add(idxs[li])
            elif len(locs) == 2:
                gi0, gi1 = idxs[locs[0]], idxs[locs[1]]
                if float(sim[gi0, gi1]) < OUTLIER_THRESHOLD_2GRP:
                    outlier.add(gi0)
                    outlier.add(gi1)
    return outlier


def build_threat_color_map(threats: list) -> dict:
    import matplotlib.pyplot as plt

    def tkey(t):
        m = re.match(r"T(\d+)", t)
        return int(m.group(1)) if m else 999

    unique = sorted(set(threats), key=tkey)
    cmap = plt.cm.tab20
    return {t: cmap(i % 20) for i, t in enumerate(unique)}


# ───────────────────── main ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--s-folder", required=True)
    ap.add_argument("--date", default="", help="YYYY-MM-DD. Default: latest.")
    ap.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    ap.add_argument("--scenarios-root", default=DEFAULT_SCENARIOS_ROOT)
    ap.add_argument("--cache-dir", default=".")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    # Resolve date
    agent_attack = Path(args.logs_root) / args.agent / "attack"
    if args.date:
        log_date = args.date
    else:
        dates = [d.name for d in sorted(agent_attack.iterdir())
                 if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)]
        if not dates:
            raise RuntimeError(f"No date folders in {agent_attack}")
        log_date = dates[-1]

    s_dir = agent_attack / log_date / args.s_folder
    judge_path = s_dir / "judge_results.json"
    if not judge_path.exists():
        raise RuntimeError(f"judge_results.json not found: {judge_path}")

    cache_path = Path(args.cache_dir) / f"trace_emb_{args.agent}.npz"
    if not cache_path.exists():
        raise RuntimeError(
            f"Embedding cache not found: {cache_path}\n"
            "Run CCS_Trace_Reproducibility.py first.")

    print(f"agent:      {args.agent}")
    print(f"s_folder:   {args.s_folder}")
    print(f"log_date:   {log_date}")
    print()

    # Load
    verdicts = load_verdicts(judge_path)
    cached = np.load(cache_path, allow_pickle=True)
    all_keys = cached["trace_keys"].tolist()
    all_emb = cached["embeddings"]
    print(f"Loaded {len(verdicts)} verdicts · {len(all_keys)} cached traces")

    # Filter + align
    s_prefix = f"{args.agent}|{args.s_folder}|"
    emb_list, sids, verdict_list = [], [], []
    for i, key in enumerate(all_keys):
        if not key.startswith(s_prefix):
            continue
        parts = key.split("|")
        if len(parts) != 4:
            continue
        _, _, sid, ts = parts
        fname = f"attack-{sid}_{ts}.jsonl"
        v = verdicts.get(fname)
        if v is None:
            continue
        emb_list.append(all_emb[i])
        sids.append(sid)
        verdict_list.append(v)

    if not emb_list:
        raise RuntimeError("No matched traces for this (agent, s-folder).")

    emb = np.vstack(emb_list)
    n = len(emb_list)
    threats = [parse_scenario_id(s)[0] for s in sids]
    cases = [parse_scenario_id(s)[1] for s in sids]
    print(f"Analyzing {n} traces\n")

    # Outlier detection + per-scenario stats
    sim = cosine_sim_matrix(emb)
    by_scen = defaultdict(list)
    for i, s in enumerate(sids):
        by_scen[s].append(i)

    outlier_global = set()
    split_counter = Counter()
    for s, idxs in by_scen.items():
        vloc = [verdict_list[i] for i in idxs]
        vc = Counter(vloc)
        cnts = sorted(vc.values(), reverse=True)
        if len(vc) == 1:
            split_counter[f"{cnts[0]}:0"] += 1
        elif cnts == [4, 1]:
            split_counter["4:1"] += 1
        elif cnts == [3, 2]:
            split_counter["3:2"] += 1
        else:
            split_counter[":".join(str(c) for c in cnts)] += 1
        outlier_global |= flag_outliers(idxs, vloc, sim)

    print(f"Verdict splits: {dict(split_counter)}")
    print(f"Outliers: {len(outlier_global)}/{n} "
          f"({len(outlier_global)/n*100:.1f}%)\n")

    # ── UMAP (tuned for spread) ─────────────────────────────────────────────
    import umap
    reducer = umap.UMAP(
        n_neighbors=8,     # smaller → more local structure, more spread
        min_dist=0.6,      # larger → clusters farther apart
        spread=1.5,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(emb)

    # ── Plot ────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        from adjustText import adjust_text
        has_adjusttext = True
    except ImportError:
        print("⚠ adjustText not installed. pip install adjustText")
        has_adjusttext = False

    BG = "#141414"
    GRID = "#2e2e2e"
    TEXT = "#ececec"
    LEGEND_BG = "#1f1f1f"
    LEGEND_EDGE = "#4a4a4a"
    OUTLIER_EDGE = "#ff3b3b"

    threat_colors = build_threat_color_map(threats)

    fig, ax = plt.subplots(figsize=(16, 12), facecolor=BG)
    ax.set_facecolor(BG)

    # Scatter each trace
    for i in range(n):
        color = threat_colors.get(threats[i], "#999999")
        marker = CASE_MARKERS.get(cases[i], CASE_MARKER_FALLBACK)
        is_out = i in outlier_global
        size = 130 if is_out else 70
        lw = 2.0 if is_out else 0.4
        edge = OUTLIER_EDGE if is_out else "#0a0a0a"
        ax.scatter(coords[i, 0], coords[i, 1],
                    c=[color], s=size, marker=marker,
                    alpha=0.95, edgecolors=edge, linewidths=lw,
                    zorder=3 if is_out else 2)

    # ── Build labels: one per (scenario, verdict) group ─────────────────────
    # Group traces by (scenario_id, verdict); each group gets ONE label
    # placed at the group centroid.
    group_traces = defaultdict(list)   # (sid, verdict) → list of global idx
    for i in range(n):
        group_traces[(sids[i], verdict_list[i])].append(i)

    labels_info = []  # list of (label_text, centroid_xy, source_point_idx)
    for (sid, v), members in group_traces.items():
        cnt = len(members)
        label_text = f"{sid} {cnt}{v}"
        cx = float(np.mean(coords[members, 0]))
        cy = float(np.mean(coords[members, 1]))
        # Use the member closest to centroid as the "source point" for
        # leader-line drawing.
        dists = [np.hypot(coords[m, 0] - cx, coords[m, 1] - cy)
                 for m in members]
        src = members[int(np.argmin(dists))]
        labels_info.append((label_text, (cx, cy), src))

    print(f"Labels: {len(labels_info)} (one per scenario×verdict group)")

    texts = []
    src_points = []
    for label_text, (cx, cy), src in labels_info:
        t = ax.text(cx, cy, label_text,
                     fontsize=7, color=TEXT,
                     ha="center", va="center", zorder=5)
        texts.append(t)
        src_points.append((coords[src, 0], coords[src, 1]))

    if has_adjusttext and texts:
        adjust_text(
            texts,
            expand_points=(1.3, 1.4),
            expand_text=(1.1, 1.2),
            force_text=(0.5, 0.6),
            force_points=(0.2, 0.3),
            only_move={"text": "xy"},
        )

        # Leader lines: rendered label center → source point
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        xr = coords[:, 0].max() - coords[:, 0].min()
        yr = coords[:, 1].max() - coords[:, 1].min()
        thresh = float(np.hypot(xr, yr)) * 0.015
        for t, (ox, oy) in zip(texts, src_points):
            bbox = t.get_window_extent(renderer=renderer)
            cx = (bbox.x0 + bbox.x1) / 2
            cy = (bbox.y0 + bbox.y1) / 2
            tx, ty = inv.transform((cx, cy))
            if np.hypot(tx - ox, ty - oy) > thresh:
                ax.plot([ox, tx], [oy, ty],
                         color="#888888", lw=0.4, alpha=0.6,
                         zorder=1.5, solid_capstyle="round")

    # Legend 1: Threat Type (upper-left)
    def tkey(t):
        m = re.match(r"T(\d+)", t)
        return int(m.group(1)) if m else 999

    threat_handles = [
        Patch(facecolor=threat_colors[t], edgecolor="#111111",
              linewidth=0.5, label=t)
        for t in sorted(threat_colors.keys(), key=tkey)
    ]
    leg1 = ax.legend(
        handles=threat_handles, title="Threat Type",
        loc="upper left", fontsize=8, title_fontsize=9,
        framealpha=0.92, facecolor=LEGEND_BG,
        edgecolor=LEGEND_EDGE, labelcolor=TEXT,
    )
    leg1.get_title().set_color(TEXT)
    ax.add_artist(leg1)

    # Legend 2: Case + Outlier (upper-right)
    present_cases = sorted(
        set(cases),
        key=lambda c: int(re.match(r"C(\d+)", c).group(1))
                      if re.match(r"C(\d+)", c) else 999,
    )
    case_handles = [
        Line2D([0], [0], marker=CASE_MARKERS.get(c, CASE_MARKER_FALLBACK),
                color="none", markerfacecolor="#cccccc",
                markeredgecolor="#111111", markersize=11, label=c)
        for c in present_cases
    ]
    case_handles.append(
        Line2D([0], [0], marker="o", color="none",
                markerfacecolor="#cccccc", markeredgecolor=OUTLIER_EDGE,
                markeredgewidth=2.2, markersize=12, label="Outlier")
    )
    leg2 = ax.legend(
        handles=case_handles, title="Case / Outlier",
        loc="upper right", fontsize=9, title_fontsize=9,
        framealpha=0.92, facecolor=LEGEND_BG,
        edgecolor=LEGEND_EDGE, labelcolor=TEXT,
    )
    leg2.get_title().set_color(TEXT)

    # Axes / title
    n_o = sum(1 for v in verdict_list if v == "O")
    ax.set_title(
        f"{args.agent} / {args.s_folder}   trace-level UMAP\n"
        f"{n} traces · {n_o} O · {n - n_o} X · "
        f"{len(outlier_global)} outliers",
        fontsize=12, color=TEXT, pad=14,
    )
    ax.set_xlabel("UMAP-1", color=TEXT)
    ax.set_ylabel("UMAP-2", color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(LEGEND_EDGE)
    ax.grid(True, alpha=0.25, linestyle="--", color=GRID)

    # Save
    out_dir = Path(args.output_dir) if args.output_dir \
        else Path(args.scenarios_root) / args.agent
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"umap_{args.s_folder}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved → {png_path}")


if __name__ == "__main__":
    main()