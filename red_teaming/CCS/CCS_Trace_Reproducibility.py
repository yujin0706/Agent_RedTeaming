#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace content reproducibility — CUMULATIVE analysis with outlier-aware metrics.

For each agent we measure intra-scenario reproducibility as the number of
S-folders included grows cumulatively:

    N=1: S1                          → reproducibility mean
    N=2: S1 ∪ S2                     → reproducibility mean
    ...
    N=5: S1 ∪ S2 ∪ S3 ∪ S4 ∪ S5      → reproducibility mean

Four metrics per N:
  - mean_all       : pairwise cosine similarity mean (baseline, includes outliers)
  - mean_clean     : pairwise mean after excluding outlier traces
  - perfect_rate   : fraction of scenarios with 5:0 verdict (unanimous)
  - outlier_rate   : fraction of traces flagged as outliers

Outlier identification (verdict-based, same logic as CCS_Embedding_Visualizer.py):
  5:0        → no outlier
  4:1 / 1:4  → minority (1 trace) is outlier (Case A)
  3:2 / 2:3  → 3-group: trace whose mean sim is below group mean - delta (Case C)
              2-group: both traces if pair sim < threshold (Case B)

Traces marked [API_ERROR] by the judge are excluded.

Scenario IDs can collide across S-folders; internal key is "<S-folder>|<scenario_id>".

Embedding: OpenAI text-embedding-ada-002 on raw jsonl content.
  - Inputs are truncated to MAX_EMBED_TOKENS via tiktoken (cl100k_base) to
    stay under the 8,191-token per-input limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import tiktoken
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Outlier thresholds — MUST match CCS_Embedding_Visualizer.py
# ═══════════════════════════════════════════════════════════════════════════
OUTLIER_DELTA_3GRP = 0.002      # 3-group: avg-sim delta vs group mean
OUTLIER_THRESHOLD_2GRP = 0.995  # 2-group: pair similarity threshold
# ═══════════════════════════════════════════════════════════════════════════

# Embedding input truncation (ada-002 hard limit is 8,191 tokens per input)
MAX_EMBED_TOKENS = 8000

# Lazy-initialised tiktoken encoder (cl100k_base is compatible with ada-002)
_ENC = None


# ---------- API key loading --------------------------------------------------

DEFAULT_KEY_PATH = (
    r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security\API_Key\OpenAI_key.txt"
)


def load_api_key(key_path=None) -> str:
    candidates = []
    if key_path:
        candidates.append(Path(key_path))
    candidates.append(Path(DEFAULT_KEY_PATH))

    for p in candidates:
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            if key:
                return key

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise RuntimeError(
        "OpenAI API key not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\n  - OPENAI_API_KEY environment variable"
    )


# ---------- Trace discovery --------------------------------------------------

TRACE_FILENAME_RE = re.compile(r"^attack-(T\d+-C\d+-S\d+)_(\d+)\.jsonl$")


def parse_trace_filename(path: Path) -> Optional[tuple]:
    m = TRACE_FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def build_embedding_text_from_path(path: Path) -> str:
    """Raw jsonl content — timestamps/IDs/tool_result JSON all included."""
    return path.read_text(encoding="utf-8")


def truncate_to_tokens(
    text: str,
    max_tokens: int = MAX_EMBED_TOKENS,
) -> tuple[str, int, bool]:
    """
    Truncate `text` to at most `max_tokens` tokens using cl100k_base.

    Returns
    -------
    (text_out, n_tokens, was_truncated)
    """
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens), False
    return _ENC.decode(tokens[:max_tokens]), max_tokens, True


def discover_traces(
    logs_root: Path,
    agents: Optional[list] = None,
    date: str = "",
) -> list:
    records = []
    if not logs_root.exists():
        return records

    agent_dirs = [
        d for d in sorted(logs_root.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]
    if agents:
        agent_dirs = [d for d in agent_dirs if d.name in agents]

    for adir in agent_dirs:
        attack_dir = adir / "attack"
        if not attack_dir.exists():
            continue

        date_dirs = [
            d for d in sorted(attack_dir.iterdir())
            if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)
        ]
        if date:
            date_dirs = [d for d in date_dirs if d.name == date]
        else:
            if date_dirs:
                date_dirs = [date_dirs[-1]]

        for ddir in date_dirs:
            for jsonl_path in ddir.rglob("attack-*.jsonl"):
                parent_name = jsonl_path.parent.name
                if not re.match(r"^S\d+$", parent_name):
                    continue
                s_folder = parent_name

                parsed = parse_trace_filename(jsonl_path)
                if not parsed:
                    continue
                sid, ts = parsed
                records.append({
                    "agent": adir.name,
                    "date": ddir.name,
                    "s_folder": s_folder,
                    "scenario_id": sid,
                    "scenario_key": f"{s_folder}|{sid}",
                    "timestamp": ts,
                    "path": jsonl_path,
                })
    return records


# ---------- Verdict loading --------------------------------------------------

def load_verdicts_for_sfolder(s_dir: Path) -> dict:
    """
    Reads judge_results.json in s_dir.
    Returns {filename: verdict_char}, where verdict_char is 'O' or 'X'.
    Traces marked [API_ERROR] are EXCLUDED from the mapping.
    """
    judge_path = s_dir / "judge_results.json"
    if not judge_path.exists():
        return {}
    try:
        data = json.loads(judge_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠ failed to parse {judge_path.name}: {e}")
        return {}

    out = {}
    for r in data.get("results", []):
        log_file = r.get("log_file", "")
        fname = Path(log_file).name
        reason = r.get("reason", "") or ""
        if reason.startswith("[API_ERROR]"):
            continue  # exclude API errors
        out[fname] = r.get("judge", "X")
    return out


# ---------- Outlier identification (mirrors Visualizer) ----------------------

def identify_outliers_in_scenario(
    trace_globals_and_verdicts: list,
    sim: np.ndarray,
    delta_3grp: float = OUTLIER_DELTA_3GRP,
    threshold_2grp: float = OUTLIER_THRESHOLD_2GRP,
) -> set:
    """
    trace_globals_and_verdicts : list of (global_idx, verdict_char)
    sim                        : full pairwise cosine-sim matrix
    Returns set of GLOBAL indices flagged as outliers.
    """
    n = len(trace_globals_and_verdicts)
    if n < 2:
        return set()

    verdicts = [v for _, v in trace_globals_and_verdicts]
    v_counter = Counter(verdicts)
    counts = sorted(v_counter.values(), reverse=True)

    groups = defaultdict(list)   # verdict -> list of local indices (0..n-1)
    for li, (_, v) in enumerate(trace_globals_and_verdicts):
        groups[v].append(li)

    outlier_globals = set()

    if len(v_counter) == 1:
        # 5:0 unanimous → no outlier
        return outlier_globals

    if counts == [4, 1] or (n == 5 and counts[-1] == 1 and counts[0] == n - 1):
        # 4:1 (or n:1 for other n) → minority is outlier (Case A)
        minority = min(v_counter, key=v_counter.get)
        for li in groups[minority]:
            outlier_globals.add(trace_globals_and_verdicts[li][0])
        return outlier_globals

    if counts == [3, 2]:
        for v, locs in groups.items():
            if len(locs) == 3:
                # Case C: 3-group
                per_mean = {}
                for li in locs:
                    others = [lj for lj in locs if lj != li]
                    gi_self = trace_globals_and_verdicts[li][0]
                    sims = [sim[gi_self, trace_globals_and_verdicts[lj][0]]
                            for lj in others]
                    per_mean[li] = float(np.mean(sims))
                grp_mean = float(np.mean(list(per_mean.values())))
                for li, m in per_mean.items():
                    if m < grp_mean - delta_3grp:
                        outlier_globals.add(trace_globals_and_verdicts[li][0])
            elif len(locs) == 2:
                # Case B: 2-group, if pair sim is low both are outliers
                gi0 = trace_globals_and_verdicts[locs[0]][0]
                gi1 = trace_globals_and_verdicts[locs[1]][0]
                if float(sim[gi0, gi1]) < threshold_2grp:
                    outlier_globals.add(gi0)
                    outlier_globals.add(gi1)
        return outlier_globals

    # Other splits (rare with n=5): no outlier flagged
    return outlier_globals


# ---------- Embedding (cached, with rate-limit handling) ---------------------

def embed_texts(texts, api_key, model="text-embedding-ada-002",
                batch_size=50, max_retries=6,
                tpm_limit=1_000_000):
    from openai import OpenAI
    try:
        from openai import (
            RateLimitError, APIError, APIConnectionError, BadRequestError,
        )
    except Exception:
        RateLimitError = APIError = APIConnectionError = Exception
        BadRequestError = type("BadRequestError", (Exception,), {})  # sentinel

    client = OpenAI(api_key=api_key)
    vectors = []
    token_window = []

    def _approx_tokens(s):
        return int(len(s) * 0.55)

    def _window_tokens():
        now = time.time()
        while token_window and now - token_window[0][0] > 60:
            token_window.pop(0)
        return sum(t for _, t in token_window)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_tokens = sum(_approx_tokens(t) for t in batch)

        safety_cap = int(tpm_limit * 0.85)
        while _window_tokens() + batch_tokens > safety_cap:
            if token_window:
                wait = 60 - (time.time() - token_window[0][0]) + 0.5
                if wait > 0:
                    print(f"  ⏸ TPM pacing: sleeping {wait:.1f}s "
                          f"(window {_window_tokens():,} + batch {batch_tokens:,})")
                    time.sleep(wait)
            else:
                break

        for attempt in range(max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vectors.extend(d.embedding for d in resp.data)
                token_window.append((time.time(), batch_tokens))
                break
            except Exception as e:
                # 400 BadRequest is NOT retryable — fail fast with context.
                if isinstance(e, BadRequestError):
                    print(f"  ✗ BadRequestError at batch {i}-{i + len(batch)}: {e}")
                    raise
                msg = str(e).lower()
                retryable = isinstance(e, (RateLimitError, APIError, APIConnectionError)) \
                    or any(s in msg for s in ("429", "500", "502", "503", "504",
                                              "rate limit", "timeout"))
                if attempt >= max_retries or not retryable:
                    raise
                delay = min(2.0 * (2 ** attempt), 60.0)
                print(f"  ⚠ {type(e).__name__}: retrying in {delay:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)

        done = min(i + batch_size, len(texts))
        print(f"  embedded {done}/{len(texts)}")

    return np.array(vectors, dtype=np.float32)


def _content_hash(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_or_compute_embeddings(trace_keys, trace_texts, cache_path, api_key):
    text_hash = _content_hash(trace_texts)

    if cache_path.exists():
        print(f"Checking cached embeddings at {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        cached_keys = cached["trace_keys"].tolist()
        cached_hash = (str(cached["content_hash"])
                       if "content_hash" in cached.files else None)
        if cached_keys == trace_keys and cached_hash == text_hash:
            print(f"  cache OK (hash {text_hash[:8]}…) — reusing embeddings")
            return cached["embeddings"]
        if cached_keys != trace_keys:
            print("  trace list changed — re-embedding")
        elif cached_hash is None:
            print("  legacy cache without hash — re-embedding")
        else:
            print(f"  content changed "
                  f"(cached {cached_hash[:8]}… vs current {text_hash[:8]}…)")

    print(f"Embedding {len(trace_texts)} traces via OpenAI API...")
    emb = embed_texts(trace_texts, api_key=api_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=emb,
        trace_keys=np.array(trace_keys),
        content_hash=np.array(text_hash),
    )
    print(f"  saved → {cache_path}")
    return emb


# ---------- Similarity + outlier-aware statistics ----------------------------

def cosine_sim_matrix(emb):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.clip(norm, 1e-12, None)
    return normed @ normed.T


def intra_scenario_stats(emb, scenario_keys, verdicts=None):
    """
    Per-scenario intra-similarity, with optional verdict-based outlier info.

    Parameters
    ----------
    emb             : (N, D) array
    scenario_keys   : list of length N
    verdicts        : optional list of length N, verdict char ('O'/'X') or None

    Returns
    -------
    dict of {scenario_key: {
        "mean_all", "std", "n_traces",
        # If verdicts provided:
        "mean_clean",          # pairwise mean after excluding outlier traces
        "n_outliers",          # number of outlier traces in scenario
        "is_perfect",          # True if all verdicts are same (5:0)
        "outlier_global_idx",  # list of global indices flagged as outlier
    }}
    """
    sim = cosine_sim_matrix(emb)
    by_scen = defaultdict(list)
    for i, sk in enumerate(scenario_keys):
        by_scen[sk].append(i)

    per_scenario = {}
    for sk, idxs in by_scen.items():
        if len(idxs) < 2:
            continue

        pair_sims_all = [sim[i, j] for i, j in combinations(idxs, 2)]
        result = {
            "mean_all": float(np.mean(pair_sims_all)),
            "std": float(np.std(pair_sims_all, ddof=1))
                    if len(pair_sims_all) > 1 else 0.0,
            "n_traces": len(idxs),
        }

        if verdicts is not None:
            scen_verdicts = [(gi, verdicts[gi]) for gi in idxs]
            # Skip traces without verdict (e.g. excluded API_ERROR)
            scen_verdicts_known = [(gi, v) for gi, v in scen_verdicts
                                    if v is not None]

            if len(scen_verdicts_known) == len(scen_verdicts) and scen_verdicts_known:
                outlier_globals = identify_outliers_in_scenario(
                    scen_verdicts_known, sim,
                )
                clean_idxs = [gi for gi in idxs if gi not in outlier_globals]

                if len(clean_idxs) >= 2:
                    pair_sims_clean = [sim[i, j]
                                       for i, j in combinations(clean_idxs, 2)]
                    result["mean_clean"] = float(np.mean(pair_sims_clean))
                else:
                    # All or nearly all flagged — fall back to mean_all
                    result["mean_clean"] = result["mean_all"]

                v_set = set(v for _, v in scen_verdicts_known)
                result["is_perfect"] = (len(v_set) == 1)
                result["n_outliers"] = len(outlier_globals)
                result["outlier_global_idx"] = sorted(outlier_globals)
            else:
                # Some verdicts missing → don't flag outliers
                result["mean_clean"] = result["mean_all"]
                result["is_perfect"] = None
                result["n_outliers"] = 0
                result["outlier_global_idx"] = []

        per_scenario[sk] = result
    return per_scenario


def summarize_per_N(per_scenario: dict, n_traces_total: int) -> dict:
    """
    Aggregate scenario-level stats into N-level summary.
    """
    if not per_scenario:
        return {}

    means_all = np.array([v["mean_all"] for v in per_scenario.values()])
    summary = {
        "n_scenarios": len(per_scenario),
        "mean_all": float(means_all.mean()),
        "std_all": float(means_all.std(ddof=1)) if len(means_all) > 1 else 0.0,
        "min_all": float(means_all.min()),
        "max_all": float(means_all.max()),
    }

    # Clean-mean (only if we actually have outlier info)
    clean_vals = [v.get("mean_clean") for v in per_scenario.values()
                  if v.get("mean_clean") is not None]
    has_outlier_info = any("n_outliers" in v for v in per_scenario.values())

    if has_outlier_info:
        means_clean = np.array(clean_vals) if clean_vals else means_all
        summary["mean_clean"] = float(means_clean.mean())
        summary["std_clean"] = (float(means_clean.std(ddof=1))
                                if len(means_clean) > 1 else 0.0)

        n_perfect = sum(1 for v in per_scenario.values()
                        if v.get("is_perfect") is True)
        summary["perfect_rate"] = n_perfect / len(per_scenario)

        n_outlier_scen = sum(1 for v in per_scenario.values()
                             if v.get("n_outliers", 0) > 0)
        summary["outlier_scenario_rate"] = n_outlier_scen / len(per_scenario)

        total_outlier_traces = sum(v.get("n_outliers", 0)
                                    for v in per_scenario.values())
        total_traces = sum(v["n_traces"] for v in per_scenario.values())
        summary["outlier_trace_rate"] = (total_outlier_traces / total_traces
                                          if total_traces else 0.0)
        summary["n_outlier_traces"] = total_outlier_traces
        summary["n_outlier_scenarios"] = n_outlier_scen

    return summary


# ---------- Main -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs_root", nargs="?",
                   default=r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
                           r"\red_teaming\CCS\run\logs")
    p.add_argument("--agent", nargs="*", default=None,
                   help="Agents to analyse. Default: all.")
    p.add_argument("--date", default="2026-04-20",
                   help="Log date (YYYY-MM-DD). Default: 2026-04-20.")
    p.add_argument("--cache-dir", default=".",
                   help="Directory for embedding caches.")
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--scenarios-root",
                   default=r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security"
                           r"\red_teaming\CCS\generated_scenarios")
    p.add_argument("--output-json", default=None,
                   help="Override save path. Default: per-agent mode → "
                        "<scenarios-root>/<agent>/trace_reproducibility.json; "
                        "aggregate mode → ./trace_reproducibility.json.")
    args = p.parse_args()

    logs_root = Path(args.logs_root)
    if not logs_root.exists():
        raise RuntimeError(f"logs_root not found: {logs_root}")

    trace_records = discover_traces(
        logs_root, agents=args.agent, date=args.date,
    )
    if not trace_records:
        raise RuntimeError("No trace files found.")

    by_agent = defaultdict(list)
    for rec in trace_records:
        by_agent[rec["agent"]].append(rec)

    print(f"Discovered {len(trace_records)} trace files "
          f"across {len(by_agent)} agents.")
    print()

    api_key = load_api_key(args.api_key_file)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for agent, recs in sorted(by_agent.items()):
        print(f"=== {agent} ({len(recs)} total traces) ===")

        recs = sorted(recs, key=lambda r: (
            r["s_folder"], r["scenario_id"], r["timestamp"]
        ))

        # ── Load verdicts per S-folder ──────────────────────────────────────
        verdict_by_sfolder = {}   # s_folder -> {filename: verdict}
        for rec in recs:
            sf = rec["s_folder"]
            if sf not in verdict_by_sfolder:
                s_dir = rec["path"].parent
                verdict_by_sfolder[sf] = load_verdicts_for_sfolder(s_dir)

        n_with_verdict = sum(1 for rec in recs
                             if verdict_by_sfolder.get(rec["s_folder"], {})
                                 .get(rec["path"].name) is not None)
        print(f"  verdicts loaded: {n_with_verdict}/{len(recs)} traces"
              f" ({len(verdict_by_sfolder)} S-folders)")

        # ── Build parallel arrays (key, text, scenario_key, verdict) ────────
        trace_keys = []
        trace_texts = []
        scenario_keys = []
        s_folder_list = []
        trace_verdicts = []
        skipped_api_error = 0
        n_truncated = 0

        for rec in recs:
            fname = rec["path"].name
            v = verdict_by_sfolder.get(rec["s_folder"], {}).get(fname)

            # If verdict map exists for this S-folder but this trace is missing,
            # it was excluded (API_ERROR). Skip it so it doesn't pollute stats.
            if verdict_by_sfolder.get(rec["s_folder"]) and v is None:
                skipped_api_error += 1
                continue

            try:
                text = build_embedding_text_from_path(rec["path"])
            except Exception as e:
                print(f"  skip (read error): {fname} — {e}")
                continue

            text, n_tok, was_truncated = truncate_to_tokens(text)
            if was_truncated:
                n_truncated += 1
                print(f"  ✂ truncated: {fname} → {n_tok:,} tokens")

            trace_keys.append(
                f"{rec['agent']}|{rec['s_folder']}|{rec['scenario_id']}|{rec['timestamp']}"
            )
            trace_texts.append(text)
            scenario_keys.append(rec["scenario_key"])
            s_folder_list.append(rec["s_folder"])
            trace_verdicts.append(v)

        if skipped_api_error:
            print(f"  ⚠ excluded {skipped_api_error} API_ERROR traces")
        if n_truncated:
            print(f"  ✂ {n_truncated} traces truncated to {MAX_EMBED_TOKENS} tokens")

        if len(trace_texts) < 2:
            print(f"  insufficient traces — skipping")
            print()
            continue

        cache_path = cache_dir / f"trace_emb_{agent}.npz"
        emb = load_or_compute_embeddings(trace_keys, trace_texts,
                                          cache_path, api_key)

        # ── Cumulative analysis: N=1, 2, ..., len(available_s) ──────────────
        available_s = sorted(
            set(s_folder_list), key=lambda x: int(x.lstrip("S"))
        )
        print(f"  S-folders available: {available_s}")

        per_N = {}
        s_folder_arr = np.array(s_folder_list)
        scenario_keys_arr = np.array(scenario_keys)
        verdicts_arr = trace_verdicts  # list aligned with emb rows

        for N in range(1, len(available_s) + 1):
            included = set(available_s[:N])
            mask = np.array([sf in included for sf in s_folder_arr])
            sub_emb = emb[mask]
            sub_keys = scenario_keys_arr[mask].tolist()
            # Re-index verdicts to sub range (need list aligned with sub_emb rows)
            sub_verdicts = [verdicts_arr[i]
                            for i in range(len(verdicts_arr)) if mask[i]]

            per_scenario = intra_scenario_stats(sub_emb, sub_keys, sub_verdicts)
            if not per_scenario:
                print(f"  N={N}: no valid scenarios")
                continue

            summary = summarize_per_N(per_scenario, int(mask.sum()))
            per_N[N] = {
                "included_s_folders": sorted(included),
                "n_traces": int(mask.sum()),
                **summary,
                "per_scenario": per_scenario,
            }
            msg = (f"  N={N}: {sorted(included)}  "
                   f"traces={mask.sum()}  scenarios={summary['n_scenarios']}  "
                   f"mean_all={summary['mean_all']:.4f}  "
                   f"std={summary['std_all']:.4f}")
            if "mean_clean" in summary:
                msg += (f"\n        "
                        f"mean_clean={summary['mean_clean']:.4f}  "
                        f"perfect={summary['perfect_rate']*100:.1f}%  "
                        f"outlier_trace={summary['outlier_trace_rate']*100:.2f}% "
                        f"({summary['n_outlier_traces']} traces)")
            print(msg)

        all_results[agent] = {
            "agent": agent,
            "n_total_traces": len(trace_texts),
            "s_folders_available": available_s,
            "thresholds": {
                "outlier_delta_3grp": OUTLIER_DELTA_3GRP,
                "outlier_threshold_2grp": OUTLIER_THRESHOLD_2GRP,
            },
            "per_N": per_N,
        }
        print()

    # ── Cumulative matrix tables (print + CSV body) ─────────────────────────
    csv_body = ""
    if all_results:
        def agent_final_mean_all(a):
            ns = all_results[a]["per_N"]
            return float(ns[max(ns.keys())]["mean_all"]) if ns else 0.0

        ordered_agents = sorted(all_results.keys(),
                                 key=agent_final_mean_all, reverse=True)

        # CSV: one row per agent per N
        csv_lines = ["agent,N,included_s,n_traces,n_scenarios,"
                     "mean_all,std_all,mean_clean,perfect_rate,"
                     "n_outlier_traces,outlier_trace_rate"]
        for a in ordered_agents:
            ns = all_results[a]["per_N"]
            for N in sorted(ns.keys()):
                r = ns[N]
                csv_lines.append(",".join([
                    a, str(N),
                    "|".join(r["included_s_folders"]),
                    str(r["n_traces"]),
                    str(r.get("n_scenarios", "")),
                    f"{r.get('mean_all', 0):.4f}",
                    f"{r.get('std_all', 0):.4f}",
                    f"{r.get('mean_clean', ''):.4f}" if r.get('mean_clean') is not None else "",
                    f"{r.get('perfect_rate', ''):.4f}" if r.get('perfect_rate') is not None else "",
                    str(r.get('n_outlier_traces', '')) if r.get('n_outlier_traces') is not None else "",
                    f"{r.get('outlier_trace_rate', ''):.4f}" if r.get('outlier_trace_rate') is not None else "",
                ]))
        csv_body = "\n".join(csv_lines) + "\n"

    # ── Save ────────────────────────────────────────────────────────────────
    per_agent_mode = args.agent is not None and len(args.agent) > 0

    if per_agent_mode:
        scenarios_root = Path(args.scenarios_root)
        saved_files = []
        for agent_name in args.agent:
            if agent_name not in all_results:
                print(f"[{agent_name}] no results — skipping save")
                continue

            agent_dir = scenarios_root / agent_name
            agent_dir.mkdir(parents=True, exist_ok=True)

            if args.output_json:
                json_path = Path(args.output_json)
                if len(args.agent) > 1:
                    json_path = json_path.with_name(
                        f"{json_path.stem}_{agent_name}{json_path.suffix}"
                    )
            else:
                json_path = agent_dir / "trace_reproducibility.json"

            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps({agent_name: all_results[agent_name]},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_files.append(json_path)

            # Per-agent CSV: long-form (one row per N)
            csv_path = json_path.with_suffix(".csv")
            ns = all_results[agent_name]["per_N"]
            header = ("N,included_s,n_traces,n_scenarios,"
                      "mean_all,std_all,mean_clean,perfect_rate,"
                      "outlier_scenario_rate,outlier_trace_rate")
            lines = [header]
            for N in sorted(ns.keys()):
                r = ns[N]
                lines.append(",".join([
                    str(N),
                    "|".join(r["included_s_folders"]),
                    str(r["n_traces"]),
                    str(r.get("n_scenarios", "")),
                    f"{r.get('mean_all', 0):.4f}",
                    f"{r.get('std_all', 0):.4f}",
                    f"{r.get('mean_clean', ''):.4f}" if r.get('mean_clean') is not None else "",
                    f"{r.get('perfect_rate', ''):.4f}" if r.get('perfect_rate') is not None else ""
                ]))
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            saved_files.append(csv_path)

        print()
        for fp in saved_files:
            print(f"Saved → {fp}")

    else:
        out_json = Path(args.output_json) if args.output_json \
            else Path("trace_reproducibility.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved → {out_json}")

        if csv_body:
            csv_path = out_json.with_suffix(".csv")
            csv_path.write_text(csv_body, encoding="utf-8")
            print(f"Saved matrix → {csv_path}")


if __name__ == "__main__":
    main()