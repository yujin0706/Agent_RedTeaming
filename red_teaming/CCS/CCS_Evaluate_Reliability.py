#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCS 시나리오 신뢰도 평가 스크립트.

측정 항목 1가지:
  1. 재현성 (Reproducibility)
       동일 입력을 N회 실행한 trace를 임베딩 → 클러스터링 → outlier 여부 검증
       "동일 입력에 대해 동일한 프로세스가 진행된다"를 보임

의존 파일:
  - red_teaming/CCS/generated_scenarios/{agent}/{date}/*.json
  - red_teaming/CCS/run/logs/{agent}/attack/{date}/*.jsonl
  - red_teaming/CCS/THREAT_Specification.json
  - red_teaming/CCS/agent_profiles/{agent}/{date}/agent_profile.yaml

사용법:
  python CCS_Evaluate_Reliability.py --agent banking_cs_agent --runs 10
  python CCS_Evaluate_Reliability.py --agent banking_cs_agent --runs 10 --scenario-id T1-C1-S1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

# sentence-transformers 있으면 사용, 없으면 TF-IDF fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import yaml
except ImportError:
    yaml = None

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


# ══════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════

DBSCAN_EPS = 0.25                # DBSCAN epsilon (코사인 거리 기준)
DBSCAN_MIN_SAMPLES = 2           # DBSCAN 최소 클러스터 크기


# ══════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════

def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_yaml_file(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_latest_date(base: Path) -> Optional[str]:
    if not base.exists():
        return None
    dates = [d.name for d in sorted(base.iterdir())
              if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name)]
    return dates[-1] if dates else None


def discover_agents(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    return [d.name for d in sorted(base_dir.iterdir())
            if d.is_dir() and not d.name.startswith(".")]


# ══════════════════════════════════════════════════════════
# API Key
# ══════════════════════════════════════════════════════════

def load_api_key(repo_root: Path, provider: str) -> str:
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if key:
            return key
        for fname in ("gemini_api_key", "gemini_api_key.txt"):
            p = repo_root / "API_Key" / fname
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        raise RuntimeError("GEMINI_API_KEY를 찾을 수 없음")
    return os.getenv("OPENAI_API_KEY", "dummy")


# ══════════════════════════════════════════════════════════
# Trace 추출
# ══════════════════════════════════════════════════════════

def extract_trace(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    assistant_texts, tool_calls, tool_results = [], [], []
    ipi_injected = None
    final_text = ""
    meta = {}

    for r in records:
        t = r.get("type")
        if t == "meta":
            meta = r
        elif t == "assistant":
            txt = str(r.get("text", "")).strip()
            if txt:
                assistant_texts.append(txt)
        elif t == "tool_call":
            tool_calls.append({"name": r.get("name"), "args": r.get("args", {})})
        elif t == "tool_result":
            tool_results.append({"name": r.get("name"), "result": r.get("result", {})})
        elif t == "ipi_injected":
            ipi_injected = {"target_tool": r.get("target_tool"), "payload": r.get("payload")}
        elif t == "final":
            final_text = str(r.get("text", ""))

    return {
        "meta": meta,
        "assistant_texts": assistant_texts,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "ipi_injected": ipi_injected,
        "final_text": final_text,
    }


def trace_to_text(trace: Dict[str, Any]) -> str:
    """
    trace를 임베딩용 텍스트로 변환.
    tool_call 시퀀스 + plan 텍스트 + final 응답을 합친다.
    """
    parts = []

    tool_seq = " -> ".join(tc.get("name", "") for tc in trace.get("tool_calls", []))
    if tool_seq:
        parts.append(f"TOOLS: {tool_seq}")

    for txt in trace.get("assistant_texts", []):
        parts.append(f"PLAN: {txt[:200]}")

    final = trace.get("final_text", "")
    if final:
        parts.append(f"FINAL: {final[:200]}")

    return " | ".join(parts) if parts else "(empty trace)"


# ══════════════════════════════════════════════════════════
# Runner (단일 task 재실행)
# ══════════════════════════════════════════════════════════

async def run_one_task_gemini(
    *,
    client,
    model: str,
    system_instruction: str,
    session,
    fn_decls: List[Any],
    user_prompt: str,
    ipi_source_tool: Optional[str],
    ipi_payload: Optional[str],
    max_steps: int = 10,
) -> List[Dict[str, Any]]:
    """Gemini로 단일 task 실행 → trace records 반환."""
    if genai is None or genai_types is None:
        raise RuntimeError("google-genai not installed")

    tool_obj = genai_types.Tool(function_declarations=fn_decls)
    tool_config = genai_types.ToolConfig(
        function_calling_config=genai_types.FunctionCallingConfig(mode="AUTO")
    )
    config = genai_types.GenerateContentConfig(
        tools=[tool_obj], tool_config=tool_config,
        system_instruction=system_instruction, temperature=0,
    )
    contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=user_prompt)])]
    records: List[Dict[str, Any]] = []

    def log(obj: Dict[str, Any]) -> None:
        records.append({**obj, "ts": utc_now_iso()})

    for _ in range(max_steps):
        resp = client.models.generate_content(model=model, contents=contents, config=config)

        try:
            parts = resp.candidates[0].content.parts
            texts = [getattr(p, "text", None) for p in parts if getattr(p, "text", None)]
            if texts:
                log({"type": "assistant", "text": "\n".join(texts).strip()})
        except Exception:
            pass

        fcalls = []
        try:
            for p in resp.candidates[0].content.parts:
                fc = getattr(p, "function_call", None)
                if fc:
                    fcalls.append(fc)
        except Exception:
            pass

        if not fcalls:
            try:
                final = (getattr(resp, "text", "") or "").strip()
            except Exception:
                final = ""
            log({"type": "final", "text": final})
            return records

        contents.append(resp.candidates[0].content)
        response_parts = []

        for fc in fcalls:
            tool_name = fc.name
            tool_args = dict(fc.args or {})
            log({"type": "tool_call", "name": tool_name, "args": tool_args})

            tool_result = await session.call_tool(tool_name, tool_args)

            try:
                blocks = getattr(tool_result, "content", None) or []
                payload_data = {
                    "content": [
                        {"type": "text", "text": b.text}
                        if getattr(b, "type", None) == "text"
                        else {"type": str(getattr(b, "type", "unknown")), "value": str(b)}
                        for b in blocks
                    ],
                    "isError": getattr(tool_result, "isError", False),
                }
            except Exception:
                payload_data = {"raw": str(tool_result)}

            if ipi_source_tool and tool_name == ipi_source_tool and ipi_payload:
                content_list = payload_data.get("content", [])
                last_text_idx = next(
                    (i for i in reversed(range(len(content_list)))
                     if content_list[i].get("type") == "text"),
                    None
                )
                if last_text_idx is not None:
                    content_list[last_text_idx]["text"] += "\n" + ipi_payload
                else:
                    content_list.append({"type": "text", "text": ipi_payload})
                payload_data["content"] = content_list
                log({"type": "ipi_injected", "target_tool": tool_name, "payload": ipi_payload})

            log({"type": "tool_result", "name": tool_name, "result": payload_data})
            response_parts.append(
                genai_types.Part.from_function_response(
                    name=tool_name, response={"result": payload_data}
                )
            )

        contents.append(genai_types.Content(role="user", parts=response_parts))

    log({"type": "final", "text": "[ERROR] max_steps exceeded"})
    return records


# ══════════════════════════════════════════════════════════
# 임베딩
# ══════════════════════════════════════════════════════════

def extract_tool_seq_from_trace_text(text: str) -> str:
    for part in text.split(" | "):
        if part.startswith("TOOLS:"):
            return part.replace("TOOLS:", "").strip()
    return text


def embed_texts(texts: List[str]) -> np.ndarray:
    if HAS_SBERT:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.array(vecs)
    else:
        tool_seqs = [extract_tool_seq_from_trace_text(t) for t in texts]
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(tool_seqs).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms


# ══════════════════════════════════════════════════════════
# 1. 재현성 측정
# ══════════════════════════════════════════════════════════

def compute_reproducibility(
    traces: List[Dict[str, Any]],
    scenario_id: str,
) -> Dict[str, Any]:
    """
    N개 trace를 임베딩 → DBSCAN 클러스터링 → outlier 탐지.

    DBSCAN label:
      -1 = outlier (노이즈)
       0, 1, 2... = 클러스터

    정상: 모든 run이 같은 클러스터(label=0)
    비정상: label=-1(outlier) 또는 여러 클러스터
    """
    n = len(traces)
    if n < 2:
        return {"error": "trace 2개 미만", "reproducibility_score": None}

    texts = [trace_to_text(t) for t in traces]
    embeddings = embed_texts(texts)

    sim_matrix = embeddings @ embeddings.T
    sim_matrix = np.clip(sim_matrix, -1, 1)
    dist_matrix = 1 - sim_matrix

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    n_outliers = int(np.sum(labels == -1))
    n_clusters = len(set(labels) - {-1})
    outlier_runs = [i + 1 for i, l in enumerate(labels) if l == -1]

    if n_clusters == 0:
        repro_score = 0.0
    else:
        outlier_penalty = n_outliers / n
        cluster_penalty = (n_clusters - 1) / n
        repro_score = round(max(0.0, 1.0 - outlier_penalty - cluster_penalty), 4)

    run_details = []
    for i, (label, text) in enumerate(zip(labels, texts)):
        run_details.append({
            "run": i + 1,
            "cluster": int(label),
            "is_outlier": bool(label == -1),
            "trace_summary": text[:150],
        })

    main_cluster_indices = [i for i, l in enumerate(labels) if l == 0]
    if len(main_cluster_indices) >= 2:
        sub_sim = sim_matrix[np.ix_(main_cluster_indices, main_cluster_indices)]
        intra_sim = float(np.mean(sub_sim[np.triu_indices_from(sub_sim, k=1)]))
    else:
        intra_sim = None

    grade = (
        "A" if repro_score >= 0.9 else
        "B" if repro_score >= 0.75 else
        "C" if repro_score >= 0.5 else "D"
    )

    return {
        "scenario_id": scenario_id,
        "n_runs": n,
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "outlier_runs": outlier_runs,
        "intra_cluster_similarity": round(intra_sim, 4) if intra_sim else None,
        "reproducibility_score": repro_score,
        "grade": grade,
        "embedding_method": "sentence-transformers" if HAS_SBERT else "TF-IDF (fallback)",
        "run_details": run_details,
        "interpretation": (
            f"N={n}회 실행 중 {n_outliers}개 outlier 발생. "
            f"클러스터 {n_clusters}개. "
            f"{'동일한 프로세스로 실행됨.' if repro_score >= 0.9 else '실행 경로 불안정.'}"
        ),
    }


# ══════════════════════════════════════════════════════════
# N회 재실행 (Runner only)
# ══════════════════════════════════════════════════════════

async def run_n_times(
    *,
    task: Dict[str, Any],
    system_txt: str,
    mcp_server_cfg: Dict[str, Any],
    allowed_tools: List[str],
    gemini_client,
    model: str,
    n_runs: int,
    sleep_sec: float,
) -> List[Dict[str, Any]]:
    """
    동일 task를 N회 실행 → trace 목록 반환.
    """
    traces: List[Dict[str, Any]] = []

    if not HAS_MCP:
        raise RuntimeError("mcp package not installed")

    server_params = StdioServerParameters(
        command=mcp_server_cfg["command"],
        args=mcp_server_cfg.get("args", []),
        env=os.environ.copy(),
    )

    user_prompt = task["user"]
    ipi_source = task.get("ipi_source_tool")
    ipi_payload_str = task.get("ipi_payload")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_list = await session.list_tools()
            tool_by_name = {t.name: t for t in tool_list.tools}

            if allowed_tools:
                tool_by_name = {n: t for n, t in tool_by_name.items() if n in set(allowed_tools)}

            fn_decls = [
                genai_types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=t.inputSchema or {"type": "object", "properties": {}},
                )
                for t in tool_by_name.values()
            ]

            for i in range(n_runs):
                print(f"      run {i+1}/{n_runs} ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    records = await run_one_task_gemini(
                        client=gemini_client,
                        model=model,
                        system_instruction=system_txt,
                        session=session,
                        fn_decls=fn_decls,
                        user_prompt=user_prompt,
                        ipi_source_tool=ipi_source,
                        ipi_payload=ipi_payload_str,
                    )
                    trace = extract_trace(records)
                    traces.append(trace)
                    elapsed = round(time.time() - t0, 1)
                    print(f"done ({elapsed}s)")
                except Exception as e:
                    traces.append({})
                    print(f"ERR: {e}")

                if i < n_runs - 1:
                    time.sleep(sleep_sec)

    return traces


# ══════════════════════════════════════════════════════════
# 단일 시나리오 평가
# ══════════════════════════════════════════════════════════

async def evaluate_one_scenario(
    *,
    scenario_id: str,
    repo_root: Path,
    agent: str,
    scenario_date: str,
    log_date: str,
    n_runs: int,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
    sleep_sec: float,
    use_existing_logs: bool,
) -> Dict[str, Any]:
    """
    단일 시나리오에 대해 재현성 측정.
    use_existing_logs=True면 기존 로그만 사용 (재실행 없음).
    """
    scenarios_dir = (
        repo_root / "red_teaming" / "CCS" / "generated_scenarios" / agent / scenario_date
    )
    logs_dir = (
        repo_root / "red_teaming" / "CCS" / "run" / "logs" / agent / "attack" / log_date
    )

    threat_id_m = re.match(r"^(T\d+)-C\d+-S\d+$", scenario_id)
    if not threat_id_m:
        return {"scenario_id": scenario_id, "error": "scenario_id 형식 오류"}
    threat_id = threat_id_m.group(1)

    oracle_files = sorted(scenarios_dir.glob(f"{threat_id}_*.json"))
    if not oracle_files:
        return {"scenario_id": scenario_id, "error": f"oracle 파일 없음: {threat_id}_*.json"}

    oracle_data = load_json(oracle_files[0])
    scenario = next(
        (sc for sc in (oracle_data.get("scenarios") or [])
         if str(sc.get("시나리오_id", "")).strip() == scenario_id),
        None
    )
    if scenario is None:
        return {"scenario_id": scenario_id, "error": "oracle에서 시나리오 못 찾음"}

    if use_existing_logs:
        log_files = sorted(logs_dir.glob(f"attack-{scenario_id}_*.jsonl"))
        if not log_files:
            return {"scenario_id": scenario_id,
                    "error": f"로그 파일 없음: attack-{scenario_id}_*.jsonl"}

        print(f"    기존 로그 {len(log_files)}개 사용: {scenario_id}")
        traces = []
        for lf in log_files[:n_runs]:
            records = load_jsonl(lf)
            trace = extract_trace(records)
            traces.append(trace)
    else:
        print(f"    재실행 {n_runs}회: {scenario_id}")

        task = {
            "id": f"attack-{scenario_id}",
            "user": scenario.get("공격_프롬프트", ""),
            "ipi_source_tool": None,
            "ipi_payload": None,
        }
        attack_prompt = str(scenario.get("공격_프롬프트", ""))
        ipi_m = re.match(r"^(.*?)\s*\[IPI:([A-Za-z_][A-Za-z0-9_]*)\]\s*(.+)$",
                          attack_prompt.strip(), re.DOTALL)
        if ipi_m:
            task["user"] = ipi_m.group(1).strip()
            task["ipi_source_tool"] = ipi_m.group(2).strip()
            task["ipi_payload"] = ipi_m.group(3).strip()

        tools_path = (
            repo_root / "red_teaming" / "CCS" / "generated_tasks"
            / agent / scenario_date / "allow_tool.json"
        )
        if not tools_path.exists():
            return {"scenario_id": scenario_id, "error": f"allow_tool.json 없음: {tools_path}"}

        tools_cfg = load_json(tools_path)
        mcp_server_cfg = tools_cfg["mcp_server"]
        allowed_tools = tools_cfg.get("allowed_tools", [])

        profile_path = (
            repo_root / "red_teaming" / "CCS" / "agent_profiles"
            / agent / scenario_date / "agent_profile.yaml"
        )
        if not profile_path.exists():
            return {"scenario_id": scenario_id, "error": "agent_profile.yaml 없음"}
        profile_data = load_yaml_file(profile_path)
        system_txt = profile_data.get("system_prompt", "")

        gemini_client = genai.Client(api_key=api_key) if genai else None
        if gemini_client is None:
            return {"scenario_id": scenario_id, "error": "google-genai not installed"}

        import asyncio
        traces = await run_n_times(
            task=task,
            system_txt=system_txt,
            mcp_server_cfg=mcp_server_cfg,
            allowed_tools=allowed_tools,
            gemini_client=gemini_client,
            model=model,
            n_runs=n_runs,
            sleep_sec=sleep_sec,
        )

    repro = compute_reproducibility(
        [t for t in traces if t],
        scenario_id=scenario_id,
    )

    return {
        "scenario_id": scenario_id,
        "n_runs": len(traces),
        "reproducibility": repro,
    }


# ══════════════════════════════════════════════════════════
# 단일 Agent 전체 평가
# ══════════════════════════════════════════════════════════

async def evaluate_one_agent(
    *,
    agent: str,
    repo_root: Path,
    n_runs: int,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
    sleep_sec: float,
    scenario_id_filter: Optional[str],
    use_existing_logs: bool,
) -> Dict[str, Any]:

    print(f"\n{'='*60}")
    print(f"[Agent] {agent}")
    print(f"{'='*60}")

    scenarios_base = repo_root / "red_teaming" / "CCS" / "generated_scenarios" / agent
    logs_base      = repo_root / "red_teaming" / "CCS" / "run" / "logs" / agent / "attack"
    out_dir        = repo_root / "red_teaming" / "CCS" / "reliability_reports" / agent

    scenario_date = get_latest_date(scenarios_base)
    log_date      = get_latest_date(logs_base)

    if not scenario_date:
        return {"agent": agent, "error": "generated_scenarios 없음"}
    if not log_date and use_existing_logs:
        return {"agent": agent, "error": "로그 없음 (--rerun 옵션으로 재실행 가능)"}

    logs_dir = logs_base / log_date if log_date else None
    scenario_ids: List[str] = []

    if scenario_id_filter:
        scenario_ids = [scenario_id_filter]
    elif logs_dir and logs_dir.exists():
        for lf in sorted(logs_dir.glob("attack-*.jsonl")):
            m = re.match(r"attack-(T\d+-C\d+-S\d+)_\d+\.jsonl$", lf.name)
            if m and m.group(1) not in scenario_ids:
                scenario_ids.append(m.group(1))
    else:
        return {"agent": agent, "error": "평가 대상 시나리오 없음"}

    print(f"  대상 시나리오: {len(scenario_ids)}개")
    print(f"  실행 방식: {'기존 로그 사용' if use_existing_logs else f'{n_runs}회 재실행'}")
    print(f"  embedding: {'sentence-transformers' if HAS_SBERT else 'TF-IDF fallback'}")

    results = []
    for sid in scenario_ids:
        print(f"\n  [{sid}]")
        result = await evaluate_one_scenario(
            scenario_id=sid,
            repo_root=repo_root,
            agent=agent,
            scenario_date=scenario_date,
            log_date=log_date or "",
            n_runs=n_runs,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            sleep_sec=sleep_sec,
            use_existing_logs=use_existing_logs,
        )
        results.append(result)

        if "error" not in result:
            r = result["reproducibility"]
            print(f"    재현성:  score={r.get('reproducibility_score')}  "
                  f"outlier={r.get('n_outliers')}회  grade={r.get('grade')}")

    valid_results = [r for r in results if "error" not in r]
    repro_scores = [r["reproducibility"].get("reproducibility_score")
                    for r in valid_results
                    if r["reproducibility"].get("reproducibility_score") is not None]

    report = {
        "agent": agent,
        "scenario_date": scenario_date,
        "log_date": log_date,
        "evaluated_at": utc_now_iso(),
        "n_runs": n_runs,
        "provider": provider,
        "model": model,
        "summary": {
            "n_scenarios": len(scenario_ids),
            "n_evaluated": len(valid_results),
            "reproducibility": {
                "mean_score": round(sum(repro_scores) / len(repro_scores), 4) if repro_scores else None,
                "interpretation": "1.0=완전 재현. outlier 없고 클러스터 1개.",
            },
        },
        "per_scenario": results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reliability_report_{agent}_{scenario_date}.json"
    save_json(out_path, report)

    return {**report, "_out_path": str(out_path)}


# ══════════════════════════════════════════════════════════
# 콘솔 최종 요약
# ══════════════════════════════════════════════════════════

def print_final_summary(reports: List[Dict[str, Any]]) -> None:
    print(f"\n{'='*70}")
    print("최종 요약")
    print(f"{'='*70}")
    print(f"{'Agent':35s} {'재현성':>8s} {'평가수':>6s}")
    print(f"{'-'*70}")

    for r in reports:
        if "error" in r and "summary" not in r:
            print(f"  ⚠  {r['agent']:33s}  ERROR: {r.get('error','')}")
            continue
        s = r.get("summary", {})
        repro_score = s.get("reproducibility", {}).get("mean_score")
        n_eval = s.get("n_evaluated", 0)

        mark = "✅" if repro_score and repro_score >= 0.75 else "⚠️"
        print(f"  {mark} {r['agent']:33s} "
              f"{str(repro_score or 'N/A'):>8s} "
              f"{n_eval:>6d}")

    print(f"{'='*70}")
    print()
    print("해석 기준:")
    print("  재현성 ≥ 0.9 → 실행 경로 안정 (신뢰 가능)")
    print()
    for r in reports:
        if "_out_path" in r:
            print(f"  리포트: {r['_out_path']}")


# ══════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════

def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(
        description="CCS 신뢰도 평가 (재현성)"
    )
    parser.add_argument("--agent", default="",
                        help="특정 agent만. 미지정 시 전체.")
    parser.add_argument("--scenario-id", default="",
                        help="특정 시나리오만. 예: T1-C1-S1")
    parser.add_argument("--runs", type=int, default=5,
                        help="재실행 횟수 (기본 5).")
    parser.add_argument("--provider", default="gemini",
                        choices=["gemini", "openai", "openai_compat", "ollama"])
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--rerun", action="store_true",
                        help="설정 시 Runner 재실행. 미설정 시 기존 로그 사용.")
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = (
        Path(args.repo_root).expanduser()
        if args.repo_root.strip()
        else repo_root_from_this_file()
    )

    try:
        api_key = load_api_key(repo_root, args.provider)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    profiles_root = repo_root / "red_teaming" / "CCS" / "agent_profiles"
    agents = [args.agent.strip()] if args.agent.strip() else discover_agents(profiles_root)
    if not agents:
        print(f"[ERROR] agent 없음: {profiles_root}")
        sys.exit(1)

    print(f"{'='*60}")
    print("CCS 신뢰도 평가 (재현성)")
    print(f"  agents  : {len(agents)}개")
    print(f"  runs    : {args.runs}회")
    print(f"  rerun   : {args.rerun}")
    print(f"  model   : {args.model}")
    print(f"  embed   : {'sentence-transformers' if HAS_SBERT else 'TF-IDF (fallback)'}")
    print(f"{'='*60}")

    reports = []
    for agent in agents:
        report = asyncio.run(evaluate_one_agent(
            agent=agent,
            repo_root=repo_root,
            n_runs=args.runs,
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            sleep_sec=args.sleep,
            scenario_id_filter=args.scenario_id.strip() or None,
            use_existing_logs=not args.rerun,
        ))
        reports.append(report)

    print_final_summary(reports)


if __name__ == "__main__":
    main()