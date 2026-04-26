#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generated_scenarios/{agent}/{date}/{suffix}/*.json 에서
tasks_attack.jsonl만 다시 생성한다.

사용법:
  python CCS_Rebuild_Tasks.py --agent banking_cs_agent --date 2026-04-19 --suffix S5
  python CCS_Rebuild_Tasks.py --agent banking_cs_agent --date 2026-04-19  # suffix 기본값 S5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_REPO_ROOT = Path(r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security")


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_latest_profile_date(agent_profiles_dir: Path, agent: str) -> Optional[str]:
    base = agent_profiles_dir / agent
    if not base.exists():
        return None
    dates = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", d.name):
            if (d / "agent_profile.yaml").exists():
                dates.append(d.name)
    return dates[-1] if dates else None


def extract_known_tools(repo_root: Path, agent: str, profile_date: str) -> List[str]:
    profile_path = repo_root / "red_teaming" / "CCS" / "agent_profiles" / agent / profile_date / "agent_profile.yaml"
    if not profile_path.exists():
        raise RuntimeError(f"agent profile not found: {profile_path}")

    profile = load_yaml(profile_path)
    tools = profile.get("tools") or []
    names: List[str] = []
    for t in tools:
        if isinstance(t, dict):
            name = str(t.get("name", "")).strip()
            if name and name not in names:
                names.append(name)
    return names


IPI_PATTERN = re.compile(r"^(.*?)\s*\[IPI:([A-Za-z_][A-Za-z0-9_]*)\]\s*(.+)$", re.DOTALL)


def parse_ipi_from_attack_prompt(attack_prompt: str):
    m = IPI_PATTERN.match(attack_prompt.strip())
    if m:
        user_input = m.group(1).strip()
        ipi_tool = m.group(2).strip()
        ipi_payload = m.group(3).strip()
        if user_input and ipi_tool and ipi_payload:
            return user_input, ipi_tool, ipi_payload
    return attack_prompt, None, None


def parse_expected_tools(flow_text: str, known_tools: List[str]) -> List[str]:
    if not flow_text:
        return []
    found: List[str] = []
    for tool in known_tools:
        if tool in flow_text and tool not in found:
            found.append(tool)
    return found


def rebuild_tasks(
    *,
    repo_root: Path,
    agent: str,
    date: str,
    suffix: str,
) -> Path:
    scenario_dir = repo_root / "red_teaming" / "CCS" / "generated_scenarios" / agent / date / suffix
    if not scenario_dir.exists():
        raise FileNotFoundError(f"시나리오 폴더 없음: {scenario_dir}")

    # agent profile 최신 날짜
    agent_profiles_dir = repo_root / "red_teaming" / "CCS" / "agent_profiles"
    profile_date = get_latest_profile_date(agent_profiles_dir, agent)
    if not profile_date:
        raise RuntimeError(f"agent profile 없음: {agent}")

    known_tools = extract_known_tools(repo_root, agent, profile_date)
    print(f"  [profile] {agent} / {profile_date} | 도구 {len(known_tools)}개")

    tasks: List[Dict[str, Any]] = []
    json_files = sorted(scenario_dir.glob("*.json"))
    print(f"  [시나리오] {len(json_files)}개 파일 발견")

    for path in json_files:
        data = load_json(path)
        scenarios = data.get("scenarios") or []
        if not isinstance(scenarios, list):
            continue

        for sc in scenarios:
            scenario_id = str(sc.get("시나리오_id", "")).strip()
            attack_prompt = str(sc.get("공격_프롬프트", "")).strip()
            attack_flow = str(sc.get("공격_흐름", "")).strip()
            normal_flow = str(sc.get("정상_흐름", "")).strip()
            case_id = str(sc.get("case_id", "")).strip()

            if not scenario_id or not attack_prompt:
                continue

            user_input, ipi_tool, ipi_payload = parse_ipi_from_attack_prompt(attack_prompt)

            expected_tools = parse_expected_tools(attack_flow, known_tools)
            if not expected_tools:
                expected_tools = parse_expected_tools(normal_flow, known_tools)
            if not expected_tools:
                expected_tools = known_tools[:]
                print(f"    [WARN] {scenario_id}: expected_tools 없음, 전체 도구 사용")

            task: Dict[str, Any] = {
                "id": f"attack-{scenario_id}",
                "case_id": case_id,
                "user": user_input,
                "expected_tools": expected_tools,
            }
            if ipi_tool:
                task["ipi_source_tool"] = ipi_tool
                task["ipi_payload"] = ipi_payload

            tasks.append(task)
            print(f"    [OK] {scenario_id}")

    out_dir = repo_root / "red_teaming" / "CCS" / "generated_tasks" / agent / date
    out_path = out_dir / "tasks_attack.jsonl"
    write_jsonl(out_path, tasks)
    print(f"\n  [저장] {len(tasks)}개 task → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="tasks_attack.jsonl 재생성")
    parser.add_argument("--agent", required=True, help="에이전트 이름")
    parser.add_argument("--date", required=True, help="시나리오 날짜 (예: 2026-04-19)")
    parser.add_argument("--suffix", default="S5", help="시나리오 서브폴더 (기본값: S5)")
    parser.add_argument("--repo-root", default="", help="프로젝트 루트. 미지정 시 자동 탐지.")
    args = parser.parse_args()

    if args.repo_root.strip():
        repo_root = Path(args.repo_root).expanduser()
    else:
        repo_root = DEFAULT_REPO_ROOT
        if not repo_root.exists():
            repo_root = repo_root_from_this_file()

    print(f"{'='*60}")
    print(f"tasks_attack.jsonl 재생성")
    print(f"  agent:  {args.agent}")
    print(f"  date:   {args.date}")
    print(f"  suffix: {args.suffix}")
    print(f"  root:   {repo_root}")
    print(f"{'='*60}\n")

    try:
        out_path = rebuild_tasks(
            repo_root=repo_root,
            agent=args.agent,
            date=args.date,
            suffix=args.suffix,
        )
        print(f"\n[완료] {out_path}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()