#!/usr/bin/env python3
"""
5회 반복 실행 래퍼
- 429 에러 감지 시 해당 회차에서 생성된 로그 파일 삭제 후 재시작
"""

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = "red_teaming/CCS/CCS_Run_Agent_Vulnerability_Analysis.py"
AGENTS = [
    "banking_cs_agent",
    "ecommerce_operations_agent",
    "education_admin_agent",
    "government_service_agent",
    "hr_onboarding_agent",
    "insurance_claims_agent",
    "logistics_operations_agent",
    "medical_consultation_agent",
    "telecom_cs_agent",
    "travel_reservation_agent",
]
LOG_BASE = Path("red_teaming/CCS/run/logs")
TOTAL_RUNS = 5
RETRY_SLEEP = 30  # 429 후 대기 초


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def delete_logs_since(start_time: float, today: str) -> int:
    deleted = 0
    for agent in AGENTS:
        log_dir = LOG_BASE / agent / "attack" / today
        if not log_dir.exists():
            continue
        for f in log_dir.rglob("*.jsonl"):
            try:
                if f.stat().st_mtime >= start_time:
                    print(f"  [삭제] {f}")
                    f.unlink()
                    deleted += 1
            except OSError:
                pass
    return deleted


def is_429(output: str) -> bool:
    keywords = ["429", "too many requests", "quota", "rate limit", "ratelimit", "resource_exhausted"]
    lower = output.lower()
    return any(k in lower for k in keywords)


def run_once(run_num: int) -> bool:
    """실행 한 회차. 성공 시 True, 429 감지 시 False 반환."""
    today = today_utc()
    print(f"\n{'='*60}")
    print(f"  실행 #{run_num} / {TOTAL_RUNS}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}")

    start_time = time.time()

    cmd = [sys.executable, SCRIPT, "--agent"] + AGENTS
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)

    proc.wait()
    full_output = "".join(output_lines)

    if is_429(full_output):
        print(f"\n[!] 429 Rate Limit 감지 — 이번 회차 로그 삭제 중...")
        deleted = delete_logs_since(start_time, today)
        print(f"[!] {deleted}개 파일 삭제 완료. {RETRY_SLEEP}초 후 재시작...")
        time.sleep(RETRY_SLEEP)
        return False

    return True


def main():
    completed = 0
    while completed < TOTAL_RUNS:
        run_num = completed + 1
        success = run_once(run_num)
        if success:
            completed += 1
            print(f"\n[완료] 실행 #{run_num} 성공 (누적: {completed}/{TOTAL_RUNS})")
        else:
            print(f"[재시작] 실행 #{run_num} 재시도...")

    print(f"\n{'='*60}")
    print(f"  모든 {TOTAL_RUNS}회 실행 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
