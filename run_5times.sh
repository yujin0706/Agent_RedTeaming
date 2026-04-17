#!/usr/bin/env bash
# 5회 실행 래퍼 — 429 에러 시 해당 회차 로그 삭제 후 재시작

set -euo pipefail

SCRIPT="red_teaming/CCS/CCS_Run_Agent_Vulnerability_Analysis.py"
AGENTS="banking_cs_agent ecommerce_operations_agent education_admin_agent government_service_agent hr_onboarding_agent insurance_claims_agent logistics_operations_agent medical_consultation_agent telecom_cs_agent travel_reservation_agent"
LOG_BASE="red_teaming/CCS/run/logs"
TODAY=$(date -u +%Y-%m-%d)
TOTAL_RUNS=5

completed=0

while [ "$completed" -lt "$TOTAL_RUNS" ]; do
    run_num=$((completed + 1))
    echo ""
    echo "============================================================"
    echo "  실행 #${run_num} / ${TOTAL_RUNS}  ($(date -u '+%Y-%m-%d %H:%M:%S UTC'))"
    echo "============================================================"

    # 이번 실행 전 타임스탬프 기록
    start_ts=$(date -u +%s)

    # 실행 및 출력 캡처
    tmp_log=$(mktemp)
    set +e
    python "$SCRIPT" --agent $AGENTS 2>&1 | tee "$tmp_log"
    exit_code=$?
    set -e

    output=$(cat "$tmp_log")
    rm -f "$tmp_log"

    # 429 에러 확인
    if echo "$output" | grep -qi "429\|Too Many Requests\|quota\|rate.limit"; then
        echo ""
        echo "[!] 429 Rate Limit 감지 — 이번 회차 로그 삭제 후 재시작"

        # 이번 실행에서 생성된 로그 파일 삭제 (start_ts 이후 생성된 파일)
        for agent in $AGENTS; do
            dir="$LOG_BASE/$agent/attack/$TODAY"
            if [ -d "$dir" ]; then
                find "$dir" -name "*.jsonl" -newer /proc/self/fd/0 2>/dev/null | while read -r f; do
                    echo "  삭제: $f"
                    rm -f "$f"
                done
                # find -newer 대신 수정 시간 기반으로 삭제
                find "$dir" -name "*.jsonl" | while read -r f; do
                    file_ts=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)
                    if [ "$file_ts" -ge "$start_ts" ]; then
                        echo "  삭제: $f"
                        rm -f "$f"
                    fi
                done
            fi
        done

        echo "[!] 재시작합니다..."
        sleep 30
        continue
    fi

    completed=$((completed + 1))
    echo ""
    echo "[완료] 실행 #${run_num} 성공 (누적: ${completed}/${TOTAL_RUNS})"
done

echo ""
echo "============================================================"
echo "  모든 ${TOTAL_RUNS}회 실행 완료!"
echo "============================================================"
