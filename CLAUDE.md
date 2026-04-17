# Agent AI Security — CCS Red Teaming

## 프로젝트 개요
AI 에이전트의 보안 취약점(Prompt Injection, 권한 남용 등)을 자동으로 평가하는 Red Teaming 프레임워크.

## 가상환경
```bash
source agent/Scripts/activate
```

## 주요 실행 명령어

### 취약점 분석 실행 (10개 에이전트 일괄)
```bash
python red_teaming/CCS/CCS_Run_Agent_Vulnerability_Analysis.py \
  --agent banking_cs_agent ecommerce_operations_agent education_admin_agent \
  government_service_agent hr_onboarding_agent insurance_claims_agent \
  logistics_operations_agent medical_consultation_agent telecom_cs_agent \
  travel_reservation_agent
```

### 5회 반복 실행 규칙
- 총 **5회** 연속 실행
- 실행 중 **429 에러(Rate Limit)** 발생 시:
  1. 해당 실행에서 생성된 로그 파일 전부 삭제
  2. 해당 회차를 처음부터 재시작
- 로그 경로: `red_teaming/CCS/run/logs/<agent>/attack/<YYYY-MM-DD>/S4/`

### 시나리오 생성
```bash
python red_teaming/CCS/CCS_Generate_Adversarial_Evaluation_Scenario.py
```

## 디렉토리 구조
```
configs/CCS/          # 에이전트별 YAML 설정 (LLM, MCP 서버, 경로 등)
red_teaming/CCS/
  generated_tasks/    # 태스크 JSONL (날짜별 서브디렉토리)
  generated_scenarios/# 시나리오 파일
  run/logs/           # 실행 로그 JSONL
agent_profiles/       # 에이전트 시스템 프롬프트
```

## 에러 대응
| 에러 | 원인 | 조치 |
|------|------|------|
| 429 Too Many Requests | API Rate Limit | 해당 실행 로그 삭제 후 재시작 |
| missing tools [...] | MCP 툴 미등록 | configs/CCS/<agent>.yml 확인 |
| <DATE> 미해석 | generated_tasks 없음 | 시나리오/태스크 생성 먼저 실행 |
