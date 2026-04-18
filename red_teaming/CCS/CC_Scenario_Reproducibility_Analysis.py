"""
시나리오 생성 재현성 분석 스크립트

1. 각 에이전트의 생성된 시나리오를 로드
2. SentenceTransformer로 임베딩
3. 각 (에이전트, case)별 재현성 지표 계산
4. 대표 에이전트 선정 (평균 재현성이 중앙값에 가장 가까운 에이전트)
5. 대표 에이전트의 UMAP scatter plot 생성
6. 10개 에이전트 집계 표 출력

설치 필요:
    pip install sentence-transformers umap-learn scikit-learn pandas matplotlib
"""

from __future__ import annotations
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG — 실제 환경에 맞춰 이 부분만 수정하세요
# ============================================================

# 시나리오 파일이 있는 루트 경로
# 예상 구조:
#   BASE_PATH/
#     {agent_name}/
#       scenarios/
#         T1-C1-S1.json
#         T1-C1-S2.json
#         ...
BASE_PATH = Path("./data/agents")

# 파일명 패턴 (정규식) - T{숫자}-C{숫자}-S{숫자} 형태를 추출
FILENAME_PATTERN = re.compile(r"(T\d+)-(C\d+)-(S\d+)")

# 임베딩에 쓸 시나리오 JSON 필드 (여러 개면 join 됨)
TEXT_FIELDS = ["공격_조건", "가정사항_패턴", "판정_기준"]

# 임베딩 모델 (한국어/영어 혼합이면 multilingual 권장)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Outlier 판정: 클러스터 중심에서 (mean + n*std) 이상 떨어진 점
OUTLIER_SIGMA = 1.5

# UMAP 파라미터
UMAP_N_NEIGHBORS = 5
UMAP_MIN_DIST = 0.3
UMAP_RANDOM_STATE = 42

# 출력 경로
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 위협별 색 (필요시 추가/수정)
THREAT_COLORS = {
    "T1": "#534AB7",   # purple
    "T2": "#1D9E75",   # teal
    "T3": "#D85A30",   # coral
    "T4": "#BA7517",   # amber
    "T5": "#D4537E",   # pink
    "T6": "#185FA5",   # blue
    "T7": "#639922",   # green
    "T9": "#5F5E5A",   # gray
    "T10": "#A32D2D",  # red
    "T15": "#0F6E56",  # dark teal
    "T17": "#7F77DD",  # lavender
}

# case 번호별 마커 (C1=원, C2=삼각형)
CASE_MARKERS = {"C1": "o", "C2": "^", "C3": "s"}


# ============================================================
# 1. 데이터 로딩
# ============================================================

def load_scenarios(base_path: Path) -> pd.DataFrame:
    """
    디렉토리를 순회하며 시나리오 JSON을 로드한다.

    반환: DataFrame with columns
        - agent: 에이전트 이름
        - threat: T1, T2, ...
        - case_id: T1-C1, T1-C2, ...
        - scenario_id: T1-C1-S1, ...
        - text: 임베딩 대상 텍스트
    """
    rows = []
    for agent_dir in sorted(base_path.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name

        # 시나리오 파일 경로 - 실제 구조에 맞게 조정
        scenarios_dir = agent_dir / "scenarios"
        if not scenarios_dir.exists():
            # 다른 위치에 있을 수도 있으니 직접 탐색
            scenarios_dir = agent_dir

        for scenario_file in scenarios_dir.rglob("*.json"):
            match = FILENAME_PATTERN.search(scenario_file.stem)
            if not match:
                continue
            threat, case_num, scenario_num = match.groups()
            case_id = f"{threat}-{case_num}"
            scenario_id = f"{threat}-{case_num}-{scenario_num}"

            try:
                with open(scenario_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [skip] {scenario_file.name}: {e}")
                continue

            # 텍스트 추출 - dict나 list 모두 처리
            texts = []
            for field in TEXT_FIELDS:
                value = _get_nested(data, field)
                if value:
                    texts.append(str(value))
            text = " ".join(texts).strip()
            if not text:
                continue

            rows.append({
                "agent": agent,
                "threat": threat,
                "case_id": case_id,
                "case_num": case_num,
                "scenario_id": scenario_id,
                "text": text,
            })

    df = pd.DataFrame(rows)
    print(f"[load] {len(df)} scenarios, {df['agent'].nunique()} agents, "
          f"{df['case_id'].nunique()} cases")
    return df


def _get_nested(data, field):
    """중첩 JSON에서도 field를 찾아 반환."""
    if isinstance(data, dict):
        if field in data:
            return data[field]
        for v in data.values():
            found = _get_nested(v, field)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _get_nested(item, field)
            if found:
                return found
    return None


# ============================================================
# 2. 임베딩
# ============================================================

def embed_scenarios(df: pd.DataFrame, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """시나리오 텍스트를 임베딩 (정규화 포함)."""
    print(f"[embed] loading model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity = dot product
        batch_size=32,
    )
    print(f"[embed] shape={embeddings.shape}")
    return embeddings


# ============================================================
# 3. (agent, case)별 재현성 계산
# ============================================================

def detect_outliers(embeddings: np.ndarray, sigma: float = OUTLIER_SIGMA) -> np.ndarray:
    """
    클러스터 중심으로부터 cosine distance 기준 outlier 검출.
    반환: boolean array (True = outlier).
    """
    n = len(embeddings)
    if n < 3:
        return np.zeros(n, dtype=bool)

    centroid = embeddings.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

    sims = embeddings @ centroid_norm  # cosine similarity (정규화 가정)
    dists = 1 - sims
    threshold = dists.mean() + sigma * dists.std()
    return dists > threshold


def compute_case_reproducibility(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    (agent, case_id)별 재현성 지표 계산.

    - intra_sim: 그룹 내 평균 pairwise cosine similarity
    - n_outliers: outlier 개수
    - reproducibility: (n - n_outliers) / n
    """
    results = []
    for (agent, case_id), group in df.groupby(["agent", "case_id"]):
        idx = group.index.to_numpy()
        embs = embeddings[idx]
        n = len(embs)

        # Pairwise cosine similarity (정규화된 임베딩의 내적)
        sim_matrix = embs @ embs.T
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        intra_sim = float(sim_matrix[mask].mean()) if mask.any() else 1.0

        outliers = detect_outliers(embs)
        n_outliers = int(outliers.sum())
        reproducibility = (n - n_outliers) / n

        results.append({
            "agent": agent,
            "case_id": case_id,
            "threat": case_id.split("-")[0],
            "n_scenarios": n,
            "intra_sim": round(intra_sim, 4),
            "n_outliers": n_outliers,
            "reproducibility": round(reproducibility, 4),
        })

    return pd.DataFrame(results)


# ============================================================
# 4. 에이전트별 집계 + 대표 선정
# ============================================================

def compute_agent_summary(
    case_df: pd.DataFrame, df: pd.DataFrame, embeddings: np.ndarray
) -> pd.DataFrame:
    """에이전트별 mean reproducibility, outlier ratio, silhouette."""
    rows = []
    for agent, grp in case_df.groupby("agent"):
        mean_repro = grp["reproducibility"].mean()
        outlier_ratio = grp["n_outliers"].sum() / grp["n_scenarios"].sum()

        # Silhouette: case_id를 레이블로 사용
        mask = (df["agent"] == agent).values
        agent_embs = embeddings[mask]
        agent_labels = df.loc[mask, "case_id"].values
        n_unique_cases = len(set(agent_labels))

        if n_unique_cases > 1 and len(agent_embs) > n_unique_cases:
            sil = float(silhouette_score(agent_embs, agent_labels, metric="cosine"))
        else:
            sil = float("nan")

        rows.append({
            "agent": agent,
            "mean_reproducibility": round(mean_repro, 4),
            "outlier_ratio": round(outlier_ratio, 4),
            "silhouette": round(sil, 4) if not np.isnan(sil) else None,
        })

    return pd.DataFrame(rows).sort_values("mean_reproducibility", ascending=False).reset_index(drop=True)


def select_representative_agent(agent_summary: pd.DataFrame) -> str:
    """평균 재현성이 중앙값에 가장 가까운 에이전트 반환."""
    median_score = agent_summary["mean_reproducibility"].median()
    distances = (agent_summary["mean_reproducibility"] - median_score).abs()
    rep_agent = agent_summary.loc[distances.idxmin(), "agent"]
    print(f"[rep] median={median_score:.4f}, selected={rep_agent}")
    return rep_agent


# ============================================================
# 5. 대표 에이전트 scatter plot
# ============================================================

def plot_representative_scatter(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    agent: str,
    save_path: Path,
):
    """대표 에이전트의 UMAP 2D scatter plot 저장."""
    mask = (df["agent"] == agent).values
    sub_df = df[mask].reset_index(drop=True)
    sub_embs = embeddings[mask]

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
        metric="cosine",
    )
    coords = reducer.fit_transform(sub_embs)
    sub_df["x"] = coords[:, 0]
    sub_df["y"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    threats = sorted(sub_df["threat"].unique())
    cases = sorted(sub_df["case_num"].unique())

    # 포인트 찍기
    for threat in threats:
        color = THREAT_COLORS.get(threat, "#888888")
        for case in cases:
            marker = CASE_MARKERS.get(case, "D")
            sel = (sub_df["threat"] == threat) & (sub_df["case_num"] == case)
            if not sel.any():
                continue
            ax.scatter(
                sub_df.loc[sel, "x"], sub_df.loc[sel, "y"],
                c=color, marker=marker, s=70,
                edgecolors="white", linewidths=0.6, alpha=0.9,
                zorder=3,
            )

    # 클러스터 중심에 라벨
    for case_id, grp in sub_df.groupby("case_id"):
        cx, cy = grp["x"].mean(), grp["y"].mean()
        ax.annotate(
            case_id, (cx, cy), xytext=(0, 12),
            textcoords="offset points", ha="center",
            fontsize=9, weight="medium",
            color="#333", alpha=0.85, zorder=4,
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=THREAT_COLORS.get(t, "#888"),
               markersize=9, label=t, markeredgecolor="white")
        for t in threats
    ]
    legend_elements.append(Line2D([0], [0], marker="", color="w", label=""))
    for case in cases:
        legend_elements.append(
            Line2D([0], [0], marker=CASE_MARKERS.get(case, "D"), color="w",
                   markerfacecolor="#888", markersize=9,
                   label=case, markeredgecolor="white")
        )
    ax.legend(
        handles=legend_elements, loc="center left",
        bbox_to_anchor=(1.01, 0.5), fontsize=9,
        frameon=False, labelspacing=0.6,
    )

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(
        f"Scenario Generation Reproducibility — {agent} (representative)",
        fontsize=12, pad=12,
    )
    ax.grid(True, alpha=0.08, zorder=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    # 1. 로드
    df = load_scenarios(BASE_PATH)
    if df.empty:
        print("[error] 시나리오를 찾지 못했습니다. BASE_PATH를 확인하세요.")
        return

    # 2. 임베딩
    embeddings = embed_scenarios(df)

    # 3. case별 재현성
    case_df = compute_case_reproducibility(df, embeddings)
    case_df.to_csv(OUTPUT_DIR / "case_reproducibility.csv", index=False, encoding="utf-8-sig")
    print(f"[save] case-level: {OUTPUT_DIR/'case_reproducibility.csv'}")

    # 4. agent 집계
    agent_summary = compute_agent_summary(case_df, df, embeddings)
    agent_summary.to_csv(OUTPUT_DIR / "agent_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[save] agent-level: {OUTPUT_DIR/'agent_summary.csv'}")
    print("\n=== Agent Summary ===")
    print(agent_summary.to_string(index=False))

    # 5. 대표 에이전트 선정
    rep_agent = select_representative_agent(agent_summary)

    # 6. 대표 scatter
    plot_representative_scatter(
        df, embeddings, rep_agent,
        OUTPUT_DIR / f"scatter_{rep_agent}.png",
    )

    # 최악 에이전트도 부록용으로 뽑기
    worst_agent = agent_summary.iloc[-1]["agent"]
    if worst_agent != rep_agent:
        plot_representative_scatter(
            df, embeddings, worst_agent,
            OUTPUT_DIR / f"scatter_{worst_agent}_worst.png",
        )

    # 최종 요약
    print("\n=== Final Summary ===")
    print(f"Total scenarios:  {len(df)}")
    print(f"Total agents:     {df['agent'].nunique()}")
    print(f"Mean reproducibility (across agents): "
          f"{agent_summary['mean_reproducibility'].mean():.4f} "
          f"(σ={agent_summary['mean_reproducibility'].std():.4f})")
    print(f"Representative agent: {rep_agent}")
    print(f"Worst agent:          {worst_agent}")


if __name__ == "__main__":
    main()