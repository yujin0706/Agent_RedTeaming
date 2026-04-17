#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCS Attack Trace 임베딩 시각화
- tool sequence / final 텍스트를 각각 임베딩 후 가중 평균 벡터 생성
- UMAP으로 2D 축소
- T번호별 색상, C번호별 마커로 scatter plot 시각화
- KNN 기반 그룹 내 outlier 탐지

사용법:
  python CCS_Embedding_Visualizer.py
  python CCS_Embedding_Visualizer.py --tool-weight 0.7 --final-weight 0.3 --threshold 0.035
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("pip install openai")

try:
    import umap
except ImportError:
    raise RuntimeError("pip install umap-learn")

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_distances
except ImportError:
    raise RuntimeError("pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
except ImportError:
    raise RuntimeError("pip install matplotlib")

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────

DEFAULT_KEY_PATH = r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security\API_Key\OpenAI_key.txt"
DEFAULT_LOGS_DIR = r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security\red_teaming\CCS\run\logs\banking_cs_agent\attack\2026-04-15\S1"


# ─────────────────────────────────────────────
# JSONL 파싱
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_trace(records: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    tool sequence 텍스트와 final 텍스트를 분리 추출.
    Returns: (tool_text, final_text)
    """
    tool_calls = [r["name"] for r in records if r.get("type") == "tool_call"]
    final_text = next((r["text"] for r in records if r.get("type") == "final"), "")
    tool_seq = " -> ".join(tool_calls) if tool_calls else "(no tools)"
    return f"TOOL SEQUENCE: {tool_seq}", final_text[:500]


def extract_scenario_id(path: Path) -> str:
    m = re.search(r"(T\d+-C\d+-S\d+)", path.name)
    return m.group(1) if m else path.stem


def extract_threat_num(scenario_id: str) -> str:
    m = re.match(r"(T\d+)", scenario_id)
    return m.group(1) if m else "T?"


def extract_case_num(scenario_id: str) -> str:
    m = re.search(r"(C\d+)", scenario_id)
    return m.group(1) if m else "C?"


# ─────────────────────────────────────────────
# 임베딩 (ada-002)
# ─────────────────────────────────────────────

def get_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    # 빈 문자열은 placeholder로 대체 (ada-002는 빈 입력 거부)
    safe_texts = [t if t.strip() else "(empty)" for t in texts]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=safe_texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def get_weighted_vectors(
    tool_texts: List[str],
    final_texts: List[str],
    client: OpenAI,
    tool_weight: float = 0.7,
    final_weight: float = 0.3,
) -> np.ndarray:
    """
    tool sequence와 final 텍스트를 각각 임베딩 후 가중 평균 벡터 생성.
    weighted_vec = normalize(tool_vec) * tool_weight + normalize(final_vec) * final_weight
    """
    print(f"  [ada-002] tool sequence {len(tool_texts)}개 임베딩 중...")
    tool_vecs = normalize(get_embeddings(tool_texts, client), norm="l2")

    print(f"  [ada-002] final 텍스트 {len(final_texts)}개 임베딩 중...")
    final_vecs = normalize(get_embeddings(final_texts, client), norm="l2")

    weighted = tool_vecs * tool_weight + final_vecs * final_weight
    weighted = normalize(weighted, norm="l2")
    print(f"  → 가중 평균 벡터 shape: {weighted.shape} (tool={tool_weight}, final={final_weight})")
    return weighted


# ─────────────────────────────────────────────
# UMAP 2D 축소
# ─────────────────────────────────────────────

def reduce_umap(
    vectors: np.ndarray,
    n_neighbors: int = 5,
    min_dist: float = 0.3,
    spread: float = 3.0,
) -> np.ndarray:
    print(f"  [UMAP] 2D 축소 중... (n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread})")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        metric="cosine",
        random_state=42,
    )
    embedded = reducer.fit_transform(vectors)
    print(f"  → 완료: shape {embedded.shape}")
    return embedded


# ─────────────────────────────────────────────
# KNN Outlier 탐지
# ─────────────────────────────────────────────

def detect_outliers_knn(
    vectors: np.ndarray,
    scenario_ids: List[str],
    threshold: float = 0.035,
) -> List[bool]:
    """
    같은 scenario_id 그룹 내에서 outlier 탐지.
    - n=2: 두 점 간 코사인 거리 > threshold → 둘 다 outlier
    - n>=3: K=그룹크기-1 KNN 평균 거리 > threshold → outlier
    """
    vecs = normalize(vectors, norm="l2")
    n = len(vecs)
    is_outlier = [False] * n

    groups = defaultdict(list)
    for i, sid in enumerate(scenario_ids):
        groups[sid].append(i)

    for sid, idxs in groups.items():
        if len(idxs) < 2:
            continue

        group_vecs = vecs[idxs]

        if len(idxs) == 2:
            dist = float(cosine_distances(group_vecs[[0]], group_vecs[[1]])[0][0])
            print(f"  [그룹] {sid} (2개) | cosine_dist={dist:.4f}")
            if dist > threshold:
                is_outlier[idxs[0]] = True
                is_outlier[idxs[1]] = True
                print(f"    → [outlier] 둘 다 | {dist:.4f} > threshold={threshold}")
            continue

        k = len(idxs) - 1
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(group_vecs)
        distances, _ = nn.kneighbors(group_vecs)
        avg_distances = distances.mean(axis=1)

        print(f"  [그룹] {sid} ({len(idxs)}개) | avg_dist: {[round(float(d),4) for d in avg_distances]}")

        for local_i, global_i in enumerate(idxs):
            if float(avg_distances[local_i]) > threshold:
                is_outlier[global_i] = True
                print(f"    → [outlier] run{local_i+1} | {avg_distances[local_i]:.4f} > threshold={threshold}")

    n_outliers = sum(is_outlier)
    print(f"  [KNN Outlier] threshold={threshold} → {n_outliers}/{n}개 outlier 탐지")
    return is_outlier


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────

COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#A8DADC", "#6A0572", "#264653",
    "#8338EC", "#FB5607", "#3A86FF", "#FFBE0B",
]

MARKERS = ["o", "s", "^", "D", "v", "P", "h", "p", "<", ">", "8", "H"]


def plot_embeddings(
    coords: np.ndarray,
    labels: List[str],
    threat_labels: List[str],
    file_names: List[str],
    is_outlier: List[bool],
    tool_weight: float,
    final_weight: float,
    output_path: Path,
) -> None:
    unique_threats = sorted(set(threat_labels), key=lambda x: int(x[1:]))
    case_labels = [extract_case_num(sid) for sid in labels]
    unique_cases = sorted(set(case_labels))

    threat_color_map = {t: COLORS[i % len(COLORS)] for i, t in enumerate(unique_threats)}
    marker_map = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(unique_cases)}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    _annotations = []

    for i, (x, y) in enumerate(coords):
        threat = threat_labels[i]
        case = case_labels[i]
        color = threat_color_map[threat]
        marker = marker_map[case]
        outlier = is_outlier[i]

        if outlier:
            ax.scatter(x, y, c=color, marker=marker, s=200, alpha=0.95,
                       edgecolors="#FF3333", linewidths=2.5, zorder=4)
            ax.scatter(x, y, c="none", marker="x", s=220, alpha=1.0,
                       edgecolors="#FF3333", linewidths=2.5, zorder=5)
        else:
            ax.scatter(x, y, c=color, marker=marker, s=130, alpha=0.88,
                       edgecolors="white", linewidths=0.5, zorder=3)

        _annotations.append((x, y, labels[i], "#FF3333" if outlier else "white", 0.95 if outlier else 0.75))

    texts = []
    for (x, y, lbl, color, alpha) in _annotations:
        t = ax.text(x, y, lbl, fontsize=7, color=color, alpha=alpha)
        texts.append(t)

    if HAS_ADJUST_TEXT:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5),
            expand_points=(1.5, 1.5),
            expand_text=(1.3, 1.3),
        )

    # 범례 1: T번호 (색상)
    threat_patches = [mpatches.Patch(color=threat_color_map[t], label=t) for t in unique_threats]
    legend1 = ax.legend(handles=threat_patches, loc="upper left", title="Threat Type",
                        framealpha=0.3, labelcolor="white", facecolor="#0d0d1a",
                        edgecolor="gray", fontsize=9, title_fontsize=9)
    legend1.get_title().set_color("white")
    ax.add_artist(legend1)

    # 범례 2: C번호 (마커)
    case_handles = [
        Line2D([0], [0], marker=marker_map[c], color="w", markerfacecolor="gray",
               markersize=8, label=c, linestyle="None")
        for c in unique_cases
    ]
    legend2 = ax.legend(handles=case_handles, loc="upper right", title="Case",
                        framealpha=0.3, labelcolor="white", facecolor="#0d0d1a",
                        edgecolor="gray", fontsize=9, title_fontsize=9)
    legend2.get_title().set_color("white")
    ax.add_artist(legend2)

    # 범례 3: outlier
    outlier_handle = Line2D([0], [0], marker="x", color="#FF3333",
                            markerfacecolor="#FF3333", markersize=10,
                            label="Outlier (KNN)", linestyle="None", markeredgewidth=2.5)
    ax.legend(handles=[outlier_handle], loc="lower right", framealpha=0.3,
              labelcolor="white", facecolor="#0d0d1a", edgecolor="gray", fontsize=9)

    title = (f"Attack Trace Embedding Space\n"
             f"(ada-002 weighted avg  tool={tool_weight} / final={final_weight}  |  UMAP)")
    ax.set_title(title, color="white", fontsize=13, pad=15)
    ax.set_xlabel("UMAP-1", color="gray", fontsize=10)
    ax.set_ylabel("UMAP-2", color="gray", fontsize=10)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, alpha=0.15, color="white", linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[OK] 저장 완료: {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# API 키 로드
# ─────────────────────────────────────────────

def load_api_key(cli_key: str) -> str:
    if cli_key:
        return cli_key.strip()
    key_path = Path(DEFAULT_KEY_PATH)
    if key_path.exists():
        key = key_path.read_text(encoding="utf-8").strip()
        print(f"[KEY] {key_path} 에서 로드")
        return key
    raise RuntimeError(f"API 키를 찾을 수 없습니다: {DEFAULT_KEY_PATH}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CCS Attack Trace 임베딩 시각화 (ada-002 가중 평균 + UMAP)")
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument("--openai-key", default="")
    parser.add_argument("--output", default="embedding_plot.png")
    parser.add_argument("--tool-weight", type=float, default=0.7, help="tool sequence 가중치 (기본 0.7)")
    parser.add_argument("--final-weight", type=float, default=0.3, help="final 텍스트 가중치 (기본 0.3)")
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--min-dist", type=float, default=0.3)
    parser.add_argument("--spread", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.035, help="KNN outlier threshold (기본 0.035)")
    args = parser.parse_args()

    api_key = load_api_key(args.openai_key)
    client = OpenAI(api_key=api_key)

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise FileNotFoundError(f"디렉토리 없음: {logs_dir}")

    log_files = sorted(logs_dir.glob("attack-T*.jsonl"))
    if not log_files:
        raise FileNotFoundError(f"attack-T*.jsonl 파일 없음: {logs_dir}")

    print(f"[파일] {len(log_files)}개 발견\n")

    tool_texts, final_texts, scenario_ids, threat_nums, file_names = [], [], [], [], []
    for f in log_files:
        records = load_jsonl(f)
        tool_text, final_text = parse_trace(records)
        sid = extract_scenario_id(f)
        tnum = extract_threat_num(sid)

        tool_texts.append(tool_text)
        final_texts.append(final_text)
        scenario_ids.append(sid)
        threat_nums.append(tnum)
        file_names.append(f.name)
        print(f"  {sid:20s} | {tool_text[:60]}...")

    print(f"\n총 {len(tool_texts)}개 trace 추출 완료\n")

    # 가중 평균 임베딩
    vectors = get_weighted_vectors(
        tool_texts, final_texts, client,
        tool_weight=args.tool_weight,
        final_weight=args.final_weight,
    )

    # UMAP
    coords = reduce_umap(vectors, n_neighbors=args.n_neighbors, min_dist=args.min_dist, spread=args.spread)

    # KNN Outlier 탐지
    is_outlier = detect_outliers_knn(vectors, scenario_ids, threshold=args.threshold)

    # 시각화
    output_path = Path(args.output)
    plot_embeddings(coords, scenario_ids, threat_nums, file_names, is_outlier,
                    args.tool_weight, args.final_weight, output_path)

    # CSV 저장
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("scenario_id,threat,umap_x,umap_y,outlier,file\n")
        for sid, t, (x, y), out, fn in zip(scenario_ids, threat_nums, coords, is_outlier, file_names):
            f.write(f"{sid},{t},{x:.6f},{y:.6f},{out},{fn}\n")
    print(f"[OK] 좌표 CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()