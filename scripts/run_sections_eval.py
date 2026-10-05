#!/usr/bin/env python3
"""Evaluation script for song-form section planner.

Compares heuristic-only vs LLM-refined section planning on fixtures/sections_truth.json.
Measures:
- Role accuracy (per-measure)
- Boundary F1 (with +/-1 measure tolerance)
- Latency (seconds per song)
Saves results to out/sections/eval_results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import ParsedSheet, SectionPlan
from app.arrange.sections import plan_sections

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_sections_eval")

FIXTURE_PATHS: dict[str, Path] = {
    "diaole": PROJECT_ROOT / "out" / "gt_eval" / "diaole.json",
    "xiaobaichuan": PROJECT_ROOT / "fixtures" / "sheets" / "xiaobaichuan.json",
    "qianlizhiwai": PROJECT_ROOT / "fixtures" / "sheets" / "qianlizhiwai.json",
    "tinghai": PROJECT_ROOT / "fixtures" / "sheets" / "tinghai.json",
    "diandao": PROJECT_ROOT / "fixtures" / "sheets" / "diandao.json",
    "yujian": PROJECT_ROOT / "fixtures" / "sheets" / "yujian.json",
}


def load_sheet(name: str) -> ParsedSheet:
    p = FIXTURE_PATHS[name]
    if not p.exists():
        raise FileNotFoundError(f"Fixture file for {name} not found at {p}")
    with open(p, encoding="utf-8") as f:
        return ParsedSheet.model_validate(json.load(f))


def extract_sections_from_plans(plans: list[SectionPlan]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    curr: dict[str, Any] | None = None
    for p in plans:
        if curr is None or curr["section_id"] != p.section_id:
            if curr is not None:
                sections.append(curr)
            curr = {
                "section_id": p.section_id,
                "role": p.role,
                "start_measure": p.measure_index,
                "end_measure": p.measure_index,
                "energy": p.energy,
            }
        else:
            curr["end_measure"] = p.measure_index
    if curr is not None:
        sections.append(curr)
    return sections


def evaluate_song_sections(
    truth_sections: list[dict[str, Any]],
    predicted_plans: list[SectionPlan],
) -> dict[str, float]:
    """Calculate role accuracy and boundary F1 (tolerance +/- 1 measure)."""
    max_truth_measure = truth_sections[-1]["end_measure"]

    # Truth per-measure role mapping
    truth_m_roles: dict[int, str] = {}
    for sec in truth_sections:
        for m in range(sec["start_measure"], sec["end_measure"] + 1):
            truth_m_roles[m] = sec["role"]

    # Predicted per-measure role mapping
    pred_m_roles: dict[int, str] = {
        p.measure_index: p.role for p in predicted_plans if p.measure_index <= max_truth_measure
    }

    # 1. Role Accuracy
    total_eval_measures = max_truth_measure + 1
    matching_roles = sum(
        1 for m in range(total_eval_measures) if truth_m_roles.get(m) == pred_m_roles.get(m)
    )
    role_acc = matching_roles / total_eval_measures if total_eval_measures > 0 else 0.0

    # 2. Boundary F1 (+/- 1 tolerance)
    truth_boundaries = [s["start_measure"] for s in truth_sections if s["start_measure"] > 0]
    pred_sections = extract_sections_from_plans(predicted_plans)
    pred_boundaries = [
        s["start_measure"]
        for s in pred_sections
        if 0 < s["start_measure"] <= max_truth_measure
    ]

    matched_truth = 0
    for tb in truth_boundaries:
        if any(abs(tb - pb) <= 1 for pb in pred_boundaries):
            matched_truth += 1

    matched_pred = 0
    for pb in pred_boundaries:
        if any(abs(pb - tb) <= 1 for tb in truth_boundaries):
            matched_pred += 1

    precision = matched_pred / len(pred_boundaries) if pred_boundaries else 1.0
    recall = matched_truth / len(truth_boundaries) if truth_boundaries else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "role_accuracy": round(role_acc, 4),
        "boundary_precision": round(precision, 4),
        "boundary_recall": round(recall, 4),
        "boundary_f1": round(f1, 4),
    }


def run_evaluation(mode: str = "heuristic") -> dict[str, Any]:
    truth_path = PROJECT_ROOT / "fixtures" / "sections_truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    with open(truth_path, encoding="utf-8") as f:
        truth_all: dict[str, list[dict[str, Any]]] = json.load(f)

    out_dir = PROJECT_ROOT / "out" / "sections"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    eval_modes = ["heuristic"]
    if mode in ("llm", "both") and os.environ.get("QA_OFFLINE") != "1":
        eval_modes.append("llm")

    for m in eval_modes:
        use_llm = (m == "llm")
        results[m] = {"songs": {}, "aggregate": {}}
        total_acc = 0.0
        total_f1 = 0.0
        total_time = 0.0
        count = len(truth_all)

        for song_name, truth_secs in truth_all.items():
            sheet = load_sheet(song_name)
            t0 = time.perf_counter()
            plans = plan_sections(sheet, use_llm=use_llm)
            elapsed = time.perf_counter() - t0

            metrics = evaluate_song_sections(truth_secs, plans)
            metrics["latency_s"] = round(elapsed, 4)
            results[m]["songs"][song_name] = metrics

            total_acc += metrics["role_accuracy"]
            total_f1 += metrics["boundary_f1"]
            total_time += elapsed

            # Save per-song plan
            plan_file = out_dir / f"{song_name}_{m}_plan.json"
            with open(plan_file, "w", encoding="utf-8") as pf:
                json.dump([p.model_dump() for p in plans], pf, indent=2, ensure_ascii=False)

        results[m]["aggregate"] = {
            "mean_role_accuracy": round(total_acc / count, 4),
            "mean_boundary_f1": round(total_f1 / count, 4),
            "total_latency_s": round(total_time, 4),
            "mean_latency_s": round(total_time / count, 4),
        }

    # Save summary
    summary_path = out_dir / "eval_results.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(results, sf, indent=2, ensure_ascii=False)

    return results


def print_markdown_table(results: dict[str, Any]) -> None:
    print("\n### Section Planning Evaluation Results\n")
    for m, data in results.items():
        print(f"#### Mode: {m.upper()}\n")
        print("| Song | Role Accuracy | Boundary Precision | Boundary Recall | Boundary F1 | Latency (s) |")
        print("|---|---|---|---|---|---|")
        for song, met in data["songs"].items():
            print(
                f"| {song} | {met['role_accuracy']*100:.1f}% | {met['boundary_precision']*100:.1f}% | "
                f"{met['boundary_recall']*100:.1f}% | {met['boundary_f1']*100:.1f}% | {met['latency_s']:.3f}s |"
            )
        agg = data["aggregate"]
        print(
            f"| **MEAN / TOTAL** | **{agg['mean_role_accuracy']*100:.1f}%** | - | - | "
            f"**{agg['mean_boundary_f1']*100:.1f}%** | **{agg['total_latency_s']:.3f}s** |\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate song-form section planner.")
    parser.add_argument("--mode", choices=["heuristic", "llm", "both"], default="heuristic", help="Evaluation mode.")
    args = parser.parse_args()

    results = run_evaluation(mode=args.mode)
    print_markdown_table(results)


if __name__ == "__main__":
    main()
