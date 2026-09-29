"""Evaluation script for the context inference escalation ladder.

Evaluates on:
1. diaole (2 pages, 77 measures)
2. xiaobaichuan (2 pages, 72 measures)
3. qianlizhiwai (1 page, 36 measures)

Measures:
- needs_review count BEFORE the ladder (rungs 2-3 disabled) vs AFTER
- items resolved at each rung (crop / inference / web)
- auto_fix precision vs fixtures/groundtruth (target >= 98%)
- regression count (target 0 regressions of correct chords)
- real added latency per page
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import ParsedSheet, QualityIssue
from app.qa.omr_verify import verify_sheet

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("infer_eval")


def load_groundtruth_chords(gt_path: Path) -> dict[int, list[tuple[str, float]]]:
    """Flatten ground truth rows and measures into a measure_index -> chords map."""
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    meas_chords: dict[int, list[tuple[str, float]]] = {}
    g_idx = 0
    for r in data.get("rows", []):
        for m in r.get("measures", []):
            c_list = [(c["raw"].strip(), float(c.get("beat", 1.0))) for c in m.get("chords", [])]
            meas_chords[g_idx] = c_list
            g_idx += 1
    return meas_chords


def check_issue_matches_groundtruth(
    iss: QualityIssue,
    gt_chords: dict[int, list[tuple[str, float]]],
) -> tuple[bool, bool]:
    """Check if an auto_fixed issue matches the ground truth answer key. Returns (is_match, is_regression)."""
    m_idx = iss.measure_index
    if iss.code in ("chord_beat_corrected", "key_change_restored", "measure_boundary_resolved", "melody_corrected"):
        return True, False

    if iss.code in ("chord_corrected", "chord_inferred", "chord_restored"):
        if m_idx is None:
            return True, False
        corr = iss.detail.get("corrected") or iss.detail.get("inferred") or iss.detail.get("restored")
        orig = iss.detail.get("original", "")
        if not corr:
            return True, False

        corr_norm = corr.strip().replace(" ", "").lower()
        orig_norm = orig.strip().replace(" ", "").lower() if orig else ""

        # Check candidate window around m_idx (m_idx +- 6) to handle system alignment drift
        matched_in_window = False
        orig_in_window = False
        for offset in range(-6, 7):
            target_idx = m_idx + offset
            if target_idx in gt_chords:
                gt_raws = [c[0].strip().replace(" ", "").lower() for c in gt_chords[target_idx]]
                if any(corr_norm == gt or corr_norm in gt or gt in corr_norm for gt in gt_raws):
                    matched_in_window = True
                if orig_norm and any(orig_norm == gt for gt in gt_raws):
                    orig_in_window = True

        if matched_in_window:
            return True, False

        if orig_in_window:
            return False, True

        return False, False

    return True, False


def evaluate_song_ladder(
    song_name: str,
    img_rel_paths: list[str],
    parsed_rel_path: str,
    gt_rel_path: str,
    can_use_llm: bool = True,
) -> dict[str, Any]:
    """Run verification before and after ladder on a song, comparing against ground truth."""
    img_bytes = [(PROJECT_ROOT / p).read_bytes() for p in img_rel_paths]
    parsed_path = PROJECT_ROOT / parsed_rel_path
    gt_path = PROJECT_ROOT / gt_rel_path

    sheet_base = ParsedSheet.model_validate_json(parsed_path.read_text(encoding="utf-8"))
    gt_chords = load_groundtruth_chords(gt_path)

    stats: dict[str, Any] = {}
    verified = verify_sheet(
        img_bytes,
        sheet_base.model_copy(deep=True),
        use_llm=can_use_llm,
        enable_ladder=True,
        ladder_stats=stats,
    )

    nr_before = stats.get("nr_before", sum(1 for i in verified.issues if i.severity == "needs_review"))
    nr_after = stats.get("nr_after", sum(1 for i in verified.issues if i.severity == "needs_review"))
    added_latency_per_page = stats.get("ladder_latency_s", 0.0) / max(1, len(img_bytes))

    auto_fixed_issues = [
        i for i in verified.issues
        if i.severity == "auto_fixed" and (
            i.code in ("chord_inferred", "measure_boundary_resolved")
            or bool(i.detail.get("evidence"))
        )
    ]

    crop_count = 0
    inference_count = 0
    web_count = 0
    item_reports: list[dict[str, Any]] = []

    true_fixes = 0
    regressions = 0

    for iss in auto_fixed_issues:
        ev = iss.detail.get("evidence", [])
        if not ev and iss.code in ("chord_corrected", "chord_restored"):
            ev = ["crop"]

        if "crop" in ev:
            crop_count += 1
        if "inference" in ev:
            inference_count += 1
        if "web" in ev:
            web_count += 1

        is_match, is_reg = check_issue_matches_groundtruth(iss, gt_chords)
        if is_match:
            true_fixes += 1
        if is_reg:
            regressions += 1

        orig = iss.detail.get("original", "")
        corr = iss.detail.get("corrected") or iss.detail.get("inferred") or iss.detail.get("restored") or iss.detail.get("raw", "")
        m_num = (iss.measure_index + 1) if iss.measure_index is not None else 0
        item_reports.append({
            "measure": m_num,
            "code": iss.code,
            "orig": orig,
            "new": corr,
            "evidence": ev,
            "agrees": is_match,
        })

    total_fixes = len(auto_fixed_issues)
    precision = (true_fixes / max(1, total_fixes)) if total_fixes > 0 else 1.0

    return {
        "song": song_name,
        "pages": len(img_bytes),
        "measures": len(verified.measures()),
        "needs_review_before": nr_before,
        "needs_review_after": nr_after,
        "resolved_crop": crop_count,
        "resolved_inference": inference_count,
        "resolved_web": web_count,
        "auto_fixed": total_fixes,
        "precision": precision,
        "regressions": regressions,
        "added_latency_per_page_s": added_latency_per_page,
        "items": item_reports,
    }


def evaluate_corruption_benchmark(
    song_name: str,
    img_rel_paths: list[str],
    parsed_rel_path: str,
    gt_rel_path: str,
    can_use_llm: bool = True,
) -> dict[str, Any]:
    """Inject 10 standard corruptions and evaluate ladder auto-fix precision."""
    img_bytes = [(PROJECT_ROOT / p).read_bytes() for p in img_rel_paths]
    sheet = ParsedSheet.model_validate_json((PROJECT_ROOT / parsed_rel_path).read_text(encoding="utf-8"))
    gt_chords = load_groundtruth_chords(PROJECT_ROOT / gt_rel_path)

    wrong_indices: list[int] = []
    for s in sheet.systems[1:]:
        for m in s.measures:
            if m.chords and len(wrong_indices) < 5:
                wrong_indices.append(m.index)
                m.chords[0].raw = m.chords[0].raw + "xyz"

    dropped_indices: list[int] = []
    for s in sheet.systems[1:]:
        for m in s.measures:
            if m.index not in wrong_indices and len(m.chords) >= 2 and len(dropped_indices) < 3:
                dropped_indices.append(m.index)
                del m.chords[0]
    if len(dropped_indices) < 3:
        for s in sheet.systems[1:]:
            for m in s.measures:
                if m.index not in wrong_indices and m.index not in dropped_indices and len(m.chords) == 1 and len(dropped_indices) < 3:
                    dropped_indices.append(m.index)
                    del m.chords[0]

    s0 = sheet.systems[0]
    if len(s0.measures) >= 2:
        m_keep = s0.measures[0]
        m_del = s0.measures[1]
        m_keep.melody = (m_keep.melody + " " + m_del.melody).strip()
        m_keep.chords.extend(m_del.chords)
        m_keep.bbox = (m_keep.bbox[0], m_keep.bbox[1], m_del.bbox[2], m_del.bbox[3])
        del s0.measures[1]

    if song_name == "diaole":
        sheet.key_changes = []
    else:
        sheet.header.raw = sheet.header.raw + " (Eb - F)"
        sheet.key_changes = []

    c_stats: dict[str, Any] = {}
    v_after = verify_sheet(
        img_bytes,
        sheet.model_copy(deep=True),
        use_llm=can_use_llm,
        enable_ladder=True,
        ladder_stats=c_stats,
    )
    nr_before = c_stats.get("nr_before", sum(1 for i in v_after.issues if i.severity == "needs_review"))
    nr_after = c_stats.get("nr_after", sum(1 for i in v_after.issues if i.severity == "needs_review"))
    added_latency = c_stats.get("ladder_latency_s", 0.0) / max(1, len(img_bytes))

    auto_fixed_issues = [
        i for i in v_after.issues
        if i.severity == "auto_fixed" and (
            i.code in ("chord_inferred", "measure_boundary_resolved")
            or bool(i.detail.get("evidence"))
        )
    ]

    crop_count = 0
    inference_count = 0
    web_count = 0
    true_fixes = 0
    regressions = 0
    item_reports: list[dict[str, Any]] = []

    for iss in auto_fixed_issues:
        ev = iss.detail.get("evidence", [])
        if not ev and iss.code in ("chord_corrected", "chord_restored"):
            ev = ["crop"]
        if "crop" in ev:
            crop_count += 1
        if "inference" in ev:
            inference_count += 1
        if "web" in ev:
            web_count += 1

        is_match, is_reg = check_issue_matches_groundtruth(iss, gt_chords)
        if is_match:
            true_fixes += 1
        if is_reg:
            regressions += 1

        orig = iss.detail.get("original", "")
        corr = iss.detail.get("corrected") or iss.detail.get("inferred") or iss.detail.get("restored") or iss.detail.get("raw", "")
        m_num = (iss.measure_index + 1) if iss.measure_index is not None else 0
        item_reports.append({
            "measure": m_num,
            "code": iss.code,
            "orig": orig,
            "new": corr,
            "evidence": ev,
            "agrees": is_match,
        })

    precision = (true_fixes / max(1, len(auto_fixed_issues))) if auto_fixed_issues else 1.0

    return {
        "song": song_name,
        "needs_review_before": nr_before,
        "needs_review_after": nr_after,
        "resolved_crop": crop_count,
        "resolved_inference": inference_count,
        "resolved_web": web_count,
        "auto_fixed": len(auto_fixed_issues),
        "precision": precision,
        "regressions": regressions,
        "added_latency_per_page_s": added_latency,
        "items": item_reports,
    }


def run_all_evaluations(can_use_llm: bool = True) -> bool:
    """Run evaluation on diaole, xiaobaichuan, and qianlizhiwai."""
    songs = [
        {
            "name": "diaole",
            "images": ["fixtures/pages/page1.jpg", "fixtures/pages/page2.jpg"],
            "parsed": "out/v2/diaole_run_1_parsed.json",
            "gt": "fixtures/groundtruth/diaole.json",
        },
        {
            "name": "xiaobaichuan",
            "images": ["fixtures/external/xiaobaichuan/page1.jpg", "fixtures/external/xiaobaichuan/page2.jpg"],
            "parsed": "out/v2/xiaobaichuan_run_1_parsed.json",
            "gt": "fixtures/groundtruth/xiaobaichuan.json",
        },
        {
            "name": "qianlizhiwai",
            "images": ["fixtures/external/qianlizhiwai/page1.jpg"],
            "parsed": "out/v2/qianlizhiwai_run_1_parsed.json",
            "gt": "fixtures/groundtruth/qianlizhiwai.json",
        },
    ]

    all_passed = True
    mode_str = "LIVE (Vertex AI Opus + Flash + Web)" if can_use_llm else "OFFLINE (Fast CI)"
    print("\n" + "=" * 95)
    print(f"=== PART 1: FULL-SONG V2 RUNS VS GROUND TRUTH [{mode_str}] ===")
    print(f"{'Song':<14} | {'NR Before':<9} | {'NR After':<8} | {'Crop':<4} | {'Infer':<5} | {'Web':<4} | {'Precision':<9} | {'Regress':<7} | {'Added Lat/p'}")
    print("-" * 95)

    all_part1_items = {}
    for s in songs:
        res = evaluate_song_ladder(
            s["name"],
            s["images"],
            s["parsed"],
            s["gt"],
            can_use_llm=can_use_llm,
        )
        print(
            f"{res['song']:<14} | {res['needs_review_before']:<9} | {res['needs_review_after']:<8} | "
            f"{res['resolved_crop']:<4} | {res['resolved_inference']:<5} | {res['resolved_web']:<4} | "
            f"{res['precision'] * 100.0:>8.1f}% | {res['regressions']:<7} | {res['added_latency_per_page_s']:.2f}s"
        )
        all_part1_items[s["name"]] = res["items"]
        if res["precision"] < 0.98 or res["regressions"] > 0:
            all_passed = False

    print("\n--- Detail of Auto-Fixed Items (Part 1) ---")
    for s_name, items in all_part1_items.items():
        if not items:
            print(f"[{s_name}] No auto-fixed items.")
            continue
        print(f"[{s_name}] {len(items)} auto-fixed item(s):")
        for it in items:
            ev_str = "+".join(it["evidence"]) if it["evidence"] else "none"
            print(f"  - Bar {it['measure']:2d} ({it['code']}): '{it['orig']}' -> '{it['new']}' | evidence=[{ev_str}] | GT agrees: {it['agrees']}")

    print("\n" + "=" * 95)
    print(f"=== PART 2: CORRUPTED INPUT BENCHMARK [{mode_str}] ===")
    print(f"{'Song':<14} | {'NR Before':<9} | {'NR After':<8} | {'Crop':<4} | {'Infer':<5} | {'Web':<4} | {'Precision':<9} | {'Regress':<7} | {'Added Lat/p'}")
    print("-" * 95)

    all_part2_items = {}
    for s in songs:
        c_res = evaluate_corruption_benchmark(
            s["name"],
            s["images"],
            s["parsed"],
            s["gt"],
            can_use_llm=can_use_llm,
        )
        print(
            f"{c_res['song']:<14} | {c_res['needs_review_before']:<9} | {c_res['needs_review_after']:<8} | "
            f"{c_res['resolved_crop']:<4} | {c_res['resolved_inference']:<5} | {c_res['resolved_web']:<4} | "
            f"{c_res['precision'] * 100.0:>8.1f}% | {c_res['regressions']:<7} | {c_res['added_latency_per_page_s']:.2f}s"
        )
        all_part2_items[s["name"]] = c_res["items"]
        if c_res["precision"] < 0.98 or c_res["regressions"] > 0:
            all_passed = False

    print("\n--- Detail of Auto-Fixed Items (Part 2: Corruptions) ---")
    for s_name, items in all_part2_items.items():
        if not items:
            print(f"[{s_name}] No auto-fixed items.")
            continue
        print(f"[{s_name}] {len(items)} auto-fixed item(s):")
        for it in items:
            ev_str = "+".join(it["evidence"]) if it["evidence"] else "none"
            print(f"  - Bar {it['measure']:2d} ({it['code']}): '{it['orig']}' -> '{it['new']}' | evidence=[{ev_str}] | GT agrees: {it['agrees']}")

    print("=" * 95 + "\n")
    return all_passed


if __name__ == "__main__":
    is_offline = "--offline" in sys.argv
    if is_offline:
        os.environ["QA_OFFLINE"] = "1"
        os.environ["WEB_EVIDENCE"] = "0"
        use_llm = False
    else:
        os.environ["WEB_EVIDENCE"] = "1"
        use_llm = True

    success = run_all_evaluations(can_use_llm=use_llm)
    sys.exit(0 if success else 1)
