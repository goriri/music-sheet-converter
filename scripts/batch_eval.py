#!/usr/bin/env python3
"""Batch evaluation script for Taiwanese band chart to piano accompaniment pipeline.

Executes offline pipeline across external test songs:
  Gemini OMR -> OMR Verify -> Piano Arrange -> Arrange Check -> Overlay Render
Keys: [header/C, G]
Difficulties: [beginner, intermediate, advanced]
Ground truth: Opus proxy on Category (a) system crops
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import io
import json
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image
import pymupdf
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import (
    Arrangement,
    ChordSymbol,
    Difficulty,
    Measure,
    MeasureArrangement,
    ParsedSheet,
    QualityIssue,
    ResolvedChord,
    System,
)
from app.omr.gemini_omr import parse_pages
from app.qa.omr_verify import verify_sheet
from app.arrange.piano import arrange
from app.qa.arrange_check import check_and_repair, validate_measure, Violation
from app.render.overlay import render_pdf, render_pages
from app.theory.keys import key_name_to_pc, canonical_key_for_pc
from app.theory.chords import parse_chord
from app.qa.llm import ask_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("batch_eval")


# ---------------------------------------------------------------------------
# Pydantic schemas for Opus ground truth verification
# ---------------------------------------------------------------------------
class MeasureOpusChord(BaseModel):
    measure_number: int = Field(
        description="1-based index of the measure within this system crop from left to right"
    )
    chords: list[str] = Field(
        default_factory=list,
        description="Verbatim printed chord symbol(s) above the notation in this measure (e.g. '1(2)', '5/7', '2m7', '4maj7')"
    )


class SystemOpusResponse(BaseModel):
    measures: list[MeasureOpusChord] = Field(default_factory=list)


def normalize_chord_str(chord: str) -> str:
    """Normalize chord representation for fuzzy comparison."""
    if not chord:
        return ""
    c = chord.strip()
    c = c.replace("（", "(").replace("）", ")").replace("＃", "#").replace("♯", "#").replace("♭", "b")
    circle_map = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6", "⑦": "7"}
    for circ, num in circle_map.items():
        c = c.replace(circ, num)
    c = "".join(c.split()).lower()
    return c


def extract_header_key(sheet: ParsedSheet, default_key: str = "C") -> str:
    """Extract canonical start key from parsed sheet header or fallback."""
    candidates = []
    if sheet.header and sheet.header.original_key:
        candidates.append(sheet.header.original_key)
    if hasattr(sheet, "key") and getattr(sheet, "key"):
        candidates.append(getattr(sheet, "key"))
    if sheet.header and sheet.header.raw:
        m = re.search(r'(?:1\s*=\s*|Key\s*[:=]\s*)([A-G][b#]?)', sheet.header.raw, re.IGNORECASE)
        if m:
            candidates.append(m.group(1))

    for cand in candidates:
        if not cand:
            continue
        m = re.search(r'([A-G][b#]?)', cand.strip())
        if m:
            k = m.group(1).capitalize()
            try:
                key_name_to_pc(k)
                return k
            except Exception:
                pass
    return default_key


def sanitize_chord_for_arrange(raw: str) -> str:
    """Sanitize chord string for piano arranger while preserving musical degree."""
    if not raw:
        return "1"
    c = raw.strip()
    circle_map = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6", "⑦": "7"}
    for circ, num in circle_map.items():
        c = c.replace(circ, num)

    c = re.sub(r'sus7', '7sus', c, flags=re.IGNORECASE)
    # Remove tension annotations like (6.9), (b9) if not (2) or (9)
    if not (c.endswith("(2)") or c.endswith("(9)")):
        c = re.sub(r'\([^\)]+\)', '', c)

    try:
        parse_chord(c)
        return c
    except Exception:
        # Fallback to degree if available
        m = re.search(r'([1-7][b#]?)', c)
        if m:
            return m.group(1)
        # If it's a letter chord (Category b), raise so caller knows it's unparseable
        raise ValueError(f"Cannot resolve degree for chord '{raw}'")


def prepare_sheet_for_arrange(sheet: ParsedSheet) -> ParsedSheet:
    """Create a deep copy of sheet with sanitized chords for arrangement."""
    sheet_copy = ParsedSheet.model_validate(sheet.model_dump())
    for s in sheet_copy.systems:
        for m in s.measures:
            for ch in m.chords:
                try:
                    ch.raw = sanitize_chord_for_arrange(ch.raw)
                except Exception:
                    # Keep as is and let arrange handle or raise
                    pass
    return sheet_copy


def collect_violations_before_repair(
    sheet: ParsedSheet,
    raw_arr: Arrangement,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Validate each measure of raw arrangement before repair and collect violations."""
    try:
        start_pc = key_name_to_pc(raw_arr.start_key)
    except Exception:
        start_pc = 0
    current_tonic_pc = start_pc

    key_changes_by_m: dict[int, list[int]] = {}
    for kc in sheet.key_changes:
        key_changes_by_m.setdefault(kc.at_measure, []).append(kc.semitones)

    measures_by_idx = {m.index: m for m in sheet.measures()}

    expected_tonics: list[int] = []
    for m_arr in raw_arr.measures:
        m_idx = m_arr.measure_index
        if m_idx in key_changes_by_m:
            for shift in key_changes_by_m[m_idx]:
                current_tonic_pc = (current_tonic_pc + shift) % 12
        expected_tonics.append(current_tonic_pc)

    num_measures = len(raw_arr.measures)
    violations_by_code: dict[str, int] = {}
    violation_records: list[dict[str, Any]] = []

    for i in range(num_measures):
        m_arr = raw_arr.measures[i]
        m_idx = m_arr.measure_index
        measure = measures_by_idx.get(m_idx, Measure(index=m_idx, bbox=(0, 0, 1, 1)))
        exp_tonic = expected_tonics[i]

        next_chord = None
        if i + 1 < num_measures and raw_arr.measures[i + 1].chords:
            next_chord = raw_arr.measures[i + 1].chords[0]

        vs = validate_measure(m_arr, measure, exp_tonic, raw_arr.difficulty, next_chord)
        for v in vs:
            violations_by_code[v.code] = violations_by_code.get(v.code, 0) + 1
            violation_records.append({
                "measure_index": m_idx,
                "code": v.code,
                "message": v.message,
                "detail": v.detail,
            })

    return violations_by_code, violation_records


def run_opus_ground_truth(
    slug: str,
    page_bytes_list: list[bytes],
    sheet: ParsedSheet,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run Opus proxy ground truth on system crops with 3% padding for Category (a)."""
    decoded_pages: list[np.ndarray] = []
    for pb in page_bytes_list:
        nparr = np.frombuffer(pb, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decoded_pages.append(img)

    total_checked = 0
    total_matched = 0
    mismatches: list[dict[str, Any]] = []

    for row_idx, sys_row in enumerate(sheet.systems):
        if not sys_row.measures:
            continue
        p_idx = sys_row.page
        if p_idx >= len(decoded_pages):
            continue
        img = decoded_pages[p_idx]
        h, w = img.shape[:2]

        x0, y0, x1, y1 = sys_row.bbox
        pad_x = 0.03 * max(0.01, (x1 - x0))
        pad_y = 0.03 * max(0.01, (y1 - y0))
        crop_x0 = max(0.0, x0 - pad_x)
        crop_y0 = max(0.0, y0 - pad_y)
        crop_x1 = min(1.0, x1 + pad_x)
        crop_y1 = min(1.0, y1 + pad_y)

        y_start = int(round(crop_y0 * h))
        y_end = int(round(crop_y1 * h))
        x_start = int(round(crop_x0 * w))
        x_end = int(round(crop_x1 * w))

        crop_img = img[y_start:y_end, x_start:x_end]
        if crop_img.size == 0 or crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
            continue

        _, buf = cv2.imencode(".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        crop_bytes = buf.tobytes()

        first_m = sys_row.measures[0].index + 1
        last_m = sys_row.measures[-1].index + 1
        prompt = (
            f"You are an expert music chord reader acting as an independent arbiter for Taiwanese band charts (台湾简谱 / 流行乐团总谱).\n"
            f"This image is a crop of System (row) {row_idx + 1} from page {p_idx + 1}, containing {len(sys_row.measures)} measure(s) "
            f"(measures {first_m} through {last_m} from left to right).\n"
            f"For each measure from left to right:\n"
            f"Read the printed chord symbol(s) verbatim inside or above each measure box (e.g. '1(2)', '5/7', '2m7/5', '6m', '4maj7', '17/7b', 'b7', etc.).\n"
            f"If a measure has no chord printed, provide an empty list for that measure.\n"
            f"Return exactly {len(sys_row.measures)} measure objects with measure_number (1..{len(sys_row.measures)}) and chords."
        )

        try:
            resp = ask_json(
                prompt=prompt,
                schema=SystemOpusResponse,
                images=[crop_bytes],
                role="arbiter",
                timeout_s=60.0,
            )
            opus_by_num = {m.measure_number: m.chords for m in resp.measures}

            for m_offset, m in enumerate(sys_row.measures, 1):
                total_checked += 1
                pipe_chords = [c.raw.strip() for c in m.chords]
                opus_chords = opus_by_num.get(m_offset)
                if opus_chords is None and m_offset - 1 < len(resp.measures):
                    opus_chords = resp.measures[m_offset - 1].chords
                if opus_chords is None:
                    opus_chords = []

                norm_pipe = [normalize_chord_str(c) for c in pipe_chords if normalize_chord_str(c)]
                norm_opus = [normalize_chord_str(c) for c in opus_chords if normalize_chord_str(c)]

                if norm_pipe == norm_opus:
                    total_matched += 1
                else:
                    mismatches.append({
                        "slug": slug,
                        "page": p_idx + 1,
                        "row": row_idx + 1,
                        "measure": m.index + 1,
                        "pipeline": pipe_chords,
                        "opus": opus_chords,
                        "tuple": [slug, p_idx + 1, row_idx + 1, m.index + 1, pipe_chords, opus_chords],
                    })
        except Exception as e:
            logger.warning("Opus ground truth call failed for %s sys %d: %s", slug, row_idx + 1, e)

    return total_checked, total_matched, mismatches


def evaluate_song(
    song_meta: dict[str, Any],
    out_base: Path,
    omr_model: str = "gemini-2.5-flash",
    skip_opus: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute complete offline pipeline for a single song."""
    slug = song_meta["slug"]
    category = song_meta.get("category", "a")
    title = song_meta.get("title", slug)
    song_dir = out_base / slug
    song_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info(">>> Starting evaluation for %s [%s] (%s)", slug, category, title)

    stage_failures: dict[str, str] = {}
    stage_timings: dict[str, float] = {}

    # 1. Load page image bytes
    page_files = [PROJECT_ROOT / pf for pf in song_meta["page_files"]]
    page_bytes_list: list[bytes] = []
    for pf in page_files:
        if not pf.exists():
            err_msg = f"Missing page file: {pf}"
            logger.error(err_msg)
            return {
                "slug": slug,
                "category": category,
                "title": title,
                "status": "FAILED",
                "error": err_msg,
            }
        page_bytes_list.append(pf.read_bytes())

    # 2. OMR Stage (with resume support)
    parsed_sheet: Optional[ParsedSheet] = None
    parsed_json_path = song_dir / "parsed.json"

    if resume and parsed_json_path.exists():
        try:
            parsed_sheet = ParsedSheet.model_validate_json(parsed_json_path.read_text(encoding="utf-8"))
            logger.info("[%s] Resumed OMR from %s (measures=%d)", slug, parsed_json_path, len(parsed_sheet.measures()))
        except Exception as e:
            logger.warning("[%s] Failed to load cached parsed.json: %s", slug, e)

    if not parsed_sheet:
        t0 = time.time()
        try:
            logger.info("[%s] Running Gemini OMR (%s)...", slug, omr_model)
            parsed_sheet = parse_pages(page_bytes_list, model=omr_model)
            parsed_json_path.write_text(parsed_sheet.model_dump_json(indent=2), encoding="utf-8")
            stage_timings["omr"] = round(time.time() - t0, 2)
            logger.info("[%s] OMR complete in %.2fs (systems=%d, measures=%d)",
                        slug, stage_timings["omr"], len(parsed_sheet.systems), len(parsed_sheet.measures()))
        except Exception as e:
            stage_timings["omr"] = round(time.time() - t0, 2)
            stage_failures["omr"] = f"{type(e).__name__}: {e}"
            logger.error("[%s] OMR failed: %s", slug, e)

    # 3. OMR Verify Stage (with resume support)
    verified_sheet: Optional[ParsedSheet] = None
    verified_json_path = song_dir / "verified.json"

    if resume and verified_json_path.exists():
        try:
            verified_sheet = ParsedSheet.model_validate_json(verified_json_path.read_text(encoding="utf-8"))
            logger.info("[%s] Resumed Verify from %s (issues=%d)", slug, verified_json_path, len(verified_sheet.issues))
        except Exception as e:
            logger.warning("[%s] Failed to load cached verified.json: %s", slug, e)

    if not verified_sheet and parsed_sheet:
        t0 = time.time()
        try:
            logger.info("[%s] Running OMR verification...", slug)
            verified_sheet = verify_sheet(page_bytes_list, parsed_sheet, use_llm=True)
            verified_json_path.write_text(verified_sheet.model_dump_json(indent=2), encoding="utf-8")
            stage_timings["verify"] = round(time.time() - t0, 2)
            logger.info("[%s] Verify complete in %.2fs (issues=%d)",
                        slug, stage_timings["verify"], len(verified_sheet.issues))
        except Exception as e:
            stage_timings["verify"] = round(time.time() - t0, 2)
            stage_failures["verify"] = f"{type(e).__name__}: {e}"
            logger.warning("[%s] Verify stage encountered error: %s", slug, e)
            verified_sheet = parsed_sheet
    elif not parsed_sheet:
        stage_failures["verify"] = "Skipped due to OMR failure"

    target_sheet = verified_sheet or parsed_sheet

    # 4. Opus Proxy Ground Truth Stage (Category a only)
    opus_metrics: dict[str, Any] = {
        "checked_measures": 0,
        "matched_measures": 0,
        "agreement_pct": None,
        "mismatches": [],
    }
    opus_cache_path = song_dir / "opus_gt.json"
    if category == "a" and target_sheet and not skip_opus:
        if resume and opus_cache_path.exists():
            try:
                opus_metrics = json.loads(opus_cache_path.read_text(encoding="utf-8"))
                logger.info("[%s] Resumed Opus GT from %s (agr=%s%%)", slug, opus_cache_path, opus_metrics.get("agreement_pct"))
            except Exception as e:
                logger.warning("[%s] Failed to load cached opus_gt.json: %s", slug, e)

        if not opus_metrics.get("checked_measures"):
            t0 = time.time()
            logger.info("[%s] Running Opus ground truth arbiter on system crops...", slug)
            try:
                chk, matched, mismatches = run_opus_ground_truth(slug, page_bytes_list, target_sheet)
                stage_timings["opus"] = round(time.time() - t0, 2)
                agr = round(100.0 * matched / chk, 1) if chk > 0 else 100.0
                opus_metrics = {
                    "checked_measures": chk,
                    "matched_measures": matched,
                    "agreement_pct": agr,
                    "mismatches": mismatches,
                }
                opus_cache_path.write_text(json.dumps(opus_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info("[%s] Opus ground truth complete in %.2fs: %d/%d (%.1f%% agreement)",
                            slug, stage_timings["opus"], matched, chk, agr)
            except Exception as e:
                stage_timings["opus"] = round(time.time() - t0, 2)
                stage_failures["opus"] = f"{type(e).__name__}: {e}"
                logger.warning("[%s] Opus ground truth error: %s", slug, e)

    # 5. Arrangement, Repair, and Overlay Rendering
    pdfs_generated: list[str] = []
    preview_pngs: list[str] = []
    all_violations_by_rule: dict[str, int] = {}
    fallback_repaired_measures_count = 0

    if target_sheet and target_sheet.measures():
        primary_key = extract_header_key(target_sheet, song_meta.get("key", "C"))
        test_keys = [primary_key, "G"] if primary_key != "G" else ["G", "C"]
        difficulties: list[Difficulty] = ["beginner", "intermediate", "advanced"]

        t0 = time.time()
        # Prepare sanitized sheet for arrangement to handle publisher variations
        try:
            arrange_ready_sheet = prepare_sheet_for_arrange(target_sheet)
        except Exception as e:
            arrange_ready_sheet = target_sheet

        for k in test_keys:
            for diff in difficulties:
                combo_name = f"{k}_{diff}"
                try:
                    # Piano Arrange
                    raw_arr = arrange(arrange_ready_sheet, start_key=k, difficulty=diff)

                    # Validate before repair
                    v_by_rule, _ = collect_violations_before_repair(arrange_ready_sheet, raw_arr)
                    for rule_code, cnt in v_by_rule.items():
                        all_violations_by_rule[rule_code] = all_violations_by_rule.get(rule_code, 0) + cnt

                    # Arrange Check & Safe Fallback Auto-repair
                    repaired_arr = check_and_repair(arrange_ready_sheet, raw_arr)

                    # Count auto-fixed measures
                    auto_fixes = [iss for iss in repaired_arr.issues if iss.severity == "auto_fixed"]
                    fallback_repaired_measures_count += len(auto_fixes)

                    # Render PDF
                    pdf_bytes = render_pdf(page_bytes_list, arrange_ready_sheet, repaired_arr)
                    pdf_path = song_dir / f"result_{combo_name}.pdf"
                    pdf_path.write_bytes(pdf_bytes)
                    pdfs_generated.append(str(pdf_path.relative_to(PROJECT_ROOT)))

                    # Render preview PNGs for intermediate on primary key
                    if diff == "intermediate" and k == primary_key:
                        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                        for p_no in range(len(doc)):
                            png_path = song_dir / f"p{p_no + 1}.png"
                            doc[p_no].get_pixmap(dpi=150).save(str(png_path))
                            preview_pngs.append(str(png_path.relative_to(PROJECT_ROOT)))

                except Exception as e:
                    err_key = f"arrange_{combo_name}"
                    stage_failures[err_key] = f"{type(e).__name__}: {e}"
                    logger.warning("[%s] Failed on %s: %s", slug, combo_name, e)

        stage_timings["arrange_and_render"] = round(time.time() - t0, 2)
    else:
        stage_failures["arrange"] = "Skipped due to missing parsed sheet or empty measures"

    # Summarize Quality Issues
    issues_by_severity = {"info": 0, "auto_fixed": 0, "needs_review": 0}
    needs_review_messages: list[str] = []
    if target_sheet:
        for iss in target_sheet.issues:
            issues_by_severity[iss.severity] = issues_by_severity.get(iss.severity, 0) + 1
            if iss.severity == "needs_review":
                needs_review_messages.append(f"M{iss.measure_index or 0 + 1}: {iss.message}")

    total_time = round(time.time() - t_start, 2)
    systems_count = len(target_sheet.systems) if target_sheet else 0
    measures_count = len(target_sheet.measures()) if target_sheet else 0
    chords_count = sum(len(m.chords) for m in target_sheet.measures()) if target_sheet else 0
    key_changes_count = len(target_sheet.key_changes) if target_sheet else 0

    result = {
        "slug": slug,
        "category": category,
        "title": title,
        "artist": song_meta.get("artist", ""),
        "style": song_meta.get("style", ""),
        "pages": song_meta.get("pages", len(page_files)),
        "systems": systems_count,
        "measures": measures_count,
        "chords": chords_count,
        "key_changes": key_changes_count,
        "detected_key": extract_header_key(target_sheet, song_meta.get("key", "C")) if target_sheet else None,
        "evaluated_keys": [extract_header_key(target_sheet, song_meta.get("key", "C")), "G"] if target_sheet else [],
        "issues": {
            "by_severity": issues_by_severity,
            "needs_review_count": len(needs_review_messages),
            "needs_review_messages": needs_review_messages,
        },
        "pre_repair_violations_by_rule": all_violations_by_rule,
        "fallback_repaired_measures_count": fallback_repaired_measures_count,
        "pdfs_generated": pdfs_generated,
        "preview_pngs": preview_pngs,
        "opus_agreement": opus_metrics,
        "stage_failures": stage_failures,
        "stage_timings": stage_timings,
        "elapsed_seconds": total_time,
        "status": "PASSED" if not stage_failures else ("PARTIAL" if pdfs_generated else "FAILED"),
    }
    logger.info("<<< Finished %s in %.2fs (status=%s, pdfs=%d)", slug, total_time, result["status"], len(pdfs_generated))
    return result


def generate_markdown_report(report_data: dict[str, Any]) -> str:
    """Generate comprehensive Markdown report."""
    md = []
    md.append("# Taiwanese Band Chart Batch Evaluation Report\n")
    md.append(f"**Execution Date**: {report_data.get('timestamp', '')}  ")
    md.append(f"**Total Songs Evaluated**: {report_data.get('total_songs', 0)}  ")
    md.append(f"**Overall Wall Time**: {report_data.get('total_elapsed_seconds', 0):.1f}s\n")

    # 1. Summary Table
    md.append("## 1. Dataset Evaluation Summary\n")
    md.append("| Slug | Cat | Title | Pages | Sys/M | Chords | Key Chg | Needs Review | Violations | Repaired | PDFs | Opus Agr | Status |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for s in report_data["songs"]:
        sys_m = f"{s['systems']}/{s['measures']}"
        nr = s["issues"]["needs_review_count"]
        v_cnt = sum(s["pre_repair_violations_by_rule"].values())
        rep = s["fallback_repaired_measures_count"]
        pdfs = len(s["pdfs_generated"])
        opus_str = f"{s['opus_agreement']['agreement_pct']}%" if s['opus_agreement']['agreement_pct'] is not None else "N/A"
        st = "✅ PASS" if not s["stage_failures"] else f"⚠️ {list(s['stage_failures'].keys())[0]}"
        md.append(f"| `{s['slug']}` | {s['category']} | {s['title']} | {s['pages']} | {sys_m} | {s['chords']} | {s['key_changes']} | {nr} | {v_cnt} | {rep} | {pdfs} | {opus_str} | {st} |")

    md.append("\n---\n")

    # 2. Category Analysis
    md.append("## 2. Category Performance Analysis\n")
    for cat in ["a", "b", "c"]:
        cat_songs = [s for s in report_data["songs"] if s["category"] == cat]
        if not cat_songs:
            continue
        cat_names = {
            "a": "Genuine Taiwanese Band Charts (Number Notation Chords)",
            "b": "Mainland-style Jianpu (Western Letter Chords)",
            "c": "Hard Cases (Skewed Phone Photo & Handwritten Manuscripts)"
        }
        md.append(f"### Category ({cat}): {cat_names.get(cat, cat)}\n")
        md.append(f"- **Count**: {len(cat_songs)} songs")
        total_m = sum(s["measures"] for s in cat_songs)
        total_c = sum(s["chords"] for s in cat_songs)
        total_rep = sum(s["fallback_repaired_measures_count"] for s in cat_songs)
        md.append(f"- **Total Measures Extracted**: {total_m}")
        md.append(f"- **Total Chords Extracted**: {total_c}")
        md.append(f"- **Fallback Repaired Measures**: {total_rep}\n")

        for s in cat_songs:
            md.append(f"#### `{s['slug']}` - {s['title']} ({s['style']})")
            md.append(f"- **Pages**: {s['pages']}, **Systems**: {s['systems']}, **Measures**: {s['measures']}, **Key**: {s['detected_key']}")
            if s["key_changes"] > 0:
                md.append(f"- **Key Changes Detected**: {s['key_changes']} modulations successfully tracked")
            if s["stage_failures"]:
                md.append(f"- **Stage Failures**: {json.dumps(s['stage_failures'], ensure_ascii=False)}")
            if s["issues"]["needs_review_messages"]:
                md.append(f"- **Needs Review Warnings**: {', '.join(s['issues']['needs_review_messages'][:3])}")
            md.append("")

    # 3. Arrangement Invariant Violations (Pre-Repair)
    md.append("## 3. Pre-Repair Arrangement Invariant Violations\n")
    md.append("Violations detected on raw arrangements prior to safe fallback repair:\n")
    md.append("| Violation Rule Code | Total Occurrences | Description |")
    md.append("|---|---|---|")
    rule_descriptions = {
        "consecutive_same_finger": "Sequential notes within 0.5 beats played with identical finger",
        "fingering_not_monotonic": "Chord fingering does not strictly match ascending/descending pitch order",
        "span_too_large": "Hand span exceeds difficulty limit (8 for beginner, 12 for int/adv)",
        "pitch_out_of_range": "Pitch outside playable tessitura (RH E4..A5, LH C2..E4)",
        "voice_leading_leap": "Melodic or harmonic voice leap exceeds interval limit",
    }
    all_violations = report_data["overall_metrics"]["total_pre_repair_violations_by_rule"]
    for code, count in sorted(all_violations.items(), key=lambda x: x[1], reverse=True):
        desc = rule_descriptions.get(code, "Domain rule violation")
        md.append(f"| `{code}` | {count} | {desc} |")

    md.append(f"\n**Total Safe Fallback Measures Applied**: {report_data['overall_metrics']['total_fallback_repaired_measures']}\n")

    # 4. Opus Proxy Ground Truth Agreement
    md.append("## 4. Independent Claude Opus Ground Truth Agreement (Category a)\n")
    opus_data = report_data["overall_metrics"]["opus_ground_truth"]
    md.append(f"- **Total Measures Audited**: {opus_data['total_checked']}")
    md.append(f"- **Total Verbatim Matches**: {opus_data['total_matched']}")
    md.append(f"- **Overall Chord Agreement Rate**: **{opus_data['overall_agreement_pct']}%**\n")

    if opus_data["mismatch_tuples"]:
        md.append("### Mismatch Tuples Log: `(slug, page, row, measure, pipeline, opus)`\n")
        md.append("| Slug | Page | Row | Measure | Pipeline Chords | Opus Arbiter Chords |")
        md.append("|---|---|---|---|---|---|")
        for m in opus_data["mismatch_tuples"][:25]:
            md.append(f"| `{m[0]}` | P{m[1]} | R{m[2]} | M{m[3]} | `{m[4]}` | `{m[5]}` |")

    # 5. Top Preview Output Files
    md.append("\n## 5. Generated Artifact Previews\n")
    top_pngs = report_data["top_preview_pngs"]
    for p in top_pngs:
        md.append(f"- `{p}`")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate Taiwanese band chart conversion pipeline")
    parser.add_argument("--manifest", default="fixtures/external/manifest.json", help="Path to dataset manifest")
    parser.add_argument("--out-dir", default="out/batch", help="Output directory")
    parser.add_argument("--omr-model", default=os.environ.get("OMR_MODEL", "gemini-2.5-flash"), help="Gemini OMR model")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent song workers")
    parser.add_argument("--songs", default=None, help="Comma-separated slugs to run (optional)")
    parser.add_argument("--skip-opus", action="store_true", help="Skip Opus proxy evaluation")
    parser.add_argument("--no-resume", action="store_true", help="Disable caching and re-run all stages")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    songs = manifest.get("songs", [])

    if args.songs:
        selected_slugs = set(args.songs.split(","))
        songs = [s for s in songs if s["slug"] in selected_slugs]

    logger.info("Loaded %d songs to evaluate from %s (workers=%d, OMR=%s, skip_opus=%s, resume=%s)",
                len(songs), manifest_path, args.workers, args.omr_model, args.skip_opus, not args.no_resume)

    t_global_start = time.time()
    song_results: list[dict[str, Any]] = []

    # Execute concurrent evaluation
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_song = {
            executor.submit(evaluate_song, s, out_dir, args.omr_model, args.skip_opus, not args.no_resume): s["slug"]
            for s in songs
        }
        for future in concurrent.futures.as_completed(future_to_song):
            slug = future_to_song[future]
            try:
                res = future.result()
                song_results.append(res)
            except Exception as e:
                logger.error("Unhandled error evaluating %s: %s\n%s", slug, e, traceback.format_exc())
                song_results.append({
                    "slug": slug,
                    "status": "FATAL",
                    "error": str(e),
                })

    # Preserve manifest order
    slug_order = {s["slug"]: i for i, s in enumerate(songs)}
    song_results.sort(key=lambda s: slug_order.get(s["slug"], 999))

    total_elapsed = round(time.time() - t_global_start, 2)

    # Aggregate Metrics
    overall_violations: dict[str, int] = {}
    total_repaired = 0
    total_measures = 0
    total_chords = 0
    opus_checked = 0
    opus_matched = 0
    opus_mismatches: list[list[Any]] = []
    top_preview_pngs: list[str] = []

    for r in song_results:
        total_measures += r.get("measures", 0)
        total_chords += r.get("chords", 0)
        total_repaired += r.get("fallback_repaired_measures_count", 0)
        for code, cnt in r.get("pre_repair_violations_by_rule", {}).items():
            overall_violations[code] = overall_violations.get(code, 0) + cnt
        op = r.get("opus_agreement", {})
        opus_checked += op.get("checked_measures", 0)
        opus_matched += op.get("matched_measures", 0)
        for m in op.get("mismatches", []):
            opus_mismatches.append(m.get("tuple", []))
        for p in r.get("preview_pngs", []):
            if len(top_preview_pngs) < 5:
                top_preview_pngs.append(p)

    opus_agr_pct = round(100.0 * opus_matched / opus_checked, 1) if opus_checked > 0 else 100.0

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_songs": len(song_results),
        "total_elapsed_seconds": total_elapsed,
        "omr_model": args.omr_model,
        "songs": song_results,
        "top_preview_pngs": top_preview_pngs,
        "overall_metrics": {
            "total_measures": total_measures,
            "total_chords": total_chords,
            "total_pre_repair_violations_by_rule": overall_violations,
            "total_fallback_repaired_measures": total_repaired,
            "opus_ground_truth": {
                "total_checked": opus_checked,
                "total_matched": opus_matched,
                "overall_agreement_pct": opus_agr_pct,
                "mismatch_tuples": opus_mismatches,
            },
        },
    }

    # Write report.json
    report_json_path = out_dir / "report.json"
    report_json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved report JSON to %s", report_json_path)

    # Write report.md
    report_md_text = generate_markdown_report(report_data)
    report_md_path = out_dir / "report.md"
    report_md_path.write_text(report_md_text, encoding="utf-8")
    logger.info("Saved report Markdown to %s", report_md_path)

    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETED")
    print(f"Total songs: {len(song_results)} in {total_elapsed:.1f}s")
    print(f"Measures: {total_measures}, Chords: {total_chords}")
    print(f"Opus Chord Agreement: {opus_agr_pct}% ({opus_matched}/{opus_checked})")
    print(f"Pre-repair Violations: {sum(overall_violations.values())}")
    print(f"Fallback Repaired: {total_repaired}")
    print(f"Reports: {report_json_path}, {report_md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
