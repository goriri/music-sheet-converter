#!/usr/bin/env python3
"""Draft ground-truth builder for OMR evaluation.

Crops printed rows from sheet images (with generous margins), upscales 3x,
and queries TWO independent LLM readers (Claude Opus arbiter and Gemini 2.5 Pro reader)
to transcribe per measure: chord symbols verbatim with beat.
Caches LLM results, compares and flags disagreements for human resolution.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.qa.llm import ask_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("make_gt_draft")

# ---------------------------------------------------------------------------
# Pydantic Schemas for Dual Reader Transcription
# ---------------------------------------------------------------------------
class TranscribedChord(BaseModel):
    raw: str = Field(
        description="Chord symbol verbatim as printed inside box/circle or above measure, "
                    "e.g. '1(2)', '5/7', '2m7/5', '1add9', '6m', '4maj7', '17/7b', 'b7', 'C', 'G/B'. '?' if unreadable."
    )
    beat: float = Field(
        1.0,
        description="1-based beat inside measure where chord starts (e.g. 1.0, 2.0, 3.0, 4.0, or 2.5, 3.5, etc.)"
    )


class TranscribedMeasure(BaseModel):
    measure_number: int = Field(
        description="1-based measure index in this row from left to right"
    )
    chords: list[TranscribedChord] = Field(
        default_factory=list,
        description="Chords in this measure in left-to-right order. Empty list if no chord printed."
    )


class RowTranscription(BaseModel):
    measure_count: int = Field(
        description="Total number of measures separated by bar lines (| or ||) in this row"
    )
    measures: list[TranscribedMeasure] = Field(
        description="List of all measures from left to right"
    )


# ---------------------------------------------------------------------------
# Song Row Geometries (generous margins, verified by inspection)
# ---------------------------------------------------------------------------
SONG_DEFINITIONS: dict[str, dict[str, Any]] = {
    "diaole": {
        "pages": 2,
        "files": ["fixtures/pages/page1.jpg", "fixtures/pages/page2.jpg"],
        "header": {
            "time_signature": "4/4",
            "original_key": "F#",
            "chord_notation": "number",
        },
        "key_changes": [
            {"page": 2, "row": 3, "measure": 3, "semitones": 2, "text": "(轉成2調)(Ab)"}
        ],
        # Rows defined from omr_sample.json systems with padding
        "use_omr_sample": True,
    },
    "qianlizhiwai": {
        "pages": 1,
        "files": ["fixtures/external/qianlizhiwai/page1.jpg"],
        "header": {
            "time_signature": "4/4",
            "original_key": "D",
            "chord_notation": "number",
        },
        "key_changes": [],
        "row_ranges": {
            1: [
                (170, 335),   # Row 1: 3 measures
                (335, 465),   # Row 2: 3 measures
                (470, 640),   # Row 3: 4 measures
                (645, 785),   # Row 4: 4 measures
                (790, 955),   # Row 5: 4 measures
                (960, 1080),  # Row 6: 4 measures
                (1085, 1245), # Row 7: 4 measures
                (1250, 1385), # Row 8: 4 measures
                (1390, 1545), # Row 9: 4 measures
                (1550, 1680), # Row 10: 4 measures
            ]
        },
    },
    "tinghai": {
        "pages": 2,
        "files": ["fixtures/external/tinghai/page1.jpg", "fixtures/external/tinghai/page2.jpg"],
        "header": {
            "time_signature": "4/4",
            "original_key": "Bb",
            "chord_notation": "number",
        },
        "key_changes": [],
        "row_ranges": {
            1: [
                (310, 490),   # Row 1
                (495, 650),   # Row 2
                (655, 785),   # Row 3
                (790, 920),   # Row 4
                (925, 1055),  # Row 5
                (1060, 1205), # Row 6
                (1210, 1355), # Row 7
                (1360, 1505), # Row 8
                (1510, 1655), # Row 9
                (1660, 1805), # Row 10
                (1810, 1935), # Row 11
                (1940, 2040), # Row 12
            ],
            2: [
                (280, 460),   # Row 1
                (460, 620),   # Row 2
                (620, 800),   # Row 3
                (800, 980),   # Row 4
                (980, 1150),  # Row 5
                (1150, 1320), # Row 6
                (1320, 1490), # Row 7
                (1490, 1670), # Row 8
            ]
        },
    },
    "diandao": {
        "pages": 2,
        "files": ["fixtures/external/diandao/page1.jpg", "fixtures/external/diandao/page2.jpg"],
        "header": {
            "time_signature": "4/4",
            "original_key": "C",
            "chord_notation": "number",
        },
        "key_changes": [],
        "row_ranges": {
            1: [
                (270, 480),   # Row 1: 前奏
                (520, 700),   # Row 2: 主歌
                (720, 890),   # Row 3
                (900, 1060),  # Row 4: 桥段
                (1070, 1240), # Row 5
                (1250, 1340), # Row 6: 副歌
                (1350, 1500), # Row 7
                (1510, 1660), # Row 8: 间奏
                (1670, 1860), # Row 9: 尾奏
            ],
            2: [
                (220, 480),   # Row 1: 主歌
                (490, 720),   # Row 2
                (730, 930),   # Row 3: 桥段
                (940, 1130),  # Row 4
                (1140, 1330), # Row 5: 副歌
                (1340, 1530), # Row 6
                (1530, 1650), # Row 7: 1 measure
            ]
        },
    },
    "xiaobaichuan": {
        "pages": 2,
        "files": ["fixtures/external/xiaobaichuan/page1.jpg", "fixtures/external/xiaobaichuan/page2.jpg"],
        "header": {
            "time_signature": "3/4",
            "original_key": "Eb",
            "chord_notation": "number",
        },
        "key_changes": [],
        "row_ranges": {
            1: [
                (300, 550),   # Row 1: 前奏 1
                (560, 750),   # Row 2: 前奏 2
                (760, 950),   # Row 3: 主歌1 1
                (960, 1150),  # Row 4: 主歌1 2
                (1160, 1350), # Row 5: 主歌2 1
                (1360, 1550), # Row 6: 主歌2 2
                (1560, 1750), # Row 7: 副歌1 1
                (1760, 1950), # Row 8: 副歌1 2
                (1960, 2150), # Row 9: 副歌2 1
                (2160, 2330), # Row 10: 副歌2 2
            ],
            2: [
                (350, 610),   # Row 1: 主歌1 1
                (610, 850),   # Row 2: 主歌1 2
                (850, 1090),  # Row 3: 主歌2 1
                (1090, 1330), # Row 4: 主歌2 2
                (1330, 1570), # Row 5: 副歌1 1
                (1570, 1810), # Row 6: 副歌1 2
                (1810, 2050), # Row 7: 副歌2 1
                (2050, 2290), # Row 8: 副歌2 2
            ]
        },
    },
    "huochuai": {
        "pages": 2,
        "files": ["fixtures/external/huochuai/page1.jpg", "fixtures/external/huochuai/page2.jpg"],
        "header": {
            "time_signature": "12/8",
            "original_key": "C",
            "chord_notation": "number",
        },
        "key_changes": [],
        "row_ranges": {
            1: [
                (750, 920),   # Row 1: (a) rhythm
                (980, 1130),  # Row 2: 《分散伴奏》① and ②
                (1150, 1310), # Row 3: (b) 節奏
                (1340, 1510), # Row 4: ① and ② (Slow Rock accompaniment pattern)
                (1540, 1820), # Row 5: variation of ①
            ],
            2: [
                (480, 750),   # Row 1: (例) 4 measures melody example
            ]
        },
    },
    "liusha": {
        "pages": 2,
        "files": ["fixtures/external/liusha/page1.jpg", "fixtures/external/liusha/page2.jpg"],
        "header": {
            "time_signature": "4/4",
            "original_key": "Ab",
            "chord_notation": "number",
        },
        "key_changes": [],
        "row_ranges": {
            1: [
                (240, 380),   # Row 1: 前奏 (pickup + 4 measures)
                (380, 500),   # Row 2: 4 measures
                (500, 650),   # Row 3: 主歌 4 measures
                (650, 780),   # Row 4: 4 measures
                (780, 930),   # Row 5: 桥段 (pickup + 4 measures)
                (930, 1060),  # Row 6: 2 measures
                (1060, 1250), # Row 7: 副歌 4 measures
                (1250, 1460), # Row 8: 4 measures
                (1460, 1680), # Row 9: 间奏 4 measures
                (1680, 1860), # Row 10: 变奏 4 measures
                (1860, 2050), # Row 11: 2 measures
                (2050, 2220), # Row 12: 尾奏 4 measures
                (2220, 2350), # Row 13: 1 measure
            ],
            2: [
                (280, 480),   # Row 1: 主歌 4 measures
                (490, 720),   # Row 2: 4 measures
                (730, 920),   # Row 3: 桥段 4 measures
                (930, 1100),  # Row 4: 2 measures
                (1110, 1300), # Row 5: 副歌 4 measures
                (1310, 1500), # Row 6: 4 measures
                (1510, 1650), # Row 7: 1 measure
            ]
        },
    },
}


def normalize_chord_str(chord: str) -> str:
    """Normalize chord string for comparison."""
    if not chord:
        return ""
    c = chord.strip()
    c = c.replace("（", "(").replace("）", ")").replace("＃", "#").replace("♯", "#").replace("♭", "b")
    circle_map = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6", "⑦": "7"}
    for circ, num in circle_map.items():
        c = c.replace(circ, num)
    c = "".join(c.split()).lower()
    return c


def crop_and_upscale_row(
    img: np.ndarray,
    y0: int,
    y1: int,
    x0: int = 0,
    x1: Optional[int] = None,
    scale: int = 3,
) -> np.ndarray:
    """Crop row and upscale by scale factor."""
    h, w = img.shape[:2]
    x_end = x1 if x1 is not None else w
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x_end)]
    if scale > 1:
        crop = cv2.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def ask_dual_readers(
    crop_path: Path,
    crop_bytes: bytes,
    slug: str,
    page: int,
    row: int,
    force_refresh: bool = False,
) -> tuple[RowTranscription, RowTranscription]:
    """Query Claude Opus and Gemini 2.5 Pro independently with local disk caching."""
    cache_dir = PROJECT_ROOT / f"out/gt/{slug}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_opus = cache_dir / f"cache_opus_p{page}_r{row:02d}.json"
    cache_gemini = cache_dir / f"cache_gemini_p{page}_r{row:02d}.json"

    prompt = (
        "You are an expert music chord reader transcribing Taiwanese band charts / jianpu music sheets.\n"
        "Inspect this row crop carefully from left to right.\n"
        "Identify each measure bounded by bar lines (| or ||). Include pickup measures if separated by a bar line.\n"
        "For each measure from left to right:\n"
        "Transcribe all printed chord symbols exactly as printed inside the chord box or circle "
        "(e.g. '1(2)', '5/7', '2m7/5', '1add9', '6m', '4maj7', '17/7b', 'b7', 'C', 'G/B', etc.), "
        "along with their 1-based starting beat in the measure (1.0, 2.0, 3.0, 4.0, or half-beats like 2.5, 3.5).\n"
        "If a measure has no chord printed, its chords list must be empty: [].\n"
        "If any chord is unreadable or uncertain, mark raw as '?' rather than guessing.\n"
        "Return: measure_count (total number of measures), and measures list in left-to-right order."
    )

    # 1. Gemini 2.5 Pro
    if cache_gemini.exists() and not force_refresh:
        try:
            gemini_data = json.loads(cache_gemini.read_text(encoding="utf-8"))
            gemini_res = RowTranscription.model_validate(gemini_data)
        except Exception:
            gemini_res = None
    else:
        gemini_res = None

    if gemini_res is None:
        logger.info("[%s p%d r%d] Querying Gemini 2.5 Pro...", slug, page, row)
        for attempt in range(4):
            try:
                gemini_res = ask_json(
                    prompt,
                    RowTranscription,
                    images=[crop_bytes],
                    role="reader",
                    model="gemini-2.5-pro",
                    timeout_s=90.0,
                )
                break
            except Exception as e:
                if attempt < 3 and "429" in str(e):
                    time.sleep(3.0 * (attempt + 1))
                else:
                    raise
        cache_gemini.write_text(
            json.dumps(gemini_res.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # 2. Claude Opus (arbiter)
    if cache_opus.exists() and not force_refresh:
        try:
            opus_data = json.loads(cache_opus.read_text(encoding="utf-8"))
            opus_res = RowTranscription.model_validate(opus_data)
        except Exception:
            opus_res = None
    else:
        opus_res = None

    if opus_res is None:
        logger.info("[%s p%d r%d] Querying Claude Opus...", slug, page, row)
        for attempt in range(4):
            try:
                opus_res = ask_json(
                    prompt,
                    RowTranscription,
                    images=[crop_bytes],
                    role="arbiter",
                    timeout_s=90.0,
                )
                break
            except Exception as e:
                if attempt < 3 and "429" in str(e):
                    time.sleep(3.0 * (attempt + 1))
                else:
                    raise
        cache_opus.write_text(
            json.dumps(opus_res.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return opus_res, gemini_res


def compare_transcriptions(
    opus: RowTranscription,
    gemini: RowTranscription,
) -> tuple[bool, list[str]]:
    """Compare Opus and Gemini transcriptions. Return (is_match, list_of_diffs)."""
    diffs = []
    if opus.measure_count != gemini.measure_count:
        diffs.append(
            f"Measure count mismatch: Opus={opus.measure_count}, Gemini={gemini.measure_count}"
        )

    max_m = max(len(opus.measures), len(gemini.measures))
    for m_idx in range(max_m):
        m_num = m_idx + 1
        m_op = opus.measures[m_idx] if m_idx < len(opus.measures) else None
        m_ge = gemini.measures[m_idx] if m_idx < len(gemini.measures) else None

        if m_op is None:
            diffs.append(f"M{m_num}: missing in Opus, Gemini has {len(m_ge.chords)} chords")
            continue
        if m_ge is None:
            diffs.append(f"M{m_num}: missing in Gemini, Opus has {len(m_op.chords)} chords")
            continue

        op_chords = [normalize_chord_str(c.raw) for c in m_op.chords]
        ge_chords = [normalize_chord_str(c.raw) for c in m_ge.chords]

        if op_chords != ge_chords:
            diffs.append(
                f"M{m_num} chord mismatch: Opus={[c.raw for c in m_op.chords]}, "
                f"Gemini={[c.raw for c in m_ge.chords]}"
            )
        else:
            # Check beats
            for c_op, c_ge in zip(m_op.chords, m_ge.chords):
                if abs(c_op.beat - c_ge.beat) > 0.5:
                    diffs.append(
                        f"M{m_num} beat mismatch for '{c_op.raw}': Opus={c_op.beat}, Gemini={c_ge.beat}"
                    )

    return len(diffs) == 0, diffs


def _draft_worker(task_info):
    slug, p_no, r_no, c_path, c_bytes, force_refresh = task_info
    op_res, ge_res = ask_dual_readers(
        c_path, c_bytes, slug, p_no, r_no, force_refresh=force_refresh
    )
    return (p_no, r_no, c_path, op_res, ge_res)


def process_song_draft(slug: str, force_refresh: bool = False) -> dict[str, Any]:
    """Process a single song, crop rows, ask dual readers, merge and record disagreements."""
    meta = SONG_DEFINITIONS.get(slug)
    if not meta:
        raise ValueError(f"Unknown slug: {slug}")

    out_gt_dir = PROJECT_ROOT / f"out/gt/{slug}"
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = PROJECT_ROOT / "fixtures/groundtruth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Special handling for diaole if use_omr_sample is True
    if meta.get("use_omr_sample"):
        logger.info("[%s] Processing from trusted omr_sample.json...", slug)
        omr_sample_path = PROJECT_ROOT / "fixtures/omr_sample.json"
        with open(omr_sample_path, encoding="utf-8") as f:
            sample_data = json.load(f)

        rows = []
        for s_idx, sys_obj in enumerate(sample_data["systems"], 1):
            p_idx = sys_obj["page"] + 1
            # Row index within page
            p_sys = [s for s in sample_data["systems"] if s["page"] == sys_obj["page"]]
            r_in_page = p_sys.index(sys_obj) + 1

            meas_list = []
            for m in sys_obj["measures"]:
                ch_list = []
                for c in m["chords"]:
                    raw_str = c["raw"].strip()
                    beat_val = float(c.get("beat", 1.0))
                    ch_list.append({"raw": raw_str, "beat": beat_val})
                meas_list.append({"chords": ch_list})
            rows.append({
                "page": p_idx,
                "row": r_in_page,
                "measures": meas_list
            })

        gt_data = {
            "slug": slug,
            "pages": meta["pages"],
            "header": meta["header"],
            "key_changes": meta["key_changes"],
            "rows": rows
        }
        return {
            "gt_data": gt_data,
            "disagreements": [],
            "stats": {
                "rows": len(rows),
                "measures": sum(len(r["measures"]) for r in rows),
                "chords": sum(len(m["chords"]) for r in rows for m in r["measures"]),
                "disagreements": 0,
                "unreadable": 0,
            }
        }

    # Standard processing with dual LLM readers
    row_ranges = meta["row_ranges"]
    all_rows = []
    disagreements_record = []
    total_chords = 0
    total_unreadable = 0

    tasks_to_run = []
    for page_no, img_file in enumerate(meta["files"], 1):
        img_path = PROJECT_ROOT / img_file
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        ranges = row_ranges.get(page_no, [])
        for row_no, (y0, y1) in enumerate(ranges, 1):
            crop = crop_and_upscale_row(img, y0, y1, scale=3)
            crop_path = out_gt_dir / f"row_p{page_no}_r{row_no:02d}.jpg"
            cv2.imwrite(str(crop_path), crop)

            _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            crop_bytes = buf.tobytes()
            tasks_to_run.append((slug, page_no, row_no, crop_path, crop_bytes, force_refresh))

    import concurrent.futures

    results_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futs = {executor.submit(_draft_worker, t): t for t in tasks_to_run}
        for fut in concurrent.futures.as_completed(futs):
            p_no, r_no, c_path, op_res, ge_res = fut.result()
            results_map[(p_no, r_no)] = (c_path, op_res, ge_res)

    # Reassemble in order
    for page_no, img_file in enumerate(meta["files"], 1):
        ranges = row_ranges.get(page_no, [])
        for row_no in range(1, len(ranges) + 1):
            crop_path, opus_res, gemini_res = results_map[(page_no, row_no)]
            is_match, diffs = compare_transcriptions(opus_res, gemini_res)

            if not is_match:
                logger.warning(
                    "[%s p%d r%d] Disagreement between Opus and Gemini:\n  %s",
                    slug, page_no, row_no, "\n  ".join(diffs)
                )
                disagreements_record.append({
                    "slug": slug,
                    "page": page_no,
                    "row": row_no,
                    "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                    "diffs": diffs,
                    "opus": opus_res.model_dump(),
                    "gemini": gemini_res.model_dump(),
                })

            chosen = opus_res
            row_measures = []
            for m in chosen.measures:
                m_chords = []
                for c in m.chords:
                    raw = c.raw.strip()
                    if raw == "?":
                        total_unreadable += 1
                    total_chords += 1
                    m_chords.append({"raw": raw, "beat": c.beat})
                row_measures.append({"chords": m_chords})

            all_rows.append({
                "page": page_no,
                "row": row_no,
                "measures": row_measures,
            })

    gt_data = {
        "slug": slug,
        "pages": meta["pages"],
        "header": meta["header"],
        "key_changes": meta["key_changes"],
        "rows": all_rows,
    }

    return {
        "gt_data": gt_data,
        "disagreements": disagreements_record,
        "stats": {
            "rows": len(all_rows),
            "measures": sum(len(r["measures"]) for r in all_rows),
            "chords": total_chords,
            "disagreements": len(disagreements_record),
            "unreadable": total_unreadable,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Build draft ground-truth for OMR evaluation")
    parser.add_argument("--slug", type=str, help="Specific song slug to process")
    parser.add_argument("--all", action="store_true", help="Process all 7 songs")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached LLM responses")
    args = parser.parse_args()

    slugs = [args.slug] if args.slug else list(SONG_DEFINITIONS.keys()) if args.all else ["qianlizhiwai"]

    for slug in slugs:
        logger.info("=== Processing slug: %s ===", slug)
        res = process_song_draft(slug, force_refresh=args.force_refresh)
        out_draft_path = PROJECT_ROOT / f"out/gt/{slug}/draft.json"
        out_draft_path.parent.mkdir(parents=True, exist_ok=True)
        out_draft_path.write_text(
            json.dumps(res["gt_data"], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(
            "[%s] Draft saved to %s (stats: %s)",
            slug, out_draft_path, res["stats"]
        )


if __name__ == "__main__":
    main()
