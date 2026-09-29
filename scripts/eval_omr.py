#!/usr/bin/env python3
"""OMR evaluation tool against hand-checked ground truth.

Computes:
- row count accuracy
- measures-per-row accuracy
- chord exact-match accuracy with alignment by (row, measure)
- secondary sequence-alignment score per row (to separate 'shifted by one measure' from 'misread')
- beat accuracy (±0.5)
- key-change accuracy

Outputs a human-readable table and a machine-readable JSON line 'EVAL_JSON:'.
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import ParsedSheet

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_omr")


def normalize_chord(raw: str) -> str:
    """Normalize chord representation for comparison."""
    if not raw:
        return ""
    c = raw.strip()
    c = c.replace("（", "(").replace("）", ")").replace("＃", "#").replace("♯", "#").replace("♭", "b")
    circle_map = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6", "⑦": "7"}
    for circ, num in circle_map.items():
        c = c.replace(circ, num)
    # Remove whitespace and lower case
    c = "".join(c.split()).lower()
    return c


def longest_common_subsequence(seq1: list[str], seq2: list[str]) -> int:
    """Length of LCS between two chord sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def evaluate_song(
    gt_data: dict[str, Any],
    candidate_sheet: ParsedSheet,
) -> dict[str, Any]:
    """Evaluate candidate ParsedSheet against ground truth."""
    gt_rows = gt_data.get("rows", [])
    cand_systems = candidate_sheet.systems

    # 1. Row count accuracy
    total_gt_rows = len(gt_rows)
    total_cand_rows = len(cand_systems)
    row_count_match = (total_gt_rows == total_cand_rows)
    row_count_acc = 1.0 if row_count_match else min(total_cand_rows, total_gt_rows) / max(total_cand_rows, total_gt_rows, 1)

    # 2. Measures-per-row accuracy
    matched_row_meas_count = 0
    # Also evaluate chord matches per row
    total_gt_chords = 0
    exact_matched_chords = 0
    total_lcs_matched_chords = 0
    beat_eligible_chords = 0
    beat_matched_chords = 0

    # Build index of candidate measures by (page_1based, row_in_page_1based)
    cand_by_page_row: dict[tuple[int, int], list[Any]] = {}
    for s_idx, sys_obj in enumerate(cand_systems):
        p_idx = sys_obj.page + 1
        p_sys = [s for s in cand_systems if s.page == sys_obj.page]
        r_in_page = p_sys.index(sys_obj) + 1
        cand_by_page_row[(p_idx, r_in_page)] = sys_obj.measures

    for r_idx, r_gt in enumerate(gt_rows):
        p_no = r_gt.get("page", 1)
        row_no = r_gt.get("row", r_idx + 1)
        gt_measures = r_gt.get("measures", [])
        num_gt_m = len(gt_measures)

        # Candidate measures for this row
        cand_measures = cand_by_page_row.get((p_no, row_no))
        if cand_measures is None and r_idx < len(cand_systems):
            cand_measures = cand_systems[r_idx].measures
        if cand_measures is None:
            cand_measures = []

        if len(cand_measures) == num_gt_m:
            matched_row_meas_count += 1

        # Evaluate chords in this row
        row_gt_chords_norm: list[str] = []
        row_cand_chords_norm: list[str] = []

        for m_idx, m_gt in enumerate(gt_measures):
            gt_m_chords = m_gt.get("chords", [])
            cand_m_chords = cand_measures[m_idx].chords if m_idx < len(cand_measures) else []

            norm_cand_m = [(normalize_chord(c.raw), c.beat) for c in cand_m_chords]
            cand_raws = [c[0] for c in norm_cand_m]

            for c_gt in gt_m_chords:
                raw_gt = c_gt.get("raw", "")
                if raw_gt == "?":
                    continue
                total_gt_chords += 1
                norm_gt = normalize_chord(raw_gt)
                gt_beat = float(c_gt.get("beat", 1.0))
                row_gt_chords_norm.append(norm_gt)

                # Check exact match by (row, measure)
                if norm_gt in cand_raws:
                    exact_matched_chords += 1
                    beat_eligible_chords += 1
                    # Check beat match (±0.5)
                    cand_match_idx = cand_raws.index(norm_gt)
                    cand_beat = norm_cand_m[cand_match_idx][1]
                    if abs(cand_beat - gt_beat) <= 0.5:
                        beat_matched_chords += 1
                    # Remove matched candidate to prevent double counting
                    cand_raws.pop(cand_match_idx)
                    norm_cand_m.pop(cand_match_idx)

        for m_cand in cand_measures:
            for c in m_cand.chords:
                row_cand_chords_norm.append(normalize_chord(c.raw))

        # Secondary sequence alignment score per row (LCS)
        lcs_len = longest_common_subsequence(row_gt_chords_norm, row_cand_chords_norm)
        total_lcs_matched_chords += lcs_len

    meas_per_row_acc = matched_row_meas_count / total_gt_rows if total_gt_rows > 0 else 1.0
    chord_exact_acc = exact_matched_chords / total_gt_chords if total_gt_chords > 0 else 1.0
    chord_seq_acc = total_lcs_matched_chords / total_gt_chords if total_gt_chords > 0 else 1.0
    beat_acc = beat_matched_chords / beat_eligible_chords if beat_eligible_chords > 0 else 1.0

    # 3. Key change accuracy
    gt_key_changes = gt_data.get("key_changes", [])
    cand_key_changes = candidate_sheet.key_changes

    # Map GT key change locations to global measure index
    gt_kc_global = []
    # Calculate cumulative measure offsets for GT
    curr_m = 0
    gt_meas_lookup: dict[tuple[int, int, int], int] = {}
    for r_idx, r_gt in enumerate(gt_rows):
        p_no = r_gt.get("page", 1)
        row_no = r_gt.get("row", r_idx + 1)
        for m_idx in range(len(r_gt.get("measures", []))):
            gt_meas_lookup[(p_no, row_no, m_idx + 1)] = curr_m
            curr_m += 1

    for kc in gt_key_changes:
        p = kc.get("page", 1)
        r = kc.get("row", 1)
        m = kc.get("measure", 1)
        glob_m = gt_meas_lookup.get((p, r, m), 0)
        gt_kc_global.append({
            "measure": glob_m,
            "semitones": kc.get("semitones", 0),
            "text": kc.get("text", "")
        })

    cand_kc_list = [
        {"measure": kc.at_measure, "semitones": kc.semitones, "raw": kc.raw}
        for kc in cand_key_changes
    ]

    if not gt_kc_global and not cand_kc_list:
        kc_acc = 1.0
    elif not gt_kc_global and cand_kc_list:
        kc_acc = 0.0  # False alarms
    elif gt_kc_global and not cand_kc_list:
        kc_acc = 0.0  # Missed
    else:
        # Match by semitones and close measure position (±2 measures)
        kc_matches = 0
        cand_kc_copy = list(cand_kc_list)
        for g_kc in gt_kc_global:
            for idx, c_kc in enumerate(cand_kc_copy):
                if g_kc["semitones"] == c_kc["semitones"] and abs(g_kc["measure"] - c_kc["measure"]) <= 2:
                    kc_matches += 1
                    cand_kc_copy.pop(idx)
                    break
        kc_acc = kc_matches / max(len(gt_kc_global), len(cand_kc_list))

    return {
        "slug": gt_data.get("slug", ""),
        "row_count": {
            "gt": total_gt_rows,
            "candidate": total_cand_rows,
            "match": row_count_match,
            "acc": round(row_count_acc, 3),
        },
        "meas_per_row": {
            "matched_rows": matched_row_meas_count,
            "total_rows": total_gt_rows,
            "acc": round(meas_per_row_acc, 3),
        },
        "chord_exact": {
            "matched": exact_matched_chords,
            "total": total_gt_chords,
            "acc": round(chord_exact_acc, 3),
        },
        "chord_seq": {
            "matched": total_lcs_matched_chords,
            "total": total_gt_chords,
            "acc": round(chord_seq_acc, 3),
        },
        "beat": {
            "matched": beat_matched_chords,
            "eligible": beat_eligible_chords,
            "acc": round(beat_acc, 3),
        },
        "key_change": {
            "gt_count": len(gt_kc_global),
            "cand_count": len(cand_kc_list),
            "acc": round(kc_acc, 3),
        }
    }


def run_evaluation(
    gt_dir: Path,
    batch_dir: Optional[Path] = None,
    candidate_file: Optional[Path] = None,
    slug: Optional[str] = None,
    live: bool = False,
) -> dict[str, Any]:
    """Run evaluation across songs and print report."""
    results = {}
    gt_files = list(gt_dir.glob("*.json"))
    if slug:
        gt_files = [f for f in gt_files if f.stem == slug]

    if not gt_files:
        raise FileNotFoundError(f"No groundtruth JSON files found in {gt_dir}")

    for gt_path in sorted(gt_files):
        s_slug = gt_path.stem
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))

        cand_sheet: Optional[ParsedSheet] = None
        if live:
            from app.omr.gemini_omr import parse_pages
            # Load images
            meta_pages = gt_data.get("pages", 1)
            img_bytes_list = []
            if s_slug == "diaole":
                paths = ["fixtures/pages/page1.jpg", "fixtures/pages/page2.jpg"]
            else:
                paths = [f"fixtures/external/{s_slug}/page{i}.jpg" for i in range(1, meta_pages + 1)]
            for p in paths:
                img_bytes_list.append((PROJECT_ROOT / p).read_bytes())
            cand_sheet = parse_pages(img_bytes_list)
        elif candidate_file:
            cand_sheet = ParsedSheet.model_validate_json(candidate_file.read_text(encoding="utf-8"))
        elif batch_dir:
            cand_path = batch_dir / s_slug / "verified.json"
            if not cand_path.exists():
                # Try fallback for diaole
                if s_slug == "diaole":
                    cand_path = PROJECT_ROOT / "fixtures/omr_sample.json"
            if cand_path.exists():
                cand_sheet = ParsedSheet.model_validate_json(cand_path.read_text(encoding="utf-8"))
            else:
                logger.warning("[%s] Candidate file not found at %s", s_slug, cand_path)
                continue

        if cand_sheet is not None:
            metrics = evaluate_song(gt_data, cand_sheet)
            results[s_slug] = metrics

    # Aggregate overall metrics
    if results:
        overall = {
            "songs_evaluated": len(results),
            "row_count_acc": round(sum(r["row_count"]["acc"] for r in results.values()) / len(results), 3),
            "meas_per_row_acc": round(sum(r["meas_per_row"]["acc"] for r in results.values()) / len(results), 3),
            "chord_exact_acc": round(
                sum(r["chord_exact"]["matched"] for r in results.values()) /
                max(1, sum(r["chord_exact"]["total"] for r in results.values())),
                3
            ),
            "chord_seq_acc": round(
                sum(r["chord_seq"]["matched"] for r in results.values()) /
                max(1, sum(r["chord_seq"]["total"] for r in results.values())),
                3
            ),
            "beat_acc": round(
                sum(r["beat"]["matched"] for r in results.values()) /
                max(1, sum(r["beat"]["eligible"] for r in results.values())),
                3
            ),
            "key_change_acc": round(sum(r["key_change"]["acc"] for r in results.values()) / len(results), 3),
        }
    else:
        overall = {}

    # Print Table
    print("\n" + "=" * 92)
    print(f"{'Slug':<15} | {'Rows':<8} | {'M/Row Acc':<10} | {'Chord(Row,M)':<13} | {'Chord(Seq)':<11} | {'Beat Acc':<9} | {'KeyChg Acc':<10}")
    print("-" * 92)
    for s_slug, res in results.items():
        r_str = f"{res['row_count']['candidate']}/{res['row_count']['gt']}"
        m_acc = f"{res['meas_per_row']['acc']*100:.1f}%"
        c_acc = f"{res['chord_exact']['acc']*100:.1f}%"
        s_acc = f"{res['chord_seq']['acc']*100:.1f}%"
        b_acc = f"{res['beat']['acc']*100:.1f}%"
        kc_acc = f"{res['key_change']['acc']*100:.1f}%"
        print(f"{s_slug:<15} | {r_str:<8} | {m_acc:<10} | {c_acc:<13} | {s_acc:<11} | {b_acc:<9} | {kc_acc:<10}")
    print("-" * 92)
    if overall:
        print(f"{'OVERALL':<15} | {'-':<8} | {overall['meas_per_row_acc']*100:.1f}%{'':<5} | {overall['chord_exact_acc']*100:.1f}%{'':<8} | {overall['chord_seq_acc']*100:.1f}%{'':<6} | {overall['beat_acc']*100:.1f}%{'':<4} | {overall['key_change_acc']*100:.1f}%")
    print("=" * 92 + "\n")

    eval_output = {"overall": overall, "songs": results}
    print(f"EVAL_JSON: {json.dumps(eval_output, ensure_ascii=False)}")
    return eval_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR against Ground Truth")
    parser.add_argument("--gt-dir", type=Path, default=PROJECT_ROOT / "fixtures/groundtruth")
    parser.add_argument("--batch-dir", type=Path, default=PROJECT_ROOT / "out/batch")
    parser.add_argument("--candidate", type=Path, help="Single candidate ParsedSheet JSON")
    parser.add_argument("--slug", type=str, help="Evaluate single song")
    parser.add_argument("--live", action="store_true", help="Run live parse_pages")
    args = parser.parse_args()

    run_evaluation(
        gt_dir=args.gt_dir,
        batch_dir=args.batch_dir,
        candidate_file=args.candidate,
        slug=args.slug,
        live=args.live,
    )


if __name__ == "__main__":
    main()
