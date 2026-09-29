#!/usr/bin/env python3
"""Evaluation script for web evidence corroboration against ground truth sheets."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import ChordSymbol, KeyChange, Measure, PageInfo, ParsedSheet, SongHeader, System
from app.qa.web_evidence import (
    ALIGNMENT_QUALITY_THRESHOLD,
    align_reference,
    find_reference,
    parse_chord_to_degree,
)
from app.theory.keys import key_name_to_pc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("web_evidence_eval")

GROUNDTRUTH_SONGS: list[dict[str, Any]] = [
    {
        "slug": "diandao",
        "title": "倒带",
        "artist": "蔡依林",
        "default_key": "C",
    },
    {
        "slug": "diaole",
        "title": "掉了",
        "artist": "张惠妹",
        "default_key": "F#",
    },
    {
        "slug": "qianlizhiwai",
        "title": "千里之外",
        "artist": "周杰伦 / 费玉清",
        "default_key": "D",
    },
    {
        "slug": "tinghai",
        "title": "听海",
        "artist": "张惠妹",
        "default_key": "Bb",
    },
    {
        "slug": "xiaobaichuan",
        "title": "小白船",
        "artist": "朝鲜童谣",
        "default_key": "Eb",
    },
    {
        "slug": "liusha",
        "title": "流沙",
        "artist": "陶喆",
        "default_key": "A",
    },
    {
        "slug": "huochuai",
        "title": "活出爱",
        "artist": "盛晓玫",
        "default_key": "C",
    },
]


def sheet_from_gt_json(gt_path: Path, title: str, default_key: str) -> tuple[ParsedSheet, int]:
    """Reconstruct a ParsedSheet from a ground truth JSON file."""
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    header_data = gt_data.get("header", {})
    orig_key_str = header_data.get("original_key") or default_key
    start_key_pc = key_name_to_pc(orig_key_str)

    header = SongHeader(
        title=title,
        original_key=orig_key_str,
        chord_notation=header_data.get("chord_notation", "number"),
        time_signature=header_data.get("time_signature", "4/4"),
    )

    systems: list[System] = []
    global_m_idx = 0
    for r_idx, r_gt in enumerate(gt_data.get("rows", [])):
        p_no = r_gt.get("page", 1) - 1
        measures: list[Measure] = []
        for m_gt in r_gt.get("measures", []):
            chords: list[ChordSymbol] = []
            for c_gt in m_gt.get("chords", []):
                raw = c_gt.get("raw", "")
                if not raw or raw == "?":
                    continue
                chords.append(
                    ChordSymbol(
                        raw=raw,
                        beat=float(c_gt.get("beat", 1.0)),
                        stacked=bool(c_gt.get("stacked", False)),
                    )
                )
            measures.append(
                Measure(
                    index=global_m_idx,
                    bbox=(0.0, 0.0, 1.0, 1.0),
                    chords=chords,
                    melody=m_gt.get("melody", ""),
                    lyrics=m_gt.get("lyrics", ""),
                )
            )
            global_m_idx += 1
        systems.append(
            System(
                page=p_no,
                bbox=(0.0, 0.0, 1.0, 1.0),
                measures=measures,
            )
        )

    key_changes: list[KeyChange] = []
    for kc in gt_data.get("key_changes", []):
        key_changes.append(
            KeyChange(
                at_measure=kc.get("measure", 0),
                raw=kc.get("text", ""),
                semitones=kc.get("semitones", 0),
            )
        )

    sheet = ParsedSheet(
        header=header,
        pages=[PageInfo(width=1000, height=1400)],
        systems=systems,
        key_changes=key_changes,
    )
    return sheet, start_key_pc


def evaluate_song(song_info: dict[str, Any], gt_dir: Path) -> dict[str, Any]:
    """Run find_reference + align_reference against answer key sheet for one song."""
    slug = song_info["slug"]
    title = song_info["title"]
    artist = song_info["artist"]
    default_key = song_info["default_key"]

    gt_path = gt_dir / f"{slug}.json"
    if not gt_path.is_file():
        logger.warning("Groundtruth file not found: %s", gt_path)
        return {
            "slug": slug,
            "title": title,
            "artist": artist,
            "found": False,
            "sources": 0,
            "alignment_quality": 0.0,
            "root_agreement": 0.0,
            "exact_agreement": 0.0,
            "total_measures": 0,
            "evaluated_measures": 0,
        }

    sheet, start_key_pc = sheet_from_gt_json(gt_path, title, default_key)
    measures = sheet.measures()
    measures_with_chords = [m for m in measures if any(c.raw and c.raw != "?" for c in m.chords)]
    total_eval_measures = len(measures_with_chords)

    ref = find_reference(title, artist)
    if ref is None or not ref.sources:
        return {
            "slug": slug,
            "title": title,
            "artist": artist,
            "found": False,
            "sources": 0,
            "alignment_quality": 0.0,
            "root_agreement": 0.0,
            "exact_agreement": 0.0,
            "total_measures": len(measures),
            "evaluated_measures": total_eval_measures,
        }

    source_count = len(ref.sources)
    candidates_by_measure = align_reference(sheet, ref, start_key_pc)

    if not candidates_by_measure:
        return {
            "slug": slug,
            "title": title,
            "artist": artist,
            "found": True,
            "sources": source_count,
            "alignment_quality": 0.0,
            "root_agreement": 0.0,
            "exact_agreement": 0.0,
            "total_measures": len(measures),
            "evaluated_measures": total_eval_measures,
        }

    # Compute alignment quality and agreement
    quality_sum = sum(cand[0].support for cand in candidates_by_measure.values())
    alignment_quality = quality_sum / max(1, total_eval_measures)

    root_matches = 0
    exact_matches = 0

    for m in measures_with_chords:
        primary_gt = m.chords[0].raw
        m_tonic = (
            start_key_pc + sum(kc.semitones for kc in sheet.key_changes if kc.at_measure <= m.index)
        ) % 12
        gt_deg = parse_chord_to_degree(
            primary_gt,
            tonic_pc=m_tonic,
            is_letter_hint=(sheet.header.chord_notation == "letter"),
        )
        if gt_deg is None:
            continue

        cands = candidates_by_measure.get(m.index)
        if not cands:
            continue

        top_cand = cands[0]
        ref_deg = parse_chord_to_degree(top_cand.chord, tonic_pc=m_tonic)
        if ref_deg is None:
            continue

        # Root agreement (same root degree & bass)
        if ref_deg.root_semitone == gt_deg.root_semitone:
            root_matches += 1

        # Exact agreement (same root, same bass, same quality)
        if (
            ref_deg.root_semitone == gt_deg.root_semitone
            and ref_deg.bass_semitone == gt_deg.bass_semitone
            and ref_deg.quality == gt_deg.quality
        ):
            exact_matches += 1

    root_acc = root_matches / max(1, total_eval_measures)
    exact_acc = exact_matches / max(1, total_eval_measures)

    return {
        "slug": slug,
        "title": title,
        "artist": artist,
        "found": True,
        "sources": source_count,
        "alignment_quality": round(alignment_quality, 3),
        "root_agreement": round(root_acc, 3),
        "exact_agreement": round(exact_acc, 3),
        "total_measures": len(measures),
        "evaluated_measures": total_eval_measures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate web evidence corroboration against ground truth.")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=PROJECT_ROOT / "fixtures/groundtruth",
        help="Path to groundtruth directory",
    )
    args = parser.parse_args()

    print(f"Evaluating {len(GROUNDTRUTH_SONGS)} groundtruth songs against web evidence...\n")

    results: list[dict[str, Any]] = []
    for song in GROUNDTRUTH_SONGS:
        print(f"Processing '{song['title']}' ({song['artist']})...", flush=True)
        res = evaluate_song(song, args.gt_dir)
        results.append(res)

    print("\n" + "=" * 90)
    print(
        f"{'Song':<10} | {'Artist':<16} | {'Found':<5} | {'#Src':<4} | {'Align Qual':<10} | {'Root Agree':<10} | {'Exact Agree':<11}"
    )
    print("-" * 90)
    for r in results:
        found_str = "YES" if r["found"] else "NO"
        print(
            f"{r['title']:<10} | {r['artist']:<16} | {found_str:<5} | {r['sources']:<4} | {r['alignment_quality']:<10.3f} | {r['root_agreement']:<10.1%} | {r['exact_agreement']:<11.1%}"
        )
    print("=" * 90)

    # Save summary report
    out_dir = PROJECT_ROOT / "out/web"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved evaluation summary to: {report_path}")


if __name__ == "__main__":
    main()
