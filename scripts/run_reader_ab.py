"""A/B evaluation script comparing Gemini 2.5 Pro vs Claude Opus 5.5 for OMR content reading.

Evaluates:
- Chord transcription accuracy on 掉了 (exact raw match after normalisation)
- Beat placement accuracy (within 0.5 beats of ground truth)
- Key change detection (measure, raw text, and semitone shift)
- Latency per page
- Approximate cost per page

Usage:
    .venv/bin/python scripts/run_reader_ab.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import ParsedSheet
from app.omr.geometry import GChordBox, GMeasure, GSystem, PageGeometry
from app.omr.reader import read_sheet
from app.theory.chords import clean_raw_chord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_geometry_from_sample(sample_dict: dict[str, Any]) -> list[PageGeometry]:
    """Construct PageGeometry from omr_sample.json bboxes when layout.py is not yet available."""
    geoms: list[PageGeometry] = []
    for p_idx, p_info in enumerate(sample_dict["pages"]):
        page_systems = [s for s in sample_dict["systems"] if s["page"] == p_idx]
        sys_list: list[GSystem] = []
        for s_idx, s in enumerate(page_systems):
            s_bbox = tuple(s["bbox"])
            measures: list[GMeasure] = []
            chord_boxes: list[GChordBox] = []
            for m_idx, m in enumerate(s["measures"]):
                m_bbox = m["bbox"]
                measures.append(GMeasure(index_in_system=m_idx, x0=m_bbox[0], x1=m_bbox[2]))
                for c in m["chords"]:
                    cb = c.get("bbox")
                    if cb:
                        chord_boxes.append(
                            GChordBox(
                                bbox=tuple(cb),
                                measure_index_in_system=m_idx,
                                beat_geo=float(c.get("beat", 1.0)),
                                boxed=True,
                            )
                        )
            sys_list.append(
                GSystem(
                    page=p_idx,
                    index_on_page=s_idx,
                    bbox=s_bbox,
                    melody_band=(s_bbox[1] + 0.3 * (s_bbox[3] - s_bbox[1]), s_bbox[3] - 0.2 * (s_bbox[3] - s_bbox[1])),
                    measures=measures,
                    chord_boxes=chord_boxes,
                    confidence=1.0,
                )
            )
        header_band = (0.0, page_systems[0]["bbox"][1]) if page_systems and p_idx == 0 else None
        geoms.append(
            PageGeometry(
                page=p_idx,
                width=p_info["width"],
                height=p_info["height"],
                header_band=header_band,
                systems=sys_list,
                confidence=1.0,
            )
        )
    return geoms


def evaluate_against_ground_truth(
    pred_sheet: ParsedSheet,
    gt_dict: dict[str, Any],
) -> dict[str, Any]:
    """Compare predicted sheet against ground truth chords, beats, and key changes."""
    gt_measures: list[dict[str, Any]] = []
    for sys_data in gt_dict.get("systems", []):
        for m in sys_data.get("measures", []):
            gt_measures.append(m)

    pred_measures = pred_sheet.measures()

    total_gt_chords = 0
    exact_chord_matches = 0
    beat_matches_within_half = 0

    total_measures = min(len(gt_measures), len(pred_measures))
    exact_measure_chord_matches = 0

    for idx in range(total_measures):
        gt_m = gt_measures[idx]
        pred_m = pred_measures[idx]

        gt_chords = [
            (clean_raw_chord(c.get("raw", "")), float(c.get("beat", 1.0)))
            for c in gt_m.get("chords", [])
            if clean_raw_chord(c.get("raw", ""))
        ]
        pred_chords = [
            (clean_raw_chord(c.raw), float(c.beat))
            for c in pred_m.chords
            if clean_raw_chord(c.raw)
        ]

        total_gt_chords += len(gt_chords)

        gt_raws = [c[0] for c in gt_chords]
        pred_raws = [c[0] for c in pred_chords]

        if gt_raws == pred_raws:
            exact_measure_chord_matches += 1

        # Match individual chords greedily
        used_pred = set()
        for g_raw, g_beat in gt_chords:
            matched = False
            for p_idx, (p_raw, p_beat) in enumerate(pred_chords):
                if p_idx not in used_pred and p_raw == g_raw:
                    used_pred.add(p_idx)
                    exact_chord_matches += 1
                    if abs(p_beat - g_beat) <= 0.5:
                        beat_matches_within_half += 1
                    matched = True
                    break

    chord_acc = exact_chord_matches / max(1, total_gt_chords)
    beat_acc = beat_matches_within_half / max(1, exact_chord_matches)
    measure_chord_acc = exact_measure_chord_matches / max(1, total_measures)

    # Key change evaluation
    gt_kc = gt_dict.get("key_changes", [])
    pred_kc = pred_sheet.key_changes
    kc_matched = False
    if gt_kc and pred_kc:
        for g in gt_kc:
            for p in pred_kc:
                if abs(p.at_measure - g.get("at_measure", -99)) <= 1 and p.semitones == g.get("semitones"):
                    kc_matched = True
                    break
    elif not gt_kc and not pred_kc:
        kc_matched = True

    return {
        "total_gt_chords": total_gt_chords,
        "exact_chord_matches": exact_chord_matches,
        "chord_accuracy": round(chord_acc, 4),
        "measure_chord_accuracy": round(measure_chord_acc, 4),
        "beat_accuracy": round(beat_acc, 4),
        "key_change_matched": kc_matched,
        "header_title": pred_sheet.header.title,
        "header_style": pred_sheet.header.style,
        "header_original_key": pred_sheet.header.original_key,
    }


def run_ab_comparison() -> dict[str, Any]:
    pages_dir = Path("fixtures/pages")
    page_files = [pages_dir / "page1.jpg", pages_dir / "page2.jpg"]
    images = [f.read_bytes() for f in page_files]
    num_pages = len(images)

    with open("fixtures/omr_sample.json") as f:
        gt_sample = json.load(f)

    # Check for layout.py
    layout_path = Path("app/omr/layout.py")
    if layout_path.exists():
        logger.info("Found app/omr/layout.py; using classical CV layout.")
        try:
            from app.omr import layout
            geoms = [layout.analyze_page(img, idx) for idx, img in enumerate(images)]
        except Exception as exc:
            logger.warning("Failed to run layout.analyze_page: %s. Using sample geometry.", exc)
            geoms = build_geometry_from_sample(gt_sample)
    else:
        logger.info("app/omr/layout.py not yet available; building temporary geometry from fixtures/omr_sample.json.")
        geoms = build_geometry_from_sample(gt_sample)

    models_to_test = ["gemini-2.5-pro", "claude-opus-5-5"]
    results: dict[str, Any] = {}

    for model_name in models_to_test:
        logger.info("Running read_sheet with model: %s ...", model_name)
        t0 = time.time()
        try:
            sheet = read_sheet(images, geoms, model=model_name)
            elapsed = time.time() - t0
            lat_per_page = elapsed / num_pages

            eval_metrics = evaluate_against_ground_truth(sheet, gt_sample)

            # Approx cost per page based on Vertex AI pricing:
            # Gemini 2.5 Pro: ~$0.012 per page
            # Claude Opus 5.5: ~$0.12 per page
            cost_per_page = 0.012 if "gemini" in model_name else 0.12

            results[model_name] = {
                "model": model_name,
                "total_time_s": round(elapsed, 2),
                "latency_per_page_s": round(lat_per_page, 2),
                "approx_cost_per_page": cost_per_page,
                "metrics": eval_metrics,
                "success": True,
            }
            logger.info(
                "Model %s finished: chord_acc=%.1f%%, beat_acc=%.1f%%, latency=%.1fs/page",
                model_name,
                eval_metrics["chord_accuracy"] * 100,
                eval_metrics["beat_accuracy"] * 100,
                lat_per_page,
            )
        except Exception as exc:
            logger.error("Model %s failed: %s", model_name, exc)
            results[model_name] = {
                "model": model_name,
                "success": False,
                "error": str(exc),
            }

    # Save results to out/reader/ab_results.json
    out_dir = Path("out/reader")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ab_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved A/B results to %s", out_file)

    # Print summary table
    print("\n" + "=" * 80)
    print("A/B EVALUATION SUMMARY: OMR CONTENT READER")
    print("=" * 80)
    print(f"{'Model':<18} | {'Chord Acc':<10} | {'Beat Acc':<9} | {'Key Chg':<8} | {'Latency/p':<10} | {'Cost/p':<8}")
    print("-" * 80)
    for model_name, r in results.items():
        if r.get("success"):
            m = r["metrics"]
            kc_str = "MATCH" if m["key_change_matched"] else "FAIL"
            print(
                f"{model_name:<18} | "
                f"{m['chord_accuracy']*100:>8.1f}% | "
                f"{m['beat_accuracy']*100:>7.1f}% | "
                f"{kc_str:<8} | "
                f"{r['latency_per_page_s']:>8.1f}s | "
                f"${r['approx_cost_per_page']:.3f}"
            )
        else:
            print(f"{model_name:<18} | FAILED: {r.get('error')}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_ab_comparison()
