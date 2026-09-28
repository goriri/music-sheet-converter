"""Evaluation runner for OMR verification layer against clean and corrupted fixtures.

Measures:
- Clean fixture: false positives (needs_review count <= 2, wrong auto-fixes == 0)
- Corrupted fixtures:
    * detected count
    * auto-fixed-correctly (target >= 80%)
    * auto-fixed-wrongly (target == 0)
    * flagged-for-review
    * missed
    * total fixed or flagged (target >= 95%)
- Added latency per page (target <= 30 s/page)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, ".")

from app.models import ParsedSheet
from app.qa.omr_verify import verify_sheet


def load_images() -> list[bytes]:
    p1 = Path("fixtures/pages/page1.jpg").read_bytes()
    p2 = Path("fixtures/pages/page2.jpg").read_bytes()
    return [p1, p2]


def run_evaluation() -> dict[str, Any]:
    images = load_images()
    clean_path = Path("fixtures/omr_sample.json")
    manifest_path = Path("fixtures/corrupted/manifest.json")

    with open(clean_path, encoding="utf-8") as f:
        clean_sheet = ParsedSheet.model_validate_json(f.read())

    with open(manifest_path, encoding="utf-8") as f:
        manifest: dict[str, list[dict[str, Any]]] = json.load(f)

    results: dict[str, Any] = {}

    print("=" * 80)
    print("STEP 1: EVALUATING CLEAN FIXTURE")
    print("=" * 80)

    t0 = time.time()
    verified_clean = verify_sheet(images, clean_sheet, use_llm=True)
    clean_duration = time.time() - t0
    clean_lat_per_page = clean_duration / len(images)

    clean_needs_review = [i for i in verified_clean.issues if i.severity == "needs_review"]
    clean_auto_fixed = [i for i in verified_clean.issues if i.severity == "auto_fixed"]

    results["clean"] = {
        "duration_s": round(clean_duration, 2),
        "latency_per_page_s": round(clean_lat_per_page, 2),
        "needs_review_count": len(clean_needs_review),
        "auto_fixed_count": len(clean_auto_fixed),
        "issues": [i.model_dump() for i in verified_clean.issues],
    }

    print(f"Clean fixture verified in {clean_duration:.2f}s ({clean_lat_per_page:.2f}s/page)")
    print(f"  needs_review count: {len(clean_needs_review)} (Target: <= 2)")
    print(f"  auto_fixed count:   {len(clean_auto_fixed)} (Target: 0)")
    for iss in verified_clean.issues:
        print(f"    [{iss.severity}] m{iss.measure_index}: {iss.message}")

    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING CORRUPTED FIXTURES")
    print("=" * 80)

    total_injected_all = 0
    total_detected_all = 0
    total_fixed_correct_all = 0
    total_fixed_wrong_all = 0
    total_flagged_all = 0
    total_missed_all = 0

    corrupted_results: dict[str, Any] = {}

    for fname, err_list in manifest.items():
        fpath = Path("fixtures/corrupted") / fname
        if not fpath.exists():
            continue

        with open(fpath, encoding="utf-8") as f:
            corrupted_sheet = ParsedSheet.model_validate_json(f.read())

        t_start = time.time()
        verified_corrupted = verify_sheet(images, corrupted_sheet, use_llm=True)
        dur = time.time() - t_start
        lat_per_page = dur / len(images)

        # Lookup measures in verified sheet
        meas_lookup = {m.index: m for m in verified_corrupted.measures()}

        detected = 0
        fixed_correct = 0
        fixed_wrong = 0
        flagged = 0
        missed = 0

        # Check each injected error
        for err in err_list:
            m_idx = err["measure_index"]
            c_idx = err["chord_index"]
            field = err["field"]
            exp_val = err["expected_value"]
            bad_val = err["corrupted_value"]
            err_type = err["error_type"]

            # Findings on this measure
            m_issues = [i for i in verified_corrupted.issues if i.measure_index == m_idx]
            v_meas = meas_lookup.get(m_idx)

            is_detected = False
            is_fixed_correct = False
            is_fixed_wrong = False
            is_flagged = False

            if field == "chord.raw" and v_meas and c_idx is not None and c_idx < len(v_meas.chords):
                actual_chord = v_meas.chords[c_idx].raw
                if exp_val is not None:
                    if actual_chord == exp_val:
                        is_detected = True
                        is_fixed_correct = True
                    elif actual_chord != bad_val:
                        is_detected = True
                        is_fixed_wrong = True
                    elif any(i.severity == "needs_review" for i in m_issues):
                        is_detected = True
                        is_flagged = True
                    else:
                        is_detected = False
                else:
                    # Garbage chord injected: correctly flagged if needs_review
                    if any(i.severity == "needs_review" for i in m_issues):
                        is_detected = True
                        is_flagged = True
                    elif actual_chord != bad_val:
                        is_detected = True
                        is_fixed_wrong = True
                    else:
                        is_detected = False
            elif field == "chord.beat" and v_meas and c_idx is not None and c_idx < len(v_meas.chords):
                actual_beat = v_meas.chords[c_idx].beat
                if actual_beat == exp_val:
                    is_detected = True
                    is_fixed_correct = True
                elif actual_beat != bad_val:
                    is_detected = True
                    is_fixed_wrong = True
                elif any(i.severity == "needs_review" for i in m_issues):
                    is_detected = True
                    is_flagged = True
            elif field == "key_changes":
                if len(verified_corrupted.key_changes) > 0:
                    is_detected = True
                    is_fixed_correct = True
                elif any(i.severity == "needs_review" for i in m_issues):
                    is_detected = True
                    is_flagged = True
            elif field == "measure.melody":
                if any(i.code == "melody_beat_sum_mismatch" for i in m_issues):
                    is_detected = True
                    is_flagged = True  # Melody corruption flagged for review
            elif field == "measure.chords":
                # Missing / deleted chord in measure: check if restored or flagged
                if v_meas and c_idx is not None and c_idx < len(v_meas.chords):
                    if v_meas.chords[c_idx].raw == exp_val:
                        is_detected = True
                        is_fixed_correct = True
                    elif any(i.severity == "needs_review" for i in m_issues):
                        is_detected = True
                        is_flagged = True
                elif any(i.severity == "needs_review" for i in m_issues):
                    is_detected = True
                    is_flagged = True
                else:
                    is_detected = False

            if is_fixed_correct:
                fixed_correct += 1
                detected += 1
            elif is_fixed_wrong:
                fixed_wrong += 1
                detected += 1
            elif is_flagged:
                flagged += 1
                detected += 1
            else:
                missed += 1

        total_injected = len(err_list)
        total_injected_all += total_injected
        total_detected_all += detected
        total_fixed_correct_all += fixed_correct
        total_fixed_wrong_all += fixed_wrong
        total_flagged_all += flagged
        total_missed_all += missed

        fix_rate = (fixed_correct / total_injected * 100.0) if total_injected > 0 else 0.0
        covered_rate = ((fixed_correct + flagged) / total_injected * 100.0) if total_injected > 0 else 0.0

        corrupted_results[fname] = {
            "injected": total_injected,
            "detected": detected,
            "fixed_correct": fixed_correct,
            "fixed_wrong": fixed_wrong,
            "flagged": flagged,
            "missed": missed,
            "fix_rate_pct": round(fix_rate, 1),
            "covered_rate_pct": round(covered_rate, 1),
            "duration_s": round(dur, 2),
            "lat_per_page_s": round(lat_per_page, 2),
        }

        print(f"\nFile: {fname} (took {dur:.2f}s, {lat_per_page:.2f}s/page)")
        print(f"  Injected:      {total_injected}")
        print(f"  Detected:      {detected}")
        print(f"  Auto-fixed OK: {fixed_correct} ({fix_rate:.1f}%) [Target >= 80%]")
        print(f"  Auto-fixed WR: {fixed_wrong} [Target == 0]")
        print(f"  Flagged:       {flagged}")
        print(f"  Missed:        {missed}")
        print(f"  Coverage:      {covered_rate:.1f}% [Target >= 95%]")

    overall_fix_rate = (total_fixed_correct_all / total_injected_all * 100.0) if total_injected_all > 0 else 0.0
    overall_covered_rate = ((total_fixed_correct_all + total_flagged_all) / total_injected_all * 100.0) if total_injected_all > 0 else 0.0

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY TABLE")
    print("=" * 80)
    print(f"{'File':<30} | Injected | Detected | Auto-Fixed | Wrong | Flagged | Missed | Lat/page")
    print("-" * 95)
    for fname, r in corrupted_results.items():
        print(
            f"{fname:<30} | "
            f"{r['injected']:<8} | "
            f"{r['detected']:<8} | "
            f"{r['fixed_correct']:<10} | "
            f"{r['fixed_wrong']:<5} | "
            f"{r['flagged']:<7} | "
            f"{r['missed']:<6} | "
            f"{r['lat_per_page_s']:.1f}s"
        )
    print("-" * 95)
    print(
        f"{'TOTAL / OVERALL':<30} | "
        f"{total_injected_all:<8} | "
        f"{total_detected_all:<8} | "
        f"{total_fixed_correct_all:<10} | "
        f"{total_fixed_wrong_all:<5} | "
        f"{total_flagged_all:<7} | "
        f"{total_missed_all:<6} | "
        f"Fix: {overall_fix_rate:.1f}%"
    )
    print("=" * 80)

    results["corrupted"] = corrupted_results
    results["overall"] = {
        "total_injected": total_injected_all,
        "total_detected": total_detected_all,
        "total_fixed_correct": total_fixed_correct_all,
        "total_fixed_wrong": total_fixed_wrong_all,
        "total_flagged": total_flagged_all,
        "total_missed": total_missed_all,
        "fix_rate_pct": round(overall_fix_rate, 1),
        "covered_rate_pct": round(overall_covered_rate, 1),
    }

    return results


if __name__ == "__main__":
    run_evaluation()
