#!/usr/bin/env python3
"""Offline beat evaluation tool.

Re-quantises ParsedSheet candidates offline using app.omr.beats.quantize_chord_beats,
verifies that chord count and identity are unchanged, and evaluates strict beat accuracy
(beat exact match) before and after quantization against:
- fixtures/groundtruth/*.json for dev songs (out/gt_eval/*.json and out/live/diaole_live.json)
- fixtures/holdout/chords/*.json for held-out songs (via scripts/eval_omr.py)
"""
from __future__ import annotations

import copy
import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models import ParsedSheet
from app.omr.beats import quantize_chord_beats
from scripts.eval_omr import evaluate_song

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def run_dev_eval() -> dict[str, dict]:
    """Evaluate dev candidate files before and after quantization."""
    gt_dir = PROJECT_ROOT / "fixtures/groundtruth"
    dev_files = sorted(glob.glob(str(PROJECT_ROOT / "out/gt_eval/*.json")))
    live_file = PROJECT_ROOT / "out/live/diaole_live.json"
    if live_file.exists():
        dev_files.append(str(live_file))

    results = {}

    for cand_str in dev_files:
        cand_path = Path(cand_str)
        label = cand_path.stem
        slug = label.replace("_live", "").replace("_verified", "").replace("_parsed", "")
        gt_path = gt_dir / f"{slug}.json"
        if not gt_path.exists():
            continue

        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        cand_raw = json.loads(cand_path.read_text(encoding="utf-8"))

        sheet_before = ParsedSheet.model_validate(cand_raw)
        metrics_before = evaluate_song(gt_data, sheet_before)

        orig_chords = [
            (c.raw, c.confidence, c.stacked)
            for s in sheet_before.systems
            for m in s.measures
            for c in m.chords
        ]

        sheet_after = quantize_chord_beats(copy.deepcopy(sheet_before))

        new_chords = [
            (c.raw, c.confidence, c.stacked)
            for s in sheet_after.systems
            for m in s.measures
            for c in m.chords
        ]
        assert orig_chords == new_chords, f"[{label}] Chord identity or count modified by quantiser!"

        metrics_after = evaluate_song(gt_data, sheet_after)

        results[label] = {
            "before": metrics_before,
            "after": metrics_after,
        }

    return results


def run_holdout_eval() -> dict[str, dict]:
    """Evaluate held-out candidates before and after quantization via eval_omr.py."""
    gt_dir = PROJECT_ROOT / "fixtures/holdout/chords"
    gt_slugs = set(f.stem for f in gt_dir.glob("*.json"))
    holdout_files = sorted(glob.glob(str(PROJECT_ROOT / "out/holdout_*.json")))

    req_dir = PROJECT_ROOT / "out/holdout_requantized"
    req_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for f_str in holdout_files:
        f_path = Path(f_str)
        slug = f_path.stem.replace("holdout_", "")
        if slug not in gt_slugs:
            continue

        # 1. Run eval_omr before
        cmd_before = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/eval_omr.py"),
            "--gt-dir", str(gt_dir),
            "--candidate", str(f_path),
            "--slug", slug,
        ]
        proc_bef = subprocess.run(cmd_before, capture_output=True, text=True, check=True)
        metrics_before = {}
        for line in proc_bef.stdout.splitlines():
            if line.startswith("EVAL_JSON:"):
                metrics_before = json.loads(line[10:])["songs"][f_path.stem]

        # 2. Re-quantise
        cand_sheet = ParsedSheet.model_validate_json(f_path.read_text(encoding="utf-8"))
        orig_chords = [
            (c.raw, c.confidence, c.stacked)
            for s in cand_sheet.systems
            for m in s.measures
            for c in m.chords
        ]

        req_sheet = quantize_chord_beats(cand_sheet)

        new_chords = [
            (c.raw, c.confidence, c.stacked)
            for s in req_sheet.systems
            for m in s.measures
            for c in m.chords
        ]
        assert orig_chords == new_chords, f"[{slug}] Chord identity or count modified by quantiser!"

        req_file = req_dir / f_path.name
        req_file.write_text(req_sheet.model_dump_json(indent=2), encoding="utf-8")

        # 3. Run eval_omr after
        cmd_after = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/eval_omr.py"),
            "--gt-dir", str(gt_dir),
            "--candidate", str(req_file),
            "--slug", slug,
        ]
        proc_aft = subprocess.run(cmd_after, capture_output=True, text=True, check=True)
        metrics_after = {}
        for line in proc_aft.stdout.splitlines():
            if line.startswith("EVAL_JSON:"):
                metrics_after = json.loads(line[10:])["songs"][req_file.stem]

        results[slug] = {
            "before": metrics_before,
            "after": metrics_after,
        }

    return results


def print_report(dev_results: dict, holdout_results: dict) -> None:
    """Print markdown formatted comparison table."""
    sep = 75
    print("\n" + "=" * sep)
    print("CHORD BEAT QUANTIZATION OFFLINE EVALUATION REPORT")
    print("=" * sep)

    # 1. Dev set
    print("\n### 1. Dev Songs (out/gt_eval/*.json & diaole_live.json)")
    print(f"{'Song':<20} | {'Before Strict':<15} | {'After Strict':<15} | {'Delta':<10}")
    print("-" * 68)

    dev_bef_matched = sum(r["before"]["beat_strict"]["matched"] for r in dev_results.values())
    dev_bef_eligible = sum(r["before"]["beat_strict"]["eligible"] for r in dev_results.values())
    dev_aft_matched = sum(r["after"]["beat_strict"]["matched"] for r in dev_results.values())
    dev_aft_eligible = sum(r["after"]["beat_strict"]["eligible"] for r in dev_results.values())

    for label, res in dev_results.items():
        b_acc = res["before"]["beat_strict"]["acc"] * 100
        a_acc = res["after"]["beat_strict"]["acc"] * 100
        a_m = res["after"]["beat_strict"]["matched"]
        a_e = res["after"]["beat_strict"]["eligible"]
        diff = a_acc - b_acc
        print(f"{label:<20} | {b_acc:>6.1f}%          | {a_acc:>6.1f}% ({a_m:>3}/{a_e:<3}) | {diff:>+6.1f}%")

    print("-" * 68)
    dev_b_overall = (dev_bef_matched / dev_bef_eligible * 100) if dev_bef_eligible else 0.0
    dev_a_overall = (dev_aft_matched / dev_aft_eligible * 100) if dev_aft_eligible else 0.0
    dev_diff = dev_a_overall - dev_b_overall
    print(f"{'OVERALL DEV':<20} | {dev_b_overall:>6.1f}%          | {dev_a_overall:>6.1f}% ({dev_aft_matched:>3}/{dev_aft_eligible:<3}) | {dev_diff:>+6.1f}%")

    # 2. Holdout set
    print("\n### 2. Held-out Songs (out/holdout_*.json via eval_omr.py)")
    print(f"{'Song':<20} | {'Before Strict':<15} | {'After Strict':<15} | {'Delta':<10}")
    print("-" * 68)

    ho_bef_matched = sum(r["before"]["beat_strict"]["matched"] for r in holdout_results.values())
    ho_bef_eligible = sum(r["before"]["beat_strict"]["eligible"] for r in holdout_results.values())
    ho_aft_matched = sum(r["after"]["beat_strict"]["matched"] for r in holdout_results.values())
    ho_aft_eligible = sum(r["after"]["beat_strict"]["eligible"] for r in holdout_results.values())

    for slug, res in holdout_results.items():
        b_acc = res["before"]["beat_strict"]["acc"] * 100
        a_acc = res["after"]["beat_strict"]["acc"] * 100
        a_m = res["after"]["beat_strict"]["matched"]
        a_e = res["after"]["beat_strict"]["eligible"]
        diff = a_acc - b_acc
        print(f"{slug:<20} | {b_acc:>6.1f}%          | {a_acc:>6.1f}% ({a_m:>3}/{a_e:<3}) | {diff:>+6.1f}%")

    print("-" * 68)
    ho_b_overall = (ho_bef_matched / ho_bef_eligible * 100) if ho_bef_eligible else 0.0
    ho_a_overall = (ho_aft_matched / ho_aft_eligible * 100) if ho_aft_eligible else 0.0
    ho_diff = ho_a_overall - ho_b_overall
    print(f"{'OVERALL HOLDOUT':<20} | {ho_b_overall:>6.1f}%          | {ho_a_overall:>6.1f}% ({ho_aft_matched:>3}/{ho_aft_eligible:<3}) | {ho_diff:>+6.1f}%")
    print("=" * sep + "\n")


def main():
    dev_results = run_dev_eval()
    holdout_results = run_holdout_eval()
    print_report(dev_results, holdout_results)


if __name__ == "__main__":
    main()
