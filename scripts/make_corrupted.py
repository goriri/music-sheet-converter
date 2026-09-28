"""Generate corrupted OMR fixtures with realistic OCR errors for verification testing.

Usage:
    python scripts/make_corrupted.py --seed 1 --out fixtures/corrupted
    python scripts/make_corrupted.py --seed 2 --out fixtures/corrupted_seed2
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, ".")

FIXTURES_DIR = Path("fixtures")
SAMPLE_PATH = FIXTURES_DIR / "omr_sample.json"


def make_corrupted_fixtures(seed: int = 1, out_dir: str | Path = "fixtures/corrupted") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        clean_data = json.load(f)

    rng = random.Random(seed)
    manifest: dict[str, list[dict[str, Any]]] = {}

    all_measures = [m for s in clean_data["systems"] for m in s["measures"]]

    # ---------------------------------------------------------------------------
    # 1. Corrupted Accidentals (dropped flat/sharp, slash bass accidentals, etc.)
    # ---------------------------------------------------------------------------
    acc_candidates: list[tuple[int, int, str, str, str]] = []
    for m in all_measures:
        for c_idx, c in enumerate(m["chords"]):
            raw = c["raw"]
            if "/" in raw and ("b" in raw.split("/")[1] or "#" in raw.split("/")[1]):
                root, bass = raw.split("/", 1)
                corrupted = f"{root}/{bass.replace('b', '').replace('#', '')}"
                acc_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"Dropped accidental in slash bass {raw} -> {corrupted}")
                )
            elif raw in ("6b", "7b", "1#", "6m7-5"):
                corrupted = raw.replace("b", "").replace("#", "").replace("-5", "")
                acc_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"Dropped accidental {raw} -> {corrupted}")
                )
            elif raw in ("6", "7"):
                corrupted = f"{raw}b"
                acc_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"Added flat {raw} -> {corrupted}")
                )
            elif raw == "5/7":
                acc_candidates.append(
                    (m["index"], c_idx, "5/7b", raw, f"Added flat in slash bass {raw} -> 5/7b")
                )

    k_acc = min(8, len(acc_candidates))
    chosen_acc = rng.sample(acc_candidates, k_acc)
    chosen_acc.sort(key=lambda x: (x[0], x[1]))

    acc_data = copy.deepcopy(clean_data)
    acc_meas_lookup = {m["index"]: m for s in acc_data["systems"] for m in s["measures"]}
    manifest["corrupted_accidentals.json"] = []
    for m_idx, c_idx, bad_val, orig_val, desc in chosen_acc:
        m = acc_meas_lookup[m_idx]
        assert m["chords"][c_idx]["raw"] == orig_val
        m["chords"][c_idx]["raw"] = bad_val
        manifest["corrupted_accidentals.json"].append({
            "measure_index": m_idx,
            "chord_index": c_idx,
            "field": "chord.raw",
            "error_type": "dropped_accidental",
            "corrupted_value": bad_val,
            "expected_value": orig_val,
            "description": desc,
        })

    with open(out_path / "corrupted_accidentals.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # 2. Corrupted Qualities (m/M7 swaps, sus variations, lost (2))
    # ---------------------------------------------------------------------------
    qual_candidates: list[tuple[int, int, str, str, str]] = []
    for m in all_measures:
        for c_idx, c in enumerate(m["chords"]):
            raw = c["raw"]
            if "m7" in raw and not raw.endswith("-5"):
                corrupted = raw.replace("m7", "7")
                qual_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"Lost minor 'm': {raw} -> {corrupted}")
                )
            elif "M7" in raw:
                corrupted = raw.replace("M7", "7")
                qual_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"M7 misread as dominant 7: {raw} -> {corrupted}")
                )
            elif "57sus" in raw:
                corrupted = raw.replace("57sus", "5sus")
                qual_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"57sus misread as 5sus: {raw} -> {corrupted}")
                )
            elif "5sus" in raw:
                corrupted = raw.replace("5sus", "57sus")
                qual_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"5sus misread as 57sus: {raw} -> {corrupted}")
                )
            elif "m" in raw and "/" not in raw and not raw.startswith("5m"):
                corrupted = raw.replace("m", "")
                qual_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"Lost minor 'm': {raw} -> {corrupted}")
                )

    add9_candidates: list[tuple[int, int, str, str, str]] = []
    for m in all_measures:
        for c_idx, c in enumerate(m["chords"]):
            raw = c["raw"]
            if "(2)" in raw:
                corrupted = raw.replace("(2)", "")
                add9_candidates.append(
                    (m["index"], c_idx, corrupted, raw, f"Lost (2) extension: {raw} -> {corrupted}")
                )

    k_qual = min(6, len(qual_candidates))
    k_add9 = min(5, len(add9_candidates))
    chosen_qual = rng.sample(qual_candidates, k_qual)
    chosen_add9 = rng.sample(add9_candidates, k_add9)

    qual_data = copy.deepcopy(clean_data)
    qual_meas_lookup = {m["index"]: m for s in qual_data["systems"] for m in s["measures"]}
    manifest["corrupted_qualities.json"] = []

    combined_qual = chosen_qual + chosen_add9
    combined_qual.sort(key=lambda x: (x[0], x[1]))
    for m_idx, c_idx, bad_val, orig_val, desc in combined_qual:
        m = qual_meas_lookup[m_idx]
        assert m["chords"][c_idx]["raw"] == orig_val
        m["chords"][c_idx]["raw"] = bad_val
        manifest["corrupted_qualities.json"].append({
            "measure_index": m_idx,
            "chord_index": c_idx,
            "field": "chord.raw",
            "error_type": "quality_confusion" if "(2)" not in orig_val else "lost_add9",
            "corrupted_value": bad_val,
            "expected_value": orig_val,
            "description": desc,
        })

    with open(out_path / "corrupted_qualities.json", "w", encoding="utf-8") as f:
        json.dump(qual_data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # 3. Corrupted Structural (wrong beat, deleted chord, deleted key change)
    # ---------------------------------------------------------------------------
    struct_data = copy.deepcopy(clean_data)
    struct_meas_lookup = {m["index"]: m for s in struct_data["systems"] for m in s["measures"]}
    manifest["corrupted_structural.json"] = []

    two_chord_meas_indices = [
        m["index"] for m in all_measures
        if len(m["chords"]) >= 2 and m["chords"][1]["beat"] > 1.0
    ]
    rng.shuffle(two_chord_meas_indices)
    beat_meas_indices = two_chord_meas_indices[:4]
    delete_meas_indices = two_chord_meas_indices[4:8]

    for m_idx in beat_meas_indices:
        m = struct_meas_lookup[m_idx]
        orig_beat = m["chords"][1]["beat"]
        bad_beat = 1.0
        m["chords"][1]["beat"] = bad_beat
        manifest["corrupted_structural.json"].append({
            "measure_index": m_idx,
            "chord_index": 1,
            "field": "chord.beat",
            "error_type": "wrong_chord_beat",
            "corrupted_value": bad_beat,
            "expected_value": orig_beat,
            "description": f"Second chord starting beat corrupted to {bad_beat} instead of {orig_beat}",
        })

    for m_idx in delete_meas_indices:
        m = struct_meas_lookup[m_idx]
        del_c = m["chords"][1]["raw"]
        del m["chords"][1]
        manifest["corrupted_structural.json"].append({
            "measure_index": m_idx,
            "chord_index": 1,
            "field": "measure.chords",
            "error_type": "chord_deleted",
            "corrupted_value": None,
            "expected_value": del_c,
            "description": f"Second chord '{del_c}' deleted in measure {m_idx}",
        })

    orig_kc = struct_data.get("key_changes", [])
    if orig_kc:
        struct_data["key_changes"] = []
        manifest["corrupted_structural.json"].append({
            "measure_index": orig_kc[0].get("at_measure", 0),
            "chord_index": None,
            "field": "key_changes",
            "error_type": "key_change_removed",
            "corrupted_value": [],
            "expected_value": orig_kc,
            "description": "Modulation key change marking removed from sheet",
        })

    with open(out_path / "corrupted_structural.json", "w", encoding="utf-8") as f:
        json.dump(struct_data, f, indent=2, ensure_ascii=False)

    with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total_injected = sum(len(errs) for errs in manifest.values())
    print(f"Seed {seed}: Generated {len(manifest)} corrupted files with {total_injected} total injected errors in {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate randomized corrupted OMR fixtures")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--out", type=str, default="fixtures/corrupted", help="Output directory")
    args = parser.parse_args()
    make_corrupted_fixtures(seed=args.seed, out_dir=args.out)
