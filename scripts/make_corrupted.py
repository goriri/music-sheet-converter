"""Generate corrupted OMR fixtures with realistic OCR errors for verification testing.

Creates:
- fixtures/corrupted/corrupted_accidentals.json: dropped/moved accidentals ('5m/7', '6', '7', etc.)
- fixtures/corrupted/corrupted_qualities.json: m/M7 confusions and lost (2) ('27', '47', '1', '5sus')
- fixtures/corrupted/corrupted_structural.json: beat corruptions, key change removed, chord deleted, garbage
- fixtures/corrupted/manifest.json: Ground-truth list of all injected errors.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, ".")
from app.models import ParsedSheet

FIXTURES_DIR = Path("fixtures")
SAMPLE_PATH = FIXTURES_DIR / "omr_sample.json"
CORRUPTED_DIR = FIXTURES_DIR / "corrupted"


def make_corrupted_fixtures() -> None:
    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)

    with open(SAMPLE_PATH, encoding="utf-8") as f:
        clean_data = json.load(f)

    manifest: dict[str, list[dict[str, Any]]] = {}

    # ---------------------------------------------------------------------------
    # 1. Corrupted Accidentals (dropped flat/sharp, slash bass accidentals)
    # ---------------------------------------------------------------------------
    acc_data = json.loads(json.dumps(clean_data))
    acc_errors = [
        # (measure_idx, chord_idx, corrupted_val, orig_val, desc)
        (10, 0, "5m/7", "5m/7b", "Dropped flat in slash bass 5m/7b -> 5m/7"),
        (11, 1, "67/1", "67/1#", "Dropped sharp in slash bass 67/1# -> 67/1"),
        (14, 0, "6", "6b", "Dropped flat in borrowed bVI degree 6b -> 6"),
        (18, 0, "5m/7", "5m/7b", "Dropped flat in slash bass 5m/7b -> 5m/7"),
        (22, 0, "6", "6b", "Dropped flat in borrowed bVI degree 6b -> 6"),
        (42, 0, "7", "7b", "Dropped flat in borrowed bVII degree 7b -> 7"),
        (45, 0, "17/7", "17/7b", "Dropped flat in slash bass 17/7b -> 17/7"),
        (46, 0, "7", "7b", "Dropped flat in borrowed bVII degree 7b -> 7"),
    ]

    manifest["corrupted_accidentals.json"] = []
    # Find measure by global index
    meas_lookup = {
        m["index"]: m for s in acc_data["systems"] for m in s["measures"]
    }
    for m_idx, c_idx, bad_val, orig_val, desc in acc_errors:
        m = meas_lookup[m_idx]
        actual_orig = m["chords"][c_idx]["raw"]
        assert actual_orig == orig_val, f"Mismatch at m{m_idx} c{c_idx}: expected {orig_val}, got {actual_orig}"
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

    with open(CORRUPTED_DIR / "corrupted_accidentals.json", "w", encoding="utf-8") as f:
        json.dump(acc_data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # 2. Corrupted Qualities (m/M7 swaps, lost '(2)', sus variations)
    # ---------------------------------------------------------------------------
    qual_data = json.loads(json.dumps(clean_data))
    qual_errors = [
        (0, 0, "1", "1(2)", "Lost (2) extension: 1(2) -> 1"),
        (2, 0, "27", "2m7", "Lost minor 'm': 2m7 -> 27"),
        (4, 0, "1", "1(2)", "Lost (2) extension: 1(2) -> 1"),
        (12, 0, "27", "2m7", "Lost minor 'm': 2m7 -> 27"),
        (20, 0, "27", "2m7", "Lost minor 'm': 2m7 -> 27"),
        (23, 0, "5sus", "57sus", "57sus misread as 5sus"),
        (45, 1, "47", "4M7", "M7 misread as dominant 7: 4M7 -> 47"),
        (62, 0, "47", "4M7", "M7 misread as dominant 7: 4M7 -> 47"),
        (64, 0, "47", "4M7", "M7 misread as dominant 7: 4M7 -> 47"),
    ]

    manifest["corrupted_qualities.json"] = []
    qual_lookup = {
        m["index"]: m for s in qual_data["systems"] for m in s["measures"]
    }
    for m_idx, c_idx, bad_val, orig_val, desc in qual_errors:
        m = qual_lookup[m_idx]
        actual_orig = m["chords"][c_idx]["raw"]
        assert actual_orig == orig_val, f"Mismatch at m{m_idx} c{c_idx}: expected {orig_val}, got {actual_orig}"
        m["chords"][c_idx]["raw"] = bad_val
        manifest["corrupted_qualities.json"].append({
            "measure_index": m_idx,
            "chord_index": c_idx,
            "field": "chord.raw",
            "error_type": "quality_confusion",
            "corrupted_value": bad_val,
            "expected_value": orig_val,
            "description": desc,
        })

    with open(CORRUPTED_DIR / "corrupted_qualities.json", "w", encoding="utf-8") as f:
        json.dump(qual_data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # 3. Corrupted Structural (beat corruptions, key change removed, chord deleted, garbage)
    # ---------------------------------------------------------------------------
    struct_data = json.loads(json.dumps(clean_data))
    manifest["corrupted_structural.json"] = []
    struct_lookup = {
        m["index"]: m for s in struct_data["systems"] for m in s["measures"]
    }

    # Error 3a: 2nd chord wrong beat in m11 (beat 1.0 instead of 3.0)
    m11 = struct_lookup[11]
    m11["chords"][1]["beat"] = 1.0
    manifest["corrupted_structural.json"].append({
        "measure_index": 11,
        "chord_index": 1,
        "field": "chord.beat",
        "error_type": "wrong_chord_beat",
        "corrupted_value": 1.0,
        "expected_value": 3.0,
        "description": "Second chord starting beat corrupted to 1.0 instead of 3.0",
    })

    # Error 3b: Melody beat sum mismatch in m23 ('2 - 0 0' truncated to '2')
    m23 = struct_lookup[23]
    orig_melody = m23["melody"]
    m23["melody"] = "2"
    manifest["corrupted_structural.json"].append({
        "measure_index": 23,
        "chord_index": None,
        "field": "measure.melody",
        "error_type": "melody_beat_sum_mismatch",
        "corrupted_value": "2",
        "expected_value": orig_melody,
        "description": "Melody beats truncated to 1.0 beat in 4-beat measure",
    })

    # Error 3c: Key change removed
    orig_kc = struct_data.get("key_changes", [])
    struct_data["key_changes"] = []
    manifest["corrupted_structural.json"].append({
        "measure_index": 50,
        "chord_index": None,
        "field": "key_changes",
        "error_type": "key_change_removed",
        "corrupted_value": [],
        "expected_value": orig_kc,
        "description": "Modulation key change marking removed from measure 50",
    })

    # Error 3d: Deleted second chord in m31
    m31 = struct_lookup[31]
    orig_m31_c = list(m31["chords"])
    del m31["chords"][1]  # delete '5' chord
    manifest["corrupted_structural.json"].append({
        "measure_index": 31,
        "chord_index": 1,
        "field": "measure.chords",
        "error_type": "chord_deleted",
        "corrupted_value": None,
        "expected_value": "5",
        "description": "Second chord '5' deleted in measure 31",
    })

    # Error 3e: Unparseable chord OCR garbage in m3
    m3 = struct_lookup[3]
    m3["chords"].append({
        "raw": "XYZ",
        "beat": 1.0,
        "bbox": [0.72, 0.123, 0.78, 0.146],
        "confidence": 1.0,
        "alternatives": [],
    })
    manifest["corrupted_structural.json"].append({
        "measure_index": 3,
        "chord_index": 0,
        "field": "chord.raw",
        "error_type": "invalid_grammar",
        "corrupted_value": "XYZ",
        "expected_value": None,
        "description": "Inserted unparseable OCR garbage 'XYZ'",
    })

    # Error 3f: Out of bounds beat in m39
    m39 = struct_lookup[39]
    m39["chords"][1]["beat"] = 5.0
    manifest["corrupted_structural.json"].append({
        "measure_index": 39,
        "chord_index": 1,
        "field": "chord.beat",
        "error_type": "wrong_chord_beat",
        "corrupted_value": 5.0,
        "expected_value": 3.0,
        "description": "Chord beat out of bounds (beat 5.0 in 4/4 bar)",
    })

    with open(CORRUPTED_DIR / "corrupted_structural.json", "w", encoding="utf-8") as f:
        json.dump(struct_data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # 4. Write manifest
    # ---------------------------------------------------------------------------
    with open(CORRUPTED_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total_injected = sum(len(errs) for errs in manifest.values())
    print(f"Generated {len(manifest)} corrupted files with {total_injected} total injected errors.")


if __name__ == "__main__":
    make_corrupted_fixtures()
