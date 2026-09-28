"""Runner script for arrangement quality assurance and validation.

Runs arrange(sheet, key, level) for keys F, G, Bb, F# and all levels on fixtures/omr_sample.json,
prints before-repair violation counts by rule and measure, and verifies 0 violations after repair.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.arrange.piano import arrange
from app.models import ParsedSheet
from app.qa.arrange_check import check_and_repair, validate_measure
from app.theory.keys import key_name_to_pc


def run_checks(omr_path: Path) -> int:
    """Run validation across keys and levels, printing before/after summaries."""
    with open(omr_path, encoding="utf-8") as f:
        sheet = ParsedSheet.model_validate(json.load(f))

    keys = ["F", "G", "Bb", "F#"]
    levels = ["beginner", "intermediate", "advanced"]
    measures_by_idx = {m.index: m for m in sheet.measures()}

    key_changes_by_m: dict[int, list[int]] = {}
    for kc in sheet.key_changes:
        key_changes_by_m.setdefault(kc.at_measure, []).append(kc.semitones)

    grand_total_before = 0
    all_rule_counts: dict[str, int] = {}
    measure_violation_counts: dict[int, int] = {}
    failures = 0

    print("=" * 80)
    print("RUNNING ARRANGEMENT VALIDATION ON fixtures/omr_sample.json")
    print("=" * 80)

    for key in keys:
        start_pc = key_name_to_pc(key)
        for level in levels:
            print(f"\n--- Checking Key: {key:2} | Level: {level:12} ---")
            raw_arr = arrange(sheet, key, level)

            # Compute expected tonics
            curr_pc = start_pc
            expected_tonics: list[int] = []
            for m_arr in raw_arr.measures:
                if m_arr.measure_index in key_changes_by_m:
                    for shift in key_changes_by_m[m_arr.measure_index]:
                        curr_pc = (curr_pc + shift) % 12
                expected_tonics.append(curr_pc)

            # Collect before-repair violations
            before_violations = []
            for i, m_arr in enumerate(raw_arr.measures):
                m_spec = measures_by_idx.get(m_arr.measure_index)
                nxt = (
                    raw_arr.measures[i + 1].chords[0]
                    if i + 1 < len(raw_arr.measures) and raw_arr.measures[i + 1].chords
                    else None
                )
                v_list = validate_measure(m_arr, m_spec, expected_tonics[i], raw_arr.difficulty, nxt)
                for v in v_list:
                    before_violations.append((m_arr.measure_index, v))
                    all_rule_counts[v.code] = all_rule_counts.get(v.code, 0) + 1
                    measure_violation_counts[m_arr.measure_index] = (
                        measure_violation_counts.get(m_arr.measure_index, 0) + 1
                    )

            count_before = len(before_violations)
            grand_total_before += count_before
            print(f"  Before repair violations: {count_before}")
            if count_before > 0:
                summary_by_rule: dict[str, int] = {}
                for _, v in before_violations:
                    summary_by_rule[v.code] = summary_by_rule.get(v.code, 0) + 1
                print(f"  Violations by rule: {summary_by_rule}")

            # Run check_and_repair
            repaired_arr = check_and_repair(sheet, raw_arr)

            # Validate after repair
            after_violations = []
            for i, m_arr in enumerate(repaired_arr.measures):
                m_spec = measures_by_idx.get(m_arr.measure_index)
                nxt = (
                    repaired_arr.measures[i + 1].chords[0]
                    if i + 1 < len(repaired_arr.measures) and repaired_arr.measures[i + 1].chords
                    else None
                )
                v_list = validate_measure(m_arr, m_spec, expected_tonics[i], repaired_arr.difficulty, nxt)
                for v in v_list:
                    after_violations.append((m_arr.measure_index, v))

            count_after = len(after_violations)
            print(f"  After repair violations:  {count_after} (Auto-fixed issues: {len(repaired_arr.issues)})")
            if count_after != 0:
                print(f"  FAILED: Still {count_after} violations after repair!")
                failures += 1

    print("\n" + "=" * 80)
    print("BEFORE-REPAIR VIOLATION SUMMARY (ALL KEYS & LEVELS)")
    print("=" * 80)
    print(f"Total violations detected before repair: {grand_total_before}")
    print("By rule:")
    for rule, count in sorted(all_rule_counts.items(), key=lambda x: -x[1]):
        print(f"  - {rule:25}: {count}")
    print("\nBy measure (measures with violations):")
    for m_idx in sorted(measure_violation_counts.keys()):
        print(f"  - Measure {m_idx + 1:2}: {measure_violation_counts[m_idx]} violations")
    print(f"\nFinal post-repair status: {'PASS (0 violations)' if failures == 0 else f'FAIL ({failures} errors)'}")
    print("=" * 80)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    omr_file = Path("fixtures/omr_sample.json")
    if not omr_file.exists():
        print(f"Error: {omr_file} not found")
        sys.exit(1)
    sys.exit(run_checks(omr_file))
