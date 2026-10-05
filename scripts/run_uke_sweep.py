"""Run sweep across all sheet fixtures, keys, and difficulties for ukulele arranger.

Generates:
  out/uke/<slug>_<key>_<level>.json
  out/uke/report.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.arrange.uke_patterns import (
    PATTERN_REGISTRY,
    choose_section_pattern,
    get_time_signature_family,
)
from app.arrange.ukulele import arrange_ukulele
from app.models import ParsedSheet

KEYS = ["C", "G", "Eb", "F#", "Bb"]
LEVELS = ["beginner", "intermediate", "advanced"]
OUT_DIR = "out/uke"


def extract_section_instances(arr: Any) -> list[dict[str, Any]]:
    """Extract contiguous section instances with their assigned base pattern and legend."""
    sec_instances: list[dict[str, Any]] = []
    curr_sec: dict[str, Any] | None = None

    for um, sp in zip(arr.uke_measures, arr.sections):
        if curr_sec is None or sp.section_id != curr_sec["section_id"]:
            if curr_sec is not None:
                sec_instances.append(curr_sec)
            curr_sec = {
                "section_id": sp.section_id,
                "role": sp.role,
                "energy": sp.energy,
                "measures": [sp.measure_index],
                "pattern_id": um.pattern_id,
                "legend": arr.pattern_legend.get(um.pattern_id, um.pattern_id),
            }
        else:
            curr_sec["measures"].append(sp.measure_index)
            # If initial measure of section was special (stop/rest), use subsequent rhythm pattern
            if curr_sec["pattern_id"] in ("uke_stop", "uke_rest", "uke_held") and um.pattern_id not in (
                "uke_stop",
                "uke_rest",
                "uke_held",
            ):
                curr_sec["pattern_id"] = um.pattern_id
                curr_sec["legend"] = arr.pattern_legend.get(um.pattern_id, um.pattern_id)

    if curr_sec is not None:
        sec_instances.append(curr_sec)
    return sec_instances


def format_section_table(title: str, arr: Any) -> list[str]:
    """Format markdown table for a song arrangement's section instances."""
    instances = extract_section_instances(arr)
    lines = [
        f"### {title} - Capo {arr.capo} (Shape Key: {arr.shape_key})",
        "",
        "| Section ID | Role | Energy | Measures | Base Pattern ID | Rhythm Legend |",
        "|---|---|---|---|---|---|",
    ]
    for s in instances:
        m_range = f"m{s['measures'][0]}..m{s['measures'][-1]} ({len(s['measures'])})"
        lines.append(
            f"| `{s['section_id']}` | `{s['role']}` | {s['energy']} | {m_range} | `{s['pattern_id']}` | `{s['legend']}` |"
        )
    return lines


def generate_selector_mapping_table() -> list[str]:
    """Generate the full pattern selector mapping table."""
    lines = [
        "## Pattern Selector Mapping Table",
        "",
        "The arranger selects base accompaniment patterns based on `(time_family, tier, role, energy)`. ",
        "Compound time families (`6/8` and `12/8`, as well as style `'Slow Rock'`) strictly map to compound patterns (`uke_6_8_*` and `uke_12_8_*`).",
        "",
        "| Time Family | Tier | Role | Energy | Pattern ID | Technique | Derived Legend |",
        "|---|---|---|---|---|---|---|",
    ]

    ts_families = ["4/4", "3/4", "2/4", "6/8", "12/8"]
    tiers = ["beginner", "intermediate", "advanced"]
    roles = ["intro", "verse", "prechorus", "chorus", "bridge", "interlude", "outro"]
    energies = [0, 1, 2, 3]

    seen = set()
    for ts in ts_families:
        for tier in tiers:
            for role in roles:
                for energy in energies:
                    pid = choose_section_pattern(ts, tier, role, energy)
                    spec = PATTERN_REGISTRY.get(pid)
                    leg = spec.legend if spec else pid
                    tech = spec.technique if spec else "strum"
                    key = (ts, tier, role, energy, pid)
                    if key not in seen:
                        seen.add(key)
                        lines.append(
                            f"| `{ts}` | `{tier}` | `{role}` | {energy} | `{pid}` | `{tech}` | `{leg}` |"
                        )
    return lines


def run_sweep() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    sheet_paths = sorted(glob.glob("fixtures/sheets/*.json"))
    if not sheet_paths:
        print("No sheet fixtures found in fixtures/sheets/*.json", file=sys.stderr)
        return 1

    qa_checker = None
    try:
        from app.qa.uke_check import validate_uke as qa_checker
    except Exception:
        qa_checker = None

    results: list[dict[str, Any]] = []
    total_runs = 0
    total_exceptions = 0

    print(
        f"Starting ukulele sweep: {len(sheet_paths)} sheets x {len(KEYS)} keys x {len(LEVELS)} levels = {len(sheet_paths) * len(KEYS) * len(LEVELS)} runs."
    )

    for path in sheet_paths:
        slug = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            sheet_data = json.load(f)
        sheet = ParsedSheet.model_validate(sheet_data)

        song_info: dict[str, Any] = {
            "slug": slug,
            "title": sheet.header.title or slug,
            "runs": {},
            "exceptions": 0,
        }

        for k in KEYS:
            for lvl in LEVELS:
                total_runs += 1
                try:
                    arr = arrange_ukulele(sheet, k, lvl)
                    # Dump json
                    out_json_path = os.path.join(OUT_DIR, f"{slug}_{k.replace('#', 's')}_{lvl}.json")
                    with open(out_json_path, "w", encoding="utf-8") as out_fp:
                        out_fp.write(arr.model_dump_json(indent=2))

                    barre_count = sum(1 for s in arr.uke_shapes.values() if s.barre is not None)
                    shapes_used = sorted(list(arr.uke_shapes.keys()))

                    # Group patterns by section
                    sec_patterns: dict[str, str] = {}
                    for um in arr.uke_measures:
                        sec_plan = next((s for s in arr.sections if s.measure_index == um.measure_index), None)
                        sec_id = sec_plan.section_id if sec_plan else "default"
                        if sec_id not in sec_patterns:
                            sec_patterns[sec_id] = um.pattern_id

                    qa_violations = 0
                    if qa_checker is not None:
                        try:
                            v = qa_checker(sheet, arr)
                            qa_violations = len(v) if isinstance(v, list) else 0
                        except Exception:
                            pass

                    song_info["runs"][(k, lvl)] = {
                        "capo": arr.capo,
                        "shape_key": arr.shape_key,
                        "shapes_count": len(shapes_used),
                        "shapes": shapes_used,
                        "barre_count": barre_count,
                        "patterns": sec_patterns,
                        "qa_violations": qa_violations,
                        "success": True,
                    }
                except Exception as e:
                    total_exceptions += 1
                    song_info["exceptions"] += 1
                    traceback.print_exc()
                    song_info["runs"][(k, lvl)] = {
                        "success": False,
                        "error": str(e),
                    }

        results.append(song_info)

    # Generate Markdown report
    report_lines: list[str] = [
        "# Ukulele Arranger Sweep Report",
        "",
        f"- **Total Sheets**: {len(sheet_paths)}",
        f"- **Total Arrangements Generated**: {total_runs}",
        f"- **Total Exceptions**: {total_exceptions}",
        f"- **QA Checker Available**: {'Yes' if qa_checker is not None else 'No (lazy fallback)'}",
        "",
        "## Summary by Song (Fixtures Sweep)",
        "",
        "| Song Slug | Exceptions | Capo/Shape Key (C / Eb / F# beg) | Distinct Shapes (C / Eb beg) | Barres (C / Eb beg) | Patterns (Verse / Chorus) |",
        "|---|---|---|---|---|---|",
    ]

    for song in results:
        slug = song["slug"]
        exc = song["exceptions"]

        r_c_beg = song["runs"].get(("C", "beginner"), {})
        r_eb_beg = song["runs"].get(("Eb", "beginner"), {})
        r_fs_beg = song["runs"].get(("F#", "beginner"), {})
        r_c_mid = song["runs"].get(("C", "intermediate"), {})

        capo_str = (
            f"C: {r_c_beg.get('capo', 0)} ({r_c_beg.get('shape_key', 'C')}) / "
            f"Eb: {r_eb_beg.get('capo', 0)} ({r_eb_beg.get('shape_key', 'Eb')}) / "
            f"F#: {r_fs_beg.get('capo', 0)} ({r_fs_beg.get('shape_key', 'F#')})"
        )

        shapes_str = f"C: {r_c_beg.get('shapes_count', 0)} / Eb: {r_eb_beg.get('shapes_count', 0)}"
        barre_str = f"C: {r_c_beg.get('barre_count', 0)} / Eb: {r_eb_beg.get('barre_count', 0)}"

        # Patterns
        pats = r_c_mid.get("patterns", {})
        pat_str = ", ".join(f"{sec}:{pid.replace('uke_4_4_', '')}" for sec, pid in list(pats.items())[:3])

        report_lines.append(
            f"| `{slug}` | {exc} | {capo_str} | {shapes_str} | {barre_str} | {pat_str} |"
        )

    # Add Pattern Selector Table
    report_lines.append("")
    report_lines.extend(generate_selector_mapping_table())

    # Fresh Parses Case Studies (gt_eval)
    report_lines.extend([
        "",
        "## Real Song Form & Section Breakdown (Fresh Parses from `out/gt_eval`)",
        "",
        "Arranged using `plan_sections(sheet, use_llm=False)` from `app.arrange.sections`.",
        "",
    ])

    diaole_fp = "out/gt_eval/diaole.json"
    tinghai_fp = "out/gt_eval/tinghai.json"

    print("\n" + "=" * 60)
    print("SECTION / PATTERN BREAKDOWN (Fresh GT Eval Parses)")
    print("=" * 60)

    if os.path.exists(diaole_fp):
        with open(diaole_fp, encoding="utf-8") as f:
            diaole_sheet = ParsedSheet.model_validate(json.load(f))

        # 掉了 Key F Intermediate
        arr_diaole_mid = arrange_ukulele(diaole_sheet, "F", "intermediate")
        report_lines.extend(format_section_table("掉了 (Key F, Intermediate)", arr_diaole_mid))
        report_lines.append("")

        print("\n--- 掉了 (Key F, Intermediate) ---")
        for s in extract_section_instances(arr_diaole_mid):
            m_range = f"m{s['measures'][0]}..m{s['measures'][-1]} ({len(s['measures'])})"
            print(
                f"{s['section_id']:12s} {s['role']:10s} energy={s['energy']} {m_range:16s} {s['pattern_id']:20s} {s['legend']}"
            )

        # 掉了 Key F Advanced
        arr_diaole_adv = arrange_ukulele(diaole_sheet, "F", "advanced")
        report_lines.extend(format_section_table("掉了 (Key F, Advanced)", arr_diaole_adv))
        report_lines.append("")

        print("\n--- 掉了 (Key F, Advanced) ---")
        for s in extract_section_instances(arr_diaole_adv):
            m_range = f"m{s['measures'][0]}..m{s['measures'][-1]} ({len(s['measures'])})"
            print(
                f"{s['section_id']:12s} {s['role']:10s} energy={s['energy']} {m_range:16s} {s['pattern_id']:20s} {s['legend']}"
            )

    if os.path.exists(tinghai_fp):
        with open(tinghai_fp, encoding="utf-8") as f:
            tinghai_sheet = ParsedSheet.model_validate(json.load(f))

        # 听海 Key Eb Advanced
        arr_tinghai_adv = arrange_ukulele(tinghai_sheet, "Eb", "advanced")
        report_lines.extend(format_section_table("听海 (Key Eb, Advanced)", arr_tinghai_adv))
        report_lines.append("")

        print("\n--- 听海 (Key Eb, Advanced) ---")
        for s in extract_section_instances(arr_tinghai_adv):
            m_range = f"m{s['measures'][0]}..m{s['measures'][-1]} ({len(s['measures'])})"
            print(
                f"{s['section_id']:12s} {s['role']:10s} energy={s['energy']} {m_range:16s} {s['pattern_id']:20s} {s['legend']}"
            )

    report_path = os.path.join(OUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(report_lines) + "\n")

    print(f"\nSweep complete! Report written to {report_path}")
    return 0 if total_exceptions == 0 else 1


if __name__ == "__main__":
    sys.exit(run_sweep())
