#!/usr/bin/env python3
"""Compare geometry chord-box->measure assignments against groundtruth answer keys.

Checks tinghai, xiaobaichuan, diandao, and liusha.
Reports mismatches by (slug, page, row, measure_counts_geom, measure_counts_gt).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.omr.layout import analyze_page


def check_slug(slug: str) -> tuple[int, int]:
    gt_file = REPO_ROOT / "fixtures" / "groundtruth" / f"{slug}.json"
    if not gt_file.exists():
        print(f"[{slug}] Missing groundtruth file: {gt_file}")
        return 0, 0

    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    num_pages = gt.get("pages", 1)
    total_rows = 0
    matching_rows = 0

    for p_idx in range(num_pages):
        page_num = p_idx + 1
        img_file = REPO_ROOT / "fixtures" / "external" / slug / f"page{page_num}.jpg"
        if not img_file.exists():
            continue

        geom = analyze_page(img_file.read_bytes(), page=p_idx)
        gt_rows = [r for r in gt.get("rows", []) if r.get("page") == page_num]

        for s_idx, sys_obj in enumerate(geom.systems):
            if s_idx >= len(gt_rows):
                break
            total_rows += 1
            gt_row = gt_rows[s_idx]
            gt_meas = gt_row.get("measures", [])

            geom_counts = [0] * len(sys_obj.measures)
            for cb in sys_obj.chord_boxes:
                if 0 <= cb.measure_index_in_system < len(geom_counts):
                    geom_counts[cb.measure_index_in_system] += 1
                else:
                    print(f"[{slug} p{page_num} r{s_idx+1}] Chord box measure index out of range: {cb.measure_index_in_system}")

            gt_counts = [len(m.get("chords", [])) for m in gt_meas]
            if geom_counts == gt_counts:
                matching_rows += 1
            else:
                print(
                    f"[{slug} p{page_num} r{s_idx+1}] MISMATCH: geom={geom_counts} vs gt={gt_counts}"
                )

    return matching_rows, total_rows


def main() -> int:
    slugs = ["tinghai", "xiaobaichuan", "diandao", "liusha"]
    total_match = 0
    total_all = 0

    print("Checking chord box -> measure assignment against groundtruth...")
    for slug in slugs:
        matched, count = check_slug(slug)
        total_match += matched
        total_all += count
        status = "OK" if matched == count else "DIFF"
        print(f"  {slug:15s}: {matched}/{count} rows match [{status}]")

    print(f"Total: {total_match}/{total_all} rows matching ({total_match / max(1, total_all) * 100:.1f}%)")
    return 0 if total_match == total_all else 1


if __name__ == "__main__":
    sys.exit(main())
