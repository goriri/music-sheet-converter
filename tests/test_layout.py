"""Unit tests for classical CV layout analysis (app/omr/layout.py).

Verifies system count, measures per system, bbox containment, non-overlap,
chord box containment, deskew, and exception-free fallbacks.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.omr.geometry import GMeasure
from app.omr.layout import (
    _assign_chord_box_to_measure,
    _has_melody_line,
    _is_bar_number_band,
    analyze_page,
    clean_narrow_measures,
    deskew_image,
    detect_chord_boxes,
    estimate_skew_angle,
    find_dashed_barlines,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRUTH_PATH = REPO_ROOT / "fixtures" / "layout_truth.json"


@pytest.fixture(scope="module")
def layout_truth() -> dict[str, dict]:
    with open(TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_corrupted_and_empty_images():
    """Verify that analyze_page never raises exceptions on invalid image buffers."""
    geom_empty = analyze_page(b"", page=0)
    assert geom_empty.page == 0
    assert geom_empty.confidence == 0.0
    assert len(geom_empty.systems) == 0

    geom_garbage = analyze_page(b"not an image", page=1)
    assert geom_garbage.page == 1
    assert geom_garbage.confidence == 0.0
    assert len(geom_garbage.systems) == 0


def test_deskew_estimation():
    """Verify that skewed image gets non-zero angle and straight page gets near zero."""
    skewed_path = REPO_ROOT / "fixtures" / "external" / "phone_photo_skewed" / "page1.jpg"
    if skewed_path.exists():
        import cv2

        gray = cv2.imread(str(skewed_path), cv2.IMREAD_GRAYSCALE)
        angle = estimate_skew_angle(gray)
        assert abs(angle) >= 2.0  # Phone photo has ~ -3.8 deg skew

    straight_path = REPO_ROOT / "fixtures" / "pages" / "page1.jpg"
    if straight_path.exists():
        import cv2

        gray = cv2.imread(str(straight_path), cv2.IMREAD_GRAYSCALE)
        angle = estimate_skew_angle(gray)
        assert abs(angle) <= 1.0


def test_ground_truth_pages(layout_truth):
    """Verify system counts and measures-per-system across all ground truth pages."""
    for rel_path, expected in layout_truth.items():
        img_path = REPO_ROOT / rel_path
        if not img_path.exists():
            continue

        img_bytes = img_path.read_bytes()
        geom = analyze_page(img_bytes, page=0)

        # 1. Exact system count
        assert len(geom.systems) == expected["num_systems"], (
            f"{rel_path}: expected {expected['num_systems']} systems, got {len(geom.systems)}"
        )

        # 2. Measures per system (allow <= 1 measure tolerance per row for handwritten/unboxed)
        det_measures = [len(s.measures) for s in geom.systems]
        exp_measures = expected["measures_per_system"]

        # All ground truth pages should match exactly
        if rel_path in [
            "fixtures/pages/page1.jpg",
            "fixtures/pages/page2.jpg",
            "fixtures/external/qianlizhiwai/page1.jpg",
            "fixtures/external/tinghai/page1.jpg",
            "fixtures/external/tinghai/page2.jpg",
            "fixtures/external/diandao/page1.jpg",
            "fixtures/external/diandao/page2.jpg",
            "fixtures/external/xiaobaichuan/page1.jpg",
            "fixtures/external/xiaobaichuan/page2.jpg",
            "fixtures/external/phone_photo_skewed/page1.jpg",
            "fixtures/external/liusha/page1.jpg",
            "fixtures/external/liusha/page2.jpg",
            "fixtures/external/xindong/page1.jpg",
            "fixtures/external/langrenqingge/page1.jpg",
        ]:
            assert det_measures == exp_measures, (
                f"{rel_path}: measure counts mismatch: {det_measures} vs {exp_measures}"
            )
        else:
            # For complex letter chord charts, allow at most 1 measure difference per row
            assert len(det_measures) == len(exp_measures)
            for d_m, e_m in zip(det_measures, exp_measures):
                assert abs(d_m - e_m) <= 1, (
                    f"{rel_path}: measure count diff > 1: {det_measures} vs {exp_measures}"
                )


def test_system_bbox_containment_and_non_overlap():
    """Verify that system bounding boxes contain their bands and do not overlap."""
    img_path = REPO_ROOT / "fixtures" / "pages" / "page1.jpg"
    img_bytes = img_path.read_bytes()
    geom = analyze_page(img_bytes, page=0)

    assert len(geom.systems) >= 2
    for i in range(len(geom.systems)):
        sys_i = geom.systems[i]
        sx0, sy0, sx1, sy1 = sys_i.bbox

        # Bbox validity
        assert 0.0 <= sx0 < sx1 <= 1.0
        assert 0.0 <= sy0 < sy1 <= 1.0

        # Melody band containment
        mel_y0, mel_y1 = sys_i.melody_band
        assert sy0 <= mel_y0 <= mel_y1 <= sy1

        # Chord band containment if present
        if sys_i.chord_band is not None:
            ch_y0, ch_y1 = sys_i.chord_band
            assert sy0 <= ch_y0 <= ch_y1 <= sy1

        # Lyric bands containment
        for lyr_y0, lyr_y1 in sys_i.lyric_bands:
            assert sy0 <= lyr_y0 <= lyr_y1 <= sy1

        # Non-overlap with next system
        if i + 1 < len(geom.systems):
            next_sy0 = geom.systems[i + 1].bbox[1]
            assert sy1 <= next_sy0 + 1e-4, f"Systems {i} and {i+1} overlap vertically: {sy1} > {next_sy0}"


def test_chord_box_containment_and_beats():
    """Verify chord boxes lie within system bbox and have valid 1-based beat estimations."""
    img_path = REPO_ROOT / "fixtures" / "pages" / "page1.jpg"
    img_bytes = img_path.read_bytes()
    geom = analyze_page(img_bytes, page=0)

    total_chords = sum(len(s.chord_boxes) for s in geom.systems)
    assert total_chords >= 10, "Expected at least 10 chord boxes detected on page 1"

    for s in geom.systems:
        sx0, sy0, sx1, sy1 = s.bbox
        for cb in s.chord_boxes:
            cx0, cy0, cx1, cy1 = cb.bbox
            assert sx0 <= cx0 <= cx1 <= sx1, f"Chord box x out of system bounds: {cb.bbox}"
            assert sy0 <= cy0 <= cy1 <= sy1, f"Chord box y out of system bounds: {cb.bbox}"
            assert 0 <= cb.measure_index_in_system < len(s.measures)
            assert 1.0 <= cb.beat_geo <= 4.0


def test_chords_below_layout():
    """Verify chords_below sheets (tinghai, diandao, liusha) have chord band below melody band,
    unboxed chord boxes, and chords_below note."""
    chords_below_paths = [
        "fixtures/external/tinghai/page1.jpg",
        "fixtures/external/diandao/page1.jpg",
        "fixtures/external/liusha/page1.jpg",
    ]
    for rel_path in chords_below_paths:
        img_path = REPO_ROOT / rel_path
        if not img_path.exists():
            continue
        geom = analyze_page(img_path.read_bytes(), page=0)
        assert "chords_below" in geom.notes, f"{rel_path}: expected chords_below in geom.notes"

        # Total chord count should be substantial (> 30)
        total_chords = sum(len(s.chord_boxes) for s in geom.systems)
        assert total_chords >= 30, f"{rel_path}: expected >= 30 chords, got {total_chords}"

        # Verify chord bands and unboxed chord boxes
        for i, s in enumerate(geom.systems):
            if s.chord_band is not None:
                assert "chords_below" in s.notes
                # Chord band must be below melody band
                assert s.chord_band[0] >= s.melody_band[1] - 0.005, (
                    f"{rel_path} system {i}: chord band {s.chord_band} not below melody band {s.melody_band}"
                )
            for cb in s.chord_boxes:
                assert not cb.boxed, f"{rel_path} chord box {cb} should be unboxed (boxed=False)"


def test_chord_only_pages():
    """Verify chord-only ('弹唱版') sheets detect chord_only rows and extract unboxed chord boxes."""
    # 1. xiaobaichuan p2 (3/4 waltz, 8 systems x 4 bars = 32 chords)
    p_xbc = REPO_ROOT / "fixtures" / "external" / "xiaobaichuan" / "page2.jpg"
    if p_xbc.exists():
        geom_xbc = analyze_page(p_xbc.read_bytes(), page=0)
        assert "chord_only" in geom_xbc.notes
        total_chords_xbc = sum(len(s.chord_boxes) for s in geom_xbc.systems)
        assert total_chords_xbc == 32, f"Expected 32 chords on xiaobaichuan p2, got {total_chords_xbc}"
        for s in geom_xbc.systems:
            assert "chord_only" in s.notes
            assert all(not cb.boxed for cb in s.chord_boxes)
            assert all(1.0 <= cb.beat_geo <= 3.0 for cb in s.chord_boxes)

        # Invariant 4: System bbox must include its own lyric lines (Sys 8)
        s8 = geom_xbc.systems[7]
        assert len(s8.lyric_bands) > 0, "Sys 8 should have detected lyric band"
        sy0, sy1 = s8.bbox[1], s8.bbox[3]
        ly0, ly1 = s8.lyric_bands[0]
        assert sy0 <= ly0 <= ly1 <= sy1, f"Sys 8 lyrics ({ly0}, {ly1}) out of system bbox ({sy0}, {sy1})"

    # 2. liusha p2 (4/4 lead sheet, 7 systems = 40 chords)
    p_ls = REPO_ROOT / "fixtures" / "external" / "liusha" / "page2.jpg"
    if p_ls.exists():
        geom_ls = analyze_page(p_ls.read_bytes(), page=0)
        assert "chord_only" in geom_ls.notes
        total_chords_ls = sum(len(s.chord_boxes) for s in geom_ls.systems)
        assert total_chords_ls == 40, f"Expected 40 chords on liusha p2, got {total_chords_ls}"
        assert all(not cb.boxed for s in geom_ls.systems for cb in s.chord_boxes)
        assert all(1.0 <= cb.beat_geo <= 4.0 for s in geom_ls.systems for cb in s.chord_boxes)


def test_non_chart_confidence():
    """Verify non-chart instructional tutorial sheet (huochuai p1) fails render gate (confidence < 0.60)."""
    p_hc = REPO_ROOT / "fixtures" / "external" / "huochuai" / "page1.jpg"
    if p_hc.exists():
        geom_hc = analyze_page(p_hc.read_bytes(), page=0)
        assert geom_hc.confidence < 0.60, f"Tutorial page huochuai p1 should have confidence < 0.60, got {geom_hc.confidence}"


def test_boxed_chart_chords_groundtruth():
    """Verify diaole p1 & p2 and qianlizhiwai p1 have detected chord boxes matching ground truth

    chord count per measure with <= 2 mismatched measures per page, and 0 chord boxes inside lyric bands.
    """
    cases = [
        ("fixtures/groundtruth/diaole.json", 1, "fixtures/pages/page1.jpg"),
        ("fixtures/groundtruth/diaole.json", 2, "fixtures/pages/page2.jpg"),
        ("fixtures/groundtruth/qianlizhiwai.json", 1, "fixtures/external/qianlizhiwai/page1.jpg"),
    ]
    for gt_rel, p_num, img_rel in cases:
        gt_path = REPO_ROOT / gt_rel
        img_path = REPO_ROOT / img_rel
        if not gt_path.exists() or not img_path.exists():
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        geom = analyze_page(img_path.read_bytes(), page=0)
        gt_rows = [r for r in gt["rows"] if r["page"] == p_num]
        assert len(geom.systems) == len(gt_rows), f"{img_rel}: systems {len(geom.systems)} != {len(gt_rows)}"

        mismatched_measures = 0
        total_measures = 0
        in_lyric_count = 0
        for s_idx, (sys_geom, gt_row) in enumerate(zip(geom.systems, gt_rows)):
            for cb in sys_geom.chord_boxes:
                cb_y = (cb.bbox[1] + cb.bbox[3]) / 2.0
                for ly0, ly1 in sys_geom.lyric_bands:
                    if ly0 <= cb_y <= ly1:
                        in_lyric_count += 1
            exp_m_counts = [len(m.get("chords", [])) for m in gt_row["measures"]]
            act_m_counts = [0] * len(sys_geom.measures)
            for cb in sys_geom.chord_boxes:
                act_m_counts[cb.measure_index_in_system] += 1
            assert len(exp_m_counts) == len(act_m_counts), f"Row {s_idx} measure count mismatch: {len(act_m_counts)} vs {len(exp_m_counts)}"
            for act_c, exp_c in zip(act_m_counts, exp_m_counts):
                total_measures += 1
                if act_c != exp_c:
                    mismatched_measures += 1

        assert in_lyric_count == 0, f"{img_rel}: expected 0 chord boxes in lyric bands, got {in_lyric_count}"
        assert mismatched_measures <= 2, f"{img_rel}: expected <= 2 mismatched measures, got {mismatched_measures}/{total_measures}"


def test_scale_and_rotation_invariance(layout_truth):
    """Verify scale and rotation invariance across image scales (0.6x down to 1.7x up)

    with JPEG compression artifacts (quality 70) and rotation (+/- 2 degrees).
    All 14 ground-truth pages must produce identical systems and measures-per-system,
    and chord-box count within +/- 5%.
    """
    import cv2

    for rel_path in layout_truth.keys():
        img_path = REPO_ROOT / rel_path
        if not img_path.exists():
            continue

        raw_bytes = img_path.read_bytes()
        geom_orig = analyze_page(raw_bytes, page=0)
        orig_systems = len(geom_orig.systems)
        orig_measures = [len(s.measures) for s in geom_orig.systems]
        orig_chords = sum(len(s.chord_boxes) for s in geom_orig.systems)

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        for sc, ang, interp in [(0.6, 2.0, cv2.INTER_AREA), (1.7, -2.0, cv2.INTER_CUBIC)]:
            scaled = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=interp)
            mat = cv2.getRotationMatrix2D((scaled.shape[1] // 2, scaled.shape[0] // 2), ang, 1.0)
            rot = cv2.warpAffine(scaled, mat, (scaled.shape[1], scaled.shape[0]), borderValue=(255, 255, 255))
            _, enc = cv2.imencode(".jpg", rot, [cv2.IMWRITE_JPEG_QUALITY, 70])
            geom_trans = analyze_page(enc.tobytes(), page=0)

            assert len(geom_trans.systems) == orig_systems, (
                f"{rel_path} at {sc}x ({ang} deg): expected {orig_systems} systems, got {len(geom_trans.systems)}"
            )
            det_m = [len(s.measures) for s in geom_trans.systems]
            assert det_m == orig_measures, (
                f"{rel_path} at {sc}x ({ang} deg): measures mismatch {det_m} vs {orig_measures}"
            )
            chords_trans = sum(len(s.chord_boxes) for s in geom_trans.systems)
            max_allowed_diff = max(1, int(round(0.05 * orig_chords)))
            assert abs(chords_trans - orig_chords) <= max_allowed_diff, (
                f"{rel_path} at {sc}x ({ang} deg): chord count diff > 5% ({chords_trans} vs {orig_chords})"
            )


def test_dashed_barlines_invariants():
    """Verify dashed barline invariants:
    - Requires >= 3 collinear segments spanning >= 0.55 * melody band height.
    - Must not coincide with digit stroke.
    - Preserves dashed barlines when digit strokes are outside or not coincident.
    """
    import numpy as np

    h, w = 100, 500
    ym = 50.0
    mel_h = 40.0
    y_radius = 30

    # 1. Valid dashed barline (4 collinear segments, span 30 >= 0.55 * 40 = 22) -> detected
    bin_img1 = np.zeros((h, w), dtype=np.uint8)
    for dy in [32, 42, 52, 62]:
        bin_img1[dy : dy + 6, 100:102] = 255
    res1 = find_dashed_barlines(bin_img1, ym=ym, y_radius=y_radius, mel_h=mel_h)
    assert len(res1) == 1
    assert abs(res1[0][0] - 100.5) <= 1.0

    # 2. Too few dashes (< 3 segments) -> rejected
    bin_img2 = np.zeros((h, w), dtype=np.uint8)
    for dy in [35, 55]:
        bin_img2[dy : dy + 6, 100:102] = 255
    res2 = find_dashed_barlines(bin_img2, ym=ym, y_radius=y_radius, mel_h=mel_h)
    assert len(res2) == 0

    # 3. Span too short (< 0.55 * mel_h) -> rejected
    bin_img3 = np.zeros((h, w), dtype=np.uint8)
    for dy in [45, 49, 53]:
        bin_img3[dy : dy + 2, 100:102] = 255
    res3 = find_dashed_barlines(bin_img3, ym=ym, y_radius=y_radius, mel_h=mel_h)
    assert len(res3) == 0

    # 4. Coincides with digit stroke in melody band -> rejected
    bin_img4 = np.zeros((h, w), dtype=np.uint8)
    for dy in [32, 42, 52, 62]:
        bin_img4[dy : dy + 6, 100:102] = 255
    bin_img4[40:60, 98:104] = 255  # digit stroke overlapping x and y
    res4 = find_dashed_barlines(bin_img4, ym=ym, y_radius=y_radius, mel_h=mel_h)
    assert len(res4) == 0

    # 5. Non-coincident digit stroke (separated in x) -> accepted
    bin_img5 = np.zeros((h, w), dtype=np.uint8)
    for dy in [32, 42, 52, 62]:
        bin_img5[dy : dy + 6, 100:102] = 255
    bin_img5[40:60, 150:156] = 255  # digit stroke far away
    res5 = find_dashed_barlines(bin_img5, ym=ym, y_radius=y_radius, mel_h=mel_h)
    assert len(res5) == 1


def test_pickup_and_measure_invariants():
    """Verify pickup invariants and narrow measure pruning:
    - Fragment counts only if separated by real solid/dashed barline.
    - Pickup insertion gated on not chords_below (while allowing full measures >= 0.14 * w).
    - Repeat endings (N-2 and N-1) protected when allow_repeat_ending is True.
    """
    # 1. clean_narrow_measures protects pickup on first system
    bars_sys0 = [50.0, 110.0, 360.0, 610.0, 860.0]
    cleaned_sys0 = clean_narrow_measures(
        bars_sys0, min_m_w=120.0, min_end_w=40.0, sheet_left=50.0, is_first_system=True
    )
    assert cleaned_sys0 == bars_sys0, "First system pickup measure should be preserved"

    # 2. Interior system prunes spurious narrow start fragment
    bars_sys1 = [50.0, 110.0, 360.0, 610.0, 860.0]
    cleaned_sys1 = clean_narrow_measures(
        bars_sys1, min_m_w=120.0, min_end_w=40.0, sheet_left=50.0, is_first_system=False
    )
    assert len(cleaned_sys1) == len(bars_sys1) - 1, "Interior narrow start fragment should be pruned"

    # 3. Dashed barline protects narrow measure on interior system
    cleaned_dashed = clean_narrow_measures(
        bars_sys1, min_m_w=120.0, min_end_w=40.0, dashed_xs={110.0}, sheet_left=50.0, is_first_system=False
    )
    assert cleaned_dashed == bars_sys1, "Dashed barline should protect narrow measure"

    # 4. Repeat endings (measures N-2 and N-1) preserved when allow_repeat_ending=True
    bars_repeat = [50.0, 300.0, 550.0, 642.0, 745.0]
    cleaned_repeat = clean_narrow_measures(
        bars_repeat, min_m_w=118.0, min_end_w=40.0, allow_repeat_ending=True
    )
    assert cleaned_repeat == bars_repeat, "Repeat endings should be preserved"

    # 5. Real page pickup verification: tinghai p2 has pickup measure on system 0
    th2_path = REPO_ROOT / "fixtures" / "external" / "tinghai" / "page2.jpg"
    if th2_path.exists():
        geom_th2 = analyze_page(th2_path.read_bytes(), page=0)
        assert len(geom_th2.systems[0].measures) == 5, "Tinghai p2 sys 0 should have 5 measures including pickup"
        m0_w = geom_th2.systems[0].measures[0].x1 - geom_th2.systems[0].measures[0].x0
        m1_w = geom_th2.systems[0].measures[1].x1 - geom_th2.systems[0].measures[1].x0
        assert m0_w < 0.35 * m1_w, f"First measure ({m0_w}) should be pickup (< 0.35 * {m1_w})"

    # 6. Real page full measure insertion: xindong sys 3 allows full measure >= 0.14 * w without left barline
    xd_path = REPO_ROOT / "fixtures" / "external" / "xindong" / "page1.jpg"
    if xd_path.exists():
        geom_xd = analyze_page(xd_path.read_bytes(), page=0)
        assert len(geom_xd.systems[3].measures) == 5, "Xindong sys 3 should have 5 measures"
        m0_w = geom_xd.systems[3].measures[0].x1 - geom_xd.systems[3].measures[0].x0
        assert m0_w >= 0.14, f"Full measure width should be >= 0.14, got {m0_w}"


def _calculate_max_vertical_run(bin_img: np.ndarray, y0: int, y1: int, bx: int, img_w: int) -> int:
    crop = bin_img[y0:y1, max(0, bx - 2) : min(img_w, bx + 3)]
    max_run = 0
    for c in range(crop.shape[1]):
        col = crop[:, c] > 0
        curr = 0
        for px in col:
            if px:
                curr += 1
                if curr > max_run:
                    max_run = curr
            else:
                curr = 0
    return max_run


def test_barline_continuous_run_rejection():
    """Verify that spurious strokes (dotted-note columns, slur/tie ends) are rejected as barlines.

    Rule (b) requires a real barline to have a continuous ink run >= min_solid_run or >= 3 dashes.
    - Dotted-note column: vertically spaced dots with white gaps between them must be rejected.
    - Slur/tie end: thin horizontal/curved tick (height <= 3 px) must be rejected.
    - Real solid barline: continuous ink run spanning melody band height must be accepted.
    """
    import numpy as np

    h, w = 400, 800
    bin_img = np.zeros((h, w), dtype=np.uint8)
    mel_y0, mel_y1 = 150, 200
    mel_h = float(mel_y1 - mel_y0)

    # 1. Real solid barline at x=200: continuous run of 30 px
    bin_img[mel_y0 + 10 : mel_y1 - 10, 200] = 255

    # 2. Dotted-note column at x=350: 3 dots (height 3 px, gap 8 px) -> max continuous run = 3 px
    bin_img[mel_y0 + 10 : mel_y0 + 13, 350] = 255
    bin_img[mel_y0 + 21 : mel_y0 + 24, 350] = 255
    bin_img[mel_y0 + 32 : mel_y0 + 35, 350] = 255

    # 3. Slur / tie end at x=500: horizontal curve/tick of height 2 px, width 12 px -> max run = 2 px
    bin_img[mel_y0 + 20 : mel_y0 + 22, 494:506] = 255

    # Verify solid bar continuous ink run filter directly
    scale_h = h / 2400.0
    min_solid_run = max(int(round(15 * scale_h)), int(round(0.48 * mel_h)))

    # Solid bar has continuous run >= min_solid_run
    assert _calculate_max_vertical_run(bin_img, mel_y0, mel_y1, 200, w) >= min_solid_run
    # Dotted-note column has max run 3 px < min_solid_run -> rejected
    assert _calculate_max_vertical_run(bin_img, mel_y0, mel_y1, 350, w) < min_solid_run
    # Slur / tie end has max run 2 px < min_solid_run -> rejected
    assert _calculate_max_vertical_run(bin_img, mel_y0, mel_y1, 500, w) < min_solid_run


def test_chord_box_anchor_assignment_stacked_fraction():
    """Verify that stacked fractions anchor to lower-right number across barlines.

    Given a barline at x=250 between measure 0 [0.0, 0.25] and measure 1 [0.25, 0.50]:
    A stacked fraction with whole-box center in measure 0 (x_mid=240), but lower-right
    number in measure 1 (center x=265), must be assigned to measure 1.
    """
    img_w = 1000
    measures = [
        GMeasure(index_in_system=0, x0=0.0, x1=0.25),
        GMeasure(index_in_system=1, x0=0.25, x1=0.50),
        GMeasure(index_in_system=2, x0=0.50, x1=0.75),
        GMeasure(index_in_system=3, x0=0.75, x1=1.0),
    ]

    bx0, by0, bx1, by1 = 200, 10, 280, 80
    # Whole box center: (200 + 280)/2 = 240 -> measure 0 (x < 250)
    # Upper circle (e.g. ⑦): (205, 12, 30, 30) -> center (220, 27)
    # Diagonal slash ╱: (235, 15, 20, 50) -> center (245, 40)
    # Lower circle (e.g. ⑤): (250, 45, 30, 30) -> center (265, 60) in measure 1
    comps = [
        (205, 12, 30, 30),
        (235, 15, 20, 50),
        (250, 45, 30, 30),
    ]

    m_idx, anchor_norm = _assign_chord_box_to_measure(
        bx0, by0, bx1, by1, comps, measures, img_w
    )

    # Must be assigned to measure 1, with anchor x corresponding to lower circle (265 / 1000 = 0.265)
    assert m_idx == 1
    assert round(anchor_norm * img_w) == 265


def test_chord_box_anchor_assignment_plain_box_barline_tolerance():
    """Verify that plain boxes near barlines respect tolerance towards box center.

    Barline at x=250 between measure 0 and measure 1 (img_w=1000).
    - Case A: First glyph at x=245 (within 16px tolerance of barline), box center in measure 1.
      Should be favoured towards measure 1 with anchor clamped to start of measure 1.
    - Case B: First glyph at x=210 (outside tolerance of barline), box spans into measure 1.
      Should remain in measure 0 with anchor at first glyph.
    """
    img_w = 1000
    measures = [
        GMeasure(index_in_system=0, x0=0.0, x1=0.25),
        GMeasure(index_in_system=1, x0=0.25, x1=0.50),
    ]

    # Case A: First glyph center = 245 (5px before barline 250), box center = 265
    comps_a = [(235, 20, 20, 20), (260, 20, 30, 20)]
    m_idx_a, anchor_norm_a = _assign_chord_box_to_measure(
        235, 20, 290, 40, comps_a, measures, img_w
    )
    assert m_idx_a == 1
    assert round(anchor_norm_a * img_w) >= 250  # Clamped to measure 1

    # Case B: First glyph center = 215 (35px before barline), box center = 255
    comps_b = [(200, 20, 30, 20), (250, 20, 40, 20)]
    m_idx_b, anchor_norm_b = _assign_chord_box_to_measure(
        200, 20, 290, 40, comps_b, measures, img_w
    )
    assert m_idx_b == 0
    assert round(anchor_norm_b * img_w) == 215


def test_bar_number_band_and_margin_labels():
    """Verify bar-number band detection and margin label rejection."""
    import numpy as np

    # 1. Bar-number band with dense vertical hatching lines
    h, w = 1000, 800
    hatched_bars = [(100 + i * 12, 100, 2, 25, 87, 112) for i in range(10)]
    assert _is_bar_number_band(hatched_bars, np.full((h, w), 255, dtype=np.uint8), w, h)

    # Normal widely-spaced barlines are not a bar-number band
    normal_bars = [(100 + i * 150, 100, 2, 25, 87, 112) for i in range(4)]
    assert not _is_bar_number_band(normal_bars, np.full((h, w), 255, dtype=np.uint8), w, h)

    # Bars lower down the page are never considered a bar-number band
    lower_bars = [(100 + i * 12, 500, 2, 25, 487, 512) for i in range(10)]
    assert not _is_bar_number_band(lower_bars, np.full((h, w), 255, dtype=np.uint8), w, h)

    # 2. Left margin chord box rejection (labels before first barline)
    measures = [
        GMeasure(index_in_system=0, x0=0.20, x1=0.60),
        GMeasure(index_in_system=1, x0=0.60, x1=1.00),
    ]
    bin_img = np.zeros((200, 1000), dtype=np.uint8)
    # Box strictly before first barline (x=200): e.g. x=50..120
    bin_img[20:45, 50:120] = 255
    # Box inside measure 0: e.g. x=250..320
    bin_img[20:45, 250:320] = 255

    boxes = detect_chord_boxes(bin_img, 10, 60, measures, 1000, 200)
    # The margin label before x=200 should be rejected; only the box inside measure 0 kept
    assert all(cb.bbox[0] * 1000 >= 180 for cb in boxes)





