"""Unit tests for classical CV layout analysis (app/omr/layout.py).

Verifies system count, measures per system, bbox containment, non-overlap,
chord box containment, deskew, and exception-free fallbacks.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.omr.layout import analyze_page, estimate_skew_angle, deskew_image

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

        # Most pages should match exactly
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

