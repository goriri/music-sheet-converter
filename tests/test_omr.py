"""Offline unit tests for OMR module.

Covers bbox conversions, page stitching, barline refinement with OpenCV,
and ParsedSheet JSON round-trip validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.models import ParsedSheet, System, Measure, ChordSymbol, SongHeader, PageInfo, KeyChange
from app.omr.gemini_omr import box_2d_to_bbox
from app.omr.barlines import refine_measure_boxes, detect_system_barlines


REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE1_PATH = REPO_ROOT / "fixtures" / "pages" / "page1.jpg"
PAGE2_PATH = REPO_ROOT / "fixtures" / "pages" / "page2.jpg"
SAMPLE_JSON_PATH = REPO_ROOT / "fixtures" / "omr_sample.json"


def test_bbox_conversion():
    """Test Gemini box_2d [ymin, xmin, ymax, xmax] (0..1000) to normalized (x0, y0, x1, y1) in 0..1."""
    # Standard box
    box = [150, 250, 350, 650]
    bbox = box_2d_to_bbox(box)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 == pytest.approx(0.25, abs=1e-4)
    assert y0 == pytest.approx(0.15, abs=1e-4)
    assert x1 == pytest.approx(0.65, abs=1e-4)
    assert y1 == pytest.approx(0.35, abs=1e-4)

    # Inverted coordinates should be ordered x0 <= x1, y0 <= y1
    inv_box = [400, 700, 200, 100]
    bbox_inv = box_2d_to_bbox(inv_box)
    assert bbox_inv is not None
    assert bbox_inv[0] <= bbox_inv[2]
    assert bbox_inv[1] <= bbox_inv[3]

    # Boundary and clamping
    clamp_box = [-50, -100, 1200, 1500]
    bbox_clamped = box_2d_to_bbox(clamp_box)
    assert bbox_clamped == (0.0, 0.0, 1.0, 1.0)

    # Empty / None
    assert box_2d_to_bbox(None) is None
    assert box_2d_to_bbox([]) is None
    assert box_2d_to_bbox([10, 20]) is None


def test_page_stitching_and_global_indices():
    """Verify that measure indices and key change targets stitch globally across pages."""
    # Synthesize two pages of systems
    sys_p0_m0 = Measure(index=0, bbox=(0.05, 0.1, 0.45, 0.2))
    sys_p0_m1 = Measure(index=1, bbox=(0.45, 0.1, 0.95, 0.2))
    sys_p0 = System(page=0, bbox=(0.05, 0.1, 0.95, 0.2), measures=[sys_p0_m0, sys_p0_m1])

    sys_p1_m0 = Measure(index=2, bbox=(0.05, 0.1, 0.45, 0.2))
    sys_p1_m1 = Measure(index=3, bbox=(0.45, 0.1, 0.95, 0.2))
    sys_p1 = System(page=1, bbox=(0.05, 0.1, 0.95, 0.2), measures=[sys_p1_m0, sys_p1_m1])

    kc = KeyChange(at_measure=2, raw="轉成2調(Ab)", semitones=2)
    sheet = ParsedSheet(
        header=SongHeader(title="Test Song", original_key="F#"),
        pages=[PageInfo(width=1000, height=1400), PageInfo(width=1000, height=1400)],
        systems=[sys_p0, sys_p1],
        key_changes=[kc],
    )

    all_measures = sheet.measures()
    assert len(all_measures) == 4
    for i, m in enumerate(all_measures):
        assert m.index == i

    assert sheet.key_changes[0].at_measure == 2
    assert sheet.key_changes[0].semitones == 2


def test_json_schema_roundtrip_with_sample():
    """Verify that fixtures/omr_sample.json validates cleanly against ParsedSheet contract."""
    assert SAMPLE_JSON_PATH.is_file(), f"Fixture missing: {SAMPLE_JSON_PATH}"

    raw_json = SAMPLE_JSON_PATH.read_text(encoding="utf-8")
    sheet = ParsedSheet.model_validate_json(raw_json)

    # Contract invariants
    assert sheet.header.title == "掉了"
    assert sheet.header.style == "Slow Soul"
    assert sheet.header.original_key == "F#"
    assert sheet.header.tempo_bpm == 81.0
    assert sheet.header.male_key == "Bb"
    assert sheet.header.female_key == "F"

    assert len(sheet.pages) == 2
    assert sheet.pages[0].width == 1056
    assert sheet.pages[0].height == 1402
    assert sheet.pages[1].width == 1016
    assert sheet.pages[1].height == 1376

    assert len(sheet.systems) == 19
    measures = sheet.measures()
    assert len(measures) == 77

    # Check measure index continuity
    for idx, m in enumerate(measures):
        assert m.index == idx

    # Check key change
    assert len(sheet.key_changes) >= 1
    kc = sheet.key_changes[0]
    assert kc.semitones == 2
    assert "轉成2調" in kc.raw
    assert kc.at_measure == 50

    # Check spot-checked chords
    row1_chords = [c.raw for c in sheet.systems[0].measures[0].chords]
    assert row1_chords == ["1(2)"]
    row3_m4_chords = [c.raw for c in sheet.systems[2].measures[3].chords]
    assert row3_m4_chords == ["5m6/2", "67/1#"]

    # Dump and re-validate
    dumped = sheet.model_dump_json()
    sheet_reloaded = ParsedSheet.model_validate_json(dumped)
    assert sheet_reloaded.header.title == sheet.header.title
    assert len(sheet_reloaded.measures()) == len(sheet.measures())


def test_barline_refinement_on_page1():
    """Test OpenCV barline refinement on page1.jpg using parsed sheet."""
    if not PAGE1_PATH.is_file():
        pytest.skip("sample chart image not in repo (local-only fixture)")
    assert SAMPLE_JSON_PATH.is_file()

    img_bytes = PAGE1_PATH.read_bytes()
    sheet = ParsedSheet.model_validate_json(SAMPLE_JSON_PATH.read_text(encoding="utf-8"))

    # Record measure boundaries before refinement
    sys0_measures_before = [m.bbox for m in sheet.systems[0].measures]

    # Run refinement
    refine_measure_boxes(img_bytes, sheet, page=0)

    # Measure boxes must remain valid and strictly increasing in x
    for s in sheet.systems:
        if s.page != 0:
            continue
        for m in s.measures:
            x0, y0, x1, y1 = m.bbox
            assert 0.0 <= x0 < x1 <= 1.0
            assert 0.0 <= y0 < y1 <= 1.0

    # System 1 bar boundaries should span ~0.05 to ~0.93
    sys0_measures_after = [m.bbox for m in sheet.systems[0].measures]
    assert len(sys0_measures_after) == 4
    assert sys0_measures_after[0][0] == pytest.approx(0.055, abs=0.015)
    assert sys0_measures_after[-1][2] == pytest.approx(0.923, abs=0.015)


def test_barline_refinement_never_raises():
    """Ensure refine_measure_boxes never raises, even on malformed or corrupt data."""
    sheet = ParsedSheet(
        header=SongHeader(),
        pages=[PageInfo(width=100, height=100)],
        systems=[],
    )
    # Corrupt image bytes
    refine_measure_boxes(b"not_an_image", sheet, page=0)
    assert any("failed to decode image" in w for w in sheet.warnings)
