"""Unit tests for smart lead-sheet rendering (app/smart/render.py).

Verifies:
1. lead_measure_to_melody_string produces canonical jianpu strings that app.qa.melody.parse_melody
   round-trips (onsets, degrees, accidentals) for all measures in fixtures and custom rhythms.
2. render_lead_sheet returns valid PNG page bytes and ParsedSheet geometry matching downstream expectations.
3. Piano and ukulele arrangement + overlay rendering succeed end-to-end.
4. chords_only fixtures produce blank melody strings and render cleanly.
5. Robustness against odd inputs (empty sheets, overflowing notes, custom chords, unusual barlines).
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest
from PIL import Image

from app.arrange.piano import arrange as arrange_piano
from app.arrange.ukulele import arrange_ukulele
from app.qa.melody import parse_melody
from app.render.overlay import render_pages
from app.smart.models import LeadChord, LeadMeasure, LeadNote, LeadSheet
from app.smart.render import (
    NoteLayoutBox,
    get_lead_sheet_layout_boxes,
    lead_measure_to_melody_string,
    render_lead_sheet,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "lead"


def _load_lead(filename: str) -> LeadSheet:
    path = FIXTURES_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return LeadSheet.model_validate(data)


# ---------------------------------------------------------------------------
# 1. Round-trip Melody Parsing Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_name",
    ["twinkle.json", "ode_to_joy.json", "jasmine_flower.json"],
)
def test_lead_measure_to_melody_string_round_trip(fixture_name: str) -> None:
    """Verify parse_melody on lead_measure_to_melody_string matches original notes."""
    lead = _load_lead(fixture_name)
    for m in lead.measures:
        s = lead_measure_to_melody_string(m)
        if not m.notes:
            assert s == ""
            continue

        parsed = parse_melody(s, beats=m.beats, include_rests=True)
        assert parsed is not None, f"Failed to parse melody {s!r} in measure {m.index}"
        assert len(parsed.notes) == len(m.notes), (
            f"Note count mismatch in {fixture_name} m={m.index}: "
            f"got {len(parsed.notes)} vs orig {len(m.notes)} (string={s!r})"
        )

        for (parsed_onset, parsed_deg, parsed_acc), orig_note in zip(parsed.notes, m.notes):
            # parse_melody uses 1-based beat inside measure; LeadNote.onset is 0-based
            assert math.isclose(parsed_onset, orig_note.onset + 1.0, abs_tol=1e-3), (
                f"Onset mismatch in {fixture_name} m={m.index}: {parsed_onset} vs {orig_note.onset + 1.0}"
            )
            assert parsed_deg == orig_note.degree, (
                f"Degree mismatch in {fixture_name} m={m.index}: {parsed_deg} vs {orig_note.degree}"
            )
            assert parsed_acc == orig_note.accidental, (
                f"Accidental mismatch in {fixture_name} m={m.index}: {parsed_acc} vs {orig_note.accidental}"
            )


def test_lead_measure_to_melody_string_edge_cases() -> None:
    """Test individual rhythm and accidental token constructions."""
    # 1. Empty notes
    m_empty = LeadMeasure(index=0, beats=4.0, chords=[], notes=[])
    assert lead_measure_to_melody_string(m_empty) == ""

    # 2. Zero or negative beats (via model_construct bypassing validation)
    m_zero_beats = LeadMeasure.model_construct(
        index=0,
        beats=0.0,
        chords=[],
        notes=[LeadNote(degree=1, onset=0.0, duration=1.0)],
    )
    assert lead_measure_to_melody_string(m_zero_beats) == ""

    # 3. Whole note (4 beats) -> '1 - - -'
    m_whole = LeadMeasure(
        index=0,
        beats=4.0,
        chords=[],
        notes=[LeadNote(degree=5, onset=0.0, duration=4.0)],
    )
    assert lead_measure_to_melody_string(m_whole) == "5 - - -"
    parsed_whole = parse_melody("5 - - -", beats=4.0, include_rests=True)
    assert parsed_whole is not None
    assert len(parsed_whole.notes) == 1
    assert parsed_whole.notes[0] == (1.0, 5, 0)
    assert parsed_whole.beat_sum == 4.0

    # 4. Half note + two quarters -> '1 - 3 4'
    m_half = LeadMeasure(
        index=0,
        beats=4.0,
        chords=[],
        notes=[
            LeadNote(degree=1, onset=0.0, duration=2.0),
            LeadNote(degree=3, onset=2.0, duration=1.0),
            LeadNote(degree=4, onset=3.0, duration=1.0),
        ],
    )
    assert lead_measure_to_melody_string(m_half) == "1 - 3 4"

    # 5. Dotted half + quarter -> '1 - - 5'
    m_dotted_half = LeadMeasure(
        index=0,
        beats=4.0,
        chords=[],
        notes=[
            LeadNote(degree=1, onset=0.0, duration=3.0),
            LeadNote(degree=5, onset=3.0, duration=1.0),
        ],
    )
    assert lead_measure_to_melody_string(m_dotted_half) == "1 - - 5"

    # 6. Dotted eighth + sixteenth -> '1.2'
    m_dotted_eighth = LeadMeasure(
        index=0,
        beats=1.0,
        chords=[],
        notes=[
            LeadNote(degree=1, onset=0.0, duration=0.75),
            LeadNote(degree=2, onset=0.75, duration=0.25),
        ],
    )
    assert lead_measure_to_melody_string(m_dotted_eighth) == "1.2"
    parsed_de = parse_melody("1.2", beats=1.0, include_rests=True)
    assert parsed_de is not None
    assert len(parsed_de.notes) == 2
    assert parsed_de.notes[0] == (1.0, 1, 0)
    assert parsed_de.notes[1] == (1.75, 2, 0)

    # 7. Sixteenth + dotted eighth -> '12.'
    m_sixteenth_dotted = LeadMeasure(
        index=0,
        beats=1.0,
        chords=[],
        notes=[
            LeadNote(degree=1, onset=0.0, duration=0.25),
            LeadNote(degree=2, onset=0.25, duration=0.75),
        ],
    )
    assert lead_measure_to_melody_string(m_sixteenth_dotted) == "12."
    parsed_sd = parse_melody("12.", beats=1.0, include_rests=True)
    assert parsed_sd is not None
    assert len(parsed_sd.notes) == 2
    assert parsed_sd.notes[0] == (1.0, 1, 0)
    assert parsed_sd.notes[1] == (1.25, 2, 0)

    # 8. Four sixteenths -> '1234'
    m_sixteenths = LeadMeasure(
        index=0,
        beats=1.0,
        chords=[],
        notes=[
            LeadNote(degree=1, onset=0.0, duration=0.25),
            LeadNote(degree=2, onset=0.25, duration=0.25),
            LeadNote(degree=3, onset=0.5, duration=0.25),
            LeadNote(degree=4, onset=0.75, duration=0.25),
        ],
    )
    assert lead_measure_to_melody_string(m_sixteenths) == "1234"

    # 9. Accidentals
    m_acc = LeadMeasure(
        index=0,
        beats=2.0,
        chords=[],
        notes=[
            LeadNote(degree=4, accidental=1, onset=0.0, duration=1.0),
            LeadNote(degree=7, accidental=-1, onset=1.0, duration=1.0),
        ],
    )
    assert lead_measure_to_melody_string(m_acc) == "#4 b7"
    parsed_acc = parse_melody("#4 b7", beats=2.0, include_rests=True)
    assert parsed_acc is not None
    assert parsed_acc.notes[0] == (1.0, 4, 1)
    assert parsed_acc.notes[1] == (2.0, 7, -1)

    # 10. Single eighth note with rest -> '10' or '01'
    m_sub_rest1 = LeadMeasure(
        index=0,
        beats=1.0,
        chords=[],
        notes=[LeadNote(degree=3, onset=0.0, duration=0.5)],
    )
    assert lead_measure_to_melody_string(m_sub_rest1) == "30"

    m_sub_rest2 = LeadMeasure(
        index=0,
        beats=1.0,
        chords=[],
        notes=[LeadNote(degree=6, onset=0.5, duration=0.5)],
    )
    assert lead_measure_to_melody_string(m_sub_rest2) == "06"


# ---------------------------------------------------------------------------
# 2. ParsedSheet Geometry & Image Validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_name",
    ["twinkle.json", "ode_to_joy.json", "jasmine_flower.json", "chords_only.json"],
)
def test_parsed_sheet_validity(fixture_name: str) -> None:
    """Verify render_lead_sheet produces valid PNGs, non-overlapping systems, and valid bboxes."""
    lead = _load_lead(fixture_name)
    pages_bytes, sheet = render_lead_sheet(lead, page_width=2400)

    # 1. Page counts match
    assert len(pages_bytes) == len(sheet.pages)
    assert len(pages_bytes) >= 1

    # 2. Verify images are valid PNGs matching sheet.pages dimensions
    loaded_images: list[Image.Image] = []
    for p_idx, (p_bytes, p_info) in enumerate(zip(pages_bytes, sheet.pages)):
        img = Image.open(io.BytesIO(p_bytes))
        assert img.format == "PNG"
        assert img.size == (p_info.width, p_info.height)
        assert img.width == 2400
        loaded_images.append(img)

    # 3. Measures across sheet match lead.measures
    all_measures = sheet.measures()
    assert len(all_measures) == len(lead.measures)
    for idx, m in enumerate(all_measures):
        assert m.index == idx
        # Bbox in normalized [0, 1]
        x0, y0, x1, y1 = m.bbox
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
        assert m.beats == lead.measures[idx].beats

        # Chord bboxes in [0, 1]
        for c in m.chords:
            cx0, cy0, cx1, cy1 = c.bbox
            assert 0.0 <= cx0 < cx1 <= 1.0
            assert 0.0 <= cy0 < cy1 <= 1.0

    # 4. Systems per page are strictly non-overlapping vertically
    for p_idx in range(len(sheet.pages)):
        page_systems = [s for s in sheet.systems if s.page == p_idx]
        assert len(page_systems) >= 1

        for i in range(len(page_systems)):
            sx0, sy0, sx1, sy1 = page_systems[i].bbox
            assert 0.0 <= sx0 < sx1 <= 1.0
            assert 0.0 <= sy0 < sy1 <= 1.0

            if i + 1 < len(page_systems):
                next_sy0 = page_systems[i + 1].bbox[1]
                # Next system must start strictly after current system ends
                assert next_sy0 > sy1, f"System overlap on page {p_idx} between {i} and {i+1}"

                # 5. Check that the gap between systems is pure white for find_system_cut_row
                p_img = loaded_images[p_idx]
                gap_top = int(round(sy1 * p_img.height))
                gap_bottom = int(round(next_sy0 * p_img.height))
                assert gap_bottom > gap_top + 10, f"Gap too small between sys {i} and {i+1}"

                # Sample row in the middle of the gap: all pixels must be pure white (255, 255, 255)
                mid_y = (gap_top + gap_bottom) // 2
                gap_stripe = p_img.crop((0, mid_y, p_img.width, mid_y + 1))
                extrema = gap_stripe.getextrema()
                # For RGB image, extrema is ((minR, maxR), (minG, maxG), (minB, maxB))
                for channel_ext in extrema:
                    assert channel_ext[0] == 255, f"Non-white pixel found in inter-system gap row {mid_y}"


# ---------------------------------------------------------------------------
# 3. Downstream Piano Arrangement & Overlay Integration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_name", ["twinkle.json", "ode_to_joy.json"])
def test_piano_arrange_overlay_end_to_end(fixture_name: str) -> None:
    """Verify render_lead_sheet -> arrange_piano -> render_pages end-to-end pipeline."""
    lead = _load_lead(fixture_name)
    pages_bytes, sheet = render_lead_sheet(lead, page_width=2400)

    # Arrange piano
    piano_arr = arrange_piano(sheet, start_key=sheet.header.original_key or "C")
    assert len(piano_arr.measures) == len(lead.measures)

    # Render overlay
    final_pages = render_pages(pages_bytes, sheet, piano_arr)
    assert len(final_pages) == len(pages_bytes)

    for p_idx, final_img in enumerate(final_pages):
        assert final_img.width == 2400
        # Piano accompaniment expands page height to insert accompaniment staves
        orig_h = sheet.pages[p_idx].height
        assert final_img.height > orig_h, (
            f"Expected page {p_idx} height ({final_img.height}) > original ({orig_h})"
        )


# ---------------------------------------------------------------------------
# 4. Downstream Ukulele Arrangement & Overlay Integration
# ---------------------------------------------------------------------------

def test_ukulele_arrange_overlay_end_to_end() -> None:
    """Verify render_lead_sheet -> arrange_ukulele -> render_pages end-to-end pipeline."""
    lead = _load_lead("twinkle.json")
    pages_bytes, sheet = render_lead_sheet(lead, page_width=2400)

    # Arrange ukulele (in offline mode)
    uke_arr = arrange_ukulele(
        sheet,
        start_key=sheet.header.original_key or "C",
        difficulty="beginner",
    )
    assert len(uke_arr.measures) == len(lead.measures)

    # Render overlay
    final_pages = render_pages(pages_bytes, sheet, uke_arr)
    assert len(final_pages) == len(pages_bytes)

    for p_idx, final_img in enumerate(final_pages):
        assert final_img.width == 2400
        orig_h = sheet.pages[p_idx].height
        assert final_img.height > orig_h


# ---------------------------------------------------------------------------
# 5. Chords Only Fixture
# ---------------------------------------------------------------------------

def test_chords_only_fixture() -> None:
    """Verify chords_only mode renders rhythm slashes and sets empty melody strings."""
    lead = _load_lead("chords_only.json")
    assert lead.chords_only is True

    pages_bytes, sheet = render_lead_sheet(lead, page_width=2400)
    assert len(pages_bytes) >= 1

    # Every measure must have empty melody string
    for m in sheet.measures():
        assert m.melody == ""
        assert len(m.chords) >= 1

    # Verify image is valid
    img = Image.open(io.BytesIO(pages_bytes[0]))
    assert img.size == (2400, sheet.pages[0].height)


# ---------------------------------------------------------------------------
# 6. Robustness & Odd Inputs
# ---------------------------------------------------------------------------

def test_odd_input_robustness() -> None:
    """Ensure renderer never crashes on edge cases, empty input, or irregular values."""
    # 1. Completely empty sheet
    empty_lead = LeadSheet(title="Empty Sheet", key="C", time_signature="4/4", measures=[])
    p_bytes, p_sheet = render_lead_sheet(empty_lead)
    assert len(p_bytes) == 1
    assert len(p_sheet.systems) == 0
    img = Image.open(io.BytesIO(p_bytes[0]))
    assert img.width == 2400

    # 2. Sheet with single measure, exotic chord raw strings, and out-of-range notes
    odd_m = LeadMeasure(
        index=0,
        beats=4.0,
        section="Intro",
        chords=[
            LeadChord(raw="N.C.", beat=1.0, agreement="agree"),
            LeadChord(raw="C#m7(b5)/G", beat=2.5, agreement="disagree", alternatives=["Gdim"]),
            LeadChord(raw="???", beat=4.0, agreement="agree"),
        ],
        notes=[
            # Degree 0 rest
            LeadNote(degree=0, onset=0.0, duration=1.0),
            # Note with high octave dots and flat accidental
            LeadNote(degree=7, accidental=-1, octave=2, onset=1.0, duration=0.5),
            # Note with low octave dots and sharp accidental
            LeadNote(degree=4, accidental=1, octave=-2, onset=1.5, duration=0.5),
            # Note overflowing bar duration
            LeadNote(degree=1, onset=3.0, duration=5.0),
        ],
    )
    odd_lead = LeadSheet(
        title="Edge Case Sheet",
        artist="Test Artist",
        key="F#",
        time_signature="4/4",
        tempo_bpm=120,
        measures=[odd_m],
    )
    odd_bytes, odd_sheet = render_lead_sheet(odd_lead, page_width=1800)
    assert len(odd_bytes) == 1
    assert len(odd_sheet.measures()) == 1
    m_out = odd_sheet.measures()[0]
    assert len(m_out.chords) == 3
    img_odd = Image.open(io.BytesIO(odd_bytes[0]))
    assert img_odd.width == 1800

    # 3. Custom page width (3000px)
    wide_bytes, wide_sheet = render_lead_sheet(odd_lead, page_width=3000)
    assert len(wide_bytes) == 1
    img_wide = Image.open(io.BytesIO(wide_bytes[0]))
    assert img_wide.width == 3000


# ---------------------------------------------------------------------------
# 7. Engraving Spacing & Visual Geometry Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_name",
    ["twinkle.json", "ode_to_joy.json", "jasmine_flower.json", "chords_only.json"],
)
def test_no_adjacent_glyph_or_lyric_overlap(fixture_name: str) -> None:
    """Verify that no two adjacent melody digit boxes overlap and no two lyric glyph boxes overlap."""
    lead = _load_lead(fixture_name)
    boxes = get_lead_sheet_layout_boxes(lead, page_width=2400)

    # Group by measure index
    by_measure: dict[int, list[NoteLayoutBox]] = {}
    for b in boxes:
        by_measure.setdefault(b.measure_index, []).append(b)

    for m_idx, m_boxes in by_measure.items():
        for i in range(len(m_boxes) - 1):
            curr_b = m_boxes[i]
            next_b = m_boxes[i + 1]

            # 1. Melody digit boxes must not overlap horizontally
            assert curr_b.digit_bbox[2] <= next_b.digit_bbox[0], (
                f"Melody digit box overlap in {fixture_name} m={m_idx} between note {i} and {i+1}: "
                f"{curr_b.digit_bbox} vs {next_b.digit_bbox}"
            )

            # 2. Lyric glyph boxes (if both present) must not overlap horizontally
            if curr_b.lyric_bbox is not None and next_b.lyric_bbox is not None:
                assert curr_b.lyric_bbox[2] <= next_b.lyric_bbox[0], (
                    f"Lyric glyph box overlap in {fixture_name} m={m_idx} between note {i} and {i+1}: "
                    f"{curr_b.lyric_bbox} vs {next_b.lyric_bbox}"
                )


def test_adaptive_measures_per_row_drops_on_dense_content() -> None:
    """Verify that a row of dense sixteenth/eighth notes adaptively drops to 3 or 2 measures."""
    # Create 6 measures filled with sixteenth notes (16 notes per measure)
    dense_measures: list[LeadMeasure] = []
    for m_i in range(6):
        m_notes = [
            LeadNote(
                degree=((k % 7) + 1),
                onset=k * 0.25,
                duration=0.25,
                lyric=f"字{k}",
            )
            for k in range(16)
        ]
        dense_measures.append(
            LeadMeasure(
                index=m_i,
                beats=4.0,
                chords=[LeadChord(raw="C", beat=1.0), LeadChord(raw="G", beat=3.0)],
                notes=m_notes,
            )
        )

    dense_lead = LeadSheet(
        title="Dense Sixteenths Test",
        key="C",
        time_signature="4/4",
        measures=dense_measures,
    )

    _, sheet = render_lead_sheet(dense_lead, page_width=2400)
    # With 16 sixteenth notes per measure, row capacity must not force 4 measures per row
    measures_per_system = [len(sys.measures) for sys in sheet.systems]
    for count in measures_per_system:
        assert count <= 3, f"Expected dense system measure count <= 3, got {count}"
    assert any(count == 2 for count in measures_per_system), (
        f"Expected at least one system to drop to 2 measures on dense content, got {measures_per_system}"
    )


def test_section_box_clearance_above_chords() -> None:
    """Verify section label boxes have padding and do not touch chord row below."""
    lead = _load_lead("twinkle.json")
    pages_bytes, sheet = render_lead_sheet(lead, page_width=2400)

    # In twinkle, measure 0 has section "主歌A", measure 4 has "主歌B", measure 8 has "副歌"
    for sys in sheet.systems:
        sec_m = [m for m in sys.measures if lead.measures[m.index].section]
        if sec_m:
            for m in sec_m:
                # System bbox y0 to y1
                # Chord symbols in measure
                assert len(m.chords) >= 1
                # The section box top is at sys_y0 + 2px, box height 34px (ends at sys_y0 + 36px)
                # First chord y0 in normalized coords must be below section box
                for c in m.chords:
                    chord_top_px = c.bbox[1] * sheet.pages[sys.page].height
                    sys_top_px = sys.bbox[1] * sheet.pages[sys.page].height
                    # Clearance must be at least 10px below section box bottom (sys_top_px + 36px)
                    assert chord_top_px >= sys_top_px + 45.0, (
                        f"Chord too close to section box: chord_top={chord_top_px}, sys_top={sys_top_px}"
                    )


def test_extension_dash_placement_between_notes() -> None:
    """Verify that extension dashes are laid out strictly between their owning note and the next note."""
    # 1. Check all notes with duration >= 2.0 in fixtures
    for fixture_name in ["twinkle.json", "ode_to_joy.json", "jasmine_flower.json"]:
        lead = _load_lead(fixture_name)
        boxes = get_lead_sheet_layout_boxes(lead, page_width=2400)
        by_measure: dict[int, list[NoteLayoutBox]] = {}
        for b in boxes:
            by_measure.setdefault(b.measure_index, []).append(b)

        for m_idx, m_boxes in by_measure.items():
            m = lead.measures[m_idx]
            for i, note in enumerate(m.notes):
                if note.duration >= 2.0:
                    curr_b = m_boxes[i]
                    expected_dashes = int(round(note.duration)) - 1
                    assert len(curr_b.dash_xs) == expected_dashes, (
                        f"Expected {expected_dashes} dashes for {fixture_name} m={m_idx} note {i}, "
                        f"got {len(curr_b.dash_xs)}"
                    )
                    # All dashes must lie strictly after the owning note's digit
                    for dash_x in curr_b.dash_xs:
                        assert dash_x > curr_b.digit_bbox[2], (
                            f"Dash at x={dash_x} is not to the right of digit at {curr_b.digit_bbox}"
                        )
                        # If there is a next note in this measure, dash must lie strictly before it
                        if i + 1 < len(m_boxes):
                            next_b = m_boxes[i + 1]
                            assert dash_x < next_b.digit_bbox[0], (
                                f"Dash at x={dash_x} for note {i} overlaps/exceeds next note digit at {next_b.digit_bbox}"
                            )

    # 2. Check synthetic measure [3 dur2, 1 dur2] (as in Jasmine last measure)
    synth_lead1 = LeadSheet(
        title="Synthetic Two Half Notes",
        key="C",
        time_signature="4/4",
        measures=[
            LeadMeasure(
                index=0,
                beats=4.0,
                chords=[LeadChord(raw="1", beat=1.0), LeadChord(raw="5", beat=3.0)],
                notes=[
                    LeadNote(degree=3, onset=0.0, duration=2.0, lyric="花"),
                    LeadNote(degree=1, onset=2.0, duration=2.0, lyric=""),
                ],
            )
        ],
    )
    boxes1 = get_lead_sheet_layout_boxes(synth_lead1, page_width=2400)
    assert len(boxes1) == 2
    b0, b1 = boxes1[0], boxes1[1]
    assert len(b0.dash_xs) == 1
    assert len(b1.dash_xs) == 1
    # First half note's dash must be strictly between note 3 and note 1
    assert b0.digit_bbox[2] < b0.dash_xs[0] < b1.digit_bbox[0], (
        f"First dash {b0.dash_xs[0]} not between note0 ({b0.digit_bbox}) and note1 ({b1.digit_bbox})"
    )
    # Second half note's dash must be strictly after note 1
    assert b1.digit_bbox[2] < b1.dash_xs[0]

    # 3. Check synthetic measure [5 dur3, 1 dur1] (dotted half note followed by quarter note)
    synth_lead2 = LeadSheet(
        title="Synthetic Dotted Half Plus Quarter",
        key="C",
        time_signature="4/4",
        measures=[
            LeadMeasure(
                index=0,
                beats=4.0,
                chords=[LeadChord(raw="5", beat=1.0)],
                notes=[
                    LeadNote(degree=5, onset=0.0, duration=3.0, lyric="长"),
                    LeadNote(degree=1, onset=3.0, duration=1.0, lyric="音"),
                ],
            )
        ],
    )
    boxes2 = get_lead_sheet_layout_boxes(synth_lead2, page_width=2400)
    assert len(boxes2) == 2
    b0, b1 = boxes2[0], boxes2[1]
    assert len(b0.dash_xs) == 2
    # Both dashes for note 5 must lie strictly between note 5 and note 1
    d1, d2 = b0.dash_xs[0], b0.dash_xs[1]
    assert b0.digit_bbox[2] < d1 < d2 < b1.digit_bbox[0], (
        f"Dashes ({d1}, {d2}) not strictly ordered between note0 ({b0.digit_bbox}) and note1 ({b1.digit_bbox})"
    )


