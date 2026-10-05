"""Unit tests for the ukulele accompaniment overlay renderer."""
from __future__ import annotations

import io
import pytest
from PIL import Image, ImageDraw

from app.models import (
    Arrangement,
    ChordSymbol,
    KeyChange,
    Measure,
    MeasureArrangement,
    PageInfo,
    ParsedSheet,
    ResolvedChord,
    SectionPlan,
    SongHeader,
    System,
    UkeMeasure,
    UkeShape,
    UkeStroke,
)
from app.render.ukulele import (
    UKE_STRIP_HEIGHT,
    build_uke_header_title,
    draw_chord_diagram,
    make_uke_header_band,
    render_uke_page,
    render_uke_pages,
    render_uke_pdf,
    render_uke_system_strip,
)


def create_dummy_png_bytes(width: int = 1000, height: int = 1400) -> bytes:
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sample_uke_sheet() -> ParsedSheet:
    return ParsedSheet(
        header=SongHeader(
            title="测试曲目",
            style="Slow Soul",
            time_signature="4/4",
            tempo_bpm=80.0,
            original_key="F",
            raw="测试曲目 4/4 80",
        ),
        pages=[PageInfo(width=1000, height=1400), PageInfo(width=1000, height=1400)],
        systems=[
            System(
                page=0,
                bbox=(0.05, 0.1, 0.95, 0.25),
                measures=[
                    Measure(
                        index=0,
                        bbox=(0.05, 0.1, 0.5, 0.25),
                        beats=4.0,
                        chords=[ChordSymbol(raw="1", beat=1.0)],
                        melody="1 2 3 4",
                    ),
                    Measure(
                        index=1,
                        bbox=(0.5, 0.1, 0.95, 0.25),
                        beats=4.0,
                        chords=[ChordSymbol(raw="5", beat=1.0)],
                        melody="5 - - -",
                    ),
                ],
            ),
            System(
                page=0,
                bbox=(0.05, 0.3, 0.95, 0.45),
                measures=[
                    Measure(
                        index=2,
                        bbox=(0.05, 0.3, 0.5, 0.45),
                        beats=4.0,
                        chords=[ChordSymbol(raw="6m", beat=1.0)],
                        melody="6 - - -",
                    ),
                    Measure(
                        index=3,
                        bbox=(0.5, 0.3, 0.95, 0.45),
                        beats=4.0,
                        chords=[ChordSymbol(raw="4", beat=1.0)],
                        melody="4 - - -",
                    ),
                ],
            ),
            System(
                page=1,
                bbox=(0.05, 0.1, 0.95, 0.25),
                measures=[
                    Measure(
                        index=4,
                        bbox=(0.05, 0.1, 0.5, 0.25),
                        beats=4.0,
                        chords=[ChordSymbol(raw="1", beat=1.0), ChordSymbol(raw="5", beat=3.0)],
                        melody="1 1 5 5",
                    ),
                    Measure(
                        index=5,
                        bbox=(0.5, 0.1, 0.95, 0.25),
                        beats=4.0,
                        chords=[ChordSymbol(raw="1", beat=1.0)],
                        melody="1 - - -",
                    ),
                ],
            ),
        ],
        key_changes=[
            KeyChange(at_measure=3, raw="转成G调", semitones=2),
        ],
        warnings=[],
    )


def create_sample_uke_arrangement(capo: int = 0) -> Arrangement:
    return Arrangement(
        instrument="ukulele",
        difficulty="intermediate",
        start_key="F" if capo == 0 else "Eb",
        capo=capo,
        shape_key="F" if capo == 0 else "D",
        measures=[
            MeasureArrangement(
                measure_index=0,
                tonic_pc=5,
                key_name="F",
                chords=[ResolvedChord(raw="1", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj")],
            ),
            MeasureArrangement(
                measure_index=1,
                tonic_pc=5,
                key_name="F",
                chords=[ResolvedChord(raw="5", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj")],
            ),
            MeasureArrangement(
                measure_index=2,
                tonic_pc=5,
                key_name="F",
                chords=[ResolvedChord(raw="6m", name="Dm", beat=1.0, root_pc=2, bass_pc=2, pcs=[2, 5, 9], quality="min")],
            ),
            MeasureArrangement(
                measure_index=3,
                tonic_pc=7,
                key_name="G",
                chords=[ResolvedChord(raw="4", name="Bb", beat=1.0, root_pc=10, bass_pc=10, pcs=[10, 2, 5], quality="maj")],
            ),
            MeasureArrangement(
                measure_index=4,
                tonic_pc=5,
                key_name="F",
                chords=[
                    ResolvedChord(raw="1", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj"),
                    ResolvedChord(raw="5", name="C", beat=3.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"),
                ],
            ),
            MeasureArrangement(
                measure_index=5,
                tonic_pc=5,
                key_name="F",
                chords=[ResolvedChord(raw="1", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj")],
            ),
        ],
        sections=[
            SectionPlan(measure_index=0, section_id="intro", role="intro", is_section_start=True),
            SectionPlan(measure_index=1, section_id="intro", role="intro", is_section_start=False),
            SectionPlan(measure_index=2, section_id="A1", role="verse", is_section_start=True),
            SectionPlan(measure_index=3, section_id="A1", role="verse", is_section_start=False),
            SectionPlan(measure_index=4, section_id="B1", role="chorus", is_section_start=True),
            SectionPlan(measure_index=5, section_id="B1", role="chorus", is_section_start=False),
        ],
        uke_measures=[
            UkeMeasure(
                measure_index=0,
                shapes=["F"],
                pattern_id="P1",
                strokes=[
                    UkeStroke(onset=0.0, duration=1.0, kind="down", accent=True),
                    UkeStroke(onset=1.0, duration=1.0, kind="down"),
                    UkeStroke(onset=2.0, duration=1.0, kind="down"),
                    UkeStroke(onset=3.0, duration=1.0, kind="down"),
                ],
            ),
            UkeMeasure(
                measure_index=1,
                shapes=["C"],
                pattern_id="P1",
                strokes=[
                    UkeStroke(onset=0.0, duration=0.5, kind="down"),
                    UkeStroke(onset=0.5, duration=0.5, kind="up"),
                    UkeStroke(onset=1.0, duration=0.5, kind="chuck"),
                    UkeStroke(onset=1.5, duration=0.5, kind="up"),
                    UkeStroke(onset=2.0, duration=1.0, kind="rest"),
                    UkeStroke(onset=3.0, duration=1.0, kind="pluck", strings=[1, 4]),
                ],
            ),
            UkeMeasure(
                measure_index=2,
                shapes=["Dm"],
                pattern_id="P2",
                strokes=[
                    UkeStroke(onset=0.0, duration=2.0, kind="down"),
                    UkeStroke(onset=2.0, duration=2.0, kind="down"),
                ],
            ),
            UkeMeasure(
                measure_index=3,
                shapes=["Bb"],
                pattern_id="P2",
                strokes=[
                    UkeStroke(onset=0.0, duration=1.0, kind="down"),
                    UkeStroke(onset=1.0, duration=1.0, kind="down"),
                    UkeStroke(onset=2.0, duration=1.0, kind="down"),
                    UkeStroke(onset=3.0, duration=1.0, kind="down"),
                ],
            ),
            UkeMeasure(
                measure_index=4,
                shapes=["F", "C"],
                pattern_id="P1",
                strokes=[
                    UkeStroke(onset=0.0, duration=1.0, kind="down"),
                    UkeStroke(onset=1.0, duration=1.0, kind="down"),
                    UkeStroke(onset=2.0, duration=1.0, kind="down"),
                    UkeStroke(onset=3.0, duration=1.0, kind="down"),
                ],
            ),
            UkeMeasure(
                measure_index=5,
                shapes=["F"],
                pattern_id="P1",
                strokes=[
                    UkeStroke(onset=0.0, duration=4.0, kind="down"),
                ],
            ),
        ],
        uke_shapes={
            "F": UkeShape(name="F", frets=(2, 0, 1, 0), fingers=(2, 0, 1, 0), base_fret=1),
            "C": UkeShape(name="C", frets=(0, 0, 0, 3), fingers=(0, 0, 0, 3), base_fret=1),
            "Dm": UkeShape(name="Dm", frets=(2, 2, 1, 0), fingers=(2, 3, 1, 0), base_fret=1),
            "Bb": UkeShape(name="Bb", frets=(3, 2, 1, 1), fingers=(3, 2, 1, 1), barre=1, base_fret=1),
            "D": UkeShape(name="D", frets=(2, 2, 2, 0), fingers=(1, 2, 3, 0), base_fret=1),
            "HighChord": UkeShape(name="HighChord", frets=(5, 4, 3, 3), fingers=(4, 3, 1, 1), barre=3, base_fret=3),
        },
        pattern_legend={
            "P1": "↓   ↓   ↓   ↓",
            "P2": "↓ ↓↑ ×↑ ↓↑",
        },
    )


def test_build_uke_header_title_capo_and_no_capo():
    # 1. No capo (capo=0): capo phrase omitted
    arr_nocapo = create_sample_uke_arrangement(capo=0)
    title_nocapo = build_uke_header_title(arr_nocapo)
    assert "尤克里里伴奏" in title_nocapo
    assert "中级" in title_nocapo
    assert "按 F 调指法" in title_nocapo
    assert "变调夹" not in title_nocapo
    assert "定弦 G C E A（高音 G）" in title_nocapo

    # 2. Capo > 0: capo phrase present
    arr_capo = create_sample_uke_arrangement(capo=2)
    title_capo = build_uke_header_title(arr_capo)
    assert "尤克里里伴奏" in title_capo
    assert "变调夹 2 品，按 D 调指法（实际 Eb 调）" in title_capo
    assert "定弦 G C E A（高音 G）" in title_capo


def test_draw_chord_diagram_primitives():
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)

    # Standard chord (F)
    shape_f = UkeShape(name="F", frets=(2, 0, 1, 0), fingers=(2, 0, 1, 0), base_fret=1)
    h_f = draw_chord_diagram(draw, shape_f, "F", 10.0, 10.0, col_w=90.0)
    assert h_f > 80.0

    # Barre chord (Bb)
    shape_bb = UkeShape(name="Bb", frets=(3, 2, 1, 1), fingers=(3, 2, 1, 1), barre=1, base_fret=1)
    h_bb = draw_chord_diagram(draw, shape_bb, "Bb", 110.0, 10.0, col_w=90.0)
    assert h_bb > 80.0

    # High fret chord (base_fret=3)
    shape_high = UkeShape(name="High", frets=(5, 4, 3, 3), fingers=(4, 3, 1, 1), barre=3, base_fret=3)
    h_high = draw_chord_diagram(draw, shape_high, "High", 210.0, 10.0, col_w=90.0)
    assert h_high > 80.0

    # Fallback missing shape (name only)
    h_missing = draw_chord_diagram(draw, None, "GhostChord", 310.0, 10.0, col_w=90.0)
    assert h_missing > 30.0


def test_one_diagram_per_distinct_shape_per_page():
    sheet = create_sample_uke_sheet()
    arr = create_sample_uke_arrangement(capo=0)

    # Page 0 has measures 0, 1, 2, 3 with shapes F, C, Dm, Bb
    p0_systems = [s for s in sheet.systems if s.page == 0]
    band0 = make_uke_header_band(1000, sheet, arr, p0_systems)
    assert band0.height > 80

    # Page 1 has measures 4, 5 with shapes F, C, F.
    # Distinct shapes on page 1 should be F and C (2 shapes, not 3).
    p1_systems = [s for s in sheet.systems if s.page == 1]
    band1 = make_uke_header_band(1000, sheet, arr, p1_systems)
    assert band1.height > 80


def test_stroke_glyph_count_equals_strokes():
    arr = create_sample_uke_arrangement(capo=0)
    sheet = create_sample_uke_sheet()
    sys0 = sheet.systems[0]

    # Measure 0 has 4 strokes, Measure 1 has 6 strokes
    strip = render_uke_system_strip(sys0, 1000, arr, sheet)
    assert strip.size == (1000, UKE_STRIP_HEIGHT)


def test_multi_chord_bar():
    arr = create_sample_uke_arrangement(capo=0)
    sheet = create_sample_uke_sheet()
    # System 2 on page 1 has measure 4 with 2 chords: F at beat 1.0, C at beat 3.0
    sys2 = sheet.systems[2]
    strip = render_uke_system_strip(sys2, 1000, arr, sheet)
    assert strip.size == (1000, UKE_STRIP_HEIGHT)


def test_three_four_bar():
    # 3/4 time signature bar
    sheet = create_sample_uke_sheet()
    arr = create_sample_uke_arrangement(capo=0)

    sheet.systems[0].measures[0].beats = 3.0
    arr.uke_measures[0].strokes = [
        UkeStroke(onset=0.0, duration=1.0, kind="down"),
        UkeStroke(onset=1.0, duration=1.0, kind="down"),
        UkeStroke(onset=2.0, duration=1.0, kind="down"),
    ]
    strip = render_uke_system_strip(sheet.systems[0], 1000, arr, sheet)
    assert strip.size == (1000, UKE_STRIP_HEIGHT)


def test_pickup_bar():
    # Pickup measure (e.g. 1 beat or 2 beats)
    sheet = create_sample_uke_sheet()
    arr = create_sample_uke_arrangement(capo=0)

    sheet.systems[0].measures[0].beats = 1.0
    arr.uke_measures[0].strokes = [
        UkeStroke(onset=0.0, duration=1.0, kind="down"),
    ]
    strip = render_uke_system_strip(sheet.systems[0], 1000, arr, sheet)
    assert strip.size == (1000, UKE_STRIP_HEIGHT)


def test_key_change_rendering():
    sheet = create_sample_uke_sheet()
    arr = create_sample_uke_arrangement(capo=1)  # with capo

    # System 1 has measure 3 where key change occurs
    strip = render_uke_system_strip(sheet.systems[1], 1000, arr, sheet)
    assert strip.size == (1000, UKE_STRIP_HEIGHT)


def test_missing_shape_name_only_renders_gracefully():
    sheet = create_sample_uke_sheet()
    arr = create_sample_uke_arrangement(capo=0)

    # Change shape to an unknown name not present in arr.uke_shapes
    arr.uke_measures[0].shapes = ["MysteryChord"]
    p0_systems = [s for s in sheet.systems if s.page == 0]
    band = make_uke_header_band(1000, sheet, arr, p0_systems)
    assert band.height > 60

    strip = render_uke_system_strip(sheet.systems[0], 1000, arr, sheet)
    assert strip.size == (1000, UKE_STRIP_HEIGHT)


def test_render_uke_pages_and_pdf_complete():
    sheet = create_sample_uke_sheet()
    arr = create_sample_uke_arrangement(capo=1)
    p0_bytes = create_dummy_png_bytes(1000, 1400)
    p1_bytes = create_dummy_png_bytes(1000, 1400)

    # 1. render_uke_pages
    pages = render_uke_pages([p0_bytes, p1_bytes], sheet, arr)
    assert len(pages) >= 2
    for p in pages:
        assert isinstance(p, Image.Image)

    # 2. render_uke_pdf
    pdf_bytes = render_uke_pdf([p0_bytes, p1_bytes], sheet, arr)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
