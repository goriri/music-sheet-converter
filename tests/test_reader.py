"""Unit tests for OMR crop-based content reader (app/omr/reader.py).

Tests run offline with monkeypatched model calls, verifying:
- Mapping by box and measure number
- Key-change semitone computation
- Beat rescaling for 3/4 and 12/8 meters
- Letter-notation headers
- Fallback path when layout analysis is missing or low-confidence
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from app.models import ParsedSheet, SongHeader
from app.omr.geometry import GChordBox, GMeasure, GSystem, PageGeometry
from app.omr.reader import (
    compute_key_change_semitones,
    parse_time_signature_to_beats,
    read_sheet,
    resolve_chord_beat,
)
from app.omr.reader_prompts import (
    BoxReading,
    ExtraChordReading,
    HeaderReading,
    KeyChangeReading,
    MeasureContentReading,
    SystemReading,
)


# ---------------------------------------------------------------------------
# 1. Key-change semitone computation tests
# ---------------------------------------------------------------------------

def test_key_change_semitone_computation():
    # Taiwanese scale degrees: '轉N調' / '轉成N調'
    assert compute_key_change_semitones("轉成2調(Ab)") == 2
    assert compute_key_change_semitones("轉2調") == 2
    assert compute_key_change_semitones("轉3調") == 4
    assert compute_key_change_semitones("轉4調") == 5
    assert compute_key_change_semitones("轉5調") == 7
    assert compute_key_change_semitones("轉6調") == 9
    assert compute_key_change_semitones("轉b7調") == -2
    assert compute_key_change_semitones("轉7b調") == -2
    assert compute_key_change_semitones("轉1#調") == 1
    assert compute_key_change_semitones("轉#1調") == 1
    assert compute_key_change_semitones("轉b2調") == 1
    assert compute_key_change_semitones("轉b3調") == 3

    # '1=X' relative to previous key
    assert compute_key_change_semitones("轉 1=B", prev_key="C") == -1
    assert compute_key_change_semitones("1=A", prev_key="C") == -3
    assert compute_key_change_semitones("1=D", prev_key="C") == 2
    assert compute_key_change_semitones("1=G", prev_key="C") == -5

    # Key transition pair: '(Old - New)'
    assert compute_key_change_semitones("(F# - Ab)") == 2
    assert compute_key_change_semitones("F# -> Ab") == 2
    assert compute_key_change_semitones("Bb - C") == 2

    # Empty or unrecognized
    assert compute_key_change_semitones("") == 0
    assert compute_key_change_semitones("no key change") == 0


# ---------------------------------------------------------------------------
# 2. Beat rescaling tests for 3/4 and 12/8 meters
# ---------------------------------------------------------------------------

def test_beat_rescaling_and_time_signatures():
    # Time signature beat parsing
    assert parse_time_signature_to_beats("4/4") == 4.0
    assert parse_time_signature_to_beats("3/4") == 3.0
    assert parse_time_signature_to_beats("2/4") == 2.0
    assert parse_time_signature_to_beats("6/8") == 6.0
    assert parse_time_signature_to_beats("12/8") == 12.0

    # 4/4 meter (standard: uses beat_geo unless clearly different)
    assert resolve_chord_beat(beat_geo=1.0, model_beat=None, time_sig="4/4", measure_beats=4.0) == 1.0
    assert resolve_chord_beat(beat_geo=3.0, model_beat=None, time_sig="4/4", measure_beats=4.0) == 3.0
    assert resolve_chord_beat(beat_geo=1.0, model_beat=1.2, time_sig="4/4", measure_beats=4.0) == 1.0
    # Clearly different model beat (difference >= 1.0)
    assert resolve_chord_beat(beat_geo=1.0, model_beat=3.0, time_sig="4/4", measure_beats=4.0) == 3.0

    # 3/4 meter: beat_geo rescaled to 3.0 beats
    # beat_geo=1.0 (frac=0.0) -> 1.0 + 0.0 * 3.0 = 1.0
    b3_start = resolve_chord_beat(beat_geo=1.0, model_beat=None, time_sig="3/4", measure_beats=3.0)
    assert b3_start == 1.0

    # beat_geo=3.0 (frac=0.5 in 4/4) -> 1.0 + 0.5 * 3.0 = 2.5
    b3_mid = resolve_chord_beat(beat_geo=3.0, model_beat=None, time_sig="3/4", measure_beats=3.0)
    assert b3_mid == 2.5

    # 12/8 meter: beat_geo rescaled to 12.0 beats
    # beat_geo=1.0 (frac=0.0) -> 1.0
    b12_start = resolve_chord_beat(beat_geo=1.0, model_beat=None, time_sig="12/8", measure_beats=12.0)
    assert b12_start == 1.0

    # beat_geo=3.0 (frac=0.5 in 4/4) -> 1.0 + 0.5 * 12.0 = 7.0 (start of beat 3 dotted-quarter)
    b12_mid = resolve_chord_beat(beat_geo=3.0, model_beat=None, time_sig="12/8", measure_beats=12.0)
    assert b12_mid == 7.0

    # Model beat override for 3/4 when clearly different
    b3_override = resolve_chord_beat(beat_geo=1.0, model_beat=2.0, time_sig="3/4", measure_beats=3.0)
    assert b3_override == 2.0


# ---------------------------------------------------------------------------
# Helper: Mock LLM reader dispatch at module scope
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: dict[type, Any] = {}


def mock_ask_reader_dispatch(prompt: str, schema: type, images: list[bytes], **kwargs: Any) -> Any:
    """Module-level dispatch for mock ask_reader calls."""
    if schema in _MOCK_RESPONSES:
        return _MOCK_RESPONSES[schema]
    raise ValueError(f"Unexpected schema in mock_ask_reader_dispatch: {schema}")


def _create_test_image(width: int = 1000, height: int = 1400) -> bytes:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. Mapping by box and measure number
# ---------------------------------------------------------------------------

def test_read_sheet_mapping_by_box_and_measure_number():
    img_bytes = _create_test_image(1000, 1400)

    # Construct test geometry: 1 page, 1 system with 2 measures, 2 chord boxes
    measures = [
        GMeasure(index_in_system=0, x0=0.05, x1=0.45),
        GMeasure(index_in_system=1, x0=0.45, x1=0.90),
    ]
    chord_boxes = [
        GChordBox(bbox=(0.10, 0.12, 0.18, 0.15), measure_index_in_system=0, beat_geo=1.0, boxed=True),
        GChordBox(bbox=(0.50, 0.12, 0.58, 0.15), measure_index_in_system=1, beat_geo=1.0, boxed=True),
    ]
    system = GSystem(
        page=0,
        index_on_page=0,
        bbox=(0.05, 0.12, 0.90, 0.20),
        melody_band=(0.15, 0.18),
        measures=measures,
        chord_boxes=chord_boxes,
        confidence=0.95,
    )
    geom = PageGeometry(
        page=0,
        width=1000,
        height=1400,
        header_band=(0.0, 0.12),
        systems=[system],
        confidence=0.95,
    )

    mock_header = HeaderReading(
        title="掉了",
        style="Slow Soul",
        time_signature="4/4",
        tempo_bpm=81.0,
        original_key="F#",
        male_key="Bb",
        female_key="F",
        chord_notation="number",
        raw="Slow Soul 4/4 (F# - Ab)",
    )

    mock_system = SystemReading(
        section_label="Only PN(RHY)",
        chord_boxes=[
            BoxReading(box_id=1, text="1(2)"),
            BoxReading(box_id=2, text="5/7"),
        ],
        extra_chords=[
            ExtraChordReading(measure_index=1, text="2m7", beat=3.0),
        ],
        measures=[
            MeasureContentReading(measure_index=0, melody="2 2 23 21", lyrics="心疼的玫瑰"),
            MeasureContentReading(measure_index=1, melody="2 2 23 21", lyrics="半夜還開著", fill=True),
        ],
        key_change=KeyChangeReading(measure_index=1, raw="轉成2調(Ab)"),
    )

    _MOCK_RESPONSES[HeaderReading] = mock_header
    _MOCK_RESPONSES[SystemReading] = mock_system

    with patch("app.omr.reader.ask_reader", side_effect=mock_ask_reader_dispatch):
        sheet = read_sheet([img_bytes], [geom], model="gemini-2.5-pro")

    assert isinstance(sheet, ParsedSheet)
    assert sheet.header.title == "掉了"
    assert sheet.header.style == "Slow Soul"
    assert sheet.header.original_key == "F#"
    assert sheet.layout_confidence == 0.95

    assert len(sheet.systems) == 1
    sys0 = sheet.systems[0]
    assert sys0.section_label == "Only PN(RHY)"
    assert sys0.bbox == (0.05, 0.12, 0.90, 0.20)
    assert len(sys0.measures) == 2

    # Measure 0: chord box 1 mapped to '1(2)'
    m0 = sys0.measures[0]
    assert m0.index == 0
    assert m0.bbox == (0.05, 0.12, 0.45, 0.20)
    assert m0.melody == "2 2 23 21"
    assert m0.lyrics == "心疼的玫瑰"
    assert len(m0.chords) == 1
    assert m0.chords[0].raw == "1(2)"
    assert m0.chords[0].beat == 1.0
    assert m0.chords[0].bbox == (0.10, 0.12, 0.18, 0.15)

    # Measure 1: chord box 2 mapped to '5/7', extra chord '2m7' at beat 3.0
    m1 = sys0.measures[1]
    assert m1.index == 1
    assert m1.bbox == (0.45, 0.12, 0.90, 0.20)
    assert m1.fill is True
    assert len(m1.chords) == 2
    assert m1.chords[0].raw == "5/7"
    assert m1.chords[0].beat == 1.0
    assert m1.chords[1].raw == "2m7"
    assert m1.chords[1].beat == 3.0

    # Key change at measure 1
    assert len(sheet.key_changes) == 1
    kc = sheet.key_changes[0]
    assert kc.at_measure == 1
    assert kc.raw == "轉成2調(Ab)"
    assert kc.semitones == 2


# ---------------------------------------------------------------------------
# 4. Letter-notation header and chord handling
# ---------------------------------------------------------------------------

def test_read_sheet_letter_notation_header():
    img_bytes = _create_test_image(1000, 1400)

    measures = [GMeasure(index_in_system=0, x0=0.05, x1=0.50)]
    chord_boxes = [
        GChordBox(bbox=(0.10, 0.12, 0.18, 0.15), measure_index_in_system=0, beat_geo=1.0, boxed=False)
    ]
    system = GSystem(
        page=0,
        index_on_page=0,
        bbox=(0.05, 0.12, 0.90, 0.20),
        melody_band=(0.15, 0.18),
        measures=measures,
        chord_boxes=chord_boxes,
        confidence=0.9,
    )
    geom = PageGeometry(
        page=0,
        width=1000,
        height=1400,
        header_band=(0.0, 0.10),
        systems=[system],
        confidence=0.9,
    )

    mock_header = HeaderReading(
        title="小情歌",
        style="Acoustic Pop",
        time_signature="4/4",
        tempo_bpm=92.0,
        original_key="C",
        chord_notation="letter",
        raw="1=C 4/4 ♩=92",
    )

    mock_system = SystemReading(
        chord_boxes=[BoxReading(box_id=1, text="Cadd9")],
        measures=[MeasureContentReading(measure_index=0, melody="1 2 3 5")],
    )

    _MOCK_RESPONSES[HeaderReading] = mock_header
    _MOCK_RESPONSES[SystemReading] = mock_system

    with patch("app.omr.reader.ask_reader", side_effect=mock_ask_reader_dispatch):
        sheet = read_sheet([img_bytes], [geom], model="gemini-2.5-pro")

    assert sheet.header.chord_notation == "letter"
    assert sheet.header.original_key == "C"
    assert sheet.systems[0].measures[0].chords[0].raw == "Cadd9"


# ---------------------------------------------------------------------------
# 5. Fallback path when layout missing or low confidence
# ---------------------------------------------------------------------------

def test_gemini_omr_fallback_when_layout_missing(monkeypatch):
    from app.omr import gemini_omr

    img_bytes = _create_test_image(500, 700)

    # Mock parse_single_page to return a synthetic page result
    mock_single = MagicMock()
    mock_single.header = MagicMock(
        title="Fallback Song",
        style="Ballad",
        time_signature="4/4",
        tempo_bpm=80.0,
        original_key="C",
        male_key=None,
        female_key=None,
        raw="Ballad 4/4",
    )
    mock_single.systems = []
    mock_single.key_changes = []
    mock_single.warnings = []

    with patch("app.omr.gemini_omr.parse_single_page", return_value=mock_single):
        with patch("app.omr.gemini_omr.refine_measure_boxes"):
            sheet = gemini_omr.parse_pages([img_bytes])

    # Since app.omr.layout does not exist yet, it should fall back gracefully
    assert isinstance(sheet, ParsedSheet)
    assert sheet.header.title == "Fallback Song"
    assert sheet.layout_confidence < 0.6
    assert any("fell back" in w.lower() for w in sheet.warnings)


def test_gemini_omr_fallback_when_layout_low_confidence(monkeypatch):
    from app.omr import gemini_omr

    img_bytes = _create_test_image(500, 700)

    # Create a mock layout module that returns low confidence
    mock_layout = MagicMock()
    mock_geom = PageGeometry(
        page=0,
        width=500,
        height=700,
        confidence=0.4,  # Below 0.6 threshold
        systems=[GSystem(
            page=0,
            index_on_page=0,
            bbox=(0.05, 0.1, 0.95, 0.2),
            melody_band=(0.12, 0.18),
            confidence=0.4,
        )],
    )
    mock_layout.analyze_page.return_value = mock_geom

    mock_single = MagicMock()
    mock_single.header = MagicMock(
        title="Low Conf Song",
        style="Ballad",
        time_signature="4/4",
        tempo_bpm=80.0,
        original_key="C",
        male_key=None,
        female_key=None,
        raw="Ballad 4/4",
    )
    mock_single.systems = []
    mock_single.key_changes = []
    mock_single.warnings = []

    with patch.dict("sys.modules", {"app.omr.layout": mock_layout}):
        with patch("app.omr.gemini_omr.parse_single_page", return_value=mock_single):
            with patch("app.omr.gemini_omr.refine_measure_boxes"):
                sheet = gemini_omr.parse_pages([img_bytes])

    assert sheet.layout_confidence == 0.4
    assert any("fell back" in w.lower() for w in sheet.warnings)


def test_gemini_omr_uses_v2_when_layout_high_confidence(monkeypatch):
    from app.omr import gemini_omr

    img_bytes = _create_test_image(500, 700)

    mock_layout = MagicMock()
    mock_geom = PageGeometry(
        page=0,
        width=500,
        height=700,
        confidence=0.9,  # >= 0.6 threshold
        systems=[GSystem(
            page=0,
            index_on_page=0,
            bbox=(0.05, 0.1, 0.95, 0.2),
            melody_band=(0.12, 0.18),
            confidence=0.9,
        )],
    )
    mock_layout.analyze_page.return_value = mock_geom

    mock_sheet = ParsedSheet(
        header=SongHeader(title="V2 Success"),
        pages=[],
        systems=[],
        layout_confidence=0.9,
    )

    with patch.dict("sys.modules", {"app.omr.layout": mock_layout}):
        with patch("app.omr.reader.read_sheet", return_value=mock_sheet) as mock_read:
            sheet = gemini_omr.parse_pages([img_bytes])

    mock_read.assert_called_once()
    assert sheet.header.title == "V2 Success"
    assert sheet.layout_confidence == 0.9
