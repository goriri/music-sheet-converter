"""Tests for chord beat quantization and partial measure inference."""
from __future__ import annotations

import pytest

from app.models import ChordSymbol, Measure, PageInfo, ParsedSheet, QualityIssue, SongHeader, System
from app.omr.beats import (
    estimate_typeset_offsets,
    infer_partial_measure_beats,
    quantize_chord_beats,
    snap_to_grid,
)


def test_snap_to_grid():
    """Verify 0.5-beat grid snapping with whole-beat preference tolerance."""
    # Exact whole beats
    assert snap_to_grid(0.0) == 0.0
    assert snap_to_grid(1.0) == 1.0
    assert snap_to_grid(2.0) == 2.0

    # Values within 0.30 whole-beat tolerance snap to whole beat
    assert snap_to_grid(0.15) == 0.0
    assert snap_to_grid(0.28) == 0.0
    assert snap_to_grid(-0.25) == 0.0
    assert snap_to_grid(0.72) == 1.0
    assert snap_to_grid(1.25) == 1.0

    # Values outside tolerance snap to half beat (0.5)
    assert snap_to_grid(0.40) == 0.5
    assert snap_to_grid(0.50) == 0.5
    assert snap_to_grid(0.60) == 0.5
    assert snap_to_grid(1.45) == 1.5
    assert snap_to_grid(1.55) == 1.5


def test_synthetic_measures_offset_typesetting():
    """Verify that ~0.45 beat indented chord symbols snap to beats 1.0 and 3.0."""
    # System with 4 measures, 4 beats each.
    # Measure width = 0.20 (x from 0.10 to 0.30, etc.)
    # In 4 beats, 1 beat corresponds to 0.20 / 4 = 0.05 width.
    # A typesetting offset of 0.45 beats is 0.45 * 0.05 = 0.0225 width.
    measures = []
    for i in range(4):
        x0 = 0.10 + i * 0.20
        x1 = x0 + 0.20
        # First chord at offset 0.45 beats (beat 1 intended)
        c0_center = x0 + 0.45 * 0.05
        # Second chord at offset 2.45 beats (beat 3 intended)
        c1_center = x0 + 2.45 * 0.05

        c0 = ChordSymbol(
            raw="1",
            beat=1.5,  # Naive geometric assignment before quantization
            bbox=(c0_center - 0.01, 0.10, c0_center + 0.01, 0.15),
        )
        c1 = ChordSymbol(
            raw="5",
            beat=3.5,  # Naive geometric assignment before quantization
            bbox=(c1_center - 0.01, 0.10, c1_center + 0.01, 0.15),
        )
        measures.append(
            Measure(
                index=i,
                bbox=(x0, 0.10, x1, 0.20),
                beats=4.0,
                chords=[c0, c1],
            )
        )

    sys0 = System(page=0, bbox=(0.10, 0.10, 0.90, 0.20), measures=measures)
    sheet = ParsedSheet(
        header=SongHeader(title="Offset Test", time_signature="4/4"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys0],
    )

    # Offset estimation should find ~0.45
    deltas = estimate_typeset_offsets(sheet)
    assert 0.40 <= deltas[0] <= 0.50

    quantized = quantize_chord_beats(sheet)
    for m in quantized.measures():
        assert m.chords[0].beat == 1.0, f"Measure {m.index} chord 0 should snap to 1.0"
        assert m.chords[1].beat == 3.0, f"Measure {m.index} chord 1 should snap to 3.0"


def test_half_beat_anticipation_with_melody():
    """Verify that a chord on beat 2.5 stays on 2.5 if melody has an onset at 2.5."""
    # Measure width 0.40, beats=4.0 -> 0.10 per beat.
    # Offset = 0.50 beats -> 0.050 width.
    x0, x1 = 0.10, 0.50
    # Chord 1 at beat 1.0 (center = x0 + (0 + 0.50) * 0.10 = 0.15)
    c0 = ChordSymbol(raw="1", bbox=(0.14, 0.10, 0.16, 0.15))
    # Chord 2 at beat 2.5 (center = x0 + (1.5 + 0.50) * 0.10 = 0.30)
    c1 = ChordSymbol(raw="4", bbox=(0.29, 0.10, 0.31, 0.15))

    # Melody '1. 5 3 4' -> '1.' has 1.5 beats, '5' has onset at 2.5
    m0 = Measure(
        index=0,
        bbox=(x0, 0.10, x1, 0.20),
        beats=4.0,
        melody="1. 5 3 4",
        chords=[c0, c1],
    )

    sys0 = System(page=0, bbox=(0.10, 0.10, 0.50, 0.20), measures=[m0])
    sheet = ParsedSheet(
        header=SongHeader(title="Anticipation Test", time_signature="4/4"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys0],
    )

    quantized = quantize_chord_beats(sheet)
    res_m = quantized.measures()[0]
    assert res_m.chords[0].beat == 1.0
    assert res_m.chords[1].beat == 2.5


def test_meters_3_4_and_6_8():
    """Verify quantization works correctly on 3/4 and 6/8 meters."""
    # 3/4 meter: 2 measures of width 0.30 each, beats 3.0 (0.10 per beat)
    # Chord 1 at offset 0.5 beats -> beat 1.0
    # Chord 2 at offset 2.5 beats -> beat 3.0
    c0 = ChordSymbol(raw="1", bbox=(0.14, 0.10, 0.16, 0.15))
    c1 = ChordSymbol(raw="5", bbox=(0.34, 0.10, 0.36, 0.15))
    m_34_0 = Measure(
        index=0,
        bbox=(0.10, 0.10, 0.40, 0.20),
        beats=3.0,
        chords=[c0, c1],
    )
    m_34_1 = Measure(
        index=1,
        bbox=(0.40, 0.10, 0.70, 0.20),
        beats=3.0,
        chords=[ChordSymbol(raw="4", bbox=(0.44, 0.10, 0.46, 0.15))],
    )

    sheet_34 = ParsedSheet(
        header=SongHeader(title="3/4 Meter Test", time_signature="3/4"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[System(page=0, bbox=(0.10, 0.10, 0.70, 0.20), measures=[m_34_0, m_34_1])],
    )
    quantized_34 = quantize_chord_beats(sheet_34)
    meas_34 = quantized_34.measures()
    assert meas_34[0].chords[0].beat == 1.0
    assert meas_34[0].chords[1].beat == 3.0

    # 6/8 meter: 2 measures of width 0.60 each, beats 6.0 (0.10 per beat)
    # Chord 1 at offset 0.5 beats -> beat 1.0
    # Chord 2 at offset 3.5 beats -> beat 4.0
    c2 = ChordSymbol(raw="1", bbox=(0.14, 0.10, 0.16, 0.15))
    c3 = ChordSymbol(raw="4", bbox=(0.44, 0.10, 0.46, 0.15))
    m_68_0 = Measure(
        index=0,
        bbox=(0.10, 0.10, 0.70, 0.20),
        beats=6.0,
        chords=[c2, c3],
    )
    m_68_1 = Measure(
        index=1,
        bbox=(0.10, 0.25, 0.70, 0.35),
        beats=6.0,
        chords=[ChordSymbol(raw="5", bbox=(0.14, 0.25, 0.16, 0.30))],
    )
    sheet_68 = ParsedSheet(
        header=SongHeader(title="6/8 Meter Test", time_signature="6/8"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[
            System(page=0, bbox=(0.10, 0.10, 0.70, 0.20), measures=[m_68_0]),
            System(page=0, bbox=(0.10, 0.25, 0.70, 0.35), measures=[m_68_1]),
        ],
    )
    quantized_68 = quantize_chord_beats(sheet_68)
    meas_68 = quantized_68.measures()
    assert meas_68[0].chords[0].beat == 1.0
    assert meas_68[0].chords[1].beat == 4.0


def test_partial_measure_inference_pickup():
    """Verify pickup measure at row start is detected and its beats inferred."""
    # System 0:
    # m0 is pickup bar with narrow width 0.06 (ratio 0.06 / 0.24 = 0.25 <= 0.60)
    # m1, m2, m3 are normal width 0.24
    m0 = Measure(
        index=0,
        bbox=(0.04, 0.10, 0.10, 0.20),
        beats=4.0,
        melody="5",  # 1-beat melody
        chords=[ChordSymbol(raw="1", beat=1.0, bbox=(0.06, 0.10, 0.08, 0.15))],
    )
    m1 = Measure(index=1, bbox=(0.10, 0.10, 0.34, 0.20), beats=4.0, melody="1 2 3 4")
    m2 = Measure(index=2, bbox=(0.34, 0.10, 0.58, 0.20), beats=4.0, melody="1 2 3 4")
    m3 = Measure(index=3, bbox=(0.58, 0.10, 0.82, 0.20), beats=4.0, melody="1 2 3 4")

    sys0 = System(page=0, bbox=(0.04, 0.10, 0.82, 0.20), measures=[m0, m1, m2, m3])
    sheet = ParsedSheet(
        header=SongHeader(title="Pickup Test"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys0],
    )

    quantized = quantize_chord_beats(sheet)
    meas = quantized.measures()
    assert meas[0].beats == 1.0
    assert meas[0].chords[0].beat == 1.0

    # Ensure QualityIssue was recorded
    issues = [iss for iss in quantized.issues if iss.code == "partial_measure_inferred"]
    assert len(issues) == 1
    assert issues[0].measure_index == 0
    assert issues[0].detail["inferred_beats"] == 1.0


def test_full_melody_last_bar_not_shortened():
    """Verify that a full-melody measure at the end of a row is not shortened even if narrow."""
    # m0, m1 are normal width 0.35, m2 is width 0.18 (ratio 0.18 / 0.35 = 0.51 <= 0.60)
    # But m2 has full 4-beat melody: '5 - - -'
    m0 = Measure(index=0, bbox=(0.05, 0.10, 0.40, 0.20), beats=4.0, melody="1 2 3 4")
    m1 = Measure(index=1, bbox=(0.40, 0.10, 0.75, 0.20), beats=4.0, melody="1 2 3 4")
    m2 = Measure(index=2, bbox=(0.75, 0.10, 0.93, 0.20), beats=4.0, melody="5 - - -")

    sys0 = System(page=0, bbox=(0.05, 0.10, 0.93, 0.20), measures=[m0, m1, m2])
    sheet = ParsedSheet(
        header=SongHeader(title="Full Melody Last Bar"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys0],
    )

    quantized = quantize_chord_beats(sheet)
    meas = quantized.measures()
    assert meas[2].beats == 4.0


def test_monotonic_chord_ordering():
    """Verify that chords inside a measure are strictly increasing in beat order."""
    # Two chords with bounding boxes very close together that might both map to beat 1.0
    c0 = ChordSymbol(raw="1", bbox=(0.14, 0.10, 0.16, 0.15))
    c1 = ChordSymbol(raw="5", bbox=(0.15, 0.10, 0.17, 0.15))
    m0 = Measure(
        index=0,
        bbox=(0.10, 0.10, 0.50, 0.20),
        beats=4.0,
        chords=[c0, c1],
    )

    sys0 = System(page=0, bbox=(0.10, 0.10, 0.50, 0.20), measures=[m0])
    sheet = ParsedSheet(
        header=SongHeader(title="Monotonic Test"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys0],
    )

    quantized = quantize_chord_beats(sheet)
    res_chords = quantized.measures()[0].chords
    assert len(res_chords) == 2
    assert res_chords[0].beat < res_chords[1].beat
    assert res_chords[0].raw == "1"
    assert res_chords[1].raw == "5"


def test_robustness_never_raises():
    """Verify that quantize_chord_beats handles missing bboxes, empty chords, and degenerate data safely."""
    # Degenerate measures
    m0 = Measure(index=0, bbox=(0.0, 0.0, 0.0, 0.0), beats=4.0, chords=[])
    m1 = Measure(
        index=1,
        bbox=(0.10, 0.10, 0.50, 0.20),
        beats=4.0,
        melody="invalid!@#$%melody",
        chords=[ChordSymbol(raw="1", bbox=None, confidence=0.8, stacked=True)],
    )
    m2 = Measure(
        index=2,
        bbox=(0.50, 0.10, 0.20, 0.20),  # Inverted bbox
        beats=4.0,
        chords=[ChordSymbol(raw="2m", bbox=(0.3, 0.1, 0.4, 0.2))],
    )

    sys0 = System(page=0, bbox=(0.0, 0.0, 1.0, 1.0), measures=[m0, m1, m2])
    sheet = ParsedSheet(
        header=SongHeader(title="Robustness Test"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[sys0],
    )

    # Must never raise an exception
    quantized = quantize_chord_beats(sheet)
    assert len(quantized.measures()) == 3
    # Check invariant: raw, confidence, stacked preserved
    c_m1 = quantized.measures()[1].chords[0]
    assert c_m1.raw == "1"
    assert c_m1.confidence == 0.8
    assert c_m1.stacked is True


def test_quantize_chord_beats_idempotent():
    """Verify that running quantization multiple times produces identical beats."""
    c0 = ChordSymbol(raw="1", bbox=(0.14, 0.10, 0.16, 0.15))
    c1 = ChordSymbol(raw="5", bbox=(0.34, 0.10, 0.36, 0.15))
    m0 = Measure(index=0, bbox=(0.10, 0.10, 0.50, 0.20), beats=4.0, chords=[c0, c1])
    sheet = ParsedSheet(
        header=SongHeader(title="Idempotence Test"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[System(page=0, bbox=(0.10, 0.10, 0.50, 0.20), measures=[m0])],
    )

    q1 = quantize_chord_beats(sheet)
    beats_q1 = [c.beat for c in q1.measures()[0].chords]

    q2 = quantize_chord_beats(q1)
    beats_q2 = [c.beat for c in q2.measures()[0].chords]

    assert beats_q1 == beats_q2


def test_melody_onset_snaps_when_no_half_beat_onset():
    """Verify that a half-beat candidate chord snaps to nearest melody note when no half-beat onset exists."""
    # Measure with melody '1 2 3 4' (whole beat notes only)
    # Chord at continuous beat 2.4 (raw offset snaps to 2.5 without melody)
    # With melody, since 2.5 has no onset, it should snap to nearest onset (2.0 or 3.0)
    # Let's verify continuous_beat = pos - delta + 1.0
    # Measure width 0.40, beats 4.0 (0.10 per beat). Delta = 0.50 (0.05 width).
    # If cx = 0.10 + 0.19 -> frac = 0.19 / 0.40 = 0.475 -> pos = 1.90.
    # pos - delta = 1.40 -> raw snapped to 1.5 -> beat = 2.5.
    # continuous_beat = 1.90 - 0.50 + 1.0 = 2.40. Nearest melody onset in [1, 2, 3, 4] is 2.0.
    c0 = ChordSymbol(raw="1", bbox=(0.14, 0.10, 0.16, 0.15))
    c1 = ChordSymbol(raw="4", bbox=(0.285, 0.10, 0.295, 0.15))  # cx = 0.29
    m0 = Measure(
        index=0,
        bbox=(0.10, 0.10, 0.50, 0.20),
        beats=4.0,
        melody="1 2 3 4",
        chords=[c0, c1],
    )
    sheet = ParsedSheet(
        header=SongHeader(title="Melody Snapping Test"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[System(page=0, bbox=(0.10, 0.10, 0.50, 0.20), measures=[m0])],
    )
    quantized = quantize_chord_beats(sheet)
    meas = quantized.measures()[0]
    assert meas.chords[0].beat == 1.0
    assert meas.chords[1].beat in (2.0, 3.0)

