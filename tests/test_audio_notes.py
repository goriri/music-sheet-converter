"""Unit tests for app.audio.notes (melody note segmentation)."""
from __future__ import annotations

import numpy as np
import pytest

from app.audio.analysis import NoteSeg
from app.audio.notes import segment_notes


def test_vibrato_note_stays_one_note():
    """Verify that ~5.5 Hz vibrato (±0.8 semitones) does not fragment a sustained note."""
    hop = 0.01
    duration = 1.2
    t = np.arange(0.0, duration, hop)
    f0_base = 440.0 * (2.0 ** ((60.0 - 69.0) / 12.0))  # Middle C (C4, MIDI 60)

    # Vibrato oscillation: 5.5 Hz with amplitude 0.8 semitones
    vib_semi = 0.8 * np.sin(2.0 * np.pi * 5.5 * t)
    pitch_hz = f0_base * (2.0 ** (vib_semi / 12.0))
    periodicity = np.full_like(t, 0.85)
    vocal_rms = np.full_like(t, 0.08)

    notes = segment_notes(t, pitch_hz, periodicity, vocal_rms)

    assert len(notes) == 1, f"Expected 1 sustained note, got {len(notes)}: {notes}"
    assert abs(notes[0].midi - 60.0) <= 0.25
    assert notes[0].start == pytest.approx(0.0, abs=0.03)
    assert notes[0].end == pytest.approx(duration, abs=0.03)


def test_portamento_between_two_notes_gives_two_notes():
    """Verify that a monotonic pitch glide (portamento) between two notes splits into 2 notes."""
    hop = 0.01
    t = np.arange(0.0, 1.2, hop)
    semi = np.zeros_like(t)

    # 0.0 -> 0.5s: note A (MIDI 60)
    # 0.5 -> 0.6s: portamento glide 60 -> 67 (duration 0.10s)
    # 0.6 -> 1.2s: note B (MIDI 67)
    for i, cur_t in enumerate(t):
        if cur_t < 0.5:
            semi[i] = 60.0
        elif cur_t < 0.6:
            semi[i] = 60.0 + ((cur_t - 0.5) / 0.10) * 7.0
        else:
            semi[i] = 67.0

    pitch_hz = 440.0 * (2.0 ** ((semi - 69.0) / 12.0))
    periodicity = np.full_like(t, 0.9)
    vocal_rms = np.full_like(t, 0.08)

    notes = segment_notes(t, pitch_hz, periodicity, vocal_rms)

    assert len(notes) == 2, f"Expected 2 notes, got {len(notes)}: {notes}"
    assert abs(notes[0].midi - 60.0) <= 0.35
    assert abs(notes[1].midi - 67.0) <= 0.35
    assert notes[0].start == pytest.approx(0.0, abs=0.03)
    assert notes[1].end == pytest.approx(1.2, abs=0.03)


def test_repeated_same_pitch_notes_with_rms_dip():
    """Verify that repeated same-pitch notes separated by an RMS dip produce two notes."""
    hop = 0.01
    t = np.arange(0.0, 1.0, hop)
    f0_base = 440.0 * (2.0 ** ((62.0 - 69.0) / 12.0))  # D4
    pitch_hz = np.full_like(t, f0_base)
    periodicity = np.full_like(t, 0.85)

    # Note 1: 0.0 -> 0.45s (rms 0.06)
    # Energy dip (consonant/breath): 0.46 -> 0.52s (rms 0.005, < 0.35 * peak)
    # Note 2: 0.53 -> 1.0s (rms 0.06)
    vocal_rms = np.full_like(t, 0.06)
    for i, cur_t in enumerate(t):
        if 0.46 <= cur_t <= 0.52:
            vocal_rms[i] = 0.005

    notes = segment_notes(t, pitch_hz, periodicity, vocal_rms)

    assert len(notes) == 2, f"Expected 2 notes separated by RMS dip, got {len(notes)}: {notes}"
    assert abs(notes[0].midi - 62.0) <= 0.2
    assert abs(notes[1].midi - 62.0) <= 0.2
    assert notes[0].end <= 0.53
    assert notes[1].start >= 0.45


def test_short_blips_removed():
    """Verify that short isolated voiced blips (< 0.08s) are eliminated."""
    hop = 0.01
    # 0.05s blip (5 frames)
    t = np.arange(0.0, 0.05, hop)
    pitch_hz = np.full_like(t, 440.0)
    periodicity = np.full_like(t, 0.8)
    vocal_rms = np.full_like(t, 0.05)

    notes = segment_notes(t, pitch_hz, periodicity, vocal_rms)
    assert len(notes) == 0, f"Expected 0 notes for 0.05s blip, got {len(notes)}"


def test_never_raises_on_empty_nan_inf():
    """Verify robustness: empty lists, NaNs, Infs, None return gracefully without raising."""
    # Empty inputs
    assert segment_notes([], [], [], []) == []
    assert segment_notes(None, None, None, None) == []

    # NaNs and Infs
    t = [np.nan, 0.01, 0.02, np.inf]
    p = [np.nan, 440.0, np.inf, 440.0]
    c = [0.0, np.nan, 0.8, np.inf]
    r = [np.nan, 0.05, np.nan, 0.05]

    res = segment_notes(t, p, c, r)
    assert isinstance(res, list)

    # Extreme length mismatch
    res_mismatch = segment_notes([0.0, 0.01], [440.0], [0.8, 0.8, 0.8], [0.05])
    assert isinstance(res_mismatch, list)


def test_baseline_mode_reproducibility():
    """Verify that mode='baseline' runs cleanly and returns NoteSeg objects."""
    hop = 0.01
    t = np.arange(0.0, 0.5, hop)
    pitch_hz = np.full_like(t, 440.0)
    periodicity = np.full_like(t, 0.8)
    vocal_rms = np.full_like(t, 0.05)

    notes = segment_notes(t, pitch_hz, periodicity, vocal_rms, params={"mode": "baseline"})
    assert isinstance(notes, list)
    assert len(notes) == 1
    assert isinstance(notes[0], NoteSeg)
    assert abs(notes[0].midi - 69.0) <= 0.2
