"""Vocal-activity gating must be relative to the stem level (no torch needed)."""
from __future__ import annotations

import numpy as np

from app.audio.transcribe import vocal_activity_thresholds


def _gate(rms):
    active, loud = vocal_activity_thresholds(rms)
    return np.asarray(rms) >= active, active, loud


def test_continuously_voiced_stem_is_not_gated_out():
    # A stem that is sung almost everywhere (no silence) — the old p25*1.5 rule gated out
    # every frame here (live regression: synthetic 茉莉花 produced 0 notes).
    rng = np.random.default_rng(0)
    rms = 0.24 + 0.03 * rng.standard_normal(2000)
    is_active, _, _ = _gate(rms)
    assert is_active.mean() > 0.95


def test_quiet_upload_still_detected():
    # Quiet phone recording: voiced RMS ~0.004, silence ~1e-5 (old absolute floor 0.008 missed it).
    rms = np.concatenate([np.full(500, 1e-5), np.full(1000, 0.004), np.full(500, 1e-5)])
    is_active, _, _ = _gate(rms)
    assert is_active[500:1500].all()
    assert not is_active[:500].any() and not is_active[1500:].any()


def test_typical_song_separates_voice_from_residue():
    # Levels seen on RWC stems: voiced ~0.13, separation residue ~0.0004.
    rms = np.concatenate([np.full(3000, 0.0004), np.full(4000, 0.13), np.full(3000, 0.0004)])
    is_active, active, loud = _gate(rms)
    assert is_active[3000:7000].all() and not is_active[:3000].any()
    assert active < loud


def test_degenerate_inputs_do_not_raise():
    for rms in ([], [0.0] * 10, [float("nan")] * 5):
        active, loud = vocal_activity_thresholds(rms)
        assert active > 0 and loud > 0
