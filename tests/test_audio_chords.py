"""Unit tests for app.audio.chords (pure python/numpy, offline, runnable in main .venv)."""
from __future__ import annotations

import numpy as np
import pytest

from app.audio.analysis import ChordSeg, NoteSeg
from app.audio.chords import (
    BTC_ROOT_LIST,
    BTC_QUALITY_LIST,
    decode_chords,
    estimate_key,
    simplify_btc_chord,
    smooth_beat_chords,
)


def _make_mock_vocab() -> dict[int, str]:
    """Create standard 170-class BTC vocabulary."""
    vocab: dict[int, str] = {168: "X", 169: "N"}
    for i in range(168):
        root = BTC_ROOT_LIST[i // 14]
        quality = BTC_QUALITY_LIST[i % 14]
        vocab[i] = f"{root}:{quality}"
    return vocab


def test_never_raises_on_empty_input():
    """Verify decode_chords handles edge cases and empty inputs gracefully without raising."""
    vocab = _make_mock_vocab()
    empty_probs = np.zeros((0, 170), dtype=float)
    empty_chroma = np.zeros((12, 0), dtype=float)

    # Empty frames
    res1 = decode_chords(empty_probs, 0.1, [0.0, 1.0], vocab, empty_chroma)
    assert res1 == []

    # None probs
    res2 = decode_chords(None, 0.1, [0.0, 1.0], vocab, empty_chroma)
    assert res2 == []

    # Empty or single-element grid
    res3 = decode_chords(np.zeros((10, 170)), 0.1, [], vocab, empty_chroma)
    assert res3 == []
    res4 = decode_chords(np.zeros((10, 170)), 0.1, [0.0], vocab, empty_chroma)
    assert res4 == []

    # estimate_key on empty chords
    key_empty = estimate_key([])
    assert isinstance(key_empty, str)
    assert key_empty == "C"


def test_stable_chord_with_noisy_frames_stays_one_segment():
    """Verify stable chord across a 4-beat bar stays as one segment despite noisy frame posteriors."""
    vocab = _make_mock_vocab()
    # 4 beats: 0.0, 0.5, 1.0, 1.5, 2.0 -> 2.0s duration
    # hop_dur = 0.05 -> 40 frames total, 10 frames per beat
    hop_dur = 0.05
    n_frames = 40
    grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    downbeats = [0.0, 2.0]

    # C:maj is index 1 (root 0, quality 1)
    c_maj_idx = 1
    probs = np.zeros((n_frames, 170), dtype=float)

    # Dominant C:maj with small noise in other classes
    rng = np.random.default_rng(42)
    for f in range(n_frames):
        probs[f, c_maj_idx] = 0.70
        noise = rng.uniform(0.0, 0.05, size=170)
        # Add slight transient dip on beat 2 (frames 10..20)
        if 10 <= f < 20:
            probs[f, c_maj_idx] = 0.55
            probs[f, 0] = 0.15  # transient C:min leakage
        probs[f] += noise
        probs[f] /= np.sum(probs[f])

    chroma_bass = np.zeros((12, n_frames), dtype=float)
    # Root C (pc 0) is dominant in bass
    chroma_bass[0, :] = 1.0

    segs = decode_chords(
        probs,
        hop_dur,
        grid,
        vocab,
        chroma_bass,
        params={"downbeats": downbeats},
    )

    assert len(segs) == 1
    assert segs[0].label == "C:maj"
    assert segs[0].start == 0.0
    assert segs[0].end == 2.0
    assert segs[0].bass_pc is None  # root position


def test_real_change_on_beat_3_is_kept():
    """Verify a genuine harmonic transition on beat 3 (half-measure) is preserved cleanly."""
    vocab = _make_mock_vocab()
    hop_dur = 0.05
    n_frames = 40
    grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    downbeats = [0.0, 2.0]

    # Beats 1 & 2 (frames 0..20): C:maj (root 0, quality 1 -> index 1)
    # Beats 3 & 4 (frames 20..40): G:maj (root 7, quality 1 -> index 7 * 14 + 1 = 99)
    c_maj_idx = 1
    g_maj_idx = 7 * 14 + 1

    probs = np.zeros((n_frames, 170), dtype=float)
    probs[:20, c_maj_idx] = 0.85
    probs[20:, g_maj_idx] = 0.85

    # Normalize
    for f in range(n_frames):
        probs[f, 169] = 0.15
        probs[f] /= np.sum(probs[f])

    chroma_bass = np.zeros((12, n_frames), dtype=float)
    chroma_bass[0, :20] = 1.0  # C in bass
    chroma_bass[7, 20:] = 1.0  # G in bass

    segs = decode_chords(
        probs,
        hop_dur,
        grid,
        vocab,
        chroma_bass,
        params={"downbeats": downbeats},
    )

    assert len(segs) == 2
    assert segs[0].label == "C:maj"
    assert segs[0].start == 0.0
    assert segs[0].end == 1.0

    assert segs[1].label == "G:maj"
    assert segs[1].start == 1.0
    assert segs[1].end == 2.0


def test_seventh_kept_when_mass_dominates():
    """Verify seventh quality (7, maj7, min7) is selected when posterior mass supports it."""
    vocab = _make_mock_vocab()
    hop_dur = 0.05
    n_frames = 20
    grid = [0.0, 0.5, 1.0]

    # Test 1: Dominant 7th (G:7, root 7, qual 9 -> index 7*14 + 9 = 107)
    g_maj_idx = 7 * 14 + 1
    g_7_idx = 7 * 14 + 9
    probs_dom7 = np.zeros((n_frames, 170), dtype=float)
    # Triad has some activation, but 7th has strong conditional activation
    probs_dom7[:, g_maj_idx] = 0.30
    probs_dom7[:, g_7_idx] = 0.55
    probs_dom7[:, 169] = 0.15
    probs_dom7 /= np.sum(probs_dom7, axis=1, keepdims=True)

    chroma = np.zeros((12, n_frames), dtype=float)
    segs_7 = decode_chords(probs_dom7, hop_dur, grid, vocab, chroma)
    assert len(segs_7) == 1
    assert segs_7[0].label == "G:7"

    # Test 2: Major 7th (C:maj7, root 0, qual 8 -> index 8)
    c_maj_idx = 1
    c_maj7_idx = 8
    probs_maj7 = np.zeros((n_frames, 170), dtype=float)
    probs_maj7[:, c_maj_idx] = 0.35
    probs_maj7[:, c_maj7_idx] = 0.50
    probs_maj7[:, 169] = 0.15
    probs_maj7 /= np.sum(probs_maj7, axis=1, keepdims=True)

    segs_maj7 = decode_chords(probs_maj7, hop_dur, grid, vocab, chroma)
    assert len(segs_maj7) == 1
    assert segs_maj7[0].label == "C:maj7"

    # Test 3: Minor 7th (A:min7, root 9, qual 6 -> index 9*14 + 6 = 132)
    a_min_idx = 9 * 14 + 0
    a_min7_idx = 9 * 14 + 6
    probs_min7 = np.zeros((n_frames, 170), dtype=float)
    probs_min7[:, a_min_idx] = 0.30
    probs_min7[:, a_min7_idx] = 0.55
    probs_min7[:, 169] = 0.15
    probs_min7 /= np.sum(probs_min7, axis=1, keepdims=True)

    segs_min7 = decode_chords(probs_min7, hop_dur, grid, vocab, chroma)
    assert len(segs_min7) == 1
    assert segs_min7[0].label == "A:min7"


def test_slash_bass_accepted_only_for_chord_tones():
    """Verify slash bass inversion is accepted for 3rd/5th chord tones, and rejected for noise."""
    vocab = _make_mock_vocab()
    hop_dur = 0.05
    n_frames = 20
    grid = [0.0, 0.5, 1.0]

    # C:maj chord (root 0, 3rd E=4, 5th G=7)
    c_maj_idx = 1
    probs = np.zeros((n_frames, 170), dtype=float)
    probs[:, c_maj_idx] = 0.85
    probs[:, 169] = 0.15

    # Case A: Bass peak at 3rd (E, pc 4) with good energy -> First inversion C/E
    chroma_e = np.full((12, n_frames), 0.1, dtype=float)
    chroma_e[4, :] = 1.0  # E
    chroma_e[0, :] = 0.4  # C root has secondary energy
    segs_e = decode_chords(probs, hop_dur, grid, vocab, chroma_e)
    assert len(segs_e) == 1
    assert segs_e[0].label == "C:maj"
    assert segs_e[0].bass_pc == 4

    # Case B: Bass peak at 5th (G, pc 7) -> Second inversion C/G
    chroma_g = np.full((12, n_frames), 0.1, dtype=float)
    chroma_g[7, :] = 1.0  # G
    chroma_g[0, :] = 0.3  # C
    segs_g = decode_chords(probs, hop_dur, grid, vocab, chroma_g)
    assert len(segs_g) == 1
    assert segs_g[0].label == "C:maj"
    assert segs_g[0].bass_pc == 7

    # Case C: Non-chord-tone noise peak at F# (pc 6, tritone) -> Must be REJECTED
    chroma_noise = np.full((12, n_frames), 0.1, dtype=float)
    chroma_noise[6, :] = 1.0  # F# noise
    chroma_noise[0, :] = 0.3  # C
    segs_noise = decode_chords(probs, hop_dur, grid, vocab, chroma_noise)
    assert len(segs_noise) == 1
    assert segs_noise[0].label == "C:maj"
    assert segs_noise[0].bass_pc is None  # rejected!


def test_key_estimation():
    """Verify estimate_key correctly infers tonic key from chord and melody notes."""
    # Diatonic chords in C major: C, F, G, Am
    chords_c = [
        ChordSeg(start=0.0, end=2.0, label="C:maj"),
        ChordSeg(start=2.0, end=4.0, label="F:maj"),
        ChordSeg(start=4.0, end=6.0, label="G:7"),
        ChordSeg(start=6.0, end=8.0, label="A:min"),
    ]
    notes_c = [
        NoteSeg(start=0.0, end=1.0, midi=60.0),  # C4
        NoteSeg(start=1.0, end=2.0, midi=64.0),  # E4
        NoteSeg(start=2.0, end=3.0, midi=65.0),  # F4
        NoteSeg(start=3.0, end=4.0, midi=67.0),  # G4
    ]
    assert estimate_key(chords_c, notes_c) == "C"

    # Diatonic chords in G major: G, C, D, Em
    chords_g = [
        ChordSeg(start=0.0, end=2.0, label="G:maj"),
        ChordSeg(start=2.0, end=4.0, label="C:maj"),
        ChordSeg(start=4.0, end=6.0, label="D:maj"),
        ChordSeg(start=6.0, end=8.0, label="E:min"),
    ]
    assert estimate_key(chords_g) == "G"
