"""Chord beat quantization and partial measure beat inference for Taiwanese band charts.

This module resolves the systematic typesetting offset where printed chord symbols
and notes appear indented after the barline (~0.3-0.6 beats), causing naive geometric
snapping to land at beat x.5 instead of x.0.

Key entrypoints:
- quantize_chord_beats(sheet: ParsedSheet) -> ParsedSheet:
    Robust typeset offset estimation, whole-beat preference grid snapping, melody-onset
    alignment, and monotonic chord ordering.
- infer_partial_measure_beats(sheet: ParsedSheet) -> ParsedSheet:
    Infers reduced beats for pickup / partial bars at row edges based on bounding box
    width ratio and jianpu melody beat-sum.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from app.models import ChordSymbol, Measure, ParsedSheet, QualityIssue, System
from app.qa.melody import parse_melody

logger = logging.getLogger(__name__)


def snap_to_grid(raw_val: float, prefer_whole_tol: float = 0.30) -> float:
    """Snap a 0-based offset to the 0.5 beat grid, preferring whole beats within tolerance.

    Args:
        raw_val: 0-based continuous beat offset (e.g. pos - delta).
        prefer_whole_tol: Margin around whole beats (default 0.30) where a whole beat is preferred.

    Returns:
        Snapped offset (e.g. 0.0, 0.5, 1.0, 1.5, ...).
    """
    nearest_whole = round(raw_val)
    if abs(raw_val - nearest_whole) <= prefer_whole_tol:
        return float(nearest_whole)
    return float(round(raw_val * 2.0) / 2.0)


def infer_partial_measure_beats(sheet: ParsedSheet) -> ParsedSheet:
    """Infer partial / pickup measure beats from layout width and melody beat-sum.

    A measure is partial if:
    1. Its bounding box width <= ~0.60 * median full-measure width of the page
       (excluding the measure itself).
    2. It is at a candidate structural position: the first measure of a row,
       the last measure before a double/final bar, or the last measure of the song.

    The new beat count is:
    - Melody beat sum if app/qa/melody.py parses it with notes and beat_sum < full_beats.
    - Else round(width_ratio * full_beats * 2.0) / 2.0 clamped to [0.5, full - 0.5].

    Chord beats in partial measures are re-clamped to [1.0, new_beats].
    Emits an informational QualityIssue with code 'partial_measure_inferred'.
    Never raises exceptions.
    """
    try:
        all_measures = sheet.measures()
        if not all_measures:
            return sheet

        # Scan for existing pickup system warnings from layout
        pickup_systems: set[tuple[int, int]] = set()
        for w in sheet.warnings:
            m_pu = re.search(r"\[v2_pickup\]\s+page=(\d+)\s+system=(\d+)", w)
            if m_pu:
                pickup_systems.add((int(m_pu.group(1)), int(m_pu.group(2))))

        for s_idx, s in enumerate(sheet.systems):
            p_idx = s.page
            page_systems = [sys for sys in sheet.systems if sys.page == p_idx]
            page_widths = [
                m.bbox[2] - m.bbox[0]
                for sys in page_systems
                for m in sys.measures
                if m.bbox and (m.bbox[2] > m.bbox[0])
            ]

            num_measures_in_sys = len(s.measures)
            for m_idx, m in enumerate(s.measures):
                if not m.bbox or m.bbox[2] <= m.bbox[0]:
                    continue

                w = m.bbox[2] - m.bbox[0]
                other_widths = [pw for pw in page_widths if abs(pw - w) > 1e-4] or page_widths
                med_w = float(np.median(other_widths)) if other_widths else w
                if med_w <= 1e-4:
                    continue

                ratio = w / med_w
                is_first = (m_idx == 0)
                is_last = (m_idx == num_measures_in_sys - 1)
                is_song_last = (m.index == len(all_measures) - 1)
                has_double_bar = bool(
                    (m.lyrics and ("||" in m.lyrics or "|]" in m.lyrics))
                    or (m.rhythm_hint and ("||" in m.rhythm_hint or "|]" in m.rhythm_hint))
                )
                is_candidate_pos = is_first or is_last or is_song_last or has_double_bar

                if ratio <= 0.60 and is_candidate_pos:
                    full_beats = float(m.beats)
                    pmel = parse_melody(m.melody, beats=full_beats) if m.melody else None

                    # If last in row or song last, and melody has full notes/holds (e.g. '5. - - -'), do not shorten
                    if (is_last or is_song_last) and pmel and pmel.notes and pmel.beat_sum >= full_beats:
                        continue

                    # If first in row and melody has full beats without pickup warning and ratio > 0.40: do not shorten
                    if is_first and pmel and pmel.notes and pmel.beat_sum >= full_beats and ratio > 0.40 and (p_idx, s_idx) not in pickup_systems:
                        continue

                    if pmel and pmel.notes and 0.0 < pmel.beat_sum < full_beats:
                        new_beats = float(pmel.beat_sum)
                    else:
                        inferred = round((w / med_w) * full_beats * 2.0) / 2.0
                        new_beats = float(max(0.5, min(full_beats - 0.5, inferred)))

                    if abs(new_beats - m.beats) > 1e-4:
                        orig_beats = m.beats
                        m.beats = new_beats

                        # Re-clamp chords in this partial measure
                        for c in m.chords:
                            c.beat = float(max(1.0, min(new_beats, c.beat)))

                        # Record QualityIssue if not already present
                        already_recorded = any(
                            iss.code == "partial_measure_inferred" and iss.measure_index == m.index
                            for iss in sheet.issues
                        )
                        if not already_recorded:
                            sheet.issues.append(
                                QualityIssue(
                                    stage="omr",
                                    measure_index=m.index,
                                    severity="info",
                                    code="partial_measure_inferred",
                                    message=f"第{m.index + 1}小节判定为弱起/不完整小节，节拍推断为 {new_beats} 拍（原 {orig_beats} 拍）",
                                    detail={
                                        "measure_index": m.index,
                                        "original_beats": orig_beats,
                                        "inferred_beats": new_beats,
                                        "width_ratio": round(ratio, 3),
                                        "melody": m.melody,
                                    },
                                )
                            )
    except Exception as exc:
        logger.warning("infer_partial_measure_beats encountered error: %s", exc)

    return sheet


def estimate_typeset_offsets(sheet: ParsedSheet) -> dict[int, float]:
    """Estimate page-level typeset offset delta (in beats) from first chords of measures.

    Most measures begin on beat 1.0, but printed chord boxes are indented by ~0.3-0.6 beats.
    To avoid contamination from measures starting on later beats, we only sample from:
    1. Measures where the first chord is the only chord (len(m.chords) == 1), or
    2. Measures where the first chord aligns with the leftmost note at beat 1.0.
    """
    page_deltas: dict[int, float] = {}

    for s in sheet.systems:
        p_idx = s.page
        if p_idx not in page_deltas:
            page_deltas[p_idx] = 0.5

    for p_idx in list(page_deltas.keys()):
        samples: list[float] = []
        for s in sheet.systems:
            if s.page != p_idx:
                continue
            for m in s.measures:
                if not m.bbox or m.bbox[2] <= m.bbox[0]:
                    continue
                mw = m.bbox[2] - m.bbox[0]
                if m.chords and m.chords[0].bbox:
                    is_only_chord = (len(m.chords) == 1)
                    first_c = m.chords[0]
                    cb = first_c.bbox
                    cx = (cb[0] + cb[2]) / 2.0
                    frac = max(0.0, min(1.0, (cx - m.bbox[0]) / mw))
                    pos = frac * m.beats

                    is_leftmost_note = False
                    if m.melody:
                        pm = parse_melody(m.melody, beats=m.beats)
                        if pm and pm.notes:
                            is_leftmost_note = (abs(pm.notes[0][0] - 1.0) < 0.15)

                    if (is_only_chord or is_leftmost_note) and 0.0 <= pos < 1.0:
                        samples.append(pos)

        if len(samples) >= 3:
            page_deltas[p_idx] = float(np.median(samples))
        else:
            page_deltas[p_idx] = 0.5

    return page_deltas


def quantize_chord_beats(sheet: ParsedSheet) -> ParsedSheet:
    """Quantize chord beats across the sheet, compensating for typesetting indentation.

    1. Infers partial measures (pickup bars, section-edge partials) first.
    2. Computes the typeset offset delta per page via robust median of measure-opening chords.
    3. In compound meters (6/8, 9/8, 12/8), snaps to the dotted-quarter grid first.
    4. Protects chords already on a melody onset at an integer beat unless delta evidence is strong.
    5. Snaps chord positions: beat = 1 + snap(pos - delta) to 0.5 grid, preferring whole beats.
    6. Aligns half-beat chords lacking melody onsets to nearest melody note onsets.
    7. Enforces strictly increasing chord beats within each measure without dropping chords.

    Never raises exceptions. Preserves chord counts and identities.
    """
    try:
        # Save chord identity invariant
        orig_chords = [
            (c.raw, c.confidence, c.stacked)
            for s in sheet.systems
            for m in s.measures
            for c in m.chords
        ]

        # 1. Partial measure inference
        infer_partial_measure_beats(sheet)

        # 2. Typeset offset estimation per page
        page_deltas = estimate_typeset_offsets(sheet)

        # 3. Beat quantization
        for s in sheet.systems:
            delta = page_deltas.get(s.page, 0.5)

            for m in s.measures:
                if not m.chords:
                    continue

                mb = m.bbox
                has_mb = bool(mb and (mb[2] > mb[0]))
                mw = (mb[2] - mb[0]) if has_mb else 1.0
                mbeats = float(m.beats)
                is_compound = (mbeats in (6.0, 9.0, 12.0))

                # Parse melody onsets if melody is available
                mel_onsets: list[float] = []
                if m.melody:
                    pmel = parse_melody(m.melody, beats=mbeats)
                    if pmel and pmel.notes:
                        mel_onsets = [n[0] for n in pmel.notes]

                for c in m.chords:
                    orig_beat = c.beat
                    orig_is_int = (abs(orig_beat - round(orig_beat)) < 0.01)
                    orig_on_mel_int = orig_is_int and any(abs(orig_beat - o) <= 0.15 for o in mel_onsets)

                    if cb := c.bbox:
                        if has_mb:
                            cx = (cb[0] + cb[2]) / 2.0
                            frac = max(0.0, min(1.0, (cx - mb[0]) / mw))
                            pos = frac * mbeats
                            continuous_beat = pos - delta + 1.0

                            if is_compound:
                                # For compound meters, snap to dotted-quarter grid first
                                dq_points = [1.0 + k * 3.0 for k in range(int(mbeats // 3))]
                                nearest_dq = min(dq_points, key=lambda dq: abs(continuous_beat - dq))
                                if abs(continuous_beat - nearest_dq) <= 1.0:
                                    beat = nearest_dq
                                else:
                                    beat = 1.0 + snap_to_grid(pos - delta, prefer_whole_tol=0.30)
                            elif orig_is_int and abs(continuous_beat - orig_beat) < 0.70:
                                # Never move an integer beat unless delta evidence is strong
                                beat = orig_beat
                            else:
                                raw_offset = pos - delta
                                snapped_offset = snap_to_grid(raw_offset, prefer_whole_tol=0.30)
                                beat = 1.0 + snapped_offset

                                # Melody onset check:
                                # If beat lands on a half-beat, check if melody has an onset there.
                                # If not, snap to the nearest melody onset.
                                is_half_beat = (abs(beat - round(beat)) > 0.01)
                                if is_half_beat and mel_onsets:
                                    has_onset = any(abs(beat - o) <= 0.15 for o in mel_onsets)
                                    if not has_onset:
                                        nearest_onset = min(
                                            mel_onsets,
                                            key=lambda o: (abs(continuous_beat - o), o),
                                        )
                                        if abs(continuous_beat - nearest_onset) <= 1.0:
                                            beat = float(nearest_onset)

                            c.beat = float(max(1.0, min(mbeats, beat)))
                        else:
                            c.beat = float(max(1.0, min(mbeats, c.beat)))
                    else:
                        c.beat = float(max(1.0, min(mbeats, c.beat)))

                # 4. Strictly increasing chords inside measure
                if len(m.chords) > 1:
                    # Forward pass
                    for i in range(1, len(m.chords)):
                        if m.chords[i].beat <= m.chords[i - 1].beat:
                            m.chords[i].beat = min(mbeats, m.chords[i - 1].beat + 0.5)

                    # Backward pass if collided with ceiling
                    for i in range(len(m.chords) - 2, -1, -1):
                        if m.chords[i].beat >= m.chords[i + 1].beat:
                            m.chords[i].beat = max(1.0, m.chords[i + 1].beat - 0.5)

                    # Final fallback if tightly crammed
                    for i in range(1, len(m.chords)):
                        if m.chords[i].beat <= m.chords[i - 1].beat:
                            span = (mbeats - m.chords[i - 1].beat) / float(len(m.chords) - i + 1)
                            m.chords[i].beat = round(m.chords[i - 1].beat + max(0.1, span), 2)

        # Invariant assertion: chord identity and count unchanged
        new_chords = [
            (c.raw, c.confidence, c.stacked)
            for s in sheet.systems
            for m in s.measures
            for c in m.chords
        ]
        assert orig_chords == new_chords, "Chord count and identities must not change during quantization"

    except Exception as exc:
        logger.warning("quantize_chord_beats encountered error: %s", exc)

    return sheet
