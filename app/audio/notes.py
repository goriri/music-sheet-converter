"""Melody note segmentation module (pure numpy, no ML/PyTorch/scipy).

Segments continuous f0 / pitch contours into discrete NoteSeg instances with:
- Vibrato tolerance via temporal median smoothing
- Portamento ramp absorption
- Repeated note separation via vocal energy (RMS) dips
- Blip noise removal
- Baseline reproducibility mode
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from app.audio.analysis import NoteSeg


def _numpy_median_filter_1d(a: np.ndarray, size: int = 5) -> np.ndarray:
    """Fast 1D median filter in pure numpy with reflect padding."""
    if len(a) == 0:
        return a.copy()
    if len(a) < size:
        return np.full_like(a, np.median(a))
    pad_len = size // 2
    padded = np.pad(a, pad_len, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=-1)


def segment_notes(
    time: Any,
    pitch_hz: Any,
    periodicity: Any,
    vocal_rms: Any,
    *,
    beats: Optional[list[float]] = None,
    params: Optional[dict[str, Any]] = None,
) -> list[NoteSeg]:
    """Segment pitch and periodicity contours into discrete melody notes.

    Parameters
    ----------
    time : array-like
        Frame timestamp array in seconds.
    pitch_hz : array-like
        Estimated fundamental frequency in Hz for each frame.
    periodicity : array-like
        Pitch confidence / voicing score [0.0, 1.0] for each frame.
    vocal_rms : array-like
        Vocal stem RMS energy for each frame.
    beats : list[float], optional
        Optional beat onset timestamps in seconds.
    params : dict, optional
        Tunable parameters:
        - 'mode': 'improved' (default) or 'baseline'
        - 'deviation_thresh': pitch deviation threshold in semitones (default: 0.95)
        - 'deviation_count_thresh': consecutive frames for pitch split (default: 4)
        - 'min_note_dur': minimum note duration in seconds (default: 0.08)

    Returns
    -------
    list[NoteSeg]
        List of segmented melody notes with start, end, midi, confidence.
        Never raises on empty or invalid inputs; returns [] instead.
    """
    params = params or {}
    mode = params.get("mode", "improved")

    # Guard against None or empty inputs
    if time is None or pitch_hz is None or periodicity is None or vocal_rms is None:
        return []

    try:
        time_arr = np.asarray(time, dtype=float)
        pitch_arr = np.asarray(pitch_hz, dtype=float)
        per_arr = np.asarray(periodicity, dtype=float)
        rms_arr = np.asarray(vocal_rms, dtype=float)
    except Exception:
        return []

    n_frames = min(len(time_arr), len(pitch_arr), len(per_arr), len(rms_arr))
    if n_frames == 0:
        return []

    # Clean NaNs, Infs, negative values
    time_arr = np.nan_to_num(time_arr[:n_frames], nan=0.0, posinf=0.0, neginf=0.0)
    pitch_arr = np.nan_to_num(pitch_arr[:n_frames], nan=0.0, posinf=0.0, neginf=0.0)
    per_arr = np.nan_to_num(per_arr[:n_frames], nan=0.0, posinf=0.0, neginf=0.0)
    rms_arr = np.nan_to_num(rms_arr[:n_frames], nan=0.0, posinf=0.0, neginf=0.0)

    hop_dur = float(time_arr[1] - time_arr[0]) if n_frames > 1 and (time_arr[1] - time_arr[0]) > 0 else 0.01

    if mode == "baseline":
        return _segment_notes_baseline(
            time_arr, pitch_arr, per_arr, rms_arr, hop_dur, n_frames
        )

    return _segment_notes_improved(
        time_arr, pitch_arr, per_arr, rms_arr, hop_dur, n_frames, params
    )


def _segment_notes_baseline(
    time_arr: np.ndarray,
    pitch_arr: np.ndarray,
    per_arr: np.ndarray,
    rms_arr: np.ndarray,
    hop_dur: float,
    n_frames: int,
) -> list[NoteSeg]:
    """Exact reproduction of previous baseline segmentation algorithm."""
    valid_rms = rms_arr[rms_arr > 0]
    ref = float(np.percentile(valid_rms, 95)) if valid_rms.size > 0 else 1e-4
    rms_active = max(1e-4, 0.10 * ref)

    is_voiced = np.zeros(n_frames, dtype=bool)
    state = False
    for t_idx in range(n_frames):
        p_val = per_arr[t_idx]
        hz_val = pitch_arr[t_idx]
        r_val = rms_arr[t_idx]
        turn_on = 0.22 if r_val >= rms_active else 0.35
        turn_off = 0.15 if r_val >= rms_active else 0.22
        if not state:
            if p_val >= turn_on and 65.0 <= hz_val <= 1100.0:
                state = True
        else:
            if p_val < turn_off or hz_val < 60.0 or hz_val > 1200.0:
                state = False
        is_voiced[t_idx] = state

    midi_arr = np.zeros(n_frames, dtype=float)
    v_idx = np.where(is_voiced)[0]
    if len(v_idx) > 0:
        midi_arr[v_idx] = 69.0 + 12.0 * np.log2(np.maximum(1.0, pitch_arr[v_idx]) / 440.0)
        midi_arr[v_idx] = _numpy_median_filter_1d(midi_arr[v_idx], size=5)

    bridged = is_voiced.copy()
    for idx in range(1, len(is_voiced) - 4):
        if not bridged[idx]:
            gap_len = 0
            while idx + gap_len < len(is_voiced) and not is_voiced[idx + gap_len]:
                gap_len += 1
            if 1 <= gap_len <= 3 and idx + gap_len < len(is_voiced):
                prev_p = midi_arr[idx - 1]
                next_p = midi_arr[idx + gap_len]
                if prev_p > 0 and next_p > 0 and abs(next_p - prev_p) <= 1.5:
                    for g in range(gap_len):
                        bridged[idx + g] = True
                        alpha = (g + 1) / (gap_len + 1)
                        midi_arr[idx + g] = prev_p + alpha * (next_p - prev_p)
    is_voiced = bridged

    notes: list[NoteSeg] = []
    in_note = False
    note_start = 0
    cur_midis: list[float] = []
    dev_count = 0

    for idx in range(n_frames):
        if is_voiced[idx]:
            cur_m = midi_arr[idx]
            if not in_note:
                in_note = True
                note_start = idx
                cur_midis = [cur_m]
                dev_count = 0
            else:
                ref_m = np.median(cur_midis)
                if abs(cur_m - ref_m) > 0.85:
                    dev_count += 1
                else:
                    dev_count = 0
                if dev_count >= 3 and len(cur_midis) >= 8:
                    split_idx = idx - dev_count + 1
                    prev_m = cur_midis[:len(cur_midis) - dev_count + 1]
                    n_dur = (split_idx - note_start) * hop_dur
                    if n_dur >= 0.08 and prev_m:
                        notes.append(
                            NoteSeg(
                                start=round(note_start * hop_dur, 2),
                                end=round(split_idx * hop_dur, 2),
                                midi=round(float(np.median(prev_m)), 1),
                                confidence=round(float(np.mean(per_arr[note_start:split_idx])), 3),
                            )
                        )
                    note_start = split_idx
                    cur_midis = list(midi_arr[split_idx:idx + 1])
                    dev_count = 0
                else:
                    cur_midis.append(cur_m)
        else:
            if in_note:
                in_note = False
                n_dur = (idx - note_start) * hop_dur
                if n_dur >= 0.08 and cur_midis:
                    notes.append(
                        NoteSeg(
                            start=round(note_start * hop_dur, 2),
                            end=round(idx * hop_dur, 2),
                            midi=round(float(np.median(cur_midis)), 1),
                            confidence=round(float(np.mean(per_arr[note_start:idx])), 3),
                        )
                    )
                cur_midis = []
                dev_count = 0

    if in_note and len(cur_midis) >= 8:
        notes.append(
            NoteSeg(
                start=round(note_start * hop_dur, 2),
                end=round(n_frames * hop_dur, 2),
                midi=round(float(np.median(cur_midis)), 1),
                confidence=round(float(np.mean(per_arr[note_start:])), 3),
            )
        )

    if len(notes) > 1:
        merged = [notes[0]]
        for n_next in notes[1:]:
            n_prev = merged[-1]
            if abs(n_next.midi - n_prev.midi) <= 0.4 and (n_next.start - n_prev.end) <= 0.06:
                merged[-1] = NoteSeg(
                    start=n_prev.start,
                    end=n_next.end,
                    midi=round(float((n_prev.midi + n_next.midi) / 2.0), 1),
                    confidence=round(float((n_prev.confidence + n_next.confidence) / 2.0), 3),
                )
            else:
                merged.append(n_next)
        notes = merged

    return notes


def _segment_notes_improved(
    time_arr: np.ndarray,
    pitch_arr: np.ndarray,
    per_arr: np.ndarray,
    rms_arr: np.ndarray,
    hop_dur: float,
    n_frames: int,
    params: dict[str, Any],
) -> list[NoteSeg]:
    """Improved note segmentation with vibrato damping, portamento absorption, and RMS dip detection."""
    valid_rms = rms_arr[rms_arr > 0]
    ref = float(np.percentile(valid_rms, 95)) if valid_rms.size > 0 else 1e-4
    rms_active = max(1e-4, 0.10 * ref)

    is_voiced = np.zeros(n_frames, dtype=bool)
    state = False
    for t_idx in range(n_frames):
        p_val = per_arr[t_idx]
        hz_val = pitch_arr[t_idx]
        r_val = rms_arr[t_idx]
        turn_on = 0.22 if r_val >= rms_active else 0.35
        turn_off = 0.15 if r_val >= rms_active else 0.22
        if not state:
            if p_val >= turn_on and 65.0 <= hz_val <= 1100.0:
                state = True
        else:
            if p_val < turn_off or hz_val < 60.0 or hz_val > 1200.0:
                state = False
        is_voiced[t_idx] = state

    midi_arr = np.zeros(n_frames, dtype=float)
    v_idx = np.where(is_voiced)[0]
    if len(v_idx) > 0:
        midi_arr[v_idx] = 69.0 + 12.0 * np.log2(np.maximum(1.0, pitch_arr[v_idx]) / 440.0)
        # 5-frame median preserves sharp onset timing while eliminating single-frame pitch spikes
        midi_arr[v_idx] = _numpy_median_filter_1d(midi_arr[v_idx], size=5)

    # Gap bridging for short unvoiced dropouts (consonants/breaths <= 40ms)
    bridged = is_voiced.copy()
    for idx in range(1, len(is_voiced) - 5):
        if not bridged[idx]:
            gap_len = 0
            while idx + gap_len < len(is_voiced) and not is_voiced[idx + gap_len]:
                gap_len += 1
            if 1 <= gap_len <= 4 and idx + gap_len < len(is_voiced):
                prev_p = midi_arr[idx - 1]
                next_p = midi_arr[idx + gap_len]
                if prev_p > 0 and next_p > 0 and abs(next_p - prev_p) <= 1.5:
                    for g in range(gap_len):
                        bridged[idx + g] = True
                        alpha = (g + 1) / (gap_len + 1)
                        midi_arr[idx + g] = prev_p + alpha * (next_p - prev_p)
    is_voiced = bridged

    # Precompute smoothed pitch for splitting (15-frame median) to cancel vibrato (~5.5Hz oscillation)
    split_midi = midi_arr.copy()
    if len(v_idx) > 0:
        split_midi[v_idx] = _numpy_median_filter_1d(midi_arr[v_idx], size=15)

    # Detect energy dips for repeated notes of the same pitch
    rms_dips = np.zeros(n_frames, dtype=bool)
    if n_frames > 10:
        window = 30
        pad_w = window // 2
        padded_rms = np.pad(rms_arr, pad_w, mode="edge")
        win_rms = np.lib.stride_tricks.sliding_window_view(padded_rms, window)
        local_max = np.max(win_rms, axis=-1)
        for i in range(2, n_frames - 2):
            if is_voiced[i] and local_max[i] > 1e-3:
                if rms_arr[i] < 0.35 * local_max[i] and rms_arr[i] <= rms_arr[i - 1] and rms_arr[i] < rms_arr[i + 1]:
                    rms_dips[i] = True

    # Note collection: list of (NoteSeg, is_dip_split_after)
    raw_notes: list[tuple[NoteSeg, bool]] = []
    in_note = False
    note_start = 0
    cur_midis: list[float] = []
    cur_split_midis: list[float] = []
    dev_count = 0

    dev_thresh = params.get("deviation_thresh", 0.95)
    dev_count_thresh = params.get("deviation_count_thresh", 4)
    min_dur = params.get("min_note_dur", 0.08)

    for idx in range(n_frames):
        if is_voiced[idx]:
            cur_m = midi_arr[idx]
            cur_s = split_midi[idx]

            if not in_note:
                in_note = True
                note_start = idx
                cur_midis = [cur_m]
                cur_split_midis = [cur_s]
                dev_count = 0
            else:
                # Check RMS dip for repeated note attack
                if rms_dips[idx] and (idx - note_start) * hop_dur >= min_dur:
                    raw_notes.append((
                        NoteSeg(
                            start=round(note_start * hop_dur, 2),
                            end=round(idx * hop_dur, 2),
                            midi=round(float(np.median(cur_midis)), 1),
                            confidence=round(float(np.mean(per_arr[note_start:idx])), 3),
                        ),
                        True,  # RMS dip boundary
                    ))
                    note_start = idx
                    cur_midis = [cur_m]
                    cur_split_midis = [cur_s]
                    dev_count = 0
                    continue

                ref_s = float(np.median(cur_split_midis))
                if abs(cur_s - ref_s) > dev_thresh:
                    dev_count += 1
                else:
                    dev_count = 0

                if dev_count >= dev_count_thresh and len(cur_midis) >= 8:
                    split_idx = idx - dev_count + 1
                    prev_m = cur_midis[:len(cur_midis) - dev_count + 1]
                    n_dur = (split_idx - note_start) * hop_dur
                    if n_dur >= min_dur and prev_m:
                        raw_notes.append((
                            NoteSeg(
                                start=round(note_start * hop_dur, 2),
                                end=round(split_idx * hop_dur, 2),
                                midi=round(float(np.median(prev_m)), 1),
                                confidence=round(float(np.mean(per_arr[note_start:split_idx])), 3),
                            ),
                            False,
                        ))
                    note_start = split_idx
                    cur_midis = list(midi_arr[split_idx:idx + 1])
                    cur_split_midis = list(split_midi[split_idx:idx + 1])
                    dev_count = 0
                else:
                    cur_midis.append(cur_m)
                    cur_split_midis.append(cur_s)
        else:
            if in_note:
                in_note = False
                n_dur = (idx - note_start) * hop_dur
                if n_dur >= min_dur and cur_midis:
                    raw_notes.append((
                        NoteSeg(
                            start=round(note_start * hop_dur, 2),
                            end=round(idx * hop_dur, 2),
                            midi=round(float(np.median(cur_midis)), 1),
                            confidence=round(float(np.mean(per_arr[note_start:idx])), 3),
                        ),
                        False,
                    ))
                cur_midis = []
                cur_split_midis = []
                dev_count = 0

    if in_note and len(cur_midis) >= 8:
        raw_notes.append((
            NoteSeg(
                start=round(note_start * hop_dur, 2),
                end=round(n_frames * hop_dur, 2),
                midi=round(float(np.median(cur_midis)), 1),
                confidence=round(float(np.mean(per_arr[note_start:])), 3),
            ),
            False,
        ))

    # Portamento absorption:
    # If 3 consecutive notes form a monotonic pitch ramp and intermediate note is short (<= 0.16s),
    # absorb the intermediate note into the 2 surrounding notes by splitting at its midpoint.
    if len(raw_notes) >= 3:
        i = 0
        cleaned: list[tuple[NoteSeg, bool]] = []
        while i < len(raw_notes):
            if i + 2 < len(raw_notes):
                (n1, d1), (n2, d2), (n3, d3) = raw_notes[i], raw_notes[i + 1], raw_notes[i + 2]
                dur2 = n2.end - n2.start
                is_ramp_up = (n1.midi + 0.5 <= n2.midi <= n3.midi - 0.5)
                is_ramp_down = (n1.midi - 0.5 >= n2.midi >= n3.midi + 0.5)
                if (
                    (is_ramp_up or is_ramp_down)
                    and dur2 <= 0.16
                    and (n2.start - n1.end) <= 0.05
                    and (n3.start - n2.end) <= 0.05
                    and not d1
                    and not d2
                ):
                    mid_t = round((n2.start + n2.end) / 2.0, 2)
                    new_n1 = NoteSeg(start=n1.start, end=mid_t, midi=n1.midi, confidence=n1.confidence)
                    new_n3 = NoteSeg(start=mid_t, end=n3.end, midi=n3.midi, confidence=n3.confidence)
                    cleaned.append((new_n1, False))
                    raw_notes[i + 2] = (new_n3, d3)
                    i += 2
                    continue
            cleaned.append(raw_notes[i])
            i += 1
        raw_notes = cleaned

    # Merge adjacent notes with similar pitch, EXCEPT if separated by an intentional RMS dip
    if len(raw_notes) > 1:
        merged: list[tuple[NoteSeg, bool]] = [raw_notes[0]]
        for n_next, d_next in raw_notes[1:]:
            n_prev, d_prev = merged[-1]
            gap = n_next.start - n_prev.end
            pitch_diff = abs(n_next.midi - n_prev.midi)
            if not d_prev and pitch_diff <= 0.5 and gap <= 0.06:
                merged[-1] = (
                    NoteSeg(
                        start=n_prev.start,
                        end=n_next.end,
                        midi=round(float((n_prev.midi + n_next.midi) / 2.0), 1),
                        confidence=round(float((n_prev.confidence + n_next.confidence) / 2.0), 3),
                    ),
                    d_next,
                )
            else:
                merged.append((n_next, d_next))
        raw_notes = merged

    # Filter out blips < min_dur
    notes = [n for n, _ in raw_notes if (n.end - n.start) >= min_dur]

    return notes
