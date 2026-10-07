"""Conversion from AudioAnalysis to LeadSheet contract (pure python/numpy only).

Converts detected beats, notes, chords, lyrics, and sections into a well-formed LeadSheet.
Handles pickup measures, tempo drift, barline ties, rest filling, jianpu degree/octave
conversions, chord snapping/collapsing, and error recovery.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

from app.audio.analysis import AudioAnalysis, ChordSeg, LyricChar, NoteSeg, SectionSeg
from app.smart.models import LeadChord, LeadMeasure, LeadNote, LeadSheet
from app.theory.chords import clean_raw_chord, normalize_quality
from app.theory.keys import DEGREE_SEMITONES, KEY_NAME_TO_PC, key_name_to_pc

logger = logging.getLogger(__name__)

# Mapping from semitone offset (0..11) to default diatonic degree (1..7) and accidental (-1, 0, 1)
# Preferences: #4 and b7, b3, b6 per standard jianpu conventions
SEMITONE_TO_DEGREE: dict[int, tuple[int, int]] = {
    0: (1, 0),
    1: (2, -1),
    2: (2, 0),
    3: (3, -1),
    4: (3, 0),
    5: (4, 0),
    6: (4, 1),
    7: (5, 0),
    8: (6, -1),
    9: (6, 0),
    10: (7, -1),
    11: (7, 0),
}


def format_degree_to_number(root_semitone: int, quality: str, bass_semitone: int) -> str:
    """Format scale degrees (root & bass relative to tonic) into Taiwanese number notation."""
    deg, acc = SEMITONE_TO_DEGREE[root_semitone % 12]
    prefix = "b" if acc == -1 else ("#" if acc == 1 else "")
    root_str = f"{prefix}{deg}"

    if quality == "maj":
        q_suf = ""
    elif quality == "m":
        q_suf = "m"
    elif quality == "7":
        q_suf = "7"
    elif quality == "maj7":
        q_suf = "maj7"
    elif quality == "m7":
        q_suf = "m7"
    elif quality == "m7b5":
        q_suf = "m7-5"
    elif quality == "dim":
        q_suf = "dim"
    elif quality == "aug":
        q_suf = "aug"
    elif quality == "add9":
        q_suf = "(2)"
    elif quality == "sus4":
        q_suf = "sus4"
    elif quality == "7sus4":
        q_suf = "7sus"
    else:
        q_suf = quality

    root_chord = f"{root_str}{q_suf}"

    if (bass_semitone % 12) != (root_semitone % 12):
        b_deg, b_acc = SEMITONE_TO_DEGREE[bass_semitone % 12]
        b_prefix = "b" if b_acc == -1 else ("#" if b_acc == 1 else "")
        return f"{root_chord}/{b_prefix}{b_deg}"

    return root_chord


def _snap_onset(val: float) -> float:
    """Quantize onset to a grid, strongly preferring 0.5 and 1.0 beat positions."""
    half_grid = round(val * 2.0) / 2.0
    if abs(val - half_grid) <= 0.16:
        return max(0.0, half_grid)
    quarter_grid = round(val * 4.0) / 4.0
    return max(0.0, quarter_grid)


def _snap_duration(dur: float) -> float:
    """Quantize duration to a 0.25-beat grid, minimum 0.25."""
    q = round(dur * 4.0) / 4.0
    return max(0.25, q)


def _parse_time_signature(ts_str: str) -> tuple[int, int]:
    """Parse time signature string like '4/4' or '3/4' into (num, den)."""
    m = re.match(r"^(\d+)/(\d+)$", str(ts_str).strip())
    if m:
        num = int(m.group(1))
        den = int(m.group(2))
        if num > 0 and den > 0:
            return num, den
    return 4, 4


def _parse_chord_label_to_degree(
    label: str,
    tonic_pc: int,
    bass_pc: Optional[int] = None,
) -> Optional[str]:
    """Convert raw chord label into Taiwanese number notation relative to key."""
    raw = clean_raw_chord(label)
    if not raw or raw in ("N", "NONE", "NO_CHORD", "X"):
        return None

    # Handle colon format like "C:maj", "Bb:min7", "F#:7"
    if ":" in raw:
        parts = raw.split(":", 1)
        root_str, qual_part = parts[0], parts[1]
    else:
        root_str = raw
        qual_part = ""

    # Parse root
    m_root = re.match(r"^([A-Ga-g])([b#♭♯]?)", root_str)
    if not m_root:
        # Check if already a number notation chord like "1", "5/7", "2m7"
        if re.match(r"^[b#]?[1-7]", raw):
            return raw
        return None

    r_let = m_root.group(1).upper()
    r_acc = "b" if m_root.group(2) in ("b", "♭") else ("#" if m_root.group(2) in ("#", "♯") else "")
    root_name = f"{r_let}{r_acc}"
    if root_name.upper() not in KEY_NAME_TO_PC:
        return None
    r_pc = KEY_NAME_TO_PC[root_name.upper()]

    rem_root = root_str[len(m_root.group(0)):].strip()
    if rem_root and not qual_part:
        qual_part = rem_root

    qual_part = qual_part.strip().lower()
    if qual_part in ("", "maj", "major"):
        qual = "maj"
    elif qual_part in ("m", "min", "minor", "-"):
        qual = "m"
    elif qual_part in ("7", "dom7"):
        qual = "7"
    elif qual_part in ("maj7", "m7+", "^7"):
        qual = "maj7"
    elif qual_part in ("m7", "min7", "-7"):
        qual = "m7"
    elif qual_part in ("m7b5", "min7b5", "hdim7", "-7b5", "m7-5"):
        qual = "m7b5"
    elif qual_part in ("dim", "dim7", "o", "o7"):
        qual = "dim"
    elif qual_part in ("aug", "+", "aug7"):
        qual = "aug"
    elif qual_part in ("sus", "sus4"):
        qual = "sus4"
    elif qual_part in ("7sus", "7sus4", "sus7"):
        qual = "7sus4"
    elif qual_part in ("add9", "2", "add2", "9"):
        qual = "add9" if "add" in qual_part or qual_part == "2" else "9"
    else:
        try:
            qual = normalize_quality(qual_part, raw)
        except Exception:
            qual = "maj"

    rst = (r_pc - tonic_pc) % 12
    if bass_pc is not None:
        bst = (bass_pc - tonic_pc) % 12
    else:
        bst = rst

    return format_degree_to_number(rst, qual, bst)


def _build_measure_grid(
    a: AudioAnalysis,
    beats_per_bar: float,
    warnings: list[str],
) -> list[dict]:
    """Construct measure boundary specs from downbeats, beats, and duration.

    Returns a list of dicts:
      {
        'start': float,
        'end': float,
        'beats': float,
        'beat_times': list[float],
      }
    """
    sorted_beats = sorted([b for b in a.beats if b >= 0.0])
    sorted_downbeats = sorted([d for d in a.downbeats if d >= 0.0])

    # Deduplicate closely spaced timestamps (< 0.1s)
    clean_downbeats: list[float] = []
    for d in sorted_downbeats:
        if not clean_downbeats or (d - clean_downbeats[-1]) >= 0.2:
            clean_downbeats.append(d)

    # If downbeats exist:
    if clean_downbeats:
        measures: list[dict] = []
        d0 = clean_downbeats[0]

        # Check for pickup measure before first downbeat
        pre_beats = [b for b in sorted_beats if b < d0 - 0.05]
        if pre_beats:
            pickup_start = pre_beats[0]
            pickup_len_beats = max(1.0, round(float(len(pre_beats))))
            pickup_beats_times = pre_beats + [d0]
            measures.append({
                "start": pickup_start,
                "end": d0,
                "beats": float(pickup_len_beats),
                "beat_times": pickup_beats_times,
            })
        elif d0 > 0.5:
            # First downbeat is noticeably delayed, check if notes start early
            has_early_notes = any(n.start < d0 - 0.1 for n in a.notes)
            if has_early_notes:
                measures.append({
                    "start": 0.0,
                    "end": d0,
                    "beats": max(1.0, round(d0 * (a.tempo_bpm or 120.0) / 60.0)),
                    "beat_times": [0.0, d0],
                })

        # Measures between downbeats
        for i in range(len(clean_downbeats) - 1):
            t0 = clean_downbeats[i]
            t1 = clean_downbeats[i + 1]
            b_in_bar = [b for b in sorted_beats if t0 - 0.05 <= b < t1 - 0.05]
            if not b_in_bar or abs(b_in_bar[0] - t0) > 0.1:
                b_in_bar = [t0] + [b for b in b_in_bar if b > t0 + 0.05]
            b_in_bar.append(t1)

            # Determine beat count
            count = len(b_in_bar) - 1
            if abs(count - beats_per_bar) <= 1:
                m_beats = beats_per_bar
            else:
                m_beats = float(count) if count > 0 else beats_per_bar

            measures.append({
                "start": t0,
                "end": t1,
                "beats": m_beats,
                "beat_times": b_in_bar,
            })

        # Trailing measure after last downbeat if audio continues
        last_d = clean_downbeats[-1]
        trailing_end = max(a.duration, last_d)
        if trailing_end - last_d > 0.5:
            trailing_beats = [b for b in sorted_beats if b >= last_d - 0.05]
            if not trailing_beats or abs(trailing_beats[0] - last_d) > 0.1:
                trailing_beats = [last_d] + [b for b in trailing_beats if b > last_d + 0.05]
            if not trailing_beats or trailing_beats[-1] < trailing_end - 0.05:
                trailing_beats.append(trailing_end)

            count = len(trailing_beats) - 1
            m_beats = beats_per_bar if count <= 0 or abs(count - beats_per_bar) <= 1 else float(count)
            measures.append({
                "start": last_d,
                "end": trailing_end,
                "beats": m_beats,
                "beat_times": trailing_beats,
            })

        return measures

    # Fallback when downbeats are empty but beats exist
    if sorted_beats:
        measures = []
        bpb = int(round(beats_per_bar))
        for i in range(0, len(sorted_beats), bpb):
            chunk = sorted_beats[i : i + bpb]
            t0 = chunk[0]
            if i + bpb < len(sorted_beats):
                t1 = sorted_beats[i + bpb]
            else:
                t1 = max(a.duration, t0 + len(chunk) * 0.5)
            b_times = chunk + [t1]
            measures.append({
                "start": t0,
                "end": t1,
                "beats": float(len(chunk)),
                "beat_times": b_times,
            })
        return measures

    # Fallback when both downbeats and beats are empty
    warnings.append("缺少节拍检测，按固定节拍估算小节")
    bpm = a.tempo_bpm if a.tempo_bpm and a.tempo_bpm > 30.0 else 120.0
    sec_per_beat = 60.0 / bpm
    bar_dur = beats_per_bar * sec_per_beat
    tot_dur = max(a.duration, bar_dur)
    num_bars = max(1, math.ceil(tot_dur / bar_dur))

    measures = []
    for i in range(num_bars):
        t0 = i * bar_dur
        t1 = min(tot_dur, (i + 1) * bar_dur)
        measures.append({
            "start": t0,
            "end": t1,
            "beats": beats_per_bar,
            "beat_times": [t0 + k * sec_per_beat for k in range(int(beats_per_bar) + 1)],
        })
    return measures


def _time_to_measure_beat(m_spec: dict, t: float) -> float:
    """Map an absolute time t inside a measure to a fractional beat offset [0.0, beats]."""
    t0 = m_spec["start"]
    t1 = m_spec["end"]
    b_total = m_spec["beats"]
    if t <= t0:
        return 0.0
    if t >= t1:
        return b_total

    beat_times = m_spec["beat_times"]
    k_intervals = len(beat_times) - 1
    if k_intervals <= 0:
        # Fallback linear interpolation
        return (t - t0) / max(1e-5, t1 - t0) * b_total

    # Find the segment in beat_times
    for i in range(k_intervals):
        seg_start = beat_times[i]
        seg_end = beat_times[i + 1]
        if seg_start <= t <= seg_end and seg_end > seg_start:
            frac = (t - seg_start) / (seg_end - seg_start)
            # Map index i to continuous measure beat
            return (i + frac) * (b_total / k_intervals)

    return (t - t0) / max(1e-5, t1 - t0) * b_total


def analysis_to_lead(a: AudioAnalysis, title: str = "", artist: str = "") -> LeadSheet:
    """Convert an AudioAnalysis into a valid, rhythmic LeadSheet draft.

    Never raises: catches any unexpected errors and returns a minimal valid LeadSheet
    with warnings in provenance.
    """
    try:
        return _analysis_to_lead_impl(a, title=title, artist=artist)
    except Exception as exc:
        logger.exception("analysis_to_lead failed: %s", exc)
        prov = f"音频转录（待核对） · 转换异常: {exc}"
        return LeadSheet(
            title=title,
            artist=artist,
            key=a.key or "C",
            time_signature=a.time_signature or "4/4",
            tempo_bpm=a.tempo_bpm,
            measures=[
                LeadMeasure(
                    beats=4.0,
                    notes=[
                        LeadNote(
                            onset=0.0,
                            duration=4.0,
                            degree=0,
                            accidental=0,
                            octave=0,
                            tie_to_next=False,
                            lyric="",
                        )
                    ],
                    chords=[],
                    section=None,
                )
            ],
            chords_only=True,
            provenance=prov,
        )


def _analysis_to_lead_impl(a: AudioAnalysis, title: str = "", artist: str = "") -> LeadSheet:
    warnings = list(a.warnings)

    # 1. Parse Key and Tonic
    try:
        tonic_pc = key_name_to_pc(a.key)
    except Exception:
        warnings.append(f"调号无法识别: {a.key!r}，默认为 C 调")
        tonic_pc = 0
        a.key = "C"

    # 2. Time Signature & Beats per Bar
    num_beats, den_beats = _parse_time_signature(a.time_signature)
    beats_per_bar = float(num_beats) if den_beats == 4 else float(num_beats) * (4.0 / den_beats)

    # 3. Construct Measures Grid
    m_specs = _build_measure_grid(a, beats_per_bar, warnings)
    if not m_specs:
        m_specs = [{
            "start": 0.0,
            "end": max(4.0, a.duration),
            "beats": beats_per_bar,
            "beat_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        }]

    # 4. Melody Octave Centering
    valid_midis = [n.midi for n in a.notes if n.midi > 0 and n.confidence >= 0.15]
    if valid_midis:
        sorted_m = sorted(valid_midis)
        median_midi = sorted_m[len(sorted_m) // 2]
    else:
        median_midi = 60.0  # C4

    # Center octave 0 around the median pitch:
    # 1..7 degrees span ~12 semitones from tonic_pc. The center of octave 0 is tonic_pc + 5.5.
    k_oct = round((median_midi - 5.5 - tonic_pc) / 12.0)
    tonic_midi_oct0 = 12 * k_oct + tonic_pc

    # 5. Process and Quantize Notes across Measures
    # Group notes per measure segment
    measure_melody_notes: list[list[LeadNote]] = [[] for _ in range(len(m_specs))]
    note_abs_times: list[tuple[LeadNote, float]] = []

    for note_seg in a.notes:
        if note_seg.end <= note_seg.start or note_seg.midi <= 0 or note_seg.confidence < 0.15:
            continue
        if (note_seg.end - note_seg.start) < 0.07:
            continue

        m_round = int(round(note_seg.midi))
        semitone_interval = (m_round - tonic_pc) % 12
        deg, acc = SEMITONE_TO_DEGREE[semitone_interval]
        base_s = DEGREE_SEMITONES[deg - 1] + acc
        octave_val = int(round((m_round - (tonic_midi_oct0 + base_s)) / 12.0))
        octave_clamped = max(-2, min(2, octave_val))

        # Find overlapping measures
        n_start = note_seg.start
        n_end = note_seg.end

        for m_idx, spec in enumerate(m_specs):
            m_s = spec["start"]
            m_e = spec["end"]
            if n_end <= m_s or n_start >= m_e:
                continue

            seg_start_t = max(n_start, m_s)
            seg_end_t = min(n_end, m_e)
            if seg_end_t <= seg_start_t:
                continue

            # Map to fractional beats inside this measure
            b_start_raw = _time_to_measure_beat(spec, seg_start_t)
            b_end_raw = _time_to_measure_beat(spec, seg_end_t)

            b_start = _snap_onset(b_start_raw)
            raw_dur = max(0.25, b_end_raw - b_start_raw)
            dur = _snap_duration(raw_dur)

            # Ensure note does not cross the barline
            max_dur = spec["beats"] - b_start
            if max_dur < 0.25:
                continue
            if dur > max_dur:
                dur = round(max_dur * 4.0) / 4.0
                if dur < 0.25:
                    dur = 0.25

            # If note continues significantly past m_e, set tie_to_next
            has_tie = (n_end > m_e + 0.12) and (dur >= 0.5 or (spec["beats"] - b_start) <= 0.5)

            lead_note = LeadNote(
                onset=b_start,
                duration=dur,
                degree=deg,
                accidental=acc,
                octave=octave_clamped,
                tie_to_next=has_tie,
                lyric="",
            )
            measure_melody_notes[m_idx].append(lead_note)
            note_abs_times.append((lead_note, seg_start_t))

    # 6. Fill Rests and Fix Measure Notes
    final_measure_notes: list[list[LeadNote]] = []
    for m_idx, spec in enumerate(m_specs):
        m_beats = spec["beats"]
        raw_notes = measure_melody_notes[m_idx]

        if not raw_notes:
            # Fully resting measure
            final_measure_notes.append([
                LeadNote(
                    onset=0.0,
                    duration=m_beats,
                    degree=0,
                    accidental=0,
                    octave=0,
                    tie_to_next=False,
                    lyric="",
                )
            ])
            continue

        # Sort by onset
        raw_notes.sort(key=lambda n: (n.onset, -n.duration))

        # Pass 1: Merge overlapping or adjacent identical notes
        merged_raw: list[LeadNote] = []
        for n in raw_notes:
            if not merged_raw:
                merged_raw.append(n)
                continue
            prev = merged_raw[-1]
            prev_end = prev.onset + prev.duration

            # Same pitch: if adjacent or slightly separated by <= 0.25 beats, merge
            if prev.degree == n.degree and prev.accidental == n.accidental and prev.octave == n.octave:
                if n.onset <= prev_end + 0.25 + 1e-4:
                    new_dur = round(max(prev.duration, (n.onset + n.duration - prev.onset)) * 4.0) / 4.0
                    merged_raw[-1] = prev.model_copy(update={
                        "duration": min(m_beats - prev.onset, new_dur),
                        "tie_to_next": n.tie_to_next or prev.tie_to_next,
                    })
                    continue

            # Overlap with different pitch
            if n.onset < prev_end:
                new_prev_dur = round((n.onset - prev.onset) * 4.0) / 4.0
                if new_prev_dur >= 0.25:
                    merged_raw[-1] = prev.model_copy(update={"duration": new_prev_dur})
                    merged_raw.append(n)
                else:
                    merged_raw[-1] = n
            else:
                merged_raw.append(n)

        # Pass 2: Absorb tiny isolated blips into neighbors
        pass2_notes: list[LeadNote] = []
        for n in merged_raw:
            if not pass2_notes:
                pass2_notes.append(n)
                continue
            prev = pass2_notes[-1]
            if prev.duration <= 0.25 and abs(n.onset - (prev.onset + prev.duration)) < 1e-4:
                pass2_notes[-1] = n.model_copy(update={
                    "onset": prev.onset,
                    "duration": round((prev.duration + n.duration) * 4.0) / 4.0,
                })
            else:
                pass2_notes.append(n)

        # Pass 3: Construct sequence with rest-gap filling and legato breath absorption
        cur_pos = 0.0
        filled_notes: list[LeadNote] = []

        for n in pass2_notes:
            gap = round((n.onset - cur_pos) * 4.0) / 4.0
            if gap > 1e-4:
                if gap <= 0.25 and filled_notes and filled_notes[-1].degree > 0:
                    # Legato breath gap: absorb small gap into preceding note
                    prev_note = filled_notes[-1]
                    filled_notes[-1] = prev_note.model_copy(update={
                        "duration": round((prev_note.duration + gap) * 4.0) / 4.0
                    })
                    cur_pos += gap
                elif gap >= 0.25:
                    filled_notes.append(
                        LeadNote(
                            onset=cur_pos,
                            duration=gap,
                            degree=0,
                            accidental=0,
                            octave=0,
                            tie_to_next=False,
                            lyric="",
                        )
                    )
                    cur_pos += gap
                else:
                    n = n.model_copy(update={"onset": cur_pos})

            # Check remaining measure capacity
            avail = round((m_beats - cur_pos) * 4.0) / 4.0
            if avail < 0.25:
                break
            actual_dur = min(n.duration, avail)
            actual_dur = round(actual_dur * 4.0) / 4.0
            if actual_dur >= 0.25:
                filled_notes.append(n.model_copy(update={"onset": cur_pos, "duration": actual_dur}))
                cur_pos += actual_dur

        # Fill any trailing rest or extend last voiced note
        if cur_pos < m_beats - 1e-4:
            trailing_gap = round((m_beats - cur_pos) * 4.0) / 4.0
            if trailing_gap <= 0.25 and filled_notes and filled_notes[-1].degree > 0:
                last_n = filled_notes[-1]
                filled_notes[-1] = last_n.model_copy(update={"duration": round((last_n.duration + trailing_gap) * 4.0) / 4.0})
                cur_pos += trailing_gap
            elif trailing_gap >= 0.25:
                filled_notes.append(
                    LeadNote(
                        onset=cur_pos,
                        duration=trailing_gap,
                        degree=0,
                        accidental=0,
                        octave=0,
                        tie_to_next=False,
                        lyric="",
                    )
                )
                cur_pos += trailing_gap

        # Guarantee exact duration sum
        tot_dur = sum(n.duration for n in filled_notes)
        diff = round((m_beats - tot_dur) * 4.0) / 4.0
        if abs(diff) >= 0.25 and filled_notes:
            last_n = filled_notes[-1]
            adj_dur = last_n.duration + diff
            if adj_dur >= 0.25:
                filled_notes[-1] = last_n.model_copy(update={"duration": adj_dur})
            elif diff > 0:
                filled_notes.append(
                    LeadNote(
                        onset=tot_dur,
                        duration=diff,
                        degree=0,
                        accidental=0,
                        octave=0,
                        tie_to_next=False,
                        lyric="",
                    )
                )

        final_measure_notes.append(filled_notes)

    # 7. Lyrics Assignment (applied directly to final_measure_notes)
    if a.lyrics:
        all_melody_notes: list[tuple[LeadNote, float]] = []
        for m_idx, spec in enumerate(m_specs):
            m_s = spec["start"]
            m_e = spec["end"]
            m_b = max(1e-5, spec["beats"])
            for note in final_measure_notes[m_idx]:
                if note.degree > 0:
                    t_note = m_s + (note.onset / m_b) * (m_e - m_s)
                    all_melody_notes.append((note, t_note))

        all_melody_notes.sort(key=lambda item: item[1])

        assigned_note_ids = set()
        for lyric_char in a.lyrics:
            text = lyric_char.text.strip()
            if not text:
                continue

            c_center = (lyric_char.start + lyric_char.end) / 2.0
            best_note: Optional[LeadNote] = None
            min_dist = float("inf")

            for note, t_note in all_melody_notes:
                dist = abs(t_note - c_center)
                if dist < min_dist and id(note) not in assigned_note_ids:
                    min_dist = dist
                    best_note = note

            if best_note is not None and min_dist <= 2.0:
                best_note.lyric = text
                assigned_note_ids.add(id(best_note))

    # 8. Process Chords per Measure
    measure_chords: list[list[LeadChord]] = [[] for _ in range(len(m_specs))]
    for chord_seg in a.chords:
        num_str = _parse_chord_label_to_degree(chord_seg.label, tonic_pc, chord_seg.bass_pc)
        if not num_str:
            continue

        c_start = chord_seg.start
        c_end = chord_seg.end

        for m_idx, spec in enumerate(m_specs):
            m_s = spec["start"]
            m_e = spec["end"]
            if c_end <= m_s or c_start >= m_e:
                continue

            # Chord starts within or before this measure
            overlap_start = max(c_start, m_s)
            offset_beat = _time_to_measure_beat(spec, overlap_start)

            # Snap to 1-based beat
            # Default harmonic rhythm favors beat 1.0 or 3.0 in 4/4
            b_val = 1.0 + offset_beat
            if spec["beats"] == 4.0:
                if 1.0 <= b_val <= 1.45:
                    snapped_beat = 1.0
                elif 2.55 <= b_val <= 3.45:
                    snapped_beat = 3.0
                elif 1.45 < b_val < 2.55:
                    snapped_beat = 2.0
                else:
                    snapped_beat = 4.0
            else:
                snapped_beat = max(1.0, min(spec["beats"], float(round(b_val))))

            measure_chords[m_idx].append(
                LeadChord(
                    raw=num_str,
                    beat=snapped_beat,
                    agreement="unchecked",
                    alternatives=[],
                )
            )

    # Clean and collapse chords within each measure
    final_measure_chords: list[list[LeadChord]] = []
    for m_idx, raw_chords in enumerate(measure_chords):
        if not raw_chords:
            final_measure_chords.append([])
            continue

        # Sort by beat
        raw_chords.sort(key=lambda c: c.beat)

        # Deduplicate identical beat positions
        by_beat: dict[float, LeadChord] = {}
        for c in raw_chords:
            # If multiple chords on same beat, keep latest or first
            if c.beat not in by_beat:
                by_beat[c.beat] = c
        sorted_by_beat = [by_beat[b] for b in sorted(by_beat.keys())]

        # Collapse consecutive identical chords within this measure
        collapsed: list[LeadChord] = []
        for c in sorted_by_beat:
            if not collapsed or collapsed[-1].raw != c.raw:
                collapsed.append(c)

        final_measure_chords.append(collapsed)

    # 9. Sections Assignment
    measure_sections: list[Optional[str]] = [None] * len(m_specs)
    if a.sections:
        for sec in a.sections:
            sec_start = sec.start
            # Find nearest measure start
            best_idx = 0
            min_dist = float("inf")
            for idx, spec in enumerate(m_specs):
                dist = abs(spec["start"] - sec_start)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            if measure_sections[best_idx] is None:
                measure_sections[best_idx] = sec.label

    # 10. Assemble LeadMeasures and LeadSheet
    measures: list[LeadMeasure] = []
    has_any_melody = False

    for m_idx, spec in enumerate(m_specs):
        m_notes = final_measure_notes[m_idx]
        if any(n.degree > 0 for n in m_notes):
            has_any_melody = True

        measures.append(
            LeadMeasure(
                beats=spec["beats"],
                notes=m_notes,
                chords=final_measure_chords[m_idx],
                section=measure_sections[m_idx],
            )
        )

    # Provenance
    prov = "音频转录（待核对）"
    if warnings:
        prov += " · " + "；".join(warnings)

    return LeadSheet(
        title=title,
        artist=artist,
        key=a.key,
        time_signature=a.time_signature,
        tempo_bpm=a.tempo_bpm,
        measures=measures,
        chords_only=not has_any_melody,
        provenance=prov,
        sources=[],
    )
