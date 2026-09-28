"""Safe fallback measure generator for verified arrangement repair."""
from __future__ import annotations

from typing import Optional

from app.models import Difficulty, Event, Measure, Note, ResolvedChord
from app.qa.music21_ref import get_chord_reference

# Valid MIDI pitch boundaries per PLAN
RH_MIN_MIDI: int = 64  # E4
RH_MAX_MIDI: int = 81  # A5
LH_MIN_BASS: int = 36  # C2
LH_MAX_BASS: int = 52  # E3
LH_CENTER_BASS: int = 43  # G2


def _assign_rh_fingering(pitches: tuple[int, ...]) -> list[int]:
    """Assign monotonic fingers 1..5 based on interval table."""
    k = len(pitches)
    if k == 0:
        return []
    if k == 1:
        return [1]
    if k == 2:
        span = pitches[1] - pitches[0]
        if span <= 4:
            return [1, 3]
        elif span <= 7:
            return [1, 4]
        return [1, 5]
    if k == 3:
        i1 = pitches[1] - pitches[0]
        i2 = pitches[2] - pitches[1]
        if i1 <= 4 and i2 >= 5:
            return [1, 2, 5]
        elif i1 <= 2:
            return [1, 2, 5] if i2 >= 4 else [1, 2, 4]
        elif i2 <= 2:
            return [1, 3, 5] if i1 >= 4 else [1, 2, 4]
        return [1, 2, 5] if i2 > i1 else [1, 3, 5]
    if k == 4:
        i1 = pitches[1] - pitches[0]
        i3 = pitches[3] - pitches[2]
        return [1, 2, 4, 5] if i1 > i3 else [1, 2, 3, 5]
    return [1, 2, 3, 4, 5][:k]


def _generate_candidate_voicings(
    chord: ResolvedChord,
    difficulty: Difficulty,
) -> list[tuple[int, ...]]:
    """Generate candidate close-position voicings containing defining tones within an octave."""
    try:
        _, _, _, def_pcs = get_chord_reference(chord)
    except Exception:
        def_pcs = [(chord.root_pc + (3 if "m" in chord.quality else 4)) % 12]

    # Beginner drops 7th/9th; keeps 3rd/4th and b5 for m7b5/dim
    if difficulty == "beginner":
        req_defs = [def_pcs[0]] if def_pcs else [chord.root_pc]
        if chord.quality in ("m7b5", "dim"):
            b5_pc = (chord.root_pc + 6) % 12
            if b5_pc not in req_defs:
                req_defs.append(b5_pc)
    else:
        req_defs = def_pcs

    chord_pcs = set(chord.pcs) if chord.pcs else {(chord.root_pc + i) % 12 for i in (0, 4, 7)}
    candidates: list[tuple[int, ...]] = []

    lengths = (3,) if difficulty == "beginner" else (3, 4)
    for base_midi in range(RH_MIN_MIDI, RH_MAX_MIDI - 2):
        if base_midi % 12 not in chord_pcs:
            continue
        for length in lengths:
            voicing = [base_midi]
            curr = base_midi
            for _ in range(length - 1):
                nxt = curr + 1
                while nxt <= RH_MAX_MIDI and nxt % 12 not in chord_pcs:
                    nxt += 1
                if nxt <= RH_MAX_MIDI:
                    voicing.append(nxt)
                    curr = nxt
                else:
                    break
            if len(voicing) == length and voicing[-1] <= RH_MAX_MIDI:
                span = voicing[-1] - voicing[0]
                if span <= 12:
                    v_pcs = {n % 12 for n in voicing}
                    if all(d in v_pcs for d in req_defs):
                        candidates.append(tuple(voicing))

    # Fallback to guaranteed triad if candidates empty
    if not candidates:
        base_pcs = [chord.root_pc, req_defs[0], (chord.root_pc + 7) % 12]
        fb_pitches = []
        for p_pc in base_pcs:
            p = 60 + p_pc
            while p < RH_MIN_MIDI:
                p += 12
            while p > RH_MAX_MIDI:
                p -= 12
            fb_pitches.append(p)
        fb_pitches.sort()
        # Ensure distinct pitches
        unique_p = []
        for p in fb_pitches:
            if not unique_p or p > unique_p[-1]:
                unique_p.append(p)
        if unique_p[-1] - unique_p[0] <= 12:
            candidates.append(tuple(unique_p))
        else:
            candidates.append(tuple(unique_p[:2] if len(unique_p) >= 2 else unique_p))

    return candidates


def _rh_voicing_transition_cost(prev_voicing: tuple[int, ...], v: tuple[int, ...]) -> float:
    d1 = sum(min(abs(n - p) for p in prev_voicing) for n in v)
    d2 = sum(min(abs(p - n) for n in v) for p in prev_voicing)
    return (d1 + d2) / 2.0 + 0.1 * abs(v[-1] - 73)


def _pick_best_rh_voicing(
    candidates: list[tuple[int, ...]],
    prev_voicing: Optional[tuple[int, ...]],
) -> tuple[int, ...]:
    """Pick voicing closest to prev_voicing, or closest to register center (C#5 = 73)."""
    if not candidates:
        return (RH_MIN_MIDI, RH_MIN_MIDI + 4, RH_MIN_MIDI + 7)

    if prev_voicing:
        return min(candidates, key=lambda v: _rh_voicing_transition_cost(prev_voicing, v))

    return min(candidates, key=lambda v: abs(v[-1] - 73) + abs(v[0] - 67))


def _select_lh_bass(bass_pc: int) -> int:
    """Select a clean LH bass pitch in [LH_MIN_BASS, LH_MAX_BASS] closest to G2 (43)."""
    target = bass_pc % 12
    cands = [p for p in range(LH_MIN_BASS, LH_MAX_BASS + 1) if p % 12 == target]
    if not cands:
        p = LH_MIN_BASS + target
        while p < LH_MIN_BASS:
            p += 12
        while p > LH_MAX_BASS:
            p -= 12
        return p
    return min(cands, key=lambda p: abs(p - LH_CENTER_BASS))


def safe_measure(
    measure: Measure,
    resolved_chords: list[ResolvedChord],
    tonic_pc: int,
    difficulty: Difficulty,
    prev_rh_voicing: Optional[tuple[int, ...]] = None,
) -> tuple[list[Event], list[Event]]:
    """Generate a verified safe fallback measure (block chord + bass).

    RH: block close-position voicing nearest to prev_rh_voicing containing all defining tones within an octave,
        simple fingering from the interval table.
    LH: single bass note (slash bass if any) held for each chord span.
    is_stop: hit at chord onset and rest for the remainder of the span.
    """
    measure_beats = measure.beats if measure else 4.0

    if not resolved_chords:
        # Default to tonic triad
        resolved_chords = [
            ResolvedChord(
                raw="1",
                name="Tonic",
                beat=1.0,
                root_pc=tonic_pc,
                bass_pc=tonic_pc,
                pcs=[tonic_pc, (tonic_pc + 4) % 12, (tonic_pc + 7) % 12],
                quality="maj",
            )
        ]

    # Calculate chord segment boundaries
    num_chords = len(resolved_chords)
    chord_segments: list[dict] = []
    for i in range(num_chords):
        c_onset = round(resolved_chords[i].beat - 1.0, 4)
        c_onset = max(0.0, min(measure_beats - 0.25, c_onset))
        next_onset = round(resolved_chords[i + 1].beat - 1.0, 4) if i + 1 < num_chords else measure_beats
        next_onset = max(c_onset + 0.25, min(measure_beats, next_onset))
        chord_segments.append({
            "onset": c_onset,
            "duration": round(next_onset - c_onset, 4),
            "chord": resolved_chords[i],
        })

    rh_events: list[Event] = []
    lh_events: list[Event] = []

    # Lead-in rest if first chord begins after beat 1
    if chord_segments and chord_segments[0]["onset"] > 0.0:
        lead_in = chord_segments[0]["onset"]
        rh_events.append(Event(onset=0.0, duration=lead_in, notes=[]))
        lh_events.append(Event(onset=0.0, duration=lead_in, notes=[]))

    curr_rh_prev = prev_rh_voicing

    for seg in chord_segments:
        s_onset = seg["onset"]
        s_dur = seg["duration"]
        chord = seg["chord"]

        cands = _generate_candidate_voicings(chord, difficulty)
        v = _pick_best_rh_voicing(cands, curr_rh_prev)
        curr_rh_prev = v

        fingers = _assign_rh_fingering(v)
        rh_notes = [Note(midi=p, finger=f) for p, f in zip(v, fingers)]

        bass_midi = _select_lh_bass(chord.bass_pc)
        lh_notes = [Note(midi=bass_midi, finger=5)]

        if measure and measure.is_stop:
            # Hit on beat, rest for the remainder
            hit_dur = min(1.0, s_dur)
            rh_events.append(Event(onset=s_onset, duration=hit_dur, notes=rh_notes))
            lh_events.append(Event(onset=s_onset, duration=hit_dur, notes=lh_notes))
            rest_dur = round(s_dur - hit_dur, 4)
            if rest_dur > 0.0:
                rh_events.append(Event(onset=round(s_onset + hit_dur, 4), duration=rest_dur, notes=[]))
                lh_events.append(Event(onset=round(s_onset + hit_dur, 4), duration=rest_dur, notes=[]))
        else:
            # Held for the chord span
            rh_events.append(Event(onset=s_onset, duration=s_dur, notes=rh_notes))
            lh_events.append(Event(onset=s_onset, duration=s_dur, notes=lh_notes))

    return rh_events, lh_events
