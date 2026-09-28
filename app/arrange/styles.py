"""Accompaniment style patterns, texture modifiers, and measure tiling."""
from __future__ import annotations

import re
from typing import Optional

from app.models import Difficulty, Event, Note, ResolvedChord
from app.arrange.voicing import get_lh_pattern_pitches

SUPPORTED_STYLES = {"slow soul", "ballad", "slow rock", "pop", "waltz"}


def normalize_style(raw_style: Optional[str]) -> str:
    """Normalize style name string to canonical style name."""
    if not raw_style:
        return "ballad"
    s = raw_style.strip().lower()
    if "slow soul" in s or "soul" in s:
        return "slow soul"
    if "slow rock" in s:
        return "slow rock"
    if "waltz" in s or "3/4" in s:
        return "waltz"
    if "pop" in s:
        return "pop"
    if "ballad" in s:
        return "ballad"
    return "ballad"


def get_diatonic_fill_notes(top_note: int, tonic_pc: int, num_notes: int = 4) -> list[int]:
    """Generate descending diatonic fill notes starting from top_note within the current key."""
    diatonic_pcs = set((tonic_pc + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11])

    curr = top_note
    # Snap to nearest diatonic pitch <= top_note
    while curr % 12 not in diatonic_pcs:
        curr -= 1

    notes = [curr]
    for _ in range(num_notes - 1):
        curr -= 1
        while curr % 12 not in diatonic_pcs:
            curr -= 1
        notes.append(curr)

    return notes


def degree_to_bass_pitch(deg_token: str, bass_pitch: int, tonic_pc: int) -> int:
    """Convert a bass_hint degree token (e.g. '1', '2', '5', 'b7') to concrete MIDI pitch relative to chord bass."""
    diatonic_pcs = [(tonic_pc + s) % 12 for s in [0, 2, 4, 5, 7, 9, 11]]
    bass_pc = bass_pitch % 12

    clean = deg_token.strip()
    if clean == "1":
        return bass_pitch

    if bass_pc in diatonic_pcs:
        base_idx = diatonic_pcs.index(bass_pc)
    else:
        base_idx = 0

    if clean == "2":
        target_pc = diatonic_pcs[(base_idx + 1) % 7]
        p = bass_pitch + 1
        while p % 12 != target_pc:
            p += 1
        return p
    elif clean == "3":
        target_pc = diatonic_pcs[(base_idx + 2) % 7]
        p = bass_pitch + 1
        while p % 12 != target_pc:
            p += 1
        return p
    elif clean == "4":
        target_pc = diatonic_pcs[(base_idx + 3) % 7]
        p = bass_pitch + 1
        while p % 12 != target_pc:
            p += 1
        return p
    elif clean == "5":
        target_pc = diatonic_pcs[(base_idx + 4) % 7]
        p = bass_pitch + 1
        while p % 12 != target_pc:
            p += 1
        return p
    elif clean in ("b7", "7b"):
        return bass_pitch - 2
    elif clean == "7":
        return bass_pitch - 1

    digits = [c for c in clean if c.isdigit()]
    if digits:
        d = int(digits[0])
        if d == 1:
            return bass_pitch
        return bass_pitch + (d - 1) * 2

    return bass_pitch


def parse_bass_hint(
    bass_hint: str,
    bass_pitch: int,
    tonic_pc: int,
    measure_beats: float = 4.0,
) -> list[tuple[float, float, int]]:
    """Parse printed 'Bs:' notation (e.g. '11 11 11 112') into (onset, duration, midi_pitch) tuples."""
    clean = re.sub(r"[~^]", "", bass_hint).strip()
    tokens = clean.split()
    if not tokens:
        return [(i * 0.5, 0.5, bass_pitch) for i in range(int(measure_beats * 2))]

    events: list[tuple[float, float, int]] = []
    current_beat = 0.0

    for token in tokens:
        if current_beat >= measure_beats:
            break
        note_tokens = re.findall(r"[b#♭♯]?[1-7]", token)
        if not note_tokens:
            events.append((current_beat, 1.0, bass_pitch))
            current_beat += 1.0
            continue

        count = len(note_tokens)
        if count == 1:
            durations = [1.0]
        elif count == 2:
            durations = [0.5, 0.5]
        elif count == 3:
            durations = [0.5, 0.25, 0.25]
        elif count >= 4:
            durations = [0.25, 0.25, 0.25, 0.25][:count]
        else:
            durations = [1.0]

        beat_offset = 0.0
        for n_tok, d_dur in zip(note_tokens, durations):
            p = degree_to_bass_pitch(n_tok, bass_pitch, tonic_pc)
            events.append((round(current_beat + beat_offset, 4), d_dur, p))
            beat_offset += d_dur

        current_beat += 1.0

    return events


def ensure_exact_tiling(events: list[Event], measure_beats: float) -> list[Event]:
    """Ensure events exactly tile the measure [0.0, measure_beats] with continuous onsets and explicit rests."""
    if not events:
        return [Event(onset=0.0, duration=measure_beats, notes=[])]

    tiled: list[Event] = []
    curr_time = 0.0

    for ev in events:
        onset = round(ev.onset, 4)
        dur = round(ev.duration, 4)

        if onset > curr_time:
            # Fill gap with explicit rest
            gap = round(onset - curr_time, 4)
            tiled.append(Event(onset=curr_time, duration=gap, notes=[]))
            curr_time = onset

        if curr_time >= measure_beats:
            break

        actual_dur = min(dur, round(measure_beats - curr_time, 4))
        if actual_dur > 0:
            tiled.append(Event(
                onset=curr_time,
                duration=actual_dur,
                notes=ev.notes,
                tie_to_next=ev.tie_to_next,
            ))
            curr_time = round(curr_time + actual_dur, 4)

    if curr_time < measure_beats:
        rem = round(measure_beats - curr_time, 4)
        tiled.append(Event(onset=curr_time, duration=rem, notes=[]))

    return tiled


def generate_measure_events(
    chords: list[ResolvedChord],
    rh_voicings: list[tuple[int, ...]],
    lh_bass_pitches: list[int],
    difficulty: Difficulty,
    style: str,
    measure_beats: float = 4.0,
    tonic_pc: int = 0,
    key_name: str = "C",
    section_label: Optional[str] = None,
    is_stop: bool = False,
    fill: bool = False,
    bass_hint: Optional[str] = None,
) -> tuple[list[Event], list[Event]]:
    """Generate strictly tiled RH and LH events for a single measure according to style and texture."""
    # Check section label for texture modifiers
    is_sparse = False
    if section_label:
        norm_label = section_label.lower()
        if "only pn" in norm_label or "pn(rhy)" in norm_label:
            is_sparse = True
        if "+bs in" in norm_label or "og in" in norm_label or "tempo" in norm_label:
            is_sparse = False

    # 1. Stop measure: band hits beat 1 and rests for the remainder
    if is_stop:
        first_voicing = rh_voicings[0] if rh_voicings else (64, 67, 72)
        first_bass = lh_bass_pitches[0] if lh_bass_pitches else 36
        rh = [
            Event(onset=0.0, duration=1.0, notes=[Note(midi=p) for p in first_voicing]),
            Event(onset=1.0, duration=round(measure_beats - 1.0, 4), notes=[]),
        ]
        lh = [
            Event(onset=0.0, duration=1.0, notes=[Note(midi=first_bass)]),
            Event(onset=1.0, duration=round(measure_beats - 1.0, 4), notes=[]),
        ]
        return ensure_exact_tiling(rh, measure_beats), ensure_exact_tiling(lh, measure_beats)

    # Calculate chord segment boundaries
    num_chords = len(chords)
    chord_segments: list[dict] = []
    for i in range(num_chords):
        c_onset = round(chords[i].beat - 1.0, 4)
        c_onset = max(0.0, min(measure_beats - 0.25, c_onset))
        next_onset = round(chords[i + 1].beat - 1.0, 4) if i + 1 < num_chords else measure_beats
        next_onset = max(c_onset + 0.25, min(measure_beats, next_onset))
        chord_dur = round(next_onset - c_onset, 4)
        chord_segments.append({
            "onset": c_onset,
            "duration": chord_dur,
            "chord": chords[i],
            "voicing": rh_voicings[i],
            "bass": lh_bass_pitches[i],
        })

    # Lead-in rest if first chord starts after beat 1
    rh_events: list[Event] = []
    lh_events: list[Event] = []
    if chord_segments and chord_segments[0]["onset"] > 0:
        gap = chord_segments[0]["onset"]
        rh_events.append(Event(onset=0.0, duration=gap, notes=[]))
        lh_events.append(Event(onset=0.0, duration=gap, notes=[]))

    # 2. 3/4 Waltz
    if round(measure_beats, 1) == 3.0 or style == "waltz":
        for seg in chord_segments:
            s_onset = seg["onset"]
            s_dur = seg["duration"]
            v = seg["voicing"]
            b = seg["bass"]
            chord = seg["chord"]
            lh_comp = get_lh_pattern_pitches(b, chord)

            if difficulty == "beginner":
                rh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=p) for p in v]))
                lh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=b)]))
            else:
                # Oom-pah-pah: LH on 1, RH chords on 2 and 3
                if s_dur >= 3.0:
                    lh_events.append(Event(onset=s_onset, duration=1.0, notes=[Note(midi=b)]))
                    lh_events.append(Event(onset=s_onset + 1.0, duration=1.0, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 2.0, duration=1.0, notes=[Note(midi=lh_comp["octave"])]))

                    rh_events.append(Event(onset=s_onset, duration=1.0, notes=[]))
                    rh_events.append(Event(onset=s_onset + 1.0, duration=1.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 2.0, duration=1.0, notes=[Note(midi=p) for p in v]))
                else:
                    rh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=p) for p in v]))
                    lh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=b)]))

        return ensure_exact_tiling(rh_events, measure_beats), ensure_exact_tiling(lh_events, measure_beats)

    # 3. 4/4 Ballad / Slow Soul / Pop
    # Check if custom bass_hint applies to LH
    custom_lh_rhythm = None
    if bass_hint and chord_segments:
        custom_lh_rhythm = parse_bass_hint(bass_hint, chord_segments[0]["bass"], tonic_pc, measure_beats)

    for seg_idx, seg in enumerate(chord_segments):
        s_onset = seg["onset"]
        s_dur = seg["duration"]
        v = seg["voicing"]
        b = seg["bass"]
        chord = seg["chord"]
        lh_comp = get_lh_pattern_pitches(b, chord)

        is_last_seg = (seg_idx == len(chord_segments) - 1)

        # ---------------- Beginner ----------------
        if difficulty == "beginner":
            # RH: Hit chord voicing on chord change, hold for duration (max 2 attacks/bar)
            rh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=p) for p in v]))
            # LH: Bass root note on chord change, hold for duration
            lh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=b)]))

        # ---------------- Intermediate ----------------
        elif difficulty == "intermediate":
            # RH comping
            if s_dur >= 4.0:
                if is_sparse:
                    # Sparse: hit on beat 1 only
                    rh_events.append(Event(onset=s_onset, duration=4.0, notes=[Note(midi=p) for p in v]))
                else:
                    # Chords on 1 and 3
                    rh_events.append(Event(onset=s_onset, duration=2.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 2.0, duration=2.0, notes=[Note(midi=p) for p in v]))
            elif s_dur >= 2.0:
                rh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=p) for p in v]))
            else:
                rh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=p) for p in v]))

            # LH accompaniment: root-5-8 broken pattern
            if custom_lh_rhythm is not None and seg_idx == 0 and len(chord_segments) == 1:
                for r_onset, r_dur, r_pitch in custom_lh_rhythm:
                    lh_events.append(Event(onset=r_onset, duration=r_dur, notes=[Note(midi=r_pitch)]))
            else:
                if s_dur >= 4.0:
                    # 1 (0.5), 5 (0.5), 8 (1.0), 1 (0.5), 5 (0.5), 8 (1.0)
                    lh_events.append(Event(onset=s_onset, duration=0.5, notes=[Note(midi=b)]))
                    lh_events.append(Event(onset=s_onset + 0.5, duration=0.5, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 1.0, duration=1.0, notes=[Note(midi=lh_comp["octave"])]))
                    lh_events.append(Event(onset=s_onset + 2.0, duration=0.5, notes=[Note(midi=b)]))
                    lh_events.append(Event(onset=s_onset + 2.5, duration=0.5, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 3.0, duration=1.0, notes=[Note(midi=lh_comp["octave"])]))
                elif s_dur >= 2.0:
                    # 1 (0.5), 5 (0.5), 8 (1.0)
                    lh_events.append(Event(onset=s_onset, duration=0.5, notes=[Note(midi=b)]))
                    lh_events.append(Event(onset=s_onset + 0.5, duration=0.5, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 1.0, duration=s_dur - 1.0, notes=[Note(midi=lh_comp["octave"])]))
                else:
                    lh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=b)]))

        # ---------------- Advanced ----------------
        else:
            # Advanced RH: syncopated comping and fill runs
            if is_last_seg and fill:
                # Fill occurs at the end of the measure (last 1-2 beats)
                if s_dur >= 4.0:
                    rh_events.append(Event(onset=s_onset, duration=1.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 1.0, duration=0.5, notes=[]))
                    rh_events.append(Event(onset=s_onset + 1.5, duration=1.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 2.5, duration=0.5, notes=[]))
                    top_note = v[-1]
                    fill_notes = get_diatonic_fill_notes(top_note, tonic_pc, num_notes=4)
                    for k, fn in enumerate(fill_notes):
                        rh_events.append(Event(onset=s_onset + 3.0 + k * 0.25, duration=0.25, notes=[Note(midi=fn)]))
                elif s_dur >= 2.0:
                    # 1st beat of segment: chord (1.0); 2nd beat: four 16th notes fill
                    rh_events.append(Event(onset=s_onset, duration=1.0, notes=[Note(midi=p) for p in v]))
                    top_note = v[-1]
                    fill_notes = get_diatonic_fill_notes(top_note, tonic_pc, num_notes=4)
                    for k, fn in enumerate(fill_notes):
                        rh_events.append(Event(onset=s_onset + 1.0 + k * 0.25, duration=0.25, notes=[Note(midi=fn)]))
                else:
                    top_note = v[-1]
                    fill_notes = get_diatonic_fill_notes(top_note, tonic_pc, num_notes=4)
                    for k, fn in enumerate(fill_notes):
                        rh_events.append(Event(onset=s_onset + k * 0.25, duration=0.25, notes=[Note(midi=fn)]))
            elif s_dur >= 4.0:
                if is_sparse:
                    # Sparser advanced: comp on 1 and 3
                    rh_events.append(Event(onset=s_onset, duration=2.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 2.0, duration=2.0, notes=[Note(midi=p) for p in v]))
                else:
                    # Full syncopated comping: 1 (1.0), rest (0.5), 2& (1.0), rest (0.5), 4 (1.0)
                    rh_events.append(Event(onset=s_onset, duration=1.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 1.0, duration=0.5, notes=[]))
                    rh_events.append(Event(onset=s_onset + 1.5, duration=1.0, notes=[Note(midi=p) for p in v]))
                    rh_events.append(Event(onset=s_onset + 2.5, duration=0.5, notes=[]))
                    rh_events.append(Event(onset=s_onset + 3.0, duration=1.0, notes=[Note(midi=p) for p in v]))
            elif s_dur >= 2.0:
                rh_events.append(Event(onset=s_onset, duration=1.0, notes=[Note(midi=p) for p in v]))
                rh_events.append(Event(onset=s_onset + 1.0, duration=s_dur - 1.0, notes=[Note(midi=p) for p in v]))
            else:
                rh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=p) for p in v]))

            # Advanced LH: wider arpeggio (1-5-1'-2'/3')
            if custom_lh_rhythm is not None and seg_idx == 0 and len(chord_segments) == 1:
                for r_onset, r_dur, r_pitch in custom_lh_rhythm:
                    lh_events.append(Event(onset=r_onset, duration=r_dur, notes=[Note(midi=r_pitch)]))
            else:
                if s_dur >= 4.0:
                    # 1-5-1'-10th arpeggio
                    lh_events.append(Event(onset=s_onset, duration=0.5, notes=[Note(midi=b)]))
                    lh_events.append(Event(onset=s_onset + 0.5, duration=0.5, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 1.0, duration=0.5, notes=[Note(midi=lh_comp["octave"])]))
                    lh_events.append(Event(onset=s_onset + 1.5, duration=0.5, notes=[Note(midi=lh_comp["tenth"])]))
                    lh_events.append(Event(onset=s_onset + 2.0, duration=0.5, notes=[Note(midi=lh_comp["octave"])]))
                    lh_events.append(Event(onset=s_onset + 2.5, duration=0.5, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 3.0, duration=1.0, notes=[Note(midi=b)]))
                elif s_dur >= 2.0:
                    lh_events.append(Event(onset=s_onset, duration=0.5, notes=[Note(midi=b)]))
                    lh_events.append(Event(onset=s_onset + 0.5, duration=0.5, notes=[Note(midi=lh_comp["fifth"])]))
                    lh_events.append(Event(onset=s_onset + 1.0, duration=0.5, notes=[Note(midi=lh_comp["octave"])]))
                    lh_events.append(Event(onset=s_onset + 1.5, duration=s_dur - 1.5, notes=[Note(midi=lh_comp["tenth"])]))
                else:
                    lh_events.append(Event(onset=s_onset, duration=s_dur, notes=[Note(midi=b)]))

    return ensure_exact_tiling(rh_events, measure_beats), ensure_exact_tiling(lh_events, measure_beats)
