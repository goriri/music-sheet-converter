"""Piano fingering assignment for chords and accompaniment patterns."""
from __future__ import annotations

from typing import Optional
from app.models import Arrangement, Event, Note

BLACK_KEY_PCS = {1, 3, 6, 8, 10}


def assign_rh_chord_fingers(pitches: list[int]) -> list[int]:
    """Assign monotonic fingers 1..5 for a simultaneous RH chord (pitches sorted ascending)."""
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
        # Close second on bottom (e.g. add9 cluster like G4 A4 C5)
        if i1 <= 2:
            return [1, 2, 5] if i2 >= 4 else [1, 2, 4]
        # Close second on top (e.g. C5 F5 G5)
        if i2 <= 2:
            return [1, 3, 5] if i1 >= 4 else [1, 2, 4]
        # Root position triad: 3rd on bottom, 3rd on top -> 1-3-5
        if i1 <= 4 and i2 <= 4:
            return [1, 3, 5]
        # 1st inversion: 3rd on bottom, 4th on top -> 1-2-5
        if i1 <= 4 and i2 >= 5:
            return [1, 2, 5]
        # 2nd inversion: 4th on bottom, 3rd on top -> 1-3-5
        if i1 >= 5 and i2 <= 4:
            return [1, 3, 5]
        # General: wider gap at top -> 1-2-5, wider at bottom -> 1-3-5
        return [1, 2, 5] if i2 > i1 else [1, 3, 5]

    if k == 4:
        i1 = pitches[1] - pitches[0]
        i2 = pitches[2] - pitches[1]
        i3 = pitches[3] - pitches[2]
        # If upper interval is larger, gap is at top -> 1-2-3-5
        # If lower interval is larger, gap is at bottom -> 1-2-4-5
        if i3 > i1:
            return [1, 2, 3, 5]
        elif i1 > i3:
            return [1, 2, 4, 5]
        else:
            return [1, 2, 3, 5]

    # k >= 5: use all 5 fingers
    return [1, 2, 3, 4, 5][:k]


def assign_lh_chord_fingers(pitches: list[int]) -> list[int]:
    """Assign strictly decreasing fingers 5..1 for simultaneous LH notes (pitches sorted ascending)."""
    k = len(pitches)
    if k == 0:
        return []
    if k == 1:
        return [5]
    if k == 2:
        span = pitches[1] - pitches[0]
        if span >= 8:
            return [5, 1]
        elif span >= 5:
            return [5, 2]
        return [5, 3]

    if k == 3:
        span = pitches[2] - pitches[0]
        if span >= 14:
            return [5, 3, 1]  # 1-5-10
        elif span >= 11:
            return [5, 2, 1]  # 1-5-8
        return [5, 3, 1]  # close triad

    if k == 4:
        return [5, 4, 2, 1]

    # k >= 5: 5, 4, 3, 2, 1
    return [5, 4, 3, 2, 1][:k]


def assign_rh_broken_fingers(notes: list[Note], hand_voicing: Optional[tuple[int, ...]] = None) -> None:
    """Assign RH fingers for a broken pattern, aligning with block chord hand position or sequential runs."""
    if not notes:
        return

    # Check if this is a sequential descending run (e.g. 16th fill notes)
    k = len(notes)
    if k >= 2 and all(notes[i].midi > notes[i + 1].midi for i in range(k - 1)):
        if k == 4:
            desc_fingers = [4, 3, 2, 1]
        elif k <= 5:
            desc_fingers = list(range(k, 0, -1))
        else:
            desc_fingers = [5] * k
        for note, f in zip(notes, desc_fingers):
            if note.finger is None:
                note.finger = f
        return

    if k >= 2 and all(notes[i].midi < notes[i + 1].midi for i in range(k - 1)):
        asc_fingers = list(range(1, min(k, 5) + 1))
        for note, f in zip(notes, asc_fingers):
            if note.finger is None:
                note.finger = f
        return

    # If an underlying hand voicing is provided, match pitch to block chord finger
    pitch_to_finger: dict[int, int] = {}
    if hand_voicing:
        sorted_voicing = sorted(hand_voicing)
        voicing_fingers = assign_rh_chord_fingers(sorted_voicing)
        pitch_to_finger = dict(zip(sorted_voicing, voicing_fingers))

    for note in notes:
        if note.finger is not None:
            continue
        if note.midi in pitch_to_finger:
            note.finger = pitch_to_finger[note.midi]
        else:
            # Fallback finger based on register
            if note.midi <= 67:
                note.finger = 1
            elif note.midi <= 71:
                note.finger = 2
            elif note.midi <= 74:
                note.finger = 3
            elif note.midi <= 77:
                note.finger = 4
            else:
                note.finger = 5


def assign_lh_broken_pattern_fingering(
    events: list[Event],
    bass_pitch: int,
    pattern_type: str = "root_5_8",
) -> None:
    """Assign fingers to sequential single-note LH broken accompaniment events."""
    single_note_events = [ev for ev in events if len(ev.notes) == 1]

    # Check for standard 7-note wide arpeggio: 1-5-8-10-8-5-1
    if pattern_type == "arpeggio_wide" and len(single_note_events) == 7:
        intervals = [ev.notes[0].midi - bass_pitch for ev in single_note_events]
        if intervals[0] == 0 and 4 <= intervals[1] <= 9 and 10 <= intervals[2] <= 14 and intervals[3] >= 14:
            crossover_fingers = [5, 2, 1, 3, 1, 2, 5]
            for ev, f in zip(single_note_events, crossover_fingers):
                if ev.notes[0].finger is None:
                    ev.notes[0].finger = f

    # Check for standard 4-note wide arpeggio: 1-5-8-10
    if pattern_type == "arpeggio_wide" and len(single_note_events) == 4:
        intervals = [ev.notes[0].midi - bass_pitch for ev in single_note_events]
        if intervals[0] == 0 and 4 <= intervals[1] <= 9 and 10 <= intervals[2] <= 14 and intervals[3] >= 14:
            crossover_fingers = [5, 2, 1, 3]
            for ev, f in zip(single_note_events, crossover_fingers):
                if ev.notes[0].finger is None:
                    ev.notes[0].finger = f

    for ev in events:
        if not ev.notes:
            continue
        if len(ev.notes) == 1:
            note = ev.notes[0]
            if note.finger is not None:
                continue

            interval_from_bass = note.midi - bass_pitch

            if interval_from_bass <= 0:
                # Bass / root note
                note.finger = 5
            elif 1 <= interval_from_bass <= 3:
                # Passing tone / degree 2 in bass hint
                note.finger = 4
            elif 4 <= interval_from_bass <= 9:
                # 5th (or 6th in slash chords like Cm/Eb)
                note.finger = 2
            elif 10 <= interval_from_bass <= 13:
                # Octave
                note.finger = 1
            elif interval_from_bass >= 14:
                # 10th / upper extension: crossover 3 in wide arpeggios, else 1
                note.finger = 3 if pattern_type == "arpeggio_wide" else 1
            else:
                note.finger = 5
        else:
            # Chord event in LH
            sorted_notes = sorted(ev.notes, key=lambda n: n.midi)
            pitches = [n.midi for n in sorted_notes]
            fingers = assign_lh_chord_fingers(pitches)
            for n, f in zip(sorted_notes, fingers):
                n.finger = f


def assign_fingering_for_measure(
    rh_events: list[Event],
    lh_events: list[Event],
    rh_voicing: Optional[tuple[int, ...]] = None,
    bass_pitch: Optional[int] = None,
    lh_pattern_type: str = "root_5_8",
    chord_basses: Optional[list[tuple[float, int]]] = None,
) -> None:
    """Assign fingers to all notes in RH and LH events of a measure."""
    # RH fingering: handle chords and collect contiguous single-note event runs
    current_run: list[Note] = []
    for ev in rh_events:
        if len(ev.notes) == 1:
            current_run.append(ev.notes[0])
        else:
            if current_run:
                assign_rh_broken_fingers(current_run, rh_voicing)
                current_run = []
            if len(ev.notes) > 1:
                sorted_notes = sorted(ev.notes, key=lambda n: n.midi)
                pitches = [n.midi for n in sorted_notes]
                fingers = assign_rh_chord_fingers(pitches)
                for n, f in zip(sorted_notes, fingers):
                    n.finger = f

    if current_run:
        assign_rh_broken_fingers(current_run, rh_voicing)

    # LH fingering: per-chord bass segmenting if available
    if chord_basses:
        sorted_cb = sorted(chord_basses, key=lambda x: x[0])
        for idx, (c_onset, c_bass) in enumerate(sorted_cb):
            next_onset = sorted_cb[idx + 1][0] if idx + 1 < len(sorted_cb) else 999.0
            seg_events = [ev for ev in lh_events if c_onset - 1e-4 <= ev.onset < next_onset - 1e-4]
            assign_lh_broken_pattern_fingering(seg_events, c_bass, lh_pattern_type)
    elif bass_pitch is not None:
        assign_lh_broken_pattern_fingering(lh_events, bass_pitch, lh_pattern_type)
    else:
        for ev in lh_events:
            if not ev.notes:
                continue
            if len(ev.notes) == 1:
                ev.notes[0].finger = 5
            else:
                sorted_notes = sorted(ev.notes, key=lambda n: n.midi)
                pitches = [n.midi for n in sorted_notes]
                fingers = assign_lh_chord_fingers(pitches)
                for n, f in zip(sorted_notes, fingers):
                    n.finger = f

    # Final safety pass: ensure every note has a valid finger 1..5
    for ev in rh_events:
        for n in ev.notes:
            if n.finger is None or not (1 <= n.finger <= 5):
                n.finger = 1
    for ev in lh_events:
        for n in ev.notes:
            if n.finger is None or not (1 <= n.finger <= 5):
                n.finger = 5


def assign_arrangement_fingering(arr: Arrangement) -> None:
    """Assign fingers to every note in an Arrangement and validate playability."""
    for m_arr in arr.measures:
        # Check if there is an RH chord voicing in the measure
        rh_voicing: Optional[tuple[int, ...]] = None
        for ev in m_arr.rh:
            if len(ev.notes) > 1:
                rh_voicing = tuple(sorted(n.midi for n in ev.notes))
                break

        # Check bass pitch in measure
        bass_pitch: Optional[int] = None
        for ev in m_arr.lh:
            if ev.notes:
                bass_pitch = min(n.midi for n in ev.notes)
                break

        assign_fingering_for_measure(
            m_arr.rh,
            m_arr.lh,
            rh_voicing=rh_voicing,
            bass_pitch=bass_pitch,
            lh_pattern_type="arpeggio_wide" if arr.difficulty == "advanced" else "root_5_8",
        )
