"""Independent reference for chord pitch classes and defining tones using music21."""
from __future__ import annotations

import re
from typing import Optional, Union

import music21.harmony
from app.models import ResolvedChord


def normalize_to_music21_syntax(name: str) -> tuple[str, str]:
    """Normalize Taiwanese / app chord names to music21 ChordSymbol syntax.

    Handles:
      - flats: 'Bb' -> 'B-', 'Eb' -> 'E-', 'Ab' -> 'A-', etc.
      - sharps: 'F#' -> 'F#'
      - slash chords: 'G/B' -> 'G/B', 'F/Bb' -> 'F/B-', 'Cm6/G' -> 'Cm6/G'
      - chord types: 'maj9' -> 'M9', 'm7b5' -> 'm7b5', '7sus4' -> '7sus4',
                     'Cadd9' -> 'Cadd9', 'Fmaj7' -> 'Fmaj7'

    Returns:
      (m21_root_chord, m21_full_name)
    """
    parts = name.split("/")
    root_part = parts[0].strip()
    bass_part = parts[1].strip() if len(parts) > 1 else None

    # Parse root letter and accidental
    if not root_part:
        raise ValueError(f"Empty chord name: {name!r}")

    r_letter = root_part[0].upper()
    rem = root_part[1:]
    if rem.startswith("b"):
        r_acc = "-"
        suffix = rem[1:]
    elif rem.startswith("#"):
        r_acc = "#"
        suffix = rem[1:]
    else:
        r_acc = ""
        suffix = rem

    # Handle alterations/syntax specific to music21
    if suffix == "maj9":
        suffix = "M9"

    m21_root_chord = f"{r_letter}{r_acc}{suffix}"

    if bass_part:
        b_letter = bass_part[0].upper()
        b_rem = bass_part[1:]
        if b_rem.startswith("b"):
            b_acc = "-"
            b_suffix = b_rem[1:]
        elif b_rem.startswith("#"):
            b_acc = "#"
            b_suffix = b_rem[1:]
        else:
            b_acc = ""
            b_suffix = b_rem
        m21_bass = f"{b_letter}{b_acc}{b_suffix}"
        m21_full = f"{m21_root_chord}/{m21_bass}"
    else:
        m21_full = m21_root_chord

    return m21_root_chord, m21_full


def get_chord_reference(
    chord_or_name: Union[str, ResolvedChord],
) -> tuple[int, int, list[int], list[int]]:
    """Return independent reference (root_pc, bass_pc, pcs, defining_pcs) from chord name.

    Defining tones:
      - 3rd (or 4th for sus)
      - 7th if present
      - added 9th / 2nd for add9 / 9 chords
      - 6th for 6 chords
      - b5 for m7b5 / dim
    """
    name = chord_or_name.name if isinstance(chord_or_name, ResolvedChord) else str(chord_or_name)
    m21_root_chord, m21_full = normalize_to_music21_syntax(name)

    cs_full = music21.harmony.ChordSymbol(m21_full)
    cs_root = music21.harmony.ChordSymbol(m21_root_chord)

    root_pc = int(cs_full.root().pitchClass)
    bass_pc = int(cs_full.bass().pitchClass)

    # Chord pitch classes (root-first unique pcs of the harmony)
    pcs: list[int] = []
    for p in cs_root.pitches:
        pc = int(p.pitchClass)
        if pc not in pcs:
            pcs.append(pc)

    # Defining tones
    defining: list[int] = []

    # 1. 3rd (or 4th for sus)
    if "sus" in m21_root_chord:
        fourth = cs_root.getChordStep(4)
        def_pitch = int(fourth.pitchClass) if fourth else (root_pc + 5) % 12
        defining.append(def_pitch)
    else:
        third = cs_root.third
        def_pitch = int(third.pitchClass) if third else (root_pc + (3 if "m" in m21_root_chord else 4)) % 12
        defining.append(def_pitch)

    # 2. 7th if present
    seventh = cs_root.seventh
    if seventh is not None:
        defining.append(int(seventh.pitchClass))

    # 3. added 9th / 2nd for add9 and 9th chords
    if "add9" in m21_root_chord or any(k in m21_root_chord for k in ("9", "M9", "m9")):
        ninth = cs_root.getChordStep(9) or cs_root.getChordStep(2)
        if ninth is not None:
            defining.append(int(ninth.pitchClass))
        else:
            defining.append((root_pc + 2) % 12)

    # 4. 6th for 6 chords
    if "6" in m21_root_chord:
        sixth = cs_root.getChordStep(6)
        if sixth is not None:
            defining.append(int(sixth.pitchClass))
        else:
            defining.append((root_pc + 9) % 12)

    # 5. b5 for m7b5 / dim
    if "m7b5" in m21_root_chord or "dim" in m21_root_chord:
        fifth = cs_root.fifth
        if fifth is not None and (fifth.pitchClass - root_pc) % 12 == 6:
            defining.append(int(fifth.pitchClass))
        else:
            defining.append((root_pc + 6) % 12)

    defining_pcs = list(dict.fromkeys(defining))
    return root_pc, bass_pc, pcs, defining_pcs


def cross_check_chord_pcs(chord: ResolvedChord) -> tuple[bool, Optional[str]]:
    """Cross-check ResolvedChord.pcs vs music21 reference.

    Returns (is_match, mismatch_reason).
    """
    try:
        ref_root, ref_bass, ref_pcs, _ = get_chord_reference(chord.name)
    except Exception as e:
        return False, f"music21 parse error for {chord.name!r}: {e}"

    if ref_root != chord.root_pc:
        return False, f"Root mismatch for {chord.name!r}: music21={ref_root} vs resolved={chord.root_pc}"
    if ref_bass != chord.bass_pc:
        return False, f"Bass mismatch for {chord.name!r}: music21={ref_bass} vs resolved={chord.bass_pc}"

    if set(ref_pcs) != set(chord.pcs):
        return False, f"Pitch class mismatch for {chord.name!r}: music21={set(ref_pcs)} vs resolved={set(chord.pcs)}"

    return True, None
