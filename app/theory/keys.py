"""Key handling and enharmonic spelling for music theory."""
from __future__ import annotations

import re
from typing import Optional

# Standard pitch classes: C=0, C#=1, D=2, ..., B=11
KEY_NAME_TO_PC: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}

# Diatonic major scale note names (degrees 1 to 7) for standard keys
DIATONIC_SCALES: dict[str, list[str]] = {
    "C": ["C", "D", "E", "F", "G", "A", "B"],
    "G": ["G", "A", "B", "C", "D", "E", "F#"],
    "D": ["D", "E", "F#", "G", "A", "B", "C#"],
    "A": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
    "C#": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
    "F": ["F", "G", "A", "Bb", "C", "D", "E"],
    "Bb": ["Bb", "C", "D", "Eb", "F", "G", "A"],
    "Eb": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "Ab": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "Db": ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"],
    "Gb": ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"],
}

# Default canonical key name for each pitch class 0..11 in Taiwanese charts
# Note: F# (pc 6) is standard in Taiwanese pop (e.g. 掉了), Ab (pc 8), Bb (pc 10), Eb (pc 3)
PC_TO_DEFAULT_KEY_NAME: dict[int, str] = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}

# Semitone offsets from tonic for major scale degrees 1..7 (0-indexed: degree - 1)
DEGREE_SEMITONES: list[int] = [0, 2, 4, 5, 7, 9, 11]


def normalize_key_string(name: str) -> str:
    """Normalize key name, handling Chinese prefixes (降/升) and suffixes (調/调/大調/major)."""
    s = name.strip()
    # Normalize full-width characters
    trans = str.maketrans({
        "＃": "#", "♯": "#",
        "♭": "b",
        "（": "", "）": "", "(": "", ")": "",
    })
    s = s.translate(trans)
    # Remove common suffixes
    s = re.sub(r"(?:大調|大调|小調|小调|調|调|major|minor|maj|min)$", "", s, flags=re.IGNORECASE).strip()

    # Handle Chinese prefixes: 降B -> Bb, 升F -> F#
    s = re.sub(r"^降([A-Ga-g])", r"\1b", s)
    s = re.sub(r"^升([A-Ga-g])", r"\1#", s)
    # Handle postfix accidental: bB -> Bb, #F -> F#
    s = re.sub(r"^([b#])([A-Ga-g])", r"\2\1", s)

    return s.strip()


def key_name_to_pc(name: str) -> int:
    """Convert a key name string to pitch class 0..11.

    Accepts 'C', 'F#', 'Gb', 'Bb', 'Ab', '降B', '升F', 'C調', 'F#大調', etc.
    Raises ValueError on invalid key name.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Invalid key name: {name!r}")

    normalized = normalize_key_string(name)
    if not normalized:
        raise ValueError(f"Empty key name after normalization: {name!r}")

    key_upper = normalized.upper()
    if key_upper in KEY_NAME_TO_PC:
        return KEY_NAME_TO_PC[key_upper]

    raise ValueError(f"Cannot parse key name: {name!r} (normalized: {normalized!r})")


def canonical_key_for_pc(tonic_pc: int, key_name: Optional[str] = None) -> str:
    """Return the canonical key name for a tonic pitch class, respecting key_name if provided."""
    tonic_pc = tonic_pc % 12
    if key_name:
        try:
            norm = normalize_key_string(key_name)
            norm_titled = norm.capitalize() if len(norm) <= 2 else norm
            if norm_titled in DIATONIC_SCALES and KEY_NAME_TO_PC.get(norm.upper()) == tonic_pc:
                return norm_titled
        except Exception:
            pass
    return PC_TO_DEFAULT_KEY_NAME.get(tonic_pc, "C")


def apply_accidental_to_note_name(note_name: str, accidental: int) -> str:
    """Modify a note name by an accidental offset (-1 = flat, 0 = natural, +1 = sharp)."""
    if accidental == 0:
        return note_name
    elif accidental == -1:
        if note_name.endswith("#"):
            return note_name[:-1]
        elif note_name.endswith("b"):
            return note_name + "b"
        else:
            return note_name + "b"
    elif accidental == 1:
        if note_name.endswith("b"):
            return note_name[:-1]
        elif note_name.endswith("#"):
            return note_name + "#"
        else:
            return note_name + "#"
    elif accidental == -2:
        return apply_accidental_to_note_name(apply_accidental_to_note_name(note_name, -1), -1)
    elif accidental == 2:
        return apply_accidental_to_note_name(apply_accidental_to_note_name(note_name, 1), 1)
    return note_name


def resolve_degree(
    degree: int,
    accidental: int = 0,
    tonic_pc: int = 0,
    key_name: Optional[str] = None,
) -> tuple[str, int]:
    """Resolve a scale degree (1..7) with accidental to (note_name, pitch_class).

    Example in key C:
      (5, 0) -> ('G', 7)
      (7, -1) -> ('Bb', 10)
      (1, 1) -> ('C#', 1)
      (6, -1) -> ('Ab', 8)
    Example in key F#:
      (4, 0) -> ('B', 11)
      (7, -1) -> ('E', 4)
    Example in key F:
      (4, 0) -> ('Bb', 10)
    """
    if not (1 <= degree <= 7):
        raise ValueError(f"Degree must be 1..7, got {degree}")

    canonical_key = canonical_key_for_pc(tonic_pc, key_name)
    scale = DIATONIC_SCALES.get(canonical_key)
    if not scale:
        scale = DIATONIC_SCALES["C"]

    diatonic_note = scale[degree - 1]
    spelled_note = apply_accidental_to_note_name(diatonic_note, accidental)

    # Compute pitch class
    base_semitone = DEGREE_SEMITONES[degree - 1]
    pc = (tonic_pc + base_semitone + accidental) % 12

    return spelled_note, pc


def spell(pc: int, tonic_pc: int = 0, key_name: Optional[str] = None) -> str:
    """Spell a pitch class with the correct enharmonic name for the key.

    Flat keys spell with flats; chromatic degrees in C: b3, b6, b7 flats, #1, #4, #5 sharps.
    """
    pc = pc % 12
    tonic_pc = tonic_pc % 12
    canonical_key = canonical_key_for_pc(tonic_pc, key_name)

    # Check if pc matches any diatonic scale degree exactly
    scale = DIATONIC_SCALES.get(canonical_key, DIATONIC_SCALES["C"])
    for d, semitones in enumerate(DEGREE_SEMITONES):
        if (tonic_pc + semitones) % 12 == pc:
            return scale[d]

    # Chromatic note: interval from tonic
    interval = (pc - tonic_pc) % 12

    # Chromatic spelling rules:
    # Flat keys (or canonical key with flats) prefer flats
    is_flat_key = canonical_key in {"F", "Bb", "Eb", "Ab", "Db", "Gb"}
    if is_flat_key:
        flat_map = {
            1: (2, -1),   # b2
            3: (3, -1),   # b3
            6: (5, -1),   # b5
            8: (6, -1),   # b6
            10: (7, -1),  # b7
        }
        if interval in flat_map:
            deg, acc = flat_map[interval]
            note, _ = resolve_degree(deg, acc, tonic_pc, canonical_key)
            return note
    else:
        # Sharp keys or C: b3, b6, b7 flats; #1, #4, #5 sharps
        c_map = {
            1: (1, 1),    # #1 (C#)
            3: (3, -1),   # b3 (Eb)
            6: (4, 1),    # #4 (F#)
            8: (6, -1),   # b6 (Ab)
            10: (7, -1),  # b7 (Bb)
        }
        if interval in c_map:
            deg, acc = c_map[interval]
            note, _ = resolve_degree(deg, acc, tonic_pc, canonical_key)
            return note

    # Fallback to default pitch class name
    pitch_names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_names_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return pitch_names_flat[pc] if is_flat_key else pitch_names_sharp[pc]
