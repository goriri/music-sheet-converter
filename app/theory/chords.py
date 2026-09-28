"""Parsing and resolution of Taiwanese number notation chords."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from app.models import ResolvedChord
from app.theory.keys import (
    canonical_key_for_pc,
    key_name_to_pc,
    resolve_degree,
    spell,
)

# Relative intervals from root for normalized qualities (semitones)
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim": [0, 3, 6, 9],
    "aug": [0, 4, 8],
    "sus4": [0, 5, 7],
    "7sus4": [0, 5, 7, 10],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "add9": [0, 4, 7, 2],
    "9": [0, 4, 7, 10, 2],
    "m9": [0, 3, 7, 10, 2],
    "maj9": [0, 4, 7, 11, 2],
}

# Display suffixes for chord names
QUALITY_SUFFIXES: dict[str, str] = {
    "maj": "",
    "m": "m",
    "7": "7",
    "maj7": "maj7",
    "m7": "m7",
    "m7b5": "m7b5",
    "dim": "dim7",
    "aug": "aug",
    "sus4": "sus4",
    "7sus4": "7sus4",
    "6": "6",
    "m6": "m6",
    "add9": "add9",
    "9": "9",
    "m9": "m9",
    "maj9": "maj9",
}


@dataclass(frozen=True)
class ChordSpec:
    """Parsed representation of a Taiwanese number-notation chord."""

    raw: str
    degree: int  # 1..7
    accidental: int  # -1 for b, 0 for natural, +1 for #
    quality: str  # normalized quality string
    bass_degree: Optional[int] = None
    bass_accidental: Optional[int] = None


def clean_raw_chord(raw: str) -> str:
    """Normalize OMR noise, whitespace, full-width characters, and superscripts."""
    if not raw or not isinstance(raw, str):
        return ""

    s = raw.strip()
    # Remove outer surrounding quotes or brackets if any (e.g. '"5sus"', '[5/7]')
    s = s.strip("\"'“”‘’[]{}")

    # Character translation table for full-width and superscripts
    trans = str.maketrans({
        # Full-width digits
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        # Full-width Latin letters
        "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G",
        "Ｍ": "M", "ａ": "a", "ｂ": "b", "ｄ": "d", "ｇ": "g", "ｉ": "i", "ｊ": "j",
        "ｍ": "m", "ｏ": "o", "ｓ": "s", "ｕ": "u",
        # Symbols and accidentals
        "＃": "#", "♯": "#", "升": "#",
        "♭": "b", "降": "b",
        "／": "/", "＼": "/",
        "（": "(", "）": ")",
        "－": "-", "—": "-", "–": "-", "﹣": "-", "−": "-",
        "＋": "+",
        "°": "o", "º": "o",
        # Superscripts
        "⁷": "7", "²": "2", "⁹": "9", "⁴": "4", "⁵": "5",
        "⁶": "6", "³": "3", "¹": "1", "⁰": "0",
        "⁺": "+", "⁻": "-", "ᵒ": "o",
    })
    s = s.translate(trans)

    # Remove all internal whitespace around symbols and within chord
    s = re.sub(r"\s+", "", s)

    return s


def parse_accidental_and_degree(token: str) -> tuple[int, int, str]:
    """Parse accidental and degree (1..7) from the beginning of a token.

    Supports accidental before (e.g. 'b7', '#1') or after (e.g. '7b', '1#').
    Returns (degree, accidental, remainder).
    """
    if not token:
        raise ValueError("Empty chord token")

    # Case 1: Accidental before degree: 'b7', '#1', etc.
    m_before = re.match(r"^([b#])([1-7])(.*)$", token)
    if m_before:
        acc_str, deg_str, rem = m_before.groups()
        acc = -1 if acc_str == "b" else 1
        deg = int(deg_str)
        return deg, acc, rem

    # Case 2: Degree first: '7b', '1#', or '1(2)', '5m', '67'
    m_deg = re.match(r"^([1-7])(.*)$", token)
    if m_deg:
        deg_str, rem = m_deg.groups()
        deg = int(deg_str)
        # Check if remainder starts with an accidental: 'b' or '#'
        if rem and rem[0] in "b#":
            acc = -1 if rem[0] == "b" else 1
            return deg, acc, rem[1:]
        return deg, 0, rem

    raise ValueError(f"Cannot identify scale degree (1-7) in token: {token!r}")


def normalize_quality(q_str: str, raw: str) -> str:
    """Normalize raw chord quality string into canonical quality.

    Recognized qualities:
    - maj (from '', 'maj')
    - m (from 'm', 'min', '-')
    - 7 (from '7')
    - maj7 (from 'M7', 'maj7', 'Maj7', '△7')
    - m7 (from 'm7', '-7', 'min7')
    - m7b5 (from 'm7-5', 'm7b5', 'm7(-5)', 'm7(b5)', '-7-5', '-7b5')
    - dim (from 'dim', 'dim7', 'o', 'o7')
    - aug (from 'aug', '+')
    - sus4 (from 'sus', 'sus4')
    - 7sus4 (from '7sus', '7sus4')
    - 6 (from '6')
    - m6 (from 'm6', '-6')
    - add9 (from '(2)', '2', 'add2', '(9)', 'add9')
    - 9 (from '9')
    - m9 (from 'm9', '-9')
    - maj9 (from 'M9', 'maj9', 'Maj9')
    """
    q = q_str.strip()
    if q in ("", "maj", "Maj"):
        return "maj"

    # Explicit lookup order: most specific first
    if q in ("m7-5", "m7b5", "m7(-5)", "m7(b5)", "-7-5", "-7b5", "m7-5", "min7b5"):
        return "m7b5"
    if q in ("7sus4", "7sus"):
        return "7sus4"
    if q in ("sus4", "sus"):
        return "sus4"
    if q in ("maj7", "M7", "Maj7", "△7", "△"):
        return "maj7"
    if q in ("maj9", "M9", "Maj9"):
        return "maj9"
    if q in ("m7", "-7", "min7"):
        return "m7"
    if q in ("m6", "-6", "min6"):
        return "m6"
    if q in ("m9", "-9", "min9"):
        return "m9"
    if q in ("(2)", "2", "add2", "(9)", "add9", "9add"):
        return "add9"
    if q in ("dim", "dim7", "o", "o7"):
        return "dim"
    if q in ("aug", "+", "aug7"):
        return "aug"
    if q == "6":
        return "6"
    if q == "7":
        return "7"
    if q == "9":
        return "9"
    if q in ("m", "-", "min"):
        return "m"

    raise ValueError(f"Unrecognized chord quality {q_str!r} in chord {raw!r}")


def parse_chord(raw: str) -> ChordSpec:
    """Parse a Taiwanese number-notation chord symbol into ChordSpec.

    Examples:
      '5/7'    -> deg=5, qual='maj', bass_deg=7
      '5m/7b'  -> deg=5, qual='m', bass_deg=7, bass_acc=-1
      '67/1#'  -> deg=6, qual='7', bass_deg=1, bass_acc=1
      '6m7-5'  -> deg=6, qual='m7b5'
      '57sus'  -> deg=5, qual='7sus4'
      '1(2)'   -> deg=1, qual='add9'
      '4M7'    -> deg=4, qual='maj7'
      '7b'     -> deg=7, acc=-1, qual='maj'
      '6b'     -> deg=6, acc=-1, qual='maj'

    Raises ValueError with a clear message on garbage or unparseable input.
    """
    cleaned = clean_raw_chord(raw)
    if not cleaned:
        raise ValueError(f"Cannot parse chord from empty string: {raw!r}")

    # Check for slash bass
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) != 2:
            raise ValueError(f"Multiple slashes in chord: {raw!r}")
        root_part, bass_part = parts[0], parts[1]
        if not bass_part:
            raise ValueError(f"Missing bass degree after slash in chord: {raw!r}")

        bass_deg, bass_acc, bass_rem = parse_accidental_and_degree(bass_part)
        if bass_rem:
            raise ValueError(f"Trailing garbage in slash bass: {bass_part!r} in {raw!r}")
    else:
        root_part = cleaned
        bass_deg = None
        bass_acc = None

    deg, acc, q_part = parse_accidental_and_degree(root_part)
    quality = normalize_quality(q_part, raw)

    return ChordSpec(
        raw=raw,
        degree=deg,
        accidental=acc,
        quality=quality,
        bass_degree=bass_deg,
        bass_accidental=bass_acc,
    )


def format_chord_name(root_name: str, quality: str) -> str:
    """Format root note name and quality into standard chord name."""
    suffix = QUALITY_SUFFIXES.get(quality, quality)
    return f"{root_name}{suffix}"


def get_chord_pcs(root_pc: int, quality: str) -> list[int]:
    """Return pitch classes for chord tones with root first."""
    intervals = QUALITY_INTERVALS.get(quality, QUALITY_INTERVALS["maj"])
    return [(root_pc + i) % 12 for i in intervals]


def resolve_chord(
    raw: str,
    tonic_pc: int,
    beat: float = 1.0,
    key_name: Optional[str] = None,
) -> ResolvedChord:
    """Resolve a Taiwanese number-notation chord symbol into a concrete ResolvedChord.

    Spells the chord and slash bass with enharmonically correct note names for the target key.
    """
    spec = parse_chord(raw)

    root_name, root_pc = resolve_degree(spec.degree, spec.accidental, tonic_pc, key_name)

    if spec.bass_degree is not None:
        bass_acc = spec.bass_accidental if spec.bass_accidental is not None else 0
        bass_name, bass_pc = resolve_degree(spec.bass_degree, bass_acc, tonic_pc, key_name)
    else:
        bass_name, bass_pc = root_name, root_pc

    root_chord_name = format_chord_name(root_name, spec.quality)
    if spec.bass_degree is not None:
        full_name = f"{root_chord_name}/{bass_name}"
    else:
        full_name = root_chord_name

    pcs = get_chord_pcs(root_pc, spec.quality)

    return ResolvedChord(
        raw=raw,
        name=full_name,
        beat=beat,
        root_pc=root_pc,
        bass_pc=bass_pc,
        pcs=pcs,
        quality=spec.quality,
    )
