"""Parsing and resolution of Taiwanese number notation chords."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from app.models import ChordSymbol, ResolvedChord
from app.theory.keys import (
    KEY_NAME_TO_PC,
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
    "7b9": [0, 4, 7, 10, 1],
    "7#9": [0, 4, 7, 10, 3],
    "7#5": [0, 4, 8, 10],
    "7add13": [0, 4, 7, 10, 9],
    "11": [0, 4, 7, 10, 2, 5],
    "13": [0, 4, 7, 10, 2, 5, 9],
    "mM7": [0, 3, 7, 11],
    "sus2": [0, 2, 7],
    "69": [0, 4, 7, 9, 2],
    "madd9": [0, 3, 7, 2],
}

# Non-standard or auxiliary quality interval aliases
EXTRA_QUALITY_INTERVALS: dict[str, list[int]] = {
    "69": [0, 4, 7, 9, 2],
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
    "madd9": "madd9",
    "9": "9",
    "m9": "m9",
    "maj9": "maj9",
    "7b9": "7b9",
    "7#9": "7#9",
    "7#5": "7#5",
    "7add13": "7add13",
    "11": "11",
    "13": "13",
    "mM7": "mM7",
    "sus2": "sus2",
    "69": "6/9",
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


@dataclass(frozen=True)
class LetterChordSpec:
    """Parsed representation of a letter-notation chord."""

    raw: str
    root_name: str
    root_pc: int
    quality: str
    bass_name: Optional[str] = None
    bass_pc: Optional[int] = None


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
        # Superscripts and modifications
        "⁷": "7", "²": "2", "⁹": "9", "⁴": "4", "⁵": "5",
        "⁶": "6", "³": "3", "¹": "1", "⁰": "0",
        "⁺": "+", "⁻": "-", "ᵒ": "o",
        "^": "",
        # Circled numbers (standard and filled)
        "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9",
        "❶": "1", "❷": "2", "❸": "3", "❹": "4", "❺": "5", "❻": "6", "❼": "7", "❽": "8", "❾": "9",
        # Fraction / division slashes
        "╱": "/", "⁄": "/", "∕": "/",
    })
    s = s.translate(trans)

    # Remove all internal whitespace around symbols and within chord
    s = re.sub(r"\s+", "", s)

    # Normalize 6/9 to 69 so it does not collide with slash bass (e.g. 16/9, 16/9/5, C6/9, C6/9/G)
    s = re.sub(r"6/9(?![0-9])", "69", s)

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
    - m7b5 (from 'm7-5', 'm7b5', 'm7(-5)', 'm7(b5)', '-7-5', '-7b5', 'm7(9-5)')
    - dim (from 'dim', 'dim7', 'o', 'o7')
    - aug (from 'aug', '+')
    - sus4 (from 'sus', 'sus4')
    - 7sus4 (from '7sus', '7sus4', 'sus7', 'sus47')
    - 6 (from '6')
    - m6 (from 'm6', '-6')
    - add9 (from '(2)', '2', 'add2', '(9)', 'add9', '9add', '(add9)')
    - 9 (from '9', '7(9)')
    - m9 (from 'm9', '-9')
    - maj9 (from 'M9', 'maj9', 'Maj9')
    - 7b9 (from '7b9', '7(b9)', 'b9', '(b9)')
    - 7#9 (from '7#9', '7(#9)', '#9', '(#9)')
    - 7#5 (from '7#5', '7(#5)')
    - 7add13 (from '7add13', '7(13)', '713')
    - 11 (from '11', '(11)', '7(11)')
    - 13 (from '13', '(13)')
    - mM7 (from 'm(maj7)', 'mM7', 'mmaj7')
    - sus2 (from 'sus2')
    - 69 (from '69', '6/9', '6.9', '(6.9)', '(6/9)')
    """
    q = q_str.strip()
    if q in ("", "maj", "Maj"):
        return "maj"

    # Explicit lookup order: most specific first
    if q in ("m7-5", "m7b5", "m7(-5)", "m7(b5)", "-7-5", "-7b5", "min7b5", "m7-5", "min7-5", "m7(9-5)", "m-5", "mb5", "m(-5)", "m(b5)"):
        return "m7b5"
    if q in ("madd9", "minadd9", "m(add9)"):
        return "madd9"
    if q in ("7sus4", "7sus", "sus7", "sus47"):
        return "7sus4"
    if q in ("sus4", "sus"):
        return "sus4"
    if q in ("sus2",):
        return "sus2"
    if q in ("maj7", "M7", "Maj7", "△7", "△"):
        return "maj7"
    if q in ("maj9", "M9", "Maj9"):
        return "maj9"
    if q in ("m(maj7)", "mM7", "m/maj7", "m/M7", "min(maj7)", "mmaj7", "mmaj7(9)", "mmaj7/9"):
        return "mM7"
    if q in ("m7", "-7", "min7"):
        return "m7"
    if q in ("m6", "-6", "min6"):
        return "m6"
    if q in ("m9", "-9", "min9"):
        return "m9"
    if q in ("7b9", "7(b9)", "7(-9)", "7-9", "(b9)", "b9", "(-9)", "-9", "7(9b)", "79b"):
        return "7b9"
    if q in ("7#9", "7(#9)", "7(+9)", "7+9", "(#9)", "#9"):
        return "7#9"
    if q in ("7#5", "7(#5)", "7(+5)", "7+5"):
        return "7#5"
    if q in ("7add13", "7(13)", "713", "7/13"):
        return "7add13"
    if q in ("13", "(13)"):
        return "13"
    if q in ("11", "(11)", "7(11)", "7/11", "7(11-9)", "7/11-9", "11-9", "(11-9)"):
        return "11"
    if q in ("(2)", "2", "add2", "(9)", "add9", "9add", "(add9)", "add(9)", "add(2)"):
        return "add9"
    if q in ("69", "6/9", "6.9", "(6.9)", "(6/9)", "(69)", "6(9)", "6add9", "6(add9)"):
        return "69"
    if q in ("dim", "dim7", "o", "o7"):
        return "dim"
    if q in ("aug", "+", "aug7"):
        return "aug"
    if q == "6":
        return "6"
    if q in ("9", "7(9)", "7/9", "79", "(9)"):
        return "9"
    if q == "7":
        return "7"
    if q in ("m", "-", "min"):
        return "m"

    raise ValueError(f"Unrecognized chord quality {q_str!r} in chord {raw!r}")


def parse_chord(
    raw: str,
    *,
    stacked: bool = False,
    stacked_orientation: str = "bottom_is_bass",
) -> ChordSpec:
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
      '/5'     -> deg=1, qual='maj', bass_deg=5

    Raises ValueError with a clear message on garbage or unparseable input.
    """
    cleaned = clean_raw_chord(raw)
    if not cleaned:
        raise ValueError(f"Cannot parse chord from empty string: {raw!r}")

    if cleaned.startswith("/"):
        raise ValueError(f"Cannot parse chord with empty root degree: {raw!r}")

    # Convert extensions after slash (e.g. 57/9, 57/11, 57/11-9, 17/9, 1/2m7/9-5) to parenthesized extensions
    cleaned = re.sub(r"/((?:b9|#9|#11|9|11|13)[-+b#0-9]*)", r"(\1)", cleaned)

    # Check for slash bass or slash extension
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) != 2:
            raise ValueError(f"Multiple slashes in chord: {raw!r}")
        top_part, bot_part = parts[0], parts[1]
        if not bot_part:
            raise ValueError(f"Missing bass degree after slash in chord: {raw!r}")

        # Decide which part is the chord and which is the bass
        if stacked:
            if stacked_orientation == "top_is_bass":
                chord_part, bass_part = bot_part, top_part
            else:
                chord_part, bass_part = top_part, bot_part
        else:
            # Inline slash: default chord/bass (top is chord, bot is bass).
            # Only invert if bottom part cannot be a bass degree and top part can be
            # (e.g. '1/2m7-5', '1/2m7/9-5').
            bot_is_bass = bool(re.match(r"^[b#]?[1-7][b#]?m?$", bot_part))
            top_is_bass = bool(re.match(r"^[b#]?[1-7][b#]?m?$", top_part))
            if not bot_is_bass and top_is_bass:
                chord_part, bass_part = bot_part, top_part
            else:
                chord_part, bass_part = top_part, bot_part

        # Strip trailing 'm' if present on bass degree (e.g. '1/6m', '5/3m')
        if bass_part.endswith("m") and len(bass_part) > 1:
            bass_part = bass_part[:-1]
        bass_deg, bass_acc, bass_rem = parse_accidental_and_degree(bass_part)
        if bass_rem:
            raise ValueError(f"Trailing garbage in slash bass: {bass_part!r} in {raw!r}")
    else:
        chord_part = cleaned
        bass_deg = None
        bass_acc = None

    deg, acc, q_part = parse_accidental_and_degree(chord_part)
    quality = normalize_quality(q_part, raw)

    return ChordSpec(
        raw=raw,
        degree=deg,
        accidental=acc,
        quality=quality,
        bass_degree=bass_deg,
        bass_accidental=bass_acc,
    )


def parse_letter_chord(raw: str) -> LetterChordSpec:
    """Parse a letter-notation chord symbol into LetterChordSpec.

    Examples:
      'C'       -> root='C', root_pc=0, qual='maj'
      'C#m7'    -> root='C#', root_pc=1, qual='m7'
      'Bb/D'    -> root='Bb', root_pc=10, bass='D', bass_pc=2, qual='maj'
      'Am7b5'   -> root='A', root_pc=9, qual='m7b5'
      'Dm7/G'   -> root='D', root_pc=2, bass='G', bass_pc=7, qual='m7'
    """
    cleaned = clean_raw_chord(raw)
    if not cleaned:
        raise ValueError(f"Cannot parse chord from empty string: {raw!r}")

    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) != 2:
            raise ValueError(f"Multiple slashes in chord: {raw!r}")
        root_part, bass_part = parts[0], parts[1]
        if not bass_part:
            raise ValueError(f"Missing bass after slash in chord: {raw!r}")

        # Check if bass is a letter note A..G
        m_bass = re.match(r"^([A-Ga-g])([b#♭♯]?)$", bass_part)
        if m_bass:
            b_let = m_bass.group(1).upper()
            b_acc_raw = m_bass.group(2)
            b_acc = "b" if b_acc_raw in ("b", "♭") else ("#" if b_acc_raw in ("#", "♯") else "")
            bass_name = f"{b_let}{b_acc}"
            bass_pc = KEY_NAME_TO_PC.get(bass_name.upper())
            if bass_pc is None:
                raise ValueError(f"Unrecognized bass note in letter chord: {bass_part!r}")
        elif bass_part.startswith(("9", "11", "13", "b9", "#9")) or bass_part in ("6.9", "69"):
            root_part = f"{root_part}({bass_part})"
            bass_name = None
            bass_pc = None
        else:
            raise ValueError(f"Invalid bass note in letter chord: {bass_part!r} in {raw!r}")
    else:
        root_part = cleaned
        bass_name = None
        bass_pc = None

    m_root = re.match(r"^([A-Ga-g])([b#♭♯]?)(.*)$", root_part)
    if not m_root:
        raise ValueError(f"Cannot identify root note (A-G) in letter chord: {raw!r}")

    r_let = m_root.group(1).upper()
    r_acc_raw = m_root.group(2)
    r_acc = "b" if r_acc_raw in ("b", "♭") else ("#" if r_acc_raw in ("#", "♯") else "")
    root_name = f"{r_let}{r_acc}"
    root_pc = KEY_NAME_TO_PC.get(root_name.upper())
    if root_pc is None:
        raise ValueError(f"Unrecognized root note in letter chord: {root_name!r}")

    q_part = m_root.group(3)
    quality = normalize_quality(q_part, raw)

    return LetterChordSpec(
        raw=raw,
        root_name=root_name,
        root_pc=root_pc,
        quality=quality,
        bass_name=bass_name,
        bass_pc=bass_pc,
    )


def format_chord_name(root_name: str, quality: str) -> str:
    """Format root note name and quality into standard chord name."""
    suffix = QUALITY_SUFFIXES.get(quality, quality)
    return f"{root_name}{suffix}"


def get_chord_pcs(root_pc: int, quality: str) -> list[int]:
    """Return pitch classes for chord tones with root first."""
    intervals = QUALITY_INTERVALS.get(quality, EXTRA_QUALITY_INTERVALS.get(quality, QUALITY_INTERVALS["maj"]))
    return [(root_pc + i) % 12 for i in intervals]


def resolve_chord(
    raw: str,
    tonic_pc: int,
    beat: float = 1.0,
    key_name: Optional[str] = None,
    *,
    notation: str = "number",
    printed_tonic_pc: Optional[int] = None,
    prev_chord: Optional[ResolvedChord | ChordSymbol] = None,
    prev: Optional[ResolvedChord | ChordSymbol] = None,
    stacked: bool = False,
    stacked_orientation: str = "bottom_is_bass",
) -> ResolvedChord:
    """Resolve a Taiwanese number-notation or letter-notation chord symbol into a concrete ResolvedChord.

    - If notation == 'letter': transposes letter chord by (tonic_pc - printed_tonic_pc).
    - If notation == 'number': resolves scale degree relative to tonic_pc.
    - If stacked == True: resolves diagonal stacked chord using stacked_orientation.
    - If leading slash (e.g. '/5', '/#4', '/G'): resolves as preceding chord over the new bass note.
    """
    cleaned_raw = clean_raw_chord(raw)

    # Contextual leading slash chord (e.g. '/5', '/#4', '/b7', '/G', '/A')
    if cleaned_raw.startswith("/"):
        prior = prev if prev is not None else prev_chord
        if prior is None:
            raise ValueError(f"Leading slash chord {raw!r} requires preceding chord context")

        if not isinstance(prior, ResolvedChord):
            prior = resolve_chord(
                prior.raw,
                tonic_pc,
                prior.beat,
                key_name,
                notation=notation,
                printed_tonic_pc=printed_tonic_pc,
                stacked=prior.stacked,
                stacked_orientation=stacked_orientation,
            )

        bass_part = cleaned_raw[1:]
        m_let = re.match(r"^([A-Ga-g])([b#♭♯]?)$", bass_part)
        if m_let:
            b_let = m_let.group(1).upper()
            b_acc_raw = m_let.group(2)
            b_acc = "b" if b_acc_raw in ("b", "♭") else ("#" if b_acc_raw in ("#", "♯") else "")
            b_name = f"{b_let}{b_acc}"
            b_pc = KEY_NAME_TO_PC.get(b_name.upper())
            if b_pc is None:
                raise ValueError(f"Unrecognized bass note in letter chord: {raw!r}")
            shift = (tonic_pc - printed_tonic_pc) % 12 if printed_tonic_pc is not None else 0
            if shift != 0:
                b_pc = (b_pc + shift) % 12
                b_name = spell(b_pc, tonic_pc=tonic_pc, key_name=key_name)
        else:
            if bass_part.endswith("m") and len(bass_part) > 1:
                bass_part = bass_part[:-1]
            b_deg, b_acc, b_rem = parse_accidental_and_degree(bass_part)
            if b_rem:
                raise ValueError(f"Trailing garbage in slash bass: {bass_part!r} in {raw!r}")
            b_name, b_pc = resolve_degree(b_deg, b_acc, tonic_pc, key_name)

        root_chord_name = prior.name.split("/")[0]
        full_name = f"{root_chord_name}/{b_name}"
        return ResolvedChord(
            raw=raw,
            name=full_name,
            beat=beat,
            root_pc=prior.root_pc,
            bass_pc=b_pc,
            pcs=prior.pcs,
            quality=prior.quality,
        )

    # Per-token notation detection:
    # A token starting with a digit (optionally b/# + digit, or a circled digit) is a number chord.
    # A token starting with A-G is a letter chord.
    # The header field only decides transposition for letter tokens.
    is_number = bool(re.match(r"^[b#♭♯]?[1-7]", cleaned_raw))
    is_letter = bool(re.match(r"^[A-Ga-g]", cleaned_raw))

    if is_number:
        token_notation = "number"
    elif is_letter:
        token_notation = "letter"
    else:
        token_notation = notation

    if token_notation == "letter":
        spec = parse_letter_chord(raw)
        shift = (tonic_pc - printed_tonic_pc) % 12 if printed_tonic_pc is not None else 0

        target_root_pc = (spec.root_pc + shift) % 12
        target_bass_pc = ((spec.bass_pc + shift) % 12) if spec.bass_pc is not None else target_root_pc

        if shift == 0:
            target_root_name = spec.root_name
            target_bass_name = spec.bass_name if spec.bass_name is not None else target_root_name
        else:
            target_root_name = spell(target_root_pc, tonic_pc=tonic_pc, key_name=key_name)
            target_bass_name = spell(target_bass_pc, tonic_pc=tonic_pc, key_name=key_name) if spec.bass_pc is not None else target_root_name

        root_chord_name = format_chord_name(target_root_name, spec.quality)
        if spec.bass_pc is not None:
            full_name = f"{root_chord_name}/{target_bass_name}"
        else:
            full_name = root_chord_name

        pcs = get_chord_pcs(target_root_pc, spec.quality)

        return ResolvedChord(
            raw=raw,
            name=full_name,
            beat=beat,
            root_pc=target_root_pc,
            bass_pc=target_bass_pc,
            pcs=pcs,
            quality=spec.quality,
        )

    num_spec = parse_chord(raw, stacked=stacked, stacked_orientation=stacked_orientation)

    root_name, root_pc = resolve_degree(num_spec.degree, num_spec.accidental, tonic_pc, key_name)

    if num_spec.bass_degree is not None:
        bass_acc = num_spec.bass_accidental if num_spec.bass_accidental is not None else 0
        bass_name, bass_pc = resolve_degree(num_spec.bass_degree, bass_acc, tonic_pc, key_name)
    else:
        bass_name, bass_pc = root_name, root_pc

    root_chord_name = format_chord_name(root_name, num_spec.quality)
    if num_spec.bass_degree is not None:
        full_name = f"{root_chord_name}/{bass_name}"
    else:
        full_name = root_chord_name

    pcs = get_chord_pcs(root_pc, num_spec.quality)

    return ResolvedChord(
        raw=raw,
        name=full_name,
        beat=beat,
        root_pc=root_pc,
        bass_pc=bass_pc,
        pcs=pcs,
        quality=num_spec.quality,
    )
