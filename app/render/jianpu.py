"""Pure functions for movable-do jianpu notation conversion and drawing primitives.

Conventions:
- tonic_pc: 0..11 (0=C, 1=C#, 2=D, ..., 5=F, 7=G, 11=B)
- Movable-do degrees 1-7 relative to tonic
- Sensible accidental prefixes:
    b3, b6, b7 flats
    #1, #4, #5 sharps (or b2 flat)
- Reference octaves (per prompt):
    RH: the octave starting at the tonic nearest above C4 has 0 dots.
        (e.g. for Key F: F4..E5 has 0 dots; C4 has 1 dot below; F5 has 1 dot above)
    LH: reference one octave lower, i.e. notes in tonic octave starting nearest
        above C3 have 0 dots.
        (e.g. for Key F: F3..E4 has 0 dots; F2 has 1 dot below; C2 has 2 dots below)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

from PIL import ImageDraw, ImageFont

from app.models import ResolvedChord
from app.theory.keys import (
    canonical_key_for_pc,
    DIATONIC_SCALES,
    KEY_NAME_TO_PC,
)


# --------------------------------------------------------------------------- Fonts cache
_FONT_CACHE: dict[Tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def get_font(size: int = 18, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load bundled OFL font with graceful fallback."""
    key = ("bold" if bold else "regular", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
    candidate_files = [
        os.path.join(fonts_dir, "NotoSansSC.ttf"),
        os.path.join(fonts_dir, "NotoSans.ttf"),
    ]

    for fpath in candidate_files:
        if os.path.isfile(fpath):
            try:
                font = ImageFont.truetype(fpath, size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                pass

    fallback = ImageFont.load_default()
    _FONT_CACHE[key] = fallback
    return fallback


# --------------------------------------------------------------------------- Note conversion
@dataclass(frozen=True)
class JianpuNote:
    degree: int  # 1..7
    accidental: str  # "", "♯", "♭"
    octave_dots: int  # 0=reference, +1=dot above, -1=dot below, etc.
    finger: Optional[int] = None
    midi: Optional[int] = None

    @property
    def text(self) -> str:
        """String representation, e.g. '♭7' or '1'."""
        return f"{self.accidental}{self.degree}"


# Diatonic semitones from tonic:
# 0 -> 1, 2 -> 2, 4 -> 3, 5 -> 4, 7 -> 5, 9 -> 6, 11 -> 7
# Chromatic intervals (prompt: b3 b6 b7 flats, #1 #4 #5 sharps, b2 flat):
# interval 1: b2 (or #1)
# interval 3: b3
# interval 6: #4
# interval 8: b6 (or #5)
# interval 10: b7
SEMITONE_TO_JIANPU_DEFAULT = {
    0: (1, ""),
    1: (2, "♭"),  # b2 flat
    2: (2, ""),
    3: (3, "♭"),  # b3 flat
    4: (3, ""),
    5: (4, ""),
    6: (4, "♯"),  # #4 sharp
    7: (5, ""),
    8: (6, "♭"),  # b6 flat
    9: (6, ""),
    10: (7, "♭"),  # b7 flat
    11: (7, ""),
}

SEMITONE_TO_JIANPU_SHARPS = {
    0: (1, ""),
    1: (1, "♯"),  # #1 sharp
    2: (2, ""),
    3: (3, "♭"),  # b3 flat
    4: (3, ""),
    5: (4, ""),
    6: (4, "♯"),  # #4 sharp
    7: (5, ""),
    8: (5, "♯"),  # #5 sharp
    9: (6, ""),
    10: (7, "♭"),  # b7 flat
    11: (7, ""),
}


LETTERS: list[str] = ["C", "D", "E", "F", "G", "A", "B"]


def get_chord_tone_letters(chord: ResolvedChord) -> dict[int, str]:
    """Map each pitch class in chord (and slash bass) to its letter name."""
    name = chord.name
    parts = name.split("/")
    root_part = parts[0]
    bass_part = parts[1] if len(parts) > 1 else None

    m_root = re.match(r"^([A-G])", root_part)
    if not m_root:
        return {}
    root_letter = m_root.group(1)
    root_idx = LETTERS.index(root_letter)

    tone_letters: dict[int, str] = {}
    # Root pitch class
    tone_letters[chord.root_pc] = root_letter

    # If explicit slash bass (e.g. D7/F# -> F#)
    if bass_part:
        m_bass = re.match(r"^([A-G])", bass_part)
        if m_bass:
            tone_letters[chord.bass_pc] = m_bass.group(1)

    # Derived chord tones from root letter by interval:
    # 3rd = root + 2, 5th = + 4, 7th = + 6, 9th/2nd = + 1, 4th = + 3, 6th = + 5
    for pc in chord.pcs:
        if pc in tone_letters:
            continue
        semi = (pc - chord.root_pc) % 12
        if semi in (3, 4):  # minor or major 3rd
            tone_letters[pc] = LETTERS[(root_idx + 2) % 7]
        elif semi in (6, 7, 8):  # dim, perf, or aug 5th
            tone_letters[pc] = LETTERS[(root_idx + 4) % 7]
        elif semi in (10, 11):  # min or maj 7th
            tone_letters[pc] = LETTERS[(root_idx + 6) % 7]
        elif semi == 9:  # dim 7th or 6th
            if chord.quality == "dim":
                tone_letters[pc] = LETTERS[(root_idx + 6) % 7]
            else:
                tone_letters[pc] = LETTERS[(root_idx + 5) % 7]
        elif semi == 2:  # 9th / 2nd
            tone_letters[pc] = LETTERS[(root_idx + 1) % 7]
        elif semi == 5:  # 4th / 11th
            tone_letters[pc] = LETTERS[(root_idx + 3) % 7]
        elif semi == 1:  # b9
            tone_letters[pc] = LETTERS[(root_idx + 1) % 7]

    return tone_letters


def get_chord_context_spelling(
    midi: int,
    tonic_pc: int,
    key_name: Optional[str],
    chord: ResolvedChord,
) -> Optional[Tuple[int, str]]:
    """Determine (degree, accidental) of a note within its chord context."""
    note_pc = midi % 12
    canonical_key = canonical_key_for_pc(tonic_pc, key_name)
    scale = DIATONIC_SCALES.get(canonical_key, DIATONIC_SCALES["C"])
    diatonic_pcs = {KEY_NAME_TO_PC[name.upper()]: (i + 1) for i, name in enumerate(scale)}

    # If note is diatonic in the key, degree has no accidental
    if note_pc in diatonic_pcs:
        return diatonic_pcs[note_pc], ""

    # Note is chromatic: check active chord tone letters
    tone_letters = get_chord_tone_letters(chord)
    if note_pc not in tone_letters:
        return None

    letter = tone_letters[note_pc]
    tonic_letter = canonical_key[0]
    degree = ((LETTERS.index(letter) - LETTERS.index(tonic_letter)) % 7) + 1
    diatonic_note = scale[degree - 1]
    diatonic_pc = KEY_NAME_TO_PC[diatonic_note.upper()]

    diff = (note_pc - diatonic_pc) % 12
    if diff == 0:
        accidental = ""
    elif diff == 1:
        accidental = "♯"
    elif diff == 11:
        accidental = "♭"
    elif diff == 2:
        accidental = "♯♯"
    elif diff == 10:
        accidental = "♭♭"
    else:
        return None

    return degree, accidental


def midi_to_jianpu(
    midi: int,
    tonic_pc: int,
    hand: Literal["rh", "lh"] = "rh",
    prefer_sharps: bool = False,
    finger: Optional[int] = None,
    chord: Optional[ResolvedChord] = None,
    key_name: Optional[str] = None,
) -> JianpuNote:
    """Convert MIDI pitch to movable-do JianpuNote.

    Reference octave rule:
    - RH: tonic nearest at/above C4 (MIDI 60 + tonic_pc%12) has 0 dots.
    - LH: reference one octave lower (MIDI 48 + tonic_pc%12) has 0 dots.

    Chromatic spelling follows active chord context when provided.
    """
    ref_midi = (60 if hand == "rh" else 48) + (tonic_pc % 12)
    octave_dots = (midi - ref_midi) // 12

    spelling = None
    if chord is not None:
        spelling = get_chord_context_spelling(midi, tonic_pc, key_name, chord)

    if spelling is not None:
        degree, accidental = spelling
    else:
        interval = (midi - (tonic_pc % 12)) % 12
        table = SEMITONE_TO_JIANPU_SHARPS if prefer_sharps else SEMITONE_TO_JIANPU_DEFAULT
        degree, accidental = table[interval]

    return JianpuNote(
        degree=degree,
        accidental=accidental,
        octave_dots=octave_dots,
        finger=finger,
        midi=midi,
    )



# --------------------------------------------------------------------------- Layout & Metrics
@dataclass
class ChordStackLayout:
    """Computed vertical layout coordinates for a chord stack."""

    notes: list[JianpuNote]
    digit_y_positions: list[float]
    top_y: float  # Topmost visual extent (including top dots)
    bottom_y: float  # Bottommost visual extent (including bottom dots)
    base_x: float  # Horizontal reference coordinate
    max_w: float  # Max width of digits and accidentals


def compute_stack_height(max_notes: int, line_spacing: float = 18.0, scale: float = 1.0) -> float:
    """Return required height in pixels for a stack of up to max_notes."""
    base_h = 44.0 * scale
    if max_notes <= 1:
        return base_h
    return base_h + (max_notes - 1) * line_spacing


# --------------------------------------------------------------------------- Drawing Primitives
def draw_jianpu_note(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    note: JianpuNote,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    acc_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Draw a single jianpu note (accidental + digit + octave dots).

    Returns (width, top_y, bottom_y) including dots.
    """
    cur_x = x
    if note.accidental:
        draw.text((cur_x, y), note.accidental, font=acc_font, fill=color)
        bbox_acc = draw.textbbox((cur_x, y), note.accidental, font=acc_font)
        cur_x += (bbox_acc[2] - bbox_acc[0]) + 1.0 * scale

    digit_str = str(note.degree)
    draw.text((cur_x, y), digit_str, font=font, fill=color)
    bbox = draw.textbbox((cur_x, y), digit_str, font=font)
    digit_w = bbox[2] - bbox[0]
    center_x = cur_x + digit_w / 2.0

    top_y = float(bbox[1])
    bottom_y = float(bbox[3])

    dot_r = 1.8 * scale
    if note.octave_dots > 0:
        for i in range(note.octave_dots):
            dot_y = float(bbox[1]) - (4.0 + i * 5.0) * scale
            draw.ellipse(
                [center_x - dot_r, dot_y - dot_r, center_x + dot_r, dot_y + dot_r],
                fill=color,
            )
            top_y = min(top_y, dot_y - 2.0 * scale)
    elif note.octave_dots < 0:
        for i in range(abs(note.octave_dots)):
            dot_y = float(bbox[3]) + (4.0 + i * 5.0) * scale
            draw.ellipse(
                [center_x - dot_r, dot_y - dot_r, center_x + dot_r, dot_y + dot_r],
                fill=color,
            )
            bottom_y = max(bottom_y, dot_y + 2.0 * scale)

    width = (cur_x + digit_w) - x
    return width, top_y, bottom_y


def draw_chord_stack(
    draw: ImageDraw.ImageDraw,
    x: float,
    base_y: float,
    notes: Sequence[JianpuNote],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    acc_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    finger_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    hand: Literal["rh", "lh"] = "rh",
    line_spacing: float = 18.0,
    color: Tuple[int, int, int] = (0, 0, 0),
    finger_color: Tuple[int, int, int] = (25, 118, 210),  # Material Blue
    draw_finger: bool = True,
    scale: float = 1.0,
) -> ChordStackLayout:
    """Draw a stacked chord (lowest note at bottom, each with its own dots).

    Notes are sorted by MIDI ascending (lowest at base_y, higher notes above).
    """
    sorted_notes = sorted(notes, key=lambda n: n.midi if n.midi is not None else n.degree)
    num_notes = len(sorted_notes)

    digit_y_positions: list[float] = []
    max_w = 0.0
    overall_top = 999999.0
    overall_bottom = -999999.0

    # Layout digits: lowest note at base_y, next note higher by line_spacing
    for i, note in enumerate(sorted_notes):
        digit_y = base_y - i * line_spacing
        digit_y_positions.append(digit_y)
        w, top_y, bot_y = draw_jianpu_note(
            draw, x, digit_y, note, font, acc_font, color=color, scale=scale
        )
        max_w = max(max_w, w)
        overall_top = min(overall_top, top_y)
        overall_bottom = max(overall_bottom, bot_y)

    # Draw finger numbers for RH
    if draw_finger:
        if num_notes == 1:
            f_num = sorted_notes[0].finger
            if f_num is not None:
                f_str = str(f_num)
                if hand == "rh":
                    fy = overall_top - 14.0 * scale
                    draw.text((x + 2.0 * scale, fy), f_str, font=finger_font, fill=finger_color)
                    overall_top = fy
        else:
            # Multi-note chord stack: draw fingers vertically next to each note
            for i, note in enumerate(sorted_notes):
                if note.finger is not None:
                    fy = digit_y_positions[i] + 7.0 * scale
                    draw.text(
                        (x + max_w + 3.0 * scale, fy),
                        str(note.finger),
                        font=finger_font,
                        fill=finger_color,
                    )

    return ChordStackLayout(
        notes=list(sorted_notes),
        digit_y_positions=digit_y_positions,
        top_y=overall_top,
        bottom_y=overall_bottom,
        base_x=x,
        max_w=max_w,
    )


def draw_rest(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[float, float, float]:
    """Draw rest '0'. Returns (width, top_y, bottom_y)."""
    draw.text((x, y), "0", font=font, fill=color)
    bbox = draw.textbbox((x, y), "0", font=font)
    return (bbox[2] - bbox[0]), float(bbox[1]), float(bbox[3])


def draw_sustain_dash(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[float, float, float]:
    """Draw sustain dash '-' for extra beats. Returns (width, top_y, bottom_y)."""
    draw.text((x, y), "-", font=font, fill=color)
    bbox = draw.textbbox((x, y), "-", font=font)
    return (bbox[2] - bbox[0]), float(bbox[1]), float(bbox[3])


def draw_augmentation_dot(
    draw: ImageDraw.ImageDraw,
    x: float,
    center_y: float,
    dot_radius: float = 2.0,
    color: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 1.0,
) -> None:
    """Draw augmentation dot '.' for dotted rhythms."""
    r = dot_radius * scale
    draw.ellipse(
        [
            x - r,
            center_y - r,
            x + r,
            center_y + r,
        ],
        fill=color,
    )


def draw_underline_beam(
    draw: ImageDraw.ImageDraw,
    x0: float,
    x1: float,
    y: float,
    width: float = 1.8,
    color: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 1.0,
) -> None:
    """Draw a horizontal underline beam line."""
    draw.line([(x0, y), (x1, y)], fill=color, width=max(1, int(round(width * scale))))


def draw_tie(
    draw: ImageDraw.ImageDraw,
    x0: float,
    x1: float,
    y: float,
    arc_height: float = 7.0,
    color: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 1.0,
) -> None:
    """Draw a curved tie arc between two notes."""
    mid_x = (x0 + x1) / 2.0
    control_y = y - arc_height * scale
    points = []
    steps = 14
    for s in range(steps + 1):
        t = s / steps
        px = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid_x + t**2 * x1
        py = (1 - t) ** 2 * y + 2 * (1 - t) * t * control_y + t**2 * y
        points.append((px, py))
    draw.line(points, fill=color, width=max(1, int(round(1.0 * scale))))

