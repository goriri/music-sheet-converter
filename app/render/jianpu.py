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
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

from PIL import ImageDraw, ImageFont

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


def midi_to_jianpu(
    midi: int,
    tonic_pc: int,
    hand: Literal["rh", "lh"] = "rh",
    prefer_sharps: bool = False,
    finger: Optional[int] = None,
) -> JianpuNote:
    """Convert MIDI pitch to movable-do JianpuNote.

    Reference octave rule:
    - RH: tonic nearest at/above C4 (MIDI 60 + tonic_pc%12) has 0 dots.
    - LH: reference one octave lower (MIDI 48 + tonic_pc%12) has 0 dots.
    """
    ref_midi = (60 if hand == "rh" else 48) + (tonic_pc % 12)
    octave_dots = (midi - ref_midi) // 12
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


def compute_stack_height(max_notes: int, line_spacing: float = 18.0) -> float:
    """Return required height in pixels for a stack of up to max_notes."""
    base_h = 44.0
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
) -> Tuple[float, float, float]:
    """Draw a single jianpu note (accidental + digit + octave dots).

    Returns (width, top_y, bottom_y) including dots.
    """
    cur_x = x
    if note.accidental:
        draw.text((cur_x, y), note.accidental, font=acc_font, fill=color)
        bbox_acc = draw.textbbox((cur_x, y), note.accidental, font=acc_font)
        cur_x += (bbox_acc[2] - bbox_acc[0]) + 1.0

    digit_str = str(note.degree)
    draw.text((cur_x, y), digit_str, font=font, fill=color)
    bbox = draw.textbbox((cur_x, y), digit_str, font=font)
    digit_w = bbox[2] - bbox[0]
    center_x = cur_x + digit_w / 2.0

    top_y = float(bbox[1])
    bottom_y = float(bbox[3])

    if note.octave_dots > 0:
        for i in range(note.octave_dots):
            dot_y = float(bbox[1]) - 4.0 - i * 5.0
            draw.ellipse(
                [center_x - 1.8, dot_y - 1.8, center_x + 1.8, dot_y + 1.8],
                fill=color,
            )
            top_y = min(top_y, dot_y - 2.0)
    elif note.octave_dots < 0:
        for i in range(abs(note.octave_dots)):
            dot_y = float(bbox[3]) + 4.0 + i * 5.0
            draw.ellipse(
                [center_x - 1.8, dot_y - 1.8, center_x + 1.8, dot_y + 1.8],
                fill=color,
            )
            bottom_y = max(bottom_y, dot_y + 2.0)

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
        w, top_y, bot_y = draw_jianpu_note(draw, x, digit_y, note, font, acc_font, color=color)
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
                    fy = overall_top - 14.0
                    draw.text((x + 2, fy), f_str, font=finger_font, fill=finger_color)
                    overall_top = fy
        else:
            # Multi-note chord stack: draw fingers vertically next to each note
            for i, note in enumerate(sorted_notes):
                if note.finger is not None:
                    fy = digit_y_positions[i] + 7.0
                    draw.text((x + max_w + 3.0, fy), str(note.finger), font=finger_font, fill=finger_color)

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
) -> None:
    """Draw augmentation dot '.' for dotted rhythms."""
    draw.ellipse(
        [
            x - dot_radius,
            center_y - dot_radius,
            x + dot_radius,
            center_y + dot_radius,
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
) -> None:
    """Draw a horizontal underline beam line."""
    draw.line([(x0, y), (x1, y)], fill=color, width=int(round(width)))


def draw_tie(
    draw: ImageDraw.ImageDraw,
    x0: float,
    x1: float,
    y: float,
    arc_height: float = 7.0,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a curved tie arc between two notes."""
    mid_x = (x0 + x1) / 2.0
    control_y = y - arc_height
    points = []
    steps = 14
    for s in range(steps + 1):
        t = s / steps
        px = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid_x + t**2 * x1
        py = (1 - t) ** 2 * y + 2 * (1 - t) * t * control_y + t**2 * y
        points.append((px, py))
    draw.line(points, fill=color, width=1)
