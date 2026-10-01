"""Prompts and Pydantic schemas for the OMR crop-based content reader.

The content reader transcribes cropped and upscaled regions guided by CV geometry:
1. Header band crop -> SongHeader (title, style, time signature, tempo, keys, chord notation).
2. System row crops -> SystemReading (chord boxes verbatim by box ID, extra chords by measure,
   jianpu melody, lyrics, bass hint, fill/stop flags, key change markings).
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Header schemas
# ---------------------------------------------------------------------------

class HeaderReading(BaseModel):
    title: str = Field("", description="Song title, e.g. '掉了'.")
    style: str = Field("", description="Rhythm/style label, e.g. 'Slow Soul', 'Pop Ballad', 'Waltz 3/4'.")
    time_signature: str = Field("4/4", description="Time signature, e.g. '4/4', '3/4', '12/8', '6/8'.")
    tempo_bpm: Optional[float] = Field(None, description="Tempo in BPM, e.g. 81.0 from '♩ = 81'.")
    original_key: Optional[str] = Field(
        None,
        description="Starting original key as printed, e.g. 'F#' from '(F# - Ab)' or 'C' from '1=C'.",
    )
    male_key: Optional[str] = Field(None, description="Male key as printed, e.g. 'Bb' from '男調(Bb-C)'.")
    female_key: Optional[str] = Field(None, description="Female key as printed, e.g. 'F' from '女調(F-G)'.")
    chord_notation: Literal["number", "letter"] = Field(
        "number",
        description="'number' for Taiwanese scale-degree number chords (1(2), 5/7, 2m7); 'letter' for Western letter chords (C, G/B, Am7).",
    )
    raw: str = Field("", description="Verbatim transcribed text of the header line.")


HEADER_PROMPT = """You are an expert Optical Music Recognition assistant reading the top header section of a music sheet (band chart / 流行简谱).

Transcribe the header metadata into structured JSON:
- title: song title (e.g. '掉了').
- style: rhythm or style label (e.g. 'Slow Soul', 'Pop Ballad', 'Waltz 3/4', 'Slow Rock').
- time_signature: time signature (e.g. '4/4', '3/4', '12/8'). Default to '4/4' if not printed.
- tempo_bpm: numeric tempo in BPM if printed (e.g. 81.0 from '♩ = 81' or 'BPM=81').
- original_key: starting original key (e.g. 'F#' from '(F# - Ab)' or 'C' from '1=C').
- male_key: male key if printed (e.g. 'Bb' from '男調(Bb-C)').
- female_key: female key if printed (e.g. 'F' from '女調(F-G)').
- chord_notation: 'number' if the sheet uses Taiwanese scale-degree number notation (e.g. 1(2), 5/7, 2m7); 'letter' if chords are written as Western letters (C, G/B, Am7) or if indicated by '1=X' without degree chords.
- raw: verbatim text of the printed header line.
"""


# ---------------------------------------------------------------------------
# System row schemas
# ---------------------------------------------------------------------------

class BoxReading(BaseModel):
    box_id: int = Field(
        description="Box identifier matching the '#1', '#2'... badge drawn on the image."
    )
    text: str = Field(
        "",
        description=(
            "Exact verbatim chord text inside or directly under this numbered badge (e.g. '1(2)', '5/7', '2m7'). "
            "For Taiwanese number charts, NEVER convert to Western letter chords (NEVER C, Dm, G/B). "
            "Exclude any instrument annotations printed outside or near the box (e.g. 'PN', 'EG', 'AG'). "
            "If the badge marks empty space, an instrument annotation only, or no chord is present, return empty string ''."
        ),
    )
    beat: Optional[float] = Field(
        None,
        description="1-based beat inside the measure if clearly visible (e.g. 1.0, 3.0), else null.",
    )
    stacked: bool = Field(
        False,
        description="True if printed as a diagonal/stacked fraction (circled mainland style, e.g. '⑦╱⑤', '①╱②m7-5'); False if inline slash or normal.",
    )


class ExtraChordReading(BaseModel):
    measure_index: int = Field(
        description="0-based measure index in this system (e.g. 0, 1, 2, 3) where the unnumbered chord appears."
    )
    text: str = Field(
        description="Verbatim chord text not enclosed in any numbered box. Exclude voicing digit patterns (e.g. '056') and rests ('0')."
    )
    beat: float = Field(
        1.0,
        description="1-based beat inside the measure where this chord starts (e.g. 1.0, 3.0).",
    )
    stacked: bool = Field(
        False,
        description="True if printed as a diagonal/stacked fraction (circled mainland style); False if inline slash or normal.",
    )
    boxed: bool = Field(
        True,
        description="True if this chord is printed inside a visual bounding box or enclosure on the sheet; False if printed as unboxed text/digits outside any box.",
    )


class MeasureContentReading(BaseModel):
    measure_index: int = Field(
        description="0-based measure index within this system (e.g. 0, 1, 2, 3)."
    )
    melody: str = Field(
        "",
        description="Jianpu melody notes as printed, space-separated beat groups (e.g. '2 2 23 21').",
    )
    lyrics: str = Field("", description="Lyrics text printed under the melody in this measure.")
    bass_hint: Optional[str] = Field(
        None,
        description="Printed 'Bs:' rhythm pattern for this measure if present (e.g. '11 11 11 112').",
    )
    rhythm_hint: Optional[str] = Field(
        None,
        description="Slash rhythm marks printed above or inside this measure (e.g. '/ / / ~').",
    )
    fill: bool = Field(
        False,
        description="True if a drum/band fill is marked on this measure, such as '(D.r fill)'.",
    )
    is_stop: bool = Field(
        False,
        description="True if melody rests or stops (e.g. '2 - 0 0', '2 - - -') or band hits and stops.",
    )


class KeyChangeReading(BaseModel):
    measure_index: int = Field(
        description="0-based measure index in this system where the key change occurs."
    )
    raw: str = Field(
        description="Verbatim key change text as printed, e.g. '(轉成2調)(Ab)' or '轉 1=B'."
    )


class SystemReading(BaseModel):
    section_label: Optional[str] = Field(
        None,
        description=(
            "Instrumentation or texture label printed above or at the beginning of the row, "
            "e.g. 'Only PN(RHY) / / / ~', 'Only PN+AG(RHY)+Bs in', 'Tempo(小鼓2.4拍)OG in', '(小鼓2.4拍)OG in'."
        ),
    )
    measures_seen: Optional[int] = Field(
        None,
        description="Number of musical measures actually printed/visible in this row as delimited by the printed vertical barlines, IGNORING the drawn blue guides.",
    )
    chord_boxes: list[BoxReading] = Field(
        default_factory=list,
        description="Readings for each numbered chord box badge (#1, #2...) shown on the image.",
    )
    extra_chords: list[ExtraChordReading] = Field(
        default_factory=list,
        description="Any chords visible in this system that do NOT have a numbered box.",
    )
    measures: list[MeasureContentReading] = Field(
        default_factory=list,
        description="Content of each measure in this system (melody, lyrics, bass hint, fill, stop).",
    )
    key_change: Optional[KeyChangeReading] = Field(
        None,
        description="Key change modulation marking if printed in this system.",
    )


def build_system_crop_prompt(
    system_index_on_page: int,
    total_measures: int,
    num_boxes: int,
    chord_notation: str = "number",
    chord_only: bool = False,
) -> str:
    """Build prompt for transcribing an upscaled annotated system crop."""
    notation_instruction = (
        "This sheet uses Taiwanese scale-degree number chords (e.g. '1(2)', '5/7', '2m7/5', '5m/7b', '67/1#', '6m7-5', '57sus', '4M7', '17/7b', '3m/7', '6b', '5sus', '5', '57', '1/3', '4'). "
        "NEVER convert degree numbers to Western letter chords (NEVER C, Dm, G/B)! Transcribe verbatim in scale-degree notation."
        if chord_notation == "number"
        else "This sheet uses Western letter chords (e.g. 'C', 'G/B', 'Am7', 'Fmaj7'). Transcribe verbatim as printed."
    )

    chord_only_instruction = (
        "\nIMPORTANT: This row is a CHORD-ONLY system (弹唱版 / no jianpu melody printed). "
        "Read the numbered chord boxes only. For all measures, set melody='' and lyrics=''.\n"
        if chord_only
        else ""
    )

    return f"""You are an expert Optical Music Recognition assistant reading an upscaled single musical system (row) crop from a Taiwanese band chart (台湾简谱 / 流行乐队总谱).

Visual guides on the image:
- Blue vertical lines delimit measures, labeled '[Measure 0]', '[Measure 1]' ... up to [Measure {total_measures - 1}].
- Red rectangular outlines with yellow badges '#1', '#2' ... up to '#{num_boxes}' indicate detected chord boxes.

{notation_instruction}
{chord_only_instruction}

Please transcribe into structured JSON:
1. section_label: any instrumentation/texture label printed above or at the beginning of the row (e.g. 'Only PN(RHY) / / / ~', 'Only PN+AG(RHY)+Bs in', '(小鼓2.4拍)OG in').
2. measures_seen: count of musical measures printed in this row as delimited by the printed barlines, IGNORING the drawn blue guides.
3. chord_boxes: for each numbered badge (#1..#{num_boxes}), provide the verbatim chord text inside or directly under that badge.
   - Exclude instrument annotations printed outside or near the box (e.g. 'PN', 'EG', 'AG').
   - If a badge marks empty space, an instrument annotation only, or no chord is present, return text="".
   - If a chord is printed as a diagonal or stacked fraction (circled mainland style, e.g. '⑦╱⑤', '①╱②m7-5'): report text as 'TOP/BOTTOM' in printed order (e.g. '7/5', '1/2m7-5') without circles, and set stacked=true.
   - Inline chords (e.g. '1(2)', '5/7', or circled '⑤7/9') have stacked=false.
4. extra_chords: any chords printed in this row that do NOT have a red numbered box (provide measure_index 0..{total_measures - 1}, text, beat, stacked, and boxed: true if enclosed in a printed box/border on the sheet missed by the detector, false if unboxed text/digits). Exclude piano/instrument voicing digits (e.g. '056', '5616', '1242') and rests ('0') which are not chords.
5. measures: for each measure (0..{total_measures - 1}):
   - melody: jianpu melody notes as space-separated beat groups (e.g. '2 2 23 21').
   - lyrics: printed lyrics under the melody.
   - bass_hint: 'Bs: ...' rhythm pattern if printed (e.g. '11 11 11 112').
   - rhythm_hint: slash marks if printed (e.g. '/ / / ~').
   - fill: true if drum/band fill is marked (e.g. '(D.r fill)').
   - is_stop: true if resting/stopping (e.g. '2 - 0 0').
6. key_change: any key modulation marking in this system (e.g. '(轉成2調)(Ab)' or '轉 1=B') and the measure_index it is above. If none, leave null.
"""

