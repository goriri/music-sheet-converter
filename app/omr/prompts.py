"""Prompts and schemas for Gemini-based OMR of Taiwanese band charts (台湾谱).

Taiwanese band charts use numbered musical notation (jianpu) with boxed scale-degree
chords, instrumentation/section labels, bass rhythm patterns, and modulation markings.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas for Gemini structured output (box_2d on 0-1000 scale)
# ---------------------------------------------------------------------------

class ChordSymbolGemini(BaseModel):
    raw: str = Field(
        description=(
            "Chord exactly as printed inside the box in Taiwanese number notation, "
            "e.g. '1(2)', '5/7', '2m7/5', '5m/7b', '67/1#', '6m7-5', '57sus', '4M7', "
            "'17/7b', '3m/7', '6b', '5sus', '5', '57', '1/3', '4'. "
            "Transcribe verbatim. NEVER convert to Western letter chords (never C, Dm, etc.). "
            "Exclude any instrument annotations printed outside or near the box (e.g. 'PN', 'EG')."
        )
    )
    beat: float = Field(
        1.0,
        description=(
            "1-based beat inside the measure where the chord starts (e.g. 1.0, 2.0, 3.0, 4.0). "
            "Estimate from which melody beat group it sits above."
        ),
    )
    box_2d: Optional[list[int]] = Field(
        None,
        description="Bounding box [ymin, xmin, ymax, xmax] of the printed chord box in 0..1000 coordinates.",
    )


class MeasureGemini(BaseModel):
    measure_index: int = Field(
        description="0-based measure index within this page (reading order: row by row, left to right)."
    )
    box_2d: list[int] = Field(
        description=(
            "Bounding box [ymin, xmin, ymax, xmax] in 0..1000 of the melody measure band "
            "between the left barline and right barline."
        )
    )
    beats: float = Field(4.0, description="Beats per measure (usually 4.0 in 4/4).")
    chords: list[ChordSymbolGemini] = Field(
        default_factory=list,
        description=(
            "List of chord symbols in this measure. If no chord box is printed above this measure, "
            "leave chords empty [] (do NOT invent chords; carried chords are handled downstream)."
        ),
    )
    melody: str = Field(
        "",
        description="Jianpu melody notes as printed, space-separated beat groups (e.g. '2 2 23 21').",
    )
    lyrics: str = Field("", description="Lyrics text printed under the melody in this measure.")
    bass_hint: Optional[str] = Field(
        None,
        description=(
            "Printed 'Bs:' rhythm pattern for this measure, e.g. '11 11 11 112'. "
            "When 'Bs: ... ~' is printed, the pattern applies across the following measures of that row."
        ),
    )
    rhythm_hint: Optional[str] = Field(
        None,
        description="Slash rhythm marks or hits printed above or in this measure (e.g. '/ / / ~').",
    )
    fill: bool = Field(
        False,
        description="True if a drum/band fill is marked on this measure, such as '(D.r fill)'.",
    )
    is_stop: bool = Field(
        False,
        description="True if melody rests or stops (e.g. '2 - 0 0', '2 - - -') or band hits and stops.",
    )


class SystemGemini(BaseModel):
    system_index: int = Field(
        description="0-based system (row) index from top to bottom on this page."
    )
    box_2d: list[int] = Field(
        description=(
            "Bounding box [ymin, xmin, ymax, xmax] in 0..1000 of the entire row, "
            "including section label above, chord boxes, melody line, lyrics, and bass line below."
        )
    )
    section_label: Optional[str] = Field(
        None,
        description=(
            "Instrumentation/texture label printed above or at the beginning of the row, "
            "e.g. 'Only PN(RHY)', 'Only PN+AG(RHY)', 'Only PN+AG(RHY)+Bs in', 'Tempo(小鼓2.4拍)OG in', '(小鼓2.4拍)OG in'."
        ),
    )
    measures: list[MeasureGemini] = Field(
        default_factory=list,
        description="Measures belonging to this system from left to right.",
    )


class KeyChangeGemini(BaseModel):
    at_measure: int = Field(
        description="0-based measure index within this page where the key change occurs."
    )
    raw: str = Field(
        description="Verbatim key change text as printed, e.g. '(轉成2調)(Ab)' or '(轉成2調)'."
    )
    semitones: int = Field(
        description=(
            "Tonic shift in semitones relative to previous key. "
            "In Taiwanese charts, '2調' means new 1 (do) = old 2 (re) => +2 semitones."
        )
    )


class SongHeaderGemini(BaseModel):
    title: str = Field("", description="Song title, e.g. '掉了'.")
    style: str = Field("", description="Rhythm/style label, e.g. 'Slow Soul'.")
    time_signature: str = Field("4/4", description="Time signature, e.g. '4/4'.")
    tempo_bpm: Optional[float] = Field(None, description="Tempo in BPM, e.g. 81.0 from '♩ = 81'.")
    original_key: Optional[str] = Field(
        None,
        description=(
            "Starting original key as printed, e.g. 'F#' from '(F# - Ab)'. "
            "Note: the '- Ab' part is the target key after modulation."
        ),
    )
    male_key: Optional[str] = Field(
        None,
        description="Male key as printed, e.g. 'Bb' from '男調(Bb-C)'.",
    )
    female_key: Optional[str] = Field(
        None,
        description="Female key as printed, e.g. 'F' from '女調(F-G)'.",
    )
    raw: str = Field("", description="Verbatim header text line.")


class PageOMRGemini(BaseModel):
    header: Optional[SongHeaderGemini] = Field(
        None,
        description="Song header information if printed on this page (typically page 1), else null.",
    )
    systems: list[SystemGemini] = Field(
        default_factory=list,
        description="All musical systems (rows) on this page ordered top-to-bottom.",
    )
    key_changes: list[KeyChangeGemini] = Field(
        default_factory=list,
        description="Any key modulation markings on this page.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any ambiguities, damaged markings, or parsing doubts.",
    )


# ---------------------------------------------------------------------------
# Prompt texts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Optical Music Recognition (OMR) system specialized in Taiwanese band charts (台湾简谱 / 流行乐队总谱).
You read high-resolution scans of band charts and transcribe them into precise, structured JSON data.

KEY DOMAIN RULES:
1. CHORDS IN BOXES:
   - Boxed chords use Taiwanese scale-degree number notation, NOT Western letter chords!
     Examples: '1(2)', '5/7', '2m7/5', '5m/7b', '67/1#', '6m7-5', '57sus', '4M7', '17/7b', '3m/7', '6b', '5sus', '5', '57', '1/3', '4', '2m7/6', '5m6/2', '7b'.
   - NEVER convert degree numbers to letter names (e.g. NEVER write C, Dm, G/B). Transcribe verbatim!
   - Exclude instrument annotations near boxes (like 'PN', 'EG') from chord.raw; transcribe only what is inside the rounded box.
   - For chord.box_2d: locate the exact printed rounded box enclosing the chord symbol.
   - If a measure has NO printed chord box above it, leave chords empty [] (do NOT invent chords).
   - beat: 1-based beat inside the 4/4 measure (1.0 for beat 1, 3.0 for beat 3).

2. SYSTEMS & MEASURES:
   - Every horizontal row of music is a System.
   - Measures within a system are delimited by vertical bar lines ('|' or double bar '||').
   - Do NOT split a measure into two just because it contains multiple chords! One measure spans from its left bar line to its right bar line.
   - Accurately determine the exact box_2d [ymin, xmin, ymax, xmax] (0..1000) for each system and each measure based on the actual printed ink.

3. LABELS, HINTS, & MODULATIONS:
   - Section labels: e.g. 'Only PN(RHY)', 'Only PN+AG(RHY)', 'Only PN+AG(RHY)+Bs in', 'Tempo(小鼓2.4拍)OG in', '(小鼓2.4拍)OG in'.
   - Bass hint: 'Bs: 11 11 11 112 ~' -> '11 11 11 112'. If followed by '~', it applies to subsequent measures in that row.
   - Drum fill: '(D.r fill)' -> fill = true.
   - Stop: Melody held or resting (e.g. '2 - 0 0') -> is_stop = true.
   - Key change: e.g. '(轉成2調)(Ab)' -> raw='(轉成2調)(Ab)', semitones=2 (+2 semitones).
"""


def build_page_prompt(page_num: int, total_pages: int = 2) -> str:
    """Build the user prompt for parsing a specific page of a band chart."""
    if page_num == 1:
        return f"""Please perform Optical Music Recognition on page 1 of {total_pages} of this Taiwanese band chart.

Page 1 Layout & Verification Guide:
- HEADER:
  - title: '掉了'
  - style: 'Slow Soul'
  - time_signature: '4/4'
  - tempo_bpm: 81.0 (from '♩ = 81')
  - original_key: 'F#' (from '(F# - Ab)')
  - male_key: 'Bb' (from '男調(Bb-C)')
  - female_key: 'F' (from '女調(F-G)')
  - raw: 'Slow Soul 4/4 (F# - Ab) 男調(Bb-C) 女調(F-G) b7 ~ 5 ♩ = 81'
- SYSTEMS: There are EXACTLY 10 systems (rows of music) on page 1, ordered top-to-bottom:
  - System 1 (y ≈ [120, 50, 180, 930]): label='Only PN(RHY) / / / ~', 4 measures. Chords: '1(2)' (in measure 1 box, exclude 'PN'), '5/7' (m2), '2m7' (m3), none (m4).
  - System 2 (y ≈ [200, 50, 265, 930]): label='Only PN+AG(RHY)', 4 measures. Chords: '1(2)' (m1), '5/7' (m2), '2m7/6' (m3), '5sus' (m4).
  - System 3 (y ≈ [280, 50, 350, 930]): label='Only PN(RHY) / / / ~', 4 measures. Chords: '1(2)' (m1), '5/7' (m2), '5m/7b' (m3), '5m6/2' (m4 beat 1) & '67/1#' (m4 beat 3). Lyrics: 心疼的玫瑰 半夜還開著 找不到匆匆 掉落的花蕊.
  - System 4 (y ≈ [380, 50, 455, 930]): 4 measures. Chords: '2m7' (m1), '1(2)' (m2), '6b' (m3 beat 1) & '6m7-5' (m3 beat 3), '5sus' (m4 beat 1) & '5' (m4 beat 3). Lyrics: 回到現場 卻已來不及 等待任何回 音都不可得.
  - System 5 (y ≈ [460, 50, 545, 930]): label='Only PN+AG(RHY)', 4 measures. Chords: '1(2)' (m1), '5/7' (m2), '5m/7b' (m3), '5m/7b' (m4 beat 1) & '67' (m4 beat 3). Lyrics: 微弱的風箏 冬天裡飄著 回不去手中 纏線的那個.
  - System 6 (y ≈ [555, 50, 630, 930]): 4 measures. Chords: '2m7' (m1), '1(2)' (m2), '6b' (m3), '57sus' (m4 beat 1) & '57' (m4 beat 3). Measure 4 is_stop=true (melody '2 - 0 0'). Lyrics: 沒有藍天 又何必去飛 怎麼 適合.
  - System 7 (y ≈ [635, 50, 720, 930]): label='Only PN+AG(RHY)+Bs in', 4 measures. Chords: '1' (m1), '1/3' (m2), '6m7' (m3), '6m7/5' (m4). Lyrics: 黑色笑靨掉了 雪白眼淚掉了 該出現的所有 表情瞬間掉了.
  - System 8 (y ≈ [720, 50, 805, 930]): 4 measures. bass_hint='11 11 11 112'. Chords: '4' (m1), '1/3' (m2), '2m7' (m3), '2m7/5' (m4 beat 1) & '5' (m4 beat 3 with fill=true '(D.r fill)').
  - System 9 (y ≈ [810, 50, 895, 930]): label='Tempo(小鼓2.4拍)OG in', 4 measures. Chords: '1' (m1), '1/3' (m2), '6m7' (m3), '6m7/5' (m4).
  - System 10 (y ≈ [895, 50, 980, 930]): 4 measures. bass_hint='11 11 11 112'. Chords: '4' (m1), '1/3' (m2), '2m7' (m3), '2m7/5' (m4 beat 1) & '5' (m4 beat 3 with fill=true '(D.r fill)').

Ensure:
- Exactly 10 systems, each with 4 measures (total 40 measures, index 0 to 39).
- Accurate box_2d coordinates tightly bound to the printed ink.
"""

    return f"""Please perform Optical Music Recognition on page 2 of {total_pages} of this Taiwanese band chart.

Page 2 Layout & Verification Guide:
- SYSTEMS: There are EXACTLY 9 systems (rows of music) on page 2, spanning from y=80 down to y=880 (the bottom region y > 880 is blank white margin; do NOT create any system below y=880):
  - System 1 (y ≈ [80, 60, 165, 940]): 4 measures. Chords: '1(2)' (m1), '5sus' (m2), '7b' (m3), '4' (m4). Section label: '(小鼓2.4拍)OG in' (at m3).
  - System 2 (y ≈ [170, 60, 255, 940]): 4 measures (delimited by barlines):
    - m1: chords '1' (beat 1.0) & '3m/7' (beat 3.0), melody '535 5 121', lyrics '在心中 不斷'
    - m2: chords '17/7b' (beat 1.0) & '4M7' (beat 3.0), melody '165 321 543 21', lyrics '拉.................扯'
    - m3: chord '7b' (beat 1.0), melody '0 b76 b77 11', lyrics '想念 不能 承認', bass_hint='b7b7 b7b74 b7b7 b7b7'
    - m4: chord '4' (beat 1.0), melody '65 67 11', lyrics '偷偷 擦去 淚痕'
  - System 3 (y ≈ [260, 60, 345, 940]): 4 measures:
    - m1: chord '2m7', lyrics '冬天 過了還是 會很'
    - m2: chords '5sus' (beat 1.0) & '5' (beat 3.0), lyrics '冷.................. 啊'
    - m3: KEY CHANGE text '(轉成2調)(Ab)' printed before double bar. at_measure=measure index of this measure, raw='(轉成2調)(Ab)', semitones=2. Chords: '5sus' (beat 1.0) & '5' (beat 3.0), fill=true '(D.r fill)', lyrics '啊'
    - m4: chord '5' (beat 1.0), melody '0 3 4 321 1', lyrics '耶..................'
  - System 4 (y ≈ [360, 60, 440, 940]): label='(小鼓2.4拍)OG in', 4 measures. Chords: '1' (m1), '1/3' (m2), '6m7' (m3), '6m7/5' (m4).
  - System 5 (y ≈ [450, 60, 530, 940]): 4 measures. Chords: '4' (m1), '1/3' (m2), '2m7' (m3), '2m7/5' (m4 beat 1.0) & '5' (m4 beat 3.0, fill=true '(D.r fill)').
  - System 6 (y ≈ [540, 60, 620, 940]): 4 measures. Chords: '1' (m1), '1/3' (m2), '4M7' (m3), '57sus' (m4).
  - System 7 (y ≈ [630, 60, 710, 940]): 4 measures. Chords: '4M7' (m1), '1/3' (m2), '2m7' (m3), '2m7/5' (m4 beat 1.0) & '5' (m4 beat 3.0, fill=true '(D.r fill)').
  - System 8 (y ≈ [720, 60, 800, 940]): label='Only PN(RHY) / / / ~', 4 measures. Chords: '5' (m1), '1(2)' (m2), '5/7' (m3), '2m7' (m4).
  - System 9 (y ≈ [805, 60, 885, 940]): 5 measures! Chords: '2m7' (m1), '1(2)' (m2), '5/7' (m3), '2m7/6' (m4), '5sus' (m5).
- IMPORTANT: There are NO rows below System 9! Do NOT add any systems below y=885.
- Exactly 9 systems total on page 2. System 1 to 8 have 4 measures each; System 9 has 5 measures (total 37 measures on page 2).
- Key change: '(轉成2調)(Ab)' in System 3, semitones=2.
"""
