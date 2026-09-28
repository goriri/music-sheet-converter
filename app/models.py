"""Shared data contract for the whole pipeline.

Pipeline: page images -> OMR (ParsedSheet) -> arrangement (Arrangement) -> overlay renderer (PDF).

Conventions
- All bboxes are normalized floats (x0, y0, x1, y1) in [0, 1] relative to the page image,
  origin top-left.
- Chord symbols are kept exactly as printed in Taiwanese band-chart ("台湾谱") number
  notation, e.g. "1(2)", "5/7", "2m7/5", "5m/7b", "67/1#", "6m7-5", "57sus", "4M7", "17/7b".
  Accidentals may appear before or after the degree ("b7" == "7b").
- Measure.index is global across the whole song (0-based, page order then top-to-bottom).
- Beats are 1-based inside a measure for chord positions; Event.onset is 0-based.
- Pitches are MIDI numbers (60 = C4).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]

Difficulty = Literal["beginner", "intermediate", "advanced"]
Instrument = Literal["piano", "guitar", "ukulele"]


# --------------------------------------------------------------------------- OMR output
class ChordSymbol(BaseModel):
    raw: str = Field(description="Chord exactly as printed inside the box, e.g. '2m7/5'")
    beat: float = Field(1.0, description="1-based beat inside the measure where the chord starts")
    bbox: Optional[BBox] = Field(None, description="Box of the printed chord symbol")


class Measure(BaseModel):
    index: int
    bbox: BBox = Field(description="Measure region on the melody line; x-range is what matters")
    beats: float = 4.0
    chords: list[ChordSymbol] = Field(default_factory=list)
    melody: str = Field("", description="Jianpu melody as printed, space separated beat groups")
    lyrics: str = ""
    bass_hint: Optional[str] = Field(None, description="Printed 'Bs:' rhythm for this measure, e.g. '11 11 11 112'")
    rhythm_hint: Optional[str] = Field(None, description="Printed slash rhythm / hits over this measure, verbatim")
    fill: bool = Field(False, description="A drum/band fill ('D.r fill') is marked on this measure")
    is_stop: bool = Field(False, description="Melody is held/resting to a stop (e.g. '2 - 0 0'), band hits and stops")


class System(BaseModel):
    """One printed row (line) of the chart."""

    page: int = Field(description="0-based page index")
    bbox: BBox = Field(description="Whole row incl. chord boxes above and lyrics/Bs lines below")
    section_label: Optional[str] = Field(
        None, description="Instrumentation/texture label printed above the row, e.g. 'Only PN+AG(RHY)+Bs in'"
    )
    measures: list[Measure] = Field(default_factory=list)


class KeyChange(BaseModel):
    at_measure: int = Field(description="Global measure index where the new key starts")
    raw: str = Field(description="As printed, e.g. '轉成2調(Ab)'")
    semitones: int = Field(description="Shift of the tonic relative to the previous key, e.g. +2")


class SongHeader(BaseModel):
    title: str = ""
    style: str = Field("", description="e.g. 'Slow Soul'")
    time_signature: str = "4/4"
    tempo_bpm: Optional[float] = None
    original_key: Optional[str] = Field(None, description="Start key as printed, e.g. 'F#'")
    male_key: Optional[str] = None
    female_key: Optional[str] = None
    raw: str = Field("", description="Verbatim header line")


class PageInfo(BaseModel):
    width: int
    height: int


class ParsedSheet(BaseModel):
    header: SongHeader
    pages: list[PageInfo]
    systems: list[System]
    key_changes: list[KeyChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def measures(self) -> list[Measure]:
        return [m for s in self.systems for m in s.measures]


# --------------------------------------------------------------------------- theory
class ResolvedChord(BaseModel):
    raw: str
    name: str = Field(description="Concrete chord name in the target key, e.g. 'G/B', 'Dm7', 'Bbmaj7'")
    beat: float = 1.0
    root_pc: int = Field(description="Pitch class 0..11 (0=C)")
    bass_pc: int
    pcs: list[int] = Field(description="Chord tones as pitch classes, root first")
    quality: str = Field(description="Normalized quality, e.g. 'maj', 'm7', 'm7b5', '7sus4', 'add9'")


# --------------------------------------------------------------------------- arrangement output
class Note(BaseModel):
    midi: int
    finger: Optional[int] = Field(None, ge=1, le=5)


class Event(BaseModel):
    onset: float = Field(description="0-based beat offset inside the measure")
    duration: float = Field(description="Length in beats")
    notes: list[Note] = Field(default_factory=list, description="Empty = rest; >1 = chord")
    tie_to_next: bool = False


class MeasureArrangement(BaseModel):
    measure_index: int
    tonic_pc: int = Field(description="Tonic pitch class in force for this measure (for movable-do jianpu)")
    key_name: str
    chords: list[ResolvedChord]
    rh: list[Event] = Field(default_factory=list)
    lh: list[Event] = Field(default_factory=list)


class Arrangement(BaseModel):
    instrument: Instrument = "piano"
    difficulty: Difficulty = "intermediate"
    start_key: str
    style: str = ""
    measures: list[MeasureArrangement]
    notes: list[str] = Field(default_factory=list, description="Human-readable remarks for the player")
