"""Data contract for 智能创建 (smart sheet creation).

A LeadSheet is the song in exact rhythmic form (unlike ParsedSheet.melody, which is the lossy
printed jianpu string). Sources that produce a LeadSheet: the audio transcription job, the
multi-source web search (chords-only), and the cross-check that merges them. The from-scratch
renderer (app.smart.render) turns a LeadSheet into page images + a ParsedSheet with bboxes, so the
existing confirm → arrange → overlay → PDF → history pipeline is reused unchanged.

Conventions
- Chords are Taiwanese number notation relative to LeadSheet.key, the same grammar as
  ChordSymbol.raw ("1", "5/7", "2m7", "4maj7", "57sus4", "6m7-5", "b7"), parseable by
  app.theory.chords.parse_chord.
- Beats: LeadNote.onset is 0-based inside the measure; LeadChord.beat is 1-based (like ChordSymbol).
- Durations are in beats (quarter note = 1.0 in x/4; eighth note = 1.0 in x/8) and are multiples of 0.25.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Agreement = Literal["agree", "disagree", "audio_only", "search_only", "unchecked"]


class LeadNote(BaseModel):
    onset: float = Field(ge=0.0, description="0-based beat offset inside the measure")
    duration: float = Field(gt=0.0, description="Length in beats, multiple of 0.25; never crosses the barline (use tie_to_next)")
    degree: int = Field(ge=0, le=7, description="0 = rest, 1..7 = movable-do scale degree in LeadSheet.key")
    accidental: int = Field(0, ge=-1, le=1, description="-1 flat, 0 natural, +1 sharp")
    octave: int = Field(0, ge=-2, le=2, description="Octave dots: +1 = one dot above, -1 = one dot below")
    tie_to_next: bool = Field(False, description="Tied to the next note (same pitch), possibly across the barline")
    lyric: str = Field("", description="Syllable sung on this note (usually one CJK character); '' for melisma/rest")


class LeadChord(BaseModel):
    raw: str = Field(description="Taiwanese number chord relative to LeadSheet.key, e.g. '1', '5/7', '2m7'")
    beat: float = Field(1.0, ge=1.0, description="1-based beat inside the measure where the chord starts")
    agreement: Agreement = "unchecked"
    alternatives: list[str] = Field(default_factory=list, description="Competing readings from the other evidence source")


class LeadMeasure(BaseModel):
    beats: float = Field(4.0, gt=0.0, description="Length of this measure in beats (pickup bars are shorter)")
    notes: list[LeadNote] = Field(default_factory=list, description="Melody; empty = no melody known (chords-only)")
    chords: list[LeadChord] = Field(default_factory=list)
    section: Optional[str] = Field(None, description="Section label printed at the start of a section: '前奏', '主歌', '副歌', '间奏', '桥段', '尾奏', or 'A'/'B'…")


class SourceLink(BaseModel):
    title: str = ""
    url: str


class LeadSheet(BaseModel):
    title: str = ""
    artist: str = ""
    key: str = Field("C", description="Sounding key of '1', e.g. 'Bb', 'F#'")
    time_signature: str = "4/4"
    tempo_bpm: Optional[float] = None
    measures: list[LeadMeasure] = Field(default_factory=list)
    chords_only: bool = Field(False, description="True when there is no melody (title/artist-only mode)")
    provenance: str = Field("", description="User-facing origin, e.g. '音频转录 + 3 个来源核对' or '网络来源整理 · 未经音频核对'")
    sources: list[SourceLink] = Field(default_factory=list, description="Links only; third-party chart text is never stored")
