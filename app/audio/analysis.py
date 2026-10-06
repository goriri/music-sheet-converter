"""Data models for audio transcription analysis (pure python/numpy only)."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChordSeg(BaseModel):
    """A detected chord segment."""
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    label: str = Field(description="Chord label, e.g. 'C:maj', 'A:min7', 'G:7', 'N'")
    bass_pc: Optional[int] = Field(None, ge=0, le=11, description="Bass pitch class 0..11 if detected")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")


class NoteSeg(BaseModel):
    """A detected melodic note segment."""
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    midi: float = Field(description="MIDI note pitch (e.g. 60.0 for C4)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Voicing/detection confidence")


class LyricChar(BaseModel):
    """A transcribed lyric character with timestamp."""
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    text: str = Field(description="Single lyric character or syllable")


class SectionSeg(BaseModel):
    """A detected structural section boundary."""
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    label: str = Field(description="Section label: '前奏', '主歌', '副歌', '间奏', '尾奏', etc.")


class AudioAnalysis(BaseModel):
    """Full analysis output of the audio transcription pipeline."""
    duration: float = Field(ge=0.0, description="Audio duration in seconds")
    tempo_bpm: Optional[float] = Field(None, gt=0.0, description="Estimated tempo in BPM")
    beats: list[float] = Field(default_factory=list, description="Beat onset timestamps in seconds")
    downbeats: list[float] = Field(default_factory=list, description="Downbeat (bar start) timestamps in seconds")
    time_signature: str = Field("4/4", description="Estimated time signature, e.g. '4/4', '3/4'")
    key: str = Field("C", description="Estimated tonic key, e.g. 'C', 'Bb', 'F#'")
    chords: list[ChordSeg] = Field(default_factory=list, description="Detected chord segments")
    notes: list[NoteSeg] = Field(default_factory=list, description="Detected melody note segments")
    lyrics: list[LyricChar] = Field(default_factory=list, description="Transcribed lyric characters")
    sections: list[SectionSeg] = Field(default_factory=list, description="Detected section segments")
    stage_timings: dict[str, float] = Field(default_factory=dict, description="Execution wall-time per stage in seconds")
    warnings: list[str] = Field(default_factory=list, description="Pipeline warnings or degradations")

