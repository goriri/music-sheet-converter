"""Page-geometry contract: produced by classical CV (app/omr/layout.py), consumed by the content reader
(app/omr/reader.py). All coordinates are normalized (0..1) relative to the page image, origin top-left.

Design: geometry (where rows, bars and chord boxes are) comes from the image itself; LLMs only read the
*content* of crops (chord text, melody, labels). This keeps bar/beat placement exact.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models import BBox


class GMeasure(BaseModel):
    index_in_system: int
    x0: float = Field(description="Left barline x (normalized)")
    x1: float = Field(description="Right barline x (normalized)")
    left_double_bar: bool = False
    right_double_bar: bool = False


class GChordBox(BaseModel):
    bbox: BBox
    measure_index_in_system: int
    beat_geo: float = Field(description="1-based beat estimated from x position inside the measure")
    boxed: bool = Field(True, description="False for unboxed letter/number chords detected as text")


class GSystem(BaseModel):
    page: int
    index_on_page: int
    bbox: BBox = Field(description="Full row: from above chord boxes/labels down to the last lyric/Bs line")
    melody_band: tuple[float, float] = Field(description="(y0, y1) of the jianpu melody digits line")
    chord_band: Optional[tuple[float, float]] = None
    label_band: Optional[tuple[float, float]] = Field(None, description="Section/texture labels above chords")
    lyric_bands: list[tuple[float, float]] = Field(default_factory=list)
    bass_band: Optional[tuple[float, float]] = Field(None, description="'Bs:' line if present")
    measures: list[GMeasure] = Field(default_factory=list)
    chord_boxes: list[GChordBox] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class PageGeometry(BaseModel):
    page: int
    width: int
    height: int
    header_band: Optional[tuple[float, float]] = Field(None, description="Title/header area above first row")
    systems: list[GSystem] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="min over systems, lowered by global oddities")
    notes: list[str] = Field(default_factory=list)
