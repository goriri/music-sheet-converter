"""Offline unit tests for web evidence extraction, transposition, and alignment."""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from app.models import ChordSymbol, KeyChange, Measure, PageInfo, ParsedSheet, SongHeader, System
from app.qa.web_evidence import (
    ALIGNMENT_QUALITY_THRESHOLD,
    DegreeChord,
    RefCandidate,
    ReferenceChart,
    ReferenceSource,
    align_reference,
    determine_reference_tonic_pc,
    find_reference,
    get_cache_path,
    load_cached_reference,
    needleman_wunsch,
    pairwise_chord_score,
    parse_chord_to_degree,
    save_cached_reference,
)


def _make_synthetic_sheet(
    chord_seq: list[list[str]],
    start_key: str = "C",
    key_changes: list[KeyChange] | None = None,
) -> ParsedSheet:
    """Build a minimal valid ParsedSheet from a list of measure chord strings."""
    measures = []
    for idx, chords in enumerate(chord_seq):
        symbols = [ChordSymbol(raw=c, beat=1.0) for c in chords]
        measures.append(
            Measure(
                index=idx,
                bbox=(0.0, 0.0, 1.0, 1.0),
                chords=symbols,
            )
        )
    system = System(page=0, bbox=(0.0, 0.0, 1.0, 1.0), measures=measures)
    header = SongHeader(
        title="Synthetic Test Song",
        original_key=start_key,
        chord_notation="number",
    )
    return ParsedSheet(
        header=header,
        pages=[PageInfo(width=1000, height=1400)],
        systems=[system],
        key_changes=key_changes or [],
    )


def test_transposition_with_capo():
    """Verify chords with capo are correctly transposed to scale degrees."""
    # Source with key G, capo 2 (sounding key A, play key G)
    src_capo = ReferenceSource(
        url="http://example.com/tab1",
        key="原调A 选调G",
        capo=2,
        lines=[["G", "Em", "C", "D", "G/B"]],
    )
    tonic_pc = determine_reference_tonic_pc(src_capo)
    assert tonic_pc == 7  # G

    chords = [parse_chord_to_degree(tok, tonic_pc=tonic_pc) for tok in src_capo.lines[0]]
    assert all(c is not None for c in chords)

    # G in key G -> 1
    assert chords[0].degree == 1 and chords[0].quality == "maj" and chords[0].number_str == "1"
    # Em in key G -> 6m
    assert chords[1].degree == 6 and chords[1].quality == "m" and chords[1].number_str == "6m"
    # C in key G -> 4
    assert chords[2].degree == 4 and chords[2].quality == "maj" and chords[2].number_str == "4"
    # D in key G -> 5
    assert chords[3].degree == 5 and chords[3].quality == "maj" and chords[3].number_str == "5"
    # G/B in key G -> 1/3 (root 1, bass 3)
    assert chords[4].degree == 1 and chords[4].bass_degree == 3 and chords[4].number_str == "1/3"


def test_transposition_slash_chords_and_accidentals():
    """Verify slash chords and flat/sharp accidentals convert to number strings."""
    # Key C
    d1 = parse_chord_to_degree("Bb", tonic_pc=0)
    assert d1 is not None
    assert d1.degree == 7 and d1.accidental == -1 and d1.number_str == "b7"

    d2 = parse_chord_to_degree("Fm", tonic_pc=0)
    assert d2 is not None
    assert d2.degree == 4 and d2.quality == "m" and d2.number_str == "4m"

    d3 = parse_chord_to_degree("G/B", tonic_pc=0)
    assert d3 is not None
    assert d3.number_str == "5/7"


def test_alignment_with_known_deleted_measure():
    """Verify alignment correctly handles a deleted measure in reference."""
    # Sheet has 8 measures: 1, 6m, 4, 5, 1, 6m, 4, 5
    sheet = _make_synthetic_sheet([["1"], ["6m"], ["4"], ["5"], ["1"], ["6m"], ["4"], ["5"]])

    # Ref has measure 2 ('4') deleted: 1, 6m, 5, 1, 6m, 4, 5
    ref = ReferenceChart(
        title="Synthetic Test Song",
        sources=[
            ReferenceSource(
                url="http://example.com/ref",
                key="C",
                capo=0,
                lines=[["C", "Am", "G", "C", "Am", "F", "G"]],
            )
        ],
    )

    candidates = align_reference(sheet, ref, start_key_pc=0)
    assert bool(candidates), "Alignment should succeed above threshold"

    # Measures 0, 1 should align to 1, 6m
    assert candidates[0][0].chord == "1" and candidates[0][0].support == 1.0
    assert candidates[1][0].chord == "6m" and candidates[1][0].support == 1.0

    # Measures 3, 4, 5, 6, 7 should align accurately
    assert candidates[3][0].chord == "5" and candidates[3][0].support == 1.0
    assert candidates[4][0].chord == "1" and candidates[4][0].support == 1.0
    assert candidates[5][0].chord == "6m" and candidates[5][0].support == 1.0
    assert candidates[6][0].chord == "4" and candidates[6][0].support == 1.0
    assert candidates[7][0].chord == "5" and candidates[7][0].support == 1.0


def test_alignment_with_known_inserted_measure():
    """Verify alignment correctly handles an inserted measure in reference."""
    # Sheet has 4 measures: 1, 6m, 4, 5
    sheet = _make_synthetic_sheet([["1"], ["6m"], ["4"], ["5"]])

    # Ref has an inserted chord '3m' between 1 and 6m: C, Em, Am, F, G
    ref = ReferenceChart(
        title="Synthetic Test Song",
        sources=[
            ReferenceSource(
                url="http://example.com/ref",
                key="C",
                capo=0,
                lines=[["C", "Em", "Am", "F", "G"]],
            )
        ],
    )

    candidates = align_reference(sheet, ref, start_key_pc=0)
    assert bool(candidates)
    assert candidates[0][0].chord == "1"
    assert candidates[1][0].chord == "6m"
    assert candidates[2][0].chord == "4"
    assert candidates[3][0].chord == "5"


def test_poor_alignment_returns_empty_dict():
    """Verify completely mismatched reference chords return {} due to poor alignment score."""
    sheet = _make_synthetic_sheet([["1"], ["6m"], ["4"], ["5"], ["1"], ["6m"], ["4"], ["5"]])

    # Ref with completely unrelated / discordant chords that don't match the sheet
    ref = ReferenceChart(
        title="Wrong Song",
        sources=[
            ReferenceSource(
                url="http://example.com/wrong",
                key="C",
                capo=0,
                lines=[["F#", "B", "C#", "D#m", "G#m", "C#", "F#"]],
            )
        ],
    )

    candidates = align_reference(sheet, ref, start_key_pc=0)
    assert candidates == {}, "Poor alignment quality must return empty dict {}"


def test_multiple_sources_agreement():
    """Verify multiple sources increase source count and candidate ranking."""
    sheet = _make_synthetic_sheet([["1"], ["6m"], ["4"], ["5"]])

    ref = ReferenceChart(
        title="Multi Source Song",
        sources=[
            ReferenceSource(
                url="http://example.com/src1",
                key="C",
                capo=0,
                lines=[["C", "Am", "F", "G"]],
            ),
            ReferenceSource(
                url="http://example.com/src2",
                key="C",
                capo=0,
                lines=[["C", "Am7", "Fmaj7", "G7"]],
            ),
        ],
    )

    candidates = align_reference(sheet, ref, start_key_pc=0)
    assert len(candidates) == 4
    # Measure 0: Both sources provided '1'
    cand_m0 = candidates[0][0]
    assert cand_m0.chord == "1"
    assert cand_m0.sources == 2
    assert cand_m0.support == 1.0


def test_cache_hit_offline(tmp_path, monkeypatch):
    """Verify cache hit works offline without calling LLM."""
    monkeypatch.setenv("QA_OFFLINE", "1")
    monkeypatch.delenv("WEB_EVIDENCE", raising=False)

    title = "Test Cache Hit Song"
    artist = "Test Artist"

    cached_chart = ReferenceChart(
        title=title,
        artist=artist,
        sources=[
            ReferenceSource(
                url="http://example.com/cached",
                key="C",
                capo=0,
                lines=[["1", "5/7", "6m", "4"]],
            )
        ],
    )

    # Save to cache
    save_cached_reference(cached_chart)
    cache_file = get_cache_path(title, artist)
    assert cache_file.is_file()

    # find_reference should return cached chart directly
    result = find_reference(title, artist)
    assert result is not None
    assert result.title == title
    assert len(result.sources) == 1
    assert result.sources[0].url == "http://example.com/cached"

    # Cleanup test cache file
    if cache_file.is_file():
        cache_file.unlink()


def test_web_evidence_disabled_by_env(monkeypatch):
    """Verify WEB_EVIDENCE=0 disables web evidence completely."""
    monkeypatch.setenv("WEB_EVIDENCE", "0")
    result = find_reference("Any Song", "Any Artist")
    assert result is None


def test_salvage_truncated_json_mid_line():
    """Verify a JSON cut off mid-line gets salvaged and recovers complete lines."""
    from app.qa.web_evidence import _clean_and_filter_sources, salvage_json_dict

    truncated_raw = """{
      "title": "听妈妈的话",
      "artist": "周杰伦",
      "sources": [
        {
          "url": "http://tab.com",
          "key": "Bb",
          "capo": 3,
          "lines": [
            ["G", "D/F#", "Em", "C"],
            ["G", "D/F#", "Em", "C", "G/B", "Am7", "D"],
            ["G", "B7/Eb", "Em", "Em/D
    """
    salvaged = salvage_json_dict(truncated_raw)
    assert salvaged is not None
    chart = ReferenceChart.model_validate(salvaged)
    assert chart.title == "听妈妈的话"
    assert len(chart.sources) == 1

    clean_sources = _clean_and_filter_sources(chart.sources, ["http://fallback.com"])
    assert len(clean_sources) == 1
    # Only the first 2 complete lines must be kept; the incomplete 3rd line is dropped
    assert len(clean_sources[0].lines) == 2
    assert clean_sources[0].lines[0] == ["G", "D/F#", "Em", "C"]
    assert clean_sources[0].lines[1] == ["G", "D/F#", "Em", "C", "G/B", "Am7", "D"]


def test_salvage_truncated_json_mid_source():
    """Verify a JSON cut off during second source recovers the first source."""
    from app.qa.web_evidence import salvage_json_dict

    truncated_raw = """{
      "title": "听妈妈的话",
      "artist": "周杰伦",
      "sources": [
        {
          "url": "http://src1.com",
          "key": "Bb",
          "lines": [["G", "D", "Em", "C"]]
        },
        {
          "url": "http://src2.com",
          "lines": [["G", "D"
    """
    salvaged = salvage_json_dict(truncated_raw)
    assert salvaged is not None
    chart = ReferenceChart.model_validate(salvaged)
    assert len(chart.sources) == 1
    assert chart.sources[0].url == "http://src1.com"
    assert chart.sources[0].lines == [["G", "D", "Em", "C"]]


def test_salvage_completely_broken_json_returns_none():
    """Verify completely broken non-JSON text returns None without raising any error."""
    from app.qa.web_evidence import salvage_json_dict

    assert salvage_json_dict("This is completely arbitrary text with no json <<>>") is None
    assert salvage_json_dict("") is None
    assert salvage_json_dict("{}{}{}") is None
    assert salvage_json_dict("[1, 2, 3]") is None  # not a dict

