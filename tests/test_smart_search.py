"""Unit tests for app.smart.search (smart chord chart search & consensus).

All tests in this suite are OFFLINE ONLY (no external network or live LLM calls).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.smart.models import LeadSheet
from app.smart.search import (
    ConsensusBar,
    ConsensusSection,
    SearchResult,
    SourceChart,
    SourceSection,
    build_consensus,
    get_search_cache_path,
    load_cached_search,
    normalize_source_chart,
    save_cached_search,
    search_result_to_lead,
    search_song,
    simplify_chord_for_voting,
)


def test_key_normalization_with_capo_and_transposed_sources():
    """Verify capo and played_key transpositions normalize into the sounding key."""
    # Source 1: Played G with Capo 3 -> sounding Bb
    raw_s1 = {
        "url": "https://source1.com/tab",
        "title": "Song Title",
        "key": "Bb",
        "capo": 3,
        "played_key": "G",
        "raw_notation": "letter",
        "sections": [
            {
                "label": "前奏",
                "bars": [["G"], ["Bm"], ["C"], ["D"]],
            }
        ],
    }
    src1 = normalize_source_chart(raw_s1)
    assert src1 is not None
    assert src1.key == "Bb"
    assert src1.capo == 3
    assert src1.played_key == "G"
    assert len(src1.sections) == 1
    # G in key G -> 1, Bm -> 3m, C -> 4, D -> 5
    assert src1.sections[0].bars == [["1"], ["3m"], ["4"], ["5"]]

    # Source 2: Capo 0 in Bb, chords written directly in Bb
    raw_s2 = {
        "url": "https://source2.com/tab",
        "title": "Song Title",
        "key": "Bb",
        "capo": 0,
        "played_key": "Bb",
        "raw_notation": "letter",
        "sections": [
            {
                "label": "前奏",
                "bars": [["Bb"], ["Dm"], ["Eb"], ["F"]],
            }
        ],
    }
    src2 = normalize_source_chart(raw_s2)
    assert src2 is not None
    assert src2.key == "Bb"
    # Bb in key Bb -> 1, Dm -> 3m, Eb -> 4, F -> 5
    assert src2.sections[0].bars == [["1"], ["3m"], ["4"], ["5"]]

    # Source 3: Played D with Capo 4 -> sounding F#
    raw_s3 = {
        "url": "https://source3.com/tab",
        "title": "Song Title",
        "key": "F#",
        "capo": 4,
        "played_key": "D",
        "raw_notation": "letter",
        "sections": [
            {
                "label": "主歌",
                "bars": [["D"], ["F#m"], ["G"], ["A"]],
            }
        ],
    }
    src3 = normalize_source_chart(raw_s3)
    assert src3 is not None
    assert src3.key == "F#"
    # D in key D -> 1, F#m -> 3m, G -> 4, A -> 5
    assert src3.sections[0].bars == [["1"], ["3m"], ["4"], ["5"]]


def test_letter_to_number_conversion():
    """Verify letter chords are converted to Taiwanese number notation relative to the key."""
    raw = {
        "url": "https://chord4.com/chords",
        "key": "Bb",
        "played_key": "Bb",
        "capo": 0,
        "raw_notation": "letter",
        "sections": [
            {
                "label": "前奏",
                "bars": [
                    ["Bb", "F/A"],
                    ["Gm", "Dm7"],
                    ["Ebmaj7", "Cm7"],
                    ["F7sus4", "F"],
                ],
            }
        ],
    }
    chart = normalize_source_chart(raw)
    assert chart is not None
    bars = chart.sections[0].bars
    assert len(bars) == 4
    # Bb=1, F/A=5/7
    assert bars[0] == ["1", "5/7"]
    # Gm=6m, Dm7=3m7
    assert bars[1] == ["6m", "3m7"]
    # Ebmaj7=4maj7, Cm7=2m7
    assert bars[2] == ["4maj7", "2m7"]
    # F7sus4=57sus, F=5
    assert bars[3] == ["57sus", "5"]


def test_consensus_voting_three_sources():
    """Consensus voting with 3 fake sources: one transposed, one with an extra bar."""
    # Source 1: Standard C major, 4 bars: 1, 5, 6m, 4
    s1 = SourceChart(
        url="https://s1.org/tab",
        title="Fake Song",
        key="C",
        sections=[
            SourceSection(
                label="前奏",
                bars=[["1"], ["5"], ["6m"], ["4"]],
            )
        ],
    )

    # Source 2: Transposed to G (Capo 5, sounding C) -> normalized to number notation 1, 5, 6m, 4
    s2 = SourceChart(
        url="https://s2.org/tab",
        title="Fake Song",
        key="C",
        capo=5,
        played_key="G",
        sections=[
            SourceSection(
                label="前奏",
                bars=[["1"], ["5"], ["6m"], ["4"]],
            )
        ],
    )

    # Source 3: Standard C major, but has an extra bar in position 1: 1, 1sus4, 5, 6m, 4
    s3 = SourceChart(
        url="https://s3.org/tab",
        title="Fake Song",
        key="C",
        sections=[
            SourceSection(
                label="前奏",
                bars=[["1"], ["1sus4"], ["5"], ["6m"], ["4"]],
            )
        ],
    )

    result = build_consensus([s1, s2, s3], title="Fake Song", artist="Test Artist")
    assert result.key == "C"
    assert len(result.consensus_sections) == 1
    sec = result.consensus_sections[0]
    assert sec.label == "前奏"
    assert len(sec.bars) == 4

    # All 4 reference bars should have winner chords ["1"], ["5"], ["6m"], ["4"]
    assert sec.bars[0].chords == ["1"]
    assert sec.bars[0].support == 3
    assert sec.bars[0].total == 3

    assert sec.bars[1].chords == ["5"]
    assert sec.bars[1].support == 3
    assert sec.bars[1].total == 3

    assert sec.bars[2].chords == ["6m"]
    assert sec.bars[2].support == 3
    assert sec.bars[2].total == 3

    assert sec.bars[3].chords == ["4"]
    assert sec.bars[3].support == 3
    assert sec.bars[3].total == 3

    # Check lead conversion
    lead = search_result_to_lead(result)
    assert isinstance(lead, LeadSheet)
    assert lead.chords_only is True
    assert lead.key == "C"
    assert len(lead.measures) == 4
    for m in lead.measures:
        for c in m.chords:
            assert c.agreement == "agree"


def test_chord_simplification_for_voting():
    """Verify extensions simplify properly for grouping votes while keeping full spelling."""
    # 1, 1add9, 1maj7 should share root 1, maj quality, and no bass
    sim1 = simplify_chord_for_voting("1")
    sim1_add9 = simplify_chord_for_voting("1add9")
    sim1_maj7 = simplify_chord_for_voting("1maj7")
    assert sim1 == sim1_add9 == sim1_maj7

    # 6m and 6m7 should share root 6, minor quality
    sim6m = simplify_chord_for_voting("6m")
    sim6m7 = simplify_chord_for_voting("6m7")
    assert sim6m == sim6m7

    # 5/7 and 57/7 should share root 5, bass 7
    sim5_7 = simplify_chord_for_voting("5/7")
    sim57_7 = simplify_chord_for_voting("57/7")
    assert sim5_7 == sim57_7


def test_search_result_to_lead_structure():
    """Verify search_result_to_lead produces compliant LeadSheet data structures."""
    s1 = SourceChart(url="https://s1.com", title="Source 1", key="G")
    s2 = SourceChart(url="https://s2.com", title="Source 2", key="G")
    s3 = SourceChart(url="https://s3.com", title="Source 3", key="G")

    res = SearchResult(
        title="晴天",
        artist="周杰伦",
        key="G",
        time_signature="4/4",
        tempo_bpm=86.0,
        sources=[s1, s2, s3],
        consensus_sections=[
            ConsensusSection(
                label="前奏",
                bars=[
                    ConsensusBar(chords=["1", "5/7"], support=2, total=3, alternatives=["5"]),
                    ConsensusBar(chords=["6m"], support=1, total=3, alternatives=["6m7", "4"]),
                ],
            ),
            ConsensusSection(
                label="主歌",
                bars=[
                    ConsensusBar(chords=["4"], support=3, total=3, alternatives=[]),
                    ConsensusBar(chords=["5"], support=2, total=2, alternatives=[]),
                ],
            ),
        ],
    )

    lead = search_result_to_lead(res)
    assert lead.title == "晴天"
    assert lead.artist == "周杰伦"
    assert lead.key == "G"
    assert lead.time_signature == "4/4"
    assert lead.tempo_bpm == 86.0
    assert lead.chords_only is True
    assert lead.provenance == "网络来源整理（3 个来源）· 未经音频核对"
    assert len(lead.sources) == 3
    assert lead.sources[0].url == "https://s1.com"

    # Measures inspection
    assert len(lead.measures) == 4
    # Measure 0: Start of 前奏 -> section label '前奏', 2 chords at beat 1.0 and 3.0
    m0 = lead.measures[0]
    assert m0.section == "前奏"
    assert len(m0.chords) == 2
    assert m0.chords[0].raw == "1"
    assert m0.chords[0].beat == 1.0
    assert m0.chords[0].agreement == "agree"  # support=2, total=3 -> 2/3 >= 0.5
    assert m0.chords[1].raw == "5/7"
    assert m0.chords[1].beat == 3.0
    assert m0.chords[1].agreement == "agree"

    # Measure 1: Second bar of 前奏 -> section is None
    m1 = lead.measures[1]
    assert m1.section is None
    assert len(m1.chords) == 1
    assert m1.chords[0].raw == "6m"
    assert m1.chords[0].agreement == "search_only"  # support=1 < 2
    assert m1.chords[0].alternatives == ["6m7", "4"]

    # Measure 2: Start of 主歌 -> section label '主歌'
    m2 = lead.measures[2]
    assert m2.section == "主歌"
    assert len(m2.chords) == 1
    assert m2.chords[0].raw == "4"
    assert m2.chords[0].agreement == "agree"  # support=3, total=3

    # Measure 3: Second bar of 主歌 -> section is None
    m3 = lead.measures[3]
    assert m3.section is None
    assert len(m3.chords) == 1
    assert m3.chords[0].raw == "5"
    assert m3.chords[0].agreement == "agree"


def test_search_song_never_raises_on_garbage_json(monkeypatch):
    """Verify search_song returns empty result + warnings on any bad inputs or LLM failures."""
    monkeypatch.delenv("QA_OFFLINE", raising=False)

    # Case 1: Empty input
    empty_res = search_song("", "")
    assert isinstance(empty_res, SearchResult)
    assert len(empty_res.warnings) > 0
    assert empty_res.sources == []

    # Case 2: Client raises exception
    with patch("google.genai.Client", side_effect=RuntimeError("API Network Down")):
        fail_res = search_song("Some Song", "Some Artist")
        assert isinstance(fail_res, SearchResult)
        assert len(fail_res.warnings) > 0
        assert "API Network Down" in fail_res.warnings[0]

    # Case 3: Model returns unparseable garbage text
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Error 500: Server encountered unexpected HTML <html><body>Crash</body></html>"
    mock_resp.candidates = []
    mock_client.models.generate_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        bad_res = search_song("Broken Song", "Broken Artist")
        assert isinstance(bad_res, SearchResult)
        assert len(bad_res.warnings) > 0
        assert bad_res.sources == []


def _assert_no_forbidden_keys(d: Any, forbidden_terms: set[str]) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            assert k not in forbidden_terms, f"Found forbidden key in serialized JSON: {k}"
            _assert_no_forbidden_keys(v, forbidden_terms)
    elif isinstance(d, list):
        for item in d:
            _assert_no_forbidden_keys(item, forbidden_terms)


def test_no_source_text_fields_in_models():
    """Verify no full text, raw text, lyrics, or tab content fields exist in data contract."""
    forbidden_terms = {
        "text",
        "full_text",
        "raw_text",
        "lyrics",
        "lyric",
        "content",
        "tab_text",
        "chord_text",
        "body",
    }

    models_to_check = [
        SourceSection,
        SourceChart,
        ConsensusBar,
        ConsensusSection,
        SearchResult,
    ]

    for model in models_to_check:
        fields = set(model.model_fields.keys())
        overlap = fields.intersection(forbidden_terms)
        assert not overlap, f"Model {model.__name__} has forbidden text fields: {overlap}"

    # Also verify dumped JSON dict has no forbidden keys
    dummy_res = SearchResult(
        title="Test",
        artist="Artist",
        key="C",
        sources=[
            SourceChart(
                url="https://ex.com",
                title="T",
                key="C",
                sections=[SourceSection(label="前奏", bars=[["1"], ["5"]])],
            )
        ],
        consensus_sections=[
            ConsensusSection(label="前奏", bars=[ConsensusBar(chords=["1"], support=1, total=1)])
        ],
    )
    dumped = dummy_res.model_dump()
    _assert_no_forbidden_keys(dumped, forbidden_terms)


def test_offline_mode(monkeypatch):
    """Verify search_song returns immediately with warning when QA_OFFLINE=1."""
    monkeypatch.setenv("QA_OFFLINE", "1")
    res = search_song("Song", "Artist")
    assert isinstance(res, SearchResult)
    assert res.sources == []
    assert any("QA_OFFLINE" in w for w in res.warnings)


def test_cache_save_and_load(tmp_path, monkeypatch):
    """Verify search result can be safely saved to and loaded from local cache."""
    # Redirect cache base directory to tmp_path
    monkeypatch.setattr(
        "app.smart.search.get_search_cache_path",
        lambda title, artist="": tmp_path / f"{title}__{artist}.json",
    )

    res = SearchResult(
        title="CacheSong",
        artist="CacheArtist",
        key="Bb",
        time_signature="4/4",
        sources=[
            SourceChart(
                url="https://cache.com",
                title="C",
                key="Bb",
                sections=[SourceSection(label="前奏", bars=[["1"]])],
            )
        ],
        consensus_sections=[
            ConsensusSection(label="前奏", bars=[ConsensusBar(chords=["1"], support=1, total=1)])
        ],
    )

    save_cached_search(res)
    loaded = load_cached_search("CacheSong", "CacheArtist")
    assert loaded is not None
    assert loaded.title == "CacheSong"
    assert loaded.artist == "CacheArtist"
    assert loaded.key == "Bb"
    assert len(loaded.sources) == 1
    assert loaded.sources[0].url == "https://cache.com"


def test_sources_must_have_real_urls_and_no_placeholder_hosts():
    """Verify that placeholder hosts are rejected and real chord site URLs are accepted."""
    from app.smart.search import FORBIDDEN_HOSTS, is_valid_source_url

    # Rejected placeholder / invalid URLs
    assert not is_valid_source_url("")
    assert not is_valid_source_url("not_a_url")
    assert not is_valid_source_url("ftp://example.com/sheet")
    assert not is_valid_source_url("https://example.com/chords")
    assert not is_valid_source_url("http://example.org/tab")
    assert not is_valid_source_url("https://sub.example.com/page")
    assert not is_valid_source_url("http://localhost:8000/tab")
    assert not is_valid_source_url("http://127.0.0.1/tab")
    assert not is_valid_source_url("https://vertexaisearch.cloud.google.com/grounding-api-redirect/123")

    # Accepted real chord URLs
    assert is_valid_source_url("https://tabs.ultimate-guitar.com/tab/5166403")
    assert is_valid_source_url("https://chord4.com/tabs/16433")
    assert is_valid_source_url("https://hopamchuan.com/song/28868/far-away/")
    assert is_valid_source_url("http://chords-haven.blogspot.com/2007/05/blog-post_3007.html")
    assert is_valid_source_url("https://www.tan8.com/unionscore-123.html")

    # normalize_source_chart rejects placeholder hosts by default
    placeholder_raw = {
        "url": "https://example.com/tab",
        "title": "Placeholder Tab",
        "key": "C",
        "sections": [
            {"label": "前奏", "bars": [["1"], ["5"]]},
            {"label": "主歌", "bars": [["1"], ["5"]]},
        ],
    }
    assert normalize_source_chart(placeholder_raw) is None
    # Allowed when allow_test_urls=True
    assert normalize_source_chart(placeholder_raw, allow_test_urls=True) is not None

    # normalize_source_chart accepts real URLs
    real_raw = {
        "url": "https://chord4.com/tabs/12345",
        "title": "Real Tab",
        "key": "C",
        "sections": [
            {"label": "前奏", "bars": [["1"], ["5"]]},
            {"label": "主歌", "bars": [["1"], ["5"]]},
        ],
    }
    assert normalize_source_chart(real_raw) is not None

