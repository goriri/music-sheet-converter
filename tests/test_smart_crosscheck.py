"""Unit tests for app.smart.crosscheck (smart lead sheet reconciliation).

All tests in this suite are OFFLINE ONLY (no live LLM calls or external network).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.smart.crosscheck import (
    CrossCheckReport,
    DisagreementItem,
    merge,
    transpose_chord_str,
    transpose_lead_sheet,
)
from app.smart.models import LeadChord, LeadMeasure, LeadNote, LeadSheet, SourceLink
from app.smart.search import (
    ConsensusBar,
    ConsensusSection,
    SearchResult,
    SourceChart,
    SourceSection,
)


def _make_measure(
    chord: str,
    melody_degrees: list[int] = (),
    section: str | None = None,
) -> LeadMeasure:
    notes = [
        LeadNote(onset=float(i), duration=1.0, degree=d, accidental=0, octave=0)
        for i, d in enumerate(melody_degrees)
    ]
    chords = [LeadChord(raw=chord, beat=1.0)] if chord else []
    return LeadMeasure(beats=4.0, notes=notes, chords=chords, section=section)


def test_transpose_chord_str():
    # In key C: 1 = C. In key D: C is b7.
    assert transpose_chord_str("1", from_key="C", to_key="D") == "b7"
    # In key D: 1 = D. In key C: D is 2.
    assert transpose_chord_str("1", from_key="D", to_key="C") == "2"
    # In key D: 6m = Bm. In key C: Bm is 7m.
    assert transpose_chord_str("6m", from_key="D", to_key="C") == "7m"
    # Same key should return same chord
    assert transpose_chord_str("4maj7", from_key="G", to_key="G") == "4maj7"
    # Invalid key or chord should not raise
    assert transpose_chord_str("???", from_key="C", to_key="D") == "???"


def test_search_only_mode():
    search = SearchResult(
        title="晴天",
        artist="周杰伦",
        key="G",
        sources=[SourceChart(url="https://source1.com", title="晴天吉他谱", key="G")],
        consensus_sections=[
            ConsensusSection(
                label="主歌",
                bars=[
                    ConsensusBar(chords=["1"], support=1, total=1),
                    ConsensusBar(chords=["5/7"], support=1, total=1),
                ],
            )
        ],
    )

    lead, report = merge(audio=None, search=search)

    assert report.mode == "search_only"
    assert "网络来源整理" in lead.provenance
    assert report.agreement_ratio == 1.0
    assert len(lead.measures) == 2
    assert lead.measures[0].section == "主歌"
    assert lead.measures[0].chords[0].raw == "1"


def test_audio_only_mode():
    audio = LeadSheet(
        title="晴天",
        artist="周杰伦",
        key="G",
        measures=[
            _make_measure("1", [1, 2, 3]),
            _make_measure("5", [5, 5]),
        ],
    )

    lead, report = merge(audio=audio, search=None)

    assert report.mode == "audio_only"
    assert "音频转录 · 未与网络来源核对" in lead.provenance
    assert report.agreement_ratio == 0.0
    assert report.n_audio_only == 2
    assert all(c.agreement == "audio_only" for m in lead.measures for c in m.chords)


def test_key_reconciliation_audio_wins_by_default():
    # Audio in C: 1, 6m, 4, 5
    audio = LeadSheet(
        title="测试曲目",
        key="C",
        measures=[
            _make_measure("1", [1]),
            _make_measure("6m", [6]),
            _make_measure("4", [4]),
            _make_measure("5", [5]),
        ],
    )
    # Search in D: chords were extracted relative to D: 1, 6m, 4, 5.
    # But wait! If search is in D and writes "2", in key C that is D major ("2").
    # If search in D has chords "2", "7m", "5", "6", when transposed to C:
    # 2 in D is E -> 3 in C.
    # Here let's test search consensus in key D with chords ["2"] (which is E in D):
    # When transposed from D to C, D's "1" becomes "2" in C!
    search = SearchResult(
        title="测试曲目",
        key="D",
        sources=[SourceChart(url="https://s1.com", key="D")],
        consensus_sections=[
            ConsensusSection(
                label="前奏",
                bars=[
                    ConsensusBar(chords=["1"], support=1, total=1),  # D in key D -> 2 in key C
                ],
            )
        ],
    )

    lead, report = merge(audio=audio, search=search)

    assert report.mode == "audio_only" or report.mode == "audio+search"
    assert "采用音频转录调性 C" in report.key_decision
    assert lead.key == "C"


def test_key_reconciliation_search_wins_when_audio_confidence_low():
    # Audio has key_confidence attribute set to 0.3
    audio = LeadSheet(
        title="低置信度测试",
        key="C",
        measures=[
            _make_measure("1", [1]),
            _make_measure("5", [5]),
        ],
    )
    # Attach key_confidence
    object.__setattr__(audio, "key_confidence", 0.3)

    search = SearchResult(
        title="低置信度测试",
        key="D",
        sources=[
            SourceChart(url="https://s1.com", key="D"),
            SourceChart(url="https://s2.com", key="D"),
        ],
        consensus_sections=[
            ConsensusSection(
                label="前奏",
                bars=[
                    ConsensusBar(chords=["2"], support=2, total=2),
                    ConsensusBar(chords=["6"], support=2, total=2),
                ],
            )
        ],
    )

    lead, report = merge(audio=audio, search=search)

    assert "采用网络来源调性 D" in report.key_decision
    assert lead.key == "D"
    # Audio chord "1" in C (C major) transposed to D becomes "b7"
    assert lead.measures[0].chords[0].raw == "b7"


def test_extra_audio_bars_alignment():
    # Audio has 4 bars (including 2 intro bars), search only has 2 bars
    audio = LeadSheet(
        title="Extra Bars",
        key="C",
        measures=[
            _make_measure("1", section="前奏"),
            _make_measure("5"),
            _make_measure("6m", section="主歌"),
            _make_measure("4"),
        ],
    )
    search = SearchResult(
        title="Extra Bars",
        key="C",
        sources=[SourceChart(url="https://s1.com", key="C")],
        consensus_sections=[
            ConsensusSection(
                label="主歌",
                bars=[
                    ConsensusBar(chords=["6m"], support=1, total=1),
                    ConsensusBar(chords=["4"], support=1, total=1),
                ],
            )
        ],
    )

    lead, report = merge(audio=audio, search=search)

    assert report.mode == "audio+search"
    assert report.n_agree == 2
    assert report.n_audio_only >= 1
    # Bars 2 and 3 agreed
    assert lead.measures[2].chords[0].agreement == "agree"
    assert lead.measures[3].chords[0].agreement == "agree"


def test_disagreement_arbitration_heuristic():
    # Offline environment (QA_OFFLINE=1)
    # Measure 0: Audio has chord '4', search consensus has '2m' (support 3/3).
    # Melody in measure 0: [2, 4, 6] -> Root of audio chord '4' IS 4, which is in melody!
    # Heuristic: audio chord root is in melody -> audio chord kept!
    #
    # Measure 1: Audio has chord '5', search consensus has '1' (support 3/3).
    # Melody in measure 1: [1, 3] -> Root of audio chord '5' is 5, NOT in melody!
    # Search support >= 2 and root not in melody -> search chord chosen!
    audio = LeadSheet(
        title="Arbitration Test",
        key="C",
        measures=[
            _make_measure("4", [4, 6, 2]),
            _make_measure("5", [1, 3]),
        ],
    )
    search = SearchResult(
        title="Arbitration Test",
        key="C",
        sources=[
            SourceChart(url="https://s1.com", key="C"),
            SourceChart(url="https://s2.com", key="C"),
            SourceChart(url="https://s3.com", key="C"),
        ],
        consensus_sections=[
            ConsensusSection(
                label="主歌",
                bars=[
                    ConsensusBar(chords=["2m"], support=3, total=3),
                    ConsensusBar(chords=["1"], support=3, total=3),
                ],
            )
        ],
    )

    with patch("app.smart.crosscheck.llm_available", return_value=False):
        lead, report = merge(audio=audio, search=search)

    assert report.n_disagree == 2
    assert len(report.disagreements) == 2

    # Measure 0: audio chord kept because root 4 is in melody
    d0 = report.disagreements[0]
    assert d0.measure_index == 0
    assert d0.chosen == "4"
    assert "包含于小节旋律音中" in d0.reason
    assert lead.measures[0].chords[0].raw == "4"
    assert lead.measures[0].chords[0].alternatives == ["2m"]
    assert lead.measures[0].chords[0].agreement == "disagree"

    # Measure 1: search chord chosen because root 5 not in melody and support >= 2
    d1 = report.disagreements[1]
    assert d1.measure_index == 1
    assert d1.chosen == "1"
    assert "音频和弦根音不在旋律音中" in d1.reason
    assert lead.measures[1].chords[0].raw == "1"
    assert lead.measures[1].chords[0].alternatives == ["5"]
    assert lead.measures[1].chords[0].agreement == "disagree"


def test_full_audio_search_agreement_metrics():
    # 4 bars: 3 agree, 1 disagree
    audio = LeadSheet(
        title="晴天",
        artist="周杰伦",
        key="G",
        measures=[
            _make_measure("1", [1]),
            _make_measure("5/7", [5]),
            _make_measure("6m", [6]),
            _make_measure("4", [4]),
        ],
    )
    search = SearchResult(
        title="晴天",
        artist="周杰伦",
        key="G",
        sources=[
            SourceChart(url="https://source1.com", key="G"),
            SourceChart(url="https://source2.com", key="G"),
        ],
        consensus_sections=[
            ConsensusSection(
                label="主歌",
                bars=[
                    ConsensusBar(chords=["1"], support=2, total=2),
                    ConsensusBar(chords=["5/7"], support=2, total=2),
                    ConsensusBar(chords=["6m"], support=2, total=2),
                    ConsensusBar(chords=["2m"], support=2, total=2),  # Disagrees with 4
                ],
            )
        ],
    )

    with patch("app.smart.crosscheck.llm_available", return_value=False):
        lead, report = merge(audio=audio, search=search)

    assert report.n_agree == 3
    assert report.n_disagree == 1
    assert report.agreement_ratio == 0.75
    assert "一致 75%" in lead.provenance
    assert len(lead.sources) == 2


def test_malformed_and_empty_inputs_never_raise():
    # Both None
    lead, report = merge(None, None)
    assert isinstance(lead, LeadSheet)
    assert isinstance(report, CrossCheckReport)
    assert len(report.warnings) > 0

    # Empty audio and empty search
    audio = LeadSheet(title="", measures=[])
    search = SearchResult(title="", consensus_sections=[])
    lead, report = merge(audio, search)
    assert isinstance(lead, LeadSheet)
    assert isinstance(report, CrossCheckReport)

    # Audio with corrupt chord tokens
    audio_corrupt = LeadSheet(
        title="Corrupt",
        key="InvalidKey#123",
        measures=[
            LeadMeasure(
                beats=4.0,
                chords=[LeadChord(raw="NOT_A_CHORD!!!", beat=1.0)],
            )
        ],
    )
    lead, report = merge(audio_corrupt, None)
    assert isinstance(lead, LeadSheet)
    assert isinstance(report, CrossCheckReport)
