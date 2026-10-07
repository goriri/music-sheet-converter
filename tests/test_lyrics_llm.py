"""Unit tests for Gemini-based lyric correction and alignment (offline, mocked)."""
from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from app.audio.analysis import LyricChar, NoteSeg
from app.audio.lyrics_llm import (
    align_line_chars,
    clean_lyrics_text,
    contains_generic_hallucination,
    correct_lyrics,
    has_repeated_pattern,
    is_char_count_implausible,
    levenshtein_distance,
    salvage_lyrics_json,
    to_simplified_chinese,
)


# ---------------------------------------------------------------------------
# 1. Text Normalization and Edit Distance Tests
# ---------------------------------------------------------------------------
def test_to_simplified_chinese():
    traditional = "這就是夢想從今後看著你好"
    simplified = to_simplified_chinese(traditional)
    assert simplified == "这就是梦想从今后看着你好"


def test_clean_lyrics_text():
    raw = " 这就，不是 梦？！ “Hello” （world）… "
    cleaned = clean_lyrics_text(raw)
    assert cleaned == "这就不是梦Helloworld"


def test_levenshtein_distance():
    assert levenshtein_distance("今天", "今天") == 0
    assert levenshtein_distance("今天", "明天") == 1
    assert levenshtein_distance("今天天气好", "今天好") == 2
    assert levenshtein_distance("", "abc") == 3


# ---------------------------------------------------------------------------
# 2. Timing Alignment Tests (Substitutions, Insertions, Deletions, Monotonicity)
# ---------------------------------------------------------------------------
def test_align_line_chars_substitutions_keep_timestamps():
    draft = [
        LyricChar(start=0.0, end=0.68, text="这"),
        LyricChar(start=0.68, end=1.07, text="就"),
        LyricChar(start=1.07, end=1.46, text="是"),
        LyricChar(start=1.46, end=3.08, text="宝"),
    ]
    target_text = "这就是梦"
    aligned = align_line_chars(draft, target_text, 0.0, 3.08)

    assert len(aligned) == 4
    assert "".join(c.text for c in aligned) == "这就是梦"
    # "梦" substituted for "宝" should preserve "宝"'s timestamps
    assert aligned[3].text == "梦"
    assert aligned[3].start == 1.46
    assert aligned[3].end == 3.08


def test_align_line_chars_insertions_interpolated():
    draft = [
        LyricChar(start=0.0, end=0.5, text="这"),
        LyricChar(start=1.0, end=1.5, text="是"),
    ]
    target_text = "这不是"
    notes = [
        NoteSeg(start=0.0, end=0.5, midi=60.0),
        NoteSeg(start=0.7, end=0.9, midi=62.0),
        NoteSeg(start=1.0, end=1.5, midi=64.0),
    ]
    aligned = align_line_chars(draft, target_text, 0.0, 1.5, notes=notes)

    assert len(aligned) == 3
    assert "".join(c.text for c in aligned) == "这不是"
    # "不" is inserted between "这" (end=0.5) and "是" (start=1.0)
    assert aligned[1].text == "不"
    assert aligned[1].start >= 0.5
    assert aligned[1].end <= 1.0
    # Note at 0.7 should guide onset
    assert aligned[1].start == 0.7


def test_align_line_chars_deletions_dropped():
    draft = [
        LyricChar(start=0.0, end=0.5, text="今"),
        LyricChar(start=0.5, end=1.0, text="天"),
        LyricChar(start=1.0, end=1.5, text="很"),
        LyricChar(start=1.5, end=2.0, text="好"),
    ]
    target_text = "今天好"
    aligned = align_line_chars(draft, target_text, 0.0, 2.0)

    assert len(aligned) == 3
    assert "".join(c.text for c in aligned) == "今天好"
    # "很" was dropped, "今", "天", "好" keep their timestamps
    assert aligned[0].text == "今" and aligned[0].start == 0.0
    assert aligned[1].text == "天" and aligned[1].start == 0.5
    assert aligned[2].text == "好" and aligned[2].start == 1.5


def test_align_line_chars_empty_draft_interpolates():
    draft: list[LyricChar] = []
    target_text = "天黑的时候"
    aligned = align_line_chars(draft, target_text, 1.0, 3.5)

    assert len(aligned) == 5
    assert "".join(c.text for c in aligned) == "天黑的时候"
    for i in range(len(aligned) - 1):
        assert aligned[i].start < aligned[i].end
        assert aligned[i + 1].start >= aligned[i].start


def test_align_line_chars_monotonicity_guarantee():
    # Stress test overlapping / adjacent boundaries
    draft = [
        LyricChar(start=0.0, end=1.0, text="一"),
        LyricChar(start=1.0, end=1.0, text="二"),
        LyricChar(start=1.0, end=2.0, text="三"),
    ]
    target_text = "一二三四五"
    aligned = align_line_chars(draft, target_text, 0.0, 3.0)

    assert len(aligned) == 5
    for i in range(len(aligned)):
        assert aligned[i].start < aligned[i].end
        if i > 0:
            assert aligned[i].start >= aligned[i - 1].start


# ---------------------------------------------------------------------------
# 3. Tolerant JSON Salvage Tests
# ---------------------------------------------------------------------------
def test_salvage_lyrics_json_markdown():
    text = '```json\n{"lines": [{"start": 0.5, "end": 2.0, "text": "你好"}]}\n```'
    lines = salvage_lyrics_json(text)
    assert len(lines) == 1
    assert lines[0]["text"] == "你好"
    assert lines[0]["start"] == 0.5


def test_salvage_lyrics_json_list():
    text = '[{"start": 1.0, "end": 2.5, "text": "世界"}]'
    lines = salvage_lyrics_json(text)
    assert len(lines) == 1
    assert lines[0]["text"] == "世界"


def test_salvage_lyrics_json_truncated():
    # Broken JSON missing closing brace
    text = '{"lines": [{"start": 0.0, "end": 3.0, "text": "测试截断"'
    lines = salvage_lyrics_json(text)
    # Regex salvage should recover the object if fields are present
    text_salvageable = '{"lines": [{"start": 0.0, "end": 3.0, "text": "测试截断"}'
    lines = salvage_lyrics_json(text_salvageable)
    assert len(lines) == 1
    assert lines[0]["text"] == "测试截断"


# ---------------------------------------------------------------------------
# 4. Fallbacks (RECITATION, SAFETY, Timeout, Garbage, Never-Raise)
# ---------------------------------------------------------------------------
def test_correct_lyrics_non_zh_gate(tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy audio")
    draft = [LyricChar(start=0.0, end=1.0, text="A")]

    chars, warnings = correct_lyrics(str(fake_audio), draft, [], language="en")
    assert chars == draft
    assert warnings == []


def test_correct_lyrics_missing_audio_file():
    draft = [LyricChar(start=0.0, end=1.0, text="A")]
    chars, warnings = correct_lyrics("/path/to/nonexistent/audio.wav", draft, [])
    assert chars == draft
    assert any("not found" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_correct_lyrics_recitation_fallback(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "RECITATION"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    mock_client.models.generate_content.return_value = mock_resp

    draft = [LyricChar(start=0.0, end=1.0, text="我")]
    chars, warnings = correct_lyrics(str(fake_audio), draft, [])

    assert chars == draft
    assert any("recitation" in w.lower() or "filter" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_correct_lyrics_safety_fallback(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "SAFETY"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    mock_client.models.generate_content.return_value = mock_resp

    draft = [LyricChar(start=0.0, end=1.0, text="我")]
    chars, warnings = correct_lyrics(str(fake_audio), draft, [])

    assert chars == draft
    assert any("safety" in w.lower() or "filter" in w.lower() for w in warnings)


def _slow_call_mock(*args, **kwargs):
    raise concurrent.futures.TimeoutError("Call timed out")


@patch("google.genai.Client")
def test_correct_lyrics_timeout_fallback(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.side_effect = _slow_call_mock

    draft = [LyricChar(start=0.0, end=1.0, text="我")]
    chars, warnings = correct_lyrics(str(fake_audio), draft, [], timeout_s=0.1)

    assert chars == draft
    assert any("timed out" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_correct_lyrics_garbage_fallback(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    mock_resp.text = "NOT JSON AT ALL AND NO VALID FIELDS"
    mock_client.models.generate_content.return_value = mock_resp

    draft = [LyricChar(start=0.0, end=1.0, text="我")]
    chars, warnings = correct_lyrics(str(fake_audio), draft, [])

    assert chars == draft
    assert any("parse" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# 5. Hallucination Guardrail Tests
# ---------------------------------------------------------------------------
@patch("google.genai.Client")
def test_hallucination_guard_no_vocal_activity(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    # Gemini hallucinates lyrics at [10.0, 15.0] where there are no notes and no draft
    mock_resp.text = '{"lines": [{"start": 10.0, "end": 15.0, "text": "突然出现一句幻觉歌词"}]}'
    mock_client.models.generate_content.return_value = mock_resp

    draft = [LyricChar(start=0.0, end=1.0, text="我")]
    notes = [NoteSeg(start=0.0, end=1.0, midi=60.0)]

    chars, warnings = correct_lyrics(str(fake_audio), draft, notes)
    # The hallucinated line should be dropped; all candidate lines filtered out -> fallback to draft
    assert chars == draft
    assert any("silent" in w.lower() or "filtered" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_hallucination_guard_low_similarity_vs_confident_draft(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    # Gemini hallucinates completely unrelated text for the line
    mock_resp.text = '{"lines": [{"start": 0.0, "end": 3.0, "text": "床前明月光疑是地上霜"}]}'
    mock_client.models.generate_content.return_value = mock_resp

    # Confident draft with 8 characters
    draft_chars = "今天天气真好真高兴"
    draft = [
        LyricChar(start=round(i * 0.3, 2), end=round((i + 1) * 0.3, 2), text=ch)
        for i, ch in enumerate(draft_chars)
    ]
    notes = [
        NoteSeg(start=round(i * 0.3, 2), end=round((i + 1) * 0.3, 2), midi=60.0)
        for i in range(len(draft_chars))
    ]

    chars, warnings = correct_lyrics(str(fake_audio), draft, notes)
    # Candidate line should be rejected due to low similarity (< 0.15), preserving draft
    assert "".join(c.text for c in chars) == draft_chars
    assert any("rejected" in w.lower() or "similarity" in w.lower() for w in warnings)


def test_correct_lyrics_never_raise():
    # Pass garbage types into correct_lyrics to ensure it never crashes
    chars, warnings = correct_lyrics(None, None, None)  # type: ignore
    assert chars is None or isinstance(chars, list)
    assert isinstance(warnings, list)


def test_has_repeated_pattern_cases():
    assert has_repeated_pattern("南无阿弥陀佛南无阿弥陀佛") is True
    assert has_repeated_pattern("啦啦啦") is True
    assert has_repeated_pattern("哈利路亚哈利路亚") is True
    assert has_repeated_pattern("这就不是梦") is False
    assert has_repeated_pattern("休息的时候才猛然的发现梦想根本不在这里") is False


def test_is_char_count_implausible_cases():
    # >= 6 notes: chars > 1.5 * notes + 2 or < 0.3 * notes
    assert is_char_count_implausible(25, 6) is True
    assert is_char_count_implausible(1, 10) is True
    assert is_char_count_implausible(8, 8) is False
    assert is_char_count_implausible(12, 10) is False


def test_contains_generic_hallucination_cases():
    assert contains_generic_hallucination("本视频由字幕组听写制作") is True
    assert contains_generic_hallucination("谢谢观看本期视频") is True
    assert contains_generic_hallucination("南无阿弥陀佛") is True
    assert contains_generic_hallucination("https://bilibili.com") is True
    assert contains_generic_hallucination("幸福和快乐是结局") is False


@patch("google.genai.Client")
def test_hallucination_guard_repeated_pattern_rejected(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    # Simulate the exact failure mode: empty draft, notes exist, but Gemini outputs repeated chanting
    mock_resp.text = '{"lines": [{"start": 0.0, "end": 4.0, "text": "南无阿弥陀佛南无阿弥陀佛", "confidence": 0.95}]}'
    mock_client.models.generate_content.return_value = mock_resp

    draft: list[LyricChar] = []
    notes = [NoteSeg(start=round(i * 0.4, 2), end=round((i + 1) * 0.4, 2), midi=60.0) for i in range(10)]

    chars, warnings = correct_lyrics(str(fake_audio), draft, notes)
    # The repeated pattern chanting must be dropped!
    assert chars == []
    assert any("repetitive" in w.lower() or "hallucination" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_hallucination_guard_draft_less_low_confidence(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    # Unconfident hallucination when draft is empty
    mock_resp.text = '{"lines": [{"start": 0.0, "end": 3.0, "text": "某句模糊歌词", "confidence": 0.42}]}'
    mock_client.models.generate_content.return_value = mock_resp

    draft: list[LyricChar] = []
    notes = [NoteSeg(start=0.0, end=3.0, midi=60.0)]

    chars, warnings = correct_lyrics(str(fake_audio), draft, notes)
    assert chars == []
    assert any("low-confidence" in w.lower() or "confidence" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_hallucination_guard_generic_subtitle_pattern(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    mock_resp.text = '{"lines": [{"start": 0.0, "end": 3.0, "text": "谢谢观看", "confidence": 0.9}]}'
    mock_client.models.generate_content.return_value = mock_resp

    draft: list[LyricChar] = []
    notes = [NoteSeg(start=0.0, end=3.0, midi=60.0)]

    chars, warnings = correct_lyrics(str(fake_audio), draft, notes)
    assert chars == []
    assert any("hallucination" in w.lower() or "dropped" in w.lower() for w in warnings)


@patch("google.genai.Client")
def test_hallucination_guard_duplicate_lines(mock_client_cls, tmp_path):
    fake_audio = tmp_path / "vocal.flac"
    fake_audio.write_bytes(b"dummy")

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_cand = MagicMock()
    mock_cand.finish_reason = "STOP"
    mock_resp = MagicMock()
    mock_resp.candidates = [mock_cand]
    # Identical repeated lines
    mock_resp.text = '{"lines": [{"start": 0.0, "end": 2.0, "text": "天黑的时候", "confidence": 0.9}, {"start": 2.2, "end": 4.0, "text": "天黑的时候", "confidence": 0.9}]}'
    mock_client.models.generate_content.return_value = mock_resp

    draft: list[LyricChar] = []
    notes = [NoteSeg(start=round(i * 0.4, 2), end=round((i + 1) * 0.4, 2), midi=60.0) for i in range(10)]

    chars, warnings = correct_lyrics(str(fake_audio), draft, notes)
    # First line should be accepted, second duplicate dropped
    assert "".join(c.text for c in chars) == "天黑的时候"
    assert any("duplicate" in w.lower() for w in warnings)

