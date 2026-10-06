"""Unit tests for app.audio.to_lead (pure python/numpy, offline, runnable in main .venv)."""
from __future__ import annotations

import pytest

from app.audio.analysis import AudioAnalysis, ChordSeg, LyricChar, NoteSeg, SectionSeg
from app.audio.to_lead import analysis_to_lead
from app.smart.models import LeadSheet


class MockWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class MockSeg:
    def __init__(
        self,
        text: str,
        words: list = None,
        avg_logprob: float = 0.0,
        no_speech_prob: float = 0.0,
        compression_ratio: float = 1.0,
    ):
        self.text = text
        self.words = words or []
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio


def test_steady_tempo_and_basic_melody():
    """Verify standard 4/4 song with steady tempo (120 bpm, 2.0s per bar)."""
    # 2 measures, 120 bpm -> 0.5s per beat, 2.0s per bar
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    downbeats = [0.0, 2.0, 4.0]
    # Key: C major (tonic C4 = 60)
    # Measure 1: C4 (60) dur 0.5s (1 beat), D4 (62) dur 0.5s (1 beat), E4 (64) dur 1.0s (2 beats)
    # Measure 2: F4 (65) dur 1.0s (2 beats), G4 (67) dur 1.0s (2 beats)
    notes = [
        NoteSeg(start=0.0, end=0.5, midi=60.0),
        NoteSeg(start=0.5, end=1.0, midi=62.0),
        NoteSeg(start=1.0, end=2.0, midi=64.0),
        NoteSeg(start=2.0, end=3.0, midi=65.0),
        NoteSeg(start=3.0, end=4.0, midi=67.0),
    ]
    chords = [
        ChordSeg(start=0.0, end=2.0, label="C:maj"),
        ChordSeg(start=2.0, end=3.0, label="F:maj"),
        ChordSeg(start=3.0, end=4.0, label="G:7"),
    ]

    analysis = AudioAnalysis(
        duration=4.0,
        tempo_bpm=120.0,
        beats=beats,
        downbeats=downbeats,
        time_signature="4/4",
        key="C",
        chords=chords,
        notes=notes,
    )

    lead = analysis_to_lead(analysis, title="Test Song", artist="Test Artist")

    assert isinstance(lead, LeadSheet)
    assert lead.key == "C"
    assert lead.time_signature == "4/4"
    assert len(lead.measures) == 2

    # Measure 1 checks
    m1 = lead.measures[0]
    assert m1.beats == 4.0
    assert sum(n.duration for n in m1.notes) == pytest.approx(4.0)
    assert len(m1.notes) == 3
    assert m1.notes[0].degree == 1 and m1.notes[0].octave == 0 and m1.notes[0].duration == 1.0
    assert m1.notes[1].degree == 2 and m1.notes[1].octave == 0 and m1.notes[1].duration == 1.0
    assert m1.notes[2].degree == 3 and m1.notes[2].octave == 0 and m1.notes[2].duration == 2.0
    assert len(m1.chords) == 1
    assert m1.chords[0].raw == "1" and m1.chords[0].beat == 1.0

    # Measure 2 checks
    m2 = lead.measures[1]
    assert m2.beats == 4.0
    assert sum(n.duration for n in m2.notes) == pytest.approx(4.0)
    assert len(m2.notes) == 2
    assert m2.notes[0].degree == 4 and m2.notes[0].duration == 2.0
    assert m2.notes[1].degree == 5 and m2.notes[1].duration == 2.0
    assert len(m2.chords) == 2
    assert m2.chords[0].raw == "4" and m2.chords[0].beat == 1.0
    assert m2.chords[1].raw == "57" and m2.chords[1].beat == 3.0


def test_pickup_measure():
    """Verify handling of pickup beats prior to the first full downbeat."""
    # First downbeat is at 1.0s. Two beats precede it at 0.0s and 0.5s.
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    downbeats = [1.0, 3.0]
    notes = [
        NoteSeg(start=0.0, end=1.0, midi=55.0),  # G3 in pickup
        NoteSeg(start=1.0, end=3.0, midi=60.0),  # C4 in bar 1
    ]

    analysis = AudioAnalysis(
        duration=3.0,
        tempo_bpm=120.0,
        beats=beats,
        downbeats=downbeats,
        time_signature="4/4",
        key="C",
        notes=notes,
    )

    lead = analysis_to_lead(analysis)
    assert len(lead.measures) >= 2

    # Measure 0 should be pickup measure
    pickup = lead.measures[0]
    assert pickup.beats == 2.0  # 2 beats pickup
    assert sum(n.duration for n in pickup.notes) == pytest.approx(pickup.beats)
    assert pickup.notes[0].degree == 5 and pickup.notes[0].octave == -1  # G3 is 5 in octave -1

    # Measure 1 should be full measure
    m1 = lead.measures[1]
    assert m1.beats == 4.0
    assert sum(n.duration for n in m1.notes) == pytest.approx(4.0)


def test_notes_crossing_barlines_and_ties():
    """Verify notes that cross the barline are split with tie_to_next=True."""
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    downbeats = [0.0, 2.0, 4.0]
    # Note starts at beat 3.5 (1.75s) and ends at beat 1.5 of bar 2 (2.75s)
    notes = [
        NoteSeg(start=1.75, end=2.75, midi=64.0),  # E4 across barline 2.0s
    ]

    analysis = AudioAnalysis(
        duration=4.0,
        tempo_bpm=120.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        notes=notes,
    )

    lead = analysis_to_lead(analysis)
    assert len(lead.measures) == 2

    m1 = lead.measures[0]
    m2 = lead.measures[1]

    # In measure 1: rest from 0 to 3.5, then note of dur 0.5 with tie_to_next=True
    assert sum(n.duration for n in m1.notes) == pytest.approx(4.0)
    note_seg1 = [n for n in m1.notes if n.degree > 0][0]
    assert note_seg1.onset == 3.5
    assert note_seg1.duration == 0.5
    assert note_seg1.tie_to_next is True

    # In measure 2: note of dur 1.5 at onset 0.0 with tie_to_next=False, followed by rest
    assert sum(n.duration for n in m2.notes) == pytest.approx(4.0)
    note_seg2 = [n for n in m2.notes if n.degree > 0][0]
    assert note_seg2.onset == 0.0
    assert note_seg2.duration == 1.5
    assert note_seg2.tie_to_next is False
    assert note_seg1.degree == note_seg2.degree == 3


def test_drifting_tempo():
    """Verify that tempo changes/drifts map to the correct local beat offsets."""
    # Bar 1 is fast (1.6s total, 0.4s/beat)
    # Bar 2 is slow (2.4s total, 0.6s/beat)
    downbeats = [0.0, 1.6, 4.0]
    beats = [0.0, 0.4, 0.8, 1.2, 1.6, 2.2, 2.8, 3.4]
    notes = [
        NoteSeg(start=0.4, end=0.8, midi=60.0),  # Beat 1 of bar 1
        NoteSeg(start=2.8, end=3.4, midi=62.0),  # Beat 2 of bar 2
    ]

    analysis = AudioAnalysis(
        duration=4.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        notes=notes,
    )

    lead = analysis_to_lead(analysis)
    assert len(lead.measures) == 2

    m1_note = [n for n in lead.measures[0].notes if n.degree > 0][0]
    assert m1_note.onset == 1.0  # Exactly beat 1.0 despite 0.4s duration

    m2_note = [n for n in lead.measures[1].notes if n.degree > 0][0]
    assert m2_note.onset == 2.0  # Exactly beat 2.0 despite 0.6s duration


def test_syncopation_and_rest_filling():
    """Verify 1/4-beat syncopation and exact rest filling."""
    beats = [0.0, 0.5, 1.0, 1.5]
    downbeats = [0.0, 2.0]
    # Note at onset 0.25 beats (0.125s) with duration 0.5 beats (0.25s)
    notes = [
        NoteSeg(start=0.125, end=0.375, midi=60.0),
    ]

    analysis = AudioAnalysis(
        duration=2.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        notes=notes,
    )

    lead = analysis_to_lead(analysis)
    m = lead.measures[0]
    assert m.beats == 4.0
    assert sum(n.duration for n in m.notes) == pytest.approx(4.0)

    # Expected: rest 0.25, note 0.5, rest 3.25
    assert len(m.notes) == 3
    assert m.notes[0].degree == 0 and m.notes[0].duration == 0.25
    assert m.notes[1].degree == 1 and m.notes[1].duration == 0.5 and m.notes[1].onset == 0.25
    assert m.notes[2].degree == 0 and m.notes[2].duration == 3.25 and m.notes[2].onset == 0.75


def test_accidental_spelling_and_chromatics():
    """Verify chromatic degrees #4, b7, b3, b6 in Bb key."""
    # Key Bb: tonic is Bb (pc 10)
    # #4 is E natural (midi 64, pc 4) -> (4 - 10) % 12 = 6
    # b7 is Ab (midi 68, pc 8) -> (8 - 10) % 12 = 10
    # b3 is Db (midi 61, pc 1) -> (1 - 10) % 12 = 3
    # b6 is Gb (midi 66, pc 6) -> (6 - 10) % 12 = 8
    downbeats = [0.0, 2.0]
    beats = [0.0, 0.5, 1.0, 1.5]
    notes = [
        NoteSeg(start=0.0, end=0.5, midi=64.0),  # #4
        NoteSeg(start=0.5, end=1.0, midi=68.0),  # b7
        NoteSeg(start=1.0, end=1.5, midi=61.0),  # b3
        NoteSeg(start=1.5, end=2.0, midi=66.0),  # b6
    ]

    analysis = AudioAnalysis(
        duration=2.0,
        beats=beats,
        downbeats=downbeats,
        key="Bb",
        notes=notes,
    )

    lead = analysis_to_lead(analysis)
    m = lead.measures[0]
    assert m.notes[0].degree == 4 and m.notes[0].accidental == 1   # #4
    assert m.notes[1].degree == 7 and m.notes[1].accidental == -1  # b7
    assert m.notes[2].degree == 3 and m.notes[2].accidental == -1  # b3
    assert m.notes[3].degree == 6 and m.notes[3].accidental == -1  # b6


def test_chord_snapping_and_collapsing():
    """Verify snapping to beats and collapsing consecutive identical chords."""
    downbeats = [0.0, 2.0]
    beats = [0.0, 0.5, 1.0, 1.5]
    # Chords: C at 0.0, C at 0.5 (beat 2), G at 1.0 (beat 3)
    chords = [
        ChordSeg(start=0.0, end=0.5, label="C:maj"),
        ChordSeg(start=0.5, end=1.0, label="C:maj"),
        ChordSeg(start=1.0, end=2.0, label="G:maj"),
    ]

    analysis = AudioAnalysis(
        duration=2.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        chords=chords,
    )

    lead = analysis_to_lead(analysis)
    m = lead.measures[0]
    # In measure: C:maj at beat 1 and C:maj at beat 2 should collapse, leaving only '1' at beat 1, '5' at beat 3
    assert len(m.chords) == 2
    assert m.chords[0].raw == "1" and m.chords[0].beat == 1.0
    assert m.chords[1].raw == "5" and m.chords[1].beat == 3.0


def test_slash_chord_and_bass_pc():
    """Verify slash chord formatting when bass_pc differs from root."""
    downbeats = [0.0, 2.0]
    beats = [0.0, 0.5, 1.0, 1.5]
    # G:maj (root G, pc 7) with bass B (pc 11) in key C -> '5/7'
    chords = [
        ChordSeg(start=0.0, end=2.0, label="G:maj", bass_pc=11),
    ]

    analysis = AudioAnalysis(
        duration=2.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        chords=chords,
    )

    lead = analysis_to_lead(analysis)
    m = lead.measures[0]
    assert len(m.chords) == 1
    assert m.chords[0].raw == "5/7"


def test_lyrics_alignment():
    """Verify lyrics characters are assigned to the nearest melody notes."""
    downbeats = [0.0, 2.0]
    beats = [0.0, 0.5, 1.0, 1.5]
    notes = [
        NoteSeg(start=0.0, end=0.5, midi=60.0),
        NoteSeg(start=0.5, end=1.0, midi=62.0),
        NoteSeg(start=1.0, end=1.5, midi=64.0),
        NoteSeg(start=1.5, end=2.0, midi=65.0),
    ]
    lyrics = [
        LyricChar(start=0.0, end=0.4, text="听"),
        LyricChar(start=0.5, end=0.9, text="海"),
        LyricChar(start=1.0, end=1.4, text="哭"),
    ]

    analysis = AudioAnalysis(
        duration=2.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        notes=notes,
        lyrics=lyrics,
    )

    lead = analysis_to_lead(analysis)
    m = lead.measures[0]
    assert m.notes[0].lyric == "听"
    assert m.notes[1].lyric == "海"
    assert m.notes[2].lyric == "哭"
    assert m.notes[3].lyric == ""  # Melisma / no lyric


def test_sections_assignment():
    """Verify section labels are assigned to the correct measures."""
    downbeats = [0.0, 2.0, 4.0, 6.0]
    beats = [i * 0.5 for i in range(12)]
    sections = [
        SectionSeg(start=0.0, end=2.0, label="前奏"),
        SectionSeg(start=2.0, end=6.0, label="主歌"),
    ]

    analysis = AudioAnalysis(
        duration=6.0,
        beats=beats,
        downbeats=downbeats,
        key="C",
        sections=sections,
    )

    lead = analysis_to_lead(analysis)
    assert lead.measures[0].section == "前奏"
    assert lead.measures[1].section == "主歌"
    assert lead.measures[2].section is None


def test_never_raises_on_invalid_or_empty_input():
    """Verify the function handles degenerate cases without crashing."""
    # Completely empty
    a_empty = AudioAnalysis(duration=0.0)
    lead_empty = analysis_to_lead(a_empty)
    assert isinstance(lead_empty, LeadSheet)
    assert len(lead_empty.measures) >= 1
    assert "音频转录" in lead_empty.provenance

    # Corrupt / unusual values
    a_bad = AudioAnalysis(
        duration=0.1,
        key="InvalidKeyXYZ",
        time_signature="invalid",
        tempo_bpm=10.0,
        notes=[NoteSeg(start=0.0, end=0.05, midi=-100.0)],
        chords=[ChordSeg(start=0.0, end=0.05, label="Corrupt!!")],
    )
    lead_bad = analysis_to_lead(a_bad)
    assert isinstance(lead_bad, LeadSheet)
    assert len(lead_bad.measures) >= 1

    # Completely arbitrary non-AudioAnalysis or object causing internal failure
    class MockBad:
        key = None
        time_signature = None
        duration = None
        tempo_bpm = None
        notes = None
        chords = None
        beats = None
        downbeats = None
        lyrics = None
        sections = None
        warnings = []

    lead_mock = analysis_to_lead(MockBad())  # type: ignore
    assert isinstance(lead_mock, LeadSheet)
    assert len(lead_mock.measures) >= 1
    assert "转换异常" in lead_mock.provenance


def test_key_detection_f_sharp_major():
    """Verify key detection on 1-5-6m-4 chord sequence in F# major."""
    from app.audio.transcribe import _detect_key_krumhansl

    chords = [
        ChordSeg(start=0.0, end=2.0, label="F#:maj"),
        ChordSeg(start=2.0, end=4.0, label="C#:maj"),
        ChordSeg(start=4.0, end=6.0, label="D#:min"),
        ChordSeg(start=6.0, end=8.0, label="B:maj"),
    ]
    warnings: list[str] = []
    key = _detect_key_krumhansl(chords, [], warnings)
    assert key == "F#"


def test_lyrics_hallucination_filter():
    """Verify Whisper hallucination patterns and out-of-melody words are rejected."""
    from app.audio.transcribe import filter_lyrics_segments

    # Melody note from 1.0s to 3.0s
    notes = [NoteSeg(start=1.0, end=3.0, midi=60.0)]

    # Various hallucinations: repeated n-grams, blacklist, low confidence, high no-speech
    hallucinated_segs = [
        MockSeg("ZitherHarpZitherHarp…字幕by索兰娅", [
            MockWord("ZitherHarp", 1.0, 1.5),
            MockWord("字幕by索兰娅", 1.5, 2.0),
        ]),
        MockSeg("谢谢观看，请关注订阅点赞！", [MockWord("谢谢观看", 1.0, 2.0)]),
        MockSeg("啦啦啦啦啦啦啦啦啦啦", avg_logprob=-1.5),
        MockSeg("低置信度字符", avg_logprob=-1.2),
        MockSeg("高静音概率", no_speech_prob=0.8),
        # Note overlap check: word at 10.0s..11.0s does not overlap any note in `notes`
        MockSeg("正常文字但无人声", [MockWord("无人声", 10.0, 11.0)]),
    ]

    filtered = filter_lyrics_segments(hallucinated_segs, notes, language="zh")
    assert len(filtered) == 0

    # Legitimate singing that overlaps the melody note
    legit_words = [MockWord("听", 1.0, 1.5), MockWord("海", 1.5, 2.0)]
    legit_segs = [MockSeg("听海", words=legit_words, avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1)]
    res_legit = filter_lyrics_segments(legit_segs, notes, language="zh")
    assert [l.text for l in res_legit] == ["听", "海"]


def test_section_labeller():
    """Verify section labeller identifies intro, verse, chorus, interlude, bridge, outro."""
    from app.audio.transcribe import _detect_sections

    # 77 bars, 3.0s each
    downbeats = [i * 3.0 for i in range(77)]
    duration = 77 * 3.0
    notes = []
    # Intro: 0..7 (bars 0-7: no notes)
    # Verse: 8..23 (bars 8-23: midi 58.0)
    for b in range(8, 24):
        notes.append(NoteSeg(start=b * 3.0 + 0.5, end=b * 3.0 + 2.5, midi=58.0))
    # Chorus: 24..39 (bars 24-39: midi 70.0)
    for b in range(24, 40):
        notes.append(NoteSeg(start=b * 3.0 + 0.5, end=b * 3.0 + 2.5, midi=70.0))
    # Interlude: 40..41 (bars 40-41: no notes)
    # Bridge: 42..51 (bars 42-51: midi 62.0)
    for b in range(42, 52):
        notes.append(NoteSeg(start=b * 3.0 + 0.5, end=b * 3.0 + 2.5, midi=62.0))
    # Chorus: 52..67 (bars 52-67: midi 70.0)
    for b in range(52, 68):
        notes.append(NoteSeg(start=b * 3.0 + 0.5, end=b * 3.0 + 2.5, midi=70.0))
    # Outro: 68..76 (bars 68-76: no notes)

    sections = _detect_sections(downbeats, duration, notes, [])
    labels = [s.label for s in sections]
    # The lower-register section after the interlude is labelled 主歌: a verse after an interlude
    # is usually a second verse, so 桥段 is never inferred from audio alone (the web cross-check
    # may supply it).
    assert labels == ["前奏", "主歌", "副歌", "间奏", "主歌", "副歌", "尾奏"]

    # Verify boundary times
    assert sections[0].start == 0.0 and sections[0].end == 24.0
    assert sections[1].start == 24.0 and sections[1].end == 72.0
    assert sections[2].start == 72.0 and sections[2].end == 120.0
    assert sections[3].start == 120.0 and sections[3].end == 126.0
    assert sections[4].start == 126.0 and sections[4].end == 156.0
    assert sections[5].start == 156.0 and sections[5].end == 204.0
    assert sections[6].start == 204.0 and sections[6].end == 231.0


def test_section_labeller_single_high_bar_does_not_split_verse():
    """One high bar inside a low verse phrase must not become its own 副歌 section."""
    from app.audio.transcribe import _detect_sections

    downbeats = [i * 3.0 for i in range(20)]
    notes = []
    for b in range(2, 18):
        midi = 70.0 if b >= 10 else (69.0 if b == 5 else 58.0)
        notes.append(NoteSeg(start=b * 3.0 + 0.5, end=b * 3.0 + 2.5, midi=midi))
    labels = [s.label for s in _detect_sections(downbeats, 60.0, notes, [])]
    assert labels == ["前奏", "主歌", "副歌", "尾奏"]


