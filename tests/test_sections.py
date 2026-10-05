"""Unit tests for song-form analysis and section planner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from app.models import (
    ChordSymbol,
    KeyChange,
    Measure,
    PageInfo,
    ParsedSheet,
    SectionPlan,
    SongHeader,
    System,
)
from app.arrange.sections import parse_label, plan_sections

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_minimal_sheet(num_measures: int = 16) -> ParsedSheet:
    measures = [
        Measure(
            index=i,
            bbox=(0.0, 0.0, 1.0, 1.0),
            beats=4.0,
            chords=[ChordSymbol(raw="1", beat=1.0)],
            lyrics="测试歌词" if 4 <= i < 12 else "",
        )
        for i in range(num_measures)
    ]
    systems = [
        System(page=0, bbox=(0, 0, 1, 1), section_label="前奏" if i == 0 else ("主歌" if i == 1 else "副歌"), measures=measures[i * 4 : (i + 1) * 4])
        for i in range(num_measures // 4)
    ]
    return ParsedSheet(
        header=SongHeader(title="测试歌曲", original_key="C", time_signature="4/4"),
        pages=[PageInfo(width=1000, height=1400)],
        systems=systems,
        key_changes=[],
    )


class TestLabelParsing:
    """Test parse_label for Mainland labels and Taiwanese instrumentation cues."""

    @pytest.mark.parametrize(
        "label,expected_role,expected_energy,is_auth",
        [
            ("前奏", "intro", 0, True),
            ("主歌1", "verse", 1, True),
            ("Verse 2", "verse", 1, True),
            ("导歌", "prechorus", 2, True),
            ("Pre-chorus", "prechorus", 2, True),
            ("副歌", "chorus", 2, True),
            ("Chorus", "chorus", 2, True),
            ("间奏", "interlude", 1, True),
            ("尾奏", "outro", 0, True),
            ("Ending", "outro", 0, True),
            ("桥段", "prechorus", 2, True),
        ],
    )
    def test_mainland_labels(self, label, expected_role, expected_energy, is_auth):
        role, energy, auth = parse_label(label)
        assert role == expected_role
        assert energy == expected_energy
        assert auth is is_auth

    @pytest.mark.parametrize(
        "label,expected_energy",
        [
            ("Only PN(RHY) / / / ~", 0),
            ("Only (鐵琴)(arp)", 0),
            ("Only PN(arp)", 0),
            ("Only PN", 0),
            ("Tempo(鼓邊2.4拍) Strings in", 2),
            ("(小鼓2.4拍)OG in", 2),
            ("Only PN+AG(RHY)+Bs in", 2),
            ("+AG", None),  # Neither sparse nor dense on its own
        ],
    )
    def test_taiwanese_instrumentation_cues(self, label, expected_energy):
        role, energy, auth = parse_label(label)
        assert auth is False
        if expected_energy is not None:
            assert energy == expected_energy

    def test_empty_or_none_label(self):
        assert parse_label(None) == (None, None, False)
        assert parse_label("") == (None, None, False)
        assert parse_label("   ") == (None, None, False)


class TestSectionPlanningHeuristics:
    """Offline heuristic tests for song-form section planner."""

    def test_intro_verse_chorus_detection(self):
        sheet = make_minimal_sheet(16)
        plans = plan_sections(sheet, use_llm=False)
        assert len(plans) == 16
        # System 0: 前奏 (m0..3)
        for i in range(4):
            assert plans[i].role == "intro"
            assert plans[i].section_id == "intro"
            assert plans[i].energy == 0
        assert plans[0].is_section_start is True
        assert plans[3].is_section_end is True

        # System 1: 主歌 (m4..7)
        for i in range(4, 8):
            assert plans[i].role == "verse"
            assert plans[i].section_id == "A1"
            assert plans[i].energy == 1
        assert plans[4].is_section_start is True
        assert plans[7].is_section_end is True

        # System 2: 副歌 (m8..11)
        for i in range(8, 12):
            assert plans[i].role == "chorus"
            assert plans[i].section_id == "B1"
            assert plans[i].energy == 2
        assert plans[8].is_section_start is True
        assert plans[11].is_section_end is True

    def test_empty_sheet_returns_empty_and_never_raises(self):
        empty_sheet = ParsedSheet(
            header=SongHeader(title="空谱", original_key="C", time_signature="4/4"),
            pages=[],
            systems=[],
        )
        plans = plan_sections(empty_sheet, use_llm=False)
        assert plans == []

    def test_single_system_sheet(self):
        measures = [Measure(index=i, bbox=(0, 0, 1, 1), beats=4.0, chords=[]) for i in range(4)]
        sys = System(page=0, bbox=(0, 0, 1, 1), section_label=None, measures=measures)
        sheet = ParsedSheet(
            header=SongHeader(title="单行", original_key="C", time_signature="4/4"),
            pages=[PageInfo(width=1000, height=1400)],
            systems=[sys],
        )
        plans = plan_sections(sheet, use_llm=False)
        assert len(plans) == 4
        assert all(p.role == "verse" for p in plans)
        assert plans[0].is_section_start is True
        assert plans[-1].is_section_end is True

    def test_key_change_boosts_chorus_energy(self):
        # Build 12 measures: m0..3 intro, m4..7 verse, m8..11 chorus with key change at m8
        measures = [Measure(index=i, bbox=(0, 0, 1, 1), beats=4.0, chords=[ChordSymbol(raw="1", beat=1.0)]) for i in range(12)]
        systems = [
            System(page=0, bbox=(0, 0, 1, 1), section_label="前奏", measures=measures[0:4]),
            System(page=0, bbox=(0, 0, 1, 1), section_label="主歌", measures=measures[4:8]),
            System(page=0, bbox=(0, 0, 1, 1), section_label="(小鼓2.4拍)OG in", measures=measures[8:12]),
        ]
        sheet = ParsedSheet(
            header=SongHeader(title="转调", original_key="C", time_signature="4/4"),
            pages=[PageInfo(width=1000, height=1400)],
            systems=systems,
            key_changes=[KeyChange(at_measure=8, raw="转D调", semitones=2)],
        )
        plans = plan_sections(sheet, use_llm=False)
        assert len(plans) == 12
        # Chorus starting at m8 after key change should have energy 3 (climax)
        assert plans[8].role == "chorus"
        assert plans[8].energy == 3


class TestSectionsTruthAccuracy:
    """Test section planner on the 6 groundtruth songs from fixtures/sections_truth.json."""

    @pytest.fixture
    def truth_data(self):
        truth_file = PROJECT_ROOT / "fixtures" / "sections_truth.json"
        with open(truth_file, encoding="utf-8") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "song_name,path_rel",
        [
            ("diaole", "out/gt_eval/diaole.json"),
            ("xiaobaichuan", "fixtures/sheets/xiaobaichuan.json"),
            ("qianlizhiwai", "fixtures/sheets/qianlizhiwai.json"),
            ("tinghai", "fixtures/sheets/tinghai.json"),
            ("diandao", "fixtures/sheets/diandao.json"),
            ("yujian", "fixtures/sheets/yujian.json"),
        ],
    )
    def test_truth_match_per_song(self, song_name, path_rel, truth_data):
        file_path = PROJECT_ROOT / path_rel
        assert file_path.exists(), f"Missing fixture file {file_path}"
        with open(file_path, encoding="utf-8") as f:
            sheet = ParsedSheet.model_validate(json.load(f))

        plans = plan_sections(sheet, use_llm=False)
        assert len(plans) == len(sheet.measures())

        truth_sections = truth_data[song_name]
        max_eval_m = truth_sections[-1]["end_measure"]

        truth_m_roles = {}
        for sec in truth_sections:
            for m in range(sec["start_measure"], sec["end_measure"] + 1):
                truth_m_roles[m] = sec["role"]

        eval_plans = [p for p in plans if p.measure_index <= max_eval_m]
        matches = sum(1 for p in eval_plans if p.role == truth_m_roles.get(p.measure_index))
        role_acc = matches / max(1, len(eval_plans))

        min_acc = 0.65 if song_name == "diaole" else (0.80 if song_name == "diandao" else 0.95)
        assert role_acc >= min_acc, f"[{song_name}] Role accuracy {role_acc:.1%} below threshold {min_acc:.1%}"

        # 2. Extract predicted sections and evaluate boundary F1
        pred_secs = []
        curr = None
        for p in plans:
            if curr is None or curr["section_id"] != p.section_id:
                if curr is not None:
                    pred_secs.append(curr)
                curr = {
                    "section_id": p.section_id,
                    "role": p.role,
                    "start_measure": p.measure_index,
                    "end_measure": p.measure_index,
                    "energy": p.energy,
                }
            else:
                curr["end_measure"] = p.measure_index
        if curr is not None:
            pred_secs.append(curr)

        eval_preds = [pr for pr in pred_secs if pr["start_measure"] <= max_eval_m]

        truth_starts = {sec["start_measure"] for sec in truth_sections}
        pred_starts = {pr["start_measure"] for pr in eval_preds}
        tp = sum(1 for ts in truth_starts if any(abs(ts - ps) <= 1 for ps in pred_starts))
        prec = sum(1 for ps in pred_starts if any(abs(ts - ps) <= 1 for ts in truth_starts)) / max(1, len(pred_starts))
        rec = tp / max(1, len(truth_starts))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        min_f1 = 0.45 if song_name == "diaole" else 0.70
        assert f1 >= min_f1, f"[{song_name}] Boundary F1 {f1:.1%} below threshold {min_f1:.1%}"


class TestAllSheetsNonRaising:
    """Sweep all fixture sheets and verify plan_sections never raises."""

    def test_sweep_all_fixture_sheets(self):
        sheets_dir = PROJECT_ROOT / "fixtures" / "sheets"
        for p in sorted(sheets_dir.glob("*.json")):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            sheet = ParsedSheet.model_validate(data)
            plans = plan_sections(sheet, use_llm=False)
            assert len(plans) == len(sheet.measures())
            for i, p_item in enumerate(plans):
                assert p_item.measure_index == sheet.measures()[i].index
                assert p_item.energy in (0, 1, 2, 3)
                assert p_item.role in ("intro", "verse", "prechorus", "chorus", "bridge", "interlude", "outro")
