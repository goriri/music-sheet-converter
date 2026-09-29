"""Tests for context-aware inference and escalation ladder (app/qa/infer.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models import ChordSymbol, Measure, ParsedSheet, QualityIssue, SongHeader, System
from app.qa.infer import (
    AmbiguousItem,
    ItemInference,
    SystemInferenceResponse,
    evaluate_ladder_decision,
    extract_root_degree,
    get_web_reference_candidates,
    run_escalation_ladder,
    verify_vertical_barline_stroke,
)


class DummyRef:
    def __init__(self, raw: str):
        self.raw = raw


def _mock_ask_json_infer(prompt, schema, images=None, role="arbiter", timeout_s=60.0):
    if schema == SystemInferenceResponse:
        return SystemInferenceResponse(
            inferences=[
                ItemInference(
                    item_id="iss_0_m1",
                    target_type="chord",
                    measure_index=1,
                    proposed_value="5",
                    confidence=0.95,
                    musical_justification="配合旋律主音 5",
                )
            ]
        )
    raise ValueError(f"Unexpected schema: {schema}")


def test_extract_root_degree():
    assert extract_root_degree("5/7") == "5"
    assert extract_root_degree("2m7/5") == "2"
    assert extract_root_degree("1(2)") == "1"
    assert extract_root_degree("4M7") == "4"
    assert extract_root_degree("/5") == "5"
    assert extract_root_degree("G/B") == "G"
    assert extract_root_degree("C#m7") == "C#"
    assert extract_root_degree("Bb") == "Bb"
    assert extract_root_degree("") is None


def test_ladder_decision_high_confidence_accepted():
    item = AmbiguousItem(
        item_id="m12_chord",
        target_type="chord",
        measure_index=11,
        system_page=0,
        system_index=1,
        system_bbox=(0.05, 0.20, 0.95, 0.35),
        original_value="5/1",
        candidate_set=["5/7", "5", "5/1"],
        crop_reading="5/7",
        crop_valid=True,
        prior_scores={"5/7": 0.85, "5": 0.75, "5/1": 0.20},
    )
    inf = ItemInference(
        item_id="m12_chord",
        target_type="chord",
        measure_index=11,
        proposed_value="5/7",
        confidence=0.92,
        musical_justification="配合旋律主音 5 构成标准正规属七挂四导和弦进行",
    )
    dec = evaluate_ladder_decision(item, inf, web_candidates=[])
    assert dec.accepted is True
    assert dec.severity == "auto_fixed"
    assert dec.new_value == "5/7"
    assert "inference" in dec.evidence
    assert "priors" in dec.evidence
    assert "crop" in dec.evidence
    assert "第12小节和弦依据旋律推断为 5/7（原读 5/1）" in dec.message


def test_ladder_decision_crop_contradiction_rejected():
    item = AmbiguousItem(
        item_id="m5_chord",
        target_type="chord",
        measure_index=4,
        system_page=0,
        system_index=0,
        system_bbox=(0.05, 0.05, 0.95, 0.20),
        original_value="2m7",
        candidate_set=["4", "2m7"],
        crop_reading="2m7",
        crop_valid=True,
        prior_scores={"4": 0.80, "2m7": 0.70},
    )
    # Model mistakenly infers 4 when crop clearly read 2m7
    inf = ItemInference(
        item_id="m5_chord",
        target_type="chord",
        measure_index=4,
        proposed_value="4",
        confidence=0.88,
        musical_justification="旋律与4更贴合",
    )
    dec = evaluate_ladder_decision(item, inf, web_candidates=[])
    assert dec.accepted is False
    assert dec.severity == "needs_review"
    assert dec.detail["reason"] == "contradicted_by_crop"


def test_ladder_decision_web_evidence_root_agreement():
    item = AmbiguousItem(
        item_id="m8_chord",
        target_type="missing_chord",
        measure_index=7,
        system_page=0,
        system_index=1,
        system_bbox=(0.05, 0.20, 0.95, 0.35),
        original_value="",
        candidate_set=["5/7", "5"],
        crop_reading=None,
        crop_valid=False,
        prior_scores={"5/7": 0.65, "5": 0.60},
    )
    inf = ItemInference(
        item_id="m8_chord",
        target_type="missing_chord",
        measure_index=7,
        proposed_value="5/7",
        confidence=0.75,
        musical_justification="低音导音上行",
    )
    web_cands = [DummyRef("5")]
    dec = evaluate_ladder_decision(item, inf, web_candidates=web_cands)
    assert dec.accepted is True
    assert dec.severity == "auto_fixed"
    assert "web" in dec.evidence
    assert "inference" in dec.evidence


def test_verify_vertical_barline_stroke():
    # Synthetic image with a black vertical line at norm_x = 0.5 (x = 500)
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    cv2.line(img, (500, 100), (500, 400), (0, 0, 0), thickness=3)

    sys_bbox = (0.05, 0.10, 0.95, 0.40)
    assert verify_vertical_barline_stroke(img, norm_x=0.50, sys_bbox=sys_bbox) is True
    # At norm_x = 0.20 (no stroke), should return False
    assert verify_vertical_barline_stroke(img, norm_x=0.20, sys_bbox=sys_bbox) is False


def test_ladder_decision_measure_boundary_stroke_verified():
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    cv2.line(img, (500, 150), (500, 350), (0, 0, 0), thickness=3)

    item = AmbiguousItem(
        item_id="m3_boundary",
        target_type="measure_boundary",
        measure_index=2,
        system_page=0,
        system_index=0,
        system_bbox=(0.05, 0.10, 0.95, 0.40),
        original_value="1 2 3 4 5 6 7 1",
        metadata={"expected_beats": 4.0, "current_melody_beats": 8.0},
    )
    # Proposed split at beat 4.0 with barline at norm_x = 0.50
    inf = ItemInference(
        item_id="m3_boundary",
        target_type="measure_boundary",
        measure_index=2,
        proposed_value="split_at_beat_4.0",
        confidence=0.89,
        musical_justification="8拍长小节应在第4拍处拆分为两个4/4小节",
        split_beat=4.0,
        barline_norm_x=0.50,
    )
    dec = evaluate_ladder_decision(item, inf, web_candidates=[], img_bgr=img)
    assert dec.accepted is True
    assert dec.severity == "auto_fixed"
    assert "pixel_stroke" in dec.evidence

    # If image lacks stroke, boundary split must be rejected -> needs_review
    blank_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    dec_no_stroke = evaluate_ladder_decision(item, inf, web_candidates=[], img_bgr=blank_img)
    assert dec_no_stroke.accepted is False
    assert dec_no_stroke.severity == "needs_review"


def test_run_escalation_ladder_end_to_end_mocked(monkeypatch):
    sheet = ParsedSheet(
        header=SongHeader(title="Test Song", original_key="C", time_signature="4/4"),
        pages=[],
        systems=[
            System(
                page=0,
                bbox=(0.05, 0.10, 0.95, 0.40),
                section_label="Verse",
                measures=[
                    Measure(
                        index=0,
                        bbox=(0.05, 0.10, 0.50, 0.40),
                        beats=4.0,
                        chords=[ChordSymbol(raw="1(2)", beat=1.0)],
                        melody="1 2 3 4",
                    ),
                    Measure(
                        index=1,
                        bbox=(0.50, 0.10, 0.95, 0.40),
                        beats=4.0,
                        chords=[ChordSymbol(raw="5xyz", beat=1.0)],
                        melody="5 6 7 1",
                    ),
                ],
            )
        ],
    )
    issues = [
        QualityIssue(
            stage="omr",
            measure_index=1,
            severity="needs_review",
            code="invalid_chord_grammar",
            message="第2小节和弦 '5xyz' 格式无法识别，请核对",
            detail={"raw": "5xyz", "alternatives": ["5", "5/7", "5sus"]},
        )
    ]
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    monkeypatch.setattr("app.qa.infer.ask_json", _mock_ask_json_infer)

    updated_issues = run_escalation_ladder(sheet, [img], issues, can_use_llm=True)
    assert len(updated_issues) == 1
    assert updated_issues[0].severity == "auto_fixed"
    assert updated_issues[0].code == "chord_inferred"
    assert sheet.systems[0].measures[1].chords[0].raw == "5"
