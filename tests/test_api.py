"""API integration tests using FastAPI TestClient with LocalStorage."""
from __future__ import annotations

import io
import json
import os
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.models import (
    Arrangement,
    ChordSymbol,
    Difficulty,
    Event,
    KeyChange,
    Measure,
    MeasureArrangement,
    Note,
    PageInfo,
    ParsedSheet,
    ResolvedChord,
    SongHeader,
    System,
)
from app.storage import LocalStorage, set_storage


def create_sample_parsed_sheet() -> ParsedSheet:
    return ParsedSheet(
        header=SongHeader(
            title="掉了",
            style="Slow Soul",
            time_signature="4/4",
            tempo_bpm=81.0,
            original_key="F#",
            male_key="Bb",
            female_key="F",
            raw="掉了 Slow Soul 4/4 81 原調 F#",
        ),
        pages=[PageInfo(width=1000, height=1400)],
        systems=[
            System(
                page=0,
                bbox=(0.05, 0.1, 0.95, 0.25),
                section_label="Intro",
                measures=[
                    Measure(
                        index=0,
                        bbox=(0.05, 0.1, 0.5, 0.25),
                        beats=4.0,
                        chords=[ChordSymbol(raw="1", beat=1.0)],
                        melody="1 2 3 4",
                        lyrics="",
                    ),
                    Measure(
                        index=1,
                        bbox=(0.5, 0.1, 0.95, 0.25),
                        beats=4.0,
                        chords=[ChordSymbol(raw="5/7", beat=1.0)],
                        melody="5 - - -",
                        lyrics="",
                    ),
                ],
            )
        ],
        key_changes=[
            KeyChange(at_measure=1, raw="轉成2調(Ab)", semitones=2)
        ],
        warnings=[],
    )


def create_sample_arrangement() -> Arrangement:
    return Arrangement(
        instrument="piano",
        difficulty="intermediate",
        start_key="F#",
        style="Slow Soul",
        measures=[
            MeasureArrangement(
                measure_index=0,
                tonic_pc=6,
                key_name="F#",
                chords=[
                    ResolvedChord(
                        raw="1",
                        name="F#",
                        beat=1.0,
                        root_pc=6,
                        bass_pc=6,
                        pcs=[6, 10, 1],
                        quality="maj",
                    )
                ],
                rh=[Event(onset=0.0, duration=2.0, notes=[Note(midi=66, finger=1)])],
                lh=[Event(onset=0.0, duration=4.0, notes=[Note(midi=42, finger=5)])],
            )
        ],
        notes=["Test arrangement remarks"],
    )


def create_dummy_png_bytes(width: int = 200, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_valid_pdf_bytes() -> bytes:
    import pymupdf
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    return doc.tobytes()


class ArrangementQAError(Exception):
    pass


def _mock_fail_verify(*args, **kwargs):
    raise RuntimeError("OMR QA service unavailable")


def _mock_failing_repair(sheet, arr, use_llm=False):
    raise ArrangementQAError("Safe fallback failed voice-leading constraints")


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Set up temporary local storage and mock external OMR / arrange / render modules."""
    test_storage_dir = tmp_path / "storage"
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(test_storage_dir))
    monkeypatch.delenv("BUCKET", raising=False)
    store = LocalStorage(test_storage_dir)
    set_storage(store)

    sample_sheet = create_sample_parsed_sheet()
    sample_arr = create_sample_arrangement()

    # Mock app.omr.gemini_omr module
    omr_mod = types.ModuleType("app.omr.gemini_omr")
    omr_mod.parse_pages = lambda images: sample_sheet
    monkeypatch.setitem(sys.modules, "app.omr.gemini_omr", omr_mod)

    # Mock app.arrange.piano module
    arrange_mod = types.ModuleType("app.arrange.piano")
    arrange_mod.arrange = lambda sheet, start_key, difficulty: sample_arr
    monkeypatch.setitem(sys.modules, "app.arrange.piano", arrange_mod)

    # Mock app.render.overlay module
    render_mod = types.ModuleType("app.render.overlay")
    render_mod.render_pdf = lambda pages, sheet, arrangement: make_valid_pdf_bytes()
    render_mod.render_pages = lambda pages, sheet, arrangement: [Image.new("RGB", (150, 150), "white")]
    monkeypatch.setitem(sys.modules, "app.render.overlay", render_mod)

    # Mock app.qa.omr_verify module
    omr_qa_mod = types.ModuleType("app.qa.omr_verify")
    omr_qa_mod.verify_sheet = lambda pages, sheet, use_llm=True: sheet
    monkeypatch.setitem(sys.modules, "app.qa.omr_verify", omr_qa_mod)

    # Mock app.qa.arrange_check module
    arrange_qa_mod = types.ModuleType("app.qa.arrange_check")
    arrange_qa_mod.ArrangementQAError = ArrangementQAError
    arrange_qa_mod.check_and_repair = lambda sheet, arr, use_llm=False: arr
    monkeypatch.setitem(sys.modules, "app.qa.arrange_check", arrange_qa_mod)

    yield store
    set_storage(None)


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_static_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "台湾谱伴奏生成器" in resp.text


def test_bad_file_type_rejected(client):
    files = [("files", ("bad.exe", b"binarycontent", "application/octet-stream"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_file_count_limit_rejected(client):
    png_data = create_dummy_png_bytes(50, 50)
    files = [("files", (f"page_{i}.png", png_data, "image/png")) for i in range(13)]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 400
    assert "Upload limit exceeded" in resp.json()["detail"]


def test_file_size_limit_rejected(client, monkeypatch):
    from app import main
    # Temporarily reduce limit or pass oversize file
    large_data = b"X" * (20 * 1024 * 1024 + 10)
    files = [("files", ("large.png", large_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 400
    assert "exceeds the 20MB limit" in resp.json()["detail"]


def test_full_pipeline_upload_poll_put_render(client):
    # 1. Upload
    png_data = create_dummy_png_bytes(300, 200)
    files = [("files", ("sheet1.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]
    assert sheet_id

    # 2. Poll until ready
    max_wait = 5.0
    start_time = time.time()
    state = None
    while time.time() - start_time < max_wait:
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        assert get_resp.status_code == 200
        state = get_resp.json()
        if state.get("status") == "ready":
            break
        time.sleep(0.05)

    assert state is not None
    assert state.get("status") == "ready"
    assert state.get("parsed") is not None
    parsed = state["parsed"]
    assert parsed["header"]["title"] == "掉了"
    assert len(parsed["systems"]) == 1

    # Check page serving endpoint
    page_resp = client.get(f"/api/pages/{sheet_id}/0")
    assert page_resp.status_code == 200
    assert page_resp.headers["content-type"] == "image/png"

    # 3. PUT corrected parsed sheet
    parsed["header"]["title"] = "掉了（修改版）"
    parsed["systems"][0]["measures"][0]["chords"][0]["raw"] = "1(2)"
    put_resp = client.put(f"/api/sheets/{sheet_id}/parsed", json=parsed)
    assert put_resp.status_code == 200
    assert put_resp.json()["status"] == "ok"

    # Verify updated parsed sheet
    get_resp2 = client.get(f"/api/sheets/{sheet_id}")
    assert get_resp2.json()["parsed"]["header"]["title"] == "掉了（修改版）"
    assert get_resp2.json()["parsed"]["systems"][0]["measures"][0]["chords"][0]["raw"] == "1(2)"

    # 4. Render non-piano instrument -> HTTP 400 '暂未支持'
    render_non_piano = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "F#", "difficulty": "intermediate", "instrument": "guitar"},
    )
    assert render_non_piano.status_code == 400
    assert render_non_piano.json()["detail"] == "暂未支持"

    # 5. Render piano
    render_resp = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "F#", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp.status_code == 200
    render_data = render_resp.json()
    assert "pdf_url" in render_data
    assert "preview_urls" in render_data
    assert len(render_data["preview_urls"]) > 0
    assert render_data["arrangement"]["instrument"] == "piano"

    # 6. File serving
    pdf_resp = client.get(render_data["pdf_url"])
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-")

    prev_resp = client.get(render_data["preview_urls"][0])
    assert prev_resp.status_code == 200
    assert prev_resp.headers["content-type"] == "image/png"


def test_pdf_upload_and_splitting(client):
    import pymupdf
    doc = pymupdf.open()
    doc.new_page(width=500, height=700)
    doc.new_page(width=500, height=700)
    pdf_bytes = doc.tobytes()

    files = [("files", ("twopage.pdf", pdf_bytes, "application/pdf"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]

    # Verify both pages were extracted and stored
    get_resp = client.get(f"/api/sheets/{sheet_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["page_count"] == 2

    # Check pages can be fetched
    p0 = client.get(f"/api/pages/{sheet_id}/0")
    assert p0.status_code == 200
    p1 = client.get(f"/api/pages/{sheet_id}/1")
    assert p1.status_code == 200


def test_large_image_downscaled(client, setup_test_environment):
    store = setup_test_environment
    large_img = Image.new("RGB", (3200, 2000), color="blue")
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")
    files = [("files", ("huge.png", buf.getvalue(), "image/png"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]

    page_data = store.get_bytes(f"sheets/{sheet_id}/pages/page_0.png")
    saved_img = Image.open(io.BytesIO(page_data))
    assert max(saved_img.size) == 2400
    assert saved_img.width == 2400


def test_path_traversal_rejected(client):
    resp1 = client.get("/api/files/app/main.py")
    assert resp1.status_code == 400
    assert "Invalid path" in resp1.json()["detail"]

    resp2 = client.get("/api/files/sheets/%2e%2e/main.py")
    assert resp2.status_code == 400
    assert "Invalid path" in resp2.json()["detail"]

    resp3 = client.get("/api/files/sheets/..%2fmain.py")
    assert resp3.status_code == 400
    assert "Invalid path" in resp3.json()["detail"]

    resp4 = client.get("/api/files/etc/passwd")
    assert resp4.status_code == 400
    assert "Invalid path" in resp4.json()["detail"]


def test_parsing_timeout_reported_as_error(client, setup_test_environment):
    store = setup_test_environment
    sheet_id = "timeout123"
    old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    state = {
        "sheet_id": sheet_id,
        "status": "parsing",
        "progress": 0.5,
        "error": None,
        "created_at": old_timestamp,
        "updated_at": old_timestamp,
    }
    store.put_json(f"sheets/{sheet_id}/state.json", state)

    resp = client.get(f"/api/sheets/{sheet_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "timed out" in data["error"].lower()


def test_qa_missing_job_still_ready_with_info_issue(client, monkeypatch):
    omr_qa_mod = types.ModuleType("app.qa.omr_verify")
    omr_qa_mod.verify_sheet = _mock_fail_verify
    monkeypatch.setitem(sys.modules, "app.qa.omr_verify", omr_qa_mod)

    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        data = get_resp.json()
        if data.get("status") == "ready":
            break
        time.sleep(0.05)

    assert data.get("status") == "ready"
    assert data.get("parsed") is not None
    issues = data["parsed"].get("issues", [])
    assert any(i.get("code") == "omr_qa_unavailable" and i.get("severity") == "info" for i in issues)


def test_arrangement_qa_error_returns_500_and_no_pdf_stored(client, setup_test_environment, monkeypatch):
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    arrange_qa_mod = types.ModuleType("app.qa.arrange_check")
    arrange_qa_mod.ArrangementQAError = ArrangementQAError
    arrange_qa_mod.check_and_repair = _mock_failing_repair
    monkeypatch.setitem(sys.modules, "app.qa.arrange_check", arrange_qa_mod)

    render_resp = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "F#", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp.status_code == 500
    detail = render_resp.json()["detail"]
    assert "编配自动修复失败" in detail

    pdf_path = f"sheets/{sheet_id}/renders/F#_intermediate_piano/score.pdf"
    assert not store.exists(pdf_path)


def test_put_clears_needs_review_and_marks_confidence(client, setup_test_environment):
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]
    from app.models import QualityIssue
    issue = QualityIssue(
        stage="omr",
        measure_index=0,
        severity="needs_review",
        code="chord_disagreement",
        message="第1小节和弦存疑：识别为 1(2)，候选为 5/7",
    )
    parsed["issues"].append(issue.model_dump())
    parsed["systems"][0]["measures"][0]["chords"][0]["confidence"] = 0.5
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    state_before = client.get(f"/api/sheets/{sheet_id}").json()
    assert any(i["severity"] == "needs_review" and i["measure_index"] == 0 for i in state_before["parsed"]["issues"])

    parsed["systems"][0]["measures"][0]["chords"][0]["raw"] = "5/7"
    put_resp = client.put(f"/api/sheets/{sheet_id}/parsed", json=parsed)
    assert put_resp.status_code == 200

    state_after = client.get(f"/api/sheets/{sheet_id}").json()
    updated_chord = state_after["parsed"]["systems"][0]["measures"][0]["chords"][0]
    assert updated_chord["confidence"] == 1.0
    assert updated_chord["raw"] == "5/7"
    assert not any(i["severity"] == "needs_review" and i["measure_index"] == 0 for i in state_after["parsed"]["issues"])


def test_put_garbage_chord_returns_422(client):
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]
    parsed["systems"][0]["measures"][0]["chords"][0]["raw"] = "xyz_garbage"
    put_resp = client.put(f"/api/sheets/{sheet_id}/parsed", json=parsed)
    assert put_resp.status_code == 422
    assert "语法错误" in str(put_resp.json()) or "xyz_garbage" in str(put_resp.json())


def test_appendix_page_present_iff_issues(client, setup_test_environment):
    import pymupdf
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]

    # Case A: No needs_review or auto_fixed issues
    parsed["issues"] = []
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    render_resp1 = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "C", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp1.status_code == 200
    pdf_url1 = render_resp1.json()["pdf_url"]
    pdf_bytes1 = client.get(pdf_url1).content
    doc1 = pymupdf.open(stream=pdf_bytes1, filetype="pdf")
    assert len(doc1) == 1

    # Case B: With needs_review / auto_fixed issue
    from app.models import QualityIssue
    parsed["issues"] = [
        QualityIssue(
            stage="omr",
            measure_index=0,
            severity="needs_review",
            code="chord_disagreement",
            message="第1小节和弦需确认",
        ).model_dump()
    ]
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    render_resp2 = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "C", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp2.status_code == 200
    pdf_url2 = render_resp2.json()["pdf_url"]
    pdf_bytes2 = client.get(pdf_url2).content
    doc2 = pymupdf.open(stream=pdf_bytes2, filetype="pdf")
    assert len(doc2) == 2


def test_render_gate_409_on_low_layout_confidence(client, setup_test_environment):
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]
    from app.models import QualityIssue
    parsed["layout_confidence"] = 0.5
    parsed["issues"].append(
        QualityIssue(
            stage="omr",
            measure_index=0,
            severity="needs_review",
            code="barline_count_mismatch",
            message="第1页第1行小节线检测数量不符，请核对小节划分",
        ).model_dump()
    )
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    render_resp = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "C", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp.status_code == 409
    assert "此谱版面识别不可靠，暂不生成" in render_resp.json()["detail"]


def test_render_gate_409_cleared_after_confirming_structural_issues(client, setup_test_environment):
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]
    from app.models import QualityIssue
    parsed["layout_confidence"] = 0.5
    parsed["issues"].append(
        QualityIssue(
            stage="omr",
            measure_index=0,
            severity="needs_review",
            code="barline_count_mismatch",
            message="第1页第1行小节线检测数量不符，请核对小节划分",
        ).model_dump()
    )
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    # 1. First render should fail with 409
    render_resp1 = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "C", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp1.status_code == 409

    # 2. Confirm structural issue
    confirm_resp = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={"issue_code": "barline_count_mismatch", "measure_index": 0, "action": "confirm"},
    )
    assert confirm_resp.status_code == 200
    assert confirm_resp.json()["status"] == "ok"

    # 3. Second render should now succeed with 200
    render_resp2 = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "C", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp2.status_code == 200
    assert "pdf_url" in render_resp2.json()


def test_confirm_endpoint_clears_needs_review_for_issue(client, setup_test_environment):
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]
    from app.models import QualityIssue
    parsed["issues"] = [
        QualityIssue(
            stage="omr",
            measure_index=0,
            severity="needs_review",
            code="barline_count_mismatch",
            message="第1页第1行小节线检测数量不符，请核对小节划分",
        ).model_dump(),
        QualityIssue(
            stage="omr",
            measure_index=1,
            severity="needs_review",
            code="chord_ambiguous",
            message="第2小节和弦存疑",
        ).model_dump(),
    ]
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    # Confirm the structural issue with correction note
    conf1 = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "barline_count_mismatch",
            "measure_index": 0,
            "action": "correct",
            "measure_count": 4,
            "note": "第1页第1行小节数应为 4",
        },
    )
    assert conf1.status_code == 200
    issues1 = conf1.json()["parsed"]["issues"]
    assert not any(i["code"] == "barline_count_mismatch" and i["severity"] == "needs_review" for i in issues1)
    warnings1 = conf1.json()["parsed"]["warnings"]
    assert any("第1页第1行小节数应为 4" in w for w in warnings1)

    # Confirm chord issue with specific chord
    conf2 = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "chord_ambiguous",
            "measure_index": 1,
            "action": "confirm",
            "chord": "5m7/2",
        },
    )
    assert conf2.status_code == 200
    p2 = conf2.json()["parsed"]
    assert not any(i["code"] == "chord_ambiguous" and i["severity"] == "needs_review" for i in p2["issues"])
    c2 = p2["systems"][0]["measures"][1]["chords"][0]
    assert c2["raw"] == "5m7/2"
    assert c2["confidence"] == 1.0


def test_put_and_confirm_payloads_validated(client, setup_test_environment):
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    # 1. Invalid chord syntax to confirm -> 422
    bad_chord = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={"chord": "not_a_valid_chord_!@#$"},
    )
    assert bad_chord.status_code == 422
    assert "和弦语法错误" in bad_chord.json()["detail"]

    # 2. Invalid measure_count (<= 0) to confirm -> 422
    bad_count = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={"measure_count": 0},
    )
    assert bad_count.status_code == 422
    assert "小节数必须大于 0" in bad_count.json()["detail"]

    # 3. Invalid beat (<= 0) to confirm -> 422
    bad_beat = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={"beat": -1.0},
    )
    assert bad_beat.status_code == 422
    assert "拍数必须大于 0" in bad_beat.json()["detail"]

    # 4. Non-existent sheet to confirm -> 404
    not_found = client.post(
        "/api/sheets/nonexistent123/confirm",
        json={"action": "confirm"},
    )
    assert not_found.status_code == 404


def test_qianlizhiwai_review_and_gate_workflow(client, setup_test_environment):
    batch_dir = Path("out/batch/qianlizhiwai")
    verified_file = batch_dir / "verified.json"
    if not verified_file.exists():
        pytest.skip("out/batch/qianlizhiwai/verified.json not found")

    store = setup_test_environment
    sheet_id = "test_qianlizhiwai_suite"

    with open(verified_file) as f:
        parsed_data = json.load(f)

    store.put_json(
        f"sheets/{sheet_id}/state.json",
        {
            "sheet_id": sheet_id,
            "status": "ready",
            "created_at": "2026-09-28T00:00:00Z",
            "page_count": 2,
            "structural_confirmed": False,
        },
    )
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed_data)
    p1_path = batch_dir / "p1.png"
    p2_path = batch_dir / "p2.png"
    if p1_path.exists():
        store.put_bytes(f"sheets/{sheet_id}/pages/page_0.png", p1_path.read_bytes(), "image/png")
    if p2_path.exists():
        store.put_bytes(f"sheets/{sheet_id}/pages/page_1.png", p2_path.read_bytes(), "image/png")

    # 1. Sheet state retrieval
    get_res = client.get(f"/api/sheets/{sheet_id}")
    assert get_res.status_code == 200
    state = get_res.json()
    assert state["status"] == "ready"
    assert state["parsed"]["header"]["title"] == "千里之外"

    # 2. Page image serving
    if p1_path.exists():
        page0_res = client.get(f"/api/pages/{sheet_id}/0")
        assert page0_res.status_code == 200
        assert page0_res.headers["content-type"] == "image/png"

    # 3. Confirm ambiguous chord
    r_c1 = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "chord_ambiguous",
            "measure_index": 5,
            "action": "confirm",
            "chord": "3m",
        },
    )
    assert r_c1.status_code == 200
    p_c1 = r_c1.json()["parsed"]
    assert not any(
        i["code"] == "chord_ambiguous" and i.get("measure_index") == 5 and i["severity"] == "needs_review"
        for i in p_c1["issues"]
    )

    # 4. Confirm missing chord
    r_c2 = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "missing_chord_suspected",
            "measure_index": 26,
            "action": "confirm",
            "chord": "57sus",
        },
    )
    assert r_c2.status_code == 200
    p_c2 = r_c2.json()["parsed"]
    assert not any(
        i["code"] == "missing_chord_suspected" and i.get("measure_index") == 26 and i["severity"] == "needs_review"
        for i in p_c2["issues"]
    )

    # 5. Confirm unlocated key change
    r_c3 = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "key_change_unlocated",
            "action": "confirm",
            "at_measure": 16,
            "semitones": 1,
        },
    )
    assert r_c3.status_code == 200
    p_c3 = r_c3.json()["parsed"]
    assert not any(
        i["code"] == "key_change_unlocated" and i["severity"] == "needs_review"
        for i in p_c3["issues"]
    )
    assert any(kc["at_measure"] == 16 for kc in p_c3["key_changes"])

    # 6. Confirm melody beat mismatch
    r_c4 = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "melody_beat_sum_mismatch",
            "measure_index": 22,
            "action": "confirm",
            "beat": 1.0,
        },
    )
    assert r_c4.status_code == 200
    p_c4 = r_c4.json()["parsed"]
    assert not any(
        i["code"] == "melody_beat_sum_mismatch" and i.get("measure_index") == 22 and i["severity"] == "needs_review"
        for i in p_c4["issues"]
    )


def test_sheet_with_only_warnings_renders_without_409(client, setup_test_environment):
    """QualityIssue severity 'warning' never blocks render with 409 and is never mutated by confirm/edit flows."""
    store = setup_test_environment
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        get_resp = client.get(f"/api/sheets/{sheet_id}")
        if get_resp.json().get("status") == "ready":
            break
        time.sleep(0.05)

    parsed = get_resp.json()["parsed"]
    from app.models import QualityIssue

    warning_issue = QualityIssue(
        stage="omr",
        measure_index=None,
        severity="warning",
        code="key_change_unlocated_warning",
        message="谱头标示 D→Eb 转调，但所给页面中未找到转调记号…",
        detail={},
    )
    arr_warning = QualityIssue(
        stage="arrange",
        measure_index=0,
        severity="warning",
        code="bass_only_without_context",
        message="第1小节低音无上下文参照，采用默认级数",
        detail={"measure": 0},
    )
    parsed["issues"] = [warning_issue.model_dump(), arr_warning.model_dump()]
    store.put_json(f"sheets/{sheet_id}/parsed.json", parsed)

    # 1. Render must succeed with 200 (not 409)
    render_resp = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "C", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp.status_code == 200
    assert "pdf_url" in render_resp.json()

    # 2. Confirm flows must not mutate or drop warning issues
    conf_resp = client.post(
        f"/api/sheets/{sheet_id}/confirm",
        json={
            "issue_code": "key_change_unlocated_warning",
            "action": "confirm",
        },
    )
    assert conf_resp.status_code == 200
    p_after_conf = conf_resp.json()["parsed"]
    warn_codes = [i["code"] for i in p_after_conf["issues"] if i["severity"] == "warning"]
    assert "key_change_unlocated_warning" in warn_codes
    assert "bass_only_without_context" in warn_codes

    # 3. Edit flows (PUT /api/sheets/{sheet_id}/parsed) must not drop warning issues
    put_resp = client.put(f"/api/sheets/{sheet_id}/parsed", json=p_after_conf)
    assert put_resp.status_code == 200
    p_after_put = put_resp.json()["parsed"]
    warn_codes_put = [i["code"] for i in p_after_put["issues"] if i["severity"] == "warning"]
    assert "key_change_unlocated_warning" in warn_codes_put
    assert "bass_only_without_context" in warn_codes_put


