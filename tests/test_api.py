"""API integration tests using FastAPI TestClient with LocalStorage."""
from __future__ import annotations

import io
import os
import sys
import time
import types
from datetime import datetime, timedelta, timezone

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
    render_mod.render_pdf = lambda pages, sheet, arrangement: b"%PDF-1.4 test accompaniment pdf"
    render_mod.render_pages = lambda pages, sheet, arrangement: [Image.new("RGB", (150, 150), "white")]
    monkeypatch.setitem(sys.modules, "app.render.overlay", render_mod)

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
    assert b"%PDF-1.4 test accompaniment pdf" in pdf_resp.content

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
