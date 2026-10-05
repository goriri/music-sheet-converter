"""Integration and unit tests for public history indexing, pagination, soft delete, and backfill."""
from __future__ import annotations

import io
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.history import (
    derive_render_meta,
    ensure_thumbnail,
    extract_title,
    generate_thumbnail_bytes,
    get_sheet_history,
    get_sheet_renders,
    list_history,
    make_history_key,
    make_inv_ts,
    restore_sheet,
    run_backfill,
    soft_delete_render,
    soft_delete_sheet,
)
from app.models import (
    Arrangement,
    ChordSymbol,
    Measure,
    MeasureArrangement,
    PageInfo,
    ParsedSheet,
    ResolvedChord,
    SongHeader,
    System,
)
from app.storage import LocalStorage, set_storage


def create_dummy_png_bytes(width: int = 200, height: int = 200, color: str = "white") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_sample_parsed_sheet(title: str = "晴天") -> ParsedSheet:
    return ParsedSheet(
        header=SongHeader(
            title=title,
            style="Slow Soul",
            time_signature="4/4",
            tempo_bpm=80.0,
            original_key="G",
            raw=f"{title} Slow Soul 4/4 80 原調 G",
        ),
        pages=[PageInfo(width=800, height=1200)],
        systems=[
            System(
                page=0,
                bbox=(0.05, 0.1, 0.95, 0.3),
                measures=[
                    Measure(
                        index=0,
                        bbox=(0.05, 0.1, 0.5, 0.3),
                        beats=4.0,
                        chords=[ChordSymbol(raw="1", beat=1.0)],
                        melody="1 2 3 4",
                        lyrics="故事的小黄花",
                    )
                ],
            )
        ],
        warnings=[],
    )


def create_sample_arrangement() -> Arrangement:
    return Arrangement(
        instrument="piano",
        difficulty="intermediate",
        start_key="G",
        style="Slow Soul",
        measures=[
            MeasureArrangement(
                measure_index=0,
                tonic_pc=7,
                key_name="G",
                chords=[
                    ResolvedChord(
                        raw="1",
                        name="G",
                        beat=1.0,
                        root_pc=7,
                        bass_pc=7,
                        pcs=[7, 11, 2],
                        quality="maj",
                    )
                ],
            )
        ],
        notes=[],
    )


def make_valid_pdf_bytes() -> bytes:
    import pymupdf
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    return doc.tobytes()


@pytest.fixture(autouse=True)
def setup_history_test_env(tmp_path, monkeypatch):
    """Set up isolated local storage and mock external heavy services."""
    test_storage_dir = tmp_path / "storage"
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(test_storage_dir))
    monkeypatch.delenv("BUCKET", raising=False)
    store = LocalStorage(test_storage_dir)
    set_storage(store)

    sample_sheet = create_sample_parsed_sheet()

    omr_mod = types.ModuleType("app.omr.gemini_omr")
    omr_mod.parse_pages = lambda images: sample_sheet
    monkeypatch.setitem(sys.modules, "app.omr.gemini_omr", omr_mod)

    render_mod = types.ModuleType("app.render.overlay")
    render_mod.render_pdf = lambda pages, sheet, arrangement: make_valid_pdf_bytes()
    render_mod.render_pages = lambda pages, sheet, arrangement: [Image.new("RGB", (150, 150), "white")]
    monkeypatch.setitem(sys.modules, "app.render.overlay", render_mod)

    omr_qa_mod = types.ModuleType("app.qa.omr_verify")
    omr_qa_mod.verify_sheet = lambda pages, sheet, use_llm=True: sheet
    monkeypatch.setitem(sys.modules, "app.qa.omr_verify", omr_qa_mod)

    arrange_qa_mod = types.ModuleType("app.qa.arrange_check")
    arrange_qa_mod.ArrangementQAError = Exception
    arrange_qa_mod.check_and_repair = lambda sheet, arr, use_llm=False: arr
    monkeypatch.setitem(sys.modules, "app.qa.arrange_check", arrange_qa_mod)

    yield store
    set_storage(None)


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_inv_ts_ordering():
    """Verify newer timestamps produce smaller inv_ts strings for newest-first sorting."""
    dt_old = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt_new = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    inv_old = make_inv_ts(dt_old)
    inv_new = make_inv_ts(dt_new)

    assert len(inv_old) == 13
    assert len(inv_new) == 13
    # Newer must sort BEFORE older in standard string sort
    assert inv_new < inv_old


def test_upload_creates_thumbnail_and_history_entry(client, setup_history_test_env):
    """Uploading a sheet should immediately create a thumbnail and a history entry."""
    store = setup_history_test_env
    png_data = create_dummy_png_bytes(400, 300)
    files = [("files", ("sheet1.png", png_data, "image/png"))]

    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]

    # Thumbnail exists
    thumb_path = f"sheets/{sheet_id}/thumb.jpg"
    assert store.exists(thumb_path)
    thumb_img = Image.open(io.BytesIO(store.get_bytes(thumb_path)))
    assert thumb_img.width == 320

    # History key exists
    state = store.get_json(f"sheets/{sheet_id}/state.json")
    history_key = state.get("history_key")
    assert history_key is not None
    assert store.exists(history_key)

    h_entry = store.get_json(history_key)
    assert h_entry["sheet_id"] == sheet_id
    assert h_entry["thumb_path"] == thumb_path


def test_parse_updates_history_title(client, setup_history_test_env):
    """After background OMR parsing, the history entry title must be updated."""
    store = setup_history_test_env
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet1.png", png_data, "image/png"))]

    resp = client.post("/api/sheets", files=files)
    assert resp.status_code == 200
    sheet_id = resp.json()["sheet_id"]

    # Wait for parse to finish
    for _ in range(50):
        get_res = client.get(f"/api/sheets/{sheet_id}")
        if get_res.json().get("status") == "ready":
            break
        time.sleep(0.05)

    assert get_res.json().get("status") == "ready"

    # History entry should now reflect "晴天" (from mocked ParsedSheet)
    state = store.get_json(f"sheets/{sheet_id}/state.json")
    h_entry = store.get_json(state["history_key"])
    assert h_entry["title"] == "晴天"


def test_render_writes_meta_and_lists_in_history(client, setup_history_test_env):
    """Rendering should write {render_folder}/meta.json and appear in sheet history."""
    store = setup_history_test_env
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet1.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        if client.get(f"/api/sheets/{sheet_id}").json().get("status") == "ready":
            break
        time.sleep(0.05)

    render_resp = client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "G", "difficulty": "intermediate", "instrument": "piano"},
    )
    assert render_resp.status_code == 200

    meta_path = f"sheets/{sheet_id}/renders/G_intermediate_piano/meta.json"
    assert store.exists(meta_path)
    meta = store.get_json(meta_path)
    assert meta["instrument"] == "piano"
    assert meta["difficulty"] == "intermediate"
    assert meta["start_key"] == "G"
    assert meta["deleted"] is False

    # Check GET /api/history/{sheet_id}
    sheet_hist = client.get(f"/api/history/{sheet_id}").json()
    assert sheet_hist["sheet_id"] == sheet_id
    assert len(sheet_hist["renders"]) == 1
    assert sheet_hist["renders"][0]["name"] == "G_intermediate_piano"


def test_history_pagination_newest_first(client, setup_history_test_env):
    """Verify GET /api/history pagination works with cursor and newest-first order."""
    store = setup_history_test_env

    # Synthesize 5 sheets with controlled timestamps
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        s_id = f"sheet_{i:02d}"
        s_time = datetime.fromtimestamp(base_time.timestamp() + i * 3600, tz=timezone.utc).isoformat()
        store.put_json(
            f"sheets/{s_id}/state.json",
            {
                "sheet_id": s_id,
                "status": "ready",
                "page_count": 1,
                "created_at": s_time,
            },
        )
        store.put_bytes(f"sheets/{s_id}/thumb.jpg", b"fake_jpeg")
        inv = make_inv_ts(datetime.fromisoformat(s_time))
        h_key = f"history/{inv}_{s_id}.json"
        store.put_json(
            h_key,
            {
                "sheet_id": s_id,
                "title": f"Song {i}",
                "created_at": s_time,
                "page_count": 1,
                "thumb_path": f"sheets/{s_id}/thumb.jpg",
            },
        )

    # Page 1 with limit=2
    res1 = client.get("/api/history?limit=2")
    assert res1.status_code == 200
    data1 = res1.json()
    assert len(data1["items"]) == 2
    # Newest first: sheet_04 was created last (+4 hours), sheet_03 (+3 hours)
    assert data1["items"][0]["sheet_id"] == "sheet_04"
    assert data1["items"][1]["sheet_id"] == "sheet_03"
    assert data1["next_cursor"] is not None

    # Page 2 with cursor
    res2 = client.get(f"/api/history?limit=2&cursor={data1['next_cursor']}")
    assert res2.status_code == 200
    data2 = res2.json()
    assert len(data2["items"]) == 2
    assert data2["items"][0]["sheet_id"] == "sheet_02"
    assert data2["items"][1]["sheet_id"] == "sheet_01"
    assert data2["next_cursor"] is not None

    # Page 3
    res3 = client.get(f"/api/history?limit=2&cursor={data2['next_cursor']}")
    assert res3.status_code == 200
    data3 = res3.json()
    assert len(data3["items"]) == 1
    assert data3["items"][0]["sheet_id"] == "sheet_00"
    assert data3["next_cursor"] is None


def test_soft_delete_sheet(client, setup_history_test_env):
    """Soft delete removes history index entry and hides sheet from public endpoints."""
    store = setup_history_test_env
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet1.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        if client.get(f"/api/sheets/{sheet_id}").json().get("status") == "ready":
            break
        time.sleep(0.05)

    # Before delete: present in history
    hist_before = client.get("/api/history").json()
    assert any(item["sheet_id"] == sheet_id for item in hist_before["items"])

    # Perform soft delete
    del_resp = client.delete(f"/api/history/{sheet_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["sheet_id"] == sheet_id

    # Verify state.json marked deleted
    state = store.get_json(f"sheets/{sheet_id}/state.json")
    assert state.get("deleted") is True
    assert "deleted_at" in state

    # After delete: hidden from history list
    hist_after = client.get("/api/history").json()
    assert not any(item["sheet_id"] == sheet_id for item in hist_after["items"])

    # GET /api/history/{sheet_id} returns 404
    assert client.get(f"/api/history/{sheet_id}").status_code == 404

    # GET /api/sheets/{sheet_id} returns 404
    assert client.get(f"/api/sheets/{sheet_id}").status_code == 404


def test_soft_delete_single_render(client, setup_history_test_env):
    """Soft delete of a single render hides it from the sheet's render list."""
    store = setup_history_test_env
    png_data = create_dummy_png_bytes(200, 200)
    files = [("files", ("sheet1.png", png_data, "image/png"))]
    resp = client.post("/api/sheets", files=files)
    sheet_id = resp.json()["sheet_id"]

    for _ in range(50):
        if client.get(f"/api/sheets/{sheet_id}").json().get("status") == "ready":
            break
        time.sleep(0.05)

    # Render piano
    client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "G", "difficulty": "intermediate", "instrument": "piano"},
    )
    # Render beginner
    client.post(
        f"/api/sheets/{sheet_id}/render",
        json={"start_key": "G", "difficulty": "beginner", "instrument": "piano"},
    )

    detail_before = client.get(f"/api/history/{sheet_id}").json()
    assert len(detail_before["renders"]) == 2

    # Delete the intermediate render
    del_render = client.delete(f"/api/history/{sheet_id}/renders/G_intermediate_piano")
    assert del_render.status_code == 200

    detail_after = client.get(f"/api/history/{sheet_id}").json()
    assert len(detail_after["renders"]) == 1
    assert detail_after["renders"][0]["name"] == "G_beginner_piano"


def test_backfill_and_restore(client, setup_history_test_env):
    """Backfill properly recovers unindexed sheets and --restore re-enables soft-deleted sheets."""
    store = setup_history_test_env
    legacy_id = "legacy_001"

    # Create a legacy sheet folder without history entry, thumb, or render meta
    store.put_json(
        f"sheets/{legacy_id}/state.json",
        {
            "sheet_id": legacy_id,
            "status": "ready",
            "page_count": 1,
            "created_at": "2026-02-01T10:00:00Z",
        },
    )
    store.put_bytes(f"sheets/{legacy_id}/pages/page_0.png", create_dummy_png_bytes(300, 400), "image/png")
    store.put_json(f"sheets/{legacy_id}/parsed.json", create_sample_parsed_sheet("青花瓷").model_dump())

    # Add a legacy render without meta.json
    store.put_json(
        f"sheets/{legacy_id}/renders/C_intermediate_piano/arrangement.json",
        {"capo": 0, "shape_key": "C", "issues": []},
    )
    store.put_bytes(f"sheets/{legacy_id}/renders/C_intermediate_piano/score.pdf", make_valid_pdf_bytes())
    store.put_bytes(f"sheets/{legacy_id}/renders/C_intermediate_piano/preview_0.png", create_dummy_png_bytes(100, 100))

    # Before backfill: not in history
    items_before, _ = list_history(store)
    assert not any(i["sheet_id"] == legacy_id for i in items_before)

    # Run backfill
    stats = run_backfill(store, dry_run=False)
    assert stats["sheets_scanned"] == 1
    assert stats["history_entries_created"] == 1
    assert stats["thumbnails_created"] == 1
    assert stats["render_meta_created"] == 1

    # After backfill: listed in history with recovered title and thumbnail
    h_after = client.get("/api/history").json()
    item = next(i for i in h_after["items"] if i["sheet_id"] == legacy_id)
    assert item["title"] == "青花瓷"
    assert item["thumb_url"] is not None

    # Check detail has derived render
    detail = client.get(f"/api/history/{legacy_id}").json()
    assert len(detail["renders"]) == 1
    assert detail["renders"][0]["name"] == "C_intermediate_piano"

    # Soft delete then restore
    client.delete(f"/api/history/{legacy_id}")
    assert client.get(f"/api/history/{legacy_id}").status_code == 404

    # Restore via restore_sheet
    restored = restore_sheet(legacy_id, store)
    assert restored is True
    assert client.get(f"/api/history/{legacy_id}").status_code == 200


def test_invalid_ids_rejected(client):
    """Path traversal and malformed sheet_id / render_name return 400 or 404."""
    assert client.get("/api/history/../../etc/passwd").status_code == 404
    assert client.get("/api/history/valid_id%2e%2e").status_code == 400
    assert client.delete("/api/history/invalid@id").status_code == 400
    assert client.delete("/api/history/sheet1/renders/bad@render").status_code == 400
    assert client.delete("/api/history/sheet1/renders/bad%2e%2e").status_code == 400
    assert client.delete("/api/history/nonexistent_sheet_id").status_code == 404


def test_broken_sheet_does_not_500(client, setup_history_test_env):
    """A corrupted history entry or missing state file must be safely skipped."""
    store = setup_history_test_env
    # Corrupt entry
    store.put_bytes("history/9999999999999_corrupt.json", b"NOT VALID JSON")

    resp = client.get("/api/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["items"], list)
