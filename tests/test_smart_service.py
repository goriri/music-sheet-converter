"""Unit and integration tests for app.smart.service and /api/smart endpoint (M5).

All tests in this suite are OFFLINE ONLY (no live LLM calls, no Cloud Run network calls).
"""
from __future__ import annotations

import functools
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.smart.models import LeadChord, LeadMeasure, LeadNote, LeadSheet
from app.smart.search import ConsensusBar, ConsensusSection, SearchResult, SourceChart
from app.smart.service import _run_smart_pipeline, start_smart, trigger_cloud_run_job
from app.storage import LocalStorage, set_storage

client = TestClient(app)


def _sample_audio_lead(title: str = "晴天") -> LeadSheet:
    measures = [
        LeadMeasure(
            beats=4.0,
            notes=[LeadNote(onset=0.0, duration=1.0, degree=1, accidental=0, octave=0)],
            chords=[LeadChord(raw="1", beat=1.0)],
            section="主歌",
        ),
        LeadMeasure(
            beats=4.0,
            notes=[LeadNote(onset=0.0, duration=1.0, degree=5, accidental=0, octave=0)],
            chords=[LeadChord(raw="5/7", beat=1.0)],
        ),
    ]
    return LeadSheet(title=title, artist="周杰伦", key="G", measures=measures)


def _sample_search_result(title: str = "晴天") -> SearchResult:
    return SearchResult(
        title=title,
        artist="周杰伦",
        key="G",
        sources=[SourceChart(url="https://tab.com/1", title="晴天和弦", key="G")],
        consensus_sections=[
            ConsensusSection(
                label="主歌",
                bars=[
                    ConsensusBar(chords=["1"], support=2, total=2),
                    ConsensusBar(chords=["5/7"], support=2, total=2),
                ],
            )
        ],
    )


def _execute_mock_audio_job(storage: LocalStorage, jid: str) -> None:
    lead = _sample_audio_lead()
    storage.put_json(
        f"private/smart/{jid}/audio_lead.json",
        lead.model_dump(),
    )
    storage.put_json(
        f"private/smart/{jid}/status.json",
        {"status": "done", "progress": 1.0, "stage": "转录完成"},
    )


def _execute_mock_failing_audio_job(storage: LocalStorage, jid: str) -> None:
    storage.put_json(
        f"private/smart/{jid}/status.json",
        {"status": "error", "progress": 0.2, "stage": "转录失败", "error": "Model failed"},
    )


def test_smart_service_audio_and_search_success(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sheet_id = "test_smart_01"
    audio_data = b"ID3_MOCK_AUDIO_DATA_FOR_TESTING"

    runner = functools.partial(_execute_mock_audio_job, storage)

    with patch("app.smart.service.search_song", return_value=_sample_search_result()):
        start_smart(
            sheet_id=sheet_id,
            title="晴天",
            artist="周杰伦",
            audio_bytes=audio_data,
            audio_ext=".mp3",
            storage=storage,
            job_runner=runner,
            background=False,  # Run synchronously for test
            poll_interval=0.01,
            poll_timeout=2.0,
        )

    # 1. State verification
    state = storage.get_json(f"sheets/{sheet_id}/state.json")
    assert state["status"] == "ready"
    assert state["progress"] == 1.0
    assert state["progress_text"] == "已完成"
    assert state["page_count"] >= 1

    # 2. File outputs verification
    assert storage.exists(f"sheets/{sheet_id}/pages/page_0.png")
    assert storage.exists(f"sheets/{sheet_id}/thumb.jpg")
    assert storage.exists(f"sheets/{sheet_id}/parsed.json")
    assert storage.exists(f"sheets/{sheet_id}/smart_report.json")

    # 3. Report verification
    report = storage.get_json(f"sheets/{sheet_id}/smart_report.json")
    assert report["mode"] == "audio+search"
    assert report["agreement_ratio"] == 1.0
    assert report["n_agree"] == 2


def test_private_audio_isolation(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sheet_id = "test_iso_02"
    audio_data = b"CONFIDENTIAL_USER_AUDIO_DATA"

    with patch("app.smart.service.search_song", return_value=_sample_search_result()), \
         patch("app.smart.service.trigger_cloud_run_job", return_value=True):
        start_smart(
            sheet_id=sheet_id,
            title="Isolated",
            audio_bytes=audio_data,
            audio_ext=".wav",
            storage=storage,
            background=False,
            poll_interval=0.01,
            poll_timeout=0.05,
        )

    # Audio MUST be in private/smart/
    assert storage.exists(f"private/smart/{sheet_id}/input.wav")
    assert storage.get_bytes(f"private/smart/{sheet_id}/input.wav") == audio_data

    # Audio MUST NOT be anywhere under sheets/
    sheet_files = storage.list(f"sheets/{sheet_id}")
    for k in sheet_files:
        assert not k.endswith(".wav")
        assert not k.endswith(".mp3")
        assert "input" not in k

    # Public endpoint check: /api/files/private/... must return 403 or 400
    with patch("app.main.get_storage", return_value=storage):
        resp = client.get(f"/api/files/private/smart/{sheet_id}/input.wav")
        assert resp.status_code in (400, 403)


def test_smart_service_audio_failure_fallback_to_search(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sheet_id = "test_fallback_03"
    audio_data = b"BROKEN_AUDIO_DATA"

    runner = functools.partial(_execute_mock_failing_audio_job, storage)

    with patch("app.smart.service.search_song", return_value=_sample_search_result()):
        start_smart(
            sheet_id=sheet_id,
            title="晴天",
            audio_bytes=audio_data,
            storage=storage,
            job_runner=runner,
            background=False,
            poll_interval=0.01,
            poll_timeout=0.05,
        )

    state = storage.get_json(f"sheets/{sheet_id}/state.json")
    # Gracefully degraded to search-only
    assert state["status"] == "ready"
    assert storage.exists(f"sheets/{sheet_id}/smart_report.json")
    report = storage.get_json(f"sheets/{sheet_id}/smart_report.json")
    assert report["mode"] == "search_only"


def test_smart_service_search_only_no_audio(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sheet_id = "test_search_only_04"

    with patch("app.smart.service.search_song", return_value=_sample_search_result()):
        start_smart(
            sheet_id=sheet_id,
            title="晴天",
            artist="周杰伦",
            audio_bytes=None,
            storage=storage,
            background=False,
        )

    state = storage.get_json(f"sheets/{sheet_id}/state.json")
    assert state["status"] == "ready"
    assert storage.exists(f"sheets/{sheet_id}/smart_report.json")
    report = storage.get_json(f"sheets/{sheet_id}/smart_report.json")
    assert report["mode"] == "search_only"


def test_smart_service_both_fail_gives_error_state(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sheet_id = "test_fail_05"

    with patch("app.smart.service.search_song", return_value=None):
        start_smart(
            sheet_id=sheet_id,
            title="Unknown Song",
            audio_bytes=None,
            storage=storage,
            background=False,
        )

    state = storage.get_json(f"sheets/{sheet_id}/state.json")
    assert state["status"] == "error"
    assert "未成功" in state["error"]


def test_api_smart_endpoint_validation():
    # 1. Reject empty submission
    resp = client.post("/api/smart", data={"title": "", "artist": ""})
    assert resp.status_code == 400
    assert "请提供" in resp.json()["detail"]

    # 2. Reject unsupported audio format
    file_bad = io.BytesIO(b"fake_exe_content")
    resp = client.post(
        "/api/smart",
        files={"file": ("virus.exe", file_bad, "application/octet-stream")},
        data={"title": "Song"},
    )
    assert resp.status_code == 400
    assert "不支持的音频" in resp.json()["detail"]

    # 3. Accept valid title/artist
    with patch("app.smart.service.start_smart") as mock_start:
        resp = client.post("/api/smart", data={"title": "菊花台", "artist": "周杰伦"})
        assert resp.status_code == 200
        sheet_id = resp.json().get("sheet_id")
        assert sheet_id is not None
        mock_start.assert_called_once()

    # 4. Accept valid audio file
    file_ok = io.BytesIO(b"fake_mp3_content")
    with patch("app.smart.service.start_smart") as mock_start:
        resp = client.post(
            "/api/smart",
            files={"file": ("track.mp3", file_ok, "audio/mpeg")},
            data={"title": "Song"},
        )
        assert resp.status_code == 200
        assert resp.json().get("sheet_id") is not None


def test_api_get_sheet_status_returns_smart_report(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    sheet_id = "test_api_get_report"
    storage.put_json(
        f"sheets/{sheet_id}/state.json",
        {
            "sheet_id": sheet_id,
            "status": "ready",
            "progress": 1.0,
            "progress_text": "已完成",
        },
    )
    storage.put_json(
        f"sheets/{sheet_id}/smart_report.json",
        {
            "mode": "audio+search",
            "key_decision": "调性一致",
            "agreement_ratio": 0.95,
            "n_agree": 19,
            "n_disagree": 1,
            "n_audio_only": 0,
            "disagreements": [],
            "sources": [],
            "warnings": [],
        },
    )

    with patch("app.main.get_storage", return_value=storage):
        resp = client.get(f"/api/sheets/{sheet_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["smart_report"] is not None
        assert data["smart_report"]["agreement_ratio"] == 0.95
