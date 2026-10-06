"""Orchestration and pipeline service for 智能创建 (smart sheet creation M5).

Coordinates:
- Audio file storage in isolated private storage (private/smart/{id}/input.<ext>).
- Cloud Run Job triggering (smart-audio) via Run Admin v2 REST API.
- Multi-source web search & consensus (app.smart.search).
- Audio job polling with status and progress text updates.
- Cross-check reconciliation (app.smart.crosscheck).
- From-scratch LeadSheet engraving to page PNGs (app.smart.render).
- ParsedSheet, thumbnail, report, and history index entry creation.
- Seamless handoff to the existing confirm/arrange/render pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional

from app.smart.crosscheck import merge
from app.smart.models import LeadSheet
from app.smart.render import render_lead_sheet
from app.smart.search import SearchResult, search_song
from app.storage import Storage, get_storage

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "cellular-cider-495602-r9"
DEFAULT_REGION = "us-central1"
DEFAULT_JOB_NAME = "smart-audio"


def trigger_cloud_run_job(
    job_id: str,
    project: Optional[str] = None,
    region: Optional[str] = None,
    job_name: Optional[str] = None,
) -> bool:
    """Trigger a Cloud Run Job execution via Google Cloud Run Admin v2 REST API."""
    proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    reg = region or os.environ.get("CLOUD_RUN_REGION", os.environ.get("REGION", DEFAULT_REGION))
    jname = job_name or os.environ.get("SMART_AUDIO_JOB", DEFAULT_JOB_NAME)

    url = f"https://run.googleapis.com/v2/projects/{proj}/locations/{reg}/jobs/{jname}:run"
    body = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {"name": "SMART_JOB_ID", "value": job_id},
                    ]
                }
            ]
        }
    }

    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
        resp = session.post(url, json=body, timeout=30.0)
        if resp.status_code in (200, 201, 202):
            logger.info("Successfully triggered Cloud Run Job %s for %s", jname, job_id)
            return True
        else:
            logger.warning(
                "Cloud Run Job trigger returned HTTP %d: %s",
                resp.status_code,
                resp.text[:300],
            )
            return False
    except Exception as exc:
        logger.warning("Failed triggering Cloud Run Job %s: %s", jname, exc)
        return False


def _update_smart_state(
    store: Storage,
    state_path: str,
    state: dict[str, Any],
    prog: float,
    text: str,
    status: str = "smart_running",
) -> None:
    state["status"] = status
    state["progress"] = round(prog, 3)
    state["progress_text"] = text
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        store.put_json(state_path, state)
    except Exception as exc:
        logger.warning("Failed updating state to %s: %s", state_path, exc)


def _run_smart_pipeline(
    sheet_id: str,
    title: str,
    artist: str,
    has_audio: bool,
    storage: Storage,
    poll_interval: float = 2.0,
    poll_timeout: float = 900.0,
) -> None:
    """Background execution pipeline for smart sheet creation."""
    store = storage
    state_path = f"sheets/{sheet_id}/state.json"
    prefix = f"private/smart/{sheet_id}"

    try:
        state = store.get_json(state_path)
    except Exception:
        now_iso = datetime.now(timezone.utc).isoformat()
        state = {
            "sheet_id": sheet_id,
            "status": "smart_running",
            "progress": 0.05,
            "progress_text": "初始化中",
            "mode": "smart",
            "created_at": now_iso,
            "updated_at": now_iso,
        }

    search_res: Optional[SearchResult] = None
    audio_lead: Optional[LeadSheet] = None

    try:
        # 1. Multi-source Web Search
        if title or artist:
            _update_smart_state(store, state_path, state, 0.10, "正在搜索网络和弦来源...")
            try:
                search_res = search_song(title=title, artist=artist)
            except Exception as s_exc:
                logger.warning("Web search failed for '%s - %s': %s", title, artist, s_exc)
                search_res = None

        # 2. Audio Processing (Wait for Cloud Run Job)
        if has_audio:
            _update_smart_state(store, state_path, state, 0.20, "音频处理中: 等待计算节点启动")
            status_file = f"{prefix}/status.json"
            lead_file = f"{prefix}/audio_lead.json"

            start_t = time.time()
            audio_finished = False
            last_stage = "等待计算节点启动"

            while time.time() - start_t < poll_timeout:
                if store.exists(status_file):
                    try:
                        s_data = store.get_json(status_file)
                        s_status = s_data.get("status")
                        s_prog = float(s_data.get("progress", 0.0))
                        s_stage = s_data.get("stage", last_stage)
                        last_stage = s_stage

                        scaled_prog = 0.20 + 0.60 * s_prog
                        _update_smart_state(store, state_path, state, scaled_prog, f"音频处理中: {s_stage}")

                        if s_status == "done":
                            audio_finished = True
                            break
                        elif s_status == "error":
                            logger.warning(
                                "Audio job reported error: %s",
                                s_data.get("error", "未知错误"),
                            )
                            break
                    except Exception as poll_exc:
                        logger.debug("Error reading audio status %s: %s", status_file, poll_exc)

                time.sleep(poll_interval)

            if audio_finished and store.exists(lead_file):
                try:
                    lead_dict = store.get_json(lead_file)
                    audio_lead = LeadSheet.model_validate(lead_dict)
                except Exception as l_exc:
                    logger.warning("Failed loading audio lead sheet: %s", l_exc)
                    audio_lead = None
            else:
                logger.info("Audio processing incomplete or timed out; proceeding to fallback")

        # 3. Graceful Degradation Check
        if audio_lead is None and search_res is None:
            _update_smart_state(
                store,
                state_path,
                state,
                0.0,
                "处理失败: 音频转录与网络搜索均未成功，请检查输入或稍后重试",
                status="error",
            )
            state["error"] = "音频转录与网络搜索均未成功，请检查输入或稍后重试"
            store.put_json(state_path, state)
            return

        # 4. Cross-Check Reconciliation
        _update_smart_state(store, state_path, state, 0.85, "正在交叉核对与对齐小节...")
        final_lead, report = merge(audio_lead, search_res)

        # Store cross-check report
        report_path = f"sheets/{sheet_id}/smart_report.json"
        store.put_json(report_path, report.model_dump())

        # 5. Typesetting & Render Lead Sheet
        _update_smart_state(store, state_path, state, 0.92, "正在排版生成乐谱图片...")
        png_bytes_list, parsed_sheet = render_lead_sheet(final_lead, page_width=2400)

        # Save pages to sheets/{sheet_id}/pages/page_{idx}.png
        page_paths: list[str] = []
        for idx, page_bytes in enumerate(png_bytes_list):
            p_path = f"sheets/{sheet_id}/pages/page_{idx}.png"
            store.put_bytes(p_path, page_bytes, content_type="image/png")
            page_paths.append(p_path)

        # Save thumbnail to sheets/{sheet_id}/thumb.jpg
        thumb_path = f"sheets/{sheet_id}/thumb.jpg"
        try:
            from app.history import generate_thumbnail_bytes

            thumb_bytes = generate_thumbnail_bytes(png_bytes_list[0])
            store.put_bytes(thumb_path, thumb_bytes, content_type="image/jpeg")
        except Exception as t_err:
            logger.warning("Failed generating thumbnail for %s: %s", sheet_id, t_err)

        # Save parsed.json
        parsed_path = f"sheets/{sheet_id}/parsed.json"
        store.put_json(parsed_path, parsed_sheet.model_dump())

        # 6. History Entry Creation
        song_title = final_lead.title or title or "智能创建乐谱"
        history_key = None
        try:
            from app.history import create_history_entry

            history_key = create_history_entry(
                sheet_id=sheet_id,
                created_at=state.get("created_at") or datetime.now(timezone.utc).isoformat(),
                page_count=len(page_paths),
                storage=store,
                thumb_path=thumb_path,
                title=song_title,
            )
        except Exception as h_err:
            logger.warning("Failed to create history entry for %s: %s", sheet_id, h_err)

        # 7. Final Ready State
        state["status"] = "ready"
        state["progress"] = 1.0
        state["progress_text"] = "已完成"
        state["message"] = "已完成"
        state["error"] = None
        state["page_count"] = len(page_paths)
        state["pages"] = page_paths
        state["history_key"] = history_key
        state["thumb_path"] = thumb_path
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)
        logger.info("Smart sheet pipeline completed successfully for %s", sheet_id)

    except Exception as exc:
        logger.exception("Error in smart pipeline for %s: %s", sheet_id, exc)
        state["status"] = "error"
        state["error"] = f"智能创建发生异常: {exc}"
        state["progress_text"] = "创建失败"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)


def start_smart(
    sheet_id: str,
    title: str = "",
    artist: str = "",
    audio_bytes: Optional[bytes] = None,
    audio_ext: str = ".mp3",
    storage: Optional[Storage] = None,
    job_runner: Optional[Any] = None,
    background: bool = True,
    poll_interval: float = 2.0,
    poll_timeout: float = 900.0,
) -> Optional[threading.Thread]:
    """Start smart sheet creation pipeline.

    Args:
        sheet_id: 12-char unique identifier.
        title: Song title.
        artist: Artist / singer.
        audio_bytes: Optional uploaded audio file bytes.
        audio_ext: File extension, e.g. '.mp3'.
        storage: Storage implementation (default: get_storage()).
        job_runner: Optional custom callable or object with .run(sheet_id) to trigger audio job.
        background: If True, executes pipeline in background thread.
        poll_interval: Audio polling interval in seconds.
        poll_timeout: Maximum seconds to wait for audio transcription job.
    """
    store = storage or get_storage()
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Initialize state.json under sheets/{sheet_id}/state.json
    state = {
        "sheet_id": sheet_id,
        "status": "smart_running",
        "progress": 0.05,
        "progress_text": "初始化中",
        "mode": "smart",
        "has_audio": bool(audio_bytes),
        "title": title,
        "artist": artist,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    store.put_json(f"sheets/{sheet_id}/state.json", state)

    # 2. Store audio in private isolated storage: private/smart/{sheet_id}/input.<ext>
    if audio_bytes:
        ext = audio_ext if audio_ext.startswith(".") else f".{audio_ext}"
        private_input = f"private/smart/{sheet_id}/input{ext}"
        store.put_bytes(private_input, audio_bytes)
        store.put_json(
            f"private/smart/{sheet_id}/meta.json",
            {"title": title, "artist": artist},
        )
        store.put_json(
            f"private/smart/{sheet_id}/status.json",
            {
                "status": "queued",
                "progress": 0.0,
                "stage": "等待计算节点启动",
                "updated_at": now_iso,
            },
        )

        # 3. Trigger Cloud Run Job
        if job_runner is not None:
            try:
                if callable(job_runner):
                    job_runner(sheet_id)
                else:
                    try:
                        runner_fn = job_runner.run
                        runner_fn(sheet_id)
                    except AttributeError:
                        pass
            except Exception as j_err:
                logger.warning("Custom job runner failed: %s", j_err)
        else:
            trigger_cloud_run_job(sheet_id)

    # 4. Launch background pipeline
    if background:
        t = threading.Thread(
            target=_run_smart_pipeline,
            args=(sheet_id, title, artist, bool(audio_bytes), store, poll_interval, poll_timeout),
            daemon=True,
        )
        t.start()
        return t
    else:
        _run_smart_pipeline(
            sheet_id, title, artist, bool(audio_bytes), store, poll_interval, poll_timeout
        )
        return None
