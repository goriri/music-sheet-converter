"""FastAPI web application for Taiwanese band chart to piano score conversion."""
from __future__ import annotations

import mimetypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.models import Difficulty, ParsedSheet
from app.pipeline import ingest_uploaded_files, render as pipeline_render, start_parse
from app.storage import get_storage

app = FastAPI(title="Music Sheet Converter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_FILES = 12


class RenderRequest(BaseModel):
    start_key: str
    difficulty: Difficulty = "intermediate"
    instrument: str = "piano"


@app.get("/healthz")
def healthz():
    """Health check endpoint for Cloud Run."""
    return {"status": "ok"}


@app.post("/api/sheets")
async def upload_sheets(
    files: Optional[list[UploadFile]] = File(default=None),
    files_bracket: Optional[list[UploadFile]] = File(default=None, alias="files[]"),
):
    """Upload up to 12 sheet images or PDFs (<= 20MB each) and start background OMR."""
    uploaded = files or files_bracket
    if not uploaded or len(uploaded) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(uploaded) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload limit exceeded: maximum {MAX_FILES} files allowed (received {len(uploaded)})",
        )

    file_data: list[tuple[str, bytes]] = []
    for file in uploaded:
        filename = file.filename or "upload"
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' for file '{filename}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' exceeds the 20MB limit ({len(content)} bytes)",
            )
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"File '{filename}' is empty",
            )
        file_data.append((filename, content))

    import uuid

    sheet_id = uuid.uuid4().hex[:12]
    storage = get_storage()

    try:
        ingest_uploaded_files(sheet_id, file_data, storage=storage)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded files: {exc}")

    start_parse(sheet_id, background=True, storage=storage)

    return {"sheet_id": sheet_id}


@app.get("/api/sheets/{sheet_id}")
def get_sheet_status(sheet_id: str):
    """Retrieve sheet parsing status and ParsedSheet when ready."""
    storage = get_storage()
    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(state_path):
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_id}' not found")

    try:
        state = storage.get_json(state_path)
    except Exception:
        import time
        time.sleep(0.02)
        state = storage.get_json(state_path)

    # Check for timeout if currently parsing (> 15 minutes)
    if state.get("status") == "parsing":
        updated_at_str = state.get("updated_at") or state.get("created_at")
        if updated_at_str:
            try:
                updated_at = datetime.fromisoformat(updated_at_str)
                # Ensure timezone aware
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - updated_at > timedelta(minutes=15):
                    state["status"] = "error"
                    state["error"] = "Processing timed out (>15 minutes)"
                    storage.put_json(state_path, state)
            except Exception:
                pass

    if state.get("status") == "ready":
        parsed_path = f"sheets/{sheet_id}/parsed.json"
        if storage.exists(parsed_path):
            state["parsed"] = storage.get_json(parsed_path)
        else:
            state["parsed"] = None
    else:
        state["parsed"] = None

    return state


@app.put("/api/sheets/{sheet_id}/parsed")
def update_parsed_sheet(sheet_id: str, parsed: ParsedSheet):
    """Save user-corrected ParsedSheet."""
    storage = get_storage()
    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(state_path):
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_id}' not found")

    parsed_dict = parsed.model_dump()
    storage.put_json(f"sheets/{sheet_id}/parsed.json", parsed_dict)
    return {"status": "ok", "parsed": parsed_dict}


@app.post("/api/sheets/{sheet_id}/render")
def render_sheet(sheet_id: str, request: RenderRequest):
    """Render accompaniment PDF and preview images."""
    if request.instrument != "piano":
        raise HTTPException(status_code=400, detail="暂未支持")

    storage = get_storage()
    parsed_path = f"sheets/{sheet_id}/parsed.json"
    if not storage.exists(parsed_path):
        raise HTTPException(
            status_code=404,
            detail="Parsed sheet not found. Please wait for parsing to finish or re-upload.",
        )

    return pipeline_render(
        sheet_id=sheet_id,
        start_key=request.start_key,
        difficulty=request.difficulty,
        instrument=request.instrument,
        storage=storage,
    )


@app.get("/api/pages/{sheet_id}/{n}")
def get_page_image(sheet_id: str, n: int):
    """Retrieve original preprocessed page image."""
    storage = get_storage()
    page_path = f"sheets/{sheet_id}/pages/page_{n}.png"
    if not storage.exists(page_path):
        page_path = f"sheets/{sheet_id}/page_{n}.png"

    if not storage.exists(page_path):
        # Check state.json
        state_path = f"sheets/{sheet_id}/state.json"
        if storage.exists(state_path):
            state = storage.get_json(state_path)
            pages = state.get("pages", [])
            if 0 <= n < len(pages) and storage.exists(pages[n]):
                page_path = pages[n]

    if not storage.exists(page_path):
        raise HTTPException(status_code=404, detail=f"Page {n} not found for sheet '{sheet_id}'")

    data = storage.get_bytes(page_path)
    return Response(content=data, media_type="image/png")


@app.get("/api/files/{path:path}")
def serve_file(path: str, request: Request):
    """Serve generated files strictly under sheets/ (rejects path traversal)."""
    raw_path = request.scope.get("raw_path", b"").decode("utf-8", errors="ignore")
    if ".." in raw_path or "%2e%2e" in raw_path.lower():
        raise HTTPException(status_code=400, detail="Invalid path or path traversal detected")

    clean_path = unquote(path).strip("/")

    # Security check: must reside inside sheets/ and must not contain directory traversal
    parts = clean_path.split("/")
    if parts[0] != "sheets" or ".." in parts or "." in parts:
        raise HTTPException(status_code=400, detail="Invalid path or path traversal detected")

    storage = get_storage()
    if not storage.exists(clean_path):
        raise HTTPException(status_code=404, detail="File not found")

    data = storage.get_bytes(clean_path)

    media_type, _ = mimetypes.guess_type(clean_path)
    if not media_type:
        if clean_path.endswith(".pdf"):
            media_type = "application/pdf"
        elif clean_path.endswith(".png"):
            media_type = "image/png"
        elif clean_path.endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        elif clean_path.endswith(".json"):
            media_type = "application/json"
        else:
            media_type = "application/octet-stream"

    headers = {}
    if clean_path.endswith(".pdf"):
        headers["Content-Disposition"] = "inline; filename=accompaniment.pdf"

    return Response(content=data, media_type=media_type, headers=headers)


# Mount static UI files at root
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
