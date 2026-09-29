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

from app.models import Difficulty, ParsedSheet, QualityIssue
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


class ConfirmRequest(BaseModel):
    issue_code: Optional[str] = None
    measure_index: Optional[int] = None
    issue_index: Optional[int] = None
    action: Optional[str] = "confirm"
    chord: Optional[str] = None
    beat: Optional[float] = None
    measure_count: Optional[int] = None
    semitones: Optional[int] = None
    at_measure: Optional[int] = None
    note: Optional[str] = None


def _is_structural_issue(issue: QualityIssue | dict) -> bool:
    code = issue.code if isinstance(issue, QualityIssue) else issue.get("code", "")
    return (
        code == "barline_count_mismatch"
        or code.startswith("barline_")
        or code.startswith("structural_")
        or "measure_count" in code
    )


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

    # Re-validate chord grammar via app.theory.chords.parse_chord
    from app.theory.chords import parse_chord
    for m in parsed.measures():
        for c in m.chords:
            try:
                parse_chord(c.raw)
            except Exception as err:
                raise HTTPException(
                    status_code=422,
                    detail=f"小节 #{m.index + 1} 和弦语法错误：'{c.raw}' ({err})",
                )

    # When the user edits a chord, mark it confidence=1.0 and drop needs_review issues for that measure
    parsed_path = f"sheets/{sheet_id}/parsed.json"
    measures_edited: set[int] = set()

    if storage.exists(parsed_path):
        existing = storage.get_json(parsed_path)
        old_measures = {
            m["index"]: [c.get("raw") for c in m.get("chords", [])]
            for s in existing.get("systems", [])
            for m in s.get("measures", [])
        }
        for m in parsed.measures():
            old_raws = old_measures.get(m.index)
            new_raws = [c.raw for c in m.chords]
            if old_raws is None or old_raws != new_raws:
                measures_edited.add(m.index)
                for c in m.chords:
                    c.confidence = 1.0

    # Also check any chord that was marked with confidence=1.0
    for m in parsed.measures():
        if any(c.confidence == 1.0 for c in m.chords):
            measures_edited.add(m.index)

    # Drop needs_review chord issues for confirmed/edited measures (preserve structural issues)
    parsed.issues = [
        issue for issue in parsed.issues
        if not (
            issue.severity == "needs_review"
            and issue.measure_index in measures_edited
            and not _is_structural_issue(issue)
        )
    ]

    parsed_dict = parsed.model_dump()
    storage.put_json(parsed_path, parsed_dict)

    # Check unconfirmed structural issues to update state.json
    has_unconfirmed_structural = any(
        i.severity == "needs_review" and _is_structural_issue(i)
        for i in parsed.issues
    )
    state = storage.get_json(state_path) if storage.exists(state_path) else {}
    state["structural_confirmed"] = not has_unconfirmed_structural
    storage.put_json(state_path, state)

    return {"status": "ok", "parsed": parsed_dict}


@app.post("/api/sheets/{sheet_id}/confirm")
@app.put("/api/sheets/{sheet_id}/confirm")
def confirm_sheet_issue(sheet_id: str, request: ConfirmRequest):
    """Confirm or correct a QA review issue on the parsed sheet."""
    storage = get_storage()
    parsed_path = f"sheets/{sheet_id}/parsed.json"
    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(parsed_path):
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_id}' not found")

    from app.theory.chords import parse_chord
    if request.chord is not None:
        try:
            parse_chord(request.chord)
        except Exception as err:
            raise HTTPException(status_code=422, detail=f"和弦语法错误：'{request.chord}' ({err})")
    if request.measure_count is not None and request.measure_count <= 0:
        raise HTTPException(status_code=422, detail="小节数必须大于 0")
    if request.beat is not None and request.beat <= 0:
        raise HTTPException(status_code=422, detail="拍数必须大于 0")

    from app.models import ChordSymbol, KeyChange, QualityIssue
    parsed_dict = storage.get_json(parsed_path)
    sheet = ParsedSheet.model_validate(parsed_dict)
    state = storage.get_json(state_path) if storage.exists(state_path) else {}

    target_issues: list[QualityIssue] = []
    if request.issue_index is not None and 0 <= request.issue_index < len(sheet.issues):
        cand = sheet.issues[request.issue_index]
        if cand.severity == "needs_review":
            target_issues.append(cand)
    else:
        for iss in sheet.issues:
            if iss.severity != "needs_review":
                continue
            code_match = (request.issue_code is None) or (iss.code == request.issue_code)
            meas_match = (
                (request.measure_index is None)
                or (iss.measure_index == request.measure_index)
                or (iss.measure_index is None and request.issue_code == iss.code)
            )
            if code_match and meas_match:
                target_issues.append(iss)

    is_structural_req = (
        request.issue_code == "barline_count_mismatch"
        or (request.issue_code and request.issue_code.startswith("barline_"))
        or (request.issue_code and request.issue_code.startswith("structural_"))
        or (request.issue_code and "measure_count" in request.issue_code)
    )

    for iss in target_issues:
        code = iss.code
        # 1. Structural issues
        if code == "barline_count_mismatch" or code.startswith("barline_") or code.startswith("structural_") or "measure_count" in code:
            if request.action == "correct" or request.measure_count is not None:
                cnt = request.measure_count or 4
                note_text = request.note or f"小节数更正为 {cnt}"
                sheet.warnings.append(note_text)
                iss.severity = "auto_fixed"
                iss.detail["confirmed_with_correction"] = True
                iss.detail["corrected_measure_count"] = cnt
                iss.message = f"{iss.message} (已记录更正：{note_text})"
            else:
                iss.severity = "info"
                iss.detail["confirmed"] = True
        # 2. Chord issues
        elif code in ("chord_ambiguous", "chord_disagreement", "invalid_chord_grammar", "missing_chord_suspected") or "chord" in code:
            if request.chord is not None and iss.measure_index is not None:
                for m in sheet.measures():
                    if m.index == iss.measure_index:
                        if m.chords:
                            m.chords[0].raw = request.chord
                            m.chords[0].confidence = 1.0
                            if request.beat is not None:
                                m.chords[0].beat = request.beat
                        else:
                            m.chords.append(ChordSymbol(raw=request.chord, beat=request.beat or 1.0, confidence=1.0))
            elif iss.measure_index is not None:
                for m in sheet.measures():
                    if m.index == iss.measure_index:
                        for c in m.chords:
                            c.confidence = 1.0
            iss.severity = "info"
            iss.detail["confirmed"] = True
        # 3. Key change issues
        elif code.startswith("key_change"):
            at_m = (
                request.at_measure
                if request.at_measure is not None
                else (request.measure_index if request.measure_index is not None else (iss.measure_index if iss.measure_index is not None else 0))
            )
            semi = request.semitones if request.semitones is not None else 2
            raw_kc = request.note or f"转调 ({semi:+d})"
            existing_kc = next((kc for kc in sheet.key_changes if kc.at_measure == at_m), None)
            if existing_kc:
                existing_kc.semitones = semi
                existing_kc.raw = raw_kc
            else:
                sheet.key_changes.append(KeyChange(at_measure=at_m, raw=raw_kc, semitones=semi))
            iss.severity = "info"
            iss.detail["confirmed"] = True
        # 4. Melody / Beat issues
        elif code.startswith("melody_") or "beat" in code:
            if request.beat is not None and iss.measure_index is not None:
                for m in sheet.measures():
                    if m.index == iss.measure_index and m.chords:
                        m.chords[0].beat = request.beat
            iss.severity = "info"
            iss.detail["confirmed"] = True
        else:
            iss.severity = "info"
            iss.detail["confirmed"] = True

    # Drop confirmed issues from needs_review (preserve warnings and unconfirmed issues)
    sheet.issues = [
        iss for iss in sheet.issues
        if iss.severity == "warning" or not (
            iss.detail.get("confirmed")
            or iss.detail.get("confirmed_with_correction")
            or (iss in target_issues and iss.severity != "needs_review")
        )
    ]

    # Check structural confirmation
    has_unconfirmed_structural = any(
        i.severity == "needs_review" and (
            i.code == "barline_count_mismatch"
            or i.code.startswith("barline_")
            or i.code.startswith("structural_")
            or "measure_count" in i.code
        )
        for i in sheet.issues
    )
    if is_structural_req or not has_unconfirmed_structural:
        state["structural_confirmed"] = True
        if "用户已确认版面结构划分" not in sheet.warnings:
            sheet.warnings.append("用户已确认版面结构划分")

    out_dict = sheet.model_dump()
    storage.put_json(parsed_path, out_dict)
    storage.put_json(state_path, state)
    return {"status": "ok", "parsed": out_dict}


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
