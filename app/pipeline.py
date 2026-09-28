"""Pipeline logic: file ingestion, background OMR parsing, and score rendering."""
from __future__ import annotations

import io
import threading
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

from fastapi import HTTPException
from PIL import Image, ImageOps

from app.models import Difficulty, ParsedSheet
from app.storage import Storage, get_storage


def extract_pages_from_upload(filename: str, file_bytes: bytes) -> list[bytes]:
    """Extract page images (PNG bytes) from an uploaded image or PDF file.

    PDFs are split into pages at 200 dpi via PyMuPDF.
    Images are downscaled so the longer side <= 2400 px.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    page_pngs: list[bytes] = []

    if ext == "pdf":
        import pymupdf

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200)
            page_pngs.append(pix.tobytes("png"))
    else:
        # Standard image (jpg, png, webp)
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > 2400:
            scale = 2400.0 / max_dim
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Convert palette/alpha to RGBA if needed or RGB
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page_pngs.append(buf.getvalue())

    return page_pngs


def ingest_uploaded_files(
    sheet_id: str,
    files: list[tuple[str, bytes]],
    storage: Optional[Storage] = None,
) -> list[str]:
    """Extract page PNGs from all uploaded files and write to storage."""
    store = storage or get_storage()
    page_paths: list[str] = []

    all_page_bytes: list[bytes] = []
    for filename, data in files:
        pages = extract_pages_from_upload(filename, data)
        all_page_bytes.extend(pages)

    if not all_page_bytes:
        raise ValueError("No valid pages found in uploaded files")

    for idx, page_bytes in enumerate(all_page_bytes):
        path = f"sheets/{sheet_id}/pages/page_{idx}.png"
        store.put_bytes(path, page_bytes, content_type="image/png")
        page_paths.append(path)

    now_iso = datetime.now(timezone.utc).isoformat()
    state = {
        "sheet_id": sheet_id,
        "status": "queued",
        "progress": 0.0,
        "error": None,
        "page_count": len(page_paths),
        "pages": page_paths,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    store.put_json(f"sheets/{sheet_id}/state.json", state)
    return page_paths


def _do_parse_sheet(sheet_id: str, storage: Optional[Storage] = None) -> None:
    """Worker task that runs OMR on the sheet's page images."""
    store = storage or get_storage()
    state_path = f"sheets/{sheet_id}/state.json"

    try:
        state = store.get_json(state_path)
    except Exception:
        state = {
            "sheet_id": sheet_id,
            "status": "parsing",
            "progress": 0.1,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        # Mark as parsing
        state["status"] = "parsing"
        state["progress"] = 0.1
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)

        # Load page images
        page_paths = state.get("pages")
        if not page_paths:
            page_paths = [p for p in store.list(f"sheets/{sheet_id}/pages/") if p.endswith(".png")]
            state["pages"] = page_paths
            state["page_count"] = len(page_paths)

        if not page_paths:
            raise FileNotFoundError(f"No pages found for sheet {sheet_id}")

        page_images = [store.get_bytes(p) for p in page_paths]

        state["progress"] = 0.3
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)

        # Lazy import OMR module
        from app.omr.gemini_omr import parse_pages

        parsed_sheet = parse_pages(page_images)

        state["progress"] = 0.8
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)

        # Save parsed sheet
        if isinstance(parsed_sheet, dict):
            parsed_dict = parsed_sheet
        else:
            parsed_dict = parsed_sheet.model_dump()
        store.put_json(f"sheets/{sheet_id}/parsed.json", parsed_dict)

        # Mark ready
        state["status"] = "ready"
        state["progress"] = 1.0
        state["error"] = None
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)

    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)


def start_parse(
    sheet_id: str,
    background: bool = True,
    storage: Optional[Storage] = None,
) -> Optional[threading.Thread]:
    """Start OMR parse for sheet_id.

    Runs in a background thread if background=True, else synchronously.
    """
    store = storage or get_storage()
    if background:
        thread = threading.Thread(
            target=_do_parse_sheet,
            args=(sheet_id, store),
            daemon=True,
            name=f"omr-parse-{sheet_id}",
        )
        thread.start()
        return thread
    else:
        _do_parse_sheet(sheet_id, store)
        return None


def render(
    sheet_id: str,
    start_key: str,
    difficulty: Difficulty,
    instrument: str,
    storage: Optional[Storage] = None,
) -> dict[str, Any]:
    """Render accompaniment PDF and preview images.

    Instrument other than 'piano' -> HTTP 400 '暂未支持'.
    """
    if instrument != "piano":
        raise HTTPException(status_code=400, detail="暂未支持")

    store = storage or get_storage()

    parsed_path = f"sheets/{sheet_id}/parsed.json"
    if not store.exists(parsed_path):
        raise HTTPException(status_code=404, detail="Parsed sheet not found")

    parsed_dict = store.get_json(parsed_path)
    sheet = ParsedSheet.model_validate(parsed_dict)

    # Load page images
    state_path = f"sheets/{sheet_id}/state.json"
    page_paths: list[str] = []
    if store.exists(state_path):
        state = store.get_json(state_path)
        page_paths = state.get("pages", [])

    if not page_paths:
        page_paths = [p for p in store.list(f"sheets/{sheet_id}/pages/") if p.endswith(".png")]

    if not page_paths:
        raise HTTPException(status_code=404, detail=f"No page images found for sheet {sheet_id}")

    page_images = [store.get_bytes(p) for p in page_paths]

    # Lazy import arrangement & render modules
    from app.arrange.piano import arrange
    from app.render.overlay import render_pages, render_pdf

    arrangement = arrange(sheet, start_key, difficulty)
    pdf_bytes = render_pdf(page_images, sheet, arrangement)
    preview_pages = render_pages(page_images, sheet, arrangement)

    render_folder = f"sheets/{sheet_id}/renders/{start_key}_{difficulty}_{instrument}"

    # Store PDF
    pdf_path = f"{render_folder}/score.pdf"
    store.put_bytes(pdf_path, pdf_bytes, content_type="application/pdf")

    # Store preview images
    preview_urls: list[str] = []
    for idx, page in enumerate(preview_pages):
        if isinstance(page, Image.Image):
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        elif isinstance(page, bytes):
            img_bytes = page
        else:
            raise TypeError(f"Unsupported preview image type: {type(page)}")

        preview_path = f"{render_folder}/preview_{idx}.png"
        store.put_bytes(preview_path, img_bytes, content_type="image/png")
        preview_urls.append(f"/api/files/{quote(preview_path, safe='/')}")

    # Store arrangement JSON
    if isinstance(arrangement, dict):
        arr_dict = arrangement
    else:
        arr_dict = arrangement.model_dump()
    store.put_json(f"{render_folder}/arrangement.json", arr_dict)

    return {
        "pdf_url": f"/api/files/{quote(pdf_path, safe='/')}",
        "preview_urls": preview_urls,
        "arrangement": arr_dict,
    }
