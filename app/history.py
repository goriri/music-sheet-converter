"""Public history indexing, metadata management, soft deletion, and backfill."""
from __future__ import annotations

import io
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

from fastapi import HTTPException
from PIL import Image, ImageOps

from app.models import ParsedSheet
from app.storage import LocalStorage, Storage

logger = logging.getLogger(__name__)

SHEET_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
RENDER_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_#+-]{1,64}$")

_BACKFILL_LOCK = threading.Lock()
_BACKFILL_IN_PROGRESS = False


def _preview_index(path: str) -> int:
    """Extract numeric page index from preview_N.png path."""
    match = re.search(r"preview_(\d+)\.png", path)
    return int(match.group(1)) if match else 0


def _lazy_backfill_worker(storage: Storage) -> None:
    """Worker task executed in background thread for lazy backfill."""
    global _BACKFILL_IN_PROGRESS
    _BACKFILL_IN_PROGRESS = True
    try:
        run_backfill(storage, dry_run=False)
    except Exception as exc:
        logger.error("Lazy backfill encountered error: %s", exc, exc_info=True)
    finally:
        _BACKFILL_IN_PROGRESS = False


def validate_sheet_id(sheet_id: str) -> None:
    """Validate sheet_id to prevent path traversal or injection."""
    if not sheet_id or not SHEET_ID_REGEX.match(sheet_id) or ".." in sheet_id:
        raise HTTPException(status_code=400, detail="Invalid sheet_id format")


def validate_render_name(render_name: str) -> None:
    """Validate render_name to prevent path traversal or injection."""
    if not render_name or not RENDER_NAME_REGEX.match(render_name) or ".." in render_name:
        raise HTTPException(status_code=400, detail="Invalid render_name format")


def make_inv_ts(dt: Optional[datetime] = None) -> str:
    """Return zero-padded 13-digit inverse timestamp (10**13 - epoch_ms).

    Smaller string values correspond to more recent timestamps, giving
    newest-first ordering in lexicographic sort.
    """
    if dt is None:
        epoch_ms = int(time.time() * 1000)
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        epoch_ms = int(dt.timestamp() * 1000)
    inv_ts = (10**13) - epoch_ms
    return f"{inv_ts:013d}"


def make_history_key(sheet_id: str, dt: Optional[datetime] = None) -> str:
    """Construct storage path for a sheet's history index entry."""
    return f"history/{make_inv_ts(dt)}_{sheet_id}.json"


def generate_thumbnail_bytes(page_bytes: bytes, target_width: int = 320) -> bytes:
    """Generate JPEG thumbnail (~320px wide, quality 80) from page image bytes."""
    img = Image.open(io.BytesIO(page_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")

    w, h = img.size
    if w > 0:
        new_w = target_width
        new_h = max(1, int(round(h * (target_width / w))))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def ensure_thumbnail(sheet_id: str, storage: Storage) -> Optional[str]:
    """Ensure thumbnail exists at sheets/{sheet_id}/thumb.jpg; generate lazily if missing."""
    thumb_path = f"sheets/{sheet_id}/thumb.jpg"
    if storage.exists(thumb_path):
        return thumb_path

    page_path = f"sheets/{sheet_id}/pages/page_0.png"
    if not storage.exists(page_path):
        page_path = f"sheets/{sheet_id}/page_0.png"
    if not storage.exists(page_path):
        try:
            state = storage.get_json(f"sheets/{sheet_id}/state.json")
            pages = state.get("pages", [])
            if pages and storage.exists(pages[0]):
                page_path = pages[0]
        except Exception:
            pass

    if storage.exists(page_path):
        try:
            page_bytes = storage.get_bytes(page_path)
            thumb_bytes = generate_thumbnail_bytes(page_bytes)
            storage.put_bytes(thumb_path, thumb_bytes, content_type="image/jpeg")
            return thumb_path
        except Exception as exc:
            logger.warning("Failed to generate thumbnail for sheet %s: %s", sheet_id, exc)

    return None


def extract_title(parsed: Any) -> str:
    """Extract song title from ParsedSheet or dict, falling back to lyrics or default."""
    if parsed is None:
        return "未命名曲谱"

    if isinstance(parsed, dict):
        header = parsed.get("header") or {}
        title = (header.get("title") or "").strip()
        if title:
            return title
        for sys in parsed.get("systems", []):
            for m in sys.get("measures", []):
                lyrics = (m.get("lyrics") or "").strip()
                if lyrics:
                    return lyrics
    elif isinstance(parsed, ParsedSheet):
        if parsed.header and parsed.header.title.strip():
            return parsed.header.title.strip()
        for m in parsed.measures():
            if m.lyrics and m.lyrics.strip():
                return m.lyrics.strip()

    return "未命名曲谱"


def create_history_entry(
    sheet_id: str,
    created_at: str,
    page_count: int,
    storage: Storage,
    thumb_path: Optional[str] = None,
    title: str = "未命名曲谱",
) -> str:
    """Write history/{inv_ts}_{sheet_id}.json index entry."""
    dt = None
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at)
        except Exception:
            dt = None

    history_key = make_history_key(sheet_id, dt)
    if not thumb_path:
        thumb_path = f"sheets/{sheet_id}/thumb.jpg"

    entry = {
        "sheet_id": sheet_id,
        "created_at": created_at,
        "title": title,
        "page_count": page_count,
        "thumb_path": thumb_path,
    }
    storage.put_json(history_key, entry)
    return history_key


def update_history_title(sheet_id: str, title: str, storage: Storage) -> None:
    """Update title in history entry for sheet_id."""
    state_path = f"sheets/{sheet_id}/state.json"
    history_key = None
    if storage.exists(state_path):
        try:
            state = storage.get_json(state_path)
            history_key = state.get("history_key")
        except Exception:
            pass

    if history_key and storage.exists(history_key):
        try:
            entry = storage.get_json(history_key)
            entry["title"] = title
            storage.put_json(history_key, entry)
            return
        except Exception as exc:
            logger.warning("Failed to update history title in %s: %s", history_key, exc)

    # Fallback: find any matching history entry
    try:
        for k in storage.list("history/"):
            if k.endswith(f"_{sheet_id}.json"):
                entry = storage.get_json(k)
                entry["title"] = title
                storage.put_json(k, entry)
                return
    except Exception as exc:
        logger.warning("Error searching history entry to update title for %s: %s", sheet_id, exc)


def write_render_meta(
    sheet_id: str,
    start_key: str,
    difficulty: str,
    instrument: str,
    capo: int,
    shape_key: str,
    pdf_url: str,
    preview_urls: list[str],
    needs_review_count: int,
    storage: Storage,
) -> dict[str, Any]:
    """Write {render_folder}/meta.json for a completed render."""
    folder_name = f"{start_key}_{difficulty}_{instrument}"
    render_folder = f"sheets/{sheet_id}/renders/{folder_name}"
    meta_path = f"{render_folder}/meta.json"
    meta = {
        "name": folder_name,
        "instrument": instrument,
        "difficulty": difficulty,
        "start_key": start_key,
        "capo": capo,
        "shape_key": shape_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pdf_url": pdf_url,
        "preview_urls": preview_urls,
        "needs_review_count": needs_review_count,
        "deleted": False,
    }
    storage.put_json(meta_path, meta)
    return meta


def derive_render_meta(sheet_id: str, folder_name: str, storage: Storage) -> Optional[dict[str, Any]]:
    """Derive render metadata for legacy renders without meta.json."""
    parts = folder_name.split("_")
    if len(parts) >= 3:
        start_key = parts[0]
        difficulty = parts[1]
        instrument = "_".join(parts[2:])
    else:
        start_key = parts[0] if parts else ""
        difficulty = parts[1] if len(parts) > 1 else ""
        instrument = "piano"

    render_folder = f"sheets/{sheet_id}/renders/{folder_name}"
    arr_path = f"{render_folder}/arrangement.json"
    capo = 0
    shape_key = start_key
    needs_review_count = 0
    if storage.exists(arr_path):
        try:
            arr = storage.get_json(arr_path)
            capo = arr.get("capo", 0)
            shape_key = arr.get("shape_key") or start_key
            issues = arr.get("issues", [])
            needs_review_count = sum(1 for i in issues if i.get("severity") == "needs_review")
        except Exception:
            pass

    pdf_path = f"{render_folder}/score.pdf"
    pdf_url = f"/api/files/{quote(pdf_path, safe='/')}" if storage.exists(pdf_path) else ""

    all_files = storage.list(f"{render_folder}/")
    preview_files = [f for f in all_files if "/preview_" in f and f.endswith(".png")]

    preview_files.sort(key=_preview_index)
    preview_urls = [f"/api/files/{quote(p, safe='/')}" for p in preview_files]

    created_at = None
    if isinstance(storage, LocalStorage):
        try:
            target = storage._resolve(pdf_path if storage.exists(pdf_path) else arr_path)
            if target.exists():
                created_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            pass

    if not created_at:
        try:
            st = storage.get_json(f"sheets/{sheet_id}/state.json")
            created_at = st.get("created_at")
        except Exception:
            pass

    if not created_at:
        created_at = datetime.now(timezone.utc).isoformat()

    return {
        "name": folder_name,
        "instrument": instrument,
        "difficulty": difficulty,
        "start_key": start_key,
        "capo": capo,
        "shape_key": shape_key,
        "created_at": created_at,
        "pdf_url": pdf_url,
        "preview_urls": preview_urls,
        "needs_review_count": needs_review_count,
        "deleted": False,
    }


def get_sheet_renders(sheet_id: str, storage: Storage) -> list[dict[str, Any]]:
    """Retrieve all non-deleted renders for a sheet, sorted newest first."""
    render_files = storage.list(f"sheets/{sheet_id}/renders/")
    render_folders: set[str] = set()
    for rf in render_files:
        parts = rf.split("/")
        if len(parts) >= 4 and parts[0] == "sheets" and parts[2] == "renders":
            render_folders.add(parts[3])

    renders: list[dict[str, Any]] = []
    for folder in render_folders:
        meta_path = f"sheets/{sheet_id}/renders/{folder}/meta.json"
        meta = None
        if storage.exists(meta_path):
            try:
                meta = storage.get_json(meta_path)
            except Exception as exc:
                logger.warning("Error reading meta for render %s: %s", meta_path, exc)

        if meta is None:
            meta = derive_render_meta(sheet_id, folder, storage)

        if meta and not meta.get("deleted"):
            renders.append(meta)

    # Sort newest first
    renders.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return renders


def list_history(
    storage: Storage,
    cursor: Optional[str] = None,
    limit: int = 20,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """List public history items newest-first with cursor pagination."""
    all_keys = [
        k for k in storage.list("history/")
        if k.endswith(".json") and not k.split("/")[-1].startswith("_")
    ]
    all_keys.sort()

    start_idx = 0
    if cursor:
        if cursor in all_keys:
            start_idx = all_keys.index(cursor) + 1
        else:
            import bisect
            start_idx = bisect.bisect_right(all_keys, cursor)

    def _build_item(key: str) -> Optional[dict[str, Any]]:
        try:
            entry = storage.get_json(key)
            sheet_id = entry.get("sheet_id")
            if not sheet_id:
                return None

            state_path = f"sheets/{sheet_id}/state.json"
            if not storage.exists(state_path):
                return None

            state = storage.get_json(state_path)
            if state.get("deleted"):
                # Clean up dangling history key
                try:
                    storage.delete(key)
                except Exception:
                    pass
                return None

            title = entry.get("title") or "未命名曲谱"
            if title == "未命名曲谱" and storage.exists(f"sheets/{sheet_id}/parsed.json"):
                try:
                    parsed = storage.get_json(f"sheets/{sheet_id}/parsed.json")
                    new_title = extract_title(parsed)
                    if new_title != "未命名曲谱":
                        title = new_title
                        entry["title"] = title
                        storage.put_json(key, entry)
                except Exception:
                    pass

            created_at = entry.get("created_at") or state.get("created_at")
            page_count = entry.get("page_count") or state.get("page_count", 0)
            status = state.get("status", "unknown")

            thumb_path = entry.get("thumb_path") or f"sheets/{sheet_id}/thumb.jpg"
            thumb_ok = storage.exists(thumb_path)
            if not thumb_ok:
                gen_thumb = ensure_thumbnail(sheet_id, storage)
                if gen_thumb:
                    thumb_path = gen_thumb
                    thumb_ok = True
                    entry["thumb_path"] = thumb_path
                    try:
                        storage.put_json(key, entry)
                    except Exception:
                        pass

            thumb_url = f"/api/files/{quote(thumb_path, safe='/')}" if thumb_ok else None

            # Calculate render_count (excluding soft-deleted renders)
            render_count = len(get_sheet_renders(sheet_id, storage))

            return {
                "sheet_id": sheet_id,
                "title": title,
                "created_at": created_at,
                "page_count": page_count,
                "thumb_url": thumb_url,
                "status": status,
                "render_count": render_count,
            }
        except Exception as exc:
            logger.warning("Error processing history entry %s: %s", key, exc)
            return None

    from concurrent.futures import ThreadPoolExecutor

    items: list[dict[str, Any]] = []
    last_processed_key = None
    idx = start_idx
    with ThreadPoolExecutor(max_workers=16) as pool:
        while idx < len(all_keys) and len(items) < limit:
            batch = all_keys[idx: idx + (limit - len(items))]
            results = list(pool.map(_build_item, batch))
            for key, item in zip(batch, results):
                last_processed_key = key
                if item is not None:
                    items.append(item)
            idx += len(batch)

    next_cursor = None
    if len(items) >= limit and last_processed_key:
        curr_pos = all_keys.index(last_processed_key)
        if curr_pos + 1 < len(all_keys):
            next_cursor = last_processed_key

    return items, next_cursor


def get_sheet_history(sheet_id: str, storage: Storage) -> dict[str, Any]:
    """Retrieve full history details for a single sheet."""
    validate_sheet_id(sheet_id)
    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(state_path):
        raise HTTPException(status_code=404, detail="Sheet not found")

    state = storage.get_json(state_path)
    if state.get("deleted"):
        raise HTTPException(status_code=404, detail="Sheet not found")

    title = "未命名曲谱"
    if storage.exists(f"sheets/{sheet_id}/parsed.json"):
        try:
            parsed = storage.get_json(f"sheets/{sheet_id}/parsed.json")
            title = extract_title(parsed)
        except Exception:
            pass
    elif state.get("history_key") and storage.exists(state["history_key"]):
        try:
            h = storage.get_json(state["history_key"])
            title = h.get("title") or title
        except Exception:
            pass

    thumb_path = f"sheets/{sheet_id}/thumb.jpg"
    if not storage.exists(thumb_path):
        ensure_thumbnail(sheet_id, storage)
    thumb_url = f"/api/files/{quote(thumb_path, safe='/')}" if storage.exists(thumb_path) else None

    renders = get_sheet_renders(sheet_id, storage)

    return {
        "sheet_id": sheet_id,
        "title": title,
        "created_at": state.get("created_at"),
        "page_count": state.get("page_count", 0),
        "thumb_url": thumb_url,
        "status": state.get("status", "unknown"),
        "renders": renders,
    }


def soft_delete_sheet(sheet_id: str, storage: Storage) -> dict[str, Any]:
    """Soft delete a sheet: remove history index entry and set deleted=true in state.json."""
    validate_sheet_id(sheet_id)
    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(state_path):
        raise HTTPException(status_code=404, detail="Sheet not found")

    state = storage.get_json(state_path)
    if state.get("deleted"):
        return {"status": "ok", "sheet_id": sheet_id}

    history_key = state.get("history_key")
    if history_key and storage.exists(history_key):
        try:
            storage.delete(history_key)
        except Exception as exc:
            logger.warning("Error removing history key %s: %s", history_key, exc)

    # Clean up any history entry referencing this sheet_id
    try:
        for k in storage.list("history/"):
            if k.endswith(f"_{sheet_id}.json"):
                storage.delete(k)
    except Exception:
        pass

    state["deleted"] = True
    state["deleted_at"] = datetime.now(timezone.utc).isoformat()
    storage.put_json(state_path, state)
    return {"status": "ok", "sheet_id": sheet_id}


def soft_delete_render(sheet_id: str, render_name: str, storage: Storage) -> dict[str, Any]:
    """Soft delete a single render by setting deleted=true in its meta.json."""
    validate_sheet_id(sheet_id)
    validate_render_name(render_name)

    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(state_path):
        raise HTTPException(status_code=404, detail="Sheet not found")

    render_folder = f"sheets/{sheet_id}/renders/{render_name}"
    meta_path = f"{render_folder}/meta.json"
    if storage.exists(meta_path):
        meta = storage.get_json(meta_path)
        meta["deleted"] = True
        meta["deleted_at"] = datetime.now(timezone.utc).isoformat()
        storage.put_json(meta_path, meta)
        return {"status": "ok", "render_name": render_name}

    # Check if render files exist
    files = storage.list(f"{render_folder}/")
    if not files:
        raise HTTPException(status_code=404, detail="Render not found")

    meta = derive_render_meta(sheet_id, render_name, storage)
    if not meta:
        raise HTTPException(status_code=404, detail="Render not found")

    meta["deleted"] = True
    meta["deleted_at"] = datetime.now(timezone.utc).isoformat()
    storage.put_json(meta_path, meta)
    return {"status": "ok", "render_name": render_name}


def restore_sheet(sheet_id: str, storage: Storage) -> bool:
    """Restore a soft-deleted sheet and re-add its history index entry."""
    validate_sheet_id(sheet_id)
    state_path = f"sheets/{sheet_id}/state.json"
    if not storage.exists(state_path):
        return False

    state = storage.get_json(state_path)
    state["deleted"] = False
    state.pop("deleted_at", None)

    created_at = state.get("created_at") or datetime.now(timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(created_at)
    except Exception:
        dt = datetime.now(timezone.utc)

    inv_ts = make_inv_ts(dt)
    history_key = f"history/{inv_ts}_{sheet_id}.json"

    title = "未命名曲谱"
    if storage.exists(f"sheets/{sheet_id}/parsed.json"):
        try:
            parsed = storage.get_json(f"sheets/{sheet_id}/parsed.json")
            title = extract_title(parsed)
        except Exception:
            pass

    thumb_path = f"sheets/{sheet_id}/thumb.jpg"
    if not storage.exists(thumb_path):
        ensure_thumbnail(sheet_id, storage)

    entry = {
        "sheet_id": sheet_id,
        "created_at": created_at,
        "title": title,
        "page_count": state.get("page_count", 0),
        "thumb_path": thumb_path,
    }
    storage.put_json(history_key, entry)
    state["history_key"] = history_key
    storage.put_json(state_path, state)
    return True


def run_backfill(
    storage: Storage,
    dry_run: bool = False,
    restore_id: Optional[str] = None,
) -> dict[str, int]:
    """Idempotently backfill missing history entries, thumbnails, and render meta."""
    with _BACKFILL_LOCK:
        if restore_id:
            ok = restore_sheet(restore_id, storage)
            return {"restored": 1 if ok else 0}

        stats = {
            "sheets_scanned": 0,
            "history_entries_created": 0,
            "thumbnails_created": 0,
            "render_meta_created": 0,
        }

        all_files = storage.list("sheets/")
        sheet_ids = set()
        for f in all_files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0] == "sheets":
                sheet_ids.add(parts[1])

        for s_id in sorted(sheet_ids):
            state_path = f"sheets/{s_id}/state.json"
            if not storage.exists(state_path):
                continue

            try:
                state = storage.get_json(state_path)
            except Exception:
                continue

            if state.get("deleted"):
                continue

            stats["sheets_scanned"] += 1

            # 1. Thumbnail
            thumb_path = f"sheets/{s_id}/thumb.jpg"
            if not storage.exists(thumb_path):
                if not dry_run:
                    gen = ensure_thumbnail(s_id, storage)
                    if gen:
                        stats["thumbnails_created"] += 1
                else:
                    stats["thumbnails_created"] += 1

            # 2. History entry
            has_history = False
            if state.get("history_key") and storage.exists(state["history_key"]):
                has_history = True
            else:
                # Check if any history entry matches this sheet
                for k in storage.list("history/"):
                    if k.endswith(f"_{s_id}.json"):
                        has_history = True
                        if not state.get("history_key"):
                            state["history_key"] = k
                            if not dry_run:
                                storage.put_json(state_path, state)
                        break

            if not has_history:
                created_at = state.get("created_at") or datetime.now(timezone.utc).isoformat()
                title = "未命名曲谱"
                if storage.exists(f"sheets/{s_id}/parsed.json"):
                    try:
                        parsed = storage.get_json(f"sheets/{s_id}/parsed.json")
                        title = extract_title(parsed)
                    except Exception:
                        pass

                if not dry_run:
                    h_key = create_history_entry(
                        sheet_id=s_id,
                        created_at=created_at,
                        page_count=state.get("page_count", 0),
                        storage=storage,
                        thumb_path=thumb_path,
                        title=title,
                    )
                    state["history_key"] = h_key
                    storage.put_json(state_path, state)
                stats["history_entries_created"] += 1

            # 3. Render metadata
            render_files = storage.list(f"sheets/{s_id}/renders/")
            r_folders: set[str] = set()
            for rf in render_files:
                parts = rf.split("/")
                if len(parts) >= 4 and parts[0] == "sheets" and parts[2] == "renders":
                    r_folders.add(parts[3])

            for rfld in r_folders:
                meta_path = f"sheets/{s_id}/renders/{rfld}/meta.json"
                if not storage.exists(meta_path):
                    derived = derive_render_meta(s_id, rfld, storage)
                    if derived and not dry_run:
                        storage.put_json(meta_path, derived)
                    stats["render_meta_created"] += 1

        if not dry_run:
            storage.put_json("history/_backfill_done", {
                "done_at": datetime.now(timezone.utc).isoformat(),
                "stats": stats,
            })

        return stats


def ensure_lazy_backfill(storage: Storage) -> None:
    """Trigger one-time background backfill if history/_backfill_done is missing."""
    if storage.exists("history/_backfill_done"):
        return

    global _BACKFILL_IN_PROGRESS
    with _BACKFILL_LOCK:
        if storage.exists("history/_backfill_done") or _BACKFILL_IN_PROGRESS:
            return

        thread = threading.Thread(
            target=_lazy_backfill_worker,
            args=(storage,),
            daemon=True,
            name="history-lazy-backfill",
        )
        thread.start()
