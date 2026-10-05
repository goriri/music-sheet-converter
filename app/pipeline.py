"""Pipeline logic: file ingestion, background OMR parsing, and score rendering."""
from __future__ import annotations

import io
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

from fastapi import HTTPException
from PIL import Image, ImageOps

from app.models import Arrangement, Difficulty, ParsedSheet, QualityIssue, SectionPlan
from app.storage import Storage, get_storage

logger = logging.getLogger(__name__)


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


def _do_parse_sheet(
    sheet_id: str,
    storage: Optional[Storage] = None,
    parse_fn: Optional[Any] = None,
) -> None:
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

        # Use passed-in parse_fn (e.g. mocked in tests) or import from OMR module
        if parse_fn is not None:
            parsed_sheet = parse_fn(page_images)
        else:
            from app.omr.gemini_omr import parse_pages
            parsed_sheet = parse_pages(page_images)

        if isinstance(parsed_sheet, dict):
            parsed_sheet = ParsedSheet.model_validate(parsed_sheet)

        state["progress"] = 0.85
        state["progress_text"] = "校验中"
        state["message"] = "校验中"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        store.put_json(state_path, state)

        # Wire OMR QA verify
        try:
            from app.qa.omr_verify import verify_sheet
            parsed_sheet = verify_sheet(page_images, parsed_sheet, use_llm=True)
            if isinstance(parsed_sheet, dict):
                parsed_sheet = ParsedSheet.model_validate(parsed_sheet)
        except Exception as qa_exc:
            from app.models import QualityIssue
            parsed_sheet.issues.append(
                QualityIssue(
                    stage="omr",
                    severity="info",
                    code="omr_qa_unavailable",
                    message="OMR自动校验服务暂不可用",
                    detail={"error": str(qa_exc)},
                )
            )

        # Save parsed sheet
        parsed_dict = parsed_sheet.model_dump()
        store.put_json(f"sheets/{sheet_id}/parsed.json", parsed_dict)

        # Mark ready
        state["status"] = "ready"
        state["progress"] = 1.0
        state["progress_text"] = "已完成"
        state["message"] = "已完成"
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
    parse_fn: Optional[Any] = None,
) -> Optional[threading.Thread]:
    """Start OMR parse for sheet_id.

    Runs in a background thread if background=True, else synchronously.
    """
    store = storage or get_storage()
    if parse_fn is None:
        try:
            from app.omr.gemini_omr import parse_pages
            parse_fn = parse_pages
        except Exception:
            parse_fn = None

    if background:
        thread = threading.Thread(
            target=_do_parse_sheet,
            args=(sheet_id, store, parse_fn),
            daemon=True,
            name=f"omr-parse-{sheet_id}",
        )
        thread.start()
        return thread
    else:
        _do_parse_sheet(sheet_id, store, parse_fn)
        return None


def append_qa_appendix_pdf(pdf_bytes: bytes, issues: list[Any]) -> bytes:
    """Create a white A4 PDF page listing QA issues and append to pdf_bytes using PyMuPDF."""
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1240, 1754  # A4 proportion
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    font_path = Path(__file__).parent / "render" / "fonts" / "NotoSansSC.ttf"
    try:
        title_font = ImageFont.truetype(str(font_path), 32)
        header_font = ImageFont.truetype(str(font_path), 20)
        body_font = ImageFont.truetype(str(font_path), 18)
        tag_font = ImageFont.truetype(str(font_path), 16)
    except Exception:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        tag_font = ImageFont.load_default()

    margin_x = 80
    y = 80

    draw.text((margin_x, y), "校验说明", fill="#0f172a", font=title_font)
    y += 50
    draw.text((margin_x, y), "以下为本乐谱自动识别与编配质量校验记录：", fill="#64748b", font=header_font)
    y += 40

    draw.line([(margin_x, y), (width - margin_x, y)], fill="#cbd5e1", width=2)
    y += 25

    for issue in issues:
        if y > height - 100:
            break
        severity = issue.severity if isinstance(issue, QualityIssue) else issue.get("severity", "")
        if severity == "needs_review":
            tag_text = "[需要确认]"
            tag_color = "#d97706"
        elif severity == "auto_fixed":
            tag_text = "[已自动修正]"
            tag_color = "#16a34a"
        else:
            tag_text = "[提示]"
            tag_color = "#2563eb"

        m_idx = issue.measure_index if isinstance(issue, QualityIssue) else issue.get("measure_index")
        m_info = f"第 {m_idx + 1} 小节" if m_idx is not None else "整曲"
        code = issue.code if isinstance(issue, QualityIssue) else issue.get("code", "")
        header_text = f"{tag_text}  {m_info}"
        draw.text((margin_x, y), header_text, fill=tag_color, font=tag_font)
        y += 28

        msg = issue.message if isinstance(issue, QualityIssue) else issue.get("message", "")
        draw.text((margin_x + 10, y), msg, fill="#1e293b", font=body_font)
        y += 36

    buf = io.BytesIO()
    img.save(buf, format="PDF")
    appendix_pdf_bytes = buf.getvalue()

    import pymupdf
    main_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    app_doc = pymupdf.open(stream=appendix_pdf_bytes, filetype="pdf")
    main_doc.insert_pdf(app_doc)
    return main_doc.tobytes()


def _is_structural(iss: QualityIssue | dict) -> bool:
    c = iss.code if isinstance(iss, QualityIssue) else iss.get("code", "")
    return (
        c == "barline_count_mismatch"
        or c.startswith("barline_")
        or c.startswith("structural_")
        or "measure_count" in c
    )


def parse_unparseable_note(note: str) -> tuple[int, str, str] | None:
    """Parse measure index, raw chord string, and failure reason from an UNPARSEABLE note."""
    if not note.startswith("UNPARSEABLE:"):
        return None
    m_meas = re.search(r"measure=(\d+)", note)
    if not m_meas:
        return None
    m_idx = int(m_meas.group(1))

    m_raw = re.search(r"raw=('(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"|\S+)", note)
    if m_raw:
        raw_token = m_raw.group(1)
        if (raw_token.startswith("'") and raw_token.endswith("'")) or (
            raw_token.startswith('"') and raw_token.endswith('"')
        ):
            try:
                import ast
                raw_val = ast.literal_eval(raw_token)
            except Exception:
                raw_val = raw_token[1:-1]
        else:
            raw_val = raw_token
    else:
        raw_val = ""

    m_reason = re.search(r"reason=(.*)", note)
    reason = m_reason.group(1).strip() if m_reason else ""
    return m_idx, raw_val, reason


def check_output_sanity(
    sheet: ParsedSheet,
    arrangement: Arrangement,
    start_key: str,
    sheet_id: str,
    store: Storage,
    trace_id: str,
) -> None:
    """Validate arrangement output sanity against the parsed sheet.

    1. Any UNPARSEABLE note in Arrangement.notes becomes a 'chord_unresolvable'
       needs_review issue on ParsedSheet, blocking render until fixed or confirmed.
    2. Global sanity gate: if >= 20% of measures with printed chords resolve to
       a chord with a different root than the printed chord implies, OR if >= 3
       UNPARSEABLE notes exist, intercept with HTTP 409 and log an error.
    """
    from app.theory.chords import resolve_chord
    from app.theory.keys import canonical_key_for_pc, key_name_to_pc
    from app.theory.stacked import infer_stacked_orientation

    # 1. Collect UNPARSEABLE notes
    unparseable_records: list[tuple[int, str, str]] = []
    if arrangement.notes:
        for n in arrangement.notes:
            if isinstance(n, str) and n.startswith("UNPARSEABLE:"):
                parsed = parse_unparseable_note(n)
                if parsed:
                    unparseable_records.append(parsed)

    # 2. Check root mismatches for measures carrying printed chords
    start_pc = key_name_to_pc(start_key)
    current_tonic_pc = start_pc
    current_key_name = canonical_key_for_pc(current_tonic_pc, start_key)
    notation = sheet.header.chord_notation if (sheet.header and sheet.header.chord_notation) else "number"
    printed_tonic_pc: Optional[int] = None

    if notation == "letter":
        orig_key = sheet.header.original_key if sheet.header else None
        if orig_key:
            try:
                printed_tonic_pc = key_name_to_pc(orig_key)
            except ValueError:
                printed_tonic_pc = start_pc
        else:
            printed_tonic_pc = start_pc

    key_changes_by_m: dict[int, list[int]] = {}
    for kc in sheet.key_changes:
        key_changes_by_m.setdefault(kc.at_measure, []).append(kc.semitones)

    stacked_res = infer_stacked_orientation(sheet)
    stacked_orientation = stacked_res.orientation

    arr_measures_by_idx = {
        m.measure_index: m
        for m in (arrangement.measures or [])
    }

    total_printed_measures = 0
    mismatched_measures = 0

    for m in sheet.measures():
        if m.index in key_changes_by_m:
            for shift in key_changes_by_m[m.index]:
                current_tonic_pc = (current_tonic_pc + shift) % 12
                current_key_name = canonical_key_for_pc(current_tonic_pc, start_key)

        if not m.chords:
            continue

        total_printed_measures += 1
        m_arr = arr_measures_by_idx.get(m.index)

        if not m_arr or not m_arr.chords:
            mismatched_measures += 1
            continue

        has_root_mismatch = False
        for cs in m.chords:
            try:
                expected_rc = resolve_chord(
                    cs.raw,
                    current_tonic_pc,
                    cs.beat,
                    current_key_name,
                    notation=notation,
                    printed_tonic_pc=printed_tonic_pc,
                    stacked=cs.stacked,
                    stacked_orientation=stacked_orientation,
                )
            except ValueError:
                # Unparseable chord, handled via unparseable_records
                continue

            arranged_c = next((c for c in m_arr.chords if abs(c.beat - cs.beat) < 0.01), None)
            if arranged_c is None:
                candidates = [c for c in m_arr.chords if c.beat <= cs.beat + 0.01]
                arranged_c = candidates[-1] if candidates else m_arr.chords[0]

            if expected_rc.root_pc != arranged_c.root_pc:
                has_root_mismatch = True
                break

        if has_root_mismatch:
            mismatched_measures += 1

    mismatch_ratio = (mismatched_measures / total_printed_measures) if total_printed_measures > 0 else 0.0

    # 3. Global sanity check: >= 20% root mismatch OR >= 3 unparseable notes
    if mismatch_ratio >= 0.20 or len(unparseable_records) >= 3:
        if unparseable_records:
            sheet_updated = False
            for m_idx, raw_val, reason in unparseable_records:
                exists = any(
                    iss.severity == "needs_review"
                    and iss.code == "chord_unresolvable"
                    and iss.measure_index == m_idx
                    for iss in sheet.issues
                )
                if not exists:
                    sheet.issues.append(
                        QualityIssue(
                            stage="arrange",
                            measure_index=m_idx,
                            severity="needs_review",
                            code="chord_unresolvable",
                            message=f"第{m_idx + 1}小节和弦「{raw_val}」无法识别，请修改",
                            detail={"raw": raw_val, "reason": reason, "measure_index": m_idx},
                        )
                    )
                    sheet_updated = True
            if sheet_updated:
                store.put_json(f"sheets/{sheet_id}/parsed.json", sheet.model_dump())

        logger.error(
            f"[{trace_id}] Global sanity check failed: mismatch_ratio={mismatch_ratio:.1%} "
            f"({mismatched_measures}/{total_printed_measures}), unparseable_count={len(unparseable_records)}"
        )
        raise HTTPException(status_code=409, detail="伴奏生成异常，已拦截")

    # 4. Handle unconfirmed unparseable notes (< 3)
    if unparseable_records:
        state_path = f"sheets/{sheet_id}/state.json"
        state_data = store.get_json(state_path) if store.exists(state_path) else {}
        confirmed_unres = set(state_data.get("confirmed_unresolvable", []))

        unconfirmed = []
        sheet_updated = False
        for m_idx, raw_val, reason in unparseable_records:
            is_confirmed = (m_idx in confirmed_unres) or any(
                f"第{m_idx + 1}小节和弦沿用前一和弦" in w for w in sheet.warnings
            )
            if not is_confirmed:
                unconfirmed.append((m_idx, raw_val, reason))
                exists = any(
                    iss.severity == "needs_review"
                    and iss.code == "chord_unresolvable"
                    and iss.measure_index == m_idx
                    for iss in sheet.issues
                )
                if not exists:
                    sheet.issues.append(
                        QualityIssue(
                            stage="arrange",
                            measure_index=m_idx,
                            severity="needs_review",
                            code="chord_unresolvable",
                            message=f"第{m_idx + 1}小节和弦「{raw_val}」无法识别，请修改",
                            detail={"raw": raw_val, "reason": reason, "measure_index": m_idx},
                        )
                    )
                    sheet_updated = True

        if sheet_updated:
            store.put_json(f"sheets/{sheet_id}/parsed.json", sheet.model_dump())

        if unconfirmed:
            first_m, first_raw, _ = unconfirmed[0]
            logger.warning(
                f"[{trace_id}] Output sanity check blocked: unconfirmed unresolvable chord at measure {first_m + 1} ({first_raw})"
            )
            raise HTTPException(
                status_code=409,
                detail=f"伴奏生成异常，已拦截：第{first_m + 1}小节和弦「{first_raw}」无法识别，请修改",
            )


def render(
    sheet_id: str,
    start_key: str,
    difficulty: Difficulty,
    instrument: str,
    storage: Optional[Storage] = None,
) -> dict[str, Any]:
    """Render accompaniment PDF and preview images.

    Supports 'piano' and 'ukulele'. Other instruments -> HTTP 400 '暂未支持'.
    """
    if instrument not in ("piano", "ukulele"):
        raise HTTPException(status_code=400, detail="暂未支持")

    store = storage or get_storage()

    parsed_path = f"sheets/{sheet_id}/parsed.json"
    if not store.exists(parsed_path):
        raise HTTPException(status_code=404, detail="Parsed sheet not found")

    parsed_dict = store.get_json(parsed_path)
    sheet = ParsedSheet.model_validate(parsed_dict)

    # Layout gate: if layout_confidence < 0.6 or > 25% of measures carry needs_review
    all_meas = sheet.measures()
    total_meas = len(all_meas)
    rev_meas = {
        i.measure_index for i in sheet.issues
        if (i.severity if isinstance(i, QualityIssue) else i.get("severity")) == "needs_review"
        and (i.measure_index if isinstance(i, QualityIssue) else i.get("measure_index")) is not None
    }
    rev_ratio = (len(rev_meas) / total_meas) if total_meas > 0 else 0.0
    is_low_conf = sheet.layout_confidence < 0.6
    is_high_rev = rev_ratio > 0.25

    unconfirmed_structural = [
        iss for iss in sheet.issues
        if (iss.severity if isinstance(iss, QualityIssue) else iss.get("severity")) == "needs_review"
        and _is_structural(iss)
    ]

    state_path = f"sheets/{sheet_id}/state.json"
    state_data = store.get_json(state_path) if store.exists(state_path) else {}
    structural_confirmed = state_data.get("structural_confirmed", False) or any(
        "已确认版面结构" in w or "structural_confirmed" in w for w in sheet.warnings
    )

    gate_triggered = False
    if is_low_conf and (unconfirmed_structural or not structural_confirmed):
        gate_triggered = True
    elif is_high_rev and unconfirmed_structural:
        gate_triggered = True

    if gate_triggered:
        reasons = []
        if is_low_conf:
            reasons.append(f"版面置信度过低 ({sheet.layout_confidence:.2f} < 0.6)")
        if is_high_rev:
            reasons.append(f"待核对小节比例过高 ({rev_ratio:.1%} > 25%)")
        if unconfirmed_structural:
            reasons.append(f"存在 {len(unconfirmed_structural)} 处未确认的版面结构问题")
        reasons_text = f"（{', '.join(reasons)}）" if reasons else ""
        raise HTTPException(
            status_code=409,
            detail=f"此谱版面识别不可靠，暂不生成{reasons_text}",
        )

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

    trace_id = uuid.uuid4().hex[:12]

    try:
        from app.render.overlay import render_pages, render_pdf

        if instrument == "piano":
            # Lazy import piano arrangement
            from app.arrange.piano import arrange

            arrangement = arrange(sheet, start_key, difficulty)

            # QA arrangement check & repair
            import os
            try:
                from app.qa.arrange_check import ArrangementQAError, check_and_repair
                try:
                    arrangement = check_and_repair(
                        sheet,
                        arrangement,
                        use_llm=os.environ.get("QA_LLM_REVIEW") == "1",
                    )
                except ArrangementQAError as qa_err:
                    raise HTTPException(status_code=500, detail=f"编配自动修复失败：{qa_err}")
            except HTTPException:
                raise
            except Exception as exc:
                if isinstance(arrangement, Arrangement):
                    arrangement.issues.append(
                        QualityIssue(
                            stage="arrange",
                            severity="info",
                            code="arrange_qa_unavailable",
                            message="编配自动校验服务暂不可用",
                            detail={"error": str(exc)},
                        )
                    )
        elif instrument == "ukulele":
            # 1. plan_sections cached at sheets/{id}/sections.json
            sections = None
            sections_path = f"sheets/{sheet_id}/sections.json"
            if store.exists(sections_path):
                try:
                    sec_data = store.get_json(sections_path)
                    sections = [SectionPlan.model_validate(s) for s in sec_data]
                except Exception as exc:
                    logger.warning(f"Failed to load cached sections from {sections_path}: {exc}")
                    sections = None

            if sections is None:
                try:
                    from app.arrange.sections import plan_sections
                    try:
                        sections = plan_sections(sheet)
                        if sections:
                            store.put_json(
                                sections_path,
                                [s.model_dump() if isinstance(s, SectionPlan) else s for s in sections],
                            )
                    except Exception as exc:
                        logger.warning(f"plan_sections failed: {exc}")
                        sections = None
                except ImportError:
                    sections = None

            # 2. arrange_ukulele
            from app.arrange.ukulele import arrange_ukulele
            arrangement = arrange_ukulele(sheet, start_key, difficulty, sections=sections)

            # 3. check_and_repair_uke (if importable)
            try:
                from app.qa.uke_check import check_and_repair_uke
                arrangement = check_and_repair_uke(sheet, arrangement)
            except ImportError:
                pass
            except Exception as exc:
                logger.warning(f"check_and_repair_uke failed: {exc}")
                if isinstance(arrangement, Arrangement):
                    arrangement.issues.append(
                        QualityIssue(
                            stage="arrange",
                            severity="info",
                            code="uke_qa_unavailable",
                            message="尤克里里编配自动校验服务暂不可用",
                            detail={"error": str(exc)},
                        )
                    )

        # Output sanity check gate
        check_output_sanity(
            sheet=sheet,
            arrangement=arrangement,
            start_key=start_key,
            sheet_id=sheet_id,
            store=store,
            trace_id=trace_id,
        )

        pdf_bytes = render_pdf(page_images, sheet, arrangement)
        preview_pages = render_pages(page_images, sheet, arrangement)

        # Append QA appendix page iff needs_review or auto_fixed issues exist
        all_issues = []
        if isinstance(sheet, ParsedSheet):
            all_issues.extend(sheet.issues)
        if isinstance(arrangement, Arrangement):
            all_issues.extend(arrangement.issues)

        relevant_issues = [
            i for i in all_issues
            if (i.severity if isinstance(i, QualityIssue) else i.get("severity", "")) in ("needs_review", "auto_fixed")
        ]
        if relevant_issues:
            try:
                pdf_bytes = append_qa_appendix_pdf(pdf_bytes, relevant_issues)
            except Exception:
                pass

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

        issues_data = [
            i.model_dump() if isinstance(i, QualityIssue) else i
            for i in all_issues
        ]

        return {
            "pdf_url": f"/api/files/{quote(pdf_path, safe='/')}",
            "preview_urls": preview_urls,
            "arrangement": arr_dict,
            "issues": issues_data,
            "capo": arrangement.capo or 0,
            "shape_key": arrangement.shape_key or start_key,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[{trace_id}] 伴奏生成处理异常: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"伴奏生成处理失败，请稍后重试或调整设置（错误跟踪号: {trace_id}）",
        )

