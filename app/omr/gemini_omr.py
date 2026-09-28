"""Gemini-based Optical Music Recognition for Taiwanese band charts.

Calls Gemini on Vertex AI with structured JSON output, stitches pages,
converts bounding boxes, and refines measure boundaries using OpenCV.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import os
import time
from typing import Optional
from PIL import Image
from google import genai
from google.genai import types

from app.models import (
    ChordSymbol,
    KeyChange,
    Measure,
    PageInfo,
    ParsedSheet,
    SongHeader,
    System,
)
from app.omr.barlines import refine_measure_boxes
from app.omr.prompts import (
    PageOMRGemini,
    SYSTEM_PROMPT,
    build_page_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_OMR_MODEL = "gemini-2.5-pro"
DEFAULT_PROJECT = "cellular-cider-495602-r9"
LOCATION = "global"


def get_client(project: Optional[str] = None) -> genai.Client:
    """Create and return a Vertex AI Gemini client."""
    proj = project or os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    return genai.Client(vertexai=True, project=proj, location=LOCATION)


def box_2d_to_bbox(
    box: Optional[list[int]],
) -> Optional[tuple[float, float, float, float]]:
    """Convert Gemini box_2d [ymin, xmin, ymax, xmax] in 0..1000 to normalized (x0, y0, x1, y1) in 0..1."""
    if not box or len(box) < 4:
        return None
    ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
    x0 = max(0.0, min(1.0, round(float(xmin) / 1000.0, 4)))
    y0 = max(0.0, min(1.0, round(float(ymin) / 1000.0, 4)))
    x1 = max(0.0, min(1.0, round(float(xmax) / 1000.0, 4)))
    y1 = max(0.0, min(1.0, round(float(ymax) / 1000.0, 4)))
    # Ensure x0 <= x1 and y0 <= y1
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def parse_single_page(
    client: genai.Client,
    model: str,
    img_bytes: bytes,
    page_idx: int,
    total_pages: int,
    max_retries: int = 3,
) -> PageOMRGemini:
    """Parse a single page image using Gemini with retries."""
    prompt = build_page_prompt(page_num=page_idx + 1, total_pages=total_pages)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            logger.info("Parsing page %d/%d (attempt %d)...", page_idx + 1, total_pages, attempt + 1)
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=PageOMRGemini,
                    temperature=0.0,
                ),
            )
            raw_text = resp.text.strip() if resp.text else "{}"
            parsed = PageOMRGemini.model_validate_json(raw_text)
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning("Error parsing page %d (attempt %d/%d): %s", page_idx + 1, attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                time.sleep(2.0 ** attempt)

    logger.error("Failed to parse page %d after %d retries: %s", page_idx + 1, max_retries, last_error)
    return PageOMRGemini(
        warnings=[f"Failed to parse page {page_idx + 1}: {last_error}"]
    )


def parse_pages(
    images: list[bytes],
    model: Optional[str] = None,
    project: Optional[str] = None,
) -> ParsedSheet:
    """Parse multiple band chart pages and stitch into a single ParsedSheet.

    Args:
        images: List of image file bytes, one per page.
        model: Optional Gemini model name override (defaults to env OMR_MODEL or gemini-2.5-pro).
        project: Optional GCP project ID override.

    Returns:
        ParsedSheet containing stitched systems, global measure indices, and key changes.
    """
    omr_model = model or os.environ.get("OMR_MODEL", DEFAULT_OMR_MODEL)
    client = get_client(project=project)
    total_pages = len(images)

    # 1. Inspect image dimensions using Pillow
    pages_info: list[PageInfo] = []
    for idx, img_bytes in enumerate(images):
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                w, h = pil_img.size
                pages_info.append(PageInfo(width=w, height=h))
        except Exception as exc:
            logger.warning("Could not read image dimensions for page %d: %s", idx + 1, exc)
            pages_info.append(PageInfo(width=1000, height=1400))

    # 2. Parse pages concurrently with a thread pool
    page_results: list[PageOMRGemini] = [None] * total_pages  # type: ignore
    max_workers = min(total_pages, 4) if total_pages > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                parse_single_page, client, omr_model, images[idx], idx, total_pages
            ): idx
            for idx in range(total_pages)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                page_results[idx] = future.result()
            except Exception as exc:
                logger.error("Exception from page %d task: %s", idx + 1, exc)
                page_results[idx] = PageOMRGemini(
                    warnings=[f"Page {idx + 1} processing failed: {exc}"]
                )

    # 3. Stitch pages, maintain global measure indices, and extract header
    header = SongHeader()
    key_changes: list[KeyChange] = []
    all_systems: list[System] = []
    all_warnings: list[str] = []

    global_measure_idx = 0

    for page_idx, page_omr in enumerate(page_results):
        page_start_measure_idx = global_measure_idx

        # Collect warnings
        all_warnings.extend(page_omr.warnings)

        # Header: take from the first page that has one
        if page_omr.header and not header.title:
            h = page_omr.header
            header = SongHeader(
                title=h.title,
                style=h.style,
                time_signature=h.time_signature or "4/4",
                tempo_bpm=h.tempo_bpm,
                original_key=h.original_key,
                male_key=h.male_key,
                female_key=h.female_key,
                raw=h.raw,
            )

        # Systems and measures
        for sys_gemini in page_omr.systems:
            sys_bbox = box_2d_to_bbox(sys_gemini.box_2d) or (0.05, 0.1, 0.95, 0.2)
            measures_list: list[Measure] = []

            for m_gemini in sys_gemini.measures:
                m_bbox = box_2d_to_bbox(m_gemini.box_2d) or (
                    sys_bbox[0],
                    sys_bbox[1] + 0.2 * (sys_bbox[3] - sys_bbox[1]),
                    sys_bbox[2],
                    sys_bbox[3] - 0.2 * (sys_bbox[3] - sys_bbox[1]),
                )

                chords_list: list[ChordSymbol] = []
                for c_gemini in m_gemini.chords:
                    c_bbox = box_2d_to_bbox(c_gemini.box_2d)
                    chords_list.append(
                        ChordSymbol(
                            raw=c_gemini.raw,
                            beat=c_gemini.beat,
                            bbox=c_bbox,
                        )
                    )

                measure = Measure(
                    index=global_measure_idx,
                    bbox=m_bbox,
                    beats=m_gemini.beats,
                    chords=chords_list,
                    melody=m_gemini.melody,
                    lyrics=m_gemini.lyrics,
                    bass_hint=m_gemini.bass_hint,
                    rhythm_hint=m_gemini.rhythm_hint,
                    fill=m_gemini.fill,
                    is_stop=m_gemini.is_stop,
                )
                measures_list.append(measure)
                global_measure_idx += 1

            system = System(
                page=page_idx,
                bbox=sys_bbox,
                section_label=sys_gemini.section_label,
                measures=measures_list,
            )
            all_systems.append(system)

        # Key changes: translate page-local measure index to global measure index
        for kc in page_omr.key_changes:
            global_at = page_start_measure_idx + kc.at_measure
            key_changes.append(
                KeyChange(
                    at_measure=global_at,
                    raw=kc.raw,
                    semitones=kc.semitones,
                )
            )

    sheet = ParsedSheet(
        header=header,
        pages=pages_info,
        systems=all_systems,
        key_changes=key_changes,
        warnings=all_warnings,
    )

    # 4. Refine measure boxes using OpenCV barline detection
    for page_idx, img_bytes in enumerate(images):
        refine_measure_boxes(img_bytes, sheet, page=page_idx)

    return sheet
