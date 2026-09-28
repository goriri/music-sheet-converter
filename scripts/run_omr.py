"""CLI runner for Taiwanese band chart OMR.

Usage:
    python scripts/run_omr.py fixtures/pages/page1.jpg fixtures/pages/page2.jpg -o fixtures/omr_sample.json --debug-dir out/omr_debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.omr.gemini_omr import parse_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def draw_debug_overlay(
    image_bytes: bytes,
    page_idx: int,
    sheet,
    output_path: str,
) -> None:
    """Draw system (blue), measure (green), and chord (red) boxes on page image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not decode image for debug overlay: page %d", page_idx + 1)
        return

    h, w = img.shape[:2]

    # Colors in BGR
    COLOR_SYSTEM = (255, 100, 0)   # Blue
    COLOR_MEASURE = (0, 200, 0)   # Green
    COLOR_CHORD = (0, 0, 255)     # Red

    # Find systems for this page
    page_systems = [s for s in sheet.systems if s.page == page_idx]

    # 1. Draw Systems (Blue)
    for s in page_systems:
        x0 = int(s.bbox[0] * w)
        y0 = int(s.bbox[1] * h)
        x1 = int(s.bbox[2] * w)
        y1 = int(s.bbox[3] * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_SYSTEM, 2)
        if s.section_label:
            cv2.putText(
                img,
                s.section_label[:20],
                (x0 + 5, max(15, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_SYSTEM,
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Measures (Green)
        for m in s.measures:
            mx0 = int(m.bbox[0] * w)
            my0 = int(m.bbox[1] * h)
            mx1 = int(m.bbox[2] * w)
            my1 = int(m.bbox[3] * h)
            cv2.rectangle(img, (mx0, my0), (mx1, my1), COLOR_MEASURE, 2)
            cv2.putText(
                img,
                f"m{m.index}",
                (mx0 + 2, my0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_MEASURE,
                1,
                cv2.LINE_AA,
            )

            # 3. Draw Chords (Red)
            for c in m.chords:
                if c.bbox:
                    cx0 = int(c.bbox[0] * w)
                    cy0 = int(c.bbox[1] * h)
                    cx1 = int(c.bbox[2] * w)
                    cy1 = int(c.bbox[3] * h)
                    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), COLOR_CHORD, 2)
                    cv2.putText(
                        img,
                        c.raw,
                        (cx0, max(12, cy0 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        COLOR_CHORD,
                        1,
                        cv2.LINE_AA,
                    )

    # Key changes overlay
    for kc in sheet.key_changes:
        # Check if this key change belongs to a measure on this page
        for s in page_systems:
            for m in s.measures:
                if m.index == kc.at_measure:
                    kx0 = int(m.bbox[0] * w)
                    ky0 = int(m.bbox[1] * h)
                    cv2.putText(
                        img,
                        f"KEY: {kc.raw} (+{kc.semitones})",
                        (kx0, max(25, ky0 - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),  # Orange
                        2,
                        cv2.LINE_AA,
                    )

    cv2.imwrite(output_path, img)
    logger.info("Saved debug overlay to %s", output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemini OMR on Taiwanese band charts.")
    parser.add_argument("pages", nargs="+", help="Path to page images (jpg/png)")
    parser.add_argument("-o", "--output", default="fixtures/omr_sample.json", help="Output JSON path")
    parser.add_argument("--debug-dir", default=None, help="Directory to save debug images")
    parser.add_argument("--model", default=None, help="Gemini model override")
    args = parser.parse_args()

    page_paths = [Path(p) for p in args.pages]
    for p in page_paths:
        if not p.is_file():
            logger.error("Page file not found: %s", p)
            return 1

    logger.info("Reading %d page(s)...", len(page_paths))
    images: list[bytes] = [p.read_bytes() for p in page_paths]

    logger.info("Calling parse_pages...")
    sheet = parse_pages(images, model=args.model)

    # Save output JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved parsed sheet to %s", out_path)

    # Save debug overlays if requested
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_bytes in enumerate(images):
            debug_img_path = debug_dir / f"page_{idx + 1}_debug.jpg"
            draw_debug_overlay(img_bytes, idx, sheet, str(debug_img_path))

    # Print summary
    print("\n--- OMR Extraction Summary ---")
    print(f"Header: title='{sheet.header.title}', style='{sheet.header.style}', key='{sheet.header.original_key}', tempo={sheet.header.tempo_bpm}")
    print(f"Pages: {len(sheet.pages)}")
    print(f"Total systems: {len(sheet.systems)}")
    print(f"Total measures: {len(sheet.measures())}")
    print(f"Key changes: {len(sheet.key_changes)}")
    for kc in sheet.key_changes:
        print(f"  - at measure {kc.at_measure}: {kc.raw} (+{kc.semitones} semitones)")
    print(f"Warnings: {len(sheet.warnings)}")
    for w in sheet.warnings[:10]:
        print(f"  - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
