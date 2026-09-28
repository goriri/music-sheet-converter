#!/usr/bin/env python3
"""Offline end-to-end rendering script.

Loads fixtures/omr_sample.json and page1.jpg, page2.jpg.
Runs arrange(...) for difficulties (beginner, intermediate, advanced) in keys F and G.
Renders multi-page PDF and preview PNGs into out/e2e/{key}_{difficulty}/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.models import ParsedSheet
from app.arrange.piano import arrange
from app.render.overlay import render_pages, render_pdf


def run_e2e():
    omr_sample_path = PROJECT_ROOT / "fixtures" / "omr_sample.json"
    page1_path = PROJECT_ROOT / "fixtures" / "pages" / "page1.jpg"
    page2_path = PROJECT_ROOT / "fixtures" / "pages" / "page2.jpg"

    assert omr_sample_path.is_file(), f"Missing {omr_sample_path}"
    assert page1_path.is_file(), f"Missing {page1_path}"
    assert page2_path.is_file(), f"Missing {page2_path}"

    with open(omr_sample_path, "r", encoding="utf-8") as f:
        sheet = ParsedSheet.model_validate_json(f.read())

    pages_bytes = [page1_path.read_bytes(), page2_path.read_bytes()]

    keys = ["F", "G"]
    difficulties = ["beginner", "intermediate", "advanced"]

    out_base = PROJECT_ROOT / "out" / "e2e"
    out_base.mkdir(parents=True, exist_ok=True)

    results = {}

    for key in keys:
        for diff in difficulties:
            combo = f"{key}_{diff}"
            combo_dir = out_base / combo
            combo_dir.mkdir(parents=True, exist_ok=True)

            print(f"=== Rendering {combo} (key={key}, diff={diff}) ===")
            arr = arrange(sheet, start_key=key, difficulty=diff)

            # Render pages as PNGs
            imgs = render_pages(pages_bytes, sheet, arr)
            saved_pngs = []
            for idx, img in enumerate(imgs):
                png_path = combo_dir / f"page_{idx + 1}.png"
                img.save(png_path, "PNG")
                saved_pngs.append(str(png_path))
                print(f"  Saved page {idx + 1}: {png_path} ({img.size[0]}x{img.size[1]})")

            # Render PDF
            pdf_bytes = render_pdf(pages_bytes, sheet, arr)
            pdf_path = combo_dir / "result.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"  Saved PDF: {pdf_path} ({len(pdf_bytes)} bytes)")

            assert pdf_bytes.startswith(b"%PDF"), f"{pdf_path} does not start with %PDF"
            results[combo] = {
                "pdf": str(pdf_path),
                "pngs": saved_pngs,
                "pages_count": len(imgs),
            }

    print("\nE2E Rendering completed successfully.")
    for combo, data in results.items():
        print(f"  {combo}: {data['pages_count']} pages, PDF: {data['pdf']}")


if __name__ == "__main__":
    run_e2e()
