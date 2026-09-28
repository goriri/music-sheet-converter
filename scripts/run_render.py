#!/usr/bin/env python3
"""CLI script to render accompaniment overlay onto sheet music pages.

Usage:
  python scripts/run_render.py --sheet fixtures/synthetic_sheet.json \
      --arr fixtures/synthetic_arrangement.json \
      --pages fixtures/pages/page1.jpg fixtures/pages/page2.jpg \
      -o out/render/
"""
import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Arrangement, ParsedSheet
from app.render import render_pages, render_pdf


def parse_args():
    parser = argparse.ArgumentParser(description="Render piano accompaniment onto Taiwanese band charts.")
    parser.add_argument("--sheet", required=True, help="Path to ParsedSheet JSON")
    parser.add_argument("--arr", required=True, help="Path to Arrangement JSON")
    parser.add_argument("--pages", nargs="+", required=True, help="Paths to page image files")
    parser.add_argument("-o", "--out", default="out/render/", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.sheet, "r", encoding="utf-8") as f:
        sheet = ParsedSheet.model_validate_json(f.read())

    with open(args.arr, "r", encoding="utf-8") as f:
        arrangement = Arrangement.model_validate_json(f.read())

    pages_bytes = []
    for p in args.pages:
        with open(p, "rb") as f:
            pages_bytes.append(f.read())

    os.makedirs(args.out, exist_ok=True)

    # Render pages as PIL images
    rendered_images = render_pages(pages_bytes, sheet, arrangement)
    print(f"Rendered {len(rendered_images)} page(s):")

    for idx, img in enumerate(rendered_images):
        png_path = os.path.join(args.out, f"preview_page_{idx + 1}.png")
        img.save(png_path, "PNG")
        print(f"  Saved preview PNG: {png_path} ({img.size[0]}x{img.size[1]})")

    # Render multi-page PDF
    pdf_bytes = render_pdf(pages_bytes, sheet, arrangement)
    pdf_path = os.path.join(args.out, "result.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"  Saved PDF: {pdf_path} ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    main()
