"""Remote end-to-end verification script against Cloud Run."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
import pymupdf

def main():
    base_url = "https://sheet-converter-766918001064.asia-east1.run.app"
    sheet_id = "c8c3d5be94a2"
    out_dir = Path("out/remote")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. GET sheet and inspect issues
    sheet_url = f"{base_url}/api/sheets/{sheet_id}"
    print(f"Fetching sheet from {sheet_url}...")
    with urllib.request.urlopen(sheet_url) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    parsed = data.get("parsed", {})
    issues = parsed.get("issues", [])
    print(f"Total issues reported: {len(issues)}")

    severity_counts: dict[str, int] = {}
    needs_review_msgs: list[str] = []
    for iss in issues:
        sev = iss.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        if sev == "needs_review":
            code = iss.get("code")
            msg = iss.get("message")
            needs_review_msgs.append(f"[{code}] {msg}")

    print("Issues by severity:", severity_counts)
    print("needs_review messages:")
    for msg in needs_review_msgs:
        print(f"  * {msg}")

    # 2. Render F / intermediate
    print("\n--- Rendering F / intermediate ---")
    render_url = f"{base_url}/api/sheets/{sheet_id}/render"
    payload_f = json.dumps({"start_key": "F", "difficulty": "intermediate", "instrument": "piano"}).encode("utf-8")
    req_f = urllib.request.Request(render_url, data=payload_f, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_f) as resp:
        res_f = json.loads(resp.read().decode("utf-8"))

    pdf_path_f = out_dir / "F_intermediate.pdf"
    full_pdf_url_f = f"{base_url}{res_f['pdf_url']}"
    print(f"Downloading F/intermediate PDF from {full_pdf_url_f}...")
    with urllib.request.urlopen(full_pdf_url_f) as resp:
        pdf_bytes_f = resp.read()
        pdf_path_f.write_bytes(pdf_bytes_f)

    doc_f = pymupdf.open(stream=pdf_bytes_f, filetype="pdf")
    print(f"F/intermediate PDF: size={len(pdf_bytes_f)} bytes, pages={len(doc_f)}, %PDF={pdf_bytes_f.startswith(b'%PDF')}")

    # Convert page 1 to out/remote/p1.png
    doc_f[0].get_pixmap(dpi=150).save(str(out_dir / "p1.png"))
    print("Saved out/remote/p1.png")

    # Find the page containing key change (bar 51)
    # Let's inspect page 3 / 4 to see which has bar 51
    # Page count is typically 5 (4 music pages + 1 appendix) or similar
    # In earlier e2e_render, input page 2 (measures 40..76) was split into output pages 3 and 4.
    # Bar 50/51 was in system 12 on output page 3!
    # Let's save page 3 as pKC.png (0-indexed page 2)
    doc_f[2].get_pixmap(dpi=150).save(str(out_dir / "pKC.png"))
    print("Saved out/remote/pKC.png (PDF page 3)")

    # Appendix page is the last page
    doc_f[-1].get_pixmap(dpi=150).save(str(out_dir / "appendix.png"))
    print(f"Saved out/remote/appendix.png (PDF page {len(doc_f)})")

    # 3. Render G / advanced
    print("\n--- Rendering G / advanced ---")
    payload_g = json.dumps({"start_key": "G", "difficulty": "advanced", "instrument": "piano"}).encode("utf-8")
    req_g = urllib.request.Request(render_url, data=payload_g, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req_g) as resp:
        res_g = json.loads(resp.read().decode("utf-8"))

    pdf_path_g = out_dir / "G_advanced.pdf"
    full_pdf_url_g = f"{base_url}{res_g['pdf_url']}"
    print(f"Downloading G/advanced PDF from {full_pdf_url_g}...")
    with urllib.request.urlopen(full_pdf_url_g) as resp:
        pdf_bytes_g = resp.read()
        pdf_path_g.write_bytes(pdf_bytes_g)

    doc_g = pymupdf.open(stream=pdf_bytes_g, filetype="pdf")
    print(f"G/advanced PDF: size={len(pdf_bytes_g)} bytes, pages={len(doc_g)}, %PDF={pdf_bytes_g.startswith(b'%PDF')}")

if __name__ == "__main__":
    main()
