"""Playwright-based UI screenshot generator for review cards and layout gate."""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from playwright.sync_api import sync_playwright

from app.models import QualityIssue


def find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Poll URL until server responds or timeout expires."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def _to_png(jpg_data: bytes) -> bytes:
    """Convert JPEG bytes to PNG bytes."""
    img = Image.open(io.BytesIO(jpg_data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def setup_test_sheets(storage_dir: Path) -> tuple[str, str]:
    """Prepare diaole_qa and low_conf_gate sheets in local storage."""
    from app.storage import LocalStorage

    store = LocalStorage(storage_dir)

    # 1. Convert page1.jpg and page2.jpg to PNG bytes
    p1_jpg = Path("fixtures/pages/page1.jpg").read_bytes()
    p2_jpg = Path("fixtures/pages/page2.jpg").read_bytes()

    p1_png = _to_png(p1_jpg)
    p2_png = _to_png(p2_jpg)

    with open("fixtures/omr_sample.json") as f:
        base_omr = json.load(f)

    # --- Sheet A: 掉了 with one issue of each kind ---
    sheet_a_id = "diaole_qa"
    store.put_bytes(f"sheets/{sheet_a_id}/pages/page_0.png", p1_png, "image/png")
    store.put_bytes(f"sheets/{sheet_a_id}/pages/page_1.png", p2_png, "image/png")

    parsed_a = dict(base_omr)
    parsed_a["layout_confidence"] = 0.95
    # Ensure measure 5 chord bbox matches real printed chord [5/7] on page 1
    for sys_item in parsed_a.get("systems", []):
        for m in sys_item.get("measures", []):
            if m.get("index") == 5:
                for c in m.get("chords", []):
                    if c.get("raw") == "5/7":
                        c["bbox"] = [0.2727, 0.2175, 0.3191, 0.2304]

    parsed_a["issues"] = [
        # 1. Structural issue: barline_count_mismatch on page 1 row 10 / measure 37
        QualityIssue(
            stage="omr",
            measure_index=37,
            severity="needs_review",
            code="barline_count_mismatch",
            message="第1页第10行小节线检测数量存疑（当前检测到 4 个小节），请核对小节划分",
            detail={"page": 0, "system_index": 9, "detected": 4, "expected": 4, "transcribed_count": 4},
        ).model_dump(),
        # 2. Chord low-confidence issue with 3 alternatives
        QualityIssue(
            stage="omr",
            measure_index=5,
            severity="needs_review",
            code="chord_ambiguous",
            message="第6小节和弦置信度偏低（当前：5/7，备选：5, 5m, 1/7），请核对原谱",
            detail={"raw": "5/7", "beat": 1.0, "chord_index": 0, "alternatives": ["5", "5m", "1/7"]},
        ).model_dump(),
        # 3. Key change issue
        QualityIssue(
            stage="omr",
            measure_index=None,
            severity="needs_review",
            code="key_change_unlocated",
            message="谱头显示有转调但未能定位，请设置转调小节与半音差",
            detail={"expected_semitones": 2, "detected_semitones": 0},
        ).model_dump(),
        # 4. Melody beat issue
        QualityIssue(
            stage="omr",
            measure_index=20,
            severity="needs_review",
            code="melody_beat_sum_mismatch",
            message="第21小节旋律拍数（2.0拍）与小节拍数（4.0拍）不符且和弦起始拍存疑，请核对",
            detail={"melody": "3 5 1 -", "beat_sum": 2.0, "expected_beats": 4.0},
        ).model_dump(),
    ]
    store.put_json(f"sheets/{sheet_a_id}/parsed.json", parsed_a)
    store.put_json(
        f"sheets/{sheet_a_id}/state.json",
        {
            "sheet_id": sheet_a_id,
            "status": "ready",
            "page_count": 2,
            "pages": [f"sheets/{sheet_a_id}/pages/page_0.png", f"sheets/{sheet_a_id}/pages/page_1.png"],
            "created_at": "2026-09-28T12:00:00Z",
            "structural_confirmed": False,
        },
    )

    # --- Sheet B: Low layout confidence gate sheet ---
    sheet_b_id = "low_conf_gate"
    store.put_bytes(f"sheets/{sheet_b_id}/pages/page_0.png", p1_png, "image/png")
    store.put_bytes(f"sheets/{sheet_b_id}/pages/page_1.png", p2_png, "image/png")

    parsed_b = dict(base_omr)
    parsed_b["layout_confidence"] = 0.42
    parsed_b["issues"] = [
        QualityIssue(
            stage="omr",
            measure_index=0,
            severity="needs_review",
            code="barline_count_mismatch",
            message="第1页第1行小节线检测数量不符，请核对小节划分",
            detail={"page": 0, "system_index": 0, "detected": 3, "expected": 4},
        ).model_dump()
    ]
    store.put_json(f"sheets/{sheet_b_id}/parsed.json", parsed_b)
    store.put_json(
        f"sheets/{sheet_b_id}/state.json",
        {
            "sheet_id": sheet_b_id,
            "status": "ready",
            "page_count": 2,
            "pages": [f"sheets/{sheet_b_id}/pages/page_0.png", f"sheets/{sheet_b_id}/pages/page_1.png"],
            "created_at": "2026-09-28T12:00:00Z",
            "structural_confirmed": False,
        },
    )

    return sheet_a_id, sheet_b_id


def capture_screenshots(port: int, sheet_a: str, sheet_b: str, out_dir: Path) -> list[str]:
    """Capture screenshots using Playwright at desktop and mobile resolutions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot_paths: list[str] = []

    devices = [
        {
            "name": "desktop",
            "viewport": {"width": 1440, "height": 900},
            "is_mobile": False,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        {
            "name": "mobile",
            "viewport": {"width": 390, "height": 844},
            "is_mobile": True,
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
        },
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for dev in devices:
            dev_name = dev["name"]
            context = browser.new_context(
                viewport=dev["viewport"],
                is_mobile=dev["is_mobile"],
                user_agent=dev["user_agent"],
                device_scale_factor=2,  # Retina crisp screenshots
            )
            page = context.new_page()
            page.on("console", lambda msg: print(f"[{dev_name} Console] {msg.type}: {msg.text}"))
            page.on("pageerror", lambda err: print(f"[{dev_name} PageError] {err}"))

            # -------------------------------------------------------------
            # Part 1: Diaole Sheet with 4 review cards + Lightbox
            # -------------------------------------------------------------
            page.goto(f"http://127.0.0.1:{port}/?sheet_id={sheet_a}", wait_until="networkidle")
            page.wait_for_selector("#qa-needs-review-list .qa-review-card", timeout=8000)
            page.wait_for_timeout(1500)  # Wait for canvases to complete drawImage

            # 1. Full page overview
            p_full = str(out_dir / f"{dev_name}_full_review.png")
            page.screenshot(path=p_full, full_page=True)
            screenshot_paths.append(p_full)

            # 2. Structural Card
            struct_card = page.locator(".qa-review-card.structural")
            if struct_card.count() > 0:
                struct_card.first.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                # Verify number input default derives from targetSystem measures (4)
                num_input = struct_card.first.locator("input[type=number]").first
                assert num_input.input_value() == "4", f"Expected numInput to be '4', got {num_input.input_value()}"
                p_struct = str(out_dir / f"{dev_name}_card_structural.png")
                struct_card.first.screenshot(path=p_struct)
                screenshot_paths.append(p_struct)

            # 3. Chord Card
            chord_card = page.locator(".qa-review-card.chord")
            if chord_card.count() > 0:
                chord_card.first.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                # Verify confirm button text matches flagged chord (5/7)
                confirm_btn = chord_card.first.locator(".qa-confirm-chord-btn").first
                btn_text = confirm_btn.inner_text()
                assert '确认 "5/7" 正确' in btn_text, f"Unexpected chord confirm button text: {btn_text}"
                p_chord = str(out_dir / f"{dev_name}_card_chord.png")
                chord_card.first.screenshot(path=p_chord)
                screenshot_paths.append(p_chord)

            # 4. Key Change Card
            key_card = page.locator(".qa-review-card.key-change")
            if key_card.count() > 0:
                key_card.first.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                p_key = str(out_dir / f"{dev_name}_card_key_change.png")
                key_card.first.screenshot(path=p_key)
                screenshot_paths.append(p_key)

            # 5. Melody Beat Card
            melody_card = page.locator(".qa-review-card.melody-beat")
            if melody_card.count() > 0:
                melody_card.first.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                p_melody = str(out_dir / f"{dev_name}_card_melody_beat.png")
                melody_card.first.screenshot(path=p_melody)
                screenshot_paths.append(p_melody)

            # 6. Lightbox with zoom
            if chord_card.count() > 0:
                crop_canvas = chord_card.first.locator(".qa-crop-canvas")
                if crop_canvas.count() > 0:
                    crop_canvas.first.click()
                    page.wait_for_selector("#lightbox-modal:not(.hidden)", timeout=3000)
                    page.wait_for_timeout(400)
                    # Click zoom in twice
                    page.locator("#lightbox-zoom-in").click()
                    page.wait_for_timeout(200)
                    page.locator("#lightbox-zoom-in").click()
                    page.wait_for_timeout(300)

                    p_lightbox = str(out_dir / f"{dev_name}_lightbox_zoom.png")
                    page.screenshot(path=p_lightbox, full_page=False)
                    screenshot_paths.append(p_lightbox)

                    # Close lightbox
                    page.locator("#lightbox-close").click()
                    page.wait_for_selector("#lightbox-modal", state="hidden", timeout=3000)
                    page.wait_for_timeout(300)

            # -------------------------------------------------------------
            # Part 2: Low Confidence Sheet with Layout Gate & Render 409
            # -------------------------------------------------------------
            page.goto(f"http://127.0.0.1:{port}/?sheet_id={sheet_b}", wait_until="networkidle")
            page.wait_for_selector("#qa-layout-gate-banner:not(.hidden)", timeout=8000)
            page.wait_for_timeout(800)

            # 7. Gate Banner screenshot
            gate_banner = page.locator("#qa-layout-gate-banner")
            gate_banner.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            p_gate = str(out_dir / f"{dev_name}_gate_banner.png")
            gate_banner.screenshot(path=p_gate)
            screenshot_paths.append(p_gate)

            # 8. Render 409 message: enable and click render button to trigger backend 409
            page.evaluate("() => { const b = document.getElementById('btn-trigger-render'); b.disabled = false; b.click(); }")
            page.wait_for_selector("#render-error-alert:not(.hidden)", timeout=8000)
            page.wait_for_timeout(500)

            error_alert = page.locator("#render-error-alert")
            error_alert.scroll_into_view_if_needed()
            p_err = str(out_dir / f"{dev_name}_render_409.png")
            error_alert.screenshot(path=p_err)
            screenshot_paths.append(p_err)

            # Also a viewport screenshot showing both gate banner and 409 alert together
            p_gate_and_err = str(out_dir / f"{dev_name}_gate_and_409_view.png")
            page.screenshot(path=p_gate_and_err, full_page=False)
            screenshot_paths.append(p_gate_and_err)

            context.close()

        browser.close()

    return screenshot_paths


def main():
    out_dir = Path("out/ui").resolve()
    storage_dir = out_dir / "storage"
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up test sheets in {storage_dir}...")
    sheet_a, sheet_b = setup_test_sheets(storage_dir)

    port = find_free_port()
    print(f"Starting local FastAPI server on port {port}...")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    env["LOCAL_STORAGE_DIR"] = str(storage_dir)
    env["QA_OFFLINE"] = "1"
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--host", "127.0.0.1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        health_url = f"http://127.0.0.1:{port}/healthz"
        if not wait_for_server(health_url, timeout=12.0):
            stderr = server_proc.stderr.read().decode("utf-8", errors="ignore") if server_proc.stderr else ""
            raise RuntimeError(f"Server failed to start on port {port}. Stderr:\n{stderr}")

        print("Server is ready. Capturing screenshots with Playwright...")
        saved_paths = capture_screenshots(port, sheet_a, sheet_b, out_dir)
        print(f"\nSuccessfully generated {len(saved_paths)} screenshots:")
        for path in saved_paths:
            print(f" - {path}")

    finally:
        print("Stopping local server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
