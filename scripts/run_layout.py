"""Batch evaluation and debug overlay visualization for layout analysis.

Processes music sheet pages from fixtures/, runs analyze_page(),
draws color-coded debug overlays (blue system bbox, green melody band,
red barlines, magenta chord boxes with beat label, yellow lyric bands),
saves them to out/layout/<slug>_p<N>.png, and prints a markdown summary table.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from app.omr.layout import analyze_page, deskew_image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "out" / "layout"
TRUTH_PATH = REPO_ROOT / "fixtures" / "layout_truth.json"


def draw_overlay(
    img_bgr: np.ndarray,
    geom,
) -> np.ndarray:
    """Draw color-coded debug annotations on page image."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Draw systems
    for sys_idx, s in enumerate(geom.systems):
        sx0, sy0, sx1, sy1 = s.bbox
        x0_px = int(sx0 * w)
        y0_px = int(sy0 * h)
        x1_px = int(sx1 * w)
        y1_px = int(sy1 * h)

        # 1. System bbox: Blue (BGR: 255, 0, 0), thickness 2
        cv2.rectangle(vis, (x0_px, y0_px), (x1_px, y1_px), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"Sys {sys_idx + 1} ({len(s.measures)}m)",
            (x0_px + 5, y0_px + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # 2. Melody band: Green (BGR: 0, 255, 0)
        my0_px = int(s.melody_band[0] * h)
        my1_px = int(s.melody_band[1] * h)
        cv2.rectangle(vis, (x0_px, my0_px), (x1_px, my1_px), (0, 255, 0), 1)

        # 3. Chord band if present
        if s.chord_band:
            cy0_px = int(s.chord_band[0] * h)
            cy1_px = int(s.chord_band[1] * h)
            cv2.rectangle(vis, (x0_px, cy0_px), (x1_px, cy1_px), (180, 180, 0), 1)

        # 4. Lyric bands: Yellow (BGR: 0, 255, 255)
        for lyr_band in s.lyric_bands:
            ly0_px = int(lyr_band[0] * h)
            ly1_px = int(lyr_band[1] * h)
            cv2.rectangle(vis, (x0_px, ly0_px), (x1_px, ly1_px), (0, 255, 255), 1)

        # 5. Barlines: Red vertical lines (BGR: 0, 0, 255)
        for m in s.measures:
            mx0_px = int(m.x0 * w)
            mx1_px = int(m.x1 * w)
            # Left barline
            cv2.line(vis, (mx0_px, my0_px - 5), (mx0_px, my1_px + 5), (0, 0, 255), 2 if m.left_double_bar else 1)
            # Right barline
            cv2.line(vis, (mx1_px, my0_px - 5), (mx1_px, my1_px + 5), (0, 0, 255), 2 if m.right_double_bar else 1)

        # 6. Chord boxes: Magenta rectangle (BGR: 255, 0, 255) with beat_geo label
        for cb in s.chord_boxes:
            cx0, cy0, cx1, cy1 = cb.bbox
            bx0 = int(cx0 * w)
            by0 = int(cy0 * h)
            bx1 = int(cx1 * w)
            by1 = int(cy1 * h)
            cv2.rectangle(vis, (bx0, by0), (bx1, by1), (255, 0, 255), 2 if cb.boxed else 1)
            beat_label = f"b{cb.beat_geo:g}"
            cv2.putText(
                vis,
                beat_label,
                (bx0, max(12, by0 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return vis


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth = {}
    if TRUTH_PATH.exists():
        with open(TRUTH_PATH, "r", encoding="utf-8") as f:
            truth = json.load(f)

    # Gather images to evaluate
    image_paths: list[Path] = []
    # Standard pages
    pages_dir = REPO_ROOT / "fixtures" / "pages"
    if pages_dir.exists():
        image_paths.extend(sorted(pages_dir.glob("*.jpg")))
    # External fixtures
    ext_dir = REPO_ROOT / "fixtures" / "external"
    if ext_dir.exists():
        image_paths.extend(sorted(ext_dir.glob("*/*.jpg")))

    rows: list[dict] = []
    print(f"Running layout analysis on {len(image_paths)} pages...\n")

    for img_p in image_paths:
        rel_key = str(img_p.relative_to(REPO_ROOT))
        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is None:
            continue

        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        deskewed_gray, deskew_ang = deskew_image(raw_gray)
        if abs(deskew_ang) >= 0.3:
            h, w = raw_bgr.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, deskew_ang, 1.0)
            vis_bgr = cv2.warpAffine(
                raw_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
            )
        else:
            vis_bgr = raw_bgr

        t0 = time.perf_counter()
        img_bytes = img_p.read_bytes()
        geom = analyze_page(img_bytes, page=0)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Draw debug overlay
        overlay = draw_overlay(vis_bgr, geom)
        slug = img_p.parent.name if img_p.parent.name != "pages" else "taiwan"
        p_name = img_p.stem
        out_name = f"{slug}_{p_name}.png"
        out_path = OUT_DIR / out_name
        cv2.imwrite(str(out_path), overlay)

        # Ground truth comparison
        tr = truth.get(rel_key, {})
        exp_sys = tr.get("num_systems", "-")
        exp_m = tr.get("measures_per_system", "-")
        total_exp_m = sum(exp_m) if isinstance(exp_m, list) else "-"

        det_sys = len(geom.systems)
        det_m = [len(s.measures) for s in geom.systems]
        total_det_m = sum(det_m)
        total_chords = sum(len(s.chord_boxes) for s in geom.systems)

        status = "PASS"
        if exp_sys != "-" and det_sys != exp_sys:
            status = "FAIL"
        elif isinstance(exp_m, list) and det_m != exp_m:
            diffs = [abs(d - e) for d, e in zip(det_m, exp_m)]
            status = "EXACT" if all(d == 0 for d in diffs) else ("APPROX" if all(d <= 1 for d in diffs) else "FAIL")
        elif exp_sys != "-":
            status = "EXACT"

        rows.append({
            "rel_key": rel_key,
            "out_file": out_name,
            "shape": f"{vis_bgr.shape[1]}x{vis_bgr.shape[0]}",
            "deskew": f"{deskew_ang:.1f}°" if abs(deskew_ang) >= 0.3 else "0.0°",
            "sys": f"{det_sys}/{exp_sys}",
            "measures": f"{total_det_m}/{total_exp_m}",
            "chords": total_chords,
            "conf": f"{geom.confidence:.2f}",
            "time_ms": f"{dt_ms:.1f}ms",
            "status": status,
        })

    # Print Markdown table
    header = "| Image | Shape | Skew | Systems | Measures | Chords | Conf | Time | Status | Overlay |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| `{r['rel_key']}` | {r['shape']} | {r['deskew']} | {r['sys']} | {r['measures']} | {r['chords']} | {r['conf']} | {r['time_ms']} | **{r['status']}** | `out/layout/{r['out_file']}` |"
        )
    print(f"\nSaved {len(rows)} debug overlays to `{OUT_DIR}`.")


if __name__ == "__main__":
    main()
