"""Classical computer vision page-layout analysis for numbered musical notation (jianpu)
and Taiwanese band charts.

Detects deskew angle, suppresses watermarks, detects systems (music rows),
physical barlines, measures, chord boxes/tokens, melody bands, lyric bands,
and estimates beats geometrically without using any deep learning or LLMs.
"""
from __future__ import annotations

import logging
from typing import Optional
import cv2
import numpy as np

from app.models import BBox
from app.omr.geometry import GChordBox, GMeasure, GSystem, PageGeometry

logger = logging.getLogger(__name__)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page rotation skew angle in degrees using projection profile variance."""
    h, w = gray.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    sh, sw = small.shape
    center = (sw // 2, sh // 2)
    _, s_bin = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(-5.0, 5.25, 0.5):
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            s_bin, M, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        score = float(np.var(np.sum(rotated, axis=1, dtype=np.float64)))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    fine_angle = best_angle
    for angle in np.arange(best_angle - 0.4, best_angle + 0.45, 0.1):
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            s_bin, M, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        score = float(np.var(np.sum(rotated, axis=1, dtype=np.float64)))
        if score > best_score:
            best_score = score
            fine_angle = float(angle)

    return float(round(fine_angle, 2))


def deskew_image(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate image to correct page skew if estimated angle is non-negligible."""
    angle = estimate_skew_angle(gray)
    if abs(angle) >= 0.3:
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        return rotated, angle
    return gray, 0.0


def binarize_sheet(gray: np.ndarray) -> np.ndarray:
    """Background illumination normalization, watermark suppression, and Otsu binarization."""
    # Background illumination estimation via morphological closing with a large kernel
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    norm = cv2.divide(gray, bg, scale=255.0).astype(np.uint8)

    # Suppress watermark stamps and faint scanner artifacts (pixels with normalized intensity > 185)
    suppressed = np.where(norm > 185, 255, norm).astype(np.uint8)

    # Otsu thresholding for inverse binary (ink = 255, paper = 0)
    _, binary = cv2.threshold(suppressed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_chord_boxes(
    bin_img: np.ndarray,
    y0_px: int,
    y1_px: int,
    measures: list[GMeasure],
    img_w: int,
    img_h: int,
) -> list[GChordBox]:
    """Detect printed chord boxes or chord text tokens in the chord band above melody."""
    if y1_px <= y0_px or not measures:
        return []

    crop = bin_img[y0_px:y1_px, :]
    crop_h, crop_w = crop.shape
    if crop_h < 8 or crop_w < 50:
        return []

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, bool]] = []

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if bw < 14 or bh < 8 or bw > 0.40 * img_w or bh > 0.06 * img_h:
            continue

        rect_ratio = area / float(bw * bh)
        # Printed chord boxes have high rectangularity and aspect ratio between 1.0 and 4.8
        is_boxed = (bw >= 24 and bh >= 12 and rect_ratio >= 0.50 and (bw / float(bh)) <= 4.8)
        # Unboxed chord text tokens or circled numbers
        is_unboxed = (bw >= 14 and bh >= 8 and (rect_ratio >= 0.20 or area >= 35))

        if is_boxed or is_unboxed:
            candidates.append((bx, by + y0_px, bw, bh, is_boxed))

    # Deduplicate overlapping boxes
    candidates.sort(key=lambda b: b[0])
    deduped: list[tuple[int, int, int, int, bool]] = []
    for c in candidates:
        if not deduped:
            deduped.append(c)
        else:
            prev = deduped[-1]
            overlap_x = max(0, min(prev[0] + prev[2], c[0] + c[2]) - max(prev[0], c[0]))
            if overlap_x > 0.60 * min(prev[2], c[2]):
                # Keep the larger / boxed one
                if c[4] and not prev[4]:
                    deduped[-1] = c
                elif c[2] * c[3] > prev[2] * prev[3]:
                    deduped[-1] = c
            else:
                deduped.append(c)

    results: list[GChordBox] = []
    for bx, by, bw, bh, boxed in deduped:
        x_mid_norm = (bx + bw / 2.0) / float(img_w)
        # Find which measure contains x_mid
        target_m_idx = 0
        min_dist = 1e9
        for m in measures:
            if m.x0 <= x_mid_norm <= m.x1:
                target_m_idx = m.index_in_system
                break
            dist = min(abs(x_mid_norm - m.x0), abs(x_mid_norm - m.x1))
            if dist < min_dist:
                min_dist = dist
                target_m_idx = m.index_in_system

        m_obj = measures[target_m_idx]
        m_width = max(1e-4, m_obj.x1 - m_obj.x0)
        frac = max(0.0, min(0.99, (x_mid_norm - m_obj.x0) / m_width))
        raw_beat = 1.0 + frac * 4.0
        beat_geo = float(max(1.0, min(4.0, round(raw_beat * 2.0) / 2.0)))

        bbox_norm: BBox = (
            round(bx / float(img_w), 4),
            round(by / float(img_h), 4),
            round((bx + bw) / float(img_w), 4),
            round((by + bh) / float(img_h), 4),
        )
        results.append(
            GChordBox(
                bbox=bbox_norm,
                measure_index_in_system=target_m_idx,
                beat_geo=beat_geo,
                boxed=boxed,
            )
        )

    return results


def _bar_group_key(g: list[tuple]) -> tuple[float, int]:
    """Sort key for barline clusters prioritizing wider horizontal span then stroke count."""
    xs_g = [b[0] for b in g]
    span_g = (max(xs_g) - min(xs_g)) if len(xs_g) > 1 else 0.0
    return (span_g, len(g))


def analyze_page(image: bytes, page: int = 0) -> PageGeometry:
    """Analyze a music sheet page using classical CV to extract layout geometry.

    Guarantees returning a PageGeometry object without raising exceptions.
    """
    try:
        nparr = np.frombuffer(image, np.uint8)
        gray_raw = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray_raw is None:
            return PageGeometry(
                page=page,
                width=0,
                height=0,
                confidence=0.0,
                notes=["Failed to decode image from bytes buffer"],
            )

        gray, skew_angle = deskew_image(gray_raw)
        h, w = gray.shape
        notes: list[str] = []
        if abs(skew_angle) >= 0.3:
            notes.append(f"Deskewed by {skew_angle:.2f} degrees")

        bin_img = binarize_sheet(gray)

        # Detect whether page is a landscape chord accompaniment chart with tall barlines
        k45 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
        v45 = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k45)
        num45, _, stats45, _ = cv2.connectedComponentsWithStats(v45)
        tall_count = sum(1 for i in range(1, num45) if stats45[i, 2] <= 15 and stats45[i, 3] >= 45)
        is_tall_bar = (tall_count >= 15)

        # Kernel height calibrated to sheet type and resolution
        if is_tall_bar:
            kh = 45
        else:
            kh = max(20, min(26, int(0.015 * h)))

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)

        num, _, stats, centroids = cv2.connectedComponentsWithStats(vert)
        bars = []
        max_w = max(10, int(0.008 * w))
        for i in range(1, num):
            x, y, sw, sh, _ = stats[i]
            cx, cy = centroids[i]
            if sw <= max_w and sh >= kh:
                if 0.02 * w <= cx <= 0.995 * w and 0.04 * h <= cy <= 0.98 * h:
                    bars.append((cx, cy, sw, sh, y, y + sh))

        bars.sort(key=lambda b: b[1])
        raw_clusters: list[list[tuple]] = []
        y_gap = max(35, int(0.035 * h))
        for b in bars:
            if not raw_clusters or abs(b[1] - np.mean([x[1] for x in raw_clusters[-1]])) > y_gap:
                raw_clusters.append([b])
            else:
                raw_clusters[-1].append(b)

        # Process clusters into candidate music rows
        processed_cands: list[dict] = []
        for c in raw_clusters:
            xs = sorted([b[0] for b in c])
            merged: list[float] = []
            for x in xs:
                if not merged or x - merged[-1] > 15:
                    merged.append(x)
            span = (merged[-1] - merged[0]) / float(w) if len(merged) > 1 else 0.0
            y_mean = float(np.mean([b[1] for b in c]))
            if span >= 0.35 or (is_tall_bar and span >= 0.15 and len(merged) >= 2):
                processed_cands.append({
                    "y_mean": y_mean,
                    "bars": c,
                    "merged_xs": merged,
                    "span": span,
                })

        # Inter-system vertical suppression (>= 45 px separation)
        suppressed_cands: list[dict] = []
        min_dist = max(45, int(0.045 * h))
        for cand in processed_cands:
            if not suppressed_cands:
                suppressed_cands.append(cand)
            else:
                prev = suppressed_cands[-1]
                if cand["y_mean"] - prev["y_mean"] < min_dist:
                    if cand["span"] > prev["span"] or len(cand["merged_xs"]) > len(prev["merged_xs"]):
                        suppressed_cands[-1] = cand
                else:
                    suppressed_cands.append(cand)

        # Filter out header cluster if it starts well after standard margin and has no full span
        if len(suppressed_cands) > 1:
            lefts = [c["merged_xs"][0] for c in suppressed_cands]
            med_left = float(np.median(lefts[1:]))
            c0 = suppressed_cands[0]
            if c0["y_mean"] < 0.12 * h and (c0["merged_xs"][0] > med_left + 0.05 * w or c0["span"] < 0.70):
                suppressed_cands.pop(0)

        # Filter out footer cluster in portrait lead sheets
        if len(suppressed_cands) > 1:
            c_last = suppressed_cands[-1]
            if not is_tall_bar and c_last["y_mean"] > 0.90 * h and c_last["span"] < 0.55:
                suppressed_cands.pop(-1)

        # Recover missed systems in large gaps between systems
        if len(suppressed_cands) >= 2:
            diffs = [
                suppressed_cands[i + 1]["y_mean"] - suppressed_cands[i]["y_mean"]
                for i in range(len(suppressed_cands) - 1)
            ]
            med_gap = float(np.median(diffs))
            recovered: list[dict] = []
            for i in range(len(suppressed_cands) - 1):
                recovered.append(suppressed_cands[i])
                gap = suppressed_cands[i + 1]["y_mean"] - suppressed_cands[i]["y_mean"]
                if gap > 1.45 * med_gap:
                    gy0 = int(suppressed_cands[i]["y_mean"] + 0.25 * med_gap)
                    gy1 = int(suppressed_cands[i + 1]["y_mean"] - 0.25 * med_gap)
                    gap_proj = np.sum(bin_img[gy0:gy1, int(0.1 * w):int(0.9 * w)] > 0, axis=1)
                    if len(gap_proj) > 0 and np.max(gap_proj) > 50:
                        peak_y = gy0 + int(np.argmax(gap_proj))
                        recovered.append({
                            "y_mean": float(peak_y),
                            "bars": [],
                            "merged_xs": [],
                            "span": 0.80,
                        })
            recovered.append(suppressed_cands[-1])
            suppressed_cands = recovered

        if not suppressed_cands:
            return PageGeometry(
                page=page,
                width=w,
                height=h,
                confidence=0.0,
                notes=["No musical systems detected on page"],
            )

        # Common printable margins from full systems
        full_lefts = [
            c["merged_xs"][0] for c in suppressed_cands if len(c["merged_xs"]) >= 2 and c["span"] >= 0.60
        ]
        full_rights = [
            c["merged_xs"][-1] for c in suppressed_cands if len(c["merged_xs"]) >= 2 and c["span"] >= 0.60
        ]
        sheet_left = float(np.median(full_lefts)) if full_lefts else 0.05 * w
        sheet_right = float(np.median(full_rights)) if full_rights else 0.95 * w

        # If barlines only exist inside the page (unbordered left margin, common in letter-chord sheets):
        if sheet_left > 0.12 * w:
            left_proj = np.sum(bin_img[:, : int(sheet_left - 0.04 * w)] > 0, axis=0)
            ink_cols = np.where(left_proj > 20)[0]
            if len(ink_cols) > 0 and len(ink_cols) >= int(0.05 * w):
                sheet_left = max(0.035 * w, float(ink_cols[0]))

        # Build GSystem objects with non-overlapping bounding boxes
        systems: list[GSystem] = []
        num_sys = len(suppressed_cands)

        # Compute vertical split boundaries between systems
        split_ys: list[int] = [0]
        for i in range(num_sys - 1):
            y_curr = int(suppressed_cands[i]["y_mean"])
            y_next = int(suppressed_cands[i + 1]["y_mean"])
            sub_proj = np.sum(bin_img[y_curr:y_next, int(0.1 * w):int(0.9 * w)] > 0, axis=1)
            if len(sub_proj) > 0:
                min_idx = int(np.argmin(sub_proj))
                split_ys.append(y_curr + min_idx)
            else:
                split_ys.append((y_curr + y_next) // 2)
        split_ys.append(h)

        # Global header band
        first_y_mean = int(suppressed_cands[0]["y_mean"])
        header_y1 = max(0, first_y_mean - (80 if is_tall_bar else 45))
        header_band: Optional[tuple[float, float]] = (
            (0.0, round(header_y1 / float(h), 4)) if header_y1 > 0.05 * h else None
        )

        for s_idx, sys_cand in enumerate(suppressed_cands):
            ym = float(sys_cand["y_mean"])
            y_top = split_ys[s_idx]
            y_bot = split_ys[s_idx + 1]

            # In the first system, clip y_top to just above the chord/label band
            if s_idx == 0:
                y_top = max(y_top, int(ym - (90 if is_tall_bar else 55)))
            # In the last system, clip y_bot to below lyric/bass lines
            if s_idx == num_sys - 1:
                y_bot = min(y_bot, int(ym + (90 if is_tall_bar else 65)))

            # Melody band
            if is_tall_bar:
                mel_y0 = max(y_top, int(ym - 35))
                mel_y1 = min(y_bot, int(ym + 35))
            else:
                mel_y0 = max(y_top, int(ym - 15))
                mel_y1 = min(y_bot, int(ym + 18))

            melody_band_norm = (round(mel_y0 / float(h), 4), round(mel_y1 / float(h), 4))

            # Barlines and measures
            m_bars: list[float] = []
            if sys_cand["bars"]:
                # Sub-cluster bars within the system by Y to isolate the melody barline line from lyrics
                sb_bars = sorted(sys_cand["bars"], key=lambda b: b[1])
                sub_groups: list[list[tuple]] = []
                for b in sb_bars:
                    if not sub_groups or abs(b[1] - np.mean([x[1] for x in sub_groups[-1]])) > (15 if is_tall_bar else 10):
                        sub_groups.append([b])
                    else:
                        sub_groups[-1].append(b)

                best_group = max(sub_groups, key=_bar_group_key)
                raw_xs = sorted([b[0] for b in best_group])

                # Merge double bars within 0.04 * w
                for bx in raw_xs:
                    if not m_bars or bx - m_bars[-1] > (0.04 * w):
                        m_bars.append(bx)

                # Filter spurious elements outside printable region
                m_bars = [bx for bx in m_bars if sheet_left - 0.04 * w <= bx <= sheet_right + 0.04 * w]

                # Filter out bars closer than 0.165 * w to enforce minimum measure width
                m_bars_spaced: list[float] = []
                for bx in m_bars:
                    if not m_bars_spaced or bx - m_bars_spaced[-1] >= (0.165 * w):
                        m_bars_spaced.append(bx)
                m_bars = m_bars_spaced

                # Check if system starts at sheet_left
                if m_bars and m_bars[0] > sheet_left + 0.12 * w:
                    m_bars.insert(0, sheet_left)
                # Check if system continues to sheet_right
                if m_bars and m_bars[-1] < sheet_right - 0.12 * w:
                    # In portrait lead sheets, every line goes to sheet_right
                    if not is_tall_bar:
                        m_bars.append(sheet_right)
                    else:
                        # In landscape sheets, check ink presence
                        rem_ink = np.sum(
                            bin_img[int(ym - 25):int(ym + 25), int(m_bars[-1] + 25):int(sheet_right + 10)] > 0
                        )
                        if rem_ink > 80:
                            m_bars.append(sheet_right)
            else:
                # Fallback for gap-recovered systems: 4 measures
                m_bars = list(np.linspace(sheet_left, sheet_right, 5))

            if len(m_bars) < 2:
                m_bars = [sheet_left, sheet_right]

            # Construct GMeasure objects
            measures: list[GMeasure] = []
            for m_i in range(len(m_bars) - 1):
                mx0 = m_bars[m_i]
                mx1 = m_bars[m_i + 1]

                # Double bar check at boundaries
                left_double = False
                right_double = False
                if sys_cand["bars"]:
                    left_double = any(0 < abs(b[0] - mx0) <= 15 for b in sys_cand["bars"])
                    right_double = any(0 < abs(b[0] - mx1) <= 15 for b in sys_cand["bars"])

                measures.append(
                    GMeasure(
                        index_in_system=m_i,
                        x0=round(mx0 / float(w), 4),
                        x1=round(mx1 / float(w), 4),
                        left_double_bar=left_double,
                        right_double_bar=right_double,
                    )
                )

            # Chord band
            chord_y0_px = max(y_top, int(mel_y0 - (60 if is_tall_bar else 35)))
            chord_y1_px = mel_y0
            chord_band_norm: Optional[tuple[float, float]] = None
            if chord_y1_px > chord_y0_px:
                chord_band_norm = (round(chord_y0_px / float(h), 4), round(chord_y1_px / float(h), 4))

            # Detect chord boxes
            chord_boxes = detect_chord_boxes(
                bin_img,
                y0_px=chord_y0_px,
                y1_px=chord_y1_px,
                measures=measures,
                img_w=w,
                img_h=h,
            )

            # Lyric bands below melody
            lyric_y0_px = mel_y1
            lyric_y1_px = y_bot
            lyric_bands: list[tuple[float, float]] = []
            if lyric_y1_px > lyric_y0_px:
                lyric_proj = np.sum(bin_img[lyric_y0_px:lyric_y1_px, int(sheet_left):int(sheet_right)] > 0, axis=1)
                if len(lyric_proj) > 0 and np.max(lyric_proj) > 30:
                    lyr_peak = lyric_y0_px + int(np.argmax(lyric_proj))
                    lyric_bands.append((
                        round(max(lyric_y0_px, lyr_peak - 12) / float(h), 4),
                        round(min(lyric_y1_px, lyr_peak + 12) / float(h), 4),
                    ))

            # Full row bbox: guarantees containing melody, chords, lyrics, and non-overlapping
            sys_x0 = max(0.0, round((sheet_left - 10) / float(w), 4))
            sys_x1 = min(1.0, round((sheet_right + 10) / float(w), 4))
            sys_bbox: BBox = (
                sys_x0,
                round(y_top / float(h), 4),
                sys_x1,
                round(y_bot / float(h), 4),
            )

            # Confidence evaluation
            diffs = [m.x1 - m.x0 for m in measures]
            regularity = float(np.std(diffs)) if len(diffs) > 1 else 0.0
            sys_conf = float(max(0.60, min(1.0, 1.0 - regularity * 2.0)))

            systems.append(
                GSystem(
                    page=page,
                    index_on_page=s_idx,
                    bbox=sys_bbox,
                    melody_band=melody_band_norm,
                    chord_band=chord_band_norm,
                    label_band=None,
                    lyric_bands=lyric_bands,
                    bass_band=None,
                    measures=measures,
                    chord_boxes=chord_boxes,
                    confidence=sys_conf,
                    notes=[],
                )
            )

        # Global page confidence
        global_conf = float(min(s.confidence for s in systems)) if systems else 0.0

        return PageGeometry(
            page=page,
            width=w,
            height=h,
            header_band=header_band,
            systems=systems,
            confidence=global_conf,
            notes=notes,
        )

    except Exception as exc:
        logger.warning("Error in analyze_page: %s", exc)
        return PageGeometry(
            page=page,
            width=0,
            height=0,
            confidence=0.0,
            notes=[f"Error during layout analysis: {exc}"],
        )
