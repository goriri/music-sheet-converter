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
    for angle in np.arange(-8.0, 8.25, 0.5):
        m_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            s_bin, m_rot, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        score = float(np.var(np.sum(rotated, axis=1, dtype=np.float64)))
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    fine_angle = best_angle
    for angle in np.arange(best_angle - 0.4, best_angle + 0.45, 0.1):
        m_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            s_bin, m_rot, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
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
        m_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
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


def _find_dashes_in_binary(
    img_c: np.ndarray,
    y0: int,
    mel_h: float,
    crop: np.ndarray,
) -> list[tuple[float, float, int, int, float, float]]:
    num, _, stats, centroids = cv2.connectedComponentsWithStats(img_c)
    dash_candidates: list[tuple[float, float, int, int, int]] = []
    for i in range(1, num):
        x, y, sw, sh, area = stats[i]
        cx, cy = centroids[i]
        if sw <= 5 and 1 <= sh <= 16 and area <= 65:
            dash_candidates.append((float(cx), float(cy + y0), int(sh), int(y + y0), int(y + y0 + sh)))

    dash_candidates.sort(key=lambda d: d[0])
    groups: list[list[tuple[float, float, int, int, int]]] = []
    for d in dash_candidates:
        if not groups or abs(d[0] - np.mean([x[0] for x in groups[-1]])) > 2.0:
            groups.append([d])
        else:
            groups[-1].append(d)

    num_crop, _, stats_crop, _ = cv2.connectedComponentsWithStats(crop)
    digit_strokes: list[tuple[int, int, int, int]] = []
    for i in range(1, num_crop):
        bx, by, bw, bh, area = stats_crop[i]
        if bh >= 0.45 * mel_h or (bh >= 10 and bw >= 5):
            digit_strokes.append((bx, bx + bw, by + y0, by + y0 + bh))

    bars: list[tuple[float, float, int, int, float, float]] = []
    min_span = max(12.0, 0.55 * mel_h)
    for g in groups:
        if len(g) >= 3:
            top_y = min(d[3] for d in g)
            bot_y = max(d[4] for d in g)
            span_y = bot_y - top_y
            if span_y >= min_span:
                avg_x = float(np.mean([d[0] for d in g]))
                avg_y = float(np.mean([d[1] for d in g]))
                if not (0.02 * crop.shape[1] <= avg_x <= 0.98 * crop.shape[1]):
                    continue
                coincides = any(
                    bx0 - 2 <= avg_x <= bx1 + 2 and max(top_y, dy0) < min(bot_y, dy1)
                    for bx0, bx1, dy0, dy1 in digit_strokes
                )
                if not coincides:
                    bars.append((avg_x, avg_y, 2, int(span_y), float(top_y), float(bot_y)))
    return bars


def find_dashed_barlines(
    bin_img: np.ndarray,
    ym: float,
    y_radius: int = 24,
    u_curr: float = 20.0,
    mel_h: float = 33.0,
    gray: Optional[np.ndarray] = None,
) -> list[tuple[float, float, int, int, float, float]]:
    """Detect dashed vertical barlines (e.g. pickup barlines or repeat barlines)."""
    h, w = bin_img.shape
    y0 = max(0, int(ym - y_radius))
    y1 = min(h, int(ym + y_radius))
    crop = bin_img[y0:y1, :]

    res = _find_dashes_in_binary(crop, y0, mel_h, crop)
    if not res and gray is not None:
        crop_gray = gray[y0:y1, :]
        bg = float(np.median(crop_gray))
        for t_val in [min(215.0, bg - 35.0), min(195.0, bg - 55.0)]:
            _, b_alt = cv2.threshold(crop_gray, t_val, 255, cv2.THRESH_BINARY_INV)
            res = _find_dashes_in_binary(b_alt, y0, mel_h, crop)
            if res:
                break
    return res


def detect_chord_boxes(
    bin_img: np.ndarray,
    y0_px: int,
    y1_px: int,
    measures: list[GMeasure],
    img_w: int,
    img_h: int,
    chords_below: bool = False,
    is_boxed_page: bool = False,
    scaled_page: bool = False,
) -> list[GChordBox]:
    """Detect printed chord boxes or chord text tokens in the chord band."""
    if y1_px <= y0_px or not measures:
        return []

    crop = bin_img[y0_px:y1_px, :]
    crop_h, crop_w = crop.shape
    if crop_h < 8 or crop_w < 50:
        return []

    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[list] = []

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if bw < 8 or bh < 8 or bw > 0.40 * img_w or bh > 0.08 * img_h:
            continue

        rect_ratio = area / float(bw * bh) if bw * bh > 0 else 0.0
        aspect = bw / float(bh)

        if is_boxed_page:
            # On boxed charts, closed rounded rectangles or hollow outlines strictly in chord band
            min_bh = 18 if scaled_page else 17
            is_boxed = (
                bw >= 20
                and bh >= min_bh
                and 1.05 <= aspect <= 4.8
                and (
                    rect_ratio >= 0.65
                    or (rect_ratio >= 0.22 and area >= 150)
                )
            )
            if is_boxed:
                candidates.append([bx, by + y0_px, bw, bh, True])
        else:
            # Printed chord boxes (Taiwanese charts)
            is_boxed = (
                bw >= 24 and bh >= 12 and rect_ratio >= 0.50 and aspect <= 4.8 and not chords_below
            )
            # Unboxed chord text tokens or circled numbers (Mainland charts)
            is_unboxed = (bw >= 10 and bh >= 8 and (rect_ratio >= 0.20 or area >= 25))

            if is_boxed or is_unboxed:
                candidates.append([bx, by + y0_px, bw, bh, is_boxed])

    if chords_below:
        # Merge horizontally close components (<= 15px) for circled numbers, slashes, extensions
        candidates.sort(key=lambda b: b[0])
        merged: list[list] = []
        for c in candidates:
            if not merged:
                merged.append(c)
            else:
                prev = merged[-1]
                if c[0] - (prev[0] + prev[2]) <= max(18.0, 0.016 * img_w):
                    x0 = min(prev[0], c[0])
                    y0 = min(prev[1], c[1])
                    x1 = max(prev[0] + prev[2], c[0] + c[2])
                    y1 = max(prev[1] + prev[3], c[1] + c[3])
                    merged[-1] = [x0, y0, x1 - x0, y1 - y0, False]
                else:
                    merged.append(c)
        candidates = merged
    else:
        # Deduplicate overlapping boxes
        candidates.sort(key=lambda b: b[0])
        deduped: list[list] = []
        for c in candidates:
            if not deduped:
                deduped.append(c)
            else:
                prev = deduped[-1]
                overlap_x = max(0, min(prev[0] + prev[2], c[0] + c[2]) - max(prev[0], c[0]))
                if overlap_x > 0.60 * min(prev[2], c[2]):
                    if c[4] and not prev[4]:
                        deduped[-1] = c
                    elif c[2] * c[3] > prev[2] * prev[3]:
                        deduped[-1] = c
                else:
                    deduped.append(c)
        candidates = deduped

    results: list[GChordBox] = []
    for bx, by, bw, bh, boxed in candidates:
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


def _bar_group_key(g: list[tuple], chords_below: bool = False) -> tuple[int, float, int]:
    """Sort key for barline clusters prioritizing valid barline count, regularity, span, then stroke count."""
    xs_g = sorted([b[0] for b in g])
    merged: list[float] = []
    for x in xs_g:
        if not merged or x - merged[-1] > 15:
            merged.append(x)
    span_g = (merged[-1] - merged[0]) if len(merged) > 1 else 0.0
    if span_g <= 0:
        return (0, 0.0, 0)
    if chords_below:
        if len(merged) >= 6 and (span_g / (len(merged) - 1)) < 110.0:
            return (0, 0.0, 0)
    elif len(merged) >= 8 and (span_g / (len(merged) - 1)) < 90.0:
        return (0, 0.0, 0)
    cleaned = clean_narrow_measures(merged, 0.12 * span_g)
    if not (2 <= len(cleaned) <= 8):
        return (0, 0.0, 0)
    diffs = [cleaned[i + 1] - cleaned[i] for i in range(len(cleaned) - 1)]
    rel_std = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 1.0
    score = span_g * (1.0 - min(0.8, rel_std))
    if len(cleaned) == 2:
        score *= 0.5
    elif len(cleaned) == 3 and span_g > 400:
        score *= 0.8
    return (1, score, len(g))


def _cluster_by_detrended_y(bars: list[tuple], y_sub_thresh: float, is_boxed_page: bool = False) -> list[list[tuple]]:
    if not bars:
        return []
    if len(bars) <= 1:
        return [list(bars)]
    ys = [b[1] for b in bars]
    med_y = float(np.median(ys))
    inliers = [i for i, y in enumerate(ys) if abs(y - med_y) <= 15]
    slopes = []
    for i_idx in range(len(inliers)):
        i = inliers[i_idx]
        for j_idx in range(i_idx + 1, len(inliers)):
            j = inliers[j_idx]
            dx = bars[j][0] - bars[i][0]
            dy = ys[j] - ys[i]
            if abs(dx) > 100 and abs(dy) <= 15:
                slopes.append(dy / dx)
    slope = float(np.median(slopes)) if slopes else 0.0
    thresh_slope = 0.0035
    if abs(slope) < thresh_slope:
        slope = 0.0
    else:
        slope = max(-0.045, min(0.045, slope))
    det_bars = []
    sh_med = float(np.median([b[3] for b in bars])) if bars else 0.0
    reg_bars = [b for b in bars if b[3] <= 1.35 * sh_med and len(b) >= 6]
    med_bot = float(np.median([b[5] - slope * b[0] for b in reg_bars])) if reg_bars else None
    med_top = float(np.median([b[4] - slope * b[0] for b in reg_bars])) if reg_bars else None
    for b in bars:
        eff_y = b[1]
        if is_boxed_page and med_bot is not None and b[3] > 1.35 * sh_med and len(b) >= 6:
            b_bot = b[5] - slope * b[0]
            b_top = b[4] - slope * b[0]
            if abs(b_bot - med_bot) < abs(b_top - med_top) and abs(b_bot - med_bot) <= 10:
                eff_y = b[5] - sh_med / 2.0
            elif abs(b_top - med_top) <= 10:
                eff_y = b[4] + sh_med / 2.0
        det_bars.append((b, eff_y - slope * b[0]))
    det_bars.sort(key=lambda item: item[1])
    sub_groups: list[list[tuple]] = []
    first_ydets: list[float] = []
    for b, y_det in det_bars:
        if not sub_groups or (y_det - first_ydets[-1]) > y_sub_thresh:
            sub_groups.append([b])
            first_ydets.append(y_det)
        else:
            sub_groups[-1].append(b)
    return sub_groups


def extract_chord_only_boxes(
    bin_img: np.ndarray,
    y0_px: int,
    y1_px: int,
    measures: list[GMeasure],
    img_w: int,
    img_h: int,
    force: bool = False,
) -> tuple[bool, list[GChordBox]]:
    """Detect if notation band is chord-only ('弹唱版') and extract circled chord boxes.

    Circled degree numbers (e.g. ①, ⑤, stacked ⑦/⑤, ⑥m, ♭⑦) sitting in the melody
    band are merged with their accidentals and suffixes into unboxed GChordBox objects.
    """
    crop = bin_img[max(0, y0_px - 10):min(img_h, y1_px + 10), :]
    nl, _, stats, _ = cv2.connectedComponentsWithStats(crop)

    seeds: list[tuple[int, int, int, int]] = []
    digits: list[tuple[int, int, int, int]] = []

    for ci in range(1, nl):
        cx, cy, cw, ch, area = stats[ci]
        is_circle = (28 <= cw <= 80 and 28 <= ch <= 80 and 0.75 <= cw / float(ch) <= 1.35 and 200 <= area <= 1200)
        is_slash = (30 <= cw <= 80 and 50 <= ch <= 110 and area >= 200)
        if is_circle or is_slash:
            seeds.append((cx, y0_px - 10 + cy, cw, ch))
        elif 6 <= cw <= 24 and 16 <= ch <= 35 and area <= 280:
            digits.append((cx, y0_px - 10 + cy, cw, ch))

    if not force:
        if len(seeds) < 2 or (len(seeds) < 3 and len(digits) > len(seeds)):
            return False, []
    elif not seeds:
        return False, []

    # Cluster seeds into chord tokens horizontally within 25px
    seeds.sort(key=lambda s: s[0])
    grouped: list[list[int]] = []
    for s in seeds:
        if not grouped or s[0] - grouped[-1][1] > 25:
            grouped.append([s[0], s[0] + s[2], s[1], s[1] + s[3]])
        else:
            grouped[-1][1] = max(grouped[-1][1], s[0] + s[2])
            grouped[-1][2] = min(grouped[-1][2], s[1])
            grouped[-1][3] = max(grouped[-1][3], s[1] + s[3])

    # Expand tokens to include accidentals on the left and suffixes on the right
    for grp in grouped:
        for ci in range(1, nl):
            cx, cy, cw, ch, area = stats[ci]
            abs_y0 = y0_px - 10 + cy
            abs_y1 = abs_y0 + ch
            if abs_y1 >= grp[2] - 5 and abs_y0 <= grp[3] + 5:
                if 0 <= grp[0] - (cx + cw) <= 18 and cw <= 25:
                    grp[0] = min(grp[0], cx)
                if 0 <= cx - grp[1] <= 25 and cw <= 40:
                    grp[1] = max(grp[1], cx + cw)

    chord_boxes: list[GChordBox] = []
    for bx0, bx1, by0, by1 in grouped:
        x_mid_norm = (bx0 + bx1) / 2.0 / float(img_w)
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
        m_w = max(1e-4, m_obj.x1 - m_obj.x0)
        frac = max(0.0, min(0.99, (x_mid_norm - m_obj.x0) / m_w))
        beat_geo = float(max(1, min(4, int(frac * 4) + 1)))

        bbox_norm: BBox = (
            round(bx0 / float(img_w), 4),
            round(by0 / float(img_h), 4),
            round(bx1 / float(img_w), 4),
            round(by1 / float(img_h), 4),
        )
        chord_boxes.append(
            GChordBox(
                bbox=bbox_norm,
                measure_index_in_system=target_m_idx,
                beat_geo=beat_geo,
                boxed=False,
            )
        )

    return True, chord_boxes


def clean_narrow_measures(
    bars: list[float],
    min_m_w: float,
    min_end_w: float | None = None,
    dashed_xs: set[float] | None = None,
    sheet_left: float | None = None,
    is_first_system: bool = False,
    allow_repeat_ending: bool = False,
) -> list[float]:
    """Prune spurious barlines from boxed charts while preserving true measure boundaries."""
    if len(bars) <= 2:
        return bars
    cleaned = list(bars)
    while len(cleaned) > 2:
        diffs = [cleaned[i + 1] - cleaned[i] for i in range(len(cleaned) - 1)]
        total_w = max(1.0, cleaned[-1] - cleaned[0])
        too_narrow = []
        for i, d in enumerate(diffs):
            if min_end_w is not None and (i == len(diffs) - 1 or (allow_repeat_ending and i == len(diffs) - 2)):
                thresh = min_end_w
            elif min_end_w is not None and i == 0:
                if is_first_system or (sheet_left is not None and cleaned[0] < sheet_left - 0.02 * total_w):
                    thresh = min_end_w
                else:
                    thresh = min_m_w
            else:
                thresh = min_m_w
            if dashed_xs:
                b_left, b_right = cleaned[i], cleaned[i + 1]
                if any(abs(x - b_left) <= 5.0 or abs(x - b_right) <= 5.0 for x in dashed_xs):
                    thresh = min(thresh, 0.035 * total_w)
            if d < thresh:
                too_narrow.append(i)
        if not too_narrow:
            break
        i_min = min(too_narrow, key=lambda i: diffs[i])
        c_a = cleaned[:i_min] + cleaned[i_min + 1:]
        c_b = cleaned[:i_min + 1] + cleaned[i_min + 2:]
        diffs_a = [c_a[j + 1] - c_a[j] for j in range(len(c_a) - 1)]
        diffs_b = [c_b[j + 1] - c_b[j] for j in range(len(c_b) - 1)]
        cleaned = c_a if np.std(diffs_a) <= np.std(diffs_b) else c_b
    return cleaned


def measure_intrinsic_unit(bin_img: np.ndarray) -> float:
    """Estimate intrinsic notation scale unit (nominal digit/stroke height) from connected components."""
    h, w = bin_img.shape
    nl, _, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    cands: list[int] = []
    for i in range(1, nl):
        sw = stats[i, 2]
        sh = stats[i, 3]
        area = stats[i, 4]
        if (
            0.004 * h <= sh <= 0.04 * h
            and 0.003 * w <= sw <= 0.04 * w
            and 0.25 <= sw / float(sh) <= 1.8
            and 0.15 <= area / float(sw * sh) <= 0.85
            and area >= max(6, int(0.000005 * w * h))
        ):
            cands.append(sh)
    return float(np.median(cands)) if cands else float(max(15.0, 0.01 * h))


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
        orig_h, orig_w = gray.shape[:2]
        notes: list[str] = []
        if abs(skew_angle) >= 0.3:
            notes.append(f"Deskewed by {skew_angle:.2f} degrees")

        bin_pre = binarize_sheet(gray)
        u_intrinsic = measure_intrinsic_unit(bin_pre)
        TARGET_UNIT = 20.0
        scaled_page = False
        if u_intrinsic < 15.0 or u_intrinsic > 26.0:
            scale = TARGET_UNIT / max(6.0, u_intrinsic)
            target_w = int(round(orig_w * scale))
            target_h = int(round(orig_h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            gray = cv2.resize(gray, (target_w, target_h), interpolation=interp)
            bin_img = binarize_sheet(gray)
            h, w = gray.shape
            scaled_page = True
            u_curr = TARGET_UNIT
        else:
            bin_img = bin_pre
            h, w = orig_h, orig_w
            u_curr = u_intrinsic

        # Detect whether page is a landscape chord accompaniment chart with tall barlines
        k45 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
        v45 = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k45)
        num45, _, stats45, _ = cv2.connectedComponentsWithStats(v45)
        tall_count = sum(1 for i in range(1, num45) if stats45[i, 2] <= 15 and stats45[i, 3] >= 45)
        is_tall_bar = (tall_count >= 15)

        # Detect whether page is a boxed Taiwanese chart
        cnts_page, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        closed_boxes = 0
        for cnt in cnts_page:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw < 32 or bh < 16 or bw > 0.30 * w or bh > 0.08 * h:
                continue
            aspect = bw / float(bh)
            if 1.05 <= aspect <= 4.8:
                area = cv2.contourArea(cnt)
                rect_ratio = area / float(bw * bh) if bw * bh > 0 else 0
                if rect_ratio >= 0.65:
                    closed_boxes += 1
        is_boxed_page = (closed_boxes >= 20)


        # Kernel height calibrated to sheet type and resolution
        if is_tall_bar:
            kh = 45
        else:
            kh = (24 if scaled_page else max(20, min(26, int(0.015 * h))))

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        vert = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k)

        num, _, stats, centroids = cv2.connectedComponentsWithStats(vert)
        bars: list[tuple] = []
        max_w = max(14, int(0.012 * w))
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

        # Decide chords_below vs chords_above from evidence (strictly False on boxed pages)
        if is_boxed_page:
            chords_below = False
        else:
            boxed_above_count = 0
            tokens_below_count = 0
            for c in raw_clusters[:8]:
                ym = int(np.mean([b[1] for b in c]))
                crop_above = bin_img[max(0, ym - int(0.025 * h)):ym - int(0.006 * h), int(0.1 * w):int(0.9 * w)]
                crop_below = bin_img[ym + int(0.006 * h):min(h, ym + int(0.025 * h)), int(0.1 * w):int(0.9 * w)]

                cnts_above, _ = cv2.findContours(crop_above, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts_above:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    rect = cv2.contourArea(cnt) / float(bw * bh) if bw * bh > 0 else 0
                    if bw >= 20 and bh >= 10 and rect >= 0.50:
                        boxed_above_count += 1

                cnts_below, _ = cv2.findContours(crop_below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts_below:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if 12 <= bw <= 55 and 12 <= bh <= 35 and 0.5 <= (bw / float(bh)) <= 2.8:
                        tokens_below_count += 1

            chords_below = (tokens_below_count >= 15 and (boxed_above_count < 5 or tokens_below_count >= 2 * boxed_above_count))
        if chords_below:
            notes.append("chords_below")

        # Process clusters into candidate music rows
        has_text_lines = False
        processed_cands: list[dict] = []
        for c in raw_clusters:
            if scaled_page:
                y_sub_thresh = 12 if is_tall_bar else (6.5 if is_boxed_page else (3.5 if chords_below else 6.0))
                sub_groups = _cluster_by_detrended_y(c, y_sub_thresh, is_boxed_page=is_boxed_page)
                best_group = max(sub_groups, key=lambda sg: _bar_group_key(sg, chords_below=chords_below))
                y_mean = float(np.mean([b[1] for b in best_group]))
                xs = sorted([b[0] for b in best_group])
                c_bars = best_group
            else:
                y_mean = float(np.mean([b[1] for b in c]))
                xs = sorted([b[0] for b in c])
                c_bars = c

            merged: list[float] = []
            for x in xs:
                if not merged or x - merged[-1] > (18 if scaled_page else 15):
                    merged.append(x)
            span = (merged[-1] - merged[0]) / float(w) if len(merged) > 1 else 0.0

            # Reject text lines: in paragraph text lines, vertical strokes of CJK characters
            # yield 11 or more strokes across the row, or excessive density (avg measure width < 0.065)
            avg_m_w = span / max(1, len(merged) - 1)
            limit = 12 if scaled_page else 11
            is_text = False
            if len(merged) >= limit or (len(merged) >= 8 and avg_m_w < 0.065):
                if is_boxed_page and span >= 0.35:
                    cleaned_m = clean_narrow_measures(merged, 0.12 * span * w)
                    if not (2 <= len(cleaned_m) <= 8):
                        is_text = True
                else:
                    is_text = True
            if is_text:
                has_text_lines = True
                continue

            # Allow short systems (1-2 measures) if span >= 0.16 and len(merged) >= 2
            if span >= 0.35 or (span >= 0.16 and len(merged) >= 2):
                processed_cands.append({
                    "y_mean": y_mean,
                    "bars": c_bars,
                    "merged_xs": merged,
                    "span": span,
                })

        # Inter-system vertical suppression (>= 40 px separation)
        suppressed_cands: list[dict] = []
        min_dist = max(40, int(0.035 * h))
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

        # Common printable margins from full systems
        full_lefts = [
            c["merged_xs"][0] for c in suppressed_cands if len(c["merged_xs"]) >= 2 and c["span"] >= 0.60
        ]
        full_rights = [
            c["merged_xs"][-1] for c in suppressed_cands if len(c["merged_xs"]) >= 2 and c["span"] >= 0.60
        ]
        sheet_left = float(np.median(full_lefts)) if full_lefts else 0.05 * w
        sheet_right = float(np.median(full_rights)) if full_rights else 0.95 * w

        if sheet_left > 0.16 * w:
            left_proj = np.sum(bin_img[:, : int(sheet_left - 0.04 * w)] > 0, axis=0)
            ink_cols = np.where(left_proj > 20)[0]
            if len(ink_cols) > 0 and len(ink_cols) >= int(0.05 * w):
                sheet_left = max(0.035 * w, float(ink_cols[0]))

        # Filter out footer cluster: keep if aligned with sheet_left and has measure barlines
        if len(suppressed_cands) > 1:
            c_last = suppressed_cands[-1]
            if not is_tall_bar and c_last["y_mean"] > 0.90 * h and c_last["span"] < 0.55:
                starts_at_margin = abs(c_last["merged_xs"][0] - sheet_left) < 0.08 * w
                if not (starts_at_margin and len(c_last["merged_xs"]) >= 2):
                    suppressed_cands.pop(-1)

        if not suppressed_cands:
            return PageGeometry(
                page=page,
                width=orig_w,
                height=orig_h,
                confidence=0.0,
                notes=["No musical systems detected on page"],
            )



        # Recover missed systems in large gaps between systems (only on traditional chords_above sheets)
        if not chords_below and len(suppressed_cands) >= 2:
            diffs = [
                suppressed_cands[i + 1]["y_mean"] - suppressed_cands[i]["y_mean"]
                for i in range(len(suppressed_cands) - 1)
            ]
            med_gap = float(np.median(diffs))
            recovered: list[dict] = []
            for i in range(len(suppressed_cands) - 1):
                recovered.append(suppressed_cands[i])
                gap = suppressed_cands[i + 1]["y_mean"] - suppressed_cands[i]["y_mean"]
                if gap > 1.35 * med_gap:
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

        systems: list[GSystem] = []
        num_sys = len(suppressed_cands)

        # Compute vertical split boundaries between systems
        split_ys: list[int] = [0]
        for i in range(num_sys - 1):
            y_curr = int(suppressed_cands[i]["y_mean"])
            y_next = int(suppressed_cands[i + 1]["y_mean"])
            if chords_below:
                # System i+1 starts just above its own melody line (preventing absorbing chords)
                split_ys.append(max(int(y_curr + 55), int(y_next - (35 if is_tall_bar else 22))))
            else:
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

            if s_idx == 0:
                y_top = max(y_top, int(ym - (90 if is_tall_bar else max(55, int(0.055 * h)))))
            if s_idx == num_sys - 1:
                typical_margin = (
                    int(np.median([split_ys[i + 1] - int(suppressed_cands[i]["y_mean"]) for i in range(num_sys - 1)]))
                    if num_sys > 1
                    else (160 if is_tall_bar else 90)
                )
                max_search_y = min(h - int(0.02 * h), int(ym + max(typical_margin, (160 if is_tall_bar else 90))))
                sub_proj = np.sum(bin_img[int(ym):max_search_y, int(sheet_left):int(sheet_right)] > 0, axis=1)
                content_rows = np.where(sub_proj > 20)[0]
                if len(content_rows) > 0:
                    y_bot = min(h, int(ym + content_rows[-1] + 25))
                else:
                    y_bot = min(y_bot, int(ym + (90 if is_tall_bar else 65)))

            # Melody band
            if is_tall_bar:
                mel_y0 = max(y_top, int(ym - 35))
                mel_y1 = min(y_bot, int(ym + 35))
            else:
                mel_y0 = max(y_top, int(ym - 15))
                mel_y1 = min(y_bot, int(ym + 18))

            melody_band_norm = (round(mel_y0 / float(h), 4), round(mel_y1 / float(h), 4))

            # Candidate barlines: combine solid bars and dashed barlines
            candidate_bars: list[tuple] = list(sys_cand["bars"])
            dashed_bars = (
                []
                if is_boxed_page
                else find_dashed_barlines(
                    bin_img,
                    ym,
                    y_radius=int(0.015 * h),
                    u_curr=u_curr,
                    mel_h=float(mel_y1 - mel_y0),
                    gray=gray,
                )
            )
            for db in dashed_bars:
                # If db is on the left (pickup candidate)
                if db[0] < sheet_left + 0.05 * w:
                    if is_tall_bar and s_idx > 0:
                        continue
                    if not is_tall_bar:
                        scale_f = (h / 2400.0) * (w / 1700.0)
                        left_ink = np.sum(bin_img[y_top:y_bot, int(db[0]):int(sheet_left - 2)] > 0)
                        min_ink = 350 if s_idx == 0 else 650
                        if (left_ink / scale_f) < min_ink:
                            continue
                candidate_bars.append(db)

            m_bars: list[float] = []
            if candidate_bars:
                sub_groups = _cluster_by_detrended_y(candidate_bars, 20 if is_tall_bar else 15, is_boxed_page=is_boxed_page)

                best_group = max(sub_groups, key=lambda sg: _bar_group_key(sg, chords_below=chords_below))
                raw_xs = sorted([b[0] for b in best_group])

                # Merge double bars within 16px (0.02 * w)
                for bx in raw_xs:
                    if not m_bars or bx - m_bars[-1] > max(16.0, 0.02 * w):
                        m_bars.append(bx)

                # Filter spurious elements outside printable region
                m_bars = [bx for bx in m_bars if sheet_left - 0.08 * w <= bx <= sheet_right + 0.04 * w]

                # Evidence-based minimum measure spacing: 0.042 * w
                m_bars_spaced: list[float] = []
                for bx in m_bars:
                    if not m_bars_spaced or bx - m_bars_spaced[-1] >= (0.042 * w):
                        m_bars_spaced.append(bx)
                m_bars = m_bars_spaced

                scale_f = (h / 2400.0) * (w / 1700.0)
                # Check if system starts with a pickup before m_bars[0]
                if m_bars and m_bars[0] > sheet_left + 0.05 * w:
                    is_full_meas = (m_bars[0] - sheet_left >= 0.14 * w)
                    if is_full_meas or not chords_below:
                        pickup_crop = bin_img[mel_y0:mel_y1, int(sheet_left - 0.02 * w):int(m_bars[0] - 10)]
                        if np.sum(pickup_crop > 0) > 80:
                            m_bars.insert(0, sheet_left)

                if is_boxed_page and scaled_page:
                    m_bars = clean_narrow_measures(m_bars, 0.14 * w)
                elif chords_below:
                    m_bars = clean_narrow_measures(
                        m_bars,
                        min_m_w=0.12 * w,
                        min_end_w=0.04 * w,
                        dashed_xs={db[0] for db in dashed_bars},
                        sheet_left=sheet_left,
                    )
                else:
                    m_bars = clean_narrow_measures(
                        m_bars,
                        min_m_w=0.118 * w,
                        min_end_w=0.04 * w,
                        dashed_xs={db[0] for db in dashed_bars},
                        sheet_left=sheet_left,
                        is_first_system=(s_idx == 0),
                        allow_repeat_ending=True,
                    )

                # Check if system continues to sheet_right
                if m_bars and m_bars[-1] < sheet_right - 0.12 * w:
                    if not is_tall_bar and sys_cand["span"] >= 0.50:
                        m_bars.append(sheet_right)
                    elif len(m_bars) == 2:
                        rem_crop = bin_img[
                            mel_y0:mel_y1,
                            int(m_bars[-1] + 25):int(min(sheet_right - 0.10 * w, m_bars[-1] + (m_bars[-1] - m_bars[0]) + 15)),
                        ]
                        if np.sum(rem_crop > 0) > 120 * scale_f:
                            m_bars.append(m_bars[-1] + (m_bars[-1] - m_bars[0]))
                    elif is_tall_bar:
                        rem_ink = np.sum(
                            bin_img[int(ym - 25):int(ym + 25), int(m_bars[-1] + 25):int(sheet_right + 10)] > 0
                        )
                        if rem_ink > 80:
                            m_bars.append(sheet_right)
            else:
                m_bars = list(np.linspace(sheet_left, sheet_right, 5))

            if len(m_bars) < 2:
                m_bars = [sheet_left, sheet_right]

            # Construct GMeasure objects
            measures: list[GMeasure] = []
            for m_i in range(len(m_bars) - 1):
                mx0 = m_bars[m_i]
                mx1 = m_bars[m_i + 1]

                left_double = any(0 < abs(b[0] - mx0) <= 16 for b in candidate_bars)
                right_double = any(0 < abs(b[0] - mx1) <= 16 for b in candidate_bars)

                measures.append(
                    GMeasure(
                        index_in_system=m_i,
                        x0=round(mx0 / float(w), 4),
                        x1=round(mx1 / float(w), 4),
                        left_double_bar=left_double,
                        right_double_bar=right_double,
                    )
                )

            # Chord band definition based on chords_below
            if chords_below:
                chord_y0_px = mel_y1
                chord_y1_px = min(y_bot, mel_y1 + int(round(1.85 * u_curr)))
            elif is_boxed_page:
                chord_y0_px = max(y_top, int(mel_y0 - max(45, int(0.035 * h))))
                chord_y1_px = min(mel_y1, int(mel_y0 + 23))
            else:
                chord_y0_px = max(y_top, int(mel_y0 - (60 if is_tall_bar else (int(round(1.85 * u_curr)) if scaled_page else 35))))
                chord_y1_px = mel_y0

            chord_band_norm: Optional[tuple[float, float]] = None
            if chord_y1_px > chord_y0_px:
                chord_band_norm = (round(chord_y0_px / float(h), 4), round(chord_y1_px / float(h), 4))

            chord_boxes = detect_chord_boxes(
                bin_img,
                y0_px=chord_y0_px,
                y1_px=chord_y1_px,
                measures=measures,
                img_w=w,
                img_h=h,
                chords_below=chords_below,
                is_boxed_page=is_boxed_page,
                scaled_page=scaled_page,
            )

            is_chord_only = False
            if not is_boxed_page:
                if len(chord_boxes) == 0 or is_tall_bar:
                    is_co, co_boxes = extract_chord_only_boxes(bin_img, mel_y0, mel_y1, measures, w, h)
                    if is_co and (len(chord_boxes) == 0 or len(co_boxes) >= len(chord_boxes) or is_tall_bar):
                        is_chord_only = True
                        chord_boxes = co_boxes
                        chord_band_norm = melody_band_norm

            # Lyric bands
            lyric_bands: list[tuple[float, float]] = []
            lyric_y0_px = mel_y1 if is_chord_only else (chord_y1_px if chords_below else mel_y1)
            lyric_y1_px = y_bot
            if lyric_y1_px > lyric_y0_px:
                lyric_proj = np.sum(bin_img[lyric_y0_px:lyric_y1_px, int(sheet_left):int(sheet_right)] > 0, axis=1)
                if len(lyric_proj) > 0 and np.max(lyric_proj) > 30:
                    lyr_peak = lyric_y0_px + int(np.argmax(lyric_proj))
                    lyric_bands.append((
                        round(max(lyric_y0_px, lyr_peak - 12) / float(h), 4),
                        round(min(lyric_y1_px, lyr_peak + 12) / float(h), 4),
                    ))

            # Exclude any chord box falling inside lyric bands
            if lyric_bands and chord_boxes:
                valid_chords = []
                for cb in chord_boxes:
                    cb_y_mid = (cb.bbox[1] + cb.bbox[3]) / 2.0
                    in_lyric = any(ly0 <= cb_y_mid <= ly1 for ly0, ly1 in lyric_bands)
                    if not in_lyric:
                        valid_chords.append(cb)
                chord_boxes = valid_chords

            # Full row bbox: guarantees containing melody, chords, lyrics, and non-overlapping
            sys_x0 = max(0.0, round((sheet_left - 10) / float(w), 4))
            sys_x1 = min(1.0, round((sheet_right + 10) / float(w), 4))
            sys_bbox: BBox = (
                sys_x0,
                round(y_top / float(h), 4),
                sys_x1,
                round(y_bot / float(h), 4),
            )

            diffs = [m.x1 - m.x0 for m in measures]
            regularity = float(np.std(diffs)) if len(diffs) > 1 else 0.0
            sys_conf = float(max(0.15, min(1.0, 1.0 - regularity * 3.5)))
            if not candidate_bars:
                sys_conf = min(sys_conf, 0.35)

            sys_notes: list[str] = []
            if is_chord_only:
                sys_notes.append("chord_only")
            elif chords_below:
                sys_notes.append("chords_below")

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
                    notes=sys_notes,
                )
            )

        # Page-level chord_only post-pass: if page is chord-only, ensure short/cadence rows also extract
        chord_only_sys_count = sum(1 for s in systems if "chord_only" in s.notes)
        if len(systems) > 0 and (chord_only_sys_count >= max(2, len(systems) // 2) or is_tall_bar):
            if "chord_only" not in notes:
                notes.append("chord_only")
            for s_idx, s in enumerate(systems):
                if "chord_only" not in s.notes:
                    my0_px = int(s.melody_band[0] * h)
                    my1_px = int(s.melody_band[1] * h)
                    is_co, co_boxes = extract_chord_only_boxes(bin_img, my0_px, my1_px, s.measures, w, h, force=True)
                    if is_co and co_boxes:
                        s.chord_boxes = co_boxes
                        s.chord_band = s.melody_band
                        s.notes.append("chord_only")

        global_conf = float(min(s.confidence for s in systems)) if systems else 0.0
        if has_text_lines:
            global_conf = min(global_conf, 0.55)

        return PageGeometry(
            page=page,
            width=orig_w,
            height=orig_h,
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
