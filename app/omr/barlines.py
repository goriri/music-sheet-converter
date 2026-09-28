"""Barline detection and measure box refinement using OpenCV.

Snaps Gemini-predicted measure bounding boxes to actual physical vertical bar lines
detected on the melody band of each system.
"""

from __future__ import annotations

import logging
from typing import Optional
import cv2
import numpy as np

from app.models import ParsedSheet, System, Measure

logger = logging.getLogger(__name__)


def _cluster_peaks(peaks: np.ndarray, max_gap: int = 12) -> list[int]:
    """Cluster column indices within max_gap and return cluster centers."""
    if len(peaks) == 0:
        return []
    clusters: list[list[int]] = []
    for p in peaks:
        if not clusters or p - clusters[-1][-1] > max_gap:
            clusters.append([int(p)])
        else:
            clusters[-1].append(int(p))
    return [int(np.mean(c)) for c in clusters]


def _merge_double_bars(barlines: list[int], double_bar_gap: int = 16) -> list[int]:
    """Merge double bar lines ('||') that appear within a small pixel distance."""
    if len(barlines) <= 1:
        return barlines
    merged: list[int] = []
    i = 0
    while i < len(barlines):
        if i + 1 < len(barlines) and barlines[i + 1] - barlines[i] <= double_bar_gap:
            # For double bars, use the midpoint position
            merged.append(int((barlines[i] + barlines[i + 1]) / 2.0))
            i += 2
        else:
            merged.append(barlines[i])
            i += 1
    return merged


def detect_system_barlines(
    gray: np.ndarray,
    sys_bbox: tuple[float, float, float, float],
    measures: list[Measure],
) -> list[int]:
    """Detect vertical bar lines in the melody band of a single system.

    Uses a sliding window search within the melody region to locate vertical barlines.
    Returns detected barline X coordinates in image pixels.
    """
    img_h, img_w = gray.shape[:2]
    expected_m = len(measures)

    # Melody band search range
    if measures:
        ym0_norm = min(m.bbox[1] for m in measures)
        ym1_norm = max(m.bbox[3] for m in measures)
        pad = 0.015
        search_y0 = max(0, int((ym0_norm - pad) * img_h))
        search_y1 = min(img_h, int((ym1_norm + pad) * img_h))
    else:
        sys_y0 = sys_bbox[1]
        sys_y1 = sys_bbox[3]
        search_y0 = max(0, int((sys_y0 + 0.15 * (sys_y1 - sys_y0)) * img_h))
        search_y1 = min(img_h, int((sys_y0 + 0.85 * (sys_y1 - sys_y0)) * img_h))

    if search_y1 - search_y0 < 15:
        return []

    best_bars: list[int] = []
    best_score = -1e9

    win_h = max(20, min(42, int((search_y1 - search_y0) * 0.45)))
    step = 2

    for wy0 in range(search_y0, max(search_y0 + 1, search_y1 - win_h), step):
        wy1 = wy0 + win_h
        crop = gray[wy0:wy1, :]
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kh = max(16, int(win_h * 0.70))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, kh))
        vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        proj = np.sum(vert, axis=0)
        thresh = kh * 255 * 0.40
        peaks = np.where(proj > thresh)[0]

        clustered = _cluster_peaks(peaks, max_gap=12)
        merged = _merge_double_bars(clustered, double_bar_gap=16)

        # Retain candidate barlines within the printable area
        valid = [x for x in merged if 0.03 * img_w <= x <= 0.98 * img_w]

        # Scoring
        if len(valid) == expected_m + 1:
            span = valid[-1] - valid[0]
            if span > 0.60 * img_w:
                diffs = np.diff(valid)
                regularity = -float(np.std(diffs))
                score = 1000.0 + regularity
            else:
                score = 50.0
        else:
            score = -10.0 * abs(len(valid) - (expected_m + 1))

        if score > best_score:
            best_score = score
            best_bars = valid

    return best_bars


def refine_measure_boxes(image: bytes, sheet: ParsedSheet, page: int) -> None:
    """Refine measure bounding boxes by snapping x-coordinates to detected bar lines.

    Never raises an exception; logs or appends warnings on failure or mismatch.
    """
    try:
        nparr = np.frombuffer(image, np.uint8)
        gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            sheet.warnings.append(f"Page {page + 1}: failed to decode image for barline detection.")
            return

        img_h, img_w = gray.shape[:2]

        for sys_idx, system in enumerate(sheet.systems):
            if system.page != page:
                continue
            measures = system.measures
            if not measures:
                continue

            M = len(measures)
            barlines = detect_system_barlines(gray, system.bbox, measures)

            # Target is M + 1 boundaries (b_0, b_1, ..., b_M)
            matched_boundaries: Optional[list[int]] = None

            if len(barlines) == M + 1:
                # Direct match if span is wide enough
                if barlines[-1] > barlines[0] and (barlines[-1] - barlines[0]) > 0.55 * img_w:
                    matched_boundaries = barlines
            elif len(barlines) > M + 1:
                # More barlines detected than expected; match closest to expected boundaries
                orig_boundaries = [int(measures[0].bbox[0] * img_w)]
                for m in measures:
                    orig_boundaries.append(int(m.bbox[2] * img_w))

                tolerance = int(0.06 * img_w)  # ~60px tolerance
                selected: list[int] = []
                valid_match = True

                for ob in orig_boundaries:
                    candidates = [b for b in barlines if abs(b - ob) <= tolerance]
                    if not candidates:
                        valid_match = False
                        break
                    best = min(candidates, key=lambda b: abs(b - ob))
                    if selected and best <= selected[-1]:
                        valid_match = False
                        break
                    selected.append(best)

                if valid_match and len(selected) == M + 1:
                    matched_boundaries = selected

            if matched_boundaries is not None:
                # Snap measure x-ranges
                for i, m in enumerate(measures):
                    new_x0 = max(0.0, min(1.0, matched_boundaries[i] / float(img_w)))
                    new_x1 = max(0.0, min(1.0, matched_boundaries[i + 1] / float(img_w)))
                    # Preserve original y-range
                    m.bbox = (round(new_x0, 4), m.bbox[1], round(new_x1, 4), m.bbox[3])
            else:
                sheet.warnings.append(
                    f"Page {page + 1} System {sys_idx + 1}: detected {len(barlines)} barlines "
                    f"for {M} measures (expected {M + 1}); kept Gemini measure boundaries."
                )

    except Exception as exc:
        logger.warning("Error in refine_measure_boxes: %s", exc)
        sheet.warnings.append(f"Page {page + 1}: barline refinement exception: {exc}")
