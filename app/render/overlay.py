"""Overlay accompaniment strips onto original sheet music pages.

Slices original page images between system bounding boxes at the whitest rows,
inserts movable-do Jianpu accompaniment strips with RH and LH rows, chord labels,
header information, and exports sight-readable multi-page PDFs.
"""
from __future__ import annotations

import io
import math
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models import Arrangement, ChordSymbol, Event, KeyChange, ParsedSheet, ResolvedChord, System
from app.render.jianpu import (
    compute_stack_height,
    draw_augmentation_dot,
    draw_chord_stack,
    draw_rest,
    draw_sustain_dash,
    draw_tie,
    draw_underline_beam,
    get_font,
    midi_to_jianpu,
)


def find_whitest_row(
    img_gray: np.ndarray,
    nominal_y: int,
    delta: int,
    min_y: int,
    max_y: int,
) -> int:
    """Find row index with maximum average brightness within [nominal_y - delta, nominal_y + delta]."""
    y_start = max(min_y, nominal_y - delta)
    y_end = min(max_y, nominal_y + delta + 1)
    if y_end <= y_start:
        return max(0, min(nominal_y, img_gray.shape[0] - 1))

    sub = img_gray[y_start:y_end, :]
    row_means = sub.mean(axis=1)
    best_offset = int(np.argmax(row_means))
    return y_start + best_offset


def _cut_run_priority(r: Tuple[int, int], nom_offset: float) -> Tuple[int, float]:
    length = r[1] - r[0] + 1
    center = (r[0] + r[1]) / 2.0
    dist = abs(center - nom_offset)
    return (length, -dist)


def find_system_cut_row(
    img_gray: np.ndarray,
    curr_y1: int,
    next_y0: int,
    nominal_y: Optional[int] = None,
    delta: Optional[int] = None,
    ink_threshold: int = 140,
) -> int:
    """Find the best cut row between systems to prevent slicing through musical notation.

    Binarizes image (ink = gray < ink_threshold), searches the region between systems,
    finds runs of near-empty rows (ink <= max(2, 0.002 * w)), and cuts at the center
    of the longest run (breaking ties by distance to nominal_y).
    Falls back to argmax of brightness if no empty run is found.
    """
    h, w = img_gray.shape
    if delta is None:
        delta = int(round(0.03 * h))
    if nominal_y is None:
        nominal_y = int(round((curr_y1 + next_y0) / 2.0))

    if curr_y1 < next_y0:
        y_start = max(0, curr_y1)
        y_end = min(h, next_y0)
    else:
        y_start = max(0, nominal_y - delta)
        y_end = min(h, nominal_y + delta + 1)

    if y_end <= y_start:
        return max(0, min(nominal_y, h - 1))

    sub = img_gray[y_start:y_end, :]
    ink_mask = sub < ink_threshold
    ink_counts = ink_mask.sum(axis=1)
    ink_limit = max(2, int(round(0.002 * w)))
    is_empty = ink_counts <= ink_limit

    runs: list[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for idx, empty in enumerate(is_empty):
        if empty:
            if not in_run:
                in_run = True
                run_start = idx
        else:
            if in_run:
                in_run = False
                runs.append((run_start, idx - 1))
    if in_run:
        runs.append((run_start, len(is_empty) - 1))

    if runs:
        nom_offset = float(nominal_y - y_start)
        best_run = max(runs, key=lambda r: _cut_run_priority(r, nom_offset))
        best_offset = int(round((best_run[0] + best_run[1]) / 2.0))
        return y_start + best_offset

    row_means = sub.mean(axis=1)
    best_offset = int(np.argmax(row_means))
    return y_start + best_offset


def images_to_pdf_bytes(images: Sequence[Image.Image]) -> bytes:
    """Embed images into multi-page PDF with exact A4 width (595.0 pt) and JPEG q90 4:4:4."""
    if not images:
        return b""
    import pymupdf

    doc = pymupdf.open()
    for img in images:
        w_px, h_px = img.size
        page_w = 595.0
        page_h = 595.0 * (float(h_px) / float(w_px))
        page = doc.new_page(width=page_w, height=page_h)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, subsampling=0)
        page.insert_image(pymupdf.Rect(0, 0, page_w, page_h), stream=buf.getvalue())
    pdf_bytes = doc.tobytes(deflate=True)
    doc.close()
    return pdf_bytes



def calc_event_x(mx0: float, mw: float, pad: float, onset: float, beats: float) -> float:
    """Calculate horizontal pixel coordinate of an event inside a measure."""
    return mx0 + pad + (onset / beats) * (mw - 2.0 * pad)


def extract_key_change_label(raw: str) -> str:
    """Extract a concise key change label like '1=G' from raw text."""
    if "1=" in raw:
        idx = raw.find("1=")
        part = raw[idx:].split()[0].split("(")[0].strip()
        if part:
            return part
    elif "(" in raw and ")" in raw:
        in_paren = raw.split("(")[1].split(")")[0].strip()
        return f"1={in_paren}"
    return raw


def get_active_chord(
    chords: Sequence[ResolvedChord],
    onset: float,
) -> Optional[ResolvedChord]:
    """Find the chord active at the given event onset (0-based beat)."""
    if not chords:
        return None
    event_beat = onset + 1.0
    active = None
    for c in chords:
        if c.beat <= event_beat + 0.01:
            active = c
        else:
            break
    if active is None:
        active = chords[0]
    return active


def draw_measure_chords(
    draw: ImageDraw.ImageDraw,
    chords: Sequence[ResolvedChord],
    mx0: float,
    mx1: float,
    beats: float,
    chord_y: float,
    min_x: float,
    scale: float = 1.0,
) -> None:
    """Draw chord names in the chord row for a measure with collision avoidance.

    - Color: #B00000 (176, 0, 0), bold.
    - Default size 18 (~0.9x jianpu digit height).
    - If collision detected, shrink to 70% (size 13) and shift right.
    - Never cross measure right bar line (mx1) unless unavoidable.
    """
    if not chords:
        return

    font_normal = get_font(size=int(round(18 * scale)), bold=True)
    font_small = get_font(size=int(round(13 * scale)), bold=True)
    chord_color = (176, 0, 0)
    mw = mx1 - mx0
    pad = max(4.0 * scale, mw * 0.04)

    # 1. Compute initial nominal x positions
    items: list[dict] = []
    for c in chords:
        onset = max(0.0, c.beat - 1.0)
        nom_x = calc_event_x(mx0, mw, pad, onset, beats)
        items.append({
            "name": c.name,
            "beat": c.beat,
            "nom_x": nom_x,
            "x": nom_x,
            "font": font_normal,
            "w": 0.0,
        })

    n = len(items)
    # 2. Check collision against next nominal position or measure bar line
    for i in range(n):
        tb = draw.textbbox((0, 0), items[i]["name"], font=items[i]["font"])
        w_text = float(tb[2] - tb[0])
        next_x = items[i + 1]["nom_x"] if i + 1 < n else (mx1 - 2.0 * scale)
        if items[i]["nom_x"] + w_text + 3.0 * scale > next_x or items[i]["nom_x"] + w_text > mx1 - 2.0 * scale:
            items[i]["font"] = font_small
            if i + 1 < n:
                items[i + 1]["font"] = font_small

    # 3. Determine placed x positions (shift right if colliding with previous item)
    cur_min_x = min_x
    for i in range(n):
        tb = draw.textbbox((0, 0), items[i]["name"], font=items[i]["font"])
        w_text = float(tb[2] - tb[0])
        items[i]["w"] = w_text

        placed_x = max(items[i]["nom_x"], cur_min_x)
        items[i]["x"] = placed_x
        cur_min_x = placed_x + w_text + 3.0 * scale

    # 4. Enforce: never cross measure right bar line unless unavoidable
    if items and items[-1]["x"] + items[-1]["w"] > mx1 - 2.0 * scale:
        items[-1]["x"] = max(min_x, min(items[-1]["x"], mx1 - 2.0 * scale - items[-1]["w"]))
        for i in range(n - 2, -1, -1):
            max_allowed = items[i + 1]["x"] - items[i]["w"] - 3.0 * scale
            if items[i]["x"] > max_allowed:
                items[i]["x"] = max(min_x, max_allowed)

    # 5. Render chords
    for item in items:
        draw.text((item["x"], chord_y), item["name"], font=item["font"], fill=chord_color)


def compute_event_glyph_width(
    draw: ImageDraw.ImageDraw,
    ev: Event,
    tonic_pc: int,
    hand: Literal["rh", "lh"],
    active_chord: Optional[ResolvedChord],
    key_name: Optional[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    acc_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    scale: float = 1.0,
) -> float:
    """Compute the horizontal glyph extent of an event (accidentals + digit + dots)."""
    if not ev.notes:
        tb = draw.textbbox((0, 0), "0", font=font)
        return float(tb[2] - tb[0])

    j_notes = [
        midi_to_jianpu(
            n.midi,
            tonic_pc,
            hand=hand,
            finger=n.finger,
            chord=active_chord,
            key_name=key_name,
        )
        for n in ev.notes
    ]
    max_w = 0.0
    for jn in j_notes:
        w = 0.0
        if jn.accidental:
            tb_acc = draw.textbbox((0, 0), jn.accidental, font=acc_font)
            w += float(tb_acc[2] - tb_acc[0]) + 1.0 * scale
        tb_deg = draw.textbbox((0, 0), str(jn.degree), font=font)
        w += float(tb_deg[2] - tb_deg[0])
        if w > max_w:
            max_w = w

    if hand == "rh" and len(j_notes) > 1 and any(n.finger is not None for n in ev.notes):
        max_w += 10.0 * scale

    if abs(ev.duration - 1.5) < 0.05 or abs(ev.duration - 0.75) < 0.05:
        max_w += 7.0 * scale

    return max_w


def layout_measure_row(
    draw: ImageDraw.ImageDraw,
    events: Sequence[Event],
    mx0: float,
    mx1: float,
    pad: float,
    beats: float,
    tonic_pc: int,
    hand: Literal["rh", "lh"],
    chords: Sequence[ResolvedChord],
    key_name: Optional[str],
    base_font_size: int = 20,
    min_font_size: int = 15,
    scale: float = 1.0,
) -> Tuple[
    list[float],
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
]:
    """Calculate collision-aware horizontal layout for events in a measure row.

    Maintains time-proportional target x while enforcing min_gap.
    If overflowing right boundary, compresses gaps proportionally down to min_gap,
    then scales font size down (up to 75%, size 15).
    """
    if not events:
        font_default = get_font(size=base_font_size, bold=False)
        acc_default = get_font(size=max(int(round(12 * scale)), int(round(16 * scale))), bold=False)
        finger_default = get_font(size=max(int(round(9 * scale)), int(round(11 * scale))), bold=True)
        return [], font_default, acc_default, finger_default

    mw = mx1 - mx0
    if len(events) == 1:
        ev = events[0]
        font_single = get_font(size=base_font_size, bold=False)
        acc_single = get_font(size=max(int(round(12 * scale)), int(round(16 * scale))), bold=False)
        finger_single = get_font(size=max(int(round(9 * scale)), int(round(11 * scale))), bold=True)
        chord_single = get_active_chord(chords, ev.onset)
        w = compute_event_glyph_width(
            draw,
            ev,
            tonic_pc,
            hand,
            chord_single,
            key_name,
            font_single,
            acc_single,
            scale=scale,
        )
        nom_x = calc_event_x(mx0, mw, pad, ev.onset, beats)
        placed_x = max(mx0 + pad, min(nom_x, mx1 - pad - w))
        return [placed_x], font_single, acc_single, finger_single

    target_x = [calc_event_x(mx0, mw, pad, ev.onset, beats) for ev in events]

    for sz in range(base_font_size, min_font_size - 1, -1):
        ratio = float(sz) / float(base_font_size) if base_font_size > 0 else 1.0
        font = get_font(size=sz, bold=False)
        acc_font = get_font(size=max(int(round(12 * scale)), int(round(16 * scale * ratio))), bold=False)
        finger_font = get_font(size=max(int(round(9 * scale)), int(round(11 * scale * ratio))), bold=True)
        tb_d = draw.textbbox((0, 0), "1", font=font)
        digit_w = float(tb_d[2] - tb_d[0])
        min_gap = max(2.5 * scale, 0.25 * digit_w)

        extents = [
            compute_event_glyph_width(
                draw,
                ev,
                tonic_pc,
                hand,
                get_active_chord(chords, ev.onset),
                key_name,
                font,
                acc_font,
                scale=scale,
            )
            for ev in events
        ]

        pos = [target_x[0]]
        for i in range(1, len(events)):
            pos.append(max(target_x[i], pos[i - 1] + extents[i - 1] + min_gap))

        overflow = (pos[-1] + extents[-1]) - (mx1 - pad)

        if overflow <= 0:
            return pos, font, acc_font, finger_font

        gaps = [pos[i + 1] - (pos[i] + extents[i]) for i in range(len(events) - 1)]
        slacks = [max(0.0, g - min_gap) for g in gaps]
        total_slack = sum(slacks)

        if total_slack >= overflow:
            new_pos = [pos[0]]
            for i in range(len(events) - 1):
                reduction = overflow * (slacks[i] / total_slack) if total_slack > 0 else 0.0
                new_gap = gaps[i] - reduction
                new_pos.append(new_pos[i] + extents[i] + new_gap)
            return new_pos, font, acc_font, finger_font

    # Fallback at min_font_size: compress all gaps down to min_gap
    ratio = float(min_font_size) / float(base_font_size) if base_font_size > 0 else 1.0
    font = get_font(size=min_font_size, bold=False)
    acc_font = get_font(size=max(int(round(12 * scale)), int(round(16 * scale * ratio))), bold=False)
    finger_font = get_font(size=max(int(round(9 * scale)), int(round(11 * scale * ratio))), bold=True)
    tb_d = draw.textbbox((0, 0), "1", font=font)
    digit_w = float(tb_d[2] - tb_d[0])
    min_gap = max(2.5 * scale, 0.25 * digit_w)
    extents = [
        compute_event_glyph_width(
            draw,
            ev,
            tonic_pc,
            hand,
            get_active_chord(chords, ev.onset),
            key_name,
            font,
            acc_font,
            scale=scale,
        )
        for ev in events
    ]
    new_pos = [target_x[0]]
    for i in range(len(events) - 1):
        new_pos.append(new_pos[i] + extents[i] + min_gap)

    if new_pos[-1] + extents[-1] > mx1 - 2.0 * scale:
        shift = (new_pos[-1] + extents[-1]) - (mx1 - 2.0 * scale)
        max_shift = new_pos[0] - (mx0 + 2.0 * scale)
        actual_shift = min(shift, max_shift)
        if actual_shift > 0:
            new_pos = [p - actual_shift for p in new_pos]

    if new_pos[-1] + extents[-1] > mx1 - 2.0 * scale:
        new_pos[-1] = mx1 - 2.0 * scale - extents[-1]
        for i in range(len(events) - 2, -1, -1):
            if new_pos[i] + extents[i] > new_pos[i + 1] - 1.0 * scale:
                new_pos[i] = new_pos[i + 1] - 1.0 * scale - extents[i]

    return new_pos, font, acc_font, finger_font



def make_header_band(
    width: int,
    height: int,
    sheet: Optional[ParsedSheet],
    arrangement: Arrangement,
    scale: float = 1.0,
) -> Image.Image:
    """Create a white header band prepended at the top of page 1."""
    band = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(band)

    diff_name = {
        "beginner": "初级",
        "intermediate": "中级",
        "advanced": "高级",
    }.get(arrangement.difficulty, "中级")

    inst_label = "钢琴伴奏"
    if arrangement.instrument == "guitar":
        inst_label = "吉他伴奏"
    elif arrangement.instrument == "ukulele":
        inst_label = "尤克里里伴奏"

    title = f"{inst_label} · {diff_name} · 1={arrangement.start_key}"
    orig_key = sheet.header.original_key if sheet and sheet.header else None
    if orig_key and orig_key != arrangement.start_key:
        title += f" (原谱 1={orig_key})"

    legend = "右手：无点 = 中央C所在八度起；左手：无点 = 低一个八度；蓝色小字 = 指法"

    font_title = get_font(size=int(round(16 * scale)), bold=True)
    font_legend = get_font(size=int(round(11 * scale)), bold=False)

    x_pos = max(40.0 * scale, width * 0.05)
    y_title = max(6.0 * scale, height * 0.16)
    y_legend = max(28.0 * scale, height * 0.56)

    draw.text((x_pos, y_title), title, font=font_title, fill=(40, 40, 40))
    draw.text((x_pos, y_legend), legend, font=font_legend, fill=(100, 100, 100))

    return band


def render_system_strip(
    system: System,
    page_width: int,
    arrangement: Arrangement,
    sheet: Optional[ParsedSheet] = None,
    scale: float = 1.0,
) -> Image.Image:
    """Render the Jianpu accompaniment strip for a single system."""
    arr_map = {m.measure_index: m for m in arrangement.measures}
    key_change_map = {kc.at_measure: kc for kc in sheet.key_changes} if sheet and sheet.key_changes else {}

    # Find max stack heights for RH and LH in this system
    max_rh = 1
    max_lh = 1
    for m in system.measures:
        arr_m = arr_map.get(m.index)
        if arr_m:
            for ev in arr_m.rh:
                max_rh = max(max_rh, len(ev.notes))
            for ev in arr_m.lh:
                max_lh = max(max_lh, len(ev.notes))

    line_spacing = 18.0 * scale
    chord_y = 6.0 * scale
    chord_row_h = 24.0 * scale

    # Vertical coordinates:
    # Dedicated chord-name row at top (y = 0..24)
    # RH row: y_top_rh leaves room for RH fingers and octave dots above top note
    y_top_rh = chord_row_h + 18.0 * scale
    base_y_rh = y_top_rh + (max_rh - 1) * line_spacing

    gap_between_hands = 26.0 * scale
    # In RH, lowest note bottom is base_y_rh + 24, beams can reach base_y_rh + 32
    top_lh = base_y_rh + 32.0 * scale + gap_between_hands
    y_top_lh = top_lh + 8.0 * scale
    base_y_lh = y_top_lh + (max_lh - 1) * line_spacing

    # In LH, bottom dots reach base_y_lh + 25, beams reach base_y_lh + 35, fingers reach base_y_lh + 48
    strip_h = int(math.ceil(base_y_lh + 60.0 * scale))

    strip = Image.new("RGB", (page_width, strip_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(strip)

    # Fonts
    digit_font = get_font(size=int(round(20 * scale)), bold=False)
    acc_font = get_font(size=int(round(16 * scale)), bold=False)
    finger_font = get_font(size=int(round(11 * scale)), bold=True)
    label_font = get_font(size=int(round(15 * scale)), bold=True)

    # Draw clean 'R' and 'L' labels in left margin (no cryptic 0:>=C4 / 0:>=C3)
    draw.text((22 * scale, base_y_rh + 3.0 * scale), "R", font=label_font, fill=(70, 70, 70))
    draw.text((22 * scale, base_y_lh + 3.0 * scale), "L", font=label_font, fill=(70, 70, 70))

    # Draw measure contents, chord row, and bar lines
    measures = system.measures
    bar_color = (190, 190, 190)  # Thin grey
    bar_w = max(1, int(round(1 * scale)))

    for m_idx, m in enumerate(measures):
        arr_m = arr_map.get(m.index)
        beats = m.beats or 4.0

        if m.bbox:
            mx0 = m.bbox[0] * page_width
            mx1 = m.bbox[2] * page_width
        else:
            mx0 = 60.0 * scale + (m_idx / max(1, len(measures))) * (page_width - 120.0 * scale)
            mx1 = 60.0 * scale + ((m_idx + 1) / max(1, len(measures))) * (page_width - 120.0 * scale)

        # Thin grey bar line at measure start
        draw.line([(mx0, int(round(4 * scale))), (mx0, strip_h - int(round(4 * scale)))], fill=bar_color, width=bar_w)
        if m_idx == len(measures) - 1:
            draw.line([(mx1, int(round(4 * scale))), (mx1, strip_h - int(round(4 * scale)))], fill=bar_color, width=bar_w)

        mw = mx1 - mx0
        pad = max(6.0 * scale, mw * 0.04)

        # ---------------------------------------------------- Chord row & Key-change marker
        key_marker_right = mx0 + 4.0 * scale
        if m.index in key_change_map:
            kc = key_change_map[m.index]
            if arr_m and arr_m.key_name:
                marker_text = f"1={arr_m.key_name}"
            else:
                marker_text = extract_key_change_label(kc.raw)

            kc_font = get_font(size=int(round(13 * scale)), bold=True)
            bx0 = mx0 + 4.0 * scale
            text_x = bx0 + 4.0 * scale
            text_y = chord_y
            tb = draw.textbbox((text_x, text_y), marker_text, font=kc_font)
            bx1 = tb[2] + 4.0 * scale
            by0 = tb[1] - 2.0 * scale
            by1 = tb[3] + 2.0 * scale
            # Red outline box
            draw.rectangle([bx0, by0, bx1, by1], outline=(176, 0, 0), width=bar_w)
            # Red text
            draw.text((text_x, text_y), marker_text, font=kc_font, fill=(176, 0, 0))
            key_marker_right = bx1 + 4.0 * scale

        if arr_m and arr_m.chords:
            draw_measure_chords(
                draw=draw,
                chords=arr_m.chords,
                mx0=mx0,
                mx1=mx1,
                beats=beats,
                chord_y=chord_y,
                min_x=key_marker_right,
                scale=scale,
            )

        if not arr_m:
            continue

        # ---------------------------------------------------- RH events
        rh_pos, rh_digit_font, rh_acc_font, rh_finger_font = layout_measure_row(
            draw=draw,
            events=arr_m.rh,
            mx0=mx0,
            mx1=mx1,
            pad=pad,
            beats=beats,
            tonic_pc=arr_m.tonic_pc,
            hand="rh",
            chords=arr_m.chords,
            key_name=arr_m.key_name,
            base_font_size=int(round(20 * scale)),
            min_font_size=int(round(15 * scale)),
            scale=scale,
        )

        rh_layouts = []
        for ev_idx, ev in enumerate(arr_m.rh):
            active_chord = get_active_chord(arr_m.chords, ev.onset)
            ev_x = rh_pos[ev_idx]
            if not ev.notes:
                _, t_y, b_y = draw_rest(draw, ev_x, base_y_rh, rh_digit_font)
            else:
                j_notes = [
                    midi_to_jianpu(
                        n.midi,
                        arr_m.tonic_pc,
                        hand="rh",
                        finger=n.finger,
                        chord=active_chord,
                        key_name=arr_m.key_name,
                    )
                    for n in ev.notes
                ]

                layout = draw_chord_stack(
                    draw,
                    ev_x,
                    base_y_rh,
                    j_notes,
                    rh_digit_font,
                    rh_acc_font,
                    rh_finger_font,
                    hand="rh",
                    line_spacing=line_spacing,
                    draw_finger=True,
                    scale=scale,
                )
                rh_layouts.append((ev, layout))

                # Extra sustain dashes
                if ev.duration >= 2.0:
                    extra_beats = int(math.floor(ev.duration)) - 1
                    for k in range(1, extra_beats + 1):
                        dash_x = calc_event_x(mx0, mw, pad, ev.onset + k, beats)
                        if dash_x < mx1 - pad:
                            draw_sustain_dash(draw, dash_x, base_y_rh, rh_digit_font)

                # Augmentation dot for dotted values
                if abs(ev.duration - 1.5) < 0.05 or abs(ev.duration - 0.75) < 0.05:
                    dot_x = ev_x + layout.max_w + 4.0 * scale
                    draw_augmentation_dot(draw, dot_x, base_y_rh + 16.0 * scale, scale=scale)

        # RH ties
        for i, (ev, lay) in enumerate(rh_layouts):
            if ev.tie_to_next and i + 1 < len(rh_layouts):
                next_lay = rh_layouts[i + 1][1]
                draw_tie(draw, lay.base_x + 8.0 * scale, next_lay.base_x + 2.0 * scale, lay.top_y - 2.0 * scale, scale=scale)

        # RH Underlines (beaming per beat)
        rh_sub_beat = [x for x in rh_layouts if x[0].duration < 1.0]
        rh_beats_dict: dict[int, list] = {}
        for ev, lay in rh_sub_beat:
            b_idx = int(math.floor(ev.onset))
            rh_beats_dict.setdefault(b_idx, []).append((ev, lay))

        for b_idx, ev_lays in rh_beats_dict.items():
            max_bot_y = max(lay.bottom_y for _, lay in ev_lays)
            b_y1 = max_bot_y + 4.0 * scale
            b_y2 = b_y1 + 4.0 * scale

            b1_events = [x for x in ev_lays if x[0].duration <= 0.75]
            if b1_events:
                x_start = b1_events[0][1].base_x - 2.0 * scale
                x_end = b1_events[-1][1].base_x + b1_events[-1][1].max_w
                draw_underline_beam(draw, x_start, x_end, b_y1, scale=scale)

            b2_events = [x for x in ev_lays if x[0].duration <= 0.375]
            if b2_events:
                x_start = b2_events[0][1].base_x - 2.0 * scale
                x_end = b2_events[-1][1].base_x + b2_events[-1][1].max_w
                draw_underline_beam(draw, x_start, x_end, b_y2, scale=scale)

        # ---------------------------------------------------- LH events
        lh_pos, lh_digit_font, lh_acc_font, lh_finger_font = layout_measure_row(
            draw=draw,
            events=arr_m.lh,
            mx0=mx0,
            mx1=mx1,
            pad=pad,
            beats=beats,
            tonic_pc=arr_m.tonic_pc,
            hand="lh",
            chords=arr_m.chords,
            key_name=arr_m.key_name,
            base_font_size=int(round(20 * scale)),
            min_font_size=int(round(15 * scale)),
            scale=scale,
        )

        lh_layouts = []
        for ev_idx, ev in enumerate(arr_m.lh):
            active_chord = get_active_chord(arr_m.chords, ev.onset)
            ev_x = lh_pos[ev_idx]
            if not ev.notes:
                draw_rest(draw, ev_x, base_y_lh, lh_digit_font)
            else:
                j_notes = [
                    midi_to_jianpu(
                        n.midi,
                        arr_m.tonic_pc,
                        hand="lh",
                        finger=n.finger,
                        chord=active_chord,
                        key_name=arr_m.key_name,
                    )
                    for n in ev.notes
                ]

                layout = draw_chord_stack(
                    draw,
                    ev_x,
                    base_y_lh,
                    j_notes,
                    lh_digit_font,
                    lh_acc_font,
                    lh_finger_font,
                    hand="lh",
                    line_spacing=line_spacing,
                    draw_finger=False,
                    scale=scale,
                )
                lh_layouts.append((ev, layout))

                if ev.duration >= 2.0:
                    extra_beats = int(math.floor(ev.duration)) - 1
                    for k in range(1, extra_beats + 1):
                        dash_x = calc_event_x(mx0, mw, pad, ev.onset + k, beats)
                        if dash_x < mx1 - pad:
                            draw_sustain_dash(draw, dash_x, base_y_lh, lh_digit_font)

                if abs(ev.duration - 1.5) < 0.05 or abs(ev.duration - 0.75) < 0.05:
                    dot_x = ev_x + layout.max_w + 4.0 * scale
                    draw_augmentation_dot(draw, dot_x, base_y_lh + 16.0 * scale, scale=scale)

        # LH ties
        for i, (ev, lay) in enumerate(lh_layouts):
            if ev.tie_to_next and i + 1 < len(lh_layouts):
                next_lay = lh_layouts[i + 1][1]
                draw_tie(draw, lay.base_x + 8.0 * scale, next_lay.base_x + 2.0 * scale, lay.top_y - 2.0 * scale, scale=scale)

        # LH Underlines (beaming per beat)
        lh_sub_beat = [x for x in lh_layouts if x[0].duration < 1.0]
        lh_beats_dict: dict[int, list] = {}
        for ev, lay in lh_sub_beat:
            b_idx = int(math.floor(ev.onset))
            lh_beats_dict.setdefault(b_idx, []).append((ev, lay))

        event_bottom_y: dict[float, float] = {}

        for b_idx, ev_lays in lh_beats_dict.items():
            max_bot_y = max(lay.bottom_y for _, lay in ev_lays)
            b_y1 = max_bot_y + 4.0 * scale
            b_y2 = b_y1 + 4.0 * scale

            b1_events = [x for x in ev_lays if x[0].duration <= 0.75]
            if b1_events:
                x_start = b1_events[0][1].base_x - 2.0 * scale
                x_end = b1_events[-1][1].base_x + b1_events[-1][1].max_w
                draw_underline_beam(draw, x_start, x_end, b_y1, scale=scale)
                for ev, _ in b1_events:
                    event_bottom_y[ev.onset] = b_y1 + 1.0 * scale

            b2_events = [x for x in ev_lays if x[0].duration <= 0.375]
            if b2_events:
                x_start = b2_events[0][1].base_x - 2.0 * scale
                x_end = b2_events[-1][1].base_x + b2_events[-1][1].max_w
                draw_underline_beam(draw, x_start, x_end, b_y2, scale=scale)
                for ev, _ in b2_events:
                    event_bottom_y[ev.onset] = b_y2 + 1.0 * scale

        # Draw LH fingers cleanly BELOW all beams and dots
        for ev, lay in lh_layouts:
            if not ev.notes:
                continue
            f_num = ev.notes[0].finger
            if f_num is not None:
                if ev.onset in event_bottom_y:
                    f_y = event_bottom_y[ev.onset] + 4.0 * scale
                else:
                    f_y = lay.bottom_y + 4.0 * scale
                draw.text((lay.base_x + 2.0 * scale, f_y), str(f_num), font=lh_finger_font, fill=(25, 118, 210))

    return strip



def render_page(
    page_img: Image.Image,
    systems: Sequence[System],
    sheet: ParsedSheet,
    arrangement: Arrangement,
    is_first_page: bool,
    scale: float = 1.0,
) -> list[Image.Image]:
    """Overlay accompaniment onto a single page and split if height > 1.6x original."""
    w, h = page_img.size
    orig_h = h

    header_band = make_header_band(w, int(round(0.05 * h)), sheet, arrangement, scale=scale) if is_first_page else None

    if not systems:
        if header_band is not None:
            res = Image.new("RGB", (w, h + header_band.height), color=(255, 255, 255))
            res.paste(header_band, (0, 0))
            res.paste(page_img, (0, header_band.height))
            return [res]
        return [page_img.copy()]

    sorted_systems = sorted(systems, key=lambda s: s.bbox[1] if s.bbox else 0.0)
    k = len(sorted_systems)

    # Calculate cuts on original page image (zero drawing on page_img)
    gray = np.array(page_img.convert("L"))

    delta = int(round(0.03 * h))
    cuts: list[int] = []
    for i in range(k - 1):
        s_curr = sorted_systems[i]
        s_next = sorted_systems[i + 1]
        curr_y1 = int(round((s_curr.bbox[3] if s_curr.bbox else (i + 1) / k) * h))
        next_y0 = int(round((s_next.bbox[1] if s_next.bbox else (i + 1) / k) * h))
        nominal_cut = int(round((curr_y1 + next_y0) / 2.0))
        cut_y = find_system_cut_row(gray, curr_y1, next_y0, nominal_y=nominal_cut, delta=delta)
        cuts.append(cut_y)

    last_sys = sorted_systems[-1]
    last_y1 = int(round((last_sys.bbox[3] if last_sys.bbox else 1.0) * h))
    nominal_last_cut = min(h, last_y1 + int(round(0.015 * h)))
    cut_last = find_system_cut_row(gray, last_y1, min(h, last_y1 + 2 * delta), nominal_y=nominal_last_cut, delta=delta)
    cuts.append(cut_last)

    page_slices: list[Image.Image] = []
    y_prev = 0
    for cut_y in cuts:
        page_slices.append(page_img.crop((0, y_prev, w, cut_y)))
        y_prev = cut_y
    page_slices.append(page_img.crop((0, y_prev, w, h)))

    strips = [render_system_strip(s, w, arrangement, sheet, scale=scale) for s in sorted_systems]

    header_h = header_band.height if header_band is not None else 0
    total_h = header_h + sum(sl.height for sl in page_slices) + sum(st.height for st in strips)

    if total_h <= 1.6 * orig_h or k <= 1:
        res = Image.new("RGB", (w, total_h), color=(255, 255, 255))
        cur_y = 0
        if header_band is not None:
            res.paste(header_band, (0, cur_y))
            cur_y += header_band.height
        for i in range(k):
            sl = page_slices[i]
            st = strips[i]
            res.paste(sl, (0, cur_y))
            cur_y += sl.height
            res.paste(st, (0, cur_y))
            cur_y += st.height
        res.paste(page_slices[k], (0, cur_y))
        return [res]

    # Split into two subpages at system boundary
    split_idx = k // 2

    # Part 1: systems 0 .. split_idx - 1
    p1_slices = page_slices[:split_idx]
    p1_strips = strips[:split_idx]
    p1_h = header_h + sum(sl.height for sl in p1_slices) + sum(st.height for st in p1_strips)
    res1 = Image.new("RGB", (w, p1_h), color=(255, 255, 255))
    cur_y = 0
    if header_band is not None:
        res1.paste(header_band, (0, cur_y))
        cur_y += header_band.height
    for sl, st in zip(p1_slices, p1_strips):
        res1.paste(sl, (0, cur_y))
        cur_y += sl.height
        res1.paste(st, (0, cur_y))
        cur_y += st.height

    # Part 2: systems split_idx .. k - 1 + bottom slice
    p2_slices = page_slices[split_idx:k]
    p2_strips = strips[split_idx:k]
    top_pad = int(round(20 * scale))
    p2_h = top_pad + sum(sl.height for sl in p2_slices) + sum(st.height for st in p2_strips) + page_slices[k].height
    res2 = Image.new("RGB", (w, p2_h), color=(255, 255, 255))
    cur_y = top_pad
    for sl, st in zip(p2_slices, p2_strips):
        res2.paste(sl, (0, cur_y))
        cur_y += sl.height
        res2.paste(st, (0, cur_y))
        cur_y += st.height
    res2.paste(page_slices[k], (0, cur_y))

    return [res1, res2]


def render_pages(
    pages: list[bytes],
    sheet: ParsedSheet,
    arrangement: Arrangement,
) -> list[Image.Image]:
    """Render all pages with overlaid accompaniment strips.

    Returns a list of PIL Images (each representing an output PDF page).
    """
    if arrangement.instrument == "ukulele":
        from app.render.ukulele import render_uke_pages

        return render_uke_pages(pages, sheet, arrangement)

    output_pages: list[Image.Image] = []

    for page_idx, page_bytes in enumerate(pages):
        page_img = Image.open(io.BytesIO(page_bytes)).convert("RGB")
        native_w, native_h = page_img.size
        s_factor = max(1.0, min(3.0, 2400.0 / float(native_w)))

        if abs(s_factor - 1.0) > 1e-4:
            target_w = int(round(native_w * s_factor))
            target_h = int(round(native_h * s_factor))
            from PIL import ImageFilter

            page_img = page_img.resize((target_w, target_h), Image.Resampling.LANCZOS).filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=3)
            )
        else:
            s_factor = 1.0

        page_systems = [s for s in sheet.systems if s.page == page_idx]
        is_first = (page_idx == 0)

        rendered_subpages = render_page(
            page_img=page_img,
            systems=page_systems,
            sheet=sheet,
            arrangement=arrangement,
            is_first_page=is_first,
            scale=s_factor,
        )
        output_pages.extend(rendered_subpages)

    return output_pages


def render_pdf(
    pages: list[bytes],
    sheet: ParsedSheet,
    arrangement: Arrangement,
) -> bytes:
    """Render multi-page PDF bytes from sheet music pages and accompaniment."""
    if arrangement.instrument == "ukulele":
        from app.render.ukulele import render_uke_pdf

        return render_uke_pdf(pages, sheet, arrangement)

    rendered_images = render_pages(pages, sheet, arrangement)
    return images_to_pdf_bytes(rendered_images)

