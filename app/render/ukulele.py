"""Ukulele accompaniment overlay renderer.

Renders sight-readable ukulele accompaniment overlays:
- Page top: header band with instrument info, capo/shape key, tuning,
  pattern legends, and standard vertical 4-string chord diagrams for every shape
  used on that page.
- Per-system strips: chord shape names at chord beat x, strum arrows (vector ↓/↑),
  chucks (×), rests (0), pluck string numbers (stacked vertically for pinches),
  accents (>), rhythm underline beams, section captions, and key-change marks.
"""
from __future__ import annotations

import io
import math
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.models import (
    Arrangement,
    ParsedSheet,
    SectionPlan,
    System,
    UkeMeasure,
    UkeShape,
    UkeStroke,
)
from app.render.jianpu import get_font
from app.render.overlay import calc_event_x, find_whitest_row
from app.theory.keys import KEY_NAME_TO_PC, canonical_key_for_pc

# Chinese names for song-form roles
ROLE_CN: dict[str, str] = {
    "intro": "前奏",
    "verse": "主歌",
    "prechorus": "导歌",
    "chorus": "副歌",
    "bridge": "桥段",
    "interlude": "间奏",
    "outro": "尾奏",
    "unknown": "",
}

UKE_STRIP_HEIGHT: int = 78



def build_uke_header_title(arrangement: Arrangement) -> str:
    """Build user-facing title line for the ukulele accompaniment header band.

    Format:
    '尤克里里伴奏 · {中级} · 变调夹 {n} 品，按 {shape_key} 调指法（实际 {start_key} 调）· 定弦 G C E A（高音 G）'
    (omit the capo phrase when capo=0).
    """
    diff_name = {
        "beginner": "初级",
        "intermediate": "中级",
        "advanced": "高级",
    }.get(arrangement.difficulty, "中级")

    shape_key = arrangement.shape_key or arrangement.start_key
    if arrangement.capo and arrangement.capo > 0:
        capo_phrase = f"变调夹 {arrangement.capo} 品，按 {shape_key} 调指法（实际 {arrangement.start_key} 调）"
    else:
        capo_phrase = f"按 {shape_key} 调指法"

    return f"尤克里里伴奏 · {diff_name} · {capo_phrase} · 定弦 G C E A（高音 G）"


def get_pattern_alias_map(arrangement: Arrangement) -> dict[str, str]:
    """Map internal or long pattern_ids to clean short display aliases like P1, P2, P3."""
    alias_map: dict[str, str] = {}
    p_counter = 1
    for m in arrangement.uke_measures:
        pid = m.pattern_id
        if not pid or pid in alias_map:
            continue
        if pid.startswith("P") and len(pid) <= 4 and pid[1:].isdigit():
            alias_map[pid] = pid
        elif "stop" in pid.lower():
            alias_map[pid] = "Stop"
        elif "hold" in pid.lower() or "held" in pid.lower():
            alias_map[pid] = "Hold"
        else:
            alias_map[pid] = f"P{p_counter}"
            p_counter += 1
    return alias_map


def format_section_label(sec: Optional[SectionPlan], short_pid: str) -> str:
    """Format section label without showing dummy internal IDs like 'section_all'."""
    if not sec:
        return short_pid

    role = sec.role or "unknown"
    sid = (sec.section_id or "").strip()
    if role == "unknown" and (not sid or sid.startswith("section_") or sid == "unknown"):
        return ""

    role_cn = ROLE_CN.get(role, "")

    # Clean up redundant sid when sid matches or contains the role name
    cleaned_sid = sid
    for en_prefix, zh_name in [
        ("intro", "前奏"),
        ("prechorus", "导歌"),
        ("verse", "主歌"),
        ("chorus", "副歌"),
        ("bridge", "桥段"),
        ("interlude", "间奏"),
        ("outro", "尾奏"),
    ]:
        if sid.lower().startswith(en_prefix):
            suffix = sid[len(en_prefix) :].strip()
            cleaned_sid = f"{zh_name} {suffix}".strip() if suffix else zh_name
            break

    if cleaned_sid:
        if role_cn and not cleaned_sid.startswith(role_cn):
            label = f"{role_cn} {cleaned_sid}".strip()
        else:
            label = cleaned_sid
    else:
        label = role_cn

    if short_pid and short_pid not in ("Stop", "Hold"):
        return f"{label} · {short_pid}" if label else short_pid
    return label


# --------------------------------------------------------------------------- Vector Drawing Primitives
def draw_down_arrow(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_top: float,
    y_bot: float,
    color: tuple[int, int, int] = (30, 30, 30),
    width: float = 2.2,
    head_size: float = 6.0,
) -> None:
    """Draw a clean vector down arrow ↓ with head pointing down."""
    w_int = max(1, int(round(width)))
    draw.line([(x, y_top), (x, y_bot)], fill=color, width=w_int)
    head_w = head_size * 0.75
    draw.line([(x - head_w, y_bot - head_size), (x, y_bot)], fill=color, width=w_int)
    draw.line([(x + head_w, y_bot - head_size), (x, y_bot)], fill=color, width=w_int)


def draw_up_arrow(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_top: float,
    y_bot: float,
    color: tuple[int, int, int] = (30, 30, 30),
    width: float = 2.2,
    head_size: float = 6.0,
) -> None:
    """Draw a clean vector up arrow ↑ with head pointing up."""
    w_int = max(1, int(round(width)))
    draw.line([(x, y_bot), (x, y_top)], fill=color, width=w_int)
    head_w = head_size * 0.75
    draw.line([(x - head_w, y_top + head_size), (x, y_top)], fill=color, width=w_int)
    draw.line([(x + head_w, y_top + head_size), (x, y_top)], fill=color, width=w_int)


def draw_chuck(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_mid: float,
    size: float = 11.0,
    color: tuple[int, int, int] = (30, 30, 30),
    width: float = 2.2,
) -> None:
    """Draw a bold vector cross × for chuck (percussive mute)."""
    w_int = max(1, int(round(width)))
    h = size / 2.0
    draw.line([(x - h, y_mid - h), (x + h, y_mid + h)], fill=color, width=w_int)
    draw.line([(x - h, y_mid + h), (x + h, y_mid - h)], fill=color, width=w_int)


def draw_accent(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_acc: float,
    color: tuple[int, int, int] = (30, 30, 30),
    width: float = 1.6,
    size: float = 3.2,
) -> None:
    """Draw accent mark > above the stroke glyph, kept small and clear without touching chord name."""
    w_int = max(1, int(round(width)))
    hw = size
    hh = size * 0.65
    draw.line([(x - hw, y_acc - hh), (x + hw, y_acc)], fill=color, width=w_int)
    draw.line([(x - hw, y_acc + hh), (x + hw, y_acc)], fill=color, width=w_int)


def draw_pluck(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_mid: float,
    strings: Sequence[int],
    scale: float = 1.0,
    color: tuple[int, int, int] = (30, 30, 30),
) -> None:
    """Draw string numbers for plucks.

    For pinches (multiple strings), stack digits vertically (string 1 highest pitch on top).
    Digits are sized at least to match chord x-height with clear separation.
    """
    if not strings:
        strings = [1]

    # In ukulele: string 1 is highest pitch (A string), string 4 is G string.
    # Order top-to-bottom: ascending string index 1, 2, 3, 4
    sorted_strings = sorted(strings)
    if len(sorted_strings) == 1:
        font = get_font(size=int(round(17 * scale)), bold=True)
        text = str(sorted_strings[0])
        tb = draw.textbbox((0, 0), text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        draw.text((x - tw / 2.0, y_mid - th / 2.0 - 1.0), text, font=font, fill=color)
    else:
        # Stacked pinch with clear spacing and bold legible digits
        font = get_font(size=int(round(13 * scale)), bold=True)
        n = len(sorted_strings)
        step = 13.0 * scale
        start_y = y_mid - ((n - 1) * step) / 2.0
        for i, s_num in enumerate(sorted_strings):
            text = str(s_num)
            tb = draw.textbbox((0, 0), text, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            draw.text((x - tw / 2.0, start_y + i * step - th / 2.0), text, font=font, fill=color)


def draw_rest_glyph(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_mid: float,
    scale: float = 1.0,
    color: tuple[int, int, int] = (90, 90, 90),
) -> None:
    """Draw rest digit '0' centered at x, y_mid."""
    font = get_font(size=int(round(16 * scale)), bold=False)
    tb = draw.textbbox((0, 0), "0", font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    draw.text((x - tw / 2.0, y_mid - th / 2.0 - 1.0), "0", font=font, fill=color)


def draw_sustain_dash_glyph(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_mid: float,
    scale: float = 1.0,
    color: tuple[int, int, int] = (40, 40, 40),
) -> None:
    """Draw sustain dash '–' at mid-height with same stroke weight as arrows."""
    dash_len = 11.0 * scale
    dash_w = max(2, int(round(2.2 * scale)))
    draw.line([(x - dash_len / 2.0, y_mid), (x + dash_len / 2.0, y_mid)], fill=color, width=dash_w)


# --------------------------------------------------------------------------- Chord Diagram
def draw_chord_diagram(
    draw: ImageDraw.ImageDraw,
    shape: Optional[UkeShape],
    chord_name: str,
    x: float,
    y: float,
    col_w: float = 90.0,
    scale: float = 1.0,
) -> float:
    """Draw a standard vertical 4-string, 4-fret ukulele chord diagram.

    Features:
    - Guaranteed >= 1.4 cm wide at A4 scale
    - Large dots with high-contrast bold white finger numbers inside
    - Supplemental finger numbers below the grid for maximum readability
    - Thick nut line when base_fret == 1, or 'n fr' label when base_fret > 1
    - Barre rendered as rounded bar with finger 1
    - Graceful fallback: name only if shape is None.
    """
    cx = x + col_w / 2.0

    # 1. Grid Geometry (guaranteed >= 1.4 cm wide at A4)
    grid_w = max(76.0 * scale, col_w - 32.0 * scale)
    string_spacing = grid_w / 3.0
    x_left = cx - grid_w / 2.0
    sx = [x_left + i * string_spacing for i in range(4)]

    # Chord Name above
    font_name = get_font(size=int(round(string_spacing * 0.65)), bold=True)
    tb_name = draw.textbbox((0, 0), chord_name, font=font_name)
    w_name = tb_name[2] - tb_name[0]
    draw.text((cx - w_name / 2.0, y), chord_name, font=font_name, fill=(176, 0, 0))

    if shape is None:
        font_sub = get_font(size=int(round(string_spacing * 0.40)), bold=False)
        draw.text((cx - 20.0 * scale, y + 26.0 * scale), "(无指法)", font=font_sub, fill=(130, 130, 130))
        return 50.0 * scale

    y_nut = y + 36.0 * scale
    fret_h = string_spacing * 1.05
    fret_y = [y_nut + (k + 1) * fret_h for k in range(4)]

    # 2. Indicators above nut: 'O' for open, '×' for muted
    y_indic = y_nut - 10.0 * scale
    indic_r = string_spacing * 0.18
    for i in range(4):
        fret = shape.frets[i]
        if fret == 0:
            draw.ellipse(
                [sx[i] - indic_r, y_indic - indic_r, sx[i] + indic_r, y_indic + indic_r],
                outline=(50, 50, 50),
                width=max(1, int(round(1.5 * scale))),
            )
        elif fret < 0:
            w_x = max(1, int(round(1.5 * scale)))
            draw.line(
                [(sx[i] - indic_r, y_indic - indic_r), (sx[i] + indic_r, y_indic + indic_r)],
                fill=(120, 120, 120),
                width=w_x,
            )
            draw.line(
                [(sx[i] - indic_r, y_indic + indic_r), (sx[i] + indic_r, y_indic - indic_r)],
                fill=(120, 120, 120),
                width=w_x,
            )

    # 3. Nut line and base fret label
    if shape.base_fret == 1:
        draw.line(
            [(sx[0] - 1.0, y_nut), (sx[3] + 1.0, y_nut)],
            fill=(30, 30, 30),
            width=max(2, int(round(3.5 * scale))),
        )
    else:
        draw.line(
            [(sx[0], y_nut), (sx[3], y_nut)],
            fill=(70, 70, 70),
            width=max(1, int(round(1.2 * scale))),
        )
        font_base = get_font(size=int(round(string_spacing * 0.42)), bold=True)
        draw.text((sx[3] + 5.0 * scale, y_nut - 2.0 * scale), f"{shape.base_fret}fr", font=font_base, fill=(40, 40, 40))

    # 4. Frets (horizontal lines)
    fret_line_w = max(1, int(round(1.2 * scale)))
    for fy in fret_y:
        draw.line([(sx[0], fy), (sx[3], fy)], fill=(150, 150, 150), width=fret_line_w)

    # 5. Strings (vertical lines)
    str_line_w = max(1, int(round(1.4 * scale)))
    for string_x in sx:
        draw.line([(string_x, y_nut), (string_x, fret_y[3])], fill=(150, 150, 150), width=str_line_w)

    # 6. Barre (rounded bar across barred strings)
    base_f = shape.base_fret
    if shape.barre is not None and shape.barre >= base_f:
        d_fret = shape.barre - base_f + 1
        if 1 <= d_fret <= 4:
            fret_mid_y = y_nut + (d_fret - 0.5) * fret_h
            barred_indices = [i for i in range(4) if shape.frets[i] >= shape.barre]
            if barred_indices:
                min_i = min(barred_indices)
                max_i = max(barred_indices)
                barre_pad = string_spacing * 0.35
                barre_h = string_spacing * 0.65
                bx0 = sx[min_i] - barre_pad
                bx1 = sx[max_i] + barre_pad
                by0 = fret_mid_y - barre_h / 2.0
                by1 = fret_mid_y + barre_h / 2.0
                draw.rounded_rectangle([bx0, by0, bx1, by1], radius=max(3.0, 5.0 * scale), fill=(30, 30, 30))
                # Finger 1 label on barre
                font_barre = get_font(size=int(round(string_spacing * 0.44)), bold=True)
                barre_cx = (bx0 + bx1) / 2.0
                tb_b = draw.textbbox((0, 0), "1", font=font_barre)
                bw = tb_b[2] - tb_b[0]
                bh = tb_b[3] - tb_b[1]
                draw.text(
                    (barre_cx - bw / 2.0, fret_mid_y - bh / 2.0 - 1.0),
                    "1",
                    font=font_barre,
                    fill=(255, 255, 255),
                )

    # 7. Dots with Large High-Contrast Finger Numbers
    dot_r = string_spacing * 0.36
    font_f = get_font(size=int(round(string_spacing * 0.46)), bold=True)
    font_sub_f = get_font(size=int(round(string_spacing * 0.38)), bold=True)

    for i in range(4):
        fret = shape.frets[i]
        f_num = shape.fingers[i]
        if fret > 0:
            d_fret = fret - base_f + 1
            if 1 <= d_fret <= 4:
                if not (shape.barre is not None and fret == shape.barre and f_num == 1):
                    dot_x = sx[i]
                    dot_y = y_nut + (d_fret - 0.5) * fret_h
                    draw.ellipse(
                        [dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r],
                        fill=(30, 30, 30),
                    )
                    if f_num > 0:
                        text_f = str(f_num)
                        tb_f = draw.textbbox((0, 0), text_f, font=font_f)
                        fw = tb_f[2] - tb_f[0]
                        fh = tb_f[3] - tb_f[1]
                        draw.text(
                            (dot_x - fw / 2.0, dot_y - fh / 2.0 - 1.0),
                            text_f,
                            font=font_f,
                            fill=(255, 255, 255),
                        )
            # Supplemental finger number below the grid
            if f_num > 0:
                text_sf = str(f_num)
                tb_sf = draw.textbbox((0, 0), text_sf, font=font_sub_f)
                sf_w = tb_sf[2] - tb_sf[0]
                draw.text(
                    (sx[i] - sf_w / 2.0, fret_y[3] + 3.0 * scale),
                    text_sf,
                    font=font_sub_f,
                    fill=(60, 60, 60),
                )

    return (fret_y[3] + string_spacing * 0.45 + 10.0 * scale) - y


# --------------------------------------------------------------------------- Header Band
def make_uke_header_band(
    width: int,
    sheet: Optional[ParsedSheet],
    arrangement: Arrangement,
    page_systems: Sequence[System],
) -> Image.Image:
    """Create a comprehensive top header band for a single ukulele page.

    Contains:
    1. Header line (Title · Difficulty · Capo/Shape Key · Tuning)
    2. Pattern legend line (patterns used on this page + notation guide)
    3. Chord diagrams for all distinct shapes used on THIS page (first-appearance order).
    """
    scale = max(1.0, min(1.6, width / 1100.0))
    margin = max(36.0 * scale, width * 0.04)

    # 1. Title line
    title = build_uke_header_title(arrangement)

    # 2. Collect patterns used on this page and map to clean aliases (P1, P2)
    alias_map = get_pattern_alias_map(arrangement)
    page_measure_indices = {m.index for s in page_systems for m in s.measures}
    pids: list[str] = []
    for uke_m in arrangement.uke_measures:
        if uke_m.measure_index in page_measure_indices and uke_m.pattern_id:
            if uke_m.pattern_id not in pids:
                pids.append(uke_m.pattern_id)

    legend_items: list[str] = []
    for pid in pids:
        short_pid = alias_map.get(pid, pid)
        raw_pat_text = arrangement.pattern_legend.get(pid, "")
        if raw_pat_text:
            # Clean up raw internal keys inside legend string if any
            if ":" in raw_pat_text or "：" in raw_pat_text:
                parts = raw_pat_text.replace(":", "：").split("：", 1)
                pat_strokes = parts[1].strip()
            else:
                pat_strokes = raw_pat_text.strip()
            legend_items.append(f"{short_pid}：{pat_strokes}")
        elif short_pid:
            legend_items.append(short_pid)

    legend_items.append("× = 闷音 chuck")
    legend_items.append("指弹数字 = 弦号（1 = A 弦，4 = G 弦）")
    legend_line = "   ·   ".join(legend_items)

    # 3. Collect distinct chord shapes used on this page (first appearance order)
    seen_shapes: set[str] = set()
    page_shapes: list[str] = []
    uke_map = {m.measure_index: m for m in arrangement.uke_measures}
    arr_map = {m.measure_index: m for m in arrangement.measures}

    for s in page_systems:
        for m in s.measures:
            uke_m = uke_map.get(m.index)
            if uke_m and uke_m.shapes:
                for sh in uke_m.shapes:
                    if sh and sh not in seen_shapes:
                        seen_shapes.add(sh)
                        page_shapes.append(sh)
            else:
                arr_m = arr_map.get(m.index)
                if arr_m:
                    for c in arr_m.chords:
                        if c.name and c.name not in seen_shapes:
                            seen_shapes.add(c.name)
                            page_shapes.append(c.name)

    # Layout dimensions: guaranteed >= 1.4 cm wide at A4 (A4 is 21.0 cm wide)
    target_grid_w = max(78.0 * scale, width * (1.45 / 21.0))
    col_w = target_grid_w + 32.0 * scale
    col_gap = 14.0 * scale
    usable_w = width - 2 * margin
    max_per_row = max(1, int((usable_w + col_gap) // (col_w + col_gap)))

    row_h = (target_grid_w / 3.0) * 4.2 + 62.0 * scale
    y_title = 12.0 * scale
    y_legend = 40.0 * scale
    y_diag_start = 68.0 * scale

    if page_shapes:
        num_rows = math.ceil(len(page_shapes) / max_per_row)
        total_h = int(math.ceil(y_diag_start + num_rows * row_h + 12.0 * scale))
    else:
        total_h = int(y_diag_start + 6.0 * scale)

    band = Image.new("RGB", (width, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(band)

    # Render title
    font_title = get_font(size=int(round(18 * scale)), bold=True)
    draw.text((margin, y_title), title, font=font_title, fill=(20, 20, 20))

    # Render pattern legend (darker and larger per feedback)
    font_legend = get_font(size=int(round(13.5 * scale)), bold=False)
    draw.text((margin, y_legend), legend_line, font=font_legend, fill=(30, 30, 30))

    # Render chord diagrams
    for idx, sh_name in enumerate(page_shapes):
        r_idx = idx // max_per_row
        c_idx = idx % max_per_row
        dx = margin + c_idx * (col_w + col_gap)
        dy = y_diag_start + r_idx * row_h
        sh_obj = arrangement.uke_shapes.get(sh_name)
        draw_chord_diagram(draw, sh_obj, sh_name, dx, dy, col_w=col_w, scale=scale)

    # Subtle bottom border line
    draw.line([(0, total_h - 1), (width, total_h - 1)], fill=(225, 225, 225), width=1)

    return band


# --------------------------------------------------------------------------- System Strip
def render_uke_system_strip(
    system: System,
    page_width: int,
    arrangement: Arrangement,
    sheet: Optional[ParsedSheet] = None,
) -> Image.Image:
    """Render a compact accompaniment strip for one system row.

    Layout:
    - Dedicated badge / label for section captions and key changes
    - Line 1: Shape names (bold, beat x) with collision avoidance
    - Line 2: Strokes at onset x: ↓/↑ arrows, chuck ×, rest 0, pluck digits, accent >,
      held dashes -, underline rhythm beams.
    Strip height: ~78 px (scaled with resolution, much smaller than piano's 170-200 px).
    """
    scale = max(1.0, min(1.6, page_width / 1100.0))
    strip_h = int(round(UKE_STRIP_HEIGHT * scale))
    strip = Image.new("RGB", (page_width, strip_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(strip)

    arr_map = {m.measure_index: m for m in arrangement.measures}
    uke_map = {m.measure_index: m for m in arrangement.uke_measures}
    sec_map = {s.measure_index: s for s in arrangement.sections}
    kc_map = {kc.at_measure: kc for kc in sheet.key_changes} if sheet and sheet.key_changes else {}
    alias_map = get_pattern_alias_map(arrangement)

    font_chord = get_font(size=int(round(15 * scale)), bold=True)
    font_chord_sm = get_font(size=int(round(12 * scale)), bold=True)
    chord_color = (176, 0, 0)
    bar_color = (190, 190, 190)

    measures = system.measures
    num_meas = len(measures)

    # Pass 1: Measure layout geometries and flags
    m_infos = []
    for m_idx, m in enumerate(measures):
        arr_m = arr_map.get(m.index)
        uke_m = uke_map.get(m.index)
        sec = sec_map.get(m.index)
        beats = m.beats or 4.0

        if m.bbox:
            mx0 = m.bbox[0] * page_width
            mx1 = m.bbox[2] * page_width
        else:
            mx0 = (m_idx / num_meas) * page_width
            mx1 = ((m_idx + 1) / num_meas) * page_width

        mw = mx1 - mx0
        pad = max(4.0 * scale, mw * 0.04)

        has_printed_chords = bool(m.chords)
        is_new_row = (m_idx == 0)
        is_pickup_bar = (mw < 140.0 * scale and not has_printed_chords)

        m_infos.append({
            "m_idx": m_idx,
            "m": m,
            "arr_m": arr_m,
            "uke_m": uke_m,
            "sec": sec,
            "beats": beats,
            "mx0": mx0,
            "mx1": mx1,
            "mw": mw,
            "pad": pad,
            "has_printed_chords": has_printed_chords,
            "is_new_row": is_new_row,
            "is_pickup_bar": is_pickup_bar,
        })

    # Pass 1b: badge geometry (section caption + key-change mark). Badges that sit
    # inside a measure reserve space on the chord-name line so names are never covered.
    for info in m_infos:
        m = info["m"]
        uke_m = info["uke_m"]
        arr_m = info["arr_m"]
        sec = info["sec"]
        mx0 = info["mx0"]
        info["badges"] = []
        info["badge_floor"] = None
        short_pid = alias_map.get(uke_m.pattern_id, "") if uke_m and uke_m.pattern_id else ""
        cap_text = format_section_label(sec, short_pid) if (sec and (sec.is_section_start or m.index == 0)) else ""
        badge_reserved_right = mx0 + info["pad"]
        in_measure = False
        if cap_text:
            font_sec = get_font(size=int(round(11 * scale)), bold=True)
            tb_sec = draw.textbbox((0, 0), cap_text, font=font_sec)
            sec_w = (tb_sec[2] - tb_sec[0]) + 10.0 * scale
            sec_h = 16.0 * scale
            # If margin on left is wide enough, put in margin; otherwise inside measure
            if info["m_idx"] == 0 and mx0 >= sec_w + 6.0 * scale:
                bx0 = max(4.0, mx0 - sec_w - 4.0 * scale)
                by0 = 5.0 * scale
            else:
                bx0 = mx0 + 4.0 * scale
                by0 = 4.0 * scale
                in_measure = True
                badge_reserved_right = max(badge_reserved_right, bx0 + sec_w + 6.0 * scale)
            info["badges"].append(([bx0, by0, bx0 + sec_w, by0 + sec_h], cap_text,
                                   (239, 246, 255), (191, 219, 254), (29, 78, 216), font_sec))
        if m.index in kc_map and arr_m:
            if arrangement.capo and arrangement.capo > 0:
                shape_pc = (KEY_NAME_TO_PC.get(arr_m.key_name, 0) - arrangement.capo) % 12
                sk = canonical_key_for_pc(shape_pc)
            else:
                sk = arr_m.key_name
            kc_text = f"转 {sk} 调指法"
            font_kc = get_font(size=int(round(11 * scale)), bold=True)
            tb_kc = draw.textbbox((0, 0), kc_text, font=font_kc)
            kcw = (tb_kc[2] - tb_kc[0]) + 10.0 * scale
            kx0 = badge_reserved_right
            info["badges"].append(([kx0, 4.0 * scale, kx0 + kcw, 20.0 * scale], kc_text,
                                   (254, 243, 199), (251, 191, 36), (180, 83, 9), font_kc))
            badge_reserved_right = kx0 + kcw + 6.0 * scale
            in_measure = True
        if in_measure:
            info["badge_floor"] = badge_reserved_right

    # Pass 2: Collect all chord items across the system strip
    all_chord_items: list[dict] = []
    for info in m_infos:
        arr_m = info["arr_m"]
        uke_m = info["uke_m"]
        has_printed_chords = info["has_printed_chords"]
        is_new_row = info["is_new_row"]
        is_pickup_bar = info["is_pickup_bar"]
        mx0 = info["mx0"]
        mx1 = info["mx1"]
        mw = info["mw"]
        pad = info["pad"]
        beats = info["beats"]

        # Do NOT repeat a chord name for a measure whose chord merely carries over
        # from the previous measure (no printed chord: check sheet measure.chords empty)
        # unless it starts a new row.
        # And for a narrow pickup bar starting a row with no printed chords, do not print carried-over chord.
        if not has_printed_chords:
            if not is_new_row or is_pickup_bar:
                continue

        if arr_m and arr_m.chords:
            for c_i, c in enumerate(arr_m.chords):
                shape_name = (
                    uke_m.shapes[c_i]
                    if (uke_m and c_i < len(uke_m.shapes) and uke_m.shapes[c_i])
                    else c.name
                )
                if not shape_name:
                    continue
                onset = max(0.0, c.beat - 1.0)
                nom_x = calc_event_x(mx0, mw, pad, onset, beats)
                all_chord_items.append({
                    "name": shape_name,
                    "nom_x": nom_x,
                    "x": nom_x,
                    "font": font_chord,
                    "w": 0.0,
                    "m_idx": info["m_idx"],
                    "mx0": mx0,
                    "mx1": mx1,
                    "is_carried_over": (not has_printed_chords),
                    "omit": False,
                    "floor": info["badge_floor"],
                })

    # Measure chord glyph bounding boxes
    for item in all_chord_items:
        tb = draw.textbbox((0, 0), item["name"], font=item["font"])
        item["w"] = float(tb[2] - tb[0])
        if item["floor"] is not None and item["x"] < item["floor"]:
            item["x"] = item["floor"]

    # Global collision resolution across adjacent chords
    cn = len(all_chord_items)
    for i in range(cn - 1):
        curr = all_chord_items[i]
        nxt = all_chord_items[i + 1]
        if curr["omit"] or nxt["omit"]:
            continue
        min_gap = 5.0 * scale
        if curr["x"] + curr["w"] + min_gap > nxt["x"]:
            # Collision detected!
            # 1. If one of them is carried over, omit the carried-over chord
            if curr["is_carried_over"]:
                curr["omit"] = True
                continue
            if nxt["is_carried_over"]:
                nxt["omit"] = True
                continue

            # 2. Try shrinking both fonts to font_chord_sm
            curr["font"] = font_chord_sm
            nxt["font"] = font_chord_sm
            tb_c = draw.textbbox((0, 0), curr["name"], font=font_chord_sm)
            curr["w"] = float(tb_c[2] - tb_c[0])
            tb_n = draw.textbbox((0, 0), nxt["name"], font=font_chord_sm)
            nxt["w"] = float(tb_n[2] - tb_n[0])

            if curr["x"] + curr["w"] + min_gap > nxt["x"]:
                overlap = (curr["x"] + curr["w"] + min_gap) - nxt["x"]
                shift = overlap / 2.0
                curr["x"] = max(curr["mx0"] + 2.0 * scale, curr["x"] - shift)
                nxt["x"] = min(nxt["mx1"] - nxt["w"] - 2.0 * scale, nxt["x"] + shift)

    # Strictly enforce left-to-right order without overlap (and never under a badge)
    cur_right = 0.0
    for item in all_chord_items:
        if item["omit"]:
            continue
        if item["floor"] is not None and item["x"] < item["floor"]:
            item["x"] = item["floor"]
        if item["x"] < cur_right + 4.0 * scale:
            item["x"] = cur_right + 4.0 * scale
        cur_right = item["x"] + item["w"]

    # Render non-omitted chords
    chord_y = 2.0 * scale
    for item in all_chord_items:
        if item["omit"]:
            continue
        draw.text((item["x"], chord_y), item["name"], font=item["font"], fill=chord_color)

    # Pass 3: Draw measures, barlines, badges, and strokes
    for info in m_infos:
        m_idx = info["m_idx"]
        m = info["m"]
        arr_m = info["arr_m"]
        uke_m = info["uke_m"]
        sec = info["sec"]
        beats = info["beats"]
        mx0 = info["mx0"]
        mx1 = info["mx1"]
        mw = info["mw"]
        pad = info["pad"]
        is_pickup_bar = info["is_pickup_bar"]

        # 1. Bar lines
        draw.line([(mx1, 0), (mx1, strip_h)], fill=bar_color, width=1)
        if m_idx == 0:
            draw.line([(mx0, 0), (mx0, strip_h)], fill=bar_color, width=1)

        # 2./3. Section caption + key-change badges (geometry computed before chords)
        for (rect, text, fill, outline, tcolor, font_b) in info["badges"]:
            draw.rounded_rectangle(rect, radius=3.0 * scale, fill=fill, outline=outline)
            draw.text((rect[0] + 5.0 * scale, rect[1] + 1.0 * scale), text, font=font_b, fill=tcolor)

        # 4. Line 2: Strokes & Rhythm
        if not uke_m or not uke_m.strokes:
            continue

        strokes = uke_m.strokes
        y_acc = 23.5 * scale
        y_top = 28.0 * scale
        y_bot = 48.0 * scale
        y_mid = 38.0 * scale
        arrow_w = 2.2 * scale
        head_s = 6.0 * scale

        col_scale = max(0.65, min(1.0, (mw - 2.0 * pad) / max(1.0, (len(strokes) * 14.0 * scale))))
        eff_arrow_w = arrow_w * col_scale
        eff_head_s = head_s * col_scale

        for s in strokes:
            sx = calc_event_x(mx0, mw, pad, s.onset, beats)

            # Accent mark above (only where UkeStroke.accent is true, small and clear)
            if bool(s.accent):
                draw_accent(draw, sx, y_acc=y_acc, width=1.6 * scale * col_scale, size=3.2 * scale * col_scale)

            # Stroke Kind
            if s.kind == "down":
                draw_down_arrow(draw, sx, y_top=y_top, y_bot=y_bot, width=eff_arrow_w, head_size=eff_head_s)
            elif s.kind == "up":
                draw_up_arrow(draw, sx, y_top=y_top, y_bot=y_bot, width=eff_arrow_w, head_size=eff_head_s)
            elif s.kind == "chuck":
                draw_chuck(draw, sx, y_mid=y_mid, size=11.0 * scale * col_scale, width=eff_arrow_w)
            elif s.kind == "rest":
                draw_rest_glyph(draw, sx, y_mid=y_mid, scale=scale * col_scale)
            elif s.kind == "pluck":
                draw_pluck(draw, sx, y_mid=y_mid, strings=s.strings, scale=scale * col_scale)

            # Extended quarter beats: '0' for rests, '–' (sustain dash) for sounding strokes
            if s.duration >= 2.0:
                extra_beats = int(math.floor(s.duration)) - 1
                for k in range(1, extra_beats + 1):
                    beat_x = calc_event_x(mx0, mw, pad, s.onset + k, beats)
                    if beat_x < mx1 - pad:
                        if s.kind == "rest":
                            draw_rest_glyph(draw, beat_x, y_mid=y_mid, scale=scale * col_scale)
                        else:
                            draw_sustain_dash_glyph(draw, beat_x, y_mid=y_mid, scale=scale)

        # 6. Underline Beaming (Jianpu style rhythm)
        beam_y1 = 53.0 * scale
        beam_y2 = 58.0 * scale
        beam_w = max(1, int(round(2.0 * scale)))

        # Group sub-beat strokes by beat integer
        beat_dict: dict[int, list[UkeStroke]] = {}
        for s in strokes:
            if s.duration < 1.0:
                b_idx = int(math.floor(s.onset))
                beat_dict.setdefault(b_idx, []).append(s)

        for b_idx, beat_strokes in beat_dict.items():
            b1 = [s for s in beat_strokes if s.duration <= 0.75]
            if len(b1) >= 2:
                x_start = calc_event_x(mx0, mw, pad, b1[0].onset, beats) - 4.5 * scale
                x_end = calc_event_x(mx0, mw, pad, b1[-1].onset, beats) + 4.5 * scale
                draw.line([(x_start, beam_y1), (x_end, beam_y1)], fill=(0, 0, 0), width=beam_w)
            elif len(b1) == 1:
                sx = calc_event_x(mx0, mw, pad, b1[0].onset, beats)
                draw.line([(sx - 4.5 * scale, beam_y1), (sx + 4.5 * scale, beam_y1)], fill=(0, 0, 0), width=beam_w)

            b2 = [s for s in beat_strokes if s.duration <= 0.375]
            if len(b2) >= 2:
                x_start = calc_event_x(mx0, mw, pad, b2[0].onset, beats) - 4.5 * scale
                x_end = calc_event_x(mx0, mw, pad, b2[-1].onset, beats) + 4.5 * scale
                draw.line([(x_start, beam_y2), (x_end, beam_y2)], fill=(0, 0, 0), width=beam_w)
            elif len(b2) == 1:
                sx = calc_event_x(mx0, mw, pad, b2[0].onset, beats)
                draw.line([(sx - 4.5 * scale, beam_y2), (sx + 4.5 * scale, beam_y2)], fill=(0, 0, 0), width=beam_w)

    return strip


# --------------------------------------------------------------------------- Page & PDF Rendering
def render_uke_page(
    page_img: Image.Image,
    systems: Sequence[System],
    sheet: ParsedSheet,
    arrangement: Arrangement,
) -> list[Image.Image]:
    """Overlay ukulele accompaniment strips onto a single page and split if height > 1.6x original."""
    w, h = page_img.size
    orig_h = h

    if not systems:
        header_band = make_uke_header_band(w, sheet, arrangement, [])
        res = Image.new("RGB", (w, h + header_band.height), color=(255, 255, 255))
        res.paste(header_band, (0, 0))
        res.paste(page_img, (0, header_band.height))
        return [res]

    sorted_systems = sorted(systems, key=lambda s: s.bbox[1] if s.bbox else 0.0)
    k = len(sorted_systems)

    # Calculate cuts on original page image between systems at whitest rows
    gray = np.array(page_img.convert("L"))
    delta = int(round(0.03 * h))
    cuts: list[int] = []
    min_cut = 1

    for i in range(k - 1):
        s_curr = sorted_systems[i]
        s_next = sorted_systems[i + 1]
        curr_y1 = int(round((s_curr.bbox[3] if s_curr.bbox else (i + 1) / k) * h))
        next_y0 = int(round((s_next.bbox[1] if s_next.bbox else (i + 1) / k) * h))

        if curr_y1 >= next_y0:
            nominal_cut = (curr_y1 + next_y0) // 2
            cut_y = nominal_cut
        else:
            nominal_cut = int(round((curr_y1 + next_y0) / 2.0))
            cut_y = find_whitest_row(gray, nominal_cut, delta, curr_y1, next_y0)

        cut_y = max(min_cut, min(h - (k - i), cut_y))
        cuts.append(cut_y)
        min_cut = cut_y + 1

    last_sys = sorted_systems[-1]
    last_y1 = int(round((last_sys.bbox[3] if last_sys.bbox else 1.0) * h))
    nominal_last_cut = min(h, last_y1 + int(round(0.015 * h)))
    cut_last = find_whitest_row(gray, nominal_last_cut, delta, last_y1, h)
    cut_last = max(min_cut, min(h, cut_last))
    cuts.append(cut_last)

    page_slices: list[Image.Image] = []
    y_prev = 0
    for cut_y in cuts:
        page_slices.append(page_img.crop((0, y_prev, w, cut_y)))
        y_prev = cut_y
    page_slices.append(page_img.crop((0, y_prev, w, h)))

    header_band = make_uke_header_band(w, sheet, arrangement, sorted_systems)
    strips = [render_uke_system_strip(s, w, arrangement, sheet) for s in sorted_systems]

    total_h = header_band.height + sum(sl.height for sl in page_slices) + sum(st.height for st in strips)

    if total_h <= 1.6 * orig_h or k <= 1:
        res = Image.new("RGB", (w, total_h), color=(255, 255, 255))
        cur_y = 0
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

    # Split into two subpages if overflow > 1.6x original
    split_idx = k // 2

    # Subpage 1
    p1_systems = sorted_systems[:split_idx]
    p1_slices = page_slices[:split_idx]
    p1_strips = strips[:split_idx]
    p1_header = make_uke_header_band(w, sheet, arrangement, p1_systems)
    p1_h = p1_header.height + sum(sl.height for sl in p1_slices) + sum(st.height for st in p1_strips)
    res1 = Image.new("RGB", (w, p1_h), color=(255, 255, 255))
    cur_y = 0
    res1.paste(p1_header, (0, cur_y))
    cur_y += p1_header.height
    for sl, st in zip(p1_slices, p1_strips):
        res1.paste(sl, (0, cur_y))
        cur_y += sl.height
        res1.paste(st, (0, cur_y))
        cur_y += st.height

    # Subpage 2
    p2_systems = sorted_systems[split_idx:k]
    p2_slices = page_slices[split_idx:k]
    p2_strips = strips[split_idx:k]
    p2_header = make_uke_header_band(w, sheet, arrangement, p2_systems)
    p2_h = p2_header.height + sum(sl.height for sl in p2_slices) + sum(st.height for st in p2_strips) + page_slices[k].height
    res2 = Image.new("RGB", (w, p2_h), color=(255, 255, 255))
    cur_y = 0
    res2.paste(p2_header, (0, cur_y))
    cur_y += p2_header.height
    for sl, st in zip(p2_slices, p2_strips):
        res2.paste(sl, (0, cur_y))
        cur_y += sl.height
        res2.paste(st, (0, cur_y))
        cur_y += st.height
    res2.paste(page_slices[k], (0, cur_y))

    return [res1, res2]


def render_uke_pages(
    pages: list[bytes],
    sheet: ParsedSheet,
    arrangement: Arrangement,
) -> list[Image.Image]:
    """Render all pages with overlaid ukulele accompaniment strips.

    Returns a list of PIL Images (each representing an output PDF page).
    """
    output_pages: list[Image.Image] = []

    for page_idx, page_bytes in enumerate(pages):
        page_img = Image.open(io.BytesIO(page_bytes)).convert("RGB")
        page_systems = [s for s in sheet.systems if s.page == page_idx]

        rendered_subpages = render_uke_page(
            page_img=page_img,
            systems=page_systems,
            sheet=sheet,
            arrangement=arrangement,
        )
        output_pages.extend(rendered_subpages)

    return output_pages


def render_uke_pdf(
    pages: list[bytes],
    sheet: ParsedSheet,
    arrangement: Arrangement,
) -> bytes:
    """Render multi-page PDF bytes from sheet music pages and ukulele accompaniment."""
    rendered_images = render_uke_pages(pages, sheet, arrangement)
    if not rendered_images:
        return b""

    buf = io.BytesIO()
    first_img = rendered_images[0]
    remaining = rendered_images[1:]
    first_img.save(
        buf,
        format="PDF",
        save_all=True,
        append_images=remaining,
        resolution=150.0,
    )
    return buf.getvalue()
