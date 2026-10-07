"""From-scratch Jianpu (numbered notation) lead-sheet typesetter.

Renders a LeadSheet into crisp 2400-pixel wide A4 PNG page images and constructs
a matching ParsedSheet whose geometry aligns with the printed sheet music so that
downstream accompaniment arrangement (piano/ukulele) and overlay rendering pipeline
work unchanged.
"""
from __future__ import annotations

import io
import math
from typing import NamedTuple, Optional, Sequence, Tuple
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

from app.models import (
    ChordSymbol,
    Measure,
    PageInfo,
    ParsedSheet,
    SongHeader,
    System,
)
from app.render.jianpu import get_font
from app.smart.models import LeadChord, LeadMeasure, LeadNote, LeadSheet


class NoteLayoutBox(NamedTuple):
    """Layout debug bounding box for a single melody note."""
    measure_index: int
    note_index: int
    digit_bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page pixels
    lyric_bbox: Optional[tuple[float, float, float, float]]  # (x0, y0, x1, y1) in page pixels
    dash_xs: tuple[float, ...] = ()  # X center coordinates of extension dashes owned by this note


def _format_note_token(n: LeadNote) -> str:
    """Format a single LeadNote into token representation for parse_melody."""
    if n.degree == 0:
        return "0"
    acc = "#" if n.accidental == 1 else ("b" if n.accidental == -1 else "")
    return f"{acc}{n.degree}"


def lead_measure_to_melody_string(m: LeadMeasure) -> str:
    """Derive canonical jianpu string that app.qa.melody.parse_melody round-trips.

    Produces standard token groupings separated by whitespace:
    - Quarters: '1 2 3 4'
    - Halves: '1 - 3 4'
    - Dotted half: '1 - - 5'
    - Whole: '1 - - -'
    - Dotted quarter + eighth: '1. 5 3 2'
    - Eighth pairs: '12 34'
    - Sixteenths: '1234'
    - Dotted eighth + sixteenth: '1.2'
    - Sixteenth + dotted eighth: '12.'
    - Sub-beat note + rest: '10' or '01'
    - Rests: '0'
    - Accidentals: '#4', 'b7'
    """
    if not m.notes:
        return ""

    total_beats = int(round(m.beats))
    if total_beats <= 0:
        return ""

    tokens: list[str] = []
    b = 0
    while b < total_beats:
        notes_in_b = [n for n in m.notes if b - 1e-4 <= n.onset < b + 1.0 - 1e-4]
        if not notes_in_b:
            tokens.append("0")
            b += 1
            continue

        first = notes_in_b[0]

        # Multi-beat notes (half, dotted half, whole)
        if abs(first.onset - b) < 1e-4 and first.duration >= 2.0:
            tokens.append(_format_note_token(first))
            extra = int(round(first.duration)) - 1
            for _ in range(extra):
                tokens.append("-")
            b += int(round(first.duration))
            continue

        # Dotted quarter + eighth (1.5 + 0.5) across integer beat boundary
        if abs(first.onset - b) < 1e-4 and abs(first.duration - 1.5) < 1e-4:
            next_notes = [n for n in m.notes if abs(n.onset - (b + 1.5)) < 1e-4]
            if next_notes:
                tokens.append(f"{_format_note_token(first)}.")
                tokens.append(_format_note_token(next_notes[0]))
                b += 2
                continue

        # Plain single note filling the beat (quarter note or rest)
        if len(notes_in_b) == 1 and abs(first.onset - b) < 1e-4 and abs(first.duration - 1.0) < 1e-4:
            tokens.append(_format_note_token(first))
            b += 1
            continue

        # Two notes inside beat (eighths or dotted eighth/sixteenth)
        if len(notes_in_b) == 2:
            n1, n2 = notes_in_b[0], notes_in_b[1]
            if abs(n1.duration - 0.75) < 1e-4 and abs(n2.duration - 0.25) < 1e-4:
                tokens.append(f"{_format_note_token(n1)}.{_format_note_token(n2)}")
            elif abs(n1.duration - 0.25) < 1e-4 and abs(n2.duration - 0.75) < 1e-4:
                tokens.append(f"{_format_note_token(n1)}{_format_note_token(n2)}.")
            else:
                tokens.append(f"{_format_note_token(n1)}{_format_note_token(n2)}")
            b += 1
            continue

        # Four sixteenth notes inside beat
        if len(notes_in_b) == 4:
            tokens.append("".join(_format_note_token(n) for n in notes_in_b))
            b += 1
            continue

        # Single sub-beat note with half rest
        if len(notes_in_b) == 1:
            if abs(first.onset - b) < 1e-4:
                tokens.append(f"{_format_note_token(first)}0")
            else:
                tokens.append(f"0{_format_note_token(first)}")
            b += 1
            continue

        tokens.append("".join(_format_note_token(n) for n in notes_in_b))
        b += 1

    return " ".join(tokens)


def _compute_measure_content_need(
    m: LeadMeasure,
    chords_only: bool,
    font_digit: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_acc: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_lyric: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_chord: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> float:
    """Calculate the engraving horizontal width needed by a measure."""
    chord_w = sum(
        (len(c.raw) * 22.0 + 35.0) for c in m.chords
    )

    if chords_only or not m.notes:
        n_beats = max(1.0, m.beats)
        return max(chord_w, n_beats * 95.0) + 70.0

    total_need = 0.0
    t_cur = 0.0

    for n in m.notes:
        if n.onset > t_cur + 1e-4:
            gap_dur = n.onset - t_cur
            total_need += (20.0 * 1.30 + 48.0 * math.sqrt(max(0.125, min(1.0, gap_dur))))

        acc_w = 22.0 if n.accidental else 0.0
        d_w = 26.0
        dot_w = 12.0 if (abs(n.duration - 1.5) < 0.05 or abs(n.duration - 0.75) < 0.05) else 0.0
        mel_w = acc_w + d_w + dot_w

        syl = n.lyric.strip()
        lyric_w = (len(syl) * 26.0) if syl else 0.0

        min_ev = max(mel_w, lyric_w) * 1.30 + 8.0
        dur_term = 48.0 * math.sqrt(max(0.125, min(1.0, n.duration)))
        total_need += (min_ev + dur_term)

        if n.duration >= 2.0:
            extra = int(round(n.duration)) - 1
            for _ in range(extra):
                total_need += (20.0 * 1.30 + 8.0 + 48.0)

        t_cur = max(t_cur, n.onset + n.duration)

    return max(total_need, chord_w, m.beats * 70.0) + 70.0


def _split_into_systems(
    measures: list[LeadMeasure],
    chords_only: bool,
    content_w: float,
    font_digit: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_acc: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_lyric: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_chord: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> list[list[int]]:
    """Partition measures into systems adaptively based on content need."""
    if not measures:
        return []

    needs = [
        _compute_measure_content_need(m, chords_only, font_digit, font_acc, font_lyric, font_chord)
        for m in measures
    ]

    systems: list[list[int]] = []
    cur_row: list[int] = []

    for i, m in enumerate(measures):
        if not cur_row:
            cur_row.append(i)
            continue

        cand = cur_row + [i]
        cand_need = sum(needs[idx] for idx in cand)

        # 1. 4 measures per row standard cap
        if len(cur_row) >= 4:
            systems.append(cur_row)
            cur_row = [i]
            continue

        # 2. Content need exceeds row capacity: drops to 2 or 3 measures
        if len(cur_row) >= 2 and cand_need > content_w * 0.96:
            systems.append(cur_row)
            cur_row = [i]
            continue

        if len(cur_row) >= 3 and cand_need > content_w * 0.90:
            systems.append(cur_row)
            cur_row = [i]
            continue

        # 3. New section starting after at least 3 measures
        if m.section and len(cur_row) >= 3:
            systems.append(cur_row)
            cur_row = [i]
            continue

        cur_row.append(i)

    if cur_row:
        systems.append(cur_row)

    return systems


def _calc_tie_y(melody_y: float, octave: int) -> float:
    """Calculate tie arc endpoint Y strictly above digit top and octave dots."""
    if octave > 0:
        safe_top = melody_y + 5.0 - (octave - 1) * 9.0
        return safe_top - 8.0
    else:
        # Digit ink top is melody_y + 17.0.
        # Place tie endpoint at melody_y + 7.0 (10 px ABOVE top of digit)
        return melody_y + 7.0


def _draw_curved_tie(
    draw: ImageDraw.ImageDraw,
    x0: float,
    x1: float,
    y: float,
    arc_height: float,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
) -> None:
    """Draw a smooth curved upward tie arc between x0 and x1 at baseline y."""
    mid_x = (x0 + x1) / 2.0
    control_y = y - arc_height
    points = []
    steps = 16
    for s in range(steps + 1):
        t = s / steps
        px = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid_x + t**2 * x1
        py = (1 - t) ** 2 * y + 2 * (1 - t) * t * control_y + t**2 * y
        points.append((px, py))
    draw.line(points, fill=color, width=width)


def _draw_outgoing_tie(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    arc_height: float = 14.0,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
) -> None:
    """Draw an outgoing tie arc heading towards a system break."""
    # Curves upward from (x0, y0) and levels off to horizontal at (x1, y0 - arc_height)
    p0 = (x0, y0)
    p1 = (x0 + (x1 - x0) * 0.35, y0 - arc_height)
    p2 = (x1, y0 - arc_height)
    points = []
    steps = 14
    for s in range(steps + 1):
        t = s / steps
        px = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        py = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        points.append((px, py))
    draw.line(points, fill=color, width=width)


def _draw_incoming_tie(
    draw: ImageDraw.ImageDraw,
    x0: float,
    x1: float,
    y1: float,
    arc_height: float = 14.0,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
) -> None:
    """Draw an incoming tie arc continuing from a system break."""
    # Starts horizontal at (x0, y1 - arc_height) and curves downward into (x1, y1)
    p0 = (x0, y1 - arc_height)
    p1 = (x1 - (x1 - x0) * 0.35, y1 - arc_height)
    p2 = (x1, y1)
    points = []
    steps = 14
    for s in range(steps + 1):
        t = s / steps
        px = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        py = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        points.append((px, py))
    draw.line(points, fill=color, width=width)


def _compute_sheet_layout(
    lead: LeadSheet,
    page_width: int = 2400,
) -> tuple[
    int,  # page_height
    float,  # margin_x
    float,  # content_w
    list[list[dict]],  # pages_systems
    list[NoteLayoutBox],  # note_layout_boxes
    dict[tuple[int, int], tuple[int, float, float, int]],  # note_positions
    dict[int, tuple[float, float]],  # measure_bounds: m_idx -> (mx0, mx1)
    dict[int, list[dict]],  # measure_events_global: m_idx -> list of events
]:
    """Compute complete typesetting geometry, events, and bounding boxes."""
    page_height = int(round(page_width * 1.41421356))
    margin_x = 160.0
    content_w = float(page_width) - 2.0 * margin_x

    font_chord = get_font(size=34, bold=True)
    font_digit = get_font(size=42, bold=True)
    font_acc = get_font(size=24, bold=True)
    font_lyric = get_font(size=24, bold=False)

    has_any_lyrics = any(
        bool(n.lyric.strip()) for m in lead.measures for n in m.notes
    )

    system_measure_indices = _split_into_systems(
        lead.measures,
        lead.chords_only,
        content_w,
        font_digit,
        font_acc,
        font_lyric,
        font_chord,
    )

    system_specs: list[dict] = []
    for s_idx, row_indices in enumerate(system_measure_indices):
        row_measures = [lead.measures[idx] for idx in row_indices]
        has_section = any(bool(m.section) for m in row_measures)

        # Height breakdown:
        # - section: 36 px (if present)
        # - chord row starts with clean clearance below section box
        sec_h = 36.0 if has_section else 0.0
        chord_y_rel = (sec_h + 16.0) if has_section else 8.0
        # Melody row begins comfortably below chords
        melody_y_rel = chord_y_rel + 62.0
        # Lyrics row below beams and octave dots
        lyrics_y_rel = melody_y_rel + 96.0 if has_any_lyrics else None

        if has_any_lyrics:
            sys_h = lyrics_y_rel + 42.0 + 16.0
        else:
            sys_h = melody_y_rel + 76.0 + 16.0

        system_specs.append({
            "measure_indices": row_indices,
            "has_section": has_section,
            "sec_h": sec_h,
            "chord_y_rel": chord_y_rel,
            "melody_y_rel": melody_y_rel,
            "lyrics_y_rel": lyrics_y_rel,
            "height": sys_h,
        })

    # Page vertical distribution
    gap_between_systems = 140.0
    header_bottom = 320.0
    page_bottom_margin = 120.0

    pages_systems: list[list[dict]] = []
    cur_page_systems: list[dict] = []
    cur_y = header_bottom

    for spec in system_specs:
        needed_h = spec["height"]
        if cur_page_systems:
            candidate_y1 = cur_y + gap_between_systems + needed_h
        else:
            candidate_y1 = cur_y + needed_h

        if candidate_y1 > page_height - page_bottom_margin and cur_page_systems:
            pages_systems.append(cur_page_systems)
            cur_page_systems = []
            cur_y = 140.0
            candidate_y1 = cur_y + needed_h

        if cur_page_systems:
            cur_y += gap_between_systems

        spec_placed = dict(spec)
        spec_placed["y0"] = cur_y
        spec_placed["y1"] = cur_y + needed_h
        cur_page_systems.append(spec_placed)
        cur_y = spec_placed["y1"]

    if cur_page_systems:
        pages_systems.append(cur_page_systems)

    if not pages_systems:
        pages_systems = [[]]

    # Pre-calculate horizontal measure widths for each system based on content needs
    dummy_img = Image.new("RGB", (100, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)

    measure_bounds_global: dict[int, tuple[float, float]] = {}

    for page_sys in pages_systems:
        for spec in page_sys:
            row_ms = [lead.measures[idx] for idx in spec["measure_indices"]]
            row_needs = [
                _compute_measure_content_need(
                    m,
                    lead.chords_only,
                    font_digit,
                    font_acc,
                    font_lyric,
                    font_chord,
                )
                for m in row_ms
            ]
            total_need = max(1.0, sum(row_needs))
            widths = [round(content_w * (nd / total_need), 2) for nd in row_needs]
            diff = round(content_w - sum(widths), 2)
            if widths:
                widths[-1] += diff

            mx = margin_x
            m_bounds = []
            for m_idx, w in zip(spec["measure_indices"], widths):
                m_bounds.append((mx, mx + w))
                measure_bounds_global[m_idx] = (mx, mx + w)
                mx += w
            spec["m_bounds"] = m_bounds

    # Layout notes cumulatively inside each measure
    note_layout_boxes: list[NoteLayoutBox] = []
    note_positions: dict[tuple[int, int], tuple[int, float, float, int]] = {}
    measure_events_global: dict[int, list[dict]] = {}

    for p_idx, page_sys in enumerate(pages_systems):
        for spec in page_sys:
            melody_y = spec["y0"] + spec["melody_y_rel"]
            lyrics_y = spec["y0"] + spec["lyrics_y_rel"] if spec["lyrics_y_rel"] is not None else None

            for m_pos, global_m_idx in enumerate(spec["measure_indices"]):
                m = lead.measures[global_m_idx]
                mx0, mx1 = spec["m_bounds"][m_pos]
                mw = mx1 - mx0

                if lead.chords_only or not m.notes:
                    continue

                pad = max(24.0, mw * 0.04)
                avail_w = mw - 2.0 * pad

                # Build events
                events: list[dict] = []
                t_cur = 0.0

                for n_idx, n in enumerate(m.notes):
                    if n.onset > t_cur + 1e-4:
                        gap_dur = n.onset - t_cur
                        gap_need = 20.0 * 1.30 + 48.0 * math.sqrt(max(0.125, min(1.0, gap_dur)))
                        events.append({"type": "gap", "onset": t_cur, "dur": gap_dur, "need": gap_need})

                    acc_w = (dummy_draw.textbbox((0, 0), "♯", font=font_acc)[2] + 4.0) if n.accidental else 0.0
                    d_w = dummy_draw.textbbox((0, 0), str(n.degree), font=font_digit)[2]
                    dot_w = 12.0 if (abs(n.duration - 1.5) < 0.05 or abs(n.duration - 0.75) < 0.05) else 0.0
                    mel_w = acc_w + d_w + dot_w

                    syl = n.lyric.strip()
                    lyric_w = dummy_draw.textbbox((0, 0), syl, font=font_lyric)[2] if syl else 0.0

                    min_ev = max(mel_w, lyric_w) * 1.30 + 8.0
                    dur_term = 48.0 * math.sqrt(max(0.125, min(1.0, n.duration)))
                    ev_need = min_ev + dur_term

                    events.append({
                        "type": "note",
                        "note": n,
                        "n_idx": n_idx,
                        "onset": n.onset,
                        "dur": min(1.0, n.duration),
                        "need": ev_need,
                        "d_w": d_w,
                        "acc_w": acc_w,
                        "dot_w": dot_w,
                        "lyric_w": lyric_w,
                    })

                    if n.duration >= 2.0:
                        extra = int(round(n.duration)) - 1
                        for k in range(extra):
                            dash_need = 20.0 * 1.30 + 8.0 + 48.0
                            events.append({
                                "type": "dash",
                                "owning_n_idx": n_idx,
                                "onset": n.onset + 1.0 + k,
                                "dur": 1.0,
                                "need": dash_need,
                            })

                    t_cur = max(t_cur, n.onset + n.duration)

                # Sort events strictly by onset so dashes follow owning note in beat order
                events.sort(key=lambda e: e["onset"])

                tot_need = max(1.0, sum(e["need"] for e in events))
                scale = avail_w / tot_need

                cur_ev_x = mx0 + pad
                dashes_by_note: dict[int, list[float]] = {}

                for e in events:
                    w = e["need"] * scale
                    cx = cur_ev_x + w / 2.0
                    e["cx"] = cx
                    e["x0"] = cur_ev_x
                    e["x1"] = cur_ev_x + w

                    if e["type"] == "dash":
                        dashes_by_note.setdefault(e["owning_n_idx"], []).append(round(cx, 2))

                    cur_ev_x += w

                # Construct NoteLayoutBox now that all dash coordinates are collected
                for e in events:
                    if e["type"] == "note":
                        n = e["note"]
                        d_cx = e["cx"]
                        cur_x = d_cx - e["d_w"] / 2.0

                        # Exact glyph bounding box
                        g_x0 = cur_x - e["acc_w"]
                        g_x1 = cur_x + e["d_w"] + e["dot_w"]
                        digit_bbox = (round(g_x0, 2), round(melody_y + 17.0, 2), round(g_x1, 2), round(melody_y + 50.0, 2))

                        if e["lyric_w"] > 0 and lyrics_y is not None:
                            l_x0 = d_cx - e["lyric_w"] / 2.0
                            l_x1 = d_cx + e["lyric_w"] / 2.0
                            lyric_bbox = (round(l_x0, 2), round(lyrics_y, 2), round(l_x1, 2), round(lyrics_y + 28.0, 2))
                        else:
                            lyric_bbox = None

                        note_dash_xs = tuple(dashes_by_note.get(e["n_idx"], []))

                        note_layout_boxes.append(
                            NoteLayoutBox(
                                measure_index=global_m_idx,
                                note_index=e["n_idx"],
                                digit_bbox=digit_bbox,
                                lyric_bbox=lyric_bbox,
                                dash_xs=note_dash_xs,
                            )
                        )
                        note_positions[(global_m_idx, e["n_idx"])] = (p_idx, d_cx, melody_y, n.octave)

                measure_events_global[global_m_idx] = events

    return (
        page_height,
        margin_x,
        content_w,
        pages_systems,
        note_layout_boxes,
        note_positions,
        measure_bounds_global,
        measure_events_global,
    )


def get_lead_sheet_layout_boxes(
    lead: LeadSheet,
    page_width: int = 2400,
) -> list[NoteLayoutBox]:
    """Retrieve per-note layout geometry bounding boxes for validation."""
    _, _, _, _, note_boxes, _, _, _ = _compute_sheet_layout(lead, page_width)
    return note_boxes


def render_lead_sheet(
    lead: LeadSheet,
    page_width: int = 2400,
) -> tuple[list[bytes], ParsedSheet]:
    """Render a LeadSheet into A4 PNG images and matching ParsedSheet.

    Args:
        lead: The LeadSheet data structure with notes, chords, and metadata.
        page_width: Page width in pixels (default 2400 for standard A4).

    Returns:
        tuple of (list of PNG bytes, ParsedSheet)
    """
    (
        page_height,
        margin_x,
        content_w,
        pages_systems,
        _,
        note_positions,
        measure_bounds_global,
        measure_events_global,
    ) = _compute_sheet_layout(lead, page_width)

    # Typography fonts
    font_title = get_font(size=60, bold=True)
    font_meta = get_font(size=26, bold=True)
    font_meta_reg = get_font(size=24, bold=False)
    font_prov = get_font(size=20, bold=False)
    font_section = get_font(size=22, bold=True)
    font_chord = get_font(size=34, bold=True)  # Clearly distinct ~80% size
    font_digit = get_font(size=42, bold=True)
    font_acc = get_font(size=24, bold=True)
    font_lyric = get_font(size=24, bold=False)
    font_slash = get_font(size=36, bold=False)

    chord_color = (31, 63, 122)  # Dark blue #1f3f7a

    page_images: list[Image.Image] = []
    output_systems: list[System] = []

    for p_idx, page_sys in enumerate(pages_systems):
        img = Image.new("RGB", (page_width, page_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 1. Header (Page 0) or Running Header (Page > 0)
        if p_idx == 0:
            header_top = 80.0
            title_text = lead.title or "无标题"
            tb_title = draw.textbbox((0, 0), title_text, font=font_title)
            title_w = tb_title[2] - tb_title[0]
            draw.text(((page_width - title_w) / 2.0, header_top), title_text, font=font_title, fill=(0, 0, 0))

            tempo_str = ""
            if lead.tempo_bpm:
                t_val = int(lead.tempo_bpm) if abs(lead.tempo_bpm - int(lead.tempo_bpm)) < 1e-3 else lead.tempo_bpm
                tempo_str = f"  ♩={t_val}"
            meta_left = f"1={lead.key}  {lead.time_signature}{tempo_str}"
            draw.text((margin_x, header_top + 80.0), meta_left, font=font_meta, fill=(0, 0, 0))

            if lead.artist:
                artist_text = f"演唱/曲：{lead.artist}"
                tb_art = draw.textbbox((0, 0), artist_text, font=font_meta_reg)
                art_w = tb_art[2] - tb_art[0]
                draw.text((page_width - margin_x - art_w, header_top + 82.0), artist_text, font=font_meta_reg, fill=(40, 40, 40))

            prov_y = header_top + 124.0
            if lead.provenance:
                draw.text((margin_x, prov_y), lead.provenance, font=font_prov, fill=(110, 110, 110))

            if lead.sources:
                domains = []
                for s in lead.sources:
                    if s.title:
                        domains.append(s.title)
                    elif s.url:
                        netloc = urlparse(s.url).netloc
                        domains.append(netloc if netloc else s.url)
                if domains:
                    src_text = "参考来源：" + " · ".join(domains)
                    draw.text((margin_x, prov_y + 30.0), src_text, font=font_prov, fill=(125, 125, 125))
        else:
            running_text = f"{lead.title}  - {p_idx + 1} -"
            tb_run = draw.textbbox((0, 0), running_text, font=font_prov)
            run_w = tb_run[2] - tb_run[0]
            draw.text(((page_width - run_w) / 2.0, 70.0), running_text, font=font_prov, fill=(120, 120, 120))

        # 2. Render systems on this page
        for spec in page_sys:
            sys_y0 = spec["y0"]
            sys_y1 = spec["y1"]
            chord_y = sys_y0 + spec["chord_y_rel"]
            melody_y = sys_y0 + spec["melody_y_rel"]
            lyrics_y = sys_y0 + spec["lyrics_y_rel"] if spec["lyrics_y_rel"] is not None else None

            bar_top = chord_y - 4.0
            bar_bottom = (lyrics_y + 30.0) if lyrics_y is not None else (melody_y + 50.0 + 20.0)

            parsed_measures: list[Measure] = []

            for m_pos, global_m_idx in enumerate(spec["measure_indices"]):
                m = lead.measures[global_m_idx]
                mx0, mx1 = spec["m_bounds"][m_pos]
                mw = mx1 - mx0
                pad = max(24.0, mw * 0.04)
                avail_w = mw - 2.0 * pad

                # ----------------- Section Label Box -----------------
                if m.section:
                    tb_sec = draw.textbbox((0, 0), m.section, font=font_section)
                    sec_tw = tb_sec[2] - tb_sec[0]
                    bx0 = mx0 + 4.0
                    by0 = sys_y0 + 2.0
                    box_h = 34.0
                    bx1 = bx0 + sec_tw + 24.0
                    by1 = by0 + box_h
                    draw.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
                    tx = (bx0 + bx1) / 2.0 - (tb_sec[0] + tb_sec[2]) / 2.0
                    ty = (by0 + by1) / 2.0 - (tb_sec[1] + tb_sec[3]) / 2.0
                    draw.text((tx, ty), m.section, font=font_section, fill=(0, 0, 0))

                # ----------------- Melody / Rhythm Row -----------------
                is_chords_only_bar = lead.chords_only or not m.notes
                note_x_by_idx: dict[int, float] = {}

                if is_chords_only_bar:
                    num_slashes = max(1, int(round(m.beats)))
                    for b_idx in range(num_slashes):
                        sx = mx0 + pad + (b_idx + 0.5) / num_slashes * avail_w - 6.0
                        draw.text((sx, melody_y), "/", font=font_slash, fill=(160, 160, 160))
                else:
                    m_evs = measure_events_global.get(global_m_idx, [])
                    note_draw_data: list[dict] = []

                    # Draw events strictly in chronological order so dashes appear in beat sequence
                    for e in m_evs:
                        if e["type"] == "dash":
                            tb_dash = draw.textbbox((0, 0), "–", font=font_digit)
                            dash_w = tb_dash[2] - tb_dash[0]
                            draw.text((e["cx"] - dash_w / 2.0, melody_y), "–", font=font_digit, fill=(0, 0, 0))

                        elif e["type"] == "note":
                            n = e["note"]
                            n_idx = e["n_idx"]
                            d_cx = e["cx"]

                            # Accidental
                            acc_w = 0.0
                            if n.accidental != 0:
                                acc_str = "♯" if n.accidental == 1 else "♭"
                                tb_acc = draw.textbbox((0, 0), acc_str, font=font_acc)
                                acc_w = tb_acc[2] - tb_acc[0]

                            deg_str = str(n.degree)
                            tb_d = draw.textbbox((0, 0), deg_str, font=font_digit)
                            d_w = tb_d[2] - tb_d[0]

                            cur_x = d_cx - d_w / 2.0
                            note_x_by_idx[n_idx] = d_cx

                            if n.accidental != 0:
                                acc_x = cur_x - acc_w - 3.0
                                draw.text((acc_x, melody_y + 12.0), acc_str, font=font_acc, fill=(0, 0, 0))

                            draw.text((cur_x, melody_y), deg_str, font=font_digit, fill=(0, 0, 0))

                            # High octave dots (above digit)
                            if n.octave > 0:
                                for oct_i in range(n.octave):
                                    dot_cy = melody_y + 8.0 - oct_i * 9.0
                                    draw.ellipse([d_cx - 3.0, dot_cy - 3.0, d_cx + 3.0, dot_cy + 3.0], fill=(0, 0, 0))

                            # Augmentation dot at digit's mid-height
                            if abs(n.duration - 1.5) < 0.05 or abs(n.duration - 0.75) < 0.05:
                                aug_x = cur_x + d_w + 6.0
                                aug_y = melody_y + 33.5
                                draw.ellipse([aug_x - 3.0, aug_y - 3.0, aug_x + 3.0, aug_y + 3.0], fill=(0, 0, 0))

                            note_draw_data.append({
                                "note": n,
                                "digit_x": cur_x,
                                "digit_w": d_w,
                                "digit_cx": d_cx,
                                "n_idx": n_idx,
                            })

                    # Underline beaming per beat group
                    beat_groups: dict[int, list[dict]] = {}
                    for nd in note_draw_data:
                        b_group = int(math.floor(nd["note"].onset))
                        beat_groups.setdefault(b_group, []).append(nd)

                    beam1_y = melody_y + 55.0
                    beam2_y = melody_y + 63.0

                    for b_group, nds in beat_groups.items():
                        sub_b1 = [x for x in nds if x["note"].duration <= 0.75]
                        if sub_b1:
                            x_start = sub_b1[0]["digit_x"] - 2.0
                            x_end = sub_b1[-1]["digit_x"] + sub_b1[-1]["digit_w"] + 2.0
                            draw.line([(x_start, beam1_y), (x_end, beam1_y)], fill=(0, 0, 0), width=2)

                        sub_b2 = [x for x in nds if x["note"].duration <= 0.375]
                        if sub_b2:
                            x_start = sub_b2[0]["digit_x"] - 2.0
                            x_end = sub_b2[-1]["digit_x"] + sub_b2[-1]["digit_w"] + 2.0
                            draw.line([(x_start, beam2_y), (x_end, beam2_y)], fill=(0, 0, 0), width=2)

                    # Low octave dots below beams
                    for nd in note_draw_data:
                        n = nd["note"]
                        if n.octave < 0:
                            num_dots = abs(n.octave)
                            if n.duration <= 0.375:
                                base_dot_y = beam2_y + 9.0
                            elif n.duration <= 0.75:
                                base_dot_y = beam1_y + 9.0
                            else:
                                base_dot_y = melody_y + 50.0 + 8.0

                            for oct_i in range(num_dots):
                                dot_cy = base_dot_y + oct_i * 9.0
                                draw.ellipse([nd["digit_cx"] - 3.0, dot_cy - 3.0, nd["digit_cx"] + 3.0, dot_cy + 3.0], fill=(0, 0, 0))

                    # Lyrics row (centered under digit, well below dots)
                    if lyrics_y is not None:
                        for nd in note_draw_data:
                            syl = nd["note"].lyric.strip()
                            if syl:
                                tb_l = draw.textbbox((0, 0), syl, font=font_lyric)
                                lw = tb_l[2] - tb_l[0]
                                lx = nd["digit_cx"] - lw / 2.0
                                draw.text((lx, lyrics_y), syl, font=font_lyric, fill=(30, 30, 30))

                # ----------------- Chords Row -----------------
                parsed_chords: list[ChordSymbol] = []
                prev_cx = -999.0
                prev_cw = 0.0

                for c in m.chords:
                    c_onset = max(0.0, min(m.beats - 0.25, c.beat - 1.0))

                    # Find target X: align with note onset if exists, or interpolate
                    cx = mx0 + pad + (c_onset / m.beats) * avail_w
                    if not is_chords_only_bar and m.notes:
                        cand_notes = [n_i for n_i, n in enumerate(m.notes) if abs(n.onset - c_onset) < 0.2]
                        if cand_notes and cand_notes[0] in note_x_by_idx:
                            cx = note_x_by_idx[cand_notes[0]] - 10.0

                    # Prevent chord-on-chord overlap
                    if cx < prev_cx + prev_cw + 14.0:
                        cx = prev_cx + prev_cw + 14.0

                    draw.text(
                        (cx, chord_y),
                        c.raw,
                        font=font_chord,
                        fill=chord_color,
                        stroke_width=1,
                        stroke_fill=chord_color,
                    )
                    tb_c = draw.textbbox((cx, chord_y), c.raw, font=font_chord, stroke_width=1)
                    cw = tb_c[2] - tb_c[0]
                    prev_cx = cx
                    # Reserve room for the disagree dot so the next chord never touches it.
                    prev_cw = cw + (12.0 if c.agreement == "disagree" else 0.0) + 8.0

                    # Orange dot indicator for disagree
                    if c.agreement == "disagree":
                        dot_cx = tb_c[2] + 5.0
                        dot_cy = tb_c[1] + 5.0
                        draw.ellipse([dot_cx - 3.0, dot_cy - 3.0, dot_cx + 3.0, dot_cy + 3.0], fill=(217, 119, 6))

                    parsed_chords.append(
                        ChordSymbol(
                            raw=c.raw,
                            beat=c.beat,
                            bbox=(
                                round(tb_c[0] / page_width, 5),
                                round(tb_c[1] / page_height, 5),
                                round((tb_c[2] + (8.0 if c.agreement == "disagree" else 0.0)) / page_width, 5),
                                round(tb_c[3] / page_height, 5),
                            ),
                            confidence=1.0,
                            alternatives=list(c.alternatives),
                        )
                    )

                # ----------------- Barlines -----------------
                if m_pos == 0:
                    draw.line([(mx0, bar_top), (mx0, bar_bottom)], fill=(0, 0, 0), width=2)

                is_song_last_measure = (global_m_idx == len(lead.measures) - 1)
                if is_song_last_measure:
                    draw.line([(mx1 - 8.0, bar_top), (mx1 - 8.0, bar_bottom)], fill=(0, 0, 0), width=2)
                    draw.line([(mx1, bar_top), (mx1, bar_bottom)], fill=(0, 0, 0), width=6)
                else:
                    draw.line([(mx1, bar_top), (mx1, bar_bottom)], fill=(0, 0, 0), width=2)

                # Construct Measure model
                melody_str = "" if is_chords_only_bar else lead_measure_to_melody_string(m)
                lyrics_str = "".join(n.lyric for n in m.notes if n.lyric)

                parsed_measures.append(
                    Measure(
                        index=global_m_idx,
                        bbox=(
                            round(mx0 / page_width, 5),
                            round(sys_y0 / page_height, 5),
                            round(mx1 / page_width, 5),
                            round(sys_y1 / page_height, 5),
                        ),
                        beats=m.beats,
                        chords=parsed_chords,
                        melody=melody_str,
                        lyrics=lyrics_str,
                    )
                )

            sys_bbox = (
                round(margin_x / page_width, 5),
                round(sys_y0 / page_height, 5),
                round((page_width - margin_x) / page_width, 5),
                round(sys_y1 / page_height, 5),
            )
            sys_sec_label = lead.measures[spec["measure_indices"][0]].section if (lead.measures and lead.measures[spec["measure_indices"][0]].section) else None

            output_systems.append(
                System(
                    page=p_idx,
                    bbox=sys_bbox,
                    section_label=sys_sec_label,
                    measures=parsed_measures,
                )
            )

        # 3. Draw ties across notes
        for m_idx, m in enumerate(lead.measures):
            for n_idx, n in enumerate(m.notes):
                if not n.tie_to_next:
                    continue

                pos_a = note_positions.get((m_idx, n_idx))
                pos_b = None
                target_m_idx = m_idx

                if n_idx + 1 < len(m.notes):
                    pos_b = note_positions.get((m_idx, n_idx + 1))
                else:
                    for next_m_idx in range(m_idx + 1, len(lead.measures)):
                        if lead.measures[next_m_idx].notes:
                            pos_b = note_positions.get((next_m_idx, 0))
                            target_m_idx = next_m_idx
                            break

                if not pos_a or not pos_b:
                    continue

                p_a, xa, ya, oct_a = pos_a
                p_b, xb, yb, oct_b = pos_b

                mx0_a, mx1_a = measure_bounds_global.get(m_idx, (margin_x, page_width - margin_x))
                mx0_b, mx1_b = measure_bounds_global.get(target_m_idx, (margin_x, page_width - margin_x))

                if p_a == p_b and abs(ya - yb) < 10.0:
                    # Same row on same page: tie endpoints strictly above digit top and octave dots
                    if p_a == p_idx:
                        tie_y = min(_calc_tie_y(ya, oct_a), _calc_tie_y(yb, oct_b))
                        h = max(8.0, min(22.0, (xb - xa) * 0.14))
                        _draw_curved_tie(draw, xa + 6.0, xb - 6.0, tie_y, arc_height=h)
                else:
                    # Cross-row or cross-page
                    if p_a == p_idx:
                        tie_ya = _calc_tie_y(ya, oct_a)
                        arc_x1 = max(xa + 45.0, min(xa + 95.0, mx1_a - 4.0))
                        _draw_outgoing_tie(draw, xa + 6.0, tie_ya, arc_x1, arc_height=14.0)

                    if p_b == p_idx:
                        tie_yb = _calc_tie_y(yb, oct_b)
                        arc_x0 = min(xb - 45.0, max(xb - 95.0, mx0_b + 4.0))
                        _draw_incoming_tie(draw, arc_x0, xb - 6.0, tie_yb, arc_height=14.0)

        page_images.append(img)

    png_bytes_list: list[bytes] = []
    for img in page_images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes_list.append(buf.getvalue())

    header = SongHeader(
        title=lead.title,
        time_signature=lead.time_signature,
        tempo_bpm=lead.tempo_bpm,
        original_key=lead.key,
        raw=f"{lead.title} 1={lead.key} {lead.time_signature}" + (f" ♩={lead.tempo_bpm}" if lead.tempo_bpm else ""),
        chord_notation="number",
    )

    page_infos = [PageInfo(width=page_width, height=page_height) for _ in page_images]

    parsed_sheet = ParsedSheet(
        header=header,
        pages=page_infos,
        systems=output_systems,
        key_changes=[],
        warnings=[],
        issues=[],
        layout_confidence=1.0,
        layout_source="cv",
    )

    return png_bytes_list, parsed_sheet
