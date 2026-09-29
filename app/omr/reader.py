"""Content reader stage for Taiwanese band charts and jianpu sheets.

Takes authoritative PageGeometry from the CV layout stage (app/omr/layout.py) and
reads sheet content (header metadata, verbatim chord symbols by box ID, melody,
lyrics, rhythm/texture hints, and key change modulations) using LLMs on upscaled crops.
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import os
import re
from typing import Optional, TypeVar
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from app.models import (
    ChordSymbol,
    KeyChange,
    Measure,
    PageInfo,
    ParsedSheet,
    SongHeader,
    System,
)
from app.omr.geometry import GChordBox, GSystem, PageGeometry
from app.omr.reader_prompts import (
    HEADER_PROMPT,
    HeaderReading,
    SystemReading,
    build_system_crop_prompt,
)
from app.qa.llm import LLMUnavailable, _ask_opus, ask_json
from app.theory.chords import clean_raw_chord
from app.theory.keys import key_name_to_pc

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_READER_MODEL = os.environ.get("OMR_READER_MODEL", "claude-opus-5-5")

DEGREE_SEMITONES: dict[str, int] = {
    "1": 0,
    "1#": 1, "#1": 1, "b2": 1, "2b": 1,
    "2": 2,
    "2#": 3, "#2": 3, "b3": 3, "3b": 3,
    "3": 4,
    "4": 5,
    "4#": 6, "#4": 6, "b5": 6, "5b": 6,
    "5": 7,
    "5#": 8, "#5": 8, "b6": 8, "6b": 8,
    "6": 9,
    "6#": 10, "#6": 10, "b7": -2, "7b": -2,
    "7": 11,
}


CIRCLE_DIGIT_MAP: dict[str, str] = {
    "①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5",
    "⑥": "6", "⑦": "7", "⑧": "8", "⑨": "9", "⑩": "10",
    "❶": "1", "❷": "2", "❸": "3", "❹": "4", "❺": "5",
    "❻": "6", "❼": "7", "❽": "8", "❾": "9", "❿": "10",
}
DIAGONAL_SLASHES = ("╱", "⁄", "∕")


def parse_chord_symbol(raw_text: str, model_stacked: bool = False) -> tuple[str, bool]:
    """Parse chord text into (clean_raw, stacked).

    When a chord is printed as a diagonal/stacked fraction (circled mainland style,
    e.g. '⑦╱⑤', '①╱②m7-5'), output raw 'TOP/BOTTOM' in printed order WITHOUT circles
    (e.g. '7/5', '1/2m7-5') and set stacked=True.
    Circled '⑤7/9' is inline (stacked=False, raw '57/9').
    """
    if not raw_text or not isinstance(raw_text, str):
        return "", False

    s = raw_text.strip()
    has_diag_slash = any(ds in s for ds in DIAGONAL_SLASHES) or "\n" in s

    stacked = bool(model_stacked or has_diag_slash)
    if not stacked and "/" in s:
        parts = s.split("/", 1)
        has_circle_top = any(c in parts[0] for c in CIRCLE_DIGIT_MAP)
        has_circle_bottom = any(c in parts[1] for c in CIRCLE_DIGIT_MAP)
        if has_circle_top and has_circle_bottom:
            stacked = True

    for ds in DIAGONAL_SLASHES:
        s = s.replace(ds, "/")
    s = s.replace("\n", "/")

    for circ, digit in CIRCLE_DIGIT_MAP.items():
        s = s.replace(circ, digit)

    # Normalize internal whitespace around slash
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, stacked


def compute_key_change_semitones(raw: str, prev_key: Optional[str] = None) -> int:
    """Compute semitone shift from modulation text.

    Supports:
    - Taiwanese scale degree notation: '轉N調', '轉成N調' (2->+2, 3->+4, 4->+5, b7->-2)
    - Letter key notation: '1=X' relative to previous key (e.g. '轉 1=B' from C -> -1)
    - Key transition pair: '(Old - New)' (e.g. '(F# - Ab)' -> +2)
    """
    if not raw or not isinstance(raw, str):
        return 0

    s = raw.strip()

    # 1. Degree modulation: '轉N調' or '轉成N調'
    m_deg = re.search(r"轉(?:成)?\s*([#b]?[1-7][#b]?)\s*調", s)
    if m_deg:
        deg_str = m_deg.group(1).strip()
        if deg_str in DEGREE_SEMITONES:
            return DEGREE_SEMITONES[deg_str]

    # 2. '1=X' relative to prev_key
    m_key = re.search(r"1\s*=\s*([A-Ga-g][#b]?)", s)
    if m_key and prev_key:
        new_key = m_key.group(1).capitalize()
        try:
            p_prev = key_name_to_pc(prev_key)
            p_new = key_name_to_pc(new_key)
            diff = (p_new - p_prev) % 12
            if diff > 6:
                diff -= 12
            return diff
        except Exception:
            pass

    # 3. '(Old - New)' or '(Old -> New)'
    m_pair = re.search(r"([A-Ga-g][#b]?)\s*(?:-|->|→)\s*([A-Ga-g][#b]?)", s)
    if m_pair:
        k_old, k_new = m_pair.group(1).capitalize(), m_pair.group(2).capitalize()
        try:
            p_old = key_name_to_pc(k_old)
            p_new = key_name_to_pc(k_new)
            diff = (p_new - p_old) % 12
            if diff > 6:
                diff -= 12
            return diff
        except Exception:
            pass

    return 0


def parse_time_signature_to_beats(time_sig: str) -> float:
    """Parse time signature into beats per measure.

    Examples:
        '4/4' -> 4.0, '3/4' -> 3.0, '2/4' -> 2.0, '6/8' -> 6.0, '12/8' -> 12.0
    """
    if not time_sig or "/" not in time_sig:
        return 4.0
    parts = time_sig.strip().split("/")
    try:
        num = float(parts[0])
        return num
    except ValueError:
        return 4.0


def resolve_chord_beat(
    beat_geo: float,
    model_beat: Optional[float],
    time_sig: str,
    measure_beats: float,
    box_bbox: Optional[tuple[float, float, float, float]] = None,
    measure_x0_x1: Optional[tuple[float, float]] = None,
) -> float:
    """Determine final chord beat by combining geometry with model reading.

    Beats: use beat_geo unless the model reports a clearly different beat
    AND time signature != 4/4 handling requires rescaling (3/4, 6/8, 12/8: rescale
    beat_geo to Measure.beats).
    """
    time_sig_clean = time_sig.strip() if time_sig else "4/4"
    is_4_4 = time_sig_clean in {"4/4", "C"}

    if not is_4_4:
        # Rescale geometry beat to Measure.beats
        if box_bbox and measure_x0_x1 and (measure_x0_x1[1] > measure_x0_x1[0]):
            box_cx = (box_bbox[0] + box_bbox[2]) / 2.0
            frac = max(0.0, min(1.0, (box_cx - measure_x0_x1[0]) / (measure_x0_x1[1] - measure_x0_x1[0])))
        else:
            frac = max(0.0, min(1.0, (beat_geo - 1.0) / 4.0))

        rescaled = 1.0 + frac * measure_beats
        rescaled = max(1.0, min(measure_beats, round(rescaled * 2.0) / 2.0))

        if model_beat is not None and abs(model_beat - rescaled) >= 1.0 and 1.0 <= model_beat <= measure_beats:
            return model_beat
        return rescaled
    else:
        # 4/4 meter
        if model_beat is not None and abs(model_beat - beat_geo) >= 1.0 and 1.0 <= model_beat <= measure_beats:
            return model_beat
        return beat_geo


def ask_reader(
    prompt: str,
    schema: type[T],
    images: list[bytes],
    *,
    model: str,
    timeout_s: float = 90.0,
) -> T:
    """Call the LLM with structured output, supporting Gemini and Claude Opus."""
    if os.environ.get("QA_OFFLINE") == "1":
        raise LLMUnavailable("QA_OFFLINE=1: LLM operations are disabled.")

    if "claude" in model or "opus" in model:
        reg = os.environ.get("QA_ARBITER_REGION", "global")
        return _ask_opus(
            prompt,
            schema,
            images,
            model=model,
            region=reg,
            timeout_s=timeout_s,
        )
    else:
        return ask_json(
            prompt,
            schema,
            images,
            role="reader",
            model=model,
            timeout_s=timeout_s,
        )


def annotate_system_crop(
    page_img: Image.Image,
    sys_geom: GSystem,
    page_idx: int,
) -> tuple[bytes, int]:
    """Crop a system row from the page, upscale long side >= 2000 px, and draw guides."""
    pw, ph = page_img.size
    s_bbox = sys_geom.bbox

    # Add a slight vertical margin of 6px
    margin_y = 6.0 / max(1, ph)
    y0_crop = max(0.0, s_bbox[1] - margin_y)
    y1_crop = min(1.0, s_bbox[3] + margin_y)
    x0_crop = max(0.0, s_bbox[0] - 0.01)
    x1_crop = min(1.0, s_bbox[2] + 0.01)

    px0 = int(round(x0_crop * pw))
    py0 = int(round(y0_crop * ph))
    px1 = int(round(x1_crop * pw))
    py1 = int(round(y1_crop * ph))

    cw = max(1, px1 - px0)
    ch = max(1, py1 - py0)
    scale = max(1.0, 2000.0 / max(cw, ch))
    new_w = int(round(cw * scale))
    new_h = int(round(ch * scale))

    crop = page_img.crop((px0, py0, px1, py1)).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Top padding for guide badges
    top_pad = 32
    annotated = Image.new("RGB", (new_w, new_h + top_pad), (255, 255, 255))
    annotated.paste(crop, (0, top_pad))
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("app/render/fonts/NotoSansSC.ttf", size=15)
    except Exception:
        font = ImageFont.load_default()

    # 1. Draw measure guides
    for m_idx, m in enumerate(sys_geom.measures):
        mx0 = (m.x0 * pw - px0) * scale
        mx1 = (m.x1 * pw - px0) * scale
        draw.line([(mx0, top_pad), (mx0, top_pad + new_h)], fill=(0, 100, 220), width=2)
        draw.line([(mx1, top_pad), (mx1, top_pad + new_h)], fill=(0, 100, 220), width=2)
        draw.text((mx0 + 6, 6), f"[Measure {m_idx}]", fill=(0, 80, 200), font=font)

    # 2. Draw numbered chord boxes
    for b_idx, cb in enumerate(sys_geom.chord_boxes, 1):
        bx0 = (cb.bbox[0] * pw - px0) * scale
        by0 = top_pad + (cb.bbox[1] * ph - py0) * scale
        bx1 = (cb.bbox[2] * pw - px0) * scale
        by1 = top_pad + (cb.bbox[3] * ph - py0) * scale

        draw.rectangle([bx0 - 2, by0 - 2, bx1 + 2, by1 + 2], outline=(220, 0, 0), width=2)
        badge_y = max(4, by0 - 22)
        badge_w = 36
        draw.rectangle([bx0, badge_y, bx0 + badge_w, badge_y + 18], fill=(255, 230, 0), outline=(200, 0, 0))
        draw.text((bx0 + 3, badge_y), f"#{b_idx}", fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), len(sys_geom.chord_boxes)


def _read_single_system(
    task_item: tuple[int, int, GSystem, bytes, int],
    chord_notation: str,
    target_model: str,
) -> tuple[tuple[int, int], SystemReading]:
    p_idx, s_idx, sys_geom, crop_bytes, num_boxes = task_item
    is_chord_only = "chord_only" in sys_geom.notes
    prompt = build_system_crop_prompt(
        system_index_on_page=s_idx,
        total_measures=len(sys_geom.measures),
        num_boxes=num_boxes,
        chord_notation=chord_notation,
        chord_only=is_chord_only,
    )
    res = ask_reader(prompt, SystemReading, images=[crop_bytes], model=target_model)
    return (p_idx, s_idx), res


def read_sheet(
    images: list[bytes],
    geoms: list[PageGeometry],
    *,
    model: Optional[str] = None,
) -> ParsedSheet:
    """Read sheet content using geometry as authoritative structure.

    Args:
        images: List of raw page image bytes.
        geoms: List of PageGeometry objects from CV layout.
        model: Optional model name ('gemini-2.5-pro', 'claude-opus-5-5', etc.).

    Returns:
        ParsedSheet with geometry-anchored bboxes, read chords, melody, and key changes.
    """
    target_model = model or os.environ.get("OMR_READER_MODEL", DEFAULT_READER_MODEL)

    if not images or not geoms:
        return ParsedSheet(
            header=SongHeader(),
            pages=[],
            systems=[],
            layout_confidence=0.0,
            warnings=["No images or geometry provided."],
        )

    # 1. Parse header from page 0
    header = SongHeader()
    warnings: list[str] = []

    try:
        p0_img = Image.open(io.BytesIO(images[0])).convert("RGB")
        pw0, ph0 = p0_img.size
        h_band = geoms[0].header_band
        if h_band:
            hy0 = max(0, int(round(h_band[0] * ph0)))
            hy1 = min(ph0, int(round(h_band[1] * ph0)))
        else:
            first_sys_y0 = geoms[0].systems[0].bbox[1] if geoms[0].systems else 0.15
            hy0 = 0
            hy1 = max(10, int(round(min(first_sys_y0, 0.15) * ph0)))

        h_crop = p0_img.crop((0, hy0, pw0, hy1))
        h_scale = max(1.0, 2000.0 / max(1, h_crop.width))
        h_crop_up = h_crop.resize(
            (int(round(h_crop.width * h_scale)), int(round(h_crop.height * h_scale))),
            Image.Resampling.LANCZOS,
        )
        h_buf = io.BytesIO()
        h_crop_up.save(h_buf, format="JPEG", quality=92)
        h_bytes = h_buf.getvalue()

        h_read = ask_reader(HEADER_PROMPT, HeaderReading, images=[h_bytes], model=target_model)
        header = SongHeader(
            title=h_read.title,
            style=h_read.style,
            time_signature=h_read.time_signature or "4/4",
            tempo_bpm=h_read.tempo_bpm,
            original_key=h_read.original_key,
            male_key=h_read.male_key,
            female_key=h_read.female_key,
            chord_notation=h_read.chord_notation,
            raw=h_read.raw,
        )
        # Handle '1=X' for letter charts if original_key not set
        if not header.original_key and h_read.raw:
            m_1x = re.search(r"1\s*=\s*([A-Ga-g][#b]?)", h_read.raw)
            if m_1x:
                header.original_key = m_1x.group(1).capitalize()
    except Exception as exc:
        logger.warning("Header extraction failed: %s", exc)
        warnings.append(f"Header extraction failed: {exc}")
        header = SongHeader(time_signature="4/4", chord_notation="number")

    measure_beats = parse_time_signature_to_beats(header.time_signature)

    # 2. Prepare crops and tasks for all systems
    system_tasks: list[tuple[int, int, GSystem, bytes, int]] = []
    # (page_idx, s_idx, sys_geom, crop_bytes, num_boxes)

    for p_idx, (img_bytes, geom) in enumerate(zip(images, geoms)):
        page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        for s_idx, sys_geom in enumerate(geom.systems):
            crop_bytes, num_boxes = annotate_system_crop(page_img, sys_geom, p_idx)
            system_tasks.append((p_idx, s_idx, sys_geom, crop_bytes, num_boxes))

    # 3. Read systems concurrently
    system_readings: dict[tuple[int, int], SystemReading] = {}
    max_workers = min(len(system_tasks), 6) if system_tasks else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_read_single_system, task, header.chord_notation, target_model): task
            for task in system_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task_key = future_to_task[future][:2]
            try:
                key, reading = future.result()
                system_readings[key] = reading
            except Exception as exc:
                logger.error("Failed to read system %s: %s", task_key, exc)
                warnings.append(f"System {task_key} reading failed: {exc}")
                system_readings[task_key] = SystemReading()

    # 4. Assemble ParsedSheet with global measure indices and KeyChanges
    all_systems: list[System] = []
    key_changes: list[KeyChange] = []
    global_measure_idx = 0
    current_key = header.original_key

    for p_idx, geom in enumerate(geoms):
        for s_idx, sys_geom in enumerate(geom.systems):
            reading = system_readings.get((p_idx, s_idx), SystemReading())
            sys_start_measure_idx = global_measure_idx
            measures_list: list[Measure] = []

            # Pre-index box readings by box_id
            box_readings_map: dict[int, str] = {}
            box_beats_map: dict[int, Optional[float]] = {}
            box_stacked_map: dict[int, bool] = {}
            for br in reading.chord_boxes:
                box_readings_map[br.box_id] = br.text
                if br.beat is not None:
                    box_beats_map[br.box_id] = br.beat
                box_stacked_map[br.box_id] = br.stacked

            # Pre-group extra chords by measure index
            extra_chords_by_m: dict[int, list[tuple[str, float, bool]]] = {}
            for ec in reading.extra_chords:
                m_ec = ec.measure_index
                if m_ec not in extra_chords_by_m:
                    extra_chords_by_m[m_ec] = []
                extra_chords_by_m[m_ec].append((ec.text, ec.beat, ec.stacked))

            # Pre-index measure content readings by measure_index
            measure_content_map = {mc.measure_index: mc for mc in reading.measures}
            is_chord_only = "chord_only" in sys_geom.notes
            if is_chord_only:
                warnings.append(f"[v2_chord_only] page={p_idx} system={s_idx}")

            for m_idx, g_m in enumerate(sys_geom.measures):
                chords_in_measure: list[ChordSymbol] = []
                m_x0_x1 = (g_m.x0, g_m.x1)

                # 1. Chords from numbered boxes
                for b_idx, cb in enumerate(sys_geom.chord_boxes, 1):
                    if cb.measure_index_in_system == m_idx:
                        raw_text = box_readings_map.get(b_idx, "")
                        is_stacked = box_stacked_map.get(b_idx, False)
                        cleaned, stacked = parse_chord_symbol(raw_text, model_stacked=is_stacked)
                        if cleaned and cleaned.lower() not in {"", "none", "null", "no", "x"}:
                            beat = resolve_chord_beat(
                                beat_geo=cb.beat_geo,
                                model_beat=box_beats_map.get(b_idx),
                                time_sig=header.time_signature,
                                measure_beats=measure_beats,
                                box_bbox=cb.bbox,
                                measure_x0_x1=m_x0_x1,
                            )
                            chords_in_measure.append(
                                ChordSymbol(
                                    raw=cleaned,
                                    beat=beat,
                                    bbox=cb.bbox if cb.boxed else None,
                                    confidence=1.0,
                                    stacked=stacked,
                                )
                            )

                # 2. Extra chords
                for extra_text, extra_beat, extra_stacked in extra_chords_by_m.get(m_idx, []):
                    cleaned_extra, stacked_extra = parse_chord_symbol(extra_text, model_stacked=extra_stacked)
                    if cleaned_extra and cleaned_extra.lower() not in {"", "none", "null", "no", "x"}:
                        # Filter non-chord voicing digits and rests
                        if re.match(r"^(?:0|\d{3,})$", cleaned_extra):
                            continue
                        # Deduplicate if already present at roughly the same beat
                        already_present = any(
                            c.raw == cleaned_extra and abs(c.beat - extra_beat) < 0.5
                            for c in chords_in_measure
                        )
                        if not already_present:
                            final_b = extra_beat if (1.0 <= extra_beat <= measure_beats) else 1.0
                            chords_in_measure.append(
                                ChordSymbol(
                                    raw=cleaned_extra,
                                    beat=final_b,
                                    bbox=None,
                                    confidence=0.9,
                                    stacked=stacked_extra,
                                )
                            )

                chords_in_measure.sort(key=lambda c: c.beat)

                # Measure content
                mc = measure_content_map.get(m_idx)
                melody = "" if is_chord_only else (mc.melody if mc else "")
                lyrics = "" if is_chord_only else (mc.lyrics if mc else "")
                bass_hint = mc.bass_hint if mc else None
                rhythm_hint = mc.rhythm_hint if mc else None
                fill = mc.fill if mc else False
                is_stop = mc.is_stop if mc else False

                m_bbox = (g_m.x0, sys_geom.bbox[1], g_m.x1, sys_geom.bbox[3])
                measure = Measure(
                    index=global_measure_idx,
                    bbox=m_bbox,
                    beats=measure_beats,
                    chords=chords_in_measure,
                    melody=melody,
                    lyrics=lyrics,
                    bass_hint=bass_hint,
                    rhythm_hint=rhythm_hint,
                    fill=fill,
                    is_stop=is_stop,
                )
                measures_list.append(measure)
                global_measure_idx += 1

            system = System(
                page=p_idx,
                bbox=sys_geom.bbox,
                section_label=reading.section_label,
                measures=measures_list,
            )
            all_systems.append(system)

            if reading.measures_seen is not None:
                warnings.append(
                    f"[v2_reader_measures_seen] page={p_idx} system={s_idx} measures_seen={reading.measures_seen}"
                )

            if sys_geom.measures and sys_geom.measures[-1].right_double_bar:
                warnings.append(
                    f"[v2_final_barline] page={p_idx} system={s_idx} right_double_bar=True"
                )

            has_fine = False
            if reading.section_label and re.search(r"\b(?:fine|end|outro)\b|完", reading.section_label, re.I):
                has_fine = True
            for mc in reading.measures:
                if mc.lyrics and re.search(r"\b(?:fine|end)\b|\|\]|\|\||完", mc.lyrics, re.I):
                    has_fine = True
                if mc.rhythm_hint and re.search(r"\b(?:fine|end)\b|\|\]|\|\|", mc.rhythm_hint, re.I):
                    has_fine = True
            if has_fine:
                warnings.append(f"[v2_end_mark] page={p_idx} system={s_idx} fine=True")

            # Key change modulation in this system
            if reading.key_change and reading.key_change.raw:
                kc_raw = reading.key_change.raw.strip()
                kc_m_idx = reading.key_change.measure_index
                kc_global = sys_start_measure_idx + kc_m_idx
                semitones = compute_key_change_semitones(kc_raw, prev_key=current_key)
                key_changes.append(
                    KeyChange(
                        at_measure=kc_global,
                        raw=kc_raw,
                        semitones=semitones,
                    )
                )

    pages_info = [PageInfo(width=g.width, height=g.height) for g in geoms]
    min_confidence = min((g.confidence for g in geoms), default=1.0)
    warnings_list = list(warnings)
    warnings_list.append("[v2_geometry] Classical CV layout authoritative boundaries")

    return ParsedSheet(
        header=header,
        pages=pages_info,
        systems=all_systems,
        key_changes=key_changes,
        warnings=warnings_list,
        layout_confidence=min_confidence,
        layout_source="cv",
    )
