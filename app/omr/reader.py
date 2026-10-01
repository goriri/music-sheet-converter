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
from pydantic import BaseModel, Field

from app.models import (
    ChordSymbol,
    KeyChange,
    Measure,
    PageInfo,
    ParsedSheet,
    QualityIssue,
    SongHeader,
    System,
)
from app.omr.geometry import GChordBox, GMeasure, GSystem, PageGeometry
from app.omr.reader_prompts import (
    HEADER_PROMPT,
    HeaderReading,
    SystemReading,
    build_system_crop_prompt,
)
from app.qa.llm import LLMUnavailable, _ask_opus, ask_json
from app.qa.omr_verify import is_valid_chord_grammar
from app.theory.chords import clean_raw_chord
from app.theory.keys import key_name_to_pc

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_READER_MODEL = os.environ.get("OMR_READER_MODEL", "claude-opus-5-5")


class SystemReadingWithIndex(SystemReading):
    system_index: Optional[int] = Field(
        None,
        description="0-based index of this system on the page, matching the '[System N]' guide on the image.",
    )


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


def is_valid_standalone_chord(token: str, notation: str = "number") -> bool:
    """Check if token is a valid standalone chord symbol.

    In Taiwanese number notation:
    - Standalone slash chords must have a denominator that is a valid bass degree (1-7 with opt accidental).
    - Denominators like 9, 11, 13 are chord extensions (e.g. 57/9), NOT valid bass notes on standalone slash chords.
    """
    if not token or not token.strip():
        return False
    t = token.strip()
    for circ, digit in CIRCLE_DIGIT_MAP.items():
        t = t.replace(circ, digit)
    for ds in DIAGONAL_SLASHES:
        t = t.replace(ds, "/")
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s+", "", t)

    if notation == "number":
        if "/" in t:
            parts = t.split("/", 1)
            bot = parts[1].strip()
            # Denominator must be a valid scale degree bass 1..7 (e.g. 1..7, b7, #4, 7b)
            if not re.match(r"^[b#♭♯]?[1-7][b#♭♯]?$", bot):
                return False
        return is_valid_chord_grammar(t, notation="number")
    else:
        if "/" in t:
            parts = t.split("/", 1)
            bot = parts[1].strip()
            if not re.match(r"^[A-Ga-g][b#♭♯]?$", bot):
                return False
        return is_valid_chord_grammar(t, notation="letter")


def resolve_chord_tokens(
    raw_text: str,
    box_geom_count: Optional[int] = None,
    notation: str = "number",
) -> tuple[list[str], bool, bool]:
    """Resolve whitespace-separated chord token according to merged token rules.

    Rules:
    1. If every whitespace-separated part is a valid chord on its own AND joining them
       is not a valid single chord -> split into separate chords.
    2. If joining gives a valid chord and the parts are not both valid (e.g. '5 7/9' -> '57/9',
       since '7/9' has an invalid bass) -> join.
    3. If both readings are valid -> use the box count from geometry; if that's inconclusive -> needs_review.

    Returns:
        (tokens_list, is_joined, needs_review_flag)
    """
    if not raw_text or not raw_text.strip():
        return [], False, False

    raw = raw_text.strip()
    # Never split a digit run like '46' (leave to grammar check -> re-read -> needs_review)
    if re.match(r"^\d+$", raw):
        return [raw], False, False

    parts = raw.split()
    if len(parts) <= 1:
        return [raw], False, False

    all_parts_valid = all(is_valid_standalone_chord(p, notation) for p in parts)
    joined = re.sub(r"\s*/\s*", "/", raw)
    joined = re.sub(r"\s+", "", joined)
    joined_valid = is_valid_chord_grammar(joined, notation)

    if all_parts_valid and not joined_valid:
        return parts, False, False
    elif joined_valid and not all_parts_valid:
        return [joined], True, False
    elif all_parts_valid and joined_valid:
        if box_geom_count is not None and box_geom_count >= len(parts):
            return parts, False, False
        elif box_geom_count == 1:
            return [joined], True, False
        else:
            # Inconclusive
            return [joined], True, True
    else:
        return [raw], False, True


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

    # Normalize internal whitespace around slash and within chord symbol
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", "", s).strip()
    if s.startswith("b4m"):
        s = "4m" + s[3:]
    # Repair truncated maj7: in scale degree or letter notation, 'maj' without a number suffix is truncated 'maj7'
    # (e.g. '5/1maj' -> '5/1maj7', '1maj' -> '1maj7', '4maj' -> '4maj7')
    s = re.sub(r"maj$", "maj7", s, flags=re.IGNORECASE)
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


def count_read_chord_symbols(reading: SystemReading, notation: str = "number") -> int:
    """Count total valid chord symbols read in a system."""
    if not reading:
        return 0
    cnt = 0
    for br in reading.chord_boxes:
        txt = br.text.strip()
        if txt and txt.lower() not in {"", "none", "null", "no", "x"}:
            tokens, _, _ = resolve_chord_tokens(txt, notation=notation)
            for tok in tokens:
                c_clean, _ = parse_chord_symbol(tok)
                if c_clean and c_clean.lower() not in {"", "none", "null", "no", "x"}:
                    cnt += 1
    for ec in reading.extra_chords:
        etxt = ec.text.strip()
        if etxt and etxt.lower() not in {"", "none", "null", "no", "x"}:
            if not re.match(r"^(?:0|\d{3,})$", etxt):
                c_clean, _ = parse_chord_symbol(etxt)
                if c_clean:
                    cnt += 1
    return cnt


def annotate_system_crop(
    page_img: Image.Image,
    sys_geom: GSystem,
    page_idx: int,
    system_idx: int = 0,
) -> tuple[bytes, int]:
    """Crop a system row from the page, upscale long side >= 2000 px, and draw guides."""
    pw, ph = page_img.size
    s_bbox = sys_geom.bbox

    # Add a slight vertical margin of 6px
    margin_y = 6.0 / max(1, ph)
    raw_sx0, raw_sy0, raw_sx1, raw_sy1 = s_bbox
    sx0 = max(0.0, min(1.0, min(raw_sx0, raw_sx1)))
    sx1 = max(0.0, min(1.0, max(raw_sx0, raw_sx1)))
    sy0 = max(0.0, min(1.0, min(raw_sy0, raw_sy1)))
    sy1 = max(0.0, min(1.0, max(raw_sy0, raw_sy1)))
    if sx1 <= sx0:
        sx1 = min(1.0, sx0 + 0.1)
        if sx1 <= sx0:
            sx0 = max(0.0, sx1 - 0.1)
    if sy1 <= sy0:
        sy1 = min(1.0, sy0 + 0.05)
        if sy1 <= sy0:
            sy0 = max(0.0, sy1 - 0.05)

    y0_crop = max(0.0, sy0 - margin_y)
    y1_crop = min(1.0, sy1 + margin_y)
    x0_crop = max(0.0, sx0 - 0.01)
    x1_crop = min(1.0, sx1 + 0.01)

    px0 = max(0, min(pw - 1, int(round(x0_crop * pw))))
    py0 = max(0, min(ph - 1, int(round(y0_crop * ph))))
    px1 = max(px0 + 1, min(pw, int(round(x1_crop * pw))))
    py1 = max(py0 + 1, min(ph, int(round(y1_crop * ph))))

    cw = max(1, px1 - px0)
    ch = max(1, py1 - py0)
    scale = max(1.0, 2000.0 / max(cw, ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

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

    try:
        # Draw system indicator badge at top-left
        draw.text((6, 6), f"[System {system_idx}] (Page {page_idx + 1})", fill=(180, 0, 0), font=font)

        # 1. Draw measure guides
        for m_idx, m in enumerate(sys_geom.measures):
            mx0_raw = (min(m.x0, m.x1) * pw - px0) * scale
            mx1_raw = (max(m.x0, m.x1) * pw - px0) * scale
            mx0 = max(0.0, min(float(new_w), mx0_raw))
            mx1 = max(0.0, min(float(new_w), mx1_raw))
            draw.line([(mx0, top_pad), (mx0, top_pad + new_h)], fill=(0, 100, 220), width=2)
            draw.line([(mx1, top_pad), (mx1, top_pad + new_h)], fill=(0, 100, 220), width=2)
            draw.text((min(mx0, mx1) + 6, 6), f"[Measure {m_idx}]", fill=(0, 80, 200), font=font)

        # 2. Draw numbered chord boxes
        for b_idx, cb in enumerate(sys_geom.chord_boxes, 1):
            cb_x0, cb_y0, cb_x1, cb_y1 = cb.bbox
            raw_bx0 = (min(cb_x0, cb_x1) * pw - px0) * scale
            raw_bx1 = (max(cb_x0, cb_x1) * pw - px0) * scale
            raw_by0 = top_pad + (min(cb_y0, cb_y1) * ph - py0) * scale
            raw_by1 = top_pad + (max(cb_y0, cb_y1) * ph - py0) * scale

            pad_box_l = int(round(0.015 * pw * scale)) if not cb.boxed else 2
            bx0_draw = max(0.0, raw_bx0 - pad_box_l)
            bx1_draw = max(bx0_draw, raw_bx1 + 2)
            by0_draw = max(0.0, raw_by0 - 2)
            by1_draw = max(by0_draw, raw_by1 + 2)

            r_x0 = max(0.0, min(float(new_w), min(bx0_draw, bx1_draw)))
            r_x1 = max(r_x0, min(float(new_w), max(bx0_draw, bx1_draw)))
            r_y0 = max(0.0, min(float(new_h + top_pad), min(by0_draw, by1_draw)))
            r_y1 = max(r_y0, min(float(new_h + top_pad), max(by0_draw, by1_draw)))

            draw.rectangle([r_x0, r_y0, r_x1, r_y1], outline=(220, 0, 0), width=2)

            badge_y = max(4.0, raw_by0 - 22)
            badge_w = 36.0
            b_x0 = max(0.0, min(float(new_w), raw_bx0))
            b_x1 = max(b_x0, min(float(new_w), raw_bx0 + badge_w))
            b_y0 = max(0.0, min(float(new_h + top_pad), badge_y))
            b_y1 = max(b_y0, min(float(new_h + top_pad), badge_y + 18))
            draw.rectangle([b_x0, b_y0, b_x1, b_y1], fill=(255, 230, 0), outline=(200, 0, 0))
            draw.text((b_x0 + 3, b_y0), f"#{b_idx}", fill=(0, 0, 0), font=font)
    except Exception as exc:
        logger.warning("Error drawing annotations on system crop: %s", exc)

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), len(sys_geom.chord_boxes)


def check_reading_inconsistent(
    reading: SystemReading,
    sys_geom: GSystem,
    notation: str = "number",
) -> tuple[bool, str]:
    """Check if system reading is inconsistent with geometry.

    Geometry box count is a LOWER BOUND: layout recall on boxed charts is only
    50-75%, and reader discovering extra chords is intentional. Never flag an
    inconsistency when chord_count >= num_geom_boxes.

    Returns (is_inconsistent, reason).
    """
    num_geom_boxes = len(sys_geom.chord_boxes)
    chord_count = count_read_chord_symbols(reading, notation)

    # Geometry boxes are a lower bound:
    if num_geom_boxes > 0 and chord_count < num_geom_boxes:
        return True, f"chord symbols read ({chord_count}) < geometry boxes ({num_geom_boxes})"

    # Duplicated readings check
    read_boxes = reading.chord_boxes if reading else []
    cleaned_chords: list[str] = []
    for br in read_boxes:
        txt = br.text.strip()
        if txt and txt.lower() not in {"", "none", "null", "no", "x"}:
            tokens, _, _ = resolve_chord_tokens(txt, box_geom_count=num_geom_boxes, notation=notation)
            for tok in tokens:
                c_clean, _ = parse_chord_symbol(tok)
                if c_clean:
                    cleaned_chords.append(c_clean)

    if len(cleaned_chords) >= 2 and len(cleaned_chords) % 2 == 0:
        half = len(cleaned_chords) // 2
        if cleaned_chords[:half] == cleaned_chords[half:] and num_geom_boxes == half:
            return True, f"duplicated reading sequence {cleaned_chords[:half]} where geometry has {num_geom_boxes} boxes"

    return False, ""


def find_measure_index_for_x(cb_cx: float, measures: list[GMeasure]) -> int:
    """Find measure index by x-centre of chord box against barline x-positions."""
    if not measures:
        return 0
    if cb_cx <= measures[0].x0:
        return 0
    if cb_cx >= measures[-1].x1:
        return len(measures) - 1
    for idx, m in enumerate(measures):
        if m.x0 <= cb_cx <= m.x1:
            return idx
    # Fallback to closest measure center
    return min(
        range(len(measures)),
        key=lambda i: abs(cb_cx - (measures[i].x0 + measures[i].x1) / 2.0),
    )


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
    prompt += (
        f"\nIMPORTANT GUIDES:\n"
        f"- Top-left badge shows '[System {s_idx}]' (Page {p_idx + 1}). Include system_index={s_idx} in your structured response.\n"
    )
    prompt += (
        "\nIMPORTANT FOR ACCIDENTALS AND SUPERSCRIPTS ON CIRCLED CHORDS:\n"
        "- In circled number notation, accidentals ('b' flat or '#' sharp) are often printed directly before or above-left of the circled number (e.g. 'b⑦add9' -> 'b7add9', 'b⑦' -> 'b7', 'b⑥' -> 'b6', '#④m7-5' -> '#4m7-5'). Note: degree 4 is natural (e.g. '4m', never 'b4m').\n"
        "- ALWAYS inspect the region immediately to the left of the circled number for a small 'b' or '#' and include it as a prefix in the chord text!\n"
        "- For minor 7th chords with flat fifth (half-diminished, e.g. 'b6/2m7-5', '1/2m7-5', '6m7-5', '2m7-5'), the '-5' or '(b5)' is printed as a small superscript directly after 'm7'. ALWAYS inspect the region after 'm7' for a superscript '-5' or '(b5)' and transcribe as '-5' (e.g. 'b6/2m7-5', '6m7-5', '2m7-5')!\n"
    )
    try:
        res = ask_reader(prompt, SystemReadingWithIndex, images=[crop_bytes], model=target_model)
    except Exception:
        res = ask_reader(prompt, SystemReading, images=[crop_bytes], model=target_model)

    inconsistent, reason = check_reading_inconsistent(res, sys_geom, chord_notation)
    if inconsistent and num_boxes > 0:
        logger.info(
            "System (p=%d, s=%d) reading inconsistent with geometry (%s); re-reading row once...",
            p_idx,
            s_idx,
            reason,
        )
        retry_prompt = prompt + (
            f"\n\nIMPORTANT CORRECTION: The previous reading had {reason}.\n"
            f"Please carefully transcribe all chords, ensuring you do not miss any of the {num_boxes} numbered boxes or additional chords in the measures."
        )
        try:
            try:
                retry_res = ask_reader(retry_prompt, SystemReadingWithIndex, images=[crop_bytes], model=target_model)
            except Exception:
                retry_res = ask_reader(retry_prompt, SystemReading, images=[crop_bytes], model=target_model)
            retry_inconsistent, _ = check_reading_inconsistent(retry_res, sys_geom, chord_notation)
            cnt_retry = count_read_chord_symbols(retry_res, chord_notation)
            cnt_res = count_read_chord_symbols(res, chord_notation)
            # Never replace with a retry that drops chords (geometry is lower bound)
            if not retry_inconsistent and cnt_retry >= cnt_res:
                res = retry_res
            elif cnt_retry > cnt_res:
                res = retry_res
        except Exception as exc:
            logger.warning("System (p=%d, s=%d) re-read failed: %s", p_idx, s_idx, exc)

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

        header_prompt = HEADER_PROMPT.replace(
            "or if indicated by '1=X' without degree chords.",
            "only if chord symbols throughout the sheet are written with letters (C, G/B, etc.). '1=X' is a key signature marking present on both number and letter charts, NOT an indication of letter notation.",
        )
        h_read = ask_reader(header_prompt, HeaderReading, images=[h_bytes], model=target_model)
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
        try:
            page_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("Could not open page image %d: %s", p_idx, exc)
            continue
        for s_idx, sys_geom in enumerate(geom.systems):
            try:
                crop_bytes, num_boxes = annotate_system_crop(page_img, sys_geom, p_idx, s_idx)
                system_tasks.append((p_idx, s_idx, sys_geom, crop_bytes, num_boxes))
            except Exception as exc:
                logger.warning("Failed to prepare crop for system (%d, %d): %s", p_idx, s_idx, exc)

    # 3. Read systems concurrently
    system_readings: dict[tuple[int, int], SystemReading] = {}
    max_workers = min(len(system_tasks), 6) if system_tasks else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_read_single_system, task, header.chord_notation, target_model): task
            for task in system_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            orig_p_idx, orig_s_idx = task[0], task[1]
            try:
                key, reading = future.result()
                # Map using system_index from the reading if valid
                read_s_idx = reading.system_index if isinstance(reading, SystemReadingWithIndex) else None
                if read_s_idx is not None and 0 <= read_s_idx < len(geoms[orig_p_idx].systems):
                    assigned_key = (orig_p_idx, read_s_idx)
                else:
                    assigned_key = (orig_p_idx, orig_s_idx)
                system_readings[assigned_key] = reading
            except Exception as exc:
                logger.error("Failed to read system (%d, %d): %s", orig_p_idx, orig_s_idx, exc)
                warnings.append(f"System ({orig_p_idx}, {orig_s_idx}) reading failed: {exc}")
                system_readings[(orig_p_idx, orig_s_idx)] = SystemReading()

    # Re-request any missing systems individually
    for p_idx, geom in enumerate(geoms):
        page_img = None
        for s_idx, sys_geom in enumerate(geom.systems):
            if (p_idx, s_idx) not in system_readings:
                logger.info(
                    "System (p=%d, s=%d) missing from mapped readings; re-requesting individually...",
                    p_idx,
                    s_idx,
                )
                if page_img is None:
                    try:
                        page_img = Image.open(io.BytesIO(images[p_idx])).convert("RGB")
                    except Exception:
                        page_img = None
                if page_img is not None:
                    try:
                        crop_bytes, num_boxes = annotate_system_crop(page_img, sys_geom, p_idx, s_idx)
                        _, solo_res = _read_single_system(
                            (p_idx, s_idx, sys_geom, crop_bytes, num_boxes),
                            header.chord_notation,
                            target_model,
                        )
                        system_readings[(p_idx, s_idx)] = solo_res
                    except Exception as exc:
                        logger.warning("Individual re-read failed for system (%d, %d): %s", p_idx, s_idx, exc)
                        system_readings[(p_idx, s_idx)] = SystemReading()
                else:
                    system_readings[(p_idx, s_idx)] = SystemReading()

    # Consistency check for header.chord_notation:
    num_number_chords = 0
    num_letter_chords = 0
    for reading in system_readings.values():
        for br in reading.chord_boxes:
            txt = br.text.strip()
            if not txt or txt.lower() in {"", "none", "null", "no", "x"}:
                continue
            clean_txt, _ = parse_chord_symbol(txt)
            if re.match(r"^[b#♭♯]?[1-7]", clean_txt):
                num_number_chords += 1
            elif re.match(r"^[A-Ga-g]", clean_txt):
                num_letter_chords += 1

    if num_number_chords > num_letter_chords and header.chord_notation != "number":
        warnings.append(
            f"[v2_notation_override] Overriding header.chord_notation from {header.chord_notation!r} to 'number' "
            f"(found {num_number_chords} number chords vs {num_letter_chords} letter chords)"
        )
        header.chord_notation = "number"
    elif num_letter_chords > num_number_chords and header.chord_notation != "letter":
        warnings.append(
            f"[v2_notation_override] Overriding header.chord_notation from {header.chord_notation!r} to 'letter' "
            f"(found {num_letter_chords} letter chords vs {num_number_chords} number chords)"
        )
        header.chord_notation = "letter"

    # 4. Assemble ParsedSheet with global measure indices and KeyChanges
    all_systems: list[System] = []
    key_changes: list[KeyChange] = []
    sheet_issues: list[QualityIssue] = []
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

            # Pre-group extra chords by measure index (handling 1-based index if model reported it)
            extra_chords_by_m: dict[int, list[tuple[str, float, bool, bool]]] = {}
            for ec in reading.extra_chords:
                m_ec = ec.measure_index
                if m_ec >= len(sys_geom.measures):
                    if 1 <= m_ec <= len(sys_geom.measures):
                        m_ec = m_ec - 1
                    else:
                        m_ec = len(sys_geom.measures) - 1
                if m_ec not in extra_chords_by_m:
                    extra_chords_by_m[m_ec] = []
                extra_chords_by_m[m_ec].append((ec.text, ec.beat, ec.stacked, bool(ec.boxed)))

            # Pre-index measure content readings by measure_index
            measure_content_map = {mc.measure_index: mc for mc in reading.measures}
            is_chord_only = "chord_only" in sys_geom.notes
            if is_chord_only:
                warnings.append(f"[v2_chord_only] page={p_idx} system={s_idx}")
            if "pickup" in sys_geom.notes:
                warnings.append(f"[v2_pickup] page={p_idx} system={s_idx}")

            # Check if reading is still inconsistent with geometry; flag needs_review if so
            still_inconsistent, still_reason = check_reading_inconsistent(reading, sys_geom, header.chord_notation)
            if still_inconsistent and len(sys_geom.chord_boxes) > 0:
                warnings.append(f"[needs_review] System page={p_idx} system={s_idx}: {still_reason}")
                sheet_issues.append(
                    QualityIssue(
                        stage="omr",
                        measure_index=sys_start_measure_idx,
                        severity="needs_review",
                        code="chord_box_count_mismatch",
                        message=f"第{p_idx + 1}页第{s_idx + 1}行和弦框与读取数量不一致（{still_reason}），请核对",
                        detail={"page": p_idx, "system": s_idx, "reason": still_reason},
                    )
                )

            # Map chords to measures using the measure_index_in_system from geometry
            boxes_by_measure: dict[int, list[tuple[int, GChordBox]]] = {
                m_i: [] for m_i in range(len(sys_geom.measures))
            }
            for b_idx, cb in enumerate(sys_geom.chord_boxes, 1):
                if 0 <= cb.measure_index_in_system < len(sys_geom.measures):
                    target_m = cb.measure_index_in_system
                else:
                    cb_cx = (cb.bbox[0] + cb.bbox[2]) / 2.0
                    target_m = find_measure_index_for_x(cb_cx, sys_geom.measures)
                if target_m in boxes_by_measure:
                    boxes_by_measure[target_m].append((b_idx, cb))
                elif boxes_by_measure:
                    boxes_by_measure[0].append((b_idx, cb))

            for m_idx, g_m in enumerate(sys_geom.measures):
                m_x0_x1 = (g_m.x0, g_m.x1)
                m_width = max(1e-4, g_m.x1 - g_m.x0)
                chords_in_measure = []

                # 1. Chords from numbered boxes mapped to this measure
                for b_idx, cb in boxes_by_measure.get(m_idx, []):
                    raw_text = box_readings_map.get(b_idx, "")
                    is_stacked = box_stacked_map.get(b_idx, False)

                    # Resolve merged tokens according to rules
                    tokens, is_joined, needs_review_flag = resolve_chord_tokens(
                        raw_text,
                        box_geom_count=len(boxes_by_measure.get(m_idx, [])),
                        notation=header.chord_notation,
                    )
                    if needs_review_flag:
                        warnings.append(
                            f"[needs_review] System page={p_idx} system={s_idx} measure={m_idx}: ambiguous merged chord reading {raw_text!r}"
                        )
                        sheet_issues.append(
                            QualityIssue(
                                stage="omr",
                                measure_index=global_measure_idx + m_idx,
                                severity="needs_review",
                                code="merged_chord_inconclusive",
                                message=f"第{p_idx + 1}页第{s_idx + 1}行第{m_idx + 1}小节和弦文本 '{raw_text}' 合并/拆分存在二义性，请核对",
                                detail={"page": p_idx, "system": s_idx, "measure": m_idx, "raw": raw_text},
                            )
                        )

                    if len(tokens) > 1:
                        for t_i, tok in enumerate(tokens):
                            cleaned, stacked = parse_chord_symbol(tok, model_stacked=is_stacked)
                            if cleaned and cleaned.lower() not in {"", "none", "null", "no", "x"}:
                                if t_i == 0:
                                    beat = resolve_chord_beat(
                                        beat_geo=cb.beat_geo,
                                        model_beat=box_beats_map.get(b_idx),
                                        time_sig=header.time_signature,
                                        measure_beats=measure_beats,
                                        box_bbox=cb.bbox,
                                        measure_x0_x1=m_x0_x1,
                                    )
                                    sub_bbox = (cb.bbox[0], cb.bbox[1], (cb.bbox[0] + cb.bbox[2]) / 2.0, cb.bbox[3])
                                else:
                                    base_b = resolve_chord_beat(
                                        beat_geo=cb.beat_geo,
                                        model_beat=box_beats_map.get(b_idx),
                                        time_sig=header.time_signature,
                                        measure_beats=measure_beats,
                                        box_bbox=cb.bbox,
                                        measure_x0_x1=m_x0_x1,
                                    )
                                    beat = 3.0 if measure_beats == 4.0 else min(measure_beats, base_b + (measure_beats / len(tokens)))
                                    sub_bbox = ((cb.bbox[0] + cb.bbox[2]) / 2.0, cb.bbox[1], cb.bbox[2], cb.bbox[3])
                                chords_in_measure.append(
                                    ChordSymbol(
                                        raw=cleaned,
                                        beat=beat,
                                        bbox=sub_bbox,
                                        confidence=1.0,
                                        stacked=stacked,
                                    )
                                )
                    elif len(tokens) == 1:
                        cleaned, stacked = parse_chord_symbol(tokens[0], model_stacked=is_stacked)
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
                                    bbox=cb.bbox,
                                    confidence=1.0,
                                    stacked=stacked,
                                )
                            )

                # 2. Extra chords
                is_sys_boxed = len(sys_geom.chord_boxes) > 0
                for extra_text, extra_beat, extra_stacked, extra_boxed in extra_chords_by_m.get(m_idx, []):
                    cleaned_extra, stacked_extra = parse_chord_symbol(extra_text, model_stacked=extra_stacked)
                    if cleaned_extra and cleaned_extra.lower() not in {"", "none", "null", "no", "x"}:
                        # Filter non-chord voicing digits and rests
                        if re.match(r"^(?:0|\d{3,})$", cleaned_extra) or not is_valid_chord_grammar(cleaned_extra, notation=header.chord_notation):
                            continue
                        # On boxed systems, drop an extra if the model marks it as not boxed (e.g. PN voicing hints like '2 2')
                        if is_sys_boxed and not extra_boxed:
                            warnings.append(
                                f"[omr_verify] Dropped unboxed chord {cleaned_extra!r} in page {p_idx} system {s_idx} measure {m_idx}: not inside a printed chord box"
                            )
                            sheet_issues.append(
                                QualityIssue(
                                    stage="omr",
                                    measure_index=global_measure_idx + m_idx,
                                    severity="info",
                                    code="unboxed_chord_dropped",
                                    message=f"第{p_idx + 1}页第{s_idx + 1}行第{m_idx + 1}小节记号 '{cleaned_extra}' 未处于和弦框内，已自动丢弃",
                                    detail={"page": p_idx, "system": s_idx, "measure": m_idx, "raw": cleaned_extra, "beat": extra_beat},
                                )
                            )
                            continue
                        # On boxed systems, unboxed digits in fill measures are drum/band fill notes, not chords
                        if is_sys_boxed:
                            mc_curr = measure_content_map.get(m_idx)
                            if mc_curr and mc_curr.fill and re.match(r"^\d+$", cleaned_extra):
                                continue
                        # Deduplicate if already present at roughly the same beat
                        already_present = any(
                            c.raw == cleaned_extra and abs(c.beat - extra_beat) < 0.5
                            for c in chords_in_measure
                        )
                        if not already_present:
                            final_b = extra_beat if (1.0 <= extra_beat <= measure_beats) else 1.0
                            synth_bbox = None
                            if extra_boxed:
                                x_frac = (final_b - 1.0) / max(1.0, measure_beats)
                                cb_w = 0.05
                                cb_x0 = g_m.x0 + x_frac * (g_m.x1 - g_m.x0)
                                cb_x1 = min(g_m.x1, cb_x0 + cb_w)
                                cb_y0 = sys_geom.bbox[1]
                                cb_y1 = sys_geom.bbox[1] + 0.25 * (sys_geom.bbox[3] - sys_geom.bbox[1])
                                synth_bbox = (cb_x0, cb_y0, cb_x1, cb_y1)
                            chords_in_measure.append(
                                ChordSymbol(
                                    raw=cleaned_extra,
                                    beat=final_b,
                                    bbox=synth_bbox,
                                    confidence=0.9,
                                    stacked=stacked_extra,
                                )
                            )

                chords_in_measure.sort(key=lambda c: c.beat)
                # Deduplicate identical raw chords at the same beat (±0.25) within a measure
                if len(chords_in_measure) > 1:
                    deduped_chords = []
                    for c in chords_in_measure:
                        if not any(dc.raw == c.raw and abs(dc.beat - c.beat) <= 0.25 for dc in deduped_chords):
                            deduped_chords.append(c)
                    chords_in_measure = deduped_chords

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
                is_roadmap = bool(re.search(r"主歌|副歌|桥段|前奏|间奏|尾奏", kc_raw))
                if semitones == 0 or is_roadmap:
                    warnings.append(
                        f"[v2_key_change_filtered] page={p_idx} system={s_idx} raw={kc_raw!r} semitones={semitones} roadmap={is_roadmap}"
                    )
                else:
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
        issues=sheet_issues,
        layout_confidence=min_confidence,
        layout_source="cv",
    )
