"""Per-chart orientation inference for stacked diagonal fraction chords.

In Chinese mainland numbered notation band sheets (e.g. 听海, 小白船), chords are often
printed as diagonal/stacked fractions (ChordSymbol.stacked=True, raw 'TOP/BOTTOM' in printed order).
Which part is the bass is publisher-dependent:
- 'top_is_bass': TOP is bass degree, BOTTOM is chord harmony (e.g. ⑦/⑤ -> chord 5 over bass 7).
- 'bottom_is_bass': TOP is chord harmony, BOTTOM is bass degree (standard mathematical fraction).
"""
from __future__ import annotations

import re
from typing import Literal, NamedTuple, Optional, Sequence

from app.models import ChordSymbol, Measure, ParsedSheet
from app.theory.chords import clean_raw_chord, parse_accidental_and_degree

from dataclasses import dataclass, field

OrientationType = Literal["top_is_bass", "bottom_is_bass"]

# Slash extension pattern: /9, /11, /13, /b9, /#9, /#11, /9-5, etc.
_EXT_PATTERN = re.compile(r"/((?:b9|#9|#11|9|11|13)[-+b#0-9]*)")


@dataclass(frozen=True)
class StackedOrientationResult:
    """Result of stacked chord orientation inference."""

    orientation: OrientationType
    margin: float
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.orientation


def _token_has_chord_quality(token: str) -> bool:
    """Return True if token carries a chord quality beyond a bare degree."""
    clean = clean_raw_chord(token)
    if not clean:
        return False
    try:
        deg, acc, rem = parse_accidental_and_degree(clean)
        # Any remainder other than empty or plain 'maj'/'Maj' is a chord quality
        return bool(rem and rem.strip() not in ("", "maj", "Maj"))
    except ValueError:
        return False


def _is_slash_extension(raw: str) -> bool:
    """Return True if slash in raw is an extension (e.g. 57/9, 57/11, 17/9)."""
    return bool(_EXT_PATTERN.search(raw))


def _extract_bass_degree(chord_raw: str, orientation: OrientationType) -> Optional[int]:
    """Extract nominal bass scale degree 1..7 for a chord under given orientation."""
    cleaned = clean_raw_chord(chord_raw)
    if not cleaned:
        return None

    # Slash extensions do not create slash bass
    if _is_slash_extension(cleaned):
        cleaned = _EXT_PATTERN.sub(r"(\1)", cleaned)

    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) != 2:
            return None
        top_str, bot_str = parts[0], parts[1]

        # Quality asymmetry check: a part with quality must be the chord
        bot_has_q = _token_has_chord_quality(bot_str)
        top_has_q = _token_has_chord_quality(top_str)
        if bot_has_q and not top_has_q:
            bass_token = top_str
        elif top_has_q and not bot_has_q:
            bass_token = bot_str
        else:
            bass_token = top_str if orientation == "top_is_bass" else bot_str

        # Strip trailing m
        if bass_token.endswith("m") and len(bass_token) > 1:
            bass_token = bass_token[:-1]

        try:
            b_deg, _, _ = parse_accidental_and_degree(bass_token)
            return b_deg
        except ValueError:
            return None

    try:
        deg, _, _ = parse_accidental_and_degree(cleaned)
        return deg
    except ValueError:
        return None


def _extract_first_melody_degree(melody_str: str) -> Optional[int]:
    """Extract the first Jianpu scale degree (1..7) from a melody string."""
    if not melody_str:
        return None
    m = re.search(r"([1-7])", melody_str)
    return int(m.group(1)) if m else None


def _get_chord_degree_tones(degree: int, quality_rem: str) -> set[int]:
    """Return diatonic scale degrees (1..7) belonging to the chord."""
    # Basic triad degrees relative to root degree (1..7)
    # 1-based diatonic steps: root = degree
    third = ((degree - 1 + 2) % 7) + 1
    fifth = ((degree - 1 + 4) % 7) + 1
    tones = {degree, third, fifth}
    if "7" in quality_rem:
        seventh = ((degree - 1 + 6) % 7) + 1
        tones.add(seventh)
    if "9" in quality_rem:
        ninth = ((degree - 1 + 1) % 7) + 1
        tones.add(ninth)
    if "11" in quality_rem or "sus4" in quality_rem or "sus" in quality_rem:
        fourth = ((degree - 1 + 3) % 7) + 1
        tones.add(fourth)
    return tones


def _score_bass_sequence(bass_degrees: list[int]) -> float:
    """Score stepwise diatonic bass motion over a sequence of bass degrees."""
    if len(bass_degrees) < 2:
        return 0.0

    score = 0.0
    descending_streak = 0

    for i in range(len(bass_degrees) - 1):
        prev_b = bass_degrees[i]
        curr_b = bass_degrees[i + 1]

        # Diatonic step down: (prev_b - curr_b) % 7 == 1 (e.g. 1->7, 7->6, 6->5, 5->4, 4->3, 3->2, 2->1)
        down_step = (prev_b - curr_b) % 7 == 1
        # Diatonic step up: (curr_b - prev_b) % 7 == 1
        up_step = (curr_b - prev_b) % 7 == 1

        if down_step:
            score += 2.0
            descending_streak += 1
            if descending_streak >= 2:
                # Bonus for multi-step descending line (1-7-6-5, 6-5-4, etc.)
                score += 1.5
        elif up_step:
            score += 1.5
            descending_streak = 0
        elif prev_b == curr_b:
            # Sustained bass pedal
            score += 0.5
            descending_streak = 0
        else:
            # Disjunct leap penalty
            score -= 1.0
            descending_streak = 0

    return score


def infer_stacked_orientation(sheet: ParsedSheet) -> StackedOrientationResult:
    """Infer per-chart stacked chord orientation ('top_is_bass' vs 'bottom_is_bass').

    Evaluation steps:
      (a) Quality asymmetry: If one part has a quality and the other is a bare degree,
          the part with quality is the chord and the bare part is the bass.
      (b) Chart scoring: Score both orientations across stepwise bass motion
          (1-7-6-5, 6-5-4 lines) and strong-beat melody note fit.
      (c) Default when tied: 'bottom_is_bass'.
    """
    notes: list[str] = []
    measures = sheet.measures() if sheet is not None else []

    # Collect all candidate chords
    all_chords: list[ChordSymbol] = []
    for m in measures:
        all_chords.extend(m.chords)

    # (a) Unambiguous quality asymmetry evidence
    unambiguous_top_is_bass = 0
    unambiguous_bottom_is_bass = 0

    for c in all_chords:
        raw = clean_raw_chord(c.raw)
        if _is_slash_extension(raw):
            continue
        if "/" in raw:
            parts = raw.split("/")
            if len(parts) == 2:
                top_part, bot_part = parts[0], parts[1]
                top_has_q = _token_has_chord_quality(top_part)
                bot_has_q = _token_has_chord_quality(bot_part)

                if bot_has_q and not top_has_q:
                    unambiguous_top_is_bass += 1
                    notes.append(
                        f"Unambiguous evidence in {c.raw!r}: bottom '{bot_part}' carries quality, "
                        f"top '{top_part}' is bare degree -> top_is_bass"
                    )
                elif top_has_q and not bot_has_q:
                    unambiguous_bottom_is_bass += 1
                    notes.append(
                        f"Unambiguous evidence in {c.raw!r}: top '{top_part}' carries quality, "
                        f"bottom '{bot_part}' is bare degree -> bottom_is_bass"
                    )

    # (b) Chart scoring
    bass_top: list[int] = []
    bass_bot: list[int] = []
    melody_score_top = 0.0
    melody_score_bot = 0.0

    for m in measures:
        mel_deg = _extract_first_melody_degree(m.melody) if m.melody else None
        for c in m.chords:
            b_top = _extract_bass_degree(c.raw, "top_is_bass")
            b_bot = _extract_bass_degree(c.raw, "bottom_is_bass")
            if b_top is not None:
                bass_top.append(b_top)
            if b_bot is not None:
                bass_bot.append(b_bot)

            # Melody fit on strong beats if melody present
            if mel_deg is not None and "/" in c.raw and not _is_slash_extension(c.raw):
                parts = clean_raw_chord(c.raw).split("/")
                if len(parts) == 2:
                    top_part, bot_part = parts[0], parts[1]
                    try:
                        # under top_is_bass: chord is bot_part
                        deg_t, _, rem_t = parse_accidental_and_degree(bot_part)
                        tones_t = _get_chord_degree_tones(deg_t, rem_t)
                        if mel_deg in tones_t:
                            melody_score_top += 3.0
                        else:
                            melody_score_top -= 1.0
                    except ValueError:
                        pass

                    try:
                        # under bottom_is_bass: chord is top_part
                        deg_b, _, rem_b = parse_accidental_and_degree(top_part)
                        tones_b = _get_chord_degree_tones(deg_b, rem_b)
                        if mel_deg in tones_b:
                            melody_score_bot += 3.0
                        else:
                            melody_score_bot -= 1.0
                    except ValueError:
                        pass

    bass_score_top = _score_bass_sequence(bass_top)
    bass_score_bot = _score_bass_sequence(bass_bot)

    total_score_top = bass_score_top + melody_score_top
    total_score_bot = bass_score_bot + melody_score_bot
    chart_margin = round(total_score_top - total_score_bot, 2)

    # Resolution logic
    if unambiguous_top_is_bass > 0 and unambiguous_bottom_is_bass == 0:
        orientation: OrientationType = "top_is_bass"
        margin = max(abs(chart_margin), float(unambiguous_top_is_bass) * 10.0)
        notes.append(
            f"Resolved as top_is_bass via quality evidence ({unambiguous_top_is_bass} unambiguous instances; "
            f"bass motion margin: {chart_margin:+.1f})"
        )
    elif unambiguous_bottom_is_bass > 0 and unambiguous_top_is_bass == 0:
        orientation = "bottom_is_bass"
        margin = max(abs(chart_margin), float(unambiguous_bottom_is_bass) * 10.0)
        notes.append(
            f"Resolved as bottom_is_bass via quality evidence ({unambiguous_bottom_is_bass} unambiguous instances; "
            f"bass motion margin: {chart_margin:+.1f})"
        )
    else:
        # Fall back to scoring
        if chart_margin > 0.0:
            orientation = "top_is_bass"
            margin = chart_margin
            notes.append(f"Resolved as top_is_bass via chart scoring (margin: +{margin:.1f})")
        elif chart_margin < 0.0:
            orientation = "bottom_is_bass"
            margin = abs(chart_margin)
            notes.append(f"Resolved as bottom_is_bass via chart scoring (margin: -{margin:.1f})")
        else:
            # (c) Default when tied
            orientation = "bottom_is_bass"
            margin = 0.0
            notes.append("Tied chart score; defaulted to bottom_is_bass per specification")

    return StackedOrientationResult(orientation=orientation, margin=margin, notes=notes)
