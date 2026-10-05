"""Ukulele strumming and fingerpicking pattern library and stroke generator."""
from __future__ import annotations

from typing import Literal, Optional

from app.models import Difficulty, ResolvedChord, SectionRole, UkeStroke, UkeStrokeKind

# Stroke template: (onset, duration, kind, strings, accent)
StrokeTemplate = tuple[float, float, UkeStrokeKind, list[int], bool]


def stroke_to_char(kind: str, strings: list[int]) -> str:
    """Map a single stroke to its character or digit representation."""
    if kind == "pluck":
        if len(strings) > 1:
            return "+".join(str(s) for s in sorted(strings, reverse=True))
        elif strings:
            return str(strings[0])
        return "p"
    elif kind == "down":
        return "↓"
    elif kind == "up":
        return "↑"
    elif kind == "chuck":
        return "×"
    elif kind == "rest":
        return "0"
    return "·"


def derive_legend_from_strokes(strokes: list[StrokeTemplate], full_beats: float) -> str:
    """Generate rhythm legend string algorithmically from stroke templates.

    Rules:
    - Beat-grouped (space separated per beat)
    - Plucking formatted as string digits e.g. '4+1 2 3 1'
    - Eighth slots formatted with '-' for empty e.g. '↓ - ↓↑ - ↑ ↓↑'
    - Sixteenth slots formatted with '·' e.g. '↓·↑ ×↑ ↓↑ ×↑'
    - Compound time (triplets) formatted as e.g. '↓↓↑ ↓↓↑'
    - Single held stroke formatted as '↓'
    - Single stop stroke formatted as '↓ 0'
    - Full rest formatted as '0'
    """
    if not strokes:
        return "0"

    # Check if pure rest
    if len(strokes) == 1 and strokes[0][2] == "rest":
        return "0"

    # Check if single held stroke covering whole bar
    if len(strokes) == 1 and strokes[0][2] == "down" and strokes[0][1] >= full_beats and strokes[0][4]:
        return "↓"

    # Check if stop (single down then rest)
    if len(strokes) == 2 and strokes[0][2] == "down" and strokes[1][2] == "rest":
        return "↓ 0"

    is_picking = any(s[2] == "pluck" for s in strokes)
    if is_picking:
        return " ".join(stroke_to_char(s[2], s[3]) for s in strokes)

    is_compound = any(
        abs((s[0] % 1.0) - 0.3333) < 0.05 or abs((s[0] % 1.0) - 0.6666) < 0.05 for s in strokes
    )
    num_beats = int(round(full_beats))

    if is_compound:
        beat_groups = []
        for b in range(num_beats):
            b_strokes = [s for s in strokes if b <= s[0] < b + 1.0 - 1e-4]
            if not b_strokes:
                beat_groups.append("-")
                continue
            if len(b_strokes) == 1 and abs(b_strokes[0][0] - b) < 1e-4 and b_strokes[0][1] >= 0.9:
                beat_groups.append(f"{stroke_to_char(b_strokes[0][2], b_strokes[0][3])} - -")
                continue
            slots = ["-", "-", "-"]
            for s in b_strokes:
                pos = s[0] - b
                slot_idx = int(round(pos * 3.0))
                if 0 <= slot_idx < 3:
                    slots[slot_idx] = stroke_to_char(s[2], s[3])
            beat_groups.append("".join(slots).rstrip("-"))
        return " ".join(beat_groups)

    has_16th_sub = any(s[1] < 0.4 or abs((s[0] % 0.5) - 0.25) < 0.05 for s in strokes)

    beat_groups = []
    for b in range(num_beats):
        b_strokes = [s for s in strokes if b <= s[0] < b + 1.0 - 1e-4]
        if not b_strokes:
            beat_groups.append("-")
            continue

        if len(b_strokes) == 1 and abs(b_strokes[0][0] - b) < 1e-4:
            s = b_strokes[0]
            ch = stroke_to_char(s[2], s[3])
            if s[1] > 1.0:
                beat_groups.append(ch)
                continue
            elif s[1] == 1.0:
                # If beat 0 in an island-type pattern
                if b == 0 and any(len([x for x in strokes if int(x[0]) == ob]) > 1 for ob in range(num_beats)):
                    beat_groups.append(f"{ch} -" if full_beats >= 4.0 else ch)
                else:
                    beat_groups.append(ch)
                continue

        has_beat_16th = any(
            abs((s[0] - b) - 0.25) < 0.05 or abs((s[0] - b) - 0.75) < 0.05 or s[1] < 0.4 for s in b_strokes
        )
        if has_16th_sub and has_beat_16th:
            s_map = {}
            for s in b_strokes:
                idx = int(round((s[0] - b) * 4.0))
                s_map[idx] = stroke_to_char(s[2], s[3])
            chars = []
            for idx in range(4):
                if idx in s_map:
                    chars.append(s_map[idx])
                elif idx == 2 and not any(k in s_map for k in (1, 2)):
                    chars.append("·")
            beat_groups.append("".join(chars))
        else:
            s0 = next((s for s in b_strokes if abs(s[0] - b) < 0.05), None)
            s1 = next((s for s in b_strokes if abs(s[0] - (b + 0.5)) < 0.05), None)
            c0 = stroke_to_char(s0[2], s0[3]) if s0 else "-"
            c1 = stroke_to_char(s1[2], s1[3]) if s1 else ""
            if c1:
                if c0 == "-":
                    beat_groups.append(f"- {c1}")
                else:
                    beat_groups.append(f"{c0}{c1}")
            else:
                beat_groups.append(f"{c0} -")

    return " ".join(beat_groups)


class PatternSpec:
    """Definition of an accompaniment pattern for a full bar."""

    def __init__(
        self,
        pattern_id: str,
        legend: Optional[str],
        technique: Literal["strum", "pick", "hold", "rest"],
        strokes: list[StrokeTemplate],
        full_beats: float = 4.0,
        description: str = "",
    ):
        self.pattern_id = pattern_id
        self.technique = technique
        self.strokes = strokes
        self.full_beats = full_beats
        self.description = description
        self.legend = derive_legend_from_strokes(strokes, full_beats)


PATTERN_REGISTRY: dict[str, PatternSpec] = {
    # 4/4 Beginner
    "uke_4_4_beg_sparse": PatternSpec(
        "uke_4_4_beg_sparse",
        "↓ - - -",
        "strum",
        [(0.0, 4.0, "down", [], False)],
        4.0,
    ),
    "uke_4_4_beg_half": PatternSpec(
        "uke_4_4_beg_half",
        "↓ - ↓ -",
        "strum",
        [(0.0, 2.0, "down", [], False), (2.0, 2.0, "down", [], False)],
        4.0,
    ),
    "uke_4_4_beg_quarter": PatternSpec(
        "uke_4_4_beg_quarter",
        "↓ ↓ ↓ ↓",
        "strum",
        [
            (0.0, 1.0, "down", [], False),
            (1.0, 1.0, "down", [], False),
            (2.0, 1.0, "down", [], False),
            (3.0, 1.0, "down", [], False),
        ],
        4.0,
    ),
    # 4/4 Intermediate
    "uke_4_4_mid_ballad": PatternSpec(
        "uke_4_4_mid_ballad",
        "↓ - ↓ ↓↑",
        "strum",
        [
            (0.0, 2.0, "down", [], False),
            (2.0, 1.0, "down", [], False),
            (3.0, 0.5, "down", [], False),
            (3.5, 0.5, "up", [], False),
        ],
        4.0,
    ),
    "uke_4_4_mid_island": PatternSpec(
        "uke_4_4_mid_island",
        "↓ - ↓↑ - ↑ ↓↑",
        "strum",
        [
            (0.0, 1.0, "down", [], False),
            (1.0, 0.5, "down", [], False),
            (1.5, 1.0, "up", [], False),
            (2.5, 0.5, "up", [], False),
            (3.0, 0.5, "down", [], False),
            (3.5, 0.5, "up", [], False),
        ],
        4.0,
    ),
    "uke_4_4_mid_chuck": PatternSpec(
        "uke_4_4_mid_chuck",
        "↓ - ×↑ - ↑ ×↑",
        "strum",
        [
            (0.0, 1.0, "down", [], False),
            (1.0, 0.5, "chuck", [], False),
            (1.5, 1.0, "up", [], False),
            (2.5, 0.5, "up", [], False),
            (3.0, 0.5, "chuck", [], False),
            (3.5, 0.5, "up", [], False),
        ],
        4.0,
    ),
    # 4/4 Advanced
    "uke_4_4_adv_pick": PatternSpec(
        "uke_4_4_adv_pick",
        "4+1 2 3 1 4 2 3 1",
        "pick",
        [
            (0.0, 0.5, "pluck", [4, 1], True),
            (0.5, 0.5, "pluck", [2], False),
            (1.0, 0.5, "pluck", [3], False),
            (1.5, 0.5, "pluck", [1], False),
            (2.0, 0.5, "pluck", [4], False),
            (2.5, 0.5, "pluck", [2], False),
            (3.0, 0.5, "pluck", [3], False),
            (3.5, 0.5, "pluck", [1], False),
        ],
        4.0,
    ),
    "uke_4_4_adv_sync": PatternSpec(
        "uke_4_4_adv_sync",
        "↓·↑ ×↑ ↓↑ ×↑",
        "strum",
        [
            (0.0, 0.75, "down", [], True),
            (0.75, 0.25, "up", [], False),
            (1.0, 0.5, "chuck", [], False),
            (1.5, 0.5, "up", [], False),
            (2.0, 0.5, "down", [], False),
            (2.5, 0.5, "up", [], False),
            (3.0, 0.5, "chuck", [], False),
            (3.5, 0.5, "up", [], False),
        ],
        4.0,
    ),
    "uke_4_4_adv_buildup": PatternSpec(
        "uke_4_4_adv_buildup",
        "↓↓ ↓↓ ↓↓ ↓↓",
        "strum",
        [
            (0.0, 0.5, "down", [], False),
            (0.5, 0.5, "down", [], False),
            (1.0, 0.5, "down", [], False),
            (1.5, 0.5, "down", [], False),
            (2.0, 0.5, "down", [], True),
            (2.5, 0.5, "down", [], True),
            (3.0, 0.5, "down", [], True),
            (3.5, 0.5, "down", [], True),
        ],
        4.0,
        description="crescendo",
    ),
    "uke_4_4_adv_fill": PatternSpec(
        "uke_4_4_adv_fill",
        "↓ - ↓↑ ↓↑ ↓↑↓↑",
        "strum",
        [
            (0.0, 1.0, "down", [], False),
            (1.0, 0.5, "down", [], False),
            (1.5, 0.5, "up", [], False),
            (2.0, 0.5, "down", [], True),
            (2.5, 0.5, "up", [], False),
            (3.0, 0.25, "down", [], True),
            (3.25, 0.25, "up", [], False),
            (3.5, 0.25, "down", [], True),
            (3.75, 0.25, "up", [], False),
        ],
        4.0,
        description="fill",
    ),
    # 3/4
    "uke_3_4_beg_hold": PatternSpec(
        "uke_3_4_beg_hold",
        "↓ - -",
        "strum",
        [(0.0, 3.0, "down", [], False)],
        3.0,
    ),
    "uke_3_4_beg_quarter": PatternSpec(
        "uke_3_4_beg_quarter",
        "↓ ↓ ↓",
        "strum",
        [
            (0.0, 1.0, "down", [], True),
            (1.0, 1.0, "down", [], False),
            (2.0, 1.0, "down", [], False),
        ],
        3.0,
    ),
    "uke_3_4_mid_waltz": PatternSpec(
        "uke_3_4_mid_waltz",
        "↓ ↓↑ ↓↑",
        "strum",
        [
            (0.0, 1.0, "down", [], True),
            (1.0, 0.5, "down", [], False),
            (1.5, 0.5, "up", [], False),
            (2.0, 0.5, "down", [], False),
            (2.5, 0.5, "up", [], False),
        ],
        3.0,
    ),
    "uke_3_4_adv_pick": PatternSpec(
        "uke_3_4_adv_pick",
        "3+1 2 1 2 1 2",
        "pick",
        [
            (0.0, 0.5, "pluck", [3, 1], True),
            (0.5, 0.5, "pluck", [2], False),
            (1.0, 0.5, "pluck", [1], False),
            (1.5, 0.5, "pluck", [2], False),
            (2.0, 0.5, "pluck", [1], False),
            (2.5, 0.5, "pluck", [2], False),
        ],
        3.0,
    ),
    "uke_3_4_adv_chuck": PatternSpec(
        "uke_3_4_adv_chuck",
        "↓ ×↑ ×↑",
        "strum",
        [
            (0.0, 1.0, "down", [], True),
            (1.0, 0.5, "chuck", [], False),
            (1.5, 0.5, "up", [], False),
            (2.0, 0.5, "chuck", [], False),
            (2.5, 0.5, "up", [], False),
        ],
        3.0,
    ),
    # 2/4
    "uke_2_4_beg": PatternSpec(
        "uke_2_4_beg",
        "↓ ↓",
        "strum",
        [(0.0, 1.0, "down", [], False), (1.0, 1.0, "down", [], False)],
        2.0,
    ),
    "uke_2_4_mid": PatternSpec(
        "uke_2_4_mid",
        "↓ ↓↑",
        "strum",
        [
            (0.0, 1.0, "down", [], False),
            (1.0, 0.5, "down", [], False),
            (1.5, 0.5, "up", [], False),
        ],
        2.0,
    ),
    "uke_2_4_adv_pick": PatternSpec(
        "uke_2_4_adv_pick",
        "3+1 2 1 2",
        "pick",
        [
            (0.0, 0.5, "pluck", [3, 1], True),
            (0.5, 0.5, "pluck", [2], False),
            (1.0, 0.5, "pluck", [1], False),
            (1.5, 0.5, "pluck", [2], False),
        ],
        2.0,
    ),
    "uke_2_4_adv_chuck": PatternSpec(
        "uke_2_4_adv_chuck",
        "↓↑ ×↑",
        "strum",
        [
            (0.0, 0.5, "down", [], True),
            (0.5, 0.5, "up", [], False),
            (1.0, 0.5, "chuck", [], False),
            (1.5, 0.5, "up", [], False),
        ],
        2.0,
    ),
    # 6/8
    "uke_6_8_beg": PatternSpec(
        "uke_6_8_beg",
        "↓ ↓",
        "strum",
        [(0.0, 1.0, "down", [], False), (1.0, 1.0, "down", [], False)],
        2.0,
    ),
    "uke_6_8_mid": PatternSpec(
        "uke_6_8_mid",
        "↓↓↑ ↓↓↑",
        "strum",
        [
            (0.0, 0.3333, "down", [], True),
            (0.3333, 0.3333, "down", [], False),
            (0.6666, 0.3334, "up", [], False),
            (1.0, 0.3333, "down", [], True),
            (1.3333, 0.3333, "down", [], False),
            (1.6666, 0.3334, "up", [], False),
        ],
        2.0,
    ),
    "uke_6_8_adv_pick": PatternSpec(
        "uke_6_8_adv_pick",
        "4+1 2 1 4 2 1",
        "pick",
        [
            (0.0, 0.3333, "pluck", [4, 1], True),
            (0.3333, 0.3333, "pluck", [2], False),
            (0.6666, 0.3334, "pluck", [1], False),
            (1.0, 0.3333, "pluck", [4], True),
            (1.3333, 0.3333, "pluck", [2], False),
            (1.6666, 0.3334, "pluck", [1], False),
        ],
        2.0,
    ),
    # 12/8
    "uke_12_8_beg": PatternSpec(
        "uke_12_8_beg",
        "↓ ↓ ↓ ↓",
        "strum",
        [
            (0.0, 1.0, "down", [], False),
            (1.0, 1.0, "down", [], False),
            (2.0, 1.0, "down", [], False),
            (3.0, 1.0, "down", [], False),
        ],
        4.0,
    ),
    "uke_12_8_mid": PatternSpec(
        "uke_12_8_mid",
        "↓↓↑ ↓↓↑ ↓↓↑ ↓↓↑",
        "strum",
        [
            (0.0, 0.3333, "down", [], True),
            (0.3333, 0.3333, "down", [], False),
            (0.6666, 0.3334, "up", [], False),
            (1.0, 0.3333, "down", [], True),
            (1.3333, 0.3333, "down", [], False),
            (1.6666, 0.3334, "up", [], False),
            (2.0, 0.3333, "down", [], True),
            (2.3333, 0.3333, "down", [], False),
            (2.6666, 0.3334, "up", [], False),
            (3.0, 0.3333, "down", [], True),
            (3.3333, 0.3333, "down", [], False),
            (3.6666, 0.3334, "up", [], False),
        ],
        4.0,
    ),
    "uke_12_8_adv_pick": PatternSpec(
        "uke_12_8_adv_pick",
        "4+1 2 1 4 2 1 4 2 1 4 2 1",
        "pick",
        [
            (0.0, 0.3333, "pluck", [4, 1], True),
            (0.3333, 0.3333, "pluck", [2], False),
            (0.6666, 0.3334, "pluck", [1], False),
            (1.0, 0.3333, "pluck", [4], True),
            (1.3333, 0.3333, "pluck", [2], False),
            (1.6666, 0.3334, "pluck", [1], False),
            (2.0, 0.3333, "pluck", [4], True),
            (2.3333, 0.3333, "pluck", [2], False),
            (2.6666, 0.3334, "pluck", [1], False),
            (3.0, 0.3333, "pluck", [4], True),
            (3.3333, 0.3333, "pluck", [2], False),
            (3.6666, 0.3334, "pluck", [1], False),
        ],
        4.0,
    ),
    # Specials
    "uke_stop": PatternSpec(
        "uke_stop",
        "↓ 0",
        "hold",
        [(0.0, 1.0, "down", [], True), (1.0, 3.0, "rest", [], False)],
        4.0,
        description="hit",
    ),
    "uke_held": PatternSpec(
        "uke_held",
        "↓",
        "hold",
        [(0.0, 4.0, "down", [], True)],
        4.0,
        description="held",
    ),
    "uke_rest": PatternSpec(
        "uke_rest",
        "0",
        "rest",
        [(0.0, 4.0, "rest", [], False)],
        4.0,
    ),
}


def get_time_signature_family(time_signature: str, style: str = "") -> str:
    """Normalize time signature string and optional style to standard family."""
    ts = (time_signature or "").strip()
    st = (style or "").strip().lower()
    if "slow rock" in st:
        return "12/8"
    if "3/4" in ts:
        return "3/4"
    if "6/8" in ts:
        return "6/8"
    if "12/8" in ts:
        return "12/8"
    if "2/4" in ts:
        return "2/4"
    return "4/4"


def choose_section_pattern(
    ts_family: str,
    difficulty: Difficulty,
    role: SectionRole,
    energy: int,
    tempo_bpm: Optional[float] = None,
) -> str:
    """Select appropriate base pattern_id for a section instance."""
    is_fast = (tempo_bpm is not None and tempo_bpm > 140)

    if ts_family == "3/4":
        if difficulty == "beginner":
            return "uke_3_4_beg_quarter" if energy >= 2 else "uke_3_4_beg_hold"
        elif difficulty == "intermediate":
            return "uke_3_4_mid_waltz"
        else:  # advanced
            if role in ("intro", "verse", "interlude") and energy <= 1 and not is_fast:
                return "uke_3_4_adv_pick"
            elif energy >= 2:
                return "uke_3_4_adv_chuck"
            return "uke_3_4_mid_waltz"

    if ts_family == "2/4":
        if difficulty == "beginner":
            return "uke_2_4_beg"
        elif difficulty == "intermediate":
            return "uke_2_4_mid"
        else:  # advanced
            if role in ("intro", "verse") and energy <= 1 and not is_fast:
                return "uke_2_4_adv_pick"
            return "uke_2_4_adv_chuck"

    if ts_family == "6/8":
        if difficulty == "beginner":
            return "uke_6_8_beg"
        elif difficulty == "intermediate":
            return "uke_6_8_mid"
        else:
            return "uke_6_8_adv_pick" if energy <= 1 and not is_fast else "uke_6_8_mid"

    if ts_family == "12/8":
        if difficulty == "beginner":
            return "uke_12_8_beg"
        elif difficulty == "intermediate":
            return "uke_12_8_mid"
        else:
            return "uke_12_8_adv_pick" if energy <= 1 and not is_fast else "uke_12_8_mid"

    # Default: 4/4
    if difficulty == "beginner":
        if energy <= 0:
            return "uke_4_4_beg_sparse"
        elif energy == 1:
            return "uke_4_4_beg_half"
        else:
            return "uke_4_4_beg_quarter"

    elif difficulty == "intermediate":
        if role in ("intro", "outro") or energy <= 0:
            return "uke_4_4_mid_ballad"
        elif role == "chorus" or energy >= 2:
            return "uke_4_4_mid_chuck"
        else:
            return "uke_4_4_mid_island"

    else:  # advanced
        if (role in ("intro", "verse", "interlude") or energy <= 1) and not is_fast:
            return "uke_4_4_adv_pick"
        elif role == "chorus" or energy >= 2:
            return "uke_4_4_adv_sync"
        else:
            return "uke_4_4_mid_island"


def generate_measure_strokes(
    pattern_id: str,
    chords: list[ResolvedChord],
    measure_beats: float,
    *,
    is_stop: bool = False,
    fill: bool = False,
    is_last_measure: bool = False,
    is_section_end: bool = False,
    is_section_start: bool = False,
    next_role: Optional[SectionRole] = None,
    difficulty: Difficulty = "intermediate",
    tempo_bpm: Optional[float] = None,
) -> tuple[str, Literal["strum", "pick", "hold", "rest"], list[UkeStroke]]:
    """Generate a valid, non-overlapping, tiling list of UkeStrokes for one measure."""
    # 1. No chords in measure -> rest
    if not chords:
        s = UkeStroke(onset=0.0, duration=measure_beats, kind="rest", strings=[], accent=False)
        return "uke_rest", "rest", [s]

    # 2. Last measure of the song -> single accented down held
    if is_last_measure:
        s = UkeStroke(onset=0.0, duration=measure_beats, kind="down", strings=[], accent=True)
        return "uke_held", "hold", [s]

    # 3. Stop measure -> hit on beat 1, then rest
    if is_stop:
        if measure_beats > 1.0:
            s1 = UkeStroke(onset=0.0, duration=1.0, kind="down", strings=[], accent=True)
            s2 = UkeStroke(onset=1.0, duration=round(measure_beats - 1.0, 4), kind="rest", strings=[], accent=False)
            return "uke_stop", "hold", [s1, s2]
        else:
            s = UkeStroke(onset=0.0, duration=measure_beats, kind="down", strings=[], accent=True)
            return "uke_stop", "hold", [s]

    # 4. Fill in advanced -> roll/burst variant
    actual_pid = pattern_id
    if fill and difficulty == "advanced" and measure_beats == 4.0:
        actual_pid = "uke_4_4_adv_fill"
    elif is_section_end and difficulty == "advanced" and next_role == "chorus" and measure_beats == 4.0:
        actual_pid = "uke_4_4_adv_buildup"

    spec = PATTERN_REGISTRY.get(actual_pid, PATTERN_REGISTRY["uke_4_4_mid_island"])
    technique = spec.technique

    # Calculate chord change onsets (0-based)
    chord_onsets: list[float] = []
    for c in chords:
        onset = max(0.0, min(measure_beats, round(c.beat - 1.0, 4)))
        chord_onsets.append(onset)

    # Sort and deduplicate onsets
    unique_onsets = sorted(list(set(chord_onsets)))
    if not unique_onsets or unique_onsets[0] > 0.0:
        unique_onsets.insert(0, 0.0)

    # Single chord at onset 0.0 covering full measure matching spec beats exactly
    if len(chords) == 1 and unique_onsets == [0.0] and abs(measure_beats - spec.full_beats) < 1e-4:
        strokes: list[UkeStroke] = []
        for onset, dur, kind, strings, acc in spec.strokes:
            strokes.append(UkeStroke(onset=onset, duration=dur, kind=kind, strings=strings, accent=acc))
        return actual_pid, technique, strokes

    # Multi-chord or non-standard measure duration: slice and tile by chord intervals
    intervals: list[tuple[float, float]] = []
    for i in range(len(unique_onsets)):
        start = unique_onsets[i]
        end = unique_onsets[i + 1] if i + 1 < len(unique_onsets) else measure_beats
        if end > start:
            intervals.append((start, end))

    strokes: list[UkeStroke] = []
    is_fast = (tempo_bpm is not None and tempo_bpm > 140)

    for start, end in intervals:
        span = round(end - start, 4)
        if span <= 0.0:
            continue

        if technique == "pick":
            # Fingerpicking sub-strokes in increments of 0.5
            curr = start
            stroke_idx = 0
            pick_strings = [[4, 1], [2], [3], [1], [4], [2], [3], [1]]
            while curr < end:
                dur = min(0.5, round(end - curr, 4))
                strs = pick_strings[stroke_idx % len(pick_strings)]
                acc = (stroke_idx == 0)
                strokes.append(UkeStroke(
                    onset=curr,
                    duration=dur,
                    kind="pluck",
                    strings=strs,
                    accent=acc,
                ))
                curr = round(curr + dur, 4)
                stroke_idx += 1

        elif difficulty == "beginner":
            # Beginner: <= 4 strokes per bar, no chucks, simple down-strokes
            if span >= 2.0:
                if span == 2.0 or is_fast:
                    strokes.append(UkeStroke(onset=start, duration=span, kind="down", strings=[], accent=True))
                else:
                    half = round(span / 2.0, 4)
                    strokes.append(UkeStroke(onset=start, duration=half, kind="down", strings=[], accent=True))
                    strokes.append(UkeStroke(onset=round(start + half, 4), duration=round(span - half, 4), kind="down", strings=[], accent=False))
            else:
                strokes.append(UkeStroke(onset=start, duration=span, kind="down", strings=[], accent=(start == 0.0)))

        else:
            # Intermediate / Advanced strum
            if span >= 2.0:
                # 2-beat strum block
                s1_dur = 1.0
                s2_dur = 0.5
                s3_dur = round(span - 1.5, 4)
                has_chuck = (spec.pattern_id in ("uke_4_4_mid_chuck", "uke_4_4_adv_sync"))
                second_kind: UkeStrokeKind = "chuck" if has_chuck else "down"

                strokes.append(UkeStroke(onset=start, duration=s1_dur, kind="down", strings=[], accent=True))
                strokes.append(UkeStroke(onset=round(start + 1.0, 4), duration=s2_dur, kind=second_kind, strings=[], accent=False))
                strokes.append(UkeStroke(onset=round(start + 1.5, 4), duration=s3_dur, kind="up", strings=[], accent=False))
            elif span >= 1.0:
                half = round(span / 2.0, 4)
                strokes.append(UkeStroke(onset=start, duration=half, kind="down", strings=[], accent=True))
                strokes.append(UkeStroke(onset=round(start + half, 4), duration=round(span - half, 4), kind="up", strings=[], accent=False))
            else:
                strokes.append(UkeStroke(onset=start, duration=span, kind="down", strings=[], accent=True))

    # Tiling validation & normalization
    # Ensure continuous non-overlapping tiling from 0.0 to measure_beats
    if not strokes:
        strokes = [UkeStroke(onset=0.0, duration=measure_beats, kind="down", strings=[], accent=True)]

    strokes.sort(key=lambda s: s.onset)

    # Adjust last stroke duration to guarantee sum == measure_beats
    current_time = 0.0
    for s in strokes:
        s.onset = round(current_time, 4)
        current_time = round(current_time + s.duration, 4)

    if strokes:
        strokes[-1].duration = round(measure_beats - strokes[-1].onset, 4)

    # Enforce accent rule: only the first stroke of a section and stop/hit strokes should be accented,
    # not every chord-change stroke.
    if is_stop:
        for s in strokes:
            if s.kind == "down":
                s.accent = True
    elif is_last_measure:
        for s in strokes:
            s.accent = True
    else:
        for idx, s in enumerate(strokes):
            s.accent = bool(is_section_start and idx == 0 and s.kind != "rest")

    return actual_pid, technique, strokes
