"""Parsing and representation of jianpu (numbered musical notation) melodies.

Parses OMR jianpu strings (digits 0-7, accidentals, octave dots, extension dashes '-',
dotted notes '.', underline beat groupings) into (onset_beat, degree, accidental) tuples
and estimates total beat-sum. Never raises exceptions on unparseable inputs.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)


class ParsedMelody(NamedTuple):
    """Result of melody parsing."""

    notes: list[tuple[float, int, int]]  # (onset_beat, degree, accidental)
    beat_sum: float


def _parse_token_notes(tok: str) -> list[tuple[int, int, bool]]:
    """Parse a single whitespace-delimited token into sub-note tokens.

    Returns a list of (degree, accidental, is_dotted):
      - degree: 1..7 for scale degrees, 0 for rest, -1 for extension dash '-'
      - accidental: -1 for flat (b), 0 for natural, +1 for sharp (#)
      - is_dotted: True if followed by a duration dot '.'
    """
    if not tok:
        return []

    # Clean octave marks like ', `, "
    cleaned = tok.replace("'", "").replace("`", "").replace('"', "")
    notes: list[tuple[int, int, bool]] = []
    i = 0
    n = len(cleaned)

    while i < n:
        ch = cleaned[i]
        acc = 0

        # Check leading accidental: e.g. 'b7', '#4'
        if ch in ("#", "b", "♯", "♭") and i + 1 < n and cleaned[i + 1] in "01234567iI":
            acc = 1 if ch in ("#", "♯") else -1
            i += 1
            ch = cleaned[i]
        elif ch == "." and i + 1 < n and cleaned[i + 1] in "01234567iI":
            # Low octave dot like '.7'
            i += 1
            ch = cleaned[i]

        deg: Optional[int] = None
        if ch in ("i", "I"):
            deg = 1
        elif ch in "01234567":
            deg = int(ch)
        elif ch in ("-", "–", "—", "﹣", "−"):
            deg = -1

        if deg is not None:
            # Check trailing accidental: e.g. '7b', '1#'
            if i + 1 < n and cleaned[i + 1] in ("#", "b", "♯", "♭"):
                acc = 1 if cleaned[i + 1] in ("#", "♯") else -1
                i += 1

            # Check trailing duration dot: e.g. '1.'
            is_dot = False
            if i + 1 < n and cleaned[i + 1] == ".":
                is_dot = True
                i += 1

            notes.append((deg, acc, is_dot))
        i += 1

    return notes


def parse_melody(
    melody: Optional[str],
    beats: float = 4.0,
    *,
    include_rests: bool = False,
) -> Optional[ParsedMelody]:
    """Parse a jianpu melody string into (onset_beat, degree, accidental) tuples and beat-sum.

    Args:
        melody: Raw jianpu string from OMR (e.g. '2 2 23 21', '1. 5 55 65', '2 - 0 0').
        beats: Expected beats in the measure (default 4.0).
        include_rests: If True, rests (degree 0) are included in the notes list.

    Returns:
        ParsedMelody(notes=[(onset, deg, acc), ...], beat_sum=float) or None if completely unparseable.
    """
    if melody is None:
        return ParsedMelody(notes=[], beat_sum=0.0)

    try:
        s = str(melody).strip()
        if not s:
            return ParsedMelody(notes=[], beat_sum=0.0)

        # Check for unparseable garbage with no digits, i, or dashes
        if not re.search(r"[0-7iI\-]", s):
            return None

        tokens = s.split()
        if not tokens:
            return ParsedMelody(notes=[], beat_sum=0.0)

        parsed_tokens: list[list[tuple[int, int, bool]]] = []
        for t in tokens:
            p = _parse_token_notes(t)
            if p:
                parsed_tokens.append(p)

        if not parsed_tokens:
            return None

        N = len(parsed_tokens)
        token_durs: list[float] = [1.0] * N

        if N == int(beats) and beats == 4.0:
            # Exactly 4 tokens in a 4-beat bar:
            # Handle dotted note followed by single note: e.g. ['1.', '5'] -> [1.5, 0.5]
            for i in range(N - 1):
                tok_a = parsed_tokens[i]
                tok_b = parsed_tokens[i + 1]
                if len(tok_a) == 1 and tok_a[0][2] and len(tok_b) == 1 and not tok_b[0][2]:
                    token_durs[i] = 1.5
                    token_durs[i + 1] = 0.5
        elif N > beats and beats == 4.0:
            # Measure has more tokens than expected beats (e.g. 5 or 6 tokens).
            # Frequently, single eighth notes were split by a space in lyrics alignment.
            # Assign 0.5 to single notes when N > 4 to bring total to 4.0 if possible.
            surplus = N - int(beats)
            # Find candidate tokens that are single notes without dots
            candidates = [
                i for i, pt in enumerate(parsed_tokens)
                if len(pt) == 1 and not pt[0][2] and pt[0][0] != -1
            ]
            # Pair them into 0.5 beats each
            reductions = 0
            for idx in candidates:
                if reductions < surplus * 2:
                    token_durs[idx] = 0.5
                    reductions += 1
        elif N < beats:
            # Fewer tokens than expected (e.g. 2 tokens in a 4-beat measure).
            # Each token defaults to 1.0 beat (unless dotted or extended),
            # allowing beat_sum < beats to expose missing melody beats!
            for i, pt in enumerate(parsed_tokens):
                if len(pt) == 1 and pt[0][2]:
                    token_durs[i] = 1.5

        # Distribute token durations to individual notes
        notes: list[tuple[float, int, int]] = []
        cur_beat = 1.0  # 1-based beat inside measure

        for pnotes, tok_dur in zip(parsed_tokens, token_durs):
            n_sub = len(pnotes)
            if n_sub == 1:
                deg, acc, _ = pnotes[0]
                if deg > 0 or (include_rests and deg == 0):
                    notes.append((round(cur_beat, 3), deg, acc))
            elif n_sub > 1:
                # Multi-note group (e.g. '23' = two eighths, '3.i' = dotted eighth + sixteenth)
                has_dot = any(n[2] for n in pnotes)
                if has_dot and n_sub == 2:
                    if pnotes[0][2]:
                        sub_durs = [tok_dur * 0.75, tok_dur * 0.25]
                    else:
                        sub_durs = [tok_dur * 0.25, tok_dur * 0.75]
                else:
                    sub_durs = [tok_dur / n_sub] * n_sub

                sub_beat = cur_beat
                for (deg, acc, _), sdur in zip(pnotes, sub_durs):
                    if deg > 0 or (include_rests and deg == 0):
                        notes.append((round(sub_beat, 3), deg, acc))
                    sub_beat += sdur

            cur_beat += tok_dur

        beat_sum = round(sum(token_durs), 3)
        return ParsedMelody(notes=notes, beat_sum=beat_sum)

    except Exception as exc:
        logger.debug("Failed to parse melody string %r: %s", melody, exc)
        return None
