"""Music theory and genre priors for Taiwanese band-chart chord verification.

Provides scoring functions:
1. melody_fit: fraction of strong-beat/long melody notes that are chord tones.
2. bass_hint_fit: verifies if first printed bass note matches chord bass pitch class.
3. progression_plausibility: transition plausibility score for pop/ballad progressions.
4. generate_candidates: generates plausible chord readings for common OCR confusions.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.theory.chords import parse_chord, resolve_chord

logger = logging.getLogger(__name__)

# Scale degree to semitone offset in C (C=0)
DEGREE_TO_PC: dict[int, int] = {
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 7,
    6: 9,
    7: 11,
}

# Canonical root and quality patterns for progression scoring
COMMON_TRANSITIONS: dict[tuple[str, str], float] = {
    # V -> I and resolutions
    ("5", "1"): 1.0,
    ("57", "1"): 1.0,
    ("5sus", "1"): 0.9,
    ("57sus", "1"): 0.9,
    ("5/7", "1"): 0.85,
    ("5/7", "6m7"): 0.95,
    ("5/7", "6m"): 0.95,
    ("5/7", "5m/7b"): 1.0,
    # Chromatic descending line: 1 -> 5/7 -> 5m/7b -> 5m6/2 -> 67/1# -> 2m7
    ("1(2)", "5/7"): 1.0,
    ("1", "5/7"): 1.0,
    ("5m/7b", "5m6/2"): 1.0,
    ("5m/7b", "67/1#"): 0.95,
    ("5m/7b", "67"): 0.95,
    ("5m6/2", "67/1#"): 1.0,
    ("67/1#", "2m7"): 1.0,
    ("67/1#", "2m"): 1.0,
    ("67", "2m7"): 0.95,
    ("67", "2m"): 0.95,
    # ii -> V cadences
    ("2m7", "5"): 1.0,
    ("2m7", "5sus"): 1.0,
    ("2m7", "57sus"): 1.0,
    ("2m7", "57"): 1.0,
    ("2m", "5"): 1.0,
    ("2m7/5", "5"): 1.0,
    ("2m7/6", "5sus"): 0.95,
    ("2m7/6", "5"): 0.9,
    ("2m7", "1(2)"): 0.8,
    ("2m7", "1"): 0.8,
    # IV -> V and IV -> I
    ("4", "5"): 1.0,
    ("4M7", "5"): 1.0,
    ("4", "5sus"): 0.95,
    ("4M7", "57sus"): 0.95,
    ("4", "1/3"): 0.95,
    ("4M7", "1/3"): 0.95,
    ("4", "1"): 0.9,
    ("4M7", "1"): 0.9,
    # Ballad descending chords: 1 -> 1/3 -> 6m7 -> 6m7/5 -> 4 -> 1/3 -> 2m7 -> 5
    ("1", "1/3"): 0.95,
    ("1(2)", "1/3"): 0.9,
    ("1/3", "6m7"): 0.95,
    ("1/3", "2m7"): 0.95,
    ("1/3", "4"): 0.9,
    ("6m7", "6m7/5"): 1.0,
    ("6m", "6m/5"): 1.0,
    ("6m7/5", "4"): 0.95,
    ("6m7/5", "2m7"): 0.9,
    # Borrowed bVI and bVII chords
    ("6b", "6m7-5"): 1.0,
    ("6m7-5", "5sus"): 1.0,
    ("6m7-5", "5"): 1.0,
    ("7b", "4"): 0.95,
    ("7b", "4M7"): 0.95,
    ("7b", "1"): 0.9,
    ("17/7b", "4M7"): 1.0,
    ("17/7b", "4"): 1.0,
    ("1", "17/7b"): 0.95,
    ("1", "3m/7"): 0.95,
    ("3m/7", "17/7b"): 0.95,
    # Suspensions resolving to triad
    ("5sus", "5"): 1.0,
    ("57sus", "57"): 1.0,
    ("57sus", "5"): 0.95,
}


def _simplify_chord(raw: str) -> str:
    """Normalize chord symbol for table lookup."""
    try:
        spec = parse_chord(raw)
        root = f"{spec.degree}"
        if spec.accidental == -1:
            root += "b"
        elif spec.accidental == 1:
            root += "#"

        q = spec.quality
        suffix = ""
        if q == "m":
            suffix = "m"
        elif q == "m6":
            suffix = "m6"
        elif q == "m7":
            suffix = "m7"
        elif q == "maj7":
            suffix = "M7"
        elif q == "7":
            suffix = "7"
        elif q == "sus4":
            suffix = "sus"
        elif q == "7sus4":
            suffix = "7sus"
        elif q == "m7b5":
            suffix = "m7-5"
        elif q == "add9":
            suffix = "(2)"

        res = f"{root}{suffix}"
        if spec.bass_degree is not None:
            b_acc = "b" if spec.bass_accidental == -1 else ("#" if spec.bass_accidental == 1 else "")
            res += f"/{spec.bass_degree}{b_acc}"
        return res
    except Exception:
        return raw.strip()


def melody_fit(
    chord_raw: str,
    melody_notes: list[tuple[float, int, int]],
    chord_beat: float = 1.0,
    chord_duration: float = 2.0,
) -> float:
    """Calculate the fit of a candidate chord against melody notes in its window.

    Evaluates the fraction of strong-beat and long melody notes that are chord tones,
    using resolve_chord in key C (tonic_pc=0).

    Args:
        chord_raw: Candidate chord symbol (e.g. '1(2)', '5m/7b').
        melody_notes: List of (onset_beat, degree, accidental) tuples.
        chord_beat: 1-based beat where the chord starts.
        chord_duration: Duration in beats of the chord's active window.

    Returns:
        Float score between 0.0 and 1.0 (0.5 if no melody notes exist).
    """
    try:
        resolved = resolve_chord(chord_raw, tonic_pc=0)
    except Exception:
        return 0.0

    chord_pcs = set(resolved.pcs)

    # Filter notes belonging to this chord's time window
    window_notes = [
        (onset, deg, acc)
        for onset, deg, acc in melody_notes
        if chord_beat - 0.01 <= onset < chord_beat + chord_duration - 0.01
    ]

    # If no notes inside window, fall back to entire measure melody notes
    eval_notes = window_notes if window_notes else melody_notes
    if not eval_notes:
        return 0.5

    weighted_match = 0.0
    total_weight = 0.0

    for onset, deg, acc in eval_notes:
        if deg not in DEGREE_TO_PC:
            continue
        note_pc = (DEGREE_TO_PC[deg] + acc) % 12
        # Strong beats (beat 1.0, 3.0, or start of chord) receive double weight
        is_strong = (abs(onset - round(onset)) < 0.05) or (abs(onset - chord_beat) < 0.05)
        weight = 2.0 if is_strong else 1.0

        if note_pc in chord_pcs:
            weighted_match += weight
        total_weight += weight

    if total_weight <= 0.0:
        return 0.5

    return round(weighted_match / total_weight, 3)


def bass_hint_fit(chord_raw: str, bass_hint: Optional[str]) -> Optional[float]:
    """Check whether the first note in bass_hint matches the chord's bass pitch class.

    Args:
        chord_raw: Candidate chord symbol.
        bass_hint: Printed 'Bs:' rhythm string (e.g. '11 11 11 112', '.7.7 .7.74').

    Returns:
        1.0 if match, 0.0 if mismatch, None if bass_hint is absent or unparseable.
    """
    if not bass_hint or not bass_hint.strip():
        return None

    try:
        resolved = resolve_chord(chord_raw, tonic_pc=0)
    except Exception:
        return 0.0

    # Extract first degree/accidental token from bass_hint
    s = bass_hint.strip()
    m = re.search(r"([b#\.]?)([1-7])", s)
    if not m:
        return None

    prefix, deg_str = m.groups()
    deg = int(deg_str)
    acc = -1 if prefix in ("b", ".") else (1 if prefix == "#" else 0)
    bass_hint_pc = (DEGREE_TO_PC[deg] + acc) % 12

    return 1.0 if resolved.bass_pc == bass_hint_pc else 0.0


def progression_plausibility(
    curr_chord: str,
    prev_chord: Optional[str] = None,
    next_chord: Optional[str] = None,
) -> float:
    """Score the transition plausibility of curr_chord in context.

    Returns:
        Score between 0.0 and 1.0.
    """
    try:
        parse_chord(curr_chord)
    except Exception:
        return 0.0

    curr_simp = _simplify_chord(curr_chord)
    scores: list[float] = []

    if prev_chord:
        prev_simp = _simplify_chord(prev_chord)
        pair = (prev_simp, curr_simp)
        if pair in COMMON_TRANSITIONS:
            scores.append(COMMON_TRANSITIONS[pair])
        else:
            # Check diatonic root motion
            scores.append(_root_motion_score(prev_chord, curr_chord))

    if next_chord:
        next_simp = _simplify_chord(next_chord)
        pair = (curr_simp, next_simp)
        if pair in COMMON_TRANSITIONS:
            scores.append(COMMON_TRANSITIONS[pair])
        else:
            scores.append(_root_motion_score(curr_chord, next_chord))

    if not scores:
        return 0.7  # Default plausibility when isolated

    return round(sum(scores) / len(scores), 3)


def _root_motion_score(c1_raw: str, c2_raw: str) -> float:
    """Heuristic score based on root interval distance."""
    try:
        s1 = parse_chord(c1_raw)
        s2 = parse_chord(c2_raw)
        # Root pitch classes in C
        pc1 = (DEGREE_TO_PC[s1.degree] + s1.accidental) % 12
        pc2 = (DEGREE_TO_PC[s2.degree] + s2.accidental) % 12
        diff = (pc2 - pc1) % 12

        # Circle of fifths / fourths: down 5th (diff=5) or up 4th (diff=5), down 4th (diff=7)
        if diff in (5, 7):
            return 0.85
        # Stepwise motion: up whole/half step (diff=1, 2) or down whole/half step (diff=10, 11)
        if diff in (1, 2, 10, 11):
            return 0.80
        # Third motion (diff=3, 4, 8, 9)
        if diff in (3, 4, 8, 9):
            return 0.70
        # Tritone (diff=6)
        return 0.40
    except Exception:
        return 0.5


def generate_candidates(raw: str) -> list[str]:
    """Generate plausible chord readings for common Taiwanese OMR confusions.

    Handles:
    - Accidental dropped/added before or after digit ('5m/7' <-> '5m/7b', '67/1' <-> '67/1#', '6b' <-> '6')
    - 'm' <-> '' ('2m7' <-> '27', '2m' <-> '2', '6m7' <-> '67')
    - 'M7' <-> '7' ('4M7' <-> '47')
    - '6' <-> 'b' (visual confusion between digit 6 and flat b)
    - '7b' <-> '7'
    - '1' <-> '7' in slash bass ('5/7' <-> '5/1')
    - '5sus' <-> '57sus'
    - '(2)' lost ('1(2)' <-> '1')

    Returns:
        List of unique, valid chord candidate strings.
    """
    candidates: set[str] = set()

    clean = raw.strip()
    if clean:
        candidates.add(clean)

    # 1. '(2)' lost / added
    if "(2)" in clean:
        candidates.add(clean.replace("(2)", ""))
    elif clean.startswith("1"):
        candidates.add(clean.replace("1", "1(2)", 1))

    # 2. '5sus' <-> '57sus'
    if "57sus" in clean:
        candidates.add(clean.replace("57sus", "5sus"))
    elif "5sus" in clean:
        candidates.add(clean.replace("5sus", "57sus"))

    # 3. 'M7' <-> '7'
    if "M7" in clean:
        candidates.add(clean.replace("M7", "7"))
    elif re.search(r"[1-7]7(?![a-zA-Z])", clean):
        candidates.add(re.sub(r"([1-7])7", r"\1M7", clean))

    # 4. 'm' <-> ''
    if "m" in clean:
        candidates.add(clean.replace("m", ""))
    else:
        # Try inserting m after root degree: e.g. '27' -> '2m7', '2' -> '2m', '67' -> '6m7'
        m = re.match(r"^([b#]?[1-7])(.*)$", clean)
        if m:
            root, rem = m.groups()
            candidates.add(f"{root}m{rem}")

    # 5. Slash bass '1' <-> '7' or accidental dropped in slash
    if "/" in clean:
        parts = clean.split("/", 1)
        root_part, bass_part = parts[0], parts[1]
        # '1' <-> '7' in bass
        if "1" in bass_part:
            candidates.add(f"{root_part}/{bass_part.replace('1', '7')}")
        if "7" in bass_part:
            candidates.add(f"{root_part}/{bass_part.replace('7', '1')}")
        # Accidental dropped in bass: e.g. /7 -> /7b or /1 -> /1#
        if "b" not in bass_part and "#" not in bass_part:
            candidates.add(f"{root_part}/{bass_part}b")
            candidates.add(f"{root_part}/{bass_part}#")
        elif "b" in bass_part:
            candidates.add(f"{root_part}/{bass_part.replace('b', '')}")
        elif "#" in bass_part:
            candidates.add(f"{root_part}/{bass_part.replace('#', '')}")

    # 6. '7b' <-> '7', '6b' <-> '6', '6' <-> 'b'
    if "7b" in clean:
        candidates.add(clean.replace("7b", "7"))
    if clean == "7":
        candidates.add("7b")
    if "6b" in clean:
        candidates.add(clean.replace("6b", "6"))
    if clean == "6":
        candidates.add("6b")
        candidates.add("6m")
        candidates.add("6m7")

    # Visual confusion: '6' <-> 'b'
    if "6" in clean:
        candidates.add(clean.replace("6", "b"))
    if "b" in clean and not clean.startswith("b"):
        candidates.add(clean.replace("b", "6"))

    # 7. Accidental dropped from root degree
    # E.g. '6b' -> '6', or '6' -> '6b', '7' -> '7b'
    m_root = re.match(r"^([1-7])(.*)$", clean)
    if m_root:
        d, rest = m_root.groups()
        candidates.add(f"{d}b{rest}")
        candidates.add(f"{d}#{rest}")

    # Validate each candidate with parse_chord; keep only grammatically valid chords
    valid_candidates: list[str] = []
    # Preserve raw first if valid
    for cand in [raw] + sorted(candidates):
        cand_str = cand.strip()
        if cand_str and cand_str not in valid_candidates:
            try:
                parse_chord(cand_str)
                valid_candidates.append(cand_str)
            except Exception:
                continue

    return valid_candidates


def score_candidate(
    chord_raw: str,
    melody_notes: list[tuple[float, int, int]],
    bass_hint: Optional[str] = None,
    prev_chord: Optional[str] = None,
    next_chord: Optional[str] = None,
    chord_beat: float = 1.0,
    chord_duration: float = 2.0,
) -> float:
    """Calculate an overall combined prior score for a candidate chord in context."""
    m_fit = melody_fit(chord_raw, melody_notes, chord_beat, chord_duration)
    prog = progression_plausibility(chord_raw, prev_chord, next_chord)
    b_fit = bass_hint_fit(chord_raw, bass_hint)

    if b_fit is not None:
        score = 0.40 * m_fit + 0.35 * prog + 0.25 * b_fit
    else:
        score = 0.55 * m_fit + 0.45 * prog

    return round(score, 3)
