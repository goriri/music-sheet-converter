"""Ukulele chord shape generator and fingering library for standard tuning G4 C4 E4 A4."""
from __future__ import annotations

import re
from typing import Optional

from app.models import Difficulty, UkeShape
from app.theory.chords import QUALITY_INTERVALS, format_chord_name, get_chord_pcs
from app.theory.keys import key_name_to_pc

# Standard re-entrant tuning (high G): G4=7, C4=0, E4=4, A4=9
OPEN_PCS: tuple[int, int, int, int] = (7, 0, 4, 9)

# Conventional root names for pitch classes 0..11
DEFAULT_ROOT_NAMES: dict[int, str] = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


def simplify_quality(quality: str, difficulty: Difficulty) -> str:
    """Reduce extensions to simpler triad/7th family for beginner difficulty when needed.

    Rule: simplified chord tones ⊆ original tones and must contain root+3rd/sus.
    """
    if difficulty != "beginner":
        return quality
    simplifications: dict[str, str] = {
        "9": "7",
        "m9": "m7",
        "maj9": "maj7",
        "7b9": "7",
        "7#9": "7",
        "7#5": "aug",
        "7add13": "7",
        "11": "7",
        "13": "7",
        "madd9": "m",
        "69": "maj",
        "6": "maj",
        "m6": "m",
        "add9": "maj",
        "mM7": "m",
        "7sus4": "sus4",
    }
    return simplifications.get(quality, quality)


def get_essential_tones(root_pc: int, quality: str) -> set[int]:
    """Return mandatory pitch classes that must be present in the voicing.

    Must contain root + 3rd (or sus tone) + 7th/6th when present + b5/#5 for dim/aug + 9 for 9/add9.
    5th may be omitted; no non-chord tones.
    """
    req: set[int] = {root_pc % 12}

    # 3rd or sus tone
    if quality in ("sus4", "7sus4"):
        req.add((root_pc + 5) % 12)
    elif quality == "sus2":
        req.add((root_pc + 2) % 12)
    elif quality in ("m", "m7", "m7b5", "dim", "m6", "m9", "mM7", "madd9"):
        req.add((root_pc + 3) % 12)
    else:
        req.add((root_pc + 4) % 12)

    # 7th or 6th
    if quality in ("7", "m7", "m7b5", "7sus4", "9", "m9", "7b9", "7#9", "7#5", "7add13", "11", "13"):
        req.add((root_pc + 10) % 12)
    elif quality in ("maj7", "maj9", "mM7"):
        req.add((root_pc + 11) % 12)
    elif quality == "dim":
        req.add((root_pc + 9) % 12)
    elif quality in ("6", "m6", "69"):
        req.add((root_pc + 9) % 12)

    # b5 / #5
    if quality in ("dim", "m7b5"):
        req.add((root_pc + 6) % 12)
    elif quality in ("aug", "7#5"):
        req.add((root_pc + 8) % 12)

    # 9th
    if quality in ("add9", "9", "m9", "maj9", "69", "madd9"):
        req.add((root_pc + 2) % 12)
    elif quality == "7b9":
        req.add((root_pc + 1) % 12)
    elif quality == "7#9":
        req.add((root_pc + 3) % 12)

    return req


def check_shape_pcs(frets: tuple[int, int, int, int], root_pc: int, quality: str) -> bool:
    """Verify that a shape contains only chord tones and all essential chord tones."""
    target_all = set(get_chord_pcs(root_pc, quality))
    req = get_essential_tones(root_pc, quality)
    if len(req) > 4:
        return False

    sounding_pcs = set((OPEN_PCS[i] + frets[i]) % 12 for i in range(4) if frets[i] >= 0)
    # No non-chord tones
    if not sounding_pcs.issubset(target_all):
        return False
    # All essential tones present
    if not req.issubset(sounding_pcs):
        return False
    return True


def assign_fingering(
    frets: tuple[int, int, int, int],
    curated_fingers: Optional[tuple[int, int, int, int]] = None,
    curated_barre: Optional[int] = None,
) -> tuple[tuple[int, int, int, int], Optional[int], int, int]:
    """Assign fingers (1..4), barre fret (or None), base_fret, and total fingers used.

    frets: (G, C, E, A) frets 0..12, -1 for muted.
    Returns (fingers, barre, base_fret, num_fingers).
    """
    pos = [f for f in frets if f > 0]
    if not pos:
        return (0, 0, 0, 0), None, 1, 0

    min_pos = min(pos)
    max_pos = max(pos)

    # base_fret: 1 if any open string or fits in frets 1..4, else min_pos
    if any(f == 0 for f in frets) or max_pos <= 4:
        base_fret = 1
    else:
        base_fret = min_pos

    if curated_fingers is not None:
        used = len(set(f for f in curated_fingers if f > 0))
        return curated_fingers, curated_barre, base_fret, used

    has_open = any(f == 0 for f in frets)
    min_count = sum(1 for f in frets if f == min_pos)

    is_barre = False
    barre_fret = None

    if not has_open and min_count >= 2:
        is_barre = True
        barre_fret = min_pos
    elif min_count == 4:
        is_barre = True
        barre_fret = min_pos

    fingers = [0, 0, 0, 0]

    if is_barre:
        for s in range(4):
            if frets[s] == min_pos:
                fingers[s] = 1
        other_strings = [s for s in range(4) if frets[s] > min_pos]
        other_strings.sort(key=lambda s: (frets[s], s))
        next_finger = 2
        for s in other_strings:
            fingers[s] = min(next_finger, 4)
            next_finger += 1
        num_fingers = 1 + len(other_strings)
    else:
        fretted_strings = [s for s in range(4) if frets[s] > 0]
        fretted_strings.sort(key=lambda s: (frets[s], s))
        for idx, s in enumerate(fretted_strings):
            fingers[s] = min(idx + 1, 4)
        num_fingers = len(fretted_strings)

    return tuple(fingers), barre_fret, base_fret, num_fingers


def shape_cost(
    frets: tuple[int, int, int, int],
    fingers: tuple[int, int, int, int],
    barre: Optional[int],
    difficulty: Difficulty,
    is_curated: bool = False,
) -> float:
    """Calculate playable difficulty cost for a chord shape."""
    pos = [f for f in frets if f > 0]
    span = (max(pos) - min(pos)) if pos else 0
    max_fret = max(pos) if pos else 0
    open_count = sum(1 for f in frets if f == 0)

    # Count distinct fingers used
    distinct_fingers = len(set(f for f in fingers if f > 0))

    cost = 2.0 * distinct_fingers
    cost += 1.0 * max_fret
    if max_fret > 4:
        cost += 2.0 * (max_fret - 4)

    cost += 1.5 * span
    cost -= 1.5 * open_count

    if barre is not None:
        if difficulty == "beginner":
            cost += 12.0
        elif difficulty == "intermediate":
            cost += 2.5
        else:
            cost += 0.5

    if difficulty == "beginner":
        if span > 2:
            cost += 25.0
        if max_fret > 4:
            cost += 15.0

    if is_curated:
        cost -= 5.0

    return cost


# Standard well-known curated shapes (G, C, E, A frets)
# Format: (name, quality, frets, optional_fingers, optional_barre)
_CURATED_LIST: list[tuple[str, str, tuple[int, int, int, int], Optional[tuple[int, int, int, int]], Optional[int]]] = [
    # C
    ("C", "maj", (0, 0, 0, 3), (0, 0, 0, 3), None),
    ("Cmaj7", "maj7", (0, 0, 0, 2), (0, 0, 0, 2), None),
    ("C7", "7", (0, 0, 0, 1), (0, 0, 0, 1), None),
    ("Csus4", "sus4", (0, 0, 1, 3), (0, 0, 1, 3), None),
    ("Cadd9", "add9", (0, 2, 0, 3), (0, 1, 0, 2), None),
    ("Cm", "m", (0, 3, 3, 3), (0, 1, 2, 3), None),
    ("C6", "6", (0, 0, 0, 0), (0, 0, 0, 0), None),
    ("Cdim", "dim", (2, 3, 2, 3), (1, 3, 2, 4), None),
    # D
    ("D", "maj", (2, 2, 2, 0), (1, 2, 3, 0), None),
    ("Dm", "m", (2, 2, 1, 0), (2, 3, 1, 0), None),
    ("D7", "7", (2, 2, 2, 3), (1, 1, 1, 2), 2),
    ("Dm7", "m7", (2, 2, 1, 3), (2, 3, 1, 4), None),
    ("Dmaj7", "maj7", (2, 2, 2, 4), (1, 1, 1, 3), 2),
    ("Dsus4", "sus4", (2, 2, 3, 0), (1, 2, 3, 0), None),
    # E
    ("E", "maj", (1, 4, 0, 2), (1, 4, 0, 2), None),
    ("E", "maj", (4, 4, 4, 2), (2, 3, 4, 1), 2),
    ("Em", "m", (0, 4, 3, 2), (0, 3, 2, 1), None),
    ("E7", "7", (1, 2, 0, 2), (1, 2, 0, 3), None),
    ("Em7", "m7", (0, 2, 0, 2), (0, 1, 0, 2), None),
    ("Emaj7", "maj7", (1, 3, 0, 2), (1, 3, 0, 2), None),
    ("Esus4", "sus4", (2, 4, 0, 2), (1, 3, 0, 2), None),
    # F
    ("F", "maj", (2, 0, 1, 0), (2, 0, 1, 0), None),
    ("Fmaj7", "maj7", (2, 4, 1, 3), (2, 4, 1, 3), None),
    ("Fadd9", "add9", (0, 0, 1, 0), (0, 0, 1, 0), None),
    ("F7", "7", (2, 3, 1, 0), (2, 3, 1, 0), None),
    ("Fm", "m", (1, 0, 1, 3), (1, 0, 2, 4), None),
    ("Fsus4", "sus4", (3, 0, 1, 1), (3, 0, 1, 1), 1),
    # F# / Gb
    ("F#", "maj", (3, 1, 2, 1), (3, 1, 2, 1), 1),
    ("F#m", "m", (2, 1, 2, 0), (2, 1, 3, 0), None),
    ("F#7", "7", (3, 4, 2, 4), (2, 3, 1, 4), None),
    ("F#m7", "m7", (2, 4, 2, 4), (1, 3, 1, 4), 2),
    # G
    ("G", "maj", (0, 2, 3, 2), (0, 1, 3, 2), None),
    ("G7", "7", (0, 2, 1, 2), (0, 2, 1, 3), None),
    ("Gm", "m", (0, 2, 3, 1), (0, 2, 3, 1), None),
    ("Gm7", "m7", (0, 2, 1, 1), (0, 2, 1, 1), 1),
    ("Gmaj7", "maj7", (0, 2, 2, 2), (0, 1, 1, 1), 2),
    ("Gsus4", "sus4", (0, 2, 3, 3), (0, 1, 2, 3), None),
    ("Gadd9", "add9", (0, 2, 5, 2), (0, 1, 4, 2), None),
    ("G6", "6", (0, 2, 0, 2), (0, 1, 0, 2), None),
    # Ab / G#
    ("Ab", "maj", (5, 3, 4, 3), (4, 1, 3, 2), 3),
    ("Abmaj7", "maj7", (1, 3, 3, 3), (1, 2, 3, 4), None),
    ("Ab7", "7", (1, 3, 2, 3), (1, 3, 2, 4), None),
    # A
    ("A", "maj", (2, 1, 0, 0), (2, 1, 0, 0), None),
    ("A7", "7", (0, 1, 0, 0), (0, 1, 0, 0), None),
    ("Am", "m", (2, 0, 0, 0), (2, 0, 0, 0), None),
    ("Am7", "m7", (0, 0, 0, 0), (0, 0, 0, 0), None),
    ("Amaj7", "maj7", (1, 1, 0, 0), (1, 2, 0, 0), None),
    ("Asus4", "sus4", (2, 2, 0, 0), (1, 2, 0, 0), None),
    # Bb
    ("Bb", "maj", (3, 2, 1, 1), (3, 2, 1, 1), 1),
    ("Bbm", "m", (3, 1, 1, 1), (3, 1, 1, 1), 1),
    ("Bb7", "7", (1, 2, 1, 1), (1, 2, 1, 1), 1),
    ("Bbmaj7", "maj7", (3, 2, 1, 0), (3, 2, 1, 0), None),
    ("Bbsus4", "sus4", (3, 3, 1, 1), (2, 3, 1, 1), 1),
    # B
    ("B", "maj", (4, 3, 2, 2), (3, 2, 1, 1), 2),
    ("Bm", "m", (4, 2, 2, 2), (3, 1, 1, 1), 2),
    ("B7", "7", (2, 3, 2, 2), (1, 2, 1, 1), 2),
    ("Bm7", "m7", (2, 2, 2, 2), (1, 1, 1, 1), 2),
    ("Bmaj7", "maj7", (4, 3, 2, 1), (4, 3, 2, 1), None),
]


def _build_curated_index() -> dict[tuple[int, str], list[dict]]:
    index: dict[tuple[int, str], list[dict]] = {}
    for name, qual, frets, fingers, barre in _CURATED_LIST:
        m = re.match(r"^([A-G][b#]?)", name)
        if not m:
            continue
        root_name = m.group(1)
        root_pc = key_name_to_pc(root_name)
        key = (root_pc, qual)
        if key not in index:
            index[key] = []
        index[key].append({
            "name": name,
            "quality": qual,
            "frets": frets,
            "fingers": fingers,
            "barre": barre,
        })
    return index


CURATED_INDEX: dict[tuple[int, str], list[dict]] = _build_curated_index()


def search_algorithmic_shapes(
    root_pc: int,
    quality: str,
    max_fret: int = 12,
    max_span: int = 3,
) -> list[tuple[int, int, int, int]]:
    """Algorithmically find all valid 4-string fingerings satisfying essential tones."""
    target_all = set(get_chord_pcs(root_pc, quality))
    req = get_essential_tones(root_pc, quality)
    if len(req) > 4:
        return []

    # Valid frets per string
    candidates_per_str: list[list[int]] = []
    for s_idx, open_pc in enumerate(OPEN_PCS):
        valid: list[int] = []
        for f in range(max_fret + 1):
            if ((open_pc + f) % 12) in target_all:
                valid.append(f)
        candidates_per_str.append(valid)

    results: list[tuple[int, int, int, int]] = []
    for f0 in candidates_per_str[0]:
        for f1 in candidates_per_str[1]:
            for f2 in candidates_per_str[2]:
                for f3 in candidates_per_str[3]:
                    frets = (f0, f1, f2, f3)
                    pos = [f for f in frets if f > 0]
                    if pos:
                        if (max(pos) - min(pos)) > max_span:
                            continue
                    sounding_pcs = set((OPEN_PCS[i] + frets[i]) % 12 for i in range(4))
                    if req.issubset(sounding_pcs):
                        results.append(frets)

    return results


def candidate_shapes(
    root_pc: int,
    quality: str,
    *,
    difficulty: Difficulty = "intermediate",
    root_name: Optional[str] = None,
) -> list[UkeShape]:
    """Return all candidate UkeShapes for root_pc and quality, sorted by difficulty cost.

    Names are formatted in shape-key spelling.
    If difficulty == 'beginner' and quality is extended, also considers simplified quality shapes.
    """
    root_pc = root_pc % 12
    r_name = root_name if root_name is not None else DEFAULT_ROOT_NAMES[root_pc]
    shape_name = format_chord_name(r_name, quality)

    max_span = 2 if difficulty == "beginner" else 3

    seen_frets: set[tuple[int, int, int, int]] = set()
    scored_shapes: list[tuple[float, UkeShape]] = []

    # 1. Curated shapes for this quality
    curated_entries = CURATED_INDEX.get((root_pc, quality), [])
    for entry in curated_entries:
        frets = entry["frets"]
        if not check_shape_pcs(frets, root_pc, quality):
            continue
        pos = [f for f in frets if f > 0]
        span = (max(pos) - min(pos)) if pos else 0
        if difficulty == "beginner" and span > max_span:
            continue

        fingers, barre, base_fret, _ = assign_fingering(
            frets,
            curated_fingers=entry.get("fingers"),
            curated_barre=entry.get("barre"),
        )
        c = shape_cost(frets, fingers, barre, difficulty, is_curated=True)
        shape = UkeShape(
            name=shape_name,
            frets=frets,
            fingers=fingers,
            barre=barre,
            base_fret=base_fret,
        )
        scored_shapes.append((c, shape))
        seen_frets.add(frets)

    # 2. Algorithmic search
    algo_frets = search_algorithmic_shapes(root_pc, quality, max_fret=12, max_span=max_span)
    for frets in algo_frets:
        if frets in seen_frets:
            continue
        fingers, barre, base_fret, num_f = assign_fingering(frets)
        if num_f > 4:
            continue
        c = shape_cost(frets, fingers, barre, difficulty, is_curated=False)
        shape = UkeShape(
            name=shape_name,
            frets=frets,
            fingers=fingers,
            barre=barre,
            base_fret=base_fret,
        )
        scored_shapes.append((c, shape))
        seen_frets.add(frets)

    target_all = set(get_chord_pcs(root_pc, quality))
    # Determine 3rd or sus tone
    if quality in ("sus4", "7sus4"):
        third_or_sus = (root_pc + 5) % 12
    elif quality == "sus2":
        third_or_sus = (root_pc + 2) % 12
    elif "m" in quality and quality not in ("maj", "maj7", "maj9", "69", "add9"):
        third_or_sus = (root_pc + 3) % 12
    else:
        third_or_sus = (root_pc + 4) % 12
    min_essential = {root_pc % 12, third_or_sus}

    # 3. If no shapes found, or if difficulty is beginner and quality can be simplified:
    simplified = simplify_quality(quality, difficulty)
    if (not scored_shapes) or (difficulty == "beginner" and simplified != quality):
        sim_name = format_chord_name(r_name, simplified)
        # Search curated for simplified
        for entry in CURATED_INDEX.get((root_pc, simplified), []):
            frets = entry["frets"]
            if frets in seen_frets:
                continue
            sounding_pcs = set((OPEN_PCS[i] + frets[i]) % 12 for i in range(4) if frets[i] >= 0)
            if not sounding_pcs.issubset(target_all) or not min_essential.issubset(sounding_pcs):
                continue
            pos = [f for f in frets if f > 0]
            span = (max(pos) - min(pos)) if pos else 0
            if difficulty == "beginner" and span > max_span:
                continue
            fingers, barre, base_fret, _ = assign_fingering(
                frets,
                curated_fingers=entry.get("fingers"),
                curated_barre=entry.get("barre"),
            )
            c = shape_cost(frets, fingers, barre, difficulty, is_curated=True)
            shape = UkeShape(
                name=sim_name,
                frets=frets,
                fingers=fingers,
                barre=barre,
                base_fret=base_fret,
            )
            scored_shapes.append((c, shape))
            seen_frets.add(frets)

        # Search algo for simplified
        algo_sim = search_algorithmic_shapes(root_pc, simplified, max_fret=12, max_span=max_span)
        for frets in algo_sim:
            if frets in seen_frets:
                continue
            sounding_pcs = set((OPEN_PCS[i] + frets[i]) % 12 for i in range(4) if frets[i] >= 0)
            if not sounding_pcs.issubset(target_all) or not min_essential.issubset(sounding_pcs):
                continue
            fingers, barre, base_fret, num_f = assign_fingering(frets)
            if num_f > 4:
                continue
            c = shape_cost(frets, fingers, barre, difficulty, is_curated=False)
            shape = UkeShape(
                name=sim_name,
                frets=frets,
                fingers=fingers,
                barre=barre,
                base_fret=base_fret,
            )
            scored_shapes.append((c, shape))
            seen_frets.add(frets)

    # 4. If still empty, relax max_span to 3 while strictly preserving subset of original chord tones
    if not scored_shapes:
        for span_limit in [3, 4]:
            for frets in search_algorithmic_shapes(root_pc, quality, max_fret=12, max_span=span_limit):
                if frets in seen_frets:
                    continue
                sounding_pcs = set((OPEN_PCS[i] + frets[i]) % 12 for i in range(4) if frets[i] >= 0)
                if not sounding_pcs.issubset(target_all) or not min_essential.issubset(sounding_pcs):
                    continue
                fingers, barre, base_fret, num_f = assign_fingering(frets)
                if num_f > 4:
                    continue
                c = shape_cost(frets, fingers, barre, difficulty, is_curated=False) + 5.0
                shape = UkeShape(
                    name=shape_name,
                    frets=frets,
                    fingers=fingers,
                    barre=barre,
                    base_fret=base_fret,
                )
                scored_shapes.append((c, shape))
                seen_frets.add(frets)
            if scored_shapes:
                break

    scored_shapes.sort(key=lambda item: item[0])
    return [s for _, s in scored_shapes]
