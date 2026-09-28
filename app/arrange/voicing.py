"""Voicing generation, voice leading dynamic programming (Viterbi), and bass line selection."""
from __future__ import annotations

from typing import Optional
from app.models import Difficulty, ResolvedChord

# =============================================================================
# VOICE LEADING & REGISTER COST CONSTANTS
# =============================================================================
WEIGHT_MOVEMENT: float = 1.0          # Penalty per semitone of total voice movement between consecutive chords
REWARD_COMMON_TONE: float = 3.0       # Bonus (subtracted cost) for each exact MIDI pitch retained across chords
WEIGHT_REGISTER_DRIFT: float = 0.8    # Penalty weight for RH top note drifting away from optimal center
TARGET_TOP_MIN: int = 71              # B4: lower bound of preferred top note register
TARGET_TOP_MAX: int = 76              # E5: upper bound of preferred top note register
TARGET_TOP_CENTER: int = 73           # C#5: ideal center for top melody/voicing line
WEIGHT_SPAN: float = 1.2              # Penalty weight for large hand span (> 10 semitones)
PENALTY_BLACK_KEY_THUMB: float = 2.0  # Penalty when RH thumb (lowest note) lands on a black key
FORBIDDEN_COST: float = 1e6           # Cost threshold for physically impossible voicings

# MIDI pitch boundaries
RH_MIN_MIDI: int = 64  # E4 (lowest allowed RH note)
RH_MAX_MIDI: int = 81  # A5 (highest allowed RH note)
LH_MIN_BASS: int = 36  # C2
LH_MAX_BASS: int = 52  # E3
LH_CENTER_BASS: int = 43  # G2

BLACK_KEY_PCS = {1, 3, 6, 8, 10}


def _generate_close_voicings(
    pitch_classes: list[int],
    max_notes: int = 4,
    max_span: int = 14,
) -> list[tuple[int, ...]]:
    """Generate close voicings for a sequence of pitch classes within MIDI range."""
    results: list[tuple[int, ...]] = []
    k = len(pitch_classes)
    for inv in range(k):
        inverted_pcs = pitch_classes[inv:] + pitch_classes[:inv]
        for base_octave in range(4, 7):
            base_midi = base_octave * 12 + inverted_pcs[0]
            if base_midi < RH_MIN_MIDI:
                continue
            voicing = [base_midi]
            prev = base_midi
            valid = True
            for pc in inverted_pcs[1:]:
                note = prev + 1
                while note % 12 != pc:
                    note += 1
                if note > RH_MAX_MIDI:
                    valid = False
                    break
                voicing.append(note)
                prev = note

            if valid and voicing[-1] <= RH_MAX_MIDI:
                span = voicing[-1] - voicing[0]
                if span <= max_span and len(voicing) <= max_notes:
                    results.append(tuple(voicing))
    return results


def get_candidate_rh_voicings(chord: ResolvedChord, difficulty: Difficulty) -> list[tuple[int, ...]]:
    """Generate musically valid RH candidate voicings (as sorted MIDI pitch tuples).

    Constraints:
      - All notes in [RH_MIN_MIDI(64), RH_MAX_MIDI(81)]
      - Beginner: close-position triads, span <= 12
        (sus4/7sus4 must contain 4th; m7b5 must contain b5th; 7ths/add9 reduced to triad)
      - Intermediate & Advanced:
        (7ths must contain 7th; add9 must contain 9th; sus4/7sus4 must contain 4th; span <= 14)
    """
    pcs = chord.pcs
    candidates: list[tuple[int, ...]] = []
    max_span = 12 if difficulty == "beginner" else 14

    if difficulty == "beginner":
        # Beginner: 3-note voicings
        if chord.quality in ("sus4", "7sus4"):
            # Must keep 4th: root, 4th, 5th
            b_pcs = [pcs[0], pcs[1], pcs[2]]
        elif chord.quality == "m7b5":
            # Must keep b5: root, b3, b5
            b_pcs = [pcs[0], pcs[1], pcs[2]]
        else:
            # Triad (root, 3rd, 5th)
            b_pcs = pcs[:3] if len(pcs) >= 3 else pcs

        candidates.extend(_generate_close_voicings(b_pcs, max_notes=3, max_span=max_span))

    else:
        # Intermediate / Advanced:
        if len(pcs) <= 3:
            # Triads (maj, m, sus4, aug, dim)
            candidates.extend(_generate_close_voicings(pcs, max_notes=3, max_span=max_span))
            doubled_root_pcs = [pcs[0], pcs[1], pcs[2], pcs[0]]
            doubled_fifth_pcs = [pcs[0], pcs[2], pcs[1], pcs[2]]
            candidates.extend(_generate_close_voicings(doubled_root_pcs, max_notes=4, max_span=max_span))
            candidates.extend(_generate_close_voicings(doubled_fifth_pcs, max_notes=4, max_span=max_span))
        elif len(pcs) == 4:
            # 4-note chords (7, maj7, m7, m7b5, dim, 7sus4, add9, 6, m6):
            # 1. 4-note voicings (guaranteed to contain all 4 tones including color tone)
            candidates.extend(_generate_close_voicings(pcs, max_notes=4, max_span=max_span))

            # 2. 3-note voicings
            if chord.quality in ("m7b5", "dim"):
                # Must retain 3rd (b3), b5 (pcs[2]), and 7th (pcs[3])
                v_3note_patterns = [
                    [pcs[1], pcs[2], pcs[3]],
                ]
            else:
                color_tone = pcs[3]
                v_3note_patterns = [
                    [pcs[0], pcs[1], color_tone],  # Root + 3rd + color tone (omits 5th)
                    [pcs[1], pcs[2], color_tone],  # 3rd + 5th + color tone (omits root)
                ]
            for pat in v_3note_patterns:
                candidates.extend(_generate_close_voicings(pat, max_notes=3, max_span=max_span))
        else:
            # 5-note chords (9, m9, maj9): [root, 3rd, 5th, 7th, 9th]
            # Defining tones are 3rd (pcs[1]), 7th (pcs[3]), 9th (pcs[4]).
            # 1. 4-note voicings (omit 5th or omit root):
            candidates.extend(_generate_close_voicings([pcs[0], pcs[1], pcs[3], pcs[4]], max_notes=4, max_span=max_span))
            candidates.extend(_generate_close_voicings([pcs[1], pcs[2], pcs[3], pcs[4]], max_notes=4, max_span=max_span))

            # 2. 3-note voicings that RETAIN all defining tones (3rd, 7th, 9th):
            candidates.extend(_generate_close_voicings([pcs[1], pcs[3], pcs[4]], max_notes=3, max_span=max_span))

    # Remove duplicates preserving order
    unique_candidates = list(dict.fromkeys(candidates))

    # Fallback safety: ensure candidates is never empty
    if not unique_candidates:
        if difficulty == "beginner":
            if chord.quality in ("sus4", "7sus4"):
                fb_pcs = [pcs[0], pcs[1], pcs[2]]
            elif chord.quality in ("m7b5", "dim"):
                fb_pcs = [pcs[0], pcs[1], pcs[2]]
            else:
                fb_pcs = pcs[:3] if len(pcs) >= 3 else pcs
        else:
            if len(pcs) >= 5:
                fb_pcs = [pcs[0], pcs[1], pcs[3], pcs[4]]
            else:
                fb_pcs = pcs[:4] if len(pcs) >= 4 else pcs[:3]
        fb = []
        for pc in fb_pcs:
            note = 60 + pc
            while note < RH_MIN_MIDI:
                note += 12
            while note > RH_MAX_MIDI and note - 12 >= RH_MIN_MIDI:
                note -= 12
            fb.append(note)
        fb.sort()
        unique_candidates.append(tuple(fb))

    return unique_candidates


def calculate_static_voicing_cost(voicing: tuple[int, ...]) -> float:
    """Calculate the standalone register, span, and thumb cost of a voicing."""
    cost = 0.0
    top = voicing[-1]
    bottom = voicing[0]
    span = top - bottom

    # 1. Register drift penalty
    if top < TARGET_TOP_MIN:
        cost += WEIGHT_REGISTER_DRIFT * ((TARGET_TOP_MIN - top) ** 2)
    elif top > TARGET_TOP_MAX:
        cost += WEIGHT_REGISTER_DRIFT * ((top - TARGET_TOP_MAX) ** 2)
    else:
        cost += 0.2 * abs(top - TARGET_TOP_CENTER)

    # 2. Hand span penalty
    if span > 10:
        cost += WEIGHT_SPAN * (span - 10)

    # 3. Black-key thumb penalty
    if (bottom % 12) in BLACK_KEY_PCS:
        cost += PENALTY_BLACK_KEY_THUMB

    return cost


def calculate_transition_cost(prev: tuple[int, ...], curr: tuple[int, ...]) -> float:
    """Calculate the voice-leading transition cost between two consecutive voicings."""
    cost = 0.0

    # 1. Voice movement distance
    if len(prev) == len(curr):
        voice_dist = sum(abs(curr[i] - prev[i]) for i in range(len(curr)))
    else:
        # Match each note in curr to closest note in prev and vice-versa
        d1 = sum(min(abs(c - p) for p in prev) for c in curr)
        d2 = sum(min(abs(p - c) for c in curr) for p in prev)
        voice_dist = (d1 + d2) / 2.0 + 2.0 * abs(len(curr) - len(prev))

    cost += WEIGHT_MOVEMENT * voice_dist

    # 2. Common tone reward
    common_tones = sum(1 for p in prev if p in curr)
    cost -= REWARD_COMMON_TONE * common_tones

    return cost


def select_rh_voicings(chords: list[ResolvedChord], difficulty: Difficulty) -> list[tuple[int, ...]]:
    """Viterbi / DP algorithm to select optimal RH voicings across the sequence of chords.

    Minimizes total voice movement, rewards common tones, and enforces register boundaries.
    """
    if not chords:
        return []

    # Generate candidate voicings for each chord
    chord_candidates = [get_candidate_rh_voicings(ch, difficulty) for ch in chords]

    n = len(chords)
    # cost_table[t][c] = minimum cost to reach candidate c at step t
    cost_table: list[dict[tuple[int, ...], float]] = []
    backpointer: list[dict[tuple[int, ...], Optional[tuple[int, ...]]]] = []

    # Step 0: Initialize with static costs
    cost_table.append({c: calculate_static_voicing_cost(c) for c in chord_candidates[0]})
    backpointer.append({c: None for c in chord_candidates[0]})

    # Steps 1 .. n-1: DP transitions
    for t in range(1, n):
        curr_costs: dict[tuple[int, ...], float] = {}
        curr_bp: dict[tuple[int, ...], Optional[tuple[int, ...]]] = {}
        prev_candidates = chord_candidates[t - 1]

        for curr in chord_candidates[t]:
            curr_static = calculate_static_voicing_cost(curr)
            best_total = float("inf")
            best_prev = None

            for prev in prev_candidates:
                prev_cost = cost_table[t - 1][prev]
                trans_cost = calculate_transition_cost(prev, curr)
                total = prev_cost + trans_cost + curr_static
                if total < best_total:
                    best_total = total
                    best_prev = prev

            curr_costs[curr] = best_total
            curr_bp[curr] = best_prev

        cost_table.append(curr_costs)
        backpointer.append(curr_bp)

    # Backtrack optimal path
    last_step = cost_table[-1]
    best_last = min(last_step.keys(), key=lambda c: last_step[c])

    path = [best_last]
    curr = best_last
    for t in range(n - 1, 0, -1):
        prev = backpointer[t][curr]
        assert prev is not None
        path.append(prev)
        curr = prev

    path.reverse()
    return path


def _get_bass_candidates(chord: ResolvedChord) -> list[int]:
    """Return valid LH bass candidates for a chord in range [LH_MIN_BASS, LH_MAX_BASS]."""
    target_pc = chord.bass_pc % 12
    cand = [p for p in range(LH_MIN_BASS, LH_MAX_BASS + 1) if p % 12 == target_pc]
    if not cand:
        cand = [LH_MIN_BASS + target_pc]
    return cand


def select_lh_bass_notes(chords: list[ResolvedChord]) -> list[int]:
    """Select LH bass pitches (C2..C3, up to E3) for a chord sequence preferring stepwise motion."""
    if not chords:
        return []

    all_cands = [_get_bass_candidates(ch) for ch in chords]
    n = len(chords)

    # DP over bass notes
    costs: list[dict[int, float]] = []
    backpointer: list[dict[int, Optional[int]]] = []

    # Step 0
    costs.append({p: 0.5 * abs(p - LH_CENTER_BASS) for p in all_cands[0]})
    backpointer.append({p: None for p in all_cands[0]})

    for t in range(1, n):
        curr_costs: dict[int, float] = {}
        curr_bp: dict[int, Optional[int]] = {}

        for curr in all_cands[t]:
            best_cost = float("inf")
            best_prev = None
            for prev in all_cands[t - 1]:
                diff = abs(curr - prev)
                # Stepwise motion bonus (1-2 semitones)
                step_cost = diff * 0.8
                if diff in (1, 2):
                    step_cost -= 2.0
                elif diff in (3, 4):
                    step_cost -= 1.0
                elif diff >= 7:
                    step_cost += 3.0

                total = costs[t - 1][prev] + step_cost + 0.3 * abs(curr - LH_CENTER_BASS)
                if total < best_cost:
                    best_cost = total
                    best_prev = prev

            curr_costs[curr] = best_cost
            curr_bp[curr] = best_prev

        costs.append(curr_costs)
        backpointer.append(curr_bp)

    best_last = min(costs[-1].keys(), key=lambda p: costs[-1][p])
    path = [best_last]
    curr = best_last
    for t in range(n - 1, 0, -1):
        prev = backpointer[t][curr]
        assert prev is not None
        path.append(prev)
        curr = prev

    path.reverse()
    return path


def get_lh_pattern_pitches(bass_pitch: int, chord: ResolvedChord) -> dict[str, int]:
    """Calculate the components for broken LH accompaniment (root, 5th, 8ve, 10th).

    The lowest note (root) is the chord bass (slash bass if specified).
    All upper notes (fifth, octave, tenth) are strictly chord tones (in chord.pcs),
    chosen near the standard fifth, octave, and tenth positions.
    """
    root = bass_pitch
    chord_pcs = set(chord.pcs) if chord.pcs else {chord.root_pc}

    # 1. "Fifth" (middle tone): chord tone strictly > root, near root + 7
    cands_fifth = [p for p in range(root + 3, root + 12) if p % 12 in chord_pcs]
    if cands_fifth:
        # Prefer root or 5th, closest to root + 7
        best_fifth = min(
            cands_fifth,
            key=lambda p: abs(p - (root + 7)) - (1.5 if p % 12 == chord.root_pc else 0.0),
        )
    else:
        best_fifth = root + 7

    # 2. "Octave" (upper tone): chord tone near root + 12
    if (root + 12) % 12 in chord_pcs:
        best_octave = root + 12
    else:
        cands_oct = [p for p in range(root + 8, root + 16) if p % 12 in chord_pcs]
        if cands_oct:
            best_octave = min(cands_oct, key=lambda p: abs(p - (root + 12)))
        else:
            best_octave = root + 12

    # 3. "Tenth" (wide extension): chord tone > octave and <= 64 (below/at RH boundary)
    is_minor = chord.quality in ("m", "m7", "m6", "m9", "m7b5")
    is_sus = "sus" in chord.quality
    if is_sus:
        target_tenth = root + 17
    elif is_minor:
        target_tenth = root + 15
    else:
        target_tenth = root + 16

    cands_tenth = [p for p in range(best_octave + 1, 65) if p % 12 in chord_pcs]
    if cands_tenth:
        best_tenth = min(cands_tenth, key=lambda p: abs(p - target_tenth))
    else:
        best_tenth = best_octave

    return {
        "root": root,
        "fifth": best_fifth,
        "octave": best_octave,
        "tenth": best_tenth,
    }
