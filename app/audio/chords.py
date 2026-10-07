"""Beat-synchronous chord decoding and key estimation.

Provides pure-NumPy chord decoding from BTC-ISMIR19 frame posteriors,
supporting baseline argmax decoding and Viterbi HMM decoding with:
- Grouped root/triad harmonic mass to prevent seventh dilution
- Downbeat-aware transition penalties (chords change on beats 1 and 3)
- Chord-tone constrained bass/inversion detection with temporal hysteresis
- Key-aware diatonic boost
- Pitch-class duration based key estimation
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np

from app.audio.analysis import ChordSeg, NoteSeg
from app.theory.keys import KEY_NAME_TO_PC, canonical_key_for_pc

logger = logging.getLogger(__name__)

BTC_ROOT_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BTC_QUALITY_LIST = [
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
    "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
]

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

FAMILIES = ["maj", "min", "dim", "sus4"]
STATES: list[str] = []
STATE_ROOT: dict[str, Optional[int]] = {}
STATE_FAM: dict[str, str] = {}
for _r_idx, _r_name in enumerate(BTC_ROOT_LIST):
    for _fam in FAMILIES:
        _s = f"{_r_name}:{_fam}"
        STATES.append(_s)
        STATE_ROOT[_s] = _r_idx
        STATE_FAM[_s] = _fam
STATES.append("N")
STATE_ROOT["N"] = None
STATE_FAM["N"] = "N"
STATE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(STATES)}
NUM_STATES = len(STATES)


def simplify_btc_chord(label: str) -> str:
    """Map rare or ambiguous BTC chord qualities to common, robust pop chords."""
    if not label or label in ("N", "X"):
        return "N"
    if ":" not in label:
        return label
    root, qual = label.split(":", 1)
    mapping = {
        "min6": "min7",
        "maj6": "maj",
        "minmaj7": "min",
        "aug": "maj",
        "dim7": "dim",
        "sus2": "sus4",
    }
    return f"{root}:{mapping.get(qual, qual)}"


def smooth_beat_chords(chords: list[str], confs: list[float]) -> list[str]:
    """Filter out isolated 1-beat chord glitches using 2-beat median/neighbor smoothing."""
    if len(chords) < 3:
        return list(chords)

    res = list(chords)
    n = len(res)

    for i in range(1, n - 1):
        if res[i - 1] == res[i + 1] and res[i] != res[i - 1]:
            res[i] = res[i - 1]

    i = 0
    while i < n:
        j = i
        while j < n and res[j] == res[i]:
            j += 1
        run_len = j - i
        if run_len == 1:
            if i == 0 and n > 1:
                res[0] = res[1]
            elif i == n - 1 and n > 1:
                res[n - 1] = res[n - 2]
            elif i > 0 and i + 1 < n:
                if res[i - 1] == res[i + 1]:
                    res[i] = res[i - 1]
                else:
                    c_prev = confs[i - 1] if i - 1 < len(confs) else 0.5
                    c_next = confs[i + 1] if i + 1 < len(confs) else 0.5
                    res[i] = res[i - 1] if c_prev >= c_next else res[i + 1]
        i = j

    for i in range(1, n - 1):
        if res[i - 1] == res[i + 1] and res[i] != res[i - 1]:
            res[i] = res[i - 1]

    return res


def detect_segment_bass(
    label: str,
    t_start: float,
    t_end: float,
    chroma_bass: np.ndarray,
    hop_dur: float,
    min_energy_ratio: float = 1.20,
    min_frame_ratio: float = 0.25,
) -> Optional[int]:
    """Detect inverted bass pitch class (slash chord) constrained to valid chord tones.

    Only accepts candidates that are valid chord tones (3rd, 5th, or 7th) or passing 2nd.
    Enforces energy ratio over the segment mean and frame persistence hysteresis.
    """
    if not label or label in ("N", "X") or chroma_bass.size == 0 or chroma_bass.shape[0] < 12:
        return None

    m_root = re.match(r"^([A-Ga-g][b#]?)", label)
    if not m_root:
        return None
    r_pc = KEY_NAME_TO_PC.get(m_root.group(1).upper())
    if r_pc is None:
        return None

    sf_start = max(0, int(round(t_start / hop_dur)))
    sf_end = min(chroma_bass.shape[1], max(sf_start + 1, int(round(t_end / hop_dur))))
    if sf_start >= chroma_bass.shape[1] or sf_end <= sf_start:
        return None

    sub_bass = chroma_bass[:, sf_start:sf_end]
    if sub_bass.shape[1] == 0:
        return None

    b_vec = np.mean(sub_bass, axis=1)
    mean_e = float(np.mean(b_vec))
    if mean_e < 1e-4:
        return None

    is_min = "min" in label and "maj" not in label
    third = (r_pc + 3) % 12 if is_min else (r_pc + 4) % 12
    fifth = (r_pc + 7) % 12
    passing = (r_pc + 2) % 12
    candidates = [third, fifth, passing]
    if "7" in label:
        seventh = (r_pc + 11) % 12 if "maj7" in label else (r_pc + 10) % 12
        candidates.append(seventh)

    best_cand = max(candidates, key=lambda c: float(b_vec[c]))
    if float(b_vec[best_cand]) > min_energy_ratio * mean_e and float(b_vec[best_cand]) >= float(b_vec[r_pc]):
        frame_wins = float(np.mean(sub_bass[best_cand, :] > sub_bass[r_pc, :]))
        if frame_wins >= min_frame_ratio:
            return int(best_cand)

    return None


def estimate_key(
    chords: list[ChordSeg],
    notes: Optional[list[NoteSeg]] = None,
) -> str:
    """Estimate tonic key using Krumhansl-Schmuckler profiles over chord tones and melody.

    Returns the canonical tonic key name (e.g. 'C', 'F#', 'Bb').
    For minor keys, maps to relative major for movable-do representation.
    """
    pc_durations = np.zeros(12, dtype=float)

    for c in chords:
        if not c.label or c.label in ("N", "X") or c.end <= c.start:
            continue
        dur = c.end - c.start
        m = re.match(r"^([A-Ga-g][b#]?)", c.label)
        if not m:
            continue
        root_pc = KEY_NAME_TO_PC.get(m.group(1).upper())
        if root_pc is None:
            continue

        qual = c.label.split(":")[1] if ":" in c.label else ""
        is_min = qual.startswith("min") or (qual.startswith("m") and not qual.startswith("maj"))
        is_dim = "dim" in qual
        is_aug = "aug" in qual

        third_pc = (root_pc + (3 if (is_min or is_dim) else 4)) % 12
        fifth_pc = (root_pc + (6 if is_dim else (8 if is_aug else 7))) % 12

        pc_durations[root_pc] += dur * 1.5
        pc_durations[third_pc] += dur * 1.0
        pc_durations[fifth_pc] += dur * 1.0

        if "7" in qual:
            seventh_pc = (root_pc + (11 if "maj7" in qual else 10)) % 12
            pc_durations[seventh_pc] += dur * 0.5

    if notes:
        for n in notes:
            if n.midi > 0 and n.end > n.start:
                dur = n.end - n.start
                pc = int(round(n.midi)) % 12
                pc_durations[pc] += dur * 1.0

    if float(np.sum(pc_durations)) < 1e-4:
        return "C"

    best_corr = -999.0
    best_pc = 0

    for root in range(12):
        rot_maj = np.roll(KRUMHANSL_MAJOR, root)
        r_maj = float(np.corrcoef(pc_durations, rot_maj)[0, 1])
        if r_maj > best_corr:
            best_corr = r_maj
            best_pc = root

        rot_min = np.roll(KRUMHANSL_MINOR, root)
        r_min = float(np.corrcoef(pc_durations, rot_min)[0, 1])
        if r_min > best_corr:
            best_corr = r_min
            best_pc = (root + 3) % 12

    return canonical_key_for_pc(best_pc)


def _decode_baseline(
    probs: np.ndarray,
    hop_dur_val: float,
    grid_list: list[float],
    vocab_map: dict[int, str],
    chroma_bass: np.ndarray,
) -> list[ChordSeg]:
    """Reproduce exact Stage 3 baseline chord decoding."""
    n_frames = probs.shape[0] if probs.size > 0 else 0
    simp_vocab = [simplify_btc_chord(vocab_map.get(idx, "N")) for idx in range(170)]
    unique_labels = sorted(list(set(simp_vocab)))

    beat_chords: list[str] = []
    beat_confs: list[float] = []

    for k in range(len(grid_list) - 1):
        t0 = grid_list[k]
        t1 = grid_list[k + 1]
        f_start = max(0, int(round(t0 / hop_dur_val)))
        f_end = min(n_frames, max(f_start + 1, int(round(t1 / hop_dur_val))))

        if f_end > f_start and n_frames > 0:
            frame_dist = np.mean(probs[f_start:f_end, :], axis=0)
        elif n_frames > 0:
            frame_dist = probs[min(f_start, n_frames - 1), :]
        else:
            frame_dist = np.zeros(170)

        prob_per_label: dict[str, float] = {lbl: 0.0 for lbl in unique_labels}
        for c_idx in range(len(frame_dist)):
            prob_per_label[simp_vocab[c_idx]] += float(frame_dist[c_idx])

        best_lbl = max(prob_per_label.keys(), key=lambda l: prob_per_label[l])
        best_prob = prob_per_label[best_lbl]
        beat_chords.append(best_lbl)
        beat_confs.append(best_prob)

    smoothed_chords = smooth_beat_chords(beat_chords, beat_confs)
    chords: list[ChordSeg] = []

    if smoothed_chords:
        cur_start_idx = 0
        cur_label = smoothed_chords[0]

        for idx in range(1, len(smoothed_chords)):
            if smoothed_chords[idx] != cur_label:
                if cur_label not in ("N", "X"):
                    t_seg_start = grid_list[cur_start_idx]
                    t_seg_end = grid_list[idx]
                    seg_conf = float(np.mean(beat_confs[cur_start_idx:idx]))

                    bass_pc = None
                    m_root = re.match(r"^([A-Ga-g][b#]?)", cur_label)
                    if m_root and chroma_bass.shape[1] > 0:
                        r_pc = KEY_NAME_TO_PC.get(m_root.group(1).upper())
                        sf_start = max(0, int(round(t_seg_start / hop_dur_val)))
                        sf_end = min(chroma_bass.shape[1], max(sf_start + 1, int(round(t_seg_end / hop_dur_val))))
                        sub_bass = chroma_bass[:, sf_start:sf_end]
                        if sub_bass.shape[1] > 0:
                            b_vec = np.mean(sub_bass, axis=1)
                            b_cand = int(np.argmax(b_vec))
                            mean_e = float(np.mean(b_vec))
                            if mean_e > 1e-4 and b_vec[b_cand] > 1.35 * mean_e and b_cand != r_pc:
                                bass_pc = b_cand

                    chords.append(
                        ChordSeg(
                            start=round(t_seg_start, 2),
                            end=round(t_seg_end, 2),
                            label=cur_label,
                            bass_pc=bass_pc,
                            confidence=round(seg_conf, 3),
                        )
                    )
                cur_start_idx = idx
                cur_label = smoothed_chords[idx]

        if cur_label not in ("N", "X"):
            t_seg_start = grid_list[cur_start_idx]
            t_seg_end = grid_list[-1]
            seg_conf = float(np.mean(beat_confs[cur_start_idx:]))

            bass_pc = None
            m_root = re.match(r"^([A-Ga-g][b#]?)", cur_label)
            if m_root and chroma_bass.shape[1] > 0:
                r_pc = KEY_NAME_TO_PC.get(m_root.group(1).upper())
                sf_start = max(0, int(round(t_seg_start / hop_dur_val)))
                sf_end = min(chroma_bass.shape[1], max(sf_start + 1, int(round(t_seg_end / hop_dur_val))))
                sub_bass = chroma_bass[:, sf_start:sf_end]
                if sub_bass.shape[1] > 0:
                    b_vec = np.mean(sub_bass, axis=1)
                    b_cand = int(np.argmax(b_vec))
                    mean_e = float(np.mean(b_vec))
                    if mean_e > 1e-4 and b_vec[b_cand] > 1.35 * mean_e and b_cand != r_pc:
                        bass_pc = b_cand

            chords.append(
                ChordSeg(
                    start=round(t_seg_start, 2),
                    end=round(t_seg_end, 2),
                    label=cur_label,
                    bass_pc=bass_pc,
                    confidence=round(seg_conf, 3),
                )
            )

    return chords


def _decode_viterbi(
    probs: np.ndarray,
    hop_dur_val: float,
    grid_list: list[float],
    chroma_bass: np.ndarray,
    downbeats: Optional[list[float]] = None,
    key_hint: Optional[str] = None,
    pen_beat1: float = 0.5,
    pen_beat3: float = 2.0,
    pen_weak: float = 2.5,
    self_reward: float = 0.0,
    th_7: float = 0.32,
    th_min7: float = 0.30,
    diatonic_bonus: float = 0.15,
) -> list[ChordSeg]:
    """Beat-synchronous Viterbi HMM chord decoding."""
    n_frames = probs.shape[0] if probs.size > 0 else 0
    n_beats = len(grid_list) - 1
    if n_beats <= 0:
        return []

    log_obs = np.zeros((n_beats, NUM_STATES), dtype=float)
    conditional_7: list[dict[int, dict[str, float]]] = []

    for k in range(n_beats):
        t0 = grid_list[k]
        t1 = grid_list[k + 1]
        f0 = max(0, int(round(t0 / hop_dur_val)))
        f1 = min(n_frames, max(f0 + 1, int(round(t1 / hop_dur_val))))
        if f1 > f0 and n_frames > 0:
            dist = np.mean(probs[f0:f1, :], axis=0)
        elif n_frames > 0:
            dist = probs[min(f0, n_frames - 1), :]
        else:
            dist = np.zeros(170)

        beat_cond: dict[int, dict[str, float]] = {}
        for r_idx, r_name in enumerate(BTC_ROOT_LIST):
            base = r_idx * 14
            p_maj_triad = float(dist[base + 1] + dist[base + 5] + dist[base + 3])
            p_7 = float(dist[base + 9])
            p_maj7 = float(dist[base + 8])
            p_maj_tot = p_maj_triad + p_7 + p_maj7

            p_min_triad = float(dist[base + 0] + dist[base + 7])
            p_min7 = float(dist[base + 6] + dist[base + 4])
            p_min_tot = p_min_triad + p_min7

            p_dim = float(dist[base + 2] + dist[base + 10] + dist[base + 11])
            p_sus = float(dist[base + 12] + dist[base + 13])

            log_obs[k, STATE_TO_IDX[f"{r_name}:maj"]] = np.log(max(1e-7, p_maj_tot))
            log_obs[k, STATE_TO_IDX[f"{r_name}:min"]] = np.log(max(1e-7, p_min_tot))
            log_obs[k, STATE_TO_IDX[f"{r_name}:dim"]] = np.log(max(1e-7, p_dim))
            log_obs[k, STATE_TO_IDX[f"{r_name}:sus4"]] = np.log(max(1e-7, p_sus))

            beat_cond[r_idx] = {
                "maj_tot": p_maj_tot,
                "p_7": p_7,
                "p_maj7": p_maj7,
                "min_tot": p_min_tot,
                "p_min7": p_min7,
            }

        p_none = float(dist[168] + dist[169]) if len(dist) >= 170 else 1.0
        log_obs[k, STATE_TO_IDX["N"]] = np.log(max(1e-7, p_none))
        conditional_7.append(beat_cond)

    if key_hint and diatonic_bonus > 0.0:
        key_pc = KEY_NAME_TO_PC.get(key_hint.upper())
        if key_pc is not None:
            diatonic_chords = [
                (key_pc % 12, "maj"),
                ((key_pc + 2) % 12, "min"),
                ((key_pc + 4) % 12, "min"),
                ((key_pc + 5) % 12, "maj"),
                ((key_pc + 7) % 12, "maj"),
                ((key_pc + 9) % 12, "min"),
                ((key_pc + 11) % 12, "dim"),
            ]
            for r_pc, fam in diatonic_chords:
                r_name = BTC_ROOT_LIST[r_pc]
                s_idx = STATE_TO_IDX[f"{r_name}:{fam}"]
                log_obs[:, s_idx] += diatonic_bonus

    beat_phases = np.zeros(n_beats, dtype=int)
    db_arr = np.array(downbeats) if (downbeats and len(downbeats) > 1) else np.array([])
    for k in range(n_beats):
        t = grid_list[k]
        if len(db_arr) > 1:
            idx = int(np.searchsorted(db_arr, t, side="right")) - 1
            if 0 <= idx < len(db_arr) - 1:
                frac = (t - db_arr[idx]) / max(0.1, (db_arr[idx + 1] - db_arr[idx]))
                if frac < 0.15 or frac >= 0.85:
                    beat_phases[k] = 1
                elif 0.35 <= frac < 0.65:
                    beat_phases[k] = 3
                else:
                    beat_phases[k] = 2
            else:
                beat_phases[k] = (k % 4) + 1
        else:
            beat_phases[k] = (k % 4) + 1

    dp = np.zeros((n_beats, NUM_STATES), dtype=float)
    backpointers = np.zeros((n_beats, NUM_STATES), dtype=int)
    dp[0] = log_obs[0]

    base_trans = np.zeros((NUM_STATES, NUM_STATES), dtype=float)
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            if i != j:
                r_i = STATE_ROOT[STATES[i]]
                r_j = STATE_ROOT[STATES[j]]
                if r_i is not None and r_j is not None and r_i == r_j:
                    base_trans[i, j] = 0.8

    for k in range(1, n_beats):
        phase = beat_phases[k]
        pen = pen_beat1 if phase == 1 else (pen_beat3 if phase == 3 else pen_weak)
        trans = base_trans - pen
        np.fill_diagonal(trans, self_reward)

        cost_matrix = dp[k - 1, :, None] + trans
        dp[k] = np.max(cost_matrix, axis=0) + log_obs[k]
        backpointers[k] = np.argmax(cost_matrix, axis=0)

    path = np.zeros(n_beats, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for k in range(n_beats - 2, -1, -1):
        path[k] = backpointers[k + 1, path[k + 1]]

    beat_labels: list[str] = []
    beat_confs: list[float] = []
    for k in range(n_beats):
        s_idx = path[k]
        s_name = STATES[s_idx]
        conf = float(np.exp(min(0.0, log_obs[k, s_idx])))
        beat_confs.append(conf)

        if s_name == "N":
            beat_labels.append("N")
        else:
            r_idx = STATE_ROOT[s_name]
            if r_idx is None:
                beat_labels.append("N")
                continue
            r_name = BTC_ROOT_LIST[r_idx]
            fam = STATE_FAM[s_name]
            cond = conditional_7[k][r_idx]

            if fam == "maj":
                tot = max(1e-6, cond["maj_tot"])
                q_7 = cond["p_7"] / tot
                q_maj7 = cond["p_maj7"] / tot
                if q_maj7 >= th_7 and q_maj7 > q_7:
                    lbl = f"{r_name}:maj7"
                elif q_7 >= th_7:
                    lbl = f"{r_name}:7"
                else:
                    lbl = f"{r_name}:maj"
            elif fam == "min":
                tot = max(1e-6, cond["min_tot"])
                q_min7 = cond["p_min7"] / tot
                if q_min7 >= th_min7:
                    lbl = f"{r_name}:min7"
                else:
                    lbl = f"{r_name}:min"
            else:
                lbl = s_name
            beat_labels.append(lbl)

    segs: list[ChordSeg] = []
    if beat_labels:
        cur_start_idx = 0
        cur_lbl = beat_labels[0]
        for idx in range(1, len(beat_labels)):
            if beat_labels[idx] != cur_lbl:
                if cur_lbl not in ("N", "X"):
                    t_start = round(grid_list[cur_start_idx], 2)
                    t_end = round(grid_list[idx], 2)
                    b_pc = detect_segment_bass(cur_lbl, t_start, t_end, chroma_bass, hop_dur_val)
                    c_conf = round(float(np.mean(beat_confs[cur_start_idx:idx])), 3)
                    segs.append(ChordSeg(start=t_start, end=t_end, label=cur_lbl, bass_pc=b_pc, confidence=c_conf))
                cur_start_idx = idx
                cur_lbl = beat_labels[idx]
        if cur_lbl not in ("N", "X"):
            t_start = round(grid_list[cur_start_idx], 2)
            t_end = round(grid_list[-1], 2)
            b_pc = detect_segment_bass(cur_lbl, t_start, t_end, chroma_bass, hop_dur_val)
            c_conf = round(float(np.mean(beat_confs[cur_start_idx:])), 3)
            segs.append(ChordSeg(start=t_start, end=t_end, label=cur_lbl, bass_pc=b_pc, confidence=c_conf))

    return segs


def decode_chords(
    probs: np.ndarray,
    hop_dur: Any,
    grid: Any,
    vocab: Any,
    chroma_bass: np.ndarray,
    *,
    key_hint: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
) -> list[ChordSeg]:
    """Decode beat-synchronous chords from frame posteriors.

    Args:
        probs: (frames, 170) BTC chord posterior probabilities.
        hop_dur: Hop duration in seconds.
        grid: Beat grid boundaries in seconds [t_0, t_1, ...].
        vocab: BTC chord vocabulary mapping or list of chord names.
        chroma_bass: (12, frames) low-band chroma features for bass estimation.
        key_hint: Optional key hint for diatonic boosting.
        params: Optional parameter dictionary.

    Returns:
        List of ChordSeg instances with start, end, label, bass_pc, and confidence.
    """
    if probs is None or probs.size == 0 or len(grid) < 2:
        return []

    hop_dur_val = float(hop_dur[0]) if isinstance(hop_dur, (list, tuple, np.ndarray)) else float(hop_dur)
    grid_list = [float(b) for b in grid]

    if isinstance(vocab, dict):
        vocab_map = vocab
    elif isinstance(vocab, (list, tuple, np.ndarray)):
        vocab_map = {i: str(vocab[i]) for i in range(len(vocab))}
    else:
        vocab_map = {}

    opt = params or {}
    algorithm = opt.get("algorithm", "viterbi")

    if algorithm == "baseline":
        return _decode_baseline(probs, hop_dur_val, grid_list, vocab_map, chroma_bass)

    downbeats = opt.get("downbeats")
    pen_beat1 = float(opt.get("pen_beat1", 0.5))
    pen_beat3 = float(opt.get("pen_beat3", 2.0))
    pen_weak = float(opt.get("pen_weak", 2.5))
    self_reward = float(opt.get("self_reward", 0.0))
    th_7 = float(opt.get("th_7", 0.32))
    th_min7 = float(opt.get("th_min7", 0.30))
    diatonic_bonus = float(opt.get("diatonic_bonus", 0.15))

    initial_segs = _decode_viterbi(
        probs=probs,
        hop_dur_val=hop_dur_val,
        grid_list=grid_list,
        chroma_bass=chroma_bass,
        downbeats=downbeats,
        key_hint=key_hint,
        pen_beat1=pen_beat1,
        pen_beat3=pen_beat3,
        pen_weak=pen_weak,
        self_reward=self_reward,
        th_7=th_7,
        th_min7=th_min7,
        diatonic_bonus=diatonic_bonus if key_hint else 0.0,
    )

    if key_hint is None and diatonic_bonus > 0.0 and len(initial_segs) > 0:
        est_key = estimate_key(initial_segs)
        refined_segs = _decode_viterbi(
            probs=probs,
            hop_dur_val=hop_dur_val,
            grid_list=grid_list,
            chroma_bass=chroma_bass,
            downbeats=downbeats,
            key_hint=est_key,
            pen_beat1=pen_beat1,
            pen_beat3=pen_beat3,
            pen_weak=pen_weak,
            self_reward=self_reward,
            th_7=th_7,
            th_min7=th_min7,
            diatonic_bonus=diatonic_bonus,
        )
        return refined_segs

    return initial_segs
