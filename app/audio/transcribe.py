"""Audio transcription pipeline for LeadSheet draft generation.

Implements lazy imports so that importing app.audio or app.audio.transcribe never loads
torch or heavy ML libraries until transcribe() is explicitly invoked.
Runs 7 modular stages:
  1. Vocal separation (audio-separator / demucs fallback)
  2. Beat and downbeat tracking (beat-this / librosa fallback)
  3. Chord recognition (BTC-ISMIR19 large-voca + low-band bass chroma)
  4. Melody pitch extraction (torchcrepe on vocal stem -> note segmentation)
  5. Key detection (Krumhansl-Schmuckler correlation -> relative major)
  6. Lyrics transcription (faster-whisper on vocal stem + opencc simplification)
  7. Section segmentation (intro, verse, chorus, interlude, outro)
"""
from __future__ import annotations

import gc
import logging
import math
import os
import re
import shutil
import time
import urllib.request
from typing import Callable, Optional

from app.audio.analysis import (
    AudioAnalysis,
    ChordSeg,
    LyricChar,
    NoteSeg,
    SectionSeg,
)
from app.theory.keys import (
    DEGREE_SEMITONES,
    KEY_NAME_TO_PC,
    canonical_key_for_pc,
    key_name_to_pc,
)

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles
KRUMHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

BTC_ROOT_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BTC_QUALITY_LIST = [
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
    "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
]


def _get_btc_chord_vocab() -> dict[int, str]:
    """Return index-to-chord mapping for BTC-ISMIR19 170-class large vocabulary."""
    vocab: dict[int, str] = {168: "X", 169: "N"}
    for i in range(168):
        root = BTC_ROOT_LIST[i // 14]
        quality = BTC_QUALITY_LIST[i % 14]
        if i % 14 != 1:  # not plain major
            vocab[i] = f"{root}:{quality}"
        else:
            vocab[i] = f"{root}:maj"
    return vocab


def _ensure_btc_weights() -> str:
    """Ensure BTC-ISMIR19 large-voca weights exist on disk and return absolute path."""
    candidates = [
        os.environ.get("BTC_WEIGHTS_PATH"),
        "/app/models/btc_model_large_voca.pt",
        "/Users/jush/.gemini/jetski/brain/982974d3-46cb-4d49-ac77-56cc9b0c43e2/scratch/audio/btc_model_large_voca.pt",
        os.path.expanduser("~/.cache/btc_model_large_voca.pt"),
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.path.getsize(path) > 10_000_000:
            return path

    target = os.path.expanduser("~/.cache/btc_model_large_voca.pt")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    url = "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/btc_model_large_voca.pt"
    logger.info("Downloading BTC-ISMIR19 weights from %s to %s", url, target)
    urllib.request.urlretrieve(url, target)
    return target


def _build_btc_model():
    """Build BTCModel architecture in PyTorch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _gen_bias_mask(max_length):
        np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
        return torch.from_numpy(np_mask).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)

    def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales).astype(float) * -log_timescale_increment
        )
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant", constant_values=[0.0, 0.0])
        return torch.from_numpy(signal.reshape([1, length, channels])).type(torch.FloatTensor)

    import numpy as np

    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta

    class Conv(nn.Module):
        def __init__(self, input_size, output_size, kernel_size, pad_type):
            super().__init__()
            padding = (
                (kernel_size - 1, 0) if pad_type == "left" else (kernel_size // 2, (kernel_size - 1) // 2)
            )
            self.pad = nn.ConstantPad1d(padding, 0)
            self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

        def forward(self, inputs):
            inputs = self.pad(inputs.permute(0, 2, 1))
            return self.conv(inputs).permute(0, 2, 1)

    class PositionwiseFeedForward(nn.Module):
        def __init__(self, input_depth, filter_size, output_depth, layer_config="cc", padding="both", dropout=0.0):
            super().__init__()
            layers = []
            sizes = (
                [(input_depth, filter_size)]
                + [(filter_size, filter_size)] * (len(layer_config) - 2)
                + [(filter_size, output_depth)]
            )
            for lc, s in zip(list(layer_config), sizes):
                if lc == "l":
                    layers.append(nn.Linear(*s))
                elif lc == "c":
                    layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            self.layers = nn.ModuleList(layers)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, inputs):
            x = inputs
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = self.relu(x)
                    x = self.dropout(x)
            return x

    class MultiHeadAttention(nn.Module):
        def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, num_heads, bias_mask=None, dropout=0.0, attention_map=False):
            super().__init__()
            self.attention_map = attention_map
            self.num_heads = num_heads
            self.query_scale = (total_key_depth // num_heads) ** -0.5
            self.bias_mask = bias_mask
            self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
            self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
            self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
            self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
            self.dropout = nn.Dropout(dropout)

        def _split_heads(self, x):
            shape = x.shape
            return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

        def _merge_heads(self, x):
            shape = x.shape
            return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

        def forward(self, queries, keys, values):
            q = self.query_linear(queries)
            k = self.key_linear(keys)
            v = self.value_linear(values)
            qh = self._split_heads(q) * self.query_scale
            kh = self._split_heads(k)
            vh = self._split_heads(v)
            logits = torch.matmul(qh, kh.permute(0, 1, 3, 2))
            if self.bias_mask is not None:
                logits = logits + self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].to(logits.device)
            weights = self.dropout(F.softmax(logits, dim=-1))
            out = torch.matmul(weights, vh)
            return self.output_linear(self._merge_heads(out))

    class SelfAttentionBlock(nn.Module):
        def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
            super().__init__()
            self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, hidden_size, num_heads, bias_mask, attention_dropout)
            self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config="cc", padding="both", dropout=relu_dropout)
            self.dropout = nn.Dropout(layer_dropout)
            self.layer_norm_mha = LayerNorm(hidden_size)
            self.layer_norm_ffn = LayerNorm(hidden_size)

        def forward(self, inputs):
            x = inputs
            x_norm = self.layer_norm_mha(x)
            y = self.multi_head_attention(x_norm, x_norm, x_norm)
            x = self.dropout(x + y)
            x_norm = self.layer_norm_ffn(x)
            y = self.positionwise_convolution(x_norm)
            return self.dropout(x + y)

    class BiDirectionalSelfAttention(nn.Module):
        def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
            super().__init__()
            params = (hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, _gen_bias_mask(max_length), layer_dropout, attention_dropout, relu_dropout)
            self.attn_block = SelfAttentionBlock(*params)
            params_bwd = (hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3), layer_dropout, attention_dropout, relu_dropout)
            self.backward_attn_block = SelfAttentionBlock(*params_bwd)
            self.linear = nn.Linear(hidden_size * 2, hidden_size)

        def forward(self, inputs):
            x, lst = inputs
            enc = self.attn_block(x)
            rev = self.backward_attn_block(x)
            out = self.linear(torch.cat((enc, rev), dim=2))
            return out, lst

    class BiDirectionalSelfAttentionLayers(nn.Module):
        def __init__(self, feature_size=144, hidden_size=128, num_layers=8, num_heads=4, total_key_depth=128, total_value_depth=128, filter_size=128, max_length=108, input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
            super().__init__()
            self.timing_signal = _gen_timing_signal(max_length, hidden_size)
            params = (hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, num_heads, max_length, layer_dropout, attention_dropout, relu_dropout)
            self.embedding_proj = nn.Linear(feature_size, hidden_size, bias=False)
            self.self_attn_layers = nn.Sequential(*[BiDirectionalSelfAttention(*params) for _ in range(num_layers)])
            self.layer_norm = LayerNorm(hidden_size)
            self.input_dropout = nn.Dropout(input_dropout)

        def forward(self, inputs):
            x = self.input_dropout(inputs)
            x = self.embedding_proj(x)
            x = x + self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            y, _ = self.self_attn_layers((x, []))
            return self.layer_norm(y)

    class OutputLayer(nn.Module):
        def __init__(self, hidden_size, output_size, probs_out=False):
            super().__init__()
            self.output_size = output_size
            self.output_projection = nn.Linear(hidden_size, output_size)
            self.probs_out = probs_out
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=int(hidden_size / 2), batch_first=True, bidirectional=True)
            self.hidden_size = hidden_size

    class SoftmaxOutputLayer(OutputLayer):
        def forward(self, hidden):
            logits = self.output_projection(hidden)
            probs = F.softmax(logits, -1)
            topk, indices = torch.topk(probs, 2)
            predictions = indices[:, :, 0]
            return predictions, probs

    class BTCModel(nn.Module):
        def __init__(self, num_chords=170):
            super().__init__()
            self.self_attn_layers = BiDirectionalSelfAttentionLayers()
            self.output_layer = SoftmaxOutputLayer(hidden_size=128, output_size=num_chords)

        def forward(self, x):
            h = self.self_attn_layers(x)
            return self.output_layer(h)

    return BTCModel(num_chords=170)


def _simplify_btc_chord(label: str) -> str:
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


def _smooth_beat_chords(chords: list[str], confs: list[float]) -> list[str]:
    """Filter out isolated 1-beat chord glitches using 2-beat median/neighbor smoothing."""
    if len(chords) < 3:
        return list(chords)

    res = list(chords)
    n = len(res)

    # Pass 1: isolated 1-beat chords flanked by identical chords: [A, B, A] -> [A, A, A]
    for i in range(1, n - 1):
        if res[i - 1] == res[i + 1] and res[i] != res[i - 1]:
            res[i] = res[i - 1]

    # Pass 2: isolated 1-beat chord between two different chords: [A, B, C]
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

    # Final pass: clean any newly created sandwich glitches
    for i in range(1, n - 1):
        if res[i - 1] == res[i + 1] and res[i] != res[i - 1]:
            res[i] = res[i - 1]

    return res


def _detect_key_krumhansl(
    chords: list[ChordSeg],
    notes: list[NoteSeg],
    warnings: list[str],
) -> str:
    """Detect key using duration-weighted chord tones + melody pitch classes.

    Combines Krumhansl-Schmuckler profiles with:
    - Tonic chord duration bonus (favoring the home I chord)
    - First and final stable chord bonus
    - Non-diatonic chord penalty (penalizing candidates where frequent chords are non-diatonic,
      such as IV appearing as bVII in the dominant key)
    Always maps to major-mode tonic for movable-do Jianpu representation.
    """
    import numpy as np

    hist = np.zeros(12, dtype=float)

    # 1. Melody notes duration weighting
    for note in notes:
        if note.midi > 0 and note.end > note.start:
            dur = note.end - note.start
            pc = int(round(note.midi)) % 12
            hist[pc] += dur * 2.0

    # 2. Chords duration weighting with full chord tones (root, 3rd, 5th, 7th)
    for c in chords:
        if c.label and c.label not in ("N", "X") and c.end > c.start:
            dur = c.end - c.start
            m = re.match(r"^([A-Ga-g][b#]?)", c.label)
            if m:
                root_name = m.group(1).upper()
                root_pc = KEY_NAME_TO_PC.get(root_name)
                if root_pc is not None:
                    qual = c.label.split(":")[1] if ":" in c.label else ""
                    is_min = qual.startswith("min") or (qual.startswith("m") and not qual.startswith("maj"))
                    is_dim = "dim" in qual
                    is_aug = "aug" in qual

                    third_pc = (root_pc + (3 if (is_min or is_dim) else 4)) % 12
                    fifth_pc = (root_pc + (6 if is_dim else (8 if is_aug else 7))) % 12

                    hist[root_pc] += dur * 1.5
                    hist[third_pc] += dur * 1.0
                    hist[fifth_pc] += dur * 1.0

                    if "7" in qual:
                        seventh_pc = (root_pc + (11 if "maj7" in qual else 10)) % 12
                        hist[seventh_pc] += dur * 0.5

    if np.sum(hist) < 1e-4:
        warnings.append("音频音高信息不足，默认设置为 C 调")
        return "C"

    total_chord_dur = sum(c.end - c.start for c in chords if c.label not in ("N", "X"))
    first_stable = next((c for c in chords if (c.end - c.start >= 0.8) and c.label not in ("N", "X")), None)
    last_stable = next((c for c in reversed(chords) if (c.end - c.start >= 0.8) and c.label not in ("N", "X")), None)

    # Standard major triad diatonic qualities for degrees 0, 2, 4, 5, 7, 9, 11
    # 0(I): maj, 2(ii): min, 4(iii): min, 5(IV): maj, 7(V): maj, 9(vi): min, 11(vii°): dim
    diatonic_degrees = {0: False, 2: True, 4: True, 5: False, 7: False, 9: True, 11: True}

    best_key_pc = 0
    best_score = -float("inf")

    # Evaluate each of the 12 candidate major keys
    for tonic in range(12):
        rotated_profile = np.roll(KRUMHANSL_MAJOR, tonic)
        corr = float(np.corrcoef(hist, rotated_profile)[0, 1])
        if np.isnan(corr):
            corr = 0.0

        dur_I = 0.0
        dur_nondiatonic = 0.0

        for c in chords:
            if c.label in ("N", "X") or c.end <= c.start:
                continue
            dur = c.end - c.start
            m = re.match(r"^([A-Ga-g][b#]?)", c.label)
            if not m:
                continue
            r_name = m.group(1).upper()
            r_pc = KEY_NAME_TO_PC.get(r_name)
            if r_pc is None:
                continue

            qual = c.label.split(":")[1] if ":" in c.label else ""
            is_min = qual.startswith("min") or (qual.startswith("m") and not qual.startswith("maj"))
            deg = (r_pc - tonic) % 12

            if deg == 0 and not is_min:
                dur_I += dur

            if deg not in diatonic_degrees:
                # Root is completely non-diatonic in key (e.g. bVII, bII, bV)
                dur_nondiatonic += dur * 1.5
            else:
                expected_min = diatonic_degrees[deg]
                if expected_min != is_min and deg != 11:
                    dur_nondiatonic += dur * 0.5

        # Bonuses
        bonus_I = 0.40 * (dur_I / max(1e-3, total_chord_dur))
        if first_stable:
            m_f = re.match(r"^([A-Ga-g][b#]?)", first_stable.label)
            if m_f and KEY_NAME_TO_PC.get(m_f.group(1).upper()) == tonic and ("min" not in first_stable.label):
                bonus_I += 0.15
        if last_stable:
            m_l = re.match(r"^([A-Ga-g][b#]?)", last_stable.label)
            if m_l and KEY_NAME_TO_PC.get(m_l.group(1).upper()) == tonic and ("min" not in last_stable.label):
                bonus_I += 0.20

        # Penalty for non-diatonic chords
        penalty = 1.25 * (dur_nondiatonic / max(1e-3, total_chord_dur))
        total_score = corr + bonus_I - penalty

        if total_score > best_score:
            best_score = total_score
            best_key_pc = tonic

    return canonical_key_for_pc(best_key_pc)


LYRIC_BLACKLIST = [
    "字幕",
    "字幕by",
    "索兰娅",
    "订阅",
    "点赞",
    "谢谢观看",
    "感谢收看",
    "请关注",
    "提供",
    "制作",
    "翻唱",
    "原唱",
    "伴奏",
    "amara",
    "zitherharp",
    "youtube",
    "bilibili",
    "transcribed by",
    "captioned by",
]


def _has_repeated_ngram(text: str) -> bool:
    """Check if text contains repeated n-grams (e.g. ABCABCABC or XX XX XX)."""
    clean = re.sub(r"\s+", "", text)
    if not clean:
        return False
    if re.search(r"(.{2,8})\1{2,}", clean):
        return True
    return False


def filter_lyrics_segments(
    segments,
    notes: list[NoteSeg],
    language: str = "zh",
    cc=None,
) -> list[LyricChar]:
    """Filter Whisper segments to eliminate hallucinations and extract aligned LyricChars.

    Applies:
      1. Confidence & threshold filtering (avg_logprob >= -1.0, no_speech_prob <= 0.5, compression_ratio <= 2.4)
      2. Repetition & blacklist rejection (_has_repeated_ngram, LYRIC_BLACKLIST)
      3. Language/character set filtering (CJK characters for zh)
      4. Melody note overlap gating: char interval [c_start, c_end] must overlap a voiced note
         [n.start - 0.15, n.end + 0.15] from Stage 4 note segmentation.
    """
    lyrics: list[LyricChar] = []
    if not segments:
        return lyrics

    for seg in segments:
        try:
            avg_logprob = float(seg.avg_logprob)
        except AttributeError:
            avg_logprob = 0.0
        try:
            no_speech_prob = float(seg.no_speech_prob)
        except AttributeError:
            no_speech_prob = 0.0
        try:
            compression_ratio = float(seg.compression_ratio)
        except AttributeError:
            compression_ratio = 1.0
        try:
            seg_text = str(seg.text or "")
        except AttributeError:
            seg_text = ""

        # 1. Metric thresholds
        if avg_logprob < -1.0 or no_speech_prob > 0.5 or compression_ratio > 2.4:
            continue

        # 2. Repeated n-gram check
        if _has_repeated_ngram(seg_text):
            continue

        # 3. Blacklist check (case-insensitive)
        text_lower = seg_text.lower()
        if any(b.lower() in text_lower for b in LYRIC_BLACKLIST):
            continue

        # 4. Extract words/chars
        try:
            words = seg.words
        except AttributeError:
            words = None

        if words:
            for w in words:
                try:
                    raw_w = str(w.word or "")
                except AttributeError:
                    raw_w = ""
                w_text = cc.convert(raw_w).strip() if cc else raw_w.strip()
                if language == "zh":
                    chars = [c for c in w_text if re.match(r"[\u4e00-\u9fff]", c)]
                else:
                    chars = [c for c in w_text if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", c)]
                if not chars:
                    continue

                try:
                    w_start = float(w.start)
                except AttributeError:
                    w_start = 0.0
                try:
                    w_end = float(w.end)
                except AttributeError:
                    w_end = w_start + 0.2
                char_dur = max(0.05, (w_end - w_start) / len(chars))

                for c_idx, char in enumerate(chars):
                    c_start = w_start + c_idx * char_dur
                    c_end = min(w_end, c_start + char_dur)

                    # Gating against melody notes: must overlap [n.start - 0.15, n.end + 0.15]
                    overlaps = False
                    for n in notes:
                        if max(c_start, n.start - 0.15) < min(c_end, n.end + 0.15):
                            overlaps = True
                            break
                    if not overlaps:
                        continue

                    lyrics.append(
                        LyricChar(
                            start=round(c_start, 2),
                            end=round(c_end, 2),
                            text=char,
                        )
                    )
        else:
            try:
                seg_start = float(seg.start)
            except AttributeError:
                seg_start = 0.0
            try:
                seg_end = float(seg.end)
            except AttributeError:
                seg_end = seg_start + 0.5
            raw_t = cc.convert(seg_text).strip() if cc else seg_text.strip()
            if language == "zh":
                chars = [c for c in raw_t if re.match(r"[\u4e00-\u9fff]", c)]
            else:
                chars = [c for c in raw_t if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", c)]
            if not chars:
                continue
            char_dur = max(0.05, (seg_end - seg_start) / len(chars))
            for c_idx, char in enumerate(chars):
                c_start = seg_start + c_idx * char_dur
                c_end = min(seg_end, c_start + char_dur)
                overlaps = False
                for n in notes:
                    if max(c_start, n.start - 0.15) < min(c_end, n.end + 0.15):
                        overlaps = True
                        break
                if not overlaps:
                    continue
                lyrics.append(
                    LyricChar(
                        start=round(c_start, 2),
                        end=round(c_end, 2),
                        text=char,
                    )
                )

    return lyrics


def _detect_sections(
    downbeats: list[float],
    duration: float,
    notes: list[NoteSeg],
    chords: list[ChordSeg],
) -> list[SectionSeg]:
    """Segment song into musical sections: 前奏, 主歌, 副歌, 间奏, 桥段, 尾奏.

    Uses bar-level vocal activity to locate instrumental regions (前奏, 间奏, 尾奏),
    and pitch register / structural positioning to distinguish 主歌 (verse), 副歌 (chorus),
    and 桥段 (bridge).
    """
    import numpy as np

    if duration <= 0:
        return []

    valid_db = sorted([d for d in downbeats if 0.0 <= d < duration])
    if not valid_db:
        valid_db = [0.0]
    elif valid_db[0] > 1.0:
        valid_db = [0.0] + valid_db

    bar_starts = valid_db
    bar_ends = valid_db[1:] + [duration]
    n_bars = len(bar_starts)

    if n_bars == 0:
        return [SectionSeg(start=0.0, end=round(duration, 2), label="主歌")]

    # 1. Bar-level vocal activity and pitch
    is_vocal = [False] * n_bars
    bar_pitches: list[Optional[float]] = [None] * n_bars

    for b in range(n_bars):
        t0 = bar_starts[b]
        t1 = bar_ends[b]
        b_dur = t1 - t0

        b_notes = [n for n in notes if n.end > t0 and n.start < t1]
        if b_notes and b_dur > 0:
            total_voc = 0.0
            weighted_pitch_sum = 0.0
            for n in b_notes:
                ov = min(t1, n.end) - max(t0, n.start)
                if ov > 0:
                    total_voc += ov
                    weighted_pitch_sum += n.midi * ov
            if total_voc >= max(0.35, 0.20 * b_dur):
                is_vocal[b] = True
                bar_pitches[b] = weighted_pitch_sum / total_voc

    vocal_bars = [i for i, v in enumerate(is_vocal) if v]
    if not vocal_bars:
        return [SectionSeg(start=0.0, end=round(duration, 2), label="前奏")]

    first_vocal_bar = vocal_bars[0]
    last_vocal_bar = vocal_bars[-1]

    all_vocal_pitches = [bar_pitches[i] for i in vocal_bars if bar_pitches[i] is not None]
    p_min = float(np.min(all_vocal_pitches)) if all_vocal_pitches else 60.0
    p_max = float(np.max(all_vocal_pitches)) if all_vocal_pitches else 72.0
    p_median = float(np.median(all_vocal_pitches)) if all_vocal_pitches else 64.0
    chorus_thresh = (p_min + p_max) / 2.0 if (p_max - p_min >= 3.0) else p_median

    bar_labels: list[str] = [""] * n_bars

    # 2. Intro (前奏)
    for b in range(0, first_vocal_bar):
        bar_labels[b] = "前奏"

    # 3. Outro (尾奏)
    for b in range(last_vocal_bar + 1, n_bars):
        bar_labels[b] = "尾奏"

    # 4. Middle unvoiced bars (间奏)
    b = first_vocal_bar
    while b <= last_vocal_bar:
        if not is_vocal[b]:
            gap_start = b
            while b <= last_vocal_bar and not is_vocal[b]:
                b += 1
            gap_len = b - gap_start
            gap_dur = bar_ends[b - 1] - bar_starts[gap_start]
            if gap_len >= 2 or gap_dur >= 3.0:
                for k in range(gap_start, b):
                    bar_labels[k] = "间奏"
        else:
            b += 1

    # 5. Classify voiced bars (主歌 vs 副歌) by register.
    for b in range(first_vocal_bar, last_vocal_bar + 1):
        if bar_labels[b] != "":
            continue
        if bar_pitches[b] is None:
            prev_p = next((bar_pitches[k] for k in range(b - 1, -1, -1) if bar_pitches[k] is not None), p_median)
            next_p = next((bar_pitches[k] for k in range(b + 1, n_bars) if bar_pitches[k] is not None), p_median)
            bar_pitches[b] = (prev_p + next_p) / 2.0
        bar_labels[b] = "副歌" if bar_pitches[b] >= chorus_thresh else "主歌"

    # 6. Bridge (桥段) is not inferred from audio alone: a verse after an interlude is far more
    #    often a second verse than a bridge. The web cross-check may supply such labels.

    # 7. Smoothing: a 主歌/副歌 run shorter than MIN_SECTION_BARS (e.g. one high bar inside a verse)
    #    is absorbed into the preceding voiced section (or the following one at a region start).
    MIN_SECTION_BARS = 3
    voiced = ("主歌", "副歌")
    changed = True
    while changed:
        changed = False
        b = 0
        while b < n_bars:
            if bar_labels[b] not in voiced:
                b += 1
                continue
            r0 = b
            while b < n_bars and bar_labels[b] == bar_labels[r0]:
                b += 1
            if b - r0 >= MIN_SECTION_BARS:
                continue
            prev_lab = bar_labels[r0 - 1] if r0 > 0 else ""
            next_lab = bar_labels[b] if b < n_bars else ""
            new_lab = prev_lab if prev_lab in voiced else (next_lab if next_lab in voiced else "")
            if new_lab and new_lab != bar_labels[r0]:
                for k in range(r0, b):
                    bar_labels[k] = new_lab
                changed = True

    # 8. Merge contiguous bars with identical labels into SectionSeg
    sections: list[SectionSeg] = []
    cur_label = bar_labels[0]
    cur_start = bar_starts[0]

    for b in range(1, n_bars):
        if bar_labels[b] != cur_label:
            sections.append(
                SectionSeg(
                    start=round(cur_start, 2),
                    end=round(bar_starts[b], 2),
                    label=cur_label,
                )
            )
            cur_start = bar_starts[b]
            cur_label = bar_labels[b]

    sections.append(
        SectionSeg(
            start=round(cur_start, 2),
            end=round(duration, 2),
            label=cur_label,
        )
    )

    return sections


def transcribe(
    audio_path: str,
    work_dir: str,
    progress: Optional[Callable[[float, str], None]] = None,
) -> AudioAnalysis:
    """Run full audio transcription pipeline and return AudioAnalysis.

    Each stage is wrapped in try/except to guarantee that intermediate errors
    yield warnings and graceful fallbacks rather than a hard crash.
    """
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(work_dir, exist_ok=True)
    warnings: list[str] = []

    def report(pct: float, stage_text: str):
        logger.info("[transcribe %d%%] %s", int(pct * 100), stage_text)
        if progress:
            progress(pct, stage_text)

    # -------------------------------------------------------------------------
    # Stage 0: Audio verification & basic info
    # -------------------------------------------------------------------------
    report(0.05, "加载音频文件")
    try:
        info = sf.info(audio_path)
        duration = float(info.duration)
        sr = info.samplerate
    except Exception as exc:
        logger.warning("soundfile.info failed: %s; fallback librosa", exc)
        import librosa
        duration = float(librosa.get_duration(path=audio_path))
        sr = 22050

    vocals_path = os.path.join(work_dir, "vocals.wav")
    acc_path = os.path.join(work_dir, "accompaniment.wav")

    stage_timings: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Stage 1: Vocal Separation (audio-separator ONNX)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.10, "分离人声与伴奏")
    sep_success = False
    try:
        from audio_separator.separator import Separator

        model_dir = os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", os.path.join(work_dir, "models"))
        os.makedirs(model_dir, exist_ok=True)
        sep = Separator(model_file_dir=model_dir, output_dir=work_dir, output_format="WAV")
        sep.load_model("UVR-MDX-NET-Voc_FT.onnx")
        outputs = sep.separate(audio_path)

        for fname in outputs:
            fpath = os.path.join(work_dir, fname)
            if "Vocals" in fname:
                shutil.move(fpath, vocals_path)
            elif "Instrumental" in fname:
                shutil.move(fpath, acc_path)

        if os.path.isfile(vocals_path) and os.path.isfile(acc_path):
            sep_success = True
    except Exception as exc:
        logger.warning("audio-separator failed: %s", exc)

    if not sep_success:
        warnings.append("人声分离未能完成，使用原音频作为后备")
        shutil.copy2(audio_path, vocals_path)
        shutil.copy2(audio_path, acc_path)
    stage_timings["vocal_separation"] = round(time.perf_counter() - t_stage, 2)


    # -------------------------------------------------------------------------
    # Stage 2: Beats & Downbeats (beat-this / librosa fallback)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.35, "检测节拍与小节线")
    beats: list[float] = []
    downbeats: list[float] = []
    tempo_bpm: Optional[float] = None

    try:
        from beat_this.inference import File2Beats

        f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        raw_beats, raw_downbeats = f2b(audio_path)
        beats = [float(b) for b in raw_beats]
        downbeats = [float(d) for d in raw_downbeats]

        if len(beats) > 1:
            diffs = np.diff(beats)
            median_interval = float(np.median(diffs))
            if median_interval > 0.05:
                tempo_bpm = round(60.0 / median_interval, 1)
    except Exception as exc:
        logger.warning("beat-this failed: %s. Falling back to librosa.", exc)
        warnings.append("beat-this 失败，降级为 Librosa 节拍检测")
        try:
            import librosa
            y, sr_b = librosa.load(audio_path, sr=22050, mono=True)
            tempo, b_frames = librosa.beat.beat_track(y=y, sr=sr_b)
            tempo_bpm = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            b_times = librosa.frames_to_time(b_frames, sr=sr_b)
            beats = [float(t) for t in b_times]
            downbeats = [beats[i] for i in range(0, len(beats), 4)]
        except Exception as l_exc:
            logger.warning("librosa beat_track also failed: %s", l_exc)
            tempo_bpm = 120.0
            sec_per_beat = 0.5
            beats = [float(t) for t in np.arange(0, duration, sec_per_beat)]
            downbeats = [beats[i] for i in range(0, len(beats), 4)]
    stage_timings["beat_detection"] = round(time.perf_counter() - t_stage, 2)


    # -------------------------------------------------------------------------
    # Stage 3: Chords (BTC-ISMIR19 + Low-band Chroma Bass for Slash Chords)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.50, "识别和弦与低音转位")
    chords: list[ChordSeg] = []
    try:
        import librosa

        btc_weights_file = _ensure_btc_weights()
        btc_model = _build_btc_model()
        ckpt = torch.load(btc_weights_file, map_location="cpu", weights_only=False)
        btc_model.load_state_dict(ckpt["model"], strict=True)
        btc_model.eval()

        mean_val = float(ckpt["mean"])
        std_val = float(ckpt["std"])
        vocab = _get_btc_chord_vocab()

        # Load accompaniment (or audio_path) at 22050Hz
        chord_target_audio = acc_path if os.path.isfile(acc_path) else audio_path
        y_chord, _ = librosa.load(chord_target_audio, sr=22050, mono=True)

        # 1. CQT features for BTC
        cqt = librosa.cqt(y_chord, sr=22050, n_bins=144, bins_per_octave=24, hop_length=2048)
        feat = np.log(np.abs(cqt) + 1e-6).T
        feat = (feat - mean_val) / std_val

        # 2. Low-band chroma for bass pitch class (32Hz to 260Hz, C1 to C4)
        chroma_bass = librosa.feature.chroma_cqt(
            y=y_chord, sr=22050, fmin=librosa.note_to_hz("C1"), n_octaves=3, hop_length=2048
        )

        n_frames = feat.shape[0]
        hop_dur = 2048.0 / 22050.0

        all_probs_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n_frames, 108):
                chunk = feat[i : i + 108]
                pad_len = 108 - chunk.shape[0]
                if pad_len > 0:
                    chunk = np.pad(chunk, ((0, pad_len), (0, 0)), "constant")
                x = torch.from_numpy(chunk).float().unsqueeze(0)
                _, prob = btc_model(x)
                valid_len = min(n_frames - i, 108)
                all_probs_list.append(prob.squeeze(0)[:valid_len].cpu().numpy())

        if all_probs_list:
            all_probs = np.concatenate(all_probs_list, axis=0)
        else:
            all_probs = np.zeros((0, 170), dtype=float)

        # 3. Beat-synchronous chord decoding using Stage 2 beats
        valid_beats = sorted([b for b in beats if 0.0 <= b <= duration])
        grid: list[float] = []
        if not valid_beats or valid_beats[0] > 0.25:
            grid.append(0.0)
        for b in valid_beats:
            if not grid or (b - grid[-1]) >= 0.15:
                grid.append(b)
        if not grid:
            grid = [0.0, duration]
        elif duration - grid[-1] > 0.25:
            grid.append(duration)
        else:
            grid[-1] = duration

        simp_vocab = [_simplify_btc_chord(vocab.get(idx, "N")) for idx in range(170)]
        unique_labels = sorted(list(set(simp_vocab)))

        # Aggregate frame posteriors over each beat interval
        beat_chords: list[str] = []
        beat_confs: list[float] = []

        for k in range(len(grid) - 1):
            t0 = grid[k]
            t1 = grid[k + 1]
            f_start = max(0, int(round(t0 / hop_dur)))
            f_end = min(n_frames, max(f_start + 1, int(round(t1 / hop_dur))))

            if f_end > f_start and n_frames > 0:
                frame_dist = np.mean(all_probs[f_start:f_end, :], axis=0)
            elif n_frames > 0:
                frame_dist = all_probs[min(f_start, n_frames - 1), :]
            else:
                frame_dist = np.zeros(170)

            prob_per_label: dict[str, float] = {lbl: 0.0 for lbl in unique_labels}
            for c_idx in range(len(frame_dist)):
                prob_per_label[simp_vocab[c_idx]] += float(frame_dist[c_idx])

            best_lbl = max(prob_per_label.keys(), key=lambda l: prob_per_label[l])
            best_prob = prob_per_label[best_lbl]
            beat_chords.append(best_lbl)
            beat_confs.append(best_prob)

        # Apply 2-beat smoothing / transient filter to remove isolated 1-beat glitches
        smoothed_chords = _smooth_beat_chords(beat_chords, beat_confs)

        # Group contiguous identical chords into ChordSeg
        if smoothed_chords:
            cur_start_idx = 0
            cur_label = smoothed_chords[0]

            for idx in range(1, len(smoothed_chords)):
                if smoothed_chords[idx] != cur_label:
                    if cur_label not in ("N", "X"):
                        t_seg_start = grid[cur_start_idx]
                        t_seg_end = grid[idx]
                        seg_conf = float(np.mean(beat_confs[cur_start_idx:idx]))

                        bass_pc = None
                        m_root = re.match(r"^([A-Ga-g][b#]?)", cur_label)
                        if m_root and chroma_bass.shape[1] > 0:
                            r_pc = KEY_NAME_TO_PC.get(m_root.group(1).upper())
                            sf_start = max(0, int(round(t_seg_start / hop_dur)))
                            sf_end = min(chroma_bass.shape[1], max(sf_start + 1, int(round(t_seg_end / hop_dur))))
                            sub_bass = chroma_bass[:, sf_start:sf_end]
                            if sub_bass.shape[1] > 0:
                                b_vec = np.mean(sub_bass, axis=1)
                                b_cand = int(np.argmax(b_vec))
                                mean_e = np.mean(b_vec)
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

            # Last segment
            if cur_label not in ("N", "X"):
                t_seg_start = grid[cur_start_idx]
                t_seg_end = grid[-1]
                seg_conf = float(np.mean(beat_confs[cur_start_idx:]))

                bass_pc = None
                m_root = re.match(r"^([A-Ga-g][b#]?)", cur_label)
                if m_root and chroma_bass.shape[1] > 0:
                    r_pc = KEY_NAME_TO_PC.get(m_root.group(1).upper())
                    sf_start = max(0, int(round(t_seg_start / hop_dur)))
                    sf_end = min(chroma_bass.shape[1], max(sf_start + 1, int(round(t_seg_end / hop_dur))))
                    sub_bass = chroma_bass[:, sf_start:sf_end]
                    if sub_bass.shape[1] > 0:
                        b_vec = np.mean(sub_bass, axis=1)
                        b_cand = int(np.argmax(b_vec))
                        mean_e = np.mean(b_vec)
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

        del btc_model
        del ckpt
        gc.collect()
    except Exception as exc:
        logger.warning("Chord recognition failed: %s", exc)
        warnings.append(f"和弦识别异常: {exc}")
    stage_timings["chord_recognition"] = round(time.perf_counter() - t_stage, 2)

    # -------------------------------------------------------------------------
    # Stage 4: Melody Pitch Extraction (torchcrepe)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.65, "提取人声音高与音符切分")

    notes: list[NoteSeg] = []
    try:
        import torchcrepe
        from scipy.ndimage import median_filter

        voc_audio, v_sr = sf.read(vocals_path)
        if voc_audio.ndim > 1:
            voc_audio = np.mean(voc_audio, axis=1)

        hop_length = int(v_sr * 0.01)  # 10ms frame
        hop_dur = hop_length / float(v_sr)
        voc_t = torch.from_numpy(voc_audio).float().unsqueeze(0)

        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(
                voc_t,
                v_sr,
                hop_length=hop_length,
                model="tiny",
                device="cpu",
                return_periodicity=True,
                batch_size=2048,
            )

        pitch_arr = pitch.squeeze(0).numpy()
        per_arr = periodicity.squeeze(0).numpy()

        # Voicing detection with hysteresis
        is_voiced = np.zeros(len(pitch_arr), dtype=bool)
        state = False
        for t_idx in range(len(pitch_arr)):
            p_val = per_arr[t_idx]
            hz_val = pitch_arr[t_idx]
            if not state:
                if p_val >= 0.35 and 65.0 <= hz_val <= 1100.0:
                    state = True
            else:
                if p_val < 0.20 or hz_val < 60.0 or hz_val > 1200.0:
                    state = False
            is_voiced[t_idx] = state

        # Convert Hz to MIDI pitches for voiced frames
        midi_arr = np.zeros(len(pitch_arr), dtype=float)
        v_idx = np.where(is_voiced)[0]
        if len(v_idx) > 0:
            midi_arr[v_idx] = 69.0 + 12.0 * np.log2(np.maximum(1.0, pitch_arr[v_idx]) / 440.0)
            # Median smoothing
            midi_arr[v_idx] = median_filter(midi_arr[v_idx], size=5)

        # Note segmentation
        in_note = False
        note_start_idx = 0
        current_note_midis: list[float] = []

        for idx in range(len(pitch_arr)):
            if is_voiced[idx]:
                cur_m = midi_arr[idx]
                if not in_note:
                    in_note = True
                    note_start_idx = idx
                    current_note_midis = [cur_m]
                else:
                    ref_m = np.median(current_note_midis[-5:]) if len(current_note_midis) >= 5 else current_note_midis[0]
                    # Split on sustained pitch jump > 0.7 semitones
                    if abs(cur_m - ref_m) > 0.7 and len(current_note_midis) >= 8:
                        # Finalize previous note if length >= 0.08s
                        n_dur = (idx - note_start_idx) * hop_dur
                        if n_dur >= 0.08:
                            notes.append(
                                NoteSeg(
                                    start=round(note_start_idx * hop_dur, 2),
                                    end=round(idx * hop_dur, 2),
                                    midi=round(float(np.median(current_note_midis)), 1),
                                    confidence=round(float(np.mean(per_arr[note_start_idx:idx])), 3),
                                )
                            )
                        note_start_idx = idx
                        current_note_midis = [cur_m]
                    else:
                        current_note_midis.append(cur_m)
            else:
                if in_note:
                    in_note = False
                    n_dur = (idx - note_start_idx) * hop_dur
                    if n_dur >= 0.08:
                        notes.append(
                            NoteSeg(
                                start=round(note_start_idx * hop_dur, 2),
                                end=round(idx * hop_dur, 2),
                                midi=round(float(np.median(current_note_midis)), 1),
                                confidence=round(float(np.mean(per_arr[note_start_idx:idx])), 3),
                            )
                        )
                    current_note_midis = []

        # Trailing note
        if in_note and len(current_note_midis) >= 8:
            notes.append(
                NoteSeg(
                    start=round(note_start_idx * hop_dur, 2),
                    end=round(len(pitch_arr) * hop_dur, 2),
                    midi=round(float(np.median(current_note_midis)), 1),
                    confidence=round(float(np.mean(per_arr[note_start_idx:])), 3),
                )
            )

        del pitch
        del periodicity
        gc.collect()
    except Exception as exc:
        logger.warning("Melody pitch extraction failed: %s", exc)
        warnings.append(f"旋律提取异常: {exc}")
    stage_timings["melody_extraction"] = round(time.perf_counter() - t_stage, 2)

    # -------------------------------------------------------------------------
    # Stage 5: Key Detection (Krumhansl-Schmuckler profiles)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.80, "推断调号")
    detected_key = _detect_key_krumhansl(chords, notes, warnings)
    stage_timings["key_detection"] = round(time.perf_counter() - t_stage, 2)

    # -------------------------------------------------------------------------
    # Stage 6: Lyrics Transcription (faster-whisper + opencc)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.88, "转写歌词并对齐字时间戳")
    lyrics: list[LyricChar] = []
    try:
        from faster_whisper import WhisperModel
        import opencc

        w_model_name = os.environ.get("WHISPER_MODEL", "small")
        whisper = WhisperModel(w_model_name, device="cpu", compute_type="int8")
        cc = opencc.OpenCC("t2s")

        segments, _ = whisper.transcribe(
            vocals_path,
            language="zh",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
        )

        lyrics = filter_lyrics_segments(list(segments), notes, language="zh", cc=cc)

    except Exception as exc:
        logger.warning("Lyrics transcription failed: %s", exc)
        warnings.append(f"歌词识别异常: {exc}")
    stage_timings["lyrics_transcription"] = round(time.perf_counter() - t_stage, 2)

    # -------------------------------------------------------------------------
    # Stage 7: Sections (intro, verse, chorus, interlude, bridge, outro)
    # -------------------------------------------------------------------------
    t_stage = time.perf_counter()
    report(0.96, "分析曲式结构")
    sections: list[SectionSeg] = []
    try:
        sections = _detect_sections(downbeats, duration, notes, chords)
    except Exception as exc:
        logger.warning("Section segmentation failed: %s", exc)
        warnings.append(f"曲式结构分析异常: {exc}")
    stage_timings["section_analysis"] = round(time.perf_counter() - t_stage, 2)

    report(1.0, "音频转录完成")

    return AudioAnalysis(
        duration=round(duration, 2),
        tempo_bpm=tempo_bpm,
        beats=beats,
        downbeats=downbeats,
        time_signature="4/4",
        key=detected_key,
        chords=chords,
        notes=notes,
        lyrics=lyrics,
        sections=sections,
        stage_timings=stage_timings,
        warnings=warnings,
    )

