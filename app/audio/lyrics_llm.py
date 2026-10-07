"""Gemini-based lyric correction and alignment for audio transcription.

Sends the separated vocal stem as inline audio to Vertex AI Gemini
along with the Whisper draft hint, corrects homophones/misrecognitions,
and aligns the corrected text back to character-level timestamps using
Needleman-Wunsch edit distance and melody note onsets.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Optional

from app.audio.analysis import LyricChar, NoteSeg

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "cellular-cider-495602-r9"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 60.0
DRAFT_LESS_CONFIDENCE_THRESHOLD = 0.60

GENERIC_HALLUCINATION_PATTERNS: list[str] = [
    "字幕", "谢谢观看", "感谢观看", "谢谢收看", "感谢收看",
    "请勿转载", "提供字幕", "听写", "打轴", "微信", "公众号",
    "关注", "点赞", "投币", "收藏", "转发", "南无", "阿弥陀佛",
    "www.", "http", ".com", ".cn", ".org", "bilibili", "youtube",
]


def has_repeated_pattern(text: str) -> bool:
    """Check if text contains repeated n-grams or excessively low character diversity."""
    clean = re.sub(r"[\s_\r\n\t,.\?!，。？！、…—\"'“”‘’（）\(\)]", "", text)
    if not clean:
        return False

    # 1. Any 1-8 char n-gram repeated 3+ times (e.g. 啦啦啦, ABCABCABC)
    if re.search(r"(.{1,8})\1{2,}", clean):
        return True

    # 2. Entire text composed of 2+ exact repetitions of a multi-char phrase (e.g. 南无阿弥陀佛南无阿弥陀佛)
    if re.fullmatch(r"(.{2,10})\1+", clean):
        return True

    # 3. Low character diversity for long phrases (e.g. length >= 6 with < 35% unique chars)
    if len(clean) >= 6 and (len(set(clean)) / len(clean)) < 0.35:
        return True

    return False


def is_char_count_implausible(char_count: int, num_notes: int) -> bool:
    """Check if character count is implausible vs the number of melody notes in the span."""
    if num_notes >= 6:
        if char_count > (1.5 * num_notes + 2) or char_count < (0.3 * num_notes):
            return True
    elif num_notes > 0:
        if char_count > (2.5 * num_notes + 4):
            return True
    return False


def contains_generic_hallucination(text: str) -> bool:
    """Check if text matches well-known speech-model or subtitle hallucination patterns."""
    text_lower = text.lower()
    for pattern in GENERIC_HALLUCINATION_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    return False


def _dedupe_monotonic(chars: list[LyricChar]) -> list[LyricChar]:
    """Drop draft chars re-added by overlapping lines and enforce monotonic starts."""
    out: list[LyricChar] = []
    seen: set[int] = set()
    for c in chars:
        if id(c) in seen:
            continue
        if out and c.start < out[-1].start:
            continue
        seen.add(id(c))
        out.append(c)
    return out


def to_simplified_chinese(text: str) -> str:
    """Convert Chinese text to Simplified Chinese (opencc t2s if available, else zhconv)."""
    try:
        import opencc
        cc = opencc.OpenCC("t2s")
        return cc.convert(text)
    except Exception:
        pass
    try:
        import zhconv
        return zhconv.convert(text, "zh-cn")
    except Exception:
        pass
    return text


def clean_lyrics_text(text: str) -> str:
    """Remove spaces, punctuation, underscores from lyrics for comparison."""
    text = re.sub(r"[\s_\r\n\t,.\?!，。？！、…—\"'“”‘’（）\(\)]", "", text)
    return text.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate character-level Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def prepare_vocal_audio_payload(vocals_path: str) -> tuple[bytes, str]:
    """Convert vocal stem to 16 kHz mono FLAC or MP3 ~64kbps and return (bytes, mime_type).

    If ffmpeg is available on PATH, encodes a compact 16 kHz mono audio payload.
    Otherwise falls back to reading the raw file bytes directly.
    """
    if not os.path.isfile(vocals_path):
        raise FileNotFoundError(f"Vocal file not found: {vocals_path}")

    # If file is already small FLAC (< 12MB), read directly
    if vocals_path.endswith(".flac"):
        try:
            raw = Path(vocals_path).read_bytes()
            if len(raw) < 12 * 1024 * 1024:
                return raw, "audio/flac"
        except Exception:
            pass

    # Try ffmpeg encoding to 16kHz mono FLAC
    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_out:
        tmp_path = tmp_out.name
    try:
        res = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", vocals_path, "-ac", "1", "-ar", "16000",
                "-c:a", "flac", tmp_path,
            ],
            capture_output=True,
            timeout=20,
        )
        if res.returncode == 0 and os.path.isfile(tmp_path):
            data = Path(tmp_path).read_bytes()
            if 0 < len(data) < 15 * 1024 * 1024:
                return data, "audio/flac"
    except Exception:
        pass
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Fallback to direct read
    ext = Path(vocals_path).suffix.lower()
    mime = "audio/flac" if ext == ".flac" else ("audio/mp3" if ext in (".mp3", ".mpeg") else "audio/wav")
    return Path(vocals_path).read_bytes(), mime


def salvage_lyrics_json(text: str) -> list[dict[str, Any]]:
    """Tolerantly parse JSON, salvaging lines array if malformed or truncated."""
    if not text or not isinstance(text, str):
        return []
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("\n", 1)[0]
    cleaned = cleaned.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            res = []
            for x in data:
                if isinstance(x, dict):
                    res.append({
                        "start": float(x.get("start", 0.0)),
                        "end": float(x.get("end", 0.0)),
                        "text": str(x.get("text", "")),
                        "confidence": float(x.get("confidence", 0.8)),
                    })
            if res:
                return res
        if isinstance(data, dict):
            for key in ("lines", "lyrics"):
                if key in data and isinstance(data[key], list):
                    res = []
                    for x in data[key]:
                        if isinstance(x, dict):
                            res.append({
                                "start": float(x.get("start", 0.0)),
                                "end": float(x.get("end", 0.0)),
                                "text": str(x.get("text", "")),
                                "confidence": float(x.get("confidence", 0.8)),
                            })
                    if res:
                        return res
    except Exception:
        pass

    # 2. Regex salvage of line objects {"start": ..., "end": ..., "text": ...}
    lines: list[dict[str, Any]] = []
    pattern = re.compile(
        r'\{[^{}]*?"(?:start)"\s*:\s*([0-9.]+)[^{}]*?"(?:end)"\s*:\s*([0-9.]+)[^{}]*?"(?:text)"\s*:\s*"([^"]+)"(?:[^{}]*?"(?:confidence)"\s*:\s*([0-9.]+))?[^{}]*?\}',
        re.DOTALL,
    )
    for m in pattern.finditer(cleaned):
        try:
            s = float(m.group(1))
            e = float(m.group(2))
            t = m.group(3)
            c = float(m.group(4)) if m.group(4) else 0.8
            lines.append({"start": s, "end": e, "text": t, "confidence": c})
        except Exception:
            continue
    if lines:
        return lines

    # Try reverse pattern {"text": ..., "start": ..., "end": ...}
    pattern_alt = re.compile(
        r'\{[^{}]*?"(?:text)"\s*:\s*"([^"]+)"[^{}]*?"(?:start)"\s*:\s*([0-9.]+)[^{}]*?"(?:end)"\s*:\s*([0-9.]+)(?:[^{}]*?"(?:confidence)"\s*:\s*([0-9.]+))?[^{}]*?\}',
        re.DOTALL,
    )
    for m in pattern_alt.finditer(cleaned):
        try:
            t = m.group(1)
            s = float(m.group(2))
            e = float(m.group(3))
            c = float(m.group(4)) if m.group(4) else 0.8
            lines.append({"start": s, "end": e, "text": t, "confidence": c})
        except Exception:
            continue

    return lines


def align_line_chars(
    draft_chars: list[LyricChar],
    target_text: str,
    line_start: float,
    line_end: float,
    notes: Optional[list[NoteSeg]] = None,
) -> list[LyricChar]:
    """Align draft chars with target text characters using Needleman-Wunsch edit distance.

    Rules:
      1. Matched and substituted characters keep draft timestamps.
      2. Deleted draft characters are dropped.
      3. Inserted characters are interpolated within their surrounding anchor interval,
         using melody note onsets within that interval when available.
      4. Output is guaranteed to be strictly monotonic and rounded to 2 decimals.
    """
    notes = notes or []
    # Filter target_text to Chinese characters only
    clean_target = [c for c in target_text if re.match(r"[\u4e00-\u9fff]", c)]
    if not clean_target:
        return []

    m = len(draft_chars)
    n = len(clean_target)

    # If draft is empty, interpolate across line_start to line_end
    if m == 0:
        dur = max(0.05, (line_end - line_start) / n)
        note_onsets = sorted([note.start for note in notes if line_start - 0.1 <= note.start <= line_end + 0.1])
        res: list[LyricChar] = []
        if len(note_onsets) >= n:
            for i, ch in enumerate(clean_target):
                s = note_onsets[i]
                e = note_onsets[i + 1] if i + 1 < n else line_end
                if e <= s:
                    e = s + max(0.05, dur)
                res.append(LyricChar(start=round(s, 2), end=round(e, 2), text=ch))
        else:
            for i, ch in enumerate(clean_target):
                s = line_start + i * dur
                e = min(line_end, s + dur)
                res.append(LyricChar(start=round(s, 2), end=round(e, 2), text=ch))
        return res

    # Needleman-Wunsch DP matrix
    # dp[i][j] where i is draft_idx (0..m), j is target_idx (0..n)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = -i
    for j in range(n + 1):
        dp[0][j] = -j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = 2 if draft_chars[i - 1].text == clean_target[j - 1] else -1
            score_diag = dp[i - 1][j - 1] + match_score
            score_del = dp[i - 1][j] - 1
            score_ins = dp[i][j - 1] - 1
            dp[i][j] = max(score_diag, score_del, score_ins)

    # Backtracking
    i, j = m, n
    aligned: list[tuple[Optional[int], int]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = 2 if draft_chars[i - 1].text == clean_target[j - 1] else -1
            if dp[i][j] == dp[i - 1][j - 1] + match_score:
                aligned.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if j > 0 and dp[i][j] == dp[i][j - 1] - 1:
            aligned.append((None, j - 1))
            j -= 1
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] - 1:
            i -= 1
            continue
    aligned.reverse()

    target_assigned: list[dict[str, Any]] = [{"start": None, "end": None, "text": ch} for ch in clean_target]
    for d_idx, t_idx in aligned:
        if d_idx is not None:
            target_assigned[t_idx]["start"] = draft_chars[d_idx].start
            target_assigned[t_idx]["end"] = draft_chars[d_idx].end

    # Fill unassigned target characters (insertions)
    idx = 0
    note_onsets = sorted([note.start for note in notes])
    while idx < n:
        if target_assigned[idx]["start"] is not None:
            idx += 1
            continue
        run_start = idx
        while idx < n and target_assigned[idx]["start"] is None:
            idx += 1
        run_end = idx
        k_count = run_end - run_start

        left_t = line_start if run_start == 0 else target_assigned[run_start - 1]["end"]
        right_t = line_end if run_end == n else target_assigned[run_end]["start"]

        if right_t <= left_t:
            if run_start > 0:
                prev_s = target_assigned[run_start - 1]["start"]
                prev_e = target_assigned[run_start - 1]["end"]
                mid = prev_s + 0.6 * max(0.05, prev_e - prev_s)
                target_assigned[run_start - 1]["end"] = round(mid, 2)
                left_t = mid
            if run_end < n:
                next_s = target_assigned[run_end]["start"]
                next_e = target_assigned[run_end]["end"]
                mid = next_s + 0.4 * max(0.05, next_e - next_s)
                target_assigned[run_end]["start"] = round(mid, 2)
                right_t = mid

        span = max(0.05 * k_count, right_t - left_t)
        local_notes = [t for t in note_onsets if left_t - 0.05 <= t <= right_t + 0.05]
        if len(local_notes) >= k_count:
            for k in range(k_count):
                s = max(left_t, local_notes[k])
                e = local_notes[k + 1] if k + 1 < k_count else right_t
                if e <= s:
                    e = s + (right_t - s) / (k_count - k)
                target_assigned[run_start + k]["start"] = round(s, 2)
                target_assigned[run_start + k]["end"] = round(max(s + 0.04, e), 2)
        else:
            dt = span / k_count
            for k in range(k_count):
                s = left_t + k * dt
                e = left_t + (k + 1) * dt
                target_assigned[run_start + k]["start"] = round(s, 2)
                target_assigned[run_start + k]["end"] = round(e, 2)

    # Monotonicity adjustment
    out_chars: list[LyricChar] = []
    for k in range(n):
        s = float(target_assigned[k]["start"])
        e = float(target_assigned[k]["end"])
        if e <= s:
            e = round(s + 0.05, 2)
        if k > 0 and s < out_chars[-1].start:
            s = out_chars[-1].start
            e = max(e, round(s + 0.05, 2))
        out_chars.append(LyricChar(start=round(s, 2), end=round(e, 2), text=target_assigned[k]["text"]))

    return out_chars


def _generate_gemini_content(
    client: Any,
    model: str,
    contents: list[Any],
    response_mime_type: str = "application/json",
) -> Any:
    """Invoke Gemini model generate_content with JSON response type."""
    from google.genai import types

    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type=response_mime_type,
            temperature=0.0,
        ),
    )


def correct_lyrics(
    vocals_path: str,
    draft: list[LyricChar],
    notes: list[NoteSeg],
    *,
    language: str = "zh",
    title: str = "",
    artist: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[list[LyricChar], list[str]]:
    """Correct Whisper draft lyrics using Gemini on Vertex AI and align back to timing.

    Guarantees:
      - Never raises: intermediate errors gracefully return draft unchanged with warnings.
      - Non-zh gate: returns draft immediately for non-zh languages.
      - Hallucination guard: rejects unvoiced lines or lines wildly deviating from confident draft.
      - Output: strictly monotonic LyricChar list with simplified Chinese characters.
    """
    warnings: list[str] = []

    # 1. Gate for language
    if language != "zh":
        return draft, warnings

    # 2. Gate for vocals path
    if not vocals_path or not os.path.isfile(vocals_path):
        warnings.append(f"Vocal stem audio not found: {vocals_path}")
        return draft, warnings

    try:
        from google import genai
        from google.genai import types

        # Read environment configuration
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
        location = os.environ.get("SMART_LYRICS_LOCATION", DEFAULT_LOCATION)
        model = os.environ.get("SMART_LYRICS_MODEL", DEFAULT_MODEL)
        use_title = os.environ.get("SMART_LYRICS_USE_TITLE", "0") == "1"

        # Prepare audio payload
        vocals_bytes, mime_type = prepare_vocal_audio_payload(vocals_path)
        audio_part = types.Part.from_bytes(data=vocals_bytes, mime_type=mime_type)

        # Build prompt
        if draft:
            # Group draft chars into lines for readable prompt hint
            draft_lines = []
            cur_line = [draft[0]]
            for c in draft[1:]:
                if c.start - cur_line[-1].end > 1.2:
                    draft_lines.append(cur_line)
                    cur_line = [c]
                else:
                    cur_line.append(c)
            if cur_line:
                draft_lines.append(cur_line)

            draft_desc = []
            for line in draft_lines:
                s = line[0].start
                e = line[-1].end
                txt = "".join(x.text for x in line)
                draft_desc.append(f"[{s:.2f}s - {e:.2f}s]: {txt}")
            draft_str = "\n".join(draft_desc)
        else:
            draft_str = "(No speech-to-text draft available)"

        prompt_parts = [
            "You are an expert audio lyrics transcriber and corrector.",
            "Listen to the singing vocals in the audio track carefully.",
            "Transcribe the exact sung lyrics in Simplified Chinese.",
        ]
        if use_title and (title or artist):
            prompt_parts.append(f"Song Info: {title} - {artist}")
        prompt_parts.extend([
            "\nSpeech-to-text draft (contains speech recognition errors, homophones, or noise):",
            draft_str,
            "\nInstructions:",
            "1. Correct homophones, wrong words, and missing characters according to what is sung in the audio.",
            "2. Output the lyrics in Simplified Chinese as a JSON object with a list of lines.",
            "3. Each line must have:",
            "   - 'start': float seconds",
            "   - 'end': float seconds",
            "   - 'text': simplified Chinese string",
            "   - 'confidence': float between 0.0 and 1.0 indicating confidence in transcription.",
            "4. Only transcribe what is actually sung. Do not hallucinate or recite un-sung lyrics.",
            "\nJSON format:",
            '{\n  "lines": [\n    {"start": 0.0, "end": 3.5, "text": "...", "confidence": 0.95}\n  ]\n}',
        ])
        full_prompt = "\n".join(prompt_parts)

        client = genai.Client(vertexai=True, project=project, location=location)

        # Call with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                _generate_gemini_content,
                client,
                model,
                [audio_part, full_prompt],
                "application/json",
            )
            resp = fut.result(timeout=timeout_s)

        # Check response and filter reasons
        if not resp or not resp.candidates:
            warnings.append("Gemini returned no candidates")
            return draft, warnings

        cand = resp.candidates[0]
        finish_reason = None
        try:
            finish_reason = cand.finish_reason
        except AttributeError:
            pass
        if finish_reason and str(finish_reason).upper() in ("RECITATION", "SAFETY", "BLOCKLIST"):
            warnings.append(f"Gemini stopped due to filter: {finish_reason}")
            return draft, warnings

        raw_text = resp.text or ""
        if not raw_text.strip():
            warnings.append("Gemini returned empty text")
            return draft, warnings

        candidate_lines = salvage_lyrics_json(raw_text)
        if not candidate_lines:
            warnings.append(f"Failed to parse JSON lines from Gemini response: {raw_text[:200]}")
            return draft, warnings

        # Process and align candidate lines
        all_corrected_chars: list[LyricChar] = []
        prev_cleaned_text = ""

        for line in candidate_lines:
            l_start = float(line.get("start", 0.0))
            l_end = float(line.get("end", l_start + 1.0))
            l_raw_text = str(line.get("text", "")).strip()
            l_text = to_simplified_chinese(l_raw_text)
            l_conf = float(line.get("confidence", 0.8))
            clean_l = clean_lyrics_text(l_text)

            # Check vocal presence in [l_start - 0.3, l_end + 0.3]
            overlapping_draft = [
                c for c in draft
                if max(c.start, l_start - 0.3) < min(c.end, l_end + 0.3)
            ]
            overlapping_notes = [
                n for n in notes
                if max(n.start, l_start - 0.3) < min(n.end, l_end + 0.3)
            ]

            # Guardrail 1: Discard line if no vocal activity (no draft and no notes)
            if not overlapping_draft and not overlapping_notes:
                logger.info("Dropping hallucinated line '%s' [%.2f, %.2f]: no vocal activity", l_text, l_start, l_end)
                warnings.append(f"Dropped silent line '{l_text}'")
                continue

            # Guardrail 2: Repeated n-grams or low character diversity
            if has_repeated_pattern(l_text):
                logger.info("Dropping line with repeated pattern: '%s'", l_text)
                warnings.append(f"Dropped repetitive pattern line '{l_text}'")
                if overlapping_draft:
                    all_corrected_chars.extend(overlapping_draft)
                continue

            # Guardrail 3: Generic subtitle / hallucination patterns (unless present in draft)
            draft_str_local = "".join(c.text for c in overlapping_draft)
            if contains_generic_hallucination(l_text) and not contains_generic_hallucination(draft_str_local):
                logger.info("Dropping line with hallucination blacklist pattern: '%s'", l_text)
                warnings.append(f"Dropped generic hallucination line '{l_text}'")
                if overlapping_draft:
                    all_corrected_chars.extend(overlapping_draft)
                continue

            # Guardrail 4: Implausible char count vs melody notes
            num_chinese_chars = len([c for c in l_text if re.match(r"[\u4e00-\u9fff]", c)])
            if is_char_count_implausible(num_chinese_chars, len(overlapping_notes)):
                logger.info(
                    "Dropping rhythmically implausible line '%s' (%d chars vs %d notes)",
                    l_text, num_chinese_chars, len(overlapping_notes),
                )
                warnings.append(f"Dropped implausible char-to-note line '{l_text}'")
                if overlapping_draft:
                    all_corrected_chars.extend(overlapping_draft)
                continue

            # Guardrail 5: Stronger evidence required when draft is EMPTY
            if not overlapping_draft:
                # 5a. Minimum confidence threshold
                if l_conf < DRAFT_LESS_CONFIDENCE_THRESHOLD:
                    logger.info(
                        "Dropping draft-less line '%s' due to low confidence (%.2f < %.2f)",
                        l_text, l_conf, DRAFT_LESS_CONFIDENCE_THRESHOLD,
                    )
                    warnings.append(f"Dropped low-confidence draft-less line '{l_text}'")
                    continue
                # 5b. Reject duplicate lines across multiple lines
                if clean_l and clean_l == prev_cleaned_text:
                    logger.info("Dropping duplicate draft-less line: '%s'", l_text)
                    warnings.append(f"Dropped duplicate draft-less line '{l_text}'")
                    continue

            # Guardrail 6: Low similarity against confident draft
            if len(draft_str_local) >= 5:
                dist = levenshtein_distance(clean_lyrics_text(draft_str_local), clean_l)
                sim = 1.0 - dist / max(len(clean_lyrics_text(draft_str_local)), len(clean_l))
                if sim < 0.15:
                    logger.info("Candidate line '%s' rejected (sim=%.2f < 0.15 vs confident draft '%s')", l_text, sim, draft_str_local)
                    warnings.append(f"Rejected candidate line '{l_text}' (low similarity vs draft)")
                    # Preserve draft characters for this span
                    all_corrected_chars.extend(overlapping_draft)
                    prev_cleaned_text = clean_l
                    continue

            prev_cleaned_text = clean_l

            # Effective time span
            span_start = l_start
            span_end = l_end
            if overlapping_draft:
                span_start = min(span_start, overlapping_draft[0].start)
                span_end = max(span_end, overlapping_draft[-1].end)

            aligned = align_line_chars(
                overlapping_draft,
                l_text,
                span_start,
                span_end,
                notes=overlapping_notes,
            )
            all_corrected_chars.extend(aligned)

        if not all_corrected_chars and draft:
            warnings.append("All candidate lines filtered out; falling back to draft")
            return draft, warnings

        return _dedupe_monotonic(all_corrected_chars), warnings

    except concurrent.futures.TimeoutError:
        warnings.append(f"Gemini lyrics correction timed out after {timeout_s}s")
        return draft, warnings
    except Exception as exc:
        logger.warning("Gemini lyrics correction encountered exception: %s", exc)
        warnings.append(f"Gemini lyrics correction error: {exc}")
        return draft, warnings
