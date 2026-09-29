"""Web evidence corroboration for music sheet chord charts.

When OMR reading is ambiguous at some measures, web evidence provides independent
corroboration from published chord charts (guitar/piano tabs, 吉他谱/和弦谱).
This is EVIDENCE ONLY — it never overrides a clear reading and never gets redistributed.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models import ChordSymbol, ParsedSheet
from app.theory.chords import clean_raw_chord, parse_chord, parse_letter_chord
from app.theory.keys import (
    DEGREE_SEMITONES,
    KEY_NAME_TO_PC,
    key_name_to_pc,
    normalize_key_string,
)

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "cellular-cider-495602-r9"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
HARD_TIMEOUT_SECONDS = float(os.environ.get("WEB_EVIDENCE_TIMEOUT", "25.0"))
ALIGNMENT_QUALITY_THRESHOLD = 0.30

# Mapping from semitone offset (0..11) to default diatonic degree (1..7) and accidental (-1, 0, 1)
# in major scale. Chromatic degrees: b2, b3, #4, b6, b7
SEMITONE_TO_DEGREE: dict[int, tuple[int, int]] = {
    0: (1, 0),
    1: (2, -1),
    2: (2, 0),
    3: (3, -1),
    4: (3, 0),
    5: (4, 0),
    6: (4, 1),
    7: (5, 0),
    8: (6, -1),
    9: (6, 0),
    10: (7, -1),
    11: (7, 0),
}


# --------------------------------------------------------------------------- Data Models
class ReferenceSource(BaseModel):
    """A single published chord chart source from the web."""

    url: str
    key: Optional[str] = None
    capo: int = 0
    lines: list[list[str]] = Field(default_factory=list)
    section_labels: list[str] = Field(default_factory=list)


class ReferenceChart(BaseModel):
    """Aggregated reference chord charts for a song."""

    title: str
    artist: Optional[str] = None
    sources: list[ReferenceSource] = Field(default_factory=list)


class RefCandidate(BaseModel):
    """A reference chord candidate for a sheet measure."""

    chord: str = Field(description="Taiwanese number notation, e.g. '1', '5/7', '6m'")
    support: float = Field(description="Support score in [0.0, 1.0]")
    sources: int = Field(1, description="Number of independent web sources supporting this chord")
    beat: float = Field(1.0, description="1-based beat inside the measure")


@dataclass(frozen=True)
class DegreeChord:
    """Canonical representation of a chord relative to tonic."""

    degree: int  # 1..7
    accidental: int  # -1, 0, 1
    quality: str  # normalized quality, e.g. "maj", "m", "7", "m7", "add9"
    bass_degree: int  # 1..7
    bass_accidental: int  # -1, 0, 1
    root_semitone: int  # 0..11 relative to tonic
    bass_semitone: int  # 0..11 relative to tonic
    number_str: str  # canonical Taiwanese number chord string, e.g. "1", "5/7", "6m"


# --------------------------------------------------------------------------- Caching
def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", name.strip())
    return re.sub(r"_+", "_", cleaned)


def get_cache_path(title: str, artist: Optional[str] = None) -> Path:
    base_dir = Path("out/web/cache")
    t_clean = _sanitize_filename(title)
    a_clean = _sanitize_filename(artist or "unknown")
    return base_dir / f"{t_clean}__{a_clean}.json"


def _load_from_gcs(cache_key: str) -> Optional[str]:
    bucket_name = os.environ.get("SHEET_BUCKET")
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"web_cache/{cache_key}")
        if blob.exists():
            return blob.download_as_text()
    except Exception as exc:
        logger.debug("Failed reading from GCS bucket %s: %s", bucket_name, exc)
    return None


def _save_to_gcs(cache_key: str, content: str) -> None:
    bucket_name = os.environ.get("SHEET_BUCKET")
    if not bucket_name:
        return
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"web_cache/{cache_key}")
        blob.upload_from_string(content, content_type="application/json")
    except Exception as exc:
        logger.debug("Failed writing to GCS bucket %s: %s", bucket_name, exc)


def load_cached_reference(title: str, artist: Optional[str] = None) -> Optional[ReferenceChart]:
    """Load reference chart from local cache or GCS bucket if present."""
    local_path = get_cache_path(title, artist)
    if local_path.is_file():
        try:
            data = local_path.read_text(encoding="utf-8")
            return ReferenceChart.model_validate_json(data)
        except Exception as exc:
            logger.warning("Error reading local cache %s: %s", local_path, exc)

    # Check GCS
    cache_key = local_path.name
    gcs_data = _load_from_gcs(cache_key)
    if gcs_data:
        try:
            chart = ReferenceChart.model_validate_json(gcs_data)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(gcs_data, encoding="utf-8")
            return chart
        except Exception as exc:
            logger.warning("Error validating GCS cache for %s: %s", cache_key, exc)

    return None


def save_cached_reference(ref: ReferenceChart) -> None:
    """Save reference chart to local cache and GCS."""
    local_path = get_cache_path(ref.title, ref.artist)
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        content = ref.model_dump_json(indent=2)
        local_path.write_text(content, encoding="utf-8")
        _save_to_gcs(local_path.name, content)
    except Exception as exc:
        logger.warning("Failed saving cache for %s: %s", ref.title, exc)


# --------------------------------------------------------------------------- Theory & Transposition
def format_degree_to_number(root_semitone: int, quality: str, bass_semitone: int) -> str:
    """Format scale degrees (root & bass relative to tonic) into Taiwanese number notation."""
    deg, acc = SEMITONE_TO_DEGREE[root_semitone % 12]
    prefix = "b" if acc == -1 else ("#" if acc == 1 else "")
    root_str = f"{prefix}{deg}"

    if quality == "maj":
        q_suf = ""
    elif quality == "m":
        q_suf = "m"
    elif quality == "7":
        q_suf = "7"
    elif quality == "maj7":
        q_suf = "maj7"
    elif quality == "m7":
        q_suf = "m7"
    elif quality == "m7b5":
        q_suf = "m7-5"
    elif quality == "dim":
        q_suf = "dim"
    elif quality == "aug":
        q_suf = "aug"
    elif quality == "add9":
        q_suf = "(2)"
    elif quality == "sus4":
        q_suf = "sus4"
    elif quality == "7sus4":
        q_suf = "7sus"
    else:
        q_suf = quality

    root_chord = f"{root_str}{q_suf}"

    if (bass_semitone % 12) != (root_semitone % 12):
        b_deg, b_acc = SEMITONE_TO_DEGREE[bass_semitone % 12]
        b_prefix = "b" if b_acc == -1 else ("#" if b_acc == 1 else "")
        return f"{root_chord}/{b_prefix}{b_deg}"

    return root_chord


def parse_chord_to_degree(
    raw_chord: str,
    tonic_pc: int = 0,
    *,
    is_letter_hint: bool = False,
) -> Optional[DegreeChord]:
    """Parse any number or letter chord into a canonical DegreeChord relative to tonic."""
    raw = clean_raw_chord(raw_chord)
    if not raw:
        return None

    # Check if number notation
    is_num = bool(re.match(r"^[b#]?[1-7]", raw)) and not is_letter_hint
    if is_num:
        try:
            spec = parse_chord(raw)
            rst = (DEGREE_SEMITONES[spec.degree - 1] + spec.accidental) % 12
            if spec.bass_degree is not None:
                bst = (
                    DEGREE_SEMITONES[spec.bass_degree - 1]
                    + (spec.bass_accidental if spec.bass_accidental is not None else 0)
                ) % 12
                b_deg = spec.bass_degree
                b_acc = spec.bass_accidental if spec.bass_accidental is not None else 0
            else:
                bst = rst
                b_deg = spec.degree
                b_acc = spec.accidental

            num_str = format_degree_to_number(rst, spec.quality, bst)
            return DegreeChord(
                degree=spec.degree,
                accidental=spec.accidental,
                quality=spec.quality,
                bass_degree=b_deg,
                bass_accidental=b_acc,
                root_semitone=rst,
                bass_semitone=bst,
                number_str=num_str,
            )
        except Exception:
            pass

    # Letter notation
    try:
        lspec = parse_letter_chord(raw)
        rst = (lspec.root_pc - tonic_pc) % 12
        bst = (lspec.bass_pc - tonic_pc) % 12 if lspec.bass_pc is not None else rst

        r_deg, r_acc = SEMITONE_TO_DEGREE[rst]
        b_deg, b_acc = SEMITONE_TO_DEGREE[bst]
        num_str = format_degree_to_number(rst, lspec.quality, bst)

        return DegreeChord(
            degree=r_deg,
            accidental=r_acc,
            quality=lspec.quality,
            bass_degree=b_deg,
            bass_accidental=b_acc,
            root_semitone=rst,
            bass_semitone=bst,
            number_str=num_str,
        )
    except Exception as exc:
        logger.debug("Failed parsing chord token %s: %s", raw_chord, exc)
        return None


def determine_reference_tonic_pc(source: ReferenceSource, default_pc: int = 0) -> int:
    """Determine the tonic pitch class of the printed chords in a ReferenceSource.

    Handles capo, '选调' (play key) vs '原调' (original key), and falls back to
    chord root frequency analysis.
    """
    key_str = source.key or ""
    capo = source.capo or 0

    candidate_tonics: list[int] = []

    # 1. Parse '选调' (play key fingering) first if explicitly marked
    m_play = re.search(r"选调\s*[:：]?\s*([A-Ga-g][b#]?)", key_str)
    if m_play:
        try:
            candidate_tonics.append(key_name_to_pc(m_play.group(1)))
        except Exception:
            pass

    # 2. Parse general or '原调' key
    m_orig = re.search(r"(?:原调\s*[:：]?\s*)?([A-Ga-g][b#]?)", key_str)
    if m_orig:
        try:
            k_pc = key_name_to_pc(m_orig.group(1))
            candidate_tonics.append(k_pc)
            if capo > 0:
                candidate_tonics.append((k_pc - capo) % 12)
                candidate_tonics.append((k_pc + capo) % 12)
        except Exception:
            pass

    # Extract all letter chord roots from source lines
    chord_roots: list[int] = []
    for line in source.lines:
        for tok in line:
            m_let = re.match(r"^([A-Ga-g])([b#♭♯]?)", tok.strip())
            if m_let:
                name = m_let.group(1).upper() + (
                    "b" if m_let.group(2) in ("b", "♭") else ("#" if m_let.group(2) in ("#", "♯") else "")
                )
                if name in KEY_NAME_TO_PC:
                    chord_roots.append(KEY_NAME_TO_PC[name])

    if not chord_roots:
        return candidate_tonics[0] if candidate_tonics else default_pc

    # Diatonic roots: 0, 2, 4, 5, 7, 9, 10, 11
    diatonic_set = {0, 2, 4, 5, 7, 9, 10, 11}
    best_pc = candidate_tonics[0] if candidate_tonics else chord_roots[0]
    best_score = -1.0

    eval_tonics = candidate_tonics if candidate_tonics else list(range(12))
    for t in eval_tonics:
        match_count = sum(1 for r in chord_roots if ((r - t) % 12) in diatonic_set)
        bonus = 0.5 if ((chord_roots[0] - t) % 12) == 0 else 0.0
        score = match_count + bonus
        if score > best_score:
            best_score = score
            best_pc = t

    return best_pc


# --------------------------------------------------------------------------- Sequence Alignment (Needleman-Wunsch)
def pairwise_chord_score(s: DegreeChord, r: DegreeChord) -> float:
    """Pairwise alignment score between a sheet chord and a reference chord.

    - Same root & bass: 1.0
    - Same root (different bass): 0.7
    - Relative / substitution: 0.3
    - Mismatch: -0.6
    """
    if s.root_semitone == r.root_semitone:
        if s.bass_semitone == r.bass_semitone:
            return 1.0
        return 0.7

    diff = (s.root_semitone - r.root_semitone) % 12
    # Relative major/minor (diff = 3 or 9, e.g. C and Am, F and Dm)
    if diff in (3, 9):
        return 0.3
    # Mediant substitutions (diff = 4 or 8, e.g. C and Em)
    if diff in (4, 8):
        return 0.3
    # Tritone substitution (diff = 6, e.g. G7 and Db7)
    if diff == 6 and ("7" in s.quality or "7" in r.quality):
        return 0.3

    return -0.6


def needleman_wunsch(
    sheet_seq: list[DegreeChord],
    ref_seq: list[DegreeChord],
    gap_penalty: float = -0.4,
) -> tuple[float, list[tuple[Optional[int], Optional[int], float]]]:
    """Run Needleman-Wunsch sequence alignment between sheet chords and ref chords.

    Returns (total_score, alignment_pairs) where each pair is
    (sheet_idx_or_None, ref_idx_or_None, match_score).
    """
    n, m = len(sheet_seq), len(ref_seq)
    if n == 0 or m == 0:
        return 0.0, []

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        s_chord = sheet_seq[i - 1]
        for j in range(1, m + 1):
            r_chord = ref_seq[j - 1]
            match_score = pairwise_chord_score(s_chord, r_chord)
            dp[i][j] = max(
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j] + gap_penalty,
                dp[i][j - 1] + gap_penalty,
            )

    # Traceback
    i, j = n, m
    pairs: list[tuple[Optional[int], Optional[int], float]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = pairwise_chord_score(sheet_seq[i - 1], ref_seq[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + match_score)) < 1e-6:
                pairs.append((i - 1, j - 1, match_score))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + gap_penalty)) < 1e-6:
            pairs.append((i - 1, None, gap_penalty))
            i -= 1
        else:
            pairs.append((None, j - 1, gap_penalty))
            j -= 1

    pairs.reverse()
    return dp[n][m], pairs


# --------------------------------------------------------------------------- Gemini Grounding Implementation
def _query_gemini_search(title: str, artist: Optional[str]) -> Optional[ReferenceChart]:
    """Execute two-step Gemini grounding search and structuring."""
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    location = os.environ.get("WEB_EVIDENCE_LOCATION", DEFAULT_LOCATION)
    model = os.environ.get("WEB_EVIDENCE_MODEL", DEFAULT_MODEL)

    client = genai.Client(vertexai=True, project=project, location=location)

    search_prompt = (
        f"和弦谱 吉他谱: {title} {artist or ''}. Provide the full chord progression for all sections "
        f"(Intro, Verse, Chorus, Bridge, Outro), key, and capo. Output chord tokens in sequential order without lyrics."
    )

    resp1 = client.models.generate_content(
        model=model,
        contents=search_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )

    extracted_urls: list[str] = []
    if resp1.candidates and resp1.candidates[0].grounding_metadata:
        gm = resp1.candidates[0].grounding_metadata
        chunks = gm.grounding_chunks or []
        for chunk in chunks:
            web = chunk.web
            if web and web.uri:
                extracted_urls.append(web.uri)

    raw_text = resp1.text or ""
    if not raw_text.strip():
        return None

    structure_prompt = (
        f"Extract 1 to 2 structured chord charts for '{title}' by '{artist or ''}' from:\n"
        f"Grounding URLs: {json.dumps(extracted_urls[:3])}\n"
        f"Text:\n{raw_text}\n"
        f"Include key, capo, section_labels, and lines of chord tokens."
    )

    resp2 = client.models.generate_content(
        model=model,
        contents=structure_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReferenceChart,
            temperature=0.0,
            max_output_tokens=1000,
        ),
    )

    raw_json = resp2.text or "{}"
    chart = ReferenceChart.model_validate_json(raw_json)

    valid_sources: list[ReferenceSource] = []
    for s in chart.sources:
        clean_lines: list[list[str]] = []
        for line in s.lines:
            chords_in_line = [c.strip() for c in line if c.strip()]
            if chords_in_line:
                clean_lines.append(chords_in_line)
        if clean_lines:
            url = s.url if s.url and s.url.startswith("http") else (extracted_urls[0] if extracted_urls else "web")
            valid_sources.append(
                ReferenceSource(
                    url=url,
                    key=s.key,
                    capo=s.capo,
                    lines=clean_lines,
                    section_labels=s.section_labels,
                )
            )

    if not valid_sources:
        return None

    clean_chart = ReferenceChart(
        title=title,
        artist=artist,
        sources=valid_sources,
    )
    save_cached_reference(clean_chart)
    return clean_chart


# --------------------------------------------------------------------------- Public API
def find_reference(
    title: str,
    artist: Optional[str] = None,
    *,
    lang_hint: str = "zh",
) -> Optional[ReferenceChart]:
    """Find published chord charts for a song using Gemini Google Search grounding.

    Args:
        title: Song title.
        artist: Optional artist name.
        lang_hint: Language hint ('zh' by default).

    Returns:
        ReferenceChart with structured chord sources, or None on failure/offline.
    """
    if os.environ.get("WEB_EVIDENCE") == "0":
        return None

    # Check cache first (offline mode allowed to read local cache)
    cached = load_cached_reference(title, artist)
    if cached is not None:
        return cached

    # If offline, do not touch network
    if os.environ.get("QA_OFFLINE") == "1":
        return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_query_gemini_search, title, artist)
            return future.result(timeout=HARD_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("find_reference failed for %s (%s): %s", title, artist, exc)
        return None


def align_reference(
    sheet: ParsedSheet,
    ref: ReferenceChart,
    start_key_pc: int,
) -> dict[int, list[RefCandidate]]:
    """Align a reference chart against a ParsedSheet's chord stream using Needleman-Wunsch.

    Transposes reference chords to scale degrees relative to the reference key (handling capo),
    converts sheet chords to scale degrees, aligns chord sequences, and produces candidate
    chords per measure.

    Returns:
        dict mapping measure_index -> list[RefCandidate].
        Returns {} if overall alignment quality is poor (< ALIGNMENT_QUALITY_THRESHOLD).
    """
    if not ref.sources:
        return {}

    measures = sheet.measures()
    if not measures:
        return {}

    sheet_chords: list[tuple[int, float, DegreeChord]] = []
    for m in measures:
        measure_tonic = (
            start_key_pc + sum(kc.semitones for kc in sheet.key_changes if kc.at_measure <= m.index)
        ) % 12
        for c in m.chords:
            raw = c.raw.strip()
            if not raw or raw == "?":
                continue
            deg_chord = parse_chord_to_degree(
                raw,
                tonic_pc=measure_tonic,
                is_letter_hint=(sheet.header.chord_notation == "letter"),
            )
            if deg_chord is not None:
                sheet_chords.append((m.index, c.beat, deg_chord))

    if not sheet_chords:
        return {}

    sheet_deg_seq = [sc[2] for sc in sheet_chords]
    total_sheet_chords = len(sheet_deg_seq)

    source_candidates: dict[int, dict[str, list[float]]] = {}
    source_quality_scores: list[float] = []

    for s_idx, src in enumerate(ref.sources):
        ref_tonic_pc = determine_reference_tonic_pc(src, default_pc=start_key_pc)

        ref_deg_seq: list[DegreeChord] = []
        for line in src.lines:
            for tok in line:
                d = parse_chord_to_degree(tok, tonic_pc=ref_tonic_pc)
                if d is not None:
                    ref_deg_seq.append(d)

        if not ref_deg_seq:
            continue

        # Pass 1: Global alignment
        _, pairs = needleman_wunsch(sheet_deg_seq, ref_deg_seq)

        aligned_sheet_indices: set[int] = set()
        matched_scores: list[float] = []

        for s_i, r_j, score in pairs:
            if s_i is not None and r_j is not None and score > 0:
                m_idx, beat, s_chord = sheet_chords[s_i]
                r_chord = ref_deg_seq[r_j]
                aligned_sheet_indices.add(s_i)
                matched_scores.append(score)

                m_dict = source_candidates.setdefault(m_idx, {})
                m_dict.setdefault(r_chord.number_str, []).append(score)

        # Pass 2: Multi-pass for unaligned repeated sections
        unaligned_blocks: list[tuple[int, int]] = []
        start_u: Optional[int] = None
        for i in range(total_sheet_chords):
            if i not in aligned_sheet_indices:
                if start_u is None:
                    start_u = i
            else:
                if start_u is not None:
                    if i - start_u >= 3:
                        unaligned_blocks.append((start_u, i))
                    start_u = None
        if start_u is not None and total_sheet_chords - start_u >= 3:
            unaligned_blocks.append((start_u, total_sheet_chords))

        for u_start, u_end in unaligned_blocks:
            sub_sheet = sheet_deg_seq[u_start:u_end]
            _, sub_pairs = needleman_wunsch(sub_sheet, ref_deg_seq)
            sub_matched = 0
            for s_sub, r_sub, sub_score in sub_pairs:
                if s_sub is not None and r_sub is not None and sub_score >= 0.7:
                    sub_matched += 1
            if sub_matched / len(sub_sheet) >= 0.5:
                for s_sub, r_sub, sub_score in sub_pairs:
                    if s_sub is not None and r_sub is not None and sub_score > 0:
                        real_s_i = u_start + s_sub
                        m_idx, beat, s_chord = sheet_chords[real_s_i]
                        r_chord = ref_deg_seq[r_sub]
                        matched_scores.append(sub_score)
                        m_dict = source_candidates.setdefault(m_idx, {})
                        m_dict.setdefault(r_chord.number_str, []).append(sub_score)

        src_quality = sum(matched_scores) / max(1, total_sheet_chords)
        source_quality_scores.append(src_quality)

    if not source_quality_scores:
        return {}

    overall_alignment_quality = max(source_quality_scores)
    logger.debug(
        "Overall alignment quality: %.3f (threshold: %.2f)",
        overall_alignment_quality,
        ALIGNMENT_QUALITY_THRESHOLD,
    )

    if overall_alignment_quality < ALIGNMENT_QUALITY_THRESHOLD:
        return {}

    # Build final RefCandidate objects per measure
    result: dict[int, list[RefCandidate]] = {}
    for m_idx, chord_dict in source_candidates.items():
        candidates: list[RefCandidate] = []
        for chord_str, scores in chord_dict.items():
            avg_support = sum(scores) / len(scores)
            source_count = len(scores)
            if source_count > 1:
                avg_support = min(1.0, avg_support * (1.0 + 0.1 * (source_count - 1)))
            candidates.append(
                RefCandidate(
                    chord=chord_str,
                    support=round(avg_support, 3),
                    sources=source_count,
                )
            )
        candidates.sort(key=lambda c: (c.support, c.sources), reverse=True)
        result[m_idx] = candidates

    return result
