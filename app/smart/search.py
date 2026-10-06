"""Multi-source web chord chart search and consensus module (智能创建 M2).

Given a song title and/or artist, searches multiple public chord chart sources
on the web, extracts structured fields (key, time signature, tempo, section
structure, and chord progressions per bar), normalizes them to Taiwanese
number notation relative to the song's sounding key, and derives a consensus
lead sheet.

HARD RULES:
- Never store, cache, log, or return third-party chart full text or lyrics.
- Only extracted structured fields plus source URL and page title.
- Never download audio.
"""
from __future__ import annotations

from collections import Counter
import concurrent.futures
from dataclasses import dataclass
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.smart.models import Agreement, LeadChord, LeadMeasure, LeadSheet, SourceLink
from app.theory.chords import clean_raw_chord, parse_chord
from app.theory.keys import (
    KEY_NAME_TO_PC,
    PC_TO_DEFAULT_KEY_NAME,
    key_name_to_pc,
    normalize_key_string,
)
from app.qa.web_evidence import (
    DegreeChord,
    format_degree_to_number,
    needleman_wunsch,
    pairwise_chord_score,
    parse_chord_to_degree,
    salvage_json_dict,
)

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "cellular-cider-495602-r9"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"


# --------------------------------------------------------------------------- Data Models
class SourceSection(BaseModel):
    """A section of a source chord chart with chords in number notation."""

    label: str
    bars: list[list[str]] = Field(
        default_factory=list,
        description="Chords per bar in NUMBER notation relative to key, e.g. [['1'], ['5/7']]",
    )


class SourceChart(BaseModel):
    """A single extracted chord chart from a web source."""

    url: str
    title: str = ""
    key: Optional[str] = None
    capo: int = 0
    played_key: Optional[str] = None
    sections: list[SourceSection] = Field(default_factory=list)
    raw_notation: Literal["letter", "number"] = "letter"


class ConsensusBar(BaseModel):
    """Consensus chord reading for a single measure."""

    chords: list[str] = Field(default_factory=list)
    support: int = 1
    total: int = 1
    alternatives: list[str] = Field(default_factory=list)


class ConsensusSection(BaseModel):
    """A section of consensus measures."""

    label: str
    bars: list[ConsensusBar] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Aggregated multi-source search result with consensus chord chart."""

    title: str = ""
    artist: str = ""
    key: str = "C"
    time_signature: str = "4/4"
    tempo_bpm: Optional[float] = None
    sources: list[SourceChart] = Field(default_factory=list)
    consensus_sections: list[ConsensusSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- Caching
def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", name.strip())
    return re.sub(r"_+", "_", cleaned)


def get_search_cache_path(title: str, artist: str = "") -> Path:
    """Return local cache path for SearchResult JSON."""
    base_dir = Path("out/smart_search_cache")
    t_clean = _sanitize_filename(title or "unknown")
    a_clean = _sanitize_filename(artist or "unknown")
    return base_dir / f"{t_clean}__{a_clean}.json"


def load_cached_search(title: str, artist: str = "") -> Optional[SearchResult]:
    """Load SearchResult from storage (GCS if BUCKET is set, else local dev cache)."""
    t_clean = _sanitize_filename(title or "unknown")
    a_clean = _sanitize_filename(artist or "unknown")
    filename = f"{t_clean}__{a_clean}.json"

    # 1. Try BUCKET via app.storage.get_storage()
    if os.environ.get("BUCKET"):
        try:
            from app.storage import get_storage

            storage = get_storage()
            storage_path = f"cache/smart_search/{filename}"
            if storage.exists(storage_path):
                data = storage.get_json(storage_path)
                return SearchResult.model_validate(data)
        except Exception as exc:
            logger.warning("Error reading from storage backend for %s: %s", filename, exc)

    # 2. Local cache fallback for dev when BUCKET is unset
    local_path = get_search_cache_path(title, artist)
    if local_path.is_file():
        try:
            data = local_path.read_text(encoding="utf-8")
            return SearchResult.model_validate_json(data)
        except Exception as exc:
            logger.warning("Error reading local cache %s: %s", local_path, exc)

    return None


def save_cached_search(res: SearchResult) -> None:
    """Save SearchResult to storage (GCS if BUCKET is set, else local dev cache)."""
    t_clean = _sanitize_filename(res.title or "unknown")
    a_clean = _sanitize_filename(res.artist or "unknown")
    filename = f"{t_clean}__{a_clean}.json"

    if os.environ.get("BUCKET"):
        try:
            from app.storage import get_storage

            storage = get_storage()
            storage_path = f"cache/smart_search/{filename}"
            storage.put_json(storage_path, res.model_dump())
        except Exception as exc:
            logger.warning("Failed saving search cache to storage for %s: %s", res.title, exc)

    # Always write local copy for dev / inspection
    local_path = get_search_cache_path(res.title, res.artist)
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed saving local search cache for %s: %s", res.title, exc)


_URL_CHART_CACHE: dict[str, SourceChart] = {}


def get_url_cache_path(url: str) -> Path:
    """Return local cache path for extracted SourceChart JSON."""
    base_dir = Path("out/smart_search_cache/urls")
    url_hash = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:20]
    return base_dir / f"{url_hash}.json"


def load_cached_url_chart(url: str) -> Optional[SourceChart]:
    """Load cached SourceChart for a URL if available (structured fields only, zero text/lyrics)."""
    if not url:
        return None
    u = url.strip()
    if u in _URL_CHART_CACHE:
        return _URL_CHART_CACHE[u]

    url_hash = hashlib.sha256(u.encode("utf-8")).hexdigest()[:20]
    if os.environ.get("BUCKET"):
        try:
            from app.storage import get_storage

            storage = get_storage()
            storage_path = f"cache/smart_search/urls/{url_hash}.json"
            if storage.exists(storage_path):
                data = storage.get_json(storage_path)
                chart = SourceChart.model_validate(data)
                _URL_CHART_CACHE[u] = chart
                return chart
        except Exception as exc:
            logger.debug("Failed reading URL cache from storage for %s: %s", u, exc)

    local_path = get_url_cache_path(u)
    if local_path.is_file():
        try:
            data = local_path.read_text(encoding="utf-8")
            chart = SourceChart.model_validate_json(data)
            _URL_CHART_CACHE[u] = chart
            return chart
        except Exception as exc:
            logger.debug("Failed reading local URL cache %s: %s", local_path, exc)

    return None


def save_cached_url_chart(url: str, chart: SourceChart) -> None:
    """Save extracted SourceChart to URL cache (structured fields only, zero text/lyrics)."""
    if not url or not chart:
        return
    u = url.strip()
    _URL_CHART_CACHE[u] = chart

    url_hash = hashlib.sha256(u.encode("utf-8")).hexdigest()[:20]
    if os.environ.get("BUCKET"):
        try:
            from app.storage import get_storage

            storage = get_storage()
            storage_path = f"cache/smart_search/urls/{url_hash}.json"
            storage.put_json(storage_path, chart.model_dump())
        except Exception as exc:
            logger.debug("Failed saving URL cache to storage for %s: %s", u, exc)

    local_path = get_url_cache_path(u)
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(chart.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed saving local URL cache for %s: %s", u, exc)


# --------------------------------------------------------------------------- Theory & Normalization
def simplify_chord_for_voting(chord_str: str) -> tuple[tuple[int, int], str, Optional[tuple[int, int]]]:
    """Simplify chord extension for voting while preserving root, triad quality, and bass.

    Returns:
        ((degree, accidental), triad_quality, bass_tuple_or_none)
    """
    dc = parse_chord_to_degree(chord_str)
    if not dc:
        # Fallback heuristic for unrecognized chord tokens
        m = re.match(r"^([b#]?)([1-7])(.*)$", chord_str.strip())
        if m:
            acc = -1 if m.group(1) == "b" else (1 if m.group(1) == "#" else 0)
            deg = int(m.group(2))
            rest = m.group(3)
            q = "m" if "m" in rest and "maj" not in rest else ("dim" if "dim" in rest or "-5" in rest else "maj")
            return ((deg, acc), q, None)
        return ((1, 0), "maj", None)

    q = dc.quality
    if "dim" in q or "m7b5" in q or "-5" in q:
        triad = "dim"
    elif "aug" in q:
        triad = "aug"
    elif q.startswith("m") and not q.startswith("maj"):
        triad = "m"
    elif "sus" in q:
        triad = "sus"
    else:
        triad = "maj"

    bass = (
        (dc.bass_degree, dc.bass_accidental)
        if (dc.bass_degree, dc.bass_accidental) != (dc.degree, dc.accidental)
        else None
    )
    return ((dc.degree, dc.accidental), triad, bass)


def infer_letter_chord_roots(sections: list[dict[str, Any]]) -> list[int]:
    """Extract pitch classes of letter chord roots across raw sections."""
    roots: list[int] = []
    for s in sections:
        for b in s.get("bars", []):
            if isinstance(b, list):
                for chord in b:
                    c_str = str(chord).strip()
                    m = re.match(r"^([A-Ga-g])([b#♭♯]?)", c_str)
                    if m:
                        name = m.group(1).upper() + (
                            "b" if m.group(2) in ("b", "♭") else ("#" if m.group(2) in ("#", "♯") else "")
                        )
                        if name in KEY_NAME_TO_PC:
                            roots.append(KEY_NAME_TO_PC[name])
    return roots


FORBIDDEN_HOSTS = {
    "example.com",
    "example.org",
    "example.net",
    "localhost",
    "127.0.0.1",
    "vertexaisearch.cloud.google.com",
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "tiktok.com",
}


def is_valid_source_url(url: str, allow_test_urls: bool = False) -> bool:
    """Validate that a URL has a real http(s) scheme and non-placeholder domain."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    try:
        parsed = urlparse(u)
        host = (parsed.netloc or "").lower().split(":")[0]
        if not host or "." not in host:
            return False
        if not allow_test_urls:
            if host in FORBIDDEN_HOSTS or any(host.endswith("." + fh) for fh in FORBIDDEN_HOSTS):
                return False
        return True
    except Exception:
        return False


def get_registered_domain(url: str) -> str:
    """Extract root domain / organization domain for diversity grouping."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().split(":")[0]
        parts = host.split(".")
        if len(parts) >= 3 and parts[-2] in ("com", "co", "org", "net", "gov", "edu"):
            return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    except Exception:
        return "web"


def normalize_section_label(raw_label: str) -> str:
    """Normalize raw section label into standard Chinese musical structure names."""
    cleaned = raw_label.strip()
    if not cleaned or "示例" in cleaned or "example" in cleaned.lower():
        return "段落"
    l_lower = cleaned.lower()
    if "前奏" in cleaned or "intro" in l_lower:
        return "前奏"
    elif (
        "预副歌" in cleaned
        or "导歌" in cleaned
        or "pre-chorus" in l_lower
        or "prechorus" in l_lower
        or "pre chorus" in l_lower
    ):
        return "预副歌"
    elif "副歌" in cleaned or "chorus" in l_lower:
        return "副歌"
    elif "主歌" in cleaned or "verse" in l_lower:
        return "主歌"
    elif "间奏" in cleaned or "interlude" in l_lower:
        return "间奏"
    elif "桥段" in cleaned or "bridge" in l_lower:
        return "桥段"
    elif "尾奏" in cleaned or "outro" in l_lower or "ending" in l_lower:
        return "尾奏"
    elif cleaned.upper() in ("A", "B", "C", "D", "E"):
        return cleaned.upper()
    return cleaned


class HTMLToText(HTMLParser):
    """Memory-only HTML to text stripper preserving line breaks and table cells."""

    def __init__(self) -> None:
        super().__init__()
        self._ignore_stack: list[str] = []
        self._lines: list[str] = []
        self._current_line: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in ("script", "style", "nav", "header", "footer", "noscript", "svg"):
            self._ignore_stack.append(tag_lower)
        elif tag_lower in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "pre"):
            if self._current_line:
                self._lines.append("".join(self._current_line).strip())
                self._current_line = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._ignore_stack and self._ignore_stack[-1] == tag_lower:
            self._ignore_stack.pop()
        elif tag_lower in ("p", "div", "tr", "li", "pre"):
            if self._current_line:
                self._lines.append("".join(self._current_line).strip())
                self._current_line = []

    def handle_data(self, data: str) -> None:
        if not self._ignore_stack:
            self._current_line.append(data)

    def get_text(self) -> str:
        if self._current_line:
            self._lines.append("".join(self._current_line).strip())
        non_empty = [l for l in self._lines if l]
        return "\n".join(non_empty)


def extract_markers_from_text(text: str) -> dict[str, Any]:
    """Scan raw page text for capo and key markers before discarding text.

    Returns:
        {
            "capo_fret": Optional[int],
            "shapes_key": Optional[str],
            "stated_original_key": Optional[str],
        }
    """
    res: dict[str, Any] = {
        "capo_fret": None,
        "shapes_key": None,
        "stated_original_key": None,
    }
    if not text or not isinstance(text, str):
        return res

    # 1. Capo markers
    m_capo_zero = re.search(
        r"(?:变调夹\s*[:：]?\s*(?:不夹|无|0|none)|capo\s*[:：]?\s*(?:none|no|0|off)|\(no capo\))",
        text,
        re.IGNORECASE,
    )
    if m_capo_zero:
        res["capo_fret"] = 0
    else:
        m_capo = re.search(
            r"(?:变调夹\s*[:：]?\s*([0-9]{1,2})\s*品?|夹\s*([0-9]{1,2})\s*品|capo\s*(?:fret\s*)?[:：]?\s*([0-9]{1,2}))",
            text,
            re.IGNORECASE,
        )
        if m_capo:
            for g in m_capo.groups():
                if g is not None:
                    res["capo_fret"] = int(g)
                    break

    # 2. Stated Original Key (原调 / Key: / 1=)
    m_orig = re.search(
        r"(?:原调\s*[:：]?\s*([A-Ga-g][b#♭♯]?(?:m)?)|(?:original\s+)?key\s*[:：]\s*([A-Ga-g][b#♭♯]?(?:m)?)|1\s*=\s*([A-Ga-g][b#♭♯]?))",
        text,
        re.IGNORECASE,
    )
    if m_orig:
        for g in m_orig.groups():
            if g is not None:
                res["stated_original_key"] = g.strip()
                break

    # 3. Shapes Key (选调 / Play:)
    m_shapes = re.search(
        r"(?:选调\s*[:：]?\s*([A-Ga-g][b#♭♯]?(?:m)?)|play(?:ed)?\s*(?:key)?\s*[:：]\s*([A-Ga-g][b#♭♯]?(?:m)?))",
        text,
        re.IGNORECASE,
    )
    if m_shapes:
        for g in m_shapes.groups():
            if g is not None:
                res["shapes_key"] = g.strip()
                break

    return res


def compute_sounding_and_played_keys(
    capo_fret: int = 0,
    shapes_key: Optional[str] = None,
    stated_original_key: Optional[str] = None,
    raw_sections: Optional[list[Any]] = None,
    default_key_hint: Optional[str] = None,
) -> tuple[str, str, int, list[str]]:
    """Compute sounding key, played key, capo fret, and any warning notes.

    Rule:
    - Never ask the model for 'sounding key' directly.
    - Compute sounding = shapes_key + capo in code.
    - When stated_original_key is present and consistent with shapes+capo
      (or with shapes alone when capo == 0) prefer it.
    - If inconsistent, record a warning and prefer shapes+capo.
    """
    capo = max(0, int(capo_fret or 0))

    shapes_pc: Optional[int] = None
    if shapes_key:
        try:
            shapes_pc = key_name_to_pc(normalize_key_string(str(shapes_key)))
        except Exception:
            pass

    stated_pc: Optional[int] = None
    if stated_original_key:
        try:
            stated_pc = key_name_to_pc(normalize_key_string(str(stated_original_key)))
        except Exception:
            pass

    # Infer shapes_pc if missing
    if shapes_pc is None:
        roots = infer_letter_chord_roots(raw_sections or [])
        if roots:
            diatonic_set = {0, 2, 4, 5, 7, 9, 10, 11}
            best_pc = roots[0]
            best_score = -1.0
            for t in range(12):
                matches = sum(1 for r in roots if ((r - t) % 12) in diatonic_set)
                bonus = 0.5 if ((roots[0] - t) % 12) == 0 else 0.0
                if matches + bonus > best_score:
                    best_score = matches + bonus
                    best_pc = t
            shapes_pc = best_pc
        elif stated_pc is not None and capo == 0:
            shapes_pc = stated_pc
        elif default_key_hint:
            try:
                shapes_pc = key_name_to_pc(normalize_key_string(default_key_hint))
            except Exception:
                shapes_pc = 0
        else:
            shapes_pc = 0

    played_key = PC_TO_DEFAULT_KEY_NAME[shapes_pc]
    computed_sounding_pc = (shapes_pc + capo) % 12
    computed_sounding_key = PC_TO_DEFAULT_KEY_NAME[computed_sounding_pc]

    warnings: list[str] = []
    if stated_pc is not None:
        stated_canon = PC_TO_DEFAULT_KEY_NAME[stated_pc]
        is_consistent_with_shapes_capo = (stated_pc == computed_sounding_pc)
        is_consistent_with_shapes_alone = (capo == 0 and stated_pc == shapes_pc)

        if is_consistent_with_shapes_capo or is_consistent_with_shapes_alone:
            sounding_key = stated_canon
        else:
            msg = (
                f"Stated original key '{stated_original_key}' ({stated_canon}) is inconsistent "
                f"with shapes key '{played_key}' + capo {capo} (computed {computed_sounding_key}); "
                f"preferring shapes+capo."
            )
            logger.warning(msg)
            warnings.append(msg)
            sounding_key = computed_sounding_key
    else:
        sounding_key = computed_sounding_key

    return (sounding_key, played_key, capo, warnings)


def normalize_source_chart(
    raw_data: dict[str, Any],
    default_key_hint: Optional[str] = None,
    *,
    min_bars: int = 0,
    min_sections: int = 0,
    allow_test_urls: bool = False,
) -> Optional[SourceChart]:
    """Normalize a raw extracted chord chart dictionary into a valid SourceChart.

    Transposes letter chords or capo/played_key to Taiwanese number notation
    relative to the song's sounding key.
    """
    if not isinstance(raw_data, dict):
        return None

    url = str(raw_data.get("url", "")).strip()
    if not is_valid_source_url(url, allow_test_urls=allow_test_urls):
        return None
    title = str(raw_data.get("title", "")).strip() or get_registered_domain(url)

    # 1. Parse Capo, Shapes Key, Stated Key
    capo_raw = raw_data.get("capo_fret")
    if capo_raw is None:
        capo_raw = raw_data.get("capo", 0)
    try:
        capo_val = int(capo_raw or 0)
    except Exception:
        capo_val = 0

    shapes_key_raw = raw_data.get("shapes_key") or raw_data.get("played_key")
    stated_key_raw = raw_data.get("stated_original_key") or raw_data.get("key")

    raw_sections = raw_data.get("sections", [])
    if not isinstance(raw_sections, list) or not raw_sections:
        return None

    # Check notation style (letter vs number)
    all_tokens: list[str] = []
    for s in raw_sections:
        if isinstance(s, dict):
            for b in s.get("bars", []):
                if isinstance(b, list):
                    for tok in b:
                        if tok and isinstance(tok, str):
                            all_tokens.append(tok.strip())

    if not all_tokens:
        return None

    is_number = sum(1 for t in all_tokens if re.match(r"^[b#]?[1-7]", t)) > len(all_tokens) / 2
    raw_notation: Literal["letter", "number"] = "number" if is_number else "letter"

    sounding_key, played_key, capo_val, _key_warnings = compute_sounding_and_played_keys(
        capo_fret=capo_val,
        shapes_key=shapes_key_raw,
        stated_original_key=stated_key_raw,
        raw_sections=raw_sections,
        default_key_hint=default_key_hint,
    )

    if raw_notation == "number":
        printed_pc = 0
    else:
        try:
            printed_pc = key_name_to_pc(played_key)
        except Exception:
            printed_pc = 0

    # 3. Convert all bars into number notation
    clean_sections: list[SourceSection] = []
    for s_item in raw_sections:
        if not isinstance(s_item, dict):
            continue
        raw_label = str(s_item.get("label", "")).strip()
        label = normalize_section_label(raw_label)
        raw_bars = s_item.get("bars", [])
        if not isinstance(raw_bars, list):
            continue

        clean_bars: list[list[str]] = []
        for bar in raw_bars:
            if not isinstance(bar, list):
                continue
            bar_chords: list[str] = []
            for tok in bar:
                if not isinstance(tok, str) or not tok.strip():
                    continue
                c_str = tok.strip()
                if raw_notation == "number":
                    dc = parse_chord_to_degree(c_str)
                    bar_chords.append(dc.number_str if dc else clean_raw_chord(c_str))
                else:
                    dc = parse_chord_to_degree(c_str, tonic_pc=printed_pc)
                    if dc:
                        bar_chords.append(dc.number_str)
                    else:
                        bar_chords.append(clean_raw_chord(c_str))
            if bar_chords:
                if len(bar_chords) > 2:
                    clean_bars.extend([[c] for c in bar_chords])
                else:
                    clean_bars.append(bar_chords)

        if clean_bars:
            clean_sections.append(SourceSection(label=label, bars=clean_bars))

    if not clean_sections:
        return None

    total_bars = sum(len(sec.bars) for sec in clean_sections)
    if min_bars > 0 and total_bars < min_bars:
        return None
    if min_sections > 0 and len(clean_sections) < min_sections:
        return None

    return SourceChart(
        url=url,
        title=title,
        key=sounding_key,
        capo=capo_val,
        played_key=played_key,
        sections=clean_sections,
        raw_notation=raw_notation,
    )


# --------------------------------------------------------------------------- Bar Alignment & Voting
def _primary_degree_chord(bar: list[str]) -> DegreeChord:
    """Return the primary DegreeChord for a bar (first chord), or a dummy rest chord."""
    if bar:
        dc = parse_chord_to_degree(bar[0])
        if dc:
            return dc
    return DegreeChord(
        degree=0,
        accidental=0,
        quality="",
        bass_degree=0,
        bass_accidental=0,
        root_semitone=-1,
        bass_semitone=-1,
        number_str="",
    )


def custom_bar_score(s: DegreeChord, r: DegreeChord) -> float:
    """Pairwise alignment score between two bars' primary degree chords."""
    if s.degree == 0 and r.degree == 0:
        return 0.5
    if s.degree == 0 or r.degree == 0:
        return -0.5
    base_score = pairwise_chord_score(s, r)
    # Give a small bonus for exact quality match so identical chords align over substitutes
    if s.quality == r.quality and base_score > 0:
        base_score += 0.1
    return base_score


def align_bar_sequences(
    ref_bars: list[list[str]],
    cand_bars: list[list[str]],
    gap_penalty: float = -0.4,
) -> list[tuple[Optional[int], Optional[int], float]]:
    """Align candidate bar sequence to reference bar sequence using Needleman-Wunsch."""
    n, m = len(ref_bars), len(cand_bars)
    if n == 0 or m == 0:
        return []

    ref_dcs = [_primary_degree_chord(b) for b in ref_bars]
    cand_dcs = [_primary_degree_chord(b) for b in cand_bars]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        s = ref_dcs[i - 1]
        for j in range(1, m + 1):
            r = cand_dcs[j - 1]
            match_score = custom_bar_score(s, r)
            dp[i][j] = max(
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j] + gap_penalty,
                dp[i][j - 1] + gap_penalty,
            )

    i, j = n, m
    pairs: list[tuple[Optional[int], Optional[int], float]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = custom_bar_score(ref_dcs[i - 1], cand_dcs[j - 1])
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
    return pairs


def vote_bar(
    candidate_bars: list[list[str]],
    total_sources: int,
) -> ConsensusBar:
    """Vote on chords in a bar across multiple aligned candidate bars.

    Simplifies chord extensions for voting (comparing root, bass, and triad quality),
    preserves the most frequent full spelling for display, and records alternatives.
    """
    valid_bars = [b for b in candidate_bars if b]
    if not valid_bars:
        return ConsensusBar(chords=[], support=0, total=total_sources, alternatives=[])

    # Target number of chords in this bar by majority
    lengths = [len(b) for b in valid_bars]
    target_len = Counter(lengths).most_common(1)[0][0]

    winner_chords: list[str] = []
    pos_supports: list[int] = []
    all_alternatives: list[str] = []

    for pos in range(target_len):
        tokens_at_pos: list[str] = []
        for b in valid_bars:
            if pos < len(b):
                tokens_at_pos.append(b[pos])

        if not tokens_at_pos:
            continue

        # Map each full chord token to its simplified voting key
        sim_to_full: dict[Any, list[str]] = {}
        for tok in tokens_at_pos:
            sim = simplify_chord_for_voting(tok)
            sim_to_full.setdefault(sim, []).append(tok)

        # Count votes per simplified key
        sim_counts = {sim: len(spellings) for sim, spellings in sim_to_full.items()}
        sorted_sims = sorted(sim_counts.items(), key=lambda x: x[1], reverse=True)
        winner_sim, win_votes = sorted_sims[0]

        # For the winning simplified key, pick the most frequent full spelling
        winner_spellings = sim_to_full[winner_sim]
        winner_full = Counter(winner_spellings).most_common(1)[0][0]
        winner_chords.append(winner_full)
        pos_supports.append(win_votes)

        # Non-winning simplified keys become alternatives
        for alt_sim, _ in sorted_sims[1:]:
            alt_spellings = sim_to_full[alt_sim]
            alt_full = Counter(alt_spellings).most_common(1)[0][0]
            if alt_full not in all_alternatives and alt_full != winner_full:
                all_alternatives.append(alt_full)

    bar_support = min(pos_supports) if pos_supports else 1
    return ConsensusBar(
        chords=winner_chords,
        support=bar_support,
        total=total_sources,
        alternatives=all_alternatives,
    )


def source_completeness(source: SourceChart) -> float:
    """Compute completeness score for selecting reference source."""
    total_bars = sum(len(sec.bars) for sec in source.sections)
    total_chords = sum(sum(len(b) for b in sec.bars) for sec in source.sections)
    return float(total_bars) + 0.05 * float(total_chords)


def _ref_source_score(s: SourceChart, median_bar_count: float) -> float:
    b_count = sum(len(sec.bars) for sec in s.sections)
    dist = abs(b_count - median_bar_count)
    return -dist * 100.0 + source_completeness(s)


def choose_reference_source(sources: list[SourceChart]) -> SourceChart:
    """Choose the best reference source.

    Prefers sources near the median/majority bar count, breaking ties by completeness.
    """
    if len(sources) <= 1:
        return sources[0]
    import statistics

    bar_counts = [sum(len(sec.bars) for sec in s.sections) for s in sources]
    med = statistics.median(bar_counts)
    return max(sources, key=lambda s: _ref_source_score(s, med))


def build_consensus(
    sources: list[SourceChart],
    title: str = "",
    artist: str = "",
) -> SearchResult:
    """Build consensus sections and metadata from a list of normalized SourceCharts."""
    if not sources:
        return SearchResult(
            title=title,
            artist=artist,
            warnings=["No valid chord chart sources found"],
        )

    # 1. Consensus Key by majority
    key_pcs: list[int] = []
    for s in sources:
        if s.key:
            try:
                k_clean = normalize_key_string(s.key)
                key_pcs.append(key_name_to_pc(k_clean))
            except Exception:
                pass
    if key_pcs:
        majority_pc = Counter(key_pcs).most_common(1)[0][0]
        consensus_key = PC_TO_DEFAULT_KEY_NAME[majority_pc]
    else:
        consensus_key = sources[0].key or "C"

    # 2. Select reference source
    ref_source = choose_reference_source(sources)

    # Flatten reference source bars with metadata
    flat_ref_bars: list[list[str]] = []
    bar_to_section: list[tuple[int, str]] = []  # (section_idx, section_label)

    for sec_idx, sec in enumerate(ref_source.sections):
        for bar in sec.bars:
            flat_ref_bars.append(bar)
            bar_to_section.append((sec_idx, sec.label))

    total_sources = len(sources)
    aligned_votes_per_ref_bar: list[list[list[str]]] = [[] for _ in range(len(flat_ref_bars))]

    # 3. Align each source against reference
    for s in sources:
        if s is ref_source:
            for r_idx, b in enumerate(flat_ref_bars):
                aligned_votes_per_ref_bar[r_idx].append(b)
            continue

        cand_bars: list[list[str]] = []
        for sec in s.sections:
            for bar in sec.bars:
                cand_bars.append(bar)

        pairs = align_bar_sequences(flat_ref_bars, cand_bars)
        for r_idx, c_idx, score in pairs:
            if r_idx is not None and c_idx is not None and score > -0.5:
                aligned_votes_per_ref_bar[r_idx].append(cand_bars[c_idx])

    # 4. Vote per bar
    consensus_bars: list[ConsensusBar] = []
    for r_idx, candidate_votes in enumerate(aligned_votes_per_ref_bar):
        c_bar = vote_bar(candidate_votes, total_sources=total_sources)
        consensus_bars.append(c_bar)

    # 5. Group into ConsensusSections matching ref_source
    consensus_sections: list[ConsensusSection] = []
    current_sec_idx = -1
    current_sec_label = ""
    current_sec_bars: list[ConsensusBar] = []

    for r_idx, (s_idx, s_lbl) in enumerate(bar_to_section):
        if s_idx != current_sec_idx:
            if current_sec_bars:
                consensus_sections.append(
                    ConsensusSection(label=current_sec_label, bars=current_sec_bars)
                )
            current_sec_idx = s_idx
            current_sec_label = s_lbl
            current_sec_bars = [consensus_bars[r_idx]]
        else:
            current_sec_bars.append(consensus_bars[r_idx])

    if current_sec_bars:
        consensus_sections.append(
            ConsensusSection(label=current_sec_label, bars=current_sec_bars)
        )

    return SearchResult(
        title=title or ref_source.title,
        artist=artist,
        key=consensus_key,
        time_signature="4/4",
        tempo_bpm=None,
        sources=sources,
        consensus_sections=consensus_sections,
        warnings=[],
    )


# --------------------------------------------------------------------------- Gemini Grounded Search
class _ExtractedSection(BaseModel):
    label: str = Field(default="段落", description="Section label like 前奏, 主歌, 副歌, 间奏, 尾奏")
    bars: list[list[str]] = Field(
        default_factory=list,
        description="List of measures, each containing 1 or 2 chord strings",
    )


class _ExtractedChart(BaseModel):
    has_chart: bool = Field(default=False, description="True ONLY if actual chord progressions exist")
    title_matches: bool = Field(default=False, description="True ONLY if matches requested song title/artist")
    title: str = Field(default="", description="Website name or source title")
    capo_fret: int = Field(default=0, description="Capo fret number (0 if none or not mentioned, look for 'Capo', '变调夹', 'X品', '夹X')")
    shapes_key: Optional[str] = Field(default=None, description="Key of the chord shapes as written/fingered on the page (选调, e.g. 'C', 'G', 'D')")
    stated_original_key: Optional[str] = Field(default=None, description="Explicitly stated original song key if present ('原调', 'Key:', '1='), e.g. 'D', 'Bb', 'A'")
    raw_notation: Literal["letter", "number"] = "letter"
    sections: list[_ExtractedSection] = Field(default_factory=list)


def resolve_redirect_url(url: str, timeout_s: float = 6.0) -> str:
    """Resolve redirect URLs (such as Vertex AI Search grounding redirects) to canonical web URLs."""
    if not url or not isinstance(url, str):
        return ""
    u = url.strip()
    if "grounding-api-redirect" not in u and not u.startswith("https://vertexaisearch.cloud.google.com"):
        return u
    try:
        import httpx

        r = httpx.get(
            u,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            },
            follow_redirects=True,
            timeout=timeout_s,
        )
        return str(r.url)
    except Exception as exc:
        logger.debug("Failed resolving redirect for %s: %s", u, exc)
        return u


def _fetch_page_content_in_memory(
    client: Any,
    model: str,
    url: str,
    title: str,
    artist: str,
) -> Optional[str]:
    """Fetch raw page in memory and extract textual chord chart representation.

    Zero storage/logging of full text or lyrics. Content is parsed and discarded immediately.
    """
    import httpx

    # 1. Attempt lightweight HTTP GET
    try:
        r = httpx.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            follow_redirects=True,
            timeout=5.0,
        )
        if r.status_code == 200:
            if "tabs.ultimate-guitar.com" in url or "data-content=" in r.text:
                ug_m = re.search(r'data-content=["\'](.*?)["\']', r.text)
                if ug_m:
                    try:
                        import html as py_html

                        ug_data = json.loads(py_html.unescape(ug_m.group(1)))
                        tab_view = ug_data.get("store", {}).get("page", {}).get("data", {}).get("tab_view", {})
                        meta = tab_view.get("meta", {})
                        capo = meta.get("capo", 0)
                        tuning = meta.get("tuning", {}).get("name", "")
                        tab = tab_view.get("tab", {})
                        tonality = tab.get("tonality_name") or ""
                        wiki = tab_view.get("wiki_tab", {})
                        content = wiki.get("content", "")
                        if content:
                            clean_chords = re.sub(r"\[/?(ch|tab)\]", "", content)
                            lines = [
                                f"Key: {tonality}" if tonality else "",
                                f"Capo: {capo}",
                                f"Tuning: {tuning}" if tuning else "",
                                clean_chords,
                            ]
                            ug_text = "\n".join(l for l in lines if l)
                            if len(ug_text) >= 100:
                                return ug_text[:12000]
                    except Exception as ug_exc:
                        logger.debug("UG json parse failed for %s: %s", url, ug_exc)

            parser = HTMLToText()
            parser.feed(r.text)
            page_text = parser.get_text()
            if len(page_text) >= 120:
                return page_text[:12000]
    except Exception as exc:
        logger.debug("HTTP fetch failed for %s: %s", url, exc)

    # 2. Only fallback to Gemini url_context for dynamic/SPA pages that require JS rendering
    domain = get_registered_domain(url).lower()
    is_spa_domain = any(d in domain for d in ("ultimate-guitar.com", "songsterr.com", "blogspot.com", "hopamchuan.com"))
    if not is_spa_domain:
        return None

    try:
        from google.genai import types

        prompt = (
            f"Read the web page at {url}.\n"
            f"Does it contain a chord chart for '{title}' by '{artist}'?\n"
            f"If so, extract the original key, played key, capo, and the full ordered sections "
            f"(Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro) with their chord progressions.\n"
            f"Do not include lyrics."
        )
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(url_context=types.UrlContext())],
                temperature=0.0,
                http_options=types.HttpOptions(timeout=25000),
            ),
        )
        if resp.text and len(resp.text.strip()) > 50:
            return resp.text.strip()
    except Exception as exc:
        logger.debug("url_context fallback failed for %s: %s", url, exc)

    return None


def _extract_chart_from_content(
    client: Any,
    model: str,
    content: str,
    real_url: str,
    title: str,
    artist: str,
    default_key: Optional[str] = None,
) -> Optional[SourceChart]:
    """Extract structured SourceChart from text content using Gemini with response_schema."""
    from google.genai import types

    # 1. Run cheap regex pass over the page text for capo/原调/选调 markers before discarding
    markers = extract_markers_from_text(content)

    extract_prompt = (
        f"From the chord content below for '{title}' by '{artist}', extract the structured chord chart.\n"
        f"CRITICAL RULES:\n"
        f"- 'has_chart': set to true ONLY if there are explicit chord progressions for this song. If only lyrics, empty text, or chord diagrams without song progression, set to false.\n"
        f"- 'title_matches': set to true ONLY if this content is for '{title}' by '{artist}'. If another song or unrelated page, set to false.\n"
        f"- 'capo_fret': capo fret number as integer (0 if none or not mentioned, look for 'Capo', '变调夹', 'X品', '夹X').\n"
        f"- 'shapes_key': key of the chord shapes as written/fingered on the page (选调, e.g. 'C', 'G', 'D').\n"
        f"- 'stated_original_key': explicitly stated original song key if present ('原调', 'Key:', '1='), e.g. 'D', 'Bb', 'A'. Do NOT calculate the sounding key yourself.\n"
        f"- Group chords into individual measures (bars). Usually 1-2 chords per bar (e.g. [['C'], ['Em7']] or [['1'], ['5/7']]).\n"
        f"- Extract ALL sections in full song performance order (前奏, 主歌, 预副歌, 副歌, 间奏, 桥段, 尾奏).\n"
        f"- Do NOT invent or label sections with '示例' or 'example'.\n"
        f"- Do NOT include lyrics.\n\n"
        f"Content:\n{content[:6000]}"
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=extract_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_ExtractedChart,
                temperature=0.0,
                max_output_tokens=3500,
                http_options=types.HttpOptions(timeout=30000),
            ),
        )
        if not resp.text:
            return None
        parsed = _ExtractedChart.model_validate_json(resp.text)
        if not parsed.has_chart or not parsed.title_matches:
            return None

        chart_dict = parsed.model_dump()
        chart_dict["url"] = real_url
        if not chart_dict.get("title"):
            chart_dict["title"] = get_registered_domain(real_url)

        # Apply regex marker overrides / validations
        if markers.get("capo_fret") is not None:
            chart_dict["capo_fret"] = markers["capo_fret"]
        if markers.get("stated_original_key") and not chart_dict.get("stated_original_key"):
            chart_dict["stated_original_key"] = markers["stated_original_key"]
        if markers.get("shapes_key") and not chart_dict.get("shapes_key"):
            chart_dict["shapes_key"] = markers["shapes_key"]

        return normalize_source_chart(
            chart_dict,
            default_key_hint=default_key,
            min_bars=16,
            min_sections=2,
            allow_test_urls=False,
        )
    except Exception as exc:
        logger.debug("Extraction failed for %s from %s: %s", title, real_url, exc)
        return None


def _process_candidate_url(
    client: Any,
    model: str,
    raw_url: str,
    title: str,
    artist: str,
    default_key: Optional[str] = None,
) -> Optional[SourceChart]:
    """Resolve redirect, fetch in-memory, and extract SourceChart with quality gate check."""
    real_url = resolve_redirect_url(raw_url)
    if not is_valid_source_url(real_url, allow_test_urls=False):
        return None

    # Check per-URL cache first (structured fields only)
    cached_chart = load_cached_url_chart(real_url)
    if cached_chart is not None:
        logger.debug("Loaded cached SourceChart for %s", real_url)
        return cached_chart

    content = _fetch_page_content_in_memory(client, model, real_url, title, artist)
    if not content:
        return None
    chart = _extract_chart_from_content(
        client, model, content, real_url, title, artist, default_key=default_key
    )
    if chart is not None:
        save_cached_url_chart(real_url, chart)
    return chart


def _execute_grounded_search(
    client: Any,
    model: str,
    q: str,
) -> list[str]:
    """Execute a single Google grounded search query with Gemini and return candidate web URIs."""
    from google.genai import types

    try:
        resp = client.models.generate_content(
            model=model,
            contents=q,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
                max_output_tokens=3000,
                http_options=types.HttpOptions(timeout=30000),
            ),
        )
        uris: list[str] = []
        if resp.candidates and resp.candidates[0].grounding_metadata:
            gm = resp.candidates[0].grounding_metadata
            for c in gm.grounding_chunks or []:
                if c and c.web and c.web.uri:
                    uris.append(c.web.uri)
        return uris
    except Exception as e:
        logger.debug("Query '%s' failed: %s", q, e)
        return []


# --------------------------------------------------------------------------- Public API
def search_song(
    title: str = "",
    artist: str = "",
    *,
    min_sources: int = 3,
    timeout_s: float = 90.0,
) -> SearchResult:
    """Search multiple public chord chart sources on the web and return consensus.

    Args:
        title: Song title.
        artist: Optional artist name.
        min_sources: Target minimum independent sources.
        timeout_s: Overall timeout in seconds.

    Returns:
        SearchResult containing structured SourceCharts and consensus sections.
        Never raises exceptions; returns empty result with warnings on failure.
    """
    clean_title = title.strip()
    clean_artist = artist.strip()

    # 1. Offline Mode Check
    if os.environ.get("QA_OFFLINE") == "1":
        return SearchResult(
            title=clean_title,
            artist=clean_artist,
            warnings=["QA_OFFLINE=1: network search disabled"],
        )

    if not clean_title and not clean_artist:
        return SearchResult(warnings=["No title or artist provided"])

    # 2. Cache Check
    cached = load_cached_search(clean_title, clean_artist)
    if cached is not None:
        return cached

    start_time = time.time()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)
    location = os.environ.get("WEB_EVIDENCE_LOCATION", DEFAULT_LOCATION)
    model = os.environ.get("SMART_SEARCH_MODEL", DEFAULT_MODEL)

    try:
        from google import genai

        client = genai.Client(vertexai=True, project=project, location=location)
    except Exception as exc:
        logger.warning("Failed initializing Gemini client: %s", exc)
        return SearchResult(
            title=clean_title,
            artist=clean_artist,
            warnings=[f"Failed initializing Gemini client: {exc}"],
        )

    # 3. Grounded Search queries to discover candidate URIs
    queries = [
        f"《{clean_title}》 {clean_artist} 吉他谱 和弦谱 原调 选调 变调夹 前奏 主歌 副歌",
        f"{clean_title} {clean_artist} chords tabs intro verse chorus chords per bar ultimate-guitar guitarians chord4",
        f"《{clean_title}》 {clean_artist} 和弦走向 简谱 段落 小节",
    ]

    candidate_uris: list[str] = []
    search_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    try:
        futs = [search_executor.submit(_execute_grounded_search, client, model, q) for q in queries]
        for fut in concurrent.futures.as_completed(futs, timeout=max(10.0, timeout_s * 0.4)):
            try:
                uris = fut.result()
                candidate_uris.extend(uris)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Initial search phase encountered error: %s", exc)
    finally:
        search_executor.shutdown(wait=False, cancel_futures=True)

    # 4. Resolve and filter candidate URLs across distinct domains
    seen_raw_urls: set[str] = set()
    unique_candidate_uris: list[str] = []
    for u in candidate_uris:
        if u and u not in seen_raw_urls:
            seen_raw_urls.add(u)
            unique_candidate_uris.append(u)

    accepted_sources: list[SourceChart] = []
    accepted_domains: set[str] = set()

    # Process candidates prioritizing domain diversity
    if unique_candidate_uris:
        cand_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            futs = [
                cand_executor.submit(_process_candidate_url, client, model, u, clean_title, clean_artist)
                for u in unique_candidate_uris[:12]
            ]
            for fut in concurrent.futures.as_completed(futs, timeout=max(15.0, timeout_s * 0.45)):
                try:
                    chart = fut.result()
                    if chart and chart.sections:
                        domain = get_registered_domain(chart.url)
                        if domain not in accepted_domains:
                            accepted_domains.add(domain)
                            accepted_sources.append(chart)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Candidate processing phase encountered error: %s", exc)
        finally:
            cand_executor.shutdown(wait=False, cancel_futures=True)

    # 5. Targeted domain fallback if < min_sources and time remaining
    time_spent = time.time() - start_time
    time_left = timeout_s - time_spent
    if len(accepted_sources) < min_sources and time_left > 15.0:
        target_sites = [
            "tabs.ultimate-guitar.com",
            "chord4.com",
            "hopamchuan.com",
            "guitarians.com",
            "chords-haven.blogspot.com",
            "tan8.com",
        ]
        sites_to_try = [
            s for s in target_sites if not any(s.endswith(ad) or ad.endswith(s) for ad in accepted_domains)
        ][:4]

        if sites_to_try:
            site_queries = [
                f"site:{site} {clean_title} {clean_artist} chords"
                for site in sites_to_try
            ]
            site_executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(site_queries), 3))
            site_candidate_uris: list[str] = []
            try:
                futs = [
                    site_executor.submit(_execute_grounded_search, client, model, sq)
                    for sq in site_queries
                ]
                for fut in concurrent.futures.as_completed(futs, timeout=min(12.0, time_left * 0.4)):
                    try:
                        uris = fut.result()
                        site_candidate_uris.extend(uris)
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Targeted site search phase encountered error: %s", exc)
            finally:
                site_executor.shutdown(wait=False, cancel_futures=True)

            if site_candidate_uris:
                t_executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(len(site_candidate_uris), 4))
                try:
                    t_futs = [
                        t_executor.submit(_process_candidate_url, client, model, u, clean_title, clean_artist)
                        for u in site_candidate_uris[:6]
                    ]
                    for fut in concurrent.futures.as_completed(t_futs, timeout=min(18.0, time_left * 0.55)):
                        try:
                            chart = fut.result()
                            if chart and chart.sections:
                                domain = get_registered_domain(chart.url)
                                if domain not in accepted_domains:
                                    accepted_domains.add(domain)
                                    accepted_sources.append(chart)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.debug("Targeted candidate processing phase encountered error: %s", exc)
                finally:
                    t_executor.shutdown(wait=False, cancel_futures=True)

    # 6. Build Consensus (NO padding, NO fake charts)
    warnings: list[str] = []
    if not accepted_sources:
        warnings.append("No valid chord charts could be extracted from web sources")
        return SearchResult(
            title=clean_title,
            artist=clean_artist,
            warnings=warnings,
        )

    res = build_consensus(accepted_sources, title=clean_title, artist=clean_artist)

    # 7. Save Cache
    save_cached_search(res)
    return res


def search_result_to_lead(res: SearchResult) -> LeadSheet:
    """Convert SearchResult into LeadSheet data contract.

    - chords_only=True
    - measures from consensus sections
    - section label on first bar of each section
    - default 4/4 beats (or matching time_signature)
    - LeadChord.agreement = 'agree' if support>=2 and support/total>=0.5 else 'search_only'
    - alternatives filled
    - provenance: '网络来源整理（N 个来源）· 未经音频核对'
    - sources: SourceLink(title, url)
    """
    time_sig = res.time_signature or "4/4"
    beats_per_measure = 4.0
    m_sig = re.match(r"^(\d+)/(\d+)$", time_sig)
    if m_sig:
        numerator = float(m_sig.group(1))
        denominator = float(m_sig.group(2))
        beats_per_measure = numerator * (4.0 / denominator)

    provenance_str = f"网络来源整理（{len(res.sources)} 个来源）· 未经音频核对"
    source_links = [
        SourceLink(title=s.title or s.url, url=s.url)
        for s in res.sources
    ]

    measures: list[LeadMeasure] = []
    for sec in res.consensus_sections:
        for bar_idx, bar in enumerate(sec.bars):
            m = LeadMeasure(beats=beats_per_measure)
            if bar_idx == 0:
                m.section = sec.label
            else:
                m.section = None

            agree_status: Agreement = (
                "agree"
                if (bar.support >= 2 and (bar.support / max(1, bar.total)) >= 0.5)
                else "search_only"
            )

            if len(bar.chords) == 1:
                m.chords.append(
                    LeadChord(
                        raw=bar.chords[0],
                        beat=1.0,
                        agreement=agree_status,
                        alternatives=bar.alternatives,
                    )
                )
            elif len(bar.chords) == 2:
                m.chords.append(
                    LeadChord(
                        raw=bar.chords[0],
                        beat=1.0,
                        agreement=agree_status,
                        alternatives=bar.alternatives,
                    )
                )
                m.chords.append(
                    LeadChord(
                        raw=bar.chords[1],
                        beat=3.0 if beats_per_measure == 4.0 else (1.0 + beats_per_measure / 2.0),
                        agreement=agree_status,
                        alternatives=bar.alternatives,
                    )
                )
            elif len(bar.chords) > 2:
                step = beats_per_measure / len(bar.chords)
                for i, chord_str in enumerate(bar.chords):
                    beat_val = 1.0 + float(i) * step
                    m.chords.append(
                        LeadChord(
                            raw=chord_str,
                            beat=beat_val,
                            agreement=agree_status,
                            alternatives=bar.alternatives,
                        )
                    )

            measures.append(m)

    return LeadSheet(
        title=res.title,
        artist=res.artist,
        key=res.key or "C",
        time_signature=time_sig,
        tempo_bpm=res.tempo_bpm,
        measures=measures,
        chords_only=True,
        provenance=provenance_str,
        sources=source_links,
    )
