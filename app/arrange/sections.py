"""Song-form analysis and section planner for accompaniments."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.models import ParsedSheet, SectionPlan, SectionRole, System

logger = logging.getLogger(__name__)

# In-memory hash cache for LLM section analysis
_LLM_SECTIONS_CACHE: dict[str, list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]]] = {}

# Mainland / Western standard song-form labels
MAINLAND_LABEL_PATTERNS: list[tuple[re.Pattern[str], SectionRole, int]] = [
    (re.compile(r"前奏|intro", re.I), "intro", 0),
    (re.compile(r"导歌|pre-?chorus", re.I), "prechorus", 2),
    (re.compile(r"桥段", re.I), "prechorus", 2),
    (re.compile(r"副歌|chorus|hook", re.I), "chorus", 2),
    (re.compile(r"主歌|verse", re.I), "verse", 1),
    (re.compile(r"间奏|interlude", re.I), "interlude", 1),
    (re.compile(r"尾奏|ending|coda|outro", re.I), "outro", 0),
]

# Taiwanese / lead-sheet instrumentation cues
SPARSE_CUE_PATTERNS = [
    re.compile(r"only\s+[a-z\(\)]+", re.I),
    re.compile(r"only\s*\(?鐵琴\)?", re.I),
]

DENSE_CUE_PATTERNS = [
    re.compile(r"\+ag", re.I),
    re.compile(r"\+bs", re.I),
    re.compile(r"strings\s*in", re.I),
    re.compile(r"\bog\b", re.I),
    re.compile(r"dr\s*in", re.I),
    re.compile(r"all\s*in", re.I),
    re.compile(r"tempo", re.I),
]


class LLMSectionItem(BaseModel):
    start_measure: int = Field(description="0-based start measure index")
    end_measure: int = Field(description="Inclusive end measure index")
    role: Literal["intro", "verse", "prechorus", "chorus", "bridge", "interlude", "outro"]
    energy: int = Field(ge=0, le=3)
    evidence: str = Field(default="", description="One-line evidence, e.g. repeated lyrics")


class LLMSectionsResponse(BaseModel):
    sections: list[LLMSectionItem]


def parse_label(label: Optional[str]) -> tuple[Optional[SectionRole], Optional[int], bool]:
    """Parse section label string.

    Returns:
        (form_role, energy_hint, is_authoritative_form_label)
    """
    if not label or not isinstance(label, str):
        return None, None, False

    clean = label.strip()

    # 1. Check Mainland form labels
    for pattern, role, default_energy in MAINLAND_LABEL_PATTERNS:
        if pattern.search(clean):
            return role, default_energy, True

    # 2. Check instrumentation cues
    is_sparse = any(p.search(clean) for p in SPARSE_CUE_PATTERNS)
    is_dense = any(p.search(clean) for p in DENSE_CUE_PATTERNS)

    if is_dense:
        return None, 2, False
    if is_sparse:
        return None, 0, False

    return None, None, False


def _normalize_lyrics(text: Optional[str]) -> str:
    """Strip punctuation and whitespace from lyrics text."""
    if not text:
        return ""
    return re.sub(r"[\s\d.,!?;:\"\'/()（）\-~～]+", "", text)


def _normalize_melody(melody: Optional[str]) -> tuple[str, ...]:
    """Extract pitch degree sequence from melody string."""
    if not melody:
        return ()
    return tuple(re.findall(r"[1-7]", melody))


def _measure_chord_signature(measure: Any) -> tuple[str, ...]:
    """Extract chord degree progression tuple for a measure."""
    chords = measure.chords
    if not chords:
        return ()
    sigs: list[str] = []
    for c in chords:
        raw = c.raw.strip()
        m = re.match(r"^([1-7][b#♭♯]?(?:m|M|sus|add)?)", raw)
        sigs.append(m.group(1) if m else raw[:2])
    return tuple(sigs)


def _load_cached_sections(
    cache_key: str,
) -> Optional[list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]]]:
    if cache_key in _LLM_SECTIONS_CACHE:
        return _LLM_SECTIONS_CACHE[cache_key]

    cache_dir = Path("out/sections/cache")
    local_path = cache_dir / f"{cache_key}.json"
    content: Optional[str] = None
    if local_path.is_file():
        try:
            content = local_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Error reading local sections cache %s: %s", local_path, exc)

    if not content:
        bucket_name = os.environ.get("SHEET_BUCKET")
        if bucket_name:
            try:
                from google.cloud import storage

                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"sections_cache/{cache_key}.json")
                if blob.exists():
                    content = blob.download_as_text()
                    try:
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_text(content, encoding="utf-8")
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Failed reading sections from GCS bucket %s: %s", bucket_name, exc)

    if content:
        try:
            raw_list = json.loads(content)
            sections: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]] = [
                (item["start_measure"], item["end_measure"], item["role"], item["energy"], "llm")
                for item in raw_list
            ]
            _LLM_SECTIONS_CACHE[cache_key] = sections
            return sections
        except Exception as exc:
            logger.warning("Error parsing cached sections json: %s", exc)

    return None


def _save_cached_sections(
    cache_key: str,
    sections: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]],
) -> None:
    _LLM_SECTIONS_CACHE[cache_key] = sections
    payload = [
        {"start_measure": s[0], "end_measure": s[1], "role": s[2], "energy": s[3]}
        for s in sections
    ]
    content = json.dumps(payload, ensure_ascii=False, indent=2)

    cache_dir = Path("out/sections/cache")
    local_path = cache_dir / f"{cache_key}.json"
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.warning("Error writing local sections cache %s: %s", local_path, exc)

    bucket_name = os.environ.get("SHEET_BUCKET")
    if bucket_name:
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"sections_cache/{cache_key}.json")
            blob.upload_from_string(content, content_type="application/json")
        except Exception as exc:
            logger.debug("Failed writing sections to GCS bucket %s: %s", bucket_name, exc)


def _repair_llm_sections(
    sections: list[LLMSectionItem],
    total_measures: int,
) -> Optional[list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]]]:
    """Validate and repair gaps / overlaps in LLM section proposals."""
    if not sections or total_measures <= 0:
        return None

    valid_roles = {"intro", "verse", "prechorus", "chorus", "bridge", "interlude", "outro"}
    cleaned: list[LLMSectionItem] = []
    for s in sections:
        if s.role not in valid_roles:
            continue
        st = max(0, min(total_measures - 1, s.start_measure))
        end = max(0, min(total_measures - 1, s.end_measure))
        if st > end:
            continue
        e = max(0, min(3, s.energy))
        cleaned.append(
            LLMSectionItem(
                start_measure=st,
                end_measure=end,
                role=s.role,
                energy=e,
                evidence=s.evidence,
            )
        )

    if not cleaned:
        return None

    cleaned.sort(key=lambda s: (s.start_measure, s.end_measure))

    # Repair start gap
    if cleaned[0].start_measure > 0:
        gap = cleaned[0].start_measure
        if gap <= 4:
            cleaned[0].start_measure = 0
        else:
            cleaned.insert(
                0,
                LLMSectionItem(
                    start_measure=0,
                    end_measure=gap - 1,
                    role="intro",
                    energy=0,
                    evidence="initial gap repair",
                ),
            )

    repaired: list[LLMSectionItem] = [cleaned[0]]
    for curr in cleaned[1:]:
        prev = repaired[-1]
        if curr.start_measure <= prev.end_measure:
            curr.start_measure = prev.end_measure + 1
            if curr.start_measure > curr.end_measure:
                continue
        elif curr.start_measure > prev.end_measure + 1:
            gap_len = curr.start_measure - (prev.end_measure + 1)
            if gap_len <= 4:
                prev.end_measure = curr.start_measure - 1
            else:
                repaired.append(
                    LLMSectionItem(
                        start_measure=prev.end_measure + 1,
                        end_measure=curr.start_measure - 1,
                        role="interlude",
                        energy=1,
                        evidence="gap repair",
                    )
                )
        repaired.append(curr)

    # Repair end gap
    last = repaired[-1]
    if last.end_measure < total_measures - 1:
        gap = (total_measures - 1) - last.end_measure
        if gap <= 4:
            last.end_measure = total_measures - 1
        else:
            repaired.append(
                LLMSectionItem(
                    start_measure=last.end_measure + 1,
                    end_measure=total_measures - 1,
                    role="outro",
                    energy=0,
                    evidence="trailing gap repair",
                )
            )

    if not repaired:
        return None
    if repaired[0].start_measure != 0 or repaired[-1].end_measure != total_measures - 1:
        return None
    for idx in range(1, len(repaired)):
        if repaired[idx].start_measure != repaired[idx - 1].end_measure + 1:
            return None

    return [(s.start_measure, s.end_measure, s.role, s.energy, "llm") for s in repaired]


def _call_llm_sections(
    sheet: ParsedSheet,
    systems: list[System],
    measures: list[Any],
    total_measures: int,
) -> Optional[list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]]]:
    from app.qa.llm import ask_json

    m_to_row_label: dict[int, str] = {}
    m_to_page: dict[int, int] = {}
    for s in systems:
        lbl = s.section_label or "None"
        p = s.page
        for m in s.measures:
            m_to_row_label[m.index] = lbl
            m_to_page[m.index] = p

    lines: list[str] = []
    for m in measures:
        lbl = m_to_row_label.get(m.index, "None")
        page = m_to_page.get(m.index, 0)
        chords = " ".join(c.raw for c in m.chords) if m.chords else "-"
        melody = m.melody.strip() if m.melody and m.melody.strip() else "-"
        lyrics = m.lyrics.strip() if m.lyrics and m.lyrics.strip() else "-"
        flags = []
        if m.fill:
            flags.append("fill")
        if m.is_stop:
            flags.append("stop")
        flag_str = f" flags: {','.join(flags)}" if flags else ""
        lines.append(
            f"m{m.index} (p{page}) | row: [{lbl}] | chords: [{chords}] | melody: [{melody}] | lyrics: [{lyrics}]{flag_str}"
        )

    compact_table = "\n".join(lines)
    cache_key = hashlib.sha256(compact_table.encode("utf-8")).hexdigest()

    cached = _load_cached_sections(cache_key)
    if cached is not None:
        return cached

    header = sheet.header
    title = header.title or "Unknown"
    time_sig = header.time_signature or "4/4"
    style = header.style or ""
    kc_info = ", ".join(f"m{kc.at_measure}:{kc.raw}" for kc in sheet.key_changes) if sheet.key_changes else "None"

    prompt = (
        f"You are an expert music arranger and song-form analyst.\n"
        f"Song Title: {title}\n"
        f"Style/Time: {time_sig} {style}\n"
        f"Key Changes: {kc_info}\n"
        f"Total Measures: {total_measures} (indices 0 to {total_measures - 1})\n\n"
        f"Measure Breakdown (one line per measure):\n"
        f"{compact_table}\n\n"
        f"Song-Form Analysis Rules:\n"
        f"1. You may leverage your knowledge of the actual song (most are well-known pop songs) and repeated-lyrics cue:\n"
        f"   - Chorus: melody AND lyrics repeat across instances; emotional core / high energy.\n"
        f"   - Verse: melody repeats (or similar contour) but with new lyrics.\n"
        f"   - Pre-chorus: transitional build-up between verse and chorus; increasing tension.\n"
        f"   - Bridge: contrasting musical material appearing after one or two choruses.\n"
        f"   - Intro / Outro / Interlude: lyric-less sections at the start, end, or between vocal parts.\n"
        f"2. Note on uploads: page 2 of some uploads is an alternate version of the same song (e.g. chord-only 弹唱版) — the song form restarts there.\n"
        f"3. Printed mainland labels ('前奏', '主歌', '副歌', '间奏', '尾奏') are authoritative when present.\n"
        f"4. Segment the entire piece into contiguous non-overlapping sections covering measure 0 to {total_measures - 1}.\n"
        f"   - Legal roles: 'intro', 'verse', 'prechorus', 'chorus', 'bridge', 'interlude', 'outro'.\n"
        f"   - Energy: integer 0 to 3 (0=softest/intro/outro, 1=verse, 2=chorus/prechorus, 3=climax/last chorus/after key change).\n"
        f"   - Provide one-line evidence for each section.\n"
    )

    try:
        timeout_seconds = float(5 * 9)
        resp: LLMSectionsResponse = ask_json(
            prompt=prompt,
            schema=LLMSectionsResponse,
            role="arbiter",
            timeout_s=timeout_seconds,
        )
        repaired = _repair_llm_sections(resp.sections, total_measures)
        if repaired:
            _save_cached_sections(cache_key, repaired)
            return repaired
        logger.warning("LLM sections failed validation or repair; falling back to heuristic")
    except Exception as exc:
        logger.warning("LLM section planning failed (%s); falling back to heuristic", exc)

    return None


def plan_sections(sheet: ParsedSheet, *, use_llm: bool | None = None) -> list[SectionPlan]:
    """Plan song-form roles and energy levels for all measures in the sheet.

    Returns:
        List of SectionPlan, exactly one per measure in global order. Never raises.
    """
    try:
        return _plan_sections_impl(sheet, use_llm=use_llm)
    except Exception as exc:
        logger.error("plan_sections unexpected failure: %s", exc, exc_info=True)
        measures = sheet.measures()
        plans: list[SectionPlan] = []
        n = len(measures)
        for i, m in enumerate(measures):
            plans.append(
                SectionPlan(
                    measure_index=m.index,
                    section_id="A1",
                    role="verse",
                    energy=1,
                    is_section_start=(i == 0),
                    is_section_end=(i == n - 1),
                    source="heuristic",
                )
            )
        return plans


def _should_use_llm(use_llm: bool | None) -> bool:
    if use_llm is False:
        return False
    if os.environ.get("QA_OFFLINE") == "1":
        return False
    if os.environ.get("SECTIONS_LLM") == "0":
        return False
    if use_llm is True:
        return True
    return True


def _post_validate_instrumental_lyrics(
    sheet: ParsedSheet,
    raw_sections: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]],
) -> list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]]:
    """Generic post-validation: instrumental roles may not cover >= 2 lyric measures.

    A run of >= 2 lyric measures inside an intro/interlude/outro is split off and
    given the role of the following sung section (or 'verse' if none), with boundary
    snapped to the row (system) start. A single stray lyric measure is tolerated.
    """
    measures = sheet.measures()
    if not raw_sections or not measures:
        return raw_sections

    has_lyr = [bool(_normalize_lyrics(m.lyrics)) for m in measures]

    m_to_sys_start: dict[int, int] = {}
    for s in sheet.systems:
        if s.measures:
            s_st = s.measures[0].index
            for m in s.measures:
                m_to_sys_start[m.index] = s_st

    result: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]] = []
    any_split = False

    for idx, (st, end, role, energy, source) in enumerate(raw_sections):
        if role not in ("intro", "interlude", "outro"):
            result.append((st, end, role, energy, source))
            continue

        lyric_runs: list[tuple[int, int]] = []
        in_run = False
        r_st = -1
        for m_idx in range(st, end + 1):
            if has_lyr[m_idx]:
                if not in_run:
                    in_run = True
                    r_st = m_idx
            else:
                if in_run:
                    lyric_runs.append((r_st, m_idx - 1))
                    in_run = False
        if in_run:
            lyric_runs.append((r_st, end))

        valid_runs = [r for r in lyric_runs if (r[1] - r[0] + 1) >= 2]
        if not valid_runs:
            result.append((st, end, role, energy, source))
            continue

        any_split = True
        first_run_st = valid_runs[0][0]
        snap_st = m_to_sys_start.get(first_run_st, first_run_st)
        snap_st = max(st, snap_st)

        next_sung_role: SectionRole = "verse"
        next_sung_energy = 1
        for future_idx in range(idx + 1, len(raw_sections)):
            f_role = raw_sections[future_idx][2]
            if f_role not in ("intro", "interlude", "outro"):
                next_sung_role = f_role
                next_sung_energy = raw_sections[future_idx][3]
                break

        if snap_st > st:
            result.append((st, snap_st - 1, role, energy, source))
            result.append((snap_st, end, next_sung_role, next_sung_energy, source))
        else:
            result.append((st, end, next_sung_role, next_sung_energy, source))

    if not any_split:
        return raw_sections

    merged: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]] = []
    for s in result:
        if merged and merged[-1][2] == s[2] and merged[-1][3] == s[3] and merged[-1][1] + 1 == s[0]:
            prev = merged[-1]
            merged[-1] = (prev[0], s[1], prev[2], prev[3], prev[4])
        else:
            merged.append(s)

    return merged


def _plan_sections_impl(sheet: ParsedSheet, *, use_llm: bool | None = None) -> list[SectionPlan]:
    measures = sheet.measures()
    total_measures = len(measures)
    if total_measures == 0:
        return []

    systems = sheet.systems
    if not systems or len(systems) == 1:
        lbl_role, lbl_energy, _ = parse_label(systems[0].section_label if systems else None)
        role = lbl_role or "verse"
        energy = lbl_energy if lbl_energy is not None else 1
        return _build_final_section_plans(
            measures, [(0, total_measures - 1, role, energy, "heuristic")]
        )

    # 1. Check if LLM should be used
    if _should_use_llm(use_llm):
        llm_sections = _call_llm_sections(sheet, systems, measures, total_measures)
        if llm_sections:
            validated = _post_validate_instrumental_lyrics(sheet, llm_sections)
            return _build_final_section_plans(measures, validated)

    # 2. Heuristic fallback
    has_any_mainland = False
    sys_mainland: list[Optional[tuple[SectionRole, int, str]]] = []
    for s in systems:
        lbl = s.section_label or ""
        match = None
        for pat, role, e in MAINLAND_LABEL_PATTERNS:
            if pat.search(lbl):
                match = (role, e, lbl)
                has_any_mainland = True
                break
        sys_mainland.append(match)

    key_change_ms = {kc.at_measure for kc in sheet.key_changes}

    if has_any_mainland:
        # Mainland labeled chart (printed form labels are authoritative)
        raw_sections: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]] = []
        curr_role: Optional[SectionRole] = None
        curr_energy = 0
        curr_start = 0

        for i, s in enumerate(systems):
            s_m_start = s.measures[0].index
            lbl_match = sys_mainland[i]
            lbl_str = s.section_label or ""
            is_dense = any(p.search(lbl_str) for p in DENSE_CUE_PATTERNS)

            if lbl_match:
                if curr_role is not None:
                    raw_sections.append((curr_start, s_m_start - 1, curr_role, curr_energy, "label"))
                curr_role = lbl_match[0]
                curr_energy = lbl_match[1]
                if curr_role == "chorus" and any(kc <= s_m_start for kc in key_change_ms):
                    curr_energy = 3
                curr_start = s_m_start
            elif is_dense:
                if curr_role is not None:
                    raw_sections.append((curr_start, s_m_start - 1, curr_role, curr_energy, "heuristic"))
                curr_role = "chorus"
                curr_energy = 3 if any(kc <= s_m_start for kc in key_change_ms) else 2
                curr_start = s_m_start
            elif curr_role is None:
                curr_role = "intro"
                curr_energy = 0
                curr_start = s_m_start

        if curr_role is not None:
            raw_sections.append((curr_start, total_measures - 1, curr_role, curr_energy, "label"))

    else:
        # Generic music-theory segmentation for unlabeled / instrumentation-only charts
        raw_sections = _segment_unlabeled_song(sheet, systems, measures, total_measures, key_change_ms)

    validated = _post_validate_instrumental_lyrics(sheet, raw_sections)
    return _build_final_section_plans(measures, validated)


def _segment_unlabeled_song(
    sheet: ParsedSheet,
    systems: list[System],
    measures: list[Any],
    total_measures: int,
    key_change_ms: set[int],
) -> list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]]:
    """Generic music-theory segmentation for unlabeled or instrumentation-only charts."""
    has_lyr = [bool(_normalize_lyrics(m.lyrics)) for m in measures]
    clean_lyr = [_normalize_lyrics(m.lyrics) for m in measures]

    sys_ranges = [(s.measures[0].index, s.measures[-1].index) for s in systems]
    sys_has_lyr = [any(has_lyr[m_idx] for m_idx in range(st, end + 1)) for st, end in sys_ranges]
    sys_count = len(systems)

    # 1. Leading lyric-less systems -> Intro
    first_lyr_sys = next((i for i, h in enumerate(sys_has_lyr) if h), sys_count)
    intro_end_m = sys_ranges[first_lyr_sys - 1][1] if first_lyr_sys > 0 else -1

    # 2. Trailing lyric-less systems -> Outro
    last_lyr_sys = max((i for i, h in enumerate(sys_has_lyr) if h), default=-1)
    outro_sys_start = sys_count
    if 0 <= last_lyr_sys < sys_count - 1:
        outro_sys_start = last_lyr_sys + 1

    outro_start_m = sys_ranges[outro_sys_start][0] if outro_sys_start < sys_count else total_measures

    # Build per-system lyrics signatures to detect repeated chorus lyrics
    sys_lyrics: list[str] = []
    for st, end in sys_ranges:
        s_lyr = "".join(clean_lyr[m_idx] for m_idx in range(st, end + 1))
        sys_lyrics.append(s_lyr)

    # Identify repeated lyric systems (generic chorus detection)
    chorus_sys_set: set[int] = set()
    for i in range(first_lyr_sys, outro_sys_start):
        lyr_i = sys_lyrics[i]
        if len(lyr_i) < 4:
            continue
        for j in range(first_lyr_sys, i):
            lyr_j = sys_lyrics[j]
            if len(lyr_j) < 4:
                continue
            set_i = set(lyr_i)
            set_j = set(lyr_j)
            overlap = len(set_i & set_j) / max(len(set_i), len(set_j))
            if lyr_i in lyr_j or lyr_j in lyr_i or overlap >= 0.7:
                chorus_sys_set.add(i)
                chorus_sys_set.add(j)

    raw_sections: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]] = []
    if intro_end_m >= 0:
        raw_sections.append((0, intro_end_m, "intro", 0, "heuristic"))

    curr_role: Optional[SectionRole] = None
    curr_st = intro_end_m + 1
    curr_e = 1

    i = first_lyr_sys
    while i < outro_sys_start:
        st, end = sys_ranges[i]
        lbl = systems[i].section_label or ""
        lbl_role, lbl_energy, is_auth = parse_label(lbl)

        # Middle lyric-less run >= 2 bars -> interlude
        sys_len = end - st + 1
        if not sys_has_lyr[i] and sys_len >= 2:
            if curr_role is not None and curr_st < st:
                raw_sections.append((curr_st, st - 1, curr_role, curr_e, "heuristic"))
            raw_sections.append((st, end, "interlude", 1, "heuristic"))
            curr_role = None
            curr_st = end + 1
            curr_e = 1
            i += 1
            continue

        # Check explicit authoritative label
        if is_auth and lbl_role:
            r = lbl_role
            e = lbl_energy if lbl_energy is not None else 1
        else:
            is_dense = any(p.search(lbl) for p in DENSE_CUE_PATTERNS)
            is_sparse = any(p.search(lbl) for p in SPARSE_CUE_PATTERNS)
            is_pre = bool(re.search(r"鼓邊|rimshot|\brim\b|导歌|pre-?chorus", lbl, re.I))

            if i in chorus_sys_set or (is_dense and not is_pre):
                r = "chorus"
                e = 3 if any(kc <= st for kc in key_change_ms) else 2
            elif is_pre:
                r = "prechorus"
                e = 2
            elif is_sparse:
                r = "verse"
                e = 0
            elif curr_role in ("chorus", "bridge", "prechorus"):
                r = curr_role
                e = curr_e
            else:
                r = "verse"
                e = 1

        is_new_section = (
            (r != curr_role)
            or (r == "chorus" and curr_role == "chorus" and (st - curr_st >= 8))
            or (r == "verse" and curr_role == "verse" and lbl)
        )

        if curr_role is None:
            curr_role = r
            curr_e = e
            curr_st = st
        elif is_new_section:
            raw_sections.append((curr_st, st - 1, curr_role, curr_e, "heuristic"))
            curr_role = r
            curr_e = e
            curr_st = st

        i += 1

    if curr_role is not None and curr_st < outro_start_m:
        raw_sections.append((curr_st, outro_start_m - 1, curr_role, curr_e, "heuristic"))

    if outro_start_m < total_measures:
        raw_sections.append((outro_start_m, total_measures - 1, "outro", 0, "heuristic"))

    return raw_sections


def _build_final_section_plans(
    measures: list[Any],
    raw_sections: list[tuple[int, int, SectionRole, int, Literal["label", "heuristic", "llm"]]],
) -> list[SectionPlan]:
    """Convert raw section spans into SectionPlan instances with stable section_ids."""
    role_counts: dict[str, int] = {
        "intro": 1,
        "verse": 1,
        "prechorus": 1,
        "chorus": 1,
        "bridge": 1,
        "interlude": 1,
        "outro": 1,
    }

    sec_id_map: list[tuple[int, int, str, SectionRole, int, str]] = []
    for st, end, sec_role, energy, source in raw_sections:
        count = role_counts.get(sec_role, 1)
        if sec_role == "verse":
            sec_id = f"A{count}"
        elif sec_role == "chorus":
            sec_id = f"B{count}"
        elif sec_role == "prechorus":
            sec_id = f"prechorus{count}"
        elif sec_role == "bridge":
            sec_id = f"bridge{count}"
        elif sec_role == "intro":
            sec_id = "intro" if count == 1 else f"intro{count}"
        elif sec_role == "interlude":
            sec_id = f"interlude{count}"
        elif sec_role == "outro":
            sec_id = "outro" if count == 1 else f"outro{count}"
        else:
            sec_id = f"{sec_role}{count}"

        role_counts[sec_role] = count + 1
        sec_id_map.append((st, end, sec_id, sec_role, energy, source))

    m_to_sec: dict[int, tuple[str, SectionRole, int, bool, bool, str]] = {}
    for st, end, sec_id, role, energy, source in sec_id_map:
        for m_idx in range(st, end + 1):
            is_start = (m_idx == st)
            is_end = (m_idx == end)
            m_to_sec[m_idx] = (sec_id, role, energy, is_start, is_end, source)

    plans: list[SectionPlan] = []
    for m in measures:
        if m.index in m_to_sec:
            sec_id, role, energy, is_start, is_end, source = m_to_sec[m.index]
        else:
            sec_id, role, energy, is_start, is_end, source = ("A1", "verse", 1, False, False, "heuristic")

        plans.append(
            SectionPlan(
                measure_index=m.index,
                section_id=sec_id,
                role=role,
                energy=energy,
                is_section_start=is_start,
                is_section_end=is_end,
                source=source,
            )
        )

    return plans
