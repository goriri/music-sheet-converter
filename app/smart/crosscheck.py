"""Cross-check and reconciliation of audio transcription with web search consensus (智能创建 M4).

Merges an audio transcription (LeadSheet) and a multi-source web consensus
(SearchResult from app.smart.search).

Key steps:
1. Key reconciliation: harmonize sounding keys; transpose degree chords if needed.
2. Sequence alignment: Needleman-Wunsch bar alignment between audio and search consensus.
3. Chord comparison & section label adoption.
4. Disagreement arbitration: batch LLM call (ask_json) with graceful heuristic fallback.
5. CrossCheckReport generation with provenance, agreement metrics, and source links.

HARD RULE:
- merge() NEVER raises an exception.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.qa.llm import ask_json, llm_available
from app.qa.web_evidence import (
    DegreeChord,
    format_degree_to_number,
    pairwise_chord_score,
    parse_chord_to_degree,
)
from app.smart.models import (
    Agreement,
    LeadChord,
    LeadMeasure,
    LeadNote,
    LeadSheet,
    SourceLink,
)
from app.smart.search import SearchResult, search_result_to_lead
from app.theory.chords import clean_raw_chord
from app.theory.keys import (
    DEGREE_SEMITONES,
    PC_TO_DEFAULT_KEY_NAME,
    key_name_to_pc,
)

logger = logging.getLogger(__name__)

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


class DisagreementItem(BaseModel):
    """Detailed record of a chord disagreement between audio and search consensus."""

    measure_index: int = Field(description="0-based measure index in the lead sheet")
    audio: str = Field(description="Chord reading from audio transcription")
    search: str = Field(description="Chord reading from web consensus")
    chosen: str = Field(description="Final chord selected after arbitration")
    reason: str = Field(description="Explanation of arbitration decision in Chinese")


class CrossCheckReport(BaseModel):
    """Full cross-check report for 智能创建."""

    mode: Literal["audio+search", "audio_only", "search_only"]
    key_decision: str = ""
    agreement_ratio: float = 1.0
    n_agree: int = 0
    n_disagree: int = 0
    n_audio_only: int = 0
    disagreements: list[DisagreementItem] = Field(default_factory=list)
    sources: list[SourceLink] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BarArbitration(BaseModel):
    measure_index: int
    chosen_chord: str
    reason: str


class ArbitrationResult(BaseModel):
    decisions: list[BarArbitration]


def transpose_chord_str(chord_str: str, from_key: str, to_key: str) -> str:
    """Transpose a Taiwanese number chord string from from_key to to_key.

    If chord_str cannot be parsed, returns it unchanged.
    """
    raw = clean_raw_chord(chord_str)
    if not raw:
        return chord_str

    try:
        from_pc = key_name_to_pc(from_key)
        to_pc = key_name_to_pc(to_key)
    except Exception:
        return chord_str

    if from_pc == to_pc:
        return chord_str

    deg_chord = parse_chord_to_degree(raw, tonic_pc=0)
    if deg_chord is None:
        return chord_str

    root_pc = (from_pc + deg_chord.root_semitone) % 12
    bass_pc = (from_pc + deg_chord.bass_semitone) % 12

    new_root_semi = (root_pc - to_pc) % 12
    new_bass_semi = (bass_pc - to_pc) % 12

    return format_degree_to_number(new_root_semi, deg_chord.quality, new_bass_semi)


def transpose_lead_sheet(lead: LeadSheet, to_key: str) -> LeadSheet:
    """Transpose all notes and chords in a LeadSheet to to_key."""
    old_key = lead.key or "C"
    try:
        from_pc = key_name_to_pc(old_key)
        to_pc = key_name_to_pc(to_key)
    except Exception:
        return lead

    if from_pc == to_pc:
        lead.key = to_key
        return lead

    transposed = lead.model_copy(deep=True)
    transposed.key = to_key

    for m in transposed.measures:
        # Transpose notes
        for n in m.notes:
            if 1 <= n.degree <= 7:
                note_pc = (from_pc + DEGREE_SEMITONES[n.degree - 1] + n.accidental) % 12
                new_semi = (note_pc - to_pc) % 12
                new_deg, new_acc = SEMITONE_TO_DEGREE[new_semi]
                n.degree = new_deg
                n.accidental = new_acc

        # Transpose chords
        for c in m.chords:
            c.raw = transpose_chord_str(c.raw, old_key, to_key)
            c.alternatives = [transpose_chord_str(alt, old_key, to_key) for alt in c.alternatives]

    return transposed


def _needleman_score(a: Optional[DegreeChord], s: Optional[DegreeChord]) -> float:
    if a is None and s is None:
        return 0.2
    if a is None or s is None:
        return -0.5
    return pairwise_chord_score(a, s)


def _align_bars_needleman_wunsch(
    audio_chords: list[Optional[DegreeChord]],
    search_chords: list[Optional[DegreeChord]],
    gap_penalty: float = -0.4,
) -> list[tuple[Optional[int], Optional[int]]]:
    """Align audio measures with search bars using Needleman-Wunsch."""
    n = len(audio_chords)
    m = len(search_chords)
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [(None, j) for j in range(m)]
    if m == 0:
        return [(i, None) for i in range(n)]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, n + 1):
        a_dc = audio_chords[i - 1]
        for j in range(1, m + 1):
            s_dc = search_chords[j - 1]
            match_score = _needleman_score(a_dc, s_dc)
            dp[i][j] = max(
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j] + gap_penalty,
                dp[i][j - 1] + gap_penalty,
            )

    # Traceback
    i, j = n, m
    pairs: list[tuple[Optional[int], Optional[int]]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = _needleman_score(audio_chords[i - 1], search_chords[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + match_score)) < 1e-6:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + gap_penalty)) < 1e-6:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1

    pairs.reverse()
    return pairs


def _arbitrate_disagreements(
    disagreeing_items: list[dict[str, Any]],
    audio_lead: LeadSheet,
) -> list[DisagreementItem]:
    """Resolve disagreeing chords via LLM (ask_json) or heuristic fallback."""
    if not disagreeing_items:
        return []

    # Attempt LLM arbitration if available
    if llm_available():
        try:
            prompt_items = []
            for item in disagreeing_items:
                m_idx = item["measure_index"]
                m = audio_lead.measures[m_idx]
                melody_degs = [n.degree for n in m.notes if n.degree > 0]
                prev_chord = (
                    audio_lead.measures[m_idx - 1].chords[0].raw
                    if m_idx > 0 and audio_lead.measures[m_idx - 1].chords
                    else "None"
                )
                next_chord = (
                    audio_lead.measures[m_idx + 1].chords[0].raw
                    if m_idx + 1 < len(audio_lead.measures) and audio_lead.measures[m_idx + 1].chords
                    else "None"
                )
                prompt_items.append({
                    "measure_index": m_idx,
                    "audio_chord": item["audio_chord"].raw,
                    "search_chord": item["search_chord_str"],
                    "search_support": f"{item['search_bar']['support']}/{item['search_bar']['total']}",
                    "melody_degrees": melody_degs,
                    "prev_chord": prev_chord,
                    "next_chord": next_chord,
                })

            prompt = (
                "You are an expert music theorist and lead sheet editor. "
                "Resolve chord discrepancies between audio transcription and web consensus. "
                "For each measure, evaluate melody degrees, audio detected chord, search consensus chord (with support), "
                "and surrounding harmonic context to pick the most appropriate chord.\n\n"
                f"Discrepancies:\n{prompt_items}\n\n"
                "Return JSON with 'decisions': list of {measure_index, chosen_chord, reason}."
            )
            arb_res = ask_json(prompt, ArbitrationResult, role="arbiter", timeout_s=30.0)
            dec_map = {d.measure_index: d for d in arb_res.decisions}

            results: list[DisagreementItem] = []
            for item in disagreeing_items:
                m_idx = item["measure_index"]
                a_chord: LeadChord = item["audio_chord"]
                s_chord_str: str = item["search_chord_str"]
                orig_audio_str = item["orig_audio_str"]

                if m_idx in dec_map:
                    dec = dec_map[m_idx]
                    chosen = dec.chosen_chord.strip()
                    reason = dec.reason
                else:
                    chosen = a_chord.raw
                    reason = "LLM未返回该小节裁决，保留音频和弦"

                loser = s_chord_str if chosen == orig_audio_str else orig_audio_str
                a_chord.raw = chosen
                a_chord.alternatives = [loser]
                a_chord.agreement = "disagree"

                results.append(
                    DisagreementItem(
                        measure_index=m_idx,
                        audio=orig_audio_str,
                        search=s_chord_str,
                        chosen=chosen,
                        reason=reason,
                    )
                )
            return results

        except Exception as exc:
            logger.info("LLM arbitration unavailable or failed: %s; falling back to heuristic", exc)

    # Heuristic fallback:
    # Choose search chord if support >= 2 and audio chord root is not in melody notes of that bar;
    # otherwise keep audio chord.
    results: list[DisagreementItem] = []
    for item in disagreeing_items:
        m_idx = item["measure_index"]
        a_chord: LeadChord = item["audio_chord"]
        s_chord_str: str = item["search_chord_str"]
        s_bar: dict[str, Any] = item["search_bar"]
        orig_audio_str: str = item["orig_audio_str"]
        m = audio_lead.measures[m_idx]

        dc = parse_chord_to_degree(orig_audio_str)
        audio_root_deg = dc.degree if dc else None
        if audio_root_deg is None:
            m_root = re.match(r"^[b#]?([1-7])", orig_audio_str)
            if m_root:
                audio_root_deg = int(m_root.group(1))

        melody_degs = {n.degree for n in m.notes if n.degree > 0}
        support = s_bar.get("support", 1)
        total = s_bar.get("total", 1)

        root_in_melody = (audio_root_deg is not None) and (audio_root_deg in melody_degs)

        if support >= 2 and not root_in_melody:
            chosen = s_chord_str
            loser = orig_audio_str
            reason = f"网络 {support}/{total} 来源一致支持 {s_chord_str}，音频和弦根音不在旋律音中"
        else:
            chosen = orig_audio_str
            loser = s_chord_str
            if root_in_melody:
                reason = f"音频和弦根音 {audio_root_deg} 包含于小节旋律音中，保留音频转录"
            else:
                reason = f"网络支持度不足（{support}/{total}），保留音频转录"

        a_chord.raw = chosen
        a_chord.alternatives = [loser]
        a_chord.agreement = "disagree"

        results.append(
            DisagreementItem(
                measure_index=m_idx,
                audio=orig_audio_str,
                search=s_chord_str,
                chosen=chosen,
                reason=reason,
            )
        )

    return results


def merge(
    audio: Optional[LeadSheet],
    search: Optional[SearchResult],
) -> tuple[LeadSheet, CrossCheckReport]:
    """Merge audio LeadSheet and multi-source web SearchResult into a final verified LeadSheet.

    Guaranteed NEVER to raise an exception.
    """
    try:
        return _merge_internal(audio, search)
    except Exception as exc:
        logger.exception("Unexpected error in crosscheck merge: %s", exc)
        # Fallback to whichever input is available
        warn_msg = f"核对流程异常 ({exc})，已启用安全回退"
        if audio is not None and audio.measures:
            rep = CrossCheckReport(
                mode="audio_only",
                key_decision=f"回退采用音频调性 {audio.key}",
                agreement_ratio=0.0,
                n_audio_only=sum(len(m.chords) for m in audio.measures),
                warnings=[warn_msg],
            )
            return audio, rep
        elif search is not None and search.consensus_sections:
            lead = search_result_to_lead(search)
            rep = CrossCheckReport(
                mode="search_only",
                key_decision=f"回退采用网络调性 {lead.key}",
                agreement_ratio=1.0,
                sources=lead.sources,
                warnings=[warn_msg],
            )
            return lead, rep
        else:
            empty_lead = LeadSheet(title="", artist="", key="C", provenance="创建失败")
            rep = CrossCheckReport(
                mode="search_only",
                key_decision="无有效数据",
                agreement_ratio=0.0,
                warnings=[warn_msg, "音频与网络搜索结果均为空"],
            )
            return empty_lead, rep


def _merge_internal(
    audio: Optional[LeadSheet],
    search: Optional[SearchResult],
) -> tuple[LeadSheet, CrossCheckReport]:
    has_audio = audio is not None and len(audio.measures) > 0
    has_search = (
        search is not None
        and len(search.consensus_sections) > 0
        and any(len(sec.bars) > 0 for sec in search.consensus_sections)
    )

    # 1. Neither provided
    if not has_audio and not has_search:
        lead = LeadSheet(title="", artist="", key="C")
        rep = CrossCheckReport(
            mode="search_only",
            key_decision="未提供输入",
            agreement_ratio=0.0,
            warnings=["未提供音频转录或网络搜索结果"],
        )
        return lead, rep

    # 2. Search-only mode
    if not has_audio and has_search:
        assert search is not None
        lead = search_result_to_lead(search)
        lead.provenance = f"网络来源整理（{len(search.sources)} 个来源）· 未经音频核对"
        rep = CrossCheckReport(
            mode="search_only",
            key_decision=f"采用网络来源调性 {lead.key}",
            agreement_ratio=1.0,
            n_agree=0,
            n_disagree=0,
            n_audio_only=0,
            disagreements=[],
            sources=lead.sources,
            warnings=list(search.warnings),
        )
        return lead, rep

    # 3. Audio-only mode
    if has_audio and not has_search:
        assert audio is not None
        lead = audio.model_copy(deep=True)
        n_chords = 0
        for m in lead.measures:
            for c in m.chords:
                c.agreement = "audio_only"
                n_chords += 1
        lead.provenance = "音频转录 · 未与网络来源核对"
        lead.sources = []
        rep = CrossCheckReport(
            mode="audio_only",
            key_decision=f"采用音频转录调性 {lead.key}",
            agreement_ratio=0.0,
            n_agree=0,
            n_disagree=0,
            n_audio_only=n_chords,
            disagreements=[],
            sources=[],
            warnings=[],
        )
        return lead, rep

    # 4. Audio + Search mode
    assert audio is not None and search is not None
    audio_lead = audio.model_copy(deep=True)
    warnings: list[str] = list(search.warnings)

    # Step 1: Key Reconciliation
    audio_key = audio_lead.key or "C"
    search_key = search.key or "C"

    try:
        a_pc = key_name_to_pc(audio_key)
        s_pc = key_name_to_pc(search_key)
    except Exception:
        a_pc = 0
        s_pc = 0

    target_key = audio_key
    if a_pc == s_pc:
        key_decision = f"音频与网络来源调性一致（{audio_key}）"
    else:
        # Check audio key confidence vs search sources consensus
        audio_key_conf = 1.0
        try:
            audio_key_conf = float(audio_lead.key_confidence)  # type: ignore[attr-defined]
        except AttributeError:
            audio_key_conf = 1.0

        if audio_key_conf < 0.6 and len(search.sources) >= 2:
            # Search key wins
            target_key = search_key
            audio_lead = transpose_lead_sheet(audio_lead, target_key)
            key_decision = (
                f"采用网络来源调性 {search_key}（{len(search.sources)} 个来源一致，"
                f"音频调性 {audio_key} 置信度较低，已将音频转调至 {search_key}）"
            )
        else:
            # Audio key wins (default)
            target_key = audio_key
            key_decision = (
                f"采用音频转录调性 {audio_key}（网络来源为 {search_key}，"
                f"已将网络和弦转调至 {audio_key}）"
            )

    # Step 2: Flatten search consensus bars and transpose to target_key
    flat_search_bars: list[dict[str, Any]] = []
    for sec in search.consensus_sections:
        for b_idx, bar in enumerate(sec.bars):
            t_chords = [transpose_chord_str(c, search_key, target_key) for c in bar.chords]
            t_alts = [transpose_chord_str(c, search_key, target_key) for c in bar.alternatives]
            flat_search_bars.append({
                "section_label": sec.label if b_idx == 0 else None,
                "chords": t_chords,
                "support": bar.support,
                "total": bar.total,
                "alternatives": t_alts,
            })

    # Step 3: Needleman-Wunsch Alignment
    audio_degree_chords: list[Optional[DegreeChord]] = []
    for m in audio_lead.measures:
        if m.chords and m.chords[0].raw:
            audio_degree_chords.append(parse_chord_to_degree(m.chords[0].raw, tonic_pc=0))
        else:
            audio_degree_chords.append(None)

    search_degree_chords: list[Optional[DegreeChord]] = []
    for sb in flat_search_bars:
        if sb["chords"]:
            search_degree_chords.append(parse_chord_to_degree(sb["chords"][0], tonic_pc=0))
        else:
            search_degree_chords.append(None)

    pairs = _align_bars_needleman_wunsch(audio_degree_chords, search_degree_chords)

    # Step 4: Compare per Bar
    n_agree = 0
    n_disagree = 0
    n_audio_only = 0
    disagreeing_items: list[dict[str, Any]] = []
    unaligned_search_count = 0

    for a_idx, s_idx in pairs:
        if a_idx is None:
            if s_idx is not None:
                unaligned_search_count += 1
            continue

        m = audio_lead.measures[a_idx]
        if s_idx is None:
            # Audio measure has no aligned search bar
            if m.chords:
                for c in m.chords:
                    c.agreement = "audio_only"
                n_audio_only += len(m.chords)
            continue

        s_bar = flat_search_bars[s_idx]

        # Adopt section label if audio lacks one
        if m.section is None and s_bar["section_label"]:
            m.section = s_bar["section_label"]

        if m.chords:
            a_chord = m.chords[0]
            if s_bar["chords"]:
                s_chord_str = s_bar["chords"][0]
                c_a = clean_raw_chord(a_chord.raw)
                c_s = clean_raw_chord(s_chord_str)

                if c_a == c_s:
                    a_chord.agreement = "agree"
                    n_agree += 1
                else:
                    a_chord.agreement = "disagree"
                    a_chord.alternatives = [s_chord_str]
                    n_disagree += 1
                    disagreeing_items.append({
                        "measure_index": a_idx,
                        "audio_chord": a_chord,
                        "orig_audio_str": a_chord.raw,
                        "search_chord_str": s_chord_str,
                        "search_bar": s_bar,
                    })
            else:
                a_chord.agreement = "audio_only"
                n_audio_only += 1
        else:
            # Audio measure had no chord, search consensus has chord
            if s_bar["chords"]:
                m.chords.append(
                    LeadChord(
                        raw=s_bar["chords"][0],
                        beat=1.0,
                        agreement="search_only",
                        alternatives=s_bar["alternatives"],
                    )
                )

    if unaligned_search_count > 0:
        warnings.append(f"网络来源共有 {unaligned_search_count} 个小节未匹配到音频小节中")

    # Step 5: Disagreement Arbitration
    disagreements = _arbitrate_disagreements(disagreeing_items, audio_lead)

    # Step 6: Provenance & Report
    total_comp = n_agree + n_disagree
    agreement_ratio = round(n_agree / total_comp, 2) if total_comp > 0 else 1.0
    pct = int(round(agreement_ratio * 100))

    n_sources = len(search.sources)
    audio_lead.provenance = f"音频转录 + {n_sources} 个来源核对（一致 {pct}%）"
    audio_lead.sources = [
        SourceLink(title=s.title or s.url, url=s.url)
        for s in search.sources
    ]

    report = CrossCheckReport(
        mode="audio+search",
        key_decision=key_decision,
        agreement_ratio=agreement_ratio,
        n_agree=n_agree,
        n_disagree=n_disagree,
        n_audio_only=n_audio_only,
        disagreements=disagreements,
        sources=audio_lead.sources,
        warnings=warnings,
    )

    return audio_lead, report
