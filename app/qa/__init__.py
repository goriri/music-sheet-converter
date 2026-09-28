"""Quality Assurance module for music sheet converter.

Includes OMR verification layer, melody parsing, music priors, and LLM interfaces.
"""

from app.qa.llm import LLMUnavailable, ask_json, llm_available
from app.qa.melody import ParsedMelody, parse_melody
from app.qa.omr_verify import verify_sheet
from app.qa.priors import (
    bass_hint_fit,
    generate_candidates,
    melody_fit,
    progression_plausibility,
    score_candidate,
)

__all__ = [
    "LLMUnavailable",
    "ask_json",
    "llm_available",
    "ParsedMelody",
    "parse_melody",
    "verify_sheet",
    "bass_hint_fit",
    "generate_candidates",
    "melody_fit",
    "progression_plausibility",
    "score_candidate",
]
