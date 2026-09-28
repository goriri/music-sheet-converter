"""Music theory module for Taiwanese band-chart chord parsing and resolution."""
from app.theory.chords import (
    ChordSpec,
    format_chord_name,
    get_chord_pcs,
    parse_chord,
    resolve_chord,
)
from app.theory.keys import (
    canonical_key_for_pc,
    key_name_to_pc,
    resolve_degree,
    spell,
)

__all__ = [
    "ChordSpec",
    "format_chord_name",
    "get_chord_pcs",
    "parse_chord",
    "resolve_chord",
    "canonical_key_for_pc",
    "key_name_to_pc",
    "resolve_degree",
    "spell",
]
