#!/usr/bin/env python3
"""Run arrangement CLI script.

Usage:
    python scripts/run_arrange.py fixtures/omr_sample.json --key F --difficulty intermediate
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import (
    ChordSymbol,
    Difficulty,
    KeyChange,
    Measure,
    PageInfo,
    ParsedSheet,
    SongHeader,
    System,
)
from app.theory.keys import spell
from app.arrange.piano import arrange


def midi_to_note_name(midi: int, tonic_pc: int = 0, key_name: str = "C") -> str:
    """Format MIDI pitch number as musical note name with octave using key-aware spelling."""
    octave = (midi // 12) - 1
    pc = midi % 12
    name = spell(pc, tonic_pc, key_name)
    return f"{name}{octave}"


def build_sample_sheet() -> ParsedSheet:
    """Construct a faithful ParsedSheet fixture of the sample song '掉了' from page images."""
    header = SongHeader(
        title="掉了",
        style="Slow Soul",
        time_signature="4/4",
        tempo_bpm=81.0,
        original_key="F#",
        male_key="Bb",
        female_key="F",
        raw="Slow Soul 4/4 (F# - Ab) 男調(Bb-C) 女調(F-G) 掉了",
    )

    dummy_bbox = (0.05, 0.1, 0.95, 0.2)
    m_bbox = (0.1, 0.1, 0.3, 0.2)

    systems = [
        # System 1: Intro (measures 0-3)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN(RHY) / / / ~",
            measures=[
                Measure(index=0, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(index=1, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5/7", beat=1.0)]),
                Measure(index=2, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7", beat=1.0)]),
                Measure(index=3, bbox=m_bbox, beats=4.0, chords=[]),  # Carried chord
            ],
        ),
        # System 2: Intro part 2 (measures 4-7)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN+AG(RHY)",
            measures=[
                Measure(index=4, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(index=5, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5/7", beat=1.0)]),
                Measure(index=6, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7/6", beat=1.0)]),
                Measure(index=7, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5sus", beat=1.0)]),
            ],
        ),
        # System 3: Verse 1 (measures 8-11)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN(RHY) / / / ~",
            measures=[
                Measure(index=8, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(index=9, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5/7", beat=1.0)]),
                Measure(index=10, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="5m/7b", beat=1.0)]),
                Measure(
                    index=11,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="5m6/2", beat=1.0), ChordSymbol(raw="67/1#", beat=3.0)],
                ),
            ],
        ),
        # System 4: Verse 1 continued (measures 12-15)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label=None,
            measures=[
                Measure(index=12, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7", beat=1.0)]),
                Measure(index=13, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1(2)", beat=1.0)]),
                Measure(
                    index=14,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="6b", beat=1.0), ChordSymbol(raw="6m7-5", beat=3.0)],
                ),
                Measure(
                    index=15,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="5sus", beat=1.0), ChordSymbol(raw="5", beat=3.0)],
                ),
            ],
        ),
        # System 5: Chorus 1 with bass (measures 16-19)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label="Only PN+AG(RHY)+Bs in",
            measures=[
                Measure(
                    index=16,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="1", beat=1.0)],
                    bass_hint="11 11 11 112",
                ),
                Measure(index=17, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1/3", beat=1.0)]),
                Measure(index=18, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="6m7", beat=1.0)]),
                Measure(index=19, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="6m7/5", beat=1.0)]),
            ],
        ),
        # System 6: Chorus 1 ending with fill (measures 20-23)
        System(
            page=0,
            bbox=dummy_bbox,
            section_label=None,
            measures=[
                Measure(index=20, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="4", beat=1.0)]),
                Measure(index=21, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="1/3", beat=1.0)]),
                Measure(index=22, bbox=m_bbox, beats=4.0, chords=[ChordSymbol(raw="2m7", beat=1.0)]),
                Measure(
                    index=23,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=[ChordSymbol(raw="2m7/5", beat=1.0), ChordSymbol(raw="5", beat=3.0)],
                    fill=True,
                ),
            ],
        ),
    ]

    key_changes = [
        KeyChange(at_measure=20, raw="轉成2調(Ab)", semitones=2),
    ]

    return ParsedSheet(
        header=header,
        pages=[PageInfo(width=1000, height=1400), PageInfo(width=1000, height=1400)],
        systems=systems,
        key_changes=key_changes,
    )


def print_arrangement(arr, start_measure: int = 0, num_measures: int = 4):
    """Print readable per-measure chords, RH and LH events with note names and fingers."""
    print("=" * 68)
    print(f"PIANO ARRANGEMENT: {arr.style.upper()} | Start Key: {arr.start_key} | Difficulty: {arr.difficulty}")
    print("=" * 68)

    selected = [m for m in arr.measures if m.measure_index >= start_measure][:num_measures]

    for m_arr in selected:
        chords_str = ", ".join(f"{c.name} [{c.raw}]" for c in m_arr.chords)
        print(f"\nMeasure {m_arr.measure_index:02d} (Key: {m_arr.key_name}, Tonic PC: {m_arr.tonic_pc}) | Chords: {chords_str}")
        print("  RH Events:")
        for ev in m_arr.rh:
            if not ev.notes:
                print(f"    [{ev.onset:4.2f} - {ev.onset + ev.duration:4.2f}] Rest (dur: {ev.duration:4.2f})")
            else:
                notes_str = " ".join(
                    f"{midi_to_note_name(n.midi, m_arr.tonic_pc, m_arr.key_name)}(f:{n.finger})" for n in ev.notes
                )
                print(f"    [{ev.onset:4.2f} - {ev.onset + ev.duration:4.2f}] {notes_str} (dur: {ev.duration:4.2f})")

        print("  LH Events:")
        for ev in m_arr.lh:
            if not ev.notes:
                print(f"    [{ev.onset:4.2f} - {ev.onset + ev.duration:4.2f}] Rest (dur: {ev.duration:4.2f})")
            else:
                notes_str = " ".join(
                    f"{midi_to_note_name(n.midi, m_arr.tonic_pc, m_arr.key_name)}(f:{n.finger})" for n in ev.notes
                )
                print(f"    [{ev.onset:4.2f} - {ev.onset + ev.duration:4.2f}] {notes_str} (dur: {ev.duration:4.2f})")


def main():
    parser = argparse.ArgumentParser(description="Piano arrangement runner")
    parser.add_argument("fixture_path", nargs="?", default="fixtures/omr_sample.json", help="Path to OMR sample JSON")
    parser.add_argument("--key", default="F", help="Start key (default: F)")
    parser.add_argument("--difficulty", default="intermediate", choices=["beginner", "intermediate", "advanced"])
    parser.add_argument("--start-measure", type=int, default=0, help="Starting measure index")
    parser.add_argument("--measures", type=int, default=4, help="Number of measures to display")
    args = parser.parse_args()

    sheet = None
    if os.path.exists(args.fixture_path):
        print(f"Loading sheet from {args.fixture_path}...")
        with open(args.fixture_path, "r", encoding="utf-8") as f:
            sheet = ParsedSheet.model_validate_json(f.read())
    else:
        print(f"Fixture {args.fixture_path} not found; using built-in sample sheet for '掉了'.")
        sheet = build_sample_sheet()

    arr = arrange(sheet, start_key=args.key, difficulty=args.difficulty)
    print_arrangement(arr, start_measure=args.start_measure, num_measures=args.measures)


if __name__ == "__main__":
    main()
