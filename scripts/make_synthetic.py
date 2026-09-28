#!/usr/bin/env python3
"""Generate synthetic ParsedSheet and Arrangement fixtures.

Approximates the visual layout of fixtures/pages/page1.jpg and page2.jpg.
- Page 1 has 10 systems, Page 2 has 9 systems (total 19 systems, 76 measures).
- Key F, intermediate difficulty.
- RH chord stacks of 3-4 notes on beats 1 and 3, syncopated 8th pairs.
- LH 1-5-8 8th note arpeggios with fingers 5-2-1.
- Key change to G on page 2 (measure 50: bar 2 of System 12, matching '(轉成 2 調)(Ab)').
- Measure 10 contains rests, dotted rhythm, 16th group, and b7 accidental (Eb).
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import (
    Arrangement,
    ChordSymbol,
    Event,
    KeyChange,
    Measure,
    MeasureArrangement,
    Note,
    PageInfo,
    ParsedSheet,
    ResolvedChord,
    SongHeader,
    System,
)


def build_synthetic_sheet() -> ParsedSheet:
    pages = [
        PageInfo(width=1056, height=1402),
        PageInfo(width=1016, height=1376),
    ]

    p1_rows = [
        (150, 240),  # Sys 0: Only PN(RHY) ... 2 2 23 21
        (270, 365),  # Sys 1: Only PN+AG(RHY) ... 2 2 23 21
        (395, 515),  # Sys 2: Only PN(RHY) ... 心疼的玫瑰
        (540, 635),  # Sys 3: 4 06 66 7i ... 回到现场
        (645, 755),  # Sys 4: Only PN+AG(RHY) ... 微弱的风筝
        (785, 880),  # Sys 5: 4 06 66 7i ... 没有蓝天
        (890, 1000),  # Sys 6: Only PN+AG(RHY)+Bs in ... 黑色笑靥掉了
        (1005, 1125),  # Sys 7: Bs: 11 11 ... 瞳孔没有颜色
        (1135, 1270),  # Sys 8: Tempo ... 故事情节掉了
        (1280, 1375),  # Sys 9: Bs: 11 11 ... 胸口没有快乐
    ]
    p1_x_bars = [61, 288, 516, 744, 974]
    p1_h = 1402.0
    p1_w = 1056.0

    p2_rows = [
        (130, 260),  # Sys 0: EG ... 1(2)
        (265, 360),  # Sys 1: 在心中 不断 拉扯...
        (365, 480),  # Sys 2: 轉成2調(Ab) at bar 2 (m50)
        (495, 630),  # Sys 3: 黑色笑靥掉了...
        (635, 725),  # Sys 4: 瞳孔没有颜色...
        (755, 850),  # Sys 5: 故事情节掉了...
        (870, 970),  # Sys 6: 胸口没有快乐...
        (985, 1090),  # Sys 7: 掉 了...
        (1125, 1205),  # Sys 8: ending
    ]
    p2_x_bars = [60, 275, 500, 725, 950]
    p2_h = 1376.0
    p2_w = 1016.0

    chord_chart = [
        # Sys 0: intro
        [("1(2)", 1.0), ("5/7", 1.0), ("2m7", 1.0), None],
        # Sys 1
        [("1(2)", 1.0), ("5/7", 1.0), ("2m7/6", 1.0), ("5sus", 1.0)],
        # Sys 2: verse 1
        [("1(2)", 1.0), ("5/7", 1.0), ("5m/7b", 1.0), [("5m6/2", 1.0), ("67/1#", 3.0)]],
        # Sys 3
        [("2m7", 1.0), ("1(2)", 1.0), ("6b", 1.0), [("6m7-5", 1.0), ("5sus", 2.0), ("5", 3.0)]],
        # Sys 4: verse 2
        [("1(2)", 1.0), ("5/7", 1.0), ("5m/7b", 1.0), [("5m/7b", 1.0), ("67", 3.0)]],
        # Sys 5
        [("2m7", 1.0), ("1(2)", 1.0), ("6b", 1.0), [("57sus", 1.0), ("57", 3.0)]],
        # Sys 6: chorus
        [("1", 1.0), ("1/3", 1.0), ("6m7", 1.0), ("6m7/5", 1.0)],
        # Sys 7
        [("4", 1.0), ("1/3", 1.0), ("2m7", 1.0), [("2m7/5", 1.0), ("5", 3.0)]],
        # Sys 8
        [("1", 1.0), ("1/3", 1.0), ("6m7", 1.0), ("6m7/5", 1.0)],
        # Sys 9
        [("4", 1.0), ("1/3", 1.0), ("2m7", 1.0), [("2m7/5", 1.0), ("5", 3.0)]],
        # Page 2: Sys 10
        [("1(2)", 1.0), ("5sus", 1.0), ("7b", 1.0), ("4", 1.0)],
        # Sys 11
        [("1", 1.0), ("3m/7", 1.0), ("17/7b", 1.0), [("4M7", 1.0), ("7b", 3.0)]],
        # Sys 12: Modulation to 2调 (G) at measure 50 (bar 2)
        [("2m7", 1.0), ("5sus", 1.0), ("5sus", 1.0), ("5", 1.0)],
        # Sys 13 (in G)
        [("1", 1.0), ("1/3", 1.0), ("6m7", 1.0), ("6m7/5", 1.0)],
        # Sys 14
        [("4", 1.0), ("1/3", 1.0), ("2m7", 1.0), [("2m7/5", 1.0), ("5", 3.0)]],
        # Sys 15
        [("1", 1.0), ("1/3", 1.0), ("4M7", 1.0), ("57sus", 1.0)],
        # Sys 16
        [("4M7", 1.0), ("1/3", 1.0), ("2m7", 1.0), [("2m7/5", 1.0), ("5", 3.0)]],
        # Sys 17
        [("5", 1.0), ("1(2)", 1.0), ("5/7", 1.0), ("2m7", 1.0)],
        # Sys 18: Outro
        [("2m7", 1.0), ("1(2)", 1.0), ("5/7", 1.0), [("2m7/6", 1.0), ("5sus", 3.0)]],
    ]

    systems: list[System] = []
    global_measure_idx = 0

    has_label_p1 = {0, 1, 2, 4, 6, 8}
    for sys_idx in range(10):
        y0, y1 = p1_rows[sys_idx]
        sys_bbox = (p1_x_bars[0] / p1_w, y0 / p1_h, p1_x_bars[-1] / p1_w, y1 / p1_h)
        measures: list[Measure] = []

        chord_y_off = 22 if sys_idx in has_label_p1 else 6
        row_chords = chord_chart[sys_idx]

        for bar_idx in range(4):
            bx0 = p1_x_bars[bar_idx] / p1_w
            bx1 = p1_x_bars[bar_idx + 1] / p1_w
            m_bbox = (bx0, y0 / p1_h, bx1, y1 / p1_h)

            chords_spec = row_chords[bar_idx]
            measure_chords: list[ChordSymbol] = []
            if chords_spec is not None:
                if isinstance(chords_spec, list):
                    for raw, beat in chords_spec:
                        c_x0 = bx0 + (beat - 1.0) / 4.0 * (bx1 - bx0) + 0.005
                        c_bbox = (c_x0, (y0 + chord_y_off) / p1_h, c_x0 + 0.045, (y0 + chord_y_off + 24) / p1_h)
                        measure_chords.append(ChordSymbol(raw=raw, beat=beat, bbox=c_bbox))
                else:
                    raw, beat = chords_spec
                    c_x0 = bx0 + 0.005
                    c_bbox = (c_x0, (y0 + chord_y_off) / p1_h, c_x0 + 0.045, (y0 + chord_y_off + 24) / p1_h)
                    measure_chords.append(ChordSymbol(raw=raw, beat=beat, bbox=c_bbox))

            measures.append(
                Measure(
                    index=global_measure_idx,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=measure_chords,
                    melody="2 2 23 21" if sys_idx < 2 else "1. 5 55 65",
                    lyrics="心疼的玫瑰" if sys_idx == 2 and bar_idx == 0 else "",
                )
            )
            global_measure_idx += 1

        systems.append(
            System(
                page=0,
                bbox=sys_bbox,
                section_label="Only PN(RHY)" if sys_idx == 0 else None,
                measures=measures,
            )
        )

    for sys_idx in range(9):
        y0, y1 = p2_rows[sys_idx]
        sys_bbox = (p2_x_bars[0] / p2_w, y0 / p2_h, p2_x_bars[-1] / p2_w, y1 / p2_h)
        measures = []

        chord_y_off = 22 if sys_idx in {0, 2} else 8
        row_chords = chord_chart[10 + sys_idx]

        for bar_idx in range(4):
            bx0 = p2_x_bars[bar_idx] / p2_w
            bx1 = p2_x_bars[bar_idx + 1] / p2_w
            m_bbox = (bx0, y0 / p2_h, bx1, y1 / p2_h)

            chords_spec = row_chords[bar_idx]
            measure_chords = []
            if chords_spec is not None:
                if isinstance(chords_spec, list):
                    for raw, beat in chords_spec:
                        c_x0 = bx0 + (beat - 1.0) / 4.0 * (bx1 - bx0) + 0.005
                        c_bbox = (c_x0, (y0 + chord_y_off) / p2_h, c_x0 + 0.045, (y0 + chord_y_off + 24) / p2_h)
                        measure_chords.append(ChordSymbol(raw=raw, beat=beat, bbox=c_bbox))
                else:
                    raw, beat = chords_spec
                    c_x0 = bx0 + 0.005
                    c_bbox = (c_x0, (y0 + chord_y_off) / p2_h, c_x0 + 0.045, (y0 + chord_y_off + 24) / p2_h)
                    measure_chords.append(ChordSymbol(raw=raw, beat=beat, bbox=c_bbox))

            measures.append(
                Measure(
                    index=global_measure_idx,
                    bbox=m_bbox,
                    beats=4.0,
                    chords=measure_chords,
                    melody="i - - 0" if sys_idx == 0 else "57 12 3i",
                )
            )
            global_measure_idx += 1

        systems.append(
            System(
                page=1,
                bbox=sys_bbox,
                section_label="EG" if sys_idx == 0 else ("(轉成2調(Ab))" if sys_idx == 2 else None),
                measures=measures,
            )
        )

    # Key change at measure 50 (matches printed "(轉成 2 調)(Ab)")
    key_changes = [
        KeyChange(at_measure=50, raw="轉成2調(Ab) / 1=G", semitones=2),
    ]

    header = SongHeader(
        title="掉了",
        style="Slow Soul",
        time_signature="4/4",
        tempo_bpm=81.0,
        original_key="F#",
        male_key="Bb",
        female_key="F",
        raw="Slow Soul 4/4 (F# - Ab) 男調(Bb-C)女調(F-G) ♩=81 掉了",
    )

    return ParsedSheet(
        header=header,
        pages=pages,
        systems=systems,
        key_changes=key_changes,
    )


def resolve_chord_spec(raw: str, tonic_pc: int, beat: float = 1.0) -> ResolvedChord:
    lookup = {
        # in F (tonic_pc=5):
        (5, "1(2)"): ("Fadd9", 5, 5, [5, 9, 0, 2], "add9"),
        (5, "1"): ("F", 5, 5, [5, 9, 0], "maj"),
        (5, "1/3"): ("F/A", 5, 9, [5, 9, 0], "maj"),
        (5, "5/7"): ("C/E", 0, 4, [0, 4, 7], "maj"),
        (5, "5"): ("C", 0, 0, [0, 4, 7], "maj"),
        (5, "5sus"): ("Csus4", 0, 0, [0, 5, 7], "sus4"),
        (5, "57sus"): ("C7sus4", 0, 0, [0, 5, 7, 10], "7sus4"),
        (5, "57"): ("C7", 0, 0, [0, 4, 7, 10], "7"),
        (5, "2m7"): ("Gm7", 2, 2, [2, 5, 9, 0], "m7"),
        (5, "2m7/6"): ("Gm7/D", 2, 2, [2, 5, 9, 0], "m7"),
        (5, "2m7/5"): ("Gm7/C", 2, 0, [2, 5, 9, 0], "m7"),
        (5, "5m/7b"): ("Cm/Eb", 0, 3, [0, 3, 7], "m"),
        (5, "5m6/2"): ("Cm6/G", 0, 7, [0, 3, 7, 9], "m6"),
        (5, "67/1#"): ("D7/F#", 2, 6, [2, 6, 9, 0], "7"),
        (5, "67"): ("D7", 2, 2, [2, 6, 9, 0], "7"),
        (5, "6b"): ("Db", 1, 1, [1, 5, 8], "maj"),
        (5, "6m7-5"): ("Dm7b5", 2, 2, [2, 5, 8, 0], "m7b5"),
        (5, "6m7"): ("Dm7", 2, 2, [2, 5, 9, 0], "m7"),
        (5, "6m7/5"): ("Dm7/C", 2, 0, [2, 5, 9, 0], "m7"),
        (5, "4"): ("Bb", 10, 10, [10, 2, 5], "maj"),
        (5, "4M7"): ("Bbmaj7", 10, 10, [10, 2, 5, 9], "maj7"),
        (5, "7b"): ("Eb", 3, 3, [3, 7, 10], "maj"),
        (5, "3m/7"): ("Am/E", 9, 4, [9, 0, 4], "m"),
        (5, "17/7b"): ("F7/Eb", 5, 3, [5, 9, 0, 3], "7"),
        # in G (tonic_pc=7):
        (7, "1(2)"): ("Gadd9", 7, 7, [7, 11, 2, 4], "add9"),
        (7, "1"): ("G", 7, 7, [7, 11, 2], "maj"),
        (7, "1/3"): ("G/B", 7, 11, [7, 11, 2], "maj"),
        (7, "5/7"): ("D/F#", 2, 6, [2, 6, 9], "maj"),
        (7, "5"): ("D", 2, 2, [2, 6, 9], "maj"),
        (7, "5sus"): ("Dsus4", 2, 2, [2, 7, 9], "sus4"),
        (7, "57sus"): ("D7sus4", 2, 2, [2, 7, 9, 0], "7sus4"),
        (7, "2m7"): ("Am7", 9, 9, [9, 0, 4, 7], "m7"),
        (7, "2m7/6"): ("Am7/E", 9, 4, [9, 0, 4, 7], "m7"),
        (7, "2m7/5"): ("Am7/D", 9, 2, [9, 0, 4, 7], "m7"),
        (7, "6m7"): ("Em7", 4, 4, [4, 7, 11, 2], "m7"),
        (7, "6m7/5"): ("Em7/D", 4, 2, [4, 7, 11, 2], "m7"),
        (7, "4"): ("C", 0, 0, [0, 4, 7], "maj"),
        (7, "4M7"): ("Cmaj7", 0, 0, [0, 4, 7, 11], "maj7"),
    }
    key = (tonic_pc, raw)
    if key in lookup:
        name, root_pc, bass_pc, pcs, quality = lookup[key]
    else:
        name = raw
        root_pc = tonic_pc
        bass_pc = tonic_pc
        pcs = [tonic_pc]
        quality = "maj"

    return ResolvedChord(
        raw=raw,
        name=name,
        beat=beat,
        root_pc=root_pc,
        bass_pc=bass_pc,
        pcs=pcs,
        quality=quality,
    )


def build_synthetic_arrangement(sheet: ParsedSheet) -> Arrangement:
    measure_arrangements: list[MeasureArrangement] = []

    for m in sheet.measures():
        if m.index >= 50:
            tonic_pc = 7
            key_name = "G"
            root_midi = 43  # G2
            fifth_midi = 50  # D3
            oct_midi = 55  # G3
            rh_chord_1 = [
                Note(midi=67, finger=1),  # G4
                Note(midi=71, finger=2),  # B4
                Note(midi=74, finger=4),  # D5
            ]
            rh_chord_3 = [
                Note(midi=67, finger=1),
                Note(midi=71, finger=2),
                Note(midi=76, finger=5),  # E5
            ]
        else:
            tonic_pc = 5
            key_name = "F"
            root_midi = 41  # F2
            fifth_midi = 48  # C3
            oct_midi = 53  # F3
            rh_chord_1 = [
                Note(midi=65, finger=1),  # F4
                Note(midi=69, finger=2),  # A4
                Note(midi=72, finger=3),  # C5
                Note(midi=77, finger=5),  # F5
            ]
            rh_chord_3 = [
                Note(midi=65, finger=1),
                Note(midi=70, finger=3),  # Bb4
                Note(midi=72, finger=4),  # C5
            ]

        resolved_chords = [resolve_chord_spec(c.raw, tonic_pc, c.beat) for c in m.chords]

        if m.index == 10:
            # Special required measure:
            # Rest + dotted rhythm + 16th group + b7 accidental
            rh_events = [
                Event(onset=0.0, duration=1.0, notes=[]),  # Rest '0'
                Event(
                    onset=1.0,
                    duration=1.5,
                    notes=[Note(midi=75, finger=3)],  # b7 (Eb5) dotted rhythm!
                ),
                Event(
                    onset=2.5,
                    duration=0.5,
                    notes=[Note(midi=65, finger=5)],  # 1 (F4)
                ),
                Event(
                    onset=3.0,
                    duration=0.25,
                    notes=[Note(midi=69, finger=1)],  # 3 (A4)
                ),
                Event(
                    onset=3.25,
                    duration=0.25,
                    notes=[Note(midi=70, finger=2)],  # 4 (Bb4)
                ),
                Event(
                    onset=3.5,
                    duration=0.25,
                    notes=[Note(midi=72, finger=3)],  # 5 (C5)
                ),
                Event(
                    onset=3.75,
                    duration=0.25,
                    notes=[Note(midi=77, finger=5)],  # 1' (F5)
                ),
            ]

            lh_events = [
                Event(onset=0.0, duration=1.0, notes=[]),  # Rest '0'
                Event(onset=1.0, duration=1.0, notes=[Note(midi=51, finger=5)]),  # Eb3 (b7)
                Event(onset=2.0, duration=1.0, notes=[Note(midi=41, finger=5)]),  # F2
                Event(onset=3.0, duration=0.25, notes=[Note(midi=41, finger=5)]),
                Event(onset=3.25, duration=0.25, notes=[Note(midi=48, finger=3)]),
                Event(onset=3.5, duration=0.25, notes=[Note(midi=53, finger=1)]),
                Event(onset=3.75, duration=0.25, notes=[Note(midi=57, finger=1)]),
            ]
        else:
            # Intermediate pattern:
            # RH:
            # - Beat 1: 3-4 note chord stack
            # - Beat 2: syncopated eighth pair (with tie on beat 1.5 in measure 0)
            # - Beat 3: 3-4 note chord stack
            # - Beat 4: single note with finger
            is_tie = (m.index == 0)
            rh_events = [
                Event(onset=0.0, duration=1.0, notes=rh_chord_1),
                Event(onset=1.0, duration=0.5, notes=[Note(midi=rh_chord_1[1].midi, finger=2)], tie_to_next=is_tie),
                Event(onset=1.5, duration=0.5, notes=[Note(midi=rh_chord_1[-1].midi, finger=5)]),
                Event(onset=2.0, duration=1.0, notes=rh_chord_3),
                Event(onset=3.0, duration=1.0, notes=[Note(midi=rh_chord_3[0].midi, finger=1)]),
            ]

            # LH: 1-5-8 eighth arpeggios (8 eighths per bar)
            lh_events = [
                Event(onset=0.0, duration=0.5, notes=[Note(midi=root_midi, finger=5)]),
                Event(onset=0.5, duration=0.5, notes=[Note(midi=fifth_midi, finger=2)]),
                Event(onset=1.0, duration=0.5, notes=[Note(midi=oct_midi, finger=1)]),
                Event(onset=1.5, duration=0.5, notes=[Note(midi=fifth_midi, finger=2)]),
                Event(onset=2.0, duration=0.5, notes=[Note(midi=root_midi, finger=5)]),
                Event(onset=2.5, duration=0.5, notes=[Note(midi=fifth_midi, finger=2)]),
                Event(onset=3.0, duration=0.5, notes=[Note(midi=oct_midi, finger=1)]),
                Event(onset=3.5, duration=0.5, notes=[Note(midi=fifth_midi, finger=2)]),
            ]

        measure_arrangements.append(
            MeasureArrangement(
                measure_index=m.index,
                tonic_pc=tonic_pc,
                key_name=key_name,
                chords=resolved_chords,
                rh=rh_events,
                lh=lh_events,
            )
        )

    return Arrangement(
        instrument="piano",
        difficulty="intermediate",
        start_key="F",
        style="Slow Soul",
        measures=measure_arrangements,
        notes=["Accompaniment in key F (female key). Modulation to G at m50."],
    )


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sheet = build_synthetic_sheet()
    arr = build_synthetic_arrangement(sheet)

    fixtures_dir = os.path.join(root, "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    sheet_path = os.path.join(fixtures_dir, "synthetic_sheet.json")
    with open(sheet_path, "w", encoding="utf-8") as f:
        f.write(sheet.model_dump_json(indent=2))
    print(f"Wrote {sheet_path} ({len(sheet.systems)} systems, {len(sheet.measures())} measures)")

    arr_path = os.path.join(fixtures_dir, "synthetic_arrangement.json")
    with open(arr_path, "w", encoding="utf-8") as f:
        f.write(arr.model_dump_json(indent=2))
    print(f"Wrote {arr_path} ({len(arr.measures)} measures)")


if __name__ == "__main__":
    main()
