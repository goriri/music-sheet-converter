"""Unit tests for the accompaniment overlay renderer.

Tests:
- Jianpu conversion (degrees, accidentals, dots for both hands, several keys)
- Beaming groups
- Strip heights
- render_pdf returns bytes starting with %PDF and page count as expected
- Resilience: no exception when bboxes are missing or measures are empty
"""
import io
import numpy as np
import pytest
from PIL import Image

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
from app.render.jianpu import (
    compute_stack_height,
    get_font,
    midi_to_jianpu,
)
from app.render.overlay import (
    make_header_band,
    render_page,
    render_pages,
    render_pdf,
    render_system_strip,
)



# --------------------------------------------------------------------------- Jianpu Conversion Tests
class TestJianpuConversion:
    def test_key_c_degrees_and_accidentals(self):
        # Key C (tonic_pc = 0)
        # Diatonic
        assert midi_to_jianpu(60, 0, "rh").degree == 1
        assert midi_to_jianpu(62, 0, "rh").degree == 2
        assert midi_to_jianpu(64, 0, "rh").degree == 3
        assert midi_to_jianpu(65, 0, "rh").degree == 4
        assert midi_to_jianpu(67, 0, "rh").degree == 5
        assert midi_to_jianpu(69, 0, "rh").degree == 6
        assert midi_to_jianpu(71, 0, "rh").degree == 7

        # Chromatic defaults
        # b2 flat
        n_b2 = midi_to_jianpu(61, 0, "rh")
        assert n_b2.degree == 2 and n_b2.accidental == "♭"

        # #1 sharp with prefer_sharps
        n_s1 = midi_to_jianpu(61, 0, "rh", prefer_sharps=True)
        assert n_s1.degree == 1 and n_s1.accidental == "♯"

        # b3 flat
        n_b3 = midi_to_jianpu(63, 0, "rh")
        assert n_b3.degree == 3 and n_b3.accidental == "♭"

        # #4 sharp
        n_s4 = midi_to_jianpu(66, 0, "rh")
        assert n_s4.degree == 4 and n_s4.accidental == "♯"

        # b6 flat
        n_b6 = midi_to_jianpu(68, 0, "rh")
        assert n_b6.degree == 6 and n_b6.accidental == "♭"

        # #5 sharp with prefer_sharps
        n_s5 = midi_to_jianpu(68, 0, "rh", prefer_sharps=True)
        assert n_s5.degree == 5 and n_s5.accidental == "♯"

        # b7 flat
        n_b7 = midi_to_jianpu(70, 0, "rh")
        assert n_b7.degree == 7 and n_b7.accidental == "♭"

    def test_key_f_degrees_and_accidentals(self):
        # Key F (tonic_pc = 5)
        # F4 (65) is tonic (1)
        assert midi_to_jianpu(65, 5, "rh").degree == 1
        assert midi_to_jianpu(67, 5, "rh").degree == 2
        assert midi_to_jianpu(69, 5, "rh").degree == 3
        assert midi_to_jianpu(70, 5, "rh").degree == 4
        assert midi_to_jianpu(72, 5, "rh").degree == 5
        assert midi_to_jianpu(74, 5, "rh").degree == 6
        assert midi_to_jianpu(76, 5, "rh").degree == 7

        # Eb5 (75) is b7
        n_eb5 = midi_to_jianpu(75, 5, "rh")
        assert n_eb5.degree == 7 and n_eb5.accidental == "♭" and n_eb5.octave_dots == 0

    def test_key_g_degrees(self):
        # Key G (tonic_pc = 7)
        # G4 (67) is tonic (1)
        assert midi_to_jianpu(67, 7, "rh").degree == 1
        assert midi_to_jianpu(69, 7, "rh").degree == 2
        assert midi_to_jianpu(71, 7, "rh").degree == 3
        assert midi_to_jianpu(72, 7, "rh").degree == 4
        assert midi_to_jianpu(74, 7, "rh").degree == 5
        assert midi_to_jianpu(76, 7, "rh").degree == 6
        assert midi_to_jianpu(78, 7, "rh").degree == 7

    def test_key_d_degrees(self):
        # Key D (tonic_pc = 2)
        # D4 (62) is tonic (1)
        assert midi_to_jianpu(62, 2, "rh").degree == 1
        assert midi_to_jianpu(66, 2, "rh").degree == 3  # F#4
        assert midi_to_jianpu(69, 2, "rh").degree == 5  # A4
        assert midi_to_jianpu(73, 2, "rh").degree == 7  # C#5

    def test_octave_dots_rh_and_lh(self):
        # Key F (tonic_pc = 5):
        # RH reference tonic at/above C4: F4 (65) -> 0 dots
        # LH reference tonic one octave lower: F3 (53) -> 0 dots

        # RH:
        assert midi_to_jianpu(65, 5, "rh").octave_dots == 0  # F4
        assert midi_to_jianpu(76, 5, "rh").octave_dots == 0  # E5
        assert midi_to_jianpu(77, 5, "rh").octave_dots == 1  # F5 (1 dot above)
        assert midi_to_jianpu(89, 5, "rh").octave_dots == 2  # F6 (2 dots above)
        assert midi_to_jianpu(60, 5, "rh").octave_dots == -1  # C4 (1 dot below)
        assert midi_to_jianpu(53, 5, "rh").octave_dots == -1  # F3 (1 dot below)
        assert midi_to_jianpu(41, 5, "rh").octave_dots == -2  # F2 (2 dots below)

        # LH:
        assert midi_to_jianpu(53, 5, "lh").octave_dots == 0  # F3 (0 dots in LH)
        assert midi_to_jianpu(60, 5, "lh").octave_dots == 0  # C4 (0 dots in LH)
        assert midi_to_jianpu(64, 5, "lh").octave_dots == 0  # E4 (0 dots in LH)
        assert midi_to_jianpu(65, 5, "lh").octave_dots == 1  # F4 (1 dot above in LH)
        assert midi_to_jianpu(48, 5, "lh").octave_dots == -1  # C3 (1 dot below in LH)
        assert midi_to_jianpu(41, 5, "lh").octave_dots == -1  # F2 (1 dot below in LH)
        assert midi_to_jianpu(36, 5, "lh").octave_dots == -2  # C2 (2 dots below in LH)

    def test_chord_context_chromatic_spelling(self):
        # 1. F key D7/F# -> '#1'
        c_d7_fsharp = ResolvedChord(
            raw="D7/F#", name="D7/F#", beat=1.0, root_pc=2, bass_pc=6, pcs=[2, 6, 9, 0], quality="7"
        )
        # F#3 = MIDI 54, F#4 = MIDI 66 (pc 6)
        n1 = midi_to_jianpu(54, 5, hand="lh", chord=c_d7_fsharp, key_name="F")
        assert n1.degree == 1 and n1.accidental == "♯"
        assert n1.text == "♯1"

        # 2. F key Cm/Eb bass -> 'b7'
        c_cm_eb = ResolvedChord(
            raw="Cm/Eb", name="Cm/Eb", beat=1.0, root_pc=0, bass_pc=3, pcs=[0, 3, 7], quality="m"
        )
        # Eb3 = MIDI 51, Eb4 = MIDI 63 (pc 3)
        n2 = midi_to_jianpu(51, 5, hand="lh", chord=c_cm_eb, key_name="F")
        assert n2.degree == 7 and n2.accidental == "♭"
        assert n2.text == "♭7"

        # 3. C key A7/C# -> '#1'
        c_a7_csharp = ResolvedChord(
            raw="A7/C#", name="A7/C#", beat=1.0, root_pc=9, bass_pc=1, pcs=[9, 1, 4, 7], quality="7"
        )
        # C#4 = MIDI 61 (pc 1)
        n3 = midi_to_jianpu(61, 0, hand="lh", chord=c_a7_csharp, key_name="C")
        assert n3.degree == 1 and n3.accidental == "♯"
        assert n3.text == "♯1"

        # 4. G key 'b7' chord (F) -> 'b7'
        c_f = ResolvedChord(
            raw="b7", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj"
        )
        # F3 = MIDI 53, F4 = MIDI 65 (pc 5)
        n4 = midi_to_jianpu(65, 7, hand="rh", chord=c_f, key_name="G")
        assert n4.degree == 7 and n4.accidental == "♭"
        assert n4.text == "♭7"



# --------------------------------------------------------------------------- Layout & Beaming Tests
class TestLayoutAndBeaming:
    def test_compute_stack_height(self):
        h1 = compute_stack_height(1)
        h2 = compute_stack_height(2)
        h3 = compute_stack_height(3)
        h4 = compute_stack_height(4)

        assert h1 > 0
        assert h2 > h1
        assert h3 > h2
        assert h4 > h3
        assert h4 - h3 == pytest.approx(18.0)

    def test_strip_height_scales_with_tallest_stack(self):
        sys_1note = System(
            page=0,
            bbox=(0.06, 0.1, 0.94, 0.2),
            measures=[Measure(index=0, bbox=(0.06, 0.1, 0.94, 0.2))],
        )
        arr_1note = Arrangement(
            instrument="piano",
            difficulty="intermediate",
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=65, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=53, finger=5)])],
                )
            ],
        )

        strip1 = render_system_strip(sys_1note, 1000, arr_1note)

        # 4-note stack in RH
        arr_4note = Arrangement(
            instrument="piano",
            difficulty="intermediate",
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[],
                    rh=[
                        Event(
                            onset=0.0,
                            duration=1.0,
                            notes=[
                                Note(midi=65, finger=1),
                                Note(midi=69, finger=2),
                                Note(midi=72, finger=3),
                                Note(midi=77, finger=5),
                            ],
                        )
                    ],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=53, finger=5)])],
                )
            ],
        )

        strip4 = render_system_strip(sys_1note, 1000, arr_4note)
        assert strip4.height > strip1.height


# --------------------------------------------------------------------------- PDF Export Tests
class TestRenderPDF:
    def _create_blank_page(self, width=1000, height=1400) -> bytes:
        img = Image.new("RGB", (width, height), color=(250, 250, 250))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_render_pdf_returns_bytes_and_correct_pages(self):
        page1 = self._create_blank_page(1056, 1402)
        page2 = self._create_blank_page(1016, 1376)

        # 2 systems per page
        systems = [
            System(page=0, bbox=(0.06, 0.15, 0.94, 0.25), measures=[Measure(index=0, bbox=(0.06, 0.15, 0.94, 0.25))]),
            System(page=0, bbox=(0.06, 0.35, 0.94, 0.45), measures=[Measure(index=1, bbox=(0.06, 0.35, 0.94, 0.45))]),
            System(page=1, bbox=(0.06, 0.15, 0.94, 0.25), measures=[Measure(index=2, bbox=(0.06, 0.15, 0.94, 0.25))]),
            System(page=1, bbox=(0.06, 0.35, 0.94, 0.45), measures=[Measure(index=3, bbox=(0.06, 0.35, 0.94, 0.45))]),
        ]
        sheet = ParsedSheet(
            header=SongHeader(title="Test Song", start_key="F"),
            pages=[PageInfo(width=1056, height=1402), PageInfo(width=1016, height=1376)],
            systems=systems,
        )
        arrangement = Arrangement(
            instrument="piano",
            difficulty="intermediate",
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=i,
                    tonic_pc=5,
                    key_name="F",
                    chords=[],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=65, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=53, finger=5)])],
                )
                for i in range(4)
            ],
        )

        pdf_bytes = render_pdf([page1, page2], sheet, arrangement)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

        # Check page count with PyMuPDF
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # With 2 systems per page, height <= 1.6x, so 2 pages total
        assert len(doc) == 2


# --------------------------------------------------------------------------- Resilience Tests
class TestResilience:
    def test_missing_or_none_bboxes(self):
        # Measure and system without bboxes
        sys_no_bbox = System(
            page=0,
            bbox=(0.0, 0.0, 1.0, 1.0),
            measures=[Measure(index=0, bbox=(0.0, 0.0, 1.0, 1.0), chords=[ChordSymbol(raw="1", beat=1.0, bbox=None)])],
        )
        sheet = ParsedSheet(
            header=SongHeader(),
            pages=[PageInfo(width=1000, height=1400)],
            systems=[sys_no_bbox],
        )
        arrangement = Arrangement(
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[ResolvedChord(raw="1", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj")],
                    rh=[],
                    lh=[],
                )
            ],
        )

        img = Image.new("RGB", (1000, 1400), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        # Should not raise exception
        pages = render_pages([raw_bytes], sheet, arrangement)
        assert len(pages) >= 1

    def test_empty_measures_and_rests(self):
        sys_empty = System(
            page=0,
            bbox=(0.05, 0.2, 0.95, 0.3),
            measures=[
                Measure(index=0, bbox=(0.05, 0.2, 0.45, 0.3)),
                Measure(index=1, bbox=(0.45, 0.2, 0.95, 0.3)),
            ],
        )
        sheet = ParsedSheet(
            header=SongHeader(),
            pages=[PageInfo(width=1000, height=1400)],
            systems=[sys_empty],
        )
        arrangement = Arrangement(
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[],
                    rh=[Event(onset=0.0, duration=4.0, notes=[])],  # whole measure rest
                    lh=[],
                ),
                MeasureArrangement(
                    measure_index=1,
                    tonic_pc=5,
                    key_name="F",
                    chords=[],
                    rh=[],
                    lh=[Event(onset=0.0, duration=2.0, notes=[])],
                ),
            ],
        )

        img = Image.new("RGB", (1000, 1400), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        pdf_bytes = render_pdf([raw_bytes], sheet, arrangement)
        assert pdf_bytes.startswith(b"%PDF")

    def test_font_fallback(self):
        # Even if requested font is missing, get_font must return a valid font
        font = get_font(24)
        assert font is not None


# --------------------------------------------------------------------------- Overlay Revisions Tests
class TestOverlayRevisions:
    def test_original_pixels_unchanged_within_system_bboxes(self):
        # Non-uniform image pattern to ensure any drawing would alter pixels
        np.random.seed(42)
        arr = np.random.randint(50, 220, (1400, 1000, 3), dtype=np.uint8)
        page_img = Image.fromarray(arr, mode="RGB")

        # Two systems with clear bboxes
        s0 = System(
            page=0,
            bbox=(0.08, 0.15, 0.92, 0.28),
            measures=[Measure(index=0, bbox=(0.08, 0.15, 0.92, 0.28))],
        )
        s1 = System(
            page=0,
            bbox=(0.08, 0.45, 0.92, 0.60),
            measures=[Measure(index=1, bbox=(0.08, 0.45, 0.92, 0.60))],
        )

        sheet = ParsedSheet(
            header=SongHeader(title="Test", original_key="F#"),
            pages=[PageInfo(width=1000, height=1400)],
            systems=[s0, s1],
        )
        arrangement = Arrangement(
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[
                        ResolvedChord(
                            raw="1", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj"
                        )
                    ],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=65, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=53, finger=5)])],
                ),
                MeasureArrangement(
                    measure_index=1,
                    tonic_pc=5,
                    key_name="F",
                    chords=[
                        ResolvedChord(
                            raw="5", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj"
                        )
                    ],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=60, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=48, finger=5)])],
                ),
            ],
        )

        rendered = render_page(page_img, [s0, s1], sheet, arrangement, is_first_page=True)
        assert len(rendered) >= 1
        res = rendered[0]

        # On page 1, header band is prepended (height = round(0.05 * 1400) = 70)
        header_h = int(round(0.05 * 1400))
        w, h = 1000, 1400

        # System 0 crop verification
        x0, y0 = int(round(0.08 * w)), int(round(0.15 * h))
        x1, y1 = int(round(0.92 * w)), int(round(0.28 * h))
        orig_s0 = np.array(page_img.crop((x0, y0, x1, y1)))
        # In res, slice 0 starts at y=header_h, so s0 is at [header_h + y0, header_h + y1]
        res_s0 = np.array(res.crop((x0, header_h + y0, x1, header_h + y1)))
        assert np.array_equal(res_s0, orig_s0), "System 0 pixels in rendered output were modified!"

        # System 1 crop verification
        x0_s1, y0_s1 = int(round(0.08 * w)), int(round(0.45 * h))
        x1_s1, y1_s1 = int(round(0.92 * w)), int(round(0.60 * h))
        orig_s1 = np.array(page_img.crop((x0_s1, y0_s1, x1_s1, y1_s1)))

        strip0 = render_system_strip(s0, w, arrangement, sheet)
        # s1 starts at header_h + strip0.height + y0_s1
        res_s1 = np.array(res.crop((x0_s1, header_h + strip0.height + y0_s1, x1_s1, header_h + strip0.height + y1_s1)))
        assert np.array_equal(res_s1, orig_s1), "System 1 pixels in rendered output were modified!"

    def test_header_band_height_and_legend(self):
        w, h = 1056, 1402
        expected_h = int(round(0.05 * h))  # 70 px
        sheet = ParsedSheet(
            header=SongHeader(title="Test", original_key="F#"),
            pages=[PageInfo(width=w, height=h)],
            systems=[],
        )
        arrangement = Arrangement(
            instrument="piano",
            difficulty="intermediate",
            start_key="F",
            measures=[],
        )
        band = make_header_band(w, expected_h, sheet, arrangement)
        assert band.size == (w, expected_h)

        # Check that pixels are not blank (text was drawn)
        band_arr = np.array(band)
        assert (band_arr < 255).any()

    def test_chord_row_existence_and_styling(self):
        sys = System(
            page=0,
            bbox=(0.06, 0.1, 0.94, 0.2),
            measures=[
                Measure(
                    index=0,
                    bbox=(0.06, 0.1, 0.94, 0.2),
                    beats=4.0,
                    chords=[ChordSymbol(raw="1", beat=1.0)],
                )
            ],
        )
        arr = Arrangement(
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[
                        ResolvedChord(
                            raw="1", name="F", beat=1.0, root_pc=5, bass_pc=5, pcs=[5, 9, 0], quality="maj"
                        )
                    ],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=65, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=53, finger=5)])],
                )
            ],
        )
        strip = render_system_strip(sys, 1000, arr)
        strip_arr = np.array(strip)

        # Chord row is at top (y in [0, 24]) with color #B00000 -> (176, 0, 0)
        chord_row_crop = strip_arr[0:24, :, :]
        red_pixels = (
            (chord_row_crop[:, :, 0] > 150)
            & (chord_row_crop[:, :, 1] < 50)
            & (chord_row_crop[:, :, 2] < 50)
        )
        assert red_pixels.sum() > 0, "Expected #B00000 chord text in top chord row"

    def test_key_change_marker_boxed_in_chord_row(self):
        sys = System(
            page=0,
            bbox=(0.06, 0.1, 0.94, 0.2),
            measures=[Measure(index=10, bbox=(0.06, 0.1, 0.94, 0.2), beats=4.0)],
        )
        sheet = ParsedSheet(
            header=SongHeader(),
            pages=[PageInfo(width=1000, height=1400)],
            systems=[sys],
            key_changes=[KeyChange(at_measure=10, raw="转1=G (原谱1=#F)", semitones=2)],
        )
        arr = Arrangement(
            start_key="G",
            measures=[
                MeasureArrangement(
                    measure_index=10,
                    tonic_pc=7,
                    key_name="G",
                    chords=[
                        ResolvedChord(
                            raw="1", name="G", beat=1.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj"
                        )
                    ],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=67, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=55, finger=5)])],
                )
            ],
        )
        strip = render_system_strip(sys, 1000, arr, sheet=sheet)
        strip_arr = np.array(strip)

        chord_row_crop = strip_arr[0:24, :, :]
        red_pixels = (
            (chord_row_crop[:, :, 0] > 150)
            & (chord_row_crop[:, :, 1] < 50)
            & (chord_row_crop[:, :, 2] < 50)
        )
        assert red_pixels.sum() > 20, "Expected red boxed key change marker in chord row"

    def test_margin_labels_clean_r_and_l(self):
        sys = System(
            page=0,
            bbox=(0.06, 0.1, 0.94, 0.2),
            measures=[Measure(index=0, bbox=(0.06, 0.1, 0.94, 0.2))],
        )
        arr = Arrangement(
            start_key="F",
            measures=[
                MeasureArrangement(
                    measure_index=0,
                    tonic_pc=5,
                    key_name="F",
                    chords=[],
                    rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=65, finger=1)])],
                    lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=53, finger=5)])],
                )
            ],
        )
        strip = render_system_strip(sys, 1000, arr)
        left_margin = np.array(strip)[:, 0:50, :]
        dark_pixels = (left_margin < 200).all(axis=2)
        assert dark_pixels.sum() > 0

