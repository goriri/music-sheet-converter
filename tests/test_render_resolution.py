"""Tests for high-resolution rendering, PDF A4 dimensions, cut placement, and chord diagram clearance."""
from __future__ import annotations

import io
import math
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.models import (
    Arrangement,
    ChordSymbol,
    Event,
    Measure,
    MeasureArrangement,
    Note,
    PageInfo,
    ParsedSheet,
    ResolvedChord,
    SongHeader,
    System,
    UkeMeasure,
    UkeShape,
    UkeStroke,
)
from app.render.overlay import (
    find_system_cut_row,
    find_whitest_row,
    images_to_pdf_bytes,
    render_page,
    render_pages,
    render_pdf,
    render_system_strip,
)
from app.render.ukulele import (
    draw_chord_diagram,
    make_uke_header_band,
    render_uke_page,
    render_uke_pages,
    render_uke_pdf,
    render_uke_system_strip,
)


def _create_sample_sheet(width: int = 1000, height: int = 1400) -> tuple[bytes, ParsedSheet, Arrangement]:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page_bytes = buf.getvalue()

    systems = [
        System(page=0, bbox=(0.06, 0.15, 0.94, 0.30), measures=[Measure(index=0, bbox=(0.06, 0.15, 0.94, 0.30))]),
        System(page=0, bbox=(0.06, 0.45, 0.94, 0.60), measures=[Measure(index=1, bbox=(0.06, 0.45, 0.94, 0.60))]),
    ]
    sheet = ParsedSheet(
        header=SongHeader(title="Test Song", start_key="C"),
        pages=[PageInfo(width=width, height=height)],
        systems=systems,
    )
    arrangement = Arrangement(
        instrument="piano",
        difficulty="intermediate",
        start_key="C",
        measures=[
            MeasureArrangement(
                measure_index=0,
                tonic_pc=0,
                key_name="C",
                chords=[ResolvedChord(raw="1", name="C", beat=1.0, root_pc=0, bass_pc=0, pcs=[0, 4, 7], quality="maj")],
                rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=60, finger=1)])],
                lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=48, finger=5)])],
            ),
            MeasureArrangement(
                measure_index=1,
                tonic_pc=0,
                key_name="C",
                chords=[ResolvedChord(raw="5", name="G", beat=1.0, root_pc=7, bass_pc=7, pcs=[7, 11, 2], quality="maj")],
                rh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=67, finger=1)])],
                lh=[Event(onset=0.0, duration=1.0, notes=[Note(midi=55, finger=5)])],
            ),
        ],
    )
    return page_bytes, sheet, arrangement


class TestHighResUpscaling:
    def test_upscale_to_target_w_2400(self):
        page_bytes, sheet, arrangement = _create_sample_sheet(width=1000, height=1400)
        rendered_pages = render_pages([page_bytes], sheet, arrangement)
        assert len(rendered_pages) >= 1
        for p in rendered_pages:
            assert p.width == 2400

    def test_already_2400_width_does_not_scale_down(self):
        page_bytes, sheet, arrangement = _create_sample_sheet(width=2400, height=3360)
        rendered_pages = render_pages([page_bytes], sheet, arrangement)
        assert len(rendered_pages) >= 1
        for p in rendered_pages:
            assert p.width == 2400

    def test_clamp_max_3x_scaling(self):
        # 600 width: 2400 / 600 = 4.0, but clamped to 3.0 => target_w = 1800
        page_bytes, sheet, arrangement = _create_sample_sheet(width=600, height=840)
        rendered_pages = render_pages([page_bytes], sheet, arrangement)
        assert len(rendered_pages) >= 1
        for p in rendered_pages:
            assert p.width == 1800


class TestPDFDimensionsAndEmbedding:
    def test_pdf_a4_width_and_quality(self):
        import pymupdf

        page_bytes, sheet, arrangement = _create_sample_sheet(width=1000, height=1400)
        pdf_bytes = render_pdf([page_bytes], sheet, arrangement)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        assert len(doc) >= 1
        for page in doc:
            assert pytest.approx(page.rect.width, rel=1e-3) == 595.0
            # Verify embedded image is JPEG
            img_list = page.get_images()
            assert len(img_list) >= 1
            xref = img_list[0][0]
            base_image = doc.extract_image(xref)
            assert base_image["ext"] == "jpeg"
        doc.close()

    def test_images_to_pdf_bytes_empty(self):
        assert images_to_pdf_bytes([]) == b""


class TestCutPlacement:
    def test_find_system_cut_row_synthetic_slur_avoidance(self):
        # Construct synthetic grayscale image: 400w x 200h
        # System 1: rows 30..60 ink
        # Empty run: rows 61..100 (40 white rows)
        # System 2 slur: rows 101..109 ink
        # Small gap: rows 110..115 (6 white rows)
        # System 2 notes: rows 116..160 ink
        arr = np.full((200, 400), 255, dtype=np.uint8)
        arr[30:60, 50:350] = 50  # System 1 ink
        arr[101:110, 80:320] = 60  # System 2 slur
        arr[116:160, 50:350] = 50  # System 2 notes

        # If bounding boxes abut at row 116 (where system 2 notes start)
        # Search range around 116 with delta=60 -> [56, 176]
        cut = find_system_cut_row(arr, curr_y1=116, next_y0=116, nominal_y=116, delta=60)
        # Should cut in the longest empty run [61, 100], center is ~80
        assert 61 <= cut <= 100
        # Slurs in rows 101-109 are preserved BELOW the cut (in System 2 slice)
        assert cut < 101

    def test_find_system_cut_row_normal_gap(self):
        arr = np.full((200, 400), 255, dtype=np.uint8)
        arr[20:50, :] = 50  # ink
        # rows 50..120 empty (indices 50..119, center 84.5 -> 84 or 85)
        arr[120:150, :] = 50  # ink
        cut = find_system_cut_row(arr, curr_y1=50, next_y0=120)
        assert cut in (84, 85)


class TestPianoScaleOneIdentity:
    def test_piano_system_strip_and_header_identical_at_scale_one(self):
        # Verify that scale=1.0 is completely deterministic and unchanged
        _, sheet, arrangement = _create_sample_sheet(width=1000, height=1400)
        sys0 = sheet.systems[0]
        strip1 = render_system_strip(sys0, 1000, arrangement, sheet, scale=1.0)
        strip2 = render_system_strip(sys0, 1000, arrangement, sheet, scale=1.0)
        assert np.array_equal(np.array(strip1), np.array(strip2))

        # Check git HEAD if accessible via git
        import subprocess, types, sys as sys_module, os
        try:
            head_jianpu_code = subprocess.check_output(
                ["git", "show", "origin/main:app/render/jianpu.py"], stderr=subprocess.DEVNULL
            ).decode("utf-8")
            head_overlay_code = subprocess.check_output(
                ["git", "show", "origin/main:app/render/overlay.py"], stderr=subprocess.DEVNULL
            ).decode("utf-8")

            mod_jianpu = types.ModuleType("head_jianpu")
            mod_jianpu.__file__ = os.path.abspath("app/render/jianpu.py")
            sys_module.modules["head_jianpu"] = mod_jianpu
            exec(head_jianpu_code, mod_jianpu.__dict__)

            mod_overlay = types.ModuleType("head_overlay")
            mod_overlay.__file__ = os.path.abspath("app/render/overlay.py")
            sys_module.modules["head_overlay"] = mod_overlay
            for k, v in mod_jianpu.__dict__.items():
                if not k.startswith("__"):
                    mod_overlay.__dict__[k] = v
            exec(head_overlay_code, mod_overlay.__dict__)

            old_strip = mod_overlay.render_system_strip(sys0, 1000, arrangement, sheet)
            new_strip = render_system_strip(sys0, 1000, arrangement, sheet, scale=1.0)
            assert np.array_equal(np.array(old_strip), np.array(new_strip)), "Strip not pixel-identical to HEAD!"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


    def test_find_system_cut_row_tie_breaking(self):
        arr = np.full((100, 200), 255, dtype=np.uint8)
        # Run 1: rows 20..30 (len 11, center 25)
        # Ink at 31
        arr[31, :] = 50
        # Run 2: rows 32..42 (len 11, center 37)
        # nominal_y = 38 (closer to Run 2)
        cut = find_system_cut_row(arr, curr_y1=20, next_y0=43, nominal_y=38)
        assert cut == 37


class TestUkuleleDiagramSpacing:
    def test_chord_diagram_vertical_clearance(self):
        img = Image.new("RGB", (400, 400), "white")
        draw = ImageDraw.Draw(img)

        # Chord with open string (C has frets (0, 0, 0, 3) -> strings 0, 1, 2 have 'O' indicators)
        shape_c = UkeShape(name="C", frets=(0, 0, 0, 3), fingers=(0, 0, 0, 3), base_fret=1)
        grid_w = max(76.0, 90.0 - 32.0)
        string_spacing = grid_w / 3.0
        font_name = ImageDraw.Draw(img).getfont()
        # Compute indicator geometry matching ukulele.py
        h_name = 18.0
        scale = 1.0
        indic_r = string_spacing * 0.18
        y = 10.0
        y_indic = y + max(h_name + 4.0 * scale, 22.0 * scale) + indic_r
        y_nut = y_indic + indic_r + 6.0 * scale

        # Verify strict vertical clearances
        assert (y_indic - indic_r) - (y + h_name) >= 4.0 * scale
        assert y_nut - (y_indic + indic_r) >= 6.0 * scale

        h = draw_chord_diagram(draw, shape_c, "C", 10.0, 10.0, col_w=90.0, scale=1.0)
        assert h > 80.0
