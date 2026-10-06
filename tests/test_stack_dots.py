"""Block-chord (柱式和弦) stacks: octave dots between stacked digits must not touch the neighbour digit."""
import pytest
from PIL import Image, ImageDraw

from app.render.jianpu import JianpuNote, draw_chord_stack, draw_jianpu_note, get_font, stack_offsets


def _note(degree, dots, midi):
    return JianpuNote(degree=degree, accidental="", octave_dots=dots, midi=midi, finger=None)


def _extents(scale, notes, line_spacing):
    """Return per-note (top_y incl. dots, bottom_y incl. dots, digit_top, digit_bottom) as drawn in a stack."""
    img = Image.new("RGB", (200, 400), "white")
    d = ImageDraw.Draw(img)
    font = get_font(size=int(round(20 * scale)))
    acc = get_font(size=int(round(16 * scale)))
    fin = get_font(size=int(round(11 * scale)), bold=True)
    lay = draw_chord_stack(d, 50, 300, notes, font, acc, fin, line_spacing=line_spacing, draw_finger=False, scale=scale)
    out = []
    for n, y in zip(lay.notes, lay.digit_y_positions):
        _, top, bot = draw_jianpu_note(d, 50, y, n, font, acc, scale=scale)
        bb = d.textbbox((50, y), str(n.degree), font=font)
        out.append((top, bot, bb[1], bb[3]))
    return out


@pytest.mark.parametrize("scale", [1.0, 2.27])
@pytest.mark.parametrize(
    "notes",
    [
        # 5 1̇ 3̇ : the dot above 1̇ sits between 1̇ and 3̇
        [_note(5, 0, 67), _note(1, 1, 72), _note(3, 1, 76)],
        # 6̣ 1 3 (LH style): the dot below 6̣ is outside; 7̣ under 2
        [_note(7, -1, 59), _note(2, 0, 62), _note(4, 0, 65)],
        # 7 2̇ with dot above 7? (no) and two-dot cases
        [_note(5, 1, 79), _note(1, 2, 84)],
        [_note(3, -2, 40), _note(5, -1, 43)],
    ],
)
def test_stack_dots_clear_neighbours(scale, notes):
    ls = 18.0 * scale
    ext = _extents(scale, notes, ls)
    for (lo, hi) in zip(ext, ext[1:]):
        lo_top_incl_dots, _, lo_digit_top, _ = lo
        _, hi_bot_incl_dots, _, hi_digit_bottom = hi
        # lower note's top (incl. its dots above) must stay below the upper digit's bottom
        assert lo_top_incl_dots >= hi_digit_bottom + 1.0 * scale
        # upper note's dots below must stay above the lower digit's top
        assert hi_bot_incl_dots <= lo_digit_top - 1.0 * scale


def test_offsets_unchanged_without_inner_dots():
    notes = [_note(1, 0, 60), _note(3, 0, 64), _note(5, 0, 67)]
    assert stack_offsets(notes, 18.0, 1.0) == [0.0, 18.0, 36.0]
