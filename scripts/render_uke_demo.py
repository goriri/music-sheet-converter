import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.arrange.sections import plan_sections
from app.arrange.ukulele import arrange_ukulele
from app.models import ParsedSheet
from app.render.ukulele import render_uke_pages, render_uke_pdf


def main() -> None:
    out_dir = Path("out/uke_render")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 掉了 (key F, advanced) from fresh gt_eval
    print("Rendering 掉了 (advanced, key F)...")
    diaole_json = Path("out/gt_eval/diaole.json")
    diaole_sheet = ParsedSheet.model_validate(json.loads(diaole_json.read_text(encoding="utf-8")))

    p1_bytes = Path("fixtures/pages/page1.jpg").read_bytes()
    p2_bytes = Path("fixtures/pages/page2.jpg").read_bytes()
    diaole_pages_bytes = [p1_bytes, p2_bytes]

    diaole_sections = plan_sections(diaole_sheet)

    # Advanced
    arr_adv = arrange_ukulele(diaole_sheet, "F", "advanced", sections=diaole_sections)
    pages_adv = render_uke_pages(diaole_pages_bytes, diaole_sheet, arr_adv)
    for idx, page in enumerate(pages_adv):
        path = out_dir / f"diaole_advanced_p{idx}.png"
        page.save(path, format="PNG")
        print(f"Saved: {path} ({page.size[0]}x{page.size[1]})")

    # Beginner (p0)
    arr_beg = arrange_ukulele(diaole_sheet, "F", "beginner", sections=diaole_sections)
    pages_beg = render_uke_pages(diaole_pages_bytes, diaole_sheet, arr_beg)
    for idx, page in enumerate(pages_beg):
        path = out_dir / f"diaole_beginner_p{idx}.png"
        page.save(path, format="PNG")
        print(f"Saved: {path} ({page.size[0]}x{page.size[1]})")

    # 2. 听海 (key Eb, intermediate — capo 3 case) from fresh gt_eval
    print("Rendering 听海 (intermediate, key Eb with capo 3)...")
    tinghai_json = Path("out/gt_eval/tinghai.json")
    tinghai_sheet = ParsedSheet.model_validate(json.loads(tinghai_json.read_text(encoding="utf-8")))

    th_p1 = Path("fixtures/external/tinghai/page1.jpg").read_bytes()
    th_p2 = Path("fixtures/external/tinghai/page2.jpg").read_bytes()
    tinghai_pages_bytes = [th_p1, th_p2]

    tinghai_sections = plan_sections(tinghai_sheet)
    arr_tinghai = arrange_ukulele(tinghai_sheet, "Eb", "intermediate", sections=tinghai_sections)
    pages_tinghai = render_uke_pages(tinghai_pages_bytes, tinghai_sheet, arr_tinghai)
    for idx, page in enumerate(pages_tinghai):
        path = out_dir / f"tinghai_intermediate_p{idx}.png"
        page.save(path, format="PNG")
        print(f"Saved: {path} ({page.size[0]}x{page.size[1]})")

    # 3. 小白船 (3/4 time signature, beginner, key Eb with capo 3) from fresh gt_eval
    print("Rendering 小白船 (3/4, beginner, key Eb)...")
    xbc_json = Path("out/gt_eval/xiaobaichuan.json")
    xbc_sheet = ParsedSheet.model_validate(json.loads(xbc_json.read_text(encoding="utf-8")))

    xbc_p1 = Path("fixtures/external/xiaobaichuan/page1.jpg").read_bytes()
    xbc_p2 = Path("fixtures/external/xiaobaichuan/page2.jpg").read_bytes()
    xbc_pages_bytes = [xbc_p1, xbc_p2]

    xbc_sections = plan_sections(xbc_sheet)
    arr_xbc = arrange_ukulele(xbc_sheet, "Eb", "beginner", sections=xbc_sections)
    pages_xbc = render_uke_pages(xbc_pages_bytes, xbc_sheet, arr_xbc)
    for idx, page in enumerate(pages_xbc):
        path = out_dir / f"xiaobaichuan_beginner_p{idx}.png"
        page.save(path, format="PNG")
        print(f"Saved: {path} ({page.size[0]}x{page.size[1]})")

    print("All requested demo renders generated successfully in out/uke_render/")


if __name__ == "__main__":
    main()
