"""Update golden test fixtures for arrangement output."""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.arrange.piano import arrange
from app.models import ParsedSheet
from app.qa.arrange_check import build_compact_score_text


def update_goldens(omr_path: Path, golden_dir: Path) -> None:
    """Generate and write golden files for F/G across all 3 levels."""
    with open(omr_path, encoding="utf-8") as f:
        sheet = ParsedSheet.model_validate(json.load(f))

    golden_dir.mkdir(parents=True, exist_ok=True)
    keys = ["F", "G"]
    levels = ["beginner", "intermediate", "advanced"]

    for key in keys:
        for level in levels:
            out_file = golden_dir / f"{key}_{level}.txt"
            print(f"Generating golden for {key} {level} -> {out_file}...")
            arr = arrange(sheet, key, level)
            score_text = build_compact_score_text(arr)
            out_file.write_text(score_text, encoding="utf-8")

    print(f"Successfully generated {len(keys) * len(levels)} golden files in {golden_dir}")


if __name__ == "__main__":
    omr_sample = REPO_ROOT / "fixtures" / "omr_sample.json"
    goldens = REPO_ROOT / "tests" / "golden"
    if not omr_sample.exists():
        print(f"Error: {omr_sample} does not exist", file=sys.stderr)
        sys.exit(1)
    update_goldens(omr_sample, goldens)
