#!/usr/bin/env python3
"""Backfill script for populating history entries, thumbnails, and render meta for existing sheets."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path so app modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.history import restore_sheet, run_backfill
from app.storage import LocalStorage, get_storage, set_storage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Idempotent history index backfill and sheet restore tool."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate backfill without writing any files.",
    )
    parser.add_argument(
        "--restore",
        metavar="SHEET_ID",
        help="Restore a soft-deleted sheet and re-create its history index entry.",
    )
    parser.add_argument(
        "--storage-dir",
        metavar="DIR",
        help="Path to local storage directory (overrides LOCAL_STORAGE_DIR env var).",
    )

    args = parser.parse_args()

    if args.storage_dir:
        store = LocalStorage(args.storage_dir)
        set_storage(store)
    else:
        store = get_storage()

    if args.restore:
        sheet_id = args.restore.strip()
        print(f"Restoring sheet '{sheet_id}'...")
        success = restore_sheet(sheet_id, store)
        if success:
            print(f"✓ Sheet '{sheet_id}' successfully restored and indexed in history.")
            return 0
        else:
            print(f"✗ Failed to restore sheet '{sheet_id}' (not found or state.json missing).", file=sys.stderr)
            return 1

    mode_str = "[DRY RUN] " if args.dry_run else ""
    print(f"{mode_str}Starting history index backfill...")

    stats = run_backfill(store, dry_run=args.dry_run)

    print(f"\n{mode_str}Backfill summary:")
    print(f"  - Sheets scanned:          {stats.get('sheets_scanned', 0)}")
    print(f"  - History entries created: {stats.get('history_entries_created', 0)}")
    print(f"  - Thumbnails generated:    {stats.get('thumbnails_created', 0)}")
    print(f"  - Render meta generated:   {stats.get('render_meta_created', 0)}")
    print(f"✓ {mode_str}Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
