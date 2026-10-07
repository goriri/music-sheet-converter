#!/usr/bin/env python3
"""Fetch, select, convert, and package evaluation datasets (RWC-P and MIR-1K).

Runs inside Cloud Build with high CPU/disk.
1. Downloads RWC-P.zip (Zenodo), extracts 12 target songs (~8 Japanese + ~4 English).
2. Converts RWC tracks to 44.1kHz mono WAV with ffmpeg.
3. Clones MIR-1K repo, extracts 12 clips across 12 singers.
4. Converts MIR-1K clips to mixed mono 44.1kHz WAV with ffmpeg (-ac 1 mixes L+R).
5. Gathers MIR-1K ground truth labels (.pv and .txt).
6. Generates manifest.json.
7. Uploads audio, labels, and manifest to GCS bucket gs://cellular-cider-495602-r9-sheet-eval.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_eval_data")

EVAL_BUCKET = os.environ.get("EVAL_BUCKET", "cellular-cider-495602-r9-sheet-eval")
ZENODO_RWC_URL = os.environ.get(
    "ZENODO_RWC_URL",
    "https://zenodo.org/records/18656623/files/RWC-P.zip?download=1",
)
MIR1K_REPO_URL = os.environ.get(
    "MIR1K_REPO_URL",
    "https://github.com/ryanwhite04/MIR-1K.git",
)

# 12 RWC DEV tracks: 8 Japanese, 4 English
RWC_DEV_SELECTION = [
    {"id": "RWC_P001", "lang": "Japanese", "title": "Eien no replica", "artist": "Kazuo Nishi", "split": "dev"},
    {"id": "RWC_P012", "lang": "Japanese", "title": "Drive me crazy", "artist": "Mika", "split": "dev"},
    {"id": "RWC_P023", "lang": "Japanese", "title": "Happy Days", "artist": "Jun Sato", "split": "dev"},
    {"id": "RWC_P034", "lang": "Japanese", "title": "Kimi to ireba", "artist": "Yuko", "split": "dev"},
    {"id": "RWC_P045", "lang": "Japanese", "title": "Together", "artist": "Eishi Segawa", "split": "dev"},
    {"id": "RWC_P056", "lang": "Japanese", "title": "I sing for you", "artist": "Ayumi", "split": "dev"},
    {"id": "RWC_P067", "lang": "Japanese", "title": "Sweet Memories", "artist": "Kenji", "split": "dev"},
    {"id": "RWC_P078", "lang": "Japanese", "title": "Sayonara no yukue", "artist": "Chie", "split": "dev"},
    {"id": "RWC_P081", "lang": "English", "title": "How Deep Is Your Love?", "artist": "Lauren", "split": "dev"},
    {"id": "RWC_P085", "lang": "English", "title": "Waiting for the moment", "artist": "Chris", "split": "dev"},
    {"id": "RWC_P092", "lang": "English", "title": "Take a Chance", "artist": "Sandra", "split": "dev"},
    {"id": "RWC_P098", "lang": "English", "title": "That's the Way It Goes", "artist": "Steve", "split": "dev"},
]

# 12 RWC TEST tracks: 8 Japanese, 4 English (strictly evaluated once at the end)
RWC_TEST_SELECTION = [
    {"id": "RWC_P002", "lang": "Japanese", "title": "Magic in your eyes", "artist": "Mika", "split": "test"},
    {"id": "RWC_P011", "lang": "Japanese", "title": "Ienai", "artist": "Shinya", "split": "test"},
    {"id": "RWC_P021", "lang": "Japanese", "title": "Feeling In My Heart", "artist": "Hiroko", "split": "test"},
    {"id": "RWC_P031", "lang": "Japanese", "title": "Moving Round and Round", "artist": "Takeo", "split": "test"},
    {"id": "RWC_P041", "lang": "Japanese", "title": "Non Stop Driving", "artist": "Ken", "split": "test"},
    {"id": "RWC_P051", "lang": "Japanese", "title": "Modoranai natsu", "artist": "Tetsuo", "split": "test"},
    {"id": "RWC_P061", "lang": "Japanese", "title": "FOR YOU", "artist": "Shin", "split": "test"},
    {"id": "RWC_P071", "lang": "Japanese", "title": "Tsuki no youni", "artist": "Mayumi", "split": "test"},
    {"id": "RWC_P082", "lang": "English", "title": "Once in a life time", "artist": "David", "split": "test"},
    {"id": "RWC_P087", "lang": "English", "title": "I think of you", "artist": "Mark", "split": "test"},
    {"id": "RWC_P091", "lang": "English", "title": "Change Of Heart", "artist": "Sarah", "split": "test"},
    {"id": "RWC_P096", "lang": "English", "title": "Weekend", "artist": "Emily", "split": "test"},
]

# 12 MIR-1K DEV clips
MIR1K_DEV_SELECTION = [
    {"id": "abjones_3_09", "singer": "abjones", "split": "dev"},
    {"id": "amy_2_05", "singer": "amy", "split": "dev"},
    {"id": "Ani_3_06", "singer": "Ani", "split": "dev"},
    {"id": "annar_3_06", "singer": "annar", "split": "dev"},
    {"id": "ariel_3_05", "singer": "ariel", "split": "dev"},
    {"id": "bobon_3_08", "singer": "bobon", "split": "dev"},
    {"id": "davidson_3_07", "singer": "davidson", "split": "dev"},
    {"id": "geniusturtle_4_12", "singer": "geniusturtle", "split": "dev"},
    {"id": "heycat_3_06", "singer": "heycat", "split": "dev"},
    {"id": "jmzen_3_05", "singer": "jmzen", "split": "dev"},
    {"id": "tammy_1_05", "singer": "tammy", "split": "dev"},
    {"id": "yifen_3_02", "singer": "yifen", "split": "dev"},
]

# 12 MIR-1K TEST clips: 7 from remaining singers + 5 distinct clips
MIR1K_TEST_SELECTION = [
    {"id": "bug_3_08", "singer": "bug", "split": "test"},
    {"id": "fdps_2_11", "singer": "fdps", "split": "test"},
    {"id": "Kenshin_3_07", "singer": "Kenshin", "split": "test"},
    {"id": "khair_4_01", "singer": "khair", "split": "test"},
    {"id": "leon_5_05", "singer": "leon", "split": "test"},
    {"id": "stool_3_06", "singer": "stool", "split": "test"},
    {"id": "titon_3_06", "singer": "titon", "split": "test"},
    {"id": "abjones_3_03", "singer": "abjones", "split": "test"},
    {"id": "amy_2_01", "singer": "amy", "split": "test"},
    {"id": "Ani_3_03", "singer": "Ani", "split": "test"},
    {"id": "bobon_3_10", "singer": "bobon", "split": "test"},
    {"id": "davidson_3_10", "singer": "davidson", "split": "test"},
]


def run(cmd: list[str], cwd: str | None = None) -> None:
    logger.info("Executing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    work_dir = Path("/tmp/eval_build")
    work_dir.mkdir(parents=True, exist_ok=True)

    export_dir = Path("/tmp/export")
    rwc_out_dir = export_dir / "audio" / "rwc"
    mir1k_out_dir = export_dir / "audio" / "mir1k"
    mir1k_labels_dir = export_dir / "labels" / "mir1k"

    rwc_out_dir.mkdir(parents=True, exist_ok=True)
    mir1k_out_dir.mkdir(parents=True, exist_ok=True)
    mir1k_labels_dir.mkdir(parents=True, exist_ok=True)

    all_rwc = RWC_DEV_SELECTION + RWC_TEST_SELECTION
    all_mir1k = MIR1K_DEV_SELECTION + MIR1K_TEST_SELECTION

    manifest_entries: list[dict] = []

    # -------------------------------------------------------------------------
    # 1. Download RWC-P.zip (from GCS cache or Zenodo) and extract tracks
    # -------------------------------------------------------------------------
    zip_path = work_dir / "RWC-P.zip"
    cache_gcs_uri = f"gs://{EVAL_BUCKET}/cache/RWC-P.zip"
    downloaded_from_cache = False
    try:
        logger.info("Checking if cached RWC-P.zip exists at %s...", cache_gcs_uri)
        res = subprocess.run(["gcloud", "storage", "stat", cache_gcs_uri], capture_output=True)
        if res.returncode == 0:
            logger.info("Found cached RWC-P.zip in GCS! Downloading from GCS...")
            run(["gcloud", "storage", "cp", cache_gcs_uri, str(zip_path)])
            downloaded_from_cache = True
    except Exception as e:
        logger.warning("Cache check failed: %s", e)

    if not downloaded_from_cache:
        logger.info("=== Step 1: Downloading RWC-P.zip from %s ===", ZENODO_RWC_URL)
        run(["curl", "-fSL", "--retry", "5", "--retry-delay", "5", "-o", str(zip_path), ZENODO_RWC_URL])
        logger.info("Downloaded RWC-P.zip: %d bytes. Uploading to cache %s...", zip_path.stat().st_size, cache_gcs_uri)
        try:
            run(["gcloud", "storage", "cp", str(zip_path), cache_gcs_uri])
        except Exception as up_exc:
            logger.warning("Failed uploading zip to cache: %s", up_exc)

    logger.info("Extracting %d selected RWC-P tracks from zip...", len(all_rwc))
    rwc_raw_dir = work_dir / "rwc_raw"
    rwc_raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = set(zf.namelist())
        for track in all_rwc:
            t_id = track["id"]
            # Candidates inside zip
            cands = [f"RWC-P/{t_id}.wav", f"{t_id}.wav"]
            found_name = None
            for c in cands:
                if c in namelist:
                    found_name = c
                    break
            if not found_name:
                raise FileNotFoundError(f"Track {t_id} not found in RWC-P.zip!")

            dest_path = rwc_raw_dir / f"{t_id}.wav"
            with zf.open(found_name) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("Extracted %s -> %s (%d bytes)", found_name, dest_path, dest_path.stat().st_size)

    # Delete zip to free up ~4 GB disk immediately
    logger.info("Deleting RWC-P.zip to free disk space...")
    zip_path.unlink()

    # Convert RWC tracks to 44.1k mono WAV
    logger.info("Converting RWC tracks to 44.1kHz mono WAV with ffmpeg...")
    for track in all_rwc:
        t_id = track["id"]
        in_wav = rwc_raw_dir / f"{t_id}.wav"
        out_wav = rwc_out_dir / f"{t_id}.wav"
        run(["ffmpeg", "-y", "-i", str(in_wav), "-ac", "1", "-ar", "44100", str(out_wav)])
        logger.info("Converted %s -> %s (%d bytes)", in_wav.name, out_wav.name, out_wav.stat().st_size)

        manifest_entries.append({
            "id": t_id,
            "dataset": "rwc",
            "audio_gcs": f"gs://{EVAL_BUCKET}/audio/rwc/{t_id}.wav",
            "title": track["title"],
            "artist": track["artist"],
            "language": track["lang"],
            "split": track["split"],
        })

    # Clean up raw RWC
    shutil.rmtree(rwc_raw_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 2. Clone MIR-1K and extract clips
    # -------------------------------------------------------------------------
    logger.info("=== Step 2: Cloning MIR-1K repository from %s ===", MIR1K_REPO_URL)
    mir1k_repo = work_dir / "mir1k_repo"
    run(["git", "clone", "--depth", "1", MIR1K_REPO_URL, str(mir1k_repo)])

    logger.info("Processing %d MIR-1K clips (converting stereo L=acc/R=voc to mixed mono 44.1k)...", len(all_mir1k))
    for clip_info in all_mir1k:
        cid = clip_info["id"]
        singer = clip_info["singer"]
        wav_path = mir1k_repo / "Wavfile" / f"{cid}.wav"
        pv_path = mir1k_repo / "PitchLabel" / f"{cid}.pv"
        txt_path = mir1k_repo / "Lyrics" / f"{cid}.txt"

        if not wav_path.exists():
            raise FileNotFoundError(f"MIR-1K wav missing: {wav_path}")
        if not pv_path.exists():
            raise FileNotFoundError(f"MIR-1K pv missing: {pv_path}")
        if not txt_path.exists():
            raise FileNotFoundError(f"MIR-1K txt missing: {txt_path}")

        # Convert stereo to mixed mono: -ac 1 mixes left (accompaniment) and right (vocal)
        out_wav = mir1k_out_dir / f"{cid}.wav"
        run(["ffmpeg", "-y", "-i", str(wav_path), "-ac", "1", "-ar", "44100", str(out_wav)])

        # Copy labels
        shutil.copy2(pv_path, mir1k_labels_dir / f"{cid}.pv")
        shutil.copy2(txt_path, mir1k_labels_dir / f"{cid}.txt")

        manifest_entries.append({
            "id": cid,
            "dataset": "mir1k",
            "audio_gcs": f"gs://{EVAL_BUCKET}/audio/mir1k/{cid}.wav",
            "title": cid,
            "artist": singer,
            "language": "Mandarin",
            "split": clip_info["split"],
        })
        logger.info("Processed MIR-1K clip %s (audio + pv + txt)", cid)

    # Clean up MIR-1K repo
    shutil.rmtree(mir1k_repo, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 3. Generate manifest.json
    # -------------------------------------------------------------------------
    logger.info("=== Step 3: Generating manifest.json (%d items) ===", len(manifest_entries))
    manifest_file = export_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_entries, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------------------------
    # 4. Upload to GCS bucket
    # -------------------------------------------------------------------------
    logger.info("=== Step 4: Uploading export directory to gs://%s/ ===", EVAL_BUCKET)
    run(["gcloud", "storage", "cp", "-r", f"{export_dir}/*", f"gs://{EVAL_BUCKET}/"])

    logger.info("=== Verifying uploaded files in gs://%s/ ===", EVAL_BUCKET)
    run(["gcloud", "storage", "ls", f"gs://{EVAL_BUCKET}/"])
    run(["gcloud", "storage", "ls", f"gs://{EVAL_BUCKET}/audio/rwc/"])
    run(["gcloud", "storage", "ls", f"gs://{EVAL_BUCKET}/audio/mir1k/"])
    run(["gcloud", "storage", "ls", f"gs://{EVAL_BUCKET}/labels/mir1k/"])

    logger.info("Successfully fetched, converted, and uploaded all evaluation data!")


if __name__ == "__main__":
    main()
