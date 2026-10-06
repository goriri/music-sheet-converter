"""Cloud Run Job & Local CLI entrypoint for audio transcription pipeline.

Usage:
  1. Cloud Run Job:
     python -m app.audio.job_main
     (Reads env SMART_JOB_ID, BUCKET; downloads private/smart/{id}/input.<ext>,
      updates private/smart/{id}/status.json, writes private/smart/{id}/analysis.json
      and private/smart/{id}/audio_lead.json via app.storage).

  2. Local Run:
     python -m app.audio.job_main --local path/to/audio.mp3 --out out_dir/ [--title "Title"] [--artist "Artist"]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Optional

from app.audio.to_lead import analysis_to_lead
from app.audio.transcribe import transcribe
from app.storage import get_storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.audio.job_main")


def _norm_qual(q: str) -> str:
    """Normalize chord quality names for evaluation comparisons."""
    ql = q.lower()
    if ql in ("min", "m"):
        return "m"
    if ql in ("maj", ""):
        return "maj"
    if ql in ("min7", "m7"):
        return "m7"
    if ql in ("maj7",):
        return "maj7"
    if ql in ("7", "dom7"):
        return "7"
    if ql in ("dim", "dim7"):
        return "dim"
    if ql in ("hdim7", "m7b5", "m7-5"):
        return "m7b5"
    if ql in ("sus4", "sus"):
        return "sus4"
    return ql


def run_cloud_run_job() -> None:
    """Execute transcription pipeline inside a Cloud Run Job environment."""
    job_id = os.environ.get("SMART_JOB_ID")
    if not job_id:
        logger.error("SMART_JOB_ID environment variable is missing")
        sys.exit(1)

    bucket = os.environ.get("BUCKET")
    logger.info("Starting Cloud Run Job: SMART_JOB_ID=%s, BUCKET=%s", job_id, bucket or "(local storage fallback)")

    storage = get_storage()
    prefix = f"private/smart/{job_id}"
    status_path = f"{prefix}/status.json"
    analysis_path = f"{prefix}/analysis.json"
    lead_path = f"{prefix}/audio_lead.json"

    def write_status(status: str, progress: float, stage: str, error: Optional[str] = None):
        payload = {
            "status": status,
            "progress": round(progress, 3),
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            payload["error"] = error
        try:
            storage.put_json(status_path, payload)
        except Exception as exc:
            logger.warning("Failed updating status to %s: %s", status_path, exc)

    write_status("running", 0.02, "查找上传的音频文件")

    # Locate input audio file under private/smart/{job_id}/input.*
    candidates = storage.list(f"{prefix}/input")
    input_storage_path = None
    if candidates:
        input_storage_path = candidates[0]
    else:
        for ext in [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]:
            cand = f"{prefix}/input{ext}"
            if storage.exists(cand):
                input_storage_path = cand
                break

    if not input_storage_path:
        err_msg = f"未找到音频文件: {prefix}/input.*"
        logger.error(err_msg)
        write_status("error", 0.0, "未找到待处理的音频文件", error=err_msg)
        sys.exit(1)

    logger.info("Found input audio at: %s", input_storage_path)

    current_progress = 0.05
    last_stage_text = "下载音频文件"

    with tempfile.TemporaryDirectory(prefix=f"smart_audio_{job_id}_") as work_dir:
        local_input = os.path.join(work_dir, os.path.basename(input_storage_path))
        try:
            write_status("running", 0.05, "下载音频文件到计算节点")
            audio_bytes = storage.get_bytes(input_storage_path)
            with open(local_input, "wb") as f:
                f.write(audio_bytes)
            logger.info("Downloaded %d bytes to %s", len(audio_bytes), local_input)

            def progress_cb(pct: float, stage_text: str):
                nonlocal current_progress, last_stage_text
                current_progress = pct
                last_stage_text = stage_text
                write_status("running", pct, stage_text)

            # Optional song title/artist metadata from private/smart/{job_id}/meta.json if present
            meta_title = ""
            meta_artist = ""
            meta_path = f"{prefix}/meta.json"
            if storage.exists(meta_path):
                try:
                    meta = storage.get_json(meta_path)
                    meta_title = meta.get("title", "")
                    meta_artist = meta.get("artist", "")
                except Exception:
                    pass

            # 1. Transcribe audio to AudioAnalysis
            analysis = transcribe(local_input, work_dir, progress=progress_cb)

            # 2. Convert AudioAnalysis to LeadSheet
            write_status("running", 0.98, "生成简谱 LeadSheet 结构")
            lead = analysis_to_lead(analysis, title=meta_title, artist=meta_artist)

            # 3. Store analysis.json and audio_lead.json
            logger.info("Saving analysis.json to %s", analysis_path)
            storage.put_json(analysis_path, analysis.model_dump())

            logger.info("Saving audio_lead.json to %s", lead_path)
            storage.put_json(lead_path, lead.model_dump())

            # 4. Mark job as complete
            write_status("done", 1.0, "转录完成")
            logger.info("Job %s completed successfully!", job_id)

        except Exception as exc:
            logger.exception("Job %s failed with exception: %s", job_id, exc)
            write_status("error", current_progress, f"转录失败: {exc}", error=str(exc))
            sys.exit(1)


def run_local(
    audio_path: str,
    out_dir: str,
    title: str = "",
    artist: str = "",
) -> None:
    """Execute transcription pipeline locally and write outputs to a local directory."""
    if not os.path.isfile(audio_path):
        logger.error("Local audio file not found: %s", audio_path)
        sys.exit(1)

    out_p = Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    work_dir = str(out_p / ".work")
    os.makedirs(work_dir, exist_ok=True)

    logger.info("Processing local audio: %s -> output directory: %s", audio_path, out_p)

    def progress_cb(pct: float, stage_text: str):
        print(f"[{int(pct * 100):3d}%] {stage_text}")

    analysis = transcribe(audio_path, work_dir, progress=progress_cb)
    lead = analysis_to_lead(analysis, title=title, artist=artist)

    analysis_file = out_p / "analysis.json"
    lead_file = out_p / "audio_lead.json"

    analysis_file.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    lead_file.write_text(lead.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Done! Written %s and %s", analysis_file, lead_file)
    print(f"\nTranscription complete:")
    print(f"  Key: {lead.key}")
    print(f"  Measures: {len(lead.measures)}")
    print(f"  Chords: {sum(len(m.chords) for m in lead.measures)}")
    print(f"  Notes: {sum(len(m.notes) for m in lead.measures)}")
    print(f"  Outputs: {analysis_file}, {lead_file}")


def run_cloud_synth_eval() -> None:
    """Execute synthetic evaluation mode inside Cloud Run Job.

    Synthesizes test audio from fixtures/groundtruth/tinghai.json and diaole.json,
    runs full transcription pipeline, computes evaluation metrics against GT,
    and uploads report + analysis/lead JSONs to GCS tmp/smart_eval/<run_id>/.
    """
    import importlib.util
    import math
    import time
    import numpy as np
    import soundfile as sf
    from app.theory.chords import resolve_chord
    from app.theory.keys import DEGREE_SEMITONES, KEY_NAME_TO_PC, key_name_to_pc

    def _midi_to_hz(midi: float) -> float:
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    def _load_parse_melody():
        cand_paths = [
            Path(__file__).resolve().parent.parent / "qa" / "melody.py",
            Path("app/qa/melody.py").resolve(),
            Path("/app/app/qa/melody.py"),
        ]
        for p in cand_paths:
            if p.is_file():
                spec = importlib.util.spec_from_file_location("app_qa_melody", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.parse_melody
        raise FileNotFoundError("Could not locate app/qa/melody.py")

    parse_melody = _load_parse_melody()

    def _synthesize_song(
        title: str,
        key_str: str,
        measures_data: list[dict],
        out_wav_path: str,
        tempo_bpm: float = 80.0,
        sr: int = 22050,
    ) -> tuple[list[float], list[tuple[float, int, int]], list[tuple[float, str, int, str]]]:
        sec_per_beat = 60.0 / tempo_bpm
        base_tonic_pc = key_name_to_pc(key_str)
        total_dur = sum(float(m.get("beats", 4.0)) * sec_per_beat for m in measures_data)
        total_samples = int(math.ceil(total_dur * sr))

        audio_acc = np.zeros(total_samples, dtype=float)
        audio_voc = np.zeros(total_samples, dtype=float)

        gt_downbeats: list[float] = []
        gt_melody_notes: list[tuple[float, int, int]] = []
        gt_beat_chords: list[tuple[float, str, int, str]] = []

        cur_t = 0.0
        for m_idx, m_info in enumerate(measures_data):
            m_start_t = cur_t
            m_beats = float(m_info.get("beats", 4.0))
            m_dur = m_beats * sec_per_beat
            cur_t += m_dur
            m_tonic_pc = m_info.get("tonic_pc", base_tonic_pc)

            gt_downbeats.append(m_start_t)

            chords_in_m = m_info.get("chords", [])
            num_m_beats = int(round(m_beats))
            m_beat_chords = [None] * num_m_beats
            cur_chord = chords_in_m[0] if chords_in_m else None
            for b in range(1, num_m_beats + 1):
                for c in chords_in_m:
                    if abs(c.get("beat", 1.0) - float(b)) < 0.2:
                        cur_chord = c
                m_beat_chords[b - 1] = cur_chord

            for b_idx in range(num_m_beats):
                beat_t = m_start_t + b_idx * sec_per_beat
                c_entry = m_beat_chords[b_idx]
                raw_c = c_entry.get("raw", "1") if c_entry else "1"
                try:
                    resolved = resolve_chord(raw_c, tonic_pc=m_tonic_pc, beat=float(b_idx + 1))
                    r_pc = resolved.root_pc
                    b_pc = resolved.bass_pc if resolved.bass_pc is not None else r_pc
                    pcs = resolved.pcs
                    qual = resolved.quality
                except Exception:
                    r_pc = m_tonic_pc
                    b_pc = m_tonic_pc
                    pcs = [(m_tonic_pc + i) % 12 for i in [0, 4, 7]]
                    qual = "maj"

                gt_beat_chords.append((beat_t, raw_c, r_pc, qual))

                # Drum click
                click_len = int(0.015 * sr)
                click_start = int(beat_t * sr)
                if click_start + click_len < total_samples:
                    click_t = np.linspace(0, 0.015, click_len)
                    audio_acc[click_start : click_start + click_len] += 0.4 * np.sin(2 * np.pi * 1200 * click_t) * np.exp(-click_t / 0.003)

                # Bass note
                prev_raw = m_beat_chords[b_idx - 1].get("raw") if (b_idx > 0 and m_beat_chords[b_idx - 1]) else None
                if b_idx in (0, 2) or (b_idx > 0 and raw_c != prev_raw):
                    bass_midi = 36 + b_pc
                    bass_hz = _midi_to_hz(bass_midi)
                    bass_len = int(sec_per_beat * 1.8 * sr)
                    b_start = int(beat_t * sr)
                    b_end = min(total_samples, b_start + bass_len)
                    actual_len = b_end - b_start
                    if actual_len > 0:
                        bt = np.linspace(0, actual_len / sr, actual_len)
                        env = np.exp(-bt / (sec_per_beat * 0.8))
                        bass_wave = 0.5 * np.sin(2 * np.pi * bass_hz * bt) + 0.25 * np.sin(4 * np.pi * bass_hz * bt)
                        audio_acc[b_start:b_end] += bass_wave * env

                # Chord tones
                chord_len = int(sec_per_beat * 1.5 * sr)
                c_start = int(beat_t * sr)
                c_end = min(total_samples, c_start + chord_len)
                actual_c_len = c_end - c_start
                if actual_c_len > 0:
                    ct = np.linspace(0, actual_c_len / sr, actual_c_len)
                    env = np.exp(-ct / (sec_per_beat * 0.7))
                    chord_wave = np.zeros(actual_c_len)
                    for pc in pcs:
                        note_midi = 60 + pc
                        hz = _midi_to_hz(note_midi)
                        chord_wave += 0.15 * np.sin(2 * np.pi * hz * ct) + 0.05 * np.sin(4 * np.pi * hz * ct)
                    audio_acc[c_start:c_end] += chord_wave * env

            melody_str = m_info.get("melody")
            if melody_str:
                parsed = parse_melody(melody_str, beats=m_beats, include_rests=False)
                if parsed and parsed.notes:
                    for note_idx, (onset_beat, deg, acc) in enumerate(parsed.notes):
                        if deg <= 0:
                            continue
                        if note_idx + 1 < len(parsed.notes):
                            next_onset = parsed.notes[note_idx + 1][0]
                            dur_beat = max(0.25, next_onset - onset_beat)
                        else:
                            dur_beat = max(0.5, m_beats - onset_beat)

                        base_semitone = DEGREE_SEMITONES[deg - 1] + acc
                        note_pc = (m_tonic_pc + base_semitone) % 12
                        octave_shift = int(m_info.get("octave_shift", 0))
                        note_midi = 60 + m_tonic_pc + base_semitone + octave_shift
                        if note_midi < 54:
                            note_midi += 12
                        elif note_midi > 78:
                            note_midi -= 12

                        abs_onset = m_start_t + onset_beat * sec_per_beat
                        gt_melody_notes.append((abs_onset, note_pc, note_midi))

                        n_start_sample = int(abs_onset * sr)
                        n_dur_sec = dur_beat * sec_per_beat * 0.92
                        n_len = int(n_dur_sec * sr)
                        n_end_sample = min(total_samples, n_start_sample + n_len)
                        actual_n_len = n_end_sample - n_start_sample

                        if actual_n_len > 0:
                            nt = np.linspace(0, actual_n_len / sr, actual_n_len)
                            base_f0 = _midi_to_hz(note_midi)
                            vibrato = 0.015 * np.sin(2 * np.pi * 5.5 * nt)
                            f0_t = base_f0 * (1.0 + vibrato)
                            phase = 2 * np.pi * np.cumsum(f0_t) / sr

                            vox = np.zeros(actual_n_len)
                            for h in range(1, 9):
                                harm_hz = base_f0 * h
                                f_weight = (1.0 / h) * (
                                    1.0
                                    + 1.5 * np.exp(-((harm_hz - 800) ** 2) / (2 * 250**2))
                                    + 1.2 * np.exp(-((harm_hz - 2500) ** 2) / (2 * 400**2))
                                )
                                vox += f_weight * np.sin(h * phase)

                            attack_s = min(int(0.03 * sr), actual_n_len // 4)
                            release_s = min(int(0.04 * sr), actual_n_len // 4)
                            env = np.ones(actual_n_len)
                            if attack_s > 0:
                                env[:attack_s] = np.linspace(0, 1, attack_s)
                            if release_s > 0:
                                env[-release_s:] = np.linspace(1, 0, release_s)

                            audio_voc[n_start_sample:n_end_sample] += 0.45 * vox * env

        mix = audio_acc + audio_voc
        peak = np.max(np.abs(mix))
        if peak > 1e-4:
            mix = mix / peak * 0.85

        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        sf.write(out_wav_path, mix, sr)
        return gt_downbeats, gt_melody_notes, gt_beat_chords

    def _synthesize_lead_sheet(lead, out_wav_path: str, sr: int = 22050) -> None:
        tempo_bpm = float(lead.tempo_bpm or 80.0)
        sec_per_beat = 60.0 / tempo_bpm
        base_tonic_pc = key_name_to_pc(lead.key) if lead.key else 0
        total_dur = sum(float(m.beats) * sec_per_beat for m in lead.measures)
        total_samples = int(math.ceil(total_dur * sr))

        audio_acc = np.zeros(total_samples, dtype=float)
        audio_voc = np.zeros(total_samples, dtype=float)

        cur_t = 0.0
        for m in lead.measures:
            m_start_t = cur_t
            m_beats = float(m.beats)
            m_dur = m_beats * sec_per_beat
            cur_t += m_dur
            num_m_beats = int(round(m_beats))

            m_beat_chords = [None] * num_m_beats
            cur_chord = m.chords[0] if m.chords else None
            for b in range(1, num_m_beats + 1):
                for c in m.chords:
                    if abs(c.beat - float(b)) < 0.2:
                        cur_chord = c
                m_beat_chords[b - 1] = cur_chord

            for b_idx in range(num_m_beats):
                beat_t = m_start_t + b_idx * sec_per_beat
                c_entry = m_beat_chords[b_idx]
                raw_c = c_entry.raw if c_entry else "1"
                try:
                    resolved = resolve_chord(raw_c, tonic_pc=base_tonic_pc, beat=float(b_idx + 1))
                    r_pc = resolved.root_pc
                    b_pc = resolved.bass_pc if resolved.bass_pc is not None else r_pc
                    pcs = resolved.pcs
                except Exception:
                    r_pc = base_tonic_pc
                    b_pc = base_tonic_pc
                    pcs = [(base_tonic_pc + i) % 12 for i in [0, 4, 7]]

                click_len = int(0.015 * sr)
                click_start = int(beat_t * sr)
                if click_start + click_len < total_samples:
                    click_t = np.linspace(0, 0.015, click_len)
                    audio_acc[click_start : click_start + click_len] += 0.4 * np.sin(2 * np.pi * 1200 * click_t) * np.exp(-click_t / 0.003)

                prev_raw = m_beat_chords[b_idx - 1].raw if (b_idx > 0 and m_beat_chords[b_idx - 1]) else None
                if b_idx in (0, 2) or (b_idx > 0 and raw_c != prev_raw):
                    bass_midi = 36 + b_pc
                    bass_hz = _midi_to_hz(bass_midi)
                    bass_len = int(sec_per_beat * 1.8 * sr)
                    b_start = int(beat_t * sr)
                    b_end = min(total_samples, b_start + bass_len)
                    actual_len = b_end - b_start
                    if actual_len > 0:
                        bt = np.linspace(0, actual_len / sr, actual_len)
                        env = np.exp(-bt / (sec_per_beat * 0.8))
                        bass_wave = 0.5 * np.sin(2 * np.pi * bass_hz * bt) + 0.25 * np.sin(4 * np.pi * bass_hz * bt)
                        audio_acc[b_start:b_end] += bass_wave * env

                chord_len = int(sec_per_beat * 1.5 * sr)
                c_start = int(beat_t * sr)
                c_end = min(total_samples, c_start + chord_len)
                actual_c_len = c_end - c_start
                if actual_c_len > 0:
                    ct = np.linspace(0, actual_c_len / sr, actual_c_len)
                    env = np.exp(-ct / (sec_per_beat * 0.7))
                    chord_wave = np.zeros(actual_c_len)
                    for pc in pcs:
                        note_midi = 60 + pc
                        hz = _midi_to_hz(note_midi)
                        chord_wave += 0.15 * np.sin(2 * np.pi * hz * ct) + 0.05 * np.sin(4 * np.pi * hz * ct)
                    audio_acc[c_start:c_end] += chord_wave * env

            for note in m.notes:
                if note.degree <= 0:
                    continue
                semitone = DEGREE_SEMITONES[note.degree - 1] + note.accidental + 12 * note.octave
                note_midi = 60 + base_tonic_pc + semitone
                abs_onset = m_start_t + note.onset * sec_per_beat
                n_start_sample = int(abs_onset * sr)
                n_dur_sec = note.duration * sec_per_beat * 0.92
                n_len = int(n_dur_sec * sr)
                n_end_sample = min(total_samples, n_start_sample + n_len)
                actual_n_len = n_end_sample - n_start_sample

                if actual_n_len > 0:
                    nt = np.linspace(0, actual_n_len / sr, actual_n_len)
                    base_f0 = _midi_to_hz(note_midi)
                    vibrato = 0.015 * np.sin(2 * np.pi * 5.5 * nt)
                    f0_t = base_f0 * (1.0 + vibrato)
                    phase = 2 * np.pi * np.cumsum(f0_t) / sr

                    vox = np.zeros(actual_n_len)
                    for h in range(1, 9):
                        harm_hz = base_f0 * h
                        f_weight = (1.0 / h) * (
                            1.0
                            + 1.5 * np.exp(-((harm_hz - 800) ** 2) / (2 * 250**2))
                            + 1.2 * np.exp(-((harm_hz - 2500) ** 2) / (2 * 400**2))
                        )
                        vox += f_weight * np.sin(h * phase)

                    attack_s = min(int(0.03 * sr), actual_n_len // 4)
                    release_s = min(int(0.04 * sr), actual_n_len // 4)
                    env = np.ones(actual_n_len)
                    if attack_s > 0:
                        env[:attack_s] = np.linspace(0, 1, attack_s)
                    if release_s > 0:
                        env[-release_s:] = np.linspace(1, 0, release_s)

                    audio_voc[n_start_sample:n_end_sample] += 0.45 * vox * env

        mix = audio_acc + audio_voc
        peak = np.max(np.abs(mix))
        if peak > 1e-4:
            mix = mix / peak * 0.85

        os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
        sf.write(out_wav_path, mix, sr)

    storage = get_storage()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("Starting Cloud Run synth_eval run_id=%s", run_id)

    # Load ground truth section definitions
    sections_truth = {}
    st_cands = [
        "/app/fixtures/sections_truth.json",
        "fixtures/sections_truth.json",
    ]
    for stp in st_cands:
        if os.path.isfile(stp):
            try:
                with open(stp, "r", encoding="utf-8") as f:
                    sections_truth = json.load(f)
                logger.info("Loaded sections_truth from local file: %s", stp)
                break
            except Exception:
                pass
    if not sections_truth:
        st_gcs = "tmp/smart_eval/fixtures/sections_truth.json"
        if storage.exists(st_gcs):
            sections_truth = storage.get_json(st_gcs)
            logger.info("Loaded sections_truth from GCS: %s", st_gcs)

    eval_results = []
    verse_templates = [
        "1. 2 3 2", "1 - - 0", "2. 3 2 1", "2 - - 0",
        "1. 2 3 2", "1 - - 0", "2. 1 7_ 1", "1 - - 0",
    ]
    chorus_templates = [
        "5. 6 1^ 2^", "3^ - - 2^", "1^. 2^ 3^ 5^", "5^ - - 0",
        "6^. 5^ 3^ 2^", "3^ - - 0", "2^. 3^ 2^ 1^", "1^ - - 0",
    ]
    bridge_templates = [
        "4. 5 6 5", "4 - - 0", "5. 6 5 4", "5 - - 0",
        "4. 5 6 1^", "7. 6 5 4", "5 - - -", "5 - - 0",
    ]

    charts = [
        ("tinghai", "听海"),
        ("diaole", "掉了"),
    ]

    with tempfile.TemporaryDirectory(prefix="smart_eval_") as tmp_dir:
        for slug, song_title in charts:
            logger.info("Evaluating chart: %s (%s)", slug, song_title)
            data = None
            cand_files = [
                f"/app/fixtures/groundtruth/{slug}.json",
                f"fixtures/groundtruth/{slug}.json",
                f"/app/fixtures/{slug}.json",
                f"fixtures/{slug}.json",
            ]
            for cf in cand_files:
                if os.path.isfile(cf):
                    try:
                        with open(cf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        logger.info("Loaded GT fixture from local file: %s", cf)
                        break
                    except Exception:
                        pass

            if not data:
                gcs_path = f"tmp/smart_eval/fixtures/{slug}.json"
                if storage.exists(gcs_path):
                    logger.info("Downloading GT fixture from GCS: %s", gcs_path)
                    data = storage.get_json(gcs_path)
                else:
                    gcs_path2 = f"tmp/smart_eval/groundtruth/{slug}.json"
                    if storage.exists(gcs_path2):
                        data = storage.get_json(gcs_path2)

            if not data:
                logger.error("Could not find GT fixture for %s anywhere!", slug)
                eval_results.append({
                    "slug": slug,
                    "error": "Ground truth fixture not found",
                })
                continue

            gt_key = data.get("header", {}).get("original_key", "C")
            base_tonic_pc = key_name_to_pc(gt_key)
            key_changes = data.get("key_changes", [])

            active_tonic_pc = base_tonic_pc
            measures_to_synth = []
            flat_m_idx = 0
            slug_sections = sections_truth.get(slug, [])
            total_m_in_data = sum(len(r.get("measures", [])) for r in data.get("rows", []))

            for r in data.get("rows", []):
                p_num = r.get("page", 1)
                r_num = r.get("row", 1)
                for m_idx_in_row, m in enumerate(r.get("measures", [])):
                    m_num = m_idx_in_row + 1
                    for kc in key_changes:
                        if kc.get("page") == p_num and kc.get("row") == r_num and kc.get("measure") == m_num:
                            active_tonic_pc = (active_tonic_pc + kc.get("semitones", 0)) % 12
                            logger.info("[%s] Key change at p%d r%d m%d: new tonic_pc=%d", slug, p_num, r_num, m_num, active_tonic_pc)

                    m_copy = dict(m)
                    m_copy["page"] = p_num
                    m_copy["row"] = r_num
                    m_copy["measure_in_row"] = m_num
                    m_copy["tonic_pc"] = active_tonic_pc

                    # Align vocal synthesis with section ground truth
                    sec_match = None
                    for s in slug_sections:
                        if s.get("start_measure", 0) <= flat_m_idx <= s.get("end_measure", -1):
                            sec_match = s
                            break
                    role = sec_match.get("role") if sec_match else None

                    if role in ("intro", "interlude", "outro"):
                        # Zero out vocal melody for instrumental regions
                        m_copy["melody"] = None
                        m_copy["role"] = role
                        m_copy["octave_shift"] = 0
                    elif role == "verse":
                        m_copy["role"] = role
                        m_copy["melody"] = verse_templates[flat_m_idx % len(verse_templates)]
                        m_copy["octave_shift"] = -12
                    elif role == "bridge":
                        m_copy["role"] = role
                        m_copy["melody"] = bridge_templates[flat_m_idx % len(bridge_templates)]
                        m_copy["octave_shift"] = -12
                    elif role == "chorus":
                        m_copy["role"] = role
                        m_copy["melody"] = chorus_templates[flat_m_idx % len(chorus_templates)]
                        m_copy["octave_shift"] = 0
                    else:
                        if flat_m_idx >= total_m_in_data - 1:
                            m_copy["melody"] = None
                            m_copy["role"] = "outro"
                            m_copy["octave_shift"] = 0
                        else:
                            m_copy["role"] = "chorus"
                            m_copy["melody"] = chorus_templates[flat_m_idx % len(chorus_templates)]
                            m_copy["octave_shift"] = 0

                    measures_to_synth.append(m_copy)
                    flat_m_idx += 1

            wav_path = os.path.join(tmp_dir, f"{slug}_synth.wav")
            song_work_dir = os.path.join(tmp_dir, f"{slug}_work")
            os.makedirs(song_work_dir, exist_ok=True)

            gt_downbeats, gt_melody_notes, gt_beat_chords = _synthesize_song(
                slug, gt_key, measures_to_synth, wav_path, tempo_bpm=80.0
            )

            # Transcribe
            t0_pipe = time.perf_counter()
            analysis = transcribe(wav_path, song_work_dir)
            lead = analysis_to_lead(analysis, title=slug)
            pipe_dur = round(time.perf_counter() - t0_pipe, 2)

            # Evaluate metrics
            pred_key = lead.key
            key_correct = (pred_key.upper() == gt_key.upper()) or (
                key_name_to_pc(pred_key) == base_tonic_pc if pred_key else False
            )

            # Downbeat alignment
            pred_downbeats = analysis.downbeats
            downbeat_hits = 0
            for gt_d in gt_downbeats:
                if any(abs(pd - gt_d) <= 0.20 for pd in pred_downbeats):
                    downbeat_hits += 1
            downbeat_acc = round(downbeat_hits / max(1, len(gt_downbeats)), 3)

            sec_per_b = 60.0 / 80.0
            total_beats = len(gt_beat_chords)

            # 1. Acoustic transcription chord accuracy (analysis.chords absolute pitch class)
            tx_root_hits = 0
            tx_full_hits = 0

            for b_idx, (beat_t, gt_raw, gt_root, gt_qual) in enumerate(gt_beat_chords):
                mid_t = beat_t + 0.5 * sec_per_b
                active_tx = None
                for c in analysis.chords:
                    if c.start <= mid_t < c.end:
                        active_tx = c
                        break

                if active_tx and active_tx.label not in ("N", "X"):
                    m_c = re.match(r"^([A-Ga-g][b#]?)", active_tx.label)
                    if m_c:
                        p_root = KEY_NAME_TO_PC.get(m_c.group(1).upper())
                        p_qual = active_tx.label.split(":")[1] if ":" in active_tx.label else "maj"
                        gt_is_min = "m" in gt_qual.lower() and "maj" not in gt_qual.lower()
                        p_is_min = "min" in p_qual.lower() or ("m" in p_qual.lower() and "maj" not in p_qual.lower())

                        if p_root == gt_root and gt_is_min == p_is_min:
                            tx_root_hits += 1
                        if p_root == gt_root and _norm_qual(p_qual) == _norm_qual(gt_qual):
                            tx_full_hits += 1

            tx_root_acc = round(tx_root_hits / max(1, total_beats), 3)
            tx_full_acc = round(tx_full_hits / max(1, total_beats), 3)

            # 2. LeadSheet chord accuracy (lead.measures resolved with pred_key)
            lead_root_hits = 0
            lead_full_hits = 0
            pred_measures = lead.measures
            pred_tonic_pc = key_name_to_pc(pred_key) if pred_key else base_tonic_pc

            for b_idx, (beat_t, gt_raw, gt_root, gt_qual) in enumerate(gt_beat_chords):
                m_idx = b_idx // 4
                in_m_beat = float((b_idx % 4) + 1)
                pred_chord_raw = None
                if m_idx < len(pred_measures):
                    m = pred_measures[m_idx]
                    for c in m.chords:
                        if c.beat <= in_m_beat:
                            pred_chord_raw = c.raw

                if pred_chord_raw:
                    try:
                        pred_resolved = resolve_chord(pred_chord_raw, tonic_pc=pred_tonic_pc, beat=in_m_beat)
                        p_root = pred_resolved.root_pc
                        p_qual = pred_resolved.quality
                        gt_is_min = "m" in gt_qual.lower() and "maj" not in gt_qual.lower()
                        p_is_min = "m" in p_qual.lower() and "maj" not in p_qual.lower()
                        if p_root == gt_root and gt_is_min == p_is_min:
                            lead_root_hits += 1
                        if p_root == gt_root and _norm_qual(p_qual) == _norm_qual(gt_qual):
                            lead_full_hits += 1
                    except Exception:
                        pass

            lead_root_acc = round(lead_root_hits / max(1, total_beats), 3)
            lead_full_acc = round(lead_full_hits / max(1, total_beats), 3)

            # 3. Melody note accuracy (analysis.notes vs gt_melody_notes)
            melody_hits = 0
            pred_notes = [n for n in analysis.notes if n.midi > 0 and n.confidence >= 0.15]
            for gt_onset, gt_pc, _ in gt_melody_notes:
                matched = False
                for pn in pred_notes:
                    pn_pc = int(round(pn.midi)) % 12
                    if abs(pn.start - gt_onset) <= 0.25 * sec_per_b + 0.05 and pn_pc == gt_pc:
                        matched = True
                        break
                if matched:
                    melody_hits += 1
            melody_acc = round(melody_hits / max(1, len(gt_melody_notes)), 3)

            # 4. First 16 bars side-by-side comparison
            comparison_16 = []
            sec_per_bar = 4.0 * sec_per_b
            print(f"\n--- First 16 Bars Comparison: {slug} ---")
            for bar_idx in range(min(16, len(measures_to_synth))):
                bar_t0 = bar_idx * sec_per_bar
                bar_t1 = (bar_idx + 1) * sec_per_bar
                gt_cs = [c.get("raw", "") for c in measures_to_synth[bar_idx].get("chords", [])]
                gt_str = ", ".join(gt_cs) if gt_cs else "None"

                tx_cs = [
                    c.label for c in analysis.chords
                    if max(c.start, bar_t0) < min(c.end, bar_t1)
                ]
                tx_str = ", ".join(tx_cs) if tx_cs else "None"

                if bar_idx < len(pred_measures):
                    lm_cs = [f"{c.raw}(b{c.beat})" for c in pred_measures[bar_idx].chords]
                    lead_str = ", ".join(lm_cs) if lm_cs else "None"
                else:
                    lead_str = "None"

                row_entry = {
                    "bar": bar_idx + 1,
                    "gt": gt_str,
                    "analysis": tx_str,
                    "lead": lead_str,
                }
                comparison_16.append(row_entry)
                print(f"Bar {bar_idx + 1:02d} | GT: {gt_str:12s} | Analysis: {tx_str:18s} | LeadSheet: {lead_str}")

            # 4. Section boundary scoring against sections_truth.json
            gt_sec_list = sections_truth.get(slug, [])
            gt_role_regions = []
            for entry in gt_sec_list:
                r = entry.get("role", "unknown")
                sm = entry.get("start_measure", 0)
                em = entry.get("end_measure", 0)
                if gt_role_regions and gt_role_regions[-1]["role"] == r and gt_role_regions[-1]["end_measure"] + 1 >= sm:
                    gt_role_regions[-1]["end_measure"] = em
                else:
                    gt_role_regions.append({"role": r, "start_measure": sm, "end_measure": em})

            gt_boundaries = [r["start_measure"] for r in gt_role_regions if r["start_measure"] > 0]

            pred_sections_summary = []
            pred_boundaries = []
            for s in analysis.sections:
                m_start = int(round(s.start / max(0.1, sec_per_bar)))
                m_end = int(round(s.end / max(0.1, sec_per_bar)))
                pred_sections_summary.append({
                    "label": s.label,
                    "start_measure": m_start,
                    "end_measure": m_end,
                    "start_sec": s.start,
                    "end_sec": s.end,
                })
                if m_start > 0:
                    pred_boundaries.append(m_start)

            boundary_hits = sum(1 for gt_b in gt_boundaries if any(abs(pb - gt_b) <= 1 for pb in pred_boundaries))
            section_boundary_hit_rate = round(boundary_hits / max(1, len(gt_boundaries)), 3)

            # Upload to GCS
            analysis_gcs_path = f"tmp/smart_eval/{run_id}/{slug}_analysis.json"
            lead_gcs_path = f"tmp/smart_eval/{run_id}/{slug}_lead.json"
            storage.put_json(analysis_gcs_path, analysis.model_dump())
            storage.put_json(lead_gcs_path, lead.model_dump())
            logger.info("Saved %s and %s to GCS", analysis_gcs_path, lead_gcs_path)

            score_record = {
                "slug": slug,
                "gt_key": gt_key,
                "pred_key": pred_key,
                "key_correct": key_correct,
                "downbeat_acc": downbeat_acc,
                "transcription_chord_root_majmin_acc": tx_root_acc,
                "transcription_chord_full_acc": tx_full_acc,
                "leadsheet_chord_root_majmin_acc": lead_root_acc,
                "leadsheet_chord_full_acc": lead_full_acc,
                "chord_root_majmin_acc": tx_root_acc,
                "chord_full_acc": tx_full_acc,
                "melody_acc": melody_acc,
                "lyrics_chars": len(analysis.lyrics),
                "section_boundary_hit_rate": section_boundary_hit_rate,
                "gt_boundaries": gt_boundaries,
                "pred_boundaries": pred_boundaries,
                "predicted_sections": pred_sections_summary,
                "stage_timings": analysis.stage_timings,
                "pipe_duration_sec": pipe_dur,
                "first_16_bars_comparison": comparison_16,
            }
            eval_results.append(score_record)

    # 5. Synthesize public domain songs (ode_to_joy, jasmine_flower) and upload to GCS
    public_wavs = {}
    pub_songs = [("ode_to_joy", "欢乐颂"), ("jasmine_flower", "茉莉花")]
    from app.smart.models import LeadSheet

    for pub_slug, pub_title in pub_songs:
        pub_lead = None
        cand_lead_paths = [
            f"/app/fixtures/lead/{pub_slug}.json",
            f"fixtures/lead/{pub_slug}.json",
        ]
        for clp in cand_lead_paths:
            if os.path.isfile(clp):
                try:
                    with open(clp, "r", encoding="utf-8") as f:
                        pub_lead = LeadSheet.model_validate_json(f.read())
                    break
                except Exception:
                    pass
        if not pub_lead:
            gcs_lead_path = f"tmp/smart_eval/fixtures/lead/{pub_slug}.json"
            if storage.exists(gcs_lead_path):
                pub_lead = LeadSheet.model_validate(storage.get_json(gcs_lead_path))

        if pub_lead:
            pub_wav_local = os.path.join(tmp_dir, f"{pub_slug}.wav")
            _synthesize_lead_sheet(pub_lead, pub_wav_local)
            with open(pub_wav_local, "rb") as f:
                wav_bytes = f.read()
            gcs_pub_dest = f"tmp/smart_eval/public/{pub_slug}.wav"
            storage.put_bytes(gcs_pub_dest, wav_bytes)
            bucket_str = os.environ.get("BUCKET", "cellular-cider-495602-r9-sheet-converter")
            full_uri = f"gs://{bucket_str}/{gcs_pub_dest}"
            public_wavs[pub_slug] = full_uri
            logger.info("Synthesized and uploaded public song %s to %s (%d bytes)", pub_slug, full_uri, len(wav_bytes))
        else:
            logger.warning("Could not find fixture for public song %s", pub_slug)

    report_data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "public_wavs": public_wavs,
        "results": eval_results,
    }

    report_gcs_path = f"tmp/smart_eval/{run_id}/report.json"
    latest_gcs_path = "tmp/smart_eval/latest_report.json"
    storage.put_json(report_gcs_path, report_data)
    storage.put_json(latest_gcs_path, report_data)
    logger.info("Saved final report to %s and %s", report_gcs_path, latest_gcs_path)

    print("\n=======================================================")
    print(f"CLOUD RUN SYNTH_EVAL REPORT: {run_id}")
    print("=======================================================")
    print(json.dumps(report_data, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Smart Sheet Audio Transcription Job")
    parser.add_argument("--local", type=str, default=None, help="Path to local audio file for direct run")
    parser.add_argument("--out", type=str, default="out/smart_audio", help="Output directory for local run")
    parser.add_argument("--title", type=str, default="", help="Song title")
    parser.add_argument("--artist", type=str, default="", help="Artist name")
    args = parser.parse_args()

    mode = os.environ.get("SMART_JOB_MODE")
    if mode == "synth_eval":
        run_cloud_synth_eval()
    elif args.local:
        run_local(args.local, args.out, title=args.title, artist=args.artist)
    else:
        run_cloud_run_job()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()


