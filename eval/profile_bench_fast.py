#!/usr/bin/env python3
"""Targeted profiling script for Audio Separation and Melody Extraction.
Flushes output immediately after each measurement.
"""
import math
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch

try:
    import torchcrepe
except ImportError:
    torchcrepe = None

def p(msg):
    print(msg, flush=True)

def inspect_onnx_and_separator():
    p("\n" + "="*60)
    p("1. ONNX RUNTIME & CPU INSPECTION")
    p("="*60)
    import onnxruntime as ort
    p(f"CPU count: {os.cpu_count()}")
    try:
        p(f"Sched affinity: {sorted(list(os.sched_getaffinity(0)))}")
    except Exception as e:
        p(f"Sched affinity error: {e}")
    p(f"Torch threads: {torch.get_num_threads()}")
    p(f"ONNX version: {ort.__version__}")
    p(f"ONNX providers: {ort.get_available_providers()}")

    from audio_separator.separator import Separator
    sep = Separator()
    p("\nSeparator MDX attributes:")
    for a in ["mdx_params", "vr_params", "output_single_stem", "normalization_threshold"]:
        p(f"  {a} = {getattr(sep, a, 'N/A')}")

def benchmark_mdx_params():
    p("\n" + "="*60)
    p("2. AUDIO_SEPARATOR MDX PARAMETER BENCHMARK (30s Audio)")
    p("="*60)
    from audio_separator.separator import Separator
    sr = 44100
    dur = 30.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    synth = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    stereo = np.column_stack([synth, synth])

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "test.wav")
        sf.write(wav_path, stereo, sr)

        test_cases = [
            ("overlap=0.25 (default)", {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1}),
            ("overlap=0.15",          {"hop_length": 1024, "segment_size": 256, "overlap": 0.15, "batch_size": 1}),
            ("overlap=0.10",          {"hop_length": 1024, "segment_size": 256, "overlap": 0.10, "batch_size": 1}),
            ("overlap=0.05",          {"hop_length": 1024, "segment_size": 256, "overlap": 0.05, "batch_size": 1}),
            ("overlap=0.15, batch=2", {"hop_length": 1024, "segment_size": 256, "overlap": 0.15, "batch_size": 2}),
            ("overlap=0.15, batch=4", {"hop_length": 1024, "segment_size": 256, "overlap": 0.15, "batch_size": 4}),
        ]

        model_dir = "/app/models/audio_separator"
        for label, mdx_p in test_cases:
            out_d = os.path.join(tmp_dir, f"out_{label.replace(' ', '_').replace('=', '_')}")
            os.makedirs(out_d, exist_ok=True)
            sep = Separator(model_file_dir=model_dir, output_dir=out_d, output_format="WAV", mdx_params=mdx_p)
            sep.load_model("UVR-MDX-NET-Voc_FT.onnx")

            t0 = time.perf_counter()
            outputs = sep.separate(wav_path)
            elapsed = time.perf_counter() - t0
            p(f"[{label:24s}] 30s -> {elapsed:5.2f}s (Speed: {dur/elapsed:4.2f}x RT, 4-min song: {elapsed*8:5.1f}s)")

def benchmark_crepe(vocals_flac_path: str):
    p("\n" + "="*60)
    p("3. MELODY EXTRACTION BENCHMARK (RWC_P056: ~5.3 min)")
    p("="*60)
    if not os.path.isfile(vocals_flac_path):
        p(f"File not found: {vocals_flac_path}")
        return

    import scipy.signal
    from scipy.ndimage import binary_closing, binary_dilation

    v_audio, v_sr = sf.read(vocals_flac_path)
    if v_audio.ndim > 1:
        v_audio = np.mean(v_audio, axis=1)
    dur = len(v_audio) / v_sr
    p(f"Audio: {vocals_flac_path} ({dur:.1f}s = {dur/60.1:.2f} min)")

    target_sr = 16000
    if v_sr != target_sr:
        g = math.gcd(int(target_sr), int(v_sr))
        voc_16k = scipy.signal.resample_poly(v_audio, target_sr // g, v_sr // g)
    else:
        voc_16k = v_audio
    voc_16k = np.ascontiguousarray(voc_16k, dtype=np.float32)

    # Active vocal gating
    t0_gate = time.perf_counter()
    hop_length = 160
    num_frames = int(np.ceil(len(voc_16k) / float(hop_length)))
    target_len = num_frames * hop_length
    padded = np.pad(voc_16k, (0, max(0, target_len - len(voc_16k))))[:target_len]
    v_frames = padded.reshape(num_frames, hop_length)
    v_rms = np.sqrt(np.mean(v_frames ** 2, axis=1))

    rms_p25 = float(np.percentile(v_rms, 25))
    rms_thresh = max(0.008, rms_p25 * 1.5)
    is_act = v_rms >= rms_thresh
    act_dilated = binary_dilation(is_act, iterations=20)
    act_bridged = binary_closing(act_dilated, structure=np.ones(40, dtype=bool))

    segments = []
    in_seg = False
    seg_start = 0
    for idx, val in enumerate(act_bridged):
        if val and not in_seg:
            in_seg = True
            seg_start = idx
        elif not val and in_seg:
            in_seg = False
            segments.append((seg_start, idx))
    if in_seg:
        segments.append((seg_start, len(act_bridged)))
    t_gate = time.perf_counter() - t0_gate
    act_frames = sum(e - s for s, e in segments)
    p(f"Active gating: {t_gate*1000:.1f}ms | Active frames: {act_frames}/{num_frames} ({act_frames/num_frames*100:.1f}%), {len(segments)} segments")

    # Benchmarks to run
    benchmarks = [
        # (name, model, hop, batch_size, threads)
        ("full  | hop=10ms | bs=2048 | th=8 (CURRENT)", "full", 160, 2048, 8),
        ("full  | hop=10ms | bs=4096 | th=8",           "full", 160, 4096, 8),
        ("full  | hop=20ms | bs=2048 | th=8",           "full", 320, 2048, 8),
        ("tiny  | hop=10ms | bs=2048 | th=8",           "tiny", 160, 2048, 8),
        ("tiny  | hop=10ms | bs=4096 | th=8",           "tiny", 160, 4096, 8),
        ("tiny  | hop=20ms | bs=2048 | th=8",           "tiny", 320, 2048, 8),
    ]

    ref_pitch_10ms = None

    for label, model_name, hop, bs, th in benchmarks:
        torch.set_num_threads(th)
        t0 = time.perf_counter()
        pitches = []
        pers = []
        for s_idx, e_idx in segments:
            sample_start = s_idx * 160
            sample_end = min(len(voc_16k), e_idx * 160)
            seg_audio = voc_16k[sample_start:sample_end]
            if len(seg_audio) < hop:
                continue
            seg_t = torch.from_numpy(seg_audio).float().unsqueeze(0)
            with torch.no_grad():
                sp, sc = torchcrepe.predict(
                    seg_t,
                    target_sr,
                    hop_length=hop,
                    model=model_name,
                    device="cpu",
                    return_periodicity=True,
                    batch_size=bs,
                    decoder=torchcrepe.decode.weighted_argmax,
                )
            pitches.append(sp.squeeze(0).numpy())
            pers.append(sc.squeeze(0).numpy())

        elapsed = time.perf_counter() - t0
        flat_p = np.concatenate(pitches) if pitches else np.array([])
        flat_per = np.concatenate(pers) if pers else np.array([])
        mean_per = float(np.mean(flat_per)) if len(flat_per) > 0 else 0.0

        agree_str = ""
        if hop == 160:
            if ref_pitch_10ms is None:
                ref_pitch_10ms = flat_p
                agree_str = "Ref (100%)"
            else:
                min_l = min(len(ref_pitch_10ms), len(flat_p))
                r_sub = ref_pitch_10ms[:min_l]
                f_sub = flat_p[:min_l]
                v = (r_sub > 65.0) & (f_sub > 65.0)
                if np.sum(v) > 0:
                    cents = 1200.0 * np.abs(np.log2(f_sub[v] / r_sub[v]))
                    agree_pct = float(np.mean(cents <= 50.0) * 100.0)
                    agree_str = f"vs Full: {agree_pct:5.1f}% within 50c"
        else:
            agree_str = "(20ms hop)"

        p(f"[{label:44s}] Wall: {elapsed:5.1f}s | Speed: {dur/elapsed:4.1f}x RT | Mean Per: {mean_per:.3f} | {agree_str}")

if __name__ == "__main__":
    inspect_onnx_and_separator()
    benchmark_mdx_params()
    voc_cand = sys.argv[1] if len(sys.argv) > 1 else "/workspace/diag_stems/RWC_P056_vocals.flac"
    benchmark_crepe(voc_cand)
