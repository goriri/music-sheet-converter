#!/usr/bin/env python3
"""Profiling and benchmarking suite for audio pipeline speedup.
Measures Separation (MDX parameters) and Melody (Crepe full vs tiny, hops, batch sizes, threads, pyin).
"""
import inspect
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

def section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def inspect_environment_and_separator():
    section("1. ENVIRONMENT & CPU DIAGNOSTICS")
    print(f"Python: {sys.version}")
    print(f"os.cpu_count(): {os.cpu_count()}")
    try:
        aff = os.sched_getaffinity(0)
        print(f"os.sched_getaffinity(0): {len(aff)} cores -> {sorted(list(aff))}")
    except Exception as e:
        print(f"os.sched_getaffinity: not available ({e})")

    # Check cgroups if on Linux
    for cgroup_path in ["/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"]:
        if os.path.exists(cgroup_path):
            try:
                with open(cgroup_path) as f:
                    print(f"cgroups ({cgroup_path}): {f.read().strip()}")
            except Exception as e:
                print(f"cgroups read error: {e}")

    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"torch.__version__: {torch.__version__}")

    section("2. AUDIO_SEPARATOR SEPARATOR INSPECTION")
    try:
        from audio_separator.separator import Separator
        print("Separator class __init__ signature:")
        sig = inspect.signature(Separator.__init__)
        for name, param in sig.parameters.items():
            print(f"  {name}: default={param.default}")

        print("\nSeparator separate method signature:")
        sig_sep = inspect.signature(Separator.separate)
        for name, param in sig_sep.parameters.items():
            print(f"  {name}: default={param.default}")

        # Check default MDX attributes on an instance
        sep = Separator()
        mdx_attrs = [k for k in dir(sep) if "mdx" in k.lower() or "batch" in k.lower() or "overlap" in k.lower() or "thread" in k.lower() or "segment" in k.lower()]
        print("\nInstance MDX-related attributes:")
        for attr in sorted(mdx_attrs):
            try:
                val = getattr(sep, attr)
                if not callable(val):
                    print(f"  {attr} = {val}")
            except Exception:
                pass

        import onnxruntime as ort
        print(f"\nONNX Runtime version: {ort.__version__}")
        print(f"Available ONNX providers: {ort.get_available_providers()}")
    except Exception as e:
        print(f"Separator inspection error: {e}")

def benchmark_melody(vocals_flac_path: str):
    section("3. MELODY EXTRACTION BENCHMARK")
    if not os.path.isfile(vocals_flac_path):
        print(f"Vocal file not found: {vocals_flac_path}")
        return

    import scipy.signal
    from scipy.ndimage import binary_closing, binary_dilation

    v_audio, v_sr = sf.read(vocals_flac_path)
    if v_audio.ndim > 1:
        v_audio = np.mean(v_audio, axis=1)
    duration = len(v_audio) / v_sr
    print(f"Input audio: {vocals_flac_path}")
    print(f"Duration: {duration:.2f}s ({duration/60.0:.2f} min), sr={v_sr}, samples={len(v_audio)}")

    target_sr = 16000
    if v_sr != target_sr:
        g = math.gcd(int(target_sr), int(v_sr))
        voc_16k = scipy.signal.resample_poly(v_audio, target_sr // g, v_sr // g)
    else:
        voc_16k = v_audio
    voc_16k = np.ascontiguousarray(voc_16k, dtype=np.float32)

    # 1. Measure active gating time
    t0_gate = time.perf_counter()
    hop_length = 160
    hop_dur = 0.01
    num_frames = int(np.ceil(len(voc_16k) / float(hop_length)))
    target_len = num_frames * hop_length
    padded_voc = np.pad(voc_16k, (0, max(0, target_len - len(voc_16k))))[:target_len]
    vocal_frames = padded_voc.reshape(num_frames, hop_length)
    vocal_rms = np.sqrt(np.mean(vocal_frames ** 2, axis=1))

    rms_p25 = float(np.percentile(vocal_rms, 25))
    rms_active_thresh = max(0.008, rms_p25 * 1.5)
    is_active = vocal_rms >= rms_active_thresh
    active_dilated = binary_dilation(is_active, iterations=20)
    active_bridged = binary_closing(active_dilated, structure=np.ones(40, dtype=bool))

    segments = []
    in_seg = False
    seg_start = 0
    for idx, val in enumerate(active_bridged):
        if val and not in_seg:
            in_seg = True
            seg_start = idx
        elif not val and in_seg:
            in_seg = False
            segments.append((seg_start, idx))
    if in_seg:
        segments.append((seg_start, len(active_bridged)))
    t_gate = time.perf_counter() - t0_gate

    total_active_frames = sum(e - s for s, e in segments)
    print(f"\nActive gating wall time: {t_gate*1000:.2f} ms")
    print(f"Total frames: {num_frames} ({num_frames*0.01:.2f}s)")
    print(f"Active frames: {total_active_frames} ({total_active_frames/num_frames*100:.1f}%), segments: {len(segments)}")

    # 2. Check pyin latency on a 5s segment
    print("\n--- Testing librosa.pyin benchmark ---")
    try:
        import librosa
        test_seg_len = min(len(voc_16k), target_sr * 5)
        test_audio_5s = voc_16k[:test_seg_len]
        t0_pyin = time.perf_counter()
        _ = librosa.pyin(test_audio_5s, fmin=65.0, fmax=1100.0, sr=target_sr, hop_length=160)
        t_pyin_5s = time.perf_counter() - t0_pyin
        print(f"librosa.pyin on 5s segment: {t_pyin_5s:.2f}s (Speed: {5.0 / t_pyin_5s:.2f}x real-time)")
        print(f"Extrapolated pyin on full 4-min song: {t_pyin_5s / 5.0 * 240:.1f}s (DANGEROUSLY SLOW!)")
    except Exception as e:
        print(f"pyin test error: {e}")

    # 3. Test torchcrepe configurations across:
    # (model, batch_size, hop_length, num_threads)
    configs = [
        # Baseline
        ("full", 2048, 160, 8),
        ("full", 4096, 160, 8),
        ("full", 2048, 160, 4),
        # 20ms hop (320 samples at 16k)
        ("full", 2048, 320, 8),
        ("full", 4096, 320, 8),
        # tiny model
        ("tiny", 2048, 160, 8),
        ("tiny", 4096, 160, 8),
        ("tiny", 2048, 320, 8),
        ("tiny", 4096, 320, 8),
    ]

    print("\n--- Running Crepe Parameter Sweep on Segments ---")
    results = {}
    base_pitch = None

    for model_name, bs, hop, n_threads in configs:
        tag = f"model={model_name:4s} | bs={bs:4d} | hop={hop:3d} ({hop/16:.0f}ms) | threads={n_threads:2d}"
        torch.set_num_threads(n_threads)

        t0_cfg = time.perf_counter()
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

        dur = time.perf_counter() - t0_cfg
        flat_p = np.concatenate(pitches) if pitches else np.array([])
        flat_per = np.concatenate(pers) if pers else np.array([])

        # Compare with baseline (full, bs=2048, hop=160, threads=8)
        if base_pitch is None and model_name == "full" and hop == 160:
            base_pitch = flat_p
            agreement_pct = 100.0
            rpa_self_cents = 0.0
        elif base_pitch is not None and hop == 160:
            min_len = min(len(base_pitch), len(flat_p))
            b_sub = base_pitch[:min_len]
            f_sub = flat_p[:min_len]
            voiced = (b_sub > 65.0) & (f_sub > 65.0)
            if np.sum(voiced) > 0:
                cents = 1200.0 * np.abs(np.log2(f_sub[voiced] / b_sub[voiced]))
                agreement_pct = float(np.mean(cents <= 50.0) * 100.0)
            else:
                agreement_pct = 0.0
        else:
            agreement_pct = -1.0 # different hop

        mean_per = float(np.mean(flat_per)) if len(flat_per) > 0 else 0.0
        results[tag] = {
            "wall_sec": dur,
            "mean_periodicity": mean_per,
            "agreement_vs_full_50c": agreement_pct,
        }
        agree_str = f"Agree vs Full: {agreement_pct:5.1f}%" if agreement_pct >= 0 else "Hop 20ms"
        print(f"[{tag}] Wall: {dur:6.2f}s | Speed: {duration/dur:5.2f}x RT | Per: {mean_per:.3f} | {agree_str}")

def benchmark_separation():
    section("4. AUDIO_SEPARATOR MDX PARAMETER BENCHMARK")
    from audio_separator.separator import Separator

    # Generate a synthetic 30-second stereo audio file for separation speed profiling
    sr = 44100
    dur_sec = 30.0
    t = np.linspace(0, dur_sec, int(sr * dur_sec), endpoint=False)
    # 440 Hz vocal sine + 120 Hz bass + white noise
    synth_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.random.randn(len(t))
    synth_stereo = np.column_stack([synth_audio, synth_audio]).astype(np.float32)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_wav = os.path.join(tmp_dir, "test_30s.wav")
        sf.write(test_wav, synth_stereo, sr)
        print(f"Created 30s test wav: {test_wav}")

        # Check different overlap and segment_size settings
        # Default in Separator: mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1}
        test_settings = [
            ("Default (overlap=0.25, batch=1)", {"mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1}}),
            ("Overlap 0.15 (batch=1)",          {"mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.15, "batch_size": 1}}),
            ("Overlap 0.10 (batch=1)",          {"mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.10, "batch_size": 1}}),
            ("Overlap 0.05 (batch=1)",          {"mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.05, "batch_size": 1}}),
            ("Overlap 0.15 (batch=2)",          {"mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.15, "batch_size": 2}}),
            ("Overlap 0.15 (batch=4)",          {"mdx_params": {"hop_length": 1024, "segment_size": 256, "overlap": 0.15, "batch_size": 4}}),
        ]

        model_dir = "/app/models/audio_separator"
        for label, kwargs in test_settings:
            out_d = os.path.join(tmp_dir, f"out_{int(time.time()*1000)%10000}")
            os.makedirs(out_d, exist_ok=True)
            sep = Separator(model_file_dir=model_dir, output_dir=out_d, output_format="WAV", **kwargs)
            sep.load_model("UVR-MDX-NET-Voc_FT.onnx")

            t0 = time.perf_counter()
            outputs = sep.separate(test_wav)
            t_sep = time.perf_counter() - t0
            print(f"[{label:32s}] 30s audio -> {t_sep:5.2f}s | Speed: {dur_sec/t_sep:4.2f}x RT | Extrapolated 4-min song: {t_sep*8:5.1f}s")

if __name__ == "__main__":
    inspect_environment_and_separator()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        benchmark_melody(sys.argv[1])
    else:
        # Check standard paths for RWC_P056_vocals.flac
        for cand in [
            "/workspace/diag_stems/RWC_P056_vocals.flac",
            "scratch/real_eval/diagnosis_stems/RWC_P056_vocals.flac",
            "/Users/jush/.gemini/jetski/brain/982974d3-46cb-4d49-ac77-56cc9b0c43e2/scratch/real_eval/diagnosis_stems/RWC_P056_vocals.flac",
        ]:
            if os.path.exists(cand):
                benchmark_melody(cand)
                break
    benchmark_separation()
