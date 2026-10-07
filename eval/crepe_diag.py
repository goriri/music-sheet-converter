#!/usr/bin/env python3
"""In-depth Crepe failure reproduction and diagnosis script.
Runs across abjones_3_09, yifen_3_02, RWC_P056, and bobon_3_08 (control).
"""
import os
import sys
import tempfile
import urllib.request
import torch
import numpy as np
import soundfile as sf
import scipy.signal

try:
    import torchcrepe
except ImportError:
    print("torchcrepe not installed locally; will run in container")

def inspect_crepe_internals():
    print("=== Torchcrepe Internals ===")
    import inspect
    print("--- predict source ---")
    print(inspect.getsource(torchcrepe.predict))
    print("--- decode.viterbi source ---")
    print(inspect.getsource(torchcrepe.decode.viterbi))
    print("--- decode.weighted_argmax source ---")
    print(inspect.getsource(torchcrepe.decode.weighted_argmax))

def run_diagnostics_on_audio(name: str, audio_path: str):
    print(f"\n=======================================================")
    print(f"DIAGNOSING: {name} ({audio_path})")
    print(f"=======================================================")

    info = sf.info(audio_path)
    print(f"Input info: sr={info.samplerate}, channels={info.channels}, duration={info.duration:.2f}s, format={info.format}")
    raw_audio, orig_sr = sf.read(audio_path)
    if raw_audio.ndim > 1:
        raw_audio = np.mean(raw_audio, axis=1)
    raw_audio = np.ascontiguousarray(raw_audio, dtype=np.float32)
    print(f"Audio array: shape={raw_audio.shape}, min={raw_audio.min():.4f}, max={raw_audio.max():.4f}, rms={np.sqrt(np.mean(raw_audio**2)):.4f}")

    # Check for NaN / Inf
    if np.isnan(raw_audio).any() or np.isinf(raw_audio).any():
        print("ERROR: input audio has NaN or Inf!")

    # 1. Test CURRENT pipeline method (pass orig_sr, hop_length=orig_sr*0.01)
    hop_orig = int(orig_sr * 0.01)
    t_orig = torch.from_numpy(raw_audio).unsqueeze(0)
    print(f"\n--- [Test 1] Current pipeline method (sr={orig_sr}, hop={hop_orig}, model=full, decoder=viterbi) ---")
    with torch.no_grad():
        p1, per1 = torchcrepe.predict(
            t_orig,
            orig_sr,
            hop_length=hop_orig,
            model="full",
            device="cpu",
            return_periodicity=True,
            batch_size=2048,
            decoder=torchcrepe.decode.viterbi,
        )
    p1 = p1.squeeze(0).numpy()
    per1 = per1.squeeze(0).numpy()
    print(f"Result 1: frames={len(p1)}, pitch_hz min={p1.min():.1f}, max={p1.max():.1f}, mean={p1.mean():.1f}, median={np.median(p1):.1f}")
    print(f"Periodicity 1: min={per1.min():.4f}, max={per1.max():.4f}, mean={per1.mean():.4f}, median={np.median(per1):.4f}")
    print(f"Frames > 1950 Hz: {np.sum(p1 > 1950)} / {len(p1)} ({np.mean(p1 > 1950)*100:.1f}%)")

    # 2. Test decoder=weighted_argmax on same input
    print(f"\n--- [Test 2] Decoder = weighted_argmax (sr={orig_sr}, hop={hop_orig}) ---")
    with torch.no_grad():
        p2, per2 = torchcrepe.predict(
            t_orig,
            orig_sr,
            hop_length=hop_orig,
            model="full",
            device="cpu",
            return_periodicity=True,
            batch_size=2048,
            decoder=torchcrepe.decode.weighted_argmax,
        )
    p2 = p2.squeeze(0).numpy()
    per2 = per2.squeeze(0).numpy()
    print(f"Result 2: frames={len(p2)}, pitch_hz min={p2.min():.1f}, max={p2.max():.1f}, mean={p2.mean():.1f}, median={np.median(p2):.1f}")
    print(f"Periodicity 2: min={per2.min():.4f}, max={per2.max():.4f}, mean={per2.mean():.4f}, median={np.median(per2):.4f}")
    print(f"Frames > 1950 Hz: {np.sum(p2 > 1950)} / {len(p2)} ({np.mean(p2 > 1950)*100:.1f}%)")

    # 3. Test decoder=argmax on same input
    print(f"\n--- [Test 3] Decoder = argmax (sr={orig_sr}, hop={hop_orig}) ---")
    with torch.no_grad():
        p3, per3 = torchcrepe.predict(
            t_orig,
            orig_sr,
            hop_length=hop_orig,
            model="full",
            device="cpu",
            return_periodicity=True,
            batch_size=2048,
            decoder=torchcrepe.decode.argmax,
        )
    p3 = p3.squeeze(0).numpy()
    per3 = per3.squeeze(0).numpy()
    print(f"Result 3: frames={len(p3)}, pitch_hz min={p3.min():.1f}, max={p3.max():.1f}, mean={p3.mean():.1f}, median={np.median(p3):.1f}")
    print(f"Periodicity 3: min={per3.min():.4f}, max={per3.max():.4f}, mean={per3.mean():.4f}, median={np.median(per3):.4f}")
    print(f"Frames > 1950 Hz: {np.sum(p3 > 1950)} / {len(p3)} ({np.mean(p3 > 1950)*100:.1f}%)")

    # 4. Test pre-resampling to 16000 Hz ourselves (scipy.signal.resample_poly)
    print(f"\n--- [Test 4] Pre-resampling to 16000 Hz with resample_poly ---")
    gcd = np.gcd(orig_sr, 16000)
    up = 16000 // gcd
    down = orig_sr // gcd
    audio_16k = scipy.signal.resample_poly(raw_audio, up, down).astype(np.float32)
    audio_16k = np.ascontiguousarray(audio_16k)
    t_16k = torch.from_numpy(audio_16k).unsqueeze(0)
    hop_16k = 160 # 10ms at 16kHz
    print(f"Resampled 16k audio: len={len(audio_16k)}, rms={np.sqrt(np.mean(audio_16k**2)):.4f}")

    # 4a: with viterbi
    with torch.no_grad():
        p4a, per4a = torchcrepe.predict(
            t_16k,
            16000,
            hop_length=hop_16k,
            model="full",
            device="cpu",
            return_periodicity=True,
            batch_size=2048,
            decoder=torchcrepe.decode.viterbi,
        )
    p4a = p4a.squeeze(0).numpy()
    per4a = per4a.squeeze(0).numpy()
    print(f"Result 4a (16k + viterbi): pitch_hz min={p4a.min():.1f}, max={p4a.max():.1f}, mean={p4a.mean():.1f}, median={np.median(p4a):.1f}")
    print(f"Periodicity 4a: mean={per4a.mean():.4f}, median={np.median(per4a):.4f}")
    print(f"Frames > 1950 Hz: {np.sum(p4a > 1950)} / {len(p4a)} ({np.mean(p4a > 1950)*100:.1f}%)")

    # 4b: with weighted_argmax
    with torch.no_grad():
        p4b, per4b = torchcrepe.predict(
            t_16k,
            16000,
            hop_length=hop_16k,
            model="full",
            device="cpu",
            return_periodicity=True,
            batch_size=2048,
            decoder=torchcrepe.decode.weighted_argmax,
        )
    p4b = p4b.squeeze(0).numpy()
    per4b = per4b.squeeze(0).numpy()
    print(f"Result 4b (16k + weighted_argmax): pitch_hz min={p4b.min():.1f}, max={p4b.max():.1f}, mean={p4b.mean():.1f}, median={np.median(p4b):.1f}")
    print(f"Periodicity 4b: mean={per4b.mean():.4f}, median={np.median(per4b):.4f}")
    print(f"Frames > 1950 Hz: {np.sum(p4b > 1950)} / {len(p4b)} ({np.mean(p4b > 1950)*100:.1f}%)")

    # 5. Test model tiny vs full with weighted_argmax
    print(f"\n--- [Test 5] Model = tiny (with 16k + weighted_argmax) ---")
    with torch.no_grad():
        pt, pert = torchcrepe.predict(
            t_16k,
            16000,
            hop_length=hop_16k,
            model="tiny",
            device="cpu",
            return_periodicity=True,
            batch_size=2048,
            decoder=torchcrepe.decode.weighted_argmax,
        )
    pt = pt.squeeze(0).numpy()
    pert = pert.squeeze(0).numpy()
    print(f"Result 5 (tiny + 16k + weighted_argmax): pitch_hz min={pt.min():.1f}, max={pt.max():.1f}, mean={pt.mean():.1f}, median={np.median(pt):.1f}")
    print(f"Periodicity 5: mean={pert.mean():.4f}, median={np.median(pert):.4f}")
    print(f"Frames > 1950 Hz: {np.sum(pt > 1950)} / {len(pt)} ({np.mean(pt > 1950)*100:.1f}%)")

    # 6. Test pyin fallback
    try:
        import librosa
        print(f"\n--- [Test 6] librosa.pyin comparison ---")
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
            audio_16k,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=16000,
            hop_length=hop_16k,
        )
        valid_f0 = f0_pyin[~np.isnan(f0_pyin)]
        print(f"pyin f0: valid_frames={len(valid_f0)} / {len(f0_pyin)}, mean={valid_f0.mean():.1f} Hz, median={np.median(valid_f0):.1f} Hz")
    except Exception as pyin_e:
        print(f"librosa.pyin test error: {pyin_e}")

def main():
    if "--inspect-internals" in sys.argv:
        inspect_crepe_internals()
        return

    stems_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/diag_stems"
    os.makedirs(stems_dir, exist_ok=True)

    items = ["abjones_3_09", "bobon_3_08", "yifen_3_02", "RWC_P056"]
    for item in items:
        flac_path = os.path.join(stems_dir, f"{item}_vocals.flac")
        wav_path = os.path.join(stems_dir, f"{item}.wav")
        target_path = flac_path if os.path.exists(flac_path) else wav_path
        if os.path.exists(target_path):
            run_diagnostics_on_audio(item, target_path)
        else:
            print(f"Target audio missing for {item} in {stems_dir}")

if __name__ == "__main__":
    main()
