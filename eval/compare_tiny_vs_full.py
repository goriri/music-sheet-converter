#!/usr/bin/env python3
import time
import numpy as np
import soundfile as sf
import torch
import torchcrepe

def p(msg):
    print(msg, flush=True)

v_audio, sr = sf.read("/workspace/diag_stems/RWC_P056_vocals.flac")
if v_audio.ndim > 1:
    v_audio = np.mean(v_audio, axis=1)

slice_audio = v_audio[60*16000 : 90*16000].astype(np.float32)
dur = len(slice_audio) / 16000.0
p(f"Test slice duration: {dur:.1f}s")
slice_t = torch.from_numpy(slice_audio).unsqueeze(0)

configs = [
    ("tiny", 160, 2048, "Tiny 10ms | bs=2048"),
    ("tiny", 160, 4096, "Tiny 10ms | bs=4096"),
    ("tiny", 320, 2048, "Tiny 20ms | bs=2048"),
    ("tiny", 320, 4096, "Tiny 20ms | bs=4096"),
]

p("\n=== CREPE TINY BENCHMARK (30s) ===")
for model_name, hop, bs, desc in configs:
    t0 = time.perf_counter()
    with torch.no_grad():
        pitch, per = torchcrepe.predict(
            slice_t,
            16000,
            hop_length=hop,
            model=model_name,
            device="cpu",
            return_periodicity=True,
            batch_size=bs,
            decoder=torchcrepe.decode.weighted_argmax,
        )
    elapsed = time.perf_counter() - t0
    p_np = pitch.squeeze(0).numpy()
    per_np = per.squeeze(0).numpy()
    p(f"[{desc:24s}] Wall: {elapsed:5.2f}s | Speed: {dur/elapsed:5.1f}x RT | Extrapolated 4-min: {elapsed/dur*240:5.1f}s | Mean Per: {np.mean(per_np):.3f} | Median Pitch: {np.median(p_np[p_np>65]):.1f} Hz")
