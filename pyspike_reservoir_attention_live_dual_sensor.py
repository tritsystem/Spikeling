#!/usr/bin/env python
"""
pyspike_reservoir_attention_live_dual_sensor.py -- the real-sensor-noise
test flagged as the biggest remaining honest gap: everything up to now
ran on synthetic np.random data. This drives TWO independent streaming
reservoir+attention pipelines (same frozen weights from the hybrid
script's Stage 2, no retraining/tuning for real sensors) off:

  - the live microphone (RMS loudness per tick)
  - the live webcam (mean abs frame-difference "motion" per tick)

Each channel is online-normalized (running EMA mean/var -> z-score,
clipped) so ordinary background level sits near the [-1,1] range the
model was trained on, and a genuine loud sound / sudden motion produces
a large-magnitude value analogous to the synthetic 3.0 marker.

This is a live demo, not a controlled experiment: no ground-truth
target exists for a real clap or wave, so there is no NMSE to report.
What IS honestly checkable is the confidence gate verified in
pyspike_reservoir_attention_confidence_gate.py -- does peak hop-1
attention cross the tuned threshold (0.85) when something real actually
happens, and stay low during quiet/still background.
"""
import time
import numpy as np
import cv2
import sounddevice as sd
import torch

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, M_RESERVOIR, ReservoirBank, ReservoirAttentionReadout,
    run_stage, make_task,
)
from pyspike_reservoir_attention_streaming import StreamingReservoir, StreamingAttentionReadout

SAMPLE_RATE = 16000
BLOCK_SIZE = 2048          # ~128ms per tick
DURATION_SEC = 25
CONF_THRESH = 0.85         # balanced threshold from the confidence-gate test
EMA_ALPHA = 0.05


def main():
    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_task(rng, 800)
    test_u, test_y, _ = make_task(rng, 200)
    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR

    torch.manual_seed(0)
    model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    print("Training reference model (identical to every prior test, synthetic data only)...")
    run_stage("offline batched (reference)", model, reservoir, train_u, train_y, test_u, test_y)
    model.eval()

    audio_res = StreamingReservoir(reservoir)
    audio_attn = StreamingAttentionReadout(model, window=T)
    video_res = StreamingReservoir(reservoir)
    video_attn = StreamingAttentionReadout(model, window=T)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("WARNING: camera did not open, video channel will be flat.")
    prev_gray = None

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, dtype="float32")
    stream.start()

    audio_mean = audio_var = None
    video_mean = video_var = None

    print(f"\nLive dual-sensor test running for {DURATION_SEC}s.")
    print("Clap or make a sudden noise for the mic; wave/move suddenly for the camera.\n")
    print(f"{'tick':>4} {'aud_rms':>8} {'aud_z':>7} {'aud_conf':>8} {'vid_mot':>8} {'vid_z':>7} {'vid_conf':>8}  events")

    t0 = time.time()
    tick = 0
    audio_events, video_events = [], []
    try:
        while time.time() - t0 < DURATION_SEC:
            audio_block, _ = stream.read(BLOCK_SIZE)
            rms = float(np.sqrt(np.mean(audio_block.astype(np.float64) ** 2)) + 1e-8)

            motion = 0.0
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if prev_gray is not None:
                        motion = float(np.mean(np.abs(gray - prev_gray)))
                    prev_gray = gray

            if tick == 0:
                audio_mean, audio_var = rms, 1e-6
                video_mean, video_var = motion, 1e-6

            audio_z = (rms - audio_mean) / (audio_var ** 0.5 + 1e-6)
            video_z = (motion - video_mean) / (video_var ** 0.5 + 1e-6)

            audio_mean = (1 - EMA_ALPHA) * audio_mean + EMA_ALPHA * rms
            audio_var = (1 - EMA_ALPHA) * audio_var + EMA_ALPHA * (rms - audio_mean) ** 2
            video_mean = (1 - EMA_ALPHA) * video_mean + EMA_ALPHA * motion
            video_var = (1 - EMA_ALPHA) * video_var + EMA_ALPHA * (motion - video_mean) ** 2

            u_audio = float(np.clip(audio_z, -6, 6))
            u_video = float(np.clip(video_z, -6, 6))

            a_state = audio_res.step(u_audio)
            v_state = video_res.step(u_video)
            a_out = audio_attn.push_and_predict(a_state)
            v_out = video_attn.push_and_predict(v_state)

            a_conf = float(a_out[1].max()) if a_out is not None else float("nan")
            v_conf = float(v_out[1].max()) if v_out is not None else float("nan")

            flags = []
            if a_out is not None and a_conf >= CONF_THRESH:
                flags.append("AUDIO-EVENT")
                audio_events.append(tick)
            if v_out is not None and v_conf >= CONF_THRESH:
                flags.append("VIDEO-EVENT")
                video_events.append(tick)

            print(f"{tick:4d} {rms:8.4f} {u_audio:7.2f} {a_conf:8.3f} {motion:8.3f} {u_video:7.2f} {v_conf:8.3f}  {' '.join(flags)}")
            tick += 1
    finally:
        stream.stop()
        stream.close()
        cap.release()

    print(f"\n{tick} ticks over {DURATION_SEC}s")
    print(f"audio events flagged (tick >= threshold {CONF_THRESH}): {audio_events}")
    print(f"video events flagged (tick >= threshold {CONF_THRESH}): {video_events}")
    if not audio_events and not video_events:
        print("No events crossed the threshold -- either nothing loud/sudden happened, "
              "or real sensor noise doesn't match the synthetic training distribution "
              "closely enough for this frozen model. Both are honest, reportable outcomes.")


if __name__ == "__main__":
    main()
