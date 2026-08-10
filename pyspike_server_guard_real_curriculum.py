#!/usr/bin/env python
"""pyspike_server_guard_real_curriculum.py -- the narrower, real-data
version: tests the currently-deployed jet-engine weights (STDP-derived,
seed=1000) against REAL historical server_guard.db windows, not the
synthetic sustained-vs-blip curriculum used to select/validate them
tonight. Also checks whether BEACON_SCALE=15.0 (calibrated against the
ORIGINAL hand-tuned weights) is still appropriate for the newly deployed
weights.

REAL windows extracted from actual history, not constructed:
  - sustained: the first 20 readings of each real run of >=15 consecutive
    elevated (>=5) readings -- 39 such runs exist in the real data.
  - blip: the actual real short elevated bursts (1-8 consecutive readings
    >=5) embedded in their REAL surrounding quiet readings -- 5 such
    bursts exist naturally in the real data. Small real sample, disclosed,
    not padded out with synthetic examples to look bigger than it is.
"""
import os
import sys
import sqlite3

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

import numpy as np
import tempfile
from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime

SPK_PATH = os.path.join(SPIKELING_ROOT, "ai-apps", "jet_engine_spike_pipeline.spk")
SERVER_GUARD_DB = r"C:\Users\gbran\OneDrive\Documents\server-guard\server_guard.db"
BEACON_SCALE = 15.0
CHANNEL = "pkt.beacon_candidate_destinations"


def load_real_values(channel=CHANNEL):
    conn = sqlite3.connect(SERVER_GUARD_DB)
    cur = conn.cursor()
    cur.execute("SELECT timestamp, value FROM readings WHERE channel = ? ORDER BY timestamp", (channel,))
    rows = cur.fetchall()
    conn.close()
    return np.array([v for _, v in rows])


def extract_real_windows(values, elevated_threshold=5, sustained_min_len=15,
                          blip_max_len=8, sustained_window=20, pad=5):
    elevated = values >= elevated_threshold
    runs = []  # (start_idx, length)
    cur_start, cur_len = None, 0
    for i, e in enumerate(elevated):
        if e:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append((cur_start, cur_len))
            cur_len = 0
    if cur_len > 0:
        runs.append((cur_start, cur_len))

    sustained_windows = []
    blip_windows = []
    for start, length in runs:
        if length >= sustained_min_len:
            sustained_windows.append(values[start:start + sustained_window])
        elif 1 <= length <= blip_max_len:
            w_start = max(0, start - pad)
            w_end = min(len(values), start + length + pad)
            blip_windows.append(values[w_start:w_end])
    return sustained_windows, blip_windows


def compile_runtime(spk_path=SPK_PATH):
    ast = compile_file(spk_path, output_dir=tempfile.mkdtemp(prefix="real_curr_"))
    return SpikelingRuntime(ast), ast


def run_window(runtime, real_values, scale):
    intake_names = ["intake1", "intake2", "intake3", "intake4"]
    prev = runtime.neurons["combustion"].fire_count
    for i, val in enumerate(real_values):
        t = float(i) * 10.0
        drive = val * scale
        for name in intake_names:
            runtime.stimulate(name, t, drive=drive)
    return runtime.neurons["combustion"].fire_count - prev


def evaluate(sustained_windows, blip_windows, scale, label, spk_path=SPK_PATH, verbose=True):
    correct = 0
    total = len(sustained_windows) + len(blip_windows)
    detail = []
    for w in sustained_windows:
        runtime, _ = compile_runtime(spk_path)
        c = run_window(runtime, w, scale)
        ok = c > 0
        correct += int(ok)
        detail.append(("sustained", ok, c))
    for w in blip_windows:
        runtime, _ = compile_runtime(spk_path)
        c = run_window(runtime, w, scale)
        ok = c == 0
        correct += int(ok)
        detail.append(("blip", ok, c))
    acc = correct / total
    sustained_correct = sum(1 for k, ok, c in detail if k == "sustained" and ok)
    blip_correct = sum(1 for k, ok, c in detail if k == "blip" and ok)
    n_sustained = len(sustained_windows)
    n_blip = len(blip_windows)
    print(f"{label}: {correct}/{total} = {acc:.1%}  "
          f"(sustained: {sustained_correct}/{n_sustained} correctly ignited, "
          f"blip: {blip_correct}/{n_blip} correctly stayed quiet)")
    if verbose:
        for kind, ok, c in detail:
            print(f"    {kind:10s} {'OK  ' if ok else 'FAIL'}  combustion_fires={c}")
    return acc


if __name__ == "__main__":
    print("=" * 78)
    print("  SERVER-GUARD REAL-DATA CURRICULUM TEST")
    print("=" * 78)

    values = load_real_values()
    sustained_windows, blip_windows = extract_real_windows(values)
    print(f"real sustained windows found: {len(sustained_windows)}")
    print(f"real blip windows found: {len(blip_windows)}\n")

    BACKUP_SPK = SPK_PATH + ".backup_before_stdp"

    print(f"--- currently DEPLOYED weights (STDP seed=1000), BEACON_SCALE={BEACON_SCALE} ---")
    acc_deployed = evaluate(sustained_windows, blip_windows, BEACON_SCALE,
                             "deployed (STDP) weights, current scale", spk_path=SPK_PATH, verbose=False)

    print(f"\n--- ORIGINAL hand-tuned weights (pre-STDP backup), BEACON_SCALE={BEACON_SCALE} ---")
    acc_handtuned = evaluate(sustained_windows, blip_windows, BEACON_SCALE,
                              "hand-tuned weights, current scale", spk_path=BACKUP_SPK, verbose=False)

    print(f"\n{'='*78}")
    print(f"COMPARISON on REAL server-guard data (44 real windows: 39 sustained, 5 blip):")
    print(f"  hand-tuned weights (pre-STDP):     {acc_handtuned:.1%}")
    print(f"  deployed weights (STDP seed=1000): {acc_deployed:.1%}")
    print("=" * 78)
