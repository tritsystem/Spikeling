#!/usr/bin/env python
"""pyspike_jet_engine_stdp.py -- "intuition" instead of "reasoning": can the
jet-engine's REAL STDP mechanism (runtime.py's STDPLearner, already
existing, unmodified, local spike-timing correlation only) learn to
discriminate sustained drive from a blip WITHOUT ever being told which is
which -- no loss, no target, no backward pass, just repeated exposure?

Contrast with pyspike_jet_engine_train.py's backprop result: that's
explicit, global, error-corrected learning (told exactly what's wrong and
how to fix it). This is local, unsupervised, correlation-only learning --
weights only strengthen or weaken based on the RELATIVE TIMING of two
directly-connected neurons, using the REAL STDPLearner class as it already
exists (dt = t_pre - downstream.last_spike_time, LTP if positive/large,
LTD if negative -- read directly from runtime.py, not reimplemented).

TRAINING PHASE (the "experience"): repeatedly stimulate with SUSTAINED
drive only (the realistic/common case) -- pure exposure, no blips shown,
no correctness signal of any kind. If synapses that fire together
reliably during real ignition sequences strengthen via Hebbian
correlation, discrimination might emerge as a side effect of the network
learning "this is what a normal firing sequence looks like" -- or it
might not. Not assumed either way; measured against the exact same
held-out evaluation used for the backprop comparison.
"""
import os
import sys

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

import tempfile
import numpy as np

from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime, STDPLearner

from pyspike_jet_engine_train import (
    SPK_PATH, make_curriculum, DRIVE_LOW, DRIVE_HIGH, T_SUSTAINED, T_WINDOW,
)


def build_runtime_with_stdp(rate=0.02, tau=20.0):
    ast = compile_file(SPK_PATH, output_dir=tempfile.mkdtemp(prefix="stdp_"))
    runtime = SpikelingRuntime(ast)
    runtime.learner = STDPLearner(rate=rate, tau=tau)  # not in the .spk file -- attached directly
    return runtime, ast


def run_trial(runtime, drive_level, sustained, burst_dur=None, start=None):
    """Real stimulate() calls on the REAL runtime -- same object used in
    production (jet_engine_gate.py), STDP updates happen automatically
    inside _fire() as a side effect of real propagation, exactly as
    designed, not specially invoked."""
    intake_names = ["intake1", "intake2", "intake3", "intake4"]
    prev_counts = {n: runtime.neurons[n].fire_count for n in runtime.neurons}
    if sustained:
        for tick in range(T_SUSTAINED):
            t = float(tick) * 10.0
            for name in intake_names:
                runtime.stimulate(name, t, drive=drive_level)
    else:
        for tick in range(T_WINDOW):
            t = float(tick) * 10.0
            if start <= tick < start + burst_dur:
                for name in intake_names:
                    runtime.stimulate(name, t, drive=drive_level)
    combustion_fires = runtime.neurons["combustion"].fire_count - prev_counts["combustion"]
    return combustion_fires


def evaluate_runtime(runtime_factory, trials, learner=None):
    """Fresh runtime per trial (so evaluation doesn't itself cause further
    STDP drift) -- but if `learner` is given, weights are copied from a
    PRE-TRAINED synapse list into the fresh runtime first, so we're
    evaluating the STDP-shaped weights, not re-learning during eval."""
    correct = 0
    for spec in trials:
        runtime = runtime_factory()
        if learner is not None:
            for syn, w in zip(runtime.synapses, learner):
                syn.weight = w
        if spec["sustained"]:
            c = run_trial(runtime, spec["drive"], True)
        else:
            c = run_trial(runtime, spec["drive"], False, spec["burst_dur"], spec["start"])
        predicted_ignite = c > 0
        if predicted_ignite == spec["sustained"]:
            correct += 1
    return correct / len(trials)


def fresh_runtime():
    ast = compile_file(SPK_PATH, output_dir=tempfile.mkdtemp(prefix="stdp_eval_"))
    return SpikelingRuntime(ast)


if __name__ == "__main__":
    print("=" * 78)
    print("  JET-ENGINE STDP -- unsupervised, local, no loss, no target")
    print("=" * 78)

    held_out_trials = make_curriculum(n_sustained=8, n_blip=8, seed=99)  # SAME split as the backprop run

    # baseline: real hand-tuned weights, no STDP, no exposure phase
    acc_handtuned = evaluate_runtime(fresh_runtime, held_out_trials)
    print(f"BASELINE (hand-tuned weights, real runtime, no STDP): held-out accuracy = {acc_handtuned:.1%}")

    # UNSUPERVISED EXPOSURE PHASE: real STDP runtime, repeated sustained-only
    # experience -- no labels, no blips shown, no loss computed anywhere
    exposure_runtime, ast = build_runtime_with_stdp(rate=0.02, tau=20.0)
    print(f"\nSTDP config: rate={exposure_runtime.learner.rate}, tau={exposure_runtime.learner.tau}")

    rng = np.random.default_rng(7)
    n_exposures = 40
    print(f"exposing the network to {n_exposures} real sustained-drive trials (pure correlation, "
          f"no correctness signal)...")
    for i in range(n_exposures):
        drive_level = rng.uniform(DRIVE_LOW, DRIVE_HIGH)
        run_trial(exposure_runtime, drive_level, sustained=True)

    stdp_weights = [syn.weight for syn in exposure_runtime.synapses]
    print("\nweights after unsupervised STDP exposure (vs original hand-tuned):")
    ast_original = compile_file(SPK_PATH, output_dir=tempfile.mkdtemp(prefix="stdp_compare_"))
    for syn_orig, w_after in zip(ast_original.connections, stdp_weights):
        print(f"  {syn_orig.src:14s} -> {syn_orig.dst:14s}  {syn_orig.weight:.3f} -> {w_after:.3f}")

    acc_stdp = evaluate_runtime(fresh_runtime, held_out_trials, learner=stdp_weights)
    print(f"\nAFTER unsupervised STDP exposure: held-out accuracy = {acc_stdp:.1%}")

    print(f"\n{'='*78}")
    print("COMPARISON on the SAME held-out set used for the backprop run:")
    print(f"  hand-tuned (manual sweeps):        {acc_handtuned:.1%}")
    print(f"  backprop-trained (explicit target): 93.8%  (from pyspike_jet_engine_train.py)")
    print(f"  STDP (unsupervised, no target):    {acc_stdp:.1%}")
    print("=" * 78)
