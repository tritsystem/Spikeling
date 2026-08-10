"""Tests whether the jet-engine pipeline's feedback-reinforced region
(comp_stage2a/comp_stage2b -- the only neurons that receive BOTH direct
intake drive AND turbine feedback) is more robust to synapse dropout than
the purely feedforward region (comp_stage3, combustion, turbine1, turbine2,
exhaust -- no feedback reinforcement).

This is the analogous structural question for a staged pipeline that edge-
vs-bulk was for the SSH lattice work: not a forced metaphor, a real
structural asymmetry this specific topology actually has (feedback-
reinforced vs feedback-free), tested the same way (perturbation sweep,
many seeds, report the honest spread -- not a single run).

CONFOUND CAUGHT BEFORE RUNNING (matches the topological-phononics vault's
own repeated input-locality confound pattern): the turbine->compressor
feedback synapse (weight=2.0 -> 100 raw pulse under the fixed-pulse
mechanism) ALONE clears comp_stage2a/2b's threshold=100 once the engine
has spooled up. Random dropout that happens to hit that one synapse would
trivially "prove" feedback matters -- circular, not a real robustness test.
Split into two conditions:
  (a) dropout restricted to the 12 NON-feedback synapses -- tests whether
      RELYING on feedback leaves the reinforced region more or less
      robust to unrelated failures elsewhere, with the feedback path
      itself always intact.
  (b) dropout allowed to hit any of the 14 synapses including feedback --
      the more trivial "does removing feedback hurt" question, included
      for contrast, not as the main claim.

Prior established pattern from the topological-phononics work (checked
before running, not after): structured/protected robustness advantages
lose to the naive baseline far more often than they hold, and hold only
under narrow, specific conditions when they do. Going in expecting a null
or negative result as the norm, not the exception.

Real limitation, disclosed up front: only 14 synapses total in this
network, so a single dropout trial can be dominated by which specific
synapses happen to get hit. Needs many seeds to average that out, and the
result describes THIS topology, not staged pipelines in general.
"""
import os
import sys
import random
import statistics
import math

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime
import tempfile

_SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jet_engine_spike_pipeline.spk")

REINFORCED = {"comp_stage2a", "comp_stage2b"}
FEEDFORWARD_ONLY = {"comp_stage3", "combustion", "turbine1", "turbine2", "exhaust"}
FEEDBACK_SYNAPSES = {("turbine1", "comp_stage2a"), ("turbine2", "comp_stage2b")}

INTAKE_NEURONS = ["intake1", "intake2", "intake3", "intake4"]
N_TICKS = 20
INTAKE_DRIVE = 80.0
N_SEEDS = 50
DROPOUT_FRAC = 0.3


def run_pipeline(synapses):
    ast = compile_file(_SPK_PATH, output_dir=tempfile.mkdtemp(prefix="jet_robustness_"))
    runtime = SpikelingRuntime(ast)
    runtime.synapses = synapses  # real live list -- _fire() scans this on every call
    for tick in range(N_TICKS):
        t = float(tick) * 10.0
        for name in INTAKE_NEURONS:
            runtime.stimulate(name, t, drive=INTAKE_DRIVE)
    return {name: n.fire_count for name, n in runtime.neurons.items()}


def get_full_synapses():
    ast = compile_file(_SPK_PATH, output_dir=tempfile.mkdtemp(prefix="jet_robustness_full_"))
    runtime = SpikelingRuntime(ast)
    return list(runtime.synapses)


def dropout(synapses, frac, rng, protect_feedback):
    eligible_idx = [i for i, s in enumerate(synapses)
                    if not (protect_feedback and (s.src, s.dst) in FEEDBACK_SYNAPSES)]
    n_drop = max(1, round(len(synapses) * frac))
    n_drop = min(n_drop, len(eligible_idx))
    idx_to_drop = set(rng.sample(eligible_idx, n_drop))
    return [s for i, s in enumerate(synapses) if i not in idx_to_drop], n_drop


def ci95(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    se = s / math.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


def run_condition(full_synapses, clean_fires, protect_feedback, label):
    per_neuron_ratios = {n: [] for n in REINFORCED | FEEDFORWARD_ONLY}
    n_drop_last = None

    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        trial_synapses, n_drop = dropout(full_synapses, DROPOUT_FRAC, rng, protect_feedback)
        n_drop_last = n_drop
        trial_fires = run_pipeline(trial_synapses)
        for name in REINFORCED | FEEDFORWARD_ONLY:
            clean = clean_fires[name]
            trial = trial_fires[name]
            ratio = (trial / clean) if clean > 0 else (1.0 if trial == 0 else float("nan"))
            per_neuron_ratios[name].append(ratio)

    reinforced_ratios = [r for n in REINFORCED for r in per_neuron_ratios[n]]
    feedforward_ratios = [r for n in FEEDFORWARD_ONLY for r in per_neuron_ratios[n]]

    print(f"\n=== {label} ({N_SEEDS} seeds, {DROPOUT_FRAC*100:.0f}% dropout, "
          f"~{n_drop_last} synapses/trial, feedback synapses {'PROTECTED' if protect_feedback else 'eligible to drop'}) ===")
    for name in ["comp_stage2a", "comp_stage2b", "comp_stage3", "combustion", "turbine1", "turbine2", "exhaust"]:
        vals = per_neuron_ratios[name]
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        tag = "REINFORCED" if name in REINFORCED else "feedforward-only"
        print(f"  {name:16s} mean={mean:.3f}  stdev={stdev:.3f}  [{tag}]")

    r_m, r_lo, r_hi = ci95(reinforced_ratios)
    f_m, f_lo, f_hi = ci95(feedforward_ratios)
    print(f"  GROUP reinforced        mean={r_m:.3f}  95% CI=[{r_lo:.3f}, {r_hi:.3f}]  n={len(reinforced_ratios)}")
    print(f"  GROUP feedforward-only  mean={f_m:.3f}  95% CI=[{f_lo:.3f}, {f_hi:.3f}]  n={len(feedforward_ratios)}")
    overlap = not (r_lo > f_hi or f_lo > r_hi)
    if overlap:
        print("  VERDICT: CIs overlap -- no clear separation.")
    else:
        winner = "reinforced" if r_m > f_m else "feedforward-only"
        print(f"  VERDICT: real separation, {winner} region shows higher mean robustness.")
    return r_m, f_m, overlap


def run_no_feedback_control(full_synapses):
    """CONFOUND CONTROL: structurally removes the 2 feedback synapses so
    they never exist (not just protected from dropout), leaving
    comp_stage2a/2b at the same 1-hop distance from intake but with no
    feedback pathway at all. If comp_stage2a/2b are STILL this robust to
    dropout on the remaining 12 synapses, the depth/hop-distance confound
    -- not feedback -- is what's driving Condition A's result."""
    no_fb_synapses = [s for s in full_synapses if (s.src, s.dst) not in FEEDBACK_SYNAPSES]
    clean_fires_nofb = run_pipeline(list(no_fb_synapses))
    print("\nno-feedback-variant clean baseline fire counts:")
    for name, count in clean_fires_nofb.items():
        print(f"  {name:16s} {count}")

    per_neuron_ratios = {n: [] for n in REINFORCED | FEEDFORWARD_ONLY}
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        trial_synapses, n_drop = dropout(no_fb_synapses, DROPOUT_FRAC, rng, protect_feedback=False)
        trial_fires = run_pipeline(trial_synapses)
        for name in REINFORCED | FEEDFORWARD_ONLY:
            clean = clean_fires_nofb[name]
            trial = trial_fires[name]
            ratio = (trial / clean) if clean > 0 else (1.0 if trial == 0 else float("nan"))
            per_neuron_ratios[name].append(ratio)

    reinforced_ratios = [r for n in REINFORCED for r in per_neuron_ratios[n]]
    print(f"\n=== CONTROL: feedback synapses REMOVED ENTIRELY ({N_SEEDS} seeds, "
          f"{DROPOUT_FRAC*100:.0f}% dropout on remaining {len(no_fb_synapses)} synapses) ===")
    for name in ["comp_stage2a", "comp_stage2b"]:
        vals = per_neuron_ratios[name]
        print(f"  {name:16s} mean={statistics.mean(vals):.3f}  stdev={statistics.stdev(vals):.3f}  [same 1-hop distance, NO feedback]")
    r_m, r_lo, r_hi = ci95(reinforced_ratios)
    print(f"  GROUP comp_stage2a/2b (no feedback)  mean={r_m:.3f}  95% CI=[{r_lo:.3f}, {r_hi:.3f}]  n={len(reinforced_ratios)}")
    return r_m, r_lo, r_hi


def main():
    full_synapses = get_full_synapses()
    print(f"real synapse count: {len(full_synapses)}")

    clean_fires = run_pipeline(list(full_synapses))
    print("clean baseline fire counts:")
    for name, count in clean_fires.items():
        print(f"  {name:16s} {count}")

    r_a, f_a, overlap_a = run_condition(full_synapses, clean_fires, protect_feedback=True,
                  label="CONDITION A: feedback synapses protected (tests general robustness, not tautology)")
    run_condition(full_synapses, clean_fires, protect_feedback=False,
                  label="CONDITION B: feedback synapses eligible to drop (includes the trivial case)")
    ctrl_m, ctrl_lo, ctrl_hi = run_no_feedback_control(full_synapses)

    print(f"\n{'='*70}\nCONFOUND CHECK: is Condition A's result about feedback, or just hop-distance?\n{'='*70}")
    print(f"  Condition A (WITH feedback, protected):     reinforced mean = {r_a:.3f}")
    print(f"  Control     (WITHOUT feedback, same depth): reinforced mean = {ctrl_m:.3f}  95% CI=[{ctrl_lo:.3f}, {ctrl_hi:.3f}]")
    if ctrl_lo <= r_a <= ctrl_hi or abs(ctrl_m - r_a) < 0.05:
        print("  VERDICT: no real difference -- Condition A's result is explained by hop-distance, NOT feedback. CONFOUNDED, retract.")
    elif r_a > ctrl_m:
        print(f"  VERDICT: real gap remains after controlling for depth ({r_a:.3f} vs {ctrl_m:.3f}) -- "
              f"feedback itself HELPS robustness, not just distance.")
    else:
        print(f"  VERDICT: real gap remains after controlling for depth ({r_a:.3f} vs {ctrl_m:.3f}) -- "
              f"feedback itself HURTS robustness (reinforced neurons are LESS robust WITH feedback than "
              f"the same depth WITHOUT it). Naive hypothesis reversed, not confirmed.")


if __name__ == "__main__":
    main()
