"""Real test of jet_engine_spike_pipeline.spk -- does the staged
compressor->combustion->turbine->exhaust wiring actually produce
jet-engine-like DYNAMICS (spool-up ramp, compressor amplification through
stages, a combustion burst, and a load-bearing turbine->compressor
feedback), or does it just have the right neuron NAMES with no real
structural behavior behind them? Tested empirically, not asserted from
the wiring diagram alone.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime

SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jet_engine_spike_pipeline.spk")


def run_sustained_airflow(ast, n_ticks=20, intake_drive=90.0, disable_feedback=False):
    """Simulates sustained intake (like continuous airflow into a running
    engine) over many ticks, one stimulation of all 4 intake neurons per
    tick. Returns per-tick fire counts per stage so the real dynamics over
    time are visible, not just an end-state snapshot."""
    runtime = SpikelingRuntime(ast)
    if disable_feedback:
        runtime.synapses = [s for s in runtime.synapses
                             if not (s.src.startswith("turbine") and s.dst.startswith("comp_"))]

    stage_names = {
        "intake": ["intake1", "intake2", "intake3", "intake4"],
        "compressor": ["comp_stage2a", "comp_stage2b", "comp_stage3"],
        "combustion": ["combustion"],
        "turbine": ["turbine1", "turbine2"],
        "exhaust": ["exhaust"],
    }
    history = {stage: [] for stage in stage_names}
    prev_counts = {n: 0 for n in runtime.neurons}

    for tick in range(n_ticks):
        t = float(tick) * 10.0
        for name in stage_names["intake"]:
            runtime.stimulate(name, t, drive=intake_drive)
        for stage, names in stage_names.items():
            fired_this_tick = sum(runtime.neurons[n].fire_count - prev_counts[n] for n in names)
            history[stage].append(fired_this_tick)
        for n in runtime.neurons:
            prev_counts[n] = runtime.neurons[n].fire_count

    return history


def main():
    out_dir = tempfile.mkdtemp(prefix="jet_engine_test_")
    ast = compile_file(SPK_PATH, output_dir=out_dir)

    print("=== SPOOL-UP: sustained intake drive, 20 ticks ===")
    history = run_sustained_airflow(ast, n_ticks=20)
    for stage in ["intake", "compressor", "combustion", "turbine", "exhaust"]:
        print(f"  {stage:>12}: {history[stage]}")

    print("\n=== REAL CHECK 1: does activity actually ramp up before steady state (spool-up)? ===")
    combustion_hist = history["combustion"]
    first_fire_tick = next((i for i, c in enumerate(combustion_hist) if c > 0), None)
    if first_fire_tick is not None and first_fire_tick > 0:
        print(f"  Combustion doesn't fire until tick {first_fire_tick} -- real spool-up delay, "
              "not instant steady-state from tick 0.")
    elif first_fire_tick == 0:
        print("  Combustion fires immediately at tick 0 -- no real spool-up delay in this design.")
    else:
        print("  Combustion never fires -- pipeline doesn't reach ignition at this drive level.")

    print("\n=== REAL CHECK 2: turbine->compressor feedback -- is it load-bearing or decorative? ===")
    ast2 = compile_file(SPK_PATH, output_dir=out_dir)
    with_fb = run_sustained_airflow(ast2, n_ticks=20, disable_feedback=False)
    ast3 = compile_file(SPK_PATH, output_dir=out_dir)
    without_fb = run_sustained_airflow(ast3, n_ticks=20, disable_feedback=True)

    total_compressor_with = sum(with_fb["compressor"])
    total_compressor_without = sum(without_fb["compressor"])
    total_exhaust_with = sum(with_fb["exhaust"])
    total_exhaust_without = sum(without_fb["exhaust"])
    print(f"  compressor total fires -- WITH feedback: {total_compressor_with}  "
          f"WITHOUT feedback: {total_compressor_without}")
    print(f"  exhaust total fires    -- WITH feedback: {total_exhaust_with}  "
          f"WITHOUT feedback: {total_exhaust_without}")
    if total_compressor_with != total_compressor_without or total_exhaust_with != total_exhaust_without:
        print("  Real result: the turbine->compressor feedback measurably changes pipeline behavior -- "
              "it's load-bearing, not just present in the wiring.")
    else:
        print("  Real result: removing the feedback made NO measurable difference -- "
              "it's currently decorative at this drive level, disclosed not hidden.")

    print("\n=== REAL CHECK 3: does the compressor amplify (converging stages fire MORE densely)? ===")
    stage_totals = {s: sum(history[s]) for s in ["intake", "compressor", "combustion", "turbine", "exhaust"]}
    stage_neuron_counts = {"intake": 4, "compressor": 3, "combustion": 1, "turbine": 2, "exhaust": 1}
    for stage in ["intake", "compressor", "combustion", "turbine", "exhaust"]:
        rate = stage_totals[stage] / stage_neuron_counts[stage] / 20.0
        print(f"  {stage:>12}: {stage_totals[stage]} total fires across {stage_neuron_counts[stage]} "
              f"neuron(s), {rate:.3f} fires/neuron/tick")


if __name__ == "__main__":
    main()
