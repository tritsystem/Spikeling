#!/usr/bin/env python
"""
Cross-backend conformance harness for the first-class neuron models
(Izhikevich / AdEx). This is "B as the proof" for the deep contribution
described in core/PRE_REGISTRATION_neuron_models.md -- it checks the four
pre-registered acceptance bars and prints a verdict table.

    python core/tests/test_neuron_model_conformance.py

Exit code 0 iff every reachable bar is MET. A bar with no C compiler present
is reported UNREACHABLE (not a pass, not a fail) and does not fail the run.

Bars
----
1  LIF zero-regression : every existing example .spk produces a byte-identical
                         spike log to the pre-change golden (core/tests/golden/).
2  Izhikevich fidelity : chattering burst signature + exact spike-time match to
                         pyspike_neuron_models.IzhikevichNeuron.
3  AdEx fidelity       : spike-frequency adaptation (ISI grows) while a runtime
                         LIF neuron under matched drive does not; exact match to
                         pyspike_neuron_models.AdExNeuron.
4  Generated-C parity  : CCodeGenerator.generate_model_sim() for the chattering
                         net, compiled with gcc, matches the Python runtime
                         within +-1 tick and the same spike count.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
REPO = os.path.dirname(CORE)
for p in (CORE, HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from compiler.compiler import SpikelingParser, CCodeGenerator      # noqa: E402
from runtime.runtime import SpikelingRuntime                       # noqa: E402
from runtime.neuron_models import build_model                      # noqa: E402
from _conformance_drive import drive_and_log                       # noqa: E402

EX_DIR = os.path.join(CORE, "examples")
GOLDEN = os.path.join(HERE, "golden")


def _isis(x):
    return [x[i + 1] - x[i] for i in range(len(x) - 1)]


def _ratio(x):
    i = _isis(x)
    return (max(i) / min(i)) if len(i) > 2 else 0.0


def _parse(name):
    return SpikelingParser().parse(open(os.path.join(EX_DIR, name + ".spk")).read())


# ── Bar 1 ────────────────────────────────────────────────────────────────────
def bar1_lif_regression():
    examples = ["sound_localizer", "signal_detector", "word_learner", "agent_brain"]
    details = []
    ok = True
    for ex in examples:
        gpath = os.path.join(GOLDEN, ex + ".spikes.json")
        if not os.path.exists(gpath):
            details.append(f"{ex}: NO GOLDEN")
            ok = False
            continue
        golden = json.load(open(gpath))
        rt = SpikelingRuntime(_parse(ex))
        got = drive_and_log(rt)
        if got == golden:
            details.append(f"{ex}: {len(got)} spikes identical")
        else:
            ok = False
            first = next((k for k in range(min(len(got), len(golden)))
                          if got[k] != golden[k]), None)
            details.append(f"{ex}: DIFFERS (len {len(got)} vs {len(golden)}, "
                           f"first diff idx {first})")
    return ("MET" if ok else "NOT-MET"), details


# ── Bar 2 ────────────────────────────────────────────────────────────────────
def bar2_izhikevich():
    from pyspike_neuron_models import IzhikevichNeuron
    ast = _parse("izhikevich_chattering")
    rt = SpikelingRuntime(ast, dt_ms=0.5)
    for i in range(600):
        rt.stimulate("rs", i * 0.5, drive=10.0)
        rt.stimulate("ch", i * 0.5, drive=10.0)
    ch = [round(t, 6) for (t, nm) in rt._spike_log if nm == "ch"]
    rs = [round(t, 6) for (t, nm) in rt._spike_log if nm == "rs"]

    ref = IzhikevichNeuron("chattering")
    t = 0.0
    while t < 300.0:
        ref.step(10.0, 0.5, t)
        t += 0.5
    ref_ch = [round(x, 6) for x in ref.spike_log]

    c_count = len(ch) > 5
    c_burst = _ratio(ch) > 1.5 * _ratio(rs)
    c_exact = ch == ref_ch
    details = [
        f"chattering: {len(ch)} spikes (>5: {c_count})",
        f"ISI range ratio ch={_ratio(ch):.2f} vs rs={_ratio(rs):.2f} "
        f"(ch > 1.5x rs: {c_burst})",
        f"exact match to IzhikevichNeuron reference: {c_exact} "
        f"({len(ch)} vs {len(ref_ch)})",
    ]
    return ("MET" if (c_count and c_burst and c_exact) else "NOT-MET"), details


# ── Bar 3 ────────────────────────────────────────────────────────────────────
def bar3_adex():
    from pyspike_neuron_models import AdExNeuron
    ast = _parse("adex_adaptation")
    rt = SpikelingRuntime(ast, dt_ms=0.1)
    for i in range(5000):
        rt.stimulate("a", i * 0.1, drive=50.0)
    a = [t for (t, nm) in rt._spike_log if nm == "a"]
    ia = _isis(a)

    ref = AdExNeuron()
    t = 0.0
    while t < 500.0:
        ref.step(400.0, 0.1, t)
        t += 0.1
    ref_a = [round(x, 6) for x in ref.spike_log]
    exact = [round(x, 6) for x in a] == ref_a

    adapts = len(ia) >= 3 and ia[-1] > ia[0] * 1.2

    lif_ast = SpikelingParser().parse("neuron lif_ctrl threshold=100 leak=5 type=LIF\n")
    lrt = SpikelingRuntime(lif_ast, dt_ms=0.1)
    for i in range(5000):
        lrt.stimulate("lif_ctrl", i * 0.1, drive=8.0)
    lil = _isis([t for (t, nm) in lrt._spike_log if nm == "lif_ctrl"])
    lif_flat = len(lil) >= 3 and abs(lil[-1] - lil[0]) < lil[0] * 0.15

    details = [
        f"AdEx: {len(a)} spikes, ISI {ia[0]:.1f} -> {ia[-1]:.1f} ms "
        f"(grows > 1.2x: {adapts})",
        f"exact match to AdExNeuron reference: {exact} ({len(a)} vs {len(ref_a)})",
        f"LIF control: {len(lil)+1} spikes, ISI {lil[0]:.2f} -> {lil[-1]:.2f} ms "
        f"(flat: {lif_flat})",
    ]
    return ("MET" if (adapts and exact and lif_flat) else "NOT-MET"), details


# ── Bar 4 ────────────────────────────────────────────────────────────────────
def bar4_c_parity():
    gcc = shutil.which("gcc") or shutil.which("cc")
    if not gcc:
        return "UNREACHABLE", ["no gcc / cc on PATH"]

    ast = _parse("izhikevich_chattering")
    tmp = tempfile.mkdtemp(prefix="spk_conf_")
    try:
        cpath = CCodeGenerator().generate_model_sim(
            ast, "ch", drive=10.0, total_ms=300.0, dt_ms=0.5, output_dir=tmp)
        exe = os.path.join(tmp, "sim.exe")
        cp = subprocess.run([gcc, "-O2", "-std=c11", cpath, "-lm", "-o", exe],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            return "NOT-MET", [f"gcc failed: {cp.stderr.strip()[:200]}"]
        out = subprocess.run([exe], capture_output=True, text=True).stdout
        c_spikes = [float(x) for x in out.split()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rt = SpikelingRuntime(ast, dt_ms=0.5)
    for i in range(600):
        rt.stimulate("ch", i * 0.5, drive=10.0)
    py_spikes = [t for (t, nm) in rt._spike_log if nm == "ch"]

    same_count = len(c_spikes) == len(py_spikes)
    within = same_count and all(abs(c - p) <= 0.5 for c, p in zip(c_spikes, py_spikes))
    max_dev = max((abs(c - p) for c, p in zip(c_spikes, py_spikes)), default=0.0)
    details = [
        f"C: {len(c_spikes)} spikes, Python: {len(py_spikes)} spikes "
        f"(same count: {same_count})",
        f"max per-spike deviation: {max_dev:.3f} ms (tolerance 0.5 ms: {within})",
    ]
    return ("MET" if within else "NOT-MET"), details


def main():
    bars = [
        ("Bar 1  LIF zero-regression", bar1_lif_regression),
        ("Bar 2  Izhikevich fidelity", bar2_izhikevich),
        ("Bar 3  AdEx fidelity", bar3_adex),
        ("Bar 4  generated-C parity", bar4_c_parity),
    ]
    print("=" * 78)
    print("  NEURON-MODEL CROSS-BACKEND CONFORMANCE  (see PRE_REGISTRATION_neuron_models.md)")
    print("=" * 78)
    verdicts = []
    for title, fn in bars:
        try:
            verdict, details = fn()
        except Exception as e:  # a broken bar is a NOT-MET, with the traceback
            import traceback
            verdict, details = "NOT-MET", [f"EXCEPTION: {e}"] + \
                traceback.format_exc().strip().splitlines()[-3:]
        verdicts.append(verdict)
        print(f"\n[{verdict:11}] {title}")
        for d in details:
            print(f"    {d}")
    print("\n" + "=" * 78)
    n_met = verdicts.count("MET")
    n_unreach = verdicts.count("UNREACHABLE")
    n_fail = verdicts.count("NOT-MET")
    print(f"  {n_met} MET   {n_unreach} UNREACHABLE   {n_fail} NOT-MET")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
