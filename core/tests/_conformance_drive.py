"""Shared, fully-deterministic stimulation script for the LIF zero-regression
bar. Imported by BOTH the golden-capture step and the conformance harness so
the two runs are byte-comparable. Do not change this without regenerating the
goldens (and then it is no longer a regression test).
"""
from __future__ import annotations

import random


def drive_and_log(rt, n_ticks: int = 2000, seed: int = 1234) -> list:
    """Run `rt` (a SpikelingRuntime) through a fixed pseudo-random stimulation
    schedule and return its spike log as a list of [time, name] pairs."""
    rng = random.Random(seed)
    names = sorted(rt.neurons.keys())
    if not names:
        return []
    for i in range(n_ticks):
        t = float(i)
        target = rng.choice(names)
        drive = rng.uniform(0.0, 120.0)
        rt.stimulate(target, t, drive=drive)
        if i % 7 == 0:
            rt.tick(t)
    return [[round(float(tm), 6), nm] for (tm, nm) in rt._spike_log]
