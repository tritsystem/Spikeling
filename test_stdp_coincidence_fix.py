#!/usr/bin/env python
"""
test_stdp_coincidence_fix.py — real, dedicated regression test for the
2026-08-29 STDPLearner fix (core/runtime/runtime.py).

Real bug, confirmed via git history (commit a5a2109, this project's own
initial commit, 2026-06-22) to have been an unexamined default, never a
deliberate design choice: `if dt > 0` (strict) sent an exactly-
simultaneous pre/post firing (dt=0) to the LTD (weakening) branch,
contradicting the actual reference behavior described in Peter van der
Made's foundational BrainChip patent (US8250011B2): potentiation should
be GREATEST at the shortest pre/post interval, including zero.

This test locks in the fix (`>=`) and re-confirms the parts that were
already correct (the exponential magnitude curve, the dt<0 LTD case,
weight clamping) so a future edit can't silently reintroduce the bug.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "runtime"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from runtime.runtime import STDPLearner, Synapse  # noqa: E402

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"PASS  {label}")
    else:
        FAIL += 1
        print(f"FAIL  {label}")


def make_synapse(weight=0.5):
    return Synapse(src="A", dst="B", weight=weight, delay_ms=0.0)


learner = STDPLearner(rate=0.1, tau=20.0)

# ── the actual fix: dt == 0 must land in LTP, not LTD ───────────────────
syn = make_synapse(0.5)
new_w = learner.update(syn, dt=0.0)
expected_ltp_delta = 0.1 * math.exp(-abs(0.0) / 20.0)  # = 0.1, the max possible delta
check("dt=0 (exact coincidence) now strengthens the synapse (LTP), not weakens it",
      new_w > 0.5)
check("dt=0's weight change matches the real LTP formula exactly (rate * exp(0) = rate)",
      math.isclose(new_w, 0.5 + expected_ltp_delta, rel_tol=1e-9))

# ── the parts that were already correct, re-confirmed so they don't regress ──
syn2 = make_synapse(0.5)
new_w2 = learner.update(syn2, dt=5.0)
check("dt>0 (pre before post) still strengthens (LTP)", new_w2 > 0.5)

syn3 = make_synapse(0.5)
new_w3 = learner.update(syn3, dt=-5.0)
check("dt<0 (pre after post) still weakens (LTD)", new_w3 < 0.5)

# magnitude peaks at dt=0 and decreases with |dt|, in both directions
syn_near = make_synapse(0.5)
syn_far = make_synapse(0.5)
w_near = learner.update(syn_near, dt=1.0)
w_far = learner.update(syn_far, dt=10.0)
check("LTP magnitude is still greatest at the shortest interval (real formula shape preserved)",
      (w_near - 0.5) > (w_far - 0.5))

syn_near_neg = make_synapse(0.5)
syn_far_neg = make_synapse(0.5)
w_near_neg = learner.update(syn_near_neg, dt=-1.0)
w_far_neg = learner.update(syn_far_neg, dt=-10.0)
check("LTD magnitude is still greatest at the shortest interval on the negative side too",
      (0.5 - w_near_neg) > (0.5 - w_far_neg))

# LTD is still asymmetric (half-strength) relative to LTP at the same |dt|
syn_ltp = make_synapse(0.5)
syn_ltd = make_synapse(0.5)
w_ltp = learner.update(syn_ltp, dt=3.0)
w_ltd = learner.update(syn_ltd, dt=-3.0)
ltp_delta = w_ltp - 0.5
ltd_delta = 0.5 - w_ltd
check("LTD is still half-strength relative to LTP at the same |dt| (asymmetry preserved)",
      math.isclose(ltd_delta, ltp_delta * 0.5, rel_tol=1e-9))

# weight clamping still holds at the boundaries
syn_hi = make_synapse(0.999)
w_hi = learner.update(syn_hi, dt=0.0)
check("weight still clamps at 1.0, doesn't exceed it even with the dt=0 fix",
      w_hi <= 1.0)

syn_lo = make_synapse(0.001)
w_lo = learner.update(syn_lo, dt=-0.001)
check("weight still clamps at 0.0, doesn't go negative",
      w_lo >= 0.0)

print(f"\n=== RESULT: {PASS} passed, {FAIL} failed, {PASS + FAIL} total ===")
print("OVERALL:", "PASS" if FAIL == 0 else "FAIL")
sys.exit(0 if FAIL == 0 else 1)
