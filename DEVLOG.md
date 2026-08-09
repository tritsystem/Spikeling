# Spikeling Dev Log

## 2026-07-17 — Unit test: latency_spike_time value=0.0 edge case

Added `test_pyspike_encoding.py` with two tests for `latency_spike_time` in `pyspike_encoding.py`.

The primary test (`test_latency_spike_time_zero_never_fires`) covers the specified edge case: `value=0.0` must return exactly `n_steps` (100 in the test), which is the sentinel value the implementation uses to signal "no spike within the window." The assertion checks `result == n_steps` and `result >= n_steps`; both are needed because the contract is that any fire_time outside `[0, n_steps-1]` means no spike, and `n_steps` is the exact value the code returns.

A second test (`test_latency_spike_time_small_positive_fires_within_window`) was added as its complement: it confirms that every `value > 0.0` (tested at 1e-9, 0.01, 0.5, and 1.0) does produce a fire_time strictly inside the valid window. Without this complement the `value=0.0` test would pass vacuously if the function returned `n_steps` for all inputs.

The implementation at `pyspike_encoding.py:53-54` already handled this correctly (`if value <= 0.0: return n_steps`), so no production code changed — this is a test-only addition.

## 2026-07-17 — Docstring audit: pyspike_delay.py and pyspike_encoding.py

Reviewed both module-level docstrings and all public-function docstrings in `pyspike_delay.py` and `pyspike_encoding.py` against the code they describe.

**pyspike_delay.py — no changes made.** The module docstring accurately describes `DelayedNet` as a min-heap wrapper over a live `pyspike.Net`, with `(delivery_time, seq, dst, weight)` events, tie-breaking via a monotone `seq` counter, `.advance_to(t)` for flushing, and `.stimulate()` auto-advancing `now` by 1.0 per call. All claims match the implementation.

**pyspike_encoding.py — one function docstring corrected.** The module docstring was accurate (rate coding, latency coding, and the LIF-driver helper are all correctly described). However, the docstring for `drive_neuron_with_spike_train()` claimed "ONE stimulate() per tick (drive=spike_drive on a True tick, drive=0.0 -- just a leak tick -- on a False tick)" — implying `rt.stimulate(..., 0.0)` is called on silent ticks. The code actually calls `rt.tick()` on False ticks, not `rt.stimulate()` with zero drive. The docstring now reads "one call per tick: stimulate(drive=spike_drive) on a True tick, rt.tick() -- just a leak step -- on a False tick," which matches the implementation.

## 2026-07-17 — Review: pyspike_causal.py module docstring

Reviewed the module-level docstring (lines 2–35) of `pyspike_causal.py` for typos and unclear wording.

**Verdict: no changes made.** The docstring was already correct and clearly written. All technical claims (Lamport logical clocks, proper-time analogy, refractory-window behavioral divergence) are stated precisely and match the code below them. Punctuation, capitalization for emphasis, and the inline self-test comment were all intentional and correct.
