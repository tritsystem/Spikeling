#!/usr/bin/env python
"""
test_pyspike_encoding.py — unit tests for pyspike_encoding.py

    python test_pyspike_encoding.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyspike_encoding import latency_spike_time   # noqa: E402


def test_latency_spike_time_zero_never_fires() -> None:
    """value=0.0 must return n_steps, meaning no spike falls within [0, n_steps-1]."""
    n_steps = 100
    result = latency_spike_time(0.0, n_steps)
    assert result == n_steps, (
        f"latency_spike_time(0.0, {n_steps}) should return {n_steps} (never fires), "
        f"got {result}"
    )
    assert result >= n_steps, (
        f"fire_time {result} must be >= n_steps={n_steps} to be outside the valid window"
    )
    print(f"  [PASS] value=0.0 -> fire_time={result} (== n_steps={n_steps}, never fires within window)")


def test_latency_spike_time_small_positive_fires_within_window() -> None:
    """Any value > 0.0 must produce a fire_time strictly inside [0, n_steps-1].
    This is the complement to the value=0.0 edge case: confirms 0.0 is the
    only sentinel that suppresses the spike."""
    n_steps = 100
    for value in (1e-9, 0.01, 0.5, 1.0):
        result = latency_spike_time(value, n_steps)
        assert 0 <= result < n_steps, (
            f"latency_spike_time({value}, {n_steps}) should be in [0, {n_steps - 1}], "
            f"got {result}"
        )
    print(f"  [PASS] values > 0.0 all fire within [0, n_steps-1]")


if __name__ == "__main__":
    print("=" * 78)
    print("  PYSPIKE ENCODING -- unit tests")
    print("=" * 78)
    test_latency_spike_time_zero_never_fires()
    test_latency_spike_time_small_positive_fires_within_window()
