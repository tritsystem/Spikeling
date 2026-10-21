# Spikeling Resonator bank vs. NeuroBench's Mackey-Glass chaotic forecasting task

**Verdict: NEGATIVE.** Mean sMAPE 72.51 across the full official protocol (14
tau values × 30 repeats), vs. the published leaderboard's tuned LSTM (13.37)
and tuned ESN (14.79). Roughly 5x worse. Real mechanism identified, not just
a number — see below.

## What this is

Uses the REAL, official NeuroBench framework (`pip install neurobench`,
`NeuroBench/neurobench` on GitHub) — the real Mackey-Glass dataset (14 tau
variants, downloaded from the official HuggingFace mirror, not regenerated),
the real `Benchmark`/`TorchModel`/`SMAPE` harness, and the exact same
train/test/warmup protocol and ridge-regression readout code the official
`examples/mackey_glass/echo_state_network.py` baseline uses — so any accuracy
difference is attributable to the reservoir substrate, not a different
evaluation method.

`spikeling_resonator_reservoir.py` reuses the REAL `Resonator` class from
`Spikeling/resonator-prototype/resonator_bank.py` directly (a damped harmonic
oscillator neuron) as the reservoir unit: a bank of 200, log-spaced natural
frequencies (period 6–300 ticks), gain-normalized per that module's own
documented fix (`coupling = base_gain * omega^2`, otherwise high-frequency
units are invisible next to low-frequency ones — a real lesson that module
already learned, reused here rather than re-discovered).

## Results (full run, matching the leaderboard's own scale)

| tau | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sMAPE | 18.5 | 36.3 | 24.2 | 73.7 | 50.8 | 79.9 | 82.6 | 83.0 | 85.5 | 100.1 | 106.1 | 92.6 | 81.7 | 100.0 |

**Overall mean: 72.51.** Reference: LSTM 13.37, ESN 14.79 (both tuned via
random hyperparameter search over reservoir size / spectral radius / leakage
/ ridge param per tau — see `echo_state_network_hyperparameters.csv` in the
official repo).

Full per-run output: `full_run_output.log`.

## Why it lost — a real mechanism, not just "worse"

1. **No cross-unit connectivity.** Each resonator in the bank is driven
   independently by the same scalar input — there is no recurrent coupling
   *between* resonators, unlike a true ESN's randomly-connected recurrent
   matrix. Functionally this bank is closer to a fixed multi-channel
   band-pass filterbank (a spectrogram-like decomposition) than a genuine
   nonlinear dynamical reservoir with internal mixing. Reservoir-computing
   theory attributes much of an ESN's representational richness to exactly
   that recurrent mixing — its absence here is a real, structural gap, not a
   tuning gap.
2. **Performance degrades sharply as tau increases** (18.5 at tau=17, the
   "standard"/easiest case, up to 106 at tau=27) — consistent with #1: a
   filterbank-shaped reservoir with no internal mixing has a harder time
   tracking an attractor that gets more chaotic/higher-dimensional as tau
   grows, because it has no mechanism to combine information across its own
   channels the way a true reservoir's recurrence would.
3. **Genuinely untuned.** No search was run over reservoir size, damping,
   frequency range, or ridge parameter — all fixed defaults chosen from the
   symplectic-Euler stability bound in `resonator_bank.py`'s own comments,
   not from tuning against this task. The ESN baseline's 522–966-unit,
   per-tau-tuned reservoirs had a real advantage this comparison did not try
   to equalize.

## Honest limits

- **Not pre-registered.** This was run directly at request, not through this
  portfolio's usual pre-registration ritual for new research hypotheses —
  there is no `PREREGISTER.md` predating this run. Reported with full
  numbers and mechanism regardless, per the same "report negatives as
  carefully as positives" rule, but the process discipline itself is
  lighter-weight than a full research rung.
- **One architecture only.** Only the plain independent-oscillator bank was
  tried. A resonator bank WITH cross-unit coupling (closer to how a real ESN
  mixes channels) is the obvious next variant, not yet tested.
- **No tuning budget spent**, as stated above — this is a first honest
  number, not a ceiling on what a resonator-based reservoir could do here.

## Reproduce

```
pip install neurobench torch pandas jitcdde
python run_spikeling_benchmark.py 17,18,19,20,21,22,23,24,25,26,27,28,29,30 30
```

Mackey-Glass data auto-downloads from the official HuggingFace mirror on
first run (`NeuroBench/mackey_glass`) into `data/mackey_glass/`.

## Next steps, if pursued further

- Add real recurrent cross-coupling between resonators (each unit's `x`
  contributes to its neighbors' drive, not just the external input) and
  re-run the identical protocol — the most direct test of mechanism #1
  above.
- A real hyperparameter search (reservoir size, damping, frequency range,
  ridge param) matched in scope to the ESN baseline's, before claiming any
  tuned comparison.
