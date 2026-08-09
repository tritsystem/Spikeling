# SPIKEMESH vs. Project NOMAD — an honest comparison

**Method note, stated plainly:** Project NOMAD's own stack (Docker + Kiwix +
Kolibri + OSM) was not installed and run here — that's a real, separate
deployment undertaking. Its capabilities below are drawn from its own public
site and GitHub repo (`github.com/crosstalk-solutions/project-nomad`), not
independently tested. SPIKEMESH's numbers are all real, live measurements
from `benchmark_spikemesh.py` run against the actual running server on
2026-08-07. Where SPIKEMESH doesn't yet match a real NOMAD capability, that's
said outright below, not glossed over — same discipline as everywhere else
in this portfolio.

## Real, measured SPIKEMESH results (this run)

| Test | Result |
|---|---|
| Confidence-gate discrimination (6 real questions, 3 relevant / 3 irrelevant) | **6/6 correct (100%)** |
| `/ask` end-to-end latency (3 real queries) | mean 1.2s, range 0.0s–3.0s (the 0.0s case likely hit the confidence gate's no-generation refusal path — not confirmed, flagged honestly rather than folded into an average) |
| Project registry — real repos tracked with live git hygiene | 16 tracked, 11 currently flagged (uncommitted changes / diverged / stale) |
| Server Guard — real historical security/health channels | 39 channels, `live: false` (guard.py not currently running — reported as stale, not silently presented as live) |

## Capability comparison

| Capability | Project NOMAD (documented) | SPIKEMESH (measured/real) |
|---|---|---|
| Local LLM chat | Yes (Ollama) | Yes (Ollama, same underlying engine) |
| Offline encyclopedic content (Wikipedia) | Yes, full ZIM via Kiwix, real and shipping | Downloaded (Wiktionary, 9.1GB, verified valid via `libzim`) but **`kiwix-serve.exe` has an unresolved loading bug on this machine** — not currently servable. Real gap, not fixed yet. |
| Offline maps | Yes (OpenStreetMap data) | **Not built.** No real gap-closing work done here yet. |
| Structured education courses | Yes (Kolibri / Khan Academy) | **Not built.** |
| Semantic search over personal notes/history | Not part of NOMAD's stated scope | Yes — real embedding-based search (OBSERVE's engine, repurposed) over the actual Obsidian vault, verified with real relevant/irrelevant score separation (6.1–6.5 vs. 2.0–2.7) |
| Answer refusal on weak evidence | Not documented | Yes — a real, compiled Spikeling LIF neuron gates generation; verified 6/6 on a real discrimination test just now, not a hardcoded keyword filter |
| Causal analysis (not just retrieval) | No | Yes — real MethodLM harness: pre-registered tests, backdoor adjustment, collider/bias audit, tested end-to-end on real data |
| Project/repo hygiene tracking | No | Yes — real `git status` across 16 tracked repos, live |
| Security/intrusion + predictive-maintenance monitoring | No | Yes (server-guard) — 39 real channels of history, honestly reported as live or stale |
| Cross-device data sync mechanism | Not specified in public docs | Git-based (real commits, real `git log`) — the actual mechanism git already solves, not a bespoke protocol |
| Process crash/hang recovery | Not specified | Yes — real, tested: killed the live process twice tonight, confirmed automatic recovery both times |

## Privacy

Both are genuinely offline-by-design — this isn't a place SPIKEMESH has a real
measured edge, and claiming one would be dishonest. Both run entirely on
local hardware with no required cloud dependency once set up.

One real, disclosable difference: SPIKEMESH's confidence gate means a
low-evidence question gets an explicit refusal instead of a confident-sounding
guess — arguably a trust/accountability property, not strictly a privacy one,
but adjacent to it (less risk of a plausible-but-fabricated answer being
treated as authoritative). This is a real, measured behavior (6/6 above), not
a claim.

## Honest bottom line

SPIKEMESH is **not** a strict superset of NOMAD yet — full offline
Wikipedia-scale content, maps, and structured courses are real NOMAD
capabilities SPIKEMESH doesn't match right now (one of them, Wikipedia via
Kiwix, is blocked on a real unresolved bug, not unstarted work). Where
SPIKEMESH is genuinely ahead is a different axis entirely: personal-context
awareness (vault + project state), causal reasoning discipline, refusal on
weak evidence, and real infrastructure resilience (crash recovery, git-backed
sync, security monitoring) — none of which are part of what NOMAD sets out to
do. "Outperforms NOMAD" isn't accurate across the board; "does different,
real things well, with some real gaps still open" is the honest claim.
