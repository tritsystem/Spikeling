#!/usr/bin/env python
"""
build_environment_skills.py — one-time generator for 50 real, environment-
tailored SKILL.md files under spike_skills/, one per real project/finding/
discipline in this user's actual portfolio (drawn from the persistent
cross-session memory index, not invented).

IMPORTANT, stated honestly rather than left implicit: these 50 skills are
NOT wired into spike_skills.select_skills_for()'s automatic per-neuron
selection. That mechanism (see corrector-discipline, review-discipline,
pre-registration-discipline) is deliberately role-scoped -- a skill fires
on EVERY invocation of the neuron it names, which is correct for "how
Reviewer should behave" but wrong for "what's true about the Tribe
terrain system": gluing all 50 topic playbooks onto every Implementer
call regardless of what it's actually working on would bloat every
prompt with 49 irrelevant projects. None of these 50 descriptions name a
fixed-roster neuron (Implementer/Reviewer/PreRegister/Corrector/
TestWriter/VaultLogger/Clarifier) for exactly this reason -- verified
below and in test_spike_skills.py.

What this DOES give Spike: a real, listable, loadable library
(load_skill_listing() / load_skill_content()) of this user's actual
established context per project -- available to load explicitly by name,
or for a future task-text-matching selection phase (deliberately not
built here -- a naive keyword matcher across 50 diverse real topics
would be more likely to mismatch than help without real testing against
real task text, and that's a separate, riskier piece of work than this
generator).
"""

import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_skills")

# Each entry: (slug, one-line trigger description, body markdown).
# Real content, drawn from the actual persistent memory index for this
# user -- not generic filler. Every description avoids the 7 reserved
# fixed-roster neuron names on purpose (see module docstring).
SKILLS = [
    ("spikeling-runtime-core",
     "Use when working on Spikeling's core LIF/STDP runtime (core/runtime/runtime.py, pyspike.py) or .spk pipeline files.",
     """# Spikeling runtime core

Real, load-bearing fact: STDP exists ONLY in `core/runtime/runtime.py`,
NOT in `pyspike.py` (confirmed by direct grep -- zero STDP hooks there).
Don't assume plasticity is available just because a `.spk` file runs;
check which engine is actually executing it.

Documented STDP gotchas, found by testing, not assumed:
- STDP relaxes toward baseline instead of growing under asymmetric
  firing rates -- it doesn't monotonically strengthen just because two
  neurons co-fire often.
- Simultaneous-fire (dt = 0) lands in the LTD (weakening) branch, not
  LTP. A dt-of-zero case is easy to miss in test coverage and will
  silently do the opposite of what's expected.

Known gap: the LIF-cascade orchestrator (`spiking_orchestrator.py`) has
a promotion/structural-learning path (dynamic specialists get promoted
into the fixed roster) but no demotion path -- a promoted specialist
that stops being useful stays promoted."""),

    ("spiking-orchestrator-routing",
     "Use when modifying spiking_orchestrator.py's neuron cascade, tool-tier routing, or specialist spawning.",
     """# Spiking orchestrator routing

This is the real, verified LIF-neuron-cascade agent router --
`test_pyspike_orchestrator_parity.py` (10/10) proves the pyspike-built
brain matches the original .spk-parsed brain exactly, including the
correction loop and refractory behavior. Re-run it after ANY change
here, not just changes that look related.

Tool-tier resolution is config-driven via `spike_tool_gateway.py` +
`spike_tools.yaml` (4 tiers: review / research_review / research /
code) -- config is the sole source of truth, "auto" is a hard error, no
ambient-credential fallback. Don't reintroduce implicit auto-detection;
that's the exact failure class this replaced.

Real ordering gotcha found by testing: handlers run synchronously as
neurons fire, and the runtime runs a fired neuron's action BEFORE
propagating its spike -- so a downstream neuron firing from the SAME
cascade as its own trigger can hit that trigger's still-refractory-
locked state and silently skip a fire that should happen (found: a
dynamically-spawned specialist's synapse into Reviewer tried to fire it
at the same instant Reviewer was still refractory from firing moments
earlier, so the review silently never happened twice when it should
have)."""),

    ("observe-ternary-search",
     "Use when working on OBSERVE / 012-ternary's semantic code search, ternary compression, or the trit_mcp_server.py MCP server.",
     """# OBSERVE / 012-ternary

Semantic code search engine with ternary compression, a chunk
provenance/lineage layer (chunk_provenance.py), hybrid search, an
incremental indexer, and an entanglement (cross-project relationship)
database.

Workaround for when `mcp__observe__*` tools aren't loaded in a session:
call `trit_mcp_server.py` directly via
`docker run --entrypoint python ...` -- confirmed working, has found
real bugs this way, not just a theoretical fallback.

Before claiming something is a new finding anywhere in this portfolio,
run `research find "<topic>"` (the research-index CLI) against the whole
memory folder first -- it exists specifically to stop re-deriving a
finding by hand that's already been made and recorded."""),

    ("mcp-gateway-shared-package",
     "Use when extending audit logging or rate limiting to a new MCP server, or modifying mcp-gateway's audit_log.py / rate_limiter.py / wrap.py.",
     """# mcp-gateway

Shared audit-log/rate-limit package used by OBSERVE's and Spikeling's
MCP servers. The real point of this package existing at all: gating
(audit logging, rate limiting) should be ONE shared implementation
wired into every MCP server in this portfolio, not reimplemented per
repo. Before writing gating logic for a new MCP server, check whether
mcp-gateway already has the primitive rather than rebuilding it
locally."""),

    ("methodlm-causal-harness",
     "Use when working on methodlm's CORR/RUN/STRAT/ADJUST/INTERACT/REFUTE causal-reasoning tool set or its pre-registration ledger.",
     """# methodlm

Verifiable causal-reasoning harness: CORR (correlate), RUN (execute),
STRAT (stratify), ADJUST (control for confounds), INTERACT (test
interaction effects), REFUTE (actively try to falsify the finding) --
a real tool set, not a metaphor.

methodlm is post-hoc-only, not for experiment design -- this was an
explicit correction made mid-session when it got misapplied; don't
reach for it to design a study, only to analyze results after the fact.

Backend selection matters: local Qwen-3B vs the Claude API is a real,
distinct choice per task, not an implementation detail to gloss over.

The Cinelli-Hazlett RV was computed for the SSH edge/bulk finding
(topological-phononics) at 0.957, robust -- a real cross-substrate
bridge methodlm produced, not a hypothetical use case."""),

    ("tribe-persistence-diplomacy",
     "Use when working on Tribe's save/offline-catchup, alliances, proximity trade, or seasons systems.",
     """# Tribe persistence & diplomacy

Godot 4 trust/Spikeling-brain NPC sim, public at
github.com/tritsystem/tribe (own scoped repo, separate from the
profile-root repo -- don't confuse the two when pushing).

Real systems: save + offline-catch-up ("leave for a week" and the world
advances coherently), alliances, proximity-based trade, and a
deliberate no-forced-winner + seasons design (the game doesn't railroad
toward one tribe "winning").

Trade specifically runs through physical courier NPCs
(`trade_envoy.gd`), not an abstract instant exchange -- the player sends
offers with the `'` key and gets a real accept/decline panel
(`TribeTradeUI`) for incoming requests. This was built via the agent
pipeline then independently re-verified headless -- don't assume agent-
built Tribe systems are unverified just because they weren't hand-
written."""),

    ("tribe-terrain-generation",
     "Use when working on Tribe's terrain_gen.gd -- heightmaps, biomes, terraforming, or island mode.",
     """# Tribe terrain system

`terrain_gen.gd` real feature set: heightmap + collision generation,
biomes, live terraforming, adaptive RES (resolution scales with need),
walkable floor-snap, and a built ISLAND MODE that's off by default with
spawn-wiring still pending -- don't assume island mode is live just
because the code exists; check the default flag.

Turtle-island (a SEPARATE, now-fully-removed feature) is not this --
`turtle_island.gd`, `player_island.gd`, `troll.gd` and their tests were
deleted in a surgical removal pass (commit 496d212) that specifically
preserved `water_crossing.gd` as an unrelated general archipelago
feature. If old context mentions turtle islands, it's stale."""),

    ("tribe-performance-tuning",
     "Use when Tribe (or any Godot project) needs performance work -- profiling before changing anything.",
     """# Tribe performance discipline

The real, hard-won lesson: profile physics-frame ms FIRST, always.
Guessed wrong 4 times before profiling actually found the real cause at
Epic-scale lag.

What actually fixed it, in the real order found: LOD active-body count,
a hive-brain pattern (shared computation instead of per-NPC), LOW
graphics preset specifically for shadows-off, MultiMesh for trees
(instancing instead of individual nodes), and opaque water (transparency
was a real, non-obvious cost). None of these were the first guess --
that's the point: don't reach for a "likely" optimization without a
profile confirming it's the actual bottleneck."""),

    ("tribe-npc-core-memory",
     "Use when working on Tribe's SSH edge/bulk NPC memory (tribe_npc_core_memory.gd) or the trust/blame dialogue system.",
     """# Tribe NPC core memory

A real research result (the SSH reservoir edge-vs-bulk recall finding
from topological-phononics) wired into an actual gameplay feature, not
just a sim result left in a paper: betrayals recall reliably under NPC
panic states, petty grudges don't -- and this asymmetry is gated live
into the dialogue system's blame line, so it's an observable in-game
behavior difference, not a backend-only property.

This is one of the clearer cross-substrate bridges in this portfolio --
worth checking `topological_phononics_finding.md` and
`tribe_npc_core_memory.md` together before assuming a memory-recall bug
is Tribe-specific rather than inherited from the underlying reservoir
property."""),

    ("tribe-emergent-sync",
     "Use when working on Tribe's Kuramoto phase-lock system (TribeDrums, tribemember.gd sync) or debugging NPCs that seem to coordinate without direct communication.",
     """# Tribe emergent sync

Real, measured result: independent NPC brains phase-lock (Kuramoto
order parameter r: 0.48 -> 0.94) purely from shared drum-audio feedback
-- no direct NPC-to-NPC communication channel exists. If NPCs appear
to be coordinating, check whether they're actually synchronizing via
this audio-feedback mechanism before assuming a scripted coordination
bug.

This has a physical-hardware analog: `sync_mesh_finding.md` replicated
the same phase-lock mechanism on real microphones (2 separate devices,
6 repeated trials needed to see it reliably, ~10x smaller effect size
than the sim, gain-mismatch and single-trial-noise being the reusable
lessons for why a single trial can look like nothing happened)."""),

    ("tribe-direct-voice",
     "Use when working on Tribe's direct_voice.gd deterministic dialogue system or tribe_llm.gd's say_as_direct path.",
     """# Tribe DirectVoice

Deterministic, LLM-free "spikes-to-text" voice decoder --
`DirectVoice.compose_line(personality, trust, betrayed_count,
recall_confidence, described_memory)`. Real trust bands via
`trust_band()`: hostile / wary / neutral / warming / trusting, each
personality x band combination pulled from a real `PHRASE_BANK` (5
personalities x 5 bands).

This exists ALONGSIDE the Ollama-backed `say_as()` path in
`tribe_llm.gd`, not as a replacement for it -- `say_as_direct()` was
added, `say_as()` was left untouched. Verified 12/12 real checks both
before and after the later turtle-island removal, so it's a stable,
independently-tested piece even though the rest of the file changed
around it."""),

    ("reservoir-quasicrystal-finding",
     "Use when working on acoustic-vortex-sim's physical reservoir computing ladder or quasicrystal connectivity research.",
     """# Reservoir quasicrystal finding

Physical reservoir computing ladder in acoustic-vortex-sim, concluded
with a stress-tested even-order symmetry selection rule -- an edge
case, not a universal law (a von Karman variant was tried and declined
to fit the pattern). Treat this as a bounded, honestly-scoped result,
not "symmetry always predicts reservoir quality."

Connects to the ternary torus / Arnold-tongue finding: golden-ratio
incommensurate coupling is robustly non-parasitic, and that convergence
between two independently-derived results is worth citing together
when relevant, not treating as two unrelated findings."""),

    ("acoustic-reservoir-boundary",
     "Use when working on the acoustic reservoir / Levitator field simulation or evaluating trap-charge-parity claims.",
     """# Acoustic reservoir finding

The Levitator field is a nonlinear reservoir in simulation -- but a
specific, real boundary was found: the trap's charge-parity does NOT
govern computation. This is a negative result worth remembering
precisely -- if a future hypothesis assumes charge-parity drives
reservoir behavior in this system, that assumption was already tested
and falsified here."""),

    ("readout-vs-dynamics-finding",
     "Use when debugging reservoir-computing dead channels or evaluating whether an observable-orbit mechanism transfers between reservoirs.",
     """# Readout vs dynamics finding

A D4 reservoir's dead-channel problem splits into two genuinely
different causes: part is readout-fixable (an observation/measurement
problem -- the information is there but not being read out correctly)
and part is geometry-only (a real generation limitation, not fixable by
changing the readout). The QRC observable-orbit mechanism only
half-transfers between these -- don't assume a readout fix will resolve
a dead channel without checking which category it's actually in
first."""),

    ("ternary-torus-arnold-finding",
     "Use when working on ternary logic's topology-gate / polarity-sign structure or Arnold-tongue mode-locking analysis.",
     """# Ternary torus / Arnold tongue finding

Core structural claim: ternary = topology-gate x polarity-sign. On a
torus specifically, the parasitic/non-parasitic boundary IS Arnold-
tongue mode-locking -- and golden-ratio incommensurate coupling is
robustly non-parasitic. This converges with the independently-derived
quasicrystal finding (see reservoir-quasicrystal-finding) -- two
different routes landing on the same golden-ratio-incommensurate
result is the actual strength of this claim, cite both together."""),

    ("topological-phononics-ssh",
     "Use when working on topological-phononics' SSH reservoir, defect-tolerance experiments, or Fibonacci/quasicrystal connectivity research.",
     """# Topological phononics (SSH reservoir)

Published, public at github.com/gbranaa4-hue/topological-phononics.
Core finding, fully scoped: SSH reservoir defect-tolerance holds ONLY
when chiral symmetry holds, and REVERSES in a physical Duffing model --
don't generalize "topological protection helps" without checking which
symmetry regime and which physical model is actually in play.

Fibonacci/quasicrystal connectivity sub-thread: a 3-parameter recursive-
rule reservoir vs. O(N^2)-stored random connectivity, confound-hunted
across 9 scripts (a real construction-artifact bug was found and fixed
via verified prefix-nesting along the way). Final honest cost, smoothly
scale-characterized N=50-10000: ~1.4-2x for linear recall, ~1-17% for
NARMA10 -- a real but modest efficiency win, not reported as bigger than
it is."""),

    ("spikeling-phononic-bridge",
     "Use when checking whether a phononic-reservoir result transfers to Spikeling's real neuron models, or vice versa.",
     """# Spikeling/phononic bridge

Measured, not assumed: the noise/rank result (structured noise
cancelled exactly at readout) transfers cleanly from the phononic
reservoir sim to Spikeling's real Resonator neuron. The topological
defect-tolerance result does NOT transfer -- it's inconclusive/fragile
in both substrates when tested directly. Don't assume every phononic
finding generalizes to Spikeling just because both are reservoir-
adjacent; each transfer claim here was checked individually and they
didn't all hold."""),

    ("cross-substrate-synthesis-discipline",
     "Use when trying to unify a finding across ternary, acoustic, and phononic substrates, or when tempted to claim one law explains multiple results.",
     """# Cross-substrate synthesis discipline

A real, documented failure worth remembering precisely: an attempt to
unify phononic/ternary/acoustic results under one "symmetry-conditional
protection" rule was tried and LOST when actually measured against each
substrate's real code -- no single law held. What survived: a narrow,
real phononic ~ ternary winding-invariant rhyme (the invariant is
protected in both, but computation doesn't follow from that alone), and
acoustic turned out to be a genuinely separate nonlinearity-selection
mechanism that didn't even reproduce in the resonator bank.

The lesson to actually apply: a unifying claim across substrates needs
to be checked against each substrate's REAL code before being stated,
not inferred from the fact that the math looks similar on paper."""),

    ("consensus-scoping-ladder",
     "Use when extending 012-ternary's scoping-rule ladder or reasoning about rank-reordering behavior in ternary consensus.",
     """# Consensus scoping rule ladder

012-ternary scoping-rule ladder, completed with a rank-reordering test
(confirmed, not assumed). Open next candidates as of last check: a
Zenodo preprint write-up, and a coupled-Spikeling-reservoir follow-up
experiment -- check whether either has since been started before
re-proposing them as new ideas."""),

    ("horde-beta-perf-discipline",
     "Use when working in this repo (horde-beta-version-1) on team_id/animation/weapon-path logic or per-frame target-scanning.",
     """# Horde-beta perf & bug discipline

Real, already-fixed bugs in this exact repo: team_id, animation, and
weapon-path issues. Also a real O(n^2) per-frame target-scan
performance bug, profiled and fixed for roughly a 17x improvement --
if target-scanning logic changes again, re-profile rather than assuming
the old complexity class is still fine after the edit.

Cross-substrate note: this repo is also where Spikeling first got wired
into a game (horde-defense-beta), a first for that integration
direction -- Tribe had used Spikeling-brain NPCs already, but this was
the first time in the reverse direction of "bring Spikeling into an
existing game project" rather than building the game around it."""),

    ("breathing-modulation-discipline",
     "Use when evaluating threshold-adaptation strategies for spiking neurons -- blind periodic vs. closed-loop.",
     """# Breathing modulation finding

Real negative result: a blind periodic "breathing" threshold rhythm
LOSES to plain noise as a robustness strategy -- tested and falsified,
not just theorized against. The principled follow-up that actually won:
homeostatic, closed-loop threshold adaptation, which decisively beat
both blind breathing and noise -- deterministic recovery, and a
tighter worst-case bound than either alternative. If a future threshold-
adaptation idea is a variant of blind periodic modulation, it's already
been tried and lost; the closed-loop approach is the one with a real
track record here."""),

    ("agent-pipeline-package",
     "Use when working on the universal agent-pipeline + settings-UI package built for distribution (e.g. to family/other users).",
     """# Agent pipeline package

Universal agent-pipeline + HTML settings UI, packaged for the user's
brother as a standalone deliverable (not just an internal tool). A real
bug was found here that's worth remembering as a class, not just a
one-off: a querySelector collision that LOOKED like a caching bug at
first glance but wasn't -- worth checking selector uniqueness before
assuming a stale-cache explanation for UI state that won't update."""),

    ("pond-health-spiking-detector",
     "Use when working on pond-health's spiking-neuron anomaly detector or comparing it against the trend-based detector.",
     """# Pond-health spiking detector

Spikeling's real LIF-neuron engine wired in as a SECOND anomaly
detector alongside a linear-trend detector, honestly measured against
each other, not just added for coverage's sake. Real, asymmetric
result: the spiking detector never false-alarms, but also gives no
early warning -- the trend detector does exactly the opposite trade-off.
Neither one dominates; they're complementary by design, not one being
strictly better. Don't drop either detector without losing a real
capability the other doesn't have."""),

    ("sensor-duo-toolkit",
     "Use when a new project needs the trend+spiking+SQLite+Grafana sensor-monitoring pattern.",
     """# Sensor-duo toolkit

Pond-health's trend+spiking+SQLite+Grafana pattern, extracted into a
reusable public pip package, generic over any named channel (not
pond-specific). This is the canonical place to reach for this pattern
next time a monitoring problem shows up -- check whether sensor-duo
already covers the need before writing a new one-off monitoring
script."""),

    ("home-hub-kasa-monitoring",
     "Use when working on home-hub or real Kasa/Tapo smart-plug power monitoring and auto-cutoff.",
     """# Home-hub + Kasa power monitoring

Sensor-duo's pattern applied to real Kasa/Tapo smart-plug power
monitoring, including auto-cutoff actuation (not just passive
monitoring -- it can act). Repo is local-only. As of the last check,
the actual hardware was not yet owned -- verify current hardware status
before assuming this is running against real plugs rather than still
being developed against mocks/specs."""),

    ("ternary-vision-negative-result",
     "Use when evaluating whether to ternary-quantize a vision model, or building the fridge-cam app.",
     """# Ternary vision negative result

Real, honestly-reported negative result: ternary-quantizing
MobileNetV3-Small genuinely fails -- accuracy collapsed 78.8% -> 4.9%,
confirmed across 2 separate attempts, not a fluke. There IS a real win
(13.8x disk-size reduction) but ZERO speedup, confirmed. Given this,
the fridge-cam app's actual recommendation is to just use an off-the-
shelf Ollama vision model rather than pursuing ternary quantization
further for this use case -- don't re-attempt ternary vision
quantization for fridge-cam without a genuinely new angle, this one was
tested and closed."""),

    ("scrapyard-sensor-repurposing",
     "Use when working on the scrapyard-sensor-project turning thrift-store hardware into new sensor/monitoring tech.",
     """# Scrapyard sensor project

Turns real thrift-store hardware (a Nighthawk router, an HP printer,
etc.) into new tech, explicitly built on TOP of existing repos --
sensor-duo, pond-health, home-hub -- rather than starting fresh per
device. Before adding a new capability here, check whether it's really
a new sensor-duo channel rather than a new one-off script."""),

    ("spikeling-hardware-mcp",
     "Use when running parametric real-hardware experiments through Spikeling's SensorAdapter/BaselineDeviation framework or mcp_server.py.",
     """# Spikeling hardware MCP framework

SensorAdapter/BaselineDeviation hardware layer with a real/simulated
split that matters: acoustic, system_telemetry, and video adapters are
REAL; emg and environmental adapters are SIMULATED. Don't treat an emg
or environmental reading as physically measured without checking which
mode it's actually running in.

Exposed as an MCP server (mcp_server.py) specifically so a new one-off
hardware experiment script doesn't need to be written from scratch --
reach for this framework first."""),

    ("noise-cancelling-frontend",
     "Use when working on real 2-mic structured-noise cancellation experiments (the noise_cancelling_frontend finding).",
     """# Noise-cancelling frontend finding

First PHYSICAL test (real 2-mic C922, not just simulation) of the
reservoir "structured noise nulled at readout" sim result. Confirmed
and replicated twice, but only after catching two real bugs along the
way: a verdict-printer that lied about its own results, and a spectral-
overlap confound. The real, load-bearing condition for this to work:
selective cancellation only succeeds when the target signal is
spectrally separated from the interferer -- don't expect this to work
if the two signals overlap in frequency."""),

    ("sync-mesh-hardware-finding",
     "Use when replicating Tribe's emergent-sync mechanism on real microphones or evaluating phase-lock effect sizes across devices.",
     """# Sync-mesh finding (hardware)

First physical test of Tribe's emergent-sync mechanism (phase-lock via
shared exposure, zero direct communication channel between devices).
Strong and replicated twice on one device; real but roughly 10x SMALLER
across 2 separate microphones. Only visible via 6 repeated trials plus
lag correction -- a single trial can look like nothing happened. Gain-
mismatch between devices and single-trial noise are the reusable
lessons: don't judge a phase-lock replication attempt from one trial or
without checking gain calibration first."""),

    ("tdoa-localization-negative-result",
     "Use when attempting sound-source localization via time-difference-of-arrival with the 2-mic sync-mesh setup.",
     """# TDOA localization negative result

Honest open negative: using the same 2-mic sync-mesh setup to compute
sound-source position via TDOA landed at coin-flip accuracy (2/4) even
AFTER fixing 3 real bugs (search window too wide, wrong correlation
method for transient signals, unconstrained peak-matching). Whether the
remaining cause is room acoustics, positioning-consistency, or a still-
undiscovered bug was NOT distinguished -- don't present this as solved
or as a specific known root cause; it's a genuinely open negative
result."""),

    ("live-vision-feedback-glasses",
     "Use when working on the video-glasses prototype (worn webcam -> scene description) or evaluating claims about standing background vision.",
     """# Live vision feedback finding

Confirmed working: a worn/moved C922 webcam captures a frame, and
Claude actually describes real scene content from it (not just motion
numbers). This is turn-based ("look now") -- there is NO standing
background watcher. Don't describe this system as continuously
monitoring; every description happens on an explicit trigger."""),

    ("wireless-glasses-voice-loop",
     "Use when working on the wireless AR-glasses hardware design or the PC-side voice loop (TTS/STT).",
     """# Wireless glasses + voice loop

Full wireless AR-glasses design with a real, priced ~$166 parts list.
PC-side WiFi code is written but UNTESTED end-to-end. Voice loop TTS is
confirmed working, with two real gotchas: profile the voice by ID, not
by name, and Piper is unsupported in this setup. STT is currently
blocked on a Whisper model download inside Voicebox -- check whether
that download has completed before assuming STT is functional."""),

    ("water-intention-study-protocol",
     "Use when working on the water-intention study testing the Emoto water-crystal claim.",
     """# Water-intention study

Rigorous, pre-registered protocol testing the Emoto water-crystal
claim. Analysis pipeline validated via a real Monte Carlo simulation
(N=100/arm locked in) BEFORE any real water was touched -- the
simulation validation came first, on purpose, so the analysis method
itself is trustworthy before it's ever run on real data. As of the last
check, no real water had been tested yet -- verify current study status
before assuming results exist. Note: methodlm is post-hoc-only and was
explicitly clarified as NOT the tool for designing this experiment."""),

    ("oscillator-memory-finding",
     "Use when working on Kuramoto phase-coupled associative memory or comparing it against discrete Hopfield networks.",
     """# Oscillator-memory finding

Kuramoto phase-coupled associative memory, with 3 real bugs found and
fixed during development. Honest result, leaning negative: smaller
basin of attraction and lower capacity than discrete Hopfield memory,
measured directly, not assumed. Coupling-gain was left untested and is
the likely next lever to try -- don't treat this result as final until
that's been explored."""),

    ("spectral-hole-burning-check",
     "Use when fact-checking a pasted claim about Pr:YSO spectral storage or similar exotic-medium storage blueprints.",
     """# Spectral hole burning reality check

A real discipline example: a pasted "Spectral Vault" Pr:YSO storage
blueprint was checked against actual physics, not taken at face value.
Found: a wrong wavelength claim, and a 260TB capacity figure shown to
be disconnected from the real physics regardless of the wavelength
error. Two real simulation bugs were also found and fixed while
checking it. The reusable lesson: check pasted technical claims against
real, run numbers before repeating them, even when they sound
plausible."""),

    ("neuromorphic-survey-fugu",
     "Use when comparing Spikeling's runtime against other neuromorphic frameworks (Fugu, Lava, Otters).",
     """# Neuromorphic survey: Fugu comparison

Real, code-level comparison of Sandia's Fugu against Spikeling's
runtime. Concrete finding: Spikeling has NO synaptic delay primitive --
a real, specific gap, not a vague "could be more featureful" note.
Status of alternatives as of the last check: Lava SDK is archived: Otters
(ICLR26) code is only available via a fragile anonymous link. Verify
these statuses haven't changed before citing them as current."""),

    ("ternary-silicon-tapeout",
     "Use when working on the ternary-silicon-tapeout project's Verilog, TinyTapeout/SkyWater submission, or balanced-ternary logic design.",
     """# Ternary silicon tapeout

Real balanced-ternary half-adder built in Verilog
(`ternary_half_adder.v` + testbench), 2-bit-per-trit encoding (10 = -1,
00 = 0, 01 = +1, 11 = invalid), verified 16/16 via Icarus Verilog
simulation -- not just written, actually simulated and checked.

Scope boundary that matters: Tier 1 (balanced ternary logic in standard
CMOS) is achievable at accessible budget via TinyTapeout/SkyWater SKY130.
Tier 2 (true multi-valued voltage-level ternary requiring CNTFETs or
memristors) is NOT achievable at accessible budget -- don't scope new
tapeout work into Tier 2 territory without flagging that constraint
explicitly."""),

    ("fpga-open-toolchain",
     "Use when synthesizing or simulating designs with the open-source FPGA toolchain (Yosys/nextpnr-ice40/IceStorm).",
     """# FPGA open toolchain

Yosys/nextpnr-ice40/IceStorm (OSS CAD Suite) -- fully installed and
verified working for this portfolio's open-source ASIC/FPGA work.
Icarus Verilog is the simulation tool actually used for testbench
verification (see ternary-silicon-tapeout's 16/16 half-adder result) --
reach for this real, working toolchain rather than assuming a new
install is needed."""),

    ("iphone-hud-overlay-app",
     "Use when working on the real iPhone HUD glasses-software app (HOLD-TO-SCAN camera ID, Work Mode chat, server.py).",
     """# iPhone HUD overlay app

THE real "glasses software" for this portfolio -- HOLD-TO-SCAN camera
ID plus a Work Mode chat interface, served over HTTPS on port 8443 via
server.py. Confirmed working end-to-end.

Do NOT confuse this with `glasses_display_server.py`, which runs on a
DIFFERENT port (5759) and is a separate thing -- this mixup has
happened before and is worth actively checking against when either
system comes up."""),

    ("arduino-spikeling-sensor-grid",
     "Use when working on the physical Arduino + Spikeling sensor-grid hardware build.",
     """# Arduino + Spikeling sensor grid

CONFIRMED WORKING physical build: a 5x HC-SR04 ultrasonic sensor grid
on an Arduino Uno R3, driving a REAL Spikeling LIF network, with per-
zone audio feedback. This was the first physical hardware build in this
entire body of work -- a meaningful milestone, not just another sim.
When extending this, keep in mind it's real hardware with real timing/
noise constraints that a pure simulation wouldn't surface."""),

    ("claude-insights-jetbrains-plugin",
     "Use when working on the Claude Insights JetBrains/PyCharm plugin.",
     """# Claude Insights JetBrains plugin

Real, working PyCharm plugin, confirmed end-to-end (right-click code ->
inline Claude review). Two real Windows-specific bugs were found and
fixed: GUI-app PATH resolution (a GUI-launched process doesn't inherit
the same PATH a terminal session would), and cmd.exe mangling multi-
line CLI arguments -- fixed by passing the payload via stdin instead of
as a command-line argument. Separately, computer-use hit a real,
persistent PyCharm rendering wall during testing -- not every
verification path works for every IDE; that limitation is real, not a
one-off flake."""),

    ("research-index-tool",
     "Use before claiming any finding across this portfolio is new, or before re-deriving something by hand.",
     """# Research index tool

`research find "<topic>"` -- a real CLI that searches the entire
persistent memory folder. Use this BEFORE claiming something is a new
finding, and before re-deriving a result by hand that may already exist
recorded somewhere in this portfolio's history. This is a discipline
step, not an optional nicety -- skipping it risks presenting an old,
already-tested result as new."""),

    ("observe-docker-workaround",
     "Use when mcp__observe__* tools aren't loaded in a session but OBSERVE's search/query capability is still needed.",
     """# OBSERVE Docker CLI workaround

When the `mcp__observe__*` tools aren't loaded, call
`trit_mcp_server.py` directly via
`docker run --entrypoint python trit_mcp_server.py ...` instead of
giving up on OBSERVE's capability for that session. Confirmed working,
and has found real bugs this way -- a genuine fallback, not a
degraded/theoretical one."""),

    ("multi-repo-git-discipline",
     "Use before any push, force-push, or other destructive git action across this multi-repo portfolio.",
     """# Multi-repo git discipline

This portfolio spans many separate repos, and a past session got the
ahead/behind relationship backwards more than once before catching it.
The rule: check the repo's ACTUAL current state before any push/force/
destructive git action -- some repos have real uncommitted work, and a
couple have genuinely divergent remote history. Never assume "ahead" or
"behind" from memory or from how a similar repo in this portfolio
usually behaves; check this specific repo, this specific time."""),

    ("spike-vault-logging-discipline",
     "Use when finishing any project's work session -- logging to the user's real Obsidian Spike Memory Vault.",
     """# Spike vault logging discipline

Check and log to the user's REAL Obsidian "Spike Memory Vault" (under
Lessons / Project Work at Spikeling / vault) for ANY project's work in
this portfolio -- not just work that's literally inside the tribe
repo. This was an explicit correction: vault logging isn't tribe-
specific, it's a portfolio-wide discipline. Confirmed as the expected
behavior."""),

    ("spike-tool-gateway-config",
     "Use when adding a new tool category or provider to Spike's own tool-gateway config (spike_tools.yaml).",
     """# Spike tool gateway config discipline

`spike_tools.yaml` is the sole source of truth for tool-tier routing --
modeled directly on Nous Portal's real "config always wins over ambient
credentials" rule. Never add an "auto" value (hard-rejected at load
time by design) and never add an ambient-credential-detection fallback
path -- that reintroduces the exact failure class Hermes hit this
portfolio with (an implicit "auto" resolution silently falling through
to a stale hardcoded default nobody configured on purpose). A new
category is a real error to resolve until it's explicitly added here,
not a silent default."""),

    ("spike-account-rotation",
     "Use when working on spike.py's AccountRotator (dual-Anthropic-account rotation) or its real-run fallback behavior.",
     """# Spike account rotation

`AccountRotator` cycles across `SPIKE_ANTHROPIC_KEY_1/2` when set. Real
fixed bug worth remembering: an early version incorrectly REQUIRED
rotation keys to be set for ANY real (non-dry-run) run, silently
falling back to dry-run otherwise -- wrong, since the underlying
`spiking_orchestrator.py --real` flag works fine with just the
machine's already-logged-in Claude CLI session, no special keys needed.
Fixed so `real=True` without rotation keys still runs for real on
ambient auth; rotation is a strictly additive opt-in on top, never a
requirement."""),

    ("pc-compute-harnessing",
     "Use when a task on this machine is compute-heavy (training, large simulation, batch inference) and could use the GPU.",
     """# PC compute harnessing

This machine has an RTX 5060 with CUDA-capable PyTorch already
installed -- harness the GPU/CPU for compute-heavy work whenever viable
rather than defaulting to a slower CPU-only path or assuming no
accelerator is available. When speed and risk/quality trade off against
each other on this machine's work, the established default is to favor
quality, not the fastest shortcut."""),

    ("glasses-voice-input-discipline",
     "Use when a message may have arrived via voice/glasses (glasses_hook.py, in the Spikeling repo) and could be terse or garbled.",
     """# Glasses voice input discipline

Messages may arrive via voice/glasses through `glasses_hook.py` (in the
Spikeling repo) -- text from this path can arrive terse or garbled
compared to typed input. Confirm intent before big or risky actions
rather than guessing at unclear phrasing when a message looks like it
could be a mis-transcription rather than a deliberate short instruction.
This is a standing rule, not a one-off caution -- it applies any time
input plausibly came through this channel, not just when it's stated
explicitly."""),
]


# Real, distinctive trigger terms per skill -- proper nouns, file/path
# names, and specific jargon, deliberately NOT generic English words, to
# keep task-text matching (spike_skills.select_skills_for_task) precise.
# Some overlap between entries in the reservoir/topological-phononics
# cluster (12-19) is intentional -- they're a genuinely interconnected
# real research thread, not a keyword-choice mistake. See
# test_spike_skills.py for the automated precision/recall checks this
# list is verified against.
KEYWORDS = {
    "spikeling-runtime-core": "STDP, pyspike.py, core/runtime/runtime.py, LTD, LTP, plasticity",
    "spiking-orchestrator-routing": "spiking_orchestrator, neuron cascade, tool tier, spike_tools.yaml, refractory",
    "observe-ternary-search": "OBSERVE, 012-ternary, trit_mcp_server, ternary compression, chunk provenance, entanglement database",
    "mcp-gateway-shared-package": "mcp-gateway, audit_log.py, rate_limiter.py, wrap.py, audit logging, rate limiting",
    "methodlm-causal-harness": "methodlm, CORR/RUN/STRAT, ADJUST, INTERACT, REFUTE, causal reasoning harness",
    "tribe-persistence-diplomacy": "tribe_persist, world_tribe.gd, trade_envoy, offline catch-up, TribeTradeUI, proximity trade",
    "tribe-terrain-generation": "terrain_gen.gd, heightmap, island mode, terraforming, biomes",
    "tribe-performance-tuning": "tribe performance, physics-frame, hive brain, MultiMesh trees, LOD active-body",
    "tribe-npc-core-memory": "tribe_npc_core_memory, core memory recall, blame line, betrayal recall",
    "tribe-emergent-sync": "TribeDrums, Kuramoto, phase-lock, emergent sync, order parameter",
    "tribe-direct-voice": "direct_voice.gd, DirectVoice, PHRASE_BANK, trust_band, spikes-to-text",
    "reservoir-quasicrystal-finding": "quasicrystal, acoustic-vortex-sim, reservoir ladder, even-order symmetry",
    "acoustic-reservoir-boundary": "Levitator field, acoustic reservoir, charge-parity",
    "readout-vs-dynamics-finding": "D4 reservoir, dead channels, observable-orbit, QRC",
    "ternary-torus-arnold-finding": "Arnold tongue, mode-locking, ternary torus, topology-gate, polarity-sign",
    "topological-phononics-ssh": "SSH reservoir, topological-phononics, defect-tolerance, chiral symmetry, Duffing model",
    "spikeling-phononic-bridge": "Spikeling/phononic bridge, Resonator neuron, noise/rank result",
    "cross-substrate-synthesis-discipline": "cross-substrate synthesis, symmetry-conditional protection, winding-invariant",
    "consensus-scoping-ladder": "consensus scoping, scoping-rule ladder, rank-reordering test",
    "horde-beta-perf-discipline": "horde-beta, horde-defense-beta, team_id bug, weapon-path, target-scan",
    "breathing-modulation-discipline": "breathing modulation, homeostatic threshold, threshold adaptation",
    "agent-pipeline-package": "agent-pipeline package, settings UI, querySelector collision",
    "pond-health-spiking-detector": "pond-health, spiking anomaly detector, trend detector",
    "sensor-duo-toolkit": "sensor-duo, trend+spiking, Grafana dashboard, SQLite channel",
    "home-hub-kasa-monitoring": "home-hub, Kasa smart-plug, Tapo, auto-cutoff",
    "ternary-vision-negative-result": "MobileNetV3, ternary-quantiz, fridge-cam, vision quantization",
    "scrapyard-sensor-repurposing": "scrapyard-sensor-project, Nighthawk router, thrift-store hardware",
    "spikeling-hardware-mcp": "SensorAdapter, BaselineDeviation, mcp_server.py, hardware sensor-adapter",
    "noise-cancelling-frontend": "noise-cancelling frontend, structured noise, C922 mic, spectral overlap",
    "sync-mesh-hardware-finding": "sync-mesh, phase-lock hardware, gain-mismatch",
    "tdoa-localization-negative-result": "TDOA, time-difference-of-arrival, sound-source localization",
    "live-vision-feedback-glasses": "video glasses, live vision feedback, worn webcam",
    "wireless-glasses-voice-loop": "wireless glasses, AR-glasses, voice loop, Piper, Voicebox",
    "water-intention-study-protocol": "water-intention study, Emoto, water-crystal",
    "oscillator-memory-finding": "oscillator-memory, Kuramoto associative memory, discrete Hopfield",
    "spectral-hole-burning-check": "spectral hole burning, Pr:YSO, Spectral Vault",
    "neuromorphic-survey-fugu": "Fugu, neuromorphic survey, Lava SDK, Otters ICLR",
    "ternary-silicon-tapeout": "ternary-silicon-tapeout, TinyTapeout, SkyWater, SKY130, balanced ternary, half-adder",
    "fpga-open-toolchain": "Yosys, nextpnr-ice40, IceStorm, Icarus Verilog, OSS CAD Suite",
    "iphone-hud-overlay-app": "iPhone HUD, HOLD-TO-SCAN, Work Mode chat",
    "arduino-spikeling-sensor-grid": "Arduino Uno R3, HC-SR04, sensor grid",
    "claude-insights-jetbrains-plugin": "Claude Insights, JetBrains plugin, PyCharm plugin",
    "research-index-tool": "research find, research_index_tool",
    "observe-docker-workaround": "OBSERVE Docker, trit_mcp_server, docker run --entrypoint",
    "multi-repo-git-discipline": "force-push, divergent remote history, multi-repo git",
    "spike-vault-logging-discipline": "Spike Memory Vault, vault logging, Obsidian vault",
    "spike-tool-gateway-config": "spike_tools.yaml, spike_tool_gateway, config always wins",
    "spike-account-rotation": "AccountRotator, SPIKE_ANTHROPIC_KEY, account rotation",
    "pc-compute-harnessing": "RTX 5060, CUDA-capable PyTorch, GPU harnessing",
    "glasses-voice-input-discipline": "glasses_hook.py, voice/glasses input, terse or garbled",
}


def main():
    os.makedirs(ROOT, exist_ok=True)
    written = []
    for slug, description, body in SKILLS:
        skill_dir = os.path.join(ROOT, slug)
        os.makedirs(skill_dir, exist_ok=True)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        frontmatter = f"---\nname: {slug}\ndescription: {description}\n"
        kws = KEYWORDS.get(slug)
        if kws:
            frontmatter += f"keywords: {kws}\n"
        frontmatter += "---\n\n"
        content = frontmatter + body.strip() + "\n"
        with open(skill_md, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(slug)
    print(f"Wrote {len(written)} skill(s).")
    return written


if __name__ == "__main__":
    main()
