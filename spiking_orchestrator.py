#!/usr/bin/env python
"""
spiking_orchestrator.py — run the agent pipeline as a SPIKING NEURAL NETWORK.

The pipeline is no longer a fixed line of stages. It's the network in
core/examples/agent_brain.spk: every specialist agent is a NEURON that only
RUNS (spends tokens) when its membrane potential crosses threshold. A task is
scored into "sensory current"; the spikes that follow decide which agents fire.

    task text --score--> sensory neurons --spikes--> specialist neurons fire
                                                     --> that agent runs (tokens)

WHY THIS SAVES TOKENS (the whole point):
    A neuron below threshold never fires, so its agent never runs. A trivial
    task fires Implementer + Reviewer + Logger and nothing else — Clarifier,
    PreRegister, TestWriter, Corrector all stay dark and cost zero. A big
    ambiguous task lights the whole board. The saving is STRUCTURAL: it falls
    out of the LIF dynamics, it isn't an if-statement someone remembered to add.

WHY IT WON'T LOOP OR SHIP GARBAGE:
    refractory (in the .spk) stops an agent firing twice in one settle, so a
    correct->review->correct chain can't run away. Review gates the ledger:
    Corrector only fires when review's OUTPUT reports issues (result-driven
    stimulation, injected here — it depends on what the agent actually found).

Handlers run SYNCHRONOUSLY as neurons fire, and the runtime runs a fired
neuron's action BEFORE propagating its spike — so the flow self-sequences:
Implementer's agent finishes, THEN the spike drives Reviewer.

Run the demo (no API calls — proves the routing + the token savings):
    python spiking_orchestrator.py --demo
Run a real task through it (calls Claude via voice_commands.do_claude_code):
    python spiking_orchestrator.py "tribe: add a health bar to members"
"""

import json
import os
import re
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyspike import Net   # noqa: E402
from pyspike_causal import CausalNet   # noqa: E402

# PORTED TO PYSPIKE (2026-07-17): this used to parse core/examples/agent_brain.spk
# at every SpikingPipeline() construction. The topology is now built directly
# via pyspike.Net in _build_brain() below -- same benefits as the scheduler
# ports (no text round-trip) plus a real simplification: handlers attach
# straight to their neuron via @net.action() instead of going through the old
# neuron -> COMMAND STRING -> handler-dict indirection. Verified spike-for-
# spike identical `fired` sequences (including the Reviewer<->Corrector
# correction loop and refractory behavior) to the .spk-parsed version across
# the full demo suite before this replaced it (test_pyspike_orchestrator_parity.py).
#
# agent_brain.spk is KEPT in the repo as the human-readable topology
# reference and as the source for the C code generator / hardware backend --
# it is NOT read by this file anymore. If you change the topology, update
# BOTH _build_brain() below and agent_brain.spk, or they will drift; there is
# no automatic sync between them.
BRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "core", "examples", "agent_brain.spk")   # kept for reference/C-backend only

# Every real routing decision gets appended here (one JSON object per line).
# This is the training data for the eventual learned score_task() replacement
# -- score_task() is the one hand-tuned, unverified piece left in the whole
# pipeline (see spiking_agent_pipeline.md memory), and a classifier needs
# real task -> decision pairs to train against, not synthetic demo cases.
DECISIONS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "task_decisions.jsonl")

# neuron -> the action command the runtime fires when it spikes (mirror of the
# `action` lines in the .spk; kept here so the orchestrator can register a
# handler per command)
COMMANDS = {
    "Clarifier":   "CMD_RUN_CLARIFY",
    "PreRegister": "CMD_RUN_PREREG",
    "Implementer": "CMD_RUN_IMPLEMENT",
    "TestWriter":  "CMD_RUN_TESTS",
    "Reviewer":    "CMD_RUN_REVIEW",
    "Corrector":   "CMD_RUN_CORRECT",
    "VaultLogger": "CMD_RUN_LOG",
}


# ─────────────────────────────────────────────────────────────────────────────
# SCORING — task text -> sensory current (0..100 per sensory neuron)
#
# This is the load-bearing, honest-about-it part: something has to decide "how
# much does this task need each agent". The prototype uses transparent
# heuristics; swapping in a learned classifier later changes ONLY this function.
# ─────────────────────────────────────────────────────────────────────────────
def score_task(task: str, project: str = "spikeling") -> dict:
    t = task.lower()
    words = re.findall(r"\w+", t)
    n = len(words)

    # S_Work: is there real work? almost always yes for a non-empty task.
    work = 70 if n >= 2 else 20

    # S_Ambiguous: short, vague, or a bare question with no concrete target.
    vague_markers = ["something", "somehow", "maybe", "fix stuff", "make it better",
                     "improve", "clean up", "etc"]
    ambiguous = 0
    if n <= 4:
        ambiguous += 45
    if any(m in t for m in vague_markers):
        ambiguous += 40
    if not re.search(r"\b(add|remove|change|rename|fix|move|write|implement|create|"
                     r"delete|refactor|update|build)\b", t):
        ambiguous += 25   # no clear verb -> what am I actually doing?

    # S_Complex: long, multi-clause, or names systems/rewrites.
    complex_markers = ["system", "refactor", "rewrite", "pipeline", "architecture",
                       "multiple", "several", "across", "integrate", "and also",
                       "as well as"]
    cplx = 0
    if n >= 18:
        cplx += 40
    if t.count(" and ") >= 2:
        cplx += 25
    if any(m in t for m in complex_markers):
        cplx += 35

    # S_Tests: explicitly about tests / verification.
    tests = 70 if re.search(r"\b(test|tests|testing|verify|coverage|assert|regression)\b", t) else 10

    # S_Research: a research codebase, or method/measurement language.
    research = 65 if (project.lower() in RESEARCH_PROJECTS
                      or re.search(r"\b(measure|experiment|benchmark|reservoir|"
                                   r"hypothesis|pre-?register)\b", t)) else 5

    clip = lambda v: max(0, min(100, v))
    return {
        "S_Work": clip(work), "S_Ambiguous": clip(ambiguous), "S_Complex": clip(cplx),
        "S_Tests": clip(tests), "S_Research": clip(research),
    }


RESEARCH_PROJECTS = {"phononics", "ternary", "methodlm", "symmetry", "quasicrystal", "spikeling"}


# ─────────────────────────────────────────────────────────────────────────────
# TERNARY GATE — Reviewer output -> correct-or-not
#
# First real wiring of the trit-gate pattern validated in
# test_soft_conflicts.py's ternary_gated scheduler (see spiking_agent_pipeline.md
# memory). The old logic was a single flat keyword match -- any hit = correct,
# no hit = clean. That collapses "one nitpick" and "breaks the build" into the
# same decision. This scores a SEVERITY in [0,1] instead and bands it into
# hard_safe / ambiguous / hard_unsafe.
#
# HONEST NOTE, found by testing before trusting: the scheduler's ternary gate
# saved real cost by skipping an expensive SNN network build in the hard
# bands. The FIRST version of this gate copied that shape literally -- only
# checking escalation/hedge markers (critical, security, regression, minor,
# typo...) inside the ambiguous band -- and that was a bug, not an
# optimization: those markers are substring checks, exactly as cheap as the
# base issue-markers, so there was no real cost being saved. It also produced
# a false negative: "Found a critical security issue... this introduces a
# regression" scored severity=0.15 from "issue" alone, landed in hard_safe,
# and the escalation language was never even looked at. The domain has no
# real cost asymmetry to exploit (unlike the scheduler, where the ambiguous
# band gated actual expensive computation) -- so ALL markers are scored
# together, always, and bands are read off the combined severity.
# ─────────────────────────────────────────────────────────────────────────────
REVIEW_GATE_HI = 0.65   # severity at/above this: certain issues, correct
REVIEW_GATE_LO = 0.15   # severity at/below this: certain clean, skip

_STRONG_ISSUE_MARKERS = ["wrong", "fails", "incorrect", "breaks", "bug", "broken", "crash"]
_WEAK_ISSUE_MARKERS = ["missing", "overclaim", "issue", "does not", "doesn't", "should"]
_UPGRADE_MARKERS = ["critical", "major", "security", "data loss", "regression", "breaks the build"]
_DOWNGRADE_MARKERS = ["minor", "nit", "nitpick", "trivial", "typo", "style", "cosmetic"]


def _review_severity(review_output: str) -> float:
    low = review_output.lower()
    score = 0.0
    for m in _STRONG_ISSUE_MARKERS:
        if m in low:
            score += 0.30
    for m in _WEAK_ISSUE_MARKERS:
        if m in low:
            score += 0.15
    for m in _UPGRADE_MARKERS:
        if m in low:
            score += 0.35
    for m in _DOWNGRADE_MARKERS:
        if m in low:
            score -= 0.20
    return max(0.0, min(1.0, score))


def gate_review(review_output: str) -> dict:
    """Classify a review into a trit band and a correct/skip decision.
    Returns {decision, band, severity} -- logged so the gate's real-world
    calibration can be checked later, the same honest-benchmark discipline
    used on the scheduler (see spiking_agent_pipeline.md)."""
    severity = _review_severity(review_output)
    if severity >= REVIEW_GATE_HI:
        return {"decision": True, "band": "hard_unsafe", "severity": severity}
    if severity <= REVIEW_GATE_LO:
        return {"decision": False, "band": "hard_safe", "severity": severity}
    return {"decision": severity >= 0.5, "band": "ambiguous", "severity": severity}


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC SPECIALISTS -- the self-growing-network capability wired into the
# real pipeline (see test_self_growing_network.py, spiking_agent_pipeline.md).
#
# Today's topology is FIXED at construction: seven specialists, decided in
# advance. But an agent can discover mid-task that it needs a sub-specialist
# nobody anticipated. If it says so in its output, the pipeline spawns one
# LIVE -- a real neuron, wired in and reviewed like any other -- instead of
# either ignoring the request or forcing everything through Implementer.
#
# Convention: an agent signals this by including a line
#   NEEDS_SPECIALIST: <Name>: <what it should do>
# in its output. Deliberately simple and grep-able rather than structured,
# so it's easy for a real Claude Code call to produce reliably.
#
# GROWTH IS CAPPED, explicitly -- same lesson as DRIVE_FLOOR in the earlier
# scheduler hang: a live-growing mechanism with no stop condition is a real
# runaway risk, not hypothetical. At most MAX_DYNAMIC_SPECIALISTS new
# neurons per pipeline run, and never the same name twice.
# ─────────────────────────────────────────────────────────────────────────────
SPAWN_SPECIALIST_RE = re.compile(r"NEEDS_SPECIALIST:\s*(\w+):\s*(.+)")
MAX_DYNAMIC_SPECIALISTS = 2

# Tool-access tier for a dynamically-spawned specialist, picked from what its
# subtask actually asks for -- WITHOUT this, every spawned specialist got the
# same full-edit tool set regardless of what it's for, which is both
# unhelpful (a research task never gets Bash) and a real safety mismatch (a
# "security review" or "audit" specialist would get Write/Edit access, the
# exact thing the REAL Reviewer is deliberately denied for the same reason:
# "read-only, can't edit to make a claim pass"). Module-level and keyword-
# only (not an LLM call) so it's cheap and independently unit-testable.
_REVIEW_KEYWORDS = re.compile(r"\b(review|audit|read-?only|assess|inspect)\b", re.I)
_RESEARCH_KEYWORDS = re.compile(r"\b(benchmark|measure|research|experiment|profile)\b", re.I)


def tool_tier_for_subtask(subtask: str) -> str:
    """Returns one of 'review' / 'research_review' / 'research' / 'code'.
    Mirrors the fixed roster's own tiers (voice_commands.REVIEW_TOOLS /
    RESEARCH_REVIEW_TOOLS / RESEARCH_CODE_TOOLS / CLAUDE_CODE_TOOLS) --
    _run_real_agent() resolves the tier name to the actual tool list."""
    is_review = bool(_REVIEW_KEYWORDS.search(subtask))
    is_research = bool(_RESEARCH_KEYWORDS.search(subtask))
    if is_review and is_research:
        return "research_review"
    if is_review:
        return "review"
    if is_research:
        return "research"
    return "code"


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL LEARNING -- the fixed roster is no longer permanently fixed.
#
# Everything above operates WITHIN a single run: the network is built fresh
# every time, does its thing, and is thrown away. Nothing persists. This
# closes that gap: a dynamically-spawned specialist with a consistent track
# record of being genuinely useful gets PROMOTED into the static topology --
# future SpikingPipeline instances build it in from construction, no longer
# needing to be rediscovered live each time. The base architecture itself
# evolves across runs, driven by real accumulated evidence.
#
# Evidence comes from VaultLogger's own ledger entry: when a run spawned any
# specialists, its framing (_vault_logger_task) asks it to state, per name,
# VERDICT: <Name>: useful|unnecessary. Promotion is a WEIGHTED running score
# (+1 useful, -1 unnecessary, unchanged on no verdict), promoted once the
# score reaches PROMOTION_THRESHOLD.
#
# THIS WAS ORIGINALLY A STRICT BINARY AND-GATE (>=PROMOTION_THRESHOLD useful
# AND zero unnecessary, ever) -- changed 2026-07-17 after cross-referencing
# two real prior findings in this portfolio: consensus_scoping_rule_ladder.md
# ("weighted combination beats voting exactly as long as the calibration-
# time reliability ranking still holds") and this same file's OWN retracted
# soft-conflict finding (plain weighted inhibition beat binary/ternary
# gating on a structurally similar graded-evidence problem, same day).
# MEASURED before changing this, not assumed (test_promotion_rule_comparison.py):
# the strict AND rule only achieved 43.7% true-positive rate on specialists
# that were GENUINELY 90%-useful, because one unlucky "unnecessary" verdict
# out of several -- pure noise -- permanently blacklisted them. The weighted
# rule recovers to 98% TPR at a false-positive cost of only +1.3% on
# consistently-bad specialists (20% useful rate). This is the SAME pattern
# as the scoping-rule finding: strict all-or-nothing consensus throws away
# real signal that a weighted accumulation preserves.
#
# KNOWN GAP, found while testing this (test_structural_learning.py): there is
# NO DEMOTION PATH. Once a name is promoted, it's part of the fixed roster
# built in _build_brain() -- a later NEEDS_SPECIALIST request for that same
# name is correctly deduped against the existing neuron (see
# _maybe_spawn_specialist's dedupe check), which means it never becomes a
# "dynamically-spawned" specialist again and _record_specialist_outcomes()
# has nothing to attach a future bad verdict to. A promoted specialist that
# later turns out to be a mistake currently stays promoted forever (fixable
# by hand: edit specialist_history.json directly). Not fixed here -- flagged
# honestly rather than silently left as an assumed-solved edge case.
# ─────────────────────────────────────────────────────────────────────────────
SPECIALIST_HISTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "specialist_history.json")
PROMOTION_THRESHOLD = 3
VERDICT_RE = re.compile(r"VERDICT:\s*(\w+):\s*(useful|unnecessary)", re.I)


def load_specialist_history() -> dict:
    try:
        with open(SPECIALIST_HISTORY, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_specialist_history(history: dict) -> None:
    """Append-and-persist, fails soft -- a disk hiccup here must never break
    the pipeline, same discipline as DECISIONS_LOG."""
    try:
        with open(SPECIALIST_HISTORY, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, sort_keys=True)
    except OSError:
        pass


def promoted_specialists(history: dict = None) -> list:
    """Names whose running weighted score has reached PROMOTION_THRESHOLD --
    see the STRUCTURAL LEARNING comment above for why this is a weighted
    accumulation, not a strict AND-gate."""
    history = load_specialist_history() if history is None else history
    return sorted(
        name for name, rec in history.items()
        if rec.get("score", 0) >= PROMOTION_THRESHOLD
    )


# ─────────────────────────────────────────────────────────────────────────────
# THE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class SpikingPipeline:
    def __init__(self, task: str, project: str = "spikeling", dry_run: bool = True,
                 review_finds_issues: bool = False, spawn_request: tuple = None,
                 verdicts: dict = None):
        self.task = task
        self.project = project
        self.dry_run = dry_run
        self.review_finds_issues = review_finds_issues   # demo knob for the correct path
        # demo/test knob for dynamic specialists: (source_neuron, new_name, subtask) --
        # in dry_run mode, source_neuron's output includes the NEEDS_SPECIALIST marker
        # so the spawn mechanism is exercisable without a real agent call.
        self.spawn_request = spawn_request
        # demo/test knob for structural learning: {name: "useful"|"unnecessary"} --
        # in dry_run mode, VaultLogger's output includes matching VERDICT lines
        # so promotion is exercisable without a real agent call.
        self.verdicts = verdicts or {}
        self.fired: list[str] = []          # order agents actually ran
        self.outputs: dict[str, str] = {}
        self._followups: list[tuple] = []    # result-driven stimulations to apply after the cascade
        self._corrections = 0
        self._review_gates: list[dict] = []  # every real gate_review() call, for calibration logging
        self._specialist_tasks: dict = {}    # dynamic specialists' subtask text, for _agent_task()
        self._dynamic_specialists: list[str] = []   # spawned this run, for the growth cap + logging

        self.rt = self._build_brain()

    def _build_brain(self):
        """Build the exact topology described in agent_brain.spk (see that
        file's comments for the design rationale), directly via pyspike
        instead of parsing it as text. Every neuron/weight/leak value here
        must match agent_brain.spk -- there's no automatic sync, see the
        module docstring above.

        Built LIVE (build_live(), not build()) so a specialist's handler can
        spawn a new specialist mid-run -- see _maybe_spawn_specialist() and
        the DYNAMIC SPECIALISTS section above. This changes nothing about the
        static topology itself (verified byte-identical to the batch-built
        version in test_pyspike_orchestrator_parity.py); it only makes growth
        possible after construction."""
        self._net = net = Net(refractory_ms=40)
        rt = net.build_live()
        # CAUSAL TIME (see pyspike_causal.py): each neuron's timing is
        # measured against its OWN local tick, advanced only by itself or a
        # real causal predecessor -- never by unrelated network activity.
        # tick_size=50.0 matches the OLD self._clock += 50.0 convention
        # agent_brain.spk's refractory_ms=40 was tuned against (one
        # intervening stimulation clears refractory); this is a substrate
        # swap, not a retuning -- see test_pyspike_causal_orchestrator_parity.py
        # for the verification that behavior is unaffected on this topology.
        self._causal = CausalNet(net, rt, tick_size=50.0)

        # sensory layer (stimulated 0..100 by score_task)
        S_Work      = net.neuron("S_Work",      threshold=50, leak=1)
        S_Ambiguous = net.neuron("S_Ambiguous", threshold=50, leak=1)
        S_Complex   = net.neuron("S_Complex",   threshold=50, leak=1)
        S_Tests     = net.neuron("S_Tests",     threshold=50, leak=1)
        S_Research  = net.neuron("S_Research",  threshold=50, leak=1)   # currently unwired, same as the .spk

        # specialists -- one neuron per agent. self._specialists is kept
        # (not a local) because _maybe_spawn_specialist() adds to it later.
        self._specialists = {name: net.neuron(name, threshold=50, leak=2) for name in COMMANDS}
        specialists = self._specialists

        # flow -- sensory->specialist first-hop weights are now TRAINED
        # (pyspike_trained_routing.py, surrogate-gradient descent against
        # the 4 demo cases' independently-verified correct firing patterns),
        # not hand-picked. Verified: matches the 4 training cases exactly,
        # and lands close to the original hand-tuned values (same sign,
        # comparable magnitude) for every edge EXCEPT one, deliberately
        # NOT adopted -- see the note on S_Complex->Reviewer below.
        S_Work.to(specialists["Implementer"], weight=1.22)     # was 1.2 (hand-tuned)
        S_Ambiguous.to(specialists["Clarifier"], weight=1.20)  # was 1.2 (hand-tuned) -- unchanged
        S_Ambiguous.inhibits(specialists["Implementer"], weight=-1.50)  # was -2.0 (hand-tuned)
        S_Complex.to(specialists["PreRegister"], weight=1.33)  # was 1.2 (hand-tuned)
        S_Tests.to(specialists["TestWriter"], weight=1.05)     # was 1.2 (hand-tuned)
        # S_Complex->Reviewer: training converged to 1.26, ABOVE the ~1.0
        # solo-fire threshold -- but the training setup only modeled
        # first-hop weights and never saw Reviewer's OTHER real inputs
        # (Implementer/TestWriter/Corrector's own synapses into Reviewer,
        # wired below). Adopting 1.26 would let S_Complex alone fire
        # Reviewer without any of those, silently turning a coincidence-
        # reinforcement edge into an independent trigger -- a real design
        # change the training never actually tested for. Kept at the
        # original hand-tuned 0.7 deliberately, not from the trained value.
        S_Complex.to(specialists["Reviewer"], weight=0.7)
        specialists["Implementer"].to(specialists["Reviewer"], weight=1.2)
        specialists["TestWriter"].to(specialists["Reviewer"], weight=0.6)
        specialists["Corrector"].to(specialists["Reviewer"], weight=1.2)

        # handlers attach directly to their neuron -- no command-string
        # indirection needed with pyspike (the old .spk model required a
        # separate action->COMMAND->handler-dict chain; see COMMANDS below,
        # kept only for the agents_skipped enumeration, not for dispatch)
        for name in COMMANDS:
            net.action(specialists[name])(self._make_handler(name))

        # PROMOTED SPECIALISTS (see STRUCTURAL LEARNING section above): a
        # name with a consistent real track record of being genuinely useful
        # is built into the topology from here on, driven by S_Complex (the
        # same "extra work is needed" signal PreRegister/Reviewer use) --
        # no live spawn required, it's just part of the roster now.
        for name in promoted_specialists():
            if name in specialists:
                continue   # already a fixed name somehow -- never double-wire
            ref = net.neuron(name, threshold=50, leak=2)
            specialists[name] = ref
            S_Complex.to(ref, weight=1.2)
            ref.to(specialists["Reviewer"], weight=1.2)
            net.action(ref)(self._make_handler(name))

        self._base_specialist_names = set(specialists)   # fixed + promoted, BEFORE any live spawn
        return rt

    def _make_handler(self, neuron: str):
        def handler():
            self.fired.append(neuron)
            out = self._run_agent(neuron)
            self.outputs[neuron] = out
            self._maybe_spawn_specialist(neuron, out)
            if neuron == "VaultLogger":
                self._record_specialist_outcomes(out)
                if not self.dry_run:
                    self._log_to_obsidian_vault(out)
            # RESULT-DRIVEN ROUTING: review is the gate. Only a review that
            # reports issues wakes the Corrector, and only a couple of times.
            if neuron == "Reviewer" and self._review_reports_issues(out) and self._corrections < 2:
                self._corrections += 1
                # caused_by="Reviewer": Corrector fires because of Reviewer's
                # OUTPUT, a real causal dependency that isn't a synapse -- see
                # CausalNet.stimulate()'s docstring for why this matters
                # (without it, Corrector's independently-drifted local tick
                # can silently fail to clear Reviewer's refractory on re-review).
                self._followups.append(("Corrector", 60.0, "Reviewer"))
        return handler

    def _log_to_obsidian_vault(self, vault_output: str) -> None:
        """VaultLogger's entire PURPOSE is to write a real ledger entry into
        the Obsidian vault (vault/Project Work) -- but until now nothing
        actually called voice_commands.log_to_vault() to do that. The one
        entry that existed there (the task_decisions_dashboard.py run) only
        exists because the underlying Claude Code agent happened to choose
        to write it itself, not because this was wired in. That's the same
        "hope the LLM remembers" pattern this project has repeatedly found
        and fixed elsewhere today (e.g. tool_tier_for_subtask() instead of
        hoping a spawned specialist picks safe tools). Fixed the same way:
        a deterministic Python call, not a hope.

        Real-mode only (dry_run is checked by the caller) -- vault writes
        would be meaningless/noisy for synthetic demo runs. Fails soft,
        matching log_to_vault()'s own discipline: vault logging must never
        break the actual pipeline."""
        try:
            import voice_commands as vc
            notes_dir = os.path.join(vc.VAULT_DIR, "Project Work")
            task_label = f"[spiking_pipeline:{self.project}] {self.task}"
            vc.log_to_vault("agent_task", task_label, output=vault_output,
                            status="done_via_spiking_pipeline", notes_dir=notes_dir)
        except Exception as e:
            print(f"vault logging failed (non-fatal): {e}", flush=True)

    def _record_specialist_outcomes(self, vault_output: str) -> None:
        """Parse VaultLogger's VERDICT: <Name>: useful|unnecessary lines and
        update the persisted specialist_history.json -- this is the ONLY
        write path for that file, and it's how a dynamically-spawned
        specialist accumulates the track record that can eventually promote
        it into the fixed roster (see STRUCTURAL LEARNING section). A
        spawned specialist this run with NO verdict given (ambiguous output)
        counts as neither useful nor unnecessary -- it's evidence-neutral,
        not silently treated as a negative.

        Isolation note for tests: this always writes to whatever
        SPECIALIST_HISTORY currently points at -- ordinary dry-run usage
        (demo(), the parity tests) never spawns anything (no spawn_request
        given), so it never reaches this far and never touches the file.
        Tests that DO exercise this path should monkeypatch the module-level
        SPECIALIST_HISTORY to a temp path first, not rely on dry_run alone."""
        if not self._dynamic_specialists:
            return
        verdicts = {name.lower(): v.lower() for name, v in VERDICT_RE.findall(vault_output or "")}
        history = load_specialist_history()
        for name in self._dynamic_specialists:
            # judged_useful/unnecessary are kept as human-readable diagnostics
            # (visible in the raw JSON); "score" is the actual weighted
            # running total promoted_specialists() decides on -- see the
            # STRUCTURAL LEARNING comment for why this isn't just
            # judged_useful - unnecessary computed fresh each time (it's the
            # same value here since both start at 0, but "score" existing as
            # its own field makes the promotion rule's logic independent of
            # how the diagnostic counters are named/shaped).
            rec = history.setdefault(name, {"spawned": 0, "judged_useful": 0, "unnecessary": 0, "score": 0})
            rec["spawned"] += 1
            verdict = verdicts.get(name.lower())
            if verdict == "useful":
                rec["judged_useful"] += 1
                rec["score"] += 1
            elif verdict == "unnecessary":
                rec["unnecessary"] += 1
                rec["score"] -= 1
        save_specialist_history(history)

    def _maybe_spawn_specialist(self, source_neuron: str, output: str) -> None:
        """If `output` asks for a specialist that doesn't exist yet, spawn one
        LIVE: a real neuron, wired into Reviewer (every dynamic specialist
        gets reviewed too, same as the fixed ones), with a real handler
        attached. Capped and deduped -- see the DYNAMIC SPECIALISTS module
        comment for why that's not optional.

        DELIBERATELY NOT wired with a static synapse FROM source_neuron.
        Two things were tried and rejected by this file's own test
        (test_dynamic_specialists.py) before landing here:
          - a direct source_neuron -> ref synapse fires ref in the SAME
            cascade instant as source_neuron (handlers run before
            propagation, propagation reads a live synapse list) -- but then
            ref -> Reviewer tries to fire Reviewer AT THAT SAME INSTANT,
            and Reviewer is still refractory-locked from firing moments
            earlier in the same cascade, so the review silently never
            happens (found: Reviewer only fired once when it should fire
            twice).
          - combining that synapse WITH a followup double-fires ref itself
            (found: DBMigrator fired twice).
        The working pattern is the SAME one already used for Corrector:
        whether to spawn depends on the agent's OUTPUT, so it's injected as
        a followup stimulation at an ADVANCED clock time, not a static
        synapse -- by the time it fires, Reviewer's refractory has cleared,
        so ref -> Reviewer (a static synapse, like Corrector -> Reviewer)
        correctly cascades into a real second review.

        Scans for ALL requests in the output, not just the first -- a single
        agent turn can legitimately need more than one (e.g. a task that
        touches both infra and security). Each is still subject to the
        per-name dedupe and the total growth cap; if an output lists more
        requests than the remaining cap allows, the earliest-listed ones win
        and the rest are silently dropped (capped, not queued -- consistent
        with "explicit stop condition, no exceptions" everywhere else this
        mechanism is documented)."""
        for m in SPAWN_SPECIALIST_RE.finditer(output or ""):
            name, subtask = m.group(1), m.group(2).strip()
            if name in self._specialists:
                continue   # dedupe -- never spawn the same name twice
            if len(self._dynamic_specialists) >= MAX_DYNAMIC_SPECIALISTS:
                break      # explicit growth cap -- see DRIVE_FLOOR lesson in the memory notes

            ref = self._net.neuron(name, threshold=50, leak=2)
            self._specialists[name] = ref
            self._specialist_tasks[name] = subtask
            self._dynamic_specialists.append(name)

            self._net.action(ref)(self._make_handler(name))
            ref.to(self._specialists["Reviewer"], weight=1.2)
            # caused_by=source_neuron: this specialist exists because of
            # source_neuron's OUTPUT, a real (non-synaptic) causal dependency --
            # same reasoning as Corrector's caused_by="Reviewer" above.
            self._followups.append((name, 60.0, source_neuron))

    def _review_reports_issues(self, review_output: str) -> bool:
        if self.dry_run:
            return self.review_finds_issues and self._corrections < 1
        gate = gate_review(review_output)
        self._review_gates.append(gate)
        return gate["decision"]

    def _run_agent(self, neuron: str) -> str:
        """Fire the agent. dry_run just records intent (no tokens); real mode
        calls into voice_commands.do_claude_code with the right preamble/tools."""
        if self.dry_run:
            out = f"[dry-run] {neuron} would run on: {self.task[:60]}"
            if self.spawn_request and self.spawn_request[0] == neuron:
                _, name, subtask = self.spawn_request
                out += f"\nNEEDS_SPECIALIST: {name}: {subtask}"
            if neuron == "VaultLogger" and self.verdicts:
                for name, verdict in self.verdicts.items():
                    out += f"\nVERDICT: {name}: {verdict}"
            return out
        return self._run_real_agent(neuron)

    def _run_real_agent(self, neuron: str) -> str:
        # Imported lazily so the demo/tests never touch the Claude CLI layer.
        import voice_commands as vc
        # Map each specialist to a concrete Claude Code invocation. Reuse the
        # existing tool tiers + preambles so this rides on the validated pipeline.
        if neuron == "Clarifier":
            # Clarifier STOPS the pipeline if it wants a question answered.
            return "(clarify stage — wire to vc.AGENT_CLARIFY_PREAMBLE)"
        tools = getattr(vc, "CLAUDE_CODE_TOOLS", None)
        if neuron == "Reviewer":
            tools = getattr(vc, "REVIEW_TOOLS", tools)
        elif neuron in self._specialist_tasks:
            # dynamically-spawned specialist -- pick its tier from what its
            # subtask actually asks for, don't default everyone to full edit
            # access (see tool_tier_for_subtask()'s docstring for why that
            # was a real safety mismatch, not just an ergonomics gap)
            tier = tool_tier_for_subtask(self._specialist_tasks[neuron])
            # WIRED TO spike_tool_gateway.py (2026-08-28): the tier -> tool
            # list mapping used to be a hardcoded dict here. Now it's
            # resolve_tool_provider(tier), reading spike_tools.yaml as the
            # sole source of truth -- same 4 tiers, same real tool lists,
            # but now config-driven and auditable instead of inline code.
            # No fallback to a default tool list on a ToolGatewayError --
            # letting that raise is deliberate (see the gateway's own
            # module docstring on why silent defaulting is exactly the
            # failure mode this exists to refuse).
            from spike_tool_gateway import resolve_tool_provider
            attr_name = resolve_tool_provider(tier)
            tools = getattr(vc, attr_name, tools)
        return vc.do_claude_code(task=self._agent_task(neuron), tools=tools) or ""

    # Applies to EVERY working specialist -- Reviewer, PreRegister, Corrector,
    # and any dynamically-spawned specialist can equally discover a genuine
    # gap, not just Implementer. Excluded: VaultLogger (a summarization pass,
    # not work -- nothing for it to discover) and Clarifier (a stop-and-ask
    # gate, not real work either).
    _SPAWN_HINT = (
        "\n\nIf, while working, you discover this task genuinely needs a kind "
        "of specialist not implied above (e.g. a database migration, a "
        "security review, an API design pass) -- something a rename/"
        "implement/test/review cycle doesn't cover -- say so explicitly on "
        "its own line at the end of your output: "
        "NEEDS_SPECIALIST: <ShortName>: <what it should do>. You may list "
        "more than one line if genuinely more than one is needed. Only do "
        "this for genuinely distinct kinds of work, not routine sub-steps of "
        "what you're already doing."
    )
    _NO_SPAWN_HINT = {"VaultLogger", "Clarifier"}

    def _agent_task(self, neuron: str) -> str:
        # each specialist gets the task framed for its job
        if neuron in self._specialist_tasks:
            # a dynamically-spawned specialist gets the subtask IT was
            # requested to do, not the top-level task -- and, up to the
            # growth cap, can request one of its own the same way
            raw_task = self._specialist_tasks[neuron]
        else:
            frames = {
                "PreRegister": f"Before any edit, state ONE falsifiable claim about what this change will do: {self.task}",
                "Implementer": self.task,
                "TestWriter":  f"Add or adjust tests for: {self.task}",
                "Reviewer":    self._reviewer_task(),
                "Corrector":   self._corrector_task(),
                "VaultLogger": self._vault_logger_task(),
            }
            raw_task = frames.get(neuron, self.task)
        base = raw_task
        if neuron not in self._NO_SPAWN_HINT:
            base += self._SPAWN_HINT
        return self._apply_skills(neuron, base, raw_task)

    def _apply_skills(self, neuron: str, base: str, raw_task: str) -> str:
        """Appends matching skill(s) onto the end of the task text, via
        TWO independent, deliberately different mechanisms:

        1. select_skills_for(neuron) -- ROLE selection. A skill (review-
           discipline, pre-registration-discipline, corrector-discipline)
           self-selects by naming this neuron in its own description.
           Fires on EVERY invocation of that neuron, regardless of task
           content -- correct for "how Reviewer should behave."

        2. select_skills_for_task(raw_task) -- TOPIC selection. The 50
           environment-tailored skills self-select by keyword match
           against `raw_task` (the specialist's own framed task text,
           which embeds self.task or a dynamic specialist's real
           subtask) -- deliberately NOT matched against `base`, which
           may already carry SPAWN_HINT boilerplate or a role skill's
           own injected content; matching against that risked a skill's
           body text accidentally re-triggering another skill.

        Results are combined in that order (role skills first, then
        topic skills) and de-duplicated -- a skill matched by both paths
        is only appended once. Byte-identical to `base` when nothing
        matches on either path -- verified by test_pyspike_orchestrator_
        parity.py still passing unchanged.

        A missing/empty spike_skills/ directory is swallowed, not raised,
        on both paths: unlike spike_tool_gateway's categories (mandatory,
        pre-registered, "auto" is a hard error), not having any skill yet
        is a normal, legitimate state, not a misconfiguration -- so
        SkillError here means "nothing to add," and the specialist runs
        exactly as it did before this feature existed."""
        try:
            from spike_skills import (
                select_skills_for, select_skills_for_task, load_skill_content, SkillError,
            )
        except ImportError:
            return base
        try:
            role_matched = select_skills_for(neuron)
        except SkillError:
            role_matched = []
        try:
            topic_matched = select_skills_for_task(raw_task)
        except SkillError:
            topic_matched = []
        seen = set()
        for name in role_matched + topic_matched:
            if name in seen:
                continue
            seen.add(name)
            try:
                content = load_skill_content(name)
            except SkillError:
                continue
            base += f"\n\n--- Skill: {name} ---\n{content.strip()}"
        return base

    def _dynamic_specialists_note(self, verb: str) -> str:
        """Shared by _reviewer_task/_corrector_task/_vault_logger_task -- a
        review or correction pass that fires structurally AFTER a spawn
        (via the ref -> Reviewer synapse) still has no idea the spawn
        happened unless the task text says so; firing order alone doesn't
        tell an LLM call what changed. Returns '' when nothing was spawned,
        so the framing is byte-identical to before this feature existed."""
        if not self._dynamic_specialists:
            return ""
        return (f" This task also involved {len(self._dynamic_specialists)} "
                f"dynamically-spawned specialist(s) that weren't part of the "
                f"original plan: {', '.join(self._dynamic_specialists)}. "
                f"{verb} their work specifically, not just the original change.")

    def _reviewer_task(self) -> str:
        note = self._dynamic_specialists_note("Check")
        return f"Peer-review the change for: {self.task}.{note} Read-only. Call out overclaiming."

    def _corrector_task(self) -> str:
        base = f"Fix exactly what review flagged for: {self.task}"
        note = self._dynamic_specialists_note("Factor in")
        return f"{base}.{note}" if note else base   # no note -> byte-identical to before this feature

    def _vault_logger_task(self) -> str:
        note = ""
        if self._dynamic_specialists:
            note = (f" This run dynamically spawned {len(self._dynamic_specialists)} "
                    f"unanticipated specialist(s) mid-task: "
                    f"{', '.join(self._dynamic_specialists)} -- mention what each was "
                    f"for and whether it turned out to be genuinely needed.")
        return f"Write an honest ledger entry for the work on: {self.task}.{note}"

    def run(self) -> dict:
        # 1. score the task into sensory current
        scores = score_task(self.task, self.project)

        # 2. CLARIFY IS A GATE, NOT A STAGE. If the task scores ambiguous, the
        #    Clarifier fires FIRST and the pipeline STOPS — you don't implement a
        #    task you don't understand, you ask one question and wait. This is the
        #    same "ask & stop" the real do_agent_task pipeline does, expressed as
        #    an inhibitory gate: a strong S_Ambiguous spike blocks the work path.
        if scores["S_Ambiguous"] >= 50:
            self._drive([("S_Ambiguous", scores["S_Ambiguous"])])
            result = {
                "scores": scores, "fired": self.fired, "agents_run": len(self.fired),
                "clarified": True,
                "agents_skipped": [n for n in self._base_specialist_names if n not in self.fired],
            }
            self._log_decision(result)
            return result

        # 3. seed the work path; each sensory spike cascades synchronously through
        #    the spine (implement -> review), handlers running as neurons fire
        work_seeds = [(name, drive) for name, drive in scores.items() if name != "S_Ambiguous"]
        self._drive(work_seeds)

        # 4. drain result-driven follow-ups (corrections -> re-reviews) at an
        #    advancing clock so refractory clears for legitimate re-fires
        while self._followups:
            batch = self._followups
            self._followups = []
            self._drive(batch)

        # 5. write the ledger ONCE, at the end, after everything has settled.
        # caused_by=the last specialist to run: the ledger's causal
        # predecessor is "whatever the run just finished doing", same
        # reasoning as Corrector/dynamic-specialist followups above.
        if "Implementer" in self.fired:
            self._drive([("VaultLogger", 60.0, self.fired[-1])])

        result = {
            "scores": scores,
            "fired": self.fired,
            "agents_run": len(self.fired),
            "agents_skipped": [n for n in self._base_specialist_names if n not in self.fired],
        }
        self._log_decision(result)
        return result

    def _drive(self, stimulations: list) -> None:
        """Each entry is (name, drive) for an ordinary/sensory stimulation,
        or (name, drive, caused_by) for a result-driven one that depends on
        another neuron's OUTPUT rather than a synapse -- see caused_by's
        docstring in CausalNet.stimulate() for why the distinction matters
        under causal time.

        Entries in the SAME batch default-chain to the previous entry
        (unless a caused_by is given explicitly): the orchestrator choosing
        to stimulate S_Work, then S_Complex, then S_Tests IN THAT ORDER,
        in one call, is itself a real (if soft) causal intent -- without
        this, every sensory seed would be an independently-clocked "first
        event ever" that coincidentally lands on the SAME local tick as
        every other first event, so their cascades collide when they
        converge on a shared downstream neuron (found: this silently ate
        the coincidence-detection second Reviewer fire from S_Complex +
        TestWriter's accumulated membrane potential, a real regression
        caught by re-running the demo, not assumed fixed by the caused_by
        fix for followups alone)."""
        prev = None
        for entry in stimulations:
            name, drive = entry[0], entry[1]
            caused_by = entry[2] if len(entry) > 2 else prev
            self._causal.stimulate(name, float(drive), caused_by=caused_by)
            prev = name

    def _log_decision(self, result: dict) -> None:
        """Append this routing decision as one JSON line. Logged for BOTH
        dry-run and real invocations -- dry_run is recorded on the entry so
        training can filter to real usage only, but demo/dry-run cases are
        still useful volume for early sanity checks. This is intentionally
        append-only and fails soft (a logging hiccup must never break the
        pipeline itself)."""
        entry = {
            "ts": time.time(),
            "task": self.task,
            "project": self.project,
            "dry_run": self.dry_run,
            "scores": result["scores"],
            "fired": result["fired"],
            "agents_run": result["agents_run"],
            "clarified": result.get("clarified", False),
            "agents_skipped": result["agents_skipped"],
            "review_gates": self._review_gates,   # calibration data for the ternary review gate
            "dynamic_specialists": self._dynamic_specialists,   # live-spawned specialists this run
        }
        try:
            with open(DECISIONS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass


def _print_result(task: str, res: dict) -> None:
    print(f"\n  TASK: {task}")
    hot = {k: v for k, v in res["scores"].items() if v >= 50}
    print(f"  sensory (>=50 fires): {hot}")
    print(f"  AGENTS RUN ({res['agents_run']}): {' -> '.join(res['fired'])}")
    print(f"  skipped (0 tokens): {', '.join(res['agents_skipped']) or 'none'}")


def demo() -> None:
    print("=" * 70)
    print("  SPIKING AGENT PIPELINE — token savings are structural")
    print("=" * 70)
    cases = [
        ("rename the fire_rate variable to shot_delay", False),
        ("add a test for the club-throw damage falloff", False),
        ("fix stuff", False),
        ("refactor the whole tribe economy across trade, war, and alliances "
         "and integrate it with the new terrain system and also add tests", True),
    ]
    for task, issues in cases:
        p = SpikingPipeline(task, dry_run=True, review_finds_issues=issues)
        _print_result(task, p.run())
    print("\n  ^ note how the trivial rename runs 3 agents while the big refactor")
    print("    lights the whole board — the cheap tasks stay cheap, automatically.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="task to route (omit with --demo)")
    ap.add_argument("--demo", action="store_true", help="run the no-API routing demo")
    ap.add_argument("--project", default="spikeling")
    ap.add_argument("--real", action="store_true", help="actually run agents via Claude (spends tokens)")
    args = ap.parse_args()
    if args.demo or not args.task:
        demo()
        return
    task = " ".join(args.task)
    p = SpikingPipeline(task, project=args.project, dry_run=not args.real)
    _print_result(task, p.run())


if __name__ == "__main__":
    main()
