"""
Spiking skill-router: the Spikeling orchestrator as a top layer over the user's
custom Claude skills. One LIF neuron per skill; a keyword sensory layer turns the
task text into per-skill drive; only over-threshold skills "fire" and should be
loaded. Everything below threshold costs zero tokens -- structural gating.

    python skill_router.py "audit transformers for a cache bug and open a PR"
    -> oss-bug-audit, oss-status-precision, commit-discipline

Keep SKILLS / RULES in sync with ~/.claude/skills/.
"""

import re
import sys
import json

THRESHOLD = 50.0
LEAK = 1.0

# skill -> (base drive if any keyword hits, list of (weight, regex) sensory rules)
RULES = {
    "oss-bug-audit":            [(70, r"\b(audit|bug|bugs|pr|pull request|contribut\w*|upstream|fix)\b.*\b(library|repo|transformers|pytorch|torch|sklearn|scikit|snntorch|sinabs|lightning|numpy|scipy|open source|oss)\b"),
                                 (70, r"\b(transformers|pytorch|scikit-learn|snntorch|sinabs|tenns-core)\b.*\b(bug|fix|pr|issue)\b")],
    "oss-status-precision":     [(60, r"\b(contribut\w*|merged|pr status|landed|accepted)\b")],
    "spikeling-os-milestone":   [(75, r"\b(spikeling.?os|milestone|cc\.elf|subset.?c|kernel)\b")],
    "qemu-boot-this-pc":        [(70, r"\b(qemu|boot|whpx|bios\.img)\b"),
                                 (40, r"\bspikeling.?os\b")],
    "honest-benchmark":         [(70, r"\b(benchmark|compare|comparison|does .* beat|ablation|efficiency|measure|vs\b)\b")],
    "ternary-quant-reality-check":[(75, r"\b(ternary|quantiz\w+|low.?bit|twn|ttq|1\.58|int4)\b")],
    "research-rung":            [(70, r"\b(reservoir|substrate|snn|rung|structure helps|symmetry|phononic|quasicrystal|kuramoto|topolog\w+)\b")],
    "research-index-check":     [(60, r"\b(new experiment|hasn.?t been tried|is this novel|already done|research index)\b"),
                                 (35, r"\b(rung|new study|reservoir|substrate)\b")],
    "cross-substrate-check":    [(70, r"\b(cross.?substrate|unif\w+ (rule|law)|transfers? across|synthesis across)\b")],
    "godot-perf-tune":          [(75, r"\b(lag|laggy|stutter|fps|slow|slower|performance|optimi[sz]e|epic scale)\b"),
                                 (40, r"\b(godot|tribe|horde|npc|scene|\.gd|physics frame)\b")],
    "portfolio-publish":        [(70, r"\b(publish|share|put on (my )?github|pages|deploy the (page|report|portfolio))\b")],
    "methodlm-artifact-theme":  [(65, r"\b(methodlm.?theme|ember|furnace|methodlm styled?)\b"),
                                 (30, r"\b(publish|artifact|portfolio)\b.*\bmethodlm\b")],
    "snn-agent-routing":        [(70, r"\b(agent (pipeline|routing)|route agents|spiking orchestrator|save tokens on (a )?pipeline)\b")],
    "skill-router":             [(70, r"\b(route skills|orchestrator over claude|which skills|skill router|save tokens on skills)\b")],
    "dual-sensor-anomaly":      [(72, r"\b(monitor\w*|anomaly|sensor|telemetry|pond|hive|beehive|power draw|kasa|scrapyard)\b")],
    "physical-hardware-test":   [(72, r"\b(real hardware|physical (test|experiment)|mics?|microphone|webcam|arduino|tdoa|sync.?mesh)\b")],
    "new-research-project":     [(60, r"\b(new (study|project|investigation)|let.?s test whether|scaffold a )\b")],
    "negative-result-writeup":  [(55, r"\b(negative result|null result|falsif\w+|didn.?t work|lost to (the )?baseline|write up the negative)\b")],
    "spike-vault-log":          [(45, r"\b(vault|obsidian|log (this|the) (work|result|finding)|lessons?)\b"),
                                 (30, r"\b(rung|milestone|finding|study)\b")],
    "commit-discipline":        [(45, r"\b(commit|push|branch|git)\b"),
                                 (30, r"\b(pr|pull request|milestone)\b")],
}

# secondary skills that fire whenever a given primary skill fires (code cascade,
# not a text match -- the SNN equivalent is a fixed excitatory synapse)
CASCADE = {
    "oss-bug-audit":          ["oss-status-precision", "commit-discipline"],
    "spikeling-os-milestone": ["qemu-boot-this-pc", "commit-discipline"],
    "research-rung":          ["research-index-check", "cross-substrate-check", "spike-vault-log"],
    "new-research-project":   ["research-index-check"],
    "skill-router":           ["snn-agent-routing"],
}


def drive_for(task: str) -> dict:
    t = task.lower()
    out = {}
    for skill, rules in RULES.items():
        d = 0
        for weight, rx in rules:
            if re.search(rx, t):
                d += weight
        if d:
            out[skill] = min(d, 100)
    return out


def fire(drives: dict) -> list:
    """One LIF step per skill (leak then integrate); fire if membrane >= threshold.
    Single-shot: the sensory drive is constant, so one step decides it -- the SNN
    here is a structural gate, not a temporal simulation."""
    fired = []
    for skill, d in drives.items():
        mem = 0.0
        mem = (1 - 0) * mem - LEAK + d      # leak + integrate
        if mem >= THRESHOLD:
            fired.append((skill, mem))
    fired.sort(key=lambda x: -x[1])
    return [s for s, _ in fired]


def route(task: str) -> dict:
    drives = drive_for(task)
    fired = fire(drives)
    # cascade: a fired primary pulls in its secondaries, order preserved
    out = list(fired)
    for s in fired:
        for sec in CASCADE.get(s, []):
            if sec not in out:
                out.append(sec)
    return {"task": task, "drives": drives, "load_skills": out,
            "skipped": [s for s in RULES if s not in out]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    r = route(" ".join(sys.argv[1:]))
    print("load skills:", ", ".join(r["load_skills"]) or "(none -- handle directly)")
    print("\nscores:", json.dumps(r["drives"], indent=1))
