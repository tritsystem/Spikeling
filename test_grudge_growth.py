#!/usr/bin/env python
"""
test_grudge_growth.py — proof-of-concept for a capability neither Spikeling
project has built yet: an INDIVIDUAL mind whose topology grows itself,
permanently, from its own lived history -- not shared structural learning
(spiking_orchestrator.py's promoted_specialists() promotes into ONE shared
roster future pipeline instances all inherit) and not weight tuning on a
fixed template (tribe/tribemember.gd's PERSONALITIES dict picks numbers on
an otherwise-identical brain shape for every member).

Modeled directly on tribe/tribemember.gd's real brain
(SawContribute/SawBetray -> Trust -> Follow, see _brain_text()) and its real
betray() mechanic (Trust wipe via an inhibitory synapse) just added there.
This POC asks the next honest question: after REPEATED betrayal, should an
individual's topology itself change, permanently and specific to them, in a
way an identical-template twin who wasn't betrayed never develops?

MECHANISM: on the Nth betrayal (GRUDGE_THRESHOLD), a brand-new "Grudge"
neuron is grown live (net.neuron() + net.connect() on a build_live()
runtime, reusing test_self_growing_network.py's proven live-growth
capability) and wired Grudge -> Follow with a dominating inhibitory weight.
From then on this individual's brain has one more neuron than an
unbetrayed twin -- a real, inspectable structural difference, not a hidden
numeric one.

TIMING, learned the hard way elsewhere in this project (see
spiking_agent_pipeline's dynamic-specialist-spawning bug and its own
docstring here): this runtime's inhibition is EVENT-DRIVEN, applied only at
the instant the source neuron fires. Two events landing in the SAME
stimulate() cascade race on synapse-list order -- Trust's own direct
synapse to Follow already fires (and refractory-locks) Follow before a
same-instant Grudge->Follow synapse could apply, exactly the bug
spiking_orchestrator.py hit with Reviewer. So this Grudge is deliberately
fired on ITS OWN, SEPARATE tick right after every contribution, not
same-instant with it -- same "clock-advanced followup, not a same-cascade
synapse" fix already proven there.

    python test_grudge_growth.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyspike import Net  # noqa: E402

GRUDGE_THRESHOLD = 3   # betrayals from the player before a Grudge neuron grows


class GrudgeNPC:
    """Mirrors tribemember.gd's real brain shape and betray() mechanic,
    plus the new capability: a Grudge neuron that grows in permanently
    once GRUDGE_THRESHOLD betrayals have happened, and never disappears."""

    # NOTE units: this Python runtime scales synapse weight x50 internally
    # (runtime.py _fire(): "downstream.membrane_potential += syn.weight * 50.0"),
    # unlike tribe/spikeling.gd's GDScript engine which adds the weight
    # directly. These constructor defaults are tribemember.gd's real numbers
    # (80, 160, 120) DIVIDED by 50 to reproduce the same intended deltas on
    # THIS engine -- copying the raw GDScript numbers here silently blew
    # every neuron's threshold by ~60x (caught by this POC's own first run:
    # Follow fired MORE for the betrayed twin, not less -- the huge weights
    # trivially overpowered the inhibition too).
    def __init__(self, name: str, trust_leak: int = 2, contrib: float = 1.6, follow_w: float = 2.4):
        self.name = name
        self.net = Net(refractory_ms=2)
        self.rt = self.net.build_live()   # empty live runtime FIRST -- everything
                                            # defined below mirrors into it as it's built
        self.SawContribute = self.net.neuron("SawContribute", threshold=50, leak=20)
        self.SawBetray = self.net.neuron("SawBetray", threshold=50, leak=20)
        self.Trust = self.net.neuron("Trust", threshold=100, leak=trust_leak)
        self.Follow = self.net.neuron("Follow", threshold=100, leak=5)
        self.net.connect(self.SawContribute, self.Trust, weight=contrib)
        self.net.connect(self.SawBetray, self.Trust, weight=-3.2)
        self.net.connect(self.Trust, self.Follow, weight=follow_w)

        self.t = 0.0
        self.betrayal_count = 0
        self.grudge_grown_at: float = None   # tick the Grudge neuron was grown, or None
        self.follow_fire_log: list = []      # ticks Follow actually fired

    def _stimulate(self, name: str, drive: float) -> None:
        self.t += 1.0
        self.rt.stimulate(name, self.t, drive)
        if self.rt.neurons["Follow"].last_spike_time == self.t:
            self.follow_fire_log.append(self.t)

    def contribute(self) -> None:
        self._stimulate("SawContribute", 80.0)
        # a grown grudge re-fires on ITS OWN tick right after -- separate
        # cascade, so it isn't racing Trust's own same-instant synapse to
        # Follow (see module docstring: same-instant vetoes silently lose)
        if self.grudge_grown_at is not None:
            self._stimulate("Grudge", 20.0)

    def betray(self) -> None:
        self._stimulate("SawBetray", 80.0)
        self.betrayal_count += 1
        if self.betrayal_count >= GRUDGE_THRESHOLD and self.grudge_grown_at is None:
            self._grow_grudge()

    def _grow_grudge(self) -> None:
        Grudge = self.net.neuron("Grudge", threshold=10, leak=0)   # leak=0: never forgets
        self.net.connect(Grudge, self.Follow, weight=-6.0)          # dominates Trust's +2.4
        self.grudge_grown_at = self.t

    def neuron_count(self) -> int:
        return len(self.rt.neurons)


# ─────────────────────────────────────────────────────────────────────────────
def _selftest_twins_diverge() -> None:
    """THE core claim: two IDENTICAL-template NPCs (same personality numbers),
    one betrayed past the threshold and one never touched, given the exact
    same positive treatment afterward. Prediction: the unbetrayed twin's
    Follow eventually fires again (trust genuinely rebuilds); the betrayed
    twin's Follow NEVER fires again, no matter how many identical gifts
    follow -- because their topology, not just their numbers, has diverged."""
    twin_never_betrayed = GrudgeNPC("Twin-B")
    twin_betrayed = GrudgeNPC("Twin-A")

    for _ in range(GRUDGE_THRESHOLD):
        twin_betrayed.betray()

    # identical positive treatment for both, well past what a single
    # contribution needs to rebuild Trust and fire Follow at least once
    for _ in range(6):
        twin_never_betrayed.contribute()
        twin_betrayed.contribute()

    b_fired = len(twin_never_betrayed.follow_fire_log) > 0
    a_never_fired = len(twin_betrayed.follow_fire_log) == 0
    a_grew_grudge = twin_betrayed.grudge_grown_at is not None
    b_grew_grudge = twin_never_betrayed.grudge_grown_at is not None
    structurally_diverged = twin_betrayed.neuron_count() > twin_never_betrayed.neuron_count()

    ok = b_fired and a_never_fired and a_grew_grudge and (not b_grew_grudge) and structurally_diverged
    print(f"    Twin-B (never betrayed):  Follow fired {len(twin_never_betrayed.follow_fire_log)}x, "
          f"neurons={twin_never_betrayed.neuron_count()}, grudge_grown={b_grew_grudge}")
    print(f"    Twin-A (betrayed {GRUDGE_THRESHOLD}x):    Follow fired {len(twin_betrayed.follow_fire_log)}x, "
          f"neurons={twin_betrayed.neuron_count()}, grudge_grown={a_grew_grudge}")
    print(f"  [{'PASS' if ok else 'FAIL'}] identical-template twins diverge STRUCTURALLY from lived "
          f"history alone: the betrayed twin permanently loses Follow despite identical later "
          f"treatment, the unbetrayed twin's trust genuinely recovers")


def _selftest_below_threshold_does_not_grow() -> None:
    """CONTROL: betrayal below GRUDGE_THRESHOLD must NOT grow a Grudge neuron
    -- otherwise this is just SawBetray's existing one-shot Trust wipe
    wearing a new name, not a genuine threshold-triggered structural change.
    An NPC betrayed once should still recover Follow with enough gifts."""
    npc = GrudgeNPC("Once-Betrayed")
    npc.betray()   # below threshold (needs GRUDGE_THRESHOLD)
    for _ in range(6):
        npc.contribute()

    ok = npc.grudge_grown_at is None and len(npc.follow_fire_log) > 0
    print(f"    betrayed once (threshold={GRUDGE_THRESHOLD}): grudge_grown={npc.grudge_grown_at is not None}, "
          f"Follow fired {len(npc.follow_fire_log)}x after")
    print(f"  [{'PASS' if ok else 'FAIL'}] below-threshold betrayal does NOT grow a Grudge neuron, "
          f"and trust still genuinely recovers -- the divergence in the main test is really "
          f"threshold-triggered, not just 'any betrayal ever blocks forever'")


def _selftest_grudge_survives_many_more_contributions() -> None:
    """The grudge must be PERMANENT (leak=0, never re-evaluated away) -- not
    a temporary penalty that erodes given enough unrelated good behavior
    afterward, which would just be a slower Trust wipe, not a structural
    veto. Hammer it with 30 more contributions past growth and confirm
    Follow still never fires."""
    npc = GrudgeNPC("Hammered")
    for _ in range(GRUDGE_THRESHOLD):
        npc.betray()
    for _ in range(30):
        npc.contribute()

    ok = npc.grudge_grown_at is not None and len(npc.follow_fire_log) == 0
    print(f"    30 contributions after grudge grown: Follow fired {len(npc.follow_fire_log)}x")
    print(f"  [{'PASS' if ok else 'FAIL'}] the grudge is genuinely permanent -- 30 further identical "
          f"positive events don't erode it, matching leak=0's 'never forgets' design")


if __name__ == "__main__":
    print("=" * 78)
    print("  GRUDGE GROWTH -- proof-of-concept: individually-diverging brain topology")
    print("=" * 78)
    _selftest_twins_diverge()
    _selftest_below_threshold_does_not_grow()
    _selftest_grudge_survives_many_more_contributions()
