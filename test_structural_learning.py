#!/usr/bin/env python
"""
test_structural_learning.py — verifies the pipeline's structural-learning
capability: a dynamically-spawned specialist with a consistent real track
record of being judged useful gets PROMOTED into the fixed topology for
future SpikingPipeline instances, without needing to be rediscovered live
each time.

Isolated from the real specialist_history.json via monkeypatching --
these tests must NEVER touch the real file (it's real accumulated usage
history, gitignored on purpose).

Eight checks:
  1. VERDICT PARSING       -- VaultLogger's VERDICT lines correctly update
                               the persisted history.
  2. PROMOTION THRESHOLD    -- fewer than PROMOTION_THRESHOLD useful verdicts
                               does NOT promote.
  3. PROMOTION FIRES        -- >=PROMOTION_THRESHOLD useful verdicts, zero
                               unnecessary, DOES promote.
  4. ONE BAD VERDICT BLOCKS -- a single "unnecessary" verdict, ever, blocks
                               promotion regardless of how many useful ones
                               came before or after (strict, one-sided).
     4b. CONSISTENTLY BAD   -- an always-"unnecessary" specialist never
                               accumulates a positive score.
  5. PROMOTED IS BUILT IN   -- a NEW pipeline instance, built AFTER
                               promotion, has the name in its topology from
                               construction -- no spawn needed.
  6. NO VERDICT IS NEUTRAL  -- a spawn with no verdict given doesn't count
                               toward promotion OR block it.
  7. TIER SURVIVES PROMOTION -- a promoted specialist keeps the tool-access
                               tier + subtask framing it earned as a dynamic
                               spawn, instead of silently falling through to
                               full CLAUDE_CODE_TOOLS on the top-level task.

    python test_structural_learning.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spiking_orchestrator as so   # noqa: E402


def _isolated_history_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="test_specialist_history_")
    os.close(fd)
    os.remove(path)   # start with no file, same as a fresh install
    return path


def _run_and_verdict(name: str, verdict: str, history_path: str,
                     subtask: str = None) -> None:
    """Simulate one full run that spawns `name` and gets judged `verdict`.

    `subtask` overrides the default spawn text -- used by check 7 to give a
    specialist a subtask whose tool_tier_for_subtask() tier is NOT the
    default 'code', so the promoted-path tier check is meaningful."""
    so.SPECIALIST_HISTORY = history_path
    p = so.SpikingPipeline(
        "rename x to y", dry_run=True,
        spawn_request=("Implementer", name, subtask or f"do the {name} work"),
        verdicts={name: verdict},
    )
    p.run()


def check_1_verdict_parsing() -> bool:
    history_path = _isolated_history_path()
    _run_and_verdict("DBMigrator", "useful", history_path)
    history = so.load_specialist_history()
    rec = history.get("DBMigrator", {})
    ok = (rec.get("spawned") == 1 and rec.get("judged_useful") == 1
          and rec.get("unnecessary", 0) == 0 and rec.get("score") == 1)
    print(f"  [{'PASS' if ok else 'FAIL'}] 1. verdict parsing: DBMigrator record={rec}")
    return ok


def check_2_below_threshold_no_promotion() -> bool:
    history_path = _isolated_history_path()
    for _ in range(so.PROMOTION_THRESHOLD - 1):
        _run_and_verdict("Almost", "useful", history_path)
    promoted = so.promoted_specialists()
    ok = "Almost" not in promoted
    print(f"  [{'PASS' if ok else 'FAIL'}] 2. below threshold ({so.PROMOTION_THRESHOLD - 1} useful) "
          f"does NOT promote: promoted={promoted}")
    return ok


def check_3_promotion_fires() -> bool:
    history_path = _isolated_history_path()
    for _ in range(so.PROMOTION_THRESHOLD):
        _run_and_verdict("Reliable", "useful", history_path)
    promoted = so.promoted_specialists()
    ok = "Reliable" in promoted
    print(f"  [{'PASS' if ok else 'FAIL'}] 3. {so.PROMOTION_THRESHOLD} useful verdicts DOES promote: "
          f"promoted={promoted}")
    return ok


def check_4_recovers_from_one_early_bad_verdict() -> bool:
    # CHANGED 2026-07-17: promotion is now a WEIGHTED running score, not a
    # strict AND-gate -- see the STRUCTURAL LEARNING comment in
    # spiking_orchestrator.py for the cross-referenced finding this is
    # based on (consensus_scoping_rule_ladder.md + this project's own
    # retracted soft-conflict result: weighted evidence beats strict
    # binary/ternary gating). This check now verifies the INTENDED recovery
    # behavior directly: one early unlucky 'unnecessary' verdict on a
    # genuinely good specialist should NOT permanently block it if enough
    # good evidence follows -- the bad verdict must land BEFORE promotion
    # ever fires (once promoted, a name is deduped against future spawns,
    # a separate, documented limitation -- not what this checks).
    history_path = _isolated_history_path()
    _run_and_verdict("Recovers", "unnecessary", history_path)
    for _ in range(so.PROMOTION_THRESHOLD + 2):
        _run_and_verdict("Recovers", "useful", history_path)
    promoted = so.promoted_specialists()
    ok = "Recovers" in promoted
    print(f"  [{'PASS' if ok else 'FAIL'}] 4. recovers from one early 'unnecessary' verdict given "
          f"{so.PROMOTION_THRESHOLD + 2} useful ones afterward: promoted={promoted}")
    return ok


def check_4b_consistently_bad_never_promotes() -> bool:
    # the weighted rule's safety property: a specialist that's ALWAYS judged
    # unnecessary must never accumulate a positive score, regardless of how
    # many times it's spawned.
    history_path = _isolated_history_path()
    for _ in range(so.PROMOTION_THRESHOLD + 3):
        _run_and_verdict("NeverGood", "unnecessary", history_path)
    promoted = so.promoted_specialists()
    ok = "NeverGood" not in promoted
    print(f"  [{'PASS' if ok else 'FAIL'}] 4b. consistently 'unnecessary' verdicts never promote "
          f"(safety holds under the weighted rule too): promoted={promoted}")
    return ok


def check_5_promoted_is_built_in() -> bool:
    history_path = _isolated_history_path()
    for _ in range(so.PROMOTION_THRESHOLD):
        _run_and_verdict("GraphQLExpert", "useful", history_path)
    # a FRESH pipeline, no spawn_request at all -- GraphQLExpert should
    # already exist in its topology from construction
    so.SPECIALIST_HISTORY = history_path
    p = so.SpikingPipeline("some other task entirely", dry_run=True)
    ok = "GraphQLExpert" in p._specialists and "GraphQLExpert" in p._base_specialist_names
    print(f"  [{'PASS' if ok else 'FAIL'}] 5. promoted specialist is built into a NEW pipeline "
          f"without a spawn: in _specialists={('GraphQLExpert' in p._specialists)}")
    return ok


def check_6_no_verdict_is_neutral() -> bool:
    history_path = _isolated_history_path()
    so.SPECIALIST_HISTORY = history_path
    # spawn WITHOUT providing a verdict for it
    p = so.SpikingPipeline(
        "rename x to y", dry_run=True,
        spawn_request=("Implementer", "Ambiguous", "do ambiguous work"),
        verdicts={},   # no verdict given at all
    )
    p.run()
    history = so.load_specialist_history()
    rec = history.get("Ambiguous", {})
    ok = rec.get("spawned") == 1 and rec.get("judged_useful", 0) == 0 and rec.get("unnecessary", 0) == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] 6. no verdict given is evidence-neutral: record={rec}")
    return ok


def check_7_promoted_specialist_keeps_its_tier_and_framing() -> bool:
    # REGRESSION (2026-09-07): a promoted specialist used to miss the
    # `neuron in self._specialist_tasks` branch in both _run_real_agent()
    # (tool-tier resolution) and _agent_task() (subtask framing) -- that
    # dict was only ever populated on the live-spawn path. Net effect: a
    # specialist spawned to do read-only work (e.g. a security audit, which
    # tool_tier_for_subtask() maps to the 'review' tier) silently ran with
    # the full CLAUDE_CODE_TOOLS set on the TOP-LEVEL task once promoted --
    # the exact tool-access mismatch tool_tier_for_subtask() exists to
    # prevent. Fixed by persisting the subtask at verdict time
    # (_record_specialist_outcomes) and re-seeding _specialist_tasks for
    # promoted names in _build_brain(). Same bug shape as the 2026-09-06
    # live-mode Resonator misrouting.
    history_path = _isolated_history_path()
    audit_subtask = "audit the authentication layer for security problems"
    # sanity: this subtask must resolve to a NON-default tier, or the check
    # below proves nothing
    assert so.tool_tier_for_subtask(audit_subtask) == "review", \
        "test premise broken: audit subtask should map to the 'review' tier"
    for _ in range(so.PROMOTION_THRESHOLD):
        _run_and_verdict("SecAudit", "useful", history_path, subtask=audit_subtask)

    so.SPECIALIST_HISTORY = history_path
    p = so.SpikingPipeline("rename x to y", dry_run=True)   # fresh, no spawn

    built_in = "SecAudit" in p._specialists and "SecAudit" in p._base_specialist_names
    subtask_seeded = p._specialist_tasks.get("SecAudit") == audit_subtask
    # the thing _run_real_agent() actually keys on: a promoted specialist is
    # now recognised, and its tier resolves read-only, not full-edit
    tier_ok = ("SecAudit" in p._specialist_tasks
               and so.tool_tier_for_subtask(p._specialist_tasks["SecAudit"]) == "review")
    # _agent_task() frames it on its OWN subtask, not the top-level task
    framed = p._agent_task("SecAudit")
    framing_ok = "authentication layer" in framed and "rename x to y" not in framed

    ok = built_in and subtask_seeded and tier_ok and framing_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] 7. promoted specialist keeps its tier + framing: "
          f"built_in={built_in} subtask_seeded={subtask_seeded} "
          f"tier_ok={tier_ok} framing_ok={framing_ok}")
    return ok


def run() -> None:
    print("=" * 74)
    print("  STRUCTURAL LEARNING -- promotion of proven dynamic specialists")
    print("=" * 74)
    original_path = so.SPECIALIST_HISTORY
    try:
        results = [
            check_1_verdict_parsing(),
            check_2_below_threshold_no_promotion(),
            check_3_promotion_fires(),
            check_4_recovers_from_one_early_bad_verdict(),
            check_4b_consistently_bad_never_promotes(),
            check_5_promoted_is_built_in(),
            check_6_no_verdict_is_neutral(),
            check_7_promoted_specialist_keeps_its_tier_and_framing(),
        ]
    finally:
        so.SPECIALIST_HISTORY = original_path   # never leave the real path swapped out

    print()
    if all(results):
        print(f"  ALL {len(results)} CHECKS PASS: the pipeline's base topology genuinely "
              f"learns from real usage history, not just per-run improvisation.")
    else:
        print(f"  {results.count(False)}/{len(results)} CHECKS FAILED -- do not trust "
              f"structural learning until resolved.")


if __name__ == "__main__":
    run()
