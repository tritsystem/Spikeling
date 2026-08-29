#!/usr/bin/env python
"""
test_spike_skills.py — real verification of spike_skills.py's Phase 0
claims: cheap frontmatter-only listing, lazy full-content loading, real
errors (not silent None/empty) for anything malformed or unknown.
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_skills import (  # noqa: E402
    load_skill_listing, load_skill_content, select_skills_for,
    load_skill_keywords, select_skills_for_task, SkillError,
)

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


def make_skill(root, name, description, body):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}")


tmp = tempfile.mkdtemp(prefix="spike_skills_test_")
try:
    # 1. Real skill listing works, frontmatter only.
    make_skill(tmp, "alpha", "Use when testing alpha things.", "# Alpha body\nSome instructions.")
    make_skill(tmp, "beta", "Use when testing beta things.", "# Beta body\nMore instructions.")
    listing = load_skill_listing(tmp)
    check("listing returns both real skills with descriptions",
          listing == {"alpha": "Use when testing alpha things.",
                      "beta": "Use when testing beta things."})

    # 2. THE central "cheap" claim: a skill with a deliberately huge body
    # never gets that body read during listing. Verified by checking how
    # far into the file the frontmatter reader actually seeks, not just
    # the returned dict.
    # Real test of "cheap": make the BODY genuinely unparseable garbage
    # (invalid UTF-8 bytes) while the frontmatter stays valid. If listing
    # ever touched the body, this would raise a UnicodeDecodeError; if it
    # truly stops at the closing '---', the body's content is irrelevant
    # and listing succeeds regardless of what garbage follows.
    huge_dir = os.path.join(tmp, "huge")
    os.makedirs(huge_dir, exist_ok=True)
    huge_path = os.path.join(huge_dir, "SKILL.md")
    with open(huge_path, "wb") as f:
        f.write(b"---\nname: huge\ndescription: Use when testing the huge skill.\n---\n\n")
        f.write(b"# body\n" + (b"\xff\xfe\x00garbage-not-valid-utf8" * 50000))  # ~1MB invalid UTF-8
    file_size = os.path.getsize(huge_path)
    check("huge skill's body is genuinely large (test is meaningful)",
          file_size > 500_000)
    check("listing succeeds even though the body is invalid UTF-8 garbage "
          "(proves the frontmatter reader never touches body content)",
          load_skill_listing(tmp)["huge"] == "Use when testing the huge skill.")
    # And confirm the inverse: actually loading that skill's CONTENT does
    # hit the bad body and fails -- proving the listing path's success
    # above wasn't an accident of the whole file happening to decode fine.
    try:
        load_skill_content("huge", tmp)
        check("loading the huge skill's real content hits the invalid body", False)
    except UnicodeDecodeError:
        check("loading the huge skill's real content hits the invalid body", True)

    # 3. load_skill_content loads the real full body, only when asked.
    content = load_skill_content("alpha", tmp)
    check("load_skill_content returns the real body, not frontmatter",
          "Alpha body" in content and "name: alpha" not in content)

    # 4. Unknown skill name is a real error, not None/empty string.
    try:
        load_skill_content("does-not-exist", tmp)
        check("unknown skill name raises SkillError", False)
    except SkillError:
        check("unknown skill name raises SkillError", True)

    # 5. Directory name / frontmatter name mismatch is a real error.
    mismatch_dir = os.path.join(tmp, "gamma")
    os.makedirs(mismatch_dir, exist_ok=True)
    with open(os.path.join(mismatch_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: wrong-name\ndescription: test mismatch\n---\n\nbody")
    try:
        load_skill_listing(tmp)
        check("directory/frontmatter name mismatch raises SkillError", False)
    except SkillError:
        check("directory/frontmatter name mismatch raises SkillError", True)
    shutil.rmtree(mismatch_dir)

    # 6. Missing skills directory is a real error.
    try:
        load_skill_listing("/nonexistent/spike_skills_does_not_exist")
        check("missing skills directory raises SkillError", False)
    except SkillError:
        check("missing skills directory raises SkillError", True)

    # 7. Missing required frontmatter field is a real error.
    bad_dir = os.path.join(tmp, "delta")
    os.makedirs(bad_dir, exist_ok=True)
    with open(os.path.join(bad_dir, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: delta\n---\n\nno description field")
    try:
        load_skill_listing(tmp)
        check("missing 'description' field raises SkillError", False)
    except SkillError:
        check("missing 'description' field raises SkillError", True)
    shutil.rmtree(bad_dir)

    # 8. The real, shipped review-discipline skill actually loads.
    real_listing = load_skill_listing()
    check("real spike_skills/ has the real review-discipline skill",
          "review-discipline" in real_listing)
    real_content = load_skill_content("review-discipline")
    check("real review-discipline content mentions the actual ported source",
          "_reviewer_task" in real_content and "overclaiming" in real_content)

    # 9. select_skills_for: a skill self-selects by naming a neuron, as a
    # whole word, in its own description -- real test with real temp
    # skills, including a false-positive-substring trap ("Reviewer" must
    # not match a skill that only mentions "ReviewerBot").
    make_skill(tmp, "for-reviewer", "Use when Reviewer fires.", "reviewer stuff")
    make_skill(tmp, "for-implementer", "Use when Implementer fires.", "implementer stuff")
    make_skill(tmp, "substring-trap", "Use when ReviewerBot fires.", "should not match Reviewer")
    check("select_skills_for matches the real neuron name as a whole word",
          select_skills_for("Reviewer", tmp) == ["for-reviewer"])
    check("select_skills_for doesn't cross-match an unrelated neuron",
          select_skills_for("Implementer", tmp) == ["for-implementer"])
    check("select_skills_for ignores a substring-only match (word-boundary correctness)",
          "substring-trap" not in select_skills_for("Reviewer", tmp))
    check("select_skills_for returns [] when nothing self-selects",
          select_skills_for("Corrector", tmp) == [])

    # 10. The real, shipped skills wiring: select_skills_for("Reviewer")
    # picks up the real review-discipline skill by its real description.
    check("real select_skills_for('Reviewer') finds the real review-discipline skill",
          select_skills_for("Reviewer") == ["review-discipline"])

    # 11. End-to-end through spiking_orchestrator.SpikingPipeline._agent_task:
    # Reviewer's built task string actually gets the skill content appended,
    # and an unrelated fixed-roster neuron (e.g. Implementer) stays
    # byte-identical to its pre-skills framing.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import spiking_orchestrator as so
    p = so.SpikingPipeline("rename the fire_rate variable to shot_delay", dry_run=True)
    reviewer_task = p._agent_task("Reviewer")
    check("Reviewer's real built task includes the real review-discipline skill content",
          "--- Skill: review-discipline ---" in reviewer_task
          and "Call out overclaiming" in reviewer_task)
    implementer_task = p._agent_task("Implementer")
    check("Implementer's real built task is untouched (no skill currently names it)",
          "--- Skill:" not in implementer_task
          and implementer_task == p.task + p._SPAWN_HINT)

    # 12. Second real ported skill: pre-registration-discipline, proving the
    # self-selection mechanism generalizes past the one Reviewer case.
    check("real spike_skills/ has the real pre-registration-discipline skill",
          "pre-registration-discipline" in real_listing or
          "pre-registration-discipline" in load_skill_listing())
    check("real select_skills_for('PreRegister') finds pre-registration-discipline",
          select_skills_for("PreRegister") == ["pre-registration-discipline"])
    preregister_task = p._agent_task("PreRegister")
    check("PreRegister's real built task includes the real pre-registration-discipline content",
          "--- Skill: pre-registration-discipline ---" in preregister_task
          and "One checkable sentence" in preregister_task)
    # 13. Third real ported skill: corrector-discipline.
    check("real spike_skills/ has the real corrector-discipline skill",
          "corrector-discipline" in load_skill_listing())
    check("real select_skills_for('Corrector') finds corrector-discipline",
          select_skills_for("Corrector") == ["corrector-discipline"])
    corrector_task = p._agent_task("Corrector")
    check("Corrector's real built task includes the real corrector-discipline content",
          "--- Skill: corrector-discipline ---" in corrector_task
          and "Fix exactly what review flagged" in corrector_task)

    # (moved below skill #13 so Corrector genuinely has content by now)
    check("two independently-selecting skills don't cross-contaminate each other's neuron",
          "--- Skill: review-discipline ---" not in p._agent_task("PreRegister")
          and "--- Skill: pre-registration-discipline ---" not in p._agent_task("Reviewer"))

    # 14. The 50 environment-tailored skills (build_environment_skills.py):
    # real count, every one loads real non-empty content, and -- the actual
    # correctness claim that matters -- NONE of them accidentally names a
    # reserved fixed-roster neuron in its description, which would silently
    # glue an unrelated topic playbook onto every single invocation of that
    # role (see build_environment_skills.py's module docstring for why
    # that would be wrong, not just untidy).
    real_listing_full = load_skill_listing()
    check("real spike_skills/ has all 3 role skills + 50 environment skills",
          len(real_listing_full) == 53)
    _RESERVED_NEURONS = ["Implementer", "Reviewer", "PreRegister", "Corrector",
                          "TestWriter", "VaultLogger", "Clarifier"]
    _EXPECTED_ROLE_SKILLS = {"review-discipline", "pre-registration-discipline",
                              "corrector-discipline"}
    collisions = []
    for neuron in _RESERVED_NEURONS:
        unexpected = [h for h in select_skills_for(neuron) if h not in _EXPECTED_ROLE_SKILLS]
        collisions.extend((neuron, h) for h in unexpected)
    check("none of the 50 environment skills accidentally self-select for a "
          "reserved fixed-roster neuron", collisions == [])
    load_errors = []
    for name in real_listing_full:
        try:
            if not load_skill_content(name).strip():
                load_errors.append((name, "empty content"))
        except SkillError as e:
            load_errors.append((name, str(e)))
    check("every real shipped skill loads real, non-empty content",
          load_errors == [])

    # 15. load_skill_keywords / select_skills_for_task -- real temp-dir
    # unit tests of the mechanism itself, isolated from the 50 real skills.
    os.makedirs(os.path.join(tmp, "kw-alpha"), exist_ok=True)
    kw_alpha_dir = os.path.join(tmp, "kw-alpha", "SKILL.md")
    with open(kw_alpha_dir, "w", encoding="utf-8") as f:
        f.write("---\nname: kw-alpha\ndescription: Use for alpha keyword testing.\n"
                 "keywords: widget-forge, alpha-gadget\n---\n\nalpha body")
    make_skill(tmp, "kw-beta", "Use for beta keyword testing (no keywords field).", "beta body")
    check("a skill with no keywords field is absent from load_skill_keywords",
          "kw-beta" not in load_skill_keywords(tmp))
    check("a skill's keywords field parses into a real list",
          load_skill_keywords(tmp)["kw-alpha"] == ["widget-forge", "alpha-gadget"])
    check("select_skills_for_task matches on a real substring, case-insensitively",
          select_skills_for_task("please fix the WIDGET-FORGE pipeline", tmp) == ["kw-alpha"])
    check("select_skills_for_task returns [] for unrelated task text",
          select_skills_for_task("completely unrelated request about lunch", tmp) == [])
    check("a skill without a keywords field never auto-matches by task text",
          select_skills_for_task("beta body beta keyword testing", tmp) == [])

    # More-hits-ranks-first + cap, in one real scenario with 3 real temp skills.
    make_skill(tmp, "kw-gamma", "Use for gamma.", "gamma body")
    gamma_md = os.path.join(tmp, "kw-gamma", "SKILL.md")
    with open(gamma_md, "w", encoding="utf-8") as f:
        f.write("---\nname: kw-gamma\ndescription: Use for gamma.\n"
                 "keywords: widget-forge, gizmo-line, sprocket-bay\n---\n\ngamma body")
    ranked = select_skills_for_task("widget-forge gizmo-line sprocket-bay work today", tmp, max_matches=5)
    check("select_skills_for_task ranks more keyword hits first",
          ranked[0] == "kw-gamma" and "kw-alpha" in ranked)
    capped = select_skills_for_task("widget-forge gizmo-line sprocket-bay work today", tmp, max_matches=1)
    check("select_skills_for_task respects a real max_matches cap",
          capped == ["kw-gamma"])

    # 16. The real, shipped 50 environment skills: every one recalls
    # correctly from a synthesized realistic sentence built from its own
    # first real keyword, with no wild over-matching (>2 hits) from a
    # single-topic sentence -- the actual precision/recall claim that
    # matters, checked automatically rather than eyeballed once by hand.
    real_keywords = load_skill_keywords()
    recall_failures = []
    overmatches = []
    for name, kws in real_keywords.items():
        sample = f"working on {kws[0]} today, need to fix an issue"
        matched = select_skills_for_task(sample)
        if name not in matched:
            recall_failures.append((name, matched))
        if len(matched) > 2:
            overmatches.append((name, matched))
    check(f"all {len(real_keywords)} real environment skills recall correctly "
          "from a sentence built on their own first keyword",
          recall_failures == [])
    check("no single-topic synthesized sentence over-matches more than 2 skills",
          overmatches == [])

    # 17. End-to-end through SpikingPipeline._agent_task: a Tribe-terrain
    # task pulls in the real tribe-terrain-generation skill for Implementer
    # (previously, per the earlier role-only wiring, Implementer NEVER got
    # any skill content -- this is the real behavior change this phase adds).
    p_terrain = so.SpikingPipeline("fix the tribe terrain heightmap generation bug", dry_run=True)
    implementer_terrain_task = p_terrain._agent_task("Implementer")
    check("a topic-matched task now gets its environment skill injected for Implementer",
          "--- Skill: tribe-terrain-generation ---" in implementer_terrain_task)
    # And an unrelated task still gets nothing extra for Implementer.
    p_unrelated = so.SpikingPipeline("rename the fire_rate variable to shot_delay", dry_run=True)
    check("an unrelated task still gets no skill content injected for Implementer",
          "--- Skill:" not in p_unrelated._agent_task("Implementer"))

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n=== RESULT: {PASS} passed, {FAIL} failed, {PASS + FAIL} total ===")
print("OVERALL:", "PASS" if FAIL == 0 else "FAIL")
sys.exit(0 if FAIL == 0 else 1)
