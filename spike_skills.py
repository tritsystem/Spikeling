#!/usr/bin/env python
"""
spike_skills.py — Spike's skills layer, Phase 0.

Reference: Claude Code's own real Skills format, inspected directly on
this machine (C:\\Users\\gbran\\.claude\\plugins\\cache\\mattpocock\\
mattpocock-skills\\1.2.3\\skills\\engineering\\diagnosing-bugs\\SKILL.md).
Real, confirmed shape: one directory per skill, a SKILL.md with YAML
frontmatter (`name`, `description` -- nothing more), then a plain
Markdown body. The description doubles as the trigger condition ("Use
when...") -- one field, not two kept in sync by hand.

Two-phase loading, on purpose (see vault/Projects/spike-skills-system.md):
  - load_skill_listing() reads ONLY the frontmatter of every SKILL.md --
    cheap, safe to call often, used to decide WHICH skill fits a task.
  - load_skill_content(name) loads one skill's full body, only when that
    skill is actually being used.

Skill SELECTION is deliberately NOT config-driven, unlike
spike_tool_gateway.py's provider routing. Which skill fits a given task
is a judgment call (matching a description to what's being asked), not a
fixed category->value mapping -- forcing it through the gateway's
"config always wins" pattern would copy the wrong lesson from the wrong
system. See the scope doc for the full reasoning.
"""

import os
import re

_SKILLS_DIRNAME = "spike_skills"


class SkillError(Exception):
    """Raised for anything that isn't a clean, real skill: unknown name,
    malformed frontmatter, missing SKILL.md. No silent None/empty-string
    fallback -- same "fail loud, don't guess" discipline as
    spike_tool_gateway.ToolGatewayError."""


def _skills_root(skills_dir: str = None) -> str:
    return skills_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), _SKILLS_DIRNAME)


def _read_frontmatter(path: str) -> dict:
    """Reads ONLY the YAML frontmatter block (between the two `---`
    lines) -- stops as soon as the closing delimiter is seen, never reads
    the Markdown body. This is the actual "cheap" claim being made, not
    just "returns the right dict" -- verified in test_spike_skills.py.

    Opened in BINARY mode and decoded one line at a time, deliberately --
    a text-mode open() with a fixed encoding decodes ahead in whatever
    chunk its internal buffer reads (found via a real test: a valid
    frontmatter block followed by invalid-UTF-8 body bytes within the
    same buffer chunk raised UnicodeDecodeError before this fix, even
    though the loop below never asks for those bytes). Binary + per-line
    decode means a garbage body genuinely can't affect frontmatter
    reading, not just usually can't."""
    with open(path, "rb") as f:
        first = f.readline().decode("utf-8")
        if first.strip() != "---":
            raise SkillError(f"{path}: missing opening '---' frontmatter delimiter")
        lines = []
        for raw_line in f:
            line = raw_line.decode("utf-8")
            if line.strip() == "---":
                break
            lines.append(line)
        else:
            raise SkillError(f"{path}: missing closing '---' frontmatter delimiter")
    meta = {}
    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            raise SkillError(f"{path}: malformed frontmatter line: {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    for required in ("name", "description"):
        if required not in meta:
            raise SkillError(f"{path}: frontmatter missing required field '{required}'")
    return meta


def load_skill_listing(skills_dir: str = None) -> dict:
    """Scans spike_skills/<name>/SKILL.md for every skill directory, reads
    ONLY the frontmatter of each, returns {name: description}. Real error
    if a skill directory's name doesn't match its own frontmatter `name`
    field -- catches copy-paste drift early rather than silently trusting
    whichever one is checked."""
    root = _skills_root(skills_dir)
    if not os.path.isdir(root):
        raise SkillError(f"skills directory not found: {root}")
    listing = {}
    for entry in sorted(os.listdir(root)):
        skill_dir = os.path.join(root, entry)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        meta = _read_frontmatter(skill_md)
        if meta["name"] != entry:
            raise SkillError(
                f"{skill_md}: frontmatter name '{meta['name']}' doesn't match "
                f"its directory name '{entry}'"
            )
        listing[meta["name"]] = meta["description"]
    return listing


def load_skill_content(name: str, skills_dir: str = None) -> str:
    """Loads one skill's full Markdown body (everything after the closing
    frontmatter delimiter). Real SkillError for an unknown name -- not
    None, not ''."""
    root = _skills_root(skills_dir)
    skill_md = os.path.join(root, name, "SKILL.md")
    if not os.path.isfile(skill_md):
        raise SkillError(f"no such skill: '{name}' (looked for {skill_md})")
    with open(skill_md, "r", encoding="utf-8") as f:
        text = f.read()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError(f"{skill_md}: malformed -- expected frontmatter then body")
    return parts[2].lstrip("\n")


def load_skill_keywords(skills_dir: str = None) -> dict:
    """Returns {name: [keyword, ...]} for every skill that declares an
    optional `keywords` frontmatter field (comma-separated). A skill with
    no `keywords` field is simply absent from this dict -- it never
    auto-matches by task text. This is correct, not a gap, for the 3
    role skills (they select by neuron name via select_skills_for, not
    task content) and for any future skill that hasn't been given real
    trigger terms yet -- silence here means "not wired for topic
    matching," never "matches everything."""
    listing = load_skill_listing(skills_dir)  # also validates dir/name consistency
    root = _skills_root(skills_dir)
    keywords = {}
    for name in listing:
        meta = _read_frontmatter(os.path.join(root, name, "SKILL.md"))
        raw = meta.get("keywords")
        if raw:
            keywords[name] = [k.strip() for k in raw.split(",") if k.strip()]
    return keywords


def select_skills_for_task(task_text: str, skills_dir: str = None, max_matches: int = 3) -> list:
    """Real, deterministic, non-LLM keyword matcher for topic-scoped
    skills -- the task-text-matching phase deliberately deferred out of
    the earlier wiring pass (see vault/Projects/spike-skills-system.md).

    Unlike select_skills_for() (role selection, which mirrors Claude
    Code's own description-as-trigger format faithfully), there's no LLM
    reading descriptions and making a judgment call here -- a specialist
    fires without a dedicated "which skill fits" reasoning step, and
    adding one would mean a whole extra API call just to pick a skill
    before the real one runs. So this is plain, auditable, case-
    insensitive SUBSTRING matching against each skill's own declared
    `keywords` frontmatter field -- not word-boundary regex, since most
    of these keywords are technical terms/paths (`core/runtime/
    runtime.py`, `SPIKE_ANTHROPIC_KEY`) that regex word boundaries don't
    handle cleanly, and a skill's own keyword list is deliberately
    narrow/specific by author intent, so plain substring matching is
    precise enough without the added complexity.

    Ranked by number of matching keywords (most specific match first),
    alphabetical name as the tiebreak, then capped at `max_matches`
    (default 3) -- a real, deliberate truncation so a task that happens
    to hit many keywords still gets a small, focused set of skills
    rather than everything at once. Not silently unaccountable: a
    caller that needs to see what was dropped can pass a higher
    max_matches directly."""
    keywords = load_skill_keywords(skills_dir)
    low = task_text.lower()
    scored = []
    for name, kws in keywords.items():
        hits = sum(1 for kw in kws if kw.lower() in low)
        if hits:
            scored.append((hits, name))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [name for _, name in scored[:max_matches]]


def select_skills_for(neuron: str, skills_dir: str = None) -> list:
    """Returns the names of skills that apply to `neuron`, in listing
    (alphabetical) order.

    Selection mechanism, deliberately NOT config-driven -- unlike
    spike_tool_gateway.py's provider routing, where a category maps to
    exactly one configured value and "auto" is a hard error. Which skill
    fits a given specialist isn't that kind of fixed-lookup problem (see
    vault/Projects/spike-skills-system.md), so there's no
    spike_skills.yaml here on purpose.

    Instead, a skill opts ITSELF into a specialist by naming that
    specialist, as a whole word, in its own description -- e.g.
    review-discipline's real description literally says "Use when
    Spike's Reviewer specialist fires...", so it self-selects for
    neuron="Reviewer" with no second mapping anywhere else that could
    drift out of sync with what the file itself claims. If a skill's
    description doesn't name a neuron, it just never auto-selects (it's
    still listed, still loadable by name -- this only affects the
    automatic per-specialist wiring)."""
    listing = load_skill_listing(skills_dir)
    pattern = re.compile(rf"\b{re.escape(neuron)}\b")
    return [name for name, description in listing.items() if pattern.search(description)]


if __name__ == "__main__":
    listing = load_skill_listing()
    print(f"{len(listing)} skill(s) found:")
    for name, desc in listing.items():
        print(f"  {name}: {desc}")
