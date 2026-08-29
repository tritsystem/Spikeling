---
name: multi-repo-git-discipline
description: Use before any push, force-push, or other destructive git action across this multi-repo portfolio.
keywords: force-push, divergent remote history, multi-repo git
---

# Multi-repo git discipline

This portfolio spans many separate repos, and a past session got the
ahead/behind relationship backwards more than once before catching it.
The rule: check the repo's ACTUAL current state before any push/force/
destructive git action -- some repos have real uncommitted work, and a
couple have genuinely divergent remote history. Never assume "ahead" or
"behind" from memory or from how a similar repo in this portfolio
usually behaves; check this specific repo, this specific time.
