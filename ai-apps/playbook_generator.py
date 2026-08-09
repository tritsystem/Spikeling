"""Generates a Playbook-style study page for ANY project, the vault, or a
knowledge index -- not the hand-researched, hand-verified original (that
took real, individual research per chapter). This is real content, grounded
in actually-fetched source material (a real README, a real function pulled
from a real file, real vault notes), but the Q&A and explanations are
LLM-generated from that material, not independently verified line by line.
Every generated page says so plainly, in the page itself -- the same
disclosure discipline as everything else in this stack.

Reuses:
  - project_manager.KNOWN_PROJECTS for real project paths
  - Ollama (mesh_rag_server's own GENERATE_MODEL) for the actual generation
  - vault_engine (when available) for real vault-note retrieval
"""
import json
import os
import re
import urllib.request

import project_manager as pm

OLLAMA_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
GENERATE_MODEL = os.environ.get("SPIKEMESH_MESH_MODEL", "llava:7b")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_playbooks")

CODE_EXTS = {".py", ".gd", ".rs", ".js", ".ts", ".sv"}
PREFERRED_NAME_HINTS = ["main", "core", "server", "engine", "runtime", "app"]


def _ollama_generate(prompt, max_chars=4000):
    # Real bug, found by reproducing a live "stuck" report: this request set no
    # num_predict, so generation length was bounded only by the model's own EOS
    # decision (or its context window) -- "keep it under 300 words" in the prompt
    # is a suggestion the model doesn't reliably obey, not an enforced limit. On
    # a real run explaining a real code file, that let generation run long enough
    # to blow past the 120s timeout below and 500 the whole request. Capping
    # num_predict makes worst-case runtime predictable instead of open-ended.
    body = json.dumps({
        "model": GENERATE_MODEL, "prompt": prompt[:max_chars], "stream": False,
        "options": {"num_predict": 600},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


def _read_readme(project_path):
    for name in ("README.md", "readme.md", "README.txt", "README"):
        p = os.path.join(project_path, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                return f.read()[:6000]
    return None


def _find_representative_code(project_path):
    """Real, simple heuristic: prefer files whose name hints at being a
    core/entry file, then just take the first code file with a real
    function definition long enough to be worth walking through."""
    candidates = []
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}
                   and not d.startswith(".")]
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in CODE_EXTS:
                candidates.append(os.path.join(root, fn))
        if len(candidates) > 300:  # real safety bound -- don't walk a huge repo forever
            break

    def score(path):
        name = os.path.basename(path).lower()
        return sum(1 for hint in PREFERRED_NAME_HINTS if hint in name)

    candidates.sort(key=score, reverse=True)
    for path in candidates[:40]:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        if re.search(r"\bdef \w+\(|\bfunc \w+\(|\bfn \w+\(", text) and 200 < len(text) < 8000:
            return path, text
    return None, None


def list_real_code_files(project_name, limit=200):
    """Real candidate files for the UI's file picker -- not a single
    auto-guessed file, an actual listing the user chooses from."""
    if project_name not in pm.KNOWN_PROJECTS:
        return {"error": f"unknown project '{project_name}'"}
    path = pm.KNOWN_PROJECTS[project_name]
    if not os.path.isdir(path):
        return {"error": f"project path not found: {path}"}

    files = []
    for root, dirs, fnames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}
                   and not d.startswith(".")]
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in CODE_EXTS:
                rel = os.path.relpath(os.path.join(root, fn), path)
                try:
                    size = os.path.getsize(os.path.join(root, fn))
                except OSError:
                    size = 0
                files.append({"path": rel.replace("\\", "/"), "size": size})
        if len(files) >= limit:
            break
    files.sort(key=lambda f: f["path"])
    return {"project": project_name, "files": files}


def generate_project_playbook(project_name, code_file=None):
    """code_file: an optional real, user-chosen relative path (from
    list_real_code_files) -- when given, this is used directly instead of
    the auto-selection heuristic. Explicit user choice always wins."""
    if project_name not in pm.KNOWN_PROJECTS:
        return {"error": f"unknown project '{project_name}'"}
    path = pm.KNOWN_PROJECTS[project_name]
    if not os.path.isdir(path):
        return {"error": f"project path not found: {path}"}

    readme = _read_readme(path)

    if code_file:
        abs_path = os.path.normpath(os.path.join(path, code_file))
        # Real safety check: the resolved path must stay inside the
        # project directory -- refuse a "../../whatever" escape attempt
        # rather than silently reading outside the intended project.
        if not abs_path.startswith(os.path.normpath(path)):
            return {"error": "invalid code_file path (escapes the project directory)"}
        if not os.path.isfile(abs_path):
            return {"error": f"file not found: {code_file}"}
        with open(abs_path, encoding="utf-8", errors="ignore") as f:
            code_text = f.read()[:8000]
        code_path = abs_path
    else:
        code_path, code_text = _find_representative_code(path)

    sources = []
    if readme:
        sources.append("README.md")
    if code_path:
        sources.append(os.path.relpath(code_path, path))

    if not readme and not code_text:
        return {"error": "no README or suitable code file found to ground generation in"}

    qa_prompt = (
        "You are generating real study material about a real software project, grounded ONLY in "
        "the material below. Write exactly 4 question-and-answer pairs a learner could use to test "
        "themselves, in this exact format (one pair per block, separated by blank lines):\n"
        "Q: <question>\nA: <answer, grounded in the material, 2-3 sentences>\n\n"
        f"MATERIAL:\n{readme or '(no README found)'}\n"
    )
    qa_raw = _ollama_generate(qa_prompt)
    qa_pairs = []
    for block in re.split(r"\n\s*\n", qa_raw):
        m = re.search(r"Q:\s*(.+?)\nA:\s*(.+)", block, re.DOTALL)
        if m:
            qa_pairs.append({"q": m.group(1).strip(), "a": m.group(2).strip()})

    code_explanation = None
    if code_text:
        code_prompt = (
            "Explain this real code, function by function, in plain language for someone learning "
            "to program. Be concrete about what each part actually does. Keep it under 300 words.\n\n"
            f"FILE: {os.path.basename(code_path)}\n\nCODE:\n{code_text[:3000]}"
        )
        code_explanation = _ollama_generate(code_prompt)

    return {
        "project": project_name,
        "sources": sources,
        "readme_used": bool(readme),
        "qa_pairs": qa_pairs,
        "code_file": os.path.relpath(code_path, path) if code_path else None,
        "code_text": code_text[:3000] if code_text else None,
        "code_explanation": code_explanation,
    }


def generate_vault_playbook(query, k=6):
    """Study material grounded in real, actually-retrieved vault notes for
    a topic -- not the whole vault, a real semantic slice of it."""
    try:
        import sys
        sys.path.insert(0, r"G:\dev\observe-api")
        from search_engine import SearchEngine
        engine = SearchEngine()
        engine.load_blocking(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_semantic_index"),
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        hits = engine.search(query, k=k) if engine.ready else []
    except Exception as e:
        return {"error": f"vault search unavailable: {e}"}

    if not hits:
        return {"error": "no real vault notes matched this topic"}

    material = "\n\n".join(f"[{h['path']}] {h['preview']}" for h in hits)
    qa_prompt = (
        "You are generating real study material about a topic, grounded ONLY in the real notes "
        "below. Write exactly 4 question-and-answer pairs, format:\n"
        "Q: <question>\nA: <answer, 2-3 sentences, cite what it's grounded in>\n\n"
        f"REAL NOTES:\n{material}\n"
    )
    qa_raw = _ollama_generate(qa_prompt)
    qa_pairs = []
    for block in re.split(r"\n\s*\n", qa_raw):
        m = re.search(r"Q:\s*(.+?)\nA:\s*(.+)", block, re.DOTALL)
        if m:
            qa_pairs.append({"q": m.group(1).strip(), "a": m.group(2).strip()})

    return {
        "topic": query,
        "sources": [h["path"] for h in hits],
        "qa_pairs": qa_pairs,
    }


def render_html(data, title):
    """Renders into the same visual language as the original Playbook
    (dark, mono headings, details/summary reveal) -- but honestly labeled
    as AI-generated, not hand-verified."""
    qa_html = ""
    for pair in data.get("qa_pairs", []):
        qa_html += (
            '<details class="qa"><summary>' + _esc(pair["q"]) + '</summary>'
            '<div class="a">' + _esc(pair["a"]) + '</div></details>\n'
        )

    code_block = ""
    if data.get("code_text"):
        code_block = (
            '<h3 class="sub">Real code from this project (' + _esc(data.get("code_file", "")) + ')</h3>'
            '<pre style="background:var(--panel); border:1px solid var(--rule); border-radius:8px; '
            'padding:14px; overflow-x:auto; font-family:var(--mono); font-size:12.5px;">'
            + _esc(data["code_text"]) + '</pre>'
            '<div class="answer">' + _esc(data.get("code_explanation") or "") + '</div>'
        )

    sources = ", ".join(data.get("sources", []))

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} -- generated playbook</title>
<style>
  :root {{ --bg:#0b1210; --panel:#101a15; --ink:#e8efe9; --ink-dim:#9db2a5; --ink-faint:#61756a;
    --rule:#22322a; --accent:#e3a159; --mono: ui-monospace, "Cascadia Mono", "SF Mono", Consolas, monospace;
    --sans: -apple-system, "Segoe UI", Arial, sans-serif; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); line-height:1.6; }}
  main {{ max-width:760px; margin:0 auto; padding:40px 22px 100px; }}
  h1 {{ font-family:var(--mono); font-size:26px; }}
  .eyebrow {{ font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--accent); }}
  .disclosure {{ border:1px solid var(--rule); border-left:3px solid var(--accent); border-radius:0 8px 8px 0;
    background:var(--panel); padding:14px 18px; font-size:13.5px; color:var(--ink-dim); margin:18px 0 28px; }}
  h3.sub {{ font-family:var(--mono); font-size:13px; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-faint); margin:32px 0 12px; }}
  details.qa {{ border:1px solid var(--rule); border-radius:10px; background:var(--panel); margin-bottom:10px; overflow:hidden; }}
  details.qa summary {{ padding:14px 18px; cursor:pointer; font-weight:600; list-style:none; }}
  details.qa summary::-webkit-details-marker {{ display:none; }}
  details.qa[open] summary {{ border-bottom:1px solid var(--rule); }}
  details.qa .a {{ padding:14px 18px 18px; font-size:14.5px; color:var(--ink-dim); }}
  .answer {{ background:var(--panel); border:1px solid var(--rule); border-left:3px solid var(--accent);
    border-radius:0 8px 8px 0; padding:16px 18px; font-size:14.5px; color:var(--ink-dim); white-space:pre-wrap; margin-top:12px; }}
</style></head><body><main>
  <div class="eyebrow">Generated by SPIKEMESH's Playbook Generator</div>
  <h1>{_esc(title)}</h1>
  <div class="disclosure"><b>AI-generated study material.</b> Grounded in real source ({_esc(sources) or 'none found'}),
    but the questions/answers/explanations below were written by a local LLM (llava:7b) from that material,
    not independently hand-verified line by line like the original portfolio Playbook. Treat this as a real
    starting point for study, not a fact-checked reference -- verify anything load-bearing yourself.</div>
  <h3 class="sub">Study questions</h3>
  {qa_html or '<p style="color:var(--ink-faint);">No question/answer pairs were generated.</p>'}
  {code_block}
</main></body></html>"""


def _esc(s):
    return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_and_save(target_type, target_name, code_file=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if target_type == "project":
        data = generate_project_playbook(target_name, code_file=code_file)
        title = f"Playbook: {target_name}"
    elif target_type == "vault":
        data = generate_vault_playbook(target_name)
        title = f"Playbook: {target_name} (vault)"
    else:
        return {"error": f"unknown target_type '{target_type}'"}

    if "error" in data:
        return data

    html = render_html(data, title)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{target_type}_{target_name}")[:60]
    fname = f"{safe_name}.html"
    with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(html)
    return {"filename": fname, "qa_count": len(data.get("qa_pairs", [])), "sources": data.get("sources", [])}
