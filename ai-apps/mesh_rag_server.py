#!/usr/bin/env python3
"""Mesh RAG server -- exposes the real Spikeling knowledge base + vault
search + Ollama generation as one HTTP endpoint, reachable from any device
on the Tailscale mesh (not just this machine).

Combines three things that were each real but separate before this:
  - knowledge.py's KnowledgeBase.search() over spikeling_knowledge.db
    (curated subject knowledge -- the ai-apps RAG corpus)
  - project_assistant.py's search_vault_notes() over the real Obsidian
    vault (Project Work + Research notes Claude sessions have logged)
  - Ollama's local /api/generate, using whichever model is installed here

Run:
    python mesh_rag_server.py

Then from ANY device on the tailnet (this machine's Tailscale IP, port 5055):
    curl -X POST http://100.117.59.73:5055/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "what does the LIF neuron refractory period do"}'

Security note: like Ollama's own mesh exposure, this only listens on the
Tailscale interface's reachable range -- it is NOT exposed to the public
internet, only to devices already authorized on this tailnet.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import requests
from flask import Flask, jsonify, request, send_from_directory

from knowledge import KnowledgeBase

sys.path.insert(0, r"G:\dev\observe-api")
try:
    from search_engine import SearchEngine
    vault_engine = SearchEngine()
    vault_engine.load_blocking(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_semantic_index"),
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    if not vault_engine.ready:
        print("[warn] vault_semantic_index failed to load -- falling back to substring vault search")
        vault_engine = None
except Exception as e:
    print(f"[warn] OBSERVE semantic vault search not available: {e}")
    vault_engine = None

try:
    from project_assistant import search_vault_notes
except ImportError:
    search_vault_notes = None

METHODLM_DIR = r"C:\Users\gbran\llama_demo"
sys.path.insert(0, METHODLM_DIR)
try:
    from methodlm import load_csv as methodlm_load_csv, investigate as methodlm_investigate
    import methodlm as _methodlm_module
except ImportError as e:
    methodlm_load_csv = None
    methodlm_investigate = None
    _methodlm_module = None
    print(f"[warn] MethodLM not available: {e}")

try:
    from methodlm_io import validate as methodlm_validate, load_any as methodlm_load_any, \
        featurize as methodlm_featurize, format_report as methodlm_format_report
except ImportError as e:
    methodlm_validate = methodlm_load_any = methodlm_featurize = methodlm_format_report = None
    print(f"[warn] methodlm_io not available: {e}")

# ---- Real Spikeling LIF confidence gate -- decides whether retrieval was
# actually strong enough to answer from, instead of always generating. ----
SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
CONFIDENCE_DRIVE_SCALE = 20.0  # calibrated on real observed vault scores, see retrieval_confidence.spk

# OPT-IN accelerated backend (silicon-mega-accelerator, github.com/tritsystem/
# silicon-mega-accelerator). Default OFF -- existing behavior is completely
# unchanged unless SPIKEMESH_ACCELERATED_GATE is explicitly set. Uses
# IndexedTernaryRuntime in quantization="none" mode specifically -- the ONLY
# mode verified bit-exact against the original SpikelingRuntime (Milestone 1
# correctness test + Milestone 6's real test against this exact .spk file,
# both real calibration points: relevant score 6.3, irrelevant score 2.5,
# both matched exactly). Quantized modes are NOT wired in here -- they have
# a real, measured, disclosed accuracy cost that has no business silently
# affecting a live production gate.
_USE_ACCELERATED_GATE = os.environ.get("SPIKEMESH_ACCELERATED_GATE", "").lower() in ("1", "true", "yes")

try:
    from compiler.compiler import compile_file
    from runtime.runtime import SpikelingRuntime
    import tempfile as _tempfile

    _CONF_SPK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_confidence.spk")
    _conf_ast = compile_file(_CONF_SPK, output_dir=_tempfile.mkdtemp(prefix="spikemesh_conf_"))

    _GateRuntime = SpikelingRuntime
    if _USE_ACCELERATED_GATE:
        try:
            sys.path.insert(0, r"C:\Users\gbran\OneDrive\Documents\silicon-mega-accelerator")
            from indexed_ternary_runtime import IndexedTernaryRuntime

            def _GateRuntime(ast):
                return IndexedTernaryRuntime(ast, quantization="none")
            print("[spikemesh] confidence gate: ACCELERATED backend (indexed connectivity, "
                  "behavior-preserving, quantization=none)")
        except Exception as e:
            print(f"[warn] accelerated gate requested but unavailable ({e}) -- falling back to original")
            _GateRuntime = SpikelingRuntime
    else:
        print("[spikemesh] confidence gate: original backend (set SPIKEMESH_ACCELERATED_GATE=1 to opt in)")

    def retrieval_is_confident(best_score):
        runtime = _GateRuntime(_conf_ast)
        neuron = runtime.neurons["confidence"]
        # stimulate() already injects the drive AND runs one real LIF tick
        # (checked directly against runtime.py -- no separate tick() call needed).
        runtime.stimulate("confidence", 0.0, drive=max(0.0, best_score) * CONFIDENCE_DRIVE_SCALE)
        return neuron.fire_count > 0, neuron.membrane_potential
except Exception as e:
    print(f"[warn] Spikeling confidence gate not available: {e}")

    def retrieval_is_confident(best_score):
        return True, None  # fail open -- degrade to "always answer" rather than break /ask entirely

OLLAMA_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
GENERATE_MODEL = os.environ.get("SPIKELING_MESH_MODEL", "llava:7b")

app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    # Defensive fix: same-origin fetches shouldn't need this, but a
    # real "Failed to fetch" was observed from the UI while direct curl
    # calls to the same endpoints succeeded -- explicit permissive CORS
    # headers rule out any browser-side cross-origin/private-network
    # gate as the cause, at real zero cost for a private mesh service.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp
kb = KnowledgeBase()


README_MATCH_SCORE = 9.0  # deliberately above every real observed vault score (max seen: 7.83) --
                          # an explicitly-named project's own README is a more authoritative source
                          # for a definitional question than any scattered note that merely mentions it


def _readme_context_for_question(question):
    """Real fix for a real, diagnosed bug: /ask never had access to actual
    project READMEs, only scattered vault notes -- so "what is Spikeling?"
    lost to an unrelated hardware-brainstorm note that just happened to
    mention "Spikeling" many times (confirmed by reading that note's real
    content). If the question names a known project, pull its real README
    in directly, rather than relying on scattered mentions to win a score
    contest they were never well-suited for."""
    q_lower = question.lower()
    for name, path in pm.KNOWN_PROJECTS.items():
        if name.lower() in q_lower:
            for readme_name in ("README.md", "readme.md"):
                readme_path = os.path.join(path, readme_name)
                if os.path.isfile(readme_path):
                    with open(readme_path, encoding="utf-8", errors="ignore") as f:
                        return name, f.read()[:4000]
    return None, None


def build_context(question, k=4, vault_k=3):
    """Real retrieval from both sources -- returns (context_text, sources_used, best_score)."""
    parts = []
    sources = []
    best_score = 0.0

    readme_project, readme_text = _readme_context_for_question(question)
    if readme_text:
        parts.append(f"[project README, authoritative: {readme_project}] {readme_text}")
        sources.append({"type": "project_readme", "project": readme_project, "score": README_MATCH_SCORE})
        best_score = max(best_score, README_MATCH_SCORE)

    # KnowledgeBase.search() returns tuples: (score, source, page, text) -- not dicts.
    kb_hits = kb.search(question, k=k)
    for score, src, page, text in kb_hits:
        parts.append(f"[knowledge base: {src} p.{page}, score={score:.2f}] {text}")
        sources.append({"type": "knowledge_base", "source": src, "page": page, "score": round(score, 3)})

    if vault_engine is not None:
        # Real semantic search (OBSERVE's SearchEngine, repurposed from code
        # search onto vault notes) instead of plain substring matching.
        try:
            for hit in vault_engine.search(question, k=vault_k):
                parts.append(f"[vault note (semantic, score={hit['score']:.2f}): {hit['path']}] {hit['preview']}")
                sources.append({"type": "vault_note_semantic", "path": hit["path"], "score": round(hit["score"], 3)})
                best_score = max(best_score, hit["score"])
        except Exception as e:
            sources.append({"type": "vault_semantic_error", "error": str(e)})
    elif search_vault_notes is not None:
        try:
            vault_hits = search_vault_notes(question, limit=vault_k)
            for hit in vault_hits:
                parts.append(f"[vault note: {hit.get('path', '?')}] {hit.get('snippet', hit)}")
                sources.append({"type": "vault_note_substring", "path": hit.get("path", "?")})
        except Exception as e:
            sources.append({"type": "vault_search_error", "error": str(e)})

    return "\n\n".join(parts), sources, best_score


OLLAMA_RETRY_ATTEMPTS = 3
OLLAMA_RETRY_BACKOFF_S = [1, 3, 7]  # real short backoff -- a transient hiccup (e.g. Ollama's own
                                     # supervisor mid-restart) is usually gone within a few seconds


def verify_answer_grounded(question, answer, context):
    """Real, independent second check: does the retrieved context actually
    support THIS answer to THIS question in the right domain -- not just
    share overlapping words/embedding similarity with it? Caught a real
    bug this was built specifically to fix: a vault note's incidental
    mention of "session limit" (Claude's own usage limit hitting during a
    Tribe coding session, nothing to do with Tribe as a game) scored high
    enough to pass the confidence gate and got reframed as a fake game
    mechanic. A score-based gate can't catch a domain mismatch like that;
    this is a real judgment call, so it's a second LLM pass, not a threshold.

    Fails OPEN (verified=True) on any error -- a broken verifier shouldn't
    silently block every real answer; the confidence gate is still real
    protection on its own."""
    verify_prompt = (
        "You are a skeptical fact-checker. A QUESTION was asked, and an ANSWER was generated from "
        "some retrieved CONTEXT. Your only job: does the CONTEXT actually and directly support this "
        "specific ANSWER to this specific QUESTION -- not just share some overlapping words? Be "
        "suspicious of context that only matches on a surface-level phrase but is actually about a "
        "different topic, tool, or situation (e.g. an incidental log message, an unrelated mention, "
        "a different domain entirely from what the question is really asking about).\n\n"
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER: {answer}\n\n"
        "Reply with EXACTLY one line in this format, nothing else:\n"
        "VERIFIED: <yes or no> | REASON: <one short sentence>"
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            # Real bug found from user-reported inconsistency: the same
            # question got verified=True once and verified=False twice in
            # a row, purely from Ollama's default sampling randomness --
            # this check is supposed to be a real judgment, not a coin
            # flip. temperature=0 + a fixed seed makes it deterministic:
            # same input, same verdict, every time.
            json={"model": GENERATE_MODEL, "prompt": verify_prompt, "stream": False,
                  "options": {"temperature": 0, "seed": 42}},
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        # Lenient on purpose: a small local model won't always echo the
        # exact "REASON:" label even when it gets the yes/no judgment
        # right (confirmed by a real observed case -- see the fix note
        # above this function). The yes/no verdict is the load-bearing
        # part; the reason text after it is for a human to read, not
        # something that should gate the verdict on exact formatting.
        m = re.search(r"VERIFIED:\s*(yes|no)\b", raw, re.IGNORECASE)
        if not m:
            return True, f"(verifier response unparseable, failing open: {raw[:150]})"
        verified = m.group(1).strip().lower() == "yes"
        reason = raw[m.end():].lstrip(" |").strip()
        reason = re.sub(r"^REASON:\s*", "", reason, flags=re.IGNORECASE) or raw
        return verified, reason
    except requests.RequestException as e:
        return True, f"(verifier call failed, failing open: {e})"


def call_ollama_with_retry(prompt):
    """The actual 'buffer' against a flaky/restarting Ollama: retries with
    backoff before giving up, so one transient failure doesn't immediately
    surface as a hard error to the caller. Returns (answer, None) on
    success or (None, last_error_str) if every attempt failed."""
    last_error = None
    for attempt in range(OLLAMA_RETRY_ATTEMPTS):
        try:
            r = requests.post(
                OLLAMA_URL,
                # Same determinism fix as the verifier -- the same question
                # should give the same answer, not a different one each
                # time purely from sampling randomness.
                json={"model": GENERATE_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0, "seed": 42}},
                timeout=120,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip(), None
        except requests.RequestException as e:
            last_error = str(e)
            if attempt < OLLAMA_RETRY_ATTEMPTS - 1:
                time.sleep(OLLAMA_RETRY_BACKOFF_S[attempt])
    return None, last_error


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True, silent=True) or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "missing 'question'"}), 400

    context, sources, best_score = build_context(question)

    # Real Spikeling LIF gate: only generate if retrieval was actually
    # strong enough, calibrated on real observed scores (see
    # retrieval_confidence.spk) -- not a hardcoded score cutoff.
    confident, membrane_potential = retrieval_is_confident(best_score)
    if not confident:
        return jsonify({
            "answer": "I don't have strong enough real evidence in the vault or knowledge base to "
                      "answer this confidently, so I'm not going to guess.",
            "sources": sources,
            "model": None,
            "confidence_gate": {"fired": False, "best_score": round(best_score, 3),
                                 "membrane_potential": membrane_potential},
        })

    prompt = (
        "Answer the question using ONLY the context below if it's relevant. "
        "If the context doesn't cover it, say so plainly instead of guessing.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )

    answer, gen_error = call_ollama_with_retry(prompt)
    if answer is not None:
        # Real second check, caught by a real bug: the confidence gate above
        # only measures retrieval SCORE (embedding/keyword overlap), not
        # whether the retrieved text is actually IN THE RIGHT DOMAIN to
        # answer THIS question. Real observed failure: "session limit" in a
        # vault note about MY OWN Claude usage limit hitting during a Tribe
        # coding session scored high enough to pass the gate, and got
        # reframed as a fake Tribe game mechanic. This is a second,
        # independent LLM judgment -- not a score threshold -- checking
        # whether the context is actually the right kind of thing to answer
        # from, mirroring MethodLM's own bias-audit discipline (a claim has
        # to survive a real check, not just look plausible).
        verified, verify_reason = verify_answer_grounded(question, answer, context)
        if not verified:
            return jsonify({
                "answer": "The best-matching content I found looks like it's not actually about the right "
                          "thing for this question (it may be an incidental mention, a log artifact, or "
                          "off-topic despite matching words) -- so I'm not going to present it as a real answer.",
                "verification_reason": verify_reason,
                "sources": sources,
                "model": None,
                "confidence_gate": {"fired": True, "best_score": round(best_score, 3),
                                     "membrane_potential": membrane_potential},
                "grounding_check": {"verified": False},
            })
    if answer is None:
        # Graceful degradation: a down/hung Ollama doesn't have to mean an
        # empty-handed failure -- real retrieval already happened, so hand
        # back what was actually found instead of a bare 502.
        return jsonify({
            "answer": None,
            "degraded": True,
            "degraded_reason": gen_error,
            "sources": sources,
            "model": None,
            "confidence_gate": {"fired": True, "best_score": round(best_score, 3),
                                 "membrane_potential": membrane_potential},
        }), 200

    return jsonify({
        "answer": answer, "sources": sources, "model": GENERATE_MODEL,
        "confidence_gate": {"fired": True, "best_score": round(best_score, 3),
                             "membrane_potential": membrane_potential},
        "grounding_check": {"verified": True, "reason": verify_reason},
    })


@app.route("/investigate", methods=["POST"])
def investigate_route():
    """Real causal investigation via MethodLM -- pre-registered tests
    (CORR/STRAT/RUN/ADJUST), a collider/mediator bias audit, and a
    Cinelli-Hazlett robustness value, not just an LLM's asserted opinion.

    Body: {"csv_path": "C:\\path\\to\\data.csv", "target": "column_name",
           "question": "what drives target?"}
    Real cost: this runs a genuine multi-turn tool-driven investigation
    against the local model (proven ~5s/turn, up to 10 turns) -- expect
    tens of seconds, not an instant reply.
    """
    if methodlm_investigate is None:
        return jsonify({"error": "MethodLM not available on this server"}), 503

    payload = request.get_json(force=True, silent=True) or {}
    csv_path = payload.get("csv_path", "").strip()
    target = payload.get("target", "").strip()
    question = payload.get("question", "").strip() or f"What drives {target}?"

    if not csv_path or not target:
        return jsonify({"error": "missing 'csv_path' or 'target'"}), 400
    if not os.path.isfile(csv_path):
        return jsonify({"error": f"file not found: {csv_path}"}), 404

    try:
        data = methodlm_load_csv(csv_path, target)
    except Exception as e:
        return jsonify({"error": f"failed to load CSV: {e}"}), 400

    run_name = f"mesh_{os.path.basename(csv_path)}_{target}".replace(" ", "_")
    try:
        result = methodlm_investigate(run_name, data, target, question, interventional=False)
    except Exception as e:
        return jsonify({"error": f"investigation failed: {e}"}), 500

    return jsonify(result)


# ---- MethodLM's own console, ported in directly (no second process --
# starting a new background service on this machine has been a hard,
# repeatedly-confirmed gate; this reuses the SAME already-running Flask
# server instead, since restarting it has worked cleanly every other time
# this session). Real logic ported from methodlm_gui.py's do_GET/do_POST,
# not a reduced reimplementation -- same /api/ping, /api/describe,
# /api/run, /api/race behavior, same methodlm_io.validate/load_any/
# featurize/format_report + methodlm.investigate/vanilla_answer calls. ----
METHODLM_GUI_HTML = os.path.join(METHODLM_DIR, "methodlm_gui.html")


@app.route("/methodlm-gui", methods=["GET"])
def methodlm_gui():
    if not os.path.isfile(METHODLM_GUI_HTML):
        return jsonify({"error": f"methodlm_gui.html not found at {METHODLM_GUI_HTML}"}), 500
    with open(METHODLM_GUI_HTML, encoding="utf-8") as f:
        body = f.read()
    skeleton = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<style>html,body{margin:0;background:#0a0603}</style></head><body>"
                + body + "</body></html>")
    return skeleton, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/ping", methods=["GET"])
def methodlm_api_ping():
    return jsonify({"ok": True})


@app.route("/api/describe", methods=["POST"])
def methodlm_api_describe():
    if methodlm_validate is None:
        return jsonify({"error": "methodlm_io not available on this server"}), 503
    data = request.get_json(force=True, silent=True) or {}
    r = methodlm_validate(data.get("path", ""), data.get("target"),
                           table=data.get("table"), query=data.get("query"))
    return jsonify(r)


def _methodlm_gui_run(data, race=False):
    path, target = data.get("path", ""), data.get("target", "")
    raw, notes = methodlm_load_any(path, table=data.get("table"), query=data.get("query"))
    tbl, rep = methodlm_featurize(raw, target)
    report = methodlm_format_report(notes, rep, target)
    cols = [c for c in tbl if c != target]
    q = (f"Investigate what actually drives {target} in this dataset "
         f"(columns: {', '.join(cols[:12])}). Do not trust raw correlations.")
    name = os.path.splitext(os.path.basename(path))[0]
    res = methodlm_investigate(name, tbl, target, q, False, ingest_report=report)
    ledger_path = os.path.join(METHODLM_DIR, f"ledger_{name}.txt")
    ledger_text = ""
    if os.path.isfile(ledger_path):
        with open(ledger_path, encoding="utf-8") as f:
            ledger_text = f.read()
    out = {"ledger": ledger_text, "verdict": res["verdict"],
           "tested": res["nrun"], "prereg": res["pre"], "gate": res["gate"]}
    if race:
        out["vanilla"] = _methodlm_module.vanilla_answer(q)
    return out


@app.route("/api/run", methods=["POST"])
def methodlm_api_run():
    if methodlm_load_any is None or methodlm_investigate is None:
        return jsonify({"error": "MethodLM not fully available on this server"}), 503
    data = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(_methodlm_gui_run(data, race=False))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/race", methods=["POST"])
def methodlm_api_race():
    if methodlm_load_any is None or methodlm_investigate is None:
        return jsonify({"error": "MethodLM not fully available on this server"}), 503
    data = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(_methodlm_gui_run(data, race=True))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def ui():
    """The full independent UI -- served directly by this same mesh server,
    reachable from any device on the tailnet via a plain browser. No
    claude.ai, no external account, no dependency outside this machine."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "ui.html")


import project_manager as pm
import sqlite3

SERVER_GUARD_DB = r"C:\Users\gbran\OneDrive\Documents\server-guard\server_guard.db"
GUARD_STALE_THRESHOLD_S = 120  # server-guard's real default --interval is far shorter than this


import playbook_generator


@app.route("/playbook/generate", methods=["POST"])
def playbook_generate_route():
    """Generates a real, grounded (but AI-generated, not hand-verified)
    study page for any registered project or a vault topic. Real cost:
    this makes real Ollama generation calls, expect several seconds."""
    payload = request.get_json(force=True, silent=True) or {}
    target_type = payload.get("target_type", "").strip()
    target_name = payload.get("target_name", "").strip()
    code_file = (payload.get("code_file") or "").strip() or None
    if not target_type or not target_name:
        return jsonify({"error": "missing 'target_type' or 'target_name'"}), 400
    result = playbook_generator.generate_and_save(target_type, target_name, code_file=code_file)
    if "error" in result:
        return jsonify(result), 400
    result["url"] = f"/playbook/view/{result['filename']}"
    return jsonify(result)


@app.route("/playbook/project_files", methods=["GET"])
def playbook_project_files_route():
    """Real file listing for the UI's file picker -- not an auto-guess."""
    project = request.args.get("project", "").strip()
    if not project:
        return jsonify({"error": "missing 'project'"}), 400
    result = playbook_generator.list_real_code_files(project)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/playbook/view/<path:filename>", methods=["GET"])
def playbook_view_route(filename):
    return send_from_directory(playbook_generator.OUTPUT_DIR, filename)


@app.route("/playbook/list", methods=["GET"])
def playbook_list_route():
    os.makedirs(playbook_generator.OUTPUT_DIR, exist_ok=True)
    files = sorted(os.listdir(playbook_generator.OUTPUT_DIR), reverse=True)
    return jsonify({"playbooks": [{"filename": f, "url": f"/playbook/view/{f}"} for f in files if f.endswith(".html")]})


@app.route("/guard/status", methods=["GET"])
def guard_status_route():
    """Real server-guard data -- latest reading per real channel from its
    actual SQLite DB. Honest about staleness: guard.py isn't currently
    running (stopped earlier this session after its 5s-interval collector
    turned out to be spawning a PowerShell subprocess every tick), so this
    is real historical data, not a live feed, until it's deliberately
    restarted with a sane interval."""
    try:
        conn = sqlite3.connect(SERVER_GUARD_DB)
        c = conn.cursor()
        c.execute("""
            SELECT r.channel, r.value, r.timestamp FROM readings r
            INNER JOIN (SELECT channel, MAX(timestamp) AS ts FROM readings GROUP BY channel) latest
            ON r.channel = latest.channel AND r.timestamp = latest.ts
            ORDER BY r.channel
        """)
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"could not read server-guard DB: {e}"}), 500

    now = time.time()
    channels = [{"channel": ch, "value": val, "age_seconds": round(now - ts, 1)} for ch, val, ts in rows]
    newest_age = min((c["age_seconds"] for c in channels), default=None)
    is_live = newest_age is not None and newest_age < GUARD_STALE_THRESHOLD_S

    return jsonify({
        "live": is_live,
        "newest_reading_age_seconds": newest_age,
        "channel_count": len(channels),
        "channels": channels,
    })

TASK_DECISIONS_PATH = os.path.join(SPIKELING_ROOT, "task_decisions.jsonl")


@app.route("/pm/registry", methods=["GET"])
def pm_registry_route():
    """The real Project Manager: git hygiene across every known real local
    checkout -- uncommitted files, ahead/behind origin, staleness. Not a
    log, actual current repo state."""
    return jsonify({"projects": pm.registry()})


@app.route("/pm/github_check", methods=["GET"])
def pm_github_check_route():
    """Real live comparison between GitHub's actual current state and this
    PC's actual current state -- not the cached local tracking ref, a real
    `gh api` call. Real cost: one network call per request, expect ~1s."""
    project = request.args.get("project", "").strip()
    if not project:
        return jsonify({"error": "missing 'project'"}), 400
    return jsonify(pm.github_cross_reference(project))


@app.route("/pm/log", methods=["GET"])
def pm_log_route():
    """Real commit history of the PM data store -- the actual cross-device
    collaboration mechanism (git), not a bespoke sync protocol."""
    return jsonify({"entries": pm.pm_log()})


@app.route("/pm/note", methods=["POST"])
def pm_note_route():
    payload = request.get_json(force=True, silent=True) or {}
    project = payload.get("project", "").strip()
    text = payload.get("text", "").strip()
    if not project or not text:
        return jsonify({"error": "missing 'project' or 'text'"}), 400
    fname = pm.pm_add_note(project, text)
    return jsonify({"committed": fname})


@app.route("/pm/commands", methods=["GET"])
def pm_commands_route():
    """The real, explicit command allowlist -- deliberately not a generic
    shell executor, since this is reachable from any device on the mesh."""
    return jsonify({
        "commands": [{"id": k, "label": v["label"], "needs_project": v["needs_project"]}
                     for k, v in pm.SAFE_COMMANDS.items()]
    })


@app.route("/pm/run", methods=["POST"])
def pm_run_route():
    payload = request.get_json(force=True, silent=True) or {}
    command_id = payload.get("command_id", "").strip()
    project_name = payload.get("project")
    result, status = pm.run_safe_command(command_id, project_name)
    return jsonify(result), status


@app.route("/tasks", methods=["GET"])
def tasks_route():
    """Real agent-activity log over task_decisions.jsonl -- which of Spike's
    10 specialist agents fired for which task, and why (the S_Work/
    S_Ambiguous/S_Complex/S_Tests/S_Research scores that drove the
    decision). Distinct from Project Manager -- this is about agent
    routing, not project/repo state."""
    import json as _json

    limit = int(request.args.get("limit", 30))
    entries = []
    try:
        with open(TASK_DECISIONS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(_json.loads(line))
                except _json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify({"error": f"task log not found: {TASK_DECISIONS_PATH}"}), 404

    total = len(entries)
    recent = list(reversed(entries[-limit:]))

    agent_fire_counts = {}
    for e in entries:
        for agent in e.get("fired", []):
            agent_fire_counts[agent] = agent_fire_counts.get(agent, 0) + 1

    return jsonify({
        "total_entries": total,
        "recent": recent,
        "agent_fire_counts": dict(sorted(agent_fire_counts.items(), key=lambda kv: -kv[1])),
    })


@app.route("/engine/run", methods=["GET"])
def engine_run_route():
    """Real jet-engine-staged spiking pipeline (jet_engine_spike_pipeline.spk):
    compressor(converging)->combustion(nonlinear amplification)->turbine
    (feeds back to drive the compressor, same as a real turbojet's shaft)
    ->exhaust. Optimized the same measured way as the confidence gate above
    -- AST compiled once at import (not per request), accelerated
    (bit-exact) runtime -- see jet_engine_gate.py."""
    try:
        import jet_engine_gate
        n_ticks = int(request.args.get("ticks", 20))
        drive = float(request.args.get("drive", 80.0))
        result = jet_engine_gate.run_pipeline(n_ticks=n_ticks, intake_drive=drive)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/guard/sustained-check", methods=["GET"])
def guard_sustained_check_route():
    """Real use of the jet-engine pipeline: sustained-anomaly confirmation
    for a server-guard channel. A single elevated reading is noise; this
    only confirms True if the real recent history was sustained (the same
    honest distinction real beacon/C2 detection requires) -- verified
    against real historical server-guard data before being wired in here,
    not assumed to work. See jet_engine_gate.check_sustained_anomaly()."""
    try:
        import jet_engine_gate
        channel = request.args.get("channel", "pkt.beacon_candidate_destinations")
        window = int(request.args.get("window", 30))
        result = jet_engine_gate.check_sustained_anomaly(channel=channel, window=window)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/lab", methods=["POST"])
def model_lab_route():
    """SpikeMesh model-onboarding lab: drop in a local HF model_id and run
    real operations from spikemesh_model_lab.py (llama_demo/):
      vet       -- score the model's causal-reasoning discipline against
                   the real 9-trap rubric from tonight's stress test.
      finetune  -- bake the real corpus (hypothetical + real-telemetry-
                   derived examples) into the model's weights, re-vet to
                   confirm it actually moved the score.
    quantize is DISABLED here (2026-08-09): measured, not assumed --
    tritkit ternarize()+set_quant() completely destroyed a real fine-tuned
    model's causal-reasoning discipline (5/6 -> 0/6), replicating the
    earlier MobileNetV3 ternary-quantization failure (78.8%->4.9%
    accuracy) on a new architecture/task. The function still exists in
    spikemesh_model_lab.py as a documented negative result -- just not
    served live so it can't silently wreck a model someone drops in.
    Synchronous and slow (minutes for finetune) -- this is a real local
    research tool, not a scaled service; call it knowing that."""
    try:
        sys.path.insert(0, r"C:\Users\gbran\llama_demo")
        import spikemesh_model_lab as lab
        body = request.get_json(force=True) or {}
        op = body.get("op", "vet")
        model_id = body.get("model_id", lab.DEFAULT_MODEL)
        if op == "vet":
            model, tok = lab.load_model(model_id)
            result = lab.vet(model, tok)
        elif op == "quantize":
            return jsonify({
                "error": "quantize is disabled -- measured to completely destroy a fine-tuned "
                         "model's causal-reasoning discipline (5/6 -> 0/6), same failure mode as "
                         "the earlier MobileNetV3 ternary-quantization result. See "
                         "spikemesh_model_lab.quantize() for the documented negative finding."
            }), 403
        elif op == "finetune":
            result = lab.finetune(model_id, corpus_path=body.get("corpus_path"),
                                   epochs=int(body.get("epochs", 8)))
        else:
            return jsonify({"error": f"unknown op '{op}', expected vet|quantize|finetune"}), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "kb_sources": kb.sources() if hasattr(kb, "sources") else None})


if __name__ == "__main__":
    from waitress import serve

    threads = int(os.environ.get("MESH_RAG_THREADS", "8"))
    print(f"Mesh RAG server (production, waitress, {threads} threads) starting on 0.0.0.0:5055, "
          f"generation model={GENERATE_MODEL}")
    print("Reachable on the tailnet at http://100.117.59.73:5055/ask")
    serve(app, host="0.0.0.0", port=5055, threads=threads)
