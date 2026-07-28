import sys, time, csv, os, re, json, hashlib, getpass, datetime, subprocess, webbrowser
import urllib.request
from urllib.parse import quote, urlencode
# Windows' default console codepage (cp1252) crashes on Unicode output --
# generated code, emoji, etc. Force UTF-8 with substitution instead of
# raising, so a print() never takes the whole process down.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "core")
import speech_recognition as sr
import pyautogui
import pyttsx3
from compiler.compiler import SpikelingParser
from runtime.runtime import SpikelingRuntime

# Offline LLM fallback -- same real, already-validated local model/binary
# methodlm.py uses (LocalLlama in llama_demo/methodlm_models.py), reused
# here rather than reinvented. Tested directly before wiring in: 3/4
# correct on phrases with zero keyword overlap, ~5s latency per call, one
# real failure mode observed ("this is way too quiet, fix it" -> wrongly
# classified VOLDOWN, likely confused by the word "quiet" itself rather
# than reasoning about intent) -- a real small-model limitation, not
# hidden. Used ONLY as a fallback when keyword matching finds nothing, so
# the common case stays fast and free.
LLAMA_BIN = r"C:\Users\gbran\llama_demo\bin\llama-completion.exe"
LLAMA_GGUF = r"C:\Users\gbran\llama_demo\qwen3b.gguf"
LLM_SYSTEM_PROMPT = """You classify a spoken phrase into exactly one command, or NONE.
Commands: PLAYPAUSE, NEXT, PREVIOUS, VOLUP, VOLDOWN, MUTE, SCREENSHOT, REPORT.
Reply with ONLY the single command word, nothing else. If none fit, reply NONE.

Reason about what the person WANTS, not just which words appear. A complaint
about the CURRENT state implies the OPPOSITE action to fix it -- "it's too
quiet" or "I can't hear it" means the person wants it LOUDER (VOLUP), even
though the word "quiet" appears. Likewise "it's blasting my ears" means VOLDOWN.

PLAYPAUSE means resume/pause whatever is ALREADY loaded -- it does NOT mean
"open and play a specific new song, playlist, video, or artist." There is no
command for opening or searching for specific content. If the person is
asking to play/put on/throw on a SPECIFIC piece of content rather than just
resume or pause current playback, reply NONE.

Example: "this is way too quiet, fix it" -> VOLUP
Example: "way too loud, hurts my ears" -> VOLDOWN
Example: "put on some 90s rap classics on youtube" -> NONE
Example: "throw on some jazz for me" -> NONE
Example: "hit play" -> PLAYPAUSE"""
LLM_COMMAND_MAP = {
    "PLAYPAUSE": "CMD_PLAYPAUSE", "NEXT": "CMD_NEXT", "PREVIOUS": "CMD_PREVIOUS",
    "VOLUP": "CMD_VOLUP", "VOLDOWN": "CMD_VOLDOWN", "MUTE": "CMD_MUTE",
    "SCREENSHOT": "CMD_SCREENSHOT", "REPORT": "CMD_REPORT",
}

def llm_classify(text):
    """Returns a command string (e.g. CMD_VOLUP) or None. ~5s latency --
    only call this when keyword matching already failed."""
    prompt = (f"<|im_start|>system\n{LLM_SYSTEM_PROMPT}<|im_end|>\n"
              f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    try:
        p = subprocess.run([LLAMA_BIN, "-m", LLAMA_GGUF, "-p", prompt, "-n", "10",
                             "-t", "8", "-c", "1024", "--special", "-no-cnv", "--temp", "0.0"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=20)
        reply = p.stdout.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip().upper()
        return LLM_COMMAND_MAP.get(reply)
    except Exception as e:
        print(f"(local LLM fallback failed: {e})", flush=True)
        return None

# Knowledge fallback -- reuses the SAME local model/binary as command
# classification, just a different system prompt and more output tokens.
# Reached only when a phrase matches NEITHER a keyword NOR a device command
# via the LLM classifier above, so it doesn't compete with or slow down
# the fast device-command path. Answers are capped at ~120 tokens and
# written for text-to-speech (no markdown/code/bullets) -- this is a 3B
# quantized model running on CPU, so treat answers as a rough first pass,
# not an authoritative source; latency is ~15-30s, noticeably slower than
# the ~5s command classify call.
LLM_ANSWER_SYSTEM_PROMPT = """You are a voice assistant answering spoken
questions about software engineering, coding concepts, mathematics, science,
or literature. Talk casual, like a buddy -- contractions, "bro", mild cussing
(damn, hell, shit) where it fits naturally. Don't be robotic or corporate.
Give a short, accurate, spoken-friendly answer in 2-4 plain sentences -- no
markdown, no code blocks, no bullet points, no headers, since this is read
aloud by text-to-speech.

If the question is genuinely ambiguous or missing something you'd need to
answer well, don't guess -- ask ONE short clarifying question back instead
(e.g. "which language, bro?"). If you're unsure even after that, just say so
plainly instead of making something up."""

# Conversation memory for the knowledge path ONLY (not command
# classification -- that's meant to stay a stateless single-shot intent
# check). Real gap found via live testing: the bot asked "which type of
# shack are you building?", the user replied "one to live in", and since
# each call was completely stateless, that follow-up got sent to the
# model with zero context -- it had no idea what "one" referred to and
# asked an unrelated clarifying question back. Capped at a few turns --
# the model only runs with a 1024-token context (-c 1024), so unbounded
# history would eventually crowd out the system prompt and the question
# itself.
CONVERSATION_HISTORY = []
MAX_HISTORY_TURNS = 3

def _trim_to_complete_sentence(reply):
    """If the model's output got cut off mid-sentence (hit the token cap
    before reaching a natural stop), trim back to the last complete
    sentence instead of returning/speaking a broken fragment. Real bug
    this fixes: "teach me basic math" got cut off mid-parenthetical --
    "...Parentheses, Exponents, Multiplication and Division (from left"
    instead of finishing "...from left to right)" -- and the raw
    fragment got sent through and would've been read aloud cut off
    mid-thought too."""
    reply = reply.strip()
    if reply and reply[-1] not in ".!?":
        cut = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
        if cut != -1:
            reply = reply[:cut + 1].strip()
    return reply

def llm_answer(text):
    """Returns a plain-text answer string, or None on failure. ~15-30s
    latency -- only call this after both keyword and command-classify
    fallbacks have already missed. Includes the last few Q&A turns as
    context, so a follow-up reply to the model's OWN clarifying question
    is actually understood as a follow-up."""
    history_prompt = "".join(
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        for q, a in CONVERSATION_HISTORY[-MAX_HISTORY_TURNS:]
    )
    prompt = (f"<|im_start|>system\n{LLM_ANSWER_SYSTEM_PROMPT}<|im_end|>\n"
              f"{history_prompt}"
              f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    try:
        # -n raised 120 -> 200 -- a ceiling, not a target, so short answers
        # are unaffected; open-ended asks ("teach me X") need the headroom
        # to actually finish a thought instead of hitting the cap mid-word.
        p = subprocess.run([LLAMA_BIN, "-m", LLAMA_GGUF, "-p", prompt, "-n", "200",
                             "-t", "8", "-c", "1024", "--special", "-no-cnv", "--temp", "0.2"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=60)
        reply = p.stdout.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        reply = _trim_to_complete_sentence(reply)
        if reply:
            CONVERSATION_HISTORY.append((text, reply))
            del CONVERSATION_HISTORY[:-MAX_HISTORY_TURNS]
        return reply or None
    except Exception as e:
        print(f"(local LLM answer failed: {e})", flush=True)
        return None

# Real Claude API (Opus 4.8) with web search -- the general fix for "it
# needs to answer real-time questions" (weather, news, prices, anything
# the static local model has no way to know). Rather than hand-build a
# dedicated API integration per question category (sunset/sunrise already
# got one), this routes general knowledge questions through a model that
# can actually look things up. Preferred over the local model when
# ANTHROPIC_API_KEY is set; falls back to the offline model otherwise, so
# the system still works with zero new setup. Real ongoing API cost while
# this is configured -- Opus 4.8 pricing, not free like the local model.
CLAUDE_SYSTEM_PROMPT = """You are a voice assistant. Talk casual, like a buddy
-- contractions, "bro", mild cussing (damn, hell, shit) where it fits
naturally. Don't be robotic or corporate. Give a short, spoken-friendly
answer in 2-4 plain sentences -- no markdown, no code blocks, no bullet
points, no headers, since this is read aloud by text-to-speech.

Use web search whenever the question needs current or real-time information
-- weather, news, prices, sports scores, anything time-sensitive -- don't
guess or rely on stale training data for those.

If the question is genuinely ambiguous, ask ONE short clarifying question
back instead of guessing."""

def claude_answer(text):
    """Returns a plain-text answer string, or None if not configured / the
    call fails. Shares CONVERSATION_HISTORY with the local model's
    llm_answer() -- same chat history either way, whichever backend
    happens to answer a given turn."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        messages = []
        for q, a in CONVERSATION_HISTORY[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": text})

        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=300,
            system=CLAUDE_SYSTEM_PROMPT,
            messages=messages,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
        )
        answer = " ".join(b.text for b in response.content if b.type == "text").strip()
        if answer:
            CONVERSATION_HISTORY.append((text, answer))
            del CONVERSATION_HISTORY[:-MAX_HISTORY_TURNS]
        return answer or None
    except Exception as e:
        print(f"(Claude API call failed: {e})", flush=True)
        return None

# Claude via the installed `claude` CLI -- reuses THIS machine's existing
# Claude Code login (subscription or API key, whatever's already set up),
# so it needs no separate ANTHROPIC_API_KEY and no extra billing setup.
# Confirmed by direct testing: `claude -p` reuses the current login, knows
# the real date (Claude Code injects it), and does real web search when
# WebSearch/WebFetch are allowed (returned live Placerville weather). The
# CLI is stateless per call, so conversation history is injected into the
# prompt text the same way the local model does it. Preferred over the raw
# API path when the binary exists, because "just use the same Claude as my
# chat" is exactly what this does -- no key to manage.
import shutil
CLAUDE_CLI = shutil.which("claude")

CLAUDE_CLI_PREAMBLE = """You are a casual voice assistant being read aloud by
text-to-speech. Talk like a buddy -- contractions, "bro", mild cussing where
it fits. ABSOLUTELY NO markdown, bold, bullet points, headers, URLs, or a
"Sources:" list -- plain spoken text only, since a robot voice reads it out.

Use web search for anything current (weather, news, prices, scores). If the
user shares a URL, or asks about a GitHub repo / their repos, actually FETCH
it with web fetch (and web search to find it if needed) and summarize what
you find -- do NOT ask them which one or stall; go look.

When the user asks you to RESEARCH something -- find prices, where to buy,
the best/cheapest option, compare things -- actually search the web and
report SPECIFIC findings, grounded in what you found: name the actual
stores or sites AND the real current prices/figures you saw, not vague
advice like "check Amazon or a hardware store." Give at least a couple of
concrete options with their prices. If you genuinely can't find a specific
price or fact after searching, SAY SO plainly -- never fill the gap with a
generic guess dressed up as an answer. Specific and honest beats broad.

Keep normal answers to 2-4 sentences; a research/comparison or repo summary
can be a short paragraph. If genuinely ambiguous, ask ONE short question."""

def _strip_markdown(s):
    """Light cleanup so TTS doesn't read markdown syntax aloud."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)          # bold
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)      # links -> link text
    # drop trailing "Sources:" blocks and bare bullet/URL lines
    lines = []
    for ln in s.splitlines():
        st = ln.strip()
        if st.lower().startswith("sources:") or st.startswith(("-", "*", "http")):
            continue
        lines.append(ln)
    return " ".join(" ".join(lines).split()).strip()

CLAUDE_ROUTER_PREAMBLE = """You are an intent router for a personal assistant
bot. Map the user's message to exactly ONE command token from the list below,
or NONE. Reply with ONLY a single line of JSON: {"command": "<TOKEN>", "arg": "<text or empty>"}.

Commands:
- TRIBE_WORK: the user wants to CONTINUE or DO the next improvement/polish task
  on their "tribe" game (e.g. "keep working on tribe", "let's polish the tribe
  game more", "do the next tribe task", "keep improving the village sim"). No arg.
- TRIBE_STATUS: the user is ASKING how the tribe game is going / its progress /
  status (e.g. "how's my tribe game", "what's the state of tribe", "how far
  along is the tribe game"). No arg.
- AGENT_TASK: the user wants a SPECIFIC coding change made to a project. arg =
  the task, and if they name a project (tribe / horde / spikeling) prefix it
  like "tribe: <task>". E.g. "add a jump to the tribe game" -> arg "tribe: add a jump".
- EXPERIMENT: the user wants a throwaway script WRITTEN AND RUN (e.g. "write me
  a python script that computes X and run it"). arg = the script description.
- HEALTH_CHECK: the user asks if the bot/system is working. No arg.
- PLAYPAUSE / NEXT / PREVIOUS / VOLUP / VOLDOWN / MUTE: media/device controls --
  ONLY when the user clearly wants that media action ("pause the music", "skip
  this track", "turn it up", "mute"). No arg.
- SCREENSHOT: the user explicitly asks to take a screenshot of the screen. No arg.
- NONE: anything else -- ESPECIALLY any informational question, general chat,
  greetings, status remarks, or request for facts/knowledge. When unsure, NONE.

Be conservative: only pick a command when the user clearly wants that action.
A question ABOUT something is NONE. Casual chat is NONE. Do NOT fire a media
command just because a word like "play", "next", "up", or "loop" appears in an
unrelated sentence -- that is the exact false-trigger you must avoid."""

def claude_route_command(text):
    """Uses Claude to map a loosely-worded message to a command token the
    literal matchers missed. Returns (CMD_..., arg) or (None, None). Reuses
    the installed claude CLI (no API key). Conservative -- returns (None,None)
    on any parse failure or NONE verdict, so it never hijacks normal Q&A."""
    if not CLAUDE_CLI or not text:
        return None, None
    _TOKEN_TO_CMD = {
        "TRIBE_WORK": "CMD_TRIBE_WORK", "TRIBE_STATUS": "CMD_TRIBE_STATUS",
        "AGENT_TASK": "CMD_AGENT_TASK", "EXPERIMENT": "CMD_EXPERIMENT",
        "HEALTH_CHECK": "CMD_HEALTH_CHECK",
        "PLAYPAUSE": "CMD_PLAYPAUSE", "NEXT": "CMD_NEXT", "PREVIOUS": "CMD_PREVIOUS",
        "VOLUP": "CMD_VOLUP", "VOLDOWN": "CMD_VOLDOWN", "MUTE": "CMD_MUTE",
        "SCREENSHOT": "CMD_SCREENSHOT",
    }
    try:
        p = subprocess.run([CLAUDE_CLI, "-p"],
                           input=f"{CLAUDE_ROUTER_PREAMBLE}\n\nUser message: {text}",
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=45)
        out = p.stdout.strip()
        m = re.search(r"\{.*\}", out, re.DOTALL)   # tolerate stray prose around the JSON
        if not m:
            return None, None
        data = json.loads(m.group(0))
        cmd = _TOKEN_TO_CMD.get(str(data.get("command", "")).strip().upper())
        arg = str(data.get("arg", "")).strip()
        if cmd:
            print(f"(claude router -> {cmd}{' : ' + arg if arg else ''})", flush=True)
        return (cmd, arg) if cmd else (None, None)
    except Exception as e:
        print(f"(claude router failed, non-fatal: {e})", flush=True)
        return None, None

def search_vault(query, k=3):
    """Keyword-overlap search over the Obsidian vault's notes -- no
    embeddings/index needed, the vault is small and grows slowly. Returns up
    to k (task, status, snippet) tuples for notes sharing meaningful words
    with `query`, ranked by overlap count. This is retrieval, not training --
    it makes past experiments/screenshots findable as context, it doesn't
    change any model weights."""
    stopwords = {"a", "an", "the", "and", "or", "to", "for", "of", "in", "on",
                 "is", "it", "that", "this", "with", "me", "my", "can", "you",
                 "what", "how", "does", "do", "did", "write", "run", "make"}
    q_words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stopwords and len(w) > 2}
    if not q_words or not os.path.isdir(VAULT_DIR):
        return []
    scored = []
    for root, _, fnames in os.walk(VAULT_DIR):
        if os.path.basename(root) == "attachments":
            continue
        for fname in fnames:
            if not fname.endswith(".md"):
                continue
            try:
                with open(os.path.join(root, fname), encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            body_words = set(re.findall(r"[a-z0-9]+", content.lower()))
            overlap = len(q_words & body_words)
            if overlap:
                title_m = re.search(r"^# (.+)$", content, re.M)
                status_m = re.search(r"^status: (.+)$", content, re.M)
                task = title_m.group(1) if title_m else fname
                status = status_m.group(1) if status_m else ""
                scored.append((overlap, task, status, content[:300]))
    scored.sort(key=lambda x: -x[0])
    return [(t, s, snip) for _, t, s, snip in scored[:k]]

def claude_cli_answer(text):
    """Returns a plain-text answer via the `claude` CLI, or None if the CLI
    isn't installed / the call fails. Shares CONVERSATION_HISTORY with the
    other backends. Also pulls in relevant past experiments from the
    Obsidian vault as context, so past work is "remembered" without any
    fine-tuning -- pure retrieval."""
    if not CLAUDE_CLI:
        return None
    history = ""
    if CONVERSATION_HISTORY:
        turns = "".join(f"- They asked: {q}\n  You answered: {a}\n"
                        for q, a in CONVERSATION_HISTORY[-MAX_HISTORY_TURNS:])
        history = f"\nRecent conversation so far (for context on follow-ups):\n{turns}"
    vault_hits = search_vault(text)
    vault_context = ""
    if vault_hits:
        notes = "".join(f"- \"{t}\" (status: {s}): {snip.strip()[:200]}\n" for t, s, snip in vault_hits)
        vault_context = (f"\nRelevant past experiments from memory (only mention these if "
                          f"actually relevant to the question, don't force it in):\n{notes}")
    prompt = f"{CLAUDE_CLI_PREAMBLE}\n{history}{vault_context}\nNow answer this question out loud: {text}"
    try:
        # Prompt goes via STDIN, not as a -p argument. On Windows `claude`
        # resolves to claude.cmd (a batch shim); a multi-line argument
        # passed through cmd.exe gets truncated at the first newline (real
        # bug: Claude only ever saw the first line of the preamble). Piping
        # the whole prompt in on stdin sidesteps cmd.exe arg parsing
        # entirely and also safely carries any quotes/special chars in the
        # user's question.
        p = subprocess.run([CLAUDE_CLI, "-p", "--allowedTools", "WebSearch", "WebFetch"],
                            input=prompt, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
        answer = _strip_markdown(p.stdout.strip())
        if answer:
            CONVERSATION_HISTORY.append((text, answer))
            del CONVERSATION_HISTORY[:-MAX_HISTORY_TURNS]
        return answer or None
    except Exception as e:
        print(f"(Claude CLI call failed: {e})", flush=True)
        return None

# Decides simple/static (-> fast local model) vs real-time (-> Claude with
# web search). A real-time question is one whose honest answer depends on
# CURRENT information the static local model can't have. Bias: catch
# genuinely-live topics (a false positive just means slower + uses your
# plan; a false negative means a stale/wrong answer). Deliberately does NOT
# include bare "current" -- that collides with physics ("the current
# through a resistor") -- only "currently"/"the current <noun>" style
# signals. "today"/date questions already have their own dedicated command.
REALTIME_INTENT = re.compile(
    r"\b(weather|temperature|forecast|raining|snowing|how (hot|cold|warm) is it|"
    r"news|headlines?|breaking|"
    r"stock price|share price|stock market|crypto|bitcoin|ethereum|exchange rate|"
    r"who won|who is winning|who's winning|final score|"
    r"today|tonight|right now|currently|at the moment|this week|this morning|this evening|"
    r"latest|most recent|nowadays|as of now|"
    r"who is the current|who's the current|open right now|still open)\b",
    re.IGNORECASE)

# Requests that require FETCHING external content the offline model simply
# cannot reach -- any URL, or a GitHub/repo ask. Real bug: "summarize this
# github repo <url>" was routed to the local Qwen model, which can't fetch
# anything, so it just deflected ("just let me know which repos, bro").
# These must go to Claude (which has web fetch/search).
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
FETCH_INTENT = re.compile(
    r"\b(github|gitlab|repo|repos|repositor(?:y|ies)|pull request|readme|"
    r"this (?:link|page|site|url|article)|summarize (?:this|the)|"
    r"what does this (?:say|do))\b", re.IGNORECASE)

# Research/shopping/price asks -- these WANT the live web ("do the
# research", "where to buy X", "best price", "compare", "look it up"). Real
# bug: they routed to the offline model, which can only spit generic filler
# ("check Home Depot / Amazon") instead of actually searching. Route them
# to Claude, which searches and reports specifics.
RESEARCH_INTENT = re.compile(
    r"\bdo (?:the |some )?research\b|\bresearch (?:this|that|it|the|on|into)\b|"
    r"\blook (?:it|this|that|them) up\b|\blook up\b|"
    r"\bfind (?:me )?(?:the )?(?:best|cheapest|top|good|cheaper)\b|"
    r"\bcompare (?:prices|options|the)\b|"
    r"\bshow me (?:the )?(?:options|deals|prices|listings|reviews)\b|"
    r"\bshop(?:ping)? for\b|\bwhere (?:can|do|should|to) (?:i )?(?:buy|get|find|purchase)\b|"
    r"\bbest (?:place|price|deal|store|option)\b|\bcheapest\b|"
    r"\bhow much (?:is|are|does|do|would)\b|\bprice (?:of|for|on)\b|"
    r"\bcost (?:of|for)\b|\bdeals? on\b|\bwhere to (?:buy|get)\b", re.IGNORECASE)

def needs_web(text):
    """True if the request needs live web access -- real-time info, a URL /
    repo to fetch, OR research/shopping/price questions the offline model
    can only answer with generic filler."""
    return bool(REALTIME_INTENT.search(text) or URL_RE.search(text)
                or FETCH_INTENT.search(text) or RESEARCH_INTENT.search(text))

pyautogui.PAUSE = 0
tts = pyttsx3.init()

# Optional external "response sink" -- lets another front-end (the Discord
# bot) capture every text response this system produces, without every
# handler needing to know Discord exists. None by default -- local typed/
# spoken use never touches this. discord_bot.py registers a callback via
# set_response_sink() and gets a mirror of everything say() would have
# printed/spoken. _last_attachment_path is the companion channel for
# handlers that produce an actual file (screenshot, gif) instead of just
# text, so the bot knows what to upload alongside the reply.
_response_sink = None
_last_attachment_path = None

# Local TTS is suppressed for Discord-originated commands -- someone
# controlling this PC remotely doesn't necessarily want its speakers to
# start talking. Text (printed + relayed through the response sink) fires
# exactly the same either way; only the actual pyttsx3 audio is gated.
# Set once per process_text() call based on input_method, read by say().
_suppress_tts = False

def set_response_sink(callback):
    global _response_sink
    _response_sink = callback

def say(text):
    """Always both: printed text response AND spoken audio (unless TTS is
    suppressed for this call, e.g. a Discord-originated command), for
    every confirmation, answer, or clarifying question the system gives.
    Also mirrored to an external sink (e.g. Discord) if one is
    registered."""
    print(f"Response: {text}", flush=True)
    if not _suppress_tts:
        tts.say(text)
        tts.runAndWait()
    if _response_sink:
        _response_sink(text)

with open("voice_commands.spk") as f:
    ast = SpikelingParser().parse(f.read())
rt = SpikelingRuntime(ast)

LOG_PATH = "voice_interaction_log.csv"
LOG_FIELDS = ["timestamp", "session_elapsed_s", "transcribed_text", "word_count",
              "listen_latency_s", "matched_command", "matched_via", "input_method"]

def log_interaction(text, word_count, latency, matched_command, matched_via, input_method, session_start):
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_elapsed_s": round(time.time() - session_start, 1),
            "transcribed_text": text,
            "word_count": len(text.split()) if text else 0,
            "listen_latency_s": round(latency, 3),
            "matched_command": matched_command or "NONE",
            "matched_via": matched_via,
            "input_method": input_method,
        })

def run_self_analysis():
    """CMD_REPORT handler -- runs the real causal self-analysis on the
    REAL accumulated log (voice_log_analysis.py) and speaks a plain-
    English summary. Honest limit: with few real logged interactions so
    far, this may not have enough data yet to find anything -- it says
    so plainly rather than forcing a finding."""
    result = subprocess.run([sys.executable, "voice_log_analysis.py"],
                             capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    summary_line = ""
    for line in result.stdout.splitlines():
        if line.startswith("SPOKEN_SUMMARY:"):
            summary_line = line[len("SPOKEN_SUMMARY:"):].strip()
    say(summary_line if summary_line else "Not enough real interactions logged yet to say anything meaningful.")

# command -> (action, confirmation phrase, trigger words to listen for)
# action is either a pyautogui media key string, or a callable for non-key actions
def do_screenshot():
    global _last_attachment_path
    path = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    pyautogui.screenshot().save(path)
    _last_attachment_path = path
    return path

def do_record_gif(seconds=4, fps=4):
    """Captures a real short screen recording and saves it as an animated
    GIF -- the visual 'show me what's happening' companion to a static
    screenshot, for the Discord side. Uses PIL.ImageGrab (Pillow's already
    a dependency here), no extra recording library needed. Blocking for
    ~`seconds` -- the handler says something before calling this so the
    user isn't left wondering why nothing happened for 4 seconds."""
    global _last_attachment_path
    from PIL import ImageGrab
    frame_delay = 1.0 / fps
    frames = []
    for _ in range(int(seconds * fps)):
        frames.append(ImageGrab.grab())
        time.sleep(frame_delay)
    path = f"clip_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:],
                    duration=int(frame_delay * 1000), loop=0)
    _last_attachment_path = path
    return path

# SEARCH and COMPOSE EMAIL need a free-text argument (the query / the email
# body), which doesn't fit Spikeling's no-argument handler signature
# (register_handler callbacks take zero args -- see runtime.py's _fire
# dispatch). _pending_arg is set right before stimulate() fires the neuron,
# and read inside the handler synchronously -- same call, no race.
_pending_arg = ""
_pending_experiment_lang = "python"
# set alongside _pending_arg when the matched CLAUDE_CODE prefix was the
# "new" variant (see CLAUDE_CODE_NEW_PREFIXES) -- tells the handler to reset
# _last_claude_session before running, forcing a genuinely fresh Claude
# conversation instead of auto-continuing the last one in this project.
_pending_claude_new_session = False

def do_search():
    query = _pending_arg
    if query:
        webbrowser.open(f"https://www.google.com/search?q={quote(query)}")
    return query

def do_compose_email():
    """Opens a Gmail compose window (in-browser, same reliable
    webbrowser.open() path as do_search()) with the dictated text
    pre-filled as the body. Originally used the mailto: scheme, but on
    a machine with no default mail client registered, Windows shows a
    'how do you want to open this' app picker that lists browsers as
    candidates -- a dead end, since browsers aren't real mailto
    handlers. Going straight to Gmail's own compose URL sidesteps OS
    protocol dispatch entirely. Deliberately does NOT send anything --
    sending on the user's behalf needs their own explicit action every
    time, not a one-shot voice command that could fire on a misheard
    phrase."""
    body = _pending_arg
    webbrowser.open(f"https://mail.google.com/mail/?view=cm&fs=1&su={quote('Voice-dictated draft')}&body={quote(body)}")
    return body

# Agentic Claude Code, driven from a chat message. Per the user's explicit
# choice: EDITING allowed, but hard-scoped for safety --
#   * It's a SENSITIVE command (admin-passphrase gate; the gate runs BEFORE
#     the handler, so a locked request never spends a Claude turn).
#   * Bash/shell is NOT in the allowlist, so it physically cannot run
#     arbitrary commands -- no rm, no git push, no money, no execution.
#     That's what keeps this from blowing through the no-delete/no-money
#     guardrails (those only inspect the message wording; a shell would
#     bypass them).
#   * Confined to one project directory (CLAUDE_CODE_PROJECT_DIR); Claude
#     runs with that as cwd and no extra --add-dir, so edits stay in-scope.
#   * Changes land in the working tree only -- with no Bash it literally
#     can't commit or push; the user reviews the diff themselves.
CLAUDE_CODE_PROJECT_DIR = os.environ.get(
    "CLAUDE_CODE_PROJECT_DIR",
    os.path.dirname(os.path.abspath(__file__)))   # defaults to this project (Spikeling)

# Fixed allowlist of projects the agent pipeline is permitted to touch. This
# preserves the "confined to one directory" safety property -- the target is
# SELECTABLE, but only from this hardcoded set, never an arbitrary user path.
# Only dirs that actually exist are registered, so a missing game folder just
# isn't offered rather than erroring at cwd time.
# All 9 gbranaa4-hue GitHub repos, mapped to their local checkouts. NOTE the
# local folder names differ from the repo names in three cases (verified via
# `git remote get-url origin`, not guessed):
#   012-trit-search      -> 012-ternary
#   horde-defense-beta   -> horde-beta-version-1
#   gbranaa4-hue         -> gbranaa4-hue (profile README repo)
_PROJECT_CANDIDATES = {
    # ── games / tools ──
    "spikeling": CLAUDE_CODE_PROJECT_DIR,                                       # Spikeling
    "tribe":     r"C:\Users\gbran\OneDrive\Documents\tribe",                    # tribe
    "horde":     r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1",     # horde-defense-beta
    # ── research repos -- same pipeline, same method (pre-register -> measure
    #    -> independent peer review -> honest ledger), pointed at research code
    "phononics": r"C:\Users\gbran\OneDrive\Documents\topological-phononics",    # topological-phononics
    "ternary":   r"C:\Users\gbran\OneDrive\Documents\012-ternary",              # 012-trit-search (+ OBSERVE)
    "methodlm":  r"C:\Users\gbran\OneDrive\Documents\methodlm",                 # methodlm
    "symmetry":  r"C:\Users\gbran\OneDrive\Documents\symmetry-selection-rule",  # symmetry-selection-rule
    "quasicrystal": r"C:\Users\gbran\OneDrive\Documents\quasicrystal-mems-reservoir",
    "profile":   r"C:\Users\gbran\OneDrive\Documents\gbranaa4-hue",             # profile README
}
PROJECTS = {name: path for name, path in _PROJECT_CANDIDATES.items() if os.path.isdir(path)}

def resolve_project(task_text):
    """If `task_text` starts with a known project name followed by a colon
    (e.g. "tribe: add X"), returns (project_dir, project_name, remaining_task).
    Otherwise defaults to Spikeling and returns the task unchanged. Keeps
    project selection inside the fixed PROJECTS allowlist -- never an
    arbitrary path."""
    m = re.match(r"^\s*([a-zA-Z0-9_-]+)\s*:\s*(.+)$", task_text, re.DOTALL)
    if m and m.group(1).lower() in PROJECTS:
        name = m.group(1).lower()
        return PROJECTS[name], name, m.group(2).strip()
    return PROJECTS.get("spikeling", CLAUDE_CODE_PROJECT_DIR), "spikeling", task_text
CLAUDE_CODE_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "WebSearch", "WebFetch"]  # NO Bash
REVIEW_TOOLS = ["Read", "Grep", "Glob"]   # peer reviewer: read-only, can't edit to make a claim pass

# RESEARCH repos get a SHELL (user-approved, research-only). "Measure, don't
# infer" is impossible without running the code -- an agent that can't execute
# can only propose, never measure, which makes a research pipeline useless.
# Scope of the escalation, honestly:
#   * ONLY the 5 research repos. Games/tools stay no-Bash.
#   * Still admin-gated, still confined to one repo dir via cwd.
#   * A shell DOES bypass the wording-based destructive/financial guards --
#     the shell rules below are instructions, not enforcement. The real
#     backstops are: the confined cwd, the admin gate, and the user reviewing
#     every diff (agents are told never to commit/push).
#   * The REVIEWER gets Bash but NOT Edit/Write: it can independently re-run to
#     verify a reported number, but cannot change the code to make it pass.
RESEARCH_CODE_TOOLS = CLAUDE_CODE_TOOLS + ["Bash"]
RESEARCH_REVIEW_TOOLS = REVIEW_TOOLS + ["Bash"]

RESEARCH_SHELL_RULES = """
--- SHELL ACCESS (research repo) ---
You DO have a shell here, confined to this repository. USE IT: actually run the
code and report REAL measured numbers. A claim you did not measure is worthless.
NEVER: `git push` / `git commit` / publish anything; delete or overwrite data or
results you did not create in this task (no `rm -rf`, no force-clean); touch
anything outside this repository; send data off the machine. If you install a
package, say so plainly in your summary.
Leave all changes in the working tree -- the user reviews every diff themselves.
"""
CLAUDE_CODE_PREAMBLE = """You are Claude Code, driven remotely through a chat
bot to work on the project in your current directory. You may read and edit
files here and search the web. You do NOT have shell access -- if a step
needs a command, say so instead of trying. Make the requested change directly
in the files, then reply with a SHORT plain-text summary (2-4 sentences, no
markdown, no code blocks -- it's read aloud by text-to-speech) of exactly
what you changed and in which files, so the user can go review the diff.

LANGUAGE-SPECIFIC CARE -- this environment cannot compile or run the code, so
avoid errors a compiler would otherwise catch. In particular, for GDScript
(Godot .gd files): do NOT use `:=` type inference when the right-hand side is a
Variant -- indexing an UNTYPED Array or Dictionary (e.g. `var x := MY_ARRAY[i]`
or `var x := my_dict.get(k)`) yields a Variant and Godot 4 rejects the inferred
`:=`. Use an explicit type instead (`var x: String = MY_ARRAY[i]`) or cast.
Match the existing file's style, indentation (tabs in GDScript), and typing
conventions. Prefer changes that a reader can verify against the file without
running it."""

# LIVE PROGRESS + SESSION CONTINUITY (2026-07-28): "interact fully with
# claude remotely" -- the old do_claude_code() ran a single blocking
# subprocess.run() and stayed completely silent for the whole task (up to
# 300-900s), then dumped one final summary. That's "fire a task and wait
# blind," not interaction. Two changes:
#   1. --output-format stream-json relays each real tool call live via
#      say() AS IT HAPPENS (already TTS-suppressed for Discord/webui, see
#      _suppress_tts above -- this doesn't add speaker noise).
#   2. --resume <session_id> lets a FOLLOW-UP message continue the same
#      Claude conversation (it remembers prior turns) instead of starting
#      blind each time -- tracked per project_dir so switching projects
#      doesn't resume the wrong conversation.
_last_claude_session = {"id": None, "project_dir": None}

def _relay_claude_event(event):
    """One stream-json line -> a short live status via say(), or nothing for
    events with nothing worth announcing. The final result text is handled
    by the caller (returned, not said here), so it can be combined with the
    handler's own "done." wrapper."""
    if event.get("type") != "assistant":
        return
    for block in (event.get("message", {}) or {}).get("content", []) or []:
        if block.get("type") != "tool_use":
            continue
        name = block.get("name", "?")
        inp = block.get("input", {}) or {}
        detail = inp.get("file_path") or inp.get("path") or inp.get("pattern") or inp.get("command") or ""
        detail = str(detail)
        if "/" in detail or "\\" in detail:
            detail = os.path.basename(detail)
        detail = detail[:60]
        say(f"→ {name}({detail})" if detail else f"→ {name}")

def do_claude_code(task=None, project_dir=None, tools=None, timeout=300, resume=False):
    """Runs an agentic Claude Code task in the confined project dir with an
    editing (no-Bash) tool allowlist. Returns a short plain-text summary.
    Only ever called from its handler, which only fires after the admin
    gate has already passed in process_text(). `task` defaults to
    _pending_arg (the original single-entry-point call shape); do_agent_task
    below calls this directly with an already-clarified task string.
    `project_dir` defaults to Spikeling but may be any dir from the PROJECTS
    allowlist (resolved by the caller). `resume=True` continues the last
    Claude conversation in this SAME project_dir, if any (see
    _last_claude_session above) -- a genuine follow-up, not a fresh blind
    task."""
    task = task if task is not None else _pending_arg
    project_dir = project_dir or CLAUDE_CODE_PROJECT_DIR
    tools = tools or CLAUDE_CODE_TOOLS
    if not (CLAUDE_CLI and task):
        return None
    # The base preamble says "you have NO shell". That's a lie for a research
    # repo (which gets Bash), so swap that line out rather than contradict it.
    preamble = CLAUDE_CODE_PREAMBLE
    if "Bash" in tools:
        preamble = preamble.replace(
            "You do NOT have shell access -- if a step\nneeds a command, say so instead of trying.",
            "You DO have shell access here (see the shell rules in the task).")
    prompt = f"{preamble}\n\nTask: {task}"

    args = [CLAUDE_CLI, "-p", "--allowedTools", *tools,
            "--output-format", "stream-json", "--verbose"]
    if resume and _last_claude_session["id"] and _last_claude_session["project_dir"] == project_dir:
        args += ["--resume", _last_claude_session["id"]]

    try:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=project_dir, text=True,
                                 encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"(Claude Code task failed to start: {e})", flush=True)
        return None

    result_text = None
    start = time.time()
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
        for line in proc.stdout:
            if time.time() - start > timeout:
                proc.kill()
                print("(Claude Code task timed out, killed)", flush=True)
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            _relay_claude_event(event)
            if event.get("type") == "result":
                result_text = event.get("result")
                sid = event.get("session_id")
                if sid:
                    _last_claude_session["id"] = sid
                    _last_claude_session["project_dir"] = project_dir
    except Exception as e:
        print(f"(Claude Code task failed: {e})", flush=True)
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    return _strip_markdown(result_text.strip()) if result_text else None

# Modular "agent task" layer on top of do_claude_code(): before touching any
# files, Claude first self-assesses whether the request is actually
# unambiguous. If not, it asks ONE short clarifying question back through
# Discord and stops -- no files touched, nothing half-done -- instead of
# guessing at scope on a real edit to this project's own source. Explicitly
# scoped to Spikeling itself (CLAUDE_CODE_PROJECT_DIR default) and only ever
# runs when the user triggers it from Discord; no polling, no autonomy loop.
AGENT_CLARIFY_PREAMBLE = """You are about to do agentic coding work (read/edit
files, no shell) on a real project codebase. Before starting, decide: is this
task specific enough to act on directly, or is there a genuine ambiguity that
would make you guess at something the user actually cares about (which file/
module, which of several reasonable approaches, a missing detail that changes
the implementation)?

If it's clear enough, reply with EXACTLY: PROCEED
If there's a real ambiguity worth asking about, reply with EXACTLY:
CLARIFY: <one short, specific question>
Do not ask about trivial style choices -- only ask if guessing wrong would
mean redoing real work. Prefer PROCEED when in doubt; only CLARIFY for
genuine forks in the road."""

# gbranaa-hue method discipline applied to code changes: pre-register a
# falsifiable claim BEFORE acting, so "did it work" is checked against a
# stated prediction, not judged after the fact by whoever did the work.
PREREGISTER_PREAMBLE = """Before writing any code, state ONE short, concrete,
falsifiable claim about what this change will do once implemented -- something
a reviewer could check against the actual files afterward (e.g. "adds function
X to file Y that returns Z" or "command A now triggers on phrase B and calls
C"). Not a plan, not a list of steps -- one checkable sentence. Reply with
ONLY that sentence, nothing else."""

# Independent peer review: a SEPARATE Claude call, read-only tools, that
# verifies the pre-registered claim against the REAL current files -- not
# against the implementer's own summary. This is "measure, don't infer"
# applied to code review: check the codebase, not the report about the
# codebase.
PEER_REVIEW_PREAMBLE = """You are peer-reviewing a code change to this project.
You did NOT make the change. Using Read/Grep/Glob, actually inspect the
current files to verify: (1) does the pre-registered claim hold in the real
code right now, (2) does the implementer's self-reported summary match what
you actually find (call out any overclaiming), (3) any obvious correctness
bug in what was added/changed. Do not trust the summary text -- check the
files yourself.

If everything genuinely checks out, reply starting with exactly: VERIFIED
followed by one short sentence of what you confirmed.
If there's a real problem, reply starting with exactly: ISSUE
followed by one short, specific, actionable sentence describing exactly what's
wrong (precise enough that a fix could be attempted from your sentence alone).
Be honest and specific -- a false VERIFIED defeats the entire point of review."""

def do_agent_task(task):
    """The full modular pipeline: resolve target project -> clarify-check ->
    (ask & stop) or (pre-register -> implement -> peer review -> maybe
    correct) -> log to vault. Returns the text to say. The task may be
    prefixed with a project name ("tribe: add X") to target a game codebase
    from the PROJECTS allowlist; defaults to Spikeling."""
    if not (CLAUDE_CLI and task):
        return None
    project_dir, project_name, task = resolve_project(task)
    vault_dir = os.path.join(VAULT_DIR, "Project Work")
    task_label = f"[{project_name}] {task}"   # so the vault ledger shows which project
    try:
        clarify = subprocess.run([CLAUDE_CLI, "-p"],
                                 input=f"{AGENT_CLARIFY_PREAMBLE}\n\nTASK: {task}",
                                 cwd=project_dir, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=60)
        verdict = clarify.stdout.strip()
    except Exception:
        verdict = "PROCEED"   # the clarify pre-check failing shouldn't block real work
    if verdict.upper().startswith("CLARIFY"):
        question = verdict.split(":", 1)[-1].strip()
        log_to_vault("agent_task", task_label, output=f"NEEDS CLARIFICATION: {question}",
                      status="needs_clarification", notes_dir=vault_dir)
        return (f"quick question before i touch anything on {project_name}, bro: {question}\n"
                 f"(hit me with the answer folded into a fresh task and I'll run it for real)")

    # Cog 2: PRE-REGISTER -- gbranaa-hue method discipline (see
    # method_copilot_build memory: pre-register before acting, so success
    # is checked against a stated claim instead of judged after the fact by
    # whoever did the work). One short, concrete, falsifiable claim about
    # what the change will DO, written before any file is touched.
    try:
        prereg_run = subprocess.run([CLAUDE_CLI, "-p"],
                                     input=f"{PREREGISTER_PREAMBLE}\n\nTASK: {task}",
                                     cwd=project_dir, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=60)
        prereg = prereg_run.stdout.strip() or "(no pre-registration returned)"
    except Exception as e:
        prereg = f"(pre-registration failed: {e})"
    say(f"pre-registered the claim, implementing the change on {project_name} now (this is the slow part)...")

    # Cog 3: IMPLEMENT -- the existing agentic editor, pointed at the target
    # project. Inject any relevant hard-won lessons from the vault first, so
    # the agent avoids known traps (e.g. the GDScript := inference error)
    # instead of re-hitting them. This is the "agents use memory efficiently"
    # loop -- past failures actively shape new work.
    lessons = relevant_lessons(task)
    implement_task = task
    if lessons:
        implement_task = (f"{task}\n\n--- RELEVANT LESSONS FROM PAST WORK (heed these to avoid "
                           f"known mistakes) ---\n{lessons}")
        print(f"(injected {lessons.count('# ')} relevant lesson(s) from the vault)", flush=True)
    # For a RESEARCH project, also hand the agent what past sessions already
    # measured -- including results that were honestly retired. Without this it
    # re-derives known ground, or re-claims something already falsified.
    is_research = project_name in RESEARCH_PROJECTS
    if is_research:
        implement_task += "\n" + RESEARCH_METHOD_PREAMBLE + RESEARCH_SHELL_RULES
        findings = relevant_research(task)
        if findings:
            implement_task += (f"\n\n--- PRIOR MEASURED FINDINGS on this research (from the vault). "
                                f"Build on these; do NOT re-claim anything already retired here, and "
                                f"do NOT re-derive what's already measured. ---\n{findings}")
            print(f"(injected {findings.count('# ')} prior research finding(s) from the vault)", flush=True)
        print("(research project -- method + shell rules injected, Bash enabled)", flush=True)
    # Research gets a shell and a longer leash (real benchmarks take minutes).
    summary = do_claude_code(implement_task, project_dir=project_dir,
                             tools=RESEARCH_CODE_TOOLS if is_research else CLAUDE_CODE_TOOLS,
                             timeout=900 if is_research else 300)
    if not summary:
        log_to_vault("agent_task", task_label, output=f"PRE-REGISTERED:\n{prereg}\n\nIMPLEMENTATION: (none returned)",
                      status="failed", notes_dir=vault_dir)
        return "couldn't complete that one, bro -- Claude Code didn't return anything."

    say("change is in -- running independent peer review against the real files now...")
    # Cog 4: PEER REVIEW -- a SEPARATE, independent Claude call, read-only
    # tools (Read/Grep/Glob, no Edit/Write), that re-derives whether the
    # pre-registered claim actually holds by inspecting the real files --
    # not by trusting the implementer's own self-report. This is the
    # "measure, don't infer" half of the method: the reviewer checks the
    # codebase itself, not the summary text.
    try:
        review_prompt = (f"{PEER_REVIEW_PREAMBLE}\n\nORIGINAL TASK: {task}\n\n"
                          f"PRE-REGISTERED CLAIM: {prereg}\n\n"
                          f"IMPLEMENTER'S SELF-REPORTED SUMMARY: {summary}")
        if is_research:
            # The reviewer gets a SHELL but no Edit/Write: it can INDEPENDENTLY
            # RE-RUN the code to check a reported number, but cannot alter the
            # code to make a claim pass. That's what makes this real peer review
            # of a measurement rather than of prose.
            review_prompt += (
                "\n\nYou have a SHELL (read-only tools otherwise -- you cannot edit).\n"
                "This is a RESEARCH claim: do NOT take the reported numbers on faith. "
                "RE-RUN the relevant code yourself and compare what you get to what was "
                "reported. If a number can't be reproduced, that is an ISSUE. Also check "
                "the instrument: would a naive baseline pass this same test? Never "
                "`git commit`/`push`, never delete data.")
        review_run = subprocess.run(
            [CLAUDE_CLI, "-p", "--allowedTools", *(RESEARCH_REVIEW_TOOLS if is_research else REVIEW_TOOLS)],
            input=review_prompt, cwd=project_dir,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600 if is_research else 180)
        review = review_run.stdout.strip() or "REVIEW_FAILED: no output"
    except Exception as e:
        review = f"REVIEW_FAILED: {e}"
    review_passed = review.upper().startswith("VERIFIED")

    # Cog 5: ONE correction pass if peer review found a real discrepancy --
    # not a retry loop (bounded, so this can't spiral), and the correction
    # itself gets logged either way so the ledger stays honest even if the
    # fix doesn't fully land.
    if not review_passed and not review.upper().startswith("REVIEW_FAILED"):
        correction_task = (f"{task}\n\nA peer review of your previous attempt at this task found "
                            f"a problem: {review}\nFix it.")
        # the correction re-implements, so it needs the same tools/leash as the
        # implement step did (a research fix usually means re-running something)
        correction_summary = do_claude_code(
            correction_task, project_dir=project_dir,
            tools=RESEARCH_CODE_TOOLS if is_research else CLAUDE_CODE_TOOLS,
            timeout=900 if is_research else 300)
        final_status = "corrected_after_review" if correction_summary else "review_failed_uncorrected"
        log_to_vault("agent_task", task_label,
                     output=(f"PRE-REGISTERED:\n{prereg}\n\nIMPLEMENTATION:\n{summary}\n\n"
                             f"PEER REVIEW:\n{review}\n\nCORRECTION:\n{correction_summary or '(correction failed)'}"),
                     status=final_status, notes_dir=vault_dir)
        return (f"did it on {project_name}, but peer review caught an issue and I fixed it, bro: {review}\n"
                 f"correction: {correction_summary or 'correction attempt failed -- take a look yourself'}")

    log_to_vault("agent_task", task_label,
                 output=f"PRE-REGISTERED:\n{prereg}\n\nIMPLEMENTATION:\n{summary}\n\nPEER REVIEW:\n{review}",
                 status="completed_verified" if review_passed else "completed_review_inconclusive",
                 notes_dir=vault_dir)
    return summary + (f"\n(peer-reviewed: {review})" if review_passed else "\n(peer review was inconclusive, worth a look)")

# ── Discord-driven tribe control: a work QUEUE + status report ──
# "work on tribe" / "let's work on tribe" pops the next undone line from
# tribe_aaa_queue.txt and runs it through the full do_agent_task pipeline;
# "how's my tribe game" reports progress. FPS + high-graphics tasks are
# front-loaded in the queue file itself. The queue is edited in place (a
# done line gets " @done" appended) so progress survives restarts.
TRIBE_QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tribe_aaa_queue.txt")

def _read_queue():
    """Returns (lines, next_index) where next_index is the first undone,
    non-comment, non-blank task line -- or -1 if the queue is exhausted."""
    try:
        with open(TRIBE_QUEUE_FILE, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return [], -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s and not s.startswith("#") and "@done" not in s:
            return lines, i
    return lines, -1

def _mark_queue_done(lines, idx):
    lines[idx] = lines[idx].rstrip() + "   @done"
    try:
        with open(TRIBE_QUEUE_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"(couldn't mark queue done, non-fatal: {e})", flush=True)

def _roadmap_tasks():
    """Extract every `work on tribe: <task>` line from the AAA roadmap as a
    'tribe: <task>' string. This is the big backlog the queue cycles through.
    VAULT_DIR is resolved at call time (it's defined later in the module)."""
    roadmap_file = os.path.join(VAULT_DIR, "AAA_Roadmap_Tribe.md")
    try:
        with open(roadmap_file, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return []
    out = []
    for m in re.finditer(r"`work on (tribe:[^`]+)`", text):
        out.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return out

def _refill_tribe_queue():
    """LOOP the pipeline: when the queue has no undone task, refill it. First
    from the 100-task roadmap (tasks not already queued), then -- if even the
    roadmap is exhausted -- generate a fresh round of polish tasks via Claude.
    So 'work on tribe' never dead-ends; it cycles the 10 pillars indefinitely."""
    lines, idx = _read_queue()
    if idx >= 0:
        return   # still have work; nothing to refill
    try:
        with open(TRIBE_QUEUE_FILE, encoding="utf-8") as f:
            existing = f.read()
    except Exception:
        existing = ""
    existing_norm = re.sub(r"\s+", " ", existing.lower())
    # 1) pull roadmap tasks not already present in the queue
    fresh = [t for t in _roadmap_tasks()
             if re.sub(r"\s+", " ", t.lower())[:60] not in existing_norm]
    if not fresh and CLAUDE_CLI:
        # 2) roadmap exhausted -- generate a new round of pillar tasks
        try:
            gen = subprocess.run([CLAUDE_CLI, "-p"],
                                 input=("Generate 8 NEW, concrete, single-sentence polish/improvement tasks "
                                        "for a Godot game called 'tribe' (a survival sim with spiking-neural-net "
                                        "NPCs), spread across game feel, audio, UI, visuals, animation, camera, "
                                        "performance, and content. Each MUST start with 'tribe: ' and be a bounded, "
                                        "reviewable change. Output ONLY the 8 lines, nothing else."),
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=60)
            fresh = [ln.strip() for ln in gen.stdout.splitlines()
                     if ln.strip().lower().startswith("tribe:")][:8]
        except Exception as e:
            print(f"(queue regen failed: {e})", flush=True)
            fresh = []
    if fresh:
        try:
            with open(TRIBE_QUEUE_FILE, "a", encoding="utf-8") as f:
                f.write("\n# --- auto-refilled %s ---\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                for t in fresh[:12]:
                    f.write(t + "\n")
            print(f"(auto-refilled tribe queue with {len(fresh[:12])} task(s))", flush=True)
        except Exception as e:
            print(f"(couldn't refill queue, non-fatal: {e})", flush=True)

def do_tribe_work():
    """Pops the next queued tribe task and runs it through the full pipeline.
    The user drives the loop by repeating 'work on tribe'. The queue auto-
    refills from the roadmap (then Claude-generated tasks) so it never
    dead-ends -- it loops the 10 pillars indefinitely. Admin-gated."""
    _refill_tribe_queue()
    lines, idx = _read_queue()
    if idx < 0:
        return ("tribe queue's all done and I couldn't auto-refill it, bro -- "
                "check the roadmap or say a specific 'work on tribe: <task>'.")
    task = lines[idx].strip()
    remaining = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#") and "@done" not in ln.strip()) - 1
    result = do_agent_task(task)
    _mark_queue_done(lines, idx)   # mark done AFTER it ran (even if review was iffy -- it's committed to the ledger either way)
    return f"{result}\n\n(that was one off the tribe queue -- {remaining} left. say 'work on tribe' again to keep going.)"

def do_tribe_status():
    """Reports how the tribe game work is going: git commits so far, queue
    progress, and the most recent pipeline results from the vault."""
    tribe_dir = PROJECTS.get("tribe")
    if not tribe_dir:
        return "can't find the tribe project on disk, bro."
    # git state
    try:
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=tribe_dir,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        log = subprocess.run(["git", "log", "--oneline", "-5"], cwd=tribe_dir,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--short"], cwd=tribe_dir,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        branch, log, dirty = "?", "(git unavailable)", ""
    # queue progress
    lines, idx = _read_queue()
    total = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    done = sum(1 for ln in lines if "@done" in ln)
    nextup = lines[idx].strip().replace("tribe: ", "")[:80] if idx >= 0 else "(queue empty)"
    uncommitted = f"{len(dirty.splitlines())} file(s) changed but not committed" if dirty else "working tree clean"
    return (f"tribe's on branch '{branch}', {done}/{total} queued tasks done, {uncommitted}. "
            f"next up: {nextup}.\nrecent commits:\n{log}")

# Write-a-Python-script-and-RUN-it experiment loop. This is genuine code
# EXECUTION (a Python script can do anything a shell can) -- the powerful
# tier. Kept as tight as honestly possible: admin-gated, runs in a
# dedicated experiments dir (not the real project), 60s timeout kills
# runaways, and the destructive/financial/credential request-guards still
# refuse obviously-bad asks. What it can't do is sandbox the interpreter
# itself -- that'd need a container. Confined to the authorized user.
EXPERIMENTS_DIR = os.environ.get(
    "VOICE_EXPERIMENTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
NODE_BIN = shutil.which("node")

# Obsidian vault -- just a folder of markdown files, Obsidian needs no API
# or integration work, it opens any folder as a vault. Every experiment/
# screenshot gets a note here so there's a browsable, searchable history
# instead of a pile of anonymous timestamped files in experiments/.
VAULT_DIR = os.environ.get(
    "VOICE_VAULT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault"))
VAULT_NOTES_DIR = os.path.join(VAULT_DIR, "Experiments")
VAULT_ATTACH_DIR = os.path.join(VAULT_DIR, "attachments")
VAULT_LESSONS_DIR = os.path.join(VAULT_DIR, "Lessons")
VAULT_RESEARCH_DIR = os.path.join(VAULT_DIR, "Research")

# Research projects get PRIOR FINDINGS injected the way code projects get
# Lessons -- so an agent picking up a research thread starts from what was
# already measured (and what was already falsified) instead of re-deriving or,
# worse, re-claiming something a past session honestly retired.
RESEARCH_PROJECTS = {"phononics", "ternary", "methodlm", "symmetry", "quasicrystal"}

# The gbranaa-hue method, encoded. Injected for RESEARCH_PROJECTS so the agent
# is held to the same discipline the research itself is held to -- this is what
# makes it a research pipeline and not just an editor pointed at a science repo.
RESEARCH_METHOD_PREAMBLE = """
--- RESEARCH METHOD (this is a RESEARCH repo -- hold to this) ---
- MEASURE, DON'T INFER. Report only numbers you actually computed. You have NO
  shell here, so you cannot run anything: if a claim needs a run, WRITE the
  script and say it must be run -- never assert an outcome you didn't measure.
- PRE-REGISTER. A falsifiable prediction was stated before this work. Judge the
  result against THAT, not a story invented afterward to fit what came out.
- VALIDATE THE INSTRUMENT. Before trusting a result, check the measurement isn't
  measuring luck or an artifact. A benchmark a naive baseline already passes is
  not a test of anything.
- CORRELATION != CAUSATION. To claim X drives Y, clamp/control the confound.
- PREFER THE BORING EXPLANATION. A bug, a confound, or a construction artifact
  is far more likely than a discovery.
- FIND THE BOUNDARY. An honest result states where it STOPS holding.
- HONEST LEDGER. A negative, null, or retired result IS a result -- report it
  plainly. Never dress a null up as a win, and never quietly revive a finding a
  past session already retired (the prior findings above say which those are).
"""

def relevant_research(task, max_notes=2):
    """Returns prior research findings from vault/Research whose scope/content
    overlaps `task`. Truncated per-note -- these are long-form findings and only
    the gist is needed as context."""
    if not os.path.isdir(VAULT_RESEARCH_DIR):
        return ""
    stop = {"the", "and", "for", "add", "with", "that", "this", "into", "from",
            "run", "test", "make", "use", "a", "an", "to", "of", "in", "on", "is"}
    q = {w for w in re.findall(r"[a-z0-9]+", task.lower()) if len(w) > 2 and w not in stop}
    scored = []
    for fn in os.listdir(VAULT_RESEARCH_DIR):
        if not fn.endswith(".md"):
            continue
        try:
            with open(os.path.join(VAULT_RESEARCH_DIR, fn), encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        scope_m = re.search(r"^scope:\s*(.+)$", content, re.M)
        scope_words = set(re.findall(r"[a-z0-9]+", (scope_m.group(1).lower() if scope_m else "")))
        body_words = set(re.findall(r"[a-z0-9]+", content.lower()))
        score = len(q & scope_words) * 3 + len(q & body_words)
        if (q & scope_words) or score >= 4:
            scored.append((score, content.strip()[:2200]))
    scored.sort(key=lambda x: -x[0])
    return "\n\n---\n\n".join(c for _, c in scored[:max_notes])

def relevant_lessons(task, max_lessons=3):
    """Returns the text of Lessons/ notes whose scope/content keyword-overlaps
    `task`, so an agent can be reminded of known gotchas BEFORE implementing --
    the 'learn from past failure' loop. Empty string if none match. Kept small
    (a few short lessons) so it doesn't bloat the prompt."""
    if not os.path.isdir(VAULT_LESSONS_DIR):
        return ""
    stop = {"the", "and", "for", "add", "with", "that", "this", "into", "from",
            "tribe", "game", "make", "use", "using", "a", "an", "to", "of", "in"}
    q = {w for w in re.findall(r"[a-z0-9]+", task.lower()) if len(w) > 2 and w not in stop}
    scored = []
    for fn in os.listdir(VAULT_LESSONS_DIR):
        if not fn.endswith(".md"):
            continue
        try:
            with open(os.path.join(VAULT_LESSONS_DIR, fn), encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        # weight the `scope:` frontmatter line heavily -- it's the routing tag
        scope_m = re.search(r"^scope:\s*(.+)$", content, re.M)
        scope_words = set(re.findall(r"[a-z0-9]+", (scope_m.group(1).lower() if scope_m else "")))
        body_words = set(re.findall(r"[a-z0-9]+", content.lower()))
        score = len(q & scope_words) * 3 + len(q & body_words)
        # Require a scope-tag hit OR a couple of body hits -- one incidental
        # shared word (e.g. "script" in both a Python task and a GDScript
        # lesson) shouldn't pull an irrelevant lesson in.
        if (q & scope_words) or score >= 3:
            scored.append((score, content.strip()))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return ""
    return "\n\n---\n\n".join(c for _, c in scored[:max_lessons])

def log_to_vault(kind, task, code=None, output=None, attachment_path=None, status="", notes_dir=None):
    """Writes one markdown note per experiment/screenshot/agent-task into the
    Obsidian vault, with the code/output inline and any image copied into
    vault/attachments and embedded. `notes_dir` defaults to
    vault/Experiments; agent tasks use vault/Project Work instead. Best-effort
    -- a logging failure should never break the actual feature, so every
    error is swallowed."""
    try:
        notes_dir = notes_dir or VAULT_NOTES_DIR
        os.makedirs(notes_dir, exist_ok=True)
        os.makedirs(VAULT_ATTACH_DIR, exist_ok=True)
        ts = datetime.datetime.now()
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:60] or "untitled"
        note_path = os.path.join(notes_dir, f"{stamp}_{slug}.md")
        lines = [
            "---",
            f"date: {ts.isoformat()}",
            f"kind: {kind}",
            f"status: {status}",
            "---",
            "",
            f"# {task}",
            "",
        ]
        ext_hint = os.path.splitext(attachment_path)[1].lstrip(".").lower() if attachment_path else ""
        if attachment_path and os.path.exists(attachment_path) and ext_hint in ("png", "jpg", "jpeg", "gif"):
            att_name = f"{stamp}_{os.path.basename(attachment_path)}"
            att_dest = os.path.join(VAULT_ATTACH_DIR, att_name)
            shutil.copy2(attachment_path, att_dest)
            lines += [f"![[{att_name}]]", ""]
        if code:
            lang_hint = {"py": "python", "js": "javascript", "jsx": "jsx"}.get(ext_hint, "")
            lines += [f"```{lang_hint}", code.strip(), "```", ""]
        if output:
            lines += ["## Output", "```", output.strip()[:2000], "```", ""]
        with open(note_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"(vault logging failed, non-fatal: {e})", flush=True)
RENDER_REACT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_react.js")

def _detect_experiment_lang(task):
    """React can't actually be executed headlessly (needs a browser/bundler)
    -- write-only, honestly labeled. JS/Node is fully write-and-run like
    Python. Defaults to Python."""
    t = task.lower()
    if re.search(r"\breact\b|\bjsx\b|\bcomponent\b.*\breact\b", t):
        return "react"
    if re.search(r"\bjavascript\b|\bnode(?:\.js)?\b|\bjs\b", t):
        return "javascript"
    return "python"

CODE_GEN_PREAMBLE = """Write a complete, self-contained, runnable Python 3
script for the task below. Output ONLY the raw Python code -- no markdown,
no triple-backtick fences, no prose or explanation before or after. Include
a small self-check that prints a clear result so running it demonstrates it
works. Task:"""

CODE_GEN_PREAMBLE_JS = """Write a complete, self-contained, runnable Node.js
script (plain JavaScript, no external npm packages, no bundler) for the task
below. Output ONLY the raw JavaScript code -- no markdown, no triple-backtick
fences, no prose before or after. Include a small self-check that console.logs
a clear result so running it demonstrates it works. Task:"""

CODE_GEN_PREAMBLE_REACT = """Write a complete, self-contained React function
component (JSX) for the task below. It must have a default export, take NO
required props (use sensible default/initial state so it renders standalone),
and use only React + inline styles or plain CSS-in-JS (no external CSS files,
no non-React npm packages). Output ONLY the raw code -- no markdown, no
triple-backtick fences, no prose before or after. This WILL be rendered
headlessly and screenshotted, so make sure it renders something visible.
Task:"""

VERIFY_PREAMBLE = """You are reviewing generated code before it runs. Check
it against the task it was supposed to accomplish. Look for: does it
actually do what was asked (not something adjacent or partial), obvious
bugs/crashes, infinite loops, or anything that looks like it would run but
silently do the wrong thing. Do NOT be pedantic about style -- only flag
real correctness problems.
Reply with EXACTLY one line: "PASS" if it's fine, or "FAIL: <one short
reason>" if there's a real problem. No other text."""

def do_experiment(task, lang=None):
    """Has Claude write code for `task` (Python, JS/Node, or React), saves it
    to the experiments dir, and actually runs it -- Python/JS execute for
    real output, React gets bundled + rendered headlessly and screenshotted.
    `lang` should be pre-detected from the FULL original sentence (the
    language keyword often sits in a phrase like "write a react component
    for X", which is stripped out of the extracted task text itself) --
    falls back to detecting from `task` alone if not given. Sets
    _last_attachment_path so the user also gets the code/screenshot. Only
    called after the admin gate passes."""
    global _last_attachment_path
    if not (CLAUDE_CLI and task):
        return None
    if lang is None:
        lang = _detect_experiment_lang(task)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    preamble, ext, fence = {
        "python": (CODE_GEN_PREAMBLE, "py", "python"),
        "javascript": (CODE_GEN_PREAMBLE_JS, "js", "(?:javascript|js)?"),
        "react": (CODE_GEN_PREAMBLE_REACT, "jsx", "(?:jsx|javascript|js)?"),
    }[lang]
    # 1. Generate the code (WebSearch allowed in case the task needs a fact).
    try:
        gen = subprocess.run([CLAUDE_CLI, "-p", "--allowedTools", "WebSearch"],
                             input=f"{preamble} {task}", cwd=EXPERIMENTS_DIR,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120)
        code = gen.stdout.strip()
    except Exception as e:
        return f"couldn't generate the code, bro: {e}"
    # Strip a markdown code fence if Claude wrapped it despite instructions.
    m = re.search(r"```" + fence + r"\s*\n(.*?)```", code, re.DOTALL)
    if m:
        code = m.group(1)
    code = code.strip()
    if not code:
        return "Claude didn't return any code, bro."

    # 1.5 Verification pass -- a second, independent Claude call sanity-checks
    # the generated code against the task before anything runs. This is a
    # real check, not theater: it can and does say NO (see VERIFY_PREAMBLE),
    # in which case the code is still saved (so you can see what it tried)
    # but is NOT executed/rendered. Kept intentionally cheap -- one short
    # yes/no-style call, not a second full generation.
    try:
        verify = subprocess.run([CLAUDE_CLI, "-p"],
                                 input=f"{VERIFY_PREAMBLE}\n\nTASK: {task}\n\nCODE:\n{code}",
                                 cwd=EXPERIMENTS_DIR, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=60)
        verdict = verify.stdout.strip()
    except Exception:
        verdict = "PASS"   # verification itself failing shouldn't block a run
    verify_failed = verdict.upper().startswith("FAIL")

    path = os.path.join(EXPERIMENTS_DIR, f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code + "\n")
    _last_attachment_path = path   # attach the actual code so the user gets it

    if verify_failed:
        reason = verdict.split(":", 1)[-1].strip()
        log_to_vault(lang, task, code=code, output=f"VERIFICATION FAILED: {reason}", status="verify_failed")
        return (f"wrote {os.path.basename(path)} but held off running it, bro -- "
                 f"verification flagged a problem: {reason}. "
                 f"Code's attached if you want to look.")

    if lang == "react":
        if not NODE_BIN:
            log_to_vault(lang, task, code=code, status="no_node")
            return f"wrote {os.path.basename(path)}, but node isn't installed here so I can't render it."
        png_path = os.path.splitext(path)[0] + ".png"
        try:
            run = subprocess.run([NODE_BIN, RENDER_REACT_SCRIPT, path, png_path],
                                 cwd=EXPERIMENTS_DIR, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=60)
            if run.returncode == 0 and os.path.exists(png_path):
                _last_attachment_path = png_path   # show the rendered result, not the source
                log_to_vault(lang, task, code=code, attachment_path=png_path, status="rendered")
                return f"wrote and rendered {os.path.basename(path)}, bro -- screenshot attached."
            err = (run.stderr or run.stdout or "").strip()[:1000]
            log_to_vault(lang, task, code=code, output=f"RENDER FAILED:\n{err}", status="render_failed")
            return f"wrote {os.path.basename(path)} but the render failed, bro:\n{err}"
        except subprocess.TimeoutExpired:
            log_to_vault(lang, task, code=code, status="render_timeout")
            return f"wrote {os.path.basename(path)} but the render timed out (60s)."
        except Exception as e:
            log_to_vault(lang, task, code=code, output=f"RENDER ERROR: {e}", status="render_error")
            return f"wrote {os.path.basename(path)} but the render errored: {e}"

    # 2. Run it (Python or JS), capture output, kill runaways.
    if lang == "javascript":
        if not NODE_BIN:
            log_to_vault(lang, task, code=code, status="no_node")
            return f"wrote {os.path.basename(path)}, but node isn't installed here so I can't run it."
        cmd = [NODE_BIN, path]
    else:
        cmd = [sys.executable, path]
    # Force the child's own stdout/stderr encoding to UTF-8 -- Windows' default
    # console codepage (cp1252) crashes on generated scripts that print
    # Unicode block chars/emoji for things like progress bars.
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        run = subprocess.run(cmd, cwd=EXPERIMENTS_DIR, env=child_env,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60)
        out = (run.stdout or "")
        if run.stderr:
            out += ("\n[stderr]\n" + run.stderr)
        out = out.strip()[:1500] or "(the script produced no output)"
        status = "ran clean" if run.returncode == 0 else "ran but errored"
    except subprocess.TimeoutExpired:
        out, status = "(killed -- it ran longer than 60 seconds)", "timed out"
    except Exception as e:
        out, status = f"(couldn't run it: {e})", "failed to run"
    log_to_vault(lang, task, code=code, output=out, attachment_path=path, status=status)
    return f"wrote {os.path.basename(path)} and {status}, bro. output:\n{out}"

SCREENSHOT_URL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshot_url.js")

def do_screenshot_website(url):
    """Loads `url` in real headless Chromium and screenshots it, using the
    same Playwright install the React render path uses. Sets
    _last_attachment_path to the PNG. Admin-gated (it fetches an arbitrary
    external URL and renders it)."""
    global _last_attachment_path
    if not NODE_BIN:
        return "node isn't installed here so I can't screenshot that, bro."
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    png_path = os.path.join(EXPERIMENTS_DIR, f"shot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    try:
        run = subprocess.run([NODE_BIN, SCREENSHOT_URL_SCRIPT, url, png_path],
                             cwd=EXPERIMENTS_DIR, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=45)
        if run.returncode == 0 and os.path.exists(png_path):
            _last_attachment_path = png_path
            log_to_vault("website", url, attachment_path=png_path, status="screenshotted")
            return f"screenshotted {url}, bro -- attached."
        err = (run.stderr or run.stdout or "").strip()[:800]
        log_to_vault("website", url, output=f"FAILED:\n{err}", status="failed")
        return f"couldn't screenshot {url}, bro:\n{err}"
    except subprocess.TimeoutExpired:
        log_to_vault("website", url, status="timeout")
        return f"screenshotting {url} timed out, bro."
    except Exception as e:
        log_to_vault("website", url, output=f"ERROR: {e}", status="error")
        return f"couldn't screenshot {url}, bro: {e}"

SCREENSHOT_WEBSITE_PATTERNS = [
    re.compile(r"(?:hey\s+spike[,\s]*)?screenshot\s+(?:the\s+)?(?:website|site|page|url)?\s*:?\s*"
               r"(https?://\S+|[\w.-]+\.[a-z]{2,}\S*)\s*$", re.I),
    re.compile(r"(?:hey\s+spike[,\s]*)?take\s+a\s+screenshot\s+of\s+(?:the\s+)?(?:website|site|page)?\s*:?\s*"
               r"(https?://\S+|[\w.-]+\.[a-z]{2,}\S*)\s*$", re.I),
]

def extract_screenshot_website_url(text):
    for pat in SCREENSHOT_WEBSITE_PATTERNS:
        m = pat.match(text.strip())
        if m:
            return m.group(1).strip()
    return None

# Task-bearing triggers only -- they REQUIRE a "that/to/which/:" task clause
# after the noun, so a vague "can you write some python for me" stays
# conversational instead of firing a blank experiment.
EXPERIMENT_PATTERNS = [
    re.compile(r"(?:hey\s+spike[,\s]*)?(?:write|create|make|code|build)\s+(?:me\s+|us\s+)?"
               r"(?:a\s+|an\s+|some\s+)?(?:python\s+|py\s+|javascript\s+|js\s+|node\s*(?:\.js)?\s+|react\s+)?"
               r"(?:script|program|code|experiment|component)\s+"
               r"(?:that\s+|to\s+|which\s+|for\s+|:\s*)(.+?)(?:\s+and\s+(?:run|test|verify|check|render)\s+it\b.*)?$", re.I),
    re.compile(r"(?:hey\s+spike[,\s]*)?run\s+(?:an?\s+)?experiment\s+(?:that\s+|to\s+|:\s*)?(.+)", re.I),
    re.compile(r"(?:hey\s+spike[,\s]*)?(?:write|code)\s+(?:and\s+(?:run|test)\s+)?(?:some\s+)?"
               r"python\s+(?:that\s+|to\s+|:\s*)(.+?)(?:\s+and\s+(?:run|test|verify|check)\s+it\b.*)?$", re.I),
]

def extract_experiment_task(text):
    for pat in EXPERIMENT_PATTERNS:
        m = pat.match(text)
        if m:
            task = m.group(1).strip()
            if len(task.split()) >= 2:   # need a real task, not "it"/"me"
                return task
    return None

# Real bug found via live Discord testing: "yo put on some 90s rap classics
# playlist on youtube" fell through keyword matching (correctly -- "playlist"
# doesn't contain a word-boundary "play"), then got misclassified PLAYPAUSE
# by the offline LLM classifier, which confidently pressed a generic OS
# media key that did nothing useful -- there was no actual "play specific
# content" capability at all, just a wrong classification masking the gap.
# This adds the real capability (opens an actual YouTube search) AND is
# checked as its own prefix-style intent BEFORE keyword/LLM classify ever
# see the phrase, so it can't be misclassified into an unrelated command.
# Anchored with .match() (start of string only) and a REQUIRED action verb
# -- "how do i upload a video on youtube" must NOT match this (it's a
# genuine question, not a play request), confirmed by direct testing.
YOUTUBE_INTENT = re.compile(
    r"^(?:yo,?\s+|hey,?\s+)*(?:please\s+|can you\s+|could you\s+)*"
    r"(?:put on|play|watch|throw on|search(?: for)?|find)\s+(.+?)\s+on youtube\b",
    re.IGNORECASE)

def extract_youtube_query(text):
    m = YOUTUBE_INTENT.match(text)
    return m.group(1).strip() if m else None

def do_youtube_search():
    query = _pending_arg
    if query:
        webbrowser.open(f"https://www.youtube.com/results?search_query={quote(query)}")
    return query

# Real bug found via live Discord testing: "what day is it today" went to
# the knowledge-fallback LLM, which has NO access to the real system clock
# -- it's a static offline model with no real-time grounding -- and it
# just hallucinated a plausible-sounding wrong answer ("Today is a
# Wednesday, bro. Check your calendar...") instead of admitting it
# couldn't know. This is answerable EXACTLY and instantly from the real
# system clock, so it's checked as its own intent before the LLM path
# ever sees it, same reasoning as the YouTube/search/email prefixes.
# Anchored carefully (requires "is it"/"today"/"current" adjacent to the
# target word) so it doesn't collide with real technical questions like
# "what is the time complexity of quicksort" -- confirmed via direct
# testing before wiring in.
DATE_TIME_INTENT = re.compile(
    r"^(?:what|whats|what's)\s+(?:day|date|time)\s+is\s+it\b"
    r"|^(?:what|whats|what's)\s+(?:the|today'?s?)\s+(?:date|day|time)\b"
    r"|\bcurrent\s+(?:date|day|time)\b", re.IGNORECASE)

def do_current_datetime():
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y, %I:%M %p").replace(" 0", " ")

HEALTH_CHECK_INTENT = re.compile(
    r"\b(health\s+check|status\s+check|are\s+you\s+(?:working|ok|okay|up)|system\s+status)\b",
    re.IGNORECASE)

def do_health_check():
    playwright_ok = os.path.exists(RENDER_REACT_SCRIPT) and os.path.exists(SCREENSHOT_URL_SCRIPT)
    parts = [
        f"Claude CLI {'yes' if CLAUDE_CLI else 'no'}",
        f"Node {'yes' if NODE_BIN else 'no'}",
        f"Playwright scripts {'yes' if playwright_ok else 'no'}",
        f"admin TOTP {'yes' if ADMIN_TOTP_SECRET else 'no'}",
    ]
    return "status: " + ", ".join(parts) + "."

# Real bug found via live Discord testing: "when is the sunset today in
# placerville ca" got "I don't have real-time data" from the knowledge
# LLM -- an honest answer, but not a useful one, since sunset time is a
# precisely computable real-world fact, not something that needs
# real-time data at all. Uses Open-Meteo's free geocoding + forecast APIs
# (no API key needed) rather than guessing from the static model.
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}

# Requires "sunset"/"sunrise" PLUS an explicit "in <location>"/"for
# <location>" clause -- a bare "when is sunset" (no location) can't be
# answered anyway, so it correctly falls through to the honest-but-vague
# LLM fallback instead of guessing a location.
SUN_INTENT = re.compile(r"\b(sunset|sunrise)\b.*?\b(?:in|for)\s+(.+?)\s*$", re.IGNORECASE)

def extract_sun_query(text):
    m = SUN_INTENT.search(text)
    return (m.group(1).lower(), m.group(2).strip()) if m else None

def do_sun_lookup(kind, location_text):
    """Real sunrise/sunset for TODAY at the given location, via Open-Meteo's
    free geocoding + forecast APIs (no key needed). Multiple US towns often
    share a name (three separate real "Placerville"s turned up: CA, ID,
    CO) -- a trailing 2-letter state abbreviation is used to disambiguate
    by filtering geocoding results to that state, confirmed necessary by
    direct testing rather than assumed."""
    parts = location_text.replace(",", " ").split()
    state_hint = None
    if parts and parts[-1].lower() in US_STATES:
        state_hint = US_STATES[parts[-1].lower()]
        city = " ".join(parts[:-1])
    else:
        city = location_text
    if not city:
        return None
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode({"name": city, "count": 10})
        with urllib.request.urlopen(geo_url, timeout=10) as r:
            geo = json.loads(r.read())
        results = geo.get("results", [])
        if not results:
            return None
        if state_hint:
            filtered = [x for x in results if x.get("admin1", "").lower() == state_hint]
            results = filtered or results
        best = results[0]
        sun_url = "https://api.open-meteo.com/v1/forecast?" + urlencode({
            "latitude": best["latitude"], "longitude": best["longitude"],
            "daily": "sunrise,sunset", "timezone": best.get("timezone", "auto"),
        })
        with urllib.request.urlopen(sun_url, timeout=10) as r:
            sun = json.loads(r.read())
        dt = datetime.datetime.fromisoformat(sun["daily"][kind][0])
        place = f"{best['name']}, {best.get('admin1', best.get('country', ''))}"
        return f"{kind} in {place} today is at {dt.strftime('%I:%M %p').lstrip('0')}"
    except Exception as e:
        print(f"(sun lookup failed: {e})", flush=True)
        return None

# Confirmation phrases are casual/slang on purpose (per explicit request) --
# these are what get spoken AND printed for every fired command, so the
# persona lives here, not just in the knowledge-answer LLM prompt.
COMMANDS = {
    "CMD_PLAYPAUSE":     ("playpause",         "aight bro, hitting play",        ["play", "pause"]),
    "CMD_NEXT":          ("nexttrack",         "bet, skipping to the next one",  ["next", "skip"]),
    "CMD_PREVIOUS":      ("prevtrack",         "yeah bro, going back",           ["previous track", "go back", "skip back"]),
    # Bare "up"/"down" dropped as lone triggers -- real, demonstrated bug:
    # "how do i set up a linked list" contains "up" as a genuine standalone
    # word (not a substring artifact), so it fired CMD_VOLUP. Once the
    # knowledge fallback opened this system up to open-vocabulary technical
    # questions, common direction words stopped being safe lone triggers.
    "CMD_VOLUP":         ("volumeup",          "turning that shit up",           ["volume up", "turn up", "turn it up", "louder"]),
    "CMD_VOLDOWN":       ("volumedown",        "aight, bringing it down",        ["volume down", "turn down", "turn it down", "quieter"]),
    "CMD_MUTE":          ("volumemute",        "muted, bro",                     ["mute", "silence"]),
    "CMD_SCREENSHOT":    (do_screenshot,       "got it, screenshot's saved",     ["screenshot", "capture"]),
    "CMD_REPORT":        (run_self_analysis,   None,                             ["report", "status", "analyze"]),
    "CMD_SEARCH":        (do_search,           None,                             []),   # matched by prefix, not keyword -- see process_text
    "CMD_COMPOSE_EMAIL": (do_compose_email,    None,                             []),   # matched by prefix, not keyword -- see process_text
    "CMD_CODE_SEARCH":   (None,                 None,                            []),   # matched by prefix, not keyword -- see process_text
    "CMD_GIF":           (None,                 None,                            ["show me what's happening", "show me the screen", "record a gif", "send a gif", "make a gif"]),
    "CMD_YOUTUBE":       (None,                 None,                            []),   # matched by YOUTUBE_INTENT, not keyword -- see process_text
    "CMD_DATETIME":      (None,                 None,                            []),   # matched by DATE_TIME_INTENT, not keyword -- see process_text
    "CMD_HEALTH_CHECK":  (None,                 None,                            []),   # matched by HEALTH_CHECK_INTENT, not keyword -- see process_text
    "CMD_SUN":           (None,                 None,                            []),   # matched by SUN_INTENT, not keyword -- see process_text
    "CMD_CLAUDE_CODE":   (None,                 None,                            []),   # matched by prefix, admin-gated -- see process_text
    "CMD_FILE_SEARCH":   (None,                 None,                            []),   # matched by intent, admin-gated -- see process_text
    "CMD_SEND_FILE":     (None,                 None,                            []),   # matched by intent, admin-gated -- see process_text
    "CMD_EXPERIMENT":    (None,                 None,                            []),   # write+run python, admin-gated -- see process_text
    "CMD_SCREENSHOT_WEB": (None,                None,                            []),   # matched by intent, admin-gated -- see process_text
    "CMD_AGENT_TASK":    (None,                 None,                            []),   # matched by prefix, admin-gated -- see process_text
    "CMD_TRIBE_WORK":    (None,                 None,                            []),   # "work on tribe" -> next queued task, admin-gated
    "CMD_TRIBE_STATUS":  (None,                 None,                            []),   # "how's my tribe game" -> progress report
    "CMD_ASK_KNOWLEDGE": (None,                 None,                            []),   # terminal LLM fallback -- see process_text
}
NEURON_FOR_COMMAND = {
    "CMD_PLAYPAUSE": "PlayPause", "CMD_NEXT": "Next", "CMD_PREVIOUS": "Previous",
    "CMD_VOLUP": "VolUp", "CMD_VOLDOWN": "VolDown", "CMD_MUTE": "Mute",
    "CMD_SCREENSHOT": "Screenshot", "CMD_REPORT": "Report",
    "CMD_SEARCH": "Search", "CMD_COMPOSE_EMAIL": "ComposeEmail",
    "CMD_ASK_KNOWLEDGE": "Knowledge", "CMD_CODE_SEARCH": "CodeSearch",
    "CMD_GIF": "Gif", "CMD_YOUTUBE": "YouTube", "CMD_DATETIME": "DateTime", "CMD_HEALTH_CHECK": "HealthCheck",
    "CMD_SUN": "Sun", "CMD_CLAUDE_CODE": "ClaudeCode", "CMD_FILE_SEARCH": "FileSearch",
    "CMD_SEND_FILE": "SendFile", "CMD_EXPERIMENT": "Experiment",
    "CMD_SCREENSHOT_WEB": "ScreenshotWeb", "CMD_AGENT_TASK": "AgentTask",
    "CMD_TRIBE_WORK": "TribeWork", "CMD_TRIBE_STATUS": "TribeStatus",
}

# Real semantic code search -- reuses the ALREADY-BUILT code-minilm +
# FAISS index from the 012-ternary repo (observe_pipeline.py / trit_app.py's
# SearchEngine), the same infrastructure the observe MCP tools wrap. Not
# reinvented here.
#
# Run in a SEPARATE PROCESS (code_search_worker.py), never imported
# in-process -- confirmed by direct testing that pyttsx3.init() (Windows
# SAPI5/COM) and loading torch/FAISS into the same process segfaults
# (exit 139), reproduced with nothing else involved. Same reasoning as
# why the offline LLM classify/answer calls already shell out to
# llama-completion.exe instead of linking anything in-process.
CODE_SEARCH_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_search_worker.py")

def do_code_search():
    query = _pending_arg
    if not query:
        return None
    print("(running code search in a separate process -- ~8-10s first time)", flush=True)
    try:
        p = subprocess.run([sys.executable, CODE_SEARCH_WORKER, query],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=60)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception as e:
        print(f"(code search worker failed: {e})", flush=True)
        return None

# ── Filesystem-wide file search ("check my computer for X") ──────────────
# The MiniLM code search above only knows the indexed CODE corpus. This one
# finds ANY file across your real folders by name. Design decisions, all
# made by direct testing on this machine:
#   * NOT the Windows Search index -- it has real gaps here (missed
#     voice_commands.py / discord_bot.py entirely), so it'd give the exact
#     "can't find my own file" frustration this is meant to avoid.
#   * A direct os.walk instead -- always complete and up-to-the-second
#     fresh (found session-new files the index didn't). Full walk of ~42k
#     files ran in under 6s once the noise is skipped.
#   * Skip dependency/build/media-library dirs (node_modules, Splice
#     Samples, AppData, ...) and media file extensions -- your Splice audio
#     library alone is 24k+ .wav files that otherwise flood and starve the
#     time budget before real work files are reached.
#   * Work/project roots walked FIRST so they surface even if the budget
#     runs out; results ranked by how many query keywords hit the filename.
# Admin-gated: it lists what files exist on your machine, which over a
# remote chat channel is sensitive.
_HOME = os.path.expanduser("~")
FS_ROOTS = [p for p in (os.path.join(_HOME, d) for d in
            (r"OneDrive\Documents", "Downloads", "Desktop", r"OneDrive\Desktop", "Documents"))
            if os.path.isdir(p)]
FS_SKIP = {"node_modules", ".git", "__pycache__", "site-packages", "venv", ".venv", ".cache",
           "appdata", "$recycle.bin", "obj", "bin", ".next", "dist", "build", "splice",
           "samples", "packs", ".vscode", ".idea", "library", "program files", "windows"}
FS_MEDIA_EXT = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a", ".mp4", ".mov",
                ".avi", ".mkv", ".fbx", ".blend1", ".sample", ".rex", ".rx2"}
_FS_STOPWORDS = {"my", "the", "a", "an", "for", "on", "in", "of", "file", "files", "find",
                 "where", "is", "that", "this", "computer", "pc", "machine", "drive", "system",
                 "laptop", "stuff", "thing", "things", "some", "any", "all", "me", "please",
                 "spike", "hey", "check", "search", "look", "locate", "named", "called", "to"}

def _fs_keywords(q):
    words = re.findall(r"[A-Za-z0-9]+", q.lower())
    kws = [w for w in words if w not in _FS_STOPWORDS and len(w) > 1]
    return kws or [w for w in words if len(w) > 1][:3]

def _is_bot_artifact(fname):
    """Exclude the bot's own generated files from search -- especially
    voice_interaction_log.csv, which logs every command and so falsely
    'contains' whatever you search for."""
    fl = fname.lower()
    return (fl == "voice_interaction_log.csv"
            or fl.startswith(("screenshot_", "clip_", "admin_totp_qr")))

def do_file_search(query, limit=8, time_budget=20.0):
    """Returns a ranked list of full file paths matching the query across
    FS_ROOTS. Empty list if nothing matches or query has no keywords."""
    kws = _fs_keywords(query)
    if not kws:
        return []
    t0 = time.time()
    hits = []
    for root in FS_ROOTS:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.lower() not in FS_SKIP and not d.startswith(".")]
            for f in files:
                if os.path.splitext(f)[1].lower() in FS_MEDIA_EXT or _is_bot_artifact(f):
                    continue
                fl = f.lower()
                score = sum(1 for k in kws if k in fl)
                if score:
                    hits.append((score, os.path.join(dirpath, f)))
            if time.time() - t0 > time_budget:
                break
        else:
            continue
        break
    hits.sort(key=lambda x: (-x[0], len(x[1])))   # most keywords first, then shortest path
    return [p for _, p in hits[:limit]]

# Broad, natural triggers (per user request: "broad commands not specific").
# Each captures the search term. Order matters only for which group() we
# read; all are tried until one matches.
FILE_SEARCH_PATTERNS = [
    # "check a file on my computer for X", "check my files for X", "scan the
    # computer for X" -- the .*? after the noun absorbs an optional "on my
    # computer" so both "check a file on my computer for X" and "check my
    # computer for X" match. Real bug this broadens past: "check a file on
    # my computer for X" didn't match before (required "my/the computer").
    re.compile(r"(?:hey\s+spike[,\s]*)?(?:check|search|scan|look through|go through|look in|look at)\s+"
               r"(?:my|the|a|some|any|all(?:\s+my)?)?\s*"
               r"(?:files?|folders?|documents?|docs?|computer|pc|machine|drive|system|laptop).*?\bfor\s+(.+)", re.I),
    re.compile(r"(?:hey\s+spike[,\s]*)?(?:find|locate|look for|search for)\s+(.+?)\s+"
               r"on\s+(?:my|the)\s+(?:computer|pc|machine|drive|system|laptop|files?)", re.I),
    re.compile(r"where(?:'?s| is)\s+(.+?)\s+on\s+(?:my|the)\s+(?:computer|pc|machine|drive|system|laptop)", re.I),
    # "which file has/contains/mentions X", "what file is about X"
    re.compile(r"(?:which|what)\s+files?\s+(?:has|have|contains?|mentions?|says?|is about|are about|talk(?:s)? about)\s+(.+)", re.I),
    # "find the file that contains/about/with X"
    re.compile(r"(?:find|locate|pull up|open)\s+(?:the|a|my)\s+files?\s+"
               r"(?:that\s+)?(?:contains?|has|mentions?|about|with|says?|is about)\s+(.+)", re.I),
    re.compile(r"(?:find|locate|pull up|open)\s+(?:my|the)\s+(.+?)(?:\s+files?)?$", re.I),
    re.compile(r"where(?:'?s| is)\s+(?:my|the)\s+(.+)", re.I),
]

def extract_file_query(text):
    for pat in FILE_SEARCH_PATTERNS:
        m = pat.match(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None

# "send/show/screenshot me the file about X" -- the user's real goal when
# they said "screenshot the file and send it": find the file whose CONTENT
# is about X and hand them the actual file. Distinct from CMD_SCREENSHOT
# (which grabs the whole screen) and must be checked BEFORE the screenshot
# keyword so "screenshot the file about X" attaches the file instead of the
# desktop. The (?:...about/with/containing...)? makes the connector
# optional: "send me the file about neurons" and "send me the neurons file"
# both work.
SEND_FILE_PATTERNS = [
    # "send me the file about X" / "screenshot the file that contains X"
    # (the word "file" comes first, then a connector, then the topic)
    re.compile(r"(?:hey\s+spike[,\s]*)?(?:screenshot|send|show|get|give|pull up|open|grab)\s+"
               r"(?:me\s+)?(?:the|that|this|a|my)\s+files?\s+"
               r"(?:that\s+)?(?:mentioning|mentions?|containing|contains?|about|with|has|have|"
               r"says?|is about|on|of|for|named|called)?\s*(.+)", re.I),
    # "send me the X file" -- descriptor before the word "file"
    re.compile(r"(?:hey\s+spike[,\s]*)?(?:screenshot|send|show|get|give|pull up|open|grab)\s+"
               r"(?:me\s+)?(?:the|that|this|a|my)\s+(.+?)\s+files?\b", re.I),
]

def extract_send_file_query(text):
    for pat in SEND_FILE_PATTERNS:
        m = pat.match(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None

# Content search -- finds files by what's INSIDE them (text/code/PDF/Word/
# Excel), not just their names. Runs in a separate process
# (content_search_worker.py) for the same reason code search does: keeps
# heavy/native extractor libs out of the pyttsx3/COM process, and lets a
# pathological file get killed by the subprocess timeout. This is the
# honest fix for "find the doc that mentions taxes" -- NOT the OBSERVE
# semantic index, which was stale, code-only, and returned confident wrong
# files with scores that didn't separate relevant from irrelevant.
CONTENT_SEARCH_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content_search_worker.py")

def do_content_search(query, limit=8):
    try:
        p = subprocess.run([sys.executable, CONTENT_SEARCH_WORKER, query, str(limit)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=40)
        lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else []
    except Exception as e:
        print(f"(content search failed: {e})", flush=True)
        return []

# Prefixes checked BEFORE the generic keyword loop -- these need to extract
# free text after the trigger phrase (the search query / email body), which
# a simple "is this word present anywhere" keyword check can't do.
SEARCH_PREFIXES = ["search for ", "look up ", "google "]
EMAIL_PREFIXES = ["compose email ", "compose an email ", "write an email ", "new email "]
# Checked BEFORE the plain web-search prefixes below -- "search my code for"
# never collides with "search for " at the startswith level (there's always
# "my code"/"the code"/"codebase" in between), so order doesn't matter for
# correctness, but these are listed first since they're the more specific ask.
CODE_SEARCH_PREFIXES = ["search my code for ", "search the code for ", "search my codebase for ",
                         "search codebase for ", "find in my code ", "find in the code ", "find in code "]
# Agentic Claude Code trigger (admin-gated). Deliberately unambiguous
# prefixes so a casual message that merely mentions Claude doesn't kick off
# a code task. The user starts an agentic task with "claude code <task>".
# Plain prefix AUTO-CONTINUES the last Claude conversation in the same
# project (do_claude_code's resume=True) -- a genuine follow-up remembers
# prior turns, no need to restate context every message. The "new" variant
# forces a fresh session instead (checked first below, since it's the more
# specific match -- both share the same "claude code "/"code task " lead-in).
CLAUDE_CODE_PREFIXES = ["claude code ", "code task "]
CLAUDE_CODE_NEW_PREFIXES = ["claude code new: ", "claude code new ", "code task new: ", "code task new "]
# Modular agent-task trigger (admin-gated, distinct from raw Claude Code --
# this one runs the clarify-check first, see do_agent_task()). Deliberately
# unambiguous prefixes for the same reason as CLAUDE_CODE_PREFIXES above.
AGENT_TASK_PREFIXES = ["work on: ", "work on ", "tackle: ", "tackle ",
                        "improve the project: ", "improve the project ",
                        "build feature: ", "build feature ", "agent task: ", "agent task "]

# Tribe control (Discord-first). WORK = pop next queued tribe task and run it
# (bare "work on tribe" with NO ":<task>" after it -- a specific
# "work on tribe: X" still goes to CMD_AGENT_TASK). STATUS = report progress.
TRIBE_WORK_INTENT = re.compile(
    r"^(?:yo,?\s*|hey,?\s*)?(?:let'?s\s+|lets\s+|keep\s+|continue\s+|please\s+)?"
    r"work(?:ing)?\s+on\s+tribe\s*$"
    r"|^(?:keep\s+going\s+on\s+|continue\s+)?tribe\s*$"
    r"|^next\s+tribe\s+task\s*$", re.IGNORECASE)
TRIBE_STATUS_INTENT = re.compile(
    r"how(?:'?s|\s+is|s)?\s+(?:my\s+|the\s+|it\s+going\s+with\s+)?tribe(?:\s+game)?"
    r"|tribe\s+status"
    r"|hows?\s+(?:my\s+|the\s+)?tribe", re.IGNORECASE)

def make_search_handler():
    def handler():
        query = do_search()
        say(f"on it bro, searching for {query}" if query else "didn't catch what to search for.")
    return handler
rt.register_handler("CMD_SEARCH", make_search_handler())

def make_email_handler():
    def handler():
        do_compose_email()
        say("bet, drafted that email bro -- nothing's sent yet, you gotta hit send yourself.")
    return handler
rt.register_handler("CMD_COMPOSE_EMAIL", make_email_handler())

def make_code_search_handler():
    def handler():
        result = do_code_search()
        if result:
            preview = " ".join(result["preview"].split())
            say(f"found it bro -- {result['path']}: {preview}")
        else:
            say("couldn't find anything for that in the code index, bro.")
    return handler
rt.register_handler("CMD_CODE_SEARCH", make_code_search_handler())

def make_gif_handler():
    def handler():
        say("bet, recording a few seconds bro, hang tight")
        do_record_gif()
        say("there you go bro, that's what's happening")
    return handler
rt.register_handler("CMD_GIF", make_gif_handler())

def make_youtube_handler():
    def handler():
        query = do_youtube_search()
        say(f"bet, pulling up {query} on youtube" if query else "didn't catch what to put on.")
    return handler
rt.register_handler("CMD_YOUTUBE", make_youtube_handler())

def make_datetime_handler():
    def handler():
        say(f"it's {do_current_datetime()}, bro")
    return handler
rt.register_handler("CMD_DATETIME", make_datetime_handler())

def make_health_check_handler():
    def handler():
        say(do_health_check())
    return handler
rt.register_handler("CMD_HEALTH_CHECK", make_health_check_handler())

def make_sun_handler():
    def handler():
        kind, location = _pending_arg
        result = do_sun_lookup(kind, location)
        say(f"{result}, bro" if result else f"couldn't find {location}, bro -- try being more specific, like adding the state.")
    return handler
rt.register_handler("CMD_SUN", make_sun_handler())

def make_claude_code_handler():
    def handler():
        # Only reached AFTER the admin gate passed in process_text (this is
        # a SENSITIVE command). Runs the agentic edit task, then reports.
        if _pending_claude_new_session:
            _last_claude_session["id"] = None
            _last_claude_session["project_dir"] = None
            say("aight bro, starting a FRESH Claude Code session -- hang tight, this can take a minute.")
        else:
            say("aight bro, putting Claude Code on it -- hang tight, this can take a minute.")
        summary = do_claude_code(resume=True)
        if summary:
            say(f"done. {summary} go review the changes before you keep 'em, bro.")
        else:
            say("that didn't go through, bro -- Claude Code either isn't set up or the task failed.")
    return handler
rt.register_handler("CMD_CLAUDE_CODE", make_claude_code_handler())

def make_experiment_handler():
    def handler():
        task = _pending_arg
        lang = _pending_experiment_lang
        say(f"aight bro, writing and running a {lang} {'component' if lang == 'react' else 'script'} for '{task}' -- gimme a minute...")
        result = do_experiment(task, lang)
        say(result if result else "couldn't run that experiment, bro -- Claude Code isn't set up or it failed.")
    return handler
rt.register_handler("CMD_EXPERIMENT", make_experiment_handler())

def make_screenshot_web_handler():
    def handler():
        url = _pending_arg
        say(f"bet, screenshotting {url} bro, one sec...")
        result = do_screenshot_website(url)
        say(result if result else "couldn't get that screenshot, bro.")
    return handler
rt.register_handler("CMD_SCREENSHOT_WEB", make_screenshot_web_handler())

def make_agent_task_handler():
    def handler():
        task = _pending_arg
        say(f"aight bro, looking at '{task}' -- checking if it's clear enough to just start, gimme a sec...")
        result = do_agent_task(task)
        say(result if result else "couldn't run that agent task, bro -- Claude Code isn't set up or it failed.")
    return handler
rt.register_handler("CMD_AGENT_TASK", make_agent_task_handler())

def make_tribe_work_handler():
    def handler():
        say("aight, pulling the next thing off the tribe queue and running it through the full pipeline -- this takes a few minutes, bro...")
        result = do_tribe_work()
        say(result if result else "couldn't run the next tribe task, bro.")
    return handler
rt.register_handler("CMD_TRIBE_WORK", make_tribe_work_handler())

def make_tribe_status_handler():
    def handler():
        say(do_tribe_status())
    return handler
rt.register_handler("CMD_TRIBE_STATUS", make_tribe_status_handler())

def make_file_search_handler():
    def handler():
        # _pending_arg holds the extracted search term. Admin-gated.
        # Runs BOTH: filename match (any file, fast) AND content match
        # (reads inside text/code/PDF/Word/Excel). Content hits are shown
        # first -- they're the "what's inside" wins the user asked for --
        # then name-only hits, deduped.
        query = _pending_arg
        say(f"aight bro, searching your files for '{query}' -- by name and by content, gimme a sec...")
        home = os.path.normpath(_HOME)
        def short(p):
            p = os.path.normpath(p)
            return os.path.basename(p) + "  (" + os.path.dirname(p).replace(home, "~") + ")"

        by_name = [os.path.normpath(p) for p in do_file_search(query)]
        by_content = [os.path.normpath(h["path"]) for h in do_content_search(query)]

        seen, merged = set(), []
        for p in by_content:
            if p not in seen:
                seen.add(p); merged.append(("inside", p))
        for p in by_name:
            if p not in seen:
                seen.add(p); merged.append(("name", p))

        if merged:
            lines = [f"{short(p)} [{how}]" for how, p in merged[:8]]
            say(f"found {len(merged)}, bro: " + "  |  ".join(lines))
        else:
            say(f"nothing matching '{query}' in your files -- by name or content, bro. try different words?")
    return handler
rt.register_handler("CMD_FILE_SEARCH", make_file_search_handler())

# Discord attachment cap for non-Nitro is ~8 MB -- above this the send
# fails, so we report the path instead of attaching.
SEND_FILE_MAX_BYTES = 8_000_000

def make_send_file_handler():
    def handler():
        global _last_attachment_path
        query = _pending_arg
        say(f"aight bro, finding the file about '{query}' to send you...")
        # Prefer a content match (the file that's ABOUT this), fall back to
        # a filename match.
        content = do_content_search(query, limit=1)
        best = content[0]["path"] if content else None
        if not best:
            names = do_file_search(query, limit=1)
            best = names[0] if names else None
        if not best or not os.path.exists(best):
            say(f"couldn't find a file about '{query}' to send, bro.")
            return
        try:
            size = os.path.getsize(best)
        except OSError:
            size = 0
        name = os.path.basename(best)
        if size > SEND_FILE_MAX_BYTES:
            say(f"found it -- {name} ({os.path.dirname(best).replace(os.path.normpath(_HOME), '~')}) -- "
                f"but it's too big to send over Discord ({size // 1024 // 1024} MB), bro. that's the path.")
            return
        _last_attachment_path = best   # discord_bot.py uploads this alongside the reply
        say(f"here you go bro -- sending {name}")
    return handler
rt.register_handler("CMD_SEND_FILE", make_send_file_handler())

# Knowledge answers are computed BEFORE the neuron fires (llm_answer() takes
# 15-30s, too slow to run inside the handler itself), then handed off through
# _pending_arg same as search/email -- the handler just speaks what's already
# been computed.
def make_knowledge_handler():
    def handler():
        answer = _pending_arg
        say(answer if answer else "nah, don't got an answer for that one, bro.")
    return handler
rt.register_handler("CMD_ASK_KNOWLEDGE", make_knowledge_handler())

for command, (action, phrase, _) in COMMANDS.items():
    if command in ("CMD_SEARCH", "CMD_COMPOSE_EMAIL", "CMD_ASK_KNOWLEDGE", "CMD_CODE_SEARCH", "CMD_GIF", "CMD_YOUTUBE", "CMD_DATETIME", "CMD_HEALTH_CHECK", "CMD_SUN", "CMD_CLAUDE_CODE", "CMD_FILE_SEARCH", "CMD_SEND_FILE", "CMD_EXPERIMENT", "CMD_SCREENSHOT_WEB", "CMD_AGENT_TASK", "CMD_TRIBE_WORK", "CMD_TRIBE_STATUS"):
        continue   # already registered above with their own handlers
    def make_handler(action=action, phrase=phrase):
        def handler():
            if callable(action):
                result = action()
                if phrase:
                    say(phrase)
                elif isinstance(result, str) and os.path.exists(result):
                    pass  # do_screenshot already handled, no separate phrase needed
            else:
                pyautogui.press(action)
                say(phrase)
        return handler
    rt.register_handler(command, make_handler())

# Microphone setup is LAZY -- only touched the first time the spoken path
# is actually used. Typed-only use (e.g. no working mic, or pyaudio isn't
# installed in this particular Python environment -- a real, common
# Windows pain point since pyaudio needs the PortAudio C library) should
# never be blocked by microphone setup that isn't needed yet.
recognizer = None
mic = None

def ensure_mic_ready():
    global recognizer, mic
    if recognizer is not None:
        return
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Calibrating for background noise...", flush=True)
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

# Hard safety boundary -- checked FIRST, before prefix/keyword/LLM matching,
# so no phrasing can route around it. This system has zero delete/remove
# capability today (no handler touches the filesystem destructively), but
# now that it's getting a remote surface (Discord), "just don't implement
# it" isn't a strong enough guarantee -- this refuses the INTENT outright,
# so even a future command or a coaxed LLM classification can never turn
# into an actual delete. Deliberately broad (a false-positive refusal on a
# harmless "how does Python's del keyword work" question is an acceptable
# cost; a missed real delete request is not).
DESTRUCTIVE_INTENT = re.compile(
    r"\b(delete|remove|erase|wipe|uninstall|format|destroy|"
    r"factory reset|empty the (recycle bin|trash))\b", re.IGNORECASE)

# Same hard-boundary reasoning as DESTRUCTIVE_INTENT, extended to money and
# credentials -- this system has zero purchasing/payment/credential-entry
# capability today, and the refusal is at the INTENT level so it stays that
# way even as commands get added later or the LLM fallback gets involved.
FINANCIAL_INTENT = re.compile(
    r"\b(buy|purchase|order|check\s?out|pay for|send money|transfer money|"
    r"wire money|wire transfer|venmo|paypal|cash\s?app|zelle)\b", re.IGNORECASE)
# Informational shopping/price questions are NOT a transaction -- "where can
# I buy X", "how much is Y", "best price for Z". These are just research
# (the bot answers via web search); refusing them was a false positive.
# The refusal only fires for FINANCIAL_INTENT that is NOT financial research.
FINANCIAL_RESEARCH = re.compile(
    r"\bwhere (can|do|should|could|to)\b|\bhow much\b|\bbest (price|deal)\b|"
    r"\bcheapest\b|\bprice(s)? (of|for|on)\b|\bcost(s)? (of|for)\b|"
    r"\bcompare prices\b|\bwhat does .* cost\b|\bhow (do|can) i find\b|"
    r"\bwhere.*\b(buy|get|find|purchase)\b", re.IGNORECASE)
CREDENTIAL_INTENT = re.compile(
    r"\b(my password is|enter my password|log\s?in with (my )?password|"
    r"type my password|save my password|here'?s my password)\b", re.IGNORECASE)

# Admin gate for SENSITIVE commands -- separate from the hard refusals
# above (those are "never, no matter what"; this is "yes, but prove it's
# really you first"). Sensitive = touches real data: screen contents, code
# paths, real usage history, drafted messages, agentic editing. Media
# controls stay ungated.
#
# UNLOCK IS TOTP, NOT A FIXED PASSPHRASE. A reusable passphrase sent over
# Discord leaks into chat history permanently (this actually happened). A
# TOTP code is single-use and expires in ~30s, so even if it lingers in
# chat forever it's worthless. The secret (seed) lives in
# VOICE_ADMIN_TOTP_SECRET, set locally by set_admin_totp.py -- never sent
# anywhere. Codes are verified with pyotp.
SENSITIVE_COMMANDS = {"CMD_SCREENSHOT", "CMD_GIF", "CMD_REPORT", "CMD_COMPOSE_EMAIL", "CMD_CODE_SEARCH",
                      "CMD_CLAUDE_CODE",   # agentic editing -- always behind the admin gate
                      "CMD_FILE_SEARCH",   # lists what files exist on your machine
                      "CMD_SEND_FILE",     # sends an actual file's contents off the machine
                      "CMD_EXPERIMENT",    # writes AND executes Python -- real code execution
                      "CMD_SCREENSHOT_WEB",   # fetches and renders an arbitrary external URL
                      "CMD_AGENT_TASK",    # agentic editing on this project -- same tier as CMD_CLAUDE_CODE
                      "CMD_TRIBE_WORK"}    # runs an agentic edit on the tribe game -- same tier
                                            # (CMD_TRIBE_STATUS is read-only, intentionally NOT gated)
ADMIN_TOTP_SECRET = os.environ.get("VOICE_ADMIN_TOTP_SECRET", "")

def check_admin_code(code):
    """Verify a 6-digit TOTP code against the configured secret. False if
    TOTP isn't set up or the code is wrong/expired. valid_window=1 tolerates
    ~30s of clock skew (accepts the adjacent time step either side)."""
    if not (ADMIN_TOTP_SECRET and code):
        return False
    try:
        import pyotp
        return pyotp.TOTP(ADMIN_TOTP_SECRET).verify(code.strip(), valid_window=1)
    except Exception:
        return False

def request_admin_code_console():
    """Local (typed/spoken) path: prompt for the current TOTP code. Discord
    has no console -- discord_bot.py registers its own per-session unlock
    check (via set_admin_auth_check) against the SAME check_admin_code()."""
    if not ADMIN_TOTP_SECRET:
        say("that's a sensitive command, bro, but the admin code isn't set up yet -- run set_admin_totp.py first.")
        return False
    entered = input("Admin 6-digit code (from your authenticator app): ").strip()
    if check_admin_code(entered):
        return True
    say("wrong or expired code, bro -- not doing that.")
    return False

# Pluggable auth check, same pattern as the response sink -- local
# typed/spoken use defaults to the console TOTP prompt; discord_bot.py
# overrides this with its own per-session unlock state, since there's no
# console to prompt in a chat interface.
_admin_auth_check = request_admin_code_console

def set_admin_auth_check(callback):
    global _admin_auth_check
    _admin_auth_check = callback

def process_text(text, latency, input_method, raw_text=None):
    """Shared matching path for both spoken and typed input -- prefix
    match (search/email, which need a free-text argument) first, then
    keyword match, then offline LLM fallback, then log + fire the neuron.

    `text` is lowercased for matching; `raw_text` is the ORIGINAL-case
    input, used when extracting a free-text argument (search query, email
    body, and especially a Claude Code task, where identifier/file-name
    casing must survive). Lowercasing is a per-character transform, so
    offsets line up between text and raw_text."""
    global _pending_arg, _suppress_tts, _pending_experiment_lang, _pending_claude_new_session
    raw_text = text if raw_text is None else raw_text
    # Strip wrapping quote chars -- real bug hit live: a user pasted a
    # suggested phrase including its surrounding quotes (`"write a react
    # component..."`), and every prefix/intent regex is anchored at the
    # start of the string, so a leading quote silently broke every match
    # and fell all the way through to the generic knowledge fallback.
    _QUOTE_CHARS = "\"'“”‘’"
    text = text.strip().strip(_QUOTE_CHARS).strip()
    raw_text = raw_text.strip().strip(_QUOTE_CHARS).strip()
    now_ms = (time.time() - t0) * 1000
    matched_command, matched_via = None, "none"
    _suppress_tts = (input_method in ("discord", "webui"))

    if DESTRUCTIVE_INTENT.search(text):
        say("nah bro, i don't do deletions or removals -- that's off the table, full stop. do that one yourself.")
        log_interaction(text, len(text.split()), latency, "REFUSED_DESTRUCTIVE", "safety_guard", input_method, t0)
        return
    if FINANCIAL_INTENT.search(text) and not FINANCIAL_RESEARCH.search(text):
        say("nope, not gonna actually buy anything or move money -- that's on you, bro. but i can look up where to get it or compare prices if you ask that way.")
        log_interaction(text, len(text.split()), latency, "REFUSED_FINANCIAL", "safety_guard", input_method, t0)
        return
    if CREDENTIAL_INTENT.search(text):
        say("i'm not the place for passwords, bro -- never gonna ask for or handle one. enter that yourself.")
        log_interaction(text, len(text.split()), latency, "REFUSED_CREDENTIAL", "safety_guard", input_method, t0)
        return

    for prefix in SEARCH_PREFIXES:
        if text.startswith(prefix):
            _pending_arg = raw_text[len(prefix):].strip()
            matched_command, matched_via = "CMD_SEARCH", "prefix"
            break
    if not matched_command:
        for prefix in EMAIL_PREFIXES:
            if text.startswith(prefix):
                _pending_arg = raw_text[len(prefix):].strip()
                matched_command, matched_via = "CMD_COMPOSE_EMAIL", "prefix"
                break
    if not matched_command:
        for prefix in CODE_SEARCH_PREFIXES:
            if text.startswith(prefix):
                _pending_arg = raw_text[len(prefix):].strip()
                matched_command, matched_via = "CMD_CODE_SEARCH", "prefix"
                break
    if not matched_command:
        youtube_query = extract_youtube_query(text)
        if youtube_query:
            _pending_arg = youtube_query
            matched_command, matched_via = "CMD_YOUTUBE", "intent"
    if not matched_command and DATE_TIME_INTENT.search(text):
        matched_command, matched_via = "CMD_DATETIME", "intent"
    if not matched_command and HEALTH_CHECK_INTENT.search(text):
        matched_command, matched_via = "CMD_HEALTH_CHECK", "intent"
    if not matched_command:
        sun_query = extract_sun_query(text)
        if sun_query:
            _pending_arg = sun_query
            matched_command, matched_via = "CMD_SUN", "intent"
    if not matched_command:
        for prefix in CLAUDE_CODE_NEW_PREFIXES:   # checked first -- more specific than the plain prefixes below
            if text.startswith(prefix):
                _pending_arg = raw_text[len(prefix):].strip()   # original case -- code identifiers/paths must survive
                _pending_claude_new_session = True
                matched_command, matched_via = "CMD_CLAUDE_CODE", "prefix"
                break
    if not matched_command:
        for prefix in CLAUDE_CODE_PREFIXES:
            if text.startswith(prefix):
                _pending_arg = raw_text[len(prefix):].strip()   # original case -- code identifiers/paths must survive
                _pending_claude_new_session = False
                matched_command, matched_via = "CMD_CLAUDE_CODE", "prefix"
                break
    # Tribe control BEFORE the generic agent-task prefix -- bare "work on
    # tribe" (no ":<task>") means "pop the next queued task", not "run a task
    # literally named tribe". A specific "work on tribe: X" has a colon and
    # falls through to CMD_AGENT_TASK below.
    if not matched_command and TRIBE_STATUS_INTENT.search(text):
        matched_command, matched_via = "CMD_TRIBE_STATUS", "intent"
    if not matched_command and TRIBE_WORK_INTENT.search(text):
        matched_command, matched_via = "CMD_TRIBE_WORK", "intent"
    if not matched_command:
        for prefix in AGENT_TASK_PREFIXES:
            if text.startswith(prefix):
                _pending_arg = raw_text[len(prefix):].strip()   # original case -- code identifiers/paths must survive
                matched_command, matched_via = "CMD_AGENT_TASK", "prefix"
                break
    if not matched_command:
        exp_task = extract_experiment_task(raw_text)   # original case -- code needs it
        if exp_task:
            _pending_arg = exp_task
            # Detect language from the FULL sentence, not just the extracted
            # task -- "write a react component for X" strips "react" out of
            # the task text itself (it's part of the noun phrase, before the
            # captured clause), so detecting on the extracted text alone
            # always missed it and silently defaulted to python.
            _pending_experiment_lang = _detect_experiment_lang(raw_text)
            matched_command, matched_via = "CMD_EXPERIMENT", "intent"
    # Website screenshot checked BEFORE the send-file/CMD_SCREENSHOT paths --
    # it also contains the word "screenshot" but is more specific (requires
    # an actual URL), so checking it first keeps it from ever being
    # swallowed by the desktop-screenshot keyword loop.
    if not matched_command:
        web_url = extract_screenshot_website_url(raw_text)
        if web_url:
            _pending_arg = web_url
            matched_command, matched_via = "CMD_SCREENSHOT_WEB", "intent"
    # SEND-file checked BEFORE file-search AND before the keyword loop, so
    # "screenshot/send me the file about X" attaches the file instead of
    # firing CMD_SCREENSHOT (screen grab) or just listing matches.
    if not matched_command:
        sfq = extract_send_file_query(text)
        if sfq:
            _pending_arg = sfq
            matched_command, matched_via = "CMD_SEND_FILE", "intent"
    if not matched_command:
        fq = extract_file_query(text)   # lowercased is fine -- file matching is case-insensitive
        if fq:
            _pending_arg = fq
            matched_command, matched_via = "CMD_FILE_SEARCH", "intent"

    # Question-shaped input skips keyword matching entirely. Confirmed by
    # direct testing (not a guess): "explain how a lab report should be
    # structured" fired CMD_REPORT (ran a self-analysis), "how do plants
    # capture sunlight" fired CMD_SCREENSHOT (took a real screenshot),
    # "what is an http status code" fired CMD_REPORT again, "what is the
    # next fibonacci number" fired CMD_NEXT, "what does silence symbolize"
    # fired CMD_MUTE. Patching individual trigger words doesn't scale --
    # any common English word used as a bare command trigger will
    # eventually collide with some natural question. A question almost
    # never IS a bare device command, so routing all question-shaped
    # input straight to LLM-classify + knowledge fallback (skipping
    # keyword matching) fixes the whole class at once.
    # (?:'s|s)? catches apostrophe-dropped contractions ("whats", "hows",
    # "wheres") -- real gap found via live testing: "whats next" fell
    # straight through to the keyword loop (bare "what" doesn't match
    # inside the fused word "whats", no boundary before the trailing s)
    # and fired CMD_NEXT as a literal media-skip command instead of being
    # treated as a question. Casual typing drops apostrophes constantly,
    # especially over Discord/mobile.
    is_question = bool(re.match(r"^(?:what|why|who|when|where|which|how|explain|define|describe)(?:'s|s)?\b", text)
                        or text.rstrip().endswith("?"))

    # CLAUDE IN CONTROL of intent routing. Real bug this fixes: the keyword
    # matcher fired CMD_PLAYPAUSE ("hitting play") on a stray word in a
    # tribe/chat message. Brittle literal matching can't tell a media command
    # from a sentence that merely contains "play"/"next"/"up"/"loop". So Claude
    # -- which understands intent -- decides FIRST. If Claude returns a command,
    # use it. If Claude ran and said NONE, it is genuinely not a command, so we
    # deliberately SKIP the keyword device-matching and let it fall through to
    # the knowledge/chat path. The old keyword + offline-classify layers now
    # run ONLY as a degraded fallback when Claude is unreachable (offline).
    claude_router_ran = False
    if not matched_command and not is_question:
        if CLAUDE_CLI:
            routed_cmd, routed_arg = claude_route_command(raw_text)
            claude_router_ran = True
            if routed_cmd in COMMANDS:
                if routed_arg:
                    _pending_arg = routed_arg
                    if routed_cmd == "CMD_EXPERIMENT":
                        _pending_experiment_lang = _detect_experiment_lang(routed_arg)
                matched_command, matched_via = routed_cmd, "claude_router"

    if not matched_command and not is_question and not claude_router_ran:
        # Degraded path -- Claude unavailable. Fall back to the old literal
        # keyword loop + offline classifier (brittle, but better than nothing).
        print("(claude router unavailable -- falling back to keyword/offline classify)", flush=True)
        for command, (action, phrase, trigger_words) in COMMANDS.items():
            if trigger_words and any(re.search(rf"\b{re.escape(w)}\b", text) for w in trigger_words):
                matched_command, matched_via = command, "keyword"
                break
        if not matched_command:
            llm_command = llm_classify(text)
            if llm_command in COMMANDS:
                matched_command, matched_via = llm_command, "llm"

    if not matched_command:
        # SPLIT routing (per user request): real-time questions go to Claude
        # (which has web search and this machine's existing login -- no API
        # key), simple/static questions stay on the fast local model. Each
        # side still falls through to the other backends if its preferred
        # one returns None, so nothing dead-ends.
        def ask_claude():
            if CLAUDE_CLI:
                r = claude_cli_answer(text)
                if r: return r, "claude_cli"
            if os.environ.get("ANTHROPIC_API_KEY"):
                r = claude_answer(text)
                if r: return r, "claude_api"
            return None, None
        def ask_local():
            r = llm_answer(text)
            return (r, "llm_knowledge") if r else (None, None)

        answer, via = None, None
        # Real bug found via live testing: "did we ever run X before" isn't
        # a real-time question by needs_web()'s definition, so it routed to
        # the small local model -- which has zero vault access -- and it
        # confidently hallucinated a fake "old project" instead of saying
        # it didn't know. Only claude_cli_answer() is wired to vault
        # retrieval, so a relevant vault hit forces that path regardless of
        # the real-time check, same as needs_web() would.
        has_vault_context = bool(search_vault(text))
        if needs_web(text) or has_vault_context:
            print("(needs live web/fetch or vault has relevant history -- asking Claude, with web tools)", flush=True)
            answer, via = ask_claude()
            if answer is None:
                answer, via = ask_local()   # offline fallback if Claude is unreachable
        else:
            print("(simple question -- using the fast local model)", flush=True)
            answer, via = ask_local()
            if answer is None:
                answer, via = ask_claude()  # fall up to Claude if the local model whiffs
        if answer:
            _pending_arg = answer
            matched_command, matched_via = "CMD_ASK_KNOWLEDGE", via

    if matched_command in SENSITIVE_COMMANDS and not _admin_auth_check():
        print(f"(matched {matched_command} but admin auth failed/declined -- refusing)", flush=True)
        log_interaction(text, len(text.split()), latency, matched_command, matched_via + "_locked", input_method, t0)
        return

    if matched_command:
        rt.stimulate(NEURON_FOR_COMMAND[matched_command], now_ms, drive=100.0)
        print(f"(matched {matched_command} via {matched_via})", flush=True)
    else:
        print("(no matching command and no knowledge answer)", flush=True)
    log_interaction(text, len(text.split()), latency, matched_command, matched_via, input_method, t0)

# t0 is needed by process_text() (elapsed-time math) regardless of which
# front-end calls it, so it's set unconditionally -- only the interactive
# REPL loop below is guarded, so importing this module (e.g. from
# discord_bot.py) sets everything up WITHOUT hanging on input() forever.
# Real bug this fixes: before the guard existed, `import voice_commands`
# would block at the bottom of this file waiting for console input that
# a Discord bot's process never provides.
t0 = time.time()

if __name__ == "__main__":
    print("Ready. play/pause, next/skip, previous/back, up/louder, down/quieter,", flush=True)
    print("mute/silence, screenshot/capture, report/status/analyze,", flush=True)
    print("'search for <query>', 'compose email <message>',", flush=True)
    print("'search my code for <query>' (real MiniLM+FAISS semantic search),", flush=True)
    print("or ask a software/coding/math/science/literature question directly.", flush=True)
    print("Press Enter alone to speak, or type a phrase directly and press Enter.", flush=True)

    while True:
        typed = input("\n> ").strip()

        if typed:
            # TYPED PATH -- skip the mic/speech-recognition entirely
            text = typed.lower()
            print(f"Typed: '{text}'", flush=True)
            process_text(text, latency=0.0, input_method="typed", raw_text=typed)
            continue

        # SPOKEN PATH -- mic is only initialized here, on first real use
        ensure_mic_ready()
        print("Listening...", flush=True)
        with mic as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                print("(no speech detected)", flush=True)
                continue

        listen_start = time.time()
        try:
            text = recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            print("(couldn't understand)", flush=True)
            log_interaction("", 0, time.time() - listen_start, None, "none", "spoken", t0)
            continue
        except sr.RequestError as e:
            print(f"(speech service error: {e})", flush=True)
            continue
        latency = time.time() - listen_start

        print(f"Heard: '{text}'", flush=True)
        process_text(text, latency, input_method="spoken")
