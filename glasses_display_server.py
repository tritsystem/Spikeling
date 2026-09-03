#!/usr/bin/env python
"""
glasses_display_server.py -- the phone-browser display for the wireless AR
glasses project. Replaces the ESP32C3 OLED half of the original design
(vault/Project Work/20260729_093448_wireless-ar-glasses-voice-assistant-
full-design-parts-list-a.md), which was never built -- no ESP32C3 was ever
bought, no OLED firmware was ever written. This is the "use the phone's own
browser as the display" version instead, same architecture principle as
room-scanner (phone is a peripheral, the PC does everything real).

Speaks the EXACT protocol wireless_glasses.py's send_to_glasses_display()
and laptop-session/glasses_hook.py's Stop hook already POST to:
  POST /display   body: raw text, Content-Type: text/plain
Both of those already exist and already send to whatever CLAUDE_GLASSES_IP
points at -- this server just needs to be that IP:port. Nothing in either
caller needs to change.

Run: python glasses_display_server.py
Then, on this PC: `tailscale serve --bg 5759` to get a trusted HTTPS URL
reachable from the phone over the tailnet (same command room-scanner's
README already documents for the same purpose).
Then set CLAUDE_GLASSES_IP to this machine's tailnet IP:5759 (or the
tailscale serve hostname) so send_to_glasses_display() and glasses_hook.py
start actually reaching something real.

SECURITY: binds 0.0.0.0 so `tailscale serve` can reach it, same choice
room-scanner's app.py already made -- do not port-forward this to the
public internet. No auth: anyone who can reach this port can post text
that appears on your phone. Fine on a private tailnet, not fine exposed
further.
"""
import os
import threading
import time

from flask import Flask, request, Response, jsonify

PORT = int(os.environ.get("GLASSES_DISPLAY_PORT", "5759"))

_lock = threading.Lock()
_latest = {"text": "(nothing displayed yet)", "ts": 0.0}

app = Flask(__name__)

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Glasses Display</title>
<style>
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: #000; color: #39ff14;
    font-family: -apple-system, "SF Mono", Menlo, monospace;
  }
  #wrap {
    box-sizing: border-box; min-height: 100vh; padding: 24px;
    display: flex; flex-direction: column; justify-content: center;
  }
  #text {
    font-size: 6vw; line-height: 1.35; white-space: pre-wrap; word-break: break-word;
  }
  #age {
    position: fixed; top: 8px; right: 12px; font-size: 12px; color: #1f7a0a;
  }
</style>
</head>
<body>
<div id="age">--</div>
<div id="wrap"><div id="text">connecting...</div></div>
<script>
async function poll() {
  try {
    const r = await fetch('/api/latest', {cache: 'no-store'});
    const j = await r.json();
    document.getElementById('text').textContent = j.text;
    const age = Math.max(0, (Date.now()/1000) - j.ts);
    document.getElementById('age').textContent = j.ts ? (age.toFixed(0) + 's ago') : '--';
  } catch (e) {
    document.getElementById('age').textContent = 'disconnected';
  }
  setTimeout(poll, 1500);
}
poll();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/display", methods=["POST"])
def display():
    text = request.get_data(as_text=True) or ""
    with _lock:
        _latest["text"] = text
        _latest["ts"] = time.time()
    return "", 200


@app.route("/api/latest")
def api_latest():
    with _lock:
        return jsonify(dict(_latest))


if __name__ == "__main__":
    print(f"Glasses display server on http://0.0.0.0:{PORT}")
    print("Run `tailscale serve --bg %d` in another terminal for a phone-reachable HTTPS URL." % PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
