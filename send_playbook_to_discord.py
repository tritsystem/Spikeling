r"""
Sends The Playbook (the tritsystem portfolio study guide) to your own Discord
DM as a file attachment, via the same bot/token spikebot (discord_bot.py) and
observe-api's status-update script already use.

Run this yourself, in your own terminal, with the token set there -- it is
never pasted into chat or seen by Claude. Regenerate the token from
https://discord.com/developers/applications if it's ever exposed anywhere.

Usage (PowerShell):
    $env:DISCORD_BOT_TOKEN = "your real token"
    $env:DISCORD_AUTHORIZED_USER_ID = "your numeric discord user id"
    python send_playbook_to_discord.py "C:\path\to\the-playbook-standalone.html"

Usage (bash / WSL):
    export DISCORD_BOT_TOKEN="your real token"
    export DISCORD_AUTHORIZED_USER_ID="your numeric discord user id"
    python3 send_playbook_to_discord.py /path/to/the-playbook-standalone.html
"""
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid

API = "https://discord.com/api/v10"

# Discord's API is fronted by Cloudflare, which blocks urllib's default
# "Python-urllib/x.y" User-Agent outright (Cloudflare error 1010) before the
# request ever reaches Discord. Discord's own API docs ask for a descriptive
# UA in this shape anyway -- this satisfies both at once.
USER_AGENT = "DiscordBot (https://github.com/tritsystem/Spikeling, 1.0)"


def _post(url, token, data, content_type):
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": content_type,
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read()


def get_dm_channel(token, user_id):
    import json

    body = json.dumps({"recipient_id": user_id}).encode()
    status, resp = _post(
        f"{API}/users/@me/channels", token, body, "application/json"
    )
    channel = __import__("json").loads(resp)
    return channel["id"]


def send_file(token, channel_id, file_path, message):
    boundary = uuid.uuid4().hex
    filename = os.path.basename(file_path)
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="content"\r\n\r\n')
    parts.append(message.encode() + b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="files[0]"; filename="{filename}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    content_type = f"multipart/form-data; boundary={boundary}"
    status, resp = _post(
        f"{API}/channels/{channel_id}/messages", token, body, content_type
    )
    return status


def main():
    token = os.environ.get("DISCORD_BOT_TOKEN")
    user_id = os.environ.get("DISCORD_AUTHORIZED_USER_ID")
    if not token or not user_id:
        print("Set DISCORD_BOT_TOKEN and DISCORD_AUTHORIZED_USER_ID in your own "
              "shell first (see the docstring at the top of this file).")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python send_playbook_to_discord.py <path-to-html-file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        channel_id = get_dm_channel(token, user_id)
        status = send_file(
            token,
            channel_id,
            file_path,
            "The Playbook -- your tritsystem portfolio study guide. Open the "
            "attached .html file directly in Safari, no claude.ai login needed.",
        )
        if status in (200, 201):
            print("Sent. Check your Discord DMs.")
        else:
            print(f"Discord returned HTTP {status} -- something went wrong.")
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.read().decode(errors='replace')}")


if __name__ == "__main__":
    main()
