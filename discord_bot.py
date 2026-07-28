"""
Discord front-end for voice_commands.py's command/knowledge system.

Reuses process_text() directly -- same destructive/financial/credential
refusals, same casual persona, same knowledge fallback, same real MiniLM
code search, same sensitive-command gate. Nothing is reimplemented here,
only a new front-end (this is the same pattern typed-input parity used:
one shared matching path, many entry points).

SETUP -- these are account/token steps only you can do, not something I
can do on your behalf:

  1. Go to https://discord.com/developers/applications, "New Application",
     add a Bot to it, and copy its token. Keep it secret -- never paste it
     into chat, a commit, or anywhere else. If it ever leaks, regenerate it
     from the same page immediately.
  2. In the bot's settings, under "Privileged Gateway Intents", enable
     "Message Content Intent" (needed to read what a message actually
     says, not just that one arrived).
  3. Set two environment variables before running this script:
       DISCORD_BOT_TOKEN=<the token from step 1>
       DISCORD_AUTHORIZED_USER_ID=<your own numeric Discord user ID>
     To get your ID: Discord Settings -> Advanced -> enable Developer
     Mode, then right-click your own name anywhere -> Copy User ID.
  4. Either invite the bot to a server you control (OAuth2 URL Generator
     -> scope "bot" -> permissions "Send Messages" + "Attach Files"), or
     just DM the bot directly -- DMs work without inviting it anywhere,
     as long as you share a server with it OR it allows DMs from its
     application settings.
  5. Run: python discord_bot.py

SECURITY:
  - Only messages from DISCORD_AUTHORIZED_USER_ID are ever acted on --
    every other author (including other people in a shared server) is
    silently ignored. This is the actual permission gate for a remote
    control surface; nothing here is "public."
  - Sensitive commands (screenshot / gif / report / compose email / code
    search / claude code) require an admin unlock. Unlock is TOTP: send
    "!unlock <6-digit code>" from your authenticator app once per bot run
    to enable them for the rest of that session. Codes are single-use and
    expire in ~30s, so even though the message passes through Discord (and
    may linger in history if the bot can't delete it), the code left
    behind is already worthless -- that's why this replaced the old fixed
    passphrase, which leaked into chat history permanently. Set the TOTP
    secret up once with set_admin_totp.py.
  - Destructive/financial/credential requests are refused the same way
    they are locally -- this front-end doesn't add or remove any of
    that; it's all in process_text() itself.
  - Local text-to-speech still fires on this machine for every command,
    even when triggered remotely via Discord -- say the word if you'd
    rather it stay silent for Discord-originated commands specifically.
"""
import os
import sys
import asyncio
import discord

import voice_commands as vc

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
AUTHORIZED_USER_ID = os.environ.get("DISCORD_AUTHORIZED_USER_ID", "")

# Per-bot-run unlock state for sensitive commands -- in-memory only, never
# persisted, resets every time this script restarts. No console exists in
# a chat interface, so this replaces voice_commands.py's hidden getpass
# prompt (see set_admin_auth_check() below).
_unlocked = False
_autolock_task = None   # asyncio Task that re-locks after a timer, or None


async def _auto_lock_after(minutes, channel):
    """Re-lock after `minutes`. Cancelled if the user re-unlocks or !locks."""
    global _unlocked
    try:
        await asyncio.sleep(minutes * 60)
        _unlocked = False
        await channel.send(f"auto-locked after {minutes} min, bro. `!unlock <code>` to go again.")
    except asyncio.CancelledError:
        pass


def discord_auth_check():
    # Real bug this fixes: process_text()'s sensitive-command gate just
    # returns silently on refusal (the local console path already says
    # "wrong password" itself, inside request_admin_password_console() --
    # this callback is what stands in for that on Discord, so refusing
    # with no message at all made a MATCHED-but-locked command look
    # identical to "didn't recognize this at all" once relayed to Discord.
    if not _unlocked:
        vc.say("that's a sensitive command, bro -- send `!unlock <6-digit code>` from your authenticator first.")
    return _unlocked


vc.set_admin_auth_check(discord_auth_check)

# LIVE response posting. process_text() runs on a worker thread (see
# asyncio.to_thread below) and calls say() -> this sink at each stage. We
# post EACH say() to Discord the moment it fires -- via
# run_coroutine_threadsafe onto the bot's event loop -- and block the worker
# thread on the result so messages arrive in order and a long multi-stage
# command (e.g. "work on tribe", ~5 min) streams its progress instead of
# going silent until the very end. _responses still records them so on_message
# can tell "said nothing" from "said something".
_responses = []
_current_channel = None
_event_loop = None


def discord_response_sink(text):
    _responses.append(text)
    if _current_channel is None or _event_loop is None:
        return
    # Discord hard-caps a message at 2000 chars -- chunk defensively.
    for i in range(0, len(text), 1900):
        chunk = text[i:i + 1900]
        try:
            fut = asyncio.run_coroutine_threadsafe(_current_channel.send(chunk), _event_loop)
            fut.result(timeout=30)   # block worker thread so ordering is preserved
        except Exception as e:
            print(f"(live Discord post failed, non-fatal: {e})", flush=True)


vc.set_response_sink(discord_response_sink)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Discord bot logged in as {client.user}.", flush=True)
    if AUTHORIZED_USER_ID:
        print(f"Only acting on messages from user ID {AUTHORIZED_USER_ID}.", flush=True)
        # STARTUP NOTIFICATION (2026-07-28): "when its ready have the bot
        # message the same discord channel that its ready" -- a DM to the
        # authorized user directly (not a guess at which server channel),
        # since the setup docs' primary interaction mode IS a DM with the
        # bot. Best-effort: a failed DM (e.g. the user has DMs from this bot
        # disabled) is logged, not fatal to startup.
        try:
            user = await client.fetch_user(int(AUTHORIZED_USER_ID))
            await user.send(
                "back online, bro -- upgraded: Claude Code tasks now stream "
                "live progress as they work (not just a final summary), and "
                "`claude code <task>` auto-continues your last conversation "
                "in that project (use `claude code new: <task>` to start "
                "fresh instead).")
        except Exception as e:
            print(f"(startup DM failed, non-fatal: {e})", flush=True)
    else:
        print("WARNING: DISCORD_AUTHORIZED_USER_ID is not set -- every message will be ignored.", flush=True)


@client.event
async def on_message(message):
    global _unlocked, _responses, _autolock_task

    if message.author.bot:
        return
    if not AUTHORIZED_USER_ID or str(message.author.id) != str(AUTHORIZED_USER_ID):
        return  # hard allowlist -- silently ignore everyone else, no exceptions

    text = message.content.strip()
    if not text:
        return

    # Manual re-lock. Cancels any pending auto-lock timer too.
    if text.lower().strip() in ("!lock", "!lock now"):
        _unlocked = False
        if _autolock_task:
            _autolock_task.cancel()
            _autolock_task = None
        await message.channel.send("locked, bro. sensitive commands are gated again.")
        return

    if text.lower().startswith("!unlock"):
        # "!unlock <code>"  or  "!unlock <code> <minutes>" for an auto-lock timer.
        parts = text[len("!unlock"):].strip().split()
        code = parts[0] if parts else ""   # a 6-digit TOTP code
        minutes = None
        if len(parts) > 1:
            try:
                minutes = max(1, min(1440, int(parts[1])))   # clamp 1 min .. 24 h
            except ValueError:
                minutes = None
        # Best-effort delete, but security no longer DEPENDS on it: a TOTP
        # code is single-use and expires in ~30s, so even if the delete
        # fails (e.g. the bot lacks Manage Messages, or this is a DM where
        # bots can't delete your messages at all) the code left in history
        # is already worthless. This is the whole point of switching off a
        # reusable passphrase.
        try:
            await message.delete()
        except Exception:
            pass
        if vc.check_admin_code(code):
            _unlocked = True
            if _autolock_task:            # replace any existing timer
                _autolock_task.cancel()
                _autolock_task = None
            if minutes:
                _autolock_task = asyncio.create_task(_auto_lock_after(minutes, message.channel))
                await message.channel.send(f"bet, unlocked for {minutes} min bro -- auto-locks after that. (`!lock` to lock early)")
            else:
                await message.channel.send("bet, unlocked for the session bro. (`!lock` to lock, or `!unlock <code> <minutes>` to set an auto-lock timer)")
        else:
            await message.channel.send("nah, wrong or expired code, bro -- still locked. (codes change every 30s)")
        return

    async with message.channel.typing():
        global _current_channel, _event_loop
        _responses.clear()
        _current_channel = message.channel      # sink posts live to THIS channel
        _event_loop = asyncio.get_running_loop()
        vc._last_attachment_path = None   # only attach a file THIS command actually produced
        try:
            # process_text() blocks for real (a Claude Code task can run up
            # to ~5 min; a knowledge answer ~30-40s) -- running it on the
            # event loop thread directly would stall Discord's heartbeat and
            # risk a gateway disconnect, so it runs in a worker thread.
            # Pass BOTH the lowercased text (for matching) and the original
            # (raw_text) so a Claude Code task keeps its identifier/path
            # casing. Every say() inside streams to Discord live via the sink.
            await asyncio.to_thread(vc.process_text, text.lower(), 0.0, "discord", text)
        except Exception as e:
            await message.channel.send(f"hit an error running that, bro: {e}")
            return
        finally:
            _current_channel = None
            _event_loop = None

        # Responses were already posted live by the sink; here we only handle
        # the "matched nothing at all" case and any file attachment.
        if not _responses:
            await message.channel.send("(didn't match anything, bro)")

        if vc._last_attachment_path and os.path.exists(vc._last_attachment_path):
            await message.channel.send(file=discord.File(vc._last_attachment_path))


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("Set DISCORD_BOT_TOKEN before running this script (see the setup steps at the top of this file).", flush=True)
        sys.exit(1)
    if not AUTHORIZED_USER_ID:
        print("Set DISCORD_AUTHORIZED_USER_ID before running this script, or the bot will ignore every message it gets.", flush=True)
    client.run(DISCORD_BOT_TOKEN)
