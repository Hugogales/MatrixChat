"""Text-message alerts for the always-on Werewolf RL loop.

Settings live in a private file outside the repo (default
``~/.config/werewolf_rl/notify.env``, or the path in ``$NOTIFY_CONFIG``); see
``scripts/training/werewolf_continuous/notify.env.example``. Without that file,
or without a phone number in it, every call here does nothing, so the loop
runs the same with or without alerts.

Providers (``NOTIFY_PROVIDER``):

* ``ntfy``     -- free push notifications to the ntfy phone app: an HTTPS POST to
  ``NTFY_SERVER/NTFY_TOPIC`` (default server https://ntfy.sh). The topic name
  works like a password, so make it long and random. Needs no phone number.
* ``gateway``  -- free: emails the carrier's email-to-text address
  (``5551234567@vtext.com`` for Verizon) through your own SMTP account (e.g.
  Gmail with an app password). This is what self-hosted Textbelt does, without
  running its server. Texts are split into 160-character parts.
* ``textbelt`` -- textbelt.com; the free key ``textbelt`` allows one text per day.
* ``twilio``   -- needs a Twilio account SID, auth token and sending number.

``NOTIFY_FALLBACK`` (e.g. ``textbelt``) is tried when the main provider fails
to hand the message over. A carrier can still silently drop a gateway email
that the mail server accepted; no fallback can see that.

Rosie's compute nodes block outgoing email, so with ``NOTIFY_RELAY=cron``
(the default for ``gateway``) jobs do not send anything themselves: each alert
is queued as a file in ``NOTIFY_OUTBOX`` (``~/.config/werewolf_rl/outbox``, on
the shared /home), and a cron task on the login node runs ``flush`` every two
minutes to email it. An alert that still cannot be sent after
``NOTIFY_FALLBACK_AFTER_MINUTES`` goes out through ``NOTIFY_FALLBACK``.

Sending never raises: an alert that cannot be delivered is logged and the job
carries on. Repeats of the same alert within ``NOTIFY_REPEAT_MINUTES`` are
skipped.

CLI:
    python -m training.werewolf_rl.notify test            # queue a test text and send it now
    python -m training.werewolf_rl.notify flush           # send queued alerts (what cron runs)
    python -m training.werewolf_rl.notify install-cron    # add the 2-minute relay to this node's crontab
    python -m training.werewolf_rl.notify uninstall-cron
    python -m training.werewolf_rl.notify send --event crash --run_dir RUN "message"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_CONFIG = os.path.expanduser("~/.config/werewolf_rl/notify.env")
# Carrier email-to-text domains. AT&T (June 2025) and T-Mobile (~2024) have shut
# theirs down; Verizon plans to end vtext.com on 2027-03-31.
GATEWAYS = {
    "verizon": "vtext.com",
    "googlefi": "msg.fi.google.com",
    "uscellular": "email.uscc.net",
    "att": "txt.att.net",
    "tmobile": "tmomail.net",
}
SMS_CHARS = 160
NTFY_PRIORITY = {"stopped": "urgent", "gave_up": "high", "crash": "high", "start": "default", "test": "default"}
NTFY_TAGS = {"stopped": "rotating_light", "gave_up": "warning", "crash": "warning", "start": "rocket",
             "test": "white_check_mark"}
DEFAULT_OUTBOX = os.path.expanduser("~/.config/werewolf_rl/outbox")
CRON_MARKER = "# werewolf_rl notify relay"
DEFAULT_EVENTS = "start,crash,gave_up,stopped"


def load_config(path=None):
    """``KEY=VALUE`` lines (quotes and ``#`` comments allowed) -> dict; {} if the file is missing."""
    path = path or os.environ.get("NOTIFY_CONFIG") or DEFAULT_CONFIG
    cfg = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.split(" #", 1)[0].strip().strip('"').strip("'")
                cfg[key.strip()] = value
    except FileNotFoundError:
        return {}
    return cfg


def _post(url, fields, auth=None, timeout=15, headers=None, body=None):
    data = body.encode() if body is not None else urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    if auth:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode(errors="replace")


def sms_parts(message, limit=SMS_CHARS, max_parts=3):
    """Split ``message`` into SMS-sized parts, numbered ``1/2 ...`` when there is more than one."""
    if len(message) <= limit:
        return [message]
    body = limit - 4  # room for "1/3 "
    chunks = [message[i:i + body] for i in range(0, len(message), body)][:max_parts]
    return [f"{k + 1}/{len(chunks)} {chunk}" for k, chunk in enumerate(chunks)]


def _send_gateway(message, cfg, phone):
    import smtplib
    from email.message import EmailMessage

    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return False, f"gateway needs a 10-digit US number, got {phone!r}"
    carrier = cfg.get("NOTIFY_CARRIER", "verizon").strip().lower()
    domain = cfg.get("NOTIFY_GATEWAY_DOMAIN", "").strip() or GATEWAYS.get(carrier)
    if not domain:
        return False, f"unknown NOTIFY_CARRIER {carrier!r}; set NOTIFY_GATEWAY_DOMAIN"
    host = cfg.get("SMTP_HOST", "smtp.gmail.com")
    port = int(cfg.get("SMTP_PORT", "465"))
    user, password = cfg.get("SMTP_USER", ""), cfg.get("SMTP_PASSWORD", "").replace(" ", "")
    if not (user and password):
        return False, "gateway needs SMTP_USER and SMTP_PASSWORD (a Gmail app password)"
    sender = cfg.get("SMTP_FROM", "") or user
    to = f"{digits}@{domain}"
    if port == 465:  # implicit TLS: the only Gmail port Rosie's login node can reach
        connection = smtplib.SMTP_SSL(host, port, timeout=20)
    else:
        connection = smtplib.SMTP(host, port, timeout=20)
    with connection as smtp:
        if port != 465:
            smtp.starttls()
        smtp.login(user, password)
        for part in sms_parts(message):
            mail = EmailMessage()
            mail["From"], mail["To"], mail["Subject"] = sender, to, ""
            mail.set_content(part)
            smtp.send_message(mail)
    return True, f"gateway: emailed {to} via {host}"


def provider_of(cfg):
    return cfg.get("NOTIFY_PROVIDER", "textbelt").strip().lower()


def enabled(cfg):
    """Alerts are on when the chosen provider has what it needs to reach you."""
    if provider_of(cfg) == "ntfy":
        return bool(cfg.get("NTFY_TOPIC", "").strip())
    return bool(cfg.get("NOTIFY_PHONE", "").strip())


def send_text(message, cfg=None, event=None):
    """Send ``message`` through ``NOTIFY_PROVIDER``. Returns ``(ok, detail)``; never raises.

    If that fails and ``NOTIFY_FALLBACK`` names another provider, tries it once.
    """
    cfg = load_config() if cfg is None else cfg
    if not enabled(cfg):
        return False, "alerts not configured (no NTFY_TOPIC / NOTIFY_PHONE)"
    phone = cfg.get("NOTIFY_PHONE", "").strip()
    primary = provider_of(cfg)
    ok, detail = _send_with(primary, message[:480], cfg, phone, event)
    fallback = cfg.get("NOTIFY_FALLBACK", "").strip().lower()
    if not ok and fallback and fallback != primary and (fallback == "ntfy" or phone):
        ok2, detail2 = _send_with(fallback, message[:300], cfg, phone, event)
        return ok2, f"{detail}; fallback {detail2}"
    return ok, detail


def _send_with(provider, message, cfg, phone, event=None):
    import smtplib

    try:
        if provider == "ntfy":
            topic = cfg.get("NTFY_TOPIC", "").strip()
            if not topic:
                return False, "ntfy needs NTFY_TOPIC"
            server = cfg.get("NTFY_SERVER", "").strip().rstrip("/") or "https://ntfy.sh"
            headers = {"Title": cfg.get("NTFY_TITLE", "Werewolf RL"),
                       "Priority": NTFY_PRIORITY.get(event, "default"),
                       "Tags": NTFY_TAGS.get(event, "robot")}
            if cfg.get("NTFY_TOKEN"):
                headers["Authorization"] = f"Bearer {cfg['NTFY_TOKEN']}"
            status, body = _post(f"{server}/{topic}", None, headers=headers, body=message)
            return 200 <= status < 300, f"ntfy {status} ({server}/{topic[:6]}...)"
        if not phone:
            return False, f"{provider} needs NOTIFY_PHONE"
        if provider == "gateway":
            return _send_gateway(message, cfg, phone)
        if provider == "twilio":
            sid, token, sender = (cfg.get(k, "") for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))
            if not (sid and token and sender):
                return False, "twilio needs TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM"
            status, body = _post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                                 {"To": phone, "From": sender, "Body": message}, auth=(sid, token))
            return 200 <= status < 300, f"twilio {status}: {body[:200]}"
        if provider == "textbelt":
            status, body = _post("https://textbelt.com/text",
                                 {"phone": phone, "message": message, "key": cfg.get("TEXTBELT_KEY", "textbelt")})
            result = json.loads(body) if body.startswith("{") else {}
            return bool(result.get("success")), f"textbelt {status}: {body[:200]}"
        return False, f"unknown provider {provider!r} (use ntfy, gateway, textbelt or twilio)"
    except smtplib.SMTPException as exc:
        return False, f"{provider} SMTP error: {exc}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        detail = exc.read().decode(errors="replace")[:200] if isinstance(exc, urllib.error.HTTPError) else str(exc)
        return False, f"{provider} error: {detail}"


def outbox_dir(cfg):
    return os.path.expanduser(cfg.get("NOTIFY_OUTBOX", "") or DEFAULT_OUTBOX)


def uses_relay(cfg):
    return (cfg.get("NOTIFY_PROVIDER", "textbelt").strip().lower() == "gateway"
            and cfg.get("NOTIFY_RELAY", "cron").strip().lower() == "cron")


def queue_alert(message, cfg):
    """Drop an alert into the outbox for the login-node relay. Returns ``(ok, detail)``."""
    folder = outbox_dir(cfg)
    try:
        os.makedirs(folder, exist_ok=True)
        name = f"{time.time():.6f}_{os.getpid()}.json"
        tmp = os.path.join(folder, "." + name)
        with open(tmp, "w") as f:
            json.dump({"t": time.time(), "message": message}, f)
        os.replace(tmp, os.path.join(folder, name))
        return True, f"queued in {folder} for the login-node relay"
    except OSError as exc:
        return False, f"could not queue: {exc}"


def flush(cfg=None, now=None, log=print):
    """Send every queued alert (oldest first). Returns how many were delivered.

    Each alert is tried through ``NOTIFY_PROVIDER``; one that keeps failing for
    ``NOTIFY_FALLBACK_AFTER_MINUTES`` is sent through ``NOTIFY_FALLBACK`` instead.
    Delivered alerts move to ``outbox/sent/`` (last 200 kept).
    """
    import fcntl

    cfg = load_config() if cfg is None else cfg
    folder = outbox_dir(cfg)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".last_flush"), "w") as f:  # lets status.sh show the relay is alive
        f.write(f"{time.time()} {os.uname().nodename}\n")
    phone = cfg.get("NOTIFY_PHONE", "").strip()
    sent_dir = os.path.join(folder, "sent")
    os.makedirs(sent_dir, exist_ok=True)
    with open(os.path.join(folder, ".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # an overlapping cron run just skips
        except OSError:
            return 0
        delivered = 0
        primary = cfg.get("NOTIFY_PROVIDER", "textbelt").strip().lower()
        fallback = cfg.get("NOTIFY_FALLBACK", "").strip().lower()
        wait = float(cfg.get("NOTIFY_FALLBACK_AFTER_MINUTES", "10")) * 60
        for name in sorted(n for n in os.listdir(folder) if n.endswith(".json") and not n.startswith(".")):
            path = os.path.join(folder, name)
            try:
                with open(path) as f:
                    alert = json.load(f)
            except (OSError, ValueError):
                continue
            if not enabled(cfg):
                log(f"[relay] {name}: alerts not configured; leaving it queued")
                break
            ok, detail = _send_with(primary, alert["message"][:480], cfg, phone)
            age = (now or time.time()) - alert.get("t", 0)
            if not ok and fallback and fallback != primary and age >= wait:
                ok, fb = _send_with(fallback, alert["message"][:300], cfg, phone)
                detail = f"{detail}; after {age / 60:.0f} min, fallback {fb}"
            log(f"[relay] {time.strftime('%F %T')} {name}: {'sent' if ok else 'not sent yet'} ({detail})")
            if ok:
                delivered += 1
                os.replace(path, os.path.join(sent_dir, name))
        old = sorted(os.listdir(sent_dir))
        for name in old[:-200]:
            try:
                os.remove(os.path.join(sent_dir, name))
            except OSError:
                pass
        return delivered


def relay_status(cfg=None):
    """``(seconds since the relay last ran or None, host, queued alerts)``."""
    cfg = load_config() if cfg is None else cfg
    folder = outbox_dir(cfg)
    queued = len([n for n in os.listdir(folder) if n.endswith(".json") and not n.startswith(".")]) \
        if os.path.isdir(folder) else 0
    try:
        with open(os.path.join(folder, ".last_flush")) as f:
            t, host = f.read().split()
        return time.time() - float(t), host, queued
    except (OSError, ValueError):
        return None, None, queued


def cron_line(repo, python=None, log_path=None):
    python = python or sys.executable
    log_path = log_path or os.path.join(os.path.dirname(DEFAULT_OUTBOX), "relay.log")
    return (f"*/2 * * * * cd {repo} && {python} -m training.werewolf_rl.notify flush >> {log_path} 2>&1 "
            f"{CRON_MARKER}")


def install_cron(repo, remove=False):
    """Add (or remove) the relay line in this node's crontab; other lines are kept."""
    import subprocess

    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = [line for line in (current.stdout.splitlines() if current.returncode == 0 else [])
             if CRON_MARKER not in line]
    if not remove:
        lines.append(cron_line(repo))
    result = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"crontab failed: {result.stderr.strip()}")
    return lines


def notify(event, message, run_dir=None, cfg=None):
    """Text an alert for ``event`` (start / crash / gave_up / stopped) if it is enabled.

    With ``run_dir``, every attempt is logged to ``RUN/chain/notify.log`` and an
    identical alert sent within ``NOTIFY_REPEAT_MINUTES`` (default 30) is skipped.
    """
    cfg = load_config() if cfg is None else cfg
    if not enabled(cfg):
        return False
    events = {e.strip() for e in cfg.get("NOTIFY_EVENTS", DEFAULT_EVENTS).split(",") if e.strip()}
    if event not in events:
        return False
    log_path = os.path.join(run_dir, "chain", "notify.log") if run_dir else None
    repeat = float(cfg.get("NOTIFY_REPEAT_MINUTES", "30")) * 60
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("message") == message and rec.get("ok") and time.time() - rec.get("t", 0) < repeat:
                        return False
        except (OSError, ValueError):
            pass
    ok, detail = queue_alert(message, cfg) if uses_relay(cfg) else send_text(message, cfg, event)
    print(f"[notify] {event}: {'ok' if ok else 'FAILED'} ({detail})", flush=True)
    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "event": event, "message": message, "ok": ok,
                                    "detail": detail}) + "\n")
        except OSError:
            pass
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test", help="Send a test text to the configured phone (queue + flush on the login node).")
    sub.add_parser("flush", help="Send queued alerts (run by cron on the login node).")
    sub.add_parser("install-cron", help="Add the every-2-minutes relay to this node's crontab.")
    sub.add_parser("uninstall-cron", help="Remove the relay from this node's crontab.")
    s = sub.add_parser("send", help="Send an alert (used by the job scripts).")
    s.add_argument("--event", required=True)
    s.add_argument("--run_dir", default=None)
    s.add_argument("message")
    a = p.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if a.cmd in ("install-cron", "uninstall-cron"):
        install_cron(repo, remove=a.cmd == "uninstall-cron")
        print(("Removed" if a.cmd == "uninstall-cron" else f"Installed on {os.uname().nodename}:\n  "
               + cron_line(repo)))
        return
    if a.cmd == "flush":
        flush()
        return
    if a.cmd == "test":
        cfg = load_config()
        if not cfg:
            sys.exit(f"No config at {os.environ.get('NOTIFY_CONFIG') or DEFAULT_CONFIG}; copy "
                     "scripts/training/werewolf_continuous/notify.env.example there and fill it in.")
        if not enabled(cfg):
            sys.exit("FAILED: alerts not configured (set NTFY_TOPIC for ntfy, or NOTIFY_PHONE)")
        message = "Werewolf RL alerts are set up. You will get notifications like this one."
        if uses_relay(cfg):
            ok, detail = queue_alert(message, cfg)
            print(detail)
            if ok:
                ok = flush(cfg) > 0
                print("Sent through the relay path." if ok else
                      "Not sent yet (see the [relay] line above); cron will keep retrying it.")
        else:
            ok, detail = send_text(message, cfg, event="test")
            print(("Sent: " if ok else "FAILED: ") + detail)
        sys.exit(0 if ok else 1)
    notify(a.event, a.message, a.run_dir)


if __name__ == "__main__":
    main()
