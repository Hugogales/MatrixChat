"""Conservative grammar/spelling correction for noisy chat-log text.

Written for the Molweni adapter (Ubuntu IRC support chat, `content_bearing
=False` -- its text is NEVER used as a content-loss supervision target,
only as structural/turn-taking signal; see `data/dataset_plan.md` and
`docs/RESEARCH_REFERENCE.md` section 4.4). Cleaning it therefore cannot
change what the model is trained to GENERATE, only the quality of the raw
input tokens the shared hidden representation (which the activity head
also reads) is built from -- a real but second-order benefit. This module
exists so that caveat is explicit and testable, not assumed.

Deliberately NOT a full grammar checker (no `language_tool_python`/Java
dependency, no rewriting of sentence structure). It only fixes three
narrow, high-confidence categories of error, with several guards against
"correcting" legitimate technical jargon into nonsense:

0. PTB-tokenization spacing artifacts (Molweni's actual on-disk text is
   space-tokenized: "i 'm", "does n't", "that ,", "` ` quoted ' '" rather
   than written naturally) -- see :func:`detokenize_ptb`. Run first, since
   it changes token boundaries the other two passes then operate on.
1. A curated dictionary of unambiguous missing-apostrophe contractions
   ("dont" -> "don't"). Deliberately EXCLUDES common real-word collisions
   ("id", "ill", "its", "well") where the "typo" is at least as often the
   intended word (e.g. "the process id", "he is ill") -- guessing wrong
   there would inject a genuine grammar error, the opposite of the goal.
2. Single-word spelling corrections from ``pyspellchecker``'s dictionary,
   gated by: skip ALL-CAPS acronyms; skip tokens containing digits or any
   punctuation other than an apostrophe (paths, flags, commands, version
   strings -- e.g. "apt-get", "/etc/fstab", "v2.1"); skip anything on the
   tech/IRC whitelist below; skip if pyspellchecker has no confident
   suggestion, or the suggestion's first letter differs from the original
   (a standard heuristic against wild guesses); preserve the original
   token's capitalization pattern on any accepted correction.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Unambiguous missing-apostrophe contractions only. Excludes "id"/"ill"/
# "its"/"well" (real, common, DIFFERENT words in this domain -- see
# module docstring) even though they are sometimes also typos.
CONTRACTIONS: dict[str, str] = {
    "dont": "don't", "cant": "can't", "wont": "won't",
    "isnt": "isn't", "wasnt": "wasn't", "werent": "weren't", "arent": "aren't",
    "hasnt": "hasn't", "havent": "haven't", "hadnt": "hadn't",
    "doesnt": "doesn't", "didnt": "didn't",
    "couldnt": "couldn't", "shouldnt": "shouldn't", "wouldnt": "wouldn't",
    "youre": "you're", "theyre": "they're",
    "weve": "we've", "youve": "you've", "ive": "I've",
    "youll": "you'll", "theyll": "they'll",
    "theyve": "they've",
    "whats": "what's", "thats": "that's", "wheres": "where's",
    "hows": "how's", "whos": "who's", "lets": "let's",
    "shes": "she's", "hes": "he's", "aint": "ain't", "im": "I'm",
    # NOTE: "id"/"ill"/"its"/"well" are deliberately excluded -- each is a
    # common, correct standalone word in this domain at least as often as
    # it is a missing-apostrophe typo ("well" the adverb, "ill" = sick,
    # "id" = identifier, "its" = possessive), so guessing wrong would
    # inject a genuine grammar error rather than fix one.
}

# Common Ubuntu/Linux/IRC-support jargon that a general-purpose English
# spelling dictionary will flag as "unknown" but that must never be
# "corrected" into an unrelated real word. Not exhaustive -- extend as new
# false positives are found in real Molweni text.
TECH_WHITELIST: frozenset[str] = frozenset({
    "ubuntu", "linux", "unix", "debian", "kubuntu", "xubuntu", "lubuntu",
    "apt", "dpkg", "sudo", "bash", "zsh", "grep", "sed", "awk", "chmod",
    "chown", "fstab", "grub", "kernel", "initrd", "systemd", "sysvinit",
    "nvidia", "amd", "intel", "wifi", "bios", "uefi", "usb", "hdmi", "vga",
    "http", "https", "url", "uri", "ssh", "scp", "ftp", "sftp", "vnc",
    "gnome", "kde", "xfce", "lxde", "unity", "xorg", "wayland", "compiz",
    "deb", "rpm", "yum", "dnf", "pacman", "snap", "flatpak", "ppa",
    "netstat", "ifconfig", "iptables", "ufw", "cron", "crontab", "rsync",
    "symlink", "cli", "gui", "tty", "pty", "os", "vm", "vps", "iso",
    "vim", "emacs", "nano", "gedit", "gcc", "g++", "make", "cmake",
    "python", "perl", "php", "html", "css", "js", "json", "xml", "sql",
    "config", "configs", "conf", "plugin", "plugins", "addon", "addons",
    "firmware", "driver", "drivers", "hardware", "software", "malware",
    "laptop", "desktop", "netbook", "router", "modem", "ethernet",
    "bluetooth", "touchpad", "trackpad", "keyboard", "screenshot",
    "terminal", "shell", "login", "logout", "username", "hostname",
    "localhost", "filesystem", "partition", "mountpoint", "swapfile",
    "reboot", "dmesg", "syslog", "logfile", "backup", "backups",
    "codec", "codecs", "mp3", "mp4", "avi", "pdf", "jpg", "png",
    "wget", "curl", "tarball", "gzip", "bzip", "unzip", "unrar",
    "readme", "changelog", "makefile", "hostnames", "ipv4", "ipv6",
    "dns", "dhcp", "lan", "wan", "vpn", "proxy", "firewall", "malloc",
    "stdout", "stderr", "stdin", "runtime", "backend", "frontend",
    "database", "server", "servers", "client", "clients", "app", "apps",
    "workspace", "workspaces", "taskbar", "sidebar", "toolbar", "webcam",
})


@lru_cache(maxsize=1)
def _spellchecker():
    from spellchecker import SpellChecker  # local import: optional dependency

    return SpellChecker()


_WORD_RE = re.compile(r"[A-Za-z']+")


def _apply_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


# Characters that, when directly touching a word token (no whitespace),
# signal a path/flag/command/compound-jargon construct ("/etc/fstab",
# "apt-get", "nvidia-driver", "user@host", "key=value") even though the
# WORD-only regex below extracted just one piece of it ("etc", "apt").
# Deliberately excludes ordinary sentence punctuation (".", ",", "!", "?",
# ";", ":") so normal sentence-final/medial words are still correctable.
_JARGON_BOUNDARY_CHARS = frozenset("/-_@=~")


def _correct_word(token: str) -> str:
    lower = token.lower()
    if lower in CONTRACTIONS:
        return _apply_case(token, CONTRACTIONS[lower])
    if token.isupper() and len(token) > 1:
        return token  # acronym
    if lower in TECH_WHITELIST:
        return token
    if "'" in token:
        return token  # already has a real apostrophe; leave contractions alone
    sc = _spellchecker()
    if lower in sc:
        return token  # known word, nothing to fix
    if len(lower) < 3:
        return token  # too short to correct confidently
    candidate = sc.correction(lower)
    if candidate is None or candidate == lower:
        return token
    if candidate[0] != lower[0]:
        return token  # guards against a wild first-letter-changing guess
    return _apply_case(token, candidate)


# Molweni's actual on-disk text (both DP/ and MRC(withDiscourse)/ splits) is
# PTB-tokenized: punctuation and contraction suffixes are space-separated
# ("i 'm", "does n't", "that ,", "this ?", "`` quoted ''") rather than
# written naturally. This is arguably a BIGGER quality issue than spelling
# for this source, so it is fixed first, before spelling correction runs.
_PTB_CLOSE_PUNCT = re.compile(r"\s+([.,!?;:%)\]}])")
_PTB_OPEN_BRACKET = re.compile(r"([(\[{])\s+")
_PTB_CONTRACTION_SUFFIX = re.compile(r"(\w)\s+'\s*(s|re|ve|d|ll|m)\b", re.IGNORECASE)
_PTB_NT = re.compile(r"(\w)\s+n't\b", re.IGNORECASE)
_PTB_OPEN_QUOTE = re.compile(r"``\s*")
_PTB_CLOSE_QUOTE = re.compile(r"\s*''")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def detokenize_ptb(text: str) -> str:
    """Undo PTB-style tokenization spacing (see module note above).

    Known limitation: a bare "." tokenized with surrounding spaces cannot
    be distinguished here from a genuine sentence-final period vs. a
    filename-extension separator that happened to get a stray tokenizer
    space (e.g. "a .run file" -> "a.run file", changing what should stay
    "a .run"). Rare in practice and, since Molweni is content_bearing=
    False, never a supervised generation target either way -- left as a
    known, documented imperfection rather than adding fragile heuristics.
    """
    if not text:
        return text
    text = _PTB_OPEN_QUOTE.sub('"', text)
    text = _PTB_CLOSE_QUOTE.sub('"', text)
    text = _PTB_CONTRACTION_SUFFIX.sub(r"\1'\2", text)
    text = _PTB_NT.sub(r"\1n't", text)
    text = _PTB_CLOSE_PUNCT.sub(r"\1", text)
    text = _PTB_OPEN_BRACKET.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Apply conservative contraction + spelling fixes to ``text``.

    Only ``[A-Za-z']+`` word tokens are ever touched (via
    ``_correct_word``'s own additional guards); all other characters
    (punctuation, digits, whitespace, URLs, paths) pass through
    unmodified, preserving the original message's structure exactly. A
    token directly touching one of ``_JARGON_BOUNDARY_CHARS`` in the
    ORIGINAL text (e.g. the "etc" inside "/etc/fstab", or the "apt" inside
    "apt-get") is left untouched even though the word-only regex extracted
    just that one piece -- correcting a fragment of a path/command/
    hyphenated-compound in isolation is exactly the kind of jargon-mangling
    this module exists to avoid.
    """
    if not text or not any(ch.isalpha() for ch in text):
        return text
    if "://" in text or text.strip().startswith(("/", "~/", "$")):
        return text  # URL-like or path-like line: don't touch at all

    text = detokenize_ptb(text)

    def _replace(m: "re.Match[str]") -> str:
        before = text[m.start() - 1] if m.start() > 0 else ""
        after = text[m.end()] if m.end() < len(text) else ""
        if before in _JARGON_BOUNDARY_CHARS or after in _JARGON_BOUNDARY_CHARS:
            return m.group(0)
        return _correct_word(m.group(0))

    return _WORD_RE.sub(_replace, text)
