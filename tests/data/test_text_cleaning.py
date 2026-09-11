"""Tests for data/adapters/_text_cleaning.py's conservative grammar/spelling
correction (written for Molweni; see that module's docstring for the
content_bearing=False caveat -- this only affects raw input token quality,
not any supervised target)."""

from data.adapters._text_cleaning import clean_text, detokenize_ptb


def test_fixes_unambiguous_missing_apostrophe_contractions():
    assert clean_text("i dont think thats right") == "i don't think that's right"
    assert clean_text("youre right, its fine") == "you're right, its fine"


def test_preserves_capitalization_on_contraction_fix():
    assert clean_text("Dont worry about it") == "Don't worry about it"


def test_does_not_guess_ambiguous_real_words():
    # "id"/"ill"/"its"/"well" are common correct words here, not just typos.
    assert clean_text("whats the process id") == "what's the process id"
    assert clean_text("he is feeling ill today") == "he is feeling ill today"
    assert clean_text("check its permissions") == "check its permissions"
    assert clean_text("well just try again") == "well just try again"


def test_fixes_common_misspellings():
    assert clean_text("i didnt recieve the update") == "i didn't receive the update"
    assert "definitely" in clean_text("its definately broken")


def test_leaves_tech_jargon_untouched():
    text = "sudo apt-get install nvidia-driver then check /etc/fstab and dmesg"
    assert clean_text(text) == text


def test_leaves_acronyms_and_paths_untouched():
    assert clean_text("run LDAP then check USB") == "run LDAP then check USB"
    assert clean_text("edit /etc/X11/xorg.conf now") == "edit /etc/X11/xorg.conf now"


def test_leaves_version_strings_and_digit_tokens_untouched():
    assert clean_text("upgrading to v2.1 with gcc4") == "upgrading to v2.1 with gcc4"


def test_leaves_urls_untouched():
    text = "see https://help.ubuntu.com/teh-guide for mroe info"
    assert clean_text(text) == text


def test_empty_and_punctuation_only_text_is_unchanged():
    assert clean_text("") == ""
    assert clean_text("   ") == "   "
    assert clean_text("!!! ??? ...") == "!!! ??? ..."


def test_short_tokens_are_never_corrected():
    # Two-letter tokens are common abbreviations/typos too ambiguous to guess.
    assert clean_text("ok so hw do i fix teh gpu") == "ok so hw do i fix the gpu"


def test_correctly_spelled_text_is_unchanged():
    text = "thanks for the help, that fixed it perfectly"
    assert clean_text(text) == text


# ---------------------------------------------------------------------------
# detokenize_ptb -- Molweni's actual on-disk format is PTB-tokenized.
# ---------------------------------------------------------------------------


def test_detokenize_joins_contraction_suffixes():
    assert detokenize_ptb("i 'm not auth'ed to services") == "i'm not auth'ed to services"
    assert detokenize_ptb("cause there probably is n't one") == "cause there probably isn't one"


def test_detokenize_removes_space_before_closing_punctuation():
    assert detokenize_ptb("where do i look to find that ?") == "where do i look to find that?"
    assert detokenize_ptb("i have , but i have FILEPATH") == "i have, but i have FILEPATH"
    assert detokenize_ptb("not sure i follow . what iso is this ?") == (
        "not sure i follow. what iso is this?"
    )


def test_detokenize_handles_ptb_quotes():
    assert detokenize_ptb("it sounds like `` bridging ''") == 'it sounds like "bridging"'


def test_clean_text_runs_detokenize_before_spelling_pass():
    raw = "llutz , you understand what z3r0-0n3 wants ? i thought bridging was n't right"
    cleaned = clean_text(raw)
    assert cleaned == (
        "llutz, you understand what z3r0-0n3 wants? i thought bridging wasn't right"
    )
