import pytest
from injestion.preprocessing import normalize_text, remove_boilerplate

def test_normalize_text():
    sample_text = (
        "“Hello\u00a0\u00a0World!”\n\n"
        "This\tis   a   test\u200bstring.\n"
        "Full-width characters：ＡＢＣ１２３\n"
        "Multiple     spaces,\t tabs,\n"
        "and newlines.\n\n\n"
        "   Trailing and leading spaces.   "
    )

    text_n = None
    text_n = normalize_text(sample_text)
    assert text_n is not None
    assert isinstance(text_n, str)
    assert text_n != sample_text


def test_remove_boilerplate():
    BOILERPLATE_PATTERNS = [
        r"©\s*\d{4}",
        r"privacy policy",
        r"terms of service",
    ]

    boilerplate_text = (
        "© 2024 Example Corp. All rights reserved.\n\n"
        "Welcome to our documentation.\n"
        "Please read our Privacy Policy before continuing.\n"
        "By using this site, you agree to the Terms of Service.\n\n"
    )

    text_rb = None
    text_rb = remove_boilerplate(boilerplate_text, BOILERPLATE_PATTERNS)
    assert text_rb is not None
    assert isinstance(text_rb, str)
    assert text_rb != boilerplate_text


