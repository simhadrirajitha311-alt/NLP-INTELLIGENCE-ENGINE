"""Small, inspectable text preprocessing utilities."""

import re

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Lowercase text and return word tokens, dropping punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _TOKEN_PATTERN.findall(text.lower())