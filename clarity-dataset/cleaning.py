#a rough attempt to clean the dataset
from __future__ import annotations
import re
from typing import Dict, Optional, Tuple, List

#filler words and phrases
FILLER_TOKENS = ["um", "uh", "er","ah", "eh","hm","mm",]
FILLER_PHRASES = ["you know", "i mean", "kind of", "sort of",]
BOUNDARY_FILLERS = ["well", "so", "anyway", "anyways", "okay", "ok", "right", "look", "like", "basically", "actually", "literally",]

def _normalize(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"([!?.,:;])\1+", r"\1", text)

    # tidy spaces before punctuation like "word ,"
    text = re.sub(r"\s+([,;:!?])", r"\1", text)

    # drop leading punctuation left behind after deletions
    text = re.sub(r"(?m)^[,;:]\s*", "", text)
    return text.strip()

def _elongate_token(tok):
    parts = []
    for ch in tok:
        parts.append(re.escape(ch) + "+")
    return "".join(parts)

def _compile_basic_patterns():
    # tokens：\b(?:u+m+|u+h+|h+m+m+|...)\b
    token_patterns = ["".join(re.escape(ch) + "+" for ch in t) for t in FILLER_TOKENS]
    RX_TOKENS = re.compile(r"\b" + "|".join(token_patterns) + r"\b", re.IGNORECASE)

    # phrases like: \bi\s+mean\b | \bkind\s+of\b  (robust to 1+ spaces)
    phrase_patterns = []
    for p in FILLER_PHRASES:
        parts = [re.escape(w) for w in p.split()]  # <-- key change
        phrase_patterns.append(r"\b" + r"\s+".join(parts) + r"\b")
    RX_PHRASES = re.compile("|".join(phrase_patterns), re.IGNORECASE)

    token_alt = "(?:" + "|".join(token_patterns) + ")"
    RX_LEADING_TOKENS = re.compile(
        r"(?im)^\s*" + token_alt + r"(?:\s*(?:\.\.\.|\.{2,}|…|[.,;:!?—–-]))*\s*"
    )
    return RX_TOKENS, RX_PHRASES, RX_LEADING_TOKENS

def _compile_boundary_patterns():
    r"""
    1) At start of sentence： ^\s*(word)\s*[,;:—–-]
    2) after punctuation： ([.!?:;,\)\]\}]\s*)(word)\s*[,;:—–-]
    """
    words = "(?:" + "|".join([re.escape(w) for w in BOUNDARY_FILLERS]) + ")"

    # start-of-sentence (allow . too here)
    RX_START = re.compile(r"(?im)^[ \t]*\b" + words + r"\b[ \t]*[.,;:!?\-—–][ \t]*")

    # after punctuation (require punctuation BEFORE the word; do NOT allow '.' AFTER)
    RX_AFTER = re.compile(
        r"(?i)([.!?:;,)\]}]\s*)\b" + words + r"\b\s*[,;:!?\-—–]\s*"
    )
    return RX_START, RX_AFTER

_RX_TOKENS, _RX_PHRASES, _RX_LEADING_TOKENS = _compile_basic_patterns()
_RX_BOUNDARY_START, _RX_BOUNDARY_AFTER = _compile_boundary_patterns()

def remove_fillers(text: str) -> str:
    """
    - first PHRASES，then TOKENS
    - finally BOUNDARIES
    """
    if not text:
        return text

    text = _RX_LEADING_TOKENS.sub("", text)

    text = _RX_PHRASES.sub(" ", text)
    text = _RX_TOKENS.sub(" ", text)

    text = _RX_BOUNDARY_START.sub("", text)
    text = _RX_BOUNDARY_AFTER.sub(r"\1", text)
    return _normalize(text)


if __name__ == "__main__":
    samples = [
        "Well, we can start now.",
        "It works well.",
        "Look, this is the key.",
        "Please look at the figure.",
        "Um... I mean, it's kind of tricky.",
        "Actually, we can, right?",
    ]
    for s in samples:
        print(u"IN : " + s)
        print(u"OUT: " + remove_fillers(s))
        print("---")

#remove names of interviewee