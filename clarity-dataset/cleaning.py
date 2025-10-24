#a rough attempt to clean the dataset
from __future__ import annotations
import re
from logging import fatal
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

    # tidy spaces before punctuation like "word,"
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
        "Ummm... I mean, it's kind of tricky.",
        "Actually, we can, right?",
    ]
    for s in samples:
        print(u"IN : " + s)
        print(u"OUT: " + remove_fillers(s))
        print("---")

#remove names of interviewee
#direct address
_DIRECT_ADDRESS = re.compile(r"(?i)(^|(?<=\W))(?:sir|ma'am|madam)\b\s*(?:[.,;:!?—–-])?\s*")

#honoured title: Mr./Mister/Ms./Mrs./Madam + President
_HONOURED_TITLE = re.compile(r"(?i)(^|(?<=\W))(?:mr|mister|ms|mrs|madam)\.?\s+president\b\s*(?:[.,;:!?—–-])?\s*")

def _name_token(tok: str) -> str:
    tok = tok.strip()
    if not tok:
        return ""
    if re.fullmatch(r"[A-Za-z]\.?", tok):
        return re.escape(tok[0]) + r"\.?"

    esc = re.escape(tok)
    return esc

def _core_fullname(name: str) -> str:
    tokens = re.split(r"[\s\u00A0]+", name.strip())
    parts = [_name_token(t) for t in tokens if t]
    return r"(?:%s)" % ("".join(parts)) if parts else ""

#maching by name list
def _compile_president_name(names: List[str], aggressive_lastname: bool = False):
    cores = [_core_fullname(n) for n in names or [] if n and n.strip()]
    lastnames: List[str] = []
    for n in names or []:
        parts = [p for p in re.split(r"[\s\u00A0]+", n.strip()) if p]
        if parts:
            lastnames.append(parts[-1])

    rx_full_alt = r"(?:" + "|".join(cores) + r")" if cores else None
    rx_last_alt = (r"(?:" + "|".join(sorted(set(map(re.escape, lastnames)), key=len, reverse=True)) + r")"
                   if lastnames else None)

    patterns = []

    if rx_full_alt:
        patterns.append(re.compile(
                r"(?i)(^|(?<=\W))president\s+" + rx_full_alt +
                r"(?:\s*(?:'s|’s))?\s*(?:[.,;:!?—–-])?\s*"))

        patterns.append(re.compile(r"(?i)(^|(?<=\W))" + rx_full_alt +
                r"(?:\s*(?:'s|’s))?\s*(?:[.,;:!?—–-])?\s*"))

    if rx_last_alt:
        patterns.append(re.compile(r"(?i)(^|(?<=\W))president\s+" + rx_last_alt +
                r"(?:\s*(?:'s|’s))?\s*(?:[.,;:!?—–-])?\s*"))

        patterns.append(re.compile(r"(?i)(^|(?<=\W))(?:mr|mister|ms|mrs)\.?\s+" + rx_last_alt +
                r"(?:\s*(?:'s|’s))?\s*(?:[.,;:!?—–-])?\s*"))

        if aggressive_lastname:
            patterns.append( re.compile(
                    r"(?i)(^|(?<=\W))" + rx_last_alt +
                    r"(?:\s*(?:'s|’s))?\s*(?:[.,;:!?—–-])?\s*"))

    return patterns

def remove_presidential_mentions(text: str, president_names: List[str], aggressive_lastname: bool = False, ) -> str:
    if not text:
        return text

    out = _DIRECT_ADDRESS.sub(r"\1", text)
    out = _HONOURED_TITLE.sub(r"\1", out)

    for rx in _compile_president_name(president_names, aggressive_lastname):
        out = rx.sub(r"\1", out)

    out = re.sub(r"\s+([,;:!?])", r"\1", out)
    out = re.sub(r"(?m)^[,;:]\s*", "", out)
    out = re.sub(r"[ \t\r\f\v]+", " ", out).strip()
    return out

def remove_names(text: str, president_names: List[str], aggressive_lastname=False) -> str:
    text = remove_presidential_mentions(text, president_names, aggressive_lastname = aggressive_lastname)
    return text

if __name__ == "__main__":
    names = [
        "Joe Biden", "Joseph R. Biden", "Donald Trump", "Donald J. Trump",
        "Barack Obama", "Bill Clinton", "George W. Bush", "George H. W. Bush",
        "Ronald Reagan", "Jimmy Carter"
    ]

    samples = [
        "Q. And on TikTok, sir, if you don't—if you don't mind: Do you expect the TikTok deal after the election?",
        "Mr. President, do you expect the TikTok deal after the election?",
        "President Biden, do you expect the TikTok deal after the election?",
        "Do you expect the TikTok deal, Mr Biden?",
        "We asked President Trump about this yesterday.",
        "As President Obama said, it could go quickly.",
        "It was Mr. Clinton's view.",
        "Ford is a car brand, not a person.",  # Bush / Ford 等如担心误删，可保持 aggressive_lastname=False
    ]

    for s in samples:
        print("IN :", s)
        print("OUT:", remove_names(s, names, aggressive_lastname=False))
        print("---")