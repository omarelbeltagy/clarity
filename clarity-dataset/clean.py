#a rough attempt to clean the dataset
from __future__ import annotations
import re
from logging import fatal
from typing import Dict, Optional, Tuple, List

from datasets import load_dataset

#filler words and phrases
FILLER_TOKENS = ["um", "uh", "ah", "eh","hm","mm", "oh"]
FILLER_PHRASES = ["what i'm trying to say is", "at the end of the day", "you know what i'm saying", "as some people said", "as some people say", "as i was saying", "thank you very much", "it's kind of like", "that being said", "having said that", "in other words",
                  "by the way", "the thing is", "the point is", "the fact is", "i mean like", "sort of like", "kind of like", "i'd say that", "i feel like", "to be honest", "to be frank", "to be clear", "to be fair" "as you know", "as i said", "as i say",
                  "like you said", "like you say", "like i said", "like i say", "in a sense", "if you will", "you know what", "more or less", "all right", "you know", "i mean", "kind of", "sort of", "you see", "i guess", "i suppose", "i think", "let's say",
                  "it's like", "and stuff", "and things", "or something", "thank you", "thanks",]
BOUNDARY_FILLERS = ["literally", "basically", "actually", "anyways", "anyway", "please", "listen", "though", "still", "right", "look", "like", "okay", "ok", "well", "now", "so",]

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

# token stretching: "ummm", "uhhh"...
def _elongate_token(tok):
    parts = []
    for ch in tok:
        parts.append(re.escape(ch) + "+")
    return "".join(parts)

def _compile_basic_patterns():
    token_patterns = [_elongate_token(t) for t in FILLER_TOKENS]
    #seperate token,only those before/after punctuation will be removed
    RX_TOKENS = re.compile(r"(?<!\w)(?:" + "|".join(token_patterns) + r")(?!\w)(?=[\s.,;:!?\-—–] | $)", re.IGNORECASE)

    # phrases like: "you konw", "i mean"...
    phrase_patterns = []
    for p in FILLER_PHRASES:
        parts = [re.escape(w) for w in p.split()]
        phrase_patterns.append(r"\b" + r"\s+".join(parts) + r"\b")
    RX_PHRASES = re.compile("(?:" + "|".join(phrase_patterns) + ")", re.IGNORECASE)

    # fillers at the beginning of sentences with punctuation
    token_alt = "(?:" + "|".join(token_patterns) + ")"
    phrase_alt = "(?:" + "|".join(phrase_patterns) + ")"
    RX_LEADING_TOKENS = re.compile(
        r"(?im)^\s*" + token_alt + r"(?:\s*(?:\.\.\.|\.{2,}|…|[.,;:!?\-—–]))*\s*"
    )
    RX_LEADING_PHRASES = re.compile(
        r"(?!m)^\s*" + phrase_alt + r"(?:\s*(?:\.\.|\.{2,}|…|[.,;:!?\-—–]))*\s*"
    )
    return RX_TOKENS, RX_PHRASES, RX_LEADING_TOKENS, RX_LEADING_PHRASES

def _compile_boundary_patterns():
    r"""
    1) At start of sentence： ^\s*(word)\s*[,;:—–-]
    2) after punctuation： ([.!?:;,\)\]\}]\s*)(word)\s*[,;:—–-]
    """
    word_alts = []
    for w in BOUNDARY_FILLERS:
        parts = [re.escape(p) for p in w.split()]
        word_alts.append(r"\b" + r"\s+".join(parts) + r"\b")
    words = "(?:" + "|".join(word_alts) + ")"

    # start-of-sentence (allow . too here)
    RX_START = re.compile(
        r"(?im)^[ \t]*(?:(?:"
        + words +
        r"[ \t]*(?:\.\.\.|\.{2,}|…|[.,;:!?\-—–])\s*"
        r")+)"
    )

    # after punctuation (require punctuation BEFORE the word; do NOT allow '.' AFTER)
    RX_AFTER = re.compile(
        r"(?i)([.!?:;,)\]}]\s*)(?:(?:"
        + words +
        r"\s*[,;:!?\-—–]\s*"
        r")+)"
    )
    return RX_START, RX_AFTER

_RX_TOKENS, _RX_PHRASES, _RX_LEADING_TOKENS, _RX_LEADING_PHRASES = _compile_basic_patterns()
_RX_BOUNDARY_START, _RX_BOUNDARY_AFTER = _compile_boundary_patterns()

# the content in []
_RX_BRACKETS = re.compile(r"\[[^]]*]")
def remove_brackets(text: str) -> str:
    if not text:
        return text
    return  _RX_BRACKETS.sub("", text)

def remove_fillers(text: str) -> str:
    """
    - first PHRASES，then TOKENS
    - finally BOUNDARIES
    """
    if not text:
        return text

    text = _RX_LEADING_TOKENS.sub("", text)
    text = _RX_LEADING_PHRASES.sub("", text)

    text = _RX_PHRASES.sub(" ", text)
    text = _RX_TOKENS.sub(" ", text)

    text = _RX_BOUNDARY_START.sub("", text)
    text = _RX_BOUNDARY_AFTER.sub(r"\1", text)

    return _normalize(text)


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

# president name list
def get_president_name_list():
    ds_train = load_dataset("ailsntua/QEvasion", split="train")
    ds_test = load_dataset("ailsntua/QEvasion", split="test")

    train_presidents = [p.strip() for p in ds_train["president"] if p is not None]
    test_presidents = [p.strip() for p in ds_test["president"] if p is not None]
    unique_presidents = list(set(train_presidents + test_presidents))
    return unique_presidents


# main method to clean the dataset
def clean_single_text(text:str, president_name : Optional[str]) -> str:
    if text is None:
        return ""

    name = [president_name] if president_name else []
    text = _normalize(text)

    text = remove_names(text, name, aggressive_lastname=False)
    text = remove_brackets(text)
    text = remove_fillers(text)
    text = _normalize(text)
    return text

def apply_clean_batch(batch):
    qs_list = batch["interview_question"]
    ans_list = batch["interview_answer"]
    pres_list = batch["president"]

    qs_clean = []
    ans_clean = []

    for q, a, p in zip(qs_list, ans_list, pres_list):
        qs_clean.append(clean_single_text(q, p))
        ans_clean.append(clean_single_text(a, p))

    return {
        "interview_question_clean": qs_clean,
        "interview_answer_clean": ans_clean,
    }



#tests for fillers, names and name list
if __name__ == "__main__":
    samples = [
        "Well, we can start now.[NY News]",
        "All right, it works well, okay, please.",
        "Look, this is the key.",
        "Thank you very much! Please look at the figure.",
        "Ummm... I mean, it's kind of tricky.",
        "Actually, we can, right?",
    ]
    for s in samples:
        print(u"IN : " + s)
        print(u"OUT: " + remove_fillers(s))
        print(u"OUT: " + clean_single_text(s,None))
        print("---")

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
        "Ford is a car brand, not a person.",
    ]

    for s in samples:
        print("IN :", s)
        print("OUT:", remove_names(s, names, aggressive_lastname=False))
        print("---")

    print(get_president_name_list())