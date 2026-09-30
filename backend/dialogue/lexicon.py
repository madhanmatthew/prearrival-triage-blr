"""Rule-based language resources for the dialogue state machine (docs/01 §5, docs/07 §3).

Question templates and answer/cue lexicons for en / hi / kn. The state machine, not an LLM,
decides which question comes next; templates are the fallback wording when no LLM phraser is
plugged in. Every yes/no question is worded so that "yes" is the feature-positive answer
(conscious / difficulty breathing / bleeding / can speak / high-energy mechanism).

[ASSUMPTION] The hi / kn wording and the small lexicons below are a first draft written for
the prototype. [TODO-VERIFY] a native speaker (Sankalp's team) must review every hi/kn string
and extend the lexicons from the recorded test set before any accuracy claim is made.
Unrecognised answers are never guessed: they become "unknown".
"""
from __future__ import annotations

import re

QUESTIONS: dict[str, dict[str, str]] = {
    "Q_CONSCIOUS": {
        "en": "Is the person awake and responding to you?",
        "hi": "क्या व्यक्ति होश में है और आपकी बात का जवाब दे रहा है?",
        "kn": "ವ್ಯಕ್ತಿ ಎಚ್ಚರವಾಗಿದ್ದಾರೆಯೇ ಮತ್ತು ನಿಮಗೆ ಪ್ರತಿಕ್ರಿಯಿಸುತ್ತಿದ್ದಾರೆಯೇ?",
    },
    "Q_BREATHING": {
        "en": "Is the person having difficulty breathing?",
        "hi": "क्या व्यक्ति को सांस लेने में तकलीफ हो रही है?",
        "kn": "ವ್ಯಕ್ತಿಗೆ ಉಸಿರಾಡಲು ತೊಂದರೆ ಆಗುತ್ತಿದೆಯೇ?",
    },
    "Q_BLEEDING": {
        "en": "Is there visible bleeding? If yes, is it heavy?",
        "hi": "क्या खून बह रहा है? अगर हाँ, तो क्या बहुत ज़्यादा?",
        "kn": "ರಕ್ತಸ್ರಾವ ಕಾಣುತ್ತಿದೆಯೇ? ಹೌದಾದರೆ, ತುಂಬಾ ಇದೆಯೇ?",
    },
    "Q_SPEAK": {
        "en": "Can the person speak clearly?",
        "hi": "क्या व्यक्ति साफ़ बोल पा रहा है?",
        "kn": "ವ್ಯಕ್ತಿ ಸ್ಪಷ್ಟವಾಗಿ ಮಾತನಾಡಬಲ್ಲರೇ?",
    },
    "Q_NUM_INJURED": {
        "en": "How many people are injured?",
        "hi": "कितने लोग घायल हैं?",
        "kn": "ಎಷ್ಟು ಜನರಿಗೆ ಗಾಯವಾಗಿದೆ?",
    },
    "Q_MECHANISM": {
        "en": "Was it a high-speed crash, a fall from a height, or a similar strong impact?",
        "hi": "क्या यह तेज़ रफ़्तार टक्कर, ऊँचाई से गिरना या ऐसी कोई ज़ोरदार चोट थी?",
        "kn": "ಇದು ವೇಗದ ಅಪಘಾತ, ಎತ್ತರದಿಂದ ಬಿದ್ದದ್ದು ಅಥವಾ ಅಂತಹ ಬಲವಾದ ಹೊಡೆತವೇ?",
    },
}

# ---- answer parsing ----------------------------------------------------------------
UNKNOWN_PHRASES = ("don't know", "dont know", "not sure", "cannot tell", "can't tell", "unknown",
                   "pata nahi", "pata nahin", "पता नहीं", "gottilla", "gotilla", "ಗೊತ್ತಿಲ್ಲ")
YES_TOKENS = {"yes", "yeah", "yep", "yup", "ya", "y", "sure", "correct", "true",
              "haan", "han", "ha", "haa", "ji", "हाँ", "हां", "हा", "जी",
              "haudu", "houdu", "ಹೌದು"}
NO_TOKENS = {"no", "nope", "not", "never", "cant", "cannot", "wont", "n",
             "nahi", "nahin", "नहीं", "नही", "illa", "ಇಲ್ಲ"}
HEAVY_WORDS = ("heavy", "heavily", "a lot", "lots", "profuse", "severe", "gushing", "pooling",
               "soaked", "bahut", "zyada", "ज़्यादा", "ज्यादा", "बहुत", "tumba", "thumba", "ತುಂಬಾ")

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "ondu": 1, "eradu": 2, "mooru": 3, "naalku": 4, "aidu": 5,
    "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲ್ಕು": 4, "ಐದು": 5,
}

_SPLIT = re.compile(r"[\s,.;:!?।\"()\-]+")


def tokens(text: str) -> list[str]:
    """Whitespace/punctuation split (not \\w: Indic combining marks are not \\w in `re`).
    Apostrophes are removed so "can't" -> "cant"."""
    return [t for t in _SPLIT.split(text.lower().strip().replace("'", "")) if t]


def parse_yes_no(text: str) -> str:
    """'yes' | 'no' | 'unknown'. Mixed or unrecognised answers are 'unknown' (never guessed)."""
    low = text.lower()
    if any(p in low for p in UNKNOWN_PHRASES):
        return "unknown"
    toks = set(tokens(low))
    yes, no = bool(toks & YES_TOKENS), bool(toks & NO_TOKENS)
    if yes == no:
        return "unknown"
    return "yes" if yes else "no"


def is_heavy(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in HEAVY_WORDS)


def parse_count(text: str) -> int | None:
    """First number in the answer (digits or a number word); None if none found."""
    for t in tokens(text):
        if t.isdigit():
            return int(t)
        if t in NUMBER_WORDS:
            return NUMBER_WORDS[t]
    return None


# ---- cues in the initial report (used only to skip questions, never to invent answers) ----
CUES_UNCONSCIOUS = ("unconscious", "unresponsive", "not responding", "not waking", "passed out",
                    "behosh", "बेहोश", "ಪ್ರಜ್ಞೆ ಇಲ್ಲ", "ಪ್ರಜ್ಞಾಹೀನ")
CUES_BREATHING = ("not breathing", "can't breathe", "cant breathe", "cannot breathe",
                  "difficulty breathing", "trouble breathing", "gasping", "saans nahi",
                  "सांस नहीं", "सांस लेने में", "ಉಸಿರಾಟ", "ಉಸಿರಾಡಲು")
CUES_BLEEDING = ("bleeding", "blood", "khoon", "खून", "रक्त", "ರಕ್ತ")

INJURY_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("cardiac_medical", ("chest pain", "heart", "seizure", "fits", "stroke", "collapsed",
                         "दिल", "ಹೃದಯ")),
    ("burn", ("burn", "fire", "scald", "आग", "ಸುಟ್ಟ", "ಬೆಂಕಿ")),
    ("fall", ("fell", "fall", "fallen", "roof", "ladder", "storey", "गिर", "ಬಿದ್ದ")),
    ("road_accident", ("accident", "crash", "collision", "hit by", "bike", "car ", "truck",
                       "lorry", "bus", "scooter", "टक्कर", "दुर्घटना", "ಅಪಘಾತ", "ಡಿಕ್ಕಿ")),
]
HIGH_ENERGY_CUES = ("high speed", "highway", "truck", "lorry", "bus", "storey", "floor",
                    "roof", "height", "ladder", "flyover", "ऊँचाई", "ऊंचाई", "ಎತ್ತರ")
