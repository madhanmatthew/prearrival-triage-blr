"""Rule-based language resources for the dialogue state machine (docs/01 §5, docs/07 §3).

Question templates and answer/cue lexicons for en / hi / kn. The state machine, not an LLM,
decides which question comes next; templates are the fallback wording when no LLM phraser is
plugged in. Every yes/no question is worded so that "yes" is the feature-positive answer
(conscious / bleeding / can speak / high-energy mechanism), except Q_BREATHING, which asks
"breathing normally?" and is inverted by the state machine (stored "yes" = breathing problem).

Cues in the initial report are regexes matched on `fold()`ed text, so they cover the spellings
real Whisper output uses for code-mixed hi/kn speech (English loanwords in Devanagari/Kannada
script, e.g. ब्लीडिंग, ಆಕ್ಸಿಡೆಂಟ್). [ASSUMPTION] the variants were written from the 90 team
clips (scripts + large-v3 transcripts), so `backend/asr/eval_triage.py` on those clips is a dev
figure, not a held-out one.

[ASSUMPTION] The hi / kn wording and the small lexicons below are a first draft written for
the prototype. [TODO-VERIFY] a native speaker (Sankalp's team) must review every hi/kn string
and extend the lexicons from the recorded test set before any accuracy claim is made.
Unrecognised answers are never guessed: they become "unknown".
"""
from __future__ import annotations

import re
import unicodedata

QUESTIONS: dict[str, dict[str, str]] = {
    "Q_CONSCIOUS": {
        "en": "Is the person awake and responding to you?",
        "hi": "क्या व्यक्ति होश में है और आपकी बात का जवाब दे रहा है?",
        "kn": "ವ್ಯಕ್ತಿ ಎಚ್ಚರವಾಗಿದ್ದಾರೆಯೇ ಮತ್ತು ನಿಮಗೆ ಪ್ರತಿಕ್ರಿಯಿಸುತ್ತಿದ್ದಾರೆಯೇ?",
    },
    # Asked as "breathing normally?" (the IFRC check), NOT "difficulty breathing?": for an
    # unresponsive person "no difficulty" could mean "not breathing at all". The state machine
    # inverts the reply so the stored answer_value stays "yes = breathing problem".
    "Q_BREATHING": {
        "en": "Is the person breathing normally?",
        "hi": "क्या व्यक्ति ठीक से सांस ले रहा है?",
        "kn": "ವ್ಯಕ್ತಿ ಸರಿಯಾಗಿ ಉಸಿರಾಡುತ್ತಿದ್ದಾರೆಯೇ?",
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
              "haan", "han", "ha", "haa", "ji", "हां", "हा", "जी",
              "haudu", "houdu", "ಹೌದು", "ಹೂಂ", "ಹಾಂ", "ಹಾ"}
NO_TOKENS = {"no", "nope", "not", "never", "cant", "cannot", "wont", "n",
             "nahi", "nahin", "नहीं", "नही", "illa", "ಇಲ್ಲ", "ಇಲ್ಲಾ"}
# Kannada negates the verb with a suffix (ಉಸಿರಾಡ್ತಿಲ್ಲ "is not breathing", romanised agtilla);
# ಎಲ್ಲ / ಎಲ್ಲಾ ("all") end the same way and are excluded.
KN_NEG_SUFFIXES = ("ಲ್ಲ", "ಲ್ಲಾ", "illa")
KN_NOT_NEG = ("ಎಲ್ಲ", "ಎಲ್ಲಾ", "ella", "yella")
HEAVY_WORDS = ("heavy", "heavily", "a lot", "lots", "profuse", "severe", "gushing", "pooling",
               "soaked", "badly", "will not stop", "wont stop", "bahut", "zyada", "ज्यादा", "बहुत", "tumba", "thumba", "ತುಂಬ", "ತುಮ್ಬ")

_FOLD = str.maketrans({"‌": None, "‍": None,   # ZWNJ / ZWJ (ಬೈಕ್‌ನಿಂದ)
                       "़": None,                   # Devanagari nukta (ज़ -> ज)
                       "ँ": "ं"})              # chandrabindu -> anusvara (हाँ -> हां)


def fold(text: str) -> str:
    """NFC, lower case, and drop spelling marks Whisper uses inconsistently in hi/kn."""
    return unicodedata.normalize("NFC", text).lower().translate(_FOLD)

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "ondu": 1, "eradu": 2, "mooru": 3, "naalku": 4, "aidu": 5, "ibbaru": 2,
    "ಇಬ್ಬರು": 2, "ಇಬ್ರು": 2, "ಮೂವರು": 3,
    "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲ್ಕು": 4, "ಐದು": 5,
}

_SPLIT = re.compile(r"[\s,.;:!?।\"()\-]+")


def tokens(text: str) -> list[str]:
    """Whitespace/punctuation split (not \\w: Indic combining marks are not \\w in `re`).
    Apostrophes are removed so "can't" -> "cant"."""
    return [t for t in _SPLIT.split(fold(text).strip().replace("'", "")) if t]


def _is_kn_negated(tok: str) -> bool:
    return tok.endswith(KN_NEG_SUFFIXES) and tok not in KN_NOT_NEG and len(tok) > 3


def parse_yes_no(text: str) -> str:
    """'yes' | 'no' | 'unknown'. Mixed or unrecognised answers are 'unknown' (never guessed)."""
    low = fold(text)
    if any(p in low for p in UNKNOWN_PHRASES):
        return "unknown"
    toks = set(tokens(low))
    yes = bool(toks & YES_TOKENS)
    no = bool(toks & NO_TOKENS) or any(_is_kn_negated(t) for t in toks)
    if yes == no:
        return "unknown"
    return "yes" if yes else "no"


def is_heavy(text: str) -> bool:
    low = fold(text)
    return any(w in low for w in HEAVY_WORDS)


def has_cue(text: str, patterns: tuple[str, ...]) -> bool:
    """True if any cue regex matches the folded text (`\\S` instead of `\\w`, see tokens())."""
    low = fold(text)
    return any(re.search(p, low) for p in patterns)


def parse_count(text: str) -> int | None:
    """First number in the answer (digits or a number word); None if none found."""
    for t in tokens(text):
        if t.isdigit():
            return int(t)
        if t in NUMBER_WORDS:
            return NUMBER_WORDS[t]
    return None


# ---- cues in the initial report (used only to skip questions, never to invent answers) ----
# Regexes on fold()ed text: write them without nukta, chandrabindu or ZWNJ, and use \S, not \w.
# "Breathing problem" cues need an explicit negation/difficulty word: a bare "breathing" word
# (ಉಸಿರಾಡ್ತಿದಾಳೆ "is breathing", धीरे सांस ले रही "breathing slowly") must not skip the question.
CUES_UNCONSCIOUS = (
    "unconscious", "unresponsive", "not responding", "not answering", "not waking",
    "passed out", "fainted",
    r"उठ\s+नहीं", r"ಎದ್ದೇಳ\S*ಲ್ಲ",
    "behosh", "बेहोश", r"अनक\S*शियस", r"(रि|रे)स्प\S*\s+(\S+\s+){0,2}नहीं",
    "ಪ್ರಜ್ಞೆ ಇಲ್ಲ", "ಪ್ರಜ್ಞಾಹೀನ", "ಮೂರ್[ಛಚ]ೆ", r"ಅನ್ಕಾ\S*ಸ್", r"(ರೆ|ರಿ)ಸ್ಪಾ\S*\s+\S*ಲ್ಲ")
CUES_BREATHING = (
    "not breathing", "can't breathe", "cant breathe", "cannot breathe",
    "difficulty breathing", "trouble breathing", "gasping",
    "saans nahi", r"सां?स\s+(\S+\s+)?नहीं", r"सां?स\s+लेने\s+में\s+(तकलीफ|दिक्कत|परेशानी)",
    r"ब्रीद\S*\s+(\S+\s+){0,3}नहीं",
    r"ಉಸ\S*(ಲ್ಲ|\s+\S*ಲ್ಲ)", r"ಉಸ\S*\s+ತೊಂದರೆ")
CUES_BLEEDING = ("bleeding", "blood", "khoon", "खून", "रक्त", r"ब्?ल[ीिे]ड", r"ब्?लड",
                 "ರಕ್ತ", r"ಬ\S{0,3}ಡಿಂಗ", "ಬ್ಲಡ್")

# First match wins. Explicit accident words come before "fall" (ಆಕ್ಸಿಡೆಂಟ್ ... ಬಿದ್ದಿದಾನೆ "accident,
# lying on the road" is a road accident); bare vehicle words come after it ("fell from her
# bike" stays a fall, as in the English script).
_ROAD_STRONG = ("accident", "crash", "collision", r"\bhit\b", "turned over",
                r"[एअ]क्स\S{0,2}ड\S*ंट", "टक्कर", "टकरा", "दुर्घटना", r"हिट\s+किया", "पलट",
                "ಅಪಘಾತ", "ಡಿಕ್ಕಿ", r"[ಆಎ]ಕ್ಸಿ[ಡದ]ೆಂಟ", "ಹಿಟ್", "ಪಲ್ಟಿ")
_ROAD_WEAK = ("bike", "car ", "truck", "lorry", "bus", "scooter",
              "बाइक", "स्कूटर", "लॉरी", "लोरी", r"(^|\s)ट्रक", "ಬೈಕ್", "ಸ್ಕೂಟ", "ಲಾರಿ")
INJURY_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("cardiac_medical", ("chest pain", "heart", "seizure", r"\bfits?\b", "stroke", "collapsed",
                         "दिल", "चेस्ट", "फिट्स", "ಹೃದಯ", "ಚ[ೆೇ]ಸ್ಟ್", "ಫಿಟ್ಸ್", "ಕುಸಿದ")),
    ("burn", ("burn", "fire", "scald", "आग", r"जल\s+ग", "ಸುಟ್ಟ", "ಬ[ೆೇ]ಂಕಿ")),
    ("road_accident", _ROAD_STRONG),
    ("fall", ("fell", "fall", "fallen", "roof", "ladder", "storey", "slip",
              "गिर", "स्लिप", "ಬಿದ್?[ದಧ]", "ಸ್ಲಿಪ್")),
    ("road_accident", _ROAD_WEAK),
]
HIGH_ENERGY_CUES = ("high speed", "highway", "truck", "lorry", "bus", "storey", "floor",
                    "roof", "height", "ladder", "flyover",
                    "ऊंचाई", "हाईवे", r"फ्ल[ोॉ]र", "लॉरी", "लोरी", r"(^|\s)ट्रक", r"बस\s*ने",
                    "ಎತ್ತರ", "ಹೈವೇ", "ಫ್ಲೋರ್", "ಲಾರಿ", "ಬಸ್")
COUNT_NOUNS = ("injured", "people", "persons", "victims", "hurt", "passenger",
               "घायल", "पैसेंजर", "ಗಾಯ", "ಪ್ಯಾಸೆಂಜರ್")
