"""Bounded dialogue state machine for the reporting agent (docs/01 §5, docs/07 §3, AGENTS rule 7).

The state machine alone decides the flow: at most 6 questions (docs/07 §3), one per turn, in
the fixed priority order below, skipping any question the initial report already answered.
An LLM may only *phrase* a question (`phraser` hook); it never picks questions or reads
answers. Answers are parsed with the rule-based lexicon; anything unclear becomes "unknown"
(asked once, never repeated). No medical advice is produced here (first-aid text comes only
from RAG, rule 6). Location is never parsed from speech (rule 5).

Output: `text_features()` -> the 19-d vector of docs/07 §3, and `to_incident_report()`.
Slots 15-18 (text severity probs) come from the text classifier; until it exists they default
to a uniform placeholder [ASSUMPTION] that must not be reported as a prediction.

Design choices [ASSUMPTION]:
  * Order: CONSCIOUS, BREATHING, BLEEDING, SPEAK, NUM_INJURED, MECHANISM.
  * Initial-report cues skip a question only for clear, high-signal statements (unconscious,
    difficulty breathing, heavy bleeding). Plain "bleeding" does NOT skip Q_BLEEDING: the
    severity follow-up must still be asked (docs/01 §8).
  * If the person is known not to be conscious, Q_SPEAK is skipped and recorded can_speak=0,
    known (an unresponsive person cannot speak clearly).
  * The dialogue schema's answer_value is yes/no/unknown, so bleeding level is derived from
    the raw answer: no -> 0, yes + heavy wording -> 1.0, yes otherwise -> 0.5.
    Q_NUM_INJURED stores answer_value "yes" if a number was understood, else "unknown".
  * Unknown num_injured defaults to 1 (a report implies at least one casualty).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.dialogue import lexicon as lx
from backend.schemas import GPS, INJURY_TYPES, TEXT_FEATURE_DIM, DialogueTurn, IncidentReport

QUESTION_ORDER = ("Q_CONSCIOUS", "Q_BREATHING", "Q_BLEEDING", "Q_SPEAK", "Q_NUM_INJURED",
                  "Q_MECHANISM")
FROM_REPORT = "(from initial report)"
UNIFORM_PROBS = [0.25, 0.25, 0.25, 0.25]  # placeholder until the text classifier exists
COUNT_NOUNS = ("injured", "people", "persons", "victims", "hurt", "घायल", "ಗಾಯ")

Phraser = Callable[[str, str, str], str]  # (q_id, language, template_text) -> wording


@dataclass
class Answer:
    value: str                    # yes | no | unknown
    raw: str
    bleeding_level: Optional[float] = None
    count: Optional[int] = None


@dataclass
class DialogueManager:
    language: str = "en"
    phraser: Optional[Phraser] = None
    transcript: str = ""
    injury_type: str = "other"
    answers: dict[str, Answer] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)      # q_ids put to the reporter, in order
    _pending: Optional[str] = None
    _question_text: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.language not in ("en", "hi", "kn"):
            raise ValueError(f"unsupported language {self.language!r}")

    # ---- start ------------------------------------------------------------------
    def start(self, initial_report: str) -> None:
        """Read the initial report: injury type and the few clear cues that skip questions."""
        self.transcript = initial_report
        low = initial_report.lower()
        self.injury_type = next((t for t, cues in lx.INJURY_CUES if any(c in low for c in cues)),
                                "other")
        if any(c in low for c in lx.CUES_UNCONSCIOUS):
            self.answers["Q_CONSCIOUS"] = Answer("no", FROM_REPORT)
        if any(c in low for c in lx.CUES_BREATHING):
            self.answers["Q_BREATHING"] = Answer("yes", FROM_REPORT)
        if any(c in low for c in lx.CUES_BLEEDING) and lx.is_heavy(low):
            self.answers["Q_BLEEDING"] = Answer("yes", FROM_REPORT, bleeding_level=1.0)
        if self.injury_type in ("road_accident", "fall") and any(
                c in low for c in lx.HIGH_ENERGY_CUES):
            self.answers["Q_MECHANISM"] = Answer("yes", FROM_REPORT)
        cnt = self._count_in_report(low)
        if cnt is not None:
            self.answers["Q_NUM_INJURED"] = Answer("yes", FROM_REPORT, count=cnt)

    @staticmethod
    def _count_in_report(low: str) -> Optional[int]:
        toks = lx.tokens(low)
        for i, t in enumerate(toks[:-1]):
            n = int(t) if t.isdigit() else lx.NUMBER_WORDS.get(t)
            if n is not None and toks[i + 1] in COUNT_NOUNS:
                return n
        return None

    # ---- flow -------------------------------------------------------------------
    def _needed(self, q_id: str) -> bool:
        if q_id in self.answers:
            return False
        if q_id == "Q_SPEAK" and self.answers.get("Q_CONSCIOUS", Answer("", "")).value == "no":
            return False
        return True

    def next_question(self) -> Optional[tuple[str, str]]:
        """(q_id, wording) for the next unanswered question, or None when the dialogue is done."""
        if self._pending is not None:
            return self._pending, self._question_text[self._pending]
        for q_id in QUESTION_ORDER:
            if self._needed(q_id):
                template = lx.QUESTIONS[q_id][self.language]
                text = self.phraser(q_id, self.language, template) if self.phraser else template
                self._pending, self._question_text[q_id] = q_id, text or template
                self.asked.append(q_id)
                return q_id, self._question_text[q_id]
        return None

    @property
    def done(self) -> bool:
        return self._pending is None and not any(self._needed(q) for q in QUESTION_ORDER)

    def answer(self, raw: str) -> Answer:
        """Record the reporter's reply to the pending question."""
        if self._pending is None:
            raise RuntimeError("no question is pending; call next_question() first")
        q_id, self._pending = self._pending, None
        if q_id == "Q_NUM_INJURED":
            n = lx.parse_count(raw)
            ans = Answer("unknown" if n is None else "yes", raw, count=n)
        else:
            ans = Answer(lx.parse_yes_no(raw), raw)
            if q_id == "Q_BLEEDING":
                if ans.value == "no":
                    ans.bleeding_level = 0.0
                elif ans.value == "yes" or lx.is_heavy(raw):  # "heavy" alone also means bleeding
                    ans.value = "yes"
                    ans.bleeding_level = 1.0 if lx.is_heavy(raw) else 0.5
        self.answers[q_id] = ans
        return ans

    # ---- outputs ----------------------------------------------------------------
    @property
    def num_injured(self) -> int:
        a = self.answers.get("Q_NUM_INJURED")
        return a.count if a and a.count is not None else 1

    def text_features(self, severity_probs: Optional[list[float]] = None) -> list[float]:
        """19-d vector, docs/07 §3. Unknown answer -> value 0 and known-flag 0."""
        f = [0.0] * TEXT_FEATURE_DIM
        for q_id, idx in (("Q_CONSCIOUS", 0), ("Q_BREATHING", 2), ("Q_SPEAK", 6)):
            a = self.answers.get(q_id)
            if a and a.value in ("yes", "no"):
                f[idx], f[idx + 1] = float(a.value == "yes"), 1.0
        if self.answers.get("Q_CONSCIOUS", Answer("", "")).value == "no" and (
                "Q_SPEAK" not in self.answers):
            f[6], f[7] = 0.0, 1.0                       # unresponsive -> cannot speak (inferred)
        b = self.answers.get("Q_BLEEDING")
        if b and b.bleeding_level is not None:
            f[4], f[5] = b.bleeding_level, 1.0
        f[8] = min(self.num_injured, 5) / 5
        m = self.answers.get("Q_MECHANISM")
        f[9] = float(bool(m and m.value == "yes"))
        f[10 + INJURY_TYPES.index(self.injury_type)] = 1.0
        probs = severity_probs if severity_probs is not None else UNIFORM_PROBS
        f[15:19] = [float(p) for p in probs]
        return f

    def turns(self) -> list[DialogueTurn]:
        """Dialogue log for IncidentReport: every question answered or read from the report."""
        return [DialogueTurn(q_id=q, question=lx.QUESTIONS[q][self.language],
                             answer_raw=self.answers[q].raw, answer_value=self.answers[q].value)
                for q in QUESTION_ORDER if q in self.answers]

    def to_incident_report(self, incident_id: str, timestamp: str, gps: GPS,
                           severity_probs: Optional[list[float]] = None) -> IncidentReport:
        """GPS is passed in from the device; it is never parsed from the transcript (rule 5)."""
        probs = severity_probs if severity_probs is not None else UNIFORM_PROBS
        return IncidentReport(
            incident_id=incident_id, timestamp=timestamp, gps=gps, language=self.language,
            transcript=self.transcript, dialogue=self.turns(),
            text_features=self.text_features(probs), text_severity_probs=list(probs),
            injury_type=self.injury_type, num_injured=self.num_injured)
