"""Tests for the bounded dialogue state machine (docs/01 §5, docs/07 §3)."""
import pytest

from backend.dialogue import lexicon as lx
from backend.dialogue.state_machine import QUESTION_ORDER, DialogueManager
from backend.schemas import GPS, TEXT_FEATURE_DIM


def run(report, replies, language="en"):
    dm = DialogueManager(language=language)
    dm.start(report)
    replies = list(replies)
    asked = []
    while (q := dm.next_question()) is not None:
        asked.append(q[0])
        dm.answer(replies.pop(0) if replies else "don't know")
    return dm, asked


@pytest.mark.parametrize("text,expected", [
    ("yes", "yes"), ("Yes he is", "yes"), ("no", "no"), ("not conscious", "no"),
    ("can't", "no"), ("I don't know", "unknown"), ("maybe", "unknown"), ("", "unknown"),
    ("yes no", "unknown"), ("haan", "yes"), ("नहीं", "no"), ("हाँ", "yes"),
    ("haudu", "yes"), ("illa", "no"), ("ಹೌದು", "yes"), ("ಇಲ್ಲ", "no"),
])
def test_parse_yes_no(text, expected):
    assert lx.parse_yes_no(text) == expected


@pytest.mark.parametrize("text,n", [("two", 2), ("3 people", 3), ("ek", 1), ("ಎರಡು", 2),
                                    ("many", None), ("", None)])
def test_parse_count(text, n):
    assert lx.parse_count(text) == n


def test_all_questions_have_all_languages():
    for q in QUESTION_ORDER:
        assert set(lx.QUESTIONS[q]) == {"en", "hi", "kn"}
        assert all(lx.QUESTIONS[q].values())


def test_flow_bounded_and_ordered_no_repeats():
    dm, asked = run("someone is hurt", [])          # nothing answerable: all unknown
    assert asked == list(QUESTION_ORDER)             # <= 6, in priority order, each once
    assert dm.done and len(dm.turns()) == 6


def test_one_question_at_a_time_and_idempotent_next():
    dm = DialogueManager()
    dm.start("accident")
    q1 = dm.next_question()
    assert dm.next_question() == q1                  # same pending question, not a new one
    dm.answer("yes")
    assert dm.next_question()[0] != q1[0]


def test_answer_without_question_raises():
    dm = DialogueManager()
    dm.start("x")
    with pytest.raises(RuntimeError):
        dm.answer("yes")


def test_bad_language_rejected():
    with pytest.raises(ValueError):
        DialogueManager(language="fr")


def test_report_cues_skip_questions():
    dm, asked = run("Bike accident on the highway, 2 injured, unconscious and not breathing, "
                    "bleeding heavily", [])
    assert asked == []                               # everything clearly stated
    f = dm.text_features()
    assert f[0:2] == [0.0, 1.0] and f[2:4] == [1.0, 1.0] and f[4:6] == [1.0, 1.0]
    assert f[6:8] == [0.0, 1.0]                      # inferred: unconscious cannot speak
    assert f[8] == pytest.approx(0.4) and f[9] == 1.0


def test_plain_bleeding_still_asks_severity_followup():
    """docs/01 §8: accident with visible bleeding must ask about bleeding severity."""
    _, asked = run("road accident, he is bleeding", [])
    assert "Q_BLEEDING" in asked


def test_unconscious_skips_can_speak():
    _, asked = run("man collapsed, unresponsive", [])
    assert "Q_SPEAK" not in asked and "Q_CONSCIOUS" not in asked


def test_bleeding_levels():
    dm, _ = run("hurt", ["yes", "no", "no heavy bleeding? yes it is heavy", "yes", "1", "no"])
    # Q_BLEEDING reply is mixed yes/no wording but heavy -> level 1.0
    assert dm.answers["Q_BLEEDING"].bleeding_level == 1.0
    dm, _ = run("hurt", ["yes", "no", "yes a little", "yes", "1", "no"])
    assert dm.answers["Q_BLEEDING"].bleeding_level == 0.5
    dm, _ = run("hurt", ["yes", "no", "no", "yes", "1", "no"])
    assert dm.answers["Q_BLEEDING"].bleeding_level == 0.0
    assert dm.text_features()[4:6] == [0.0, 1.0]     # "no bleeding" is a known 0, not unknown


def test_unknown_answers_zero_value_and_flag():
    dm, _ = run("someone is hurt", [])
    f = dm.text_features()
    assert len(f) == TEXT_FEATURE_DIM
    assert f[0:8] == [0.0] * 8 and f[9] == 0.0
    assert f[8] == pytest.approx(0.2)                # unknown count defaults to 1


def test_injury_type_one_hot_and_num_injured_cap():
    dm, _ = run("fell from the roof", ["yes", "no", "no", "yes", "9", "yes"])
    f = dm.text_features()
    assert f[10:15] == [0.0, 1.0, 0.0, 0.0, 0.0] and f[8] == 1.0   # capped at 5/5
    assert dm.num_injured == 9
    dm, _ = run("chest pain, seizure", [])
    assert dm.text_features()[10:15] == [0.0, 0.0, 0.0, 1.0, 0.0]
    dm, _ = run("weird", [])
    assert dm.text_features()[10:15] == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_prob_slots_and_placeholder():
    dm, _ = run("hurt", [])
    assert dm.text_features()[15:19] == [0.25] * 4
    assert dm.text_features([0.1, 0.2, 0.3, 0.4])[15:19] == [0.1, 0.2, 0.3, 0.4]


def test_incident_report_validates_and_keeps_gps_from_device():
    dm, _ = run("Car accident, 2 injured", ["yes", "no", "yes heavy", "yes", "2", "yes"])
    rep = dm.to_incident_report("INC-20260930-0001", "2026-09-30T10:00:00Z",
                                GPS(lat=12.9166, lon=77.6231), [0.1, 0.3, 0.4, 0.2])
    assert rep.gps.lat == 12.9166 and rep.injury_type == "road_accident"
    assert len(rep.text_features) == TEXT_FEATURE_DIM and len(rep.dialogue) <= 6
    assert rep.num_injured == 2


def test_hindi_and_kannada_flow():
    for lang, yes, no in (("hi", "हाँ", "नहीं"), ("kn", "ಹೌದು", "ಇಲ್ಲ")):
        dm, asked = run("hurt", [yes, no, no, yes, "ek" if lang == "hi" else "ondu", no], lang)
        assert asked == list(QUESTION_ORDER)
        assert dm.text_features()[0:4] == [1.0, 1.0, 0.0, 1.0]


def test_phraser_only_rewords_never_changes_flow():
    dm = DialogueManager(phraser=lambda q, lang, t: f"PLEASE: {t}")
    dm.start("hurt")
    qid, text = dm.next_question()
    assert qid == "Q_CONSCIOUS" and text.startswith("PLEASE: ")
    dm2 = DialogueManager(phraser=lambda q, lang, t: "")   # empty output falls back to template
    dm2.start("hurt")
    assert dm2.next_question()[1] == lx.QUESTIONS["Q_CONSCIOUS"]["en"]
