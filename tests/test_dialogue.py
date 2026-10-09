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
        dm, asked = run("hurt", [yes, yes, no, yes, "ek" if lang == "hi" else "ondu", no], lang)
        assert asked == list(QUESTION_ORDER)
        assert dm.text_features()[0:4] == [1.0, 1.0, 0.0, 1.0]   # breathing normally -> no problem


@pytest.mark.parametrize("reply,stored", [("yes", "no"), ("no", "yes"), ("don't know", "unknown"),
                                          ("ಉಸಿರಾಡ್ತಿಲ್ಲ", "yes"), ("नहीं ले रहा", "yes")])
def test_breathing_asked_as_normal_and_inverted(reply, stored):
    """'Breathing normally?' -> stored value keeps 'yes = breathing problem' (RAG/fusion unchanged).
    Unresponsive + 'no' must therefore not look like 'no problem' (would route to recovery)."""
    assert "normally" in lx.QUESTIONS["Q_BREATHING"]["en"]
    dm = DialogueManager(language="kn" if reply.startswith("ಉ") else "en")
    dm.start("man collapsed, unresponsive")
    assert dm.next_question()[0] == "Q_BREATHING"
    assert dm.answer(reply).value == stored


@pytest.mark.parametrize("text,expected", [
    ("ಉಸಿರಾಡ್ತಿಲ್ಲ", "no"), ("agtilla", "no"), ("ಎಲ್ಲಾ", "unknown"), ("ಹೂಂ", "yes"),
    ("हाँ जी", "yes"), ("जी हां", "yes")])
def test_parse_yes_no_indic_forms(text, expected):
    assert lx.parse_yes_no(text) == expected


@pytest.mark.parametrize("lang", ["en", "hi", "kn"])
def test_tap_buttons_work_in_every_language(lang):
    assert [lx.parse_yes_no(t) for t in ("yes", "no", "don't know")] == ["yes", "no", "unknown"]


def test_fold_drops_inconsistent_marks():
    assert lx.fold("ज़्यादा हाँ") == "ज्यादा हां"
    assert lx.fold("ಬೈಕ್‌ನಿಂದ") == "ಬೈಕ್ನಿಂದ"
    assert lx.is_heavy("ज़्यादा")


def test_cue_patterns_compile_and_are_folded():
    import re
    pats = [*lx.CUES_UNCONSCIOUS, *lx.CUES_BREATHING, *lx.CUES_BLEEDING, *lx.HIGH_ENERGY_CUES,
            *(p for _, cues in lx.INJURY_CUES for p in cues)]
    for p in pats:
        re.compile(p)
        assert not set(p) & {"‌", "‍", "़", "ँ", "\x08"}, p


# Scripts of the team test clips (data/asr_test/reference.csv) and Whisper-style spellings.
@pytest.mark.parametrize("lang,report,injury,cues", [
    ("hi", "सिल्क बोर्ड के पास एक्सीडेंट हुआ है, एक आदमी रोड पर पड़ा है और वो सांस नहीं ले रहा।",
     "road_accident", {"Q_BREATHING"}),
    ("kn", "ಸಿಲ್ಕ್ ಬೋರ್ಡ್ ಹತ್ರ ಆಕ್ಸಿಡೆಂಟ್ ಆಗಿದೆ, ಒಬ್ಬ ಮನುಷ್ಯ ರೋಡ್ ಮೇಲೆ ಬಿದ್ದಿದಾನೆ, ಉಸಿರಾಡ್ತಿಲ್ಲ.",
     "road_accident", {"Q_BREATHING"}),
    ("kn", "ಕಾಲೇಜ್ ಫಂಕ್ಷನ್‌ನಲ್ಲಿ ಒಬ್ಬ ಸ್ಟೂಡೆಂಟ್ ಮೂರ್ಛೆ ಹೋಗಿದಾಳೆ, ತುಂಬಾ ನಿಧಾನವಾಗಿ ಉಸಿರಾಡ್ತಿದಾಳೆ.",
     "other", {"Q_CONSCIOUS"}),                       # "is breathing slowly" is not a no-breath cue
    ("hi", "कॉलेज फंक्शन में एक स्टूडेंट बेहोश हो गई है और बहुत धीरे सांस ले रही है।",
     "other", {"Q_CONSCIOUS"}),
    ("hi", "मेरी दादी किचन में बेहोश हो गईं, नाम बुला रहे हैं पर कोई रिस्पॉन्स नहीं दे रहीं।",
     "other", {"Q_CONSCIOUS"}),
    ("hi", "रेडर अनकॉंशियस है", "other", {"Q_CONSCIOUS"}),
    ("hi", "construction site पर किसी का हाथ कट गया है", "other", set()),   # not "ट्रक"
    ("hi", "कंस्ट्रक्शन साइट पर किसी का हाथ कट गया है", "other", set()),
    ("hi", "बेलंदूर के पास एक लेडी बाइक से गिर गई है और उसके लेग से बहुत ब्लीडिंग हो रही है।",
     "fall", {"Q_BLEEDING"}),
    ("kn", "ಒಬ್ಬ ಚಿಕ್ಕ ಹುಡುಗ ಸ್ಟೆಪ್ಸ್‌ನಿಂದ ಬಿದ್ಧಿದಾನೆ", "fall", set()),
    ("en", "My friend is having a fit, his whole body is shaking", "cardiac_medical", set()),
    ("en", "A woman fell from her bike and her leg is bleeding badly.", "fall", {"Q_BLEEDING"}),
    ("kn", "ಸಿಗ್ನಲ್ ಹತ್ರ ಒಂದು ಆಟೋ ಪಲ್ಟಿ ಆಗಿದೆ, ಇಬ್ರು ಪ್ಯಾಸೆಂಜರ್ಸ್‌ಗೆ ಗಾಯ ಆಗಿದೆ.",
     "road_accident", {"Q_NUM_INJURED"}),
    ("en", "he is responding fine", "other", set()),
    ("hi", "वो रिस्पॉन्स दे रहा है", "other", set()),
])
def test_code_mixed_report_cues(lang, report, injury, cues):
    dm = DialogueManager(language=lang)
    dm.start(report)
    assert dm.injury_type == injury
    assert set(dm.answers) == cues
    if "Q_NUM_INJURED" in cues:
        assert dm.answers["Q_NUM_INJURED"].count == 2


def test_phraser_only_rewords_never_changes_flow():
    dm = DialogueManager(phraser=lambda q, lang, t: f"PLEASE: {t}")
    dm.start("hurt")
    qid, text = dm.next_question()
    assert qid == "Q_CONSCIOUS" and text.startswith("PLEASE: ")
    dm2 = DialogueManager(phraser=lambda q, lang, t: "")   # empty output falls back to template
    dm2.start("hurt")
    assert dm2.next_question()[1] == lx.QUESTIONS["Q_CONSCIOUS"]["en"]
