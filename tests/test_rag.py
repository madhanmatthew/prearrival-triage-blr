"""Tests for backend/rag (toy corpus + bag-of-words embeddings; no model downloads)."""
import re
import zlib

import numpy as np
import pandas as pd
import pytest
from langchain_core.embeddings import Embeddings

from backend.rag import eval_rag as ev
from backend.rag import pipeline as rp
from backend.schemas import GPS, DialogueTurn, IncidentReport

TOY = {  # test fixture only, not a real corpus
    "bleeding.md": ("bleeding", "Press firmly on the wound with a clean cloth.\n"
                    "Keep pressing until help arrives.\nGive aspirin 300 mg for pain."),
    "burns.md": ("burns", "Cool the burn under cool running water.\n"
                 "Remove rings near the burn before swelling starts."),
}


class BowEmbeddings(Embeddings):
    """Hashed bag-of-words, L2-normalised (cosine = inner product)."""
    dim = 256

    def _v(self, t):
        v = np.zeros(self.dim)
        for w in re.findall(r"\w+", t.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


def _write(dirp, name, topic, body, header=True):
    meta = (f"---\ntitle: Toy {topic}\npublisher: TEST\nurl: http://example.test/{topic}\n"
            f"license: test-only\ntopic: {topic}\n---\n") if header else ""
    (dirp / name).write_text(meta + body, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    for name, (topic, body) in TOY.items():
        _write(d, name, topic, body)
    (d / "README.md").write_text("ignored", encoding="utf-8")
    return d


@pytest.fixture
def store(corpus, tmp_path):
    s, docs, chunks = rp.build(corpus, tmp_path / "idx", embeddings=BowEmbeddings(),
                               chunk_size=200, chunk_overlap=0)
    return s


def test_corpus_metadata_and_chunk_ids(corpus):
    docs = rp.load_corpus(corpus)
    assert [d.metadata["file"] for d in docs] == ["bleeding.md", "burns.md"]
    chunks = rp.split_corpus(docs, 80, 0)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)) and ids[0] == "bleeding.md#0"
    assert all(c.metadata["url"].startswith("http") for c in chunks)


def test_corpus_rejects_missing_header(tmp_path):
    _write(tmp_path, "x.md", "x", "text", header=False)
    with pytest.raises(ValueError, match="header"):
        rp.parse_corpus_file(tmp_path / "x.md")
    (tmp_path / "y.md").write_text("---\ntitle: a\n---\nbody", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata missing"):
        rp.parse_corpus_file(tmp_path / "y.md")


def test_index_save_load_roundtrip(store, tmp_path):
    s2 = rp.load_index(BowEmbeddings(), tmp_path / "idx")
    a = store.similarity_search_with_score("burn water", k=1)[0]
    b = s2.similarity_search_with_score("burn water", k=1)[0]
    assert a[0].metadata["chunk_id"] == b[0].metadata["chunk_id"]
    assert a[1] == pytest.approx(b[1]) and 0 < a[1] <= 1.0 + 1e-6
    assert (tmp_path / "idx" / "build_meta.json").exists()


def test_extractive_guidance_grounded_traceable_no_medication(store):
    g = rp.FirstAidRAG(store, min_similarity=0.1).guide("wound bleeding pressing cloth")
    assert g.status == "ok" and g.mode == "extractive"
    assert g.fully_grounded and g.groundedness == 1.0
    assert all(s.chunk_ids for s in g.sentences)
    assert all(s.chunk_ids[0].startswith("bleeding.md") for s in g.sentences[:1])
    assert not any("aspirin" in s.text.lower() for s in g.sentences)
    assert any("aspirin" in s.lower() for s in g.removed_medication)
    assert "108" in g.message


def test_not_covered_below_threshold(store):
    g = rp.FirstAidRAG(store, min_similarity=0.9).guide("person feels unwell near the bus stop")
    assert g.status == "not_covered" and g.message == rp.NOT_COVERED_MSG
    assert g.sentences == [] and g.retrieved and g.refusal_reason == "below_threshold"


def test_llm_ungrounded_sentence_dropped_and_scored(store):
    gen = lambda prompt: ("1. Press firmly on the wound with a clean cloth.\n"
                          "2. Pour vinegar on the wound.\n3. Take 2 tablets of paracetamol.")
    entail = lambda premise, hyp: 0.9 if "press" in hyp.lower() else 0.1
    g = rp.FirstAidRAG(store, generator=gen, entail=entail, min_similarity=0.1).guide(
        "bleeding wound pressing")
    assert g.mode == "llm" and g.status == "ok"
    assert g.groundedness == pytest.approx(0.5) and g.fully_grounded is False
    assert g.dropped_ungrounded == ["Pour vinegar on the wound."]
    assert len(g.removed_medication) == 1
    assert [s.text for s in g.sentences] == ["Press firmly on the wound with a clean cloth."]


def test_llm_prompt_contains_context_and_not_covered_reply(store):
    seen = {}

    def gen(prompt):
        seen["p"] = prompt
        return "NOT_COVERED"
    g = rp.FirstAidRAG(store, generator=gen, entail=lambda *_: 1.0,
                       min_similarity=0.1).guide("burn water")
    assert "burns.md#" in seen["p"] and "STRICTLY" in seen["p"]
    assert g.status == "not_covered"


def test_llm_failure_returns_fallback(store):
    def gen(prompt):
        raise TimeoutError
    g = rp.FirstAidRAG(store, generator=gen, min_similarity=0.1).guide("burn water")
    assert g.status == "llm_unavailable" and g.message == rp.LLM_DOWN_MSG


def test_check_groundedness_ratio_and_empty():
    from langchain_core.documents import Document
    ch = [Document("Cool the burn under running water.", metadata={"chunk_id": "a#0"})]
    assert rp.check_groundedness([], ch, None) == 1.0
    assert rp.check_groundedness(["Cool the burn under running water"], ch, None) == 1.0
    assert rp.check_groundedness(["Cool the burn.", "Apply butter."], ch,
                                 lambda p, h: 0.8 if "Cool" in h else 0.2) == 0.5
    # joined-context fallback: entailed only when all chunks are given together
    ch2 = ch + [Document("Remove rings.", metadata={"chunk_id": "b#0"})]
    joint = lambda p, h: 0.9 if ("rings" in p and "water" in p) else 0.0
    assert rp.sentence_support("Cool it and remove rings.", ch2, joint) == ["a#0", "b#0"]


def test_split_sentences_and_medication_rule():
    assert rp.split_sentences("1. Do A.\n- Do B! Then C.") == ["Do A.", "Do B!", "Then C."]
    for s in ("Give 5 ml syrup.", "Take a tablet.", "aspirin helps", "no dosage info"):
        assert rp.is_medication(s)
    assert not rp.is_medication("Press firmly on the wound.")


def test_generator_from_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert rp.generator_from_env() is None
    monkeypatch.setenv("LLM_PROVIDER", "openai_compat")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        rp.generator_from_env()


def _report(answers: dict, injury="road_accident"):
    turns = [DialogueTurn(q_id=q, question="q", answer_raw=a, answer_value=a) for q, a in answers.items()]
    return IncidentReport(incident_id="i", timestamp="2026-10-05T00:00:00Z", gps=GPS(lat=12.9, lon=77.6),
                          language="en", transcript="t", dialogue=turns,
                          text_features=[0.0] * 19, text_severity_probs=[0.25] * 4,
                          injury_type=injury, num_injured=1)


def test_queries_from_report_routing():
    q = rp.queries_from_report
    # unresponsive, breathing unknown -> CPR (corpus: if in doubt, assume breathing is not normal)
    out = q(_report({"Q_CONSCIOUS": "no", "Q_BLEEDING": "yes"}))
    assert [(p, f) for p, _, f in out] == [("cpr", "cpr_adult_ifrc.md"),
                                           ("bleeding", "severe_bleeding_ifrc.md"),
                                           ("road_accident", None)]
    assert q(_report({"Q_CONSCIOUS": "no", "Q_BREATHING": "yes"}))[0][2] == "cpr_adult_ifrc.md"
    assert q(_report({"Q_CONSCIOUS": "no", "Q_BREATHING": "no"}))[0][2] == "recovery_position_ifrc.md"
    assert q(_report({"Q_CONSCIOUS": "yes", "Q_BREATHING": "yes"}))[0][2] == "breathing_difficulty_ifrc.md"
    # unresponsive cardiac case: CPR replaces the chest-pain item
    out = q(_report({"Q_CONSCIOUS": "no"}, injury="cardiac_medical"))
    assert [p for p, _, _ in out] == ["cpr"]
    assert q(_report({"Q_CONSCIOUS": "yes"}, injury="burn"))[-1][2] == "burns_ifrc.md"
    # every routed file exists in the real corpus
    real = {p.name for p in rp.CORPUS_DIR.glob("*_ifrc.md")}
    routed = {f for _, f in rp.INJURY_QUERY.values() if f} | {
        "cpr_adult_ifrc.md", "recovery_position_ifrc.md", "breathing_difficulty_ifrc.md",
        "severe_bleeding_ifrc.md"}
    if real:
        assert routed <= real


def test_routed_guide_stays_in_file_and_skips_threshold(store):
    rag = rp.FirstAidRAG(store, min_similarity=0.99)
    assert rag.guide("burn water").status == "not_covered"            # free text: threshold applies
    g = rag.guide("burn water", topic_file="bleeding.md")              # routed: file filter only
    assert g.status == "ok" and {r["file"] for r in g.retrieved} == {"bleeding.md"}
    assert g.topic_file == "bleeding.md" and g.refusal_reason is None


@pytest.mark.parametrize("query,reason", [
    ("what tablet should I give for the pain", "medication"),
    ("can I give him a painkiller injection", "medication"),
    ("how long will the ambulance take", "non_medical"),
    ("which hospital has free beds", "non_medical"),
    ("a snake bit my brother", None),                     # v3: snake_bites in corpus
    ("my diabetic father is sweating", None),
    ("child has high fever", None),
    ("he got an electric shock from the wire", "excluded_topic"),   # no IFRC 2020 chapter
    ("my son has a nosebleed", "excluded_topic"),
    ("the boy is drowning in the lake", "excluded_topic"),
    ("his leg is bleeding a bit after the fall", None),
    ("pregnant woman fell and hurt her ankle", None),
    ("he is having a fit and clenching his teeth", None),
])
def test_query_guard(query, reason):
    assert rp.query_refusal_reason(query) == reason


def test_guard_refuses_before_retrieval(store):
    g = rp.FirstAidRAG(store, min_similarity=0.0).guide("how many mg of aspirin for bleeding wound")
    assert g.status == "not_covered" and g.refusal_reason == "medication" and g.retrieved == []


def test_embed_text_uses_topic_and_keywords():
    from langchain_core.documents import Document
    d = Document("Cool the burn.", metadata={"title": "Guidelines 2020 - Burns", "keywords": "scald"})
    assert rp.embed_text(d) == "Burns. scald\nCool the burn."


def test_eval_metrics_and_sweep(store, tmp_path):
    q = pd.DataFrame([
        {"query_id": "1", "query": "bleeding wound pressing", "expected_file": "bleeding.md", "split": "test"},
        {"query_id": "2", "query": "burn running water", "expected_file": "burns.md", "split": "test"},
        {"query_id": "3", "query": "zzz qqq", "expected_file": "", "split": "test"},
        {"query_id": "4", "query": "burn rings", "expected_file": "burns.md", "split": "dev"},
        {"query_id": "5", "query": "xyz", "expected_file": "", "split": "dev"},
    ])
    p = tmp_path / "q.csv"
    q.to_csv(p, index=False)
    df = ev.load_queries(p)
    rag = rp.FirstAidRAG(store, min_similarity=0.2)
    per_q, s = ev.evaluate(rag, df[df.split == "test"])
    assert s["hit_at_k"] == 1.0 and s["answer_rate"] == 1.0 and s["refusal_rate"] == 1.0
    assert s["pct_fully_grounded"] == 1.0 and s["medication_removed"] >= 1
    sw = ev.sweep_threshold(rag, df[df.split == "dev"], grid=[0.1, 0.99])
    assert sw.loc[0, "answer_rate"] == 1.0 and sw.loc[1, "refusal_rate"] == 1.0


def test_load_queries_validation(tmp_path):
    p = tmp_path / "q.csv"
    pd.DataFrame([{"query_id": "1", "query": "a", "expected_file": "", "split": "train"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="split"):
        ev.load_queries(p)


def test_key_steps_replace_routed_sentences_and_stay_traceable(corpus, store):
    (corpus / "key_steps.yaml").write_text(
        "bleeding.md:\n  - Keep pressing\n  - Press firmly\n", encoding="utf-8")
    steps = rp.load_key_steps(corpus)
    assert steps == {"bleeding.md": ["Keep pressing until help arrives.",
                                     "Press firmly on the wound with a clean cloth."]}   # yaml order
    rag = rp.FirstAidRAG(store, min_similarity=0.99, key_steps=steps)
    g = rag.guide("bleeding", topic_file="bleeding.md")
    assert g.status == "ok" and g.mode == "key_steps" and g.fully_grounded
    assert [s.text for s in g.sentences] == steps["bleeding.md"]
    assert all(s.chunk_ids and s.chunk_ids[0].startswith("bleeding.md#") for s in g.sentences)
    assert {r["file"] for r in g.retrieved} == {"bleeding.md"}
    assert rag.guide("burn water", topic_file="burns.md").mode == "extractive"   # no list: unchanged
    assert rag.guide("press wound bleeding").mode == "extractive"                  # free text: unchanged


def test_key_steps_reject_missing_or_ambiguous_prefix(corpus):
    (corpus / "key_steps.yaml").write_text("bleeding.md:\n  - Stitch the wound\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matches 0"):
        rp.load_key_steps(corpus)
    (corpus / "key_steps.yaml").write_text('bleeding.md:\n  - ""\n', encoding="utf-8")
    with pytest.raises(ValueError, match="matches 3"):          # empty prefix: every TOY sentence
        rp.load_key_steps(corpus)
    assert rp.load_key_steps(corpus, name="absent.yaml") == {}


def test_real_key_steps_resolve_inside_single_chunks():
    """Every reviewed step is one corpus sentence that sits whole inside a chunk of its file."""
    steps = rp.load_key_steps()
    assert steps and all(1 <= len(v) <= 4 for v in steps.values())
    chunks = rp.split_corpus(rp.load_corpus())
    for f, sents in steps.items():
        mine = [c for c in chunks if c.metadata["file"] == f]
        for s in sents:
            assert rp.sentence_support(s, mine, None), (f, s)
            assert not rp.is_medication(s)
