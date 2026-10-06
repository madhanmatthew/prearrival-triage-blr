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
    g = rp.FirstAidRAG(store, min_similarity=0.9).guide("snake bite on the ankle")
    assert g.status == "not_covered" and g.message == rp.NOT_COVERED_MSG
    assert g.sentences == [] and g.retrieved  # retrieval still logged


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


def test_queries_from_report_priority():
    turns = [DialogueTurn(q_id="Q_CONSCIOUS", question="q", answer_raw="no", answer_value="no"),
             DialogueTurn(q_id="Q_BLEEDING", question="q", answer_raw="yes", answer_value="yes")]
    rep = IncidentReport(incident_id="i", timestamp="2026-10-05T00:00:00Z", gps=GPS(lat=12.9, lon=77.6),
                         language="en", transcript="t", dialogue=turns,
                         text_features=[0.0] * 19, text_severity_probs=[0.25] * 4,
                         injury_type="road_accident", num_injured=1)
    assert [p for p, _ in rp.queries_from_report(rep)] == ["unconscious", "bleeding", "road_accident"]


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
