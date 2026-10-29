"""RAG first-aid guidance (docs/01 §6, docs/12 §01, docs/07 §5-6, AGENTS rule 6).

Pipeline (LangChain is used only here, never for dialogue logic, rule 7):
  curated corpus (`data/first_aid_corpus/`, format in its README) -> RecursiveCharacterTextSplitter
  -> bge-small-en-v1.5 embeddings (normalised; MiniLM in v1) -> FAISS (inner product = cosine) -> top-k chunks
  with a similarity threshold -> grounded generation (LCEL) -> medication filter -> per-sentence
  groundedness (verbatim match, else NLI entailment) -> drop ungrounded sentences at serve time.

Every guidance sentence carries the chunk ids that back it (traceability, rule 6). If no chunk
reaches `min_similarity` the answer is "not covered" (docs/07 §6), never improvised.

v2 (2026-10-06, after the v1 test run):
  * Query guard: medication questions, non-medical questions and topics deliberately left out of
    the corpus (IFRC topics not extracted, e.g. snake bite, poisoning) are refused before retrieval.
  * Dialogue routing: when the structured dialogue already identifies the situation,
    `guide_report()` retrieves only inside that topic's file (`queries_from_report`). Unresponsive
    with abnormal/unknown breathing -> CPR, per the corpus's own rule ("If there is any doubt
    about whether breathing is normal, assume it is not", IFRC 2020 p.134-145).
  * Chunks are embedded with their topic name + header `keywords` prepended (stored text unchanged).
  * Embedding model is configurable (`RAG_EMBED_MODEL`); the built index records which one it used.
If the LLM is unreachable the answer is the docs/07 §6 fallback message.

Generation modes:
  * "extractive" (default when LLM_PROVIDER is empty): guidance = sentences copied verbatim from
    the retrieved chunks, in score order. No API needed; grounded by construction.
  * "llm": an OpenAI-compatible chat endpoint (free tier only, docs/15 §4), configured from
    `.env`: LLM_PROVIDER=openai_compat, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.

[ASSUMPTION]s (tune on the dev split of the eval set, `backend/rag/eval_rag.py`):
  * min_similarity 0.55 (bge cosine; chosen on dev rows only, safety-first: no in-scope dev refusal), k 4, chunk 500 / overlap 50 chars (docs/01 §6.2 start values).
  * A sentence is grounded if it appears verbatim in a chunk, or if NLI P(entailment) >= 0.5
    for some retrieved chunk (or all chunks joined). NLI model: cross-encoder/nli-deberta-v3-small.
  * Medication/dosage sentences are removed by a keyword rule (rule 6), even if the corpus says
    them; removals are logged.
  * Guidance is English only; the corpus is English. hi/kn guidance = future work.
  * Child vs adult is not asked by the dialogue: routed CPR uses the adult file [ASSUMPTION].
  * Query-guard keyword lists below are rules, not learned; they can over-refuse (e.g. "bleeding
    after a dog bite" is refused as an excluded topic).
Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = Path("data/first_aid_corpus")
KEY_STEPS_FILE = "key_steps.yaml"   # short reviewed step list per topic (dialogue-routed guidance)
INDEX_DIR = Path("models/rag/faiss_index")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # dev hit@4 1.000 vs MiniLM 0.956
QUERY_PROMPTS = {  # instruction prefixes some retrieval models expect on queries (model cards)
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
HF_CACHE = Path("models/hf")
REQUIRED_META = ("title", "publisher", "url", "license", "topic")
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
TOP_K = 4
MIN_SIMILARITY = 0.55  # v2, bge scale; dev: keeps every in-scope query answered (see progress notes)
ENTAIL_THRESHOLD = 0.5
MAX_SENTENCES = 6

NOT_COVERED_MSG = ("Our first-aid reference does not cover this situation. "
                   "Please wait for the ambulance crew and call 108 if you have not already.")
LLM_DOWN_MSG = "Wait for professional help; call 108."  # docs/07 §6, verbatim

MEDICATION_RE = re.compile(
    r"\b(\d+(\.\d+)?\s?(mg|mcg|ml|g|iu)\b|dose|dosage|dosing|tablet|capsule|pill|inject|"
    r"medicine|medication|drug|aspirin|paracetamol|acetaminophen|ibuprofen|antibiotic|"
    r"epinephrine|adrenaline|insulin|nitroglycerin)", re.IGNORECASE)

# Query guard (v2). Medication questions are refused outright (rule 6).
MED_QUERY_RE = re.compile(r"painkiller|pain killer|injection|ointment|syrup|which medicine|"
                          r"what (tablet|medicine|drug)", re.IGNORECASE)
NON_MEDICAL_RE = re.compile(r"how long will|\beta\b|weather|traffic|which hospital|nearest hospital|"
                            r"free beds?|hospital beds?|route to|address of", re.IGNORECASE)
# Topics NOT in the corpus: IFRC 2020 topics not extracted (v3 added bites, stings, snakebite,
# poisoning, allergy, diabetic, faint, fever, abdominal pain, childbirth, heat, dehydration,
# hypothermia, asthma, knocked-out tooth) + electric shock and nosebleed, which have no IFRC 2020
# chapter. Taken from the IFRC table of contents, not from eval results.
EXCLUDED_TOPIC_RE = re.compile(
    r"jellyfish|frostbite|electric shock|electrocut|current (hit|struck|passed)|drown|overdose|croup|"
    r"nose ?bleed|bleeding from (the |his |her |my )?nose|eye injury|in (his|her|the|my) eyes?\b",
    re.IGNORECASE)


def query_refusal_reason(query: str) -> Optional[str]:
    """Why a free-text query must get "not covered" before retrieval, or None."""
    if MEDICATION_RE.search(query) or MED_QUERY_RE.search(query):
        return "medication"
    if NON_MEDICAL_RE.search(query):
        return "non_medical"
    if EXCLUDED_TOPIC_RE.search(query):
        return "excluded_topic"
    return None


GROUNDED_PROMPT = ChatPromptTemplate.from_template(
    "You give first-aid guidance to a bystander, based STRICTLY on the context below.\n"
    "Rules: use only steps stated in the context; copy wording where possible; short numbered "
    "sentences; no medication or dosage advice; if the context does not clearly cover the "
    "situation, reply exactly NOT_COVERED.\n\nContext:\n{context}\n\nSituation: {question}\n\n"
    "Guidance:")

Generator = Callable[[str], str]          # prompt -> completion
EntailFn = Callable[[str, str], float]    # (premise, hypothesis) -> P(entailment)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def parse_corpus_file(path: Path) -> Document:
    """Read one corpus file: a `---` header of `key: value` lines, then the protocol text."""
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
    if not m:
        raise ValueError(f"{path}: missing '---' metadata header (see corpus README)")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    missing = [k for k in REQUIRED_META if not meta.get(k)]
    if missing:
        raise ValueError(f"{path}: metadata missing {missing}")
    body = m.group(2).strip()
    if not body:
        raise ValueError(f"{path}: empty body")
    meta["file"] = path.name
    return Document(page_content=body, metadata=meta)


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    files = sorted(p for p in corpus_dir.glob("*") if p.suffix in (".md", ".txt")
                   and p.name.lower() != "readme.md")
    if not files:
        raise FileNotFoundError(f"no corpus files in {corpus_dir} (see its README.md)")
    return [parse_corpus_file(p) for p in files]


def split_corpus(docs: Sequence[Document], chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Chunk documents; each chunk gets a stable id `<file>#<n>`."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    out = []
    for d in docs:
        for i, c in enumerate(splitter.split_documents([d])):
            c.metadata["chunk_id"] = f"{d.metadata['file']}#{i}"
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def default_embeddings(model_name: str = EMBED_MODEL):
    from langchain_huggingface import HuggingFaceEmbeddings
    q_kwargs = {"normalize_embeddings": True}
    if model_name in QUERY_PROMPTS:
        q_kwargs["prompt"] = QUERY_PROMPTS[model_name]
    return HuggingFaceEmbeddings(model_name=model_name, cache_folder=str(HF_CACHE),
                                 encode_kwargs={"normalize_embeddings": True},
                                 query_encode_kwargs=q_kwargs)


def embed_text(chunk: Document) -> str:
    """Text that gets embedded: topic name + keywords + chunk (the stored text stays the chunk)."""
    m = chunk.metadata
    name = m.get("title", "").rsplit(" - ", 1)[-1]
    return f"{name}. {m.get('keywords', '')}\n{chunk.page_content}"


def build_index(chunks: Sequence[Document], embeddings):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    chunks = list(chunks)
    vecs = embeddings.embed_documents([embed_text(c) for c in chunks])
    return FAISS.from_embeddings(list(zip([c.page_content for c in chunks], vecs)), embeddings,
                                 metadatas=[c.metadata for c in chunks],
                                 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


def save_index(store, index_dir: Path = INDEX_DIR, meta: Optional[dict] = None) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    (index_dir / "build_meta.json").write_text(json.dumps(meta or {}, indent=2), encoding="utf-8")


def load_index(embeddings, index_dir: Path = INDEX_DIR):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    # Our own pickle, built locally by `build`; never load an index from elsewhere.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True,
                            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split on sentence ends and line breaks; strip list markers ("1.", "-", "*")."""
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)
    out = []
    for p in parts:
        s = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", p).strip()
        if len(s) > 2:
            out.append(s)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def load_key_steps(corpus_dir: Path = CORPUS_DIR, name: str = KEY_STEPS_FILE) -> dict[str, list[str]]:
    """{topic file: [full corpus sentences]} from `key_steps.yaml` (sentence prefixes per file).

    Each prefix must start exactly one distinct sentence of that file, so the steps stay verbatim
    corpus text (rule 6). No yaml file -> {} (routed guidance keeps the retrieved sentences).
    """
    import yaml
    path = corpus_dir / name
    if not path.exists():
        return {}
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for fname, prefixes in spec.items():
        sents = list(dict.fromkeys(split_sentences(parse_corpus_file(corpus_dir / fname).page_content)))
        steps = []
        for pre in prefixes:
            hits = [x for x in sents if _norm(x).startswith(_norm(pre))]
            if len(hits) != 1:
                raise ValueError(f"{name}: {fname}: prefix {pre!r} matches {len(hits)} sentences, need 1")
            steps.append(hits[0])
        out[fname] = steps
    return out


def is_medication(sentence: str) -> bool:
    return bool(MEDICATION_RE.search(sentence))


# ---------------------------------------------------------------------------
# Groundedness (docs/12 §01: NLI entailment per sentence; returns a float ratio)
# ---------------------------------------------------------------------------

class NLIJudge:
    """Cross-encoder NLI; `__call__(premise, hypothesis)` -> P(entailment). Lazy-loaded."""

    def __init__(self, model_name: str = NLI_MODEL):
        self.model_name = model_name
        self._model = None
        self._entail_idx = None

    def _load(self):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, cache_folder=str(HF_CACHE))
        id2label = self._model.model.config.id2label
        self._entail_idx = next(i for i, l in id2label.items() if l.lower().startswith("entail"))

    def __call__(self, premise: str, hypothesis: str) -> float:
        if self._model is None:
            self._load()
        import numpy as np
        logits = np.asarray(self._model.predict([(premise, hypothesis)]))[0]
        p = np.exp(logits - logits.max())
        return float((p / p.sum())[self._entail_idx])


def sentence_support(sentence: str, chunks: Sequence[Document], entail: Optional[EntailFn],
                     threshold: float = ENTAIL_THRESHOLD) -> list[str]:
    """Chunk ids that support `sentence` (verbatim first, else NLI). Empty list = ungrounded."""
    ns = _norm(sentence)
    verbatim = [c.metadata["chunk_id"] for c in chunks if ns and ns in _norm(c.page_content)]
    if verbatim or entail is None:
        return verbatim
    scored = [(c.metadata["chunk_id"], entail(c.page_content, sentence)) for c in chunks]
    hits = [cid for cid, p in scored if p >= threshold]
    if not hits and len(chunks) > 1:
        joined = "\n".join(c.page_content for c in chunks)
        if entail(joined, sentence) >= threshold:
            hits = [c.metadata["chunk_id"] for c in chunks]
    return hits


def check_groundedness(sentences: Sequence[str], chunks: Sequence[Document],
                       entail: Optional[EntailFn], threshold: float = ENTAIL_THRESHOLD) -> float:
    """Fraction of answer sentences entailed by the retrieved chunks (1.0 = fully grounded).

    An empty answer returns 1.0 (nothing unsupported was said).
    """
    if not sentences:
        return 1.0
    ok = sum(bool(sentence_support(s, chunks, entail, threshold)) for s in sentences)
    return ok / len(sentences)


# ---------------------------------------------------------------------------
# LLM (free-tier OpenAI-compatible endpoint, optional)
# ---------------------------------------------------------------------------

def openai_compat_generator(base_url: str, model: str, api_key: str,
                            timeout: float = 20.0) -> Generator:
    def gen(prompt: str) -> str:
        body = json.dumps({"model": model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    return gen


def generator_from_env() -> Optional[Generator]:
    """None (extractive mode) unless LLM_PROVIDER=openai_compat and all settings are present."""
    if os.getenv("LLM_PROVIDER", "").strip() != "openai_compat":
        return None
    url, model, key = (os.getenv(k, "").strip() for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"))
    if not (url and model and key):
        raise RuntimeError("LLM_PROVIDER=openai_compat needs LLM_BASE_URL, LLM_MODEL, LLM_API_KEY")
    return openai_compat_generator(url, model, key)


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------

@dataclass
class GuidanceSentence:
    text: str
    chunk_ids: list[str]


@dataclass
class Guidance:
    query: str
    status: str                      # ok | not_covered | llm_unavailable
    mode: str                        # extractive | llm | key_steps (routed topic, reviewed list)
    message: str                     # text shown to the reporter
    sentences: list[GuidanceSentence] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)   # chunk_id, file, publisher, url, score
    groundedness: Optional[float] = None    # before dropping ungrounded sentences
    fully_grounded: Optional[bool] = None
    dropped_ungrounded: list[str] = field(default_factory=list)
    removed_medication: list[str] = field(default_factory=list)
    refusal_reason: Optional[str] = None    # medication | non_medical | excluded_topic | below_threshold
    topic_file: Optional[str] = None        # set when routed from the dialogue

    def to_dict(self) -> dict:
        return asdict(self)


def format_docs(docs: Sequence[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['chunk_id']} | {d.metadata.get('publisher', '?')}]\n"
                       f"{d.page_content}" for d in docs)


@dataclass
class FirstAidRAG:
    store: object                               # LangChain vector store
    generator: Optional[Generator] = None       # None -> extractive
    entail: Optional[EntailFn] = None           # None -> verbatim check only
    k: int = TOP_K
    min_similarity: float = MIN_SIMILARITY
    max_sentences: int = MAX_SENTENCES
    drop_ungrounded: bool = True
    embed_model: str = EMBED_MODEL                # recorded in logs
    key_steps: dict = field(default_factory=dict)  # {topic file: [sentences]}, see load_key_steps

    @property
    def mode(self) -> str:
        return "extractive" if self.generator is None else "llm"

    def retrieve(self, query: str, topic_file: Optional[str] = None) -> list[tuple[Document, float]]:
        kw = {"filter": {"file": topic_file}, "fetch_k": 1000} if topic_file else {}
        hits = self.store.similarity_search_with_score(query, k=self.k, **kw)
        return [(d, float(s)) for d, s in hits]

    def _generate(self, query: str, chunks: Sequence[Document]) -> str:
        chain = (GROUNDED_PROMPT
                 | RunnableLambda(lambda pv: self.generator(pv.to_string()))
                 | StrOutputParser())
        return chain.invoke({"context": format_docs(chunks), "question": query})

    def _key_steps(self, g: Guidance, query: str, topic_file: str) -> Guidance:
        """Routed topic with a reviewed step list: those sentences only, each traced to its chunk."""
        hits = self.store.similarity_search_with_score(query, k=1000, filter={"file": topic_file},
                                                       fetch_k=1000)
        chunks = [d for d, _ in hits]
        kept = [GuidanceSentence(t, sentence_support(t, chunks, None)) for t in self.key_steps[topic_file]]
        kept = [k for k in kept if k.chunk_ids]          # a step split across chunks is not shown
        used = {c for k in kept for c in k.chunk_ids}
        g.mode = "key_steps"
        g.retrieved = [{"chunk_id": d.metadata["chunk_id"], "file": d.metadata["file"],
                        "publisher": d.metadata.get("publisher"), "url": d.metadata.get("url"),
                        "score": round(float(sc), 4)} for d, sc in hits if d.metadata["chunk_id"] in used]
        if not kept:
            return g
        g.sentences, g.status = kept, "ok"
        g.groundedness, g.fully_grounded = 1.0, True
        g.message = " ".join(k.text for k in kept) + " Call 108 if you have not already."
        return g

    def guide(self, query: str, topic_file: Optional[str] = None) -> Guidance:
        """Guidance for a free-text query. With `topic_file` (dialogue routing) retrieval is
        restricted to that file and the similarity threshold is not applied."""
        g = Guidance(query=query, status="not_covered", mode=self.mode, message=NOT_COVERED_MSG,
                     topic_file=topic_file)
        if topic_file is None:
            g.refusal_reason = query_refusal_reason(query)
            if g.refusal_reason:
                return g
        elif topic_file in self.key_steps:
            return self._key_steps(g, query, topic_file)
        hits = self.retrieve(query, topic_file)
        g.retrieved = [{"chunk_id": d.metadata["chunk_id"], "file": d.metadata["file"],
                        "publisher": d.metadata.get("publisher"), "url": d.metadata.get("url"),
                        "score": round(s, 4)} for d, s in hits]
        chunks = [d for d, s in hits if topic_file or s >= self.min_similarity]
        if not chunks:
            g.refusal_reason = "below_threshold"
            return g

        if self.generator is None:
            raw = [s for c in chunks for s in split_sentences(c.page_content)]
        else:
            try:
                out = self._generate(query, chunks)
            except Exception:                                  # network / quota / parse errors
                g.status, g.message = "llm_unavailable", LLM_DOWN_MSG
                return g
            if "NOT_COVERED" in out:
                return g
            raw = split_sentences(out)

        g.removed_medication = [s for s in raw if is_medication(s)]
        cand = [s for s in raw if not is_medication(s)]
        cand = list(dict.fromkeys(cand))[: self.max_sentences]
        support = [sentence_support(s, chunks, self.entail) for s in cand]
        if cand:
            g.groundedness = sum(bool(x) for x in support) / len(cand)
            g.fully_grounded = g.groundedness == 1.0
        kept = [GuidanceSentence(s, ids) for s, ids in zip(cand, support)
                if ids or not self.drop_ungrounded]
        g.dropped_ungrounded = [s for s, ids in zip(cand, support) if not ids] if self.drop_ungrounded else []
        if not kept:
            return g
        g.sentences = kept
        g.status = "ok"
        g.message = " ".join(s.text for s in kept) + " Call 108 if you have not already."
        return g

    def guide_report(self, report) -> list[tuple[str, Guidance]]:
        """Guidance per problem found by the dialogue, in priority order."""
        return [(problem, self.guide(q, topic_file=f))
                for problem, q, f in queries_from_report(report)]


# ---------------------------------------------------------------------------
# Query building from the dialogue (rule-based; the dialogue logic stays in the state machine)
# ---------------------------------------------------------------------------

# injury_type -> (query, topic file; None = free retrieval with the similarity threshold)
INJURY_QUERY = {
    "road_accident": ("road traffic accident injury", None),
    "fall": ("fall injury, possible broken bone", "fractures_ifrc.md"),
    "burn": ("burn injury, cool the burn", "burns_ifrc.md"),
    "cardiac_medical": ("chest pain, possible heart attack", "chest_pain_ifrc.md"),
    "other": ("injured person", None),
}


def queries_from_report(report) -> list[tuple[str, str, Optional[str]]]:
    """Ordered (problem, query, topic_file) from an IncidentReport: airway/breathing first.

    Unresponsive + breathing difficulty or unknown -> CPR (corpus rule: if in doubt whether
    breathing is normal, assume it is not); unresponsive + no breathing difficulty -> recovery
    position. For an unresponsive cardiac_medical report the CPR item replaces chest pain.
    """
    ans = {t.q_id: t.answer_value for t in report.dialogue}
    out = []
    unresponsive = ans.get("Q_CONSCIOUS") == "no"
    if unresponsive and ans.get("Q_BREATHING") != "no":
        out.append(("cpr", "unresponsive and not breathing normally, start chest compressions",
                    "cpr_adult_ifrc.md"))
    elif unresponsive:
        out.append(("unresponsive_breathing",
                    "unresponsive and breathing normally, recovery position",
                    "recovery_position_ifrc.md"))
    elif ans.get("Q_BREATHING") == "yes":
        out.append(("breathing", "person has difficulty breathing", "breathing_difficulty_ifrc.md"))
    if ans.get("Q_BLEEDING") == "yes":
        out.append(("bleeding", "severe bleeding, apply direct pressure", "severe_bleeding_ifrc.md"))
    query, f = INJURY_QUERY[report.injury_type]
    if not (unresponsive and report.injury_type == "cardiac_medical"):
        out.append((report.injury_type, query, f))
    return out


# ---------------------------------------------------------------------------
# CLI: build the index / ask one question
# ---------------------------------------------------------------------------

def build(corpus_dir: Path = CORPUS_DIR, index_dir: Path = INDEX_DIR, embeddings=None,
          chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
          embed_model: str = EMBED_MODEL):
    docs = load_corpus(corpus_dir)
    chunks = split_corpus(docs, chunk_size, chunk_overlap)
    store = build_index(chunks, embeddings or default_embeddings(embed_model))
    save_index(store, index_dir, {"embed_model": embed_model, "chunk_size": chunk_size,
                                  "chunk_overlap": chunk_overlap, "n_docs": len(docs),
                                  "n_chunks": len(chunks),
                                  "files": sorted(d.metadata["file"] for d in docs)})
    return store, docs, chunks


def load(index_dir: Path = INDEX_DIR, embeddings=None, nli: bool = True, **kw) -> FirstAidRAG:
    from dotenv import load_dotenv
    load_dotenv()
    if embeddings is None:  # query with the same model the index was built with
        meta_path = index_dir / "build_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        embeddings = default_embeddings(meta.get("embed_model", EMBED_MODEL))
    store = load_index(embeddings, index_dir)
    kw.setdefault("key_steps", load_key_steps())
    return FirstAidRAG(store=store, generator=generator_from_env(),
                       entail=NLIJudge() if nli else None,
                       embed_model=getattr(embeddings, "model_name", EMBED_MODEL), **kw)


def main() -> None:
    p = argparse.ArgumentParser(description="RAG first-aid guidance: build index / ask")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    b.add_argument("--index", type=Path, default=INDEX_DIR)
    b.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    b.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    b.add_argument("--embed-model", default=EMBED_MODEL)
    q = sub.add_parser("ask")
    q.add_argument("query")
    q.add_argument("--index", type=Path, default=INDEX_DIR)
    q.add_argument("--min-sim", type=float, default=MIN_SIMILARITY)
    q.add_argument("--no-nli", action="store_true")
    a = p.parse_args()

    if a.cmd == "build":
        _, docs, chunks = build(a.corpus, a.index, chunk_size=a.chunk_size,
                                chunk_overlap=a.chunk_overlap, embed_model=a.embed_model)
        print(f"indexed {len(docs)} files, {len(chunks)} chunks -> {a.index}")
    else:
        rag = load(a.index, nli=not a.no_nli, min_similarity=a.min_sim)
        print(json.dumps(rag.guide(a.query).to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
