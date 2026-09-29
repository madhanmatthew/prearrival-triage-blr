# Component 1 — Reporting Agent (Voice, Conversational AI, RAG First-Aid Guidance)

**Owner responsibility:** end-to-end reporting pipeline — from raw voice/text
input to a structured incident report + first-aid guidance output.

**Reads first:** `00_MASTER_OVERVIEW.md` for system context.

---

## 1. Purpose

This is the system's first point of contact. It must:
1. Accept a report in voice or text, in one of several languages.
2. Extract structured incident information (location via GPS, injury description,
   apparent severity signals) through **guided dialogue**, not free-form chat.
3. Feed the extracted severity signal into the fusion model (Component 2/5).
4. Provide grounded, non-hallucinated first-aid guidance to the reporter while
   help is en route.

## 2. Current State (MVP) — honest baseline

- `report_parser.py` does keyword/regex matching on text input only.
- No voice support.
- No multilingual support.
- No guided follow-up questioning.
- No first-aid guidance of any kind.

**This component needs to be rebuilt, not incrementally patched.**

## 3. Target Architecture

```
Voice Input ──► Whisper ASR ──► Transcribed Text ──┐
                                                     │
Text Input ─────────────────────────────────────────┤──► LLM Dialogue Manager
                                                     │      (structured, guided)
GPS ─────────────────────────────────────────────────      │
                                                            ▼
                                              Extracted Severity Signal
                                                            │
                                                            ▼
                                          (feeds Component 2 fusion model)

Guidance Request ──► RAG Retrieval (vector store over first-aid corpus)
                              │
                              ▼
                     LLM answers ONLY from retrieved context
                              │
                              ▼
                     Grounded first-aid instructions to reporter
```

## 4. Sub-piece 1: Multilingual Voice Intake

- **Model:** Whisper (small/tiny variant for feasible on-device or low-latency
  inference; base/small should be enough for a demo).
- **Languages for demo scope:** Kannada, Hindi, English. Architecture should stay
  language-agnostic (Whisper supports ~90+ languages) — just scope the *demo* to
  3, not the architecture.
- **Do NOT extract location from speech.** Use device GPS directly — far more
  reliable than parsing an address out of a panicked voice report. This was a
  deliberate design decision, not an oversight — do not "improve" this later by
  adding speech-based geolocation.
- Output: transcribed text, passed to the Dialogue Manager below.

## 5. Sub-piece 2: Guided Conversational Dialogue

- **NOT an open-ended chatbot.** This is a structured, multi-turn dialogue where
  the AI asks *targeted* follow-up questions based on the severity signal it
  already has, similar to how trained emergency call operators triage over the
  phone.
- Example flow:
  1. Initial report parsed for keywords/severity cues.
  2. If severity signal is ambiguous or high, ask targeted follow-ups: "Is the
     person conscious?", "Is there visible bleeding?", "Can they speak clearly?"
  3. Each answer refines the severity estimate.
- **Implementation approach:** LLM API call with a system prompt constraining it
  to ask ONE targeted question at a time from a bounded question set tied to
  triage logic — not open-ended generation. Consider a state machine (dialogue
  state = which questions have been asked/answered) wrapping the LLM call, so the
  LLM's role is generating natural phrasing, not deciding dialogue logic freely.
- Output: a structured severity feature vector (or text summary) — this becomes
  the "text severity" input to the fusion model in `05_coordinator_fusion_evaluation.md`.
- This replaces `report_parser.py` and also replaces/upgrades the existing
  TF-IDF+LogisticRegression severity classifier's INPUT quality — consider
  whether the downstream classifier still runs on the dialogue's structured
  output, or whether the dialogue's own extracted signals become the features
  directly. Decide and document this interface explicitly before building
  Component 2's fusion layer.

## 6. Sub-piece 3: RAG-Grounded First-Aid Guidance (LangChain Implementation)

**Why RAG specifically (justify this in the report, don't just say "we added
RAG"):** an LLM improvising first-aid advice is a real safety risk — it can
hallucinate incorrect medical guidance. RAG constrains the model to answer only
from retrieved, curated, trustworthy content.

**Why LangChain specifically here, and only here (not everywhere):** RAG —
document loading, chunking, embedding, retrieval, grounded generation — is the
textbook use case LangChain's abstractions were built for. Using it for this
sub-piece is a legitimate engineering choice, not a buzzword addition. It should
**not** be used to drive the dialogue-flow decision logic in Section 5 above —
that decision logic must stay in your own bounded state machine so the
conversation remains guided, not open-ended (see the caveat in Section 5).

### 6.1 Architecture

```
Curated first-aid corpus (WHO / Red Cross / trauma protocols)
        │
        ▼  DocumentLoader + TextSplitter
   Chunked documents
        │
        ▼  Embeddings (sentence-transformers)
   Vector Store (FAISS/Chroma)
        │
        ▼  Retriever (top-k similarity search, k=3-5)
   Retrieved chunks  ──┐
                        │
Incident type/severity ─┴──► Grounded-generation Prompt ──► LLM ──► Guidance
     (from Section 5)                                                    │
                                                                          ▼
                                                          Groundedness check
                                                       (Section 6.3 below)
```

### 6.2 LangChain Implementation Sketch

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load and chunk the curated first-aid corpus
loader = DirectoryLoader("./first_aid_corpus/", loader_cls=TextLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Embed and store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 3. Grounded-generation prompt — constrains the LLM to retrieved context ONLY
GROUNDED_PROMPT = ChatPromptTemplate.from_template("""
You are providing first-aid guidance based STRICTLY on the context below.
If the context does not clearly cover the situation described, say so
explicitly and recommend waiting for professional help — do NOT improvise
medical advice beyond what the context supports.

Context:
{context}

Situation: {question}

Guidance:
""")

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | GROUNDED_PROMPT
    | llm  # your chosen LLM API client wrapped as a LangChain-compatible model
    | StrOutputParser()
)

# Usage
guidance = rag_chain.invoke("Bystander reports heavy bleeding from a leg wound")
```

Built using LangChain Expression Language (LCEL, the `|` pipe syntax above) —
this is the current recommended LangChain pattern (not the older `RetrievalQA`
chain class, which is legacy).

### 6.3 Groundedness Verification (do not skip this)

LangChain gives you retrieval and generation — it does **not** automatically
guarantee the generated answer only used the retrieved content. You must verify
this yourself:

```python
def check_groundedness(answer: str, retrieved_chunks: list[str]) -> bool:
    """
    Simple lexical-overlap groundedness check for the report/demo.
    A more rigorous version could use a second LLM call to judge entailment
    (answer entailed by context) — start with this simpler version first.
    """
    combined_context = " ".join(retrieved_chunks).lower()
    answer_sentences = answer.lower().split(". ")
    grounded_count = sum(
        1 for s in answer_sentences
        if any(phrase in combined_context for phrase in s.split(",")[:1])
    )
    return grounded_count / max(len(answer_sentences), 1)
```

For evaluation (Section 8 below), run this check across your first-aid test set
and report the **% of generated answers that are fully grounded** — this is a
real, reportable metric, not a hand-wave claim of "we use RAG so it's safe."

### 6.4 Corpus, Vector Store, and Chunking Decisions

1. **Corpus:** curated first-aid/trauma protocol documents — WHO first-aid
   guidelines, Red Cross protocols, standard trauma triage instructions. Compile
   this manually from public, trustworthy sources. Keep the corpus scoped
   (bleeding control, CPR basics, fracture immobilization, shock management,
   burns) rather than trying to cover all of medicine.
2. **Chunk size (500 chars, 50 overlap above) is a starting point** — tune this
   based on how your source documents are structured; medical protocol steps
   are often short and procedural, so err toward smaller chunks that preserve
   one instruction per chunk rather than large chunks that blend multiple
   unrelated instructions.
3. **Embedding model:** `all-MiniLM-L6-v2` (via `sentence-transformers`) is a
   good default — small, fast, good enough quality for this scale of corpus.
   No need for a larger embedding model here.
4. **Vector store:** FAISS (in-memory, simplest to set up) or Chroma (adds
   persistence) — either is fine at this corpus scale; FAISS is the simpler
   starting choice.
5. **Traceability:** log which source document/chunk backed each piece of
   guidance given (the `format_docs` function above already tags each chunk
   with its source) — useful both for safety auditing and for demonstrating
   rigor in the viva.

## 7. Data / Corpus Sourcing

- WHO first-aid guidelines (publicly available)
- Red Cross first-aid protocols (publicly available)
- Standard trauma triage instructions (clinical literature, publicly available
  summaries)
- **Do not** scrape or use content you don't have clear rights to redistribute in
  an academic project — stick to publicly published guideline documents.

## 8. Evaluation

- **ASR:** word error rate (WER) on held-out multilingual test utterances (record
  your own test set — a few dozen sample reports per language is enough for a
  student project).
- **Dialogue quality:** does the guided dialogue correctly escalate follow-up
  questions when severity cues are present? Test with scripted scenarios
  (accident with visible bleeding → should ask about bleeding severity, not skip
  it).
- **RAG groundedness:** for a test set of first-aid queries, run the
  groundedness check from Section 6.3 (or a manual verification pass for a
  smaller test set) and report the % of answers that are fully grounded vs. any
  that drift. This is a required, reportable metric — not an assumption that
  follows automatically from "we used RAG."

## 9. Interface Contract With Other Components

**Output to Component 2 (Vitals/Fusion):** a text severity signal — either raw
transcribed+dialogue text, or a structured feature vector, depending on the
decision made in Section 5. Document whichever is chosen in
`05_coordinator_fusion_evaluation.md` once decided, so both components agree on
the interface.

**Output to Coordinator:** structured incident record — location (GPS), incident
type, initial severity estimate, timestamp.

## 10. Open Decisions (resolve before building, note answer here once decided)

- [ ] Exact LLM provider/API for dialogue + RAG generation (cost, latency,
  offline feasibility all matter — an ambulance may have poor connectivity).
- [ ] Whether Whisper runs on-device (edge) or via API — connectivity assumption
  affects this.
- [ ] Final language list for the demo (recommended: Kannada, Hindi, English).
- [ ] Exact interface format handed to Component 2 (raw text vs. structured
  vector).
