# AI Underwriting Agent V3.1

AI-assisted commercial property underwriting workbench built with **LangGraph, LLMs, RAG/Corrective RAG, deterministic risk rules, machine learning, and human-in-the-loop review**.

> **Portfolio / research prototype — not production underwriting software.** The bundled training data is synthetic and the risk thresholds are demonstration rules.

## Why this project?

Commercial underwriting requires reviewing heterogeneous submissions, extracting facts, checking historical losses, consulting underwriting guidelines, assessing exposure, and documenting a defensible recommendation.

V3.1 separates those responsibilities across specialized components instead of asking a single LLM to make the entire decision.

### Core design principle

**LLM understands → RAG grounds → rules calculate → ML predicts → LangGraph orchestrates → human decides.**

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     Streamlit UI    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      LangGraph      │
                         │  workflow + state   │
                         └──────────┬──────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                            │
       ▼                            ▼                            ▼
 ACORD / Submission             SOV / Loss Runs              Metadata
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                           ┌───────────────────┐
                           │    Extraction LLM │
                           └─────────┬─────────┘
                                     ▼
                           Structured submission
                                     │
                                     ▼
                              Deterministic
                                enrichment
                                     │
                                     ▼
                           ┌───────────────────┐
                           │   Supervisor LLM  │
                           │ research planning │
                           └─────────┬─────────┘
                                     ▼
                              Research queries
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │      RAG / Corrective RAG │
                       │ TF-IDF + cosine similarity│
                       └────────────┬──────────────┘
                                    ▼
                              Guideline evidence
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
             Feature Engine    Risk Rules         ML Model
                   │                │                │
                   └────────────────┼────────────────┘
                                    ▼
                           ┌───────────────────┐
                           │ Decision Synthesis│
                           │       LLM         │
                           └─────────┬─────────┘
                                     ▼
                           ┌───────────────────┐
                           │   Human Review    │
                           └─────────┬─────────┘
                                     ▼
                              Final Output
```

## End-to-end lifecycle

1. **Intake** — accepts the underwriting submission and normalizes workflow state.
2. **Extraction** — LLM converts unstructured application content into structured fields.
3. **Enrichment** — deterministic lookup logic adds industry/construction factors.
4. **Supervisor** — LLM creates targeted underwriting research questions.
5. **RAG retrieval** — searches internal guideline knowledge.
6. **Corrective RAG** — expands weak queries and retries retrieval when evidence is insufficient.
7. **Feature engineering** — creates model-ready underwriting features.
8. **Risk engine** — calculates a reproducible prototype risk score.
9. **ML predictor** — estimates referral probability.
10. **Decision synthesis** — LLM combines facts, evidence, rules, and ML signals into an explanation.
11. **Human review** — workflow pauses for an authorized human decision.
12. **Output** — produces the final underwriting result and supporting evidence.

---

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM framework | LangChain / LangChain Core |
| Local inference | Ollama |
| Local model | Llama 3.2 3B |
| Cloud LLM option | OpenAI API |
| RAG | TF-IDF + cosine similarity |
| RAG persistence | Joblib |
| ML | scikit-learn |
| ML algorithm | StandardScaler + Logistic Regression |
| Workflow persistence | SQLite checkpointing |
| Configuration | python-dotenv |
| Observability | LangSmith-compatible tracing |

---

## Project structure

```text
underwriting-agent-v3.1/
├── app/
│   ├── graph.py                 # LangGraph workflow
│   ├── llm.py                   # provider abstraction
│   ├── state.py                 # shared workflow state
│   ├── nodes/                   # specialized workflow nodes
│   ├── rag/                     # retrieval, feedback, reranker training
│   └── ml/                      # feature/model training + prediction
├── data/
│   ├── knowledge/               # underwriting guidelines
│   └── submissions/             # synthetic demo submissions
├── scripts/
│   ├── build_rag_index.py
│   ├── train_rag_reranker.py
│   ├── evaluate_batch.py
│   └── run_demo.py
├── streamlit_app.py
├── requirements.txt
├── requirements-optional.txt
├── .env.example
├── langgraph.json
├── MAC_M1_SETUP.md
└── LICENSE
```

---

## RAG implementation

### Knowledge ingestion

Guidelines are stored as Markdown/text under `data/knowledge/`.

The prototype uses **section-aware semantic chunking**. Markdown `##` sections become retrieval chunks, and document/section metadata is retained for traceability.

There is no fixed `chunk_size` / `chunk_overlap` configuration in the current prototype.

### Vectorisation

The current prototype uses a `TfidfVectorizer` to convert guideline chunks into sparse numerical vectors.

```text
Guideline text
     ↓
TF-IDF vectorization
     ↓
Sparse numerical vectors
     ↓
Persisted retrieval index
```

A research query generated by the Supervisor is transformed with the same vectorizer. Cosine similarity ranks the knowledge chunks.

### Corrective RAG

```text
Research query
     ↓
Initial retrieval
     ↓
Relevance check
     ↓
Weak evidence?
   ↙       ↘
 YES       NO
  ↓         ↓
Query       Use evidence
expansion
  ↓
Second retrieval
  ↓
Select stronger evidence
```

The system records corrective retrieval information in the workflow state for auditability.

### Important prototype limitation

This is **vectorised retrieval**, but it is not an external vector database and it is not dense semantic embedding retrieval. A production version could use embeddings + hybrid search + a persistent vector store such as pgvector, Qdrant, OpenSearch, Pinecone, or Weaviate.

---

## Feature engineering

The current referral model uses nine features:

```text
TIV
Claims count
Total incurred
Loss ratio
Industry factor
Construction factor
Combined multiplier
Year built
Sprinkler present
```

Loss ratio is calculated as:

```text
loss_ratio = total_incurred / TIV
```

The combined multiplier is derived from the deterministic enrichment factors:

```text
combined_multiplier = industry_factor × construction_factor
```

---

## Prototype risk score

The deterministic risk engine uses:

```text
base = min(loss_ratio × 100, 60)
mult = min(max((combined_multiplier - 1) × 40, 0), 40)
protection = -8 if sprinkler_present else +5

risk_score = clamp(base + mult + protection, 0, 100)
```

Current demonstration thresholds:

| Score | Prototype outcome |
|---:|---|
| `< 25` | Auto quote |
| `25–54.9` | Refer |
| `≥ 55` | Auto-decline candidate |

These thresholds are **demo rules only** and must not be treated as production underwriting policy.

---

## Machine learning

The ML layer is intentionally separate from the deterministic risk score.

```text
9 engineered features
        ↓
StandardScaler
        ↓
Logistic Regression
        ↓
Referral probability
```

The bundled 100-case dataset is synthetic. The training script does not require an LLM, making model training lightweight and deterministic.

Train locally:

```bash
python -m app.ml.train_demo
```

The generated model artifact is ignored by Git and can be recreated locally.

---

## LLM responsibilities

LLMs are used where language understanding adds value:

- Unstructured document extraction
- Supervisor/research planning
- Retrieved evidence synthesis
- Decision rationale generation
- Final human-readable output

The LLM is **not** used to perform the deterministic risk-score arithmetic or the Logistic Regression prediction.

---

## LLM providers

The gateway supports local Ollama inference and an OpenAI option.

### Local Ollama

```bash
ollama pull llama3.2:3b
```

`.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL_FAST=llama3.2:3b
OLLAMA_MODEL_REASONING=llama3.2:3b
OLLAMA_MODEL_SYNTHESIS=llama3.2:3b
OLLAMA_KEEP_ALIVE=0
```

### OpenAI

Install the optional provider:

```bash
pip install -r requirements-optional.txt
```

Then configure:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.5
```

Never commit `.env` or API keys.

---

## Installation

```bash
git clone <your-repository-url>
cd underwriting-agent-v3.1

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
```

For local Ollama mode:

```bash
ollama pull llama3.2:3b
```

Build the RAG index:

```bash
python scripts/build_rag_index.py
```

Train the demonstration ML model:

```bash
python -m app.ml.train_demo
```

Start the application:

```bash
streamlit run streamlit_app.py
```

---

## Smoke test without an LLM

For workflow/UI smoke testing, use:

```dotenv
FORCE_MOCK_LLM=true
```

This is useful when validating the graph on a constrained laptop or when Ollama is unavailable.

See `MAC_M1_SETUP.md` for local setup guidance.

---

## Observability

LangSmith tracing can be enabled through environment variables:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=underwriting-agent-v3
```

The goal is to make individual graph nodes, retrieval operations, and LLM calls inspectable during development.

---

## Human-in-the-loop

The graph intentionally pauses at the review gate before finalization.

```text
AI analysis
    ↓
Recommendation
    ↓
Human review
    ↓
Final decision
```

This design is important for a high-impact workflow where an AI recommendation should remain reviewable and auditable.

---

## RAG feedback loop

Underwriters can mark retrieved evidence as **Helpful** or **Incorrect** in the Streamlit UI.

Feedback is collected locally and can be used to train a lightweight reranker after enough labeled examples are available.

The current prototype establishes the feedback/training path; the trained reranker is not automatically applied to the primary retrieval path yet.

---

## Production roadmap

1. Replace synthetic data with sanitized historical underwriting outcomes.
2. Add document/OCR ingestion and validation.
3. Add embeddings and hybrid BM25 + vector retrieval.
4. Move RAG to a production vector store.
5. Integrate the feedback-trained reranker into retrieval.
6. Add retrieval/LLM evaluation datasets and regression tests.
7. Introduce model and prompt versioning.
8. Add authentication, authorization, secrets management, and encryption.
9. Add decision-level audit trails and guideline versioning.
10. Add ML calibration, drift monitoring, and time-based validation.
11. Deploy with scalable inference and observability.

---

## Project status

**V3.1 — Portfolio-ready prototype**

Implemented:

- LangGraph orchestration
- LLM extraction and research planning
- Persistent TF-IDF RAG
- Section-aware chunking
- Corrective retrieval
- RAG source visibility and feedback
- Deterministic feature engineering
- Prototype risk engine
- Logistic Regression referral model
- Human-in-the-loop review
- Ollama/OpenAI provider abstraction
- Streamlit UI
- SQLite workflow checkpointing
- LangSmith-compatible observability

Known limitations are intentionally documented rather than hidden.

---

## Disclaimer

This repository is an educational/portfolio prototype. It does not constitute insurance underwriting advice, production underwriting policy, or a production-ready automated decision system. Synthetic training data and demonstration risk thresholds are included solely for software demonstration.
