# FinanceRAG

A retrieval-augmented generation (RAG) system for querying corporate financial filings (10-K, 10-Q, earnings releases), evaluated against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

---

## Getting Started

### Prerequisites

- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)
- [FinanceBench](https://github.com/patronus-ai/financebench) dataset (PDFs + JSONL)

### Installation

```bash
git clone https://github.com/vshereperov/financerag.git
cd financerag
python -m venv .venv && .venv\Scripts\activate  # Unix: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your values
```

---

## Usage

**1. Ingest documents into the vector store:**
```bash
python -m src.ingest
```

**2. Ask a question:**
```bash
python -m src.cli "What was AMD's revenue in 2022?"
```

**3. Run evaluation:**
```bash
python -m src.eval
python -m src.eval --debug  # show per-question verdicts
```

---

## Evaluation

Metrics:

- **Retrieval hit-rate** — did the gold evidence page land in top-k
- **Answer correctness** — LLM-as-judge vs gold answers (judge model is
  stronger than the generator to avoid self-grading)
- **Faithfulness** — is the answer grounded in retrieved context, not hallucinated

---

## Results

The system currently reaches **63.0% answer correctness** on the
150-question FinanceBench benchmark.

> **Current pipeline:** page-level retrieval + query-aligned page summaries
> (doc2query) · hybrid retrieval · hosted reranker (`cohere/rerank-4-fast`) ·
> `text-embedding-3-large` · k=10

### Development

| #  | Configuration                              | Hit-rate@k | Correctness | Faithfulness |
|----|--------------------------------------------|-----------:|------------:|-------------:|
|    | **Phase 1 — evaluated on 34 questions**    |            |             |              |
| 1  | Dense, k=5 _(baseline)_                    |     29.4%  |      39.7%  |       88.2%  |
| 2  | + Reranker (jina, local)                   |     50.0%  |      44.1%  |       92.6%  |
| 3  | + Hybrid retrieval                         |     52.9%  |      48.5%  |       95.6%  |
| 4  | + k=10                                     |     55.9%  |      50.0%  |       85.3%  |
| 5  | + `text-embedding-3-large`                 |     61.8%  |      58.8%  |       79.4%  |
| 6  | + Page-level retrieval + summaries         |     64.7%  |      57.4%  |       80.9%  |
| 7  | + Hosted reranker (`cohere/rerank-4-fast`) |     79.4%  |      72.1%  |       94.1%  |
| 8  | + Query-aligned page summaries (doc2query) |     91.2%  |      70.6%  |       91.2%  |
|    | **Phase 2 — evaluated on 150 questions**   |            |             |              |
| 8  | + Query-aligned page summaries (doc2query) |     84.7%  |      63.0%  |       88.0%  |
