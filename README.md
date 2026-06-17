# FinanceRAG

A retrieval-augmented generation (RAG) system for querying SEC 10-K filings, evaluated against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

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

The system is evaluated against FinanceBench with a reproducible harness
(deterministic, `temperature=0`):

- **Retrieval hit-rate** — did the gold evidence page land in top-k
- **Answer correctness** — LLM-as-judge vs gold answers (judge model is
  stronger than the generator to avoid self-grading)
- **Faithfulness** — is the answer grounded in retrieved context, not hallucinated

---

## Corpus

5 companies selected by 10-K question coverage in FinanceBench, with sector diversity:

| Company | 10-K questions | Sector |
|---|---|---|
| AMD | 8 | Semiconductors |
| Boeing | 8 | Aerospace / industrial |
| American Express | 7 | Financial services |
| PepsiCo | 6 | Consumer staples |
| 3M | 5 | Industrial conglomerate |

34 benchmark questions total.
