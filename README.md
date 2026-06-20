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

## Results

Measured on 34 FinanceBench questions over 9 SEC 10-K filings. Each retrieval
change was A/B-tested.

| Config                  | hit-rate@5 | correctness | faithfulness |
|-------------------------|-----------:|------------:|-------------:|
| dense (baseline)        | 29.4%      | 39.7%       | 89.7%        |
| hybrid (dense + BM25)   | 26.5%      | 30.9%       | 98.5%        |
| dense + reranker        | 50.0%      | 44.1%       | 95.6%        |
| **hybrid + reranker**   | **52.9%**  | **51.5%**   | 89.7%        |

Notes:

- **Reranker model choice was decisive.** A 512-token cross-encoder
  (ms-marco-MiniLM) made retrieval *worse* than baseline by truncating long
  table chunks; a 1024-token model (jina-reranker-v2) nearly doubled hit-rate.
  This points to chunk length as the next bottleneck.
- **Best config is hybrid + reranker:** the reranker rescues
  hybrid's good recall from its poor ranking.
- Reranking trades latency for quality (~12 s/query on CPU); production would
  move the reranker to GPU or a hosted rerank API.

---

## Corpus

The corpus is built from SEC 10-K filings of 5 companies, evaluated against
the open-source FinanceBench benchmark.

Companies were not chosen at random. From the 150-question open FinanceBench set,
the 10-K questions per company
were counted, and the 5 companies with the highest coverage were selected,
while keeping sector diversity:

| Company           | 10-K questions | Sector                    |
|-------------------|----------------|---------------------------|
| AMD               | 8              | Semiconductors            |
| Boeing            | 8              | Aerospace / industrial    |
| American Express  | 7              | Financial services        |
| PepsiCo           | 6              | Consumer staples          |
| 3M                | 5              | Industrial conglomerate   |

This gives 34 benchmark questions total.
