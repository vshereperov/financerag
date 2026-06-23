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

The final system reaches **66.2% answer correctness**.

Each change below was A/B-tested against the previous best configuration on the
same question set.

### Summary

The final configuration lifts answer correctness from **39.7% → 66.2%**
(a **+26.5 pt** gain) over the baseline:

| Metric        | Baseline | Final | Δ          |
|---------------|---------:|------:|:----------:|
| Hit-rate@k    |  29.4%   | 61.8% | **+32.4**  |
| Correctness   |  39.7%   | 66.2% | **+26.5**  |
| Faithfulness  |  89.7%   | 86.8% | −2.9       |

> **Baseline config:** dense retrieval · `text-embedding-3-small` · k=5
>
> **Final config:** page-level retrieval + summaries · hybrid retrieval ·
> reranker · `text-embedding-3-large` · k=10

### Per-change breakdown

| #  | Configuration                          | Hit-rate@k | Correctness | Faithfulness |
|----|----------------------------------------|-----------:|------------:|-------------:|
| 1  | Dense, k=5 _(baseline)_                |     29.4%  |      39.7%  |       89.7%  |
| 2  | + Reranker                             |     50.0%  |      44.1%  |       95.6%  |
| 3  | + Hybrid retrieval                     |     52.9%  |      51.5%  |       89.7%  |
| 4  | + k=10                                 |     55.9%  |      55.9%  |       86.8%  |
| 5  | + `text-embedding-3-large`             |     61.8%  |      58.8%  |       86.8%  |
| 6  | **+ Page-level retrieval + summaries** | **61.8%**  |  **66.2%**  |       86.8%  |

Each row is cumulative: it adds one change on top of the row above.

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
