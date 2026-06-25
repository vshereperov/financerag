import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz
from openai import OpenAI

from .config import settings
from .embed import embed_texts, embed_sparse_docs
from .store import init_collection, upsert_pages, client


MIN_CHARS = 40  # Skip pages with fewer chars than this when summarizing
BATCH = 100  # Embed and write this many pages to Qdrant at a time
SUMMARY_CONCURRENCY = 8  # Run this many page-summary LLM calls in parallel

llm = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key
)

SUMMARY_SYSTEM_PROMPT = (
    "You write a retrieval index entry for one page of a company's financial "
    "filing or report (10-K, 10-Q, earnings release, and similar). The entry is "
    "embedded and matched against financial-analyst questions. A separate "
    "keyword index already covers exact wording, so capture the meaning an "
    "analyst searches for, not the page verbatim.\n\n"
    "Part 1 - one or two sentences naming the specific section or financial "
    "statement(s) on the page and the concrete line items, segments, metrics, "
    "events or disclosures it contains, in the analyst's terminology (prefer "
    "the analyst's term when it differs from the page, e.g. 'capital "
    "expenditures' for 'purchases of property, plant and equipment').\n\n"
    "Part 2 - 3-6 questions THIS page answers, each specific to the content that "
    "distinguishes this page from other pages of the same filing: name the "
    'statement, segment or line item (e.g. "What drove the change in the Data '
    "Center segment's operating income?\", 'What were total cash flows from "
    "financing activities?'). Avoid generic questions answerable from any page. "
    "The company and fiscal year are implied - do not repeat them.\n\n"
    "Use only facts on the page; do not invent line items, numbers or topics, "
    "and do not answer the questions. If the page has no substantive financial "
    "or business content (cover, signatures, exhibit index), describe what it is "
    "in one sentence and write no questions. Output only the entry."
)


def _fiscal_year(doc_name):
    """Pull the fiscal year out of a doc name (e.g. 'AMD_2022_10K', '3M_2023Q2_10Q')."""
    for part in doc_name.split("_"):
        if part[:4].isdigit():
            return part[:4]
    return ""


def _summarize(page):
    """Return a retrieval summary for one page; falls back to the text on tiny pages."""
    text = page["content"]
    if len(text) < MIN_CHARS:
        return text
    context = (
        f"Company: {page['company']}\n"
        f"Fiscal year: FY{_fiscal_year(page['doc_name'])}\n\n"
        f"PAGE:\n{text}"
    )
    response = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return (response.choices[0].message.content or "").strip()


def parse_pages(company, doc_name):
    """Return one record per non-empty page with its full text."""
    pdf_path = Path(settings.financebench_dir) / "pdfs" / f"{doc_name}.pdf"
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(1, doc.page_count + 1):
        raw = doc.load_page(page_num - 1).get_text()
        text = raw.strip() if isinstance(raw, str) else ""
        if text:
            pages.append(
                {
                    "company": company,
                    "doc_name": doc_name,
                    "page": page_num,
                    "content": text,
                }
            )
    doc.close()
    return pages


def summarize_pages(pages):
    """Summarize every page in parallel and set its `summary` and `embed_text`."""
    with ThreadPoolExecutor(max_workers=SUMMARY_CONCURRENCY) as pool:
        summaries = pool.map(_summarize, pages)
    for page, summary in zip(pages, summaries):
        fy = _fiscal_year(page["doc_name"])
        anchor = f"{page['company']} FY{fy}." if fy else f"{page['company']}."
        page["summary"] = summary
        page["embed_text"] = f"{anchor} {summary}"


def _load_docs():
    """Return every (company, doc_name) pair in the benchmark JSONL."""
    jsonl = Path(settings.financebench_dir) / "data" / "financebench_open_source.jsonl"
    pairs = set()
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                pairs.add((record["company"], record["doc_name"]))
    return sorted(pairs)


def parse_all():
    """Parse and summarize all documents into page records."""
    all_pages = []
    for company, doc_name in _load_docs():
        pages = parse_pages(company, doc_name)
        print(f"{doc_name}: {len(pages)} pages")
        all_pages.extend(pages)
    print(f"TOTAL: {len(all_pages)} pages -> summarizing with {settings.llm_model}")
    summarize_pages(all_pages)
    return all_pages


def ingest():
    """Parse documents, summarize, embed each page, and upsert into Qdrant."""
    pages = parse_all()
    init_collection()
    for start in range(0, len(pages), BATCH):
        batch = pages[start : start + BATCH]
        dense_vectors = embed_texts([p["embed_text"] for p in batch])
        sparse_vectors = embed_sparse_docs([p["content"] for p in batch])
        upsert_pages(batch, dense_vectors, sparse_vectors, start_id=start)
        print(f"ingested {start + len(batch)} / {len(pages)}")
    print("done:", client.count(settings.qdrant_collection))


if __name__ == "__main__":
    ingest()
