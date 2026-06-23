from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz
from openai import OpenAI

from .config import settings
from .embed import embed_texts, embed_sparse_docs
from .store import init_collection, upsert_pages, client

# Documents to ingest: (company, document name)
DOCS = [
    ("3M", "3M_2018_10K"),
    ("3M", "3M_2022_10K"),
    ("AMD", "AMD_2015_10K"),
    ("AMD", "AMD_2022_10K"),
    ("American Express", "AMERICANEXPRESS_2022_10K"),
    ("Boeing", "BOEING_2018_10K"),
    ("Boeing", "BOEING_2022_10K"),
    ("PepsiCo", "PEPSICO_2021_10K"),
    ("PepsiCo", "PEPSICO_2022_10K"),
]

MIN_CHARS = 40  # Skip pages with fewer chars than this when summarizing
BATCH = 100  # Embed and write this many pages to Qdrant at a time
SUMMARY_CONCURRENCY = 8  # Run this many page-summary LLM calls in parallel

llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)

SUMMARY_SYSTEM = (
    "You compress a single page of a company's 10-K filing into a dense "
    "retrieval summary. In 2-4 sentences, state which topics, financial "
    "statements, line items, events, or disclosures appear on the page, using "
    "the terminology a financial analyst would search with (e.g. 'consolidated "
    "statements of cash flows', 'risk factors', 'segment operating results', "
    "'revenue recognition policy'). Do not invent anything not on the page and "
    "do not editorialize. Output only the summary."
)


def _summarize(text):
    """Return a retrieval summary for one page, falls back to the text on tiny pages."""
    if len(text) < MIN_CHARS:
        return text
    response = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=180,
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
    """Attach an LLM summary to every page record, in parallel."""
    texts = [p["content"] for p in pages]
    with ThreadPoolExecutor(max_workers=SUMMARY_CONCURRENCY) as pool:
        for page, summary in zip(pages, pool.map(_summarize, texts)):
            page["summary"] = summary


def parse_all():
    """Parse and summarize all documents into page records."""
    all_pages = []
    for company, doc_name in DOCS:
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
        dense_vectors = embed_texts([p["summary"] for p in batch])
        sparse_vectors = embed_sparse_docs([p["content"] for p in batch])
        upsert_pages(batch, dense_vectors, sparse_vectors, start_id=start)
        print(f"ingested {start + len(batch)} / {len(pages)}")
    print("done:", client.count(settings.qdrant_collection))


if __name__ == "__main__":
    ingest()
