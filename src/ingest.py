from pathlib import Path
import fitz
from .config import FINANCEBENCH_DIR, QDRANT_COLLECTION
from .embed import embed_texts, embed_sparse_docs
from .store import init_collection, upsert_chunks, client

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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH = 100


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into chunks of a specified size with a specified overlap."""
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


def parse_doc(company, doc_name):
    """Parses a single PDF document and returns a list of chunks with metadata."""
    pdf_path = Path(FINANCEBENCH_DIR) / "pdfs" / f"{doc_name}.pdf"
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(1, doc.page_count + 1):
        page = doc.load_page(page_num - 1)
        for piece in chunk_text(page.get_text()):
            chunks.append(
                {
                    "company": company,
                    "doc_name": doc_name,
                    "page": page_num,
                    "content": piece,
                }
            )
    doc.close()
    return chunks


def parse_all():
    """Parses all documents and returns a list of all chunks with metadata."""
    all_chunks = []
    for company, doc_name in DOCS:
        chunks = parse_doc(company, doc_name)
        print(f"{doc_name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"TOTAL: {len(all_chunks)} chunks")
    return all_chunks


def ingest():
    """Main function to parse documents, embed chunks, and upsert into Qdrant."""
    chunks = parse_all()
    init_collection()
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        texts = [c["content"] for c in batch]
        dense_vectors = embed_texts(texts)
        sparse_vectors = embed_sparse_docs(texts)
        upsert_chunks(batch, dense_vectors, sparse_vectors, start_id=start)
        print(f"ingested {start + len(batch)} / {len(chunks)}")
    print("done:", client.count(QDRANT_COLLECTION))


if __name__ == "__main__":
    ingest()
