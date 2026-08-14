from types import SimpleNamespace

import pytest

from src import ingest

# _fiscal_year


@pytest.mark.parametrize(
    "doc_name,expected",
    [
        ("AMD_2022_10K", "2022"),
        ("3M_2023Q2_10Q", "2023"),
        ("COSTCO_2021_10K", "2021"),
        ("JOHNSON_JOHNSON_2022_8K", "2022"),
        ("NO_YEAR_HERE_10K", ""),
        ("", ""),
    ],
)
def test_fiscal_year(doc_name, expected):
    assert ingest._fiscal_year(doc_name) == expected


# _summarize: talks to an LLM, so the client gets replaced


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    """Stands in for the OpenAI client so no request ever leaves the test."""

    def __init__(self, content: str | None = "  Balance sheet page.  "):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def test_summarize_skips_the_llm_on_a_tiny_page(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(ingest, "llm", fake)
    page = {"company": "AMD", "doc_name": "AMD_2022_10K", "content": "Page 14"}

    assert ingest._summarize(page) == "Page 14"
    assert fake.completions.calls == []


def test_summarize_returns_the_stripped_llm_answer(monkeypatch):
    fake = FakeLLM("  Consolidated balance sheets.  ")
    monkeypatch.setattr(ingest, "llm", fake)
    page = {
        "company": "AMD",
        "doc_name": "AMD_2022_10K",
        "content": "Total assets were $67,580 million as of December 31, 2022.",
    }

    assert ingest._summarize(page) == "Consolidated balance sheets."


def test_summarize_sends_company_and_fiscal_year_to_the_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(ingest, "llm", fake)
    page = {
        "company": "AMD",
        "doc_name": "AMD_2022_10K",
        "content": "Total assets were $67,580 million as of December 31, 2022.",
    }

    ingest._summarize(page)

    kwargs = fake.completions.calls[0]
    user_message = kwargs["messages"][1]["content"]
    assert "Company: AMD" in user_message
    assert "Fiscal year: FY2022" in user_message


def test_summarize_returns_empty_string_when_the_llm_returns_nothing(monkeypatch):
    """A reasoning model that spends its budget on thinking can return content=None."""
    monkeypatch.setattr(ingest, "llm", FakeLLM(None))
    page = {
        "company": "AMD",
        "doc_name": "AMD_2022_10K",
        "content": "Total assets were $67,580 million as of December 31, 2022.",
    }

    assert ingest._summarize(page) == ""


# summarize_pages: mock out _summarize, test the wiring around it


def test_summarize_pages_anchors_embed_text_with_company_and_year(monkeypatch):
    monkeypatch.setattr(ingest, "_summarize", lambda page: "Cash flow statement.")
    pages = [{"company": "AMD", "doc_name": "AMD_2022_10K", "content": "..."}]

    ingest.summarize_pages(pages)

    assert pages[0]["summary"] == "Cash flow statement."
    assert pages[0]["embed_text"] == "AMD FY2022. Cash flow statement."


def test_summarize_pages_omits_the_year_when_the_doc_name_has_none(monkeypatch):
    monkeypatch.setattr(ingest, "_summarize", lambda page: "Cover page.")
    pages = [{"company": "AMD", "doc_name": "AMD_10K", "content": "..."}]

    ingest.summarize_pages(pages)

    assert pages[0]["embed_text"] == "AMD. Cover page."


def test_summarize_pages_keeps_summaries_aligned_with_their_pages(monkeypatch):
    """pool.map must not shuffle results: page N's summary has to land on page N."""
    monkeypatch.setattr(ingest, "_summarize", lambda page: f"summary of {page['page']}")
    pages = [
        {"company": "AMD", "doc_name": "AMD_2022_10K", "page": n, "content": "..."}
        for n in range(1, 21)
    ]

    ingest.summarize_pages(pages)

    for page in pages:
        assert page["summary"] == f"summary of {page['page']}"
