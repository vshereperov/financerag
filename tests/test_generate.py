from types import SimpleNamespace

from src import generate, usage


def point(company="AMD", doc_name="AMD_2022_10K", page=42, content="Total assets..."):
    return SimpleNamespace(
        payload={
            "company": company,
            "doc_name": doc_name,
            "page": page,
            "content": content,
        }
    )


# build_context: what the generator sees, and what it can cite


def test_build_context_labels_a_page_with_its_company_document_and_page():
    """The system prompt asks for (company, document, page) citations, so those
    three have to reach the model in the context."""
    context = generate.build_context([point(content="Total assets were $67,580M.")])

    assert context == "[AMD | AMD_2022_10K | page 42]\nTotal assets were $67,580M."


def test_build_context_separates_pages_so_they_cannot_run_together():
    context = generate.build_context(
        [point(page=1, content="first"), point(page=2, content="second")]
    )

    assert context == (
        "[AMD | AMD_2022_10K | page 1]\nfirst"
        "\n\n---\n\n"
        "[AMD | AMD_2022_10K | page 2]\nsecond"
    )


def test_build_context_of_nothing_is_empty():
    assert generate.build_context([]) == ""


# generate_answer: thin over the API, but the cost accounting is ours


class FakeCompletions:
    def __init__(self, content, usage_obj):
        self.content = content
        self.usage = usage_obj
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=self.usage,
        )


class FakeClient:
    """Stands in for the OpenAI client so no request ever leaves the test."""

    def __init__(self, content="The answer.", usage_obj=None):
        self.completions = FakeCompletions(content, usage_obj)
        self.chat = SimpleNamespace(completions=self.completions)


def test_generate_answer_passes_the_question_and_context_to_the_model(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(generate, "client", fake)

    answer = generate.generate_answer("What were total assets?", [point()])

    assert answer == "The answer."
    messages = fake.completions.calls[0]["messages"]
    assert messages[0]["content"] == generate.SYSTEM_PROMPT
    assert "What were total assets?" in messages[1]["content"]
    assert "[AMD | AMD_2022_10K | page 42]" in messages[1]["content"]


def test_generate_answer_records_the_token_usage(monkeypatch):
    usage_obj = SimpleNamespace(prompt_tokens=900, completion_tokens=120, cost=0.002)
    monkeypatch.setattr(generate, "client", FakeClient(usage_obj=usage_obj))

    generate.generate_answer("q", [point()])

    stats = usage._usage[generate.settings.generator_model]
    assert stats == {"input": 900, "output": 120, "calls": 1, "cost": 0.002}
