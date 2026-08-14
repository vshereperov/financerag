from src import judge

# _parse: the gate between the judge's reply and every eval number


def test_parse_reads_a_plain_json_verdict():
    text = '{"reason": "same figure", "verdict": "correct"}'

    assert judge._parse(text, "incorrect") == {
        "reason": "same figure",
        "verdict": "correct",
    }


def test_parse_strips_a_markdown_code_fence():
    """Models wrap JSON in ```json fences even when asked not to."""
    text = '```json\n{"reason": "grounded", "verdict": "supported"}\n```'

    assert judge._parse(text, "unsupported")["verdict"] == "supported"


def test_parse_strips_surrounding_whitespace():
    text = '\n\n  {"reason": "r", "verdict": "partial"}  \n'

    assert judge._parse(text, "incorrect")["verdict"] == "partial"


def test_parse_falls_back_when_the_output_is_not_json():
    """A parse failure counts as a bad verdict, so it shows up as low quality in the
    report rather than crashing the run. The reason keeps the raw text for debugging."""
    result = judge._parse("The answer looks right to me.", "incorrect")

    assert result["verdict"] == "incorrect"
    assert "unparseable judge output" in result["reason"]
    assert "The answer looks right to me." in result["reason"]


def test_parse_uses_the_fallback_it_is_given():
    """correctness() and faithfulness() have different verdict vocabularies."""
    assert judge._parse("nonsense", "unsupported")["verdict"] == "unsupported"


def test_parse_truncates_a_long_unparseable_reply():
    result = judge._parse("x" * 500, "incorrect")

    assert len(result["reason"]) < 120
