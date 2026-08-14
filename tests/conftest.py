import os

os.environ["OPENROUTER_API_KEY"] = "test-key-not-used"
os.environ["FINANCEBENCH_DIR"] = "tests/fixtures"

import pytest

from src import usage


@pytest.fixture(autouse=True)
def clear_usage():
    """usage keeps counters in a module-level dict, so reset it between tests."""
    usage.reset()
    yield
    usage.reset()
