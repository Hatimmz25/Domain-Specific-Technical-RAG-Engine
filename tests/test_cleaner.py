import pytest
from app.ingestion.cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_cleaner_basic_prose(cleaner):
    raw = "  FastAPI   is a modern   framework.  \n\n\n\nIt is fast.  "
    cleaned = cleaner.clean(raw)
    assert cleaned == "FastAPI is a modern framework.\n\nIt is fast."


def test_cleaner_preserves_code_blocks(cleaner):
    raw = (
        "Here is a Python function:\n\n"
        "```python\n"
        "def hello():\n"
        "    print('world')\n"
        "```\n\n"
        "End of function."
    )
    cleaned = cleaner.clean(raw)
    assert "def hello():" in cleaned
    assert "print('world')" in cleaned
    assert "```python" in cleaned


def test_cleaner_strips_html_tags(cleaner):
    raw = "<p>FastAPI provides <b>automatic</b> OpenAPI documentation.</p>"
    cleaned = cleaner.clean(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "FastAPI provides automatic OpenAPI documentation." in cleaned