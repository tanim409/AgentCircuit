"""
Drives workflow.research() through repeated, near-identical queries to
verify the three-state breaker (closed -> open -> half_open -> raise)
fires correctly. All Tavily/LLM calls are mocked so this runs fully
offline with no API keys or network access needed.
"""
import itertools

import pytest

import workflow
from Schema import EvidencePackage, EvidenceItem, circuitBacker

REPEATED_QUERY = ["what is transformer architecture"]
BASE_STATE = {
    "as_of": "2026-08-15",
    "recency_days": 3650,
    "mode": "closed_book",
    "evidence": [],
}


@pytest.fixture(autouse=True)
def mock_external_calls(monkeypatch):
    """Replace Tavily search and structured-output LLM calls with
    deterministic fakes so research() can run without any real
    network/API access, and so evidence grows a known amount per call."""

    monkeypatch.setattr(
        workflow,
        "tavily_search",
        lambda query, max_result=5: [
            {
                "title": "raw result",
                "url": "https://example.com/raw",
                "snippet": "snippet",
                "published_at": None,
                "source": "tavily",
            }
        ],
    )

    counter = itertools.count()

    def fake_invoked_structured_output(llm, schema, messages, max_retries):
        if schema is EvidencePackage:
            n = next(counter)
            return EvidencePackage(
                evidence=[
                    EvidenceItem(
                        title=f"Evidence {n}",
                        url=f"https://example.com/evidence-{n}",
                        published_at=None,
                    )
                ]
            )
        raise AssertionError(f"Unexpected schema requested in test: {schema}")

    monkeypatch.setattr(workflow,
                         "invoked_structured_output",
                         fake_invoked_structured_output)

    monkeypatch.setattr(
        workflow.embeddings,
        "embed_query",
        lambda text: (
            [1.0, 0.0, 0.0]
            if text == "query one"
            else [0.0, 1.0, 0.0]
        ),
    )


def _run(state):

    result = workflow.research(state)
    return {**state, **result}


def test_repeated_identical_queries_trip_the_breaker_through_all_states():
    state = {**BASE_STATE, "queries": REPEATED_QUERY}


    state = _run(state)
    assert state["research_status"] == "ok"
    assert state["consecutive_repeats"] == 1
    assert state["breaker_state"] == "closed"

    state = _run(state)
    assert state["research_status"] == "ok"
    assert state["consecutive_repeats"] == 2


    state = _run(state)
    assert state["breaker_state"] == "open"
    assert state["research_status"] == "loop_warning"


    state["queries"] = REPEATED_QUERY
    state = _run(state)
    assert state["breaker_state"] == "half_open"
    assert state["research_status"] == "loop_warning_final"


    state["queries"] = REPEATED_QUERY
    with pytest.raises(circuitBacker):
        _run(state)


def test_non_repeating_queries_never_trip_the_breaker():
    state = {**BASE_STATE, "queries": ["query one"]}
    state = _run(state)
    assert state["consecutive_repeats"] == 1

    state["queries"] = ["a completely different query"]
    state = _run(state)
    assert state["consecutive_repeats"] == 1
    assert state["breaker_state"] == "closed"
    assert state["research_status"] == "ok"
