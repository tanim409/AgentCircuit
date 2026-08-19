import numpy as np

from helper import hash_tool_call
from workflow import normalize_url, _cosine_similarity


class TestNormalizeUrl:
    def test_strips_www(self):
        assert normalize_url("https://www.example.com/page") == normalize_url(
            "https://example.com/page"
        )

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/page/") == normalize_url(
            "https://example.com/page"
        )

    def test_case_insensitive(self):
        assert normalize_url("HTTPS://Example.COM/Page") == normalize_url(
            "https://example.com/page"
        )

    def test_empty_or_none_returns_empty_string(self):
        assert normalize_url("") == ""
        assert normalize_url(None) == ""

    def test_different_paths_stay_different(self):
        a = normalize_url("https://example.com/a")
        b = normalize_url("https://example.com/b")
        assert a != b


class TestHashToolCall:
    def test_same_args_same_hash(self):
        h1 = hash_tool_call("tavily_search", {"query": "LangGraph tutorial"})
        h2 = hash_tool_call("tavily_search", {"query": "LangGraph tutorial"})
        assert h1 == h2

    def test_different_args_different_hash(self):
        h1 = hash_tool_call("tavily_search", {"query": "LangGraph tutorial"})
        h2 = hash_tool_call("tavily_search", {"query": "LangChain tutorial"})
        assert h1 != h2

    def test_key_order_does_not_affect_hash(self):
        h1 = hash_tool_call("tool", {"a": 1, "b": 2})
        h2 = hash_tool_call("tool", {"b": 2, "a": 1})
        assert h1 == h2

    def test_different_tool_name_different_hash(self):
        h1 = hash_tool_call("tool_a", {"query": "x"})
        h2 = hash_tool_call("tool_b", {"query": "x"})
        assert h1 != h2


class TestCosineSimilarity:
    def test_identical_vectors_similarity_one(self):
        v = [1.0, 2.0, 3.0]
        assert np.isclose(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors_similarity_zero(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert np.isclose(_cosine_similarity(v1, v2), 0.0)

    def test_opposite_vectors_similarity_negative_one(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert np.isclose(_cosine_similarity(v1, v2), -1.0)

    def test_near_duplicate_queries_score_above_threshold(self):

        v1 = [1.0, 0.1, 0.0]
        v2 = [1.0, 0.11, 0.0]
        assert _cosine_similarity(v1, v2) > 0.85
