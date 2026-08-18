import pytest
from pydantic import ValidationError

from Schema import Task, Plan, RouterDecision, EvidenceItem, Citation, SectionOutput


class TestTask:
    def test_valid_task(self):
        t = Task(
            id=1,
            title="Intro",
            goal="Reader understands the topic.",
            bullets=["point one", "point two", "point three"],
            target_words=750,
        )
        assert t.id == 1
        assert len(t.bullets) == 3

    def test_rejects_fewer_than_min_bullets(self):
        with pytest.raises(ValidationError):
            Task(
                id=1,
                title="Intro",
                goal="Reader understands the topic.",
                bullets=["only one"],
                target_words=750,
            )

    def test_rejects_more_than_max_bullets(self):
        with pytest.raises(ValidationError):
            Task(
                id=1,
                title="Intro",
                goal="Reader understands the topic.",
                bullets=["a", "b", "c", "d", "e", "f"],
                target_words=750,
            )

    def test_default_flags_are_false(self):
        t = Task(
            id=1,
            title="Intro",
            goal="goal",
            bullets=["a", "b", "c"],
            target_words=750,
        )
        assert t.requires_research is False
        assert t.requires_citations is False
        assert t.requires_code is False


class TestPlan:
    def _make_task(self, task_id=1):
        return Task(
            id=task_id,
            title="Section",
            goal="goal",
            bullets=["a", "b", "c"],
            target_words=750,
        )

    def test_valid_plan(self):
        p = Plan(
            blog_title="Title",
            audience="Developers",
            tone="practical",
            narrative_thread="thread",
            tasks=[self._make_task()],
        )
        assert p.blog_kind == "explainer"  # default

    def test_rejects_invalid_blog_kind(self):
        with pytest.raises(ValidationError):
            Plan(
                blog_title="Title",
                audience="Developers",
                tone="practical",
                blog_kind="not_a_real_kind",
                narrative_thread="thread",
                tasks=[self._make_task()],
            )


class TestRouterDecision:
    def test_valid_modes(self):
        for mode in ("closed_book", "hybrid", "open_book"):
            d = RouterDecision(need_research=True, mode=mode, queries=["q"])
            assert d.mode == mode

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            RouterDecision(need_research=True, mode="made_up_mode", queries=[])

    def test_queries_default_empty_list(self):
        d = RouterDecision(need_research=False, mode="closed_book")
        assert d.queries == []


class TestEvidenceItem:
    def test_requires_title_and_url(self):
        with pytest.raises(ValidationError):
            EvidenceItem(published_at=None)

    def test_valid_minimal(self):
        e = EvidenceItem(title="Title", url="https://example.com", published_at=None)
        assert e.snippet is None


class TestSectionOutputAndCitation:
    def test_valid_section_with_citation(self):
        s = SectionOutput(
            content="Some content [1]",
            citations=[Citation(marker_id=1, source_url="https://example.com")],
        )
        assert s.citations[0].marker_id == 1

    def test_empty_citations_allowed(self):
        s = SectionOutput(content="No citations here", citations=[])
        assert s.citations == []
