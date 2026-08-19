from workflow import route_next, research_router


class TestRouteNext:
    def test_routes_to_research_when_needed(self):
        state = {"need_research": True}
        assert route_next(state) == "research"

    def test_routes_to_orchestrator_when_not_needed(self):
        state = {"need_research": False}
        assert route_next(state) == "orchestrator"


class TestResearchRouter:
    def test_routes_to_orchestrator_on_ok_status(self):
        state = {"research_status": "ok", "breaker_state": "closed"}
        assert research_router(state) == "orchestrator"

    def test_routes_back_to_router_on_open_breaker_with_warning(self):
        state = {"research_status": "loop_warning", "breaker_state": "open"}
        assert research_router(state) == "router"

    def test_routes_back_to_router_on_half_open_breaker_with_final_warning(self):
        state = {"research_status": "loop_warning_final", "breaker_state": "half_open"}
        assert research_router(state) == "router"

    def test_does_not_retry_if_breaker_is_closed_even_with_warning_status(self):
        state = {"research_status": "loop_warning", "breaker_state": "closed"}
        assert research_router(state) == "orchestrator"

    def test_defaults_to_orchestrator_when_keys_missing(self):
        assert research_router({}) == "orchestrator"
