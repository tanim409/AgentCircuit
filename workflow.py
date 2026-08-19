import os
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
import re
from urllib.parse import urlparse, urlunparse
import numpy as np
from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from Schema import *
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv
from Prompt import ORCH_SYSTEM, RESEARCH_SYSTEM, WORKER_SYSTEM, EDITOR_SYSTEM, DECIDE_IMAGES_SYSTEM, ROUTE_SYSTEM
from langgraph.checkpoint.memory import MemorySaver

in_memory = MemorySaver()
from helper import invoked_structured_output, hash_tool_call, log_breaker_trip

load_dotenv()

llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=8192
)
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

editor_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


def router(state: State):
    print("In Router")
    topic = state['topic']
    as_of = state['as_of']
    research_status = state.get('research_status', 'ok')
    correction_note = ""

    if research_status in ("loop_warning", "loop_warning_final"):
        correction_note = (
            "\n\nIMPORTANT: Previous search queries were too repetitive and made no progress. "
            "Generate DIFFERENT, more specific queries this time."
        )

    decision = invoked_structured_output(
        llm, RouterDecision,
        [
            SystemMessage(content=ROUTE_SYSTEM),
            HumanMessage(content=f"Topic: {topic} \n as_of : {as_of}{correction_note}")
        ],
        max_retries=3
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650
    print("Exit Router")
    return {
        "need_research": decision.need_research,
        "queries": decision.queries,
        "mode": decision.mode,
        "recency_days": recency_days,
    }


def route_next(state: State):
    print("In route_next")
    status = state.get("research_status", "ok")
    breaker_state = state.get("breaker_state", "closed")
    if state['need_research']:
        return "research"
    else:
        return "orchestrator"


def normalize_url(url: str):
    if not url:
        return ""

    url = url.lower()
    if "://" in url:
        protocol, rest = url.split("://", 1)
    else:
        rest = url

    # Remove www. if present
    if rest.startswith("www."):
        rest = rest[4:]

    # Remove trailing slash
    rest = rest.rstrip("/")

    # Reconstruct URL
    return f"{protocol}://{rest}" if "://" in url else rest


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def research_router(state: State):
    print("In research_route")
    status = state.get("research_status", "ok")
    breaker_state = state.get("breaker_state", "closed")
    if status in ("loop_warning", "loop_warning_final") and breaker_state != "closed":
        return "router"
    return "orchestrator"


ESTIMATED_COST_PER_QUERY = 0.008
ESTIMATED_COST_PER_LLM_CALL = 0.01


def research(state: State):
    print("In research")
    queries = state.get('queries', [])[:10]

    if isinstance(queries, list):
        flattened_queries = []
        for q in queries:
            if isinstance(q, list):
                flattened_queries.extend([str(item) for item in q if isinstance(item, str)])
            elif isinstance(q, str):
                flattened_queries.append(q)
        query_text = " ".join(flattened_queries)
    else:
        query_text = str(queries)

    current_vector = embeddings.embed_query(query_text) if query_text.strip() else None
    current_hash = hash_tool_call("tavily_search", {"queries": queries})
    history = state.get('tool_call_history', [])
    cost_history = state.get("cost_history", [])

    is_duplicate = False
    if history:
        last_entry = history[-1]
        last_vector = last_entry.get("embedding")
        if current_hash == last_entry.get("hash"):
            is_duplicate = True
        elif last_vector is not None and current_vector is not None:
            similarity = _cosine_similarity(current_vector, last_vector)
            if similarity > 0.85:
                is_duplicate = True

    if is_duplicate:
        repeats = state.get('consecutive_repeats', 0) + 1
    else:
        repeats = 1

    update_history = history + [{"tool": "tavily_search", "hash": current_hash, "embedding": current_vector}]

    tracking_update = {
        "consecutive_repeats": repeats,
        "tool_call_history": update_history
    }

    # if repeats >= 3:
    #     raise circuitBacker(
    #         f"Loop detected: research() called {repeats}x in a row with identical/near-identical queries."
    #     )

    existing_evidence = state.get("evidence", [])
    max_result = 5
    raw_result: List[dict] = []
    progress_history = state.get("progress_history", [])
    stagnant_steps = state.get("stagnant_steps", 0)
    try:
        if not queries:
            stagnant_steps += 1
            stagnation_update = {
                "progress_history": progress_history + [len(existing_evidence)],
                "stagnant_steps": stagnant_steps,
                "cost_history": cost_history + [0.0],
            }
            if stagnant_steps >= 3:
                log_breaker_trip(f"Stagnation: {stagnant_steps} steps with no queries.",
                                 state,
                                 extra={**stagnation_update, **tracking_update})
                raise circuitBacker(f"Stagnation: {stagnant_steps} steps with no queries and no progress.")
            return {
                'evidence': existing_evidence,
                'research_status': 'no_results',
                **tracking_update,
                **stagnation_update
            }

        turn_cost = len(queries) * ESTIMATED_COST_PER_QUERY + ESTIMATED_COST_PER_LLM_CALL
        new_cost = cost_history + [turn_cost]

        for query in queries:
            if isinstance(query, str):
                raw_result.extend(tavily_search(query, max_result))
            elif isinstance(query, list):
                for q in query:
                    if isinstance(q, str):
                        raw_result.extend(tavily_search(q, max_result))

        if not raw_result:
            stagnant_steps += 1
            stagnation_update = {
                "progress_history": progress_history + [len(existing_evidence)],
                "stagnant_steps": stagnant_steps,
                "cost_history": new_cost,
            }
            if stagnant_steps >= 3:
                log_breaker_trip(f"Stagnation: {stagnant_steps} steps with zero search results.", state,
                                 extra={**stagnation_update, **tracking_update})
                raise circuitBacker(f"Stagnation: {stagnant_steps} steps with zero search results.")
            return {
                'evidence': existing_evidence,
                'research_status': 'no_results',
                **tracking_update,
                **stagnation_update
            }

        seen_urls = set()
        deduped_raw_result: List[dict] = []
        for item in raw_result:
            norm_url = normalize_url(item.get('url'))
            if norm_url and norm_url not in seen_urls:
                seen_urls.add(norm_url)
                deduped_raw_result.append(item)

        pack = invoked_structured_output(
            llm, EvidencePackage,
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=(
                    f"Raw result: \n{deduped_raw_result}"
                    f"As-of-date: {state['as_of']}"
                    f"recency_date: {state['recency_days']}"
                ))
            ],
            max_retries=3
        )
        evidences = pack.evidence
        final_seen_urls = set()
        unique_evidences: List[EvidenceItem] = []
        for e in evidences:
            norm_url = normalize_url(e.url)
            if norm_url and norm_url not in final_seen_urls:
                final_seen_urls.add(norm_url)
                unique_evidences.append(e)
        evidences = unique_evidences
        mode = state.get("mode", "closed_book")
        if mode == "open_book":
            as_of = date.fromisoformat(state["as_of"])
            cutoff = as_of - timedelta(days=int(state["recency_days"]))
            fresh: List[EvidenceItem] = []
            for e in evidences:
                d = iso_format(e.published_at)
                if d and d >= cutoff:
                    fresh.append(e)
            evidences = fresh

        current_evidence = state.get("evidence", []) + evidences
        current_progress_marker = len(current_evidence)

        if progress_history:
            last_progress_marker = progress_history[-1]

            if current_progress_marker <= last_progress_marker:
                stagnant_steps += 1
            else:
                stagnant_steps = 0

        stagnation_update = {
            "progress_history": progress_history + [current_progress_marker],
            "stagnant_steps": stagnant_steps,
            "cost_history": new_cost,
        }

        window_size = 3
        if len(new_cost) >= window_size:
            recent_cost = sum(new_cost[-window_size:])
            recent_progress = (
                    (progress_history + [current_progress_marker])[-1] -
                    (progress_history + [current_progress_marker])[-window_size]
            )
            if recent_progress <= 0 and recent_cost > 0.02:
                log_breaker_trip(
                    f"Cost velocity: ${recent_cost:.4f} spent over last {window_size} steps with zero progress.",
                    state,
                    extra={**stagnation_update, **tracking_update}
                )
                raise circuitBacker(
                    f"Cost velocity: ${recent_cost:.4f} spent over last {window_size} steps with zero progress."
                )

            cost_per_unit = recent_cost / max(recent_progress, 1)
            if cost_per_unit > 0.05:
                log_breaker_trip(
                    f"Cost velocity: ${recent_cost:.4f} spent over last {window_size} steps with zero progress.",
                    state,
                    extra={**stagnation_update, **tracking_update}
                )
                raise circuitBacker(
                    f"Cost efficiency: ${cost_per_unit:.4f} per progress unit exceeds baseline."
                )

        breaker_state = state.get("breaker_state", "closed")

        if repeats >= 3 or stagnant_steps >= 3:
            if breaker_state == "closed":
                log_breaker_trip("breaker state opened",
                                 state,
                                 extra={**stagnation_update, **tracking_update}
                                 )
                return {
                    "breaker_state": "open",
                    "research_status": "loop_warning",
                    "queries": [],
                    "evidence": current_evidence,
                    **stagnation_update,
                    **tracking_update
                }
            elif breaker_state == "open":
                log_breaker_trip("breaker state closed",
                                 state,
                                 extra={**stagnation_update, **tracking_update}
                                 )
                return {
                    "breaker_state": "half_open",
                    "research_status": "loop_warning_final",
                    "evidence": current_evidence,
                    **tracking_update,
                    **stagnation_update
                }
            else:
                log_breaker_trip("Loop persisted after correction attempts.",
                                 state,
                                 extra={**stagnation_update, **tracking_update}
                                 )
                raise circuitBacker("Loop persisted after correction attempts.")

        if not evidences:
            log_breaker_trip("No Evidence.",
                             state,
                             extra={**stagnation_update, **tracking_update}
                             )
            return {
                "breaker_state": "closed",
                'evidence': state.get("evidence", []),
                'research_status': 'no_results',
                **stagnation_update,
                **tracking_update
            }

        print("Exit research")
        return {
            "breaker_state": "closed",
            'evidence': current_evidence,
            'research_status': 'ok',
            **tracking_update,
            **stagnation_update
        }
    except circuitBacker:
        raise
    except Exception as e:
        print(f"research() error: {e}")
        return {
            "breaker_state": "closed",
            'evidence': state.get("evidence", []),
            'research_status': 'error',
            **tracking_update
        }


def tavily_search(query: str, max_result: int = 5) -> list[dict]:
    # Initialize TavilySearch with max_results
    print("In tavily_search")
    tavily = TavilySearch(max_results=max_result)

    # Run search
    response = tavily.invoke({"query": query})

    # TavilySearch usually returns a dict with a 'results' key
    results = response.get("results", response) if isinstance(response, dict) else response

    normalized: list[dict] = []
    for q in results:
        normalized.append({
            'title': q.get('title'),
            'url': q.get('url'),
            'snippet': q.get('content') or q.get('snippet'),
            "published_at": q.get("published_date") or q.get("published_at"),
            'source': q.get('source', 'tavily')
        })
    print('Exit tavily_search')
    return normalized


def iso_format(s: Optional[str]):
    print("In iso_format")
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def orchestrator(state: State):
    print("In orchestrator")
    mode = state.get('mode', 'closed_book')
    evidence = state.get('evidence', [])
    research_status = state.get('research_status', 'no_results')
    plan = invoked_structured_output(
        llm, Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"mode: {mode}\n\n"
                f"Research status: {research_status}\n\n"
                f"{[e.model_dump() for e in evidence]}"
            )
            )
        ],
        max_retries=3
    )
    print("Exit orchestrator")
    return {"plan": plan}


def fetch_task(state: State):
    print("In fetch_task")
    return [Send("worker",
                 {
                     "topic": state['topic'],
                     "plan": state['plan'].model_dump(),
                     "task": task.model_dump(),
                     "mode": state['mode'],
                     "as_of": state["as_of"],
                     "recency_days": state["recency_days"],
                     "evidence": [e.model_dump() for e in state.get('evidence', [])]
                 }
                 )
            for task in state["plan"].tasks]


def worker(payload: WorkPayload):
    print("In worker")
    topic = payload['topic']
    plan = Plan(**payload['plan'])
    task = Task(**payload['task'])
    as_of = payload.get("as_of")
    recency_days = payload.get("recency_days")
    evidence = payload.get('evidence', [])
    mode = payload.get("mode", "closed_book")
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = ""
    valid_urls = set()
    if evidence:
        evidence_items = [EvidenceItem(**e) if isinstance(e, dict) else e for e in evidence]
        valid_urls = {e.url for e in evidence_items if e.url}
        evidence_text = "\n".join(
            f"{e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip() for e in evidence_items[:10]
        )

    blog = invoked_structured_output(
        llm, SectionOutput,
        [SystemMessage(content=WORKER_SYSTEM),
         HumanMessage(
             content=(
                 f"Title: {plan.blog_title}\n"
                 f"Audience: {plan.audience}\n\n"
                 f"Tone: {plan.tone}\n\n"
                 f"Blog Kind: {plan.blog_kind}\n"
                 f"Mode: {mode}\n\n"
                 f"Topic: {topic}\n\n"
                 f"narrative_thread: {plan.narrative_thread}\n\n"
                 f"As-of: {as_of} (recency_days={recency_days})\n\n"
                 f"Task Title: {task.title}\n\n"
                 f"goals: {task.goal}"
                 f"Target words: {task.target_words}\n"
                 f"Bullets: {bullets_text}\n"
                 f"tags: {task.tags}\n"
                 f"requires_research: {task.requires_research}\n"
                 f"requires_citations: {task.requires_citations}\n"
                 f"requires_code: {task.requires_code}\n"
                 f"evidence: {evidence_text}"

                 "Return the section content and list of cited sources."
             )),
         ],
        max_retries=3
    )

    verified_citations = []
    response = str(blog.content)
    if blog.citations:
        for cite in blog.citations:
            if cite.source_url in valid_urls:
                verified_citations.append(cite)
            else:
                print(f"Dropping ungrounded hallucinated URL: {cite.source_url}")

        if verified_citations:
            references_md = "\n\n**Sources:**\n" + "\n".join(
                f"- [{c.marker_id}] {c.source_url}" for c in verified_citations
            )
            response += references_md

    print("Exit worker")
    return {'sections': [(task.id, response)]}


def merge_content(state: State):
    print("In merge_content")
    title = state['plan'].blog_title
    sorted_sections = sorted(state['sections'], key=lambda x: x[0])
    cleaned_sections = [s[1] for s in sorted_sections]
    body = "\n\n".join(cleaned_sections).strip()
    final_blog = f"# {title}\n\n{body}"
    print("Exit merge_content")
    return {"merged_md": final_blog}


def editor(state: State):
    merged_md = state['merged_md']
    response = editor_llm.invoke(
        [
            SystemMessage(content=EDITOR_SYSTEM),
            HumanMessage(content=f"Here is the draft:\n\n{merged_md}\n\nReturn the polished markdown.")
        ]
    )
    return {'merged_md': response.content}


def decide_images(state: State):
    print("In decide_images")
    merged_md = state['merged_md']
    plan = state['plan']
    try:
        image_plan = invoked_structured_output(
            llm, GlobalImagePlan,
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md[:3000]}"
                ))
            ],
            max_retries=3
        )
        print("Exit decide_images")
        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [img.model_dump() for img in image_plan.image_specs],
        }
    except Exception as e:
        print(f"decide_images failed: {e}")
        return {
            "md_with_placeholders": merged_md,
            "image_specs": []
        }



def _flux_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by black-forest-labs/FLUX.1-schnell model.
    Requires: HF_TOKEN in environment or .env
    """
    print("In _flux_generate_image_bytes")
    hf_token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=hf_token) if hf_token else InferenceClient()

    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
    buf = BytesIO()
    image.save(buf, format="PNG")
    print("Exit _flux_generate_image_bytes")
    return buf.getvalue()


def generate_place_image(state: State) -> dict:
    print("In generate_place_images")
    plan = state["plan"]
    if not plan:
        return {"final": state.get("merged_md", "")}

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    clean_title = re.sub(r'[\\/*?:"<>|]', "", plan.blog_title)
    md_filename = f"{clean_title}.md"

    blogs_dir = Path("output/markdown")
    blogs_dir.mkdir(parents=True, exist_ok=True)
    out_md_path = blogs_dir / md_filename

    
    if not image_specs:
        out_md_path.write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("output/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        img_filename = spec["filename"]
        out_path = images_dir / img_filename

       
        if not out_path.exists():
            try:
                img_bytes = _flux_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                print(f"Error generating image '{img_filename}': {e}")
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](output/images/{img_filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    out_md_path.write_text(md, encoding="utf-8")
    print(f"Exit generate_place_images. File created: {out_md_path}")
    return {"final": md}


def build_graph():
    print("In build_graph")
    graph = StateGraph(State)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("router", router)
    graph.add_node("research", research)
    graph.add_node("worker", worker)
    graph.add_node("reducer", reducer_subgraph())
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_next, {"orchestrator": "orchestrator", "research": "research"})
    graph.add_conditional_edges("research",
                                research_router,
                                {"router": "router", "orchestrator": "orchestrator"}
                                )
    graph.add_conditional_edges("orchestrator", fetch_task, ["worker"])
    graph.add_edge("worker", "reducer")
    graph.add_edge("reducer", END)
    print("Exit build_graph.")
    return graph.compile(checkpointer=in_memory)


def reducer_subgraph():
    print("In reducer_subgraph")
    graph = StateGraph(State)
    graph.add_node("merge_content", merge_content)
    graph.add_node("editor", editor)
    graph.add_node("decide_images", decide_images)
    graph.add_node("generate_place_image", generate_place_image)
    graph.add_edge(START, "merge_content")
    graph.add_edge("merge_content", "editor")
    graph.add_edge("editor", "decide_images")
    graph.add_edge("decide_images", "generate_place_image")
    graph.add_edge("generate_place_image", END)
    print("Exit reducer_subgraph.")
    return graph.compile()
