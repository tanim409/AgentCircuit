A research-aware, agentic workflow blog-writing pipeline built with LangGraph, featuring an autonomous circuit breaker system that detects and stops repetitive agent behavior before it drains your API budget.

The project has two halves:

Blog Writing Agent — plans, researches, writes, edits, and illustrates a full technical blog post from a single topic prompt.
Circuit Breaker — a self-contained loop/cost-guard layer wired directly into the agent's state graph, so runaway tool-calling or LLM cost spikes get caught and stopped automatically instead of silently burning credits.
How it works

router — decides if the topic needs live web research (closed_book / hybrid / open_book) and drafts search queries.
research — runs Tavily search, deduplicates + synthesizes evidence, and is where the circuit breaker lives (see below).
orchestrator — turns the topic + evidence into a structured Plan: title, audience, tone, narrative thread, and 5–7 section Tasks.
worker (parallel) — writes one Markdown section per task, grounding factual claims in evidence and citing sources.
reducer (sub-graph) — merges all sections → runs an editor pass for tone/coherence → plans supporting images → generates them  → writes the final .md file to output/markdown/.

State persists across node calls via a LangGraph MemorySaver checkpointer, which is what allows the circuit breaker to track history across retries within a run.

Circuit Breaker

Lives inside research() and is triggered by three independent signals, layered:

Signal	What it catches
Tool-call hashing	Exact repeated Tavily queries (back-to-back)
Embedding similarity	Paraphrased/near-duplicate queries (cosine similarity)
Progress stagnation	Evidence count not growing across steps, despite continued calls
Cost velocity	Spend per step / spend per unit of progress exceeding a baseline

Three-state breaker, not a hard on/off switch:

closed → open: first trip. Router is asked to generate different, less repetitive queries and the graph retries.
open → half_open: second trip. One last corrective retry.
half_open → raise: third trip. circuitBacker (custom exception) is raised and propagates up, halting the graph run entirely.

Every trip is logged to log_breaker.log.json (JSON Lines) with a timestamp, reason, and the state snapshot at time of trip — useful for spotting recurring failure patterns in prompts or tool design after the fact.

Reliability & quality features
Structured-output retries — every LLM call that expects structured data (RouterDecision, Plan, EvidencePackage, GlobalImagePlan, SectionOutput) goes through a wrapper that retries on schema-validation failure (up to 3 attempts, with backoff), feeding the parsing error back to the model so it can self-correct.
Citation grounding — each worker-written section's citations are checked against the actual evidence URLs. Any citation pointing to a URL not in the evidence set is dropped and logged as an ungrounded/hallucinated source before the section is finalized.
URL deduplication — evidence URLs are normalized (scheme/www/trailing-slash stripped) and deduplicated programmatically, not left to the LLM.
Research status propagation — research_status (ok / no_results / error / loop_warning / etc.) is passed into the orchestrator so plan generation can adapt when evidence is missing rather than writing as if fresh data exists.
Editor pass — a dedicated node re-reads the fully merged draft to smooth transitions, fix tone consistency, and trim repetition across independently-written sections before images are added.
Tech stack
Orchestration: LangGraph
LLMs: OpenRouter (llama-3.3-70b) for planning/structured tasks; 
Search: Tavily
Embeddings: Hugging Face (Qwen/Qwen3-Embedding-0.6B) for duplicate-query detection
Image generation: FLUX.1-schnell via Hugging Face Inference API
Schema/validation: Pydantic
