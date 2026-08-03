# AgentForge — Phase 1

A four-agent LangGraph pipeline behind a FastAPI service, with a multi-provider LLM
gateway, layered guardrails, and three-tier memory.

```
        ┌── team blackboard (shared, append-only) ──┐
        ▼                ▼           ▼              ▼
   ┌─────────┐     ┌─────────┐  ┌─────────┐   ┌─────────┐
   │   Ana   │────▶│   Dev   │─▶│  Tess   │──▶│   Dep   │
   │ Analyst │     │Architect│  │   QA    │   │ Release │
   └─────────┘     └─────────┘  └─────────┘   └─────────┘
    Analysis        Develop        Test         Deploy
        └── typed Pydantic contract at every handoff ──┘

  guardrails  ▸ regex + classifier + LLM judge, before AND after every node
  gateway     ▸ Claude → GPT-4o → Gemini → Groq → offline stub
  memory      ▸ Redis STM · pgvector LTM · semantic cache
```

## Quick start

```bash
docker compose up --build
```

Then open <http://localhost:8000/ui>. That brings up the API, Redis, and
Postgres+pgvector with no manual steps — no API keys required, because the gateway
falls through to a deterministic offline provider (`ALLOW_STUB_PROVIDER=true` in
compose). Add real keys for real output:

```bash
cp .env.example .env
```

Fill in at least one provider key, then `docker compose up --build`. Compose reads
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `GROQ_API_KEY` from
your shell or `.env`.

## Running the pipeline

```bash
curl -s localhost:8000/pipeline/run -H 'content-type: application/json' -d '{"topic":"Design a rate limiter for a public REST API"}'
```

Returns a `PipelineReport`: the four stage contracts, the team blackboard, a
per-node trace (provider, latency, guardrail verdicts), and any errors.

| Endpoint | Purpose |
|---|---|
| `POST /pipeline/run` | Run the pipeline; returns the full report |
| `GET /pipeline/stream/{run_id}` | SSE feed of coordination events (the UI uses this) |
| `GET /pipeline/session/{session_id}` | Session short-term memory |
| `GET /health` | Redis / Postgres / configured providers |
| `GET /ui` | Team coordination visualization |
| `GET /docs` | OpenAPI |

The UI generates a `run_id`, subscribes to the stream, then POSTs with that id — so
agent cards light up live as each agent picks up, hands off, and raises concerns.

## Local development (no Docker)

```bash
py -3.13 -m venv .venv
```

```bash
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

```bash
./.venv/Scripts/python.exe -m pytest -q
```

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

On Linux/macOS use `.venv/bin/python` instead of `./.venv/Scripts/python.exe`.

Redis and Postgres are optional locally — every memory tier fails open, so the
pipeline still answers without them (you lose cache hits and recall, not
availability). `GET /health` reports which backends are degraded.

## Environment variables

All credentials come from the environment; nothing is hardcoded. Full list with
defaults in [`.env.example`](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `GROQ_API_KEY` | — | Blank = that provider is skipped |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` / `GEMINI_MODEL` / `GROQ_MODEL` | see `.env.example` | Per-provider model id |
| `LLM_TIMEOUT_S` / `LLM_RETRIES` / `LLM_MAX_TOKENS` | `60` / `2` / `2048` | Retries are per provider |
| `ALLOW_STUB_PROVIDER` | `false` | Offline deterministic provider, last in chain |
| `REDIS_URL` | `redis://localhost:6379/0` | STM + semantic cache |
| `POSTGRES_DSN` | `postgresql://agentforge:agentforge@localhost:5432/agentforge` | pgvector LTM |
| `SESSION_TTL_S` / `CACHE_TTL_S` | `3600` / `900` | |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Cosine above which a rephrase is a hit |
| `GUARDRAILS_LLM_TIER` | `false` | Tier 3 costs an LLM call |
| `GUARDRAILS_BLOCK_THRESHOLD` | `0.6` | Risk score that blocks a request |
| `LOG_LEVEL` | `INFO` | Structured JSON to stdout |

## How the pieces work

**Agent team.** Each agent publishes one typed contract (`agents/contracts.py`) and
consumes exactly one upstream contract — never raw shared state, so a bad handoff
fails at the boundary that produced it. Every contract carries a `handoff` block
naming the recipient, what it consumed, and its open questions. Alongside that,
agents post to a shared blackboard (`TeamMessage`) which is what the UI renders and
what the next agent sees as team context.

**Routing.** `agents/graph.py` runs the stages in order — each stage's input is the
previous stage's output, so there is nothing to fan out. After every node the router
asks one question: did anything halt the run? A guardrail block, an exhausted
provider chain, an unparseable contract, or an agent raising a *blocking* open
question all short-circuit to `END`.

**Gateway.** `gateway/` tries each configured provider in order. A missing key skips
the provider without a request; a fatal status (401/403/404) moves on immediately;
retryable failures (429/5xx/timeout) back off exponentially in place first. Adapters
are raw `httpx`, so four providers cost zero extra dependencies.

**Guardrails.** Three tiers run on the input *and* output of every node. Tier 1
regex (PII with a Luhn check on card numbers, injection shapes) and tier 2 a scored
classifier (lexicon density, imperative ratio, homoglyph ratio, entropy) run
concurrently. Tier 3, an LLM judge, only adjudicates the uncertain band — escalating
every check would double token spend to reconfirm what regex already knew. PII is
redacted; injection is blocked before any agent sees it.

**Memory.** Redis session history (STM), pgvector run archive (LTM), and a semantic
cache keyed on a normalised-topic hash with cosine fallback for rephrasings. All
three fail open.

**Instrumentation.** Every node emits a `NodeTrace` — reasoning summary, tool calls,
provider, latency, success/failure, guardrail verdicts — logged as JSON and returned
in the report. That is the trajectory shape later phases train on.

## Deliberate Phase 1 simplifications

Each is marked with a `ponytail:` comment at its site, naming the ceiling and the
upgrade path.

| Simplification | Upgrade when |
|---|---|
| Guardrail tier 2 is a linear model with hand-set priors, not fitted weights | A labelled corpus exists — keep `features()`, fit the weights, or swap in a trained classifier behind `classify()` |
| Local hashing embedder (lexical, not semantic) | Rephrasings with no shared vocabulary need to hit — swap `embed()`, the vector column is unchanged |
| Semantic cache scans a capped Redis list | The index outgrows a few hundred entries — move to `FT.SEARCH` |
| LTM opens a connection per call | Request rate justifies `psycopg_pool.AsyncConnectionPool` |
| Event bus is in-process | The API runs more than one replica — move to Redis pub/sub |

## Not in this phase

RAG pipeline, connectors/actions, red-team dashboard, admin UI, and K8s/Helm
manifests are later phases. The coordination UI here is read-only visualization, not
an admin console.

## Layout

```
app/        config, structured logging + event bus, JSON extraction, FastAPI, UI
agents/     contracts, prompts + node execution, LangGraph wiring
gateway/    provider adapters, fallback chain + retry policy
guardrails/ regex tier, classifier tier, three-tier fusion engine
memory/     embedder, Redis STM, pgvector LTM, semantic cache
tests/      graph routing, gateway fallback, guardrails, memory
db/         initial pgvector schema
```
