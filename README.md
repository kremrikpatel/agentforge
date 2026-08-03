# AgentForge

A four-agent LangGraph pipeline behind a FastAPI service, with a multi-provider LLM
gateway, layered guardrails, three-tier memory, and a RAG retrieval subsystem the
agents call as a tool.

*Phase 1: agent pipeline, gateway, guardrails, memory. Phase 2: the RAG subsystem —
see [RAG subsystem (Phase 2)](#rag-subsystem-phase-2).*

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

## RAG subsystem (Phase 2)

A retrieval module the agents call as a LangGraph tool. Five strategies share one
funnel, and a mode decides how deep down it a request goes:

```
  HyDE rewrite ─▶ hybrid search ─▶ rerank ─▶ CRAG grading ─▶ Self-RAG loop
                  (dense + BM25)   (cross-    (corrective)    (retrieve
                   RRF-fused        encoder)                   again)
```

Qdrant defaults to **embedded local mode** — with no `QDRANT_URL` set it runs
in-process, so ingestion and every retrieval mode work with no server at all.

### Ingestion

```python
from rag.pipeline import RagPipeline
from rag.schemas import Document

pipe = RagPipeline(gateway)
await pipe.ingest("handbook", [
    Document(doc_id="rl-1", title="Rate Limiting", source="docs/rate-limiting.md",
             text=open("docs/rate-limiting.md").read()),
])
```

Documents are split on paragraph boundaries into `RAG_CHUNK_SIZE` windows with
`RAG_CHUNK_OVERLAP` carry-over, embedded, and written to Qdrant collection
`af_kb_<kb_id>` plus the lexical index. Chunk ids are deterministic
(`kb_id:doc_id:ordinal`), so re-ingesting a document upserts rather than
duplicating.

### Retrieval modes

```python
from rag.schemas import RetrievalMode

result = await pipe.retrieve("how do I stop one client hogging the API?",
                             kb_id="handbook", mode=RetrievalMode.HYBRID)
result.context      # citation-prefixed passages, ready for a prompt
result.citations    # ['[Rate Limiting#3]', ...]
result.steps        # per-stage trace: name, candidates in/out, latency
```

| Mode | What it adds | Reach for it when |
|---|---|---|
| `vector` | dense similarity only | you want pure semantics, no keyword bias |
| `lexical` | BM25 / full-text only | names, error codes, exact identifiers |
| `hybrid` *(default)* | both signals, RRF-fused, then reranked | general use |
| `hyde` | writes a hypothetical answer and embeds *that* | question and answer share little vocabulary |
| `crag` | grades relevance; corrects when it is poor | the corpus may not cover the question |
| `self_rag` | regenerates when the answer is not grounded | hallucination risk is the main worry |
| `text2sql` | natural language to SQL | the answer lives in tables, not prose |

Two details worth knowing. **HyDE rewrites only the dense query** — a hypothetical
document is poor BM25 input, so the lexical half keeps the user's own words.
**CRAG's `incorrect` verdict** means the corpus does not hold the answer, so it
reformulates and, if a `web_search` callable is configured, reaches outside;
`ambiguous` keeps the good passages and merges in corrected ones.

`self_rag` needs a generating callable — that is how the agent plugs in:

```python
async def generate(query: str, context: str) -> str:
    return await my_agent_answer(query, context)

result = await pipe.retrieve(q, "handbook", RetrievalMode.SELF_RAG, generate=generate)
result.grounding.action   # 'accept' | 'retrieve_again'
```

### Binding it to an agent

```python
from agents.tools import make_retrieval_tool

tool = make_retrieval_tool(pipe, allowed_modes={RetrievalMode.HYBRID, RetrievalMode.HYDE})
```

Returns a LangChain `StructuredTool`: the agent gets citation-prefixed prose, the
caller gets the full `RetrievalResult` on `.artifact`. `allowed_modes` caps what a
given node may ask for. Phase 1's `nodes.py`, `graph.py`, and `contracts.py` are
untouched — binding a tool is additive.

### Text2SQL approval flow

**Nothing executes without a human.** `propose()` generates and previews;
`execute()` refuses unless handed the approval token for that exact statement.

```python
from rag.text2sql import Column, SqlSchema, Table, Text2Sql

schema = SqlSchema(tables=(
    Table("orders", (Column("id", "bigint"), Column("total_cents", "bigint"))),
))
t2s = Text2Sql(gateway, schema)

proposal = await t2s.propose("revenue by month this year")
proposal.sql              # the statement
proposal.plan             # EXPLAIN output -- planned, not run
proposal.safe             # passed validation?
proposal.rejection_reason # why not, if not
proposal.approval_token   # sha256 of this exact SQL

# ---- a human reviews proposal.sql and proposal.plan here ----

rows = await t2s.execute(proposal, proposal.approval_token, approved=True)
```

Layered defences, in order:

1. **Scrub, then check.** Comments and string literals are stripped *before* any
   keyword or statement-count check, so nothing hides inside them.
2. **One statement, SELECT or WITH only.**
3. **Deny-list** of mutating and filesystem-reaching constructs (`INSERT`, `DROP`,
   `SELECT ... INTO`, `pg_read_file`, `dblink`, ...).
4. **Table allow-list** — every referenced table must appear in the declared
   `SqlSchema`; CTE names are recognised as local, not foreign.
5. **`execute()` revalidates** rather than trusting the proposal object, then runs
   in a `READ ONLY` transaction with a statement timeout and a row cap.

The token binds an approval to one statement: approving a `SELECT` and replaying
that approval against different SQL raises `ApprovalRequired`. `Text2Sql.execute`
is deliberately **not** exposed as an agent tool — a tool an agent can call is by
definition not a human approval.

### RAG configuration

Every stage's top-k is tunable; the funnel is where latency and cost live.

| Variable | Default | Notes |
|---|---|---|
| `QDRANT_URL` / `QDRANT_API_KEY` | — | Blank means embedded local mode |
| `QDRANT_PATH` | — | On-disk local mode instead of `:memory:` |
| `RAG_TOP_K_DENSE` / `RAG_TOP_K_LEXICAL` | `20` / `20` | Wide at the cheap stages |
| `RAG_TOP_K_FUSED` / `RAG_TOP_K_FINAL` | `12` / `5` | Narrow at the expensive ones |
| `RAG_RRF_K` | `60` | RRF damping constant |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `900` / `150` | Characters |
| `RAG_RERANK` / `RAG_RERANK_MODEL` | `true` / `ms-marco-MiniLM-L-6-v2` | See the extra below |
| `RAG_HYDE_QUERY_WEIGHT` | `0.3` | Query's share of the blended dense query |
| `RAG_CRAG_CORRECT_AT` / `RAG_CRAG_INCORRECT_AT` | `0.6` / `0.3` | Grading thresholds |
| `RAG_GROUNDING_THRESHOLD` / `RAG_SELF_RAG_LOOPS` | `0.6` / `2` | Self-RAG |
| `RAG_SQL_MAX_ROWS` / `RAG_SQL_TIMEOUT_MS` | `200` / `5000` | Text2SQL execution caps |
| `RAG_CACHE` / `RAG_CACHE_TTL_S` | `true` / `3600` | Embedding + rerank score cache |

Cross-encoder reranking is an optional extra, because `sentence-transformers`
pulls torch (~2GB):

```bash
pip install -e ".[rerank]"
```

Without it, a dependency-free lexical reranker fills the stage and the interface
is identical.

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

## Not yet built

Connectors/actions, red-team dashboard, admin UI, and K8s/Helm manifests are later
phases. The coordination UI here is read-only visualization, not an admin console.

Qdrant is not in `docker-compose.yml` — the embedded local mode covers development,
and Phase 2's scope was the `rag/` module. To run a server instead, add:

```yaml
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
```

and set `QDRANT_URL=http://qdrant:6333` on the `api` service.

## Layout

```
app/        config, structured logging + event bus, JSON extraction, FastAPI, UI
agents/     contracts, prompts + node execution, LangGraph wiring
gateway/    provider adapters, fallback chain + retry policy
guardrails/ regex tier, classifier tier, three-tier fusion engine
memory/     embedder, Redis STM, pgvector LTM, semantic cache
rag/        chunking + Qdrant store, BM25/FTS lexical index, reranking,
            HyDE, CRAG, Self-RAG, Text2SQL, and the retrieve() pipeline
tests/      graph routing, gateway fallback, guardrails, memory, retrieval
db/         initial pgvector schema
```

`agents/tools.py` is the RAG binding point — the only Phase 2 addition under
`agents/`.
