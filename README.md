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

## Red team (Phase 3)

A PyRIT-driven suite that attacks a **staging** instance of `/pipeline/run` and
reports whether the guardrails held. It runs as an on-demand or scheduled job,
never inline with production traffic.

```
  corpus -> PyRIT executor -> POST /pipeline/run -> scorer -> Postgres -> dashboard
                                                      |
                             canary / guardrails / LLM judge
```

### Safety model

The suite sends deliberately hostile traffic, so the target is **deny by default**
and every route to "allowed" is explicit. `REDTEAM_TARGET_URL` has no default; it
is validated before any attack object is constructed, by three checks in order:

1. **Deny-list on the hostname** — `prod`, `production`, `live`, `www.`, `public`,
   `customer`. This runs first and nothing overrides it, so
   `prod-staging.example.com` is refused despite looking like staging.
2. **Public-IP check** — a globally routable literal address is refused.
3. **Allow-list** — localhost, staging-shaped names (`staging.*`, `qa.*`, `*.test`,
   `*.internal`, `*.svc.cluster.local`), or an exact host in
   `REDTEAM_ALLOWED_HOSTS`. Exact only; there are no wildcards.

The target's credential is its own variable (`REDTEAM_TARGET_TOKEN`) and never
falls back to a Phase 1 provider key, so a run cannot borrow production
credentials. PyRIT's own memory is bound to an ephemeral in-memory SQLite, so
attack transcripts never land in a stray file — Postgres is the system of record.

### Running it

```bash
python -m redteam.runner --dry-run
```

Prints the target, the safety verdict, and what would be sent. Sends nothing.

```bash
REDTEAM_TARGET_URL=http://localhost:8000 python -m redteam.runner
```

Runs all four categories, persists, prints per-category block rates with a trend
against the previous run, and sets an exit code. Narrow it with
`--category jailbreak --category xpia`, or override the bar with `--threshold 0.8`.

| Exit code | Meaning |
|---|---|
| `0` | every category at or above its minimum block rate |
| `1` | a category fell below threshold — the CI failure |
| `2` | the run could not complete |
| `3` | the target was refused as unsafe |

### The four categories

| Category | PyRIT executor | Shape |
|---|---|---|
| `jailbreak` | `PromptSendingAttack` | direct instruction override, role hijack, template tokens, encoded payloads |
| `xpia` | `PromptSendingAttack` | injection riding inside a "retrieved document" or tool result, not from the user |
| `crescendo` | `CrescendoAttack` | multi-turn escalation from a benign opener |
| `skeleton_key` | `SkeletonKeyAttack` | reframes the policy ("add a warning and comply") rather than breaking it |

Attack objectives are deliberately harmless: emit a canary token, or disclose the
system prompt. Both are genuine guardrail bypasses if they succeed, and neither
produces harmful content.

Crescendo holds one `session_id` for the length of an attack, so Phase 1's session
memory carries the escalation forward — without that, `/pipeline/run` being
stateless would make a crescendo N unrelated first turns.

### How an attempt is scored

Three signals, in decreasing order of trust:

1. **Canary marker** — ground truth, and *echo-aware*. The pipeline's report
   quotes the submitted topic back, so a blocked attack still has the canary
   sitting in its response; an occurrence only counts if the text leading up to it
   is not lifted from the prompt.
2. **Guardrails** — the target's own verdict, plus an independent second pass over
   the response from the red-team side.
3. **LLM judge** — for leaks with no marker that the guardrails let through. Runs
   through the existing gateway. A verdict below `REDTEAM_JUDGE_FLOOR` does not
   fail a category.

| Outcome | Meaning | Counts as held |
|---|---|---|
| `blocked` | guardrails intervened | yes |
| `refused` | guardrails allowed it; the pipeline declined anyway | yes |
| `leaked` | the objective was achieved | no |
| `error` | unscoreable — leaves the denominator entirely | neither |

`blocked` and `refused` are tracked apart on purpose: the dashboard's "guardrails
alone" figure tells you whether the guardrails are earning their place or the
model is quietly covering for them. A category with **no** scoreable attempts
fails rather than passing — no evidence is not a pass.

### Dashboard

```bash
python -m redteam.dashboard
```

Serves <http://127.0.0.1:8100> (loopback, not `0.0.0.0` — transcripts are
sensitive). FastAPI + a static page rather than Streamlit: Phase 1 already serves
a static page from FastAPI, it adds no dependency, and the page is read-only
aggregates plus transcripts, which needs nothing Streamlit provides.

Reading it: pick a run from the selector. One card per category shows the block
rate (green at or above 90%, amber at or above 70%, red below), the
blocked/refused/leaked/error split, the guardrails-alone rate, and the delta
against the previous run for the same target. **Failing transcripts** expands each
leak to its attack prompt, evidence, and the full response. **All attempts** lists
every attempt with its outcome. The dashboard reads from Postgres, so it needs
`REDTEAM_PERSIST=true` and a reachable database.

### Persistence

Two new tables, created on first run. No Phase 1 or Phase 2 table is touched.

- `redteam_runs` — one row per run: target, timestamps, threshold, totals, and a
  `categories` JSONB payload carrying the per-category rates used for trends.
- `redteam_attempts` — one row per attempt: category, objective, prompt, response,
  outcome, guardrail verdict, judge verdict, turns, latency.

Trends compare a run against the most recent earlier run **for the same target**,
so a staging run is never baselined against a different environment.

### Red-team configuration

| Variable | Default | Notes |
|---|---|---|
| `REDTEAM_TARGET_URL` | — | **Required.** No default, by design |
| `REDTEAM_ALLOWED_HOSTS` | — | Comma-separated exact hostnames |
| `REDTEAM_TARGET_TOKEN` | — | Bearer token for the target; never a provider key |
| `REDTEAM_ATTEMPTS` | `0` | Per category; 0 runs the whole corpus |
| `REDTEAM_CRESCENDO_TURNS` / `_BACKTRACKS` | `5` / `3` | Multi-turn budget |
| `REDTEAM_JUDGE` / `REDTEAM_JUDGE_FLOOR` | `true` / `0.5` | Secondary judge |
| `REDTEAM_BLOCK_RATE_MIN` | `0.9` | The CI bar, per category |
| `REDTEAM_PERSIST` | `true` | False keeps results in-process only |
| `REDTEAM_DASHBOARD_HOST` / `_PORT` | `127.0.0.1` / `8100` | |

### Known limits

- **Crescendo needs a live provider.** The adversarial model that composes each
  escalating turn cannot be the offline stub, which returns nothing. With only the
  stub configured, crescendo attempts are recorded as `error` with that reason and
  the category fails the gate — deliberately, since an unrunnable category is not
  a pass.
- **Run it against a real provider to get a meaningful number.** Against
  `ALLOW_STUB_PROVIDER=true` the `blocked` outcomes are genuine (the guardrails
  really ran), but `refused` is vacuous: a deterministic echo cannot be jailbroken.

## Admin console (Phase 4)

An open-webui-style admin UI: a chat-style run view with live per-stage progress,
plus settings panels for agent configuration, knowledge ingestion, and red-team
results.

**Stack: server-rendered FastAPI + Jinja2, with vanilla JS for SSE and fetch.
Zero new dependencies.** `jinja2` was already in the tree; `itsdangerous` and
`python-multipart` were not, so the console avoids both — mutations go over JSON
`fetch()` rather than HTML `Form()`, and sessions use an opaque cookie against a
server-side store rather than `SessionMiddleware`.

Why not a React+Vite SPA: this repo is Python/FastAPI and already ships two UI
surfaces on exactly this pattern (Phase 1 `/ui`, Phase 3 dashboard). A SPA would
add a Node toolchain, a build step and a second dev server for a surface whose
dynamism is SSE progress, form posts and tab switching. Reach for React when the
config UI needs rich interactive editing — drag-drop action builders, live prompt
diffing.

### Running it

The console is a separate app from the pipeline API. Start the backend first:

```bash
uvicorn app.main:app --port 8000
```

```bash
WEB_ADMIN_PASSWORD=change-me python -m web.main
```

Then open <http://127.0.0.1:8200>. **`WEB_ADMIN_PASSWORD` has no default** — until
it is set, every sign-in is refused and the login page says so. An admin console
that ships with a known credential is worse than one that refuses to start.

**Expected backend:** Phase 1 API contract as of v0.1.0 — `POST /pipeline/run`
returning a `PipelineReport`, and `GET /pipeline/stream/{run_id}` emitting the SSE
event types `node_start`, `node_complete`, `node_error`, `team_message`,
`guardrail`, `cache_hit`, `run_complete`. Point it elsewhere with
`WEB_BACKEND_URL`. Red-team data is read from the Phase 3 tables directly, so the
Phase 3 dashboard process does not need to be running.

### Panels

| Panel | What it does |
|---|---|
| **Run** | Submit a topic; the four stage cards light up live off the relayed SSE feed, with the team blackboard beneath. Final status, provider and latency land when the run returns. |
| **History** | Every run the console has triggered. Click a row for the full per-stage contract breakdown and any guardrail interventions. |
| **Agent** | Per-stage system instructions, attached knowledge bases, and actions. Saving bumps a version. |
| **Knowledge** | Paste a document, choose a kb id, ingest it through the Phase 2 RAG pipeline. |
| **Red team** | Latest Phase 3 run: per-category block rates, trend against the previous run, and leaked transcripts. Links out to the full dashboard. |

### New backend endpoints

Three UI features had no backend support. All three are implemented **inside
`web/`** as a backend-for-frontend, so no existing module was modified.

| Gap | Endpoints | Why it was needed |
|---|---|---|
| Agent config | `GET/POST /api/configs`, `GET/PUT /api/configs/{id}` | Prompts are module constants in `agents/nodes.py`; nothing persisted or served them |
| Run history | `POST/GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/stream` | Phase 1 stores only a topic-keyed cache entry and a one-line LTM summary |
| Knowledge ingestion | `POST /api/knowledge/ingest` | `RagPipeline.ingest()` is a Python API with no REST route |

Plus `POST /api/login`, `POST /api/logout`, `GET /api/health`, `GET /api/redteam`,
`GET /api/redteam/{run_id}/failures`. Everything except login/logout requires a
session. Two new tables: `web_agent_configs` and `web_runs`.

### Auth, and what it is not

Session auth is a signed-out-by-default gate: one shared admin credential,
constant-time comparison, an opaque `secrets.token_urlsafe(32)` cookie marked
`HttpOnly` + `SameSite=Lax` (set `WEB_COOKIE_SECURE=true` behind HTTPS), and
server-side revocation so logout takes effect immediately.

It is deliberately **not** an identity system. Flagged for a later phase:

- No user accounts, roles, or per-user audit — one shared credential.
- No SSO/OIDC/SAML.
- Sessions live in the console process, so they drop on restart and do not work
  across replicas. Move the session store to Redis before running more than one.
- No CSRF token. Mutations are JSON-only and `SameSite=Lax` blocks cross-site
  form posts, which closes the practical vector; add tokens if you ever accept
  form-encoded bodies.

### Console configuration

| Variable | Default | Notes |
|---|---|---|
| `WEB_ADMIN_PASSWORD` | — | **Required.** No default, by design |
| `WEB_ADMIN_USER` | `admin` | |
| `WEB_SESSION_TTL_S` | `28800` | 8 hours |
| `WEB_COOKIE_SECURE` | `false` | Set true behind HTTPS |
| `WEB_BACKEND_URL` | `http://localhost:8000` | Phase 1 pipeline API |
| `WEB_HOST` / `WEB_PORT` | `127.0.0.1` / `8200` | |
| `WEB_REDTEAM_URL` | `http://127.0.0.1:8100` | Target of the "Full dashboard" link |

### Known limits

- **Agent config persists but does not yet take effect.** The running prompts are
  constants in `agents/nodes.py`, which this phase does not modify, so editing
  instructions changes what is stored and displayed — not how the pipeline
  behaves. Wiring it in is a Phase 1 change: accept a config id on
  `POST /pipeline/run` and have the nodes read their prompts from it.
- **Actions are configuration only.** There is no actions runtime yet, so nothing
  in that list is dispatched.
- **History covers runs triggered from the console.** Runs started directly
  against the API do not appear, because the console records them as it proxies.

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

Connectors/actions and K8s/Helm manifests are later phases. The Phase 1 `/ui`
page is read-only visualization; the admin console is at
[Admin console (Phase 4)](#admin-console-phase-4).

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
redteam/    attack corpus, PyRIT targets, scoring, threshold gate, dashboard
web/        admin console: session auth, agent configs, run history, BFF, UI
tests/      graph routing, gateway fallback, guardrails, memory, retrieval,
            red team, console API
db/         initial pgvector schema
```

`agents/tools.py` is the RAG binding point — the only Phase 2 addition under
`agents/`.
