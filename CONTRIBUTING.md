# Contributing to AgentForge

This guide covers local setup, what a change needs before it can merge, and how
releases get cut. For how the system works, start with [README.md](README.md);
for deployment, [deploy/README.md](deploy/README.md).

## Setup

Python **3.13** is required.

```bash
py -3.13 -m venv .venv
```

```bash
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

```bash
./.venv/Scripts/python.exe -m pytest -q
```

On Linux/macOS use `python3.13 -m venv .venv` and `.venv/bin/python`.

You don't need Redis, Postgres or any provider API key for development. Every
memory tier fails open, and `docker compose up --build` starts the full stack with
the offline stub provider. To use real providers, copy `.env.example` to `.env`.

## Making a change

1. **Branch from `main`.** Name it `feat/<topic>`, `fix/<topic>`, `docs/<topic>`
   and so on.
2. **Stay inside the module that owns the behaviour.** Later phases usually
   extend earlier ones rather than editing them. For example, `web/` adds backend
   endpoints as a backend-for-frontend and leaves `agents/` alone. If your change
   has to reach into another module, explain why in the PR.
3. **Test it.** New behaviour gets tests under `tests/`. A bug fix gets a test that
   fails without the fix.
4. **Document it.** If you add or change an endpoint, env var, config key or
   deploy step, update `README.md` or `deploy/README.md` too. New env vars also go
   in `.env.example`.
5. **Mark deliberate shortcuts.** Some simplifications have a known limit (a linear
   scan, an in-process store). Mark each one with a `ponytail:` comment that names
   the limit and the upgrade path, as in the existing
   [simplifications table](README.md#deliberate-phase-1-simplifications).

### Code conventions

- Ruff settings live in `pyproject.toml`: line length 96, target `py313`.
- Put type hints on public functions. Agent handoffs use the typed Pydantic
  contracts in `agents/contracts.py`, never raw shared state.
- Read credentials from the environment only. Never put one in code, tests,
  fixtures or Helm values. Use placeholders such as `REPLACE` or
  `example.invalid`.
- Prefer the dependencies the project already has. If you add a new one, justify
  it in the PR. Heavy dependencies go in an extra, the way `redteam` and `rerank`
  do, so the API image stays small.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <imperative summary>

<optional body: why, not what>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

## Pull requests

Open PRs against `main` and include this checklist:

- [ ] `pytest -q` passes locally
- [ ] Tests cover the new behaviour or the fixed bug
- [ ] No credentials, tokens or real production hostnames anywhere in the diff
- [ ] Docs and `.env.example` updated where behaviour or config changed
- [ ] If `deploy/helm/` changed, the chart renders:
      `helm template agentforge deploy/helm/agentforge -f deploy/helm/agentforge/values-dev.yaml`

Keep PRs focused. Refactors and behaviour changes are easier to review as
separate PRs.

## Security-sensitive areas

These modules need extra care in review. Don't weaken them to get a test passing.

| Area | Invariant |
|---|---|
| `guardrails/` | Input *and* output of every node is checked; injection blocks before an agent sees it |
| `instrumentation/` | PII is scrubbed inside the store's save path, so nothing can persist unscrubbed data |
| `rag/text2sql.py` | Nothing runs without the approval token for that exact statement |
| `redteam/safety.py` | The hostname deny-list runs first and nothing overrides it. Never target production |
| `web/` auth | `WEB_ADMIN_PASSWORD` has no default; cookies stay `HttpOnly` + `SameSite=Lax` |

**Report vulnerabilities privately.** Use GitHub's private vulnerability reporting
(repository **Security** tab, then **Report a vulnerability**). Don't open a public
issue.

## Releases

Maintainers cut releases by pushing a tag. [`.github/workflows/release.yml`](.github/workflows/release.yml)
does the rest.

1. Set `version` in `pyproject.toml` and `appVersion` in
   `deploy/helm/agentforge/Chart.yaml` to the same value. If the chart itself
   changed, bump the chart `version` too.
2. Merge to `main`.
3. Tag and push:

   ```bash
   git tag v0.2.0
   ```

   ```bash
   git push origin v0.2.0
   ```

The workflow then:

1. **Verifies.** The tag must equal both versions, and the test suite must pass.
   If either check fails, nothing is published.
2. **Publishes images** to GitHub Packages:
   `ghcr.io/<owner>/agentforge/{api,web,redteam,monitoring}`. Each image is tagged
   `0.2.0`, `0.2` and `latest`.
3. **Publishes the Helm chart** to `oci://ghcr.io/<owner>/charts/agentforge`.
4. **Creates a GitHub Release** with auto-generated notes. The wheel, sdist and
   chart archive are attached.

A tag with a hyphen, such as `v0.2.0-rc.1`, becomes a pre-release. It gets only its
own image tag and doesn't move `latest`. Set both versions to the full string
(`0.2.0-rc.1`) before tagging. The chart pulls its images by `appVersion`, so the
gate requires an exact match.

Install a published release:

```bash
helm install agentforge oci://ghcr.io/<owner>/charts/agentforge --version 0.2.0 -n agentforge -f deploy/helm/agentforge/values-prod.yaml --set image.registry=ghcr.io/<owner>
```
