# AgentForge deployment runbook

Production Kubernetes deployment: three images, one Helm chart, OTLP tracing to
LangSmith, LangWatch and Arize.

**Target platform:** vanilla Kubernetes **1.28+**. Uses `apps/v1`,
`autoscaling/v2`, `networking.k8s.io/v1`, `batch/v1` and `policy/v1` only — no
cloud-specific CRDs. The only optional third-party dependency is an ingress
controller (nginx by default) and, in `values-prod.yaml`, a cert-manager
`ClusterIssuer` annotation you can drop if you terminate TLS elsewhere.

**Local development is unchanged.** `docker compose up` from the repository root
still works exactly as it did in Phase 1; nothing here modifies it.

---

## 1. Build and push images

**Using a tagged release?** Skip this step. Every `v*` tag publishes all four
images to `ghcr.io/<owner>/agentforge/<service>` and the chart to
`oci://ghcr.io/<owner>/charts` (see [CONTRIBUTING.md](../CONTRIBUTING.md#releases)).
Set `REGISTRY=ghcr.io/<owner>` and go to step 2.

Build from the **repository root** (the Dockerfiles reference `-f`). Image names
must be `$REGISTRY/agentforge/<service>`, because the chart pulls
`{image.registry}/agentforge/<service>`:

```bash
export REGISTRY=ghcr.io/your-org
export TAG=0.1.0
```

```bash
docker build -f deploy/docker/api.Dockerfile        -t $REGISTRY/agentforge/api:$TAG        .
docker build -f deploy/docker/web.Dockerfile        -t $REGISTRY/agentforge/web:$TAG        .
docker build -f deploy/docker/redteam.Dockerfile    -t $REGISTRY/agentforge/redteam:$TAG    .
docker build -f deploy/docker/monitoring.Dockerfile -t $REGISTRY/agentforge/monitoring:$TAG .
```

```bash
docker push $REGISTRY/agentforge/api:$TAG
docker push $REGISTRY/agentforge/web:$TAG
docker push $REGISTRY/agentforge/redteam:$TAG
docker push $REGISTRY/agentforge/monitoring:$TAG
```

All three are multi-stage: a builder installs into a virtualenv and only that
venv is copied into a fresh `python:3.13-slim`, so no compiler, pip cache or
build tooling ships. Every image runs as uid **10001**, non-root, with a
read-only root filesystem (`/tmp` is an `emptyDir`).

**The red-team image is deliberately much larger.** It is the only one that
installs the `redteam` extra, which is where PyRIT's ~44 packages (torch,
transformers, datasets, av, pyodbc) live. The API and console images do not
import PyRIT and do not carry it.

Each Dockerfile has a sibling `*.Dockerfile.dockerignore`. BuildKit reads those,
which keeps the ignore rules inside `deploy/` — without them the build context
would include `.venv` (several GB) on every build. Ensure BuildKit is on
(`DOCKER_BUILDKIT=1`, the default on modern Docker).

---

## 2. Create the Secret — before installing

**The chart never contains a credential.** It references a Secret you create out
of band. Nothing installs correctly until this exists.

```bash
kubectl create namespace agentforge
```

```bash
kubectl create secret generic agentforge-secrets -n agentforge \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-REPLACE' \
  --from-literal=OPENAI_API_KEY='sk-REPLACE' \
  --from-literal=GEMINI_API_KEY='REPLACE' \
  --from-literal=GROQ_API_KEY='REPLACE' \
  --from-literal=POSTGRES_DSN='postgresql://user:pass@host:5432/agentforge' \
  --from-literal=REDIS_URL='redis://:pass@host:6379/0' \
  --from-literal=QDRANT_API_KEY='REPLACE' \
  --from-literal=WEB_ADMIN_PASSWORD='REPLACE' \
  --from-literal=LANGSMITH_API_KEY='REPLACE' \
  --from-literal=LANGWATCH_API_KEY='REPLACE' \
  --from-literal=ARIZE_API_KEY='REPLACE' \
  --from-literal=ARIZE_SPACE_ID='REPLACE' \
  --from-literal=POSTGRES_PASSWORD='REPLACE'
```

`POSTGRES_DSN` and `REDIS_URL` are secrets **in full**, not just their passwords,
even when the datastore runs in-cluster. That is deliberate: it means the chart
never has to compose a connection string out of a credential, so there is no
template that could accidentally render one into a manifest.

`POSTGRES_PASSWORD` is only needed when `postgres.enabled=true` (the in-cluster
StatefulSet reads it); with managed Postgres, `POSTGRES_DSN` alone is enough.

For a secrets manager, point External Secrets Operator or Vault Agent Injector at
the same Secret name and skip the `kubectl create` above.

> `secrets.createPlaceholder=true` renders an **all-empty** Secret so
> `kubectl edit secret` has the right keys to fill in. It is off by default
> because a later `helm upgrade` would overwrite real values with empty ones.
> Do not enable it in an environment you care about.

---

## 3. Install

Validate first — this renders every manifest without touching the cluster:

```bash
helm install agentforge deploy/helm/agentforge -n agentforge -f deploy/helm/agentforge/values-dev.yaml --dry-run --debug
```

Development / staging — everything in-cluster, one replica each:

```bash
helm install agentforge deploy/helm/agentforge -n agentforge -f deploy/helm/agentforge/values-dev.yaml --set image.registry=$REGISTRY --set-string api.image.tag=$TAG
```

Production — managed datastores, HPA on, anti-affinity:

```bash
helm install agentforge deploy/helm/agentforge -n agentforge -f deploy/helm/agentforge/values-prod.yaml --set image.registry=$REGISTRY --set-string api.image.tag=$TAG --atomic --timeout 10m
```

`--atomic` rolls the release back automatically if it does not become ready
inside the timeout, which is what you want for an unattended deploy.

### dev vs prod, in one line each

| | dev | prod |
|---|---|---|
| Datastores | in-cluster StatefulSets | **managed** (`enabled: false`) |
| API replicas | 1, no HPA | 3–30 with HPA |
| Red-team schedule | hourly, 2 attacks/category, bar 0.8 | nightly, full corpus, bar 0.95 |
| TLS | none | cert-manager + `ssl-redirect` |

The in-chart StatefulSets are single-replica with no HA, backups or failover.
They are a working default for staging. **Use managed Postgres/Redis/Qdrant in
production** — that is exactly what `enabled: false` is for.

---

## 4. Upgrade and rollback

```bash
helm upgrade agentforge deploy/helm/agentforge -n agentforge -f deploy/helm/agentforge/values-prod.yaml --set-string api.image.tag=$NEW_TAG --atomic --timeout 10m
```

```bash
helm history agentforge -n agentforge
```

```bash
helm rollback agentforge <REVISION> -n agentforge --wait
```

Omit the revision to go back exactly one. Watch it land:

```bash
kubectl rollout status deployment/agentforge-api -n agentforge
```

Deployments use `maxUnavailable: 0`, so a rollout never dips below current
capacity, and the API's `terminationGracePeriodSeconds: 60` lets an in-flight
pipeline run finish before SIGKILL.

**What rollback does not undo:** database schema. Every table created by
Phases 1–5 uses `CREATE TABLE IF NOT EXISTS` and no destructive migrations
exist, so rolling the app back is safe today — but if you add a migration that
drops or renames a column, `helm rollback` will not reverse it.

**Config-only change?** The API and web pods carry a `checksum/config`
annotation over the rendered ConfigMap, so editing `config.*` in values rolls
the pods automatically. Editing the *Secret* does not — restart deliberately:

```bash
kubectl rollout restart deployment/agentforge-api -n agentforge
```

---

## 5. Monitoring — where to look

Tracing is **off unless `OTEL_ENABLED=true` and at least one backend credential
is present**. The chart sets `OTEL_ENABLED` from `tracing.enabled`; the keys come
from the Secret. A backend with no key is simply absent — running LangSmith-only
is a normal configuration.

One OpenTelemetry pipeline fans out to all three over OTLP/HTTP, one span
processor per destination. There are no vendor SDKs to keep in sync, and a
fourth backend is a values change.

| Tool | Where | What you will see |
|---|---|---|
| **LangSmith** | smith.langchain.com → project `agentforge-prod` (`tracing.langsmithProject`) | One trace per run; `agentforge.pipeline.run` as root with four `agentforge.agent.*` children |
| **LangWatch** | app.langwatch.ai → your project | Same traces; per-agent latency and failure rate |
| **Arize** | app.arize.com → space from `ARIZE_SPACE_ID` | Same traces; drift and performance over `gen_ai.*` attributes |

Span attributes carry the numbers each tool charts: `agentforge.stage`,
`agentforge.provider`, `agentforge.latency_ms`, `agentforge.success`,
`agentforge.guardrail_input` / `guardrail_output`, `agentforge.halted`,
`gen_ai.system`, `gen_ai.request.model`.

> **Metrics are derived from spans, not exported separately.** All three products
> build their charts from trace data, so a second metrics pipeline would be
> duplicate plumbing for the same numbers.

> **Verify the endpoints before rollout.** `app/tracing.py` ships sensible
> OTLP/HTTP defaults, but vendor paths change. Check each against current vendor
> docs and override via `tracing.langsmithEndpoint` / `langwatchEndpoint` /
> `arizeEndpoint` if they have moved. Alternatively set `tracing.otlpEndpoint` to
> an OTel Collector and let it fan out — then the app holds no vendor
> credentials at all.

### Everything else

- **Structured logs** — JSON to stdout: `kubectl logs -l app.kubernetes.io/component=api -n agentforge`
- **Trajectories and audit log** (Phase 5) — in the console, `GET /trajectories/{run_id}` and `GET /audit-log`
- **Red-team results** (Phase 3) — the console's Red team panel; the CronJob's exit code is the CI signal (`0` held, `1` below threshold, `2` failed, `3` unsafe target)

```bash
kubectl get cronjob,job -n agentforge -l app.kubernetes.io/component=redteam
```

The red-team CronJob defaults to attacking **this release's own API service**, an
in-cluster staging-shaped name. `redteam/safety.py` refuses production-shaped
hostnames whatever `redteam.targetUrl` is set to — the chart cannot override that
guard, by design.

---

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Pods `CreateContainerConfigError` | The Secret does not exist or is missing a key. `envFrom` is `optional: false` on purpose — a pod with half its config is worse than one that will not start. |
| API never becomes ready | `/health` checks Redis and Postgres. `kubectl logs` will name the unreachable one; the app fails open but reports `degraded`. |
| Traces missing | `OTEL_ENABLED` unset, no backend key in the Secret, or a moved vendor endpoint. Logs emit `tracing.enabled` with the target list, or `tracing.no_targets`. |
| HPA shows `<unknown>` targets | metrics-server is not installed. The HPA is valid; it has nothing to read. |
| Console logs you out randomly | Sessions live in-process (Phase 4 known limit). The Service sets `sessionAffinity: ClientIP`; with multiple replicas behind an L7 proxy you may still need sticky sessions at the ingress. |
