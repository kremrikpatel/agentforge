# AgentForge API (app.main:app) -- the Phase 1 pipeline service.
#
# Build from the repository root:
#   docker build -f deploy/docker/api.Dockerfile -t agentforge/api:0.1.0 .
#
# Deliberately does NOT install the `redteam` extra: PyRIT pulls ~44 packages
# (torch, transformers, datasets, av, pyodbc) that this service never imports.
# The red-team worker image installs them; this one stays small.
#
# No secrets are baked in. Everything is injected as environment variables from
# the ConfigMap and Secret at runtime.

# ---------- builder -------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# A virtualenv is the unit copied into the runtime stage, so no build tooling,
# pip cache, or compiler ever reaches the final image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY app ./app
COPY agents ./agents
COPY gateway ./gateway
COPY guardrails ./guardrails
COPY memory ./memory
COPY rag ./rag
COPY instrumentation ./instrumentation
COPY actions ./actions
COPY web ./web

RUN pip install --upgrade pip && pip install .

# ---------- runtime -------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production

# Numeric uid so Kubernetes runAsNonRoot can verify it without a lookup.
RUN groupadd --gid 10001 agentforge \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin agentforge

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

WORKDIR /srv
USER 10001:10001

EXPOSE 8000

# Liveness/readiness are the chart's job (see api-deployment); a HEALTHCHECK
# here would only duplicate them and confuse `kubectl describe`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
