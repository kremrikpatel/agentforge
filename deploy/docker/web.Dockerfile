# AgentForge admin console (web.main:app) -- the Phase 4 frontend service.
#
#   docker build -f deploy/docker/web.Dockerfile -t agentforge/web:0.1.0 .
#
# Note: this shares the API's dependency set and differs only in its command.
# Kept as a separate image because the brief asks for one per service; if you
# would rather build once, drop this file and give the web Deployment a
# `command:` override on the API image. Nothing else needs to change.

# ---------- builder -------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
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

RUN groupadd --gid 10001 agentforge \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin agentforge

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

WORKDIR /srv
USER 10001:10001

EXPOSE 8200

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8200", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
