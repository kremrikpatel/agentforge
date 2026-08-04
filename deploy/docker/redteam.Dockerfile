# AgentForge red-team worker (redteam.runner) -- Phase 3, run on a schedule.
#
#   docker build -f deploy/docker/redteam.Dockerfile -t agentforge/redteam:0.1.0 .
#
# This is the only image that installs the `redteam` extra, which is where
# PyRIT's ~44 packages live. Expect it to be substantially larger than the API
# image; that is the point of keeping them separate.
#
# Deployed as a CronJob, not a Deployment: the suite is a scheduled batch job
# that exits with a status code, and a Deployment would restart it forever.

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
COPY redteam ./redteam

RUN pip install --upgrade pip && pip install ".[redteam]"

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

# Exit code is the CI signal: 0 all categories held, 1 a category fell below
# threshold, 2 the run failed, 3 the target was refused as unsafe.
ENTRYPOINT ["python", "-m", "redteam.runner"]
