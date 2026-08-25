# AgentForge alert sweep (monitoring.cli) -- Phase 7, run on a schedule.
#
#   docker build -f deploy/docker/monitoring.Dockerfile -t agentforge/monitoring:0.1.0 .
#
# Deployed as a CronJob for the same reason the red-team worker is: the sweep
# evaluates a window of trajectories, sends what fired, and exits. A Deployment
# would restart it forever.
#
# Note this image does NOT install the `redteam` extra even though it imports
# `redteam.thresholds` -- that module only needs `redteam.schemas`, so PyRIT's
# ~44 packages stay out of this image. The red-team gate sends its own alert
# from inside the red-team image; this one covers the trajectory-window rules.

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
# monitoring/api.py imports web.auth (one session model, not a second one) and
# monitoring/rules.py imports redteam.thresholds (the existing CI gate).
COPY web ./web
COPY redteam ./redteam
COPY monitoring ./monitoring

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

ENTRYPOINT ["python", "-m", "monitoring.cli"]
CMD ["sweep"]
