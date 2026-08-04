"""Tracing: exporter wiring for all three backends, and spans from a real run.

No vendor account is needed. Spans go to an in-memory exporter, which is the
same code path an OTLP exporter sits on -- what is asserted is that the pipeline
emits the spans, with the attributes each backend needs to chart.
"""

from __future__ import annotations

import dataclasses

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agents.graph import build_graph, initial_state
from agents.nodes import AgentDeps
from app import tracing
from app.tracing import (
    DEFAULT_ENDPOINTS,
    build_targets,
    configure_tracing,
    is_enabled,
    shutdown_tracing,
    span,
)
from gateway.client import LLMGateway
from gateway.providers import StubProvider
from guardrails.engine import GuardrailEngine


@pytest.fixture
def exporter():
    """Fresh tracer provider per test, torn down afterwards."""
    exp = InMemorySpanExporter()
    configure_tracing(service_name="agentforge-test", exporters=[exp])
    yield exp
    shutdown_tracing()


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    for key in (
        "LANGSMITH_API_KEY", "LANGWATCH_API_KEY", "ARIZE_API_KEY",
        "ARIZE_SPACE_ID", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)


def _deps(settings):
    cfg = dataclasses.replace(settings, allow_stub_provider=True)
    gateway = LLMGateway(
        cfg, providers=[StubProvider(cfg)], client=object(), backoff_base_s=0.0
    )
    return AgentDeps(
        gateway=gateway,
        guardrails=GuardrailEngine(cfg, gateway=gateway),
        stm=None,
        ltm=None,
        publish=False,
    )


# --- exporter targets ------------------------------------------------------


def test_no_credentials_means_no_targets():
    assert build_targets() == []


def test_langsmith_target_carries_its_key_and_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "agentforge-test")

    target = build_targets()[0]

    assert target.name == "langsmith"
    assert target.endpoint == DEFAULT_ENDPOINTS["langsmith"]
    assert target.headers["x-api-key"] == "ls-test-key"
    assert target.headers["Langsmith-Project"] == "agentforge-test"
    assert target.configured


def test_all_three_backends_fan_out_from_one_pipeline(monkeypatch):
    """The point of the OTLP design: three destinations, one tracing path."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setenv("LANGWATCH_API_KEY", "lw-test-key")
    monkeypatch.setenv("ARIZE_API_KEY", "az-test-key")
    monkeypatch.setenv("ARIZE_SPACE_ID", "space-123")

    targets = {t.name: t for t in build_targets()}

    assert set(targets) == {"langsmith", "langwatch", "arize"}
    assert targets["langwatch"].headers["Authorization"] == "Bearer lw-test-key"
    assert targets["arize"].headers["space_id"] == "space-123"
    assert all(t.configured for t in targets.values())


def test_arize_needs_both_key_and_space(monkeypatch):
    monkeypatch.setenv("ARIZE_API_KEY", "az-test-key")
    assert build_targets() == [], "an api key without a space id is not usable"


def test_endpoints_are_overridable(monkeypatch):
    monkeypatch.setenv("LANGWATCH_API_KEY", "lw-test-key")
    monkeypatch.setenv("LANGWATCH_OTEL_ENDPOINT", "https://collector.internal/v1/traces")

    assert build_targets()[0].endpoint == "https://collector.internal/v1/traces"


def test_a_collector_endpoint_is_its_own_target(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/v1/traces")

    target = build_targets()[0]

    assert target.name == "collector"
    assert target.configured


# --- disabled by default ---------------------------------------------------


def test_tracing_is_off_unless_enabled_and_costs_nothing():
    assert configure_tracing() is False
    assert is_enabled() is False

    # The helper must still be usable -- an unconfigured deploy cannot crash.
    with span("noop", **{"agentforge.stage": "analysis"}) as current:
        assert current is None


def test_configure_returns_false_when_enabled_but_no_backend(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")
    assert configure_tracing() is False


# --- spans -----------------------------------------------------------------


def test_span_records_attributes(exporter):
    with span("unit", **{"agentforge.stage": "analysis", "agentforge.latency_ms": 12.5}):
        pass

    finished = exporter.get_finished_spans()
    assert [s.name for s in finished] == ["unit"]
    assert finished[0].attributes["agentforge.stage"] == "analysis"
    assert finished[0].attributes["agentforge.latency_ms"] == 12.5


def test_none_attributes_are_dropped_rather_than_exported(exporter):
    with span("unit", **{"agentforge.provider": None, "agentforge.stage": "test"}):
        pass

    attrs = exporter.get_finished_spans()[0].attributes
    assert "agentforge.provider" not in attrs
    assert attrs["agentforge.stage"] == "test"


async def test_a_pipeline_run_emits_a_span_per_agent(exporter, settings):
    """Acceptance: a run produces the traces the three backends consume."""
    await build_graph(_deps(settings)).ainvoke(
        initial_state("run-1", "sess-1", "Design a rate limiter")
    )

    spans = {s.name: s for s in exporter.get_finished_spans()}
    for stage in ("analysis", "develop", "test", "deploy"):
        assert f"agentforge.agent.{stage}" in spans, f"no span for {stage}"

    analysis = spans["agentforge.agent.analysis"]
    assert analysis.attributes["agentforge.stage"] == "analysis"
    assert analysis.attributes["agentforge.run_id"] == "run-1"
    assert analysis.attributes["gen_ai.system"] == "agentforge"
    # The numbers each backend charts travel as span attributes.
    assert analysis.attributes["agentforge.provider"] == "stub"
    assert analysis.attributes["agentforge.latency_ms"] >= 0
    assert analysis.attributes["agentforge.success"] is True


async def test_agent_spans_are_nested_under_one_trace(exporter, settings):
    """A run has to be one trace, or a backend shows four unrelated calls."""
    with span("agentforge.pipeline.run", **{"agentforge.run_id": "run-2"}):
        await build_graph(_deps(settings)).ainvoke(initial_state("run-2", "s", "topic"))

    trace_ids = {s.context.trace_id for s in exporter.get_finished_spans()}
    assert len(trace_ids) == 1, "agent spans must share the run's trace"


def test_set_attributes_tolerates_a_missing_span():
    tracing.set_attributes(None, **{"agentforge.stage": "analysis"})   # must not raise
    tracing.record_error(None, ValueError("x"))
