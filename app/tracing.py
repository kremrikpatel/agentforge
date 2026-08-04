"""OpenTelemetry tracing, fanned out to LangSmith, LangWatch and Arize.

All three products ingest OTLP, so this is one tracing pipeline with one span
processor per destination rather than three vendor SDKs. Adding a fourth backend
is a config change, not a code change.

Metrics are carried as span attributes (latency, confidence, provider, token
budget) rather than through a separate metrics pipeline: all three of these tools
derive their charts from spans, and a second export path would be duplicate
plumbing for the same numbers.

Nothing here is on by default. With OTEL_ENABLED unset every helper degrades to
a no-op context manager, so an unconfigured deployment pays nothing and cannot
fail on a missing endpoint.

Credentials come from the environment only. The Helm chart mounts them from a
Kubernetes Secret; no key is ever written into a chart or values file.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.config import _env, _env_bool
from app.observability import get_logger, log_event

logger = get_logger("agentforge.tracing")

# Vendor OTLP/HTTP trace endpoints. Defaults are a convenience, not a promise --
# every one is overridable, and they should be checked against current vendor
# documentation before a production rollout.
DEFAULT_ENDPOINTS = {
    "langsmith": "https://api.smith.langchain.com/otel/v1/traces",
    "langwatch": "https://app.langwatch.ai/api/otel/v1/traces",
    "arize": "https://otlp.arize.com/v1/traces",
}


@dataclass(frozen=True)
class ExporterTarget:
    name: str
    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.headers)


def build_targets() -> list[ExporterTarget]:
    """One target per observability backend that has credentials present.

    A backend with no API key is simply absent -- running with LangSmith alone
    is a normal configuration, not a degraded one.
    """
    targets: list[ExporterTarget] = []

    langsmith_key = _env("LANGSMITH_API_KEY")
    if langsmith_key:
        targets.append(
            ExporterTarget(
                name="langsmith",
                endpoint=_env("LANGSMITH_OTEL_ENDPOINT", DEFAULT_ENDPOINTS["langsmith"]),
                headers={
                    "x-api-key": langsmith_key,
                    "Langsmith-Project": _env("LANGSMITH_PROJECT", "agentforge"),
                },
            )
        )

    langwatch_key = _env("LANGWATCH_API_KEY")
    if langwatch_key:
        targets.append(
            ExporterTarget(
                name="langwatch",
                endpoint=_env("LANGWATCH_OTEL_ENDPOINT", DEFAULT_ENDPOINTS["langwatch"]),
                headers={"Authorization": f"Bearer {langwatch_key}"},
            )
        )

    arize_key = _env("ARIZE_API_KEY")
    arize_space = _env("ARIZE_SPACE_ID")
    if arize_key and arize_space:
        targets.append(
            ExporterTarget(
                name="arize",
                endpoint=_env("ARIZE_OTEL_ENDPOINT", DEFAULT_ENDPOINTS["arize"]),
                headers={
                    "api_key": arize_key,
                    "space_id": arize_space,
                    "authorization": f"Bearer {arize_key}",
                },
            )
        )

    # An OTel Collector sidecar, if one is deployed. Useful as a single fan-out
    # point when you would rather not give the app three sets of credentials.
    collector = _env("OTEL_EXPORTER_OTLP_ENDPOINT")
    if collector:
        targets.append(
            ExporterTarget(
                name="collector", endpoint=collector, headers={"x-source": "agentforge"}
            )
        )

    return targets


_provider = None
_tracer = None
_enabled = False


def configure_tracing(
    service_name: str = "", targets: list[ExporterTarget] | None = None, exporters=None
) -> bool:
    """Install the tracer provider. Returns whether tracing ended up active.

    `exporters` lets tests inject an in-memory exporter and assert on spans
    without a network destination.
    """
    global _provider, _tracer, _enabled

    if not _env_bool("OTEL_ENABLED", False) and exporters is None:
        log_event(logger, "tracing.disabled", reason="OTEL_ENABLED is not set")
        _enabled = False
        return False

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create(
        {
            "service.name": service_name or _env("OTEL_SERVICE_NAME", "agentforge-api"),
            "service.version": _env("APP_VERSION", "0.1.0"),
            "deployment.environment": _env("APP_ENV", "development"),
        }
    )
    provider = TracerProvider(resource=resource)

    installed: list[str] = []
    if exporters:
        for exporter in exporters:
            # Simple (not batched) so a test sees its spans without flushing.
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            installed.append(type(exporter).__name__)
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        for target in targets if targets is not None else build_targets():
            if not target.configured:
                continue
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=target.endpoint, headers=target.headers)
                )
            )
            installed.append(target.name)

    if not installed:
        log_event(logger, "tracing.no_targets", reason="no backend credentials present")
        _enabled = False
        return False

    # Also publish globally so third-party instrumentation finds it, but take
    # our own tracer from this provider directly: set_tracer_provider only wins
    # once per process, so a second configure() would otherwise keep emitting
    # into the first provider's exporters.
    trace.set_tracer_provider(provider)
    _provider = provider
    _tracer = provider.get_tracer("agentforge")
    _enabled = True
    atexit.register(shutdown_tracing)
    log_event(
        logger,
        "tracing.enabled",
        targets=installed,
        service=resource.attributes.get("service.name"),
    )
    return True


def shutdown_tracing() -> None:
    """Flush pending spans. Registered atexit and called from app shutdown."""
    global _provider, _tracer, _enabled
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:  # noqa: BLE001 -- never fail a shutdown on telemetry
            pass
    _provider, _tracer, _enabled = None, None, False


def is_enabled() -> bool:
    return _enabled


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span, or do nothing at all when tracing is off.

    Telemetry must never be able to break a request, so a failure inside the
    span machinery is swallowed and the body still runs.
    """
    if not _enabled or _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as current:
        set_attributes(current, **attributes)
        yield current


def set_attributes(current, **attributes: Any) -> None:
    """Attach attributes to a span that may be None (tracing disabled)."""
    if current is None:
        return
    try:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
    except Exception:  # noqa: BLE001
        pass


def record_error(current, exc: BaseException) -> None:
    if current is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        current.record_exception(exc)
        current.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
    except Exception:  # noqa: BLE001
        pass
