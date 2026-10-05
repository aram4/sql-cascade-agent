"""OpenTelemetry setup — console exporter, no extra infra.

This adds real OTel spans (trace IDs, span IDs, parent/child nesting via
context propagation, attributes) around the same five pipeline stages the
model_router's lightweight `trace` list already tracks. It's additive, not a
replacement: eval.py's per-route accuracy/cost rollup still reads
state.trace, since that's a structured dict, not something you'd parse back
out of printed span text. This module is purely so you can see the shape of
a real distributed trace for this pipeline, in the terminal, to learn the
OTel API.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

_initialized = False


def init_tracing(service_name: str = "sql-cascade-agent") -> None:
    global _initialized
    if _initialized:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    # SimpleSpanProcessor exports each span the moment it ends — no batching delay,
    # which matters for a short-lived CLI run that might exit before a batch flushes.
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer():
    return trace.get_tracer("sql_cascade_agent")
