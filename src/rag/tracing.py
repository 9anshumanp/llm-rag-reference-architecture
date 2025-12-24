from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

@dataclass(frozen=True)
class TracingConfig:
    service_name: str
    otlp_endpoint: Optional[str] = None

def configure_tracing(cfg: TracingConfig) -> None:
    resource = Resource.create({"service.name": cfg.service_name})
    provider = TracerProvider(resource=resource)

    if cfg.otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

def get_tracer(name: str = "rag") -> trace.Tracer:
    return trace.get_tracer(name)
