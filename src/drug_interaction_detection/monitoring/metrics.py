from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class ServiceMetrics:
    requests_total: int = 0
    errors_total: int = 0
    low_confidence_total: int = 0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0
    predictions_by_label: dict[str, int] = field(default_factory=dict)
    requests_by_route: dict[str, int] = field(default_factory=dict)

    def observe(self, label: str, started_at: float, *, route: str, ok: bool = True, confidence: float | None = None, low_confidence: bool = False) -> None:
        self.requests_total += 1
        if not ok:
            self.errors_total += 1
        if low_confidence:
            self.low_confidence_total += 1
        if confidence is not None:
            self.total_confidence += confidence
        self.total_latency_ms += (time.perf_counter() - started_at) * 1000.0
        self.predictions_by_label[label] = self.predictions_by_label.get(label, 0) + 1
        self.requests_by_route[route] = self.requests_by_route.get(route, 0) + 1

    def snapshot(self) -> dict[str, object]:
        average_latency = self.total_latency_ms / self.requests_total if self.requests_total else 0.0
        average_confidence = self.total_confidence / self.requests_total if self.requests_total else 0.0
        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "low_confidence_total": self.low_confidence_total,
            "average_latency_ms": round(average_latency, 3),
            "average_confidence": round(average_confidence, 5),
            "predictions_by_label": dict(sorted(self.predictions_by_label.items())),
            "requests_by_route": dict(sorted(self.requests_by_route.items())),
        }
