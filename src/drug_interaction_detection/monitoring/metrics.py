from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class ServiceMetrics:
    requests_total: int = 0
    errors_total: int = 0
    total_latency_ms: float = 0.0
    predictions_by_label: dict[str, int] = field(default_factory=dict)

    def observe(self, label: str, started_at: float, ok: bool = True) -> None:
        self.requests_total += 1
        if not ok:
            self.errors_total += 1
        self.total_latency_ms += (time.perf_counter() - started_at) * 1000.0
        self.predictions_by_label[label] = self.predictions_by_label.get(label, 0) + 1

    def snapshot(self) -> dict[str, object]:
        average_latency = self.total_latency_ms / self.requests_total if self.requests_total else 0.0
        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "average_latency_ms": round(average_latency, 3),
            "predictions_by_label": dict(sorted(self.predictions_by_label.items())),
        }

