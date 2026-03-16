from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

from drug_interaction_detection.config.settings import ResolvedSettings
from drug_interaction_detection.inference.service import InferenceEngine
from drug_interaction_detection.monitoring.metrics import ServiceMetrics


def create_handler(settings: ResolvedSettings, engine: InferenceEngine, metrics: ServiceMetrics) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json({"status": "ok", "service": settings.settings.project.name})
                return
            if self.path == "/metrics":
                self._send_json(metrics.snapshot())
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            started_at = time.perf_counter()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = engine.predict(payload["drug_a"], payload["drug_b"], payload.get("pair_id", "api-request"))
                metrics.observe(result.interaction_label, started_at, ok=True)
                self._send_json(result.model_dump(mode="json"))
            except Exception as exc:
                metrics.observe("error", started_at, ok=False)
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    return Handler


def run_server(settings: ResolvedSettings) -> None:
    engine = InferenceEngine(settings)
    metrics = ServiceMetrics()
    handler = create_handler(settings, engine, metrics)
    server = ThreadingHTTPServer((settings.settings.service.host, settings.settings.service.port), handler)
    server.serve_forever()

