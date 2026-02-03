from __future__ import annotations

from http import client
import json
import threading
import time

from drug_interaction_detection.api.server import create_handler
from drug_interaction_detection.data.dataset import prepare_dataset
from drug_interaction_detection.inference.service import InferenceEngine
from drug_interaction_detection.modeling.pipeline import train_models
from drug_interaction_detection.monitoring.metrics import ServiceMetrics
from http.server import ThreadingHTTPServer


def test_http_api_predict_endpoint(settings):
    bundle = prepare_dataset(settings)
    artifacts = train_models(settings, bundle)
    engine = InferenceEngine(settings, artifacts=artifacts)
    metrics = ServiceMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(settings, engine, metrics))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    conn = client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    conn.request(
        "POST",
        "/predict",
        body=json.dumps({"drug_a": "warfarin", "drug_b": "ibuprofen"}),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    server.shutdown()
    thread.join(timeout=2)
    assert response.status == 200
    assert payload["interaction_label"]


def test_http_api_rejects_invalid_predict_request(settings):
    bundle = prepare_dataset(settings)
    artifacts = train_models(settings, bundle)
    engine = InferenceEngine(settings, artifacts=artifacts)
    metrics = ServiceMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(settings, engine, metrics))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    conn = client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    conn.request(
        "POST",
        "/predict",
        body=json.dumps({"drug_a": "warfarin"}),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    server.shutdown()
    thread.join(timeout=2)
    assert response.status == 400
    assert payload["error"] == "invalid request"


def test_http_api_batch_predict_endpoint(settings):
    bundle = prepare_dataset(settings)
    artifacts = train_models(settings, bundle)
    engine = InferenceEngine(settings, artifacts=artifacts)
    metrics = ServiceMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(settings, engine, metrics))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    conn = client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    conn.request(
        "POST",
        "/batch-predict",
        body=json.dumps(
            {
                "requests": [
                    {"drug_a": "warfarin", "drug_b": "ibuprofen", "pair_id": "batch-1"},
                    {"drug_a": "metformin", "drug_b": "lisinopril", "pair_id": "batch-2"},
                ]
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    server.shutdown()
    thread.join(timeout=2)
    assert response.status == 200
    assert payload["count"] == 2
    assert len(payload["results"]) == 2
