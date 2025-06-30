import json
import time
import requests
from datetime import datetime
from prometheus_client import Counter, Gauge, CollectorRegistry, make_wsgi_app
from wsgiref.simple_server import make_server
import threading

# === 1. Prometheus Custom Metrics ===
registry = CollectorRegistry()

INFERENCE_REQUESTS = Counter("inference_requests_total", "Total number of inference requests", registry=registry)
INFERENCE_SUCCESS = Counter("inference_success_total", "Total number of successful inferences", registry=registry)
INFERENCE_FAILURE = Counter("inference_failure_total", "Total number of failed inferences", registry=registry)
INFERENCE_ACCURACY = Gauge("inference_accuracy", "Accuracy of model inference (1=correct, 0=wrong)", registry=registry)
INFERENCE_ACTIVE = Gauge("inference_active", "Number of active inference processes", registry=registry)

# === 2. Start Prometheus Exporter ===
def start_custom_http_server():
    app = make_wsgi_app(registry)
    httpd = make_server('', 8000, app)
    print("✅ Prometheus exporter running at http://localhost:8000/metrics")
    httpd.serve_forever()

threading.Thread(target=start_custom_http_server, daemon=True).start()

# === 3. Dummy Inference Requests to MLflow Model ===
sample_data = {
    "columns": ["clean_text"],
    "data": [
        ["This product is amazing, I love it!"],
        ["Terrible experience. Would not recommend."]
    ]
}

endpoint_url = "http://localhost:5000/invocations"

# === 4. Simulasi Inference Berkala ===
while True:
    print(f"\n🧪 [{datetime.now().isoformat()}] Sending inference request...")
    INFERENCE_ACTIVE.inc()
    INFERENCE_REQUESTS.inc()

    try:
        response = requests.post(endpoint_url, json=sample_data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction result:", result)

            # Simulasi akurasi: jika hasil 1 untuk review positif
            simulated_accuracy = 1 if result[0] == 1 else 0
            INFERENCE_SUCCESS.inc()
            INFERENCE_ACCURACY.set(simulated_accuracy)
        else:
            print("❌ Inference failed:", response.text)
            INFERENCE_FAILURE.inc()

    except Exception as e:
        print("❌ Error during inference:", str(e))
        INFERENCE_FAILURE.inc()

    INFERENCE_ACTIVE.dec()
    time.sleep(30)
