from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time

# Define custom metrics
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total number of inference requests')
INFERENCE_SUCCESS = Counter('inference_success_total', 'Total number of successful inferences')
INFERENCE_LATENCY = Summary('inference_latency_seconds', 'Latency of inference requests in seconds')
INFERENCE_ACTIVE = Gauge('inference_active', 'Number of active inference sessions')

@INFERENCE_LATENCY.time()
def process_inference():
    INFERENCE_REQUESTS.inc()
    INFERENCE_ACTIVE.inc()
    time.sleep(random.uniform(0.1, 0.5))  # Simulated processing delay
    INFERENCE_SUCCESS.inc()
    INFERENCE_ACTIVE.dec()

if __name__ == '__main__':
    start_http_server(8000)  # Prometheus will scrape from this port
    print("✅ Prometheus exporter running at http://localhost:8000/metrics")
    while True:
        process_inference()
        time.sleep(2)
