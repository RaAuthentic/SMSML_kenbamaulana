# 📊 Sentiment Analysis & ML Monitoring Project

### 🔍 Overview

This project is part of the final submission for the **Machine Learning Operations** class at Dicoding. It implements a complete **Sentiment Analysis pipeline using the Amazon Reviews dataset** with **end-to-end MLOps integration**, including:

- **Model training & experiment tracking with MLflow**
- **Model serving with MLflow server**
- **Inference monitoring with Prometheus**
- **Visualization & alerting with Grafana**

> **Author**: Keindra Bagas Maulana (kenbamaulana)

---

### 🧠 Model & Dataset

- **Model**: Logistic Regression + TF-IDF
- **Dataset**: [Amazon Polarity Sentiment](https://huggingface.co/datasets/amazon_polarity) (Used: 200,000 samples)
- **Classes**: Positive (1), Negative (0)
- **Vectorizer**: TF-IDF (Top 3000 features)

---

### ⚙️ Project Structure

```bash
.
├── MLProject/                 # MLflow Project config
├── Membuat_Model/            # Training, tuning, and saved runs
├── Monitoring_Logging/
│   ├── inference.py          # Inference + Prometheus push
│   ├── prometheus.yml        # Prometheus config
│   ├── prometheus_exporter.py
│   └── Data/                 # JSON input for inference
├── bukti monitoring grafana/
├── bukti monitoring prometheus/
└── README.md
```


---

### 🚀 Model Serving & Inference
Model is served using MLflow local server:


```bash
mlflow models serve -m "mlruns/0/models/<run_id>/artifacts" --port 5001 --env-manager=local
```


Example inference request (10 positive samples):


```bash
import requests
import pandas as pd

data = pd.DataFrame({
    "clean_text": [
        "This is the best product I've ever used!",
        "Absolutely love it!",
        "Exceeded all expectations.",
        "Great quality and fast shipping.",
        "Customer service was outstanding!",
        "Highly recommend to everyone.",
        "Amazing performance.",
        "Super easy to use.",
        "Very helpful and reliable.",
        "Perfectly met my needs!"
    ]
})

response = requests.post(
    url="http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    json={"inputs": data["clean_text"].tolist()}
)

print(response.json())
```
### 📈 Monitoring Dashboard

#### ✅ Prometheus Metrics (localhost:9090)

Monitored custom metrics include:

- `inference_requests_total`: Total number of inference requests
- `inference_success_total`: Total number of successful inferences
- `inference_failure_total`: Total number of failed inferences
- `inference_accuracy`: Accuracy of the inference model
- `inference_latency_seconds`: Inference request latency in seconds
- `inference_active`: Number of active inference sessions

---

#### 📉 Grafana Visualization

<p align="center">
  <img src="https://github.com/RaAuthentic/SMSML_kenbamaulana/blob/e6b661637a159221ec89ce48231d36778d3cfe4d/Monitoring_Logging/bukti%20monitoring%20prometheus/inference_accuracy.png" width="700"/>
  <br/><br/>
  <img src="https://github.com/RaAuthentic/SMSML_kenbamaulana/blob/e6b661637a159221ec89ce48231d36778d3cfe4d/Monitoring_Logging/bukti%20monitoring%20grafana/inference_accuracy.png" width="700"/>
</p>

---

### 🎯 Results & Notes

- **Model Accuracy**: > 80%
- **Inference Latency**: ~0.4s (average)
- **Metrics Interval**: Custom Prometheus metrics pushed every 30 seconds
- **Dashboard**: Visualized via Grafana panel (time-based)

---

### 🙏 Acknowledgement

Thanks to the Dicoding team for their guidance.


@kenbamaulana
