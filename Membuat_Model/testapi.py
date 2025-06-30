import mlflow

mlflow.set_tracking_uri("https://dagshub.com/RaAuthentic/SMSML_kenbamaulana.mlflow")
mlflow.set_experiment("SMSML_kenbamaulana")

with mlflow.start_run():
    mlflow.log_param("test_param", 123)
