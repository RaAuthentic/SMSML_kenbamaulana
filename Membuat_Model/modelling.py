# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import os
# import zipfile
# import argparse
# import json
#
# # ========================
# # ARGUMEN ZIP & EKSTRAKSI
# # ========================
# parser = argparse.ArgumentParser()
# parser.add_argument("--zip_path", type=str, default="../Membuat_Model/amazon_cleaned_preprocessing.zip")
# args = parser.parse_args()
#
# zip_path = args.zip_path
# csv_path = "../Membuat_Model/amazon_cleaned_preprocessing.csv"
#
# if not os.path.exists(csv_path):
#     print(f"📦 Mengekstrak file dari ZIP: {zip_path}")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(os.path.dirname(csv_path))
#     print(f"✅ Ekstraksi selesai. Dataset tersedia di: {csv_path}")
# else:
#     print(f"✅ File sudah tersedia: {csv_path}")
#
# # ========================
# # LOAD & SPLIT DATASET
# # ========================
# df = pd.read_csv(csv_path)
# X = df["clean_text"]
# y = df["label"]
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # ========================
# # MODEL PIPELINE
# # ========================
# pipeline = Pipeline([
#     ("tfidf", TfidfVectorizer(max_features=3000)),
#     ("clf", LogisticRegression(max_iter=1000))
# ])
#
# # ========================
# # MLflow SETUP (Pastikan tracking & DB tersimpan)
# # ========================
# os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"  # Paksa pakai path lokal eksplisit
# mlflow.set_tracking_uri("file:./mlruns")
# mlflow.set_experiment("Amazon_Review_Classification")
#
# with mlflow.start_run(run_name="Training_LogReg_Manual"):
#
#     # Log parameter pipeline
#     mlflow.log_params({
#         "vectorizer": "TfidfVectorizer",
#         "max_features": 3000,
#         "classifier": "LogisticRegression",
#         "max_iter": 1000,
#         "dataset": csv_path
#     })
#
#     # Train model
#     pipeline.fit(X_train, y_train)
#
#     # Evaluation
#     y_pred = pipeline.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#
#     # Log metrics
#     mlflow.log_metrics({
#         "accuracy": acc,
#         "precision": prec,
#         "recall": rec,
#         "f1_score": f1
#     })
#
#
#
#     # Input example supaya MLflow buat struktur model lengkap
#     input_example = pd.DataFrame(X_train.iloc[:5].values, columns=["clean_text"])
#
#     mlflow.sklearn.log_model(
#         sk_model=pipeline,
#         artifact_path="model",
#         input_example=input_example,
#         registered_model_name="LogisticModelAmazon"
#     )
#
# print("✅ Semua parameter, metrik, dan artefak berhasil disimpan ke MLflow (localhost:5000)")

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import os
import argparse

# === Setup Path & Dataset ===
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, default="amazon_cleaned_preprocessing.csv")
args = parser.parse_args()
csv_path = args.csv_path

df = pd.read_csv(csv_path)
X = df["clean_text"]  # Series
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert X to list for safe logging (autolog won't log, but we prevent error)
X_train_list = X_train.tolist()

# === MLflow Setup ===
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Amazon_Baseline")
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=True)

# === Pipeline & Run ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("clf", LogisticRegression(max_iter=1000))
])

with mlflow.start_run(run_name="Baseline_LogReg_Autolog"):
    pipeline.fit(X_train_list, y_train)
    print("✅ Training selesai dan tercatat ke MLflow.")
