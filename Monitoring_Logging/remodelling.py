import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import argparse
import os
import gdown

# Argumen input dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="amazon_cleaned_preprocessing.csv")
args = parser.parse_args()
local_path = args.data_path

# GDrive file
gdrive_url = "https://drive.google.com/uc?id=1YZRRtPlVwra-6RiRSeKtKNDzWbZ4w7Cz"

# Cek data lokal
if not os.path.exists(local_path):
    print("⬇️ Mengunduh file dari Google Drive...")
    gdown.download(gdrive_url, local_path, quiet=False)
else:
    print("✅ File sudah tersedia secara lokal.")

# Load data
df = pd.read_csv(local_path)
X = df["clean_text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Autolog aktif
mlflow.set_tracking_uri("file:./mlruns")  # 🔥 Pastikan log ke ./mlruns
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

# Mulai training
with mlflow.start_run(run_name="Training_Pipeline_LogReg"):
    mlflow.set_tag("Dataset", local_path)
    mlflow.log_param("dataset", local_path)

    pipeline.fit(X_train, y_train)
    mlflow.sklearn.log_model(pipeline, artifact_path="logistic_pipeline")

print("✅ Pipeline model berhasil dilatih dan disimpan ke ./mlruns")
