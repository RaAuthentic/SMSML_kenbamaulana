import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import argparse
import os
import zipfile


# Argumen input ZIP
parser = argparse.ArgumentParser()
parser.add_argument("--zip_path", type=str, default="../Membuat_Model/amazon_cleaned_preprocessing.zip")
args = parser.parse_args()

zip_path = args.zip_path
csv_path = "../Membuat_Model/amazon_cleaned_preprocessing.csv"

# ✅ Cek dan ekstrak jika CSV belum tersedia
if not os.path.exists(csv_path):
    print(f"📦 Mengekstrak file dari ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(csv_path))
    print(f"✅ Ekstraksi selesai. Dataset tersedia di: {csv_path}")
else:
    print(f"✅ File sudah tersedia: {csv_path}")

# Load dataset
df = pd.read_csv(csv_path)
X = df["clean_text"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Tracking ke local ./mlruns
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Amazon Review Classification")
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

# Contoh input untuk log_model
input_example = X_train.iloc[:5].tolist()  # 5 contoh kalimat teks

# Logging run
with mlflow.start_run(run_name="Training_Pipeline_LogReg"):
    mlflow.set_tag("Dataset", csv_path)
    mlflow.log_param("dataset", csv_path)

    pipeline.fit(X_train, y_train)

    # Tambahkan log_model MANUAL untuk memastikan artefak tersimpan
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="logistic_pipeline",
        input_example=input_example
    )

print("✅ Pipeline model berhasil dilatih dan artefaknya tersimpan di MLflow.")
