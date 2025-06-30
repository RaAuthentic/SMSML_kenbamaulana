# import pandas as pd
# import time
# import mlflow
# import mlflow.sklearn
# from dagshub import dagshub_logger
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import os
# import gdown
# import joblib
# import os
# import mlflow
# from dagshub import dagshub_logger
#
#
# # Masukkan token dari DagsHub (token = Personal Access Token)
# os.environ["MLFLOW_TRACKING_USERNAME"] = "RaAuthentic"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "8bd2228a35e3b940c4b76035e2b938a1d0e09d5d"
#
# # Set tracking URI ke DagsHub
# mlflow.set_tracking_uri("https://dagshub.com/RaAuthentic/SMSML_kenbamaulana.mlflow")
# mlflow.set_experiment("SMSML_kenbamaulana")
#
# #  Load data
# # Nama file CSV
# local_path = "amazon_cleaned_preprocessing.csv"
#
# # Link Google Drive direct-download
# gdrive_url = "https://drive.google.com/uc?id=1YZRRtPlVwra-6RiRSeKtKNDzWbZ4w7Cz"
#
# # Download jika belum tersedia
# if not os.path.exists(local_path):
#     print("⬇️ Mengunduh file dari Google Drive...")
#     gdown.download(gdrive_url, local_path, quiet=False)
# else:
#     print("✅ File sudah tersedia secara lokal, tidak perlu download.")
#
# # Load ke DataFrame
# df = pd.read_csv(local_path)
# X = df["clean_text"]
# y = df["label"]
#
# # Vectorization (TF-IDF)
# vectorizer = TfidfVectorizer(max_features=3000)
# X_vectorized = vectorizer.fit_transform(X)
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vectorized, y, test_size=0.2, random_state=42
# )
#
# # Start MLflow + DagsHub logging
# with mlflow.start_run(run_name="Tuning Model") as run, dagshub_logger() as logger:
#     mlflow.log_param("dataset", "amazon_cleaned_preprocessing.csv")
#
#     model = LogisticRegression()
#
#     param_grid = {
#         "C": [0.1, 1.0],
#         "solver": ["liblinear", "saga"]
#     }
#
#     grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
#
#     start = time.time()
#     grid.fit(X_train, y_train)
#     end = time.time()
#
#     best_model = grid.best_estimator_
#     y_pred = best_model.predict(X_test)
#
#     # Metrics
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     train_time = end - start
#     n_features = X_vectorized.shape[1]
#
#     # Logging
#     mlflow.log_params(grid.best_params_)
#     mlflow.log_metric("accuracy", acc)
#     mlflow.log_metric("precision", prec)
#     mlflow.log_metric("recall", rec)
#     mlflow.log_metric("f1_score", f1)
#     mlflow.log_metric("train_time", train_time)
#     mlflow.log_metric("n_features", n_features)
#
#     # Buat folder outputs
#     os.makedirs("outputs", exist_ok=True)
#
#     # Simpan model
#     joblib.dump(best_model, "outputs/logistic_model_tuned.pkl")
#
#     # Log sebagai artifact
#     mlflow.log_artifact("outputs/logistic_model_tuned.pkl")
#
# print("✅ Model tuning & logging selesai.")


import pandas as pd
import time
import os
import gdown
import joblib
import mlflow
import mlflow.sklearn
from dagshub import dagshub_logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 🔐 Setup kredensial untuk DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "RaAuthentic"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8bd2228a35e3b940c4b76035e2b938a1d0e09d5d"

# 🌐 Set URI MLflow Tracking ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/RaAuthentic/SMSML_kenbamaulana.mlflow")
mlflow.set_experiment("SMSML_kenbamaulana")

# 📥 Download dataset jika belum tersedia
local_path = "../MLProject/amazon_cleaned_preprocessing.csv"
gdrive_url = "https://drive.google.com/uc?id=1YZRRtPlVwra-6RiRSeKtKNDzWbZ4w7Cz"

if not os.path.exists(local_path):
    print("⬇️ Mengunduh file dari Google Drive...")
    gdown.download(gdrive_url, local_path, quiet=False)
else:
    print("✅ File sudah tersedia secara lokal.")

# 📊 Load dataset
df = pd.read_csv(local_path)
X = df["clean_text"]
y = df["label"]

# ✨ Vectorisasi TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(X)

# 📦 Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# 🚀 Start training dan logging ke MLflow + DagsHub
with mlflow.start_run(run_name="Tuning Model") as run, dagshub_logger() as logger:
    mlflow.log_param("dataset", local_path)

    # 🔧 Setup & tuning
    model = LogisticRegression()
    param_grid = {
        "C": [0.1, 1.0],
        "solver": ["liblinear", "saga"]
    }
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)

    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()

    # ✅ Evaluasi model terbaik
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    train_time = end - start
    n_features = X_vectorized.shape[1]

    # 🧾 Logging ke MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_time": train_time,
        "n_features": n_features
    })

    # 💾 Simpan dan log model sebagai artifact
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/logistic_model_tuned.pkl"
    joblib.dump(best_model, model_path)

    if os.path.exists(model_path):
        print(f"📁 Model berhasil disimpan di: {model_path}")
        mlflow.log_artifact(model_path)
    else:
        print("❌ Gagal menyimpan model! File tidak ditemukan.")

print("✅ Model tuning & logging selesai.")

