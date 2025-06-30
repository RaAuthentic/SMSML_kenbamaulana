import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse
import joblib

# ======================
# Argument Handling
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str, default="../Membuat_Model/amazon_cleaned_preprocessing.csv")
args = parser.parse_args()
csv_path = args.csv_path

# ======================
# Load Dataset
# ======================
df = pd.read_csv(csv_path)
X = df["clean_text"]
y = df["label"]

# ======================
# Split Dataset
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Pipeline & Param Grid
# ======================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("clf", LogisticRegression())
])

param_grid = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)

# ======================
# MLflow DagsHub Auth Setup
# ======================
os.environ["MLFLOW_TRACKING_USERNAME"] = "raauthentic"  # Ganti sesuai username kamu
os.environ["MLFLOW_TRACKING_PASSWORD"] = "50d93ffeb34ab4dfe237fe8df77278fdebd2b876"  # Token akses pribadi DagsHub
mlflow.set_tracking_uri("https://dagshub.com/RaAuthentic/SMSML_kenbamaulana.mlflow")
mlflow.set_experiment("Amazon_Review_Classification_Tuning")

# ======================
# MLflow Logging & Training
# ======================
with mlflow.start_run(run_name="Tuning_LogReg_DagsHub"):

    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_features", 3000)
    mlflow.log_param("dataset", csv_path)
    mlflow.log_param("tuning_method", "GridSearchCV")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Logging best params hasil GridSearch
    mlflow.log_params(grid_search.best_params_)

    # Log model ke DagsHub
    model_path = "logreg_tuned_model.pkl"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

print("✅ Tuning selesai dan model terbaik disimpan ke DagsHub (MLflow).")
