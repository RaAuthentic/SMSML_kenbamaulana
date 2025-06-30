import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer



import argparse
import os
import gdown

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="amazon_cleaned_preprocessing.csv")
args = parser.parse_args()
local_path = args.data_path

gdrive_url = "https://drive.google.com/uc?id=1YZRRtPlVwra-6RiRSeKtKNDzWbZ4w7Cz"

if not os.path.exists(local_path):
    print("⬇️ Mengunduh file dari Google Drive...")
    gdown.download(gdrive_url, local_path, quiet=False)
else:
    print("✅ File sudah tersedia secara lokal.")
# Load ke DataFrame
df = pd.read_csv(local_path)

# 2. Feature and label
X = df["clean_text"]
y = df["label"]

# 3. Vectorization (ringan)
vectorizer = TfidfVectorizer(max_features=3000)
X_vect = vectorizer.fit_transform(X)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# 5. Aktifkan autolog
mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=False)

# 6. Training
with mlflow.start_run(run_name="Training model "):
    mlflow.set_tag("Dataset", "amazon_cleaned_preprocessing.csv")
    mlflow.log_param("dataset", "amazon_cleaned_preprocessing.csv")
    mlflow.set_tag("Dataset", "amazon_cleaned_preprocessing.csv")
    mlflow.log_param("dataset", "amazon_cleaned_preprocessing.csv")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # Logging manual artefak model (opsional, autolog sudah cover)
    mlflow.sklearn.log_model(model, artifact_path="logistic_model")

print("✅ Model selesai dilatih dan disimpan ke MLflow.")
