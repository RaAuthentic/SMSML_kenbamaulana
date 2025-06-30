import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import gdown


def clean_text(content):
    text = str(content).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def preprocess_amazon(input_path, output_path_csv, eda_report_path, fig_folder):
    print("📥 Memuat data...")
    df = pd.read_csv(input_path, low_memory=False, header=None, names=["label", "title", "content"])

    print("🔍 Jumlah data:", df.shape[0])
    print("📊 Distribusi label:\n", df['label'].value_counts(), "\n")

    print("❓ Missing values:\n", df.isnull().sum())
    df.dropna(inplace=True)
    print("✅ Missing values setelah drop:\n", df.isnull().sum())

    # Buang baris header duplikat jika ada
    df = df[df["label"] != "label"]
    df["label"] = df["label"].astype(int)

    # Cek ulang jumlah data cukup atau tidak
    if df[df['label'] == 0].shape[0] < 100_000 or df[df['label'] == 1].shape[0] < 100_000:
        raise ValueError("Jumlah data label 0 atau 1 kurang dari 100.000. Gagal sampling.")

    print("📊 Sampling 100.000 positif dan negatif...")
    positif = df[df['label'] == 0].sample(n=100_000, random_state=42)
    negatif = df[df['label'] == 1].sample(n=100_000, random_state=42)
    df_sampled = pd.concat([positif, negatif]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Setelah sampling, total data: {df_sampled.shape[0]}")
    print("Distribusi label:\n", df_sampled['label'].value_counts())

    # Buat ulang kolom statistik setelah sampling
    df_sampled['text_length'] = df_sampled['content'].apply(len)
    df_sampled['word_count'] = df_sampled['content'].apply(lambda x: len(str(x).split()))
    df_sampled['title_length'] = df_sampled['title'].apply(lambda x: len(str(x)))

    os.makedirs(fig_folder, exist_ok=True)

    plt.figure(figsize=(10, 4))
    sns.histplot(df_sampled['text_length'], bins=50, kde=True)
    plt.title("Distribusi Panjang Teks (Jumlah Karakter)")
    plt.xlabel("Jumlah Karakter")
    plt.ylabel("Frekuensi")
    plt.savefig(os.path.join(fig_folder, "hist_teks_panjang.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.histplot(df_sampled['word_count'], bins=50, kde=True)
    plt.title("Distribusi Panjang Teks (Jumlah Kata)")
    plt.xlabel("Jumlah Kata")
    plt.ylabel("Frekuensi")
    plt.savefig(os.path.join(fig_folder, "hist_jumlah_kata.png"))
    plt.close()

    # Buat eda report berdasarkan hasil sampling
    os.makedirs(os.path.dirname(eda_report_path), exist_ok=True)
    with open(eda_report_path, "w") as f:
        f.write("Jumlah Data: " + str(df_sampled.shape[0]) + "\n")
        f.write("Distribusi Label:\n" + df_sampled['label'].value_counts().to_string() + "\n\n")
        f.write("Panjang Teks:\n" + df_sampled['text_length'].describe().to_string() + "\n\n")
        f.write("Jumlah Kata:\n" + df_sampled['word_count'].describe().to_string() + "\n\n")
        f.write("Panjang Judul:\n" + df_sampled['title_length'].describe().to_string() + "\n")

    print("🧼 Cleaning teks...")
    df_sampled['clean_text'] = df_sampled['content'].apply(clean_text)

    os.makedirs(os.path.dirname(output_path_csv), exist_ok=True)
    df_sampled.to_csv(output_path_csv, index=False)
    print(f"✅ File preprocessing disimpan ke {output_path_csv}")



if __name__ == "__main__":
    # Lokasi file lokal
    input_path = "dataset_amazon_raw.csv"
    output_path_csv = "amazon_cleaned_preprocessing.csv"

    # Link Google Drive (dikonversi ke direct-download)
    raw_url = "https://drive.google.com/uc?id=1gnkJYuGo3Slb2euugnutV_HDFrFDmrdD"
    processed_url = "https://drive.google.com/uc?id=1YZRRtPlVwra-6RiRSeKtKNDzWbZ4w7Cz"

    # Download RAW jika belum ada
    if not os.path.exists(input_path):
        print("⬇️ Mengunduh dataset RAW dari Google Drive...")
        gdown.download(raw_url, input_path, quiet=False)
    else:
        print("✅ File RAW sudah ada, skip download.")

    # Download file hasil preprocessing jika belum ada
    if not os.path.exists(output_path_csv):
        print("⬇️ Mengunduh hasil preprocessing dari Google Drive...")
        gdown.download(processed_url, output_path_csv, quiet=False)
    else:
        print("✅ File hasil preprocessing sudah ada, skip download.")

    # Jalankan preprocessing ulang jika ingin overwrite
    preprocess_amazon(
        input_path=input_path,
        output_path_csv="..\preprocessing/amazon_cleaned_preprocessing.csv",
        eda_report_path="amazon_eda_report/eda_report.txt",
        fig_folder="amazon_eda_report/fig"
    )
