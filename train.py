import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================
# 1. KONFIGURASI
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "heart_disease_uci.csv"
OUTPUT_DIR = BASE_DIR / "model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================
# 2. LOAD DATASET
# ============================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATA_PATH}\n"
        "Periksa nama dan lokasi file CSV."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PEMERIKSAAN AWAL DATASET")
print("=" * 60)

print(f"Jumlah baris awal : {df.shape[0]}")
print(f"Jumlah kolom awal : {df.shape[1]}")

print("\nNama kolom:")
print(df.columns.tolist())

print("\nTipe data:")
print(df.dtypes)

print("\nMissing value:")
print(df.isnull().sum())

print(f"\nDuplikasi baris: {df.duplicated().sum()}")


# ============================================
# 3. DATA CLEANING
# ============================================

# Rapikan nama kolom
df.columns = df.columns.str.strip().str.lower()

# Hapus baris duplikat
df = df.drop_duplicates().copy()

# Pastikan target tersedia
if "num" not in df.columns:
    raise ValueError("Kolom target 'num' tidak ditemukan.")

# Target tidak boleh kosong
df = df.dropna(subset=["num"]).copy()

# Pastikan target valid
valid_labels = {0, 1, 2, 3, 4}

if not set(df["num"].unique()).issubset(valid_labels):
    raise ValueError(
        "Target num harus berisi nilai 0, 1, 2, 3, atau 4."
    )

# Pastikan target numerik
df["num"] = pd.to_numeric(df["num"], errors="raise")

# Hapus kolom identitas
if "id" in df.columns:
    df = df.drop(columns=["id"])

print("\n" + "=" * 60)
print("HASIL PEMBERSIHAN DATA")
print("=" * 60)

print(f"Jumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")

print("\nMissing value setelah cleaning:")
print(df.isnull().sum())

# ============================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# Seluruh hasil EDA disimpan terpisah agar tidak bercampur
# dengan file model dan hasil evaluasi.
EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# 4.1 Ringkasan struktur dataset
print("\n[EDA] Informasi dataset:")
print(f"Jumlah baris : {df.shape[0]}")
print(f"Jumlah kolom : {df.shape[1]}")
print("\nTipe data setiap kolom:")
print(df.dtypes)

# Simpan ringkasan statistik fitur numerik.
# Kolom num disimpan terpisah karena merupakan target asli,
# bukan fitur input model.
numeric_eda = df.select_dtypes(include=["number"]).drop(
    columns=["num"], errors="ignore"
)

if not numeric_eda.empty:
    numeric_summary = numeric_eda.describe().T
    numeric_summary.to_csv(EDA_DIR / "numeric_summary.csv")

    print("\n[EDA] Statistik deskriptif fitur numerik:")
    print(numeric_summary.round(2))

# 4.2 Analisis missing value
missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().mean() * 100).round(2)
}).sort_values("missing_count", ascending=False)

missing_summary.to_csv(EDA_DIR / "missing_value_summary.csv")

print("\n[EDA] Ringkasan missing value:")
print(missing_summary)

missing_plot = missing_summary[missing_summary["missing_count"] > 0]

if not missing_plot.empty:
    plt.figure(figsize=(10, 5))
    plt.bar(
        missing_plot.index,
        missing_plot["missing_count"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Kolom")
    plt.ylabel("Jumlah Missing Value")
    plt.title("Missing Value per Kolom")
    plt.tight_layout()
    plt.savefig(
        EDA_DIR / "missing_value.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# 4.3 Distribusi target asli (num)
target_counts = (
    df["num"]
    .value_counts()
    .sort_index()
    .rename_axis("num")
    .reset_index(name="jumlah")
)

target_counts.to_csv(
    EDA_DIR / "original_target_distribution.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.bar(
    target_counts["num"].astype(str),
    target_counts["jumlah"]
)
plt.xlabel("Nilai num")
plt.ylabel("Jumlah Data")
plt.title("Distribusi Target Asli (num)")
plt.tight_layout()
plt.savefig(
    EDA_DIR / "original_target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 4.4 Distribusi fitur numerik
# Kolom num tidak dimasukkan karena merupakan target.
if not numeric_eda.empty:
    numeric_columns = numeric_eda.columns.tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_columns) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows)
    )
    axes = np.array(axes).reshape(-1)

    for i, column in enumerate(numeric_columns):
        values = pd.to_numeric(
            numeric_eda[column], errors="coerce"
        ).dropna()

        axes[i].hist(values, bins=20, edgecolor="black")
        axes[i].set_title(f"Distribusi {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frekuensi")

    for i in range(len(numeric_columns), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Distribusi Fitur Numerik", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        EDA_DIR / "numeric_distributions.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# 4.5 Distribusi fitur kategorikal
categorical_eda = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

categorical_summary = []

for column in categorical_eda:
    counts = (
        df[column]
        .astype("string")
        .fillna("Missing")
        .value_counts()
        .rename_axis("kategori")
        .reset_index(name="jumlah")
    )
    counts.insert(0, "kolom", column)
    categorical_summary.append(counts)

    # Batasi jumlah kategori yang ditampilkan agar grafik tetap terbaca.
    plot_counts = counts.head(15)

    plt.figure(figsize=(9, 5))
    plt.bar(
        plot_counts["kategori"].astype(str),
        plot_counts["jumlah"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah Data")
    plt.title(f"Distribusi Kategori: {column}")
    plt.tight_layout()
    plt.savefig(
        EDA_DIR / f"categorical_{column}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

if categorical_summary:
    pd.concat(
        categorical_summary,
        ignore_index=True
    ).to_csv(
        EDA_DIR / "categorical_summary.csv",
        index=False
    )

# 4.6 Korelasi antarfitur numerik
# Target num sengaja tidak disertakan agar korelasi fitur
# tidak tercampur dengan analisis target.
if len(numeric_eda.columns) >= 2:
    correlation = numeric_eda.corr(numeric_only=True)
    correlation.to_csv(EDA_DIR / "numeric_correlation.csv")

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )
    ax.set_yticklabels(correlation.columns)

    for row in range(len(correlation.columns)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(image, ax=ax, label="Korelasi")
    ax.set_title("Matriks Korelasi Fitur Numerik")
    fig.tight_layout()
    fig.savefig(
        EDA_DIR / "numeric_correlation.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

# 4.7 Ringkasan hasil EDA
print("\n[EDA] Fitur numerik:")
print(numeric_eda.columns.tolist())

print("\n[EDA] Fitur kategorikal:")
print(categorical_eda)

print(f"\nSeluruh hasil EDA disimpan di: {EDA_DIR}")

# ============================================
# 5. FEATURE ENGINEERING
# ============================================

# Klasifikasi biner:
# num = 0     -> kelas 0
# num = 1..4  -> kelas 1

df["target"] = (df["num"] > 0).astype(int)

# Pisahkan fitur dan target
# Kolom num asli tidak boleh menjadi fitur
X = df.drop(columns=["num", "target"])
y = df["target"]

# Identifikasi fitur berdasarkan tipe data
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nFitur numerik:")
print(numeric_features)

print("\nFitur kategorikal:")
print(categorical_features)

print("\nDistribusi target:")
print(
    y.value_counts()
    .sort_index()
    .rename(index={
        0: "Tidak terindikasi",
        1: "Terindikasi"
    })
)


# ============================================
# 6. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)

print(f"Jumlah data train: {len(X_train)}")
print(f"Jumlah data test : {len(X_test)}")

print("\nProporsi kelas data train:")
print(y_train.value_counts(normalize=True).round(3))

print("\nProporsi kelas data test:")
print(y_test.value_counts(normalize=True).round(3))


# ============================================
# 7. PREPROCESSING
# ============================================

# Fitur numerik:
# Missing value diganti dengan median.
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# Fitur kategorikal:
# Missing value diganti dengan modus.
# Kategori diubah menjadi representasi One-Hot.
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# Gabungkan preprocessing numerik dan kategorikal
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================
# 8. MEMBANGUN MODEL XGBOOST
# ============================================

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Satukan preprocessing dan model.
# Seluruh pipeline nantinya disimpan dalam satu file PKL.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_model)
    ]
)


# ============================================
# 9. TRAINING MODEL
# ============================================

print("\n" + "=" * 60)
print("TRAINING MODEL XGBOOST")
print("=" * 60)

model.fit(X_train, y_train)

print("Training selesai.")


# ============================================
# 10. PREDIKSI DAN EVALUASI
# ============================================

y_pred = model.predict(X_test).astype(int)

# Probabilitas kelas positif
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "precision": float(
        precision_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test, y_pred, zero_division=0
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_test, y_proba)
    )
}

print("\n" + "=" * 60)
print("HASIL EVALUASI XGBOOST")
print("=" * 60)

for name, value in metrics.items():
    print(f"{name.upper():<12}: {value:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Tidak terindikasi",
            "Terindikasi"
        ],
        zero_division=0
    )
)


# ============================================
# 11. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Tidak terindikasi",
        "Terindikasi"
    ]
)

disp.plot(values_format="d")
plt.title("Confusion Matrix - XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 12. FEATURE IMPORTANCE
# ============================================

# Ambil nama fitur setelah One-Hot Encoding
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Ambil importance dari model XGBoost
importance_values = (
    model.named_steps["classifier"]
    .feature_importances_
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
}).sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance.to_string(index=False))

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 7))

top_features = importance.head(15).sort_values(
    by="importance",
    ascending=True
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Fitur")
plt.title("Top 15 Feature Importance - XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 13. SIMPAN MODEL DALAM FORMAT PKL
# ============================================

model_path = OUTPUT_DIR / "xgboost_heart_disease.pkl"

joblib.dump(model, model_path)

print("\nModel berhasil disimpan:")
print(model_path)


# ============================================
# 14. SIMPAN METADATA
# ============================================

metadata = {
    "algorithm": "XGBoost",
    "model_type": "binary_classification",
    "target_column": "target",
    "original_target_column": "num",
    "target_mapping": {
        "0": "Tidak terindikasi (num = 0)",
        "1": "Terindikasi (num > 0)"
    },
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "model_file": model_path.name
}

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================
# 15. RINGKASAN AKHIR
# ============================================

print("\n" + "=" * 60)
print("PROSES TRAINING SELESAI")
print("=" * 60)

print(f"Model       : {model_path}")
print(f"Metadata    : {OUTPUT_DIR / 'metadata.json'}")
print(f"Metrik      : {OUTPUT_DIR / 'metrics.json'}")
print(f"Confusion   : {OUTPUT_DIR / 'confusion_matrix.png'}")
print(f"Importance  : {OUTPUT_DIR / 'feature_importance.csv'}")
print(f"Grafik      : {OUTPUT_DIR / 'feature_importance.png'}")