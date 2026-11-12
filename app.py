from pathlib import Path
import json
import joblib
import socket
import pandas as pd
import streamlit as st

# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xgboost_heart_disease.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"
DATASET_PATH = BASE_DIR / "heart_disease_uci.csv"

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="wide"
)

st.title("Prediksi Penyakit Jantung")
st.write(
    "Aplikasi prediksi indikasi penyakit jantung "
    "menggunakan model machine learning XGBoost langsung di Streamlit."
)


# ==================================================
# 2. MEMUAT MODEL, METADATA, DAN DATASET (DENGAN CACHE)
# ==================================================
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"File metadata tidak ditemukan di: {METADATA_PATH}")
    
    model = joblib.load(MODEL_PATH)
    
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
        
    return model, metadata

@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH)

try:
    model, metadata = load_artifacts()
    feature_columns = metadata.get("feature_columns", [])
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])
    
    dataset = load_dataset()

except Exception as error:
    st.error(f"Gagal memuat artefak atau dataset: {error}")
    st.stop()


# ==================================================
# 3. FORMULIR INPUT PASIEN
# ==================================================
st.subheader("1. Masukkan Data Pasien")
st.caption("Pilihan kategori dan nilai default diambil dari metadata serta dataset.")

patient_data = {}

with st.form("prediction_form"):
    for feature in feature_columns:
        label = feature.replace("_", " ").title()

        if feature in numeric_features:
            # Ambil median dari dataset sebagai nilai default jika ada
            if not dataset.empty and feature in dataset.columns:
                col_vals = pd.to_numeric(dataset[feature], errors="coerce").dropna()
                default_val = float(col_vals.median()) if not col_vals.empty else 0.0
            else:
                default_val = 0.0

            patient_data[feature] = st.number_input(
                label,
                value=default_val,
                key=f"input_{feature}"
            )

        elif feature in categorical_features:
            if not dataset.empty and feature in dataset.columns:
                options = [
                    val.item() if hasattr(val, "item") else val
                    for val in dataset[feature].dropna().unique()
                ]
            else:
                options = []

            if options:
                patient_data[feature] = st.selectbox(
                    label,
                    options=options,
                    key=f"input_{feature}"
                )
            else:
                patient_data[feature] = st.text_input(
                    label,
                    key=f"input_{feature}"
                )
        else:
            # Fallback jika jenis fitur belum terdaftar
            patient_data[feature] = st.text_input(
                label,
                key=f"input_{feature}"
            )

    submitted = st.form_submit_button("Prediksi Penyakit Jantung", use_container_width=True)


# ==================================================
# 4. EKSEKUSI PREDIKSI & MENAMPILKAN HASIL
# ==================================================

if submitted:
    try:
        # Susun DataFrame sesuai urutan fitur saat training model
        input_df = pd.DataFrame(
            [{feature: patient_data[feature] for feature in feature_columns}],
            columns=feature_columns
        )

        # Lakukan prediksi langsung dengan model XGBoost
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        positive_index = classes.index(1)
        positive_probability = float(probabilities[positive_index])
        
        label = "Terindikasi" if prediction == 1 else "Tidak terindikasi"

        # Tampilkan hasil ke UI
        st.divider()
        st.subheader("2. Hasil Prediksi")

        if prediction == 1:
            st.warning("Model memprediksi adanya **indikasi penyakit jantung**.")
        else:
            st.success("Model **tidak memprediksi** adanya indikasi penyakit jantung.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Kelas Prediksi", prediction)
        with col2:
            st.metric("Probabilitas Indikasi", f"{positive_probability * 100:.2f}%")

        st.write(f"**Label Status:** {label}")
        
        # Tambahan informasi Instance ID / Hostname sesuai pembaruan
        st.info(f"**Served by Host / Instance ID:** `{socket.gethostname()}`")
        
        st.caption("Hasil ini merupakan prediksi model machine learning, bukan diagnosis medis resmi.")

    except Exception as error:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {error}")