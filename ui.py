"""
CardioCheck | Heart Health Screening (Streamlit UI)

Aplikasi prediksi indikasi penyakit jantung memakai model XGBoost lokal.
Tidak membutuhkan FastAPI: model dimuat langsung dari folder `model/`.

Struktur yang dipakai:
    model/xgboost_heart_disease.pkl   -> model terlatih
    model/metadata.json               -> daftar fitur (feature/numeric/categorical)
    model/metrics.json                -> (opsional) metrik evaluasi
    model/confusion_matrix.png        -> (opsional)
    model/feature_importance.png      -> (opsional)
    heart_disease_uci.csv             -> dataset referensi (default & opsi input)
"""

import html
import json
import socket
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "xgboost_heart_disease.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.png"
DATASET_PATH = BASE_DIR / "heart_disease_uci.csv"

HERO_IMAGE_URL = (
    "https://images.pexels.com/photos/5207096/"
    "pexels-photo-5207096.jpeg?cs=srgb&dl=pexels-karola-g-5207096.jpg&fm=jpg"
)

st.set_page_config(
    page_title="CardioCheck | Heart Health Screening",
    layout="wide",
    initial_sidebar_state="auto",  # otomatis tertutup di layar kecil
)


# =========================================================
# 2. CSS RESPONSIF
# =========================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    color-scheme: dark;
    --bg: #080f1d;
    --muted: #91a4bd;
    --text: #eaf1fb;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse at 80% 0%, rgba(49,111,207,.15), transparent 32%),
        radial-gradient(ellipse at 0% 35%, rgba(31,154,174,.07), transparent 30%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"]  { background: rgba(8,15,29,.82); }
[data-testid="stSidebar"] { background: #0d1728; border-right: 1px solid #20324b; }
[data-testid="stSidebar"] * { color: #dce7f6; }

.block-container {
    max-width: 1420px;
    padding: 2rem clamp(.8rem, 3vw, 2.5rem) 3.5rem;
}

h1, h2, h3, h4 { color: #f4f8ff !important; font-family: 'Manrope', sans-serif; letter-spacing: -.45px; }
p, li, label   { color: #c3d0e2; }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ---------- Grid kolom responsif: 2 kolom di desktop, 1 kolom di HP ---------- */
[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; row-gap: 1rem; }
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    flex: 1 1 280px !important;
    min-width: min(100%, 280px) !important;
}

/* ---------- Metric ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14233a, #101b2d);
    border: 1px solid #293e5c;
    border-radius: 16px;
    padding: 17px 19px;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
    overflow-wrap: anywhere;
}
[data-testid="stMetricLabel"] { color: #9fb1c9 !important; font-size: 13px; }
[data-testid="stMetricValue"] {
    color: #f4f8ff !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: clamp(1.35rem, 2.6vw, 2.1rem);
}

/* ---------- Form & input ---------- */
div[data-testid="stForm"] {
    background: linear-gradient(145deg, rgba(18,31,51,.96), rgba(13,24,41,.96));
    border: 1px solid #2a405e;
    border-radius: 20px;
    padding: clamp(14px, 2.2vw, 25px);
    box-shadow: 0 16px 40px rgba(0,0,0,.16);
}
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #0b1627 !important;
    color: #f1f6ff !important;
    border: 1px solid #344b6b !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
    background: #142238 !important; color: #f1f6ff !important;
}
[role="option"] { color: #f1f6ff !important; }
[role="option"]:hover { background: #263d5d !important; }

.stButton > button, .stFormSubmitButton > button {
    width: 100%;
    min-height: 49px;
    border: 1px solid #5595ff;
    border-radius: 11px;
    background: linear-gradient(120deg, #3478ef, #2457c5);
    color: white !important;
    font-weight: 800;
    transition: all .2s ease;
    white-space: normal;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: #8bb8ff;
    background: linear-gradient(120deg, #4a8cff, #326fe0);
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(43,111,224,.22);
}

/* ---------- Hero (grid CSS, bukan kolom Streamlit) ---------- */
.hero {
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr);
    gap: 28px;
    align-items: center;
    background:
        radial-gradient(circle at 88% 20%, rgba(57,133,232,.24), transparent 28%),
        linear-gradient(120deg, #142d4e 0%, #102139 55%, #101b2d 100%);
    border: 1px solid #315077;
    border-radius: 24px;
    padding: clamp(20px, 3.2vw, 36px);
    margin-bottom: 24px;
    box-shadow: 0 18px 50px rgba(0,0,0,.2);
}
.hero-img {
    width: 100%;
    height: 100%;
    max-height: 320px;
    object-fit: cover;
    border-radius: 18px;
    border: 1px solid #315077;
}
.eyebrow {
    color: #77b7ff; font-size: 11px; font-weight: 800;
    letter-spacing: 2.2px; text-transform: uppercase; margin-bottom: 13px;
}
.hero-title {
    color: #f7faff; font-family: 'Manrope', sans-serif;
    font-size: clamp(26px, 3.5vw, 45px); font-weight: 800;
    letter-spacing: -1.3px; line-height: 1.16; margin-bottom: 15px;
}
.hero-desc { color: #c3d3e8; font-size: 14px; line-height: 1.8; margin: 0; }
.hero-chip {
    display: inline-block; margin: 18px 7px 0 0; padding: 7px 12px;
    border: 1px solid #3b5f8a; border-radius: 999px;
    background: rgba(32,70,117,.45); color: #d6eaff;
    font-size: 11px; font-weight: 700;
}

/* ---------- Teks seksi ---------- */
.section-kicker {
    color: #62a6ff; font-size: 10px; font-weight: 800;
    letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 5px;
}
.section-heading {
    color: #f1f6ff; font-family: 'Manrope', sans-serif;
    font-size: clamp(18px, 2.2vw, 21px); font-weight: 800; margin: 3px 0 4px;
}
.section-subtitle { color: #91a4bd; font-size: 13px; line-height: 1.7; margin: 0 0 17px; }

.info-panel {
    background: linear-gradient(145deg, #14243b, #101b2d);
    border: 1px solid #2a405e; border-radius: 16px;
    padding: 19px 21px; margin: 8px 0 17px;
}
.info-panel-title { color: #f1f6ff; font-size: 14px; font-weight: 800; margin-bottom: 6px; }
.info-panel-text  { color: #a9bad0; font-size: 12px; line-height: 1.7; }

/* ---------- Kartu hasil ---------- */
.result-card { border: 1px solid; border-radius: 19px; padding: clamp(16px, 2.4vw, 27px); margin: 10px 0 19px; }
.result-positive { background: linear-gradient(120deg, rgba(112,37,53,.32), rgba(47,25,43,.38)); border-color: #87485b; }
.result-negative { background: linear-gradient(120deg, rgba(15,100,78,.25), rgba(16,48,53,.34)); border-color: #287b68; }
.result-label {
    color: #aebdd0; font-size: 10px; font-weight: 800;
    letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 10px;
}
.result-value {
    color: #f8fbff; font-family: 'Manrope', sans-serif;
    font-size: clamp(22px, 3vw, 35px); font-weight: 800;
    line-height: 1.25; margin-bottom: 9px; overflow-wrap: anywhere;
}
.result-description { color: #c5d2e2; font-size: 13px; line-height: 1.75; }

.small-note { color: #8195af; font-size: 11px; line-height: 1.75; }
.footer {
    border-top: 1px solid #263a55; margin-top: 42px; padding-top: 21px;
    color: #71849d; font-size: 11px; text-align: center; line-height: 1.9;
}

[data-testid="stAlert"]    { border-radius: 12px; }
[data-testid="stExpander"] { background: #101d30; border: 1px solid #293e5b; border-radius: 12px; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #3478ef, #36c5d9); }
hr { border-color: #263a55 !important; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Breakpoint ---------- */
@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; }
    .hero-img { max-height: 220px; }
}
@media (max-width: 640px) {
    .block-container { padding-top: 1rem; }
    .hero { border-radius: 18px; gap: 18px; }
    .hero-img { display: none; }          /* hemat ruang di HP */
    .hero-desc { font-size: 13px; }
    .info-panel { padding: 15px; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL, METADATA, DATASET (DENGAN CACHE)
# =========================================================
@st.cache_resource
def load_model():
    """Memuat model XGBoost sekali saja per sesi server."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_metrics():
    """Metrik evaluasi (opsional). Mengembalikan None jika tidak ada/rusak."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


try:
    model = load_model()
    metadata = load_metadata()
    dataset = load_dataset()
except Exception as error:
    st.error(f"Gagal memuat file aplikasi: {error}")
    st.info(
        "Pastikan `model/xgboost_heart_disease.pkl`, `model/metadata.json`, "
        "dan `heart_disease_uci.csv` berada di lokasi yang sesuai struktur proyek."
    )
    st.stop()


# =========================================================
# 4. KONFIGURASI FITUR
# =========================================================
FEATURE_COLUMNS = metadata.get("feature_columns", [])
NUMERIC_FEATURES = metadata.get("numeric_features", [])
CATEGORICAL_FEATURES = metadata.get("categorical_features", [])

if not FEATURE_COLUMNS:
    st.error("`feature_columns` tidak ditemukan di metadata.json.")
    st.stop()

missing_features = [c for c in FEATURE_COLUMNS if c not in dataset.columns]
if missing_features:
    st.error("Kolom fitur berikut tidak tersedia pada dataset: " + ", ".join(missing_features))
    st.stop()

# Fallback jika metadata belum memuat jenis fitur.
if not NUMERIC_FEATURES and not CATEGORICAL_FEATURES:
    NUMERIC_FEATURES = [
        c for c in FEATURE_COLUMNS if pd.api.types.is_numeric_dtype(dataset[c])
    ]
    CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]


# =========================================================
# 5. FUNGSI BANTUAN (INPUT)
# =========================================================
def format_feature_name(feature) -> str:
    return str(feature).replace("_", " ").replace("-", " ").strip().title()


def get_numeric_series(feature) -> pd.Series:
    return pd.to_numeric(dataset[feature], errors="coerce").dropna()


def get_numeric_default(feature) -> float:
    median_value = get_numeric_series(feature).median()
    return 0.0 if pd.isna(median_value) else float(median_value)


def get_numeric_bounds(feature) -> tuple:
    series = get_numeric_series(feature)
    if series.empty:
        return 0.0, 100.0
    minimum, maximum = float(series.min()), float(series.max())
    if minimum == maximum:
        maximum = minimum + 1.0
    return minimum, maximum


def get_category_options(feature) -> list:
    """Nilai unik kategori sebagai tipe Python native (bukan numpy)."""
    values = [
        v.item() if hasattr(v, "item") else v
        for v in dataset[feature].dropna().unique()
    ]
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def render_feature_input(feature):
    """Membuat widget input sesuai tipe fitur dan mengembalikan nilainya."""
    name = format_feature_name(feature)
    key = f"input_{feature}"

    if feature in CATEGORICAL_FEATURES:
        options = get_category_options(feature)
        if not options:
            st.warning(f"Tidak ada kategori yang tersedia untuk {name}.")
            return None
        return st.selectbox(name, options=options, key=key, help=f"Fitur kategorikal: {feature}")

    if feature in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(dataset[feature]):
        default = get_numeric_default(feature)
        minimum, maximum = get_numeric_bounds(feature)
        series = get_numeric_series(feature)
        is_integer = not series.empty and (series % 1 == 0).all()

        if is_integer:
            lo, hi = int(minimum), int(maximum)
            return st.number_input(
                name, min_value=lo, max_value=hi,
                value=max(lo, min(hi, int(round(default)))),
                step=1, key=key, help=f"Rentang berdasarkan dataset: {lo}–{hi}.",
            )

        padding = max(abs(maximum - minimum) * 0.10, 1.0)
        return st.number_input(
            name, min_value=float(minimum - padding), max_value=float(maximum + padding),
            value=float(default), step=0.1, format="%.2f", key=key,
            help="Nilai awal menggunakan median dataset.",
        )

    options = get_category_options(feature)
    if options:
        return st.selectbox(name, options=options, key=key)
    return st.text_input(name, key=key)


# =========================================================
# 6. FUNGSI PREDIKSI
# =========================================================
def predict(patient_data: dict) -> dict:
    """Menjalankan model XGBoost dan mengembalikan hasil siap tampil."""
    # Urutan kolom harus sama dengan saat training.
    input_df = pd.DataFrame([patient_data], columns=FEATURE_COLUMNS)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    positive_index = list(model.classes_).index(1)

    return {
        "prediction": prediction,
        "label": "Terindikasi penyakit jantung" if prediction == 1 else "Tidak terindikasi",
        "probability_percent": float(probabilities[positive_index]) * 100,
    }


# =========================================================
# 7. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 19px 0;">
            <div style="font-size: 36px;"></div>
            <div style="font-family:Manrope,sans-serif;font-size:24px;
                        font-weight:800;color:#f4f8ff;margin-top:4px;">CardioCheck</div>
            <div style="font-size:11px;color:#8fa4bf;margin-top:4px;">HEART HEALTH SCREENING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### Informasi Model")
    st.markdown(
        f"""
        - **Algoritma:** XGBoost
        - **Jenis:** Klasifikasi biner
        - **Jumlah fitur:** {len(FEATURE_COLUMNS)}
        - **Data referensi:** {len(dataset):,} baris
        """
    )

    st.divider()
    st.markdown("#### Cara Menggunakan")
    st.markdown(
        """
        1. Isi seluruh fitur klinis.
        2. Periksa kembali data.
        3. Klik tombol analisis.
        4. Baca hasil prediksi.
        """
    )
    st.markdown(
        """
        <div class="small-note" style="margin-top:25px;">
            CardioCheck merupakan prototipe pembelajaran.
            Hasil model bukan pengganti diagnosis atau konsultasi medis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 8. HERO
# =========================================================
st.markdown(
    f"""
    <div class="hero">
        <div>
            <div class="eyebrow">AI-POWERED HEALTH SCREENING</div>
            <div class="hero-title">Pahami Kondisi.<br>Peduli Kesehatan Jantung.</div>
            <p class="hero-desc">
                CardioCheck menggunakan model machine learning XGBoost untuk mengolah
                data klinis dan menghasilkan klasifikasi berdasarkan pola yang dipelajari
                dari data pelatihan.
            </p>
            <span class="hero-chip">✦ XGBoost Model</span>
            <span class="hero-chip">◆ Streamlit</span>
            <span class="hero-chip">● MODEL SIAP</span>
        </div>
        <img class="hero-img" src="{HERO_IMAGE_URL}" alt="Heart health"
             loading="lazy" onerror="this.style.display='none'">
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 9. OVERVIEW
# =========================================================
st.markdown('<div class="section-kicker">SYSTEM OVERVIEW</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Ringkasan Aplikasi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Informasi singkat mengenai model dan dataset.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Fitur Model", len(FEATURE_COLUMNS), help="Jumlah fitur yang digunakan model.")
with m2:
    st.metric("Data Referensi", f"{len(dataset):,}", help="Jumlah baris dataset referensi.")
with m3:
    st.metric("Algoritma", "XGBoost")

st.markdown(
    """
    <div class="info-panel">
        <div class="info-panel-title">ⓘ Tentang hasil prediksi</div>
        <div class="info-panel-text">
            Model menghasilkan klasifikasi berdasarkan fitur yang dimasukkan.
            Nilai probabilitas merupakan keluaran model, bukan kepastian seseorang
            mengalami atau tidak mengalami penyakit. Interpretasi kondisi kesehatan
            tetap harus dilakukan oleh tenaga medis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 10. FORM INPUT
# =========================================================
st.markdown('<div class="section-kicker">PATIENT INFORMATION</div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Input Data Klinis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Lengkapi semua fitur berikut. Nilai awal numerik '
    'menggunakan median dataset, sedangkan pilihan kategori diambil dari data yang tersedia.</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form", clear_on_submit=False):
    st.markdown("##### Data yang akan dianalisis")
    st.caption(f"{len(FEATURE_COLUMNS)} fitur digunakan sesuai konfigurasi model pada metadata.json.")

    patient_data = {}
    # Dua kolom di desktop; CSS flex-wrap otomatis menjadikannya satu kolom di HP.
    left_col, right_col = st.columns(2, gap="large")
    for index, feature in enumerate(FEATURE_COLUMNS):
        with (left_col if index % 2 == 0 else right_col):
            patient_data[feature] = render_feature_input(feature)

    st.divider()
    submitted = st.form_submit_button("🩺  ANALISIS DATA & JALANKAN PREDIKSI", use_container_width=True)


# =========================================================
# 11. EKSEKUSI PREDIKSI
# =========================================================
if submitted:
    if any(value is None for value in patient_data.values()):
        st.error("Ada fitur yang belum memiliki nilai. Periksa kembali formulir.")
    else:
        with st.spinner("Menjalankan model XGBoost..."):
            try:
                st.session_state["last_prediction"] = predict(patient_data)
                st.session_state["last_patient_data"] = patient_data
            except Exception as error:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {error}")


# =========================================================
# 12. HASIL
# =========================================================
if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    positive = result["prediction"] == 1
    probability_percent = result["probability_percent"]

    st.divider()
    st.markdown('<div class="section-kicker">ANALYSIS RESULT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Hasil Analisis Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Hasil klasifikasi dari model XGBoost.</div>',
        unsafe_allow_html=True,
    )

    result_class = "result-positive" if positive else "result-negative"
    description = (
        "Model memprediksi adanya indikasi penyakit jantung."
        if positive
        else "Model tidak memprediksi adanya indikasi penyakit jantung."
    )

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-label">HASIL KLASIFIKASI XGBOOST</div>
            <div class="result-value">{html.escape(result["label"])}</div>
            <div class="result-description">{html.escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.metric("Probabilitas Indikasi", f"{probability_percent:.2f}%")
        st.progress(min(max(probability_percent / 100, 0.0), 1.0))
    with r2:
        st.metric("Kelas Prediksi", str(result["prediction"]))
        st.caption("1 = terindikasi penyakit jantung, 0 = tidak terindikasi.")

    with st.expander("Lihat data input yang dianalisis"):
        last_data = st.session_state.get("last_patient_data", {})
        st.dataframe(
            pd.DataFrame(
                [(format_feature_name(k), str(v)) for k, v in last_data.items()],
                columns=["Fitur", "Nilai"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Catatan penting: CardioCheck adalah prototipe untuk pembelajaran, "
        "bukan alat diagnosis medis. Probabilitas model tidak otomatis "
        "merepresentasikan probabilitas klinis seseorang mengalami penyakit. "
        "Konsultasikan kondisi kesehatan dengan tenaga medis."
    )


# =========================================================
# 13. EVALUASI MODEL (OPSIONAL, DARI FILE DI FOLDER model/)
# =========================================================
metrics = load_metrics()
has_cm = CONFUSION_MATRIX_PATH.exists()
has_fi = FEATURE_IMPORTANCE_PATH.exists()

if metrics is not None or has_cm or has_fi:
    st.divider()
    st.markdown('<div class="section-kicker">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Hasil evaluasi saat pelatihan model.</div>',
        unsafe_allow_html=True,
    )

    if metrics is not None:
        with st.expander("Metrik evaluasi"):
            st.json(metrics)

# =========================================================
# 14. FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer">
        <strong style="color:#c8d8ed;">CARDIOCHECK</strong><br>
        Heart Disease Prediction System · XGBoost · Streamlit<br>
        Prototipe pembelajaran dan demonstrasi — bukan pengganti diagnosis medis.<br>
        <span style="color:#9fb1c9;">Served by Host / Instance ID: {html.escape(socket.gethostname())}</span>
    </div>
    """,
    unsafe_allow_html=True,
)