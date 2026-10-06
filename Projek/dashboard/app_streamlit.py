import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Maritime Wave Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL UI
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #f5f8fc;
}

[data-testid="stHeader"] {
    background: rgba(245,248,252,0.88);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071b2e 0%, #0b2d47 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
    color: #e9f3fb !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 9px 12px;
    border-radius: 10px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,.08);
}

.hero {
    padding: 30px 34px;
    border-radius: 22px;
    background:
        radial-gradient(circle at 88% 18%, rgba(42,184,218,.30), transparent 28%),
        radial-gradient(circle at 12% 80%, rgba(32,99,155,.30), transparent 30%),
        linear-gradient(135deg, #071b2e 0%, #0b3b59 58%, #087c9e 100%);
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 16px 45px rgba(5,38,62,.16);
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.35rem;
    line-height: 1.08;
    margin: 0 0 10px 0;
}

.hero p {
    color: #cfe8f4;
    margin: 0;
    font-size: 1rem;
    max-width: 850px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    color: #0b2d47;
    font-size: 1.55rem;
    font-weight: 700;
    margin: 1.4rem 0 .7rem 0;
}

.kpi {
    background: white;
    border: 1px solid #e3ebf2;
    border-radius: 17px;
    padding: 19px 20px;
    box-shadow: 0 8px 24px rgba(13,49,73,.06);
}

.kpi-label {
    color: #688092;
    font-size: .82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
}

.kpi-value {
    color: #09283f;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 4px;
}

.kpi-note {
    color: #8293a0;
    font-size: .78rem;
    margin-top: 3px;
}

.info-card {
    background: white;
    border: 1px solid #e3ebf2;
    border-radius: 17px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(13,49,73,.05);
}

.result-card {
    background: linear-gradient(135deg, #0a2f48, #087d9f);
    color: white;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 14px 38px rgba(6,66,91,.18);
}

.result-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 10px 0;
}

.small-muted {
    color: #738797;
    font-size: .86rem;
}

.footer {
    margin-top: 42px;
    padding-top: 18px;
    border-top: 1px solid #dce5ec;
    color: #81919e;
    font-size: .78rem;
    text-align: center;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3ebf2;
    padding: 12px 15px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(13,49,73,.05);
}

.stButton > button {
    border-radius: 11px;
    font-weight: 700;
    border: 0;
    min-height: 42px;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# PATHS / MODEL
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model_wave_height_production.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

load_dotenv()

def get_config(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# Support both local MySQL names and TiDB Cloud names.
DB_HOST = get_config("DB_HOST") or get_config("TIDB_HOST", "localhost")
DB_PORT = get_config("DB_PORT") or get_config("TIDB_PORT", "3306")
DB_NAME = get_config("DB_NAME") or get_config("TIDB_DATABASE", "maritim_db")
DB_USER = get_config("DB_USER") or get_config("TIDB_USER", "root")
DB_PASSWORD = get_config("DB_PASSWORD") or get_config("TIDB_PASSWORD", "")

if not MODEL_PATH.exists():
    st.error("File model tidak ditemukan: model/model_wave_height_production.pkl")
    st.stop()

if not METADATA_PATH.exists():
    st.error("File metadata model tidak ditemukan: model/model_metadata.json")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
TARGET = metadata.get("target", "wave_height")
model = joblib.load(MODEL_PATH)

# ============================================================
# DATABASE
# ============================================================
@st.cache_resource
def get_engine():
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(url, pool_pre_ping=True)

engine = get_engine()

@st.cache_data(ttl=300)
def load_integrated_data():
    query = text("""
        SELECT
            b.port_code,
            b.port_name,
            b.timestamp_utc,
            b.latitude,
            b.longitude,
            b.temp_avg,
            b.rh_avg,
            b.wind_speed,
            b.wind_gust,
            b.wave_height,
            b.wave_cat,
            b.current_speed_bmkg,
            o.temperature_2m,
            o.relative_humidity_2m,
            o.surface_pressure,
            o.precipitation,
            o.wind_speed_10m,
            o.wind_direction_10m,
            c.uo,
            c.vo,
            c.current_speed_copernicus
        FROM bmkg_maritime b
        INNER JOIN openmeteo_weather o
            ON b.port_code = o.port_code
            AND b.timestamp_utc = o.timestamp_utc
        INNER JOIN copernicus_marine c
            ON b.port_code = c.port_code
            AND b.timestamp_utc = c.timestamp_utc
        ORDER BY b.timestamp_utc, b.port_code
    """)
    return pd.read_sql(query, engine)

try:
    dataset = load_integrated_data()
except Exception as e:
    st.error("Gagal mengambil data dari database.")
    st.exception(e)
    st.stop()

if dataset.empty:
    st.warning("Dataset hasil JOIN 3 API masih kosong.")
    st.stop()

dataset["timestamp_utc"] = pd.to_datetime(
    dataset["timestamp_utc"], errors="coerce", utc=True
)
dataset = dataset.dropna(subset=["timestamp_utc"]).sort_values(
    ["timestamp_utc", "port_code"]
).reset_index(drop=True)

missing_features = [c for c in FEATURES if c not in dataset.columns]
if missing_features:
    st.error("Fitur model tidak ditemukan: " + ", ".join(missing_features))
    st.stop()

# ============================================================
# HELPERS
# ============================================================
port_lookup = (
    dataset[["port_code", "port_name"]]
    .drop_duplicates()
    .sort_values("port_name")
)
port_name_to_code = dict(zip(port_lookup["port_name"], port_lookup["port_code"]))
port_options = ["Semua Pelabuhan"] + port_lookup["port_name"].tolist()

def kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_title(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)

def clean_port_filter(df, chosen):
    if chosen == "Semua Pelabuhan":
        return df.copy()
    code = port_name_to_code[chosen]
    return df[df["port_code"] == code].copy()

def display_df_without_code(df):
    hidden = [c for c in ["port_code", "id"] if c in df.columns]
    return df.drop(columns=hidden)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 22px 4px;">
            <div style="font-size:2.1rem;">🌊</div>
            <div style="font-family:'Space Grotesk';font-size:1.25rem;font-weight:700;">
                Maritime Wave
            </div>
            <div style="font-size:.78rem;color:#a9c6d8;">
                Intelligence Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "NAVIGASI",
        [
            "📊 Monitoring",
            "🤖 Prediksi Tinggi Gelombang",
            "📈 Evaluasi Model",
            "ℹ️ Tentang Sistem",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:.8rem;color:#b9d1df;">
            <b>Data terintegrasi</b><br>
            {len(dataset):,} observasi<br>
            {dataset['port_name'].nunique()} pelabuhan<br>
            3 sumber data
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MONITORING
# ============================================================
if menu == "📊 Monitoring":
    st.markdown(
        """
        <div class="hero">
            <h1>Monitoring Kondisi Gelombang</h1>
            <p>
                Pantau tinggi gelombang dan kondisi meteorologi–oseanografi
                pada pelabuhan di Sulawesi Tenggara berdasarkan data terintegrasi.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Observasi", f"{len(dataset):,}", "hasil JOIN 3 sumber")
    with c2:
        kpi("Pelabuhan", dataset["port_name"].nunique(), "cakupan lokasi")
    with c3:
        kpi("Rata-rata Gelombang", f"{dataset['wave_height'].mean():.2f} m", "seluruh observasi")
    with c4:
        kpi("Maksimum Gelombang", f"{dataset['wave_height'].max():.2f} m", "nilai tertinggi")

    st.markdown("")
    chosen = st.selectbox("Pilih Pelabuhan", port_options, key="monitor_port")
    dfm = clean_port_filter(dataset, chosen).sort_values("timestamp_utc")

    section_title(
        "Tren Tinggi Gelombang",
        "Perubahan tinggi gelombang terhadap waktu UTC.",
    )
    fig = px.line(
        dfm,
        x="timestamp_utc",
        y="wave_height",
        markers=True,
        template="plotly_white",
    )
    fig.update_traces(line_width=2.5)
    fig.update_layout(
        height=390,
        xaxis_title="Waktu UTC",
        yaxis_title="Tinggi Gelombang (m)",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        section_title("Distribusi Tinggi Gelombang")
        fig2 = px.histogram(
            dfm, x="wave_height", nbins=20, template="plotly_white"
        )
        fig2.update_layout(
            height=340, xaxis_title="Tinggi Gelombang (m)", yaxis_title="Frekuensi"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with b:
        section_title("Angin vs Tinggi Gelombang")
        fig3 = px.scatter(
            dfm,
            x="wind_speed",
            y="wave_height",
            hover_data=["port_name", "timestamp_utc"],
            template="plotly_white",
        )
        fig3.update_layout(
            height=340,
            xaxis_title="Kecepatan Angin BMKG",
            yaxis_title="Tinggi Gelombang (m)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    section_title("Rata-rata Gelombang per Pelabuhan")
    avg_port = (
        dataset.groupby("port_name", as_index=False)["wave_height"]
        .mean()
        .sort_values("wave_height", ascending=False)
    )
    fig4 = px.bar(
        avg_port,
        x="wave_height",
        y="port_name",
        orientation="h",
        template="plotly_white",
        text_auto=".2f",
    )
    fig4.update_layout(
        height=390,
        xaxis_title="Rata-rata Tinggi Gelombang (m)",
        yaxis_title="Pelabuhan",
    )
    st.plotly_chart(fig4, use_container_width=True)

    section_title("Observasi Terbaru")
    latest = (
        dfm.sort_values("timestamp_utc", ascending=False)
        .head(15)
    )
    latest = display_df_without_code(latest)
    st.dataframe(latest, use_container_width=True, hide_index=True)

# ============================================================
# PREDICTION
# ============================================================
elif menu == "🤖 Prediksi Tinggi Gelombang":
    st.markdown(
        """
        <div class="hero">
            <h1>Prediksi Tinggi Gelombang</h1>
            <p>
                Masukkan kondisi meteorologi dan oseanografi untuk memperoleh
                estimasi tinggi gelombang menggunakan Random Forest Regression.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chosen = st.selectbox(
        "Pelabuhan",
        port_lookup["port_name"].tolist(),
        key="prediction_port",
    )
    code = port_name_to_code[chosen]
    dport = dataset[dataset["port_code"] == code].sort_values("timestamp_utc")
    latest = dport.tail(1).iloc[0]

    use_latest = st.checkbox(
        "Gunakan observasi terbaru sebagai nilai awal",
        value=True,
    )

    defaults = latest if use_latest else pd.Series(dtype=float)

    st.markdown("### Input Kondisi")
    st.caption("Nilai dapat disesuaikan sebelum prediksi.")

    left_features = FEATURES[:7]
    right_features = FEATURES[7:]

    values = {}

    c1, c2 = st.columns(2)
    with c1:
        for feature in left_features:
            default = float(defaults[feature]) if use_latest and pd.notna(defaults[feature]) else 0.0
            values[feature] = st.number_input(
                feature,
                value=default,
                format="%.4f",
                key=f"pred_{feature}",
            )
    with c2:
        for feature in right_features:
            default = float(defaults[feature]) if use_latest and pd.notna(defaults[feature]) else 0.0
            values[feature] = st.number_input(
                feature,
                value=default,
                format="%.4f",
                key=f"pred_{feature}",
            )

    input_df = pd.DataFrame([values], columns=FEATURES)

    if st.button("🌊 Hitung Tinggi Gelombang", type="primary", use_container_width=True):
        prediction = float(model.predict(input_df)[0])

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:.86rem;opacity:.82;">ESTIMASI TINGGI GELOMBANG</div>
                <div class="result-number">{prediction:.2f} m</div>
                <div style="opacity:.88;">{chosen}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.info(
            "Catatan ilmiah: model saat ini merupakan estimasi pada kondisi "
            "waktu yang sama (same-timestamp estimation). Ini belum merupakan "
            "forecast t+1 atau prediksi beberapa jam ke depan."
        )

# ============================================================
# MODEL EVALUATION
# ============================================================
elif menu == "📈 Evaluasi Model":
    st.markdown(
        """
        <div class="hero">
            <h1>Evaluasi dan Performa Model</h1>
            <p>
                Evaluasi Random Forest Regression menggunakan pemisahan data
                kronologis 80:20 untuk menghindari kebocoran informasi waktu.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_df = dataset.dropna(subset=["wave_height"]).copy()
    split_idx = int(len(eval_df) * 0.8)
    train = eval_df.iloc[:split_idx].copy()
    test = eval_df.iloc[split_idx:].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    eval_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    with st.spinner("Menghitung evaluasi model..."):
        eval_model.fit(X_train, y_train)
        pred = eval_model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("MAE", f"{mae:.4f} m", "rata-rata error absolut")
    with c2:
        kpi("RMSE", f"{rmse:.4f} m", "root mean squared error")
    with c3:
        kpi("R²", f"{r2:.4f}", "proporsi variasi yang dijelaskan")

    comparison = test[
        ["timestamp_utc", "port_name", "wave_height"]
    ].copy()
    comparison["predicted_wave_height"] = pred

    chosen = st.selectbox(
        "Pilih Pelabuhan",
        ["Semua Pelabuhan"] + sorted(comparison["port_name"].unique()),
        key="eval_port",
    )

    chart = comparison.copy()
    if chosen != "Semua Pelabuhan":
        chart = chart[chart["port_name"] == chosen]

    # FIX UTAMA:
    # Jangan melt dengan value_name="wave_height" karena kolom wave_height
    # sudah ada di dataframe. Kita ubah kedua kolom menjadi "value" terlebih dahulu.
    actual_plot = chart[
        ["timestamp_utc", "port_name", "wave_height"]
    ].rename(columns={"wave_height": "value"})
    actual_plot["series"] = "Aktual"

    predicted_plot = chart[
        ["timestamp_utc", "port_name", "predicted_wave_height"]
    ].rename(columns={"predicted_wave_height": "value"})
    predicted_plot["series"] = "Prediksi"

    plot_data = pd.concat(
        [actual_plot, predicted_plot],
        ignore_index=True,
    )

    section_title("Aktual vs Prediksi")
    fig = px.line(
        plot_data,
        x="timestamp_utc",
        y="value",
        color="series",
        markers=True,
        template="plotly_white",
    )
    fig.update_layout(
        height=450,
        xaxis_title="Waktu UTC",
        yaxis_title="Tinggi Gelombang (m)",
        legend_title="",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        section_title("Scatter Aktual vs Prediksi")
        scatter = px.scatter(
            chart,
            x="wave_height",
            y="predicted_wave_height",
            hover_data=["port_name", "timestamp_utc"],
            template="plotly_white",
        )
        lo = min(chart["wave_height"].min(), chart["predicted_wave_height"].min())
        hi = max(chart["wave_height"].max(), chart["predicted_wave_height"].max())
        scatter.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Prediksi ideal",
                line=dict(dash="dash"),
            )
        )
        scatter.update_layout(
            height=390,
            xaxis_title="Aktual (m)",
            yaxis_title="Prediksi (m)",
        )
        st.plotly_chart(scatter, use_container_width=True)

    with b:
        section_title("Distribusi Residual")
        residual = chart["wave_height"] - chart["predicted_wave_height"]
        residual_df = pd.DataFrame({"residual": residual})
        resid_fig = px.histogram(
            residual_df,
            x="residual",
            nbins=20,
            template="plotly_white",
        )
        resid_fig.add_vline(x=0, line_dash="dash")
        resid_fig.update_layout(
            height=390,
            xaxis_title="Residual (Aktual - Prediksi)",
            yaxis_title="Frekuensi",
        )
        st.plotly_chart(resid_fig, use_container_width=True)

    section_title("Performa per Pelabuhan")
    rows = []
    for port, group in comparison.groupby("port_name"):
        rows.append(
            {
                "Pelabuhan": port,
                "MAE (m)": mean_absolute_error(
                    group["wave_height"], group["predicted_wave_height"]
                ),
                "RMSE (m)": np.sqrt(
                    mean_squared_error(
                        group["wave_height"], group["predicted_wave_height"]
                    )
                ),
                "R²": r2_score(
                    group["wave_height"], group["predicted_wave_height"]
                )
                if len(group) >= 2
                else np.nan,
            }
        )
    port_metrics = pd.DataFrame(rows).sort_values("MAE (m)")
    st.dataframe(
        port_metrics.style.format(
            {"MAE (m)": "{:.4f}", "RMSE (m)": "{:.4f}", "R²": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    section_title("Feature Importance")
    fitted_rf = eval_model.named_steps["model"]
    importance = pd.DataFrame(
        {"Fitur": FEATURES, "Importance": fitted_rf.feature_importances_}
    ).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        importance,
        x="Importance",
        y="Fitur",
        orientation="h",
        template="plotly_white",
    )
    fig_imp.update_layout(height=480)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "Angka evaluasi pada halaman ini dihitung ulang dari pembagian kronologis "
        "80:20. Model production yang digunakan pada halaman prediksi adalah "
        "model yang telah dilatih menggunakan seluruh data tersedia."
    )

# ============================================================
# ABOUT
# ============================================================
else:
    st.markdown(
        """
        <div class="hero">
            <h1>Tentang Sistem</h1>
            <p>
                Platform analitik untuk memahami hubungan kondisi meteorologi
                dan oseanografi dengan tinggi gelombang di wilayah pelabuhan
                Sulawesi Tenggara.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="info-card">
                <h3>Arsitektur Data</h3>
                <p>
                Sistem menggabungkan tiga sumber data menggunakan
                <b>port_code + timestamp_utc</b> sebagai kunci integrasi.
                </p>
                <ul>
                    <li>BMKG Maritime — tinggi gelombang dan cuaca maritim</li>
                    <li>Open-Meteo — kondisi meteorologi</li>
                    <li>Copernicus Marine — komponen arus laut</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="info-card">
                <h3>Machine Learning</h3>
                <p>
                Model menggunakan <b>Random Forest Regression</b> dengan target
                <b>wave_height</b>.
                </p>
                <ul>
                    <li>14 variabel input</li>
                    <li>Evaluasi kronologis 80:20</li>
                    <li>MAE, RMSE, dan R²</li>
                    <li>Feature importance</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        """
        <div class="info-card">
            <h3>Catatan Interpretasi</h3>
            <p>
            Dashboard ini merupakan prototipe analitik dan pendukung keputusan.
            Hasil model tidak boleh dianggap sebagai satu-satunya dasar keputusan
            keselamatan pelayaran. Model saat ini mengestimasi tinggi gelombang
            pada kondisi waktu yang sama; untuk forecast masa depan diperlukan
            formulasi target berbasis horizon waktu, misalnya t+1 atau t+3 jam.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("### Status Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Observasi", f"{len(dataset):,}", "hasil JOIN")
    with c2:
        kpi("Pelabuhan", dataset["port_name"].nunique(), "Sulawesi Tenggara")
    with c3:
        kpi("Sumber", "3 API", "BMKG · Open-Meteo · Copernicus")

st.markdown(
    """
    <div class="footer">
        Maritime Wave Intelligence · Data Science Project · Sulawesi Tenggara
    </div>
    """,
    unsafe_allow_html=True,
)
