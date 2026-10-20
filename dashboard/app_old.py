import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Tinggi Gelombang",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# PATH MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "model_wave_height_production.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ============================================================
# KONFIGURASI DATABASE
# ============================================================

load_dotenv()


def get_config(key, default=None):
    """Ambil konfigurasi dari Streamlit Secrets atau environment."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


DB_HOST = get_config("DB_HOST", "localhost")
DB_PORT = get_config("DB_PORT", "3306")
DB_NAME = get_config("DB_NAME", "maritim_db")
DB_USER = get_config("DB_USER", "root")
DB_PASSWORD = get_config("DB_PASSWORD", "")


# ============================================================
# LOAD MODEL DAN METADATA
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "File model tidak ditemukan. Pastikan "
        "model/model_wave_height_production.pkl tersedia."
    )
    st.stop()

if not METADATA_PATH.exists():
    st.error(
        "File metadata model tidak ditemukan. Pastikan "
        "model/model_metadata.json tersedia."
    )
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
TARGET = metadata["target"]

model = joblib.load(MODEL_PATH)


# ============================================================
# KONEKSI MYSQL
# ============================================================

@st.cache_resource
def get_engine():
    connection_url = (
        f"mysql+mysqlconnector://"
        f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True
    )


engine = get_engine()


# ============================================================
# QUERY DATA DARI 3 API
# ============================================================

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

        ORDER BY b.timestamp_utc
    """)

    return pd.read_sql(query, engine)


# ============================================================
# LOAD DATA
# ============================================================

try:
    dataset = load_integrated_data()
except Exception as e:
    st.error("Gagal mengambil data dari MySQL.")
    st.exception(e)
    st.stop()

if dataset.empty:
    st.warning("Dataset hasil JOIN 3 API masih kosong.")
    st.stop()

dataset["timestamp_utc"] = pd.to_datetime(
    dataset["timestamp_utc"],
    errors="coerce",
    utc=True
)

dataset = dataset.sort_values(
    ["timestamp_utc", "port_code"]
).reset_index(drop=True)


# ============================================================
# VALIDASI FITUR MODEL
# ============================================================

missing_features = [
    col for col in FEATURES
    if col not in dataset.columns
]

if missing_features:
    st.error(
        "Kolom fitur model tidak ditemukan pada hasil JOIN: "
        + ", ".join(missing_features)
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🌊 Prediksi Tinggi Gelombang")
st.subheader(
    "Berdasarkan Kondisi Meteorologi dan Oseanografi "
    "di Sulawesi Tenggara"
)

st.caption(
    "Sumber data: BMKG Maritime, Open-Meteo, dan Copernicus Marine | "
    "Target model: wave_height"
)


# ============================================================
# KPI
# ============================================================

total_data = len(dataset)
total_ports = dataset["port_code"].nunique()
avg_wave = dataset["wave_height"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Data Terintegrasi", f"{total_data:,}")
col2.metric("Jumlah Pelabuhan", total_ports)
col3.metric("Rata-rata Tinggi Gelombang", f"{avg_wave:.2f} m")


# ============================================================
# FILTER PELABUHAN
# ============================================================

st.divider()

ports = (
    dataset[["port_code", "port_name"]]
    .drop_duplicates()
    .sort_values("port_name")
)

port_options = {
    row["port_name"]: row["port_code"]
    for _, row in ports.iterrows()
}

selected_port_name = st.selectbox(
    "Pilih Pelabuhan",
    list(port_options.keys())
)

selected_port = port_options[selected_port_name]

df_port = dataset[
    dataset["port_code"] == selected_port
].copy()

df_port = df_port.sort_values("timestamp_utc")


# ============================================================
# MONITORING TINGGI GELOMBANG
# ============================================================

st.header("📈 Monitoring Tinggi Gelombang")

fig_wave = px.line(
    df_port,
    x="timestamp_utc",
    y="wave_height",
    markers=True,
    title=f"Tinggi Gelombang - {selected_port_name}"
)

fig_wave.update_layout(
    xaxis_title="Waktu UTC",
    yaxis_title="Tinggi Gelombang (m)"
)

st.plotly_chart(fig_wave, use_container_width=True)


# ============================================================
# KONDISI METEOROLOGI
# ============================================================

st.header("🌦️ Kondisi Meteorologi")

col1, col2 = st.columns(2)

with col1:
    fig_wind = px.line(
        df_port,
        x="timestamp_utc",
        y="wind_speed",
        markers=True,
        title="Kecepatan Angin BMKG"
    )

    fig_wind.update_layout(
        xaxis_title="Waktu UTC",
        yaxis_title="Wind Speed"
    )

    st.plotly_chart(fig_wind, use_container_width=True)

with col2:
    fig_temp = px.line(
        df_port,
        x="timestamp_utc",
        y="temp_avg",
        markers=True,
        title="Temperatur BMKG"
    )

    fig_temp.update_layout(
        xaxis_title="Waktu UTC",
        yaxis_title="Temperatur (°C)"
    )

    st.plotly_chart(fig_temp, use_container_width=True)


# ============================================================
# KELEMBAPAN DAN ARUS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    fig_humidity = px.line(
        df_port,
        x="timestamp_utc",
        y="rh_avg",
        markers=True,
        title="Relative Humidity BMKG"
    )

    fig_humidity.update_layout(
        xaxis_title="Waktu UTC",
        yaxis_title="Relative Humidity (%)"
    )

    st.plotly_chart(fig_humidity, use_container_width=True)

with col2:
    fig_current = px.line(
        df_port,
        x="timestamp_utc",
        y="current_speed_copernicus",
        markers=True,
        title="Kecepatan Arus Laut Copernicus"
    )

    fig_current.update_layout(
        xaxis_title="Waktu UTC",
        yaxis_title="Current Speed"
    )

    st.plotly_chart(fig_current, use_container_width=True)


# ============================================================
# PREDIKSI MODEL
# ============================================================

st.divider()
st.header("🤖 Prediksi Tinggi Gelombang")

latest_data = (
    df_port
    .sort_values("timestamp_utc")
    .tail(1)
    .copy()
)

latest_input = latest_data[FEATURES]

prediction = float(
    model.predict(latest_input)[0]
)

actual_wave = float(
    latest_data["wave_height"].iloc[0]
)

latest_time = latest_data["timestamp_utc"].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Prediksi Tinggi Gelombang",
    f"{prediction:.2f} m"
)

col2.metric(
    "Gelombang Aktual Terakhir",
    f"{actual_wave:.2f} m"
)

difference = prediction - actual_wave

col3.metric(
    "Selisih Prediksi - Aktual",
    f"{difference:+.2f} m"
)

st.caption(
    f"Prediksi menggunakan data terakhir pada {latest_time}."
)


# ============================================================
# DETAIL INPUT MODEL
# ============================================================

with st.expander("Lihat variabel input model"):
    display_input = latest_data[
        ["timestamp_utc", "port_code"] + FEATURES
    ].T

    st.dataframe(
        display_input,
        use_container_width=True
    )


# ============================================================
# AKTUAL VS PREDIKSI
# ============================================================

st.header("📊 Aktual vs Prediksi")

comparison = df_port[
    ["timestamp_utc", "wave_height"] + FEATURES
].copy()

comparison = comparison.dropna(
    subset=["wave_height"]
)

comparison["predicted_wave_height"] = model.predict(
    comparison[FEATURES]
)

plot_data = comparison[
    ["timestamp_utc", "wave_height", "predicted_wave_height"]
].melt(
    id_vars="timestamp_utc",
    var_name="jenis",
    value_name="wave_height"
)

plot_data["jenis"] = plot_data["jenis"].replace({
    "wave_height": "Aktual",
    "predicted_wave_height": "Prediksi"
})

fig_comparison = px.line(
    plot_data,
    x="timestamp_utc",
    y="wave_height",
    color="jenis",
    markers=True,
    title="Tinggi Gelombang Aktual dan Prediksi"
)

fig_comparison.update_layout(
    xaxis_title="Waktu UTC",
    yaxis_title="Tinggi Gelombang (m)"
)

st.plotly_chart(fig_comparison, use_container_width=True)


# ============================================================
# PERFORMA MODEL
# ============================================================

st.header("📋 Performa Model")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metadata['evaluation_mae']:.4f} m")
col2.metric("RMSE", f"{metadata['evaluation_rmse']:.4f} m")
col3.metric("R²", f"{metadata['evaluation_r2']:.4f}")

st.info(
    "Nilai MAE, RMSE, dan R² berasal dari evaluasi model "
    "pada data testing kronologis 80:20. Model production "
    "yang digunakan dashboard dilatih ulang menggunakan "
    "seluruh data yang tersedia."
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("🔎 Feature Importance")

rf_estimator = model.named_steps["model"]

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf_estimator.feature_importances_
}).sort_values(
    "importance",
    ascending=True
)

fig_importance = px.bar(
    importance_df,
    x="importance",
    y="feature",
    orientation="h",
    title="Feature Importance Random Forest"
)

fig_importance.update_layout(
    xaxis_title="Importance",
    yaxis_title="Feature"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)


# ============================================================
# DATA TERINTEGRASI
# ============================================================

st.header("🗃️ Data Terintegrasi")

st.dataframe(
    df_port,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Dashboard melakukan INNER JOIN berdasarkan "
    "port_code + timestamp_utc antara BMKG Maritime, "
    "Open-Meteo, dan Copernicus Marine."
)
