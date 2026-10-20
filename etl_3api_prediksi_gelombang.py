import os
import time
import math
import shutil
import tempfile
import requests
import numpy as np
import pandas as pd
import xarray as xr
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import copernicusmarine

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

# ============================================================
# KONFIGURASI DATABASE
# ============================================================

# Database lokal (backup/development)
LOCAL_DATABASE_URL = os.getenv(
    "LOCAL_DATABASE_URL",
    "mysql+mysqlconnector://root:root@host.docker.internal:3306/maritime_wave_db",
)

local_engine = create_engine(
    LOCAL_DATABASE_URL,
    pool_pre_ping=True,
)

# Engine TiDB dibuat SEKALI lalu dipakai ulang (tidak membuat pool baru
# setiap kali dipanggil).
_TIDB_ENGINE = None


def get_tidb_engine():
    """Buat (sekali) koneksi TiDB Cloud dari environment variable Airflow."""
    global _TIDB_ENGINE
    if _TIDB_ENGINE is not None:
        return _TIDB_ENGINE

    host = os.getenv("TIDB_HOST")
    port = os.getenv("TIDB_PORT", "4000")
    user = os.getenv("TIDB_USER")
    password = os.getenv("TIDB_PASSWORD")
    database = os.getenv("TIDB_DATABASE", "maritime_wave_db")

    missing = [
        name
        for name, value in {
            "TIDB_HOST": host,
            "TIDB_USER": user,
            "TIDB_PASSWORD": password,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Konfigurasi TiDB Cloud belum lengkap. Variabel yang belum ada: "
            + ", ".join(missing)
        )

    # quote_plus diperlukan agar username/password aman jika mengandung
    # karakter khusus seperti @, :, /, #, atau spasi.
    url = (
        f"mysql+mysqlconnector://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}"
    )

    # TiDB Cloud Public Endpoint mewajibkan TLS.
    _TIDB_ENGINE = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"ssl_disabled": False},
    )
    return _TIDB_ENGINE


def get_db_targets():
    """Kembalikan dua target database: (nama, fungsi pembuat engine).

    Engine TiDB baru dibuat saat dipakai, sehingga konfigurasi TiDB yang
    belum lengkap hanya menggagalkan target TiDB, bukan target lokal.
    """
    return [
        ("MYSQL LOKAL", lambda: local_engine),
        ("TIDB CLOUD", get_tidb_engine),
    ]


def run_on_targets(action, errors, label):
    """Jalankan action(engine) di setiap target database.

    Kegagalan satu target dicatat di `errors` dan target lain tetap
    diproses. Target yang sudah gagal pada task ini dilewati.
    Mengembalikan dict {nama_target: hasil_action}.
    """
    results = {}
    for db_name, engine_getter in get_db_targets():
        if db_name in errors:
            continue
        try:
            results[db_name] = action(engine_getter())
        except Exception as e:
            errors[db_name] = f"{label}: {type(e).__name__}: {e}"
            print(f"   ERROR {db_name} ({label}): {e}")
    return results


def raise_if_errors(errors, task_name):
    if errors:
        detail = " | ".join(f"{k} -> {v}" for k, v in errors.items())
        raise RuntimeError(f"{task_name}: gagal menulis ke database. {detail}")

# ============================================================
# TANGGAL DATA
# ============================================================

# Tidak digunakan sebagai tanggal tetap.
# Timestamp BMKG menjadi sumber kebenaran (master timestamp).
DATE = None

# ============================================================
# KONFIGURASI 7 PELABUHAN
# ============================================================

PORTS = {
    "AF001": {
        "name": "Pelabuhan Nusantara Kendari",
        "lat": -3.973083,
        "lon": 122.583639,
        "bmkg_area": "P.AF.04",
    },
    "AF004": {
        "name": "Pelabuhan Kolaka",
        "lat": -4.057490,
        "lon": 121.589599,
        "bmkg_area": "P.AF.15",
    },
    "AF006": {
        "name": "Pelabuhan Raha",
        "lat": -4.853700,
        "lon": 122.743000,
        "bmkg_area": "P.AF.08",
    },
    "AF013": {
        "name": "Pelabuhan Sikeli",
        "lat": -5.262207,
        "lon": 121.794983,
        "bmkg_area": "P.AF.16",
    },
    "AF003": {
        "name": "Pelabuhan Murhum Baubau",
        "lat": -5.454722,
        "lon": 122.610000,
        "bmkg_area": "P.AF.11",
    },
    "AF031": {
        "name": "Pelabuhan Ereke-Waode Buri",
        "lat": -4.773011,
        "lon": 123.173311,
        "bmkg_area": "P.AF.13",
    },
    "AF033": {
        "name": "Pelabuhan Wanci",
        "lat": -3.969325,
        "lon": 122.613412,
        "bmkg_area": "P.AF.18",
    },
}

# ============================================================
# HELPER TIMESTAMP
# ============================================================

def normalize_timestamp_series(series):
    """Normalisasi timestamp ke UTC naive, presisi sampai jam."""
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.floor("h").dt.tz_localize(None)


def timestamp_strings(df):
    """Ubah timestamp dataframe menjadi list string untuk XCom."""
    return [
        ts.strftime("%Y-%m-%d %H:%M:%S")
        for ts in pd.to_datetime(df["timestamp_utc"], utc=True)
    ]

# ============================================================
# 1. AMBIL DATA BMKG
# ============================================================

def get_bmkg(port):
    area = PORTS[port]["bmkg_area"]
    url = (
        f"https://maritim.bmkg.go.id/"
        f"marine2026-data/perairan/{area}.json"
    )

    print(f"   Mengambil BMKG {port}...")

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    # PENTING:
    # Day 1 = 24 titik hourly.
    # Day 2-4 = titik 3-hourly.
    day1 = data.get("forecast_day1", []) or []
    day2_4 = data.get("forecast_day2-4", []) or []
    records = day1 + day2_4

    print(
        f"   BMKG {port}: day1={len(day1)}, "
        f"day2-4={len(day2_4)}, total={len(records)}"
    )

    rows = []

    for item in records:
        timestamp = item.get("time")
        if not timestamp:
            continue

        timestamp = str(timestamp).replace(" UTC", "")
        ts = pd.to_datetime(timestamp, errors="coerce", utc=True)

        if pd.isna(ts):
            continue

        rows.append({
            "port_code": port,
            "timestamp_utc": ts,
            "temp_avg": item.get("temp_avg"),
            "rh_avg": item.get("rh_avg"),
            "wind_speed": item.get("wind_speed"),
            "wind_gust": item.get("wind_gust"),
            "wave_cat": item.get("wave_cat"),
            "wave_height": item.get("wave_height"),
            "current_speed_bmkg": item.get("current_speed"),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"BMKG {port}: data kosong setelah parsing.")

    df["timestamp_utc"] = normalize_timestamp_series(df["timestamp_utc"])

    # Paksa kolom numerik agar nilai aneh (mis. "-" atau teks) menjadi NULL
    # dan tidak membuat INSERT ke database error.
    for col in ["temp_avg", "rh_avg", "wind_speed", "wind_gust",
                "wave_height", "current_speed_bmkg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp_utc").drop_duplicates(
        ["port_code", "timestamp_utc"], keep="last"
    ).reset_index(drop=True)

    print(
        f"   BMKG {port}: {len(df)} data, "
        f"{df['timestamp_utc'].min()} sampai {df['timestamp_utc'].max()}"
    )

    return df

# ============================================================
# 2. AMBIL DATA OPEN-METEO BERDASARKAN TIMESTAMP BMKG
# ============================================================

def get_openmeteo(port, target_timestamps):
    if not target_timestamps:
        raise ValueError(f"Open-Meteo {port}: target timestamp kosong.")

    lat = PORTS[port]["lat"]
    lon = PORTS[port]["lon"]

    target_ts = normalize_timestamp_series(pd.Series(target_timestamps))
    target_ts = pd.DatetimeIndex(target_ts.dropna().unique()).sort_values()

    if len(target_ts) == 0:
        raise ValueError(f"Open-Meteo {port}: tidak ada timestamp valid.")

    start_date = target_ts.min().strftime("%Y-%m-%d")
    end_date = target_ts.max().strftime("%Y-%m-%d")
    target_set = set(target_ts)

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "precipitation,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
        
    }

    for attempt in range(1, 4):
        try:
            print(
                f"   Mengambil Open-Meteo {port} "
                f"{start_date} s.d. {end_date} "
                f"(percobaan {attempt}/3)..."
            )

            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            hourly = data["hourly"]

            df = pd.DataFrame({
                "port_code": port,
                "timestamp_utc": pd.to_datetime(
                    hourly["time"], utc=True
                ),
                "temperature_2m": hourly["temperature_2m"],
                "relative_humidity_2m": hourly["relative_humidity_2m"],
                "surface_pressure": hourly["surface_pressure"],
                "precipitation": hourly["precipitation"],
                "wind_speed_10m": hourly["wind_speed_10m"],
                "wind_direction_10m": hourly["wind_direction_10m"],
            })

            df["timestamp_utc"] = normalize_timestamp_series(
                df["timestamp_utc"]
            )

            # MASTER TIMESTAMP: hanya ambil waktu yang diberikan BMKG.
            df = df[df["timestamp_utc"].isin(target_set)].copy()
            df = df.sort_values("timestamp_utc").drop_duplicates(
                ["port_code", "timestamp_utc"], keep="last"
            ).reset_index(drop=True)

            missing = sorted(target_set - set(df["timestamp_utc"]))
            if missing:
                missing_text = ", ".join(
                    ts.strftime("%Y-%m-%d %H:%M:%S") for ts in missing[:10]
                )
                raise ValueError(
                    f"Open-Meteo {port}: {len(missing)} timestamp BMKG "
                    f"tidak ditemukan. Contoh: {missing_text}"
                )

            print(f"   Open-Meteo {port}: {len(df)} data cocok dengan BMKG")
            return df

        except Exception as e:
            print(
                f"   WARNING Open-Meteo {port} "
                f"percobaan {attempt} gagal: {e}"
            )
            if attempt < 3:
                print("   Menunggu 3 detik...")
                time.sleep(3)
            else:
                raise RuntimeError(f"Open-Meteo gagal untuk {port}: {e}")

# ============================================================
# 3. AMBIL DATA COPERNICUS BERDASARKAN TIMESTAMP BMKG
# ============================================================

def get_copernicus(port, target_timestamps):
    if not (
        os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
        and os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")
    ):
        raise RuntimeError(
            "Credential Copernicus Marine tidak tersedia di environment "
            "Airflow. Pastikan COPERNICUSMARINE_SERVICE_USERNAME dan "
            "COPERNICUSMARINE_SERVICE_PASSWORD sudah terkonfigurasi."
        )

    if not target_timestamps:
        raise ValueError(f"Copernicus {port}: target timestamp kosong.")

    lat = PORTS[port]["lat"]
    lon = PORTS[port]["lon"]

    target_ts = normalize_timestamp_series(pd.Series(target_timestamps))
    target_ts = pd.DatetimeIndex(target_ts.dropna().unique()).sort_values()

    if len(target_ts) == 0:
        raise ValueError(f"Copernicus {port}: tidak ada timestamp valid.")

    # Folder sementara UNIK per pemanggilan, dihapus setelah selesai.
    # (Versi lama memakai ulang file .nc lama sehingga data Copernicus
    # tidak pernah diperbarui dan file menumpuk.)
    base_dir = "copernicus_runtime"
    os.makedirs(base_dir, exist_ok=True)
    output_dir = tempfile.mkdtemp(prefix=f"{port}_", dir=base_dir)

    start_ts = target_ts.min()
    end_ts = target_ts.max()

    start_text = start_ts.strftime("%Y-%m-%dT%H:%M:%S")
    end_text = end_ts.strftime("%Y-%m-%dT%H:%M:%S")

    safe_start = start_ts.strftime("%Y%m%d%H%M")
    safe_end = end_ts.strftime("%Y%m%d%H%M")
    filename = f"{port}_{safe_start}_{safe_end}.nc"
    output_path = os.path.join(output_dir, filename)

    min_lon = lon - 0.30
    max_lon = lon + 0.30
    min_lat = lat - 0.30
    max_lat = lat + 0.30

    print(
        f"   Mengambil Copernicus {port}: "
        f"{start_text} s.d. {end_text}..."
    )

    try:
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
            variables=["uo", "vo"],
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_depth=0,
            maximum_depth=1,
            start_datetime=start_text,
            end_datetime=end_text,
            output_filename=filename,
            output_directory=output_dir,
        )
        ds = xr.open_dataset(output_path)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    try:
        # ----------------------------------------------------
        # Nama koordinat
        # ----------------------------------------------------
        lat_name = None
        lon_name = None

        for name in ["latitude", "lat"]:
            if name in ds.coords:
                lat_name = name
                break

        for name in ["longitude", "lon"]:
            if name in ds.coords:
                lon_name = name
                break

        if lat_name is None or lon_name is None:
            raise ValueError("Koordinat Copernicus tidak ditemukan.")

        # ----------------------------------------------------
        # Ambil variabel
        # ----------------------------------------------------
        uo = ds["uo"]
        vo = ds["vo"]
        current = np.sqrt(uo ** 2 + vo ** 2)

        lat_values = np.asarray(ds[lat_name].values).reshape(-1)
        lon_values = np.asarray(ds[lon_name].values).reshape(-1)

        # ----------------------------------------------------
        # Cari grid laut valid terdekat
        # ----------------------------------------------------
        valid_points = []

        for ilat, grid_lat in enumerate(lat_values):
            for ilon, grid_lon in enumerate(lon_values):
                try:
                    current_values = (
                        current
                        .isel({lat_name: ilat, lon_name: ilon})
                        .values
                    )
                    current_values = np.asarray(current_values).squeeze()
                    valid_count = np.sum(~np.isnan(current_values))

                    if valid_count > 0:
                        distance = math.sqrt(
                            (grid_lat - lat) ** 2
                            + (grid_lon - lon) ** 2
                        )
                        valid_points.append((distance, ilat, ilon))
                except Exception:
                    continue

        if not valid_points:
            raise ValueError(
                f"Grid laut valid tidak ditemukan untuk {port}."
            )

        valid_points.sort(key=lambda x: x[0])
        _, ilat, ilon = valid_points[0]

        print(
            f"   Grid Copernicus {port}: "
            f"lat={lat_values[ilat]}, lon={lon_values[ilon]}"
        )

        # ----------------------------------------------------
        # Ambil data pada grid terpilih
        # ----------------------------------------------------
        selected_uo = uo.isel({lat_name: ilat, lon_name: ilon})
        selected_vo = vo.isel({lat_name: ilat, lon_name: ilon})
        selected_current = current.isel({lat_name: ilat, lon_name: ilon})

        timestamps = pd.to_datetime(
            selected_current["time"].values,
            utc=True,
        )

        uo_values = np.asarray(selected_uo.values).squeeze().reshape(-1)
        vo_values = np.asarray(selected_vo.values).squeeze().reshape(-1)
        current_values = (
            np.asarray(selected_current.values).squeeze().reshape(-1)
        )
        timestamps = np.asarray(timestamps).reshape(-1)

        if not (
            len(timestamps) == len(uo_values)
            == len(vo_values)
            == len(current_values)
        ):
            raise ValueError(
                f"Jumlah timestamp/UO/VO/current Copernicus tidak sama "
                f"untuk {port}."
            )

        df = pd.DataFrame({
            "port_code": port,
            "timestamp_utc": timestamps,
            "uo": uo_values.astype(float),
            "vo": vo_values.astype(float),
            "current_speed_copernicus": current_values.astype(float),
        })

        df["timestamp_utc"] = normalize_timestamp_series(
            df["timestamp_utc"]
        )

        target_set = set(target_ts)
        df = df[df["timestamp_utc"].isin(target_set)].copy()
        df = df.sort_values("timestamp_utc").drop_duplicates(
            ["port_code", "timestamp_utc"], keep="last"
        ).reset_index(drop=True)

        missing = sorted(target_set - set(df["timestamp_utc"]))
        if missing:
            missing_text = ", ".join(
                ts.strftime("%Y-%m-%d %H:%M:%S") for ts in missing[:10]
            )
            raise ValueError(
                f"Copernicus {port}: {len(missing)} timestamp BMKG "
                f"tidak ditemukan. Contoh: {missing_text}"
            )

        print(f"   Copernicus {port}: {len(df)} data cocok dengan BMKG")
        return df

    finally:
        ds.close()
        shutil.rmtree(output_dir, ignore_errors=True)

# ============================================================
# 4. NORMALISASI TIMESTAMP PER SUMBER
# ============================================================

def normalize_source(df, source_name):
    df = df.copy()

    if "timestamp_utc" not in df.columns:
        raise ValueError(
            f"{source_name}: kolom timestamp_utc tidak ditemukan."
        )

    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    invalid = int(ts.isna().sum())

    if invalid:
        raise ValueError(
            f"{source_name}: {invalid} timestamp tidak valid."
        )

    df["timestamp_utc"] = ts.dt.floor("h").dt.tz_localize(None)

    dup = int(df.duplicated(["port_code", "timestamp_utc"]).sum())
    if dup:
        raise ValueError(
            f"{source_name}: {dup} duplicate port_code + timestamp_utc."
        )

    return df


def clean_records(df):
    """Ubah DataFrame jadi list dict; NaN/NaT -> None, numpy -> Python."""
    cleaned = []
    for record in df.to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                row[key] = None
            elif isinstance(value, pd.Timestamp):
                row[key] = value.to_pydatetime()
            elif hasattr(value, "item"):
                row[key] = value.item()
            else:
                row[key] = value
        cleaned.append(row)
    return cleaned

# ============================================================
# 5. TABEL DATABASE - ports + 3 API (SAMA PERSIS dengan file .sql)
# ============================================================

DDL_PORTS = """
    CREATE TABLE IF NOT EXISTS ports (
        port_code  VARCHAR(10)   NOT NULL,
        port_name  VARCHAR(100)  NOT NULL,
        latitude   DECIMAL(10,6) NOT NULL,
        longitude  DECIMAL(10,6) NOT NULL,
        bmkg_area  VARCHAR(20)   NOT NULL,
        created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (port_code),
        UNIQUE KEY uq_ports_bmkg_area (bmkg_area)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_BMKG = """
    CREATE TABLE IF NOT EXISTS bmkg_maritime (
        id                 BIGINT      NOT NULL AUTO_INCREMENT,
        port_code          VARCHAR(10) NOT NULL,
        timestamp_utc      DATETIME    NOT NULL,
        temp_avg           FLOAT DEFAULT NULL,
        rh_avg             FLOAT DEFAULT NULL,
        wind_speed         FLOAT DEFAULT NULL,
        wind_gust          FLOAT DEFAULT NULL,
        wave_height        FLOAT DEFAULT NULL,
        wave_cat           VARCHAR(50) DEFAULT NULL,
        current_speed_bmkg FLOAT DEFAULT NULL,
        created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_bmkg_port_time (port_code, timestamp_utc),
        KEY idx_bmkg_timestamp (timestamp_utc),
        CONSTRAINT fk_bmkg_port FOREIGN KEY (port_code)
            REFERENCES ports (port_code) ON UPDATE CASCADE ON DELETE RESTRICT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_OPENMETEO = """
    CREATE TABLE IF NOT EXISTS openmeteo_weather (
        id                   BIGINT      NOT NULL AUTO_INCREMENT,
        port_code            VARCHAR(10) NOT NULL,
        timestamp_utc        DATETIME    NOT NULL,
        temperature_2m       FLOAT DEFAULT NULL,
        relative_humidity_2m FLOAT DEFAULT NULL,
        surface_pressure     FLOAT DEFAULT NULL,
        precipitation        FLOAT DEFAULT NULL,
        wind_speed_10m       FLOAT DEFAULT NULL,
        wind_direction_10m   FLOAT DEFAULT NULL,
        created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_openmeteo_port_time (port_code, timestamp_utc),
        KEY idx_openmeteo_timestamp (timestamp_utc),
        CONSTRAINT fk_openmeteo_port FOREIGN KEY (port_code)
            REFERENCES ports (port_code) ON UPDATE CASCADE ON DELETE RESTRICT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

DDL_COPERNICUS = """
    CREATE TABLE IF NOT EXISTS copernicus_marine (
        id                       BIGINT      NOT NULL AUTO_INCREMENT,
        port_code                VARCHAR(10) NOT NULL,
        timestamp_utc            DATETIME    NOT NULL,
        uo                       FLOAT DEFAULT NULL,
        vo                       FLOAT DEFAULT NULL,
        current_speed_copernicus FLOAT DEFAULT NULL,
        created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        UNIQUE KEY uq_copernicus_port_time (port_code, timestamp_utc),
        KEY idx_copernicus_timestamp (timestamp_utc),
        CONSTRAINT fk_copernicus_port FOREIGN KEY (port_code)
            REFERENCES ports (port_code) ON UPDATE CASCADE ON DELETE RESTRICT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

SQL_UPSERT_PORT = """
    INSERT INTO ports (port_code, port_name, latitude, longitude, bmkg_area)
    VALUES (:port_code, :port_name, :latitude, :longitude, :bmkg_area)
    ON DUPLICATE KEY UPDATE
        port_name = VALUES(port_name),
        latitude = VALUES(latitude),
        longitude = VALUES(longitude),
        bmkg_area = VALUES(bmkg_area)
"""

SQL_CHECK_OLD_SCHEMA = """
    SELECT COUNT(*) FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name IN ('bmkg_maritime', 'openmeteo_weather',
                         'copernicus_marine')
      AND column_name IN ('port_name', 'latitude', 'longitude')
"""

BATCH_SIZE = 500


def _prepare_one_database(engine):
    with engine.begin() as conn:
        # Urutan penting: tabel induk (ports) dulu, baru tabel anak.
        for ddl in (DDL_PORTS, DDL_BMKG, DDL_OPENMETEO, DDL_COPERNICUS):
            conn.execute(text(ddl))

        # Tabel lama (masih punya port_name/latitude/longitude) membuat
        # INSERT gagal. Beri pesan yang jelas.
        old_cols = conn.execute(text(SQL_CHECK_OLD_SCHEMA)).scalar()
        if old_cols:
            raise RuntimeError(
                "Skema LAMA terdeteksi (tabel sumber masih punya kolom "
                "port_name/latitude/longitude). Jalankan "
                "maritime_wave_db_skema.sql pada database ini terlebih dahulu."
            )

        # Sinkronkan master pelabuhan dari konfigurasi PORTS.
        conn.execute(
            text(SQL_UPSERT_PORT),
            [
                {
                    "port_code": code,
                    "port_name": cfg["name"],
                    "latitude": cfg["lat"],
                    "longitude": cfg["lon"],
                    "bmkg_area": cfg["bmkg_area"],
                }
                for code, cfg in PORTS.items()
            ],
        )
    return True


def prepare_database(errors):
    """Pastikan ports + 3 tabel sumber ada dan data pelabuhan sinkron."""
    for db_name, _ in get_db_targets():
        print(f"\nMengecek/membuat tabel di {db_name}...")
    ok = run_on_targets(_prepare_one_database, errors, "persiapan tabel")
    for db_name in ok:
        print(f"Tabel ports + 3 API siap di {db_name}.")


def upsert_records(engine, sql, records):
    """Upsert batch (executemany) dalam satu transaksi."""
    with engine.begin() as conn:
        for i in range(0, len(records), BATCH_SIZE):
            conn.execute(text(sql), records[i:i + BATCH_SIZE])
    return len(records)

# ============================================================
# 6. LOAD BMKG
# ============================================================

SQL_UPSERT_BMKG = """
    INSERT INTO bmkg_maritime (
        port_code, timestamp_utc,
        temp_avg, rh_avg, wind_speed, wind_gust,
        wave_height, wave_cat, current_speed_bmkg
    ) VALUES (
        :port_code, :timestamp_utc,
        :temp_avg, :rh_avg, :wind_speed, :wind_gust,
        :wave_height, :wave_cat, :current_speed_bmkg
    )
    ON DUPLICATE KEY UPDATE
        temp_avg = VALUES(temp_avg),
        rh_avg = VALUES(rh_avg),
        wind_speed = VALUES(wind_speed),
        wind_gust = VALUES(wind_gust),
        wave_height = VALUES(wave_height),
        wave_cat = VALUES(wave_cat),
        current_speed_bmkg = VALUES(current_speed_bmkg)
"""

BMKG_COLUMNS = [
    "port_code", "timestamp_utc",
    "temp_avg", "rh_avg", "wind_speed", "wind_gust",
    "wave_height", "wave_cat", "current_speed_bmkg",
]


def load_bmkg(df, errors):
    df = normalize_source(df, "BMKG")
    records = clean_records(df[BMKG_COLUMNS])
    results = run_on_targets(
        lambda eng: upsert_records(eng, SQL_UPSERT_BMKG, records),
        errors, "load BMKG",
    )
    for db_name, n in results.items():
        print(f"   BMKG -> {db_name}: DIPROSES={n}")
    return results

# ============================================================
# 7. LOAD OPEN-METEO
# ============================================================

SQL_UPSERT_OPENMETEO = """
    INSERT INTO openmeteo_weather (
        port_code, timestamp_utc,
        temperature_2m, relative_humidity_2m, surface_pressure,
        precipitation, wind_speed_10m, wind_direction_10m
    ) VALUES (
        :port_code, :timestamp_utc,
        :temperature_2m, :relative_humidity_2m, :surface_pressure,
        :precipitation, :wind_speed_10m, :wind_direction_10m
    )
    ON DUPLICATE KEY UPDATE
        temperature_2m = VALUES(temperature_2m),
        relative_humidity_2m = VALUES(relative_humidity_2m),
        surface_pressure = VALUES(surface_pressure),
        precipitation = VALUES(precipitation),
        wind_speed_10m = VALUES(wind_speed_10m),
        wind_direction_10m = VALUES(wind_direction_10m)
"""

OPENMETEO_COLUMNS = [
    "port_code", "timestamp_utc",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "precipitation", "wind_speed_10m", "wind_direction_10m",
]


def load_openmeteo(df, errors):
    df = normalize_source(df, "Open-Meteo")
    records = clean_records(df[OPENMETEO_COLUMNS])
    results = run_on_targets(
        lambda eng: upsert_records(eng, SQL_UPSERT_OPENMETEO, records),
        errors, "load Open-Meteo",
    )
    for db_name, n in results.items():
        print(f"   Open-Meteo -> {db_name}: DIPROSES={n}")
    return results

# ============================================================
# 8. LOAD COPERNICUS
# ============================================================

SQL_UPSERT_COPERNICUS = """
    INSERT INTO copernicus_marine (
        port_code, timestamp_utc,
        uo, vo, current_speed_copernicus
    ) VALUES (
        :port_code, :timestamp_utc,
        :uo, :vo, :current_speed_copernicus
    )
    ON DUPLICATE KEY UPDATE
        uo = VALUES(uo),
        vo = VALUES(vo),
        current_speed_copernicus = VALUES(current_speed_copernicus)
"""

COPERNICUS_COLUMNS = [
    "port_code", "timestamp_utc", "uo", "vo", "current_speed_copernicus",
]


def load_copernicus(df, errors):
    df = normalize_source(df, "Copernicus")
    records = clean_records(df[COPERNICUS_COLUMNS])
    results = run_on_targets(
        lambda eng: upsert_records(eng, SQL_UPSERT_COPERNICUS, records),
        errors, "load Copernicus",
    )
    for db_name, n in results.items():
        print(f"   Copernicus -> {db_name}: DIPROSES={n}")
    return results


def add_to_totals(totals, result):
    for db_name, n in result.items():
        totals[db_name] = totals.get(db_name, 0) + n


def print_etl_results(title, totals, errors):
    print(f"\n{title}")
    for db_name, n in totals.items():
        print(f"  {db_name}: {n} baris diproses")
    for db_name, msg in errors.items():
        print(f"  {db_name}: GAGAL -> {msg}")

# ============================================================
# 9. TASK AIRFLOW: ETL BMKG
# ============================================================

def etl_bmkg(**context):
    print("\n" + "=" * 90)
    print("ETL BMKG - DAY 1 + DAY 2-4")
    print("=" * 90)

    errors = {}
    prepare_database(errors)

    timestamp_map = {}
    totals = {}

    for port in PORTS:
        print(f"\n--- BMKG: {port} ---")

        df = get_bmkg(port)
        df = normalize_source(df, "BMKG")

        if df.empty:
            raise ValueError(f"BMKG {port}: data kosong.")

        # BMKG menjadi MASTER TIMESTAMP untuk dua API lainnya.
        timestamp_map[port] = timestamp_strings(df)

        add_to_totals(totals, load_bmkg(df, errors))

        print(
            f"   Timestamp BMKG {port}: {len(timestamp_map[port])} "
            f"timestamp dikirim melalui XCom."
        )

    print_etl_results("HASIL ETL BMKG", totals, errors)

    # Error penulisan database TIDAK menghentikan task ini, supaya task
    # Open-Meteo & Copernicus tetap berjalan untuk target yang sehat.
    # Error dilaporkan oleh task akhir `cek_hasil_load`.
    context["ti"].xcom_push(key="load_errors", value=errors)

    return timestamp_map

# ============================================================
# 10. TASK AIRFLOW: ETL OPEN-METEO
# ============================================================

def etl_openmeteo(**context):
    print("\n" + "=" * 90)
    print("ETL OPEN-METEO - MENGIKUTI TIMESTAMP BMKG")
    print("=" * 90)

    errors = {}
    prepare_database(errors)

    ti = context["ti"]
    timestamp_map = ti.xcom_pull(task_ids="etl_bmkg")

    if not timestamp_map:
        raise ValueError("Timestamp BMKG dari XCom tidak ditemukan.")

    totals = {}

    for port in PORTS:
        target_timestamps = timestamp_map.get(port)

        if not target_timestamps:
            raise ValueError(
                f"Timestamp BMKG untuk {port} tidak ditemukan."
            )

        print(
            f"\n--- Open-Meteo: {port} | "
            f"{len(target_timestamps)} timestamp BMKG ---"
        )

        df = get_openmeteo(port, target_timestamps)

        # Wajib sama persis dengan timestamp BMKG untuk port tersebut.
        if len(df) != len(target_timestamps):
            raise ValueError(
                f"Open-Meteo {port}: jumlah data {len(df)} tidak sama "
                f"dengan timestamp BMKG {len(target_timestamps)}."
            )

        add_to_totals(totals, load_openmeteo(df, errors))

    print_etl_results("HASIL ETL OPEN-METEO", totals, errors)
    ti.xcom_push(key="load_errors", value=errors)

# ============================================================
# 11. TASK AIRFLOW: ETL COPERNICUS
# ============================================================

def etl_copernicus(**context):
    print("\n" + "=" * 90)
    print("ETL COPERNICUS MARINE - MENGIKUTI TIMESTAMP BMKG")
    print("=" * 90)

    errors = {}
    prepare_database(errors)

    ti = context["ti"]
    timestamp_map = ti.xcom_pull(task_ids="etl_bmkg")

    if not timestamp_map:
        raise ValueError("Timestamp BMKG dari XCom tidak ditemukan.")

    totals = {}

    for port in PORTS:
        target_timestamps = timestamp_map.get(port)

        if not target_timestamps:
            raise ValueError(
                f"Timestamp BMKG untuk {port} tidak ditemukan."
            )

        print(
            f"\n--- Copernicus: {port} | "
            f"{len(target_timestamps)} timestamp BMKG ---"
        )

        df = get_copernicus(port, target_timestamps)

        if len(df) != len(target_timestamps):
            raise ValueError(
                f"Copernicus {port}: jumlah data {len(df)} tidak sama "
                f"dengan timestamp BMKG {len(target_timestamps)}."
            )

        add_to_totals(totals, load_copernicus(df, errors))

    print_etl_results("HASIL ETL COPERNICUS", totals, errors)
    ti.xcom_push(key="load_errors", value=errors)

# ============================================================
# 12. TASK AKHIR: LAPORKAN ERROR PENULISAN DATABASE
# ============================================================

def cek_hasil_load(**context):
    """Gagalkan DAG run jika ada target database yang gagal ditulis.

    Dijalankan setelah ketiga task selesai (trigger_rule=all_done), jadi
    target yang sehat sudah terisi lebih dulu. Retry task berikutnya aman
    karena semua penulisan bersifat upsert (idempotent).
    """
    ti = context["ti"]
    all_errors = {}
    for task_id in ("etl_bmkg", "etl_openmeteo", "etl_copernicus"):
        errs = ti.xcom_pull(task_ids=task_id, key="load_errors") or {}
        for db_name, msg in errs.items():
            all_errors[f"{task_id}/{db_name}"] = msg

    if all_errors:
        raise_if_errors(all_errors, "cek_hasil_load")

    print("Semua target database berhasil ditulis.")

# ============================================================
# 13. DAG AIRFLOW
# ============================================================

default_args = {
    "owner": "prediksi-gelombang",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=45),
}

with DAG(
    dag_id="etl_3api_prediksi_gelombang",
    default_args=default_args,
    description=(
        "ETL BMKG Day 1 + Day 2-4, Open-Meteo, dan Copernicus "
        "dengan BMKG sebagai master timestamp"
    ),
    start_date=datetime(2026, 10, 1),
    schedule="0 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["prediksi-gelombang", "3-api", "BMKG", "Open-Meteo", "Copernicus"],
) as dag:

    task_bmkg = PythonOperator(
        task_id="etl_bmkg",
        python_callable=etl_bmkg,
    )

    task_openmeteo = PythonOperator(
        task_id="etl_openmeteo",
        python_callable=etl_openmeteo,
    )

    task_copernicus = PythonOperator(
        task_id="etl_copernicus",
        python_callable=etl_copernicus,
    )

    task_cek = PythonOperator(
        task_id="cek_hasil_load",
        python_callable=cek_hasil_load,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # BMKG dijalankan lebih dulu karena timestamp BMKG menjadi
    # acuan timestamp Open-Meteo dan Copernicus.
    task_bmkg >> [task_openmeteo, task_copernicus] >> task_cek
