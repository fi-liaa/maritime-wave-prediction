# Prediksi Tinggi Gelombang untuk Pelayaran Menggunakan Machine Learning

## Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi tinggi gelombang laut berdasarkan data meteorologi dan oseanografi pada tujuh pelabuhan di Sulawesi Tenggara. Proyek mengintegrasikan data dari tiga sumber, yaitu **BMKG, Open-Meteo, dan Copernicus Marine**, untuk mendukung analisis kondisi perairan dan pengembangan model prediksi tinggi gelombang.

Pengambilan, pemrosesan, dan penyimpanan data dilakukan melalui pipeline **ETL (Extract, Transform, Load)** menggunakan Apache Airflow. Data yang telah dikumpulkan disimpan dalam database SQL, kemudian digunakan untuk analisis, pemodelan machine learning, dan visualisasi melalui dashboard interaktif berbasis Streamlit.

## Tujuan Proyek

1. Mengintegrasikan data meteorologi dan oseanografi dari tiga sumber data yang berbeda.
2. Mengotomatisasi proses pengambilan dan pembaruan data menggunakan Apache Airflow.
3. Menyimpan data dalam database SQL dengan struktur tabel yang terorganisasi.
4. Membangun model machine learning untuk memprediksi tinggi gelombang laut.
5. Menyajikan informasi dan hasil prediksi melalui dashboard interaktif.
6. Mendukung analisis kondisi perairan sebagai informasi tambahan dalam perencanaan pelayaran.

## Sumber Data

Proyek ini menggunakan tiga sumber data utama.

| Sumber data | Informasi yang digunakan |
|---|---|
| BMKG | Suhu rata-rata, kelembapan, kecepatan angin, hembusan angin, tinggi gelombang, kategori gelombang, dan kecepatan arus |
| Open-Meteo | Suhu udara, kelembapan relatif, tekanan permukaan, curah hujan, kecepatan angin, dan arah angin |
| Copernicus Marine | Komponen arus laut `uo` dan `vo`, serta kecepatan arus hasil perhitungan |

BMKG digunakan sebagai acuan waktu (*master timestamp*) untuk menyelaraskan data Open-Meteo dan Copernicus Marine berdasarkan kode pelabuhan dan waktu pengamatan.

## Wilayah Penelitian

Data dikumpulkan untuk tujuh pelabuhan di Sulawesi Tenggara:

1. Pelabuhan Nusantara Kendari
2. Pelabuhan Kolaka
3. Pelabuhan Raha
4. Pelabuhan Sikeli
5. Pelabuhan Murhum Baubau
6. Pelabuhan Ereke-Waode Buri
7. Pelabuhan Wanci

## Arsitektur Sistem

Alur kerja proyek secara umum adalah sebagai berikut:

1. **Extract** — Mengambil data dari API BMKG dan Open-Meteo serta layanan Copernicus Marine.
2. **Transform** — Menyesuaikan format data, membersihkan nilai numerik, menormalisasi timestamp ke UTC, dan memeriksa duplikasi berdasarkan kode pelabuhan dan timestamp.
3. **Load** — Menyimpan data ke tabel-tabel database SQL menggunakan mekanisme *upsert* untuk memperbarui data yang sudah tersedia tanpa menambahkan duplikasi berdasarkan kunci unik.
4. **Integrasi data** — Menggabungkan data dari ketiga sumber berdasarkan kode pelabuhan dan timestamp.
5. **Pemodelan** — Menggunakan data terintegrasi untuk analisis dan prediksi tinggi gelombang.
6. **Visualisasi** — Menampilkan informasi dan hasil prediksi melalui dashboard Streamlit.

Pipeline ETL dijadwalkan berjalan setiap jam menggunakan Apache Airflow.

## Teknologi yang Digunakan

- **Python** — Bahasa pemrograman utama.
- **Apache Airflow** — Orkestrasi dan penjadwalan pipeline ETL.
- **Pandas dan NumPy** — Pengolahan serta pembersihan data.
- **Requests** — Pengambilan data melalui HTTP API.
- **Xarray dan Copernicus Marine Toolbox** — Pengolahan data oseanografi dari Copernicus Marine.
- **SQLAlchemy dan MySQL Connector** — Koneksi dan pengelolaan data pada database SQL.
- **MySQL / TiDB Cloud** — Penyimpanan data.
- **Scikit-learn** — Pengembangan model machine learning.
- **Streamlit** — Pembuatan dashboard interaktif.
- **Plotly** — Visualisasi data.

## Struktur Repository

```text
maritime-wave-prediction/
├── dashboard/
│   ├── app_streamlit.py
│   ├── app.py
│   ├── app_old.py
│   ├── requirements.txt
│   ├── .env.example
│   └── model/
│       ├── model_wave_height_production.pkl
│       └── model_metadata.json
├── etl_3api_prediksi_gelombang.py
├── Notebook_Prediksi_Tinggi_Gelombang.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── Screenshot airflow.png
```

*Catatan: Struktur di atas menggambarkan susunan file utama proyek. Sesuaikan dengan isi repository terbaru apabila terdapat file yang berbeda.*

## Database

Data disimpan dalam database `maritime_wave_db`, yang memiliki beberapa tabel utama:

| Nama tabel | Fungsi |
|---|---|
| `ports` | Menyimpan informasi dan koordinat pelabuhan |
| `bmkg_maritime` | Menyimpan data meteorologi dan kondisi laut dari BMKG |
| `openmeteo_weather` | Menyimpan data cuaca dari Open-Meteo |
| `copernicus_marine` | Menyimpan data arus laut dari Copernicus Marine |

Ketiga tabel sumber menggunakan `port_code` dan `timestamp_utc` sebagai acuan pencocokan data. Kombinasi kedua kolom tersebut juga digunakan sebagai kunci unik untuk mencegah duplikasi catatan pada setiap tabel sumber.

## Pipeline ETL Apache Airflow

Pipeline ETL didefinisikan dalam file `etl_3api_prediksi_gelombang.py` dengan DAG ID `etl_3api_prediksi_gelombang`.

DAG terdiri atas empat task utama:

- `etl_bmkg` — Mengambil dan menyimpan data BMKG serta menghasilkan acuan timestamp.
- `etl_openmeteo` — Mengambil data Open-Meteo berdasarkan timestamp BMKG.
- `etl_copernicus` — Mengambil data arus laut Copernicus berdasarkan timestamp BMKG.
- `cek_hasil_load` — Memeriksa hasil proses penyimpanan data pada target database.

Jadwal eksekusi diatur menggunakan cron `0 * * * *`, yang berarti pipeline dijadwalkan berjalan setiap jam.

## Model Machine Learning

Proyek ini berfokus pada prediksi tinggi gelombang laut dengan variabel target `wave_height`.

Model menggunakan data meteorologi dan oseanografi sebagai fitur untuk memperkirakan tinggi gelombang. Notebook digunakan untuk eksplorasi data, persiapan fitur, pelatihan model, serta evaluasi hasil sesuai eksperimen yang dilakukan.

Hasil evaluasi model, seperti MAE, RMSE, dan \(R^2\), perlu dicantumkan berdasarkan hasil eksperimen yang benar-benar diperoleh.

**Catatan:** Prediksi berdasarkan data pada waktu yang sama menggambarkan estimasi tinggi gelombang berdasarkan kondisi yang tersedia. Untuk memprediksi tinggi gelombang pada waktu mendatang, target model perlu disusun berdasarkan timestamp masa depan dan dievaluasi menggunakan pemisahan data berbasis waktu.

## Dashboard

Dashboard dibuat menggunakan Streamlit untuk menyajikan data dan informasi prediksi tinggi gelombang secara interaktif.

Aplikasi dapat diakses melalui tautan berikut:

**[Dashboard Prediksi Tinggi Gelombang](https://prediksi-tinggi-gelombang.streamlit.app/)**

Ketersediaan fitur dashboard mengikuti implementasi aplikasi yang sedang digunakan.

## Konfigurasi dan Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/fi-liaa/maritime-wave-prediction.git
cd maritime-wave-prediction
```

### 2. Siapkan lingkungan Python

Buat dan aktifkan virtual environment sesuai sistem operasi, kemudian instal dependensi dari file `requirements.txt` yang sesuai dengan komponen yang ingin dijalankan.

```bash
python -m venv .venv
```

Pada Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instal dependensi:

```bash
pip install -r requirements.txt
```

### 3. Konfigurasikan koneksi database

Siapkan database `maritime_wave_db` beserta tabel-tabel yang diperlukan.

Konfigurasi koneksi database dilakukan melalui environment variable atau Streamlit Secrets sesuai lingkungan eksekusi. Untuk pipeline Airflow, variabel yang digunakan meliputi:

- `LOCAL_DATABASE_URL`
- `TIDB_HOST`
- `TIDB_PORT`
- `TIDB_USER`
- `TIDB_PASSWORD`
- `TIDB_DATABASE`

Untuk Copernicus Marine, konfigurasikan kredensial berikut:

- `COPERNICUSMARINE_SERVICE_USERNAME`
- `COPERNICUSMARINE_SERVICE_PASSWORD`

Gunakan kredensial milik sendiri. Jangan mengunggah kata sandi, token, atau file `.env` yang berisi informasi rahasia ke repository publik.

### 4. Jalankan pipeline ETL

Pastikan Apache Airflow sudah terpasang dan berjalan, dependensi DAG tersedia, serta koneksi database dan kredensial sumber data telah dikonfigurasi.

Tempatkan file DAG pada direktori DAG yang dipantau Airflow, lalu aktifkan DAG `etl_3api_prediksi_gelombang` melalui antarmuka Airflow.

Periksa status setiap task dan log eksekusi untuk memastikan pengambilan serta penyimpanan data berhasil.

### 5. Jalankan dashboard secara lokal

Jika dependensi dashboard sudah terpasang, jalankan:

```bash
streamlit run dashboard/app_streamlit.py
```

Pastikan konfigurasi database dan lokasi file model sesuai dengan lingkungan yang digunakan.

## Batasan Proyek

- Data yang digunakan bergantung pada ketersediaan dan cakupan layanan BMKG, Open-Meteo, serta Copernicus Marine.
- Ketersediaan data pada setiap timestamp perlu divalidasi sebelum proses integrasi.
- Hasil prediksi merupakan estimasi model dan bukan pengganti informasi keselamatan atau peringatan resmi dari BMKG.
- Akurasi model harus dinilai berdasarkan hasil evaluasi pada data pengujian yang tidak digunakan saat pelatihan.

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan meliputi:

- Menambahkan validasi kualitas dan kelengkapan data secara otomatis.
- Mengembangkan prediksi tinggi gelombang untuk beberapa jam ke depan.
- Membandingkan performa beberapa algoritma machine learning.
- Menambahkan pemantauan performa model dan pembaruan model secara berkala.
- Meningkatkan visualisasi tren tinggi gelombang dan kondisi meteorologi maupun oseanografi.

## Penulis

Proyek ini dikembangkan sebagai bagian dari tugas proyek Data Science, dengan fokus pada integrasi data, otomatisasi ETL, pemodelan machine learning, dan visualisasi prediksi tinggi gelombang laut di Sulawesi Tenggara.
