import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Folder tempat app.py ini berada, dipakai sebagai acuan lokasi file model
# supaya tetap ketemu meski working directory server berbeda
BASE_DIR = Path(__file__).resolve().parent

# ============================================
# Load model, encoder, dan daftar fitur
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / 'model_rf.pkl')
    encoders = joblib.load(BASE_DIR / 'encoders.pkl')
    feature_cols = joblib.load(BASE_DIR / 'feature_cols.pkl')
    return model, encoders, feature_cols

model, encoders, feature_cols = load_artifacts()

# ============================================
# Nilai default untuk fitur yang pengaruhnya kecil
# (diambil dari nilai tengah/paling umum di data asli)
# ============================================
DEFAULT_ITEM_WEIGHT = 12.6
DEFAULT_ITEM_TYPE = "Fruits and Vegetables"
DEFAULT_OUTLET_IDENTIFIER = "OUT027"
DEFAULT_OUTLET_ESTABLISHMENT_YEAR = 1999
DEFAULT_ITEM_FAT_CONTENT = "Low Fat"
DEFAULT_OUTLET_SIZE = "Medium"
DEFAULT_OUTLET_LOCATION = "Tier 3"

# ============================================
# Judul & deskripsi aplikasi
# ============================================
st.set_page_config(page_title="Prediksi Penjualan Produk Retail")
st.title("Prediksi Penjualan Produk Retail")
st.write(
    "Aplikasi ini membantu memperkirakan potensi penjualan sebuah produk di sebuah toko. "
    "Cukup isi tiga informasi di bawah ini, lalu tekan tombol untuk melihat hasilnya."
)

st.divider()

# ============================================
# Form input HANYA untuk fitur yang paling berpengaruh
# (berdasarkan hasil feature importance model)
# ============================================
st.subheader("Informasi Produk dan Toko")

item_mrp = st.number_input(
    "Harga jual produk",
    min_value=0.0, max_value=300.0, value=140.0, step=1.0,
    help="Faktor paling berpengaruh terhadap hasil prediksi."
)

outlet_type = st.selectbox(
    "Jenis toko",
    options=list(encoders['Outlet_Type'].classes_),
    help="Faktor kedua paling berpengaruh terhadap hasil prediksi."
)

item_visibility = st.slider(
    "Seberapa mencolok produk dipajang di rak toko",
    0.0, 0.35, 0.05, step=0.01,
    help="0 berarti tersembunyi, 0.35 berarti sangat mencolok. Faktor ketiga paling berpengaruh terhadap hasil prediksi."
)

st.divider()

# ============================================
# Tombol prediksi
# ============================================
if st.button("Lihat Estimasi Penjualan", type="primary", use_container_width=True):
    outlet_age = 2013 - DEFAULT_OUTLET_ESTABLISHMENT_YEAR

    input_data = pd.DataFrame([{
        'Item_Weight': DEFAULT_ITEM_WEIGHT,
        'Item_Visibility': item_visibility,
        'Item_MRP': item_mrp,
        'Outlet_Age': outlet_age,
        'Item_Fat_Content_enc': encoders['Item_Fat_Content'].transform([DEFAULT_ITEM_FAT_CONTENT])[0],
        'Item_Type_enc': encoders['Item_Type'].transform([DEFAULT_ITEM_TYPE])[0],
        'Outlet_Identifier_enc': encoders['Outlet_Identifier'].transform([DEFAULT_OUTLET_IDENTIFIER])[0],
        'Outlet_Size_enc': encoders['Outlet_Size'].transform([DEFAULT_OUTLET_SIZE])[0],
        'Outlet_Location_Type_enc': encoders['Outlet_Location_Type'].transform([DEFAULT_OUTLET_LOCATION])[0],
        'Outlet_Type_enc': encoders['Outlet_Type'].transform([outlet_type])[0],
    }])[feature_cols]

    prediction = model.predict(input_data)[0]

    st.success(f"Estimasi nilai penjualan: {prediction:,.2f}")
    st.caption(
        "Catatan: angka ini adalah perkiraan dari sistem berdasarkan pola data yang ada, "
        "bukan angka pasti. Faktor selain yang diisi di atas menggunakan nilai umum sebagai asumsi."
    )

st.divider()
st.caption("Aplikasi ini dibuat sebagai bagian dari proyek magang - Prediksi Penjualan Retail")
