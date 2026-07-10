import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load('model_rf.pkl')
    encoders = joblib.load('encoders.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, encoders, feature_cols

model, encoders, feature_cols = load_artifacts()

st.set_page_config(page_title="Prediksi Penjualan Produk Retail")
st.title("Prediksi Penjualan Produk Retail")
st.write(
    "Aplikasi ini membantu memperkirakan potensi penjualan sebuah produk di sebuah toko. "
    "Isi informasi produk dan toko di bawah ini, lalu tekan tombol untuk melihat hasilnya."
)

st.divider()
st.subheader("Informasi Produk")

item_type = st.selectbox("Jenis produk", options=list(encoders['Item_Type'].classes_))
item_mrp = st.number_input("Harga jual produk (dalam satuan mata uang dataset asli Rupee India)", min_value=0.0, max_value=300.0, value=140.0, step=1.0)
item_fat_content = st.selectbox("Kadar lemak produk", options=list(encoders['Item_Fat_Content'].classes_))

st.subheader("Informasi Toko")

outlet_type = st.selectbox("Jenis toko", options=list(encoders['Outlet_Type'].classes_))
outlet_size = st.selectbox("Ukuran toko", options=list(encoders['Outlet_Size'].classes_))
outlet_location = st.selectbox("Lokasi toko", options=list(encoders['Outlet_Location_Type'].classes_))

with st.expander("Informasi tambahan (boleh dilewati)"):
    st.caption("Bagian ini tidak wajib diisi. Sistem sudah menyiapkan nilai yang wajar secara otomatis.")
    item_weight = st.number_input("Berat produk (kg)", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
    item_visibility = st.slider("Seberapa mudah produk terlihat di rak toko (0 = tersembunyi, 0.35 = sangat mencolok)", 0.0, 0.35, 0.05, step=0.01)
    outlet_identifier = st.selectbox("Kode toko tertentu", options=list(encoders['Outlet_Identifier'].classes_))
    outlet_year = st.number_input("Tahun toko mulai beroperasi", min_value=1980, max_value=2013, value=2000, step=1)

st.divider()

if st.button("Lihat Estimasi Penjualan", type="primary", use_container_width=True):
    outlet_age = 2013 - outlet_year

    input_data = pd.DataFrame([{
        'Item_Weight': item_weight,
        'Item_Visibility': item_visibility,
        'Item_MRP': item_mrp,
        'Outlet_Age': outlet_age,
        'Item_Fat_Content_enc': encoders['Item_Fat_Content'].transform([item_fat_content])[0],
        'Item_Type_enc': encoders['Item_Type'].transform([item_type])[0],
        'Outlet_Identifier_enc': encoders['Outlet_Identifier'].transform([outlet_identifier])[0],
        'Outlet_Size_enc': encoders['Outlet_Size'].transform([outlet_size])[0],
        'Outlet_Location_Type_enc': encoders['Outlet_Location_Type'].transform([outlet_location])[0],
        'Outlet_Type_enc': encoders['Outlet_Type'].transform([outlet_type])[0],
    }])[feature_cols]

    prediction = model.predict(input_data)[0]

    st.success(f"Estimasi nilai penjualan: Rs {prediction:,.2f}")
    st.caption(
        "Catatan: angka ini adalah perkiraan dari sistem berdasarkan pola data yang ada, "
        "bukan angka pasti. Gunakan sebagai bahan pertimbangan tambahan, bukan satu-satunya acuan."
    )

st.divider()
st.caption("Prediksi Penjualan Retail")