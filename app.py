import streamlit as st
from PIL import Image
import numpy as np
import time
import os
import h5py

# Konfigurasi Halaman
st.set_page_config(
    page_title="Klasifikasi Kematangan Pisang",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Tema Kuning Pastel, Font Nunito, Clean UI
css_kustom = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Nunito', sans-serif !important;
        background-color: #FFF9C4 !important;
    }
    
    .stApp > header {
        background-color: transparent !important;
    }

    h1, h2, h3, h4, p, label, li {
        color: #1E293B !important;
    }
    
    /* Sembunyikan ikon rantai (anchor link) pada setiap judul */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
        display: none !important;
    }

    /* Mengunci Sidebar agar tidak bisa disembunyikan (hapus tombol panah/X) */
    [data-testid="collapsedControl"], 
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* TABS NAVBAR STYLING - CENTERED */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 30px;
        border-bottom: 2px solid #FDE047;
        padding-bottom: 10px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding: 10px 15px;
        color: #64748B;
        font-weight: 700;
        font-size: 1.1rem;
        border: none !important;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1E293B !important;
        border-bottom: 3px solid #1E293B !important;
        background-color: transparent !important;
    }

    /* Card styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #FDE047;
        border-radius: 12px;
        padding: 32px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .info-card {
        background-color: #FEFCE8;
        border: 1px solid #FDE047;
        border-left: 4px solid #EAB308;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
    }

    /* Status Badges */
    .status-mentah {
        color: #059669;
        background-color: #ECFDF5;
        padding: 6px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #34D399;
    }
    
    .status-matang {
        color: #D97706;
        background-color: #FFFBEB;
        padding: 6px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FBBF24;
    }

    .status-terlalu {
        color: #DC2626;
        background-color: #FEF2F2;
        padding: 6px 20px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    
    hr {
        border-color: #FDE047;
        margin: 2rem 0;
    }
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)

# Konstanta
UKURAN_INPUT = (224, 224)
CLASS_NAMES = ['matang', 'mentah', 'terlalu_matang']

# --- FUNGSI PEMBUATAN ARSITEKTUR MODEL (BACKEND PERSIS SAMA) ---
def inject_dense_weights(model, h5_path, expected_in):
    kernel_1 = None
    bias_1 = None
    kernel_2 = None
    bias_2 = None
    
    with h5py.File(h5_path, 'r') as f:
        g = f['model_weights'] if 'model_weights' in f else f
        
        def search_group(group):
            nonlocal kernel_1, bias_1, kernel_2, bias_2
            for k in group.keys():
                item = group[k]
                if isinstance(item, h5py.Group):
                    search_group(item)
                elif isinstance(item, h5py.Dataset):
                    if item.shape == (expected_in, 128):
                        kernel_1 = item[:]
                        for sub_k in group.keys():
                            if group[sub_k].shape == (128,):
                                bias_1 = group[sub_k][:]
                                break
                    elif item.shape == (128, 3):
                        kernel_2 = item[:]
                        for sub_k in group.keys():
                            if group[sub_k].shape == (3,):
                                bias_2 = group[sub_k][:]
                                break
        
        search_group(g)
        
    if kernel_1 is not None and bias_1 is not None:
        model.layers[-3].set_weights([kernel_1, bias_1])
    if kernel_2 is not None and bias_2 is not None:
        model.layers[-1].set_weights([kernel_2, bias_2])

def get_augmentation_layer():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1)
    ], name="Augmentasi_On_The_Fly")

@st.cache_resource
def load_mobilenet_v3():
    import tensorflow as tf
    try:
        data_augmentation = get_augmentation_layer()
        base_model = tf.keras.applications.MobileNetV3Large(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="MobileNetV3_Large")
        inject_dense_weights(model, "models/best_mobilenet_pisang.h5", 960)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_convnext_tiny():
    import tensorflow as tf
    try:
        data_augmentation = get_augmentation_layer()
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = tf.keras.applications.convnext.preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs, name="ConvNeXt_Tiny")
        inject_dense_weights(model, "models/best_convnext_pisang.h5", 768)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image):
    import tensorflow as tf
    import numpy as np
    img = image.resize(UKURAN_INPUT)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    return img_array

def format_hasil(class_name, confidence):
    label = class_name.replace('_', ' ').title()
    if class_name == 'mentah':
        return label, confidence, 'status-mentah'
    elif class_name == 'matang':
        return label, confidence, 'status-matang'
    else:
        return label, confidence, 'status-terlalu'

def tampilkan_gambar_visualisasi(nama_file, deskripsi):
    path_visual = f"Visualisasi/{nama_file}"
    if os.path.exists(path_visual):
        gambar = Image.open(path_visual)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(gambar, caption=deskripsi, use_column_width=True)
    else:
        st.info(f"Visualisasi {nama_file} belum tersedia.")

# --- UI LAYOUT ---

# HEADER
st.markdown("<h1 style='text-align: center; margin-top: 1rem; color: #1E293B;'>Kematangan Buah Pisang Lokal Indonesia</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- PREDIKSI UTAMA ---
if True:
    st.markdown("<br>", unsafe_allow_html=True)
    
    kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

    with kolom_tengah:
        berkas_unggah = st.file_uploader("Pilih file citra (Resolusi disarankan: > 500x500px)", type=['jpg', 'jpeg', 'png'])
        
        if berkas_unggah is not None:
            # Pindahkan pemuatan model ke sini (Lazy Loading) agar server cepat menyala
            with st.spinner("Memuat model kecerdasan buatan untuk pertama kali (membutuhkan beberapa detik)..."):
                model_mob, err_mob = load_mobilenet_v3()
                model_conv, err_conv = load_convnext_tiny()
                
            if model_mob is None or model_conv is None:
                st.warning("Peringatan: File bobot model belum terpasang dengan benar pada peladen (server).")
            else:
                citra_asli = Image.open(berkas_unggah).convert('RGB')
                st.image(citra_asli, caption="Citra Sampel Terunggah", use_column_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Auto-inference (No button)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; margin-bottom: 2rem; color: #1E293B;'>Laporan Hasil Inspeksi Komparatif</h3>", unsafe_allow_html=True)
                
                with st.spinner("Mengeksekusi jaringan saraf tiruan secara paralel..."):
                    img_tensor = preprocess_image(citra_asli)
                    
                    # ConvNeXt
                    start_conv = time.time()
                    pred_c = model_conv.predict(img_tensor, verbose=0)
                    waktu_conv = time.time() - start_conv
                    class_c = CLASS_NAMES[np.argmax(pred_c[0])]
                    conf_c = np.max(pred_c[0]) * 100
                    
                    # MobileNet
                    start_mob = time.time()
                    pred_m = model_mob.predict(img_tensor, verbose=0)
                    waktu_mob = time.time() - start_mob
                    class_m = CLASS_NAMES[np.argmax(pred_m[0])]
                    conf_m = np.max(pred_m[0]) * 100
                    
                    kelas_conv, conf_conv, style_conv = format_hasil(class_c, conf_c)
                    kelas_mob, conf_mob, style_mob = format_hasil(class_m, conf_m)

                    col_conv, col_mob = st.columns(2)
                    
                    with col_conv:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #374151; font-size: 1.2rem; margin-bottom: 1.5rem;'>Arsitektur ConvNeXt-Tiny</h3>
                            <div style='margin-bottom: 2rem;'>
                                <span class='{style_conv}'>{kelas_conv}</span>
                            </div>
                            <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Tingkat Kepercayaan (Confidence)</p>
                            <h2 style='margin: 0 0 1.5rem 0; color: #1E293B;'>{conf_conv:.2f}%</h2>
                            <div style='background-color: #FEFCE8; padding: 10px; border-radius: 6px; display: inline-block;'>
                                <span style='font-size: 0.85rem; color: #4B5563;'>Waktu Inferensi: <b>{waktu_conv:.3f} detik</b></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_mob:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #374151; font-size: 1.2rem; margin-bottom: 1.5rem;'>Arsitektur MobileNetV3-Large</h3>
                            <div style='margin-bottom: 2rem;'>
                                <span class='{style_mob}'>{kelas_mob}</span>
                            </div>
                            <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Tingkat Kepercayaan (Confidence)</p>
                            <h2 style='margin: 0 0 1.5rem 0; color: #1E293B;'>{conf_mob:.2f}%</h2>
                            <div style='background-color: #FEFCE8; padding: 10px; border-radius: 6px; display: inline-block;'>
                                <span style='font-size: 0.85rem; color: #4B5563;'>Waktu Inferensi: <b>{waktu_mob:.3f} detik</b></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

