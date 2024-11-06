import os
from io import BytesIO

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# Fungsi untuk menampilkan gambar dengan judul
def tampilkan_judul(citra, judul):
    st.image(citra, caption=judul, use_column_width=True)

# Fungsi untuk membuat dan menampilkan histogram sebagai bar plot
def tampilkan_histogram(citra):
    fig, ax = plt.subplots()
    if len(citra.shape) == 3:  # Histogram untuk gambar berwarna
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            # Mengabaikan piksel hitam (nilai 0)
            non_zero_pixels = citra[:, :, i][citra[:, :, i] > 0]
            hist = np.histogram(non_zero_pixels, bins=256, range=(0, 256))[0]
            ax.bar(np.arange(256), hist, color=col, alpha=0.5, width=1.0)
        ax.set_title('Histogram (RGB) - Tanpa Padding Hitam')
    else:  # Histogram untuk gambar grayscale
        # Mengabaikan piksel hitam (nilai 0)
        non_zero_pixels = citra.flatten()[citra.flatten() > 0]
        hist, _ = np.histogram(non_zero_pixels, bins=256, range=(0, 256))
        ax.bar(np.arange(256), hist, color='black', alpha=0.7, width=1.0)
        ax.set_title('Histogram (Grayscale) - Tanpa Padding Hitam')
    ax.set_xlim([0, 256])
    st.pyplot(fig)

# Fungsi untuk mengkonversi array numpy menjadi bytes
def convert_image_to_bytes(image_array):
    img = Image.fromarray(image_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im

# Judul Aplikasi
st.title("Pengolahan Citra Kelompok Esigma")

# Input Upload Gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membaca gambar dengan Pillow
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    # Menampilkan gambar dan histogram asli
    st.subheader("Gambar Asli dan Histogram")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(img_np, "Gambar Asli")
    with col2:
        tampilkan_histogram(img_np)

    # Sidebar untuk memilih mode pemrosesan gambar
    st.sidebar.subheader("Pilih Mode Pengolahan Citra")
    opsi = st.sidebar.selectbox("Mode Pengolahan", (
        "Normal","Citra Negatif", "Grayscale", "Rotasi&Flip", 
        "Histogram Equalization", "Black & White", "Smoothing (Gaussian Blur)", "Channel RGB"
    ))

    # Input untuk threshold jika opsi "Black & White" dipilih
    if opsi == "Black & White":
        threshold = st.sidebar.number_input("Threshold Level", min_value=0, max_value=255, value=127)

    # Button untuk memilih derajat rotasi jika opsi "Rotasi" dipilih
    if opsi == "Rotasi&Flip":
        rotasi = st.sidebar.radio("Pilih Derajat Rotasi", ("Custom", "Flip Horizontal", "Flip Vertikal"), index=0)

        custom_angle_input = st.sidebar.text_input("Masukkan derajat rotasi (0-360)", value="0")
    
        # Validasi input dari user, memastikan bahwa nilai yang dimasukkan adalah angka dan berada dalam rentang 0-360
        try:
            custom_angle = float(custom_angle_input)
            if not (0 <= custom_angle <= 360):
                st.sidebar.error("Masukkan nilai derajat antara 0 hingga 360.")
                custom_angle = 0  # Nilai default jika input di luar batas
        except ValueError:
            st.sidebar.error("Masukkan nilai numerik yang valid.")
            custom_angle = 0  # Nilai default jika input tidak valid

    # Field input untuk blur radius jika opsi "Smoothing (Gaussian Blur)" dipilih
    if opsi == "Smoothing (Gaussian Blur)":
        blur_radius = st.sidebar.text_input("Masukkan Blur Radius", value="10")
        try:
            blur_radius = float(blur_radius)
        except ValueError:
            st.sidebar.error("Masukkan nilai numerik yang valid untuk blur radius.")
            blur_radius = 10  # Default value jika input salah

    # Pilihan channel jika opsi "Channel RGB" dipilih
    if opsi == "Channel RGB":
        channel = st.sidebar.selectbox("Pilih Channel", ("Red", "Green", "Blue"))

    # Fungsi untuk mengolah gambar berdasarkan opsi
    def olah_gambar(img_np, opsi):
        if opsi == "Normal":
            return np.array(img_np)
        elif opsi == "Citra Negatif":
            return np.clip(255 - img_np.astype(np.uint8), 0, 255)
        elif opsi == "Grayscale":
            return np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))
        elif opsi == "Rotasi&Flip":
            if rotasi == "Flip Horizontal":
                return np.fliplr(img_np)
            elif rotasi == "Flip Vertikal":
                return np.flipud(img_np)
            else:  # Untuk rotasi kustom berdasarkan slider
                return np.array(Image.fromarray(img_np.astype(np.uint8)).rotate(custom_angle, expand=True))
            
        elif opsi == "Histogram Equalization":
            img_rgb = Image.fromarray(img_np.astype(np.uint8))
            r, g, b = img_rgb.split()
            r_eq = ImageOps.equalize(r)
            g_eq = ImageOps.equalize(g)
            b_eq = ImageOps.equalize(b)
            img_eq = Image.merge("RGB", (r_eq, g_eq, b_eq))
            return np.array(img_eq)

        elif opsi == "Black & White":
            gray = np.array(ImageOps.grayscale(Image.fromarray(img_np.astype(np.uint8))))
            bw = np.where(gray > threshold, 255, 0).astype(np.uint8)
            return bw
        elif opsi == "Smoothing (Gaussian Blur)":
            return np.array(Image.fromarray(img_np.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        elif opsi == "Channel RGB":
            # Preserve color of the selected channel, set other channels to zero
            img_channel = np.zeros_like(img_np)
            channel_map = {"Red": 0, "Green": 1, "Blue": 2}
            img_channel[:, :, channel_map[channel]] = img_np[:, :, channel_map[channel]]
            return img_channel

    # Pemrosesan gambar berdasarkan opsi
    hasil = olah_gambar(img_np, opsi)

    # Terapkan pengaturan ke gambar HASIL (bukan img_np)
    img_pil = Image.fromarray(hasil.astype(np.uint8))  # Ubah ini dari img_np ke hasil

    default_value = 1.0

    # Tambahkan state untuk track kapan reset ditekan
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    # Pindahkan button Reset ke ATAS sebelum slider
    if st.sidebar.button("Reset", key="reset_adjustment"):
        st.session_state.reset_counter += 1
        st.sidebar.write("Adjustment telah direset ke nilai default.")

    # Buat slider dengan key yang dinamis
    brightness = st.sidebar.slider("Brightness", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_value, 
        step=0.1,
        key=f"brightness_{st.session_state.reset_counter}")

    contrast = st.sidebar.slider("Contrast", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_value, 
        step=0.1,
        key=f"contrast_{st.session_state.reset_counter}")

    highlight = st.sidebar.slider("Highlight", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_value, 
        step=0.1,
        key=f"highlight_{st.session_state.reset_counter}")

    shadow = st.sidebar.slider("Shadow", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_value, 
        step=0.1,
        key=f"shadow_{st.session_state.reset_counter}")

    whites = st.sidebar.slider("Whites", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_value, 
        step=0.1,
        key=f"whites_{st.session_state.reset_counter}")

    blacks = st.sidebar.slider("Blacks", 
        min_value=0.0, 
        max_value=2.0, 
        value=default_value, 
        step=0.1,
        key=f"blacks_{st.session_state.reset_counter}")

    # Terapkan pengaturan ke gambar HASIL
    img_pil = Image.fromarray(hasil.astype(np.uint8))  # Menggunakan hasil pengolahan

    # Adjustment untuk Brightness dan Contrast
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(contrast)

    # Adjustment untuk Highlights dan Shadows
    img_pil = ImageEnhance.Brightness(img_pil).enhance(highlight)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(shadow)

    # Adjustment Whites dan Blacks
    img_pil = ImageEnhance.Brightness(img_pil).enhance(whites)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(blacks)

    img_np_adjusted = np.array(img_pil)

    # Menampilkan gambar yang telah disesuaikan
    st.subheader("Hasil Pengolahan Citra")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(img_np_adjusted, f"Hasil Pengolahan: {opsi}")
    with col2:
        tampilkan_histogram(img_np_adjusted)

    # Tombol untuk mengunduh gambar hasil
    st.download_button(
        label="Unduh Gambar",
        data=convert_image_to_bytes(img_np_adjusted),
        file_name="hasil_pengolahan.png",
        mime="image/png",
        key="download_image"
    )
