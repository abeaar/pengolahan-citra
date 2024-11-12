import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage


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
        "Normal","Citra Negatif", "Grayscale", "Rotasi",
        "Histogram Equalization", "Black & White", "Smoothing (Gaussian Blur)","Noise",
        "Channel RGB" ,"Edge Detection"
    ))

    # Input untuk threshold jika opsi "Black & White" dipilih
    if opsi == "Black & White":
        threshold = st.sidebar.number_input("Threshold Level", min_value=0, max_value=255, value=127)

    # Button untuk memilih derajat rotasi jika opsi "Rotasi" dipilih
    if opsi == "Rotasi":
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
    
    # Tambahkan pilihan noise di sidebar
    if opsi == "Noise":
        noise_type = st.sidebar.selectbox("Jenis Noise", ("Gaussian", "Speckle", "Salt and Pepper"))

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

    # Pilihan metode untuk edge detection
    if opsi == "Edge Detection":
        metode_edge = st.sidebar.selectbox("Metode Edge Detection", ("Sobel", "Prewitt", "Roberts"))
    # Fungsi konvolusi manual
    # def apply_convolution(image, kernel):
    #     h, w = image.shape
    #     kh, kw = kernel.shape
    #     output = np.zeros((h - kh + 1, w - kw + 1))
        
    #     for i in range(h - kh + 1):
    #         for j in range(w - kw + 1):
    #             region = image[i:i + kh, j:j + kw]
    #             output[i, j] = np.sum(region * kernel)
        
    #     return output
    # Fungsi untuk menambahkan Gaussian, Speckle, atau Salt-and-Pepper noise pada gambar
    def add_noise(image, noise_type):
        if noise_type == "Gaussian":
            row, col, ch = image.shape
            mean = 0
            var = 0.01
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss * 255
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        elif noise_type == "Speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            noisy = image + image * gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        elif noise_type == "Salt and Pepper":
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[coords[0], coords[1], :] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords[0], coords[1], :] = 0
            return out

    # Fungsi untuk mengolah gambar berdasarkan opsi
    def olah_gambar(img_np, opsi):
        if opsi == "Normal":
            return np.array(img_np)
        elif opsi == "Citra Negatif":
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # Membuat citra negatif
            negatif_img = cv2.bitwise_not(img_cv2)
            # Kembali ke format RGB untuk ditampilkan di Streamlit
            return cv2.cvtColor(negatif_img, cv2.COLOR_BGR2RGB)
        elif opsi == "Grayscale":
            gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
            return gray.astype(np.uint8)
        elif opsi == "Rotasi":
            if rotasi == "Flip Horizontal":
                return np.fliplr(img_np)
            elif rotasi == "Flip Vertikal":
                return np.flipud(img_np)
            else:  # Untuk rotasi kustom berdasarkan slider
                return np.array(Image.fromarray(img_np.astype(np.uint8)).rotate(custom_angle, expand=True))
        elif opsi == "Histogram Equalization":
            img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
                
            # Equalize the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                
            # Convert back to RGB
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            return img_output
        elif opsi == "Black & White":
            # Konversi gambar ke grayscale dan terapkan threshold
            gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
            bw = np.where(gray > threshold, 255, 0)
            return bw.astype(np.uint8)
        elif opsi == "Smoothing (Gaussian Blur)":
            return np.array(Image.fromarray(img_np.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        elif opsi == "Noise":
            return add_noise(img_np, noise_type)
        elif opsi == "Channel RGB":
            # Konversi dari RGB ke BGR (format yang digunakan OpenCV)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # Pisahkan channel BGR
            b, g, r = cv2.split(img_bgr)
            # Buat array kosong untuk setiap channel
            zeros = np.zeros(img_bgr.shape[:2], dtype="uint8")
            # Mapping channel yang akan ditampilkan
            channel_output = {
                "Red": cv2.merge([zeros, zeros, r]),    # Hanya channel merah
                "Green": cv2.merge([zeros, g, zeros]),  # Hanya channel hijau
                "Blue": cv2.merge([b, zeros, zeros])    # Hanya channel biru
            }
            # Ambil channel yang dipilih
            result = channel_output[channel]
            # Konversi kembali ke RGB untuk ditampilkan di streamlit
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
        # elif opsi == "Edge Detection":
        #     gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale first
        #     if metode_edge == "Sobel":
        #         # Sobel kernels
        #         sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        #         sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        #         gx = apply_convolution(gray, sobel_x)
        #         gy = apply_convolution(gray, sobel_y)
        #         edge_sobel = np.hypot(gx, gy)
        #         return np.clip(edge_sobel, 0, 255).astype(np.uint8)
        #     elif metode_edge == "Prewitt":
        #         # Prewitt kernels
        #         prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        #         prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        #         gx = apply_convolution(gray, prewitt_x)
        #         gy = apply_convolution(gray, prewitt_y)
        #         edge_prewitt = np.hypot(gx, gy)
        #         return np.clip(edge_prewitt, 0, 255).astype(np.uint8)
        #     elif metode_edge == "Roberts":
        #         # Roberts kernels
        #         roberts_x = np.array([[1, 0], [0, -1]])
        #         roberts_y = np.array([[0, 1], [-1, 0]])
        #         gx = apply_convolution(gray, roberts_x)
        #         gy = apply_convolution(gray, roberts_y)
        #         edge_roberts = np.hypot(gx, gy)
        #         return np.clip(edge_roberts, 0, 255).astype(np.uint8)
        
        elif opsi == "Edge Detection":
            gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale first
            if metode_edge == "Sobel":
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gx = ndimage.convolve(gray, sobel_x)
                gy = ndimage.convolve(gray, sobel_y)
                edge_sobel = np.hypot(gx, gy)
                return np.clip(edge_sobel, 0, 255).astype(np.uint8)
            elif metode_edge == "Prewitt":
                prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                gx = ndimage.convolve(gray, prewitt_x)
                gy = ndimage.convolve(gray, prewitt_y)
                edge_prewitt = np.hypot(gx, gy)
                return np.clip(edge_prewitt, 0, 255).astype(np.uint8)
            elif metode_edge == "Roberts":
                roberts_x = np.array([[1, 0], [0, -1]])
                roberts_y = np.array([[0, 1], [-1, 0]])
                gx = ndimage.convolve(gray, roberts_x)
                gy = ndimage.convolve(gray, roberts_y)
                edge_roberts = np.hypot(gx, gy)
                return np.clip(edge_roberts, 0, 255).astype(np.uint8)

    # Pemrosesan gambar berdasarkan opsi
    hasil = olah_gambar(img_np, opsi)

    # Menampilkan hasil pemrosesan dan histogram
    st.subheader(f"Hasil - {opsi}")
    col1, col2 = st.columns(2)
    with col1:
        tampilkan_judul(hasil, f"Hasil - {opsi}")
    with col2:
        tampilkan_histogram(hasil)

    # Membuat nama file untuk hasil yang akan diunduh
    original_filename = uploaded_file.name
    ext = os.path.splitext(original_filename)[1]
    nama_file_simpan = f"{os.path.splitext(original_filename)[0]}-{opsi.lower().replace(' ', '_')}{ext}"

    # Konversi hasil menjadi bytes
    hasil_bytes = convert_image_to_bytes(hasil)

    # Tombol download
    st.download_button(
        label=f"Download {opsi}",
        data=hasil_bytes,
        file_name=nama_file_simpan,
        mime=f"image/{ext[1:]}"
    )

else:
    st.write("Silakan upload gambar terlebih dahulu.")



