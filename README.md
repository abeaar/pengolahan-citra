# Aplikasi Pengolahan Citra Kelompok Esigma

Aplikasi ini adalah proyek pengolahan citra berbasis web yang dikembangkan menggunakan **Streamlit**. Fitur utama meliputi pengolahan gambar seperti grayscale, citra negatif, histogram equalization, rotasi, Gaussian blur, dan lainnya.

## Fitur
1. **Normal**: Menampilkan gambar asli tanpa perubahan.
2. **Citra Negatif**: Mengubah gambar menjadi negatif.
3. **Grayscale**: Mengubah gambar berwarna menjadi grayscale.
4. **Rotasi**: Memutar gambar sesuai sudut yang diinginkan atau melakukan flip horizontal/vertikal.
5. **Histogram Equalization**: Meningkatkan kontras gambar dengan menyesuaikan distribusi intensitas piksel.
6. **Black & White**: Mengubah gambar menjadi hitam putih berdasarkan nilai threshold.
7. **Smoothing (Gaussian Blur)**: Menghaluskan gambar dengan efek blur.
8. **Noise**: Menambahkan noise pada gambar (Gaussian, Speckle, atau Salt & Pepper).
9. **Channel RGB**: Menampilkan saluran warna spesifik (Red, Green, atau Blue).
10. **Edge Detection**: Mendeteksi tepi gambar menggunakan metode Sobel, Prewitt, atau Roberts.

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/abeaar/pengolahan-citra.git
cd pengolahan-citra
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run citra.py
```
