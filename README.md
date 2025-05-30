# Laporan Proyek Machine Learning - Yulia Harni

## Project Overview

Menonton film telah menjadi salah satu bentuk hiburan yang sangat populer di seluruh dunia. Namun, dengan jumlah film yang sangat banyak, pengguna sering mengalami kesulitan dalam memilih film yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi menjadi solusi yang sangat relevan untuk membantu pengguna menemukan film yang mereka sukai berdasarkan minat dan kesukaan sebelumnya.

Dalam proyek ini, dikembangkan sistem rekomendasi film berbasis konten (**Content-Based Filtering**) yang menggunakan informasi dari film itu sendiri, seperti genre, deskripsi, dan skor emosional, untuk merekomendasikan film yang mirip dengan yang disukai pengguna.

Penerapan sistem rekomendasi ini tidak hanya berguna bagi pengguna akhir, tetapi juga memiliki nilai bisnis bagi penyedia layanan streaming atau platform media. Dengan memberikan rekomendasi yang relevan dan personal, platform dapat meningkatkan keterlibatan pengguna, memperpanjang waktu penggunaan aplikasi, dan mendorong loyalitas pelanggan.

---

## Business Understanding

### Problem Statements

1. Bagaimana sistem dapat memberikan rekomendasi film yang relevan kepada pengguna berdasarkan preferensi kontennya, seperti genre dan deskripsi film?
2. Bagaimana sistem dapat mengidentifikasi film yang mirip secara konten untuk membantu pengguna menemukan tontonan baru yang sesuai?

### Goals

1. Membangun sistem rekomendasi berbasis konten yang mampu menyarankan film-film serupa berdasarkan genre dan karakteristik film lainnya.
2. Meningkatkan pengalaman pengguna dalam menemukan film yang sesuai dengan preferensi mereka tanpa harus mencarinya secara manual.

### Solution Statements

Untuk mencapai tujuan tersebut, pendekatan yang digunakan meliputi:

- Menggunakan metode **Content-Based Filtering** yang memanfaatkan fitur konten seperti genre dan deskripsi film.
- Menerapkan representasi fitur menggunakan teknik **TF-IDF Vectorization** pada teks genre atau deskripsi film.
- Mengukur kemiripan antar film dengan **cosine similarity**, sehingga sistem dapat merekomendasikan film yang paling mirip dari segi konten.

## 3. Data Understanding

Dataset yang digunakan dalam proyek ini berisi informasi mengenai **41.399 film** dengan total **19 fitur**. Dataset ini diambil dari file `filmtv_movies.csv` sumber: https://www.kaggle.com/datasets/stefanoleone992/filmtv-movies-dataset dan memuat berbagai informasi penting mengenai film, mulai dari detail umum hingga skor emosional.

### Informasi Dataset

Berikut adalah daftar fitur yang tersedia dalam dataset beserta deskripsinya:

| Nama Kolom     | Deskripsi                                                                 |
|----------------|---------------------------------------------------------------------------|
| `filmtv_id`    | ID unik untuk masing-masing film                                           |
| `title`        | Judul film                                                                |
| `year`         | Tahun rilis film                                                          |
| `genre`        | Genre film (bisa lebih dari satu, dipisahkan dengan koma)                 |
| `duration`     | Durasi film dalam menit                                                   |
| `country`      | Negara asal film                                                          |
| `directors`    | Nama sutradara                                                            |
| `actors`       | Nama-nama aktor utama dalam film                                          |
| `avg_vote`     | Rata-rata rating dari semua sumber                                        |
| `critics_vote` | Rating yang diberikan oleh kritikus                                       |
| `public_vote`  | Rating dari penonton umum                                                 |
| `total_votes`  | Jumlah voting total dari pengguna                                         |
| `description`  | Deskripsi singkat mengenai isi atau plot film                             |
| `notes`        | Catatan tambahan (jika ada)                                               |
| `humor`        | Skor emosi untuk unsur humor                                              |
| `rhythm`       | Skor emosi untuk ritme film                                               |
| `effort`       | Skor emosi untuk usaha atau intensitas film                               |
| `tension`      | Skor emosi untuk ketegangan                                               |
| `erotism`      | Skor emosi untuk unsur erotis                                             |

Dataset ini sangat kaya akan informasi dan cocok digunakan untuk membangun **sistem rekomendasi berbasis konten (Content-Based Filtering)**. Fokus utama pada proyek ini adalah fitur **`genre`**, yang akan digunakan sebagai dasar dalam mengukur kemiripan antar film.

### Jumlah Data

Setelah dataset dimuat menggunakan pustaka `pandas`

## 3. Data Preprocessing

Tahapan preprocessing dilakukan untuk memastikan data siap digunakan dalam pembangunan sistem rekomendasi berbasis konten. Proses preprocessing meliputi beberapa langkah sebagai berikut:

### 3.1 Cek dan Tangani Missing Values

**Menghapus baris dengan nilai kosong pada kolom `genre`**  
  Karena genre merupakan fitur utama dalam sistem rekomendasi ini, maka baris yang tidak memiliki informasi genre dianggap tidak relevan dan dihapus dari dataset.

**Mengisi nilai kosong pada kolom kategorikal tertentu:**  
  - `country`, `directors`, dan `actors` diisi dengan nilai `"Unknown"` sebagai placeholder.  
  - `description` diisi dengan string kosong `""` untuk menghindari error saat pemrosesan teks.
**Menghapus kolom yang tidak relevan:**  
  Beberapa kolom dihapus karena tidak berkontribusi langsung terhadap sistem rekomendasi:
  - `filmtv_id`
  - `notes`
  - `critics_vote`
  - `public_vote`

### 3.2 Pemeriksaan Duplikat

Dilakukan pemeriksaan terhadap data yang terduplikasi. Duplikat dihapus jika seluruh nilai pada baris tersebut sama persis, untuk memastikan tidak ada entri film yang tercatat lebih dari satu kali dengan informasi identik.

### 3.3 Pembersihan Data Teks

Langkah ini penting untuk memastikan hasil yang lebih akurat pada saat transformasi teks menggunakan TF-IDF dan penghitungan cosine similarity.

- Teks pada kolom `title`, `genre`, dan `description` dibersihkan dari simbol, angka, dan huruf kapital.
- Semua huruf dikonversi ke huruf kecil (lowercase).
- Karakter non-alfabet dihapus untuk menjaga konsistensi representasi teks.

## 4. Data Preparation

Pada tahap ini dilakukan proses penggabungan beberapa fitur penting dari dataset menjadi satu fitur gabungan bernama `combined_features`. Tujuan dari penggabungan ini adalah untuk merepresentasikan karakteristik konten tiap film dalam bentuk teks yang bisa diproses lebih lanjut oleh model, terutama saat dilakukan vektorisasi dan perhitungan kemiripan antar film.

### 🔍 Fitur yang Digunakan:

- **genre** — menggambarkan kategori atau jenis film, seperti drama, komedi, aksi, dll.
- **description** — berisi sinopsis atau isi cerita dari film.
- **directors** — menyatakan siapa yang menyutradarai film tersebut.
- **actors** — menyebutkan aktor dan aktris utama yang membintangi film.

Semua fitur ini digabungkan menjadi satu kolom teks agar bisa diolah menggunakan metode TF-IDF dan digunakan dalam penghitungan cosine similarity untuk membangun sistem rekomendasi berbasis konten.

## 5. Model Development dengan Content-Based Filtering

Pada tahap ini, dilakukan proses transformasi teks dari fitur `combined_features` menjadi bentuk numerik menggunakan teknik **TF-IDF Vectorization**, lalu menghitung kemiripan antar film menggunakan **Euclidean Distance**. Tujuannya adalah untuk merekomendasikan film yang memiliki konten serupa berdasarkan genre, sinopsis, sutradara, dan aktor.

### 5.1 TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) digunakan untuk mengubah data teks menjadi representasi numerik dalam bentuk vektor. Metode ini mempertimbangkan seberapa penting suatu kata dalam satu dokumen relatif terhadap keseluruhan dokumen lain. Semakin spesifik dan jarang muncul suatu kata, semakin tinggi bobotnya dalam vektor TF-IDF.

### 5.2 Euclidean Distance Similarity

Setelah teks diubah menjadi vektor, dilakukan perhitungan kemiripan antar film menggunakan **Euclidean Distance**. Metode ini mengukur jarak antara dua vektor dalam ruang multidimensi. Semakin kecil nilai jaraknya, maka semakin mirip isi/konten kedua film tersebut.

# 6. Evaluasi Fungsi Rekomendasi

Dibuat sebuah fungsi `recommend_same_genre_euclidean()` yang dapat memberikan rekomendasi film-film yang kontennya paling mirip dengan film input berdasarkan judulnya.

Contoh penggunaan:
```python
recommend_same_genre_euclidean("Ride a Wild Pony", df)
```
Output:
['Senza famiglia',
'Paradise',
"Sogno d'amore",
'Inga Lindström: Die Pferde von Katarinaberg',
'A Perfect Stranger']


Hasil rekomendasi menunjukkan bahwa film-film tersebut memiliki genre dan karakter cerita yang serupa dengan film input, yaitu film bergenre drama keluarga dan petualangan.

### 🔍 Insight:

- Hasil rekomendasi bersifat masuk akal dan relevan secara konten.
- Sistem sudah mampu memberikan alternatif film serupa meskipun belum ada interaksi pengguna.

