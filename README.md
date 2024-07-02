Identifikasi Kalori Makanan untuk Manajemen Diabetes
Latar Belakang
Penggunaan smartphone yang semakin marak telah memudahkan pengguna dalam mengambil dan membagikan gambar makanan di media sosial. Kebiasaan ini membuka peluang bagi pengembang aplikasi kesehatan untuk menciptakan aplikasi yang dapat mengidentifikasi kandungan kalori makanan berdasarkan gambar. Bagi penderita diabetes mellitus, mengontrol asupan kalori sangat penting untuk menjaga kesehatan dan kualitas hidup. Penelitian ini bertujuan untuk mengidentifikasi jumlah kalori makanan yang akan dikonsumsi oleh penderita diabetes menggunakan metode transfer learning, serta membantu mereka dalam merencanakan dan memantau asupan kalori harian.

Metodologi
Metode yang digunakan dalam penelitian ini adalah Transfer Learning, dengan model yang dikembangkan menggunakan dataset yang terdiri dari 32 jenis olahan makanan dan minuman dengan total 3316 gambar. Arsitektur yang digunakan adalah MobileNetV2. Percobaan dilakukan dengan membagi data menjadi rasio 70:30 dan 80:20 untuk data pelatihan dan data pengujian. Selain itu, berbagai ukuran batch (16, 32, 64, dan 128) juga diuji untuk menemukan kombinasi terbaik antara pembagian data dan ukuran batch, sehingga menghasilkan model yang optimal.

Hasil
Hasil penelitian menunjukkan bahwa kombinasi antara pembagian data dengan perbandingan 70:30 dan penggunaan batch 64 memiliki kinerja model yang terbaik dengan skor akurasi 0.887, precision 0.924, recall 0.859, dan f1-score 0.89. Evaluasi dilakukan baik dengan metrik evaluasi maupun pengujian manual, menunjukkan hasil kalkulasi kalori yang akurat dan sesuai dengan referensi yang digunakan.

Kesimpulan
Hasil dari penelitian ini memiliki potensi sebagai alat bantu dalam manajemen nutrisi dan kesehatan serta dapat diintegrasikan dalam aplikasi yang lebih luas. Alat ini dapat membantu penderita diabetes dalam mengelola asupan kalori mereka dengan lebih efektif dan meningkatkan manajemen kesehatan secara keseluruhan.
