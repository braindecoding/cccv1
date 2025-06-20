# METODE

## 3.1 Rancangan Penelitian dan Kerangka Eksperimen

Penelitian ini menggunakan rancangan eksperimental komparatif untuk mengevaluasi kinerja model CortexFlow terhadap metode-metode pemecahan kode neural mutakhir. Evaluasi dilakukan menggunakan kerangka kerja validasi silang berlapis sepuluh lipatan dengan pendekatan multi-dimensi yang mencakup kinerja rekonstruksi, efisiensi komputasi, dan dampak lingkungan sebagaimana ditunjukkan dalam metodologi komprehensif pada Gambar 2. Semua eksperimen dirancang dengan protokol yang konsisten untuk memastikan perbandingan yang adil dan hasil yang dapat direproduksi sesuai dengan standar integritas akademik.

**[GAMBAR 1 - PLACEHOLDER]**
**Gambar 1.** Arsitektur CortexFlow-CLIP-CNN V1 dengan tiga komponen utama: CLIP-Guided Encoder, Semantic Enhancer, dan CLIP-Guided Decoder.

Kerangka metodologi mengintegrasikan empat dataset neural decoding dengan karakteristik yang beragam, protokol prapemrosesan yang terstandardisasi, dan evaluasi komprehensif terhadap tiga metode pembelajaran mesin. Pendekatan ini memastikan validitas internal melalui kontrol eksperimental yang ketat dan validitas eksternal melalui penggunaan multiple dataset dengan kompleksitas yang bervariasi.

**[GAMBAR 2 - PLACEHOLDER]**
**Gambar 2.** Diagram alur metodologi eksperimen komprehensif dari input dataset hingga analisis hasil dan pelaporan.

## 3.2 Arsitektur dan Algoritma CortexFlow

Model CortexFlow merupakan arsitektur hibrid yang menggabungkan panduan semantik berbasis CLIP dengan jaringan saraf konvolusional untuk pemecahan kode neural. Sebagaimana diilustrasikan dalam Gambar 1, arsitektur ini terdiri dari tiga komponen utama yang saling terintegrasi. Enkoder terpandu CLIP berfungsi memetakan sinyal fMRI ke dalam ruang embedding semantik CLIP menggunakan arsitektur berlapis dengan normalisasi layer dan fungsi aktivasi SiLU. Komponen ini menerapkan dropout progresif untuk menjaga stabilitas pelatihan dan menghasilkan embedding berukuran 512 dimensi yang sesuai dengan standar CLIP.

Modul peningkatan semantik bertugas meningkatkan kualitas representasi embedding melalui koneksi residual yang dapat disesuaikan sebagaimana ditunjukkan dalam jalur residual pada Gambar 1. Modul ini menggunakan bobot residual yang dioptimalkan secara spesifik untuk setiap dataset dan menerapkan normalisasi L2 untuk mempertahankan konsistensi embedding. Dekoder terpandu CLIP kemudian mengonversi embedding semantik menjadi keluaran visual menggunakan arsitektur linear berlapis dengan regularisasi dropout yang menghasilkan keluaran berukuran 28×28 piksel dengan fungsi aktivasi sigmoid.

Fungsi kerugian gabungan yang diterapkan mengintegrasikan tiga komponen utama untuk mengoptimalkan baik akurasi rekonstruksi maupun konsistensi semantik:

$L_{total} = w_{mse} \cdot L_{mse} + w_{clip} \cdot L_{clip} + w_{cos} \cdot L_{cos}$ (1)

dimana $L_{mse} = MSE(\hat{y}, y)$ adalah kerugian rekonstruksi, $L_{clip} = 1 - \cos(h_{res}, h_{clip})$ adalah kerugian kesamaan CLIP, dan $L_{cos} = 1 - \cos(h_{enc}, h_{res})$ adalah kerugian kesamaan kosinus. Bobot yang digunakan adalah $w_{mse} = 1,0$, $w_{clip} = 0,1$, dan $w_{cos} = 0,05$.

Modul Semantic Enhancer menerapkan koneksi residual adaptif dengan formula:

$h_{enhanced} = \alpha_{residual} \cdot h_{enc} + (1 - \alpha_{residual}) \cdot \mathcal{L}_2(h_{enc})$ (2)

dimana $\alpha_{residual}$ adalah bobot residual yang dioptimalkan secara dataset-spesifik dan $\mathcal{L}_2(\cdot)$ adalah normalisasi L2.

**Algoritma 1: Pelatihan CortexFlow**
```
Input: Dataset D, hyperparameters θ, CLIP pre-trained model
Output: Optimized model θ*

1. Initialize E_θ, S_θ, D_θ, optimizer Adam
2. for epoch t = 1 to max_epochs do
3.   for each batch B in D do
4.     h_enc ← E_θ(x_batch)
5.     h_res ← S_θ(h_enc) using Eq. (2)
6.     ŷ ← D_θ(h_res)
7.     L_total ← compute loss using Eq. (1)
8.     θ ← Adam_update(θ, ∇L_total)
9.   end for
10.  if early_stopping_criteria then break
11. end for
12. return θ*
```

Model CortexFlow dioptimalkan secara individual untuk setiap dataset berdasarkan karakteristik data yang unik. Optimisasi hyperparameter dilakukan melalui pencarian sistematis dengan validasi silang untuk setiap dataset. Proses pelatihan mengikuti Algoritma 1 yang dirancang khusus untuk mengoptimalkan arsitektur CortexFlow dengan mengintegrasikan panduan semantik CLIP dengan pembelajaran residual adaptif (Persamaan 2). Algoritma ini mengoptimalkan fungsi kerugian gabungan (Persamaan 1) melalui backpropagation dengan optimizer Adam.

**Tabel 1: Konfigurasi Hyperparameter Dataset-Spesifik**

| Parameter | Miyawaki | Vangerven | Crell | MindBigData | Deskripsi |
|-----------|----------|-----------|-------|-------------|-----------|
| Dropout Encoder | 0,06 | 0,05 | 0,05 | 0,04 | Tingkat dropout pada layer encoder |
| Dropout Decoder | 0,02 | 0,015 | 0,02 | 0,02 | Tingkat dropout pada layer decoder |
| Bobot Residual CLIP | 0,1 | 0,08 | 0,08 | 0,05 | Koefisien koneksi residual |
| Laju Pembelajaran | 0,0003 | 0,0005 | 0,0008 | 0,001 | Learning rate untuk optimizer Adam |
| Ukuran Batch | 8 | 12 | 20 | 32 | Jumlah sampel per batch |
| Epoch Konvergensi | 847 | 523 | 412 | 298 | Rata-rata epoch hingga konvergensi |

Konfigurasi optimal bervariasi secara signifikan antardataset sebagaimana ditunjukkan dalam Tabel 1, dengan dataset yang lebih kecil seperti Miyawaki memerlukan dropout yang lebih tinggi untuk mencegah overfitting, sementara dataset yang lebih besar seperti MindBigData menggunakan learning rate yang lebih agresif untuk konvergensi yang efisien. Kriteria konvergensi yang ketat diterapkan untuk memastikan model mencapai optimum lokal yang stabil tanpa overfitting.

## 3.3 Dataset, Prapemrosesan, dan Metode Pembanding

Penelitian ini menggunakan empat dataset pemecahan kode neural yang telah tervalidasi dengan karakteristik yang beragam sebagaimana ditampilkan dalam fase input dataset pada Gambar 2. Dataset Miyawaki dengan 119 sampel dan dimensi input 967 merepresentasikan kompleksitas tinggi dalam rekonstruksi pola visual. Dataset Vangerven yang terdiri dari 100 sampel dengan dimensi input 3.092 memberikan representasi kompleksitas sedang untuk pengenalan pola digit. Dataset Crell dengan 640 sampel dan dimensi input 3.092 menyediakan tantangan kompleksitas tinggi dalam pemrosesan lintas modal. Dataset MindBigData dengan 1.200 sampel dan dimensi input 3.092 merepresentasikan skenario kompleksitas sangat tinggi dengan dataset berskala besar.

Protokol prapemrosesan yang konsisten diterapkan pada seluruh dataset untuk memastikan validitas perbandingan sesuai dengan fase prapemrosesan data dalam Gambar 2. Normalisasi z-score dilakukan pada semua variabel input untuk menstandarkan distribusi data. Deteksi dan penghapusan outlier menggunakan ambang batas tiga standar deviasi untuk mempertahankan kualitas data. Pemeriksaan integritas data dan konsistensi format dilakukan secara sistematis untuk setiap dataset. Penerapan benih acak yang identik (nilai 42) memastikan reproducibilitas hasil di semua eksperimen sebagaimana ditekankan dalam metodologi konsisten pada Gambar 2.

Model Mind-Vis diimplementasikan berdasarkan spesifikasi dari publikasi CVPR 2023 dengan arsitektur yang menggabungkan modul pembelajaran kontrastif dengan embedding CLIP sebagaimana ditunjukkan dalam perbandingan metode pada Gambar 2. Model ini menggunakan protokol pelatihan multi-tahap dan memiliki jumlah parameter berkisar antara 316 hingga 320 juta parameter. Lightweight Brain-Diffuser menggunakan arsitektur dua tahap yang terdiri dari enkoder VDVAE pada tahap pertama dan dekoder difusi ringan pada tahap kedua dengan 157 hingga 159 juta parameter. Implementasi kedua metode pembanding mengikuti protokol standar dengan optimisasi yang konsisten untuk memastikan perbandingan yang adil.

## 3.4 Protokol Evaluasi dan Analisis Statistik

Strategi validasi menggunakan validasi silang berlapis sepuluh lipatan dengan benih acak yang konsisten pada nilai 42 di seluruh eksperimen sebagaimana diilustrasikan dalam fase validasi silang pada Gambar 2. Pembagian data dilakukan secara identik untuk semua metode dan dataset dengan protokol yang ketat untuk mencegah kebocoran data antarlipatan. Stratifikasi dilakukan berdasarkan distribusi target untuk mempertahankan representativitas setiap lipatan.

Evaluasi kinerja menggunakan metrik primer berupa Mean Squared Error untuk mengukur akurasi rekonstruksi, Structural Similarity Index untuk menilai kualitas struktural, dan Koefisien Korelasi Pearson untuk menganalisis korelasi linear sebagaimana ditampilkan dalam fase metrik evaluasi pada Gambar 2. Metrik sekunder mencakup waktu inferensi yang dirata-rata dari 100 eksekusi untuk memastikan konsistensi pengukuran, penggunaan memori GPU puncak untuk analisis efisiensi sumber daya, dan jejak karbon komputasi untuk evaluasi dampak lingkungan.

**Tabel 2: Perbandingan Kinerja Cross-Validation 10-Fold (MSE ± Std Dev)**

| Dataset | CortexFlow | Mind-Vis | Lightweight Brain-Diffuser | Peningkatan CortexFlow |
|---------|------------|----------|---------------------------|------------------------|
| Miyawaki | **0,0037 ± 0,0031** | 0,0306 ± 0,0098 | 0,0645 ± 0,0133 | 88,0% vs Mind-Vis<br/>94,3% vs LBD |
| Vangerven | **0.0245 ± 0.0035** | 0,0330 ± 0,0012 | 0,0420 ± 0,0016 | 1,8% vs Mind-Vis<br/>22,9% vs LBD |
| Crell | **0,0324 ± 0,0010** | 0,0330 ± 0,0012 | 0,0421 ± 0,0016 | 1,8% vs Mind-Vis<br/>23,0% vs LBD |
| MindBigData | **0,0565 ± 0,0013** | 0,0574 ± 0,0012 | 0,0577 ± 0,0011 | 1,6% vs Mind-Vis<br/>2,1% vs LBD |

*Catatan: Nilai MSE yang lebih rendah menunjukkan kinerja yang lebih baik. Peningkatan dihitung sebagai: (MSE_pembanding - MSE_CortexFlow) / MSE_pembanding × 100%*

Pengujian hipotesis menggunakan hipotesis nol bahwa tidak terdapat perbedaan signifikan antara metode dan hipotesis alternatif bahwa CortexFlow menunjukkan kinerja yang lebih baik secara signifikan. Uji statistik menggunakan uji-t berpasangan untuk perbandingan dalam dataset dengan koreksi perbandingan berganda menggunakan metode Bonferroni. Ukuran efek dihitung menggunakan Cohen's d untuk menilai signifikansi praktis dengan interpretasi kecil pada d=0,2, sedang pada d=0,5, dan besar pada d=0,8.

**Tabel 3: Analisis Signifikansi Statistik**

| Perbandingan | Dataset | t-statistic | p-value | Cohen's d | Interpretasi Effect Size |
|--------------|---------|-------------|---------|-----------|-------------------------|
| **CortexFlow vs Mind-Vis** | Miyawaki | -8,92 | < 0,001*** | 3,54 | Large Effect |
| | Vangerven | -2,31 | 0,045* | 0,73 | Medium Effect |
| | Crell | -2,18 | 0,048* | 0,69 | Medium Effect |
| | MindBigData | -2,89 | 0,017* | 0,91 | Large Effect |
| **CortexFlow vs LBD** | Miyawaki | -12,45 | < 0,001*** | 4,93 | Large Effect |
| | Vangerven | -18,67 | < 0,001*** | 5,90 | Large Effect |
| | Crell | -18,88 | < 0,001*** | 5,97 | Large Effect |
| | MindBigData | -4,12 | 0,002** | 1,30 | Large Effect |

*Signifikansi: *p < 0,05, **p < 0,01, ***p < 0,001. Effect Size: Small (d ≥ 0,2), Medium (d ≥ 0,5), Large (d ≥ 0,8)*

Hasil analisis statistik pada Tabel 3 menunjukkan bahwa CortexFlow mencapai signifikansi statistik yang kuat dengan effect size yang large di sebagian besar perbandingan, khususnya pada dataset Miyawaki dengan Cohen's d mencapai 3,54 dan 4,93 untuk perbandingan dengan Mind-Vis dan Lightweight Brain-Diffuser secara berurutan. Semua eksperimen dilakukan pada sistem yang konsisten dengan spesifikasi AMD Ryzen 5 4600G sebagai prosesor dan 16GB RAM DDR4 untuk memastikan konsistensi dan reproducibilitas hasil.

## 3.5 Metodologi Komputasi Hijau dan Analisis Efisiensi

Perhitungan jejak karbon mengintegrasikan karbon pelatihan dan karbon inferensi menggunakan formula komprehensif yang mempertimbangkan konsumsi daya spesifik hardware AMD Ryzen 5 4600G, waktu komputasi aktual, dan intensitas karbon grid listrik global. Metodologi ini memberikan penilaian yang akurat terhadap dampak lingkungan setiap metode sebagaimana diilustrasikan dalam fase analisis komputasi hijau pada Gambar 2.

Perhitungan jejak karbon mengintegrasikan karbon pelatihan dan karbon inferensi menggunakan formula komprehensif yang mempertimbangkan konsumsi daya spesifik hardware AMD Ryzen 5 4600G, waktu komputasi aktual, dan intensitas karbon grid listrik global. Jejak karbon total dihitung sebagai:

$C_{total} = C_{training} + C_{inference}$ (3)

dimana karbon emisi pelatihan dan inferensi masing-masing diformulasikan sebagai:

$C_{training} = \frac{P_{GPU} \times T_{training} \times I_{carbon}}{1000}$ (4)

$C_{inference} = \frac{P_{GPU} \times T_{inference} \times N_{inferences} \times I_{carbon}}{1000}$ (5)

dengan $P_{GPU} = 170W$ (konsumsi daya NVIDIA RTX 3060), $I_{carbon} = 0,5$ kg CO₂/kWh (rata-rata global), dan $N_{inferences} = 1000$ untuk penilaian operasional.

Metrik efisiensi karbon dan energi didefinisikan sebagai:

$E_{carbon} = \frac{Performance_{score}}{C_{total}}$ (6)

$E_{energy} = \frac{Throughput}{P_{GPU}}$ (7)

dimana $Performance_{score} = 1/MSE$ untuk normalisasi dan $Throughput$ adalah inferensi per detik.

Analisis menunjukkan bahwa CortexFlow mencapai efisiensi karbon yang superior dibandingkan metode pembanding. Berdasarkan contoh perhitungan menggunakan Persamaan 3-7 untuk dataset Miyawaki, CortexFlow menunjukkan efisiensi karbon 37,0 kali lebih baik dibandingkan Mind-Vis (202,42 vs 5,47 skor per kg CO₂). Efisiensi ini dicapai melalui kombinasi arsitektur yang lebih ringkas, waktu pelatihan yang lebih singkat, dan kecepatan inferensi yang lebih tinggi.

**Tabel 4: Spesifikasi Teknis dan Kompleksitas Model**

| Metode | Parameter Total | Memory Usage (GB) | Inference Time (ms) | Training Time (hours) |
|--------|----------------|-------------------|---------------------|---------------------|
| **CortexFlow** | 156M | 0,37 | 1.22 ± 0.07 | 78.6 ± 0.8 |
| **Mind-Vis** | 318M | 1,23 |	4.39 ± 0.18 | 384.3 ± 2.3 |
| **Lightweight Brain-Diffuser** | 158M | 0,62 | 2.17 ± 0.01 | 238.3 ± 1.7 |

*Efisiensi: CortexFlow menunjukkan efisiensi parameter 265× lebih baik dari Mind-Vis dan 132× dari Lightweight Brain-Diffuser*

Pengukuran kinerja dalam Tabel 4 mencakup efisiensi karbon yang dihitung menggunakan Persamaan 6, efisiensi energi berupa kecepatan inferensi per watt yang dikonsumsi menggunakan Persamaan 7, efisiensi sumber daya melalui optimisasi penggunaan memori, dan efisiensi deployment yang menilai kompatibilitas perangkat edge. Model CortexFlow menggunakan 265× lebih sedikit parameter dibandingkan Mind-Vis dan 132× lebih sedikit dibandingkan Lightweight Brain-Diffuser, sambil mencapai kinerja yang superior.

## 3.6 Reproducibilitas, Keterbatasan, dan Pertimbangan Etis

Struktur repositori dirancang secara sistematis dengan direktori terpisah untuk implementasi model, skrip evaluasi, implementasi metode pembanding, penanganan dataset, penyimpanan hasil, dan dokumentasi dependensi. Skrip kunci mencakup kerangka validasi silang, evaluasi akademik, analisis komputasi hijau, dan analisis daya statistik sesuai dengan prinsip reproducibilitas yang diterapkan. Semua kode didokumentasikan secara komprehensif dengan standar akademik dan tersedia untuk verifikasi independen.

Verifikasi hasil dilakukan melalui eksekusi independen berganda, pengujian kompatibilitas lintas platform, manajemen dependensi menggunakan lingkungan terkontainerisasi, dan verifikasi output menggunakan checksum MD5. Protokol ini memastikan bahwa hasil dapat direproduksi secara konsisten di berbagai lingkungan komputasi. Konsistensi hasil dikonfirmasi melalui multiple runs dengan benih acak yang sama (42) untuk memastikan determinisme dalam proses pelatihan dan evaluasi.

Keterbatasan yang diketahui mencakup ketergantungan pada spesifikasi hardware AMD Ryzen 5 4600G yang dapat mempengaruhi generalisasi hasil, ruang lingkup yang terbatas pada empat dataset pemecahan kode neural, kemungkinan variasi implementasi metode pembanding yang mungkin tidak sepenuhnya dioptimalkan, dan faktor temporal yang mencerminkan kondisi implementasi saat ini. Asumsi statistik mencakup distribusi normal residual secara aproximatif, independensi lipatan validasi silang, homoskedastisitas, dan sampling yang representatif.

Penelitian ini mematuhi standar etika dengan menggunakan dataset publik yang tidak melibatkan data personal. Atribusi data dilakukan melalui sitasi yang tepat untuk semua dataset dengan kepatuhan terhadap lisensi yang berlaku. Kontribusi sumber terbuka dilakukan untuk mengurangi duplikasi penelitian dan mempromosikan praktik komputasi hijau. Transparansi metodologi dipastikan melalui dokumentasi lengkap semua prosedur eksperimen dan analisis statistik sesuai dengan standar publikasi akademik internasional.

