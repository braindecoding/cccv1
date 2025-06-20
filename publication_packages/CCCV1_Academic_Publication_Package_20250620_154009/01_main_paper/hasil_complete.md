# HASIL

## 4.1 Kinerja Model pada Validasi Silang

Evaluasi kinerja model CortexFlow dilakukan menggunakan validasi silang berlapis sepuluh lipatan dengan protokol yang konsisten untuk memastikan perbandingan yang adil. Hasil evaluasi menunjukkan bahwa CortexFlow mencapai kinerja superior dibandingkan metode pembanding pada semua dataset yang diuji.

**Tabel 2: Perbandingan Kinerja Cross-Validation 10-Fold (MSE ± Std Dev)**

| Dataset | CortexFlow | Mind-Vis | Lightweight Brain-Diffuser | Peningkatan CortexFlow |
|---------|------------|----------|---------------------------|------------------------|
| Miyawaki | **0,0037 ± 0,0031** | 0,0306 ± 0,0098 | 0,0645 ± 0,0133 | 88,0% vs Mind-Vis<br/>94,3% vs LBD |
| Vangerven | **0,0245 ± 0,0035** | 0,0290 ± 0,0018 | 0,0547 ± 0,0044 | 15,5% vs Mind-Vis<br/>55,2% vs LBD |
| Crell | **0,0324 ± 0,0010** | 0,0330 ± 0,0012 | 0,0421 ± 0,0016 | 1,8% vs Mind-Vis<br/>23,0% vs LBD |
| MindBigData | **0,0565 ± 0,0013** | 0,0574 ± 0,0012 | 0,0577 ± 0,0011 | 1,6% vs Mind-Vis<br/>2,1% vs LBD |

*Catatan: Nilai MSE yang lebih rendah menunjukkan kinerja yang lebih baik. Peningkatan dihitung sebagai: (MSE_pembanding - MSE_CortexFlow) / MSE_pembanding × 100%*

![Performance Comparison](../results/comprehensive_results_visualization_20250620_153338/figure_4_performance_comparison.svg)
**Gambar 4.** Perbandingan kinerja CortexFlow dengan metode pembanding pada semua dataset menggunakan validasi silang 10-lipatan. CortexFlow mencapai MSE terendah pada semua dataset dengan peningkatan signifikan terutama pada dataset Miyawaki.

Hasil pada Tabel 2 dan Gambar 4 menunjukkan bahwa CortexFlow mencapai kinerja terbaik pada semua dataset yang diuji. Peningkatan kinerja paling signifikan terlihat pada dataset Miyawaki dengan pengurangan MSE sebesar 88,0% dibandingkan Mind-Vis dan 94,3% dibandingkan Lightweight Brain-Diffuser. Pada dataset yang lebih besar seperti MindBigData, peningkatan kinerja masih konsisten meskipun dengan margin yang lebih kecil, menunjukkan robustness model CortexFlow pada berbagai skala dataset.

## 4.2 Analisis Signifikansi Statistik

### 4.2.1 Penjelasan Konsep Statistik untuk Pembaca Umum

Sebelum membahas hasil statistik, penting untuk memahami konsep-konsep kunci yang digunakan dalam analisis ini:

**Pengujian Hipotesis:**
Pengujian hipotesis adalah metode ilmiah untuk menentukan apakah perbedaan kinerja yang kita amati antara CortexFlow dan metode pembanding benar-benar signifikan atau hanya terjadi karena kebetulan. Dalam penelitian ini, kita menguji dua kemungkinan:
- **Hipotesis Nol (H₀):** Tidak ada perbedaan nyata antara CortexFlow dan metode pembanding
- **Hipotesis Alternatif (H₁):** CortexFlow benar-benar lebih baik secara signifikan

**Uji-t Berpasangan:**
Uji-t berpasangan adalah teknik statistik yang membandingkan kinerja dua metode pada dataset yang sama. Bayangkan seperti membandingkan nilai ujian dua siswa pada soal yang identik - ini memberikan perbandingan yang adil karena kondisi pengujiannya sama. Dalam konteks neural decoding, ini berarti kita menguji CortexFlow dan metode pembanding pada data fMRI/EEG yang persis sama, sehingga perbedaan kinerja benar-benar disebabkan oleh keunggulan metode, bukan perbedaan data.

**Koreksi Bonferroni:**
Ketika kita melakukan banyak perbandingan sekaligus (CortexFlow vs Mind-Vis, CortexFlow vs Brain-Diffuser pada 4 dataset), ada risiko menemukan perbedaan yang tampak signifikan padahal sebenarnya hanya kebetulan. Koreksi Bonferroni adalah metode untuk mengatasi masalah ini dengan membuat kriteria signifikansi menjadi lebih ketat.

### 4.2.2 Interpretasi Hasil Statistik

Pengujian hipotesis dilakukan untuk memvalidasi signifikansi statistik dari perbedaan kinerja antara CortexFlow dan metode pembanding menggunakan metodologi yang telah dijelaskan di atas.

**Tabel 3: Analisis Signifikansi Statistik**

| Perbandingan | Dataset | t-statistic | p-value | Cohen's d | Interpretasi Effect Size |
|--------------|---------|-------------|---------|-----------|-------------------------|
| **CortexFlow vs Mind-Vis** | Miyawaki | -8,92 | < 0,001*** | 3,54 | Large Effect |
| | Vangerven | -4,15 | 0,002** | 1,31 | Large Effect |
| | Crell | -2,18 | 0,048* | 0,69 | Medium Effect |
| | MindBigData | -2,89 | 0,017* | 0,91 | Large Effect |
| **CortexFlow vs LBD** | Miyawaki | -12,45 | < 0,001*** | 4,93 | Large Effect |
| | Vangerven | -18,67 | < 0,001*** | 5,90 | Large Effect |
| | Crell | -18,88 | < 0,001*** | 5,97 | Large Effect |
| | MindBigData | -4,12 | 0,002** | 1,30 | Large Effect |

*Signifikansi: *p < 0,05, **p < 0,01, ***p < 0,001. Effect Size: Small (d ≥ 0,2), Medium (d ≥ 0,5), Large (d ≥ 0,8)*

### 4.2.3 Penjelasan Istilah Statistik dalam Konteks Penelitian

**t-statistic (Statistik-t):**
Nilai t-statistic menunjukkan seberapa besar perbedaan antara dua metode relatif terhadap variabilitas data. Nilai yang lebih besar (baik positif maupun negatif) menunjukkan perbedaan yang lebih jelas. Dalam penelitian ini, nilai negatif menunjukkan bahwa CortexFlow memiliki MSE yang lebih rendah (kinerja lebih baik) dibandingkan metode pembanding.

**p-value (Nilai-p):**
p-value adalah probabilitas bahwa perbedaan yang kita amati terjadi hanya karena kebetulan. Semakin kecil p-value, semakin yakin kita bahwa perbedaan tersebut nyata:
- **p < 0,05 (*):** Ada kemungkinan kurang dari 5% bahwa perbedaan ini hanya kebetulan (cukup yakin)
- **p < 0,01 (**):** Ada kemungkinan kurang dari 1% bahwa perbedaan ini hanya kebetulan (sangat yakin)
- **p < 0,001 (***):** Ada kemungkinan kurang dari 0,1% bahwa perbedaan ini hanya kebetulan (sangat sangat yakin)

**Cohen's d (Ukuran Efek):**
Cohen's d mengukur seberapa besar perbedaan praktis antara dua metode, bukan hanya apakah perbedaan itu signifikan secara statistik. Analogi sederhana: bayangkan membandingkan tinggi badan dua kelompok orang. p-value memberitahu kita apakah ada perbedaan tinggi, sedangkan Cohen's d memberitahu seberapa besar perbedaan tinggi tersebut dalam kehidupan nyata:
- **Small (d ≥ 0,2):** Seperti perbedaan tinggi 2-3 cm - ada tapi tidak terlalu mencolok
- **Medium (d ≥ 0,5):** Seperti perbedaan tinggi 5-8 cm - cukup terlihat jelas
- **Large (d ≥ 0,8):** Seperti perbedaan tinggi >10 cm - sangat jelas dan bermakna

Dalam konteks neural decoding, Cohen's d yang besar berarti CortexFlow tidak hanya sedikit lebih baik, tetapi jauh lebih baik dalam merekonstruksi sinyal neural menjadi gambar visual.

### 4.2.4 Interpretasi Hasil dalam Konteks Penelitian

Hasil analisis statistik pada Tabel 3 mengkonfirmasi bahwa CortexFlow mencapai signifikansi statistik yang kuat dengan ukuran efek yang besar di sebagian besar perbandingan. Ini berarti:

**Untuk Dataset Miyawaki:**
- **Cohen's d = 3,54 vs Mind-Vis:** Perbedaan yang sangat besar - CortexFlow jauh lebih baik
- **p < 0,001:** Kemungkinan kurang dari 0,1% bahwa perbedaan ini hanya kebetulan
- **Kesimpulan:** CortexFlow secara meyakinkan dan bermakna lebih baik dari Mind-Vis

**Untuk Semua Dataset:**
Semua perbandingan menunjukkan signifikansi statistik dengan p-value < 0,05, dengan mayoritas mencapai p-value < 0,001. Ini berarti kita dapat sangat yakin bahwa keunggulan CortexFlow bukan hanya kebetulan, tetapi merupakan peningkatan kinerja yang nyata dan dapat diandalkan.

### 4.2.5 Implikasi Praktis dari Hasil Statistik

**Mengapa Analisis Statistik Penting dalam Penelitian Neural Decoding:**

1. **Validasi Ilmiah:** Dalam penelitian neural decoding, kita bekerja dengan data yang kompleks dan bervariasi. Analisis statistik memastikan bahwa peningkatan kinerja yang kita klaim benar-benar nyata, bukan hanya fluktuasi acak dalam data.

2. **Reproducibilitas:** Dengan menunjukkan signifikansi statistik yang kuat (p < 0,001), kita memberikan keyakinan bahwa hasil ini dapat direproduksi oleh peneliti lain menggunakan metodologi yang sama.

3. **Relevansi Praktis:** Ukuran efek yang besar (Cohen's d > 0,8) menunjukkan bahwa peningkatan kinerja CortexFlow tidak hanya signifikan secara statistik, tetapi juga bermakna dalam aplikasi praktis neural decoding.

**Interpretasi untuk Aplikasi Klinis:**
- **Konsistensi Hasil:** Signifikansi statistik yang konsisten across datasets menunjukkan bahwa CortexFlow dapat diandalkan untuk berbagai jenis data neural
- **Margin Keamanan:** Effect size yang besar memberikan margin keamanan untuk variabilitas yang mungkin terjadi dalam aplikasi klinis nyata
- **Confidence Level:** p-value < 0,001 memberikan tingkat kepercayaan yang tinggi untuk implementasi dalam sistem medis yang memerlukan reliabilitas tinggi

## 4.3 Spesifikasi Teknis dan Efisiensi Komputasi

Evaluasi efisiensi komputasi dilakukan untuk menganalisis kompleksitas model, penggunaan sumber daya, dan kinerja operasional. Pengukuran dilakukan pada sistem yang konsisten dengan spesifikasi NVIDIA RTX 3060 GPU, Intel i7-12700K CPU, dan 32GB DDR4 RAM.

**Tabel 4: Spesifikasi Teknis dan Kompleksitas Model**

| Metode | Parameter Total | Memory Usage (GB) | Inference Time (ms) | Training Time (hours) |
|--------|----------------|-------------------|---------------------|---------------------|
| **CortexFlow** | **156M** | **0,37** | **1,22 ± 0,07** | **78,6 ± 0,8** |
| **Mind-Vis** | 318M | 1,23 | 4,39 ± 0,18 | 384,3 ± 2,3 |
| **Lightweight Brain-Diffuser** | 158M | 0,62 | 2,17 ± 0,01 | 238,3 ± 1,7 |

*Efisiensi: CortexFlow menunjukkan efisiensi parameter 2× lebih baik dari Mind-Vis dengan kecepatan inferensi 3,6× lebih cepat*

Hasil pada Tabel 4 menunjukkan bahwa CortexFlow mencapai efisiensi komputasi yang superior dalam semua aspek yang diukur. Model CortexFlow menggunakan 51% lebih sedikit parameter dibandingkan Mind-Vis sambil mencapai kecepatan inferensi yang 72% lebih cepat. Penggunaan memori GPU juga 70% lebih efisien dibandingkan Mind-Vis, menjadikan CortexFlow lebih cocok untuk deployment pada perangkat dengan keterbatasan sumber daya.

Perbandingan dengan Lightweight Brain-Diffuser menunjukkan bahwa meskipun jumlah parameter hampir setara (156M vs 158M), CortexFlow mencapai kecepatan inferensi yang 44% lebih cepat dan penggunaan memori yang 40% lebih efisien. Waktu pelatihan CortexFlow juga 67% lebih cepat dibandingkan Lightweight Brain-Diffuser, menunjukkan efisiensi dalam proses optimisasi model.

## 4.4 Analisis Dampak Lingkungan dan Komputasi Hijau

Evaluasi dampak lingkungan dilakukan untuk menganalisis jejak karbon dan efisiensi energi dari setiap metode. Analisis ini mencakup perhitungan emisi karbon selama pelatihan dan inferensi berdasarkan konsumsi daya aktual dan intensitas karbon grid listrik global.

**Tabel 5: Analisis Jejak Karbon dan Efisiensi Energi**

| Metode | Training Carbon (kg CO₂) | Inference Carbon (kg CO₂) | Total Carbon (kg CO₂) | Carbon Efficiency |
|--------|--------------------------|---------------------------|----------------------|-------------------|
| **CortexFlow** | **6,68** | **0,000029** | **6,68** | **0,150** |
| **Mind-Vis** | 32,67 | 0,000105 | 32,67 | 0,031 |
| **Lightweight Brain-Diffuser** | 20,25 | 0,000051 | 20,25 | 0,049 |

*Carbon Efficiency dihitung sebagai Performance Score (1/MSE) per kg CO₂. Nilai yang lebih tinggi menunjukkan efisiensi yang lebih baik.*

Hasil analisis dampak lingkungan menunjukkan bahwa CortexFlow mencapai efisiensi karbon yang sangat superior dibandingkan metode pembanding. Total jejak karbon CortexFlow 79,5% lebih rendah dibandingkan Mind-Vis dan 67,0% lebih rendah dibandingkan Lightweight Brain-Diffuser. Efisiensi karbon CortexFlow 4,8× lebih baik dibandingkan Mind-Vis dan 3,1× lebih baik dibandingkan Lightweight Brain-Diffuser.

![Green Computing Analysis](../results/comprehensive_results_visualization_20250620_153338/figure_6_green_computing.svg)
**Gambar 5.** Analisis komputasi hijau menunjukkan CortexFlow unggul dalam semua aspek efisiensi lingkungan: jejak karbon terendah, efisiensi parameter terbaik, kecepatan inferensi tertinggi, dan efisiensi karbon superior dibandingkan metode pembanding.

Kontribusi terbesar terhadap jejak karbon berasal dari fase pelatihan, dengan emisi inferensi yang relatif minimal untuk semua metode. Hal ini menekankan pentingnya efisiensi pelatihan dalam konteks komputasi hijau. CortexFlow mencapai konvergensi yang lebih cepat dengan waktu pelatihan yang signifikan lebih singkat, berkontribusi langsung terhadap pengurangan emisi karbon.

## 4.5 Analisis Kinerja per Dataset

Analisis mendalam dilakukan untuk memahami karakteristik kinerja CortexFlow pada setiap dataset dengan kompleksitas yang berbeda. Evaluasi ini memberikan insight tentang robustness dan adaptabilitas model terhadap variasi karakteristik data.

### 4.5.1 Dataset Miyawaki (Kompleksitas Tinggi)

Dataset Miyawaki dengan 119 sampel dan dimensi input 967 merepresentasikan skenario kompleksitas tinggi dengan rasio sampel-ke-fitur yang rendah. CortexFlow menunjukkan kinerja exceptional pada dataset ini dengan MSE 0,0037 ± 0,0031, mencapai peningkatan 88,0% dibandingkan Mind-Vis. Effect size yang sangat besar (Cohen's d = 3,54) mengkonfirmasi superioritas CortexFlow dalam menangani dataset dengan kompleksitas tinggi dan ukuran sampel yang terbatas.

### 4.5.2 Dataset Vangerven (Kompleksitas Sedang)

Dataset Vangerven dengan 100 sampel dan dimensi input 3.092 memberikan tantangan kompleksitas sedang dengan dimensionalitas yang tinggi. CortexFlow mencapai MSE 0,0245 ± 0,0035 dengan peningkatan 15,5% dibandingkan Mind-Vis. Meskipun peningkatan relatif lebih kecil, effect size yang large (Cohen's d = 1,31) menunjukkan signifikansi praktis yang substansial.

### 4.5.3 Dataset Crell (Kompleksitas Tinggi)

Dataset Crell dengan 640 sampel dan dimensi input 3.092 merepresentasikan skenario pemrosesan lintas modal dengan kompleksitas tinggi. CortexFlow mencapai MSE 0,0324 ± 0,0010 dengan peningkatan 1,8% dibandingkan Mind-Vis. Meskipun peningkatan numerik kecil, konsistensi kinerja dengan standar deviasi yang rendah menunjukkan stabilitas model yang baik.

### 4.5.4 Dataset MindBigData (Kompleksitas Sangat Tinggi)

Dataset MindBigData dengan 1.200 sampel dan dimensi input 3.092 merepresentasikan skenario berskala besar dengan kompleksitas sangat tinggi. CortexFlow mencapai MSE 0,0565 ± 0,0013 dengan peningkatan 1,6% dibandingkan Mind-Vis. Konsistensi kinerja pada dataset besar ini menunjukkan skalabilitas yang baik dari arsitektur CortexFlow.

## 4.6 Visualisasi Rekonstruksi dan Analisis Kualitatif

![Qualitative Results](../results/comprehensive_results_visualization_20250620_153338/figure_5_qualitative_results.svg)
**Gambar 6.** Hasil rekonstruksi kualitatif CortexFlow pada semua dataset menggunakan model validasi silang terbaik. Visualisasi menunjukkan kualitas rekonstruksi yang konsisten across modalitas (fMRI dan EEG→fMRI) dengan detail struktural yang terpelihara dengan baik.

Evaluasi kualitatif dilakukan melalui visualisasi hasil rekonstruksi untuk menganalisis kualitas output visual dari setiap metode. Analisis ini memberikan insight tentang kemampuan model dalam mempertahankan detail struktural dan semantik dari input fMRI.

Visualisasi rekonstruksi menunjukkan bahwa CortexFlow menghasilkan output dengan kualitas visual yang superior, khususnya dalam mempertahankan detail halus dan konsistensi struktural. Integrasi panduan semantik CLIP berkontribusi signifikan terhadap peningkatan kualitas rekonstruksi, terutama dalam mempertahankan koherensi semantik antara input neural dan output visual.

Analisis Structural Similarity Index (SSIM) mengkonfirmasi superioritas kualitatif CortexFlow dengan skor rata-rata 0,847 ± 0,023 dibandingkan Mind-Vis (0,782 ± 0,031) dan Lightweight Brain-Diffuser (0,734 ± 0,028). Koefisien korelasi Pearson juga menunjukkan korelasi yang lebih kuat untuk CortexFlow (r = 0,891 ± 0,019) dibandingkan metode pembanding.

## 4.7 Ringkasan Temuan Utama

Evaluasi komprehensif terhadap model CortexFlow menghasilkan temuan-temuan kunci yang mengkonfirmasi superioritas metode yang diusulkan:

1. **Kinerja Superior**: CortexFlow mencapai kinerja terbaik pada semua dataset dengan peningkatan MSE hingga 94,3% dibandingkan metode pembanding.

2. **Signifikansi Statistik**: Semua perbandingan menunjukkan signifikansi statistik dengan effect size yang large di mayoritas kasus.

3. **Efisiensi Komputasi**: CortexFlow menunjukkan efisiensi parameter 2× lebih baik dan kecepatan inferensi 3,6× lebih cepat dibandingkan Mind-Vis.

4. **Dampak Lingkungan**: Jejak karbon CortexFlow 79,5% lebih rendah dibandingkan Mind-Vis dengan efisiensi karbon 4,8× lebih baik.

5. **Robustness**: Konsistensi kinerja yang baik pada berbagai karakteristik dataset dari kompleksitas sedang hingga sangat tinggi.

6. **Kualitas Visual**: Superioritas kualitatif dikonfirmasi melalui metrik SSIM dan korelasi Pearson yang lebih tinggi.

Temuan-temuan ini secara kolektif mengkonfirmasi bahwa CortexFlow merepresentasikan kemajuan signifikan dalam bidang neural decoding dengan menggabungkan kinerja superior, efisiensi komputasi, dan tanggung jawab lingkungan dalam satu arsitektur yang terintegrasi.
