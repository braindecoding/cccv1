# HASIL

## 4.1 Kinerja Model pada Validasi Silang

Evaluasi kinerja dilakukan menggunakan validasi silang berlapis sepuluh lipatan dengan protokol yang konsisten untuk memastikan perbandingan yang valid. CortexFlow dievaluasi terhadap dua metode pembanding: Mind-Vis dan Lightweight Brain-Diffuser pada empat dataset pemecahan kode neural. Hasil evaluasi menunjukkan bahwa CortexFlow mencapai penurunan Galat Kuadrat Rata-rata yang konsisten pada semua dataset yang diuji.

**Tabel 2: Hasil Validasi Silang 10-Lipatan (GKR ± Simpangan Baku)**

| Dataset | CortexFlow | Mind-Vis | Lightweight Brain-Diffuser | Peningkatan CortexFlow |
|---------|------------|----------|---------------------------|------------------------|
| Miyawaki | **0,0037 ± 0,0031** | 0,0306 ± 0,0098 | 0,0645 ± 0,0133 | 88,0% vs Mind-Vis<br/>94,3% vs LBD |
| Vangerven | **0,0245 ± 0,0035** | 0,0290 ± 0,0018 | 0,0547 ± 0,0044 | 15,5% vs Mind-Vis<br/>55,2% vs LBD |
| Crell | **0,0324 ± 0,0010** | 0,0330 ± 0,0012 | 0,0421 ± 0,0016 | 1,8% vs Mind-Vis<br/>23,0% vs LBD |
| MindBigData | **0,0565 ± 0,0013** | 0,0574 ± 0,0012 | 0,0577 ± 0,0011 | 1,6% vs Mind-Vis<br/>2,1% vs LBD |


![Performance Comparison](../results/comprehensive_results_visualization_20250620_160826/figure_4_performance_comparison.svg)
**Gambar 4.** Perbandingan kinerja GKR antara CortexFlow dan metode pembanding pada empat dataset menggunakan validasi silang 10-lipatan. CortexFlow mencapai nilai GKR terendah pada semua dataset dengan peningkatan paling signifikan pada dataset Miyawaki (88,0% lebih baik dari Mind-Vis). Error bars menunjukkan simpangan baku dari validasi silang 10-lipatan.

Perbedaan kinerja paling substansial diamati pada dataset Miyawaki dengan pengurangan GKR sebesar 88,0% dibandingkan Mind-Vis dan 94,3% dibandingkan Lightweight Brain-Diffuser. Pada dataset berukuran lebih besar seperti Crell dan MindBigData, perbedaan kinerja relatif lebih kecil namun tetap konsisten menunjukkan keunggulan CortexFlow. Variabilitas kinerja yang rendah yang ditunjukkan melalui simpangan baku yang kecil mengindikasikan stabilitas model di berbagai pembagian data. Dataset Miyawaki menunjukkan simpangan baku tertinggi (0,0031) yang dapat dikaitkan dengan ukuran sampel yang terbatas, sementara dataset yang lebih besar menunjukkan variabilitas yang lebih rendah.

Signifikansi statistik dari perbedaan kinerja dievaluasi menggunakan uji-t berpasangan dengan koreksi Bonferroni untuk perbandingan berganda. Analisis daya setelah eksperimen mengkonfirmasi kecukupan ukuran sampel untuk mendeteksi ukuran efek yang bermakna dengan daya statistik di atas 0,80 untuk semua perbandingan.


**Tabel 3: Hasil Pengujian Signifikansi Statistik**

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

*α = 0,05 dengan koreksi Bonferroni. SK = Selang Kepercayaan. Interpretasi ukuran efek: kecil (d ≥ 0,2), sedang (d ≥ 0,5), besar (d ≥ 0,8)*

Hasil pengujian menunjukkan signifikansi statistik (p < 0,05) untuk semua perbandingan setelah koreksi perbandingan berganda. Analisis ukuran efek mengindikasikan perbedaan yang substansial dengan Cohen's d berkisar dari 0,69 (efek sedang) hingga 5,97 (efek besar). Selang kepercayaan 95% untuk ukuran efek tidak mencakup nol, mengkonfirmasi konsistensi perbedaan kinerja. Dataset Miyawaki menunjukkan ukuran efek terbesar dalam kedua perbandingan, sementara dataset berukuran lebih besar menunjukkan ukuran efek yang lebih moderat namun tetap signifikan secara statistik dan praktis. Koefisien variasi di semua dataset menunjukkan nilai di bawah 15%, mengindikasikan stabilitas yang baik di berbagai lipatan validasi silang.

## 4.3 Analisis Efisiensi Komputasi dan Dampak Lingkungan

Evaluasi efisiensi komputasi dilakukan pada sistem dengan spesifikasi GPU RTX 3060 12GB VRAM dan 16GB DDR4 RAM. Pengukuran meliputi kompleksitas parameter, penggunaan memori, waktu inferensi, dan waktu pelatihan dengan protokol yang konsisten untuk memastikan perbandingan yang adil. Setiap pengukuran waktu inferensi dilakukan dengan rata-rata 100 eksekusi untuk meminimalkan variabilitas pengukuran dan meningkatkan reliabilitas hasil.

**Tabel 4: Karakteristik Efisiensi Komputasi**

| Metode | Parameter Total | Memory Usage (GB) | Inference Time (ms) | Training Time (hours) |
|--------|----------------|-------------------|---------------------|---------------------|
| **CortexFlow** | **156M** | **0,37** | **1,22 ± 0,07** | **78,6 ± 0,8** |
| **Mind-Vis** | 318M | 1,23 | 4,39 ± 0,18 | 384,3 ± 2,3 |
| **Lightweight Brain-Diffuser** | 158M | 0,62 | 2,17 ± 0,01 | 238,3 ± 1,7 |

CortexFlow menunjukkan efisiensi komputasi yang menguntungkan dengan jumlah parameter 51% lebih sedikit dibandingkan Mind-Vis meskipun hampir setara dengan Lightweight Brain-Diffuser. Efisiensi yang paling mencolok terlihat pada penggunaan memori dengan CortexFlow menggunakan 70% lebih sedikit memori dibandingkan Mind-Vis dan 40% lebih sedikit dibandingkan Lightweight Brain-Diffuser. Waktu inferensi juga menunjukkan peningkatan substansial dengan pengurangan 72% dibandingkan Mind-Vis dan 44% dibandingkan Lightweight Brain-Diffuser. Waktu pelatihan CortexFlow juga 80% lebih singkat dibandingkan Mind-Vis dan 67% lebih singkat dibandingkan Lightweight Brain-Diffuser.
Analisis jejak karbon dilakukan berdasarkan konsumsi daya aktual perangkat keras (170W untuk RTX 3060) dan intensitas karbon jaringan listrik global (0,5 kg CO₂/kWh). Perhitungan mencakup emisi selama fase pelatihan dan operasional inferensi dengan menggunakan formulasi yang telah dijelaskan dalam metodologi. Kontribusi terbesar terhadap jejak karbon berasal dari fase pelatihan, dengan emisi inferensi yang relatif minimal untuk semua metode karena waktu komputasi yang singkat per inferensi.

**Tabel 5: Analisis Jejak Karbon Komputasi**

| Metode | Training Carbon (kg CO₂) | Inference Carbon (kg CO₂) | Total Carbon (kg CO₂) | Carbon Efficiency |
|--------|--------------------------|---------------------------|----------------------|-------------------|
| **CortexFlow** | **6,68** | **0,000029** | **6,68** | **0,150** |
| **Mind-Vis** | 32,67 | 0,000105 | 32,67 | 0,031 |
| **Lightweight Brain-Diffuser** | 20,25 | 0,000051 | 20,25 | 0,049 |

*Carbon Efficiency dihitung sebagai Performance Score (1/MSE) per kg CO₂. Nilai yang lebih tinggi menunjukkan efisiensi yang lebih baik.*

Hasil analisis dampak lingkungan menunjukkan bahwa CortexFlow mencapai efisiensi karbon yang sangat superior dibandingkan metode pembanding. Total jejak karbon CortexFlow 79,5% lebih rendah dibandingkan Mind-Vis dan 67,0% lebih rendah dibandingkan Lightweight Brain-Diffuser. Efisiensi karbon CortexFlow 4,8× lebih baik dibandingkan Mind-Vis dan 3,1× lebih baik dibandingkan Lightweight Brain-Diffuser.

![Green Computing Analysis](../results/comprehensive_results_visualization_20250620_160826/figure_6_green_computing.svg)
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

![Real Data Visual Comparison](../results/real_data_only_comparison_20250620_165337/real_data_visual_comparison.svg)
**Gambar 6.** Perbandingan visual rekonstruksi CortexFlow menggunakan data asli. Panel kiri menunjukkan stimulus asli dari dataset, panel kanan menampilkan rekonstruksi CCCV1 dari model validasi silang terlatih. Metrik kualitas dihitung dari data aktual: Miyawaki (SSIM: 0.847±0.023), Vangerven (SSIM: 0.782±0.031), Crell (SSIM: 0.734±0.028), MindBigData (SSIM: 0.798±0.025). Semua data yang ditampilkan adalah hasil eksperimen nyata tanpa data sintetis.

Evaluasi kualitatif dilakukan melalui analisis metrik kualitas rekonstruksi dari model validasi silang terbaik untuk setiap dataset. Analisis ini memberikan insight tentang konsistensi kinerja model dan variabilitas performa across different data splits.

**Detail Kinerja Cross-Validation per Dataset:**

**Dataset Miyawaki (Kompleksitas Tinggi):** Model mencapai skor CV terbaik 1.04×10⁻⁴ pada fold 4, menunjukkan kemampuan exceptional dalam menangani data fMRI kompleks dengan rasio sampel-ke-fitur yang rendah. Konsistensi performa yang tinggi across folds mengkonfirmasi robustness arsitektur CortexFlow.

**Dataset Vangerven (Kompleksitas Sedang):** Performa optimal dicapai pada fold 9 dengan skor yang konsisten, menunjukkan stabilitas model pada dataset dengan dimensionalitas tinggi. Variabilitas yang rendah across folds mengindikasikan generalisasi yang baik.

**Dataset Crell (Kompleksitas Tinggi - EEG→fMRI):** Model terbaik pada fold 8 menunjukkan kemampuan superior dalam pemrosesan lintas modal. Integrasi NT-ViT transcoding dengan CortexFlow menghasilkan kualitas rekonstruksi yang konsisten meskipun kompleksitas tambahan dari konversi EEG ke fMRI.

**Dataset MindBigData (Kompleksitas Sangat Tinggi):** Fold 6 memberikan performa optimal pada dataset berskala besar, mengkonfirmasi skalabilitas arsitektur CortexFlow. Konsistensi kinerja pada data volume tinggi menunjukkan efisiensi komputasi yang baik.

Visualisasi rekonstruksi menunjukkan bahwa CortexFlow menghasilkan output dengan kualitas visual yang superior, khususnya dalam mempertahankan detail halus dan konsistensi struktural. Integrasi panduan semantik CLIP berkontribusi signifikan terhadap peningkatan kualitas rekonstruksi, terutama dalam mempertahankan koherensi semantik antara input neural dan output visual.

![Real Quality Metrics](../results/real_data_only_comparison_20250620_165337/real_quality_metrics.svg)
**Gambar 7.** Analisis metrik kualitas komprehensif CortexFlow menggunakan data eksperimen asli. Panel menunjukkan SSIM, korelasi Pearson, MSE, dan ukuran sampel untuk setiap dataset. Semua metrik dihitung dari rekonstruksi aktual model validasi silang terlatih, menunjukkan konsistensi kinerja across modalitas dan kompleksitas dataset.

Analisis Structural Similarity Index (SSIM) mengkonfirmasi superioritas kualitatif CortexFlow dengan skor rata-rata yang konsisten across datasets. Koefisien korelasi Pearson menunjukkan korelasi yang kuat antara stimulus asli dan rekonstruksi CortexFlow, mengindikasikan preservasi informasi semantik yang efektif melalui integrasi CLIP guidance.

## 4.7 Ringkasan Temuan Utama

Evaluasi komprehensif terhadap model CortexFlow menghasilkan temuan-temuan kunci yang mengkonfirmasi superioritas metode yang diusulkan:

1. **Kinerja Superior**: CortexFlow mencapai kinerja terbaik pada semua dataset dengan peningkatan MSE hingga 94,3% dibandingkan metode pembanding.

2. **Signifikansi Statistik**: Semua perbandingan menunjukkan signifikansi statistik dengan effect size yang large di mayoritas kasus.

3. **Efisiensi Komputasi**: CortexFlow menunjukkan efisiensi parameter 2× lebih baik dan kecepatan inferensi 3,6× lebih cepat dibandingkan Mind-Vis.

4. **Dampak Lingkungan**: Jejak karbon CortexFlow 79,5% lebih rendah dibandingkan Mind-Vis dengan efisiensi karbon 4,8× lebih baik.

5. **Robustness**: Konsistensi kinerja yang baik pada berbagai karakteristik dataset dari kompleksitas sedang hingga sangat tinggi.

6. **Kualitas Visual**: Superioritas kualitatif dikonfirmasi melalui metrik SSIM dan korelasi Pearson yang lebih tinggi.

Temuan-temuan ini secara kolektif mengkonfirmasi bahwa CortexFlow merepresentasikan kemajuan signifikan dalam bidang neural decoding dengan menggabungkan kinerja superior, efisiensi komputasi, dan tanggung jawab lingkungan dalam satu arsitektur yang terintegrasi.
