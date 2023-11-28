[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/GpZPIs9c)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12877156&assignment_repo_type=AssignmentRepo)
# Graded Challenge 7 - Set 1

_Graded Challenge 7 ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Fulltime Program khususnya pada Phase 2 Week 1._

---

## Assignment Objectives

Graded Challenge 7 ini dibuat guna mengevaluasi konsep Computer Vision / Natural Language Processing pada pembelajaran Phase 2 sebagai berikut:

- Mampu memahami konsep Computer Vision/NLP.
- Mampu mempersiapkan data untuk digunakan dalam model Computer Vision/NLP.
- Mampu mengimplementasikan Artificial Neural Network dengan data yang dipilih.
- Mampu menganalisis dan menjelaskan performansi dari arsitektur Artificial Neural Network yang dibuat.

---

## Topik

Student dipersilakan memilih topik antara **Computer Vision atau Natural Language Processing** menggunakan dataset sendiri (scraping oleh student) atau menggunakan open dataset.

---

## Dataset

### Ketentuan Dataset
1. Pilihlah dataset yang paling nyaman digunakan karena tidak ada batasan untuk memilih dataset dalam mengerjakan *Graded Challenge 7*. 

2. **Konsultasikan terlebih dahulu dataset yang hendak digunakan ke buddy masing-masing student. Jika disetujui, maka silakan dikerjakan. Jika tidak disetujui, maka cari dataset yang lain dan konsultasikan lagi mengenai dataset yang baru ini.**

3. Student tidak boleh menggunakan dataset yang sudah dipakai dalam sesi pembelajaran saat dikelas bersama instruktur. Carilah dataset yang baru untuk tugas Graded Challenge ini.

4. **Student diizikan untuk melakukan scraping dataset**. Sertakan code mengenai scraping ini ke dalam repository.

5. Dataset tidak perlu diupload ke dalam repository. Sertakan link dataset pada file `url.txt`. Jika melakukan scraping, maka upload dataset ke dalam Google Drive dan share link mengenai dataset pada file `url.txt`. 

### Data Sources

Student dapat memilih dataset dari salah satu repository dibawah ini. Popular open data repositories :

- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle datasets](https://www.kaggle.com/datasets)
- [Amazon’s AWS datasets](https://registry.opendata.aws/)

Meta portals (they list open data repositories) :

- [Data Portals](http://dataportals.org/)
- [OpenDataMonitor](https://opendatamonitor.eu/frontend/web/index.php?r=dashboard%2Findex)
- [Quandl](https://www.quandl.com/)

Other pages listing many popular open data repositories :

- [Wikipedia’s list of Machine Learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
- [Quora.com](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)
- [The datasets subreddit](https://www.reddit.com/r/datasets)
- Sumber lain yang kredibel.

---

## Assignment Instructions

*Graded Challenge 7* dikerjakan dalam format *notebook* dan *model deployment* dengan beberapa *kriteria wajib* di bawah ini:

1. Deep Learning framework yang digunakan adalah *TensorFlow*.

2. Ada penggunaan library visualisasi, seperti *matplotlib*, *seaborn*, atau yang lain.

3. Isi *notebook* harus mengikuti *outline* di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, gambaran besar dataset yang digunakan, dan *objective* yang ingin dicapai.
   
   2. Import Libraries
      > *Cell* pertama pada *notebook* **harus berisi dan hanya berisi** semua *library* yang digunakan dalam *project*.
   
   3. Data Loading
      > Bagian ini berisi proses penyiapan data sebelum dilakukan eksplorasi data lebih lanjut. Proses Data Loading dapat berupa memberi nama baru untuk setiap kolom, mengecek ukuran dataset, dll.
   
   4. Exploratory Data Analysis (EDA)
      > Bagian ini berisi explorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.

   5. Feature Engineering
      > Bagian ini berisi proses penyiapan data untuk proses pelatihan model, seperti pembagian data menjadi train-val-test, transformasi data (normalisasi, encoding, dll.), dan proses-proses lain yang dibutuhkan.   
   
   6. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

   7. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.
   
   8. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis terkait dengan hasil pada model dan tuliskan hasil analisisnya**.

   9. Model Saving
      > Pada bagian ini, dilakukan proses penyimpanan model dan file-file lain yang terkait dengan hasil proses pembuatan model. Pilihlah 1 arsitektur ANN yang terbaik berdasarkan hasil evaluasi sebelumnya.
   
   10. Model Inference
       > Model yang sudah dilatih akan dicoba pada data yang bukan termasuk ke dalam train-set, val-set, ataupun test-set. Data ini harus dalam format yang asli, bukan data yang sudah di-scaled. **Model Inference harus berada pada notebook yang berbeda dari notebook yang dipakai untuk pembuatan model.**
   
   11. Pengambilan Kesimpulan
       > Pada bagian terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan *objective* yang sudah ditulis di bagian pengenalan.

5. *Notebook* harus diupload dalam akun GitHub Classroom masing-masing siswa untuk selanjutnya dinilai.

---

## Assignment Submission

- Simpan assignment pada sesi ini dengan nama `P2G7_<nama-student>.ipynb`, misal `P2G7_raka_ardhi.ipynb`.

- Push Assigment yang telah Anda buat ke akun Github Classroom Anda masing-masing.

- Untuk Model Deployment :
  * Buat sebuah folder bernama `deployment` dan masukkan semua file yang berkaitan dengan deployment ke folder ini.
  * Buat sebuah file bernama `url.txt` yang berisi URL Dataset dan URL deployment.
  * Contoh bentuk isi repository dengan Model Deployment.
    ```
    ├── deployment/
    │   ├── app.py
    │   ├── eda.py
    │   ├── prediction.py
    │   ├── model.h5
    │   ├── requirements.txt
    ├── P2G7_raka_ardhi.ipynb
    ├── P2G7_raka_ardhi_inference.ipynb
    ├── url.txt
    └── README.md
    ```

---

## Assignment Rubrics

### Code Review

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Feature Engineering | Mampu melakukan proses Feature Engineering sebelum melakukan proses modeling. Sertakan narasi mengenai alasan Feature Engineering tersebut dilakukan | 20 pts |
```
Penilaian terhadap code Feature Engineering : 10 points
Penilaian terhadap narasi dan keterhubungannya dengan domain yang dihadapi : 10 points
```

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| ANN Training | Membuat model Artificial Neural Network untuk menyelesaikan masalah Computer Vision atau NLP | 20 pts  |
| ANN Improvement | Membuat improvisasi model Artificial Neural Network | 20 pts  |
| Model Inference | Mencoba model yang telah dibuat dengan data baru | 10 pts |
| Runs Perfectly| Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar. | 10 pts |

```
Catatan mengenai ANN Training : 

1. Narasikan mengenai layer yang Anda buat beserta hyperparameter yang Anda tulis.
2. Sebaiknya lakukan training dengan lebih dari 20 epochs agar dapat terlihat performa model dengan lebih jelas.
3. Anda dipersilakan membuat arsitektur model dengan Sequential API/Functional API/keduanya.

Catatan mengenai ANN Improvement :
1. Narasikan mengenai apa yang Anda lakukan untuk memperbaiki performansi dari model-model yang tercipta pada ANN Training.
2. Sama seperti ANN Training, Anda dipersilakan membuat arsitektur model dengan Sequential API/Functional API/keduanya.
3. Arsitektur yang dibuat untuk rubric ANN Improvement akan dibandingkan dengan aristektur dari rubric ANN Training dan harus memiliki performansi yang lebih baik.
4. Contoh definisi performansi yang lebih baik : memiliki nilai accuracy atau metrics lain yang lebih baik, proses training lebih cepat meski nilai accuracy sama, nilai accuracy sama namun ukuran model lebih kecil, dll.
5. Model yang dipilih sebagai model terbaik haruslah model dari ANN Improvement, tidak boleh berasal dari ANN Training. Model terbaik ini akan digunakan pada Model Inference dan Model Deployment.
6. Anda diperkenankan menggunakan transfer learning. Ujilah beberapa arsitektur buatan Anda sendiri.
7. Anda dipersilakan menggunakan TensorFlow Callback.
```

### Readability

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Tertata Dengan Baik | Semua baris kode terdokumentasi dengan baik dengan menggunakan Markdown untuk penjelasan kode. | 10 pts |

```
Kriteria tertata dengan baik diantaranya adalah : 

1. Terdapat section Perkenalan yang jelas
2. Tidak menyalin markdown dari tugas lain.
3. Import library rapih (terdapat dalam 1 cell dan tidak ada unused libs).
4. Pemakaian fungsi markdown yang optimal (Heading, text formating, dll). 
5. Terdapat komentar pada setiap baris kode.
6. Adanya pemisah yang jelas antar section, dll.
7. Tidak adanya typo.
```

### Analysis

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Model Analysis | Menganalisa informasi dari model yang telah dibuat | 30 pts |
| Overall Analysis | Menarik informasi/kesimpulan dari keseluruhan kegiatan yang dilakukan | 20 pts |

```
Contoh kriteria analisa yang baik diantaranya adalah: 

1. Terdapat penjelasan macam-macam hasil metric evaluasi dan interpretasinya terhadap kasus yang diselesaikan.
2. Dapat menjelaskan kelemahan/kekurangan dan kelebihan dari model yang dibuat.
3. Dapat memberikan statement untuk improvement selanjutnya dari model yang dibuat. 
4. Sebutkan insight yang dapat diambil setelah proses EDA, dll.
```

### Deployment

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Model Deployment | Membuat webapps terhadap project yang telah dibuat. | 10 pts |

```
Catatan mengenai Model Deployment : 

1. Ketiadaan URL deployment ataupun source code deployment di repository, akan tetap diperhitungkan untuk menilai bagian Model Deployment. 
2. Tidak diperkenankan adanya informasi tambahan/informasi susulan seperti lupa memberikan URL deployment atau lupa mengupload source code via apapun (DM buddy, email, atau yang lain).
3. Student akan dianggap tidak melakukan Model Deployment jika tidak ada URL deployment dan source code deployment di repository.
4. Untuk hasil prediksi, ubah kembali menjadi label aslinya sesuai dengan dataset asalnya.
   - Contoh : saat melakukan modeling, Anda melakukan konversi target seperti 
     `car` menjadi `2`
     `bus` menjadi `1`
     `truck` menjadi `0`
   - Saat menampilkan hasil prediction, ubahlah label target ke dalam bentuk string (`car`/`bus`/`truck`).
```

---

```
Total Points : 150
```

---

## Notes

- **Deadline : P2W2D1 pukul 23:59 WIB.**

- **Keterlambatan pengumpulan tugas mengakibatkan skor Graded Challenge 7 menjadi 0.**
