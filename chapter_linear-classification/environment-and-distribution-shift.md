# Perubahan Lingkungan dan Distribusi
:label:`sec_environment-and-distribution-shift`

Pada bagian sebelumnya, kita telah melalui
beberapa aplikasi langsung dari machine learning,
memasang model untuk berbagai dataset.
Namun, kita belum sempat memikirkan
dari mana data berasal pada awalnya
atau apa yang pada akhirnya kita rencanakan
dengan output dari model kita.
Seringkali, pengembang machine learning
yang memiliki data segera mengembangkan model
tanpa berhenti untuk mempertimbangkan isu-isu mendasar ini.

Banyak kegagalan implementasi machine learning
dapat dilacak kembali ke kegagalan ini.
Kadang-kadang model tampaknya berkinerja sangat baik
saat diukur dengan akurasi set uji
tetapi gagal secara drastis saat diterapkan
ketika distribusi data tiba-tiba berubah.
Lebih berbahaya lagi, terkadang penerapan model itu sendiri
bisa menjadi pemicu yang mengganggu distribusi data.
Misalnya, jika kita melatih model
untuk memprediksi siapa yang akan membayar kembali pinjaman dan siapa yang akan gagal bayar,
dan menemukan bahwa pilihan alas kaki pemohon
berkaitan dengan risiko gagal bayar
(Oxford menunjukkan pembayaran, sepatu kets menunjukkan gagal bayar).
Kita mungkin tergoda untuk memberikan pinjaman
kepada semua pemohon yang memakai sepatu Oxford
dan menolak semua pemohon yang memakai sepatu kets.

Dalam kasus ini, lompatan yang kurang dipertimbangkan dari
pengenalan pola ke pengambilan keputusan
dan kegagalan kita dalam mempertimbangkan lingkungan secara kritis
dapat berakibat buruk.
Sebagai permulaan, begitu kita mulai
mengambil keputusan berdasarkan alas kaki,
pelanggan akan menyadarinya dan mengubah perilaku mereka.
Tak lama kemudian, semua pemohon akan memakai sepatu Oxford,
tanpa ada peningkatan yang bersamaan dalam kelayakan kredit mereka.
Luangkan waktu sejenak untuk merenungkannya karena masalah serupa banyak ditemukan
dalam berbagai aplikasi machine learning:
dengan memperkenalkan keputusan berbasis model kita ke lingkungan,
kita mungkin merusak model itu sendiri.

Meskipun kita tidak mungkin memberikan pembahasan lengkap
tentang topik ini dalam satu bagian,
tujuan kita di sini adalah mengungkap beberapa kekhawatiran umum,
dan mendorong pemikiran kritis
yang diperlukan untuk mendeteksi situasi semacam ini lebih awal,
mengurangi kerusakan, dan menggunakan machine learning secara bertanggung jawab.
Beberapa solusi sederhana
(meminta data yang "tepat"),
beberapa secara teknis sulit
(mengimplementasikan sistem reinforcement learning),
dan lainnya mengharuskan kita melangkah keluar dari ranah
prediksi statistik sepenuhnya dan
berjuang dengan pertanyaan filosofis yang sulit
mengenai aplikasi algoritma yang etis.

## Jenis Perubahan Distribusi

Untuk memulai, kita tetap menggunakan pengaturan prediksi pasif
dan mempertimbangkan berbagai cara distribusi data dapat berubah
serta apa yang dapat dilakukan untuk menyelamatkan kinerja model.
Dalam pengaturan klasik, kita mengasumsikan bahwa data pelatihan kita
diambil dari distribusi $p_S(\mathbf{x},y)$ tertentu
tetapi data uji kita akan terdiri dari
contoh tanpa label yang diambil dari
distribusi berbeda $p_T(\mathbf{x},y)$.
Sudah dari sini, kita harus menghadapi kenyataan yang mengejutkan.
Tanpa asumsi apa pun tentang bagaimana $p_S$
dan $p_T$ saling berhubungan,
mempelajari classifier yang kuat adalah hal yang mustahil.

Pertimbangkan masalah klasifikasi biner,
di mana kita ingin membedakan antara anjing dan kucing.
Jika distribusi dapat berubah dengan cara yang arbitrer,
maka pengaturan kita mengizinkan kasus patologis
di mana distribusi pada input tetap
konstan: $p_S(\mathbf{x}) = p_T(\mathbf{x})$,
tetapi semua label dibalik:
$p_S(y \mid \mathbf{x}) = 1 - p_T(y \mid \mathbf{x})$.
Dengan kata lain, jika tiba-tiba Tuhan memutuskan
bahwa semua "kucing" di masa depan sekarang adalah anjing
dan apa yang sebelumnya kita sebut sebagai "anjing" sekarang adalah kucing---tanpa
ada perubahan dalam distribusi input $p(\mathbf{x})$,
maka kita tidak mungkin membedakan pengaturan ini
dari pengaturan di mana distribusi tidak berubah sama sekali.

Untungnya, di bawah beberapa asumsi terbatas
tentang cara data kita dapat berubah di masa depan,
algoritma prinsipil dapat mendeteksi perubahan
dan terkadang bahkan beradaptasi secara langsung,
meningkatkan akurasi classifier asli.

### Covariate Shift

Di antara kategori perubahan distribusi,
covariate shift mungkin yang paling banyak dipelajari.
Di sini, kita mengasumsikan bahwa meskipun distribusi input
dapat berubah dari waktu ke waktu, fungsi pelabelan,
yaitu distribusi kondisional
$P(y \mid \mathbf{x})$, tidak berubah.
Statistikawan menyebut ini sebagai *covariate shift*
karena masalah ini muncul akibat pergeseran
dalam distribusi kovariat (fitur).
Meskipun kita terkadang bisa memahami perubahan distribusi
tanpa melibatkan kausalitas, perlu dicatat bahwa covariate shift
adalah asumsi alami yang digunakan dalam pengaturan
di mana kita percaya bahwa $\mathbf{x}$ menyebabkan $y$.

Pertimbangkan tantangan dalam membedakan kucing dan anjing.
Data pelatihan kita mungkin terdiri dari gambar-gambar seperti yang ditunjukkan pada :numref:`fig_cat-dog-train`.

![Data pelatihan untuk membedakan kucing dan anjing (ilustrasi: Lafeez Hossain / 500px / Getty Images; ilkermetinkursova / iStock / Getty Images Plus; GlobalP / iStock / Getty Images Plus; Musthafa Aboobakuru / 500px / Getty Images).](../img/cat-dog-train.png)
:label:`fig_cat-dog-train`


Saat uji, kita diminta mengklasifikasikan gambar-gambar pada :numref:`fig_cat-dog-test`.

![Data uji untuk membedakan kucing dan anjing (ilustrasi: SIBAS_minich / iStock / Getty Images Plus; Ghrzuzudu / iStock / Getty Images Plus; id-work / DigitalVision Vectors / Getty Images; Yime / iStock / Getty Images Plus).](../img/cat-dog-test.png)
:label:`fig_cat-dog-test`

Set pelatihan terdiri dari foto-foto,
sementara set uji hanya berisi kartun.
Melatih model pada dataset dengan karakteristik yang sangat berbeda
dari set uji dapat menimbulkan masalah jika tidak ada rencana yang jelas
untuk beradaptasi dengan domain baru ini.

### Perubahan Label

*Label shift* menggambarkan masalah yang berlawanan.
Di sini, kita mengasumsikan bahwa label marginal $P(y)$
dapat berubah
tetapi distribusi kondisional kelas
$P(\mathbf{x} \mid y)$ tetap sama di berbagai domain.
Label shift adalah asumsi yang masuk akal untuk dibuat
ketika kita percaya bahwa $y$ menyebabkan $\mathbf{x}$.
Misalnya, kita mungkin ingin memprediksi diagnosis
berdasarkan gejala (atau manifestasi lainnya),
meskipun prevalensi relatif diagnosis
berubah dari waktu ke waktu.
Label shift adalah asumsi yang tepat di sini
karena penyakit menyebabkan gejala.
Dalam beberapa kasus degeneratif, asumsi label shift
dan covariate shift dapat berlaku bersamaan.
Misalnya, ketika label bersifat deterministik,
asumsi covariate shift akan terpenuhi,
bahkan ketika $y$ menyebabkan $\mathbf{x}$.
Menariknya, dalam kasus-kasus ini,
seringkali lebih menguntungkan untuk bekerja dengan metode
yang berasal dari asumsi label shift.
Ini karena metode ini cenderung
melibatkan manipulasi objek yang tampak seperti label (seringkali berdimensi rendah),
berlawanan dengan objek yang tampak seperti input,
yang cenderung berdimensi tinggi dalam deep learning.

### Perubahan Konsep

Kita juga mungkin menemui masalah terkait yaitu *concept shift*,
yang muncul ketika definisi label itu sendiri dapat berubah.
Ini terdengar aneh---seekor *kucing* adalah *kucing*, bukan?
Namun, kategori lain bisa mengalami perubahan penggunaan seiring waktu.
Kriteria diagnostik untuk penyakit mental,
apa yang dianggap modis, dan gelar pekerjaan,
semuanya tunduk pada
perubahan konsep yang cukup besar.
Jika kita berkeliling Amerika Serikat,
mengubah sumber data kita berdasarkan geografi,
kita akan menemukan perubahan konsep yang signifikan terkait
penyebutan *soft drink*
seperti ditunjukkan pada :numref:`fig_popvssoda`.

![Perubahan konsep untuk nama minuman ringan di Amerika Serikat (CC-BY: Alan McConchie, PopVsSoda.com).](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Jika kita ingin membangun sistem terjemahan mesin,
distribusi $P(y \mid \mathbf{x})$ mungkin berbeda
tergantung pada lokasi kita.
Masalah ini bisa sulit terdeteksi.
Kita mungkin berharap untuk memanfaatkan pengetahuan
bahwa perubahan hanya terjadi secara bertahap
baik dalam arti temporal maupun geografis.


## Contoh Perubahan Distribusi

Sebelum mendalami formalisme dan algoritma,
kita dapat membahas beberapa situasi konkret
di mana covariate shift atau concept shift mungkin tidak tampak jelas.


### Diagnostik Medis

Bayangkan Anda ingin merancang algoritma untuk mendeteksi kanker.
Anda mengumpulkan data dari orang sehat dan orang sakit
lalu melatih algoritma Anda.
Algoritma ini bekerja dengan baik, menghasilkan akurasi tinggi
dan Anda menyimpulkan bahwa Anda siap
untuk karier yang sukses dalam diagnostik medis.
*Tunggu dulu.*

Distribusi yang menghasilkan data pelatihan
dan yang akan Anda temui di lapangan bisa berbeda jauh.
Ini terjadi pada sebuah startup yang kurang beruntung
yang beberapa dari kami penulis pernah bekerja sama bertahun-tahun lalu.
Mereka sedang mengembangkan tes darah untuk penyakit
yang terutama menyerang pria yang lebih tua
dan berharap untuk menelitinya menggunakan sampel darah
yang mereka kumpulkan dari pasien.
Namun, ternyata jauh lebih sulit
untuk mendapatkan sampel darah dari pria sehat
daripada dari pasien sakit yang sudah berada dalam sistem.
Sebagai kompensasi, startup tersebut meminta
sumbangan darah dari mahasiswa di kampus universitas
untuk dijadikan sebagai kontrol sehat dalam pengembangan tes mereka.
Kemudian mereka bertanya apakah kami bisa membantu mereka
membangun classifier untuk mendeteksi penyakit tersebut.

Seperti yang kami jelaskan kepada mereka,
akan sangat mudah membedakan
antara kohort sehat dan sakit
dengan akurasi mendekati sempurna.
Namun, ini karena subjek uji
berbeda dalam hal usia, tingkat hormon,
aktivitas fisik, diet, konsumsi alkohol,
dan banyak faktor lain yang tidak terkait dengan penyakit tersebut.
Hal ini tidak mungkin terjadi pada pasien sesungguhnya.
Karena prosedur pengambilan sampel mereka,
kita bisa berharap mengalami covariate shift yang ekstrem.
Selain itu, kasus ini kemungkinan besar tidak dapat
diperbaiki melalui metode konvensional.
Singkatnya, mereka menyia-nyiakan sejumlah besar uang.


### Mobil _Self Driving_

Misalkan sebuah perusahaan ingin memanfaatkan machine learning
untuk mengembangkan mobil _Self Driving_.
Salah satu komponen kunci di sini adalah pendeteksi pinggir jalan.
Karena data beranotasi asli sangat mahal untuk didapatkan,
mereka memiliki ide (cerdas namun dipertanyakan)
untuk menggunakan data sintetis dari mesin render permainan
sebagai data pelatihan tambahan.
Ini bekerja dengan sangat baik pada "data uji"
yang diambil dari mesin render tersebut.
Sayangnya, di dalam mobil sungguhan hasilnya sangat buruk.
Ternyata, pinggir jalan telah dirender
dengan tekstur yang sangat sederhana.
Yang lebih penting, *semua* pinggir jalan dirender
dengan tekstur yang *sama* dan pendeteksi pinggir jalan
mempelajari "fitur" ini dengan sangat cepat.

Hal serupa juga terjadi pada Angkatan Darat AS
ketika mereka pertama kali mencoba mendeteksi tank di hutan.
Mereka mengambil foto udara hutan tanpa tank,
kemudian menggerakkan tank ke dalam hutan
dan mengambil satu set foto lagi.
Classifier tampaknya bekerja dengan *sempurna*.
Sayangnya, classifier ini hanya belajar
cara membedakan pohon dengan bayangan
dari pohon tanpa bayangan---set foto pertama
diambil pada pagi hari,
set kedua pada siang hari.


### Distribusi Non-Stasioner

Situasi yang jauh lebih halus muncul
ketika distribusi berubah perlahan-lahan
(juga dikenal sebagai *distribusi non-stasioner*)
dan model tidak diperbarui dengan memadai.
Berikut adalah beberapa kasus tipikal.

* Kita melatih model periklanan komputasi lalu gagal memperbaruinya secara berkala (misalnya, kita lupa memperhitungkan bahwa ada perangkat baru bernama iPad yang baru saja diluncurkan).
* Kita membangun filter spam. Filter ini bekerja dengan baik dalam mendeteksi semua spam yang pernah kita lihat sejauh ini. Tetapi kemudian para pengirim spam menjadi lebih cerdik dan membuat pesan baru yang terlihat berbeda dari apa pun yang pernah kita lihat.
* Kita membangun sistem rekomendasi produk. Ini bekerja sepanjang musim dingin tetapi terus merekomendasikan topi Sinterklas lama setelah Natal berakhir.

### Anecdot Lainnya

* Kita membangun pendeteksi wajah. Ini bekerja dengan baik pada semua benchmark. Sayangnya, gagal pada data uji---contoh yang gagal adalah close-up di mana wajah memenuhi seluruh gambar (tidak ada data semacam ini dalam set pelatihan).
* Kita membangun mesin pencari web untuk pasar AS dan ingin menerapkannya di Inggris.
* Kita melatih classifier gambar dengan mengumpulkan dataset besar di mana setiap kelas dari sekumpulan besar kelas terwakili secara setara dalam dataset, misalnya 1000 kategori, masing-masing diwakili oleh 1000 gambar. Kemudian kita menerapkan sistem tersebut di dunia nyata, di mana distribusi label sebenarnya pada foto jelas tidak seragam.


## Koreksi Perubahan Distribusi

Seperti yang telah kita bahas, ada banyak kasus
di mana distribusi pelatihan dan uji
$P(\mathbf{x}, y)$ berbeda.
Dalam beberapa kasus, kita beruntung dan model bekerja
meskipun terjadi covariate, label, atau concept shift.
Dalam kasus lain, kita bisa mendapatkan hasil yang lebih baik dengan menerapkan
strategi-strategi yang tepat untuk menghadapi pergeseran tersebut.
Bagian selanjutnya akan semakin teknis.
Pembaca yang tidak sabar dapat melanjutkan ke bagian berikutnya
karena materi ini tidak menjadi prasyarat untuk konsep selanjutnya.


### Risiko Empiris dan Risiko
:label:`subsec_empirical-risk-and-risk`

Mari kita renungkan terlebih dahulu apa yang sebenarnya terjadi selama pelatihan model:
kita mengiterasi fitur dan label terkait
dari data pelatihan
$\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$
dan memperbarui parameter model $f$ setelah setiap minibatch.
Untuk menyederhanakan, kita tidak mempertimbangkan regularisasi,
sehingga kita terutama meminimalkan loss pada data pelatihan:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

di mana $l$ adalah fungsi loss
yang mengukur "seberapa buruk" prediksi $f(\mathbf{x}_i)$ diberikan label terkait $y_i$.
Statistikawan menyebut istilah pada :eqref:`eq_empirical-risk-min` sebagai *risiko empiris*.
*Risiko empiris* adalah rata-rata loss pada data pelatihan
untuk mendekati *risiko*,
yang merupakan ekspektasi dari loss pada seluruh populasi data yang diambil dari distribusi sebenarnya
$p(\mathbf{x},y)$:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

Namun, dalam praktiknya kita biasanya tidak dapat memperoleh seluruh populasi data.
Oleh karena itu, *minimisasi risiko empiris*,
yang meminimalkan risiko empiris dalam :eqref:`eq_empirical-risk-min`,
adalah strategi praktis dalam machine learning,
dengan harapan mendekati
minimisasi risiko.



### Koreksi Covariate Shift
:label:`subsec_covariate-shift-correction`

Asumsikan bahwa kita ingin memperkirakan
ketergantungan $P(y \mid \mathbf{x})$
untuk mana kita memiliki data berlabel $(\mathbf{x}_i, y_i)$.
Sayangnya, pengamatan $\mathbf{x}_i$ diambil
dari *distribusi sumber* $q(\mathbf{x})$
bukan dari *distribusi target* $p(\mathbf{x})$.
Untungnya,
asumsi ketergantungan berarti
bahwa distribusi kondisional tidak berubah: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$.
Jika distribusi sumber $q(\mathbf{x})$ "salah",
kita dapat memperbaikinya dengan menggunakan identitas sederhana berikut dalam risiko:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Dengan kata lain, kita perlu menimbang ulang setiap contoh data
dengan rasio probabilitas
bahwa data tersebut diambil dari distribusi yang benar dibandingkan dengan yang salah:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Dengan memasukkan bobot $\beta_i$ untuk
setiap contoh data $(\mathbf{x}_i, y_i)$
kita dapat melatih model kita menggunakan
*minimisasi risiko empiris berbobot*:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`

Sayangnya, kita tidak tahu rasio tersebut,
jadi sebelum kita dapat melakukan sesuatu yang berguna kita perlu memperkirakannya.
Banyak metode tersedia,
termasuk beberapa pendekatan operator-teoretis canggih
yang mencoba mengkalibrasi ulang operator ekspektasi secara langsung
menggunakan prinsip minimum-norm atau maksimum entropi.
Perhatikan bahwa untuk pendekatan semacam itu, kita memerlukan sampel
yang diambil dari kedua distribusi---"benar" $p$, misalnya,
melalui akses ke data uji, dan distribusi yang digunakan
untuk menghasilkan set pelatihan $q$ (yang terakhir mudah didapatkan).
Namun, perhatikan bahwa kita hanya memerlukan fitur $\mathbf{x} \sim p(\mathbf{x})$;
kita tidak perlu mengakses label $y \sim p(y)$.

Dalam kasus ini, ada pendekatan yang sangat efektif
yang akan memberikan hasil yang hampir sama baiknya dengan aslinya: yaitu, regresi logistik,
yang merupakan kasus khusus dari regresi softmax (lihat :numref:`sec_softmax`)
untuk klasifikasi biner.
Ini adalah semua yang diperlukan untuk menghitung perkiraan rasio probabilitas.
Kita melatih classifier untuk membedakan
antara data yang diambil dari $p(\mathbf{x})$
dan data yang diambil dari $q(\mathbf{x})$.
Jika tidak mungkin membedakan
antara kedua distribusi
maka itu berarti instance terkait
sama-sama mungkin berasal dari
salah satu dari dua distribusi tersebut.
Di sisi lain, instance apa pun
yang dapat dibedakan dengan baik
harus diberi bobot lebih atau kurang sesuai.

Untuk penyederhanaan, asumsikan bahwa kita memiliki
jumlah instance yang sama dari kedua distribusi
$p(\mathbf{x})$
dan $q(\mathbf{x})$.
Sekarang berikan label $z$ sebagai $1$
untuk data yang diambil dari $p$ dan $-1$ untuk data yang diambil dari $q$.
Kemudian probabilitas dalam dataset campuran diberikan oleh

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \textrm{ dan oleh karena itu } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Dengan demikian, jika kita menggunakan pendekatan regresi logistik,
di mana $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ adalah fungsi parameter),
maka

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

Hasilnya, kita perlu menyelesaikan dua masalah:
pertama, membedakan antara
data yang diambil dari kedua distribusi,
lalu masalah minimisasi risiko empiris berbobot
dalam :eqref:`eq_weighted-empirical-risk-min`
di mana kita menimbang suku dengan $\beta_i$.

Sekarang kita siap untuk menjelaskan algoritma koreksi.
Misalkan kita memiliki set pelatihan $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ dan set uji tanpa label $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$.
Untuk covariate shift,
kita mengasumsikan bahwa $\mathbf{x}_i$ untuk semua $1 \leq i \leq n$ diambil dari beberapa distribusi sumber
dan $\mathbf{u}_i$ untuk semua $1 \leq i \leq m$
diambil dari distribusi target.
Berikut ini adalah algoritma prototipikal
untuk memperbaiki covariate shift:

1. Buat set pelatihan klasifikasi biner: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
2. Latih classifier biner menggunakan regresi logistik untuk mendapatkan fungsi $h$.
3. Beri bobot data pelatihan menggunakan $\beta_i = \exp(h(\mathbf{x}_i))$ atau lebih baik $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ untuk beberapa konstanta $c$.
4. Gunakan bobot $\beta_i$ untuk melatih pada $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ dalam :eqref:`eq_weighted-empirical-risk-min`.

Perhatikan bahwa algoritma di atas bergantung pada asumsi penting.
Agar skema ini berfungsi, kita memerlukan bahwa setiap contoh data
dalam distribusi target (misalnya, pada saat uji)
memiliki probabilitas tidak nol untuk terjadi pada saat pelatihan.
Jika kita menemukan titik di mana $p(\mathbf{x}) > 0$ tetapi $q(\mathbf{x}) = 0$,
maka bobot penting yang sesuai harus tak hingga.


### Koreksi Label Shift

Asumsikan bahwa kita sedang mengerjakan tugas klasifikasi dengan $k$ kategori.
Dengan menggunakan notasi yang sama di :numref:`subsec_covariate-shift-correction`,
$q$ dan $p$ masing-masing adalah distribusi sumber (misalnya, saat pelatihan) dan distribusi target (misalnya, saat uji).
Asumsikan bahwa distribusi label berubah seiring waktu:
$q(y) \neq p(y)$, tetapi distribusi kondisional kelas
tetap sama: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$.
Jika distribusi sumber $q(y)$ "salah",
kita dapat memperbaikinya
berdasarkan
identitas berikut dalam risiko
seperti yang didefinisikan dalam
:eqref:`eq_true-risk`:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$

Di sini, bobot penting kita akan sesuai dengan
rasio likelihood label:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

Salah satu hal yang menarik tentang label shift adalah bahwa
jika kita memiliki model yang cukup baik
pada distribusi sumber,
maka kita dapat memperkirakan bobot ini secara konsisten
tanpa harus berurusan dengan dimensi tinggi.
Dalam deep learning, input cenderung
berupa objek berdimensi tinggi seperti gambar,
sementara label sering kali objek yang lebih sederhana seperti kategori.

Untuk memperkirakan distribusi label target,
pertama-tama kita mengambil classifier yang cukup baik
(yang biasanya dilatih pada data pelatihan)
dan menghitung matriks "confusion" atau kebingungan menggunakan set validasi
(juga dari distribusi pelatihan).
*Matriks kebingungan*, $\mathbf{C}$, hanyalah sebuah matriks $k \times k$,
di mana setiap kolom sesuai dengan kategori label (ground truth)
dan setiap baris sesuai dengan kategori prediksi model kita.
Nilai setiap sel $c_{ij}$ adalah fraksi dari total prediksi pada set validasi
di mana label sebenarnya adalah $j$ dan model kita memprediksi $i$.

Sekarang, kita tidak dapat menghitung matriks kebingungan
langsung pada data target
karena kita tidak dapat melihat label untuk contoh-contoh
yang kita lihat di lapangan,
kecuali kita menginvestasikan pipeline anotasi real-time yang kompleks.
Namun, yang bisa kita lakukan adalah merata-ratakan semua prediksi model kita
pada saat uji, menghasilkan output rata-rata model $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$,
di mana elemen $i^\textrm{th}$ $\mu(\hat{y}_i)$
adalah fraksi dari total prediksi pada set uji
di mana model kita memprediksi $i$.

Ternyata, di bawah kondisi tertentu---jika
classifier kita cukup akurat sejak awal,
dan jika data target hanya berisi kategori
yang telah kita lihat sebelumnya,
dan jika asumsi label shift berlaku sejak awal
(asumsi terkuat di sini)---kita dapat memperkirakan distribusi label set uji
dengan menyelesaikan sistem linier sederhana

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

karena sebagai estimasi $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ berlaku untuk semua $1 \leq i \leq k$,
di mana $p(y_j)$ adalah elemen ke-$j$ dari vektor distribusi label berdimensi $k$, $p(\mathbf{y})$.
Jika classifier kita cukup akurat sejak awal,
maka matriks kebingungan $\mathbf{C}$ akan memiliki invers,
dan kita mendapatkan solusi $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.

Karena kita mengamati label pada data sumber,
sangat mudah untuk memperkirakan distribusi $q(y)$.
Kemudian, untuk setiap contoh pelatihan $i$ dengan label $y_i$,
kita dapat mengambil rasio perkiraan kita $p(y_i)/q(y_i)$
untuk menghitung bobot $\beta_i$,
dan memasukkannya ke dalam minimisasi risiko empiris berbobot
dalam :eqref:`eq_weighted-empirical-risk-min`.



### Koreksi Concept Shift

Concept shift jauh lebih sulit untuk diperbaiki dengan cara yang prinsipil.
Misalnya, dalam situasi di mana tiba-tiba masalah berubah
dari membedakan kucing dari anjing menjadi
membedakan hewan putih dari hewan hitam,
akan tidak masuk akal untuk mengasumsikan
bahwa kita bisa melakukan yang lebih baik daripada hanya mengumpulkan label baru
dan melatih ulang dari awal.
Untungnya, dalam praktiknya, perubahan ekstrem semacam ini jarang terjadi.
Sebaliknya, yang biasanya terjadi adalah tugas tersebut berubah secara perlahan.
Untuk membuatnya lebih konkret, berikut adalah beberapa contoh:

* Dalam periklanan komputasi, produk baru diluncurkan,
produk lama menjadi kurang populer. Ini berarti distribusi iklan dan popularitasnya berubah secara bertahap dan prediktor click-through rate mana pun perlu mengikuti perubahan tersebut secara bertahap.
* Lensa kamera lalu lintas terdegradasi secara bertahap karena paparan lingkungan, memengaruhi kualitas gambar seiring waktu.
* Konten berita berubah secara bertahap (misalnya, sebagian besar berita tetap tidak berubah tetapi berita baru muncul).

Dalam kasus seperti ini, kita dapat menggunakan pendekatan yang sama yang kita gunakan untuk melatih jaringan agar dapat beradaptasi dengan perubahan data. Dengan kata lain, kita menggunakan bobot jaringan yang ada dan cukup melakukan beberapa langkah pembaruan dengan data baru alih-alih melatih dari awal.


## Taksonomi Masalah Pembelajaran

Dengan pengetahuan tentang cara menangani perubahan dalam distribusi, kita sekarang dapat mempertimbangkan beberapa aspek lain dalam perumusan masalah machine learning.


### Pembelajaran Batch

Dalam *batch learning*, kita memiliki akses ke fitur pelatihan dan label $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, yang kita gunakan untuk melatih model $f(\mathbf{x})$. Nantinya, kita menerapkan model ini untuk menilai data baru $(\mathbf{x}, y)$ yang diambil dari distribusi yang sama. Ini adalah asumsi default untuk masalah apa pun yang kita bahas di sini. Misalnya, kita mungkin melatih detektor kucing berdasarkan banyak gambar kucing dan anjing. Setelah kita melatihnya, kita mengirimkannya sebagai bagian dari sistem visi komputer smart catdoor yang hanya membiarkan kucing masuk. Sistem ini kemudian dipasang di rumah pelanggan dan tidak pernah diperbarui lagi (kecuali dalam keadaan ekstrem).


### Pembelajaran Online (_Online Learning_)

Sekarang bayangkan bahwa data $(\mathbf{x}_i, y_i)$ tiba satu per satu. Lebih spesifiknya, asumsikan bahwa kita pertama-tama mengamati $\mathbf{x}_i$, kemudian kita perlu menghasilkan perkiraan $f(\mathbf{x}_i)$. Hanya setelah kita melakukan ini, kita mengamati $y_i$ dan menerima keuntungan atau mengalami kerugian, tergantung pada keputusan kita.
Banyak masalah nyata jatuh ke dalam kategori ini. Misalnya, kita perlu memprediksi harga saham besok, yang memungkinkan kita untuk melakukan perdagangan berdasarkan perkiraan tersebut dan pada akhir hari kita mengetahui apakah perkiraan kita menghasilkan keuntungan. Dengan kata lain, dalam *pembelajaran online*, kita memiliki siklus berikut di mana kita terus meningkatkan model kita berdasarkan pengamatan baru:

$$\begin{aligned}&\textrm{model } f_t \longrightarrow \textrm{data }  \mathbf{x}_t \longrightarrow \textrm{perkiraan } f_t(\mathbf{x}_t) \longrightarrow\\ \textrm{pengamat}&\textrm{an } y_t \longrightarrow \textrm{loss } l(y_t, f_t(\mathbf{x}_t)) \longrightarrow \textrm{model } f_{t+1}\end{aligned}$$

### Bandits

*Bandit* adalah kasus khusus dari masalah di atas. Sementara dalam sebagian besar masalah pembelajaran kita memiliki fungsi yang terus diparametrisasi $f$ yang ingin kita pelajari parameternya (misalnya, jaringan deep), dalam masalah *bandit* kita hanya memiliki sejumlah lengan yang dapat kita tarik, yaitu, sejumlah tindakan yang dapat kita ambil. Tidak mengherankan bahwa untuk masalah yang lebih sederhana ini jaminan teoretis yang lebih kuat dalam hal optimalitas dapat diperoleh. Kami mencantumkannya terutama karena masalah ini sering (secara membingungkan) diperlakukan seolah-olah itu adalah pengaturan pembelajaran yang berbeda.


### Kontrol

Dalam banyak kasus lingkungan "mengingat" apa yang kita lakukan. Tidak selalu dalam cara yang melawan tetapi hanya akan mengingat dan responsnya bergantung pada apa yang terjadi sebelumnya. Misalnya, pengontrol ketel kopi akan mengamati suhu yang berbeda tergantung pada apakah sebelumnya memanaskan ketel. Algoritma pengontrol PID (proportional-integral-derivative) adalah pilihan populer di sini.
Demikian pula, perilaku pengguna di situs berita akan bergantung pada apa yang kami tunjukkan sebelumnya (misalnya, mereka hanya akan membaca sebagian besar berita sekali saja). Banyak algoritma semacam ini membentuk model dari lingkungan tempat mereka bertindak sehingga keputusan mereka tampak kurang acak.
Baru-baru ini,
teori kontrol (misalnya, varian PID) juga telah digunakan
untuk secara otomatis menyetel hyperparameter
untuk mencapai pemisahan dan rekonstruksi yang lebih baik,
dan meningkatkan keragaman teks yang dihasilkan serta kualitas rekonstruksi gambar yang dihasilkan :cite:`Shao.Yao.Sun.ea.2020`.





### Reinforcement Learning

Dalam kasus lingkungan dengan memori yang lebih umum, kita mungkin menghadapi situasi di mana lingkungan berusaha bekerja sama dengan kita (permainan kooperatif, terutama untuk permainan non-zero-sum), atau lainnya di mana lingkungan akan mencoba untuk menang. Catur, Go, Backgammon, atau StarCraft adalah beberapa contoh dalam *reinforcement learning*. Demikian pula, kita mungkin ingin membangun pengontrol yang baik untuk mobil otonom. Mobil lain kemungkinan akan merespons gaya berkendara mobil otonom dengan cara yang tidak sepele, misalnya, mencoba menghindarinya, mencoba menyebabkan kecelakaan, atau mencoba bekerja sama dengannya.

### Mempertimbangkan Lingkungan

Salah satu perbedaan utama antara berbagai situasi di atas adalah bahwa strategi yang mungkin berhasil di lingkungan stasioner mungkin tidak berhasil di lingkungan yang dapat beradaptasi. Misalnya, peluang arbitrase yang ditemukan oleh seorang pedagang kemungkinan akan menghilang setelah dieksploitasi. Kecepatan dan cara lingkungan berubah sangat menentukan jenis algoritma yang dapat kita gunakan. Misalnya, jika kita tahu bahwa sesuatu hanya bisa berubah perlahan, kita bisa memaksa setiap perkiraan untuk berubah perlahan juga. Jika kita tahu bahwa lingkungan dapat berubah secara instan, tetapi hanya jarang terjadi, kita bisa membuat pengecualian untuk hal tersebut. Jenis pengetahuan ini sangat penting bagi data scientist yang ingin menangani concept shift, yaitu ketika masalah yang sedang dipecahkan dapat berubah dari waktu ke waktu.


## Keadilan, Akuntabilitas, dan Transparansi dalam Machine Learning

Akhirnya, penting untuk diingat
bahwa ketika Anda menerapkan sistem machine learning
Anda tidak hanya mengoptimalkan model prediktif---
Anda biasanya menyediakan alat yang akan
digunakan untuk (sebagian atau sepenuhnya) mengotomatiskan keputusan.
Sistem teknis ini dapat memengaruhi kehidupan
individu yang tunduk pada keputusan yang dihasilkan.
Langkah dari mempertimbangkan prediksi hingga membuat keputusan
tidak hanya menimbulkan pertanyaan teknis baru,
tetapi juga sejumlah pertanyaan etis
yang harus dipertimbangkan dengan hati-hati.
Jika kita menerapkan sistem diagnostik medis,
kita perlu mengetahui untuk populasi mana
sistem ini mungkin bekerja dan untuk populasi mana mungkin tidak.
Mengabaikan risiko yang dapat diperkirakan terhadap kesejahteraan
suatu subpopulasi dapat menyebabkan kita memberikan perawatan yang lebih rendah.
Selain itu, setelah kita memikirkan sistem pengambilan keputusan,
kita harus mundur dan mempertimbangkan kembali bagaimana kita mengevaluasi teknologi kita.
Di antara konsekuensi lain dari perubahan lingkup ini,
kita akan menemukan bahwa *akurasi* jarang menjadi ukuran yang tepat.
Misalnya, saat menerjemahkan prediksi menjadi tindakan,
kita sering kali ingin mempertimbangkan
sensitivitas biaya dari kesalahan dalam berbagai cara.
Jika salah satu cara salah klasifikasi sebuah gambar
dapat dianggap sebagai penghinaan rasial,
sementara salah klasifikasi ke kategori lain
tidak berdampak, maka kita mungkin ingin menyesuaikan
ambang batas kita dengan mempertimbangkan nilai-nilai sosial
dalam merancang protokol pengambilan keputusan.
Kita juga ingin berhati-hati tentang
bagaimana sistem prediksi dapat menyebabkan loop umpan balik.
Misalnya, pertimbangkan sistem kepolisian prediktif,
yang menugaskan petugas patroli
ke area dengan perkiraan kejahatan tinggi.
Mudah untuk melihat bagaimana pola yang mengkhawatirkan bisa muncul:

 1. Lingkungan dengan lebih banyak kejahatan mendapatkan lebih banyak patroli.
 2. Akibatnya, lebih banyak kejahatan ditemukan di lingkungan ini, yang masuk dalam data pelatihan untuk iterasi mendatang.
 3. Terpapar lebih banyak kasus positif, model memprediksi lebih banyak kejahatan di lingkungan ini.
 4. Pada iterasi berikutnya, model yang diperbarui semakin menargetkan lingkungan yang sama sehingga lebih banyak kejahatan ditemukan, dan seterusnya.

Seringkali, berbagai mekanisme di mana
prediksi model menjadi terhubung dengan data pelatihannya
tidak diperhitungkan dalam proses pemodelan.
Hal ini dapat menyebabkan apa yang oleh para peneliti disebut *runaway feedback loops*.
Selain itu, kita ingin berhati-hati tentang
apakah kita benar-benar menyelesaikan masalah yang tepat.
Algoritma prediktif sekarang memainkan peran besar
dalam memediasi penyebaran informasi.
Apakah berita yang dihadapi seseorang
harus ditentukan oleh kumpulan halaman Facebook yang mereka *Sukai*?
Ini hanyalah beberapa di antara banyak dilema etis mendesak
yang mungkin Anda temui dalam karier di bidang machine learning.


## Ringkasan

Dalam banyak kasus, set pelatihan dan uji tidak berasal dari distribusi yang sama. Hal ini disebut *distribution shift*.
Risiko adalah ekspektasi dari loss pada seluruh populasi data yang diambil dari distribusi sebenarnya. Namun, populasi ini biasanya tidak tersedia. Risiko empiris adalah rata-rata loss pada data pelatihan untuk mendekati risiko. Dalam praktiknya, kita melakukan minimisasi risiko empiris.

Di bawah asumsi yang sesuai, *covariate shift* dan *label shift* dapat dideteksi dan diperbaiki pada saat uji. Kegagalan untuk memperhitungkan bias ini dapat menjadi masalah pada saat uji.
Dalam beberapa kasus, lingkungan dapat mengingat tindakan otomatis dan merespons dengan cara yang mengejutkan. Kita harus mempertimbangkan kemungkinan ini saat membangun model dan terus memantau sistem yang berjalan, terbuka terhadap kemungkinan bahwa model kita dan lingkungan akan terhubung dengan cara yang tidak terduga.

## Latihan

1. Apa yang bisa terjadi ketika kita mengubah perilaku mesin pencari? Apa yang mungkin dilakukan oleh pengguna? Bagaimana dengan pengiklan?
2. Implementasikan detektor *covariate shift*. Petunjuk: bangun sebuah *classifier*.
3. Implementasikan korektor *covariate shift*.
4. Selain perubahan distribusi, apa lagi yang dapat memengaruhi bagaimana risiko empiris mendekati risiko?


[Diskusi](https://discuss.d2l.ai/t/105)
