# Regresi Softmax
:label:`sec_softmax`

Pada :numref:`sec_linear_regression`, kita telah memperkenalkan regresi linear,
dengan implementasi dari awal di :numref:`sec_linear_scratch`
dan menggunakan API tingkat tinggi dari framework deep learning
di :numref:`sec_linear_concise` untuk menangani bagian yang rumit.

Regresi adalah alat yang kita gunakan ketika ingin menjawab pertanyaan *berapa banyak?* atau *seberapa banyak?*.
Jika Anda ingin memprediksi jumlah dolar (harga)
sebuah rumah akan dijual,
atau jumlah kemenangan yang mungkin diperoleh sebuah tim baseball,
atau jumlah hari seorang pasien
akan tetap dirawat sebelum dipulangkan,
maka Anda mungkin sedang mencari model regresi.
Namun, bahkan dalam model regresi,
terdapat perbedaan penting.
Misalnya, harga rumah
tidak akan pernah negatif dan perubahannya sering kali *relatif* terhadap harga dasarnya.
Oleh karena itu, mungkin lebih efektif untuk melakukan regresi
pada logaritma harga.
Demikian juga, jumlah hari seorang pasien di rumah sakit
adalah variabel acak *diskret non-negatif*.
Karena itu, metode least mean squares mungkin tidak menjadi pendekatan yang ideal.
Pemodelan seperti ini, yang berkaitan dengan *time-to-event*,
memiliki banyak aspek rumit yang ditangani dalam subbidang khusus yang disebut *pemodelan survival*.

Tujuannya di sini bukan untuk membingungkan Anda, melainkan
untuk memberi tahu bahwa ada banyak hal dalam estimasi
selain sekadar meminimalkan kesalahan kuadrat.
Lebih luas lagi, ada lebih banyak hal dalam pembelajaran terawasi selain regresi.
Pada bagian ini, kita fokus pada masalah *klasifikasi*
di mana kita mengesampingkan pertanyaan *berapa banyak?*
dan sebagai gantinya berfokus pada pertanyaan *kategori mana?*

* Apakah email ini masuk ke folder spam atau inbox?
* Apakah pelanggan ini lebih mungkin untuk mendaftar
  atau tidak untuk layanan berlangganan?
* Apakah gambar ini menampilkan seekor keledai, anjing, kucing, atau ayam jantan?
* Film mana yang paling mungkin akan ditonton Aston selanjutnya?
* Bagian mana dari buku yang akan Anda baca berikutnya?

Secara umum, praktisi machine learning
menggunakan kata *klasifikasi*
untuk menggambarkan dua masalah yang secara halus berbeda:
(i) mereka yang hanya tertarik pada
penetapan tegas contoh ke dalam kategori (kelas);
dan (ii) mereka yang ingin membuat penetapan lunak,
yaitu, menilai probabilitas bahwa setiap kategori berlaku.
Perbedaan ini sering kali menjadi kabur, sebagian karena
sering kali, meskipun kita hanya peduli pada penetapan tegas,
kita masih menggunakan model yang membuat penetapan lunak.

Lebih jauh lagi, ada kasus di mana lebih dari satu label mungkin benar.
Misalnya, sebuah artikel berita mungkin secara bersamaan mencakup
topik hiburan, bisnis, dan penerbangan luar angkasa,
tetapi tidak mencakup topik medis atau olahraga.
Maka, mengkategorikan artikel ini hanya ke dalam satu dari kategori tersebut
tidak akan sangat berguna.
Masalah ini umumnya dikenal sebagai [klasifikasi multi-label](https://en.wikipedia.org/wiki/Multi-label_classification).
Lihat :citet:`Tsoumakas.Katakis.2007` untuk gambaran umum
dan :citet:`Huang.Xu.Yu.2015`
untuk algoritma yang efektif dalam penandaan gambar.

## Klasifikasi
:label:`subsec_classification-problem`

Untuk memulai, mari kita mulai dengan
masalah klasifikasi gambar sederhana.
Di sini, setiap input terdiri dari gambar grayscale $2\times2$.
Kita dapat merepresentasikan setiap nilai piksel dengan satu skalar,
sehingga kita memiliki empat fitur $x_1, x_2, x_3, x_4$.
Selanjutnya, kita asumsikan bahwa setiap gambar termasuk dalam salah satu
dari kategori "kucing", "ayam", dan "anjing".

Selanjutnya, kita harus memilih cara untuk merepresentasikan label.
Kita memiliki dua pilihan yang jelas.
Mungkin dorongan paling alami adalah
memilih $y \in \{1, 2, 3\}$,
di mana bilangan bulat mewakili
$\{\textrm{anjing}, \textrm{kucing}, \textrm{ayam}\}$ secara berturut-turut.
Ini adalah cara yang bagus untuk *menyimpan* informasi semacam ini di komputer.
Jika kategori memiliki urutan alami,
misalnya jika kita mencoba memprediksi
$\{\textrm{bayi}, \textrm{balita}, \textrm{remaja}, \textrm{dewasa muda}, \textrm{dewasa}, \textrm{lansia}\}$,
maka mungkin masuk akal untuk mengklasifikasikan ini sebagai masalah
[regresi ordinal](https://en.wikipedia.org/wiki/Ordinal_regression)
dan menyimpan label dalam format ini.
Lihat :citet:`Moon.Smola.Chang.ea.2010` untuk gambaran umum
berbagai jenis fungsi loss peringkat
dan :citet:`Beutel.Murray.Faloutsos.ea.2014` untuk pendekatan Bayesian
yang menangani respons dengan lebih dari satu mode.

Secara umum, masalah klasifikasi tidak datang
dengan urutan alami di antara kelas-kelasnya.
Untungnya, ahli statistik lama telah menemukan cara sederhana
untuk merepresentasikan data kategori: *one-hot encoding*.
One-hot encoding adalah vektor
dengan komponen sebanyak jumlah kategori yang kita miliki.
Komponen yang sesuai dengan kategori dari instance tertentu diatur ke 1
dan semua komponen lainnya diatur ke 0.
Dalam kasus kita, label $y$ akan menjadi vektor berdimensi tiga,
dengan $(1, 0, 0)$ mewakili "kucing", $(0, 1, 0)$ untuk "ayam",
dan $(0, 0, 1)$ untuk "anjing":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

### Model Linear

Untuk memperkirakan probabilitas kondisional
yang terkait dengan semua kelas yang mungkin,
kita memerlukan model dengan banyak output, satu untuk setiap kelas.
Untuk mengatasi klasifikasi dengan model linear,
kita membutuhkan sebanyak mungkin fungsi afine sesuai dengan jumlah output.
Secara teknis, kita hanya perlu satu fungsi lebih sedikit,
karena kategori terakhir merupakan selisih antara $1$ dan jumlah kategori lainnya,
tetapi demi simetri,
kita menggunakan parameterisasi yang sedikit berlebih.
Setiap output sesuai dengan fungsi afine-nya sendiri.
Dalam kasus kita, karena kita memiliki 4 fitur dan 3 kategori output yang mungkin,
kita memerlukan 12 skalar untuk mewakili bobot ($w$ dengan subskrip),
dan 3 skalar untuk mewakili bias ($b$ dengan subskrip). Ini menghasilkan:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Diagram jaringan saraf yang sesuai
ditampilkan pada :numref:`fig_softmaxreg`.
Seperti pada regresi linear,
kita menggunakan jaringan saraf satu lapis.
Dan karena perhitungan setiap output, $o_1, o_2$, dan $o_3$,
bergantung pada setiap input, $x_1$, $x_2$, $x_3$, dan $x_4$,
lapisan output ini juga dapat disebut sebagai *fully connected layer*.

![Regresi softmax adalah jaringan saraf satu lapis.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Untuk notasi yang lebih ringkas, kita menggunakan vektor dan matriks:
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$
jauh lebih cocok untuk matematika dan kode.
Perhatikan bahwa kita telah mengumpulkan semua bobot kita dalam matriks $3 \times 4$ dan semua bias
$\mathbf{b} \in \mathbb{R}^3$ dalam sebuah vektor.

### Fungsi Softmax
:label:`subsec_softmax_operation`

Dengan asumsi terdapat fungsi loss yang sesuai,
kita bisa langsung mencoba meminimalkan perbedaan
antara $\mathbf{o}$ dan label $\mathbf{y}$.
Meskipun menganggap klasifikasi
sebagai masalah regresi vektor-valued bekerja cukup baik,
hal ini tetap tidak memuaskan dalam cara berikut:

* Tidak ada jaminan bahwa output $o_i$ akan berjumlah $1$, sebagaimana yang kita harapkan dari probabilitas.
* Tidak ada jaminan bahwa output $o_i$ bernilai non-negatif, meskipun outputnya berjumlah $1$, atau bahwa nilainya tidak melebihi $1$.

Kedua aspek ini membuat masalah estimasi sulit untuk dipecahkan
dan solusinya sangat rentan terhadap outlier.
Misalnya, jika kita berasumsi bahwa terdapat ketergantungan linear positif
antara jumlah kamar tidur dan kemungkinan
seseorang akan membeli rumah,
probabilitasnya mungkin melebihi $1$
ketika datang untuk membeli rumah mewah!
Karena itu, kita memerlukan mekanisme untuk "mengompres" output.

Ada banyak cara untuk mencapai tujuan ini.
Misalnya, kita bisa mengasumsikan bahwa output
$\mathbf{o}$ adalah versi yang telah tercemar dari $\mathbf{y}$,
di mana pencemaran terjadi melalui penambahan noise $\boldsymbol{\epsilon}$
yang diambil dari distribusi normal.
Dengan kata lain, $\mathbf{y} = \mathbf{o} + \boldsymbol{\epsilon}$,
di mana $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
Ini disebut [model probit](https://en.wikipedia.org/wiki/Probit_model),
yang pertama kali diperkenalkan oleh :citet:`Fechner.1860`.
Meskipun menarik, model ini tidak bekerja sebaik
dan tidak menghasilkan masalah optimisasi yang mudah diselesaikan,
jika dibandingkan dengan softmax.

Cara lain untuk mencapai tujuan ini
(dan untuk memastikan non-negatif) adalah menggunakan
fungsi eksponensial $P(y = i) \propto \exp o_i$.
Ini memenuhi syarat bahwa
probabilitas kelas kondisional
meningkat seiring dengan meningkatnya $o_i$, bersifat monoton,
dan semua probabilitas adalah non-negatif.
Kita kemudian dapat mentransformasikan nilai-nilai ini sehingga berjumlah $1$
dengan membagi masing-masing dengan jumlah totalnya.
Proses ini disebut *normalisasi*.
Menggabungkan dua bagian ini
menghasilkan fungsi *softmax*:

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{di mana}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$
:eqlabel:`eq_softmax_y_and_o`

Perhatikan bahwa komponen terbesar dari $\mathbf{o}$
sesuai dengan kelas yang paling mungkin menurut $\hat{\mathbf{y}}$.
Selain itu, karena operasi softmax
mempertahankan urutan di antara argumennya,
kita tidak perlu menghitung softmax
untuk menentukan kelas mana yang memiliki probabilitas tertinggi. Dengan demikian,

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

Ide softmax berasal dari :citet:`Gibbs.1902`,
yang mengadaptasi ide dari fisika.
Lebih jauh ke belakang, Boltzmann,
bapak fisika statistik modern,
menggunakan trik ini untuk memodelkan distribusi
atas keadaan energi pada molekul gas.
Secara khusus, ia menemukan bahwa prevalensi
suatu keadaan energi dalam ensemble termodinamika,
seperti molekul dalam gas,
sebanding dengan $\exp(-E/kT)$.
Di sini, $E$ adalah energi suatu keadaan,
$T$ adalah temperatur, dan $k$ adalah konstanta Boltzmann.
Ketika ahli statistik berbicara tentang meningkatkan atau menurunkan
"temperatur" dari suatu sistem statistik,
mereka mengacu pada mengubah $T$
untuk mendukung keadaan energi yang lebih rendah atau lebih tinggi.
Mengikuti ide Gibbs, energi disamakan dengan kesalahan.
Model berbasis energi :cite:`Ranzato.Boureau.Chopra.ea.2007`
menggunakan sudut pandang ini ketika menggambarkan
masalah dalam deep learning.


### Vektorisasi
:label:`subsec_softmax_vectorization`

Untuk meningkatkan efisiensi komputasi,
kita melakukan vektorisasi perhitungan pada minibatch data.
Misalkan kita memiliki minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$
dari $n$ contoh dengan dimensi (jumlah input) $d$.
Selain itu, misalkan kita memiliki $q$ kategori dalam output.
Maka bobot memenuhi $\mathbf{W} \in \mathbb{R}^{d \times q}$
dan bias memenuhi $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Hal ini mempercepat operasi utama menjadi
produk matriks--matriks $\mathbf{X} \mathbf{W}$.
Selain itu, karena setiap baris dalam $\mathbf{X}$ mewakili contoh data,
operasi softmax itu sendiri dapat dihitung secara *per baris*:
untuk setiap baris dari $\mathbf{O}$, lakukan eksponensial pada semua entri
dan kemudian normalkan mereka dengan jumlah totalnya.
Namun, perlu diperhatikan bahwa kita harus berhati-hati
untuk menghindari eksponensiasi dan mengambil logaritma dari angka besar,
karena ini dapat menyebabkan overflow atau underflow numerik.
Framework deep learning menangani ini secara otomatis.

## Fungsi Loss
:label:`subsec_softmax-regression-loss-func`

Sekarang setelah kita memiliki pemetaan dari fitur $\mathbf{x}$
ke probabilitas $\mathbf{\hat{y}}$,
kita memerlukan cara untuk mengoptimalkan akurasi pemetaan ini.
Kita akan bergantung pada estimasi maksimum likelihood,
metode yang sama yang kita temui
saat memberikan justifikasi probabilistik
untuk loss mean squared error dalam
:numref:`subsec_normal_distribution_and_squared_loss`.

### Log-Likelihood

Fungsi softmax memberi kita sebuah vektor $\hat{\mathbf{y}}$,
yang dapat kita tafsirkan sebagai probabilitas kondisional (estimasi)
dari setiap kelas, mengingat input $\mathbf{x}$,
seperti $\hat{y}_1 = P(y=\textrm{kucing} \mid \mathbf{x})$.
Selanjutnya kita mengasumsikan bahwa untuk sebuah dataset
dengan fitur $\mathbf{X}$ label $\mathbf{Y}$
direpresentasikan menggunakan vektor label one-hot encoding.
Kita dapat membandingkan estimasi dengan realitas
dengan memeriksa seberapa mungkin kelas sebenarnya
menurut model kita, berdasarkan fitur-fitur tersebut:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

Kita dapat menggunakan faktorisasi ini
karena kita mengasumsikan bahwa setiap label diambil secara independen
dari distribusi masing-masing $P(\mathbf{y}\mid\mathbf{x}^{(i)})$.
Karena memaksimalkan hasil kali dari beberapa suku agak sulit,
kita mengambil logaritma negatif untuk mendapatkan masalah yang setara
yaitu meminimalkan negative log-likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

di mana untuk setiap pasangan label $\mathbf{y}$
dan prediksi model $\hat{\mathbf{y}}$
dalam $q$ kelas, fungsi loss $l$ adalah

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

Untuk alasan yang akan dijelaskan nanti,
fungsi loss dalam :eqref:`eq_l_cross_entropy`
umumnya disebut sebagai *loss cross-entropy*.
Karena $\mathbf{y}$ adalah vektor one-hot berdimensi $q$,
penjumlahan atas semua koordinat $j$ akan bernilai nol untuk semua kecuali satu suku.
Perhatikan bahwa loss $l(\mathbf{y}, \hat{\mathbf{y}})$
dibatasi dari bawah oleh $0$
ketika $\hat{\mathbf{y}}$ adalah vektor probabilitas:
tidak ada satu entri pun yang lebih besar dari $1$,
sehingga logaritma negatif mereka tidak bisa lebih rendah dari $0$;
$l(\mathbf{y}, \hat{\mathbf{y}}) = 0$ hanya jika kita memprediksi
label aktual dengan *kepastian*.
Hal ini tidak mungkin terjadi pada pengaturan bobot yang terbatas,
karena untuk mengarahkan output softmax menuju $1$
kita perlu membuat input yang sesuai $o_i$ menuju tak terhingga
(atau semua output lain $o_j$ untuk $j \neq i$ menuju negatif tak terhingga).
Bahkan jika model kita dapat menetapkan probabilitas output sebesar $0$,
kesalahan apa pun yang dibuat dengan menetapkan kepercayaan tinggi tersebut
akan menghasilkan loss tak hingga ($-\log 0 = \infty$).



### Softmax dan Loss Cross-Entropy
:label:`subsec_softmax_and_derivatives`

Karena fungsi softmax
dan loss cross-entropy yang terkait begitu umum,
maka perlu memahami sedikit lebih dalam bagaimana mereka dihitung.
Dengan memasukkan :eqref:`eq_softmax_y_and_o` ke dalam definisi loss
di :eqref:`eq_l_cross_entropy`
dan menggunakan definisi softmax, kita peroleh

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

Untuk memahami sedikit lebih baik apa yang sedang terjadi,
pertimbangkan turunan terhadap setiap logit $o_j$. Kita mendapatkan

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

Dengan kata lain, turunan adalah perbedaan
antara probabilitas yang diberikan oleh model kita,
seperti yang diungkapkan oleh operasi softmax,
dan apa yang sebenarnya terjadi, sebagaimana diwakili
oleh elemen-elemen dalam vektor label one-hot.
Dalam hal ini, turunan ini sangat mirip
dengan yang kita lihat pada regresi,
di mana gradien adalah perbedaan
antara pengamatan $y$ dan estimasi $\hat{y}$.
Ini bukan kebetulan.
Dalam model family eksponensial apa pun,
gradien log-likelihood diberikan oleh suku ini.
Fakta ini membuat perhitungan gradien mudah dalam praktiknya.

Sekarang pertimbangkan kasus di mana kita mengamati bukan hanya satu hasil
tetapi seluruh distribusi hasil.
Kita dapat menggunakan representasi yang sama seperti sebelumnya untuk label $\mathbf{y}$.
Satu-satunya perbedaan adalah bahwa
alih-alih vektor yang hanya berisi entri biner,
misalnya $(0, 0, 1)$, sekarang kita memiliki vektor probabilitas umum,
misalnya $(0.1, 0.2, 0.7)$.
Matematika yang kita gunakan sebelumnya untuk mendefinisikan loss $l$
di :eqref:`eq_l_cross_entropy`
masih berfungsi dengan baik,
hanya saja interpretasinya sedikit lebih umum.
Ini adalah nilai ekspektasi dari loss untuk distribusi label.
Loss ini disebut *loss cross-entropy* dan merupakan
salah satu loss yang paling umum digunakan untuk masalah klasifikasi.
Kita dapat memperjelas nama ini dengan memperkenalkan dasar-dasar teori informasi.
Secara singkat, ini mengukur jumlah bit yang dibutuhkan untuk mengkodekan apa yang kita lihat, $\mathbf{y}$,
relatif terhadap apa yang kita prediksi akan terjadi, $\hat{\mathbf{y}}$.
Berikut adalah penjelasan dasar. Untuk rincian lebih lanjut tentang teori informasi, lihat
:citet:`Cover.Thomas.1999` atau :citet:`mackay2003information`.

## Dasar-dasar Teori Informasi
:label:`subsec_info_theory_basics`

Banyak makalah deep learning menggunakan intuisi dan istilah dari teori informasi.
Untuk memahaminya, kita memerlukan beberapa istilah umum.
Ini adalah panduan dasar.
*Teori informasi* berkaitan dengan masalah
pengkodean, dekode, transmisi,
dan manipulasi informasi (juga dikenal sebagai data).

### Entropi

Ide utama dalam teori informasi adalah mengkuantifikasi
jumlah informasi yang terkandung dalam data.
Ini memberi batas pada kemampuan kita untuk mengompresi data.
Untuk distribusi $P$, *entropi*-nya, $H[P]$, didefinisikan sebagai:

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

Salah satu teorema fundamental teori informasi menyatakan
bahwa untuk mengkodekan data yang diambil secara acak dari distribusi $P$,
kita membutuhkan setidaknya $H[P]$ "nat" untuk mengkodekannya :cite:`Shannon.1948`.
Jika Anda bertanya-tanya apa itu "nat", ini setara dengan bit
tetapi saat menggunakan kode berbasis $e$ daripada berbasis 2.
Jadi, satu nat adalah $\frac{1}{\log(2)} \approx 1.44$ bit.



### Surprisal

Anda mungkin bertanya-tanya apa hubungan kompresi dengan prediksi.
Bayangkan kita memiliki aliran data yang ingin kita kompresi.
Jika selalu mudah bagi kita untuk memprediksi token berikutnya,
maka data ini mudah dikompresi.
Ambil contoh ekstrem di mana setiap token dalam aliran
selalu memiliki nilai yang sama.
Itu adalah aliran data yang sangat membosankan!
Dan tidak hanya membosankan, tetapi juga mudah diprediksi.
Karena token-token tersebut selalu sama,
kita tidak perlu mengirimkan informasi apa pun
untuk mengomunikasikan isi dari aliran tersebut.
Mudah diprediksi, mudah dikompresi.

Namun, jika kita tidak dapat memprediksi setiap kejadian dengan sempurna,
maka terkadang kita mungkin terkejut.
Tingkat keterkejutan kita lebih besar ketika suatu kejadian diberi probabilitas yang lebih rendah.
Claude Shannon memilih $\log \frac{1}{P(j)} = -\log P(j)$
untuk mengkuantifikasi *keterkejutan* seseorang saat mengamati kejadian $j$
yang diberi probabilitas (subjektif) $P(j)$.
Entropi yang didefinisikan pada :eqref:`eq_softmax_reg_entropy`
kemudian menjadi *ekspektasi keterkejutan*
ketika seseorang memberikan probabilitas yang benar
yang benar-benar sesuai dengan proses pembangkitan data.

### Peninjauan Kembali Cross-Entropy

Jika entropi adalah tingkat keterkejutan yang dialami
oleh seseorang yang mengetahui probabilitas sebenarnya,
maka Anda mungkin bertanya, apa itu cross-entropy?
Cross-entropy *dari* $P$ *ke* $Q$, dilambangkan dengan $H(P, Q)$,
adalah ekspektasi keterkejutan dari seorang pengamat dengan probabilitas subjektif $Q$
saat melihat data yang sebenarnya dihasilkan menurut probabilitas $P$.
Ini diberikan oleh $H(P, Q) \stackrel{\textrm{def}}{=} \sum_j - P(j) \log Q(j)$.
Cross-entropy terendah tercapai ketika $P=Q$.
Dalam kasus ini, cross-entropy dari $P$ ke $Q$ adalah $H(P, P)= H(P)$.

Singkatnya, kita dapat melihat tujuan klasifikasi cross-entropy
dengan dua cara: (i) sebagai memaksimalkan likelihood dari data yang diamati;
dan (ii) sebagai meminimalkan keterkejutan kita (dan dengan demikian jumlah bit)
yang dibutuhkan untuk mengomunikasikan label.

## Ringkasan dan Diskusi

Pada bagian ini, kita menemukan fungsi loss pertama yang non-trivial,
yang memungkinkan kita untuk mengoptimalkan ruang output *diskret*.
Kunci dari desainnya adalah pendekatan probabilistik yang kita ambil,
memperlakukan kategori diskret sebagai contoh pengambilan sampel dari distribusi probabilitas.
Sebagai efek samping, kita menemukan softmax,
fungsi aktivasi yang nyaman untuk mentransformasikan
output dari lapisan jaringan saraf biasa
menjadi distribusi probabilitas diskret yang valid.
Kita melihat bahwa turunan dari loss cross-entropy
ketika digabungkan dengan softmax
berperilaku sangat mirip
dengan turunan dari squared error;
yakni, dengan mengambil selisih antara
perilaku yang diharapkan dan prediksinya.
Dan, meskipun kita hanya bisa menyentuh permukaannya,
kita menemukan hubungan menarik
dengan fisika statistik dan teori informasi.

Meskipun ini cukup untuk memulai Anda,
dan mudah-mudahan cukup untuk menggugah minat Anda,
kita tidak mendalami materi di sini.
Antara lain, kita melewatkan pertimbangan komputasional.
Secara spesifik, untuk lapisan fully connected dengan $d$ input dan $q$ output,
parameterisasi dan biaya komputasinya adalah $\mathcal{O}(dq)$,
yang bisa menjadi sangat mahal dalam praktiknya.
Untungnya, biaya mengubah $d$ input menjadi $q$ output
dapat dikurangi melalui pendekatan aproksimasi dan kompresi.
Sebagai contoh, Deep Fried Convnets :cite:`Yang.Moczulski.Denil.ea.2015`
menggunakan kombinasi dari permutasi,
transformasi Fourier, dan skala
untuk mengurangi biaya dari kuadratik menjadi log-linear.
Teknik serupa bekerja untuk aproksimasi matriks struktural yang lebih maju :cite:`sindhwani2015structured`.
Terakhir, kita dapat menggunakan dekomposisi seperti quaternion
untuk mengurangi biaya menjadi $\mathcal{O}(\frac{dq}{n})$,
lagi-lagi jika kita bersedia menukar sedikit akurasi
untuk biaya komputasi dan penyimpanan :cite:`Zhang.Tay.Zhang.ea.2021`
berdasarkan faktor kompresi $n$.
Ini adalah area penelitian yang aktif.
Yang membuatnya menantang adalah
kita tidak selalu berusaha
untuk mendapatkan representasi yang paling ringkas
atau jumlah operasi floating point terkecil,
tetapi lebih pada solusi
yang dapat dieksekusi secara efisien di GPU modern.

## Latihan

1. Kita dapat mengeksplorasi hubungan antara family eksponensial dan softmax lebih dalam.
    1. Hitung turunan kedua dari loss cross-entropy $l(\mathbf{y},\hat{\mathbf{y}})$ untuk softmax.
    1. Hitung varians dari distribusi yang diberikan oleh $\mathrm{softmax}(\mathbf{o})$ dan tunjukkan bahwa hasilnya sesuai dengan turunan kedua yang dihitung sebelumnya.
1. Asumsikan kita memiliki tiga kelas yang terjadi dengan probabilitas yang sama, yaitu vektor probabilitas $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    1. Apa masalahnya jika kita mencoba merancang kode biner untuk itu?
    1. Dapatkah Anda merancang kode yang lebih baik? Petunjuk: apa yang terjadi jika kita mencoba mengkodekan dua pengamatan independen? Bagaimana jika kita mengkodekan $n$ pengamatan secara bersamaan?
1. Saat mengkodekan sinyal yang ditransmisikan melalui kabel fisik, insinyur tidak selalu menggunakan kode biner. Misalnya, [PAM-3](https://en.wikipedia.org/wiki/Ternary_signal) menggunakan tiga tingkat sinyal $\{-1, 0, 1\}$ dibandingkan dengan dua tingkat $\{0, 1\}$. Berapa banyak unit ternary yang Anda butuhkan untuk mengirimkan bilangan bulat dalam rentang $\{0, \ldots, 7\}$? Mengapa ini mungkin lebih baik dalam hal elektronik?
1. Model [Bradley--Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) menggunakan
model logistik untuk menangkap preferensi. Misalkan seorang pengguna memilih antara apel dan jeruk dengan skor $o_{\textrm{apel}}$ dan $o_{\textrm{jeruk}}$. Kita ingin skor yang lebih besar mengarah pada kemungkinan lebih tinggi dalam memilih item terkait dan bahwa
item dengan skor terbesar adalah yang paling mungkin dipilih :cite:`Bradley.Terry.1952`.
    1. Buktikan bahwa softmax memenuhi persyaratan ini.
    1. Apa yang terjadi jika Anda ingin memberikan opsi default untuk tidak memilih baik apel maupun jeruk? Petunjuk: sekarang pengguna memiliki tiga pilihan.
1. Softmax mendapatkan namanya dari pemetaan berikut: $\textrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    1. Buktikan bahwa $\textrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    1. Seberapa kecil Anda dapat membuat perbedaan antara kedua fungsi ini? Petunjuk: tanpa kehilangan generalitas Anda bisa mengatur $b = 0$ dan $a \geq b$.
    1. Buktikan bahwa ini berlaku untuk $\lambda^{-1} \textrm{RealSoftMax}(\lambda a, \lambda b)$, asalkan $\lambda > 0$.
    1. Tunjukkan bahwa untuk $\lambda \to \infty$ kita memiliki $\lambda^{-1} \textrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    1. Konstruksi fungsi softmin yang serupa.
    1. Perluas ini untuk lebih dari dua angka.
1. Fungsi $g(\mathbf{x}) \stackrel{\textrm{def}}{=} \log \sum_i \exp x_i$ kadang-kadang juga disebut sebagai [fungsi partisi log](https://en.wikipedia.org/wiki/Partition_function_(mathematics)).
    1. Buktikan bahwa fungsi ini konveks. Petunjuk: untuk melakukannya, gunakan fakta bahwa turunan pertama berjumlah pada probabilitas dari fungsi softmax dan tunjukkan bahwa turunan kedua adalah varians.
    1. Tunjukkan bahwa $g$ adalah invarian translasi, yaitu, $g(\mathbf{x} + b) = g(\mathbf{x})$.
    1. Apa yang terjadi jika beberapa koordinat $x_i$ sangat besar? Apa yang terjadi jika semuanya sangat kecil?
    1. Tunjukkan bahwa jika kita memilih $b = \mathrm{max}_i x_i$ kita mendapatkan implementasi yang stabil secara numerik.
1. Asumsikan kita memiliki distribusi probabilitas $P$. Misalkan kita memilih distribusi lain $Q$ dengan $Q(i) \propto P(i)^\alpha$ untuk $\alpha > 0$.
    1. Pilihan $\alpha$ yang mana yang sesuai dengan menggandakan temperatur? Pilihan mana yang sesuai dengan membaginya dua?
    1. Apa yang terjadi jika kita membiarkan temperatur mendekati $0$?
    1. Apa yang terjadi jika kita membiarkan temperatur mendekati $\infty$?

[Diskusi](https://discuss.d2l.ai/t/46)
