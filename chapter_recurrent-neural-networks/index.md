# Recurrent Neural Networks
:label:`chap_rnn`

Sejauh ini, kita telah berfokus terutama pada data dengan panjang tetap.
Ketika memperkenalkan regresi linier dan logistik di :numref:`chap_regression` dan :numref:`chap_classification`,
serta *multilayer perceptrons* di :numref:`chap_perceptrons`,
kita berasumsi bahwa setiap vektor fitur $\mathbf{x}_i$
terdiri dari sejumlah komponen tetap $x_1, \dots, x_d$,
di mana setiap fitur numerik $x_j$
sesuai dengan atribut tertentu.
Dataset ini kadang-kadang disebut *tabular*,
karena dapat diatur dalam tabel,
di mana setiap contoh $i$ mendapatkan barisnya sendiri,
dan setiap atribut memiliki kolomnya sendiri.
Hal yang penting, dengan data tabular, kita jarang
mengasumsikan adanya struktur tertentu di antara kolom.

Selanjutnya, pada :numref:`chap_cnn`,
kita beralih ke data gambar, di mana input terdiri
dari nilai piksel mentah pada setiap koordinat dalam sebuah gambar.
Data gambar sulit diatur dalam format tabular.
Di sini, kita membutuhkan *convolutional neural networks* (CNN)
untuk menangani struktur hierarkis dan invariansi.
Namun, data kita masih memiliki panjang tetap.
Setiap gambar di Fashion-MNIST direpresentasikan
sebagai grid nilai piksel $28 \times 28$.
Selain itu, tujuan kita adalah mengembangkan model
yang hanya melihat satu gambar dan kemudian
menghasilkan satu prediksi.
Namun, apa yang harus kita lakukan ketika dihadapkan pada
urutan gambar, seperti pada video,
atau ketika ditugaskan menghasilkan prediksi
yang terstruktur secara berurutan,
seperti pada kasus pembuatan keterangan gambar?

Banyak sekali tugas pembelajaran yang membutuhkan pengelolaan data berurutan.
Pembuatan keterangan gambar, sintesis ucapan, dan pembuatan musik
semuanya memerlukan model yang menghasilkan output berupa urutan.
Di domain lain, seperti prediksi deret waktu,
analisis video, dan pengambilan informasi musik,
model harus belajar dari input yang berupa urutan.
Kebutuhan ini sering muncul bersamaan:
tugas-tugas seperti menerjemahkan teks
dari satu bahasa ke bahasa lain,
melakukan dialog, atau mengendalikan robot,
membutuhkan model yang dapat mengonsumsi dan menghasilkan
data yang terstruktur secara berurutan.

*Recurrent Neural Networks* (RNN) adalah model pembelajaran mendalam
yang menangkap dinamika urutan melalui
koneksi *recurrent*, yang dapat dianggap sebagai
siklus dalam jaringan node.
Ini mungkin terasa kontra-intuitif pada awalnya.
Lagi pula, sifat *feedforward* dari jaringan saraf
membuat urutan komputasi menjadi jelas.
Namun, koneksi *recurrent* didefinisikan secara tepat
sehingga tidak ada ambiguitas yang muncul.
RNN *diurai* ke langkah-langkah waktu (atau langkah urutan),
dengan parameter yang *sama* digunakan pada setiap langkah.
Sementara koneksi standar diterapkan secara *sinkron*
untuk menyebarkan aktivasi setiap lapisan ke lapisan berikutnya
*pada langkah waktu yang sama*,
koneksi *recurrent* bersifat *dinamis*,
mengirimkan informasi di antara langkah-langkah waktu berdekatan.
Sebagaimana terlihat pada pandangan yang diurai di :numref:`fig_unfolded-rnn`,
RNN dapat dianggap sebagai jaringan saraf *feedforward*
di mana parameter setiap lapisan (baik konvensional maupun *recurrent*)
dibagikan di seluruh langkah waktu.

![Di sebelah kiri, koneksi *recurrent* digambarkan melalui edge siklis. Di sebelah kanan, kita mengurai RNN ke langkah waktu. Di sini, koneksi *recurrent* melintasi langkah waktu berdekatan, sementara koneksi konvensional dihitung secara sinkron.](../img/unfolded-rnn.svg)
:label:`fig_unfolded-rnn`

Seperti jaringan saraf pada umumnya,
RNN memiliki sejarah panjang yang melintasi disiplin,
berawal dari model otak yang dipopulerkan
oleh ilmuwan kognitif dan kemudian diadopsi
sebagai alat pemodelan praktis yang digunakan
oleh komunitas pembelajaran mesin.
Seperti pembelajaran mendalam pada umumnya,
dalam buku ini kita mengadopsi perspektif pembelajaran mesin,
dengan fokus pada RNN sebagai alat praktis yang naik
ke popularitas pada 2010-an berkat
hasil terobosan pada berbagai tugas seperti
pengenalan tulisan tangan :cite:`graves2008novel`,
terjemahan mesin :cite:`Sutskever.Vinyals.Le.2014`,
dan pengenalan diagnosis medis :cite:`Lipton.Kale.2016`.
Pembaca yang tertarik dengan materi latar belakang yang lebih luas dapat membaca tinjauan komprehensif yang tersedia untuk umum :cite:`Lipton.Berkowitz.Elkan.2015`.
Kami juga mencatat bahwa sekuensialitas tidaklah unik bagi RNN.
Misalnya, CNN yang sudah kita bahas
dapat diadaptasi untuk menangani data dengan panjang yang bervariasi,
misalnya, gambar dengan resolusi yang berbeda-beda.
Selain itu, RNN baru-baru ini kehilangan pangsa pasar yang cukup besar terhadap model Transformer,
yang akan dibahas di :numref:`chap_attention-and-transformers`.
Namun, RNN menjadi terkenal sebagai model default
untuk menangani struktur berurutan yang kompleks dalam pembelajaran mendalam,
dan tetap menjadi model pokok untuk pemodelan sekuensial hingga hari ini.
Kisah RNN dan pemodelan urutan sangat terkait erat, dan ini adalah bab tentang dasar-dasar masalah pemodelan urutan
serta bab tentang RNN.

Satu wawasan kunci membuka jalan bagi revolusi dalam pemodelan urutan.
Sementara input dan target untuk banyak tugas dasar dalam pembelajaran mesin
tidak mudah direpresentasikan sebagai vektor dengan panjang tetap,
mereka seringkali dapat direpresentasikan sebagai
urutan dengan panjang bervariasi dari vektor dengan panjang tetap.
Sebagai contoh, dokumen dapat direpresentasikan sebagai urutan kata;
rekam medis seringkali dapat direpresentasikan sebagai urutan kejadian
(pertemuan, obat-obatan, prosedur, tes laboratorium, diagnosis);
video dapat direpresentasikan sebagai urutan gambar diam dengan panjang yang bervariasi.

Meskipun model urutan muncul dalam berbagai area aplikasi,
penelitian dasar di area ini sebagian besar didorong
oleh kemajuan dalam tugas-tugas inti pemrosesan bahasa alami.
Oleh karena itu, di sepanjang bab ini, kita akan fokus
pada eksposisi dan contoh teks data.
Jika Anda menguasai contoh-contoh ini,
maka menerapkan model-model ini pada modalitas data lain
seharusnya relatif mudah.
Pada bagian berikut, kita memperkenalkan notasi dasar untuk urutan dan beberapa ukuran evaluasi
untuk menilai kualitas output model yang terstruktur secara berurutan.
Setelah itu, kita membahas konsep dasar model bahasa
dan menggunakan diskusi ini untuk memotivasi model RNN pertama kita.
Terakhir, kita menjelaskan metode untuk menghitung gradien
saat melakukan *backpropagation* melalui RNN dan mengeksplorasi beberapa tantangan
yang sering ditemui saat melatih jaringan ini,
memotivasi arsitektur RNN modern yang akan dijelaskan
di :numref:`chap_modern_rnn`.


```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
rnn-scratch
rnn-concise
bptt
```

