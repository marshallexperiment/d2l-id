# Forward Propagation, Backward Propagation, dan Computational Graphs
:label:`sec_backprop`

Sejauh ini, kita telah melatih model kita
dengan minibatch stochastic gradient descent.
Namun, ketika kita mengimplementasikan algoritma tersebut,
kita hanya berfokus pada perhitungan yang terlibat dalam
*forward propagation* melalui model.
Ketika tiba saatnya menghitung gradien,
kita cukup memanggil fungsi backpropagation yang disediakan oleh framework deep learning.

Perhitungan otomatis gradien
secara signifikan menyederhanakan
implementasi algoritma deep learning.
Sebelum adanya diferensiasi otomatis,
bahkan perubahan kecil pada model yang rumit mengharuskan
penghitungan ulang turunan yang rumit secara manual.
Cukup sering, makalah akademis harus mengalokasikan
banyak halaman untuk menurunkan aturan pembaruan.
Meskipun kita harus terus bergantung pada diferensiasi otomatis
agar kita dapat fokus pada bagian yang lebih menarik,
Anda sebaiknya tahu bagaimana gradien ini
dihitung di balik layar
jika Anda ingin memahami lebih dalam
tentang deep learning.

Pada bagian ini, kita akan mendalami
detail dari *backward propagation*
(yang lebih umum disebut *backpropagation*).
Untuk memberikan pemahaman baik tentang
teknik maupun implementasinya,
kita akan menggunakan beberapa konsep matematika dasar dan computational graphs.
Untuk memulai, kita akan fokus pada
MLP dengan satu hidden layer
dengan *weight decay* ($\ell_2$ regularization, yang akan dijelaskan pada bab-bab selanjutnya).

## Forward Propagation

*Forward propagation* (atau *forward pass*) mengacu pada perhitungan dan penyimpanan
variabel-variabel antara (termasuk keluaran)
untuk jaringan saraf secara berurutan
dari lapisan input hingga lapisan output.
Sekarang kita akan menjelajahi mekanisme
dari jaringan saraf dengan satu hidden layer secara mendetail.
Mungkin ini terasa membosankan tetapi seperti yang dikatakan
oleh musisi funk terkenal James Brown,
Anda harus "membayar harga untuk menjadi bos."

Untuk menyederhanakan, mari kita asumsikan
bahwa contoh input adalah $\mathbf{x} \in \mathbb{R}^d$
dan hidden layer kita tidak memiliki bias.
Di sini variabel antara adalah:

$$\mathbf{z} = \mathbf{W}^{(1)} \mathbf{x},$$

dengan $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
sebagai parameter bobot dari hidden layer.
Setelah menjalankan variabel antara
$\mathbf{z} \in \mathbb{R}^h$ melalui
fungsi aktivasi $\phi$,
kita memperoleh vektor aktivasi hidden dengan panjang $h$:

$$\mathbf{h} = \phi(\mathbf{z}).$$

Keluaran dari hidden layer $\mathbf{h}$
juga merupakan variabel antara.
Dengan mengasumsikan bahwa parameter dari output layer
hanya memiliki bobot sebesar
$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$,
kita dapat memperoleh variabel output layer
dengan vektor panjang $q$:

$$\mathbf{o} = \mathbf{W}^{(2)} \mathbf{h}.$$

Dengan mengasumsikan bahwa fungsi loss adalah $l$
dan label dari contoh data adalah $y$,
kita kemudian dapat menghitung nilai loss
untuk satu contoh data,

$$L = l(\mathbf{o}, y).$$

Seperti yang akan kita lihat pada definisi $\ell_2$ regularization
yang akan diperkenalkan kemudian,
dengan hyperparameter $\lambda$,
term regularisasi adalah

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_\textrm{F}^2 + \|\mathbf{W}^{(2)}\|_\textrm{F}^2\right),$$
:eqlabel:`eq_forward-s`

di mana Frobenius norm dari matriks
adalah $\ell_2$ norm yang diterapkan
setelah meratakan matriks menjadi vektor.
Akhirnya, loss yang telah diregularisasi dari model
pada contoh data yang diberikan adalah:

$$J = L + s.$$

Kita menyebut $J$ sebagai *objective function*
dalam diskusi berikut.


## Computational Graph dari Forward Propagation

Menggambar *computational graphs* membantu kita memvisualisasikan
ketergantungan antara operator
dan variabel dalam perhitungan.
:numref:`fig_forward` menunjukkan grafik yang terkait
dengan jaringan sederhana yang dijelaskan di atas,
di mana persegi mewakili variabel dan lingkaran mewakili operator.
Sudut kiri bawah menunjukkan input
dan sudut kanan atas adalah output.
Perhatikan bahwa arah panah
(yang menggambarkan aliran data)
terutama mengarah ke kanan dan ke atas.

![Computational graph dari forward propagation.](../img/forward.svg)
:label:`fig_forward`

## Backpropagation

*Backpropagation* mengacu pada metode perhitungan
gradien dari parameter jaringan saraf.
Secara singkat, metode ini menjelajahi jaringan secara terbalik,
dari lapisan output ke lapisan input,
berdasarkan *chain rule* dari kalkulus.
Algoritma ini menyimpan setiap variabel antara
(partial derivatives) yang dibutuhkan saat menghitung gradien
terhadap beberapa parameter.
Misalkan kita memiliki fungsi
$\mathsf{Y}=f(\mathsf{X})$
dan $\mathsf{Z}=g(\mathsf{Y})$,
di mana input dan output
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$
adalah tensor dengan bentuk sembarang.
Dengan menggunakan chain rule,
kita dapat menghitung turunan
dari $\mathsf{Z}$ terhadap $\mathsf{X}$ melalui

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \textrm{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Di sini kita menggunakan operator $\textrm{prod}$
untuk mengalikan argumennya
setelah operasi yang diperlukan,
seperti transposisi dan menukar posisi input, telah dilakukan.
Untuk vektor, ini sederhana:
hanya perkalian matriks-matriks.
Untuk tensor berdimensi lebih tinggi,
kita menggunakan padanan yang sesuai.
Operator $\textrm{prod}$ menyembunyikan semua notasi tambahan.

Ingat bahwa
parameter dari jaringan sederhana dengan satu hidden layer,
yang computational graph-nya ada pada :numref:`fig_forward`,
adalah $\mathbf{W}^{(1)}$ dan $\mathbf{W}^{(2)}$.
Tujuan dari backpropagation adalah untuk
menghitung gradien $\partial J/\partial \mathbf{W}^{(1)}$
dan $\partial J/\partial \mathbf{W}^{(2)}$.
Untuk mencapai ini, kita menerapkan chain rule
dan menghitung, secara bergantian, gradien dari
setiap variabel antara dan parameter.
Urutan perhitungan terbalik
relatif terhadap yang dilakukan pada forward propagation,
karena kita perlu memulai dari hasil computational graph
dan berlanjut ke arah parameter.
Langkah pertama adalah menghitung gradien
dari fungsi objektif $J=L+s$
terhadap nilai loss $L$
dan nilai regularisasi $s$:

$$\frac{\partial J}{\partial L} = 1 \; \textrm{dan} \; \frac{\partial J}{\partial s} = 1.$$

Selanjutnya, kita menghitung gradien dari fungsi objektif
terhadap variabel dari output layer $\mathbf{o}$
sesuai dengan chain rule:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \textrm{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Berikutnya, kita menghitung gradien
dari term regularisasi
terhadap kedua parameter:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \textrm{dan} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Sekarang kita dapat menghitung gradien
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$
dari parameter model yang paling dekat dengan output layer.
Dengan menggunakan chain rule kita peroleh:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

Untuk mendapatkan gradien terhadap $\mathbf{W}^{(1)}$
kita perlu melanjutkan backpropagation
dari output layer ke hidden layer.
Gradien terhadap output hidden layer
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ diberikan oleh

$$
\frac{\partial J}{\partial \mathbf{h}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Karena fungsi aktivasi $\phi$ diterapkan secara elemen-per-elemen,
menghitung gradien $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
dari variabel antara $\mathbf{z}$
membutuhkan kita menggunakan operator perkalian elemen-per-elemen,
yang kita nyatakan dengan $\odot$:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Akhirnya, kita dapat memperoleh gradien
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
dari parameter model yang paling dekat dengan input layer.
Menurut chain rule, kita mendapatkan

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## Melatih _Neural Networks_

Saat melatih neural networks,
forward dan backward propagation saling bergantung satu sama lain.
Secara khusus, dalam forward propagation,
kita menjelajahi computational graph mengikuti arah ketergantungan
dan menghitung semua variabel pada jalurnya.
Variabel-variabel ini kemudian digunakan untuk backpropagation,
di mana urutan perhitungan pada grafik tersebut dibalik.

Ambil jaringan sederhana yang telah disebutkan sebagai contoh ilustratif.
Di satu sisi,
perhitungan regularization term :eqref:`eq_forward-s`
selama forward propagation
bergantung pada nilai parameter model saat ini $\mathbf{W}^{(1)}$ dan $\mathbf{W}^{(2)}$.
Parameter ini diberikan oleh algoritma optimisasi menurut backpropagation pada iterasi terbaru.
Di sisi lain,
perhitungan gradien untuk parameter
:eqref:`eq_backprop-J-h` selama backpropagation
bergantung pada nilai saat ini dari output hidden layer $\mathbf{h}$,
yang diberikan oleh forward propagation.

Oleh karena itu, saat melatih neural networks, setelah parameter model diinisialisasi,
kita bergantian antara forward propagation dengan backpropagation,
memperbarui parameter model menggunakan gradien yang diberikan oleh backpropagation.
Perlu dicatat bahwa backpropagation menggunakan kembali nilai antara yang disimpan dari forward propagation untuk menghindari perhitungan yang berulang.
Salah satu konsekuensinya adalah kita perlu menyimpan
nilai antara tersebut sampai backpropagation selesai.
Ini juga merupakan salah satu alasan mengapa pelatihan
membutuhkan lebih banyak memori secara signifikan dibandingkan prediksi.
Selain itu, ukuran dari nilai-nilai antara ini kira-kira
sebanding dengan jumlah lapisan jaringan dan ukuran batch.
Dengan demikian,
pelatihan jaringan yang lebih dalam dengan ukuran batch yang lebih besar
lebih mudah menyebabkan terjadinya kesalahan *out-of-memory*.


## Ringkasan

Forward propagation secara berurutan menghitung dan menyimpan variabel antara dalam computational graph yang didefinisikan oleh neural network. Proses ini berlangsung dari lapisan input ke lapisan output.
Backpropagation secara berurutan menghitung dan menyimpan gradien dari variabel antara dan parameter dalam neural network dalam urutan yang dibalik.
Saat melatih model deep learning, forward propagation dan backpropagation saling bergantung,
dan pelatihan membutuhkan memori yang jauh lebih besar dibandingkan dengan prediksi.


## Latihan

1. Asumsikan bahwa input $\mathbf{X}$ ke suatu fungsi skalar $f$ adalah matriks berukuran $n \times m$. Berapakah dimensi dari gradien $f$ terhadap $\mathbf{X}$?
2. Tambahkan bias ke hidden layer pada model yang dijelaskan dalam bagian ini (Anda tidak perlu menyertakan bias dalam term regularisasi).
    1. Gambar computational graph yang sesuai.
    2. Turunkan persamaan forward dan backward propagation.
3. Hitung penggunaan memori untuk pelatihan dan prediksi dalam model yang dijelaskan di bagian ini.
4. Asumsikan bahwa Anda ingin menghitung turunan kedua. Apa yang terjadi pada computational graph? Berapa lama kira-kira waktu yang dibutuhkan untuk perhitungan tersebut?
5. Asumsikan bahwa computational graph terlalu besar untuk GPU Anda.
    1. Dapatkah Anda mempartisinya pada lebih dari satu GPU?
    2. Apa keuntungan dan kerugian dari pelatihan pada minibatch yang lebih kecil?

[Diskusi](https://discuss.d2l.ai/t/102)
