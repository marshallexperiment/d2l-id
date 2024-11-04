```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Regresi Linear
:label:`sec_linear_regression`

Masalah *regresi* muncul ketika kita ingin memprediksi nilai numerik.
Contoh umum termasuk memprediksi harga (rumah, saham, dll.),
memperkirakan lama tinggal (pasien di rumah sakit),
memperkirakan permintaan (penjualan ritel), dan banyak lagi.
Tidak semua masalah prediksi adalah regresi klasik.
Nantinya, kita akan memperkenalkan masalah klasifikasi,
di mana tujuan adalah memprediksi keanggotaan dalam satu set kategori.

Sebagai contoh berkelanjutan, misalkan kita ingin
memperkirakan harga rumah (dalam dolar)
berdasarkan luas (dalam kaki persegi) dan usia (dalam tahun).
Untuk mengembangkan model prediksi harga rumah,
kita memerlukan data,
termasuk harga penjualan, luas, dan usia untuk setiap rumah.
Dalam terminologi pembelajaran mesin,
kumpulan data ini disebut *dataset pelatihan* atau *training set*,
dan setiap baris (yang berisi data yang sesuai dengan satu penjualan)
disebut sebagai *contoh* (atau *data point*, *instance*, *sample*).
Hal yang ingin kita prediksi (harga)
disebut *label* (atau *target*).
Variabel-variabel (usia dan luas)
yang menjadi dasar prediksi disebut *fitur* (atau *kovariat*).


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from jax import numpy as jnp
import math
import time
```

## Dasar-Dasar

*Regresi linear* adalah alat paling sederhana
dan paling populer di antara alat standar
untuk mengatasi masalah regresi.
Metode ini sudah ada sejak awal abad ke-19 :cite:`Legendre.1805,Gauss.1809`,
dan berasal dari beberapa asumsi sederhana.
Pertama, kita mengasumsikan bahwa hubungan
antara fitur $\mathbf{x}$ dan target $y$
adalah kira-kira linear,
yaitu, mean kondisional $E[Y \mid X=\mathbf{x}]$
dapat diekspresikan sebagai penjumlahan berbobot
dari fitur $\mathbf{x}$.
Pengaturan ini memungkinkan nilai target
masih bisa menyimpang dari nilai ekspektasinya
karena adanya noise pada pengamatan.
Selanjutnya, kita bisa memberlakukan asumsi bahwa noise tersebut
terdistribusi secara baik, mengikuti distribusi Gaussian.
Biasanya, kita menggunakan $n$ untuk menyatakan
jumlah contoh dalam dataset kita.
Kita menggunakan superscript untuk mengenumerasi sampel dan target,
dan subscript untuk mengindeks koordinat.
Secara lebih konkret,
$\mathbf{x}^{(i)}$ menyatakan sampel ke-$i$
dan $x_j^{(i)}$ menyatakan koordinat ke-$j$ dari sampel tersebut.

### Model
:label:`subsec_linear_model`

Di inti dari setiap solusi adalah model
yang menggambarkan bagaimana fitur dapat diubah
menjadi estimasi target.
Asumsi linearitas berarti bahwa
nilai harapan dari target (harga) dapat diekspresikan
sebagai penjumlahan berbobot dari fitur (luas dan usia):

$$\textrm{harga} = w_{\textrm{luas}} \cdot \textrm{luas} + w_{\textrm{usia}} \cdot \textrm{usia} + b.$$
:eqlabel:`eq_price-area`

Di sini, $w_{\textrm{luas}}$ dan $w_{\textrm{usia}}$
disebut *bobot*, dan $b$ disebut *bias*
(atau *offset* atau *intercept*).
Bobot menentukan pengaruh setiap fitur pada prediksi kita.
Bias menentukan nilai estimasi ketika semua fitur bernilai nol.
Meskipun kita tidak akan pernah menemukan rumah baru dengan luas tepat nol,
kita masih membutuhkan bias karena memungkinkan kita
untuk mengekspresikan semua fungsi linear dari fitur kita
(tanpa membatasi kita pada garis yang melalui titik asal).
Secara ketat, :eqref:`eq_price-area` adalah *transformasi afin* dari fitur input, yang dicirikan oleh *transformasi linear* dari fitur melalui penjumlahan berbobot, dikombinasikan dengan *translasi* melalui bias tambahan.
Diberi sebuah dataset, tujuan kita adalah memilih
bobot $\mathbf{w}$ dan bias $b$
yang, secara rata-rata, membuat prediksi model kita
mendekati harga sebenarnya yang diamati dalam data sebaik mungkin.

Dalam disiplin yang biasa berfokus
pada dataset dengan hanya beberapa fitur,
mengekspresikan model dalam bentuk panjang,
seperti pada :eqref:`eq_price-area`, adalah hal umum.
Dalam pembelajaran mesin, kita biasanya bekerja
dengan dataset berdimensi tinggi,
sehingga lebih nyaman menggunakan notasi aljabar linear yang ringkas.
Ketika input kita terdiri dari $d$ fitur,
kita dapat memberi masing-masing indeks (antara $1$ dan $d$)
dan mengekspresikan prediksi kita $\hat{y}$
(secara umum, simbol "hat" menunjukkan estimasi) sebagai

$$\hat{y} = w_1  x_1 + \cdots + w_d  x_d + b.$$

Dengan mengumpulkan semua fitur ke dalam vektor $\mathbf{x} \in \mathbb{R}^d$
dan semua bobot ke dalam vektor $\mathbf{w} \in \mathbb{R}^d$,
kita dapat mengekspresikan model kita secara ringkas melalui dot product
antara $\mathbf{w}$ dan $\mathbf{x}$:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

Dalam :eqref:`eq_linreg-y`, vektor $\mathbf{x}$
merupakan fitur dari satu contoh.
Kita sering menemukan bahwa lebih nyaman
merujuk pada fitur dari seluruh dataset berjumlah $n$ contoh
melalui *design matrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$.
Di sini, $\mathbf{X}$ memiliki satu baris untuk setiap contoh
dan satu kolom untuk setiap fitur.
Untuk sekumpulan fitur $\mathbf{X}$,
prediksi $\hat{\mathbf{y}} \in \mathbb{R}^n$
dapat diekspresikan melalui perkalian matriks-vektor:

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$
:eqlabel:`eq_linreg-y-vec`

dengan broadcasting (:numref:`subsec_broadcasting`) yang diterapkan selama penjumlahan.
Diberi fitur dari dataset pelatihan $\mathbf{X}$
dan label yang sesuai (diketahui) $\mathbf{y}$,
tujuan regresi linear adalah menemukan
vektor bobot $\mathbf{w}$ dan nilai bias $b$
sehingga, dengan fitur dari contoh data baru
yang diambil dari distribusi yang sama dengan $\mathbf{X}$,
label dari contoh baru akan (dalam ekspektasi)
dapat diprediksi dengan kesalahan terkecil.

Meskipun kita percaya bahwa model terbaik untuk
memprediksi $y$ diberi $\mathbf{x}$ adalah linear,
kita tidak akan mengharapkan untuk menemukan dataset dunia nyata dengan $n$ contoh di mana
$y^{(i)}$ persis sama dengan $\mathbf{w}^\top \mathbf{x}^{(i)}+b$
untuk semua $1 \leq i \leq n$.
Misalnya, alat apa pun yang kita gunakan untuk mengamati
fitur $\mathbf{X}$ dan label $\mathbf{y}$, mungkin ada sedikit kesalahan pengukuran.
Oleh karena itu, meskipun kita yakin
bahwa hubungan dasarnya adalah linear,
kita akan memasukkan istilah noise untuk memperhitungkan kesalahan semacam itu.

Sebelum kita bisa mulai mencari *parameter* terbaik
(atau *parameter model*) $\mathbf{w}$ dan $b$,
kita membutuhkan dua hal lagi:
(i) ukuran kualitas dari model tertentu;
dan (ii) prosedur untuk memperbarui model untuk meningkatkan kualitasnya.

### Fungsi Kerugian
:label:`subsec_linear-regression-loss-function`

Secara alami, menyesuaikan model kita dengan data memerlukan
kesepakatan tentang ukuran *kecocokan*
(atau, secara ekuivalen, *ketidakcocokan*).
*Fungsi kerugian* mengukur jarak
antara nilai *sebenarnya* dan *prediksi* dari target.
Kerugian biasanya berupa angka non-negatif
di mana nilai yang lebih kecil lebih baik
dan prediksi yang sempurna menghasilkan kerugian sebesar 0.
Untuk masalah regresi, fungsi kerugian yang paling umum adalah kesalahan kuadrat.
Ketika prediksi kita untuk contoh $i$ adalah $\hat{y}^{(i)}$
dan label sebenarnya yang sesuai adalah $y^{(i)}$,
*maka kesalahan kuadrat* diberikan oleh:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

Konstanta $\frac{1}{2}$ sebenarnya tidak memiliki pengaruh signifikan
tetapi berguna secara notasi,
karena akan hilang saat kita mengambil turunan dari kerugian.
Karena dataset pelatihan sudah diberikan kepada kita,
dan dengan demikian di luar kendali kita,
kesalahan empiris hanya merupakan fungsi dari parameter model.
Pada :numref:`fig_fit_linreg`, kita memvisualisasikan kecocokan model regresi linear
dalam masalah dengan input satu dimensi.

![Mencocokkan model regresi linear dengan data satu dimensi.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

Perhatikan bahwa perbedaan besar antara
estimasi $\hat{y}^{(i)}$ dan target $y^{(i)}$
menyebabkan kontribusi yang lebih besar terhadap kerugian,
karena bentuk kuadratnya
(bentuk kuadrat ini bisa menjadi pedang bermata dua; meskipun mendorong model untuk menghindari kesalahan besar
ini juga bisa menyebabkan sensitivitas berlebih terhadap data yang menyimpang).
Untuk mengukur kualitas model pada seluruh dataset yang berjumlah $n$ contoh,
kita cukup menghitung rata-rata (atau secara ekuivalen, jumlah)
kerugian pada set pelatihan:

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Ketika melatih model, kita mencari parameter ($\mathbf{w}^*, b^*$)
yang meminimalkan total kerugian pada seluruh contoh pelatihan:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Solusi Analitik

Tidak seperti sebagian besar model yang akan kita bahas,
regresi linear memberikan kita
masalah optimasi yang sangat mudah.
Secara khusus, kita dapat menemukan parameter optimal
(seperti yang dinilai pada data pelatihan)
secara analitik dengan menerapkan rumus sederhana sebagai berikut.
Pertama, kita dapat memasukkan bias $b$ ke dalam parameter $\mathbf{w}$
dengan menambahkan kolom ke design matrix yang berisi semua nilai 1.
Kemudian masalah prediksi kita adalah meminimalkan $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$.
Selama design matrix $\mathbf{X}$ memiliki peringkat penuh
(tidak ada fitur yang linier tergantung pada yang lain),
maka akan ada hanya satu titik kritis pada permukaan kerugian
dan titik tersebut sesuai dengan minimum kerugian pada seluruh domain.
Mengambil turunan dari kerugian terhadap $\mathbf{w}$
dan menyetarakannya dengan nol menghasilkan:

$$\begin{aligned}
    \partial_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 =
    2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0
    \textrm{ dan karena itu }
    \mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}.
\end{aligned}$$

Menyelesaikan untuk $\mathbf{w}$ memberikan kita solusi optimal
untuk masalah optimasi.
Perhatikan bahwa solusi ini 

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$$

hanya akan unik
ketika matriks $\mathbf X^\top \mathbf X$ dapat dibalik,
yaitu, ketika kolom dari design matrix
linier tidak saling bergantung :cite:`Golub.Van-Loan.1996`.


Meskipun masalah sederhana seperti regresi linear
dapat memiliki solusi analitik,
Anda sebaiknya tidak terbiasa dengan keberuntungan seperti ini.
Meskipun solusi analitik memungkinkan analisis matematis yang indah,
persyaratan solusi analitik sangat ketat
sehingga akan mengesampingkan hampir semua aspek menarik dari pembelajaran mendalam.

### Minibatch Stochastic Gradient Descent

Untungnya, bahkan dalam kasus di mana kita tidak dapat menyelesaikan model secara analitik,
kita masih sering dapat melatih model dengan efektif dalam praktik.
Terlebih lagi, untuk banyak tugas, model yang sulit dioptimalkan tersebut
ternyata jauh lebih baik sehingga mencari cara untuk melatihnya
menjadi sepadan dengan usaha.

Teknik utama untuk mengoptimalkan hampir setiap model pembelajaran mendalam,
dan yang akan kita gunakan sepanjang buku ini,
terdiri dari pengurangan kesalahan secara iteratif
dengan memperbarui parameter ke arah
yang secara bertahap mengurangi fungsi kerugian.
Algoritma ini disebut *gradient descent*.

Penerapan gradient descent yang paling sederhana
terdiri dari mengambil turunan dari fungsi kerugian,
yang merupakan rata-rata dari kerugian yang dihitung
pada setiap contoh dalam dataset.
Dalam praktiknya, ini bisa sangat lambat:
kita harus melalui seluruh dataset sebelum melakukan satu pembaruan,
meskipun langkah pembaruan mungkin sangat kuat :cite:`Liu.Nocedal.1989`.
Lebih buruk lagi, jika ada banyak redundansi dalam data pelatihan,
manfaat dari pembaruan penuh terbatas.

Di sisi lain, kita bisa mempertimbangkan hanya satu contoh pada satu waktu
dan melakukan langkah pembaruan berdasarkan satu pengamatan saja.
Algoritma yang dihasilkan, *stochastic gradient descent* (SGD)
bisa menjadi strategi yang efektif :cite:`Bottou.2010`, bahkan untuk dataset besar.
Sayangnya, SGD memiliki kelemahan, baik dari segi komputasi maupun statistik.
Salah satu masalah muncul dari fakta bahwa prosesor jauh lebih cepat
dalam melakukan operasi perkalian dan penjumlahan angka
dibandingkan memindahkan data dari memori utama ke cache prosesor.
Jauh lebih efisien untuk
melakukan perkalian matriks-vektor
daripada sejumlah operasi vektor-vektor yang setara.
Ini berarti bahwa memproses satu sampel pada satu waktu bisa memakan waktu jauh lebih lama dibandingkan batch penuh.
Masalah kedua adalah bahwa beberapa lapisan,
seperti normalisasi batch (yang akan dijelaskan di :numref:`sec_batch_norm`),
hanya bekerja dengan baik saat kita memiliki
lebih dari satu pengamatan pada satu waktu.

Solusi untuk kedua masalah ini adalah dengan memilih strategi menengah:
alih-alih mengambil batch penuh atau hanya satu sampel pada satu waktu,
kita mengambil *minibatch* dari beberapa pengamatan :cite:`Li.Zhang.Chen.ea.2014`.
Pilihan ukuran minibatch bergantung pada banyak faktor,
seperti jumlah memori, jumlah akselerator,
pilihan lapisan, dan ukuran total dataset.
Meskipun demikian, ukuran antara 32 dan 256,
sebaiknya kelipatan dari pangkat besar $2$, adalah awal yang baik.
Ini membawa kita pada *minibatch stochastic gradient descent*.

Dalam bentuknya yang paling dasar, pada setiap iterasi $t$,
kita pertama-tama mengambil sampel secara acak sebuah minibatch $\mathcal{B}_t$
yang terdiri dari sejumlah tetap $|\mathcal{B}|$ contoh pelatihan.
Kemudian kita menghitung turunan (gradien) dari kerugian rata-rata
pada minibatch tersebut terhadap parameter model.
Terakhir, kita mengalikan gradien
dengan nilai positif kecil yang telah ditentukan $\eta$,
yang disebut *learning rate*,
dan mengurangkan hasilnya dari nilai parameter saat ini.
Kita dapat mengekspresikan pembaruan ini sebagai berikut:


$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

Secara ringkas, minibatch SGD berjalan sebagai berikut:
(i) menginisialisasi nilai parameter model, biasanya secara acak;
(ii) mengambil minibatch secara acak dari data secara iteratif,
memperbarui parameter ke arah gradien negatif.
Untuk kerugian kuadrat dan transformasi afin,
ini memiliki bentuk ekspansi tertutup:

$$\begin{aligned} \mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_b l^{(i)}(\mathbf{w}, b) &&  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Karena kita memilih sebuah minibatch $\mathcal{B}$,
kita perlu menormalisasi dengan ukurannya $|\mathcal{B}|$.
Seringkali ukuran minibatch dan learning rate ditentukan oleh pengguna.
Parameter yang dapat diatur ini, yang tidak diperbarui
di dalam loop pelatihan, disebut *hyperparameter*.
Hyperparameter dapat disetel secara otomatis dengan sejumlah teknik, seperti optimisasi Bayesian
:cite:`Frazier.2018`. Pada akhirnya, kualitas solusi biasanya dievaluasi
menggunakan *validation dataset* terpisah (atau *validation set*).

Setelah melatih model untuk jumlah iterasi yang telah ditentukan
(atau hingga kriteria pemberhentian lain terpenuhi),
kita mencatat parameter model yang diestimasi,
dinyatakan sebagai $\hat{\mathbf{w}}, \hat{b}$.
Perlu dicatat bahwa bahkan jika fungsi kita benar-benar linear dan bebas noise,
parameter ini tidak akan menjadi minimizer loss yang tepat, bahkan tidak deterministik.
Meskipun algoritma perlahan konvergen ke minimizer,
umumnya tidak akan menemukannya secara tepat dalam jumlah langkah yang terbatas.
Selain itu, minibatch $\mathcal{B}$
yang digunakan untuk memperbarui parameter dipilih secara acak.
Ini membuat prosesnya tidak deterministik.

Regresi linear kebetulan merupakan masalah pembelajaran
dengan minimum global
(ketika $\mathbf{X}$ memiliki peringkat penuh, atau secara ekuivalen,
ketika $\mathbf{X}^\top \mathbf{X}$ dapat dibalik).
Namun, permukaan kerugian untuk jaringan dalam mengandung banyak saddle point dan minimum.
Untungnya, kita biasanya tidak peduli untuk menemukan
seperangkat parameter yang benar-benar tepat tetapi hanya membutuhkan parameter
yang menghasilkan prediksi akurat (dan dengan demikian kerugian rendah).
Dalam praktiknya, praktisi pembelajaran mendalam
jarang kesulitan menemukan parameter
yang meminimalkan kerugian *pada set pelatihan*
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018,Frankle.Carbin.2018`.
Tugas yang lebih menantang adalah menemukan parameter
yang menghasilkan prediksi akurat pada data yang belum pernah dilihat sebelumnya,
sebuah tantangan yang disebut *generalization*.
Kita akan kembali ke topik ini sepanjang buku ini.

### Prediksi

Dengan model $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$,
kita sekarang dapat membuat *prediksi* untuk contoh baru,
misalnya, memprediksi harga jual dari rumah yang belum pernah dilihat
berdasarkan luas $x_1$ dan usia $x_2$.
Praktisi pembelajaran mendalam sering menyebut fase prediksi ini sebagai *inference*,
tetapi ini sedikit keliru—*inference* secara umum merujuk
pada kesimpulan yang diambil berdasarkan bukti,
termasuk nilai parameter
dan label yang mungkin untuk instance yang belum terlihat.
Bahkan, dalam literatur statistik,
*inference* lebih sering merujuk pada inferensi parameter,
dan penggunaan istilah ini secara berlebihan menciptakan kebingungan yang tidak perlu
ketika praktisi pembelajaran mendalam berbicara dengan ahli statistik.
Di bagian selanjutnya, kita akan menggunakan istilah *prediksi* sebanyak mungkin.




## Vektorisasi untuk Kecepatan

Saat melatih model kita, kita biasanya ingin memproses
seluruh minibatch dari contoh secara bersamaan.
Melakukan hal ini secara efisien membutuhkan bahwa (**kita**)
(**memvektorisasi perhitungan dan memanfaatkan
perpustakaan aljabar linear yang cepat
daripada menulis loop for yang memakan waktu di Python.**)

Untuk melihat mengapa ini sangat penting,
mari (**pertimbangkan dua metode untuk menjumlahkan vektor.**)
Sebagai permulaan, kita membuat dua vektor berdimensi 10.000
yang masing-masing berisi nilai 1.
Pada metode pertama, kita melakukan loop pada vektor dengan loop for di Python.
Pada metode kedua, kita mengandalkan satu pemanggilan `+`.


```{.python .input}
%%tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Sekarang kita dapat membandingkan beban kerjanya.
Pertama, [**kita menjumlahkan vektor tersebut, satu koordinat pada satu waktu,
menggunakan loop for.**]


```{.python .input}
%%tab mxnet, pytorch
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab tensorflow
c = tf.Variable(d2l.zeros(n))
t = time.time()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab jax
# Array JAX bersifat immutable, artinya setelah dibuat, isinya
# tidak dapat diubah. Untuk memperbarui elemen individu, JAX menyediakan
# sintaks pembaruan terindeks yang mengembalikan salinan yang telah diperbarui.
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c = c.at[i].set(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```
(**Sebagai alternatif, kita mengandalkan operator `+` yang di-overload untuk menghitung jumlah elemen secara langsung.**)

```{.python .input}
%%tab all
t = time.time()
d = a + b
f'{time.time() - t:.5f} sec'
```

Metode kedua secara dramatis lebih cepat daripada metode pertama.
Melvektorisasi kode sering kali menghasilkan peningkatan kecepatan secara signifikan.
Selain itu, kita menyerahkan lebih banyak perhitungan matematika ke perpustakaan,
sehingga kita tidak perlu menulis banyak perhitungan sendiri,
mengurangi potensi kesalahan dan meningkatkan portabilitas kode.


## Distribusi Normal dan Kerugian Kuadrat
:label:`subsec_normal_distribution_and_squared_loss`

Sejauh ini kita telah memberikan motivasi fungsional
untuk tujuan kerugian kuadrat:
parameter optimal mengembalikan ekspektasi kondisional $E[Y\mid X]$
ketika pola dasarnya benar-benar linear,
dan fungsi kerugian memberikan penalti besar untuk outlier.
Kita juga dapat memberikan motivasi yang lebih formal
untuk tujuan kerugian kuadrat
dengan membuat asumsi probabilistik
tentang distribusi noise.

Regresi linear ditemukan pada pergantian abad ke-19.
Meskipun telah lama diperdebatkan apakah Gauss atau Legendre
yang pertama kali memunculkan ide tersebut,
Gauss juga menemukan distribusi normal
(yang juga disebut *Gaussian*).
Ternyata, distribusi normal
dan regresi linear dengan kerugian kuadrat
memiliki hubungan yang lebih dalam daripada sekadar kemiripan sejarah.

Sebagai permulaan, ingat bahwa distribusi normal
dengan mean $\mu$ dan varians $\sigma^2$ (simpangan baku $\sigma$)
dinyatakan sebagai

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Berikut ini [**kita mendefinisikan sebuah fungsi untuk menghitung distribusi normal**].


```{.python .input}
%%tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    if tab.selected('jax'):
        return p * jnp.exp(-0.5 * (x - mu)**2 / sigma**2)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
```

Sekarang kita dapat (**memvisualisasikan distribusi normal**).


```{.python .input}
%%tab mxnet
# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.python .input}

%%tab pytorch, tensorflow, jax
if tab.selected('jax'):
    # Use JAX NumPy for visualization
    x = jnp.arange(-7, 7, 0.01)
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    # Use NumPy again for visualization
    x = np.arange(-7, 7, 0.01)

# Pasangan mean dan simpangan baku
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

Perhatikan bahwa mengubah mean berhubungan dengan
pergeseran sepanjang sumbu $x$,
dan meningkatkan varians
menyebarkan distribusi, menurunkan puncaknya.

Salah satu cara untuk memotivasi regresi linear dengan kerugian kuadrat
adalah dengan mengasumsikan bahwa pengamatan berasal dari pengukuran dengan noise,
di mana noise $\epsilon$ mengikuti distribusi normal
$\mathcal{N}(0, \sigma^2)$:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \textrm{ di mana } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Dengan demikian, kita sekarang dapat menuliskan *likelihood*
untuk melihat nilai tertentu $y$ untuk $\mathbf{x}$ yang diberikan melalui

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Dengan demikian, likelihood dapat difaktorkan.
Menurut *prinsip maximum likelihood*,
nilai terbaik untuk parameter $\mathbf{w}$ dan $b$ adalah nilai-nilai
yang memaksimalkan *likelihood* dari seluruh dataset:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)} \mid \mathbf{x}^{(i)}).$$

Persamaan ini mengikuti karena semua pasangan $(\mathbf{x}^{(i)}, y^{(i)})$
diambil secara independen satu sama lain.
Estimator yang dipilih berdasarkan prinsip maximum likelihood
disebut *maximum likelihood estimators*.
Meskipun memaksimalkan hasil perkalian dari banyak fungsi eksponensial
terlihat sulit,
kita dapat menyederhanakan hal tersebut secara signifikan, tanpa mengubah tujuan,
dengan memaksimalkan logaritma dari likelihood.
Karena alasan historis, optimisasi lebih sering diekspresikan
sebagai minimisasi daripada maksimisasi.
Jadi, tanpa mengubah apa pun,
kita dapat *meminimalkan* *negative log-likelihood*,
yang dapat kita nyatakan sebagai berikut:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Jika kita mengasumsikan bahwa $\sigma$ tetap,
kita dapat mengabaikan suku pertama,
karena tidak bergantung pada $\mathbf{w}$ atau $b$.
Suku kedua identik
dengan kerugian kuadrat yang diperkenalkan sebelumnya,
kecuali untuk konstanta perkalian $\frac{1}{\sigma^2}$.
Untungnya, solusinya tidak bergantung pada $\sigma$ juga.
Ini berarti bahwa meminimalkan mean squared error
setara dengan estimasi maximum likelihood
dari model linear di bawah asumsi adanya noise Gaussian aditif.


## Regresi Linear sebagai Jaringan Neural

Meskipun model linear tidak cukup kaya
untuk mengekspresikan berbagai jaringan rumit
yang akan kita perkenalkan dalam buku ini,
jaringan neural (buatan) cukup kaya
untuk memasukkan model linear sebagai jaringan
di mana setiap fitur diwakili oleh neuron input,
yang semuanya terhubung langsung ke output.

:numref:`fig_single_neuron` menggambarkan
regresi linear sebagai jaringan neural.
Diagram ini menyoroti pola konektivitas,
seperti bagaimana setiap input terhubung ke output,
tetapi tidak menunjukkan nilai spesifik dari bobot atau bias.

![Regresi linear adalah jaringan neural satu lapisan.](../img/singleneuron.svg)
:label:`fig_single_neuron`

Inputnya adalah $x_1, \ldots, x_d$.
Kita menyebut $d$ sebagai *jumlah input*
atau *dimensi fitur* pada lapisan input.
Output dari jaringan adalah $o_1$.
Karena kita hanya mencoba memprediksi
satu nilai numerik,
kita hanya memiliki satu neuron output.
Perhatikan bahwa nilai input semua *diberikan*.
Hanya ada satu neuron yang *dihitung*.
Singkatnya, kita dapat memandang regresi linear
sebagai jaringan neural satu lapisan dengan koneksi penuh.
Kita akan menjumpai jaringan
dengan lebih banyak lapisan
di bab-bab berikutnya.

### Biologi

Karena regresi linear muncul sebelum ilmu saraf komputasi,
mungkin terlihat aneh untuk menggambarkan
regresi linear dalam istilah jaringan neural.
Namun demikian, ini adalah tempat yang alami untuk memulai
ketika para ahli sibernetika dan neurofisiologi
Warren McCulloch dan Walter Pitts mulai mengembangkan
model neuron buatan.
Pertimbangkan gambar kartun
dari neuron biologis pada :numref:`fig_Neuron`,
terdiri dari *dendrit* (terminal input),
*nukleus* (CPU), *akson* (kabel output),
dan *terminal akson* (terminal output),
yang memungkinkan koneksi ke neuron lain melalui *sinapsis*.

![Neuron asli (sumber: "Anatomy and Physiology" oleh US National Cancer Institute's Surveillance, Epidemiology and End Results (SEER) Program).](../img/neuron.svg)
:label:`fig_Neuron`

Informasi $x_i$ yang datang dari neuron lain
(atau sensor lingkungan) diterima di dendrit.
Secara khusus, informasi tersebut diberi bobot
oleh *bobot sinapsis* $w_i$,
yang menentukan efek dari input,
misalnya, aktivasi atau inhibisi melalui produk $x_i w_i$.
Input berbobot yang datang dari berbagai sumber
diagregasi di nukleus
sebagai jumlah berbobot $y = \sum_i x_i w_i + b$,
mungkin melalui pemrosesan nonlinier melalui fungsi $\sigma(y)$.
Informasi ini kemudian dikirim melalui akson ke terminal akson,
di mana ia mencapai tujuannya
(misalnya, penggerak seperti otot)
atau diteruskan ke neuron lain melalui dendritnya.

Tentu saja, ide tingkat tinggi bahwa banyak unit seperti ini
dapat digabungkan, asalkan memiliki konektivitas dan algoritma pembelajaran yang tepat,
untuk menghasilkan perilaku yang jauh lebih menarik dan kompleks
daripada yang bisa diungkapkan oleh satu neuron saja
berasal dari studi kita tentang sistem neural biologis nyata.
Pada saat yang sama, sebagian besar penelitian dalam pembelajaran mendalam saat ini
mengambil inspirasi dari sumber yang jauh lebih luas.
Kita mengutip :citet:`Russell.Norvig.2016`
yang menunjukkan bahwa meskipun pesawat terbang mungkin *terinspirasi* oleh burung,
ornitologi bukanlah pendorong utama
dari inovasi aeronautika selama beberapa abad terakhir.
Demikian pula, inspirasi dalam pembelajaran mendalam saat ini
datang dalam ukuran yang sama atau lebih besar
dari matematika, linguistik, psikologi,
statistik, ilmu komputer, dan banyak bidang lainnya.


## Ringkasan

Di bagian ini, kita memperkenalkan
regresi linear tradisional,
di mana parameter dari fungsi linear
dipilih untuk meminimalkan kerugian kuadrat pada set pelatihan.
Kami juga memotivasi pemilihan tujuan ini
baik melalui beberapa pertimbangan praktis
maupun melalui interpretasi
regresi linear sebagai estimasi maximum likelihood
dengan asumsi linearitas dan noise Gaussian.
Setelah membahas pertimbangan komputasi
dan koneksi ke statistik,
kita menunjukkan bagaimana model linear semacam itu dapat diekspresikan
sebagai jaringan neural sederhana di mana input
terhubung langsung ke output.
Meskipun kita akan segera beralih dari model linear,
model ini cukup untuk memperkenalkan sebagian besar komponen
yang dibutuhkan semua model kita:
bentuk parametrik, tujuan yang dapat didiferensiasikan,
optimisasi melalui minibatch stochastic gradient descent,
dan evaluasi pada data yang belum pernah dilihat sebelumnya.

## Latihan

1. Asumsikan bahwa kita memiliki data $x_1, \ldots, x_n \in \mathbb{R}$. Tujuan kita adalah menemukan konstanta $b$ sedemikian sehingga $\sum_i (x_i - b)^2$ diminimalkan.
    1. Temukan solusi analitik untuk nilai optimal $b$.
    1. Bagaimana masalah ini dan solusinya terkait dengan distribusi normal?
    1. Bagaimana jika kita mengubah kerugian dari $\sum_i (x_i - b)^2$ menjadi $\sum_i |x_i-b|$? Bisakah Anda menemukan solusi optimal untuk $b$?
1. Buktikan bahwa fungsi afin yang dapat diekspresikan dengan $\mathbf{x}^\top \mathbf{w} + b$ setara dengan fungsi linear pada $(\mathbf{x}, 1)$.
1. Asumsikan bahwa Anda ingin menemukan fungsi kuadrat dari $\mathbf{x}$, yaitu $f(\mathbf{x}) = b + \sum_i w_i x_i + \sum_{j \leq i} w_{ij} x_{i} x_{j}$. Bagaimana Anda akan memformulasikan ini dalam jaringan yang dalam?
1. Ingat bahwa salah satu kondisi agar masalah regresi linear dapat diselesaikan adalah bahwa design matrix $\mathbf{X}^\top \mathbf{X}$ memiliki peringkat penuh.
    1. Apa yang terjadi jika tidak demikian?
    1. Bagaimana Anda bisa memperbaikinya? Apa yang terjadi jika Anda menambahkan sedikit noise Gaussian independen pada setiap entri dari $\mathbf{X}$?
    1. Berapakah nilai ekspektasi dari design matrix $\mathbf{X}^\top \mathbf{X}$ dalam kasus ini?
    1. Apa yang terjadi dengan stochastic gradient descent ketika $\mathbf{X}^\top \mathbf{X}$ tidak memiliki peringkat penuh?
1. Asumsikan bahwa model noise yang mengatur noise aditif $\epsilon$ adalah distribusi eksponensial. Artinya, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Tuliskan negative log-likelihood dari data di bawah model $-\log P(\mathbf y \mid \mathbf X)$.
    1. Bisakah Anda menemukan solusi dalam bentuk tertutup?
    1. Usulkan algoritma minibatch stochastic gradient descent untuk menyelesaikan masalah ini. Apa yang mungkin salah (petunjuk: apa yang terjadi di sekitar titik stasioner saat kita terus memperbarui parameter)? Bisakah Anda memperbaikinya?
1. Asumsikan bahwa kita ingin merancang jaringan neural dengan dua lapisan dengan menggabungkan dua lapisan linear. Artinya, output dari lapisan pertama menjadi input dari lapisan kedua. Mengapa komposisi naif seperti ini tidak akan berhasil?
1. Apa yang terjadi jika Anda ingin menggunakan regresi untuk estimasi harga rumah atau harga saham yang realistis?
    1. Tunjukkan bahwa asumsi noise Gaussian aditif tidak sesuai. Petunjuk: bisakah kita memiliki harga negatif? Bagaimana dengan fluktuasi?
    1. Mengapa regresi pada logaritma harga lebih baik, yaitu $y = \log \textrm{harga}$?
    1. Apa yang perlu Anda perhatikan saat menangani saham bernilai rendah (pennystock), yaitu saham dengan harga sangat rendah? Petunjuk: bisakah Anda melakukan perdagangan pada semua harga yang mungkin? Mengapa ini menjadi masalah yang lebih besar untuk saham murah? Untuk informasi lebih lanjut, tinjau model Black-Scholes yang terkenal untuk penetapan harga opsi :cite:`Black.Scholes.1973`.
1. Misalkan kita ingin menggunakan regresi untuk memperkirakan *jumlah* apel yang terjual di sebuah toko.
    1. Apa masalahnya dengan model noise Gaussian aditif? Petunjuk: Anda menjual apel, bukan minyak.
    1. [Distribusi Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) menggambarkan distribusi pada jumlah. Distribusi ini diberikan oleh $p(k \mid \lambda) = \lambda^k e^{-\lambda}/k!$. Di sini $\lambda$ adalah fungsi laju dan $k$ adalah jumlah kejadian yang Anda lihat. Buktikan bahwa $\lambda$ adalah nilai ekspektasi dari jumlah $k$.
    1. Rancang fungsi kerugian yang terkait dengan distribusi Poisson.
    1. Rancang fungsi kerugian untuk memperkirakan $\log \lambda$.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/259)
:end_tab:
