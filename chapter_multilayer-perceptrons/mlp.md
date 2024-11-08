```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Multilayer Perceptrons 
:label:`sec_mlp`

Di :numref:`sec_softmax`, kita memperkenalkan
softmax regression,
mengimplementasikan algoritma dari awal
(:numref:`sec_softmax_scratch`) dan menggunakan API tingkat tinggi
(:numref:`sec_softmax_concise`). Ini memungkinkan kita
melatih classifier yang mampu mengenali
10 kategori pakaian dari gambar beresolusi rendah.
Sepanjang perjalanan, kita belajar cara mengelola data,
mengubah output menjadi distribusi probabilitas yang valid,
menerapkan fungsi loss yang sesuai,
dan meminimalkannya terhadap parameter model kita.
Sekarang setelah kita menguasai mekanisme ini
dalam konteks model linear sederhana,
kita dapat memulai eksplorasi kita tentang deep neural network,
kelas model yang lebih kaya secara komparatif
yang menjadi fokus utama dari buku ini.


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Hidden Layers (lapisan tersembunyi)

Kita mendeskripsikan transformasi afine di
:numref:`subsec_linear_model` sebagai
transformasi linear dengan penambahan bias.
Untuk memulai, ingat kembali arsitektur model
yang sesuai dengan contoh softmax regression kita,
seperti yang diilustrasikan pada :numref:`fig_softmaxreg`.
Model ini memetakan input langsung ke output
melalui satu transformasi afine,
diikuti dengan operasi softmax.
Jika label kita benar-benar terkait
dengan data input melalui transformasi afine sederhana,
pendekatan ini akan cukup.
Namun, linearitas (dalam transformasi afine) adalah asumsi yang *kuat*.

### kekurangan dari Linear Models

Misalnya, linearitas menyiratkan asumsi yang lebih *lemah*
yaitu *monotonisitas*, yaitu,
bahwa peningkatan dalam fitur kita
selalu menyebabkan peningkatan dalam output model kita
(jika bobot yang sesuai positif),
atau selalu menyebabkan penurunan dalam output model kita
(jika bobot yang sesuai negatif).
Kadang-kadang asumsi ini masuk akal.
Sebagai contoh, jika kita mencoba memprediksi
apakah seseorang akan membayar kembali pinjaman,
kita mungkin berasumsi bahwa semua hal lain sama,
pelamar dengan penghasilan yang lebih tinggi
selalu akan lebih mungkin untuk membayar kembali
dibandingkan dengan yang berpenghasilan lebih rendah.
Walaupun monoton, hubungan ini mungkin
tidak secara linear terkait dengan probabilitas
pembayaran kembali. Peningkatan penghasilan dari \$0 menjadi \$50.000
kemungkinan besar berhubungan dengan peningkatan yang lebih besar
dalam kemungkinan pembayaran kembali
dibandingkan dengan peningkatan dari \$1 juta menjadi \$1,05 juta.
Salah satu cara untuk menangani ini mungkin adalah dengan memproses ulang hasil kita
sehingga linearitas menjadi lebih masuk akal,
dengan menggunakan peta logistik (dan dengan demikian logaritma dari probabilitas hasil).

Perhatikan bahwa kita dapat dengan mudah menemukan contoh
yang melanggar monotonisitas.
Misalnya jika kita ingin memprediksi kesehatan berdasarkan
suhu tubuh.
Untuk individu dengan suhu tubuh normal
di atas 37°C (98,6°F),
suhu yang lebih tinggi menunjukkan risiko yang lebih besar.
Namun, jika suhu tubuh turun
di bawah 37°C, suhu yang lebih rendah menunjukkan risiko yang lebih besar!
Sekali lagi, kita bisa menyelesaikan masalah ini
dengan pra-pemrosesan yang cerdas, seperti menggunakan jarak dari 37°C
sebagai fitur.


Tetapi bagaimana dengan mengklasifikasikan gambar kucing dan anjing?
Apakah meningkatkan intensitas
piksel pada lokasi (13, 17)
selalu meningkatkan (atau selalu menurunkan)
kemungkinan bahwa gambar tersebut menunjukkan anjing?
Ketergantungan pada model linear berhubungan dengan asumsi implisit
bahwa satu-satunya persyaratan
untuk membedakan kucing dan anjing adalah menilai
kecerahan dari setiap piksel.
Pendekatan ini pasti akan gagal di dunia
di mana membalikkan gambar mempertahankan kategori.

Dan meskipun ketidakmasukakalan linearitas terlihat jelas di sini,
dibandingkan dengan contoh-contoh sebelumnya,
tidak terlalu jelas bahwa kita bisa menyelesaikan masalah ini
dengan pra-pemrosesan sederhana.
Artinya, karena signifikansi dari setiap piksel
bergantung dengan cara yang kompleks pada konteksnya
(nilai piksel-piksel di sekitarnya).
Meskipun mungkin ada representasi dari data kita
yang memperhitungkan
interaksi relevan di antara fitur-fitur kita,
di atasnya model linear akan cocok,
kita tidak tahu cara menghitungnya secara manual.
Dengan jaringan saraf dalam, kita menggunakan data observasi
untuk secara bersama-sama belajar baik representasi melalui hidden layer
maupun prediktor linear yang bekerja pada representasi tersebut.

Masalah non-linearitas ini telah dipelajari setidaknya
selama satu abad :cite:`Fisher.1928`. Misalnya, pohon keputusan
dalam bentuk paling dasarnya menggunakan serangkaian keputusan biner untuk
memutuskan keanggotaan kelas :cite:`quinlan2014c4`. Demikian juga, metode kernel
telah digunakan selama beberapa dekade untuk memodelkan ketergantungan non-linear
:cite:`Aronszajn.1950`. Hal ini telah ditemukan dalam
model spline nonparametrik :cite:`Wahba.1990` dan metode kernel
:cite:`Scholkopf.Smola.2002`. Hal ini juga sesuatu yang secara alami dipecahkan oleh otak. Bagaimanapun, neuron-neuron memberi input ke neuron-neuron lain yang,
kemudian, memberi input lagi ke neuron-neuron lain :cite:`Cajal.Azoulay.1894`.
Akibatnya, kita memiliki serangkaian transformasi yang relatif sederhana.

### Incorporating Hidden Layers

Kita dapat mengatasi keterbatasan model linear
dengan memasukkan satu atau lebih hidden layer.
Cara termudah untuk melakukan ini adalah dengan menumpuk
banyak fully connected layer di atas satu sama lain.
Setiap lapisan memberi input ke lapisan di atasnya,
hingga kita menghasilkan output.
Kita dapat menganggap $L-1$ lapisan pertama
sebagai representasi dan lapisan terakhir
sebagai prediktor linear.
Arsitektur ini sering disebut
sebagai *multilayer perceptron*,
yang sering disingkat sebagai *MLP* (:numref:`fig_mlp`).

![MLP dengan satu hidden layer berisi lima unit tersembunyi.](../img/mlp.svg)
:label:`fig_mlp`

MLP ini memiliki empat input, tiga output,
dan hidden layer-nya berisi lima unit tersembunyi.
Karena lapisan input tidak melibatkan perhitungan,
menghasilkan output dengan jaringan ini
memerlukan implementasi perhitungan
untuk hidden dan output layer;
dengan demikian, jumlah lapisan dalam MLP ini adalah dua.
Perhatikan bahwa kedua lapisan ini terhubung penuh.
Setiap input mempengaruhi setiap neuron di hidden layer,
dan masing-masing dari ini pada gilirannya mempengaruhi
setiap neuron di output layer. Namun, kita belum selesai.


### From Linear to Nonlinear

Seperti sebelumnya, kita menggunakan matriks $\mathbf{X} \in \mathbb{R}^{n \times d}$
untuk merepresentasikan sebuah minibatch yang berisi $n$ contoh di mana setiap contoh memiliki $d$ input (fitur).
Untuk MLP satu-hidden-layer di mana hidden layer memiliki $h$ unit tersembunyi,
kita menyatakan output dari hidden layer sebagai $\mathbf{H} \in \mathbb{R}^{n \times h}$
yang merupakan *representasi tersembunyi*.
Karena hidden dan output layer keduanya terhubung penuh,
kita memiliki bobot hidden layer $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ dan bias $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$,
serta bobot output layer $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ dan bias $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.
Dengan demikian, kita dapat menghitung output $\mathbf{O} \in \mathbb{R}^{n \times q}$
dari MLP satu-hidden-layer sebagai berikut:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Perhatikan bahwa setelah menambahkan hidden layer,
model kita sekarang membutuhkan kita untuk melacak dan memperbarui
sekumpulan parameter tambahan.
Jadi apa yang kita dapatkan sebagai gantinya?
Anda mungkin akan terkejut mengetahui bahwa—dalam model yang didefinisikan di atas—*kita
tidak mendapatkan apa-apa dari usaha kita*!
Alasannya jelas.
Unit tersembunyi di atas diberikan oleh
fungsi afine dari input,
dan output (sebelum softmax) hanya
merupakan fungsi afine dari unit tersembunyi.
Fungsi afine dari fungsi afine
sendiri adalah fungsi afine.
Selain itu, model linear kita sudah
mampu merepresentasikan setiap fungsi afine.

Untuk melihat ini secara formal, kita cukup menyederhanakan hidden layer dalam definisi di atas,
menghasilkan model satu-lapisan yang setara dengan parameter
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ dan $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

Untuk mewujudkan potensi arsitektur multilayer,
kita memerlukan satu elemen penting tambahan: sebuah
fungsi aktivasi *nonlinear* $\sigma$
yang diterapkan pada setiap unit tersembunyi
setelah transformasi afine. Sebagai contoh,
fungsi aktivasi ReLU (rectified linear unit) populer digunakan :cite:`Nair.Hinton.2010`
$\sigma(x) = \mathrm{max}(0, x)$ yang beroperasi pada argumennya secara elementwise.
Output dari fungsi aktivasi $\sigma(\cdot)$
disebut *aktivasi*.
Secara umum, dengan fungsi aktivasi di tempatnya,
kita tidak bisa lagi menyederhanakan MLP kita menjadi model linear:

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Karena setiap baris dalam $\mathbf{X}$ sesuai dengan satu contoh dalam minibatch,
dengan sedikit penyalahgunaan notasi, kita mendefinisikan non-linearitas
$\sigma$ yang diterapkan pada inputnya secara baris demi baris,
yaitu satu contoh pada satu waktu.
Perhatikan bahwa kita menggunakan notasi yang sama untuk softmax
saat kita menyatakan operasi baris demi baris di :numref:`subsec_softmax_vectorization`.
Fungsi aktivasi yang kita gunakan cukup sering diterapkan bukan hanya secara baris demi baris, tetapi
elementwise. Artinya, setelah menghitung bagian linear dari lapisan,
kita dapat menghitung setiap aktivasi
tanpa memperhatikan nilai yang diambil oleh unit tersembunyi lainnya.

Untuk membangun MLP yang lebih umum, kita dapat terus menumpuk
hidden layer seperti itu,
misalnya, $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
dan $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$,
satu di atas yang lain, menghasilkan model yang semakin ekspresif.


### Universal Approximators

Kita tahu bahwa otak mampu melakukan analisis statistik yang sangat canggih. Karena itu,
perlu kita tanyakan, seberapa *kuat* jaringan dalam bisa menjadi. Pertanyaan ini
telah dijawab beberapa kali, misalnya, dalam konteks MLP oleh :citet:`Cybenko.1989`,
dan dalam konteks ruang Hilbert kernel reproduksi oleh :citet:`micchelli1984interpolation` dengan cara yang bisa dilihat sebagai jaringan fungsi dasar radial (RBF) dengan satu hidden layer.
Hasil-hasil ini (dan hasil terkait lainnya) menunjukkan bahwa bahkan dengan jaringan satu-hidden-layer,
dengan jumlah node yang cukup (mungkin dalam jumlah yang sangat besar),
dan sekumpulan bobot yang tepat,
kita dapat memodelkan fungsi apapun.
Namun, mempelajari fungsi tersebut adalah bagian yang sulit.
Anda mungkin menganggap jaringan neural Anda
sedikit seperti bahasa pemrograman C.
Bahasa tersebut, seperti bahasa modern lainnya,
mampu mengekspresikan program yang dapat dihitung.
Namun, merancang program
yang memenuhi spesifikasi Anda adalah bagian yang sulit.

Selain itu, hanya karena jaringan satu-hidden-layer
*bisa* mempelajari fungsi apapun
bukan berarti Anda harus mencoba
menyelesaikan semua masalah Anda
dengan satu hidden layer. Faktanya, dalam kasus ini metode kernel
jauh lebih efektif, karena mereka mampu menyelesaikan masalah
*secara tepat* bahkan dalam ruang berdimensi tak hingga :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`.
Kita dapat mendekati banyak fungsi
dengan lebih efisien menggunakan jaringan yang lebih dalam (daripada lebih lebar) :cite:`Simonyan.Zisserman.2014`.
Kita akan membahas argumen yang lebih mendalam di bab-bab selanjutnya.


## Activation Functions
:label:`subsec_activation-functions`

Fungsi aktivasi memutuskan apakah sebuah neuron harus diaktifkan atau tidak
dengan menghitung jumlah bobot dan menambahkan bias padanya.
Fungsi ini merupakan operator diferensial untuk mentransformasi sinyal input menjadi output,
dengan sebagian besar dari mereka menambahkan non-linearitas.
Karena fungsi aktivasi sangat mendasar bagi deep learning,
(**mari kita tinjau secara singkat beberapa yang umum**).

### ReLU Function

Pilihan yang paling populer,
karena kesederhanaan implementasinya dan
kinerjanya yang baik pada berbagai tugas prediktif,
adalah *rectified linear unit* (*ReLU*) :cite:`Nair.Hinton.2010`.
[**ReLU memberikan transformasi non-linear yang sangat sederhana**].
Diberikan elemen $x$, fungsi ini didefinisikan
sebagai nilai maksimum dari elemen tersebut dan 0:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Secara informal, fungsi ReLU hanya mempertahankan elemen-elemen positif
dan mengabaikan semua elemen negatif
dengan menetapkan aktivasi yang bersangkutan menjadi 0.
Untuk mendapatkan intuisi, kita bisa memplot fungsi ini.
Seperti yang Anda lihat, fungsi aktivasi ini berbentuk linear secara bertahap.


```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

Ketika input negatif,
turunan fungsi ReLU adalah 0,
dan ketika input positif,
turunan fungsi ReLU adalah 1.
Perhatikan bahwa fungsi ReLU tidak terdiferensiasi
ketika input memiliki nilai yang tepat sama dengan 0.
Dalam kasus ini, kita menggunakan turunan sisi kiri
dan menyatakan bahwa turunan adalah 0 ketika inputnya adalah 0.
Kita bisa mengabaikannya karena
input mungkin tidak pernah benar-benar nol (ahli matematika akan
mengatakan bahwa ia tidak terdiferensiasi pada himpunan dengan ukuran nol).
Ada sebuah pepatah lama yang mengatakan bahwa jika kondisi batas yang halus penting,
kita mungkin sedang melakukan (*matematika*) yang nyata, bukan rekayasa.
Kebijaksanaan konvensional tersebut mungkin berlaku di sini, atau paling tidak, fakta bahwa
kita tidak melakukan optimisasi terbatas :cite:`Mangasarian.1965,Rockafellar.1970`.
Kami memplot turunan dari fungsi ReLU di bawah ini.


```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

Alasan penggunaan ReLU adalah bahwa
turunannya sangat baik:
baik menghilang atau hanya meneruskan argumennya.
Hal ini membuat optimisasi menjadi lebih stabil
dan mengurangi masalah vanishing gradient yang terdokumentasi dengan baik
yang mengganggu versi jaringan saraf sebelumnya (lebih lanjut tentang ini nanti).

Perlu dicatat bahwa ada banyak varian fungsi ReLU,
termasuk *parametrized ReLU* (*pReLU*) :cite:`He.Zhang.Ren.ea.2015`.
Varian ini menambahkan komponen linear ke ReLU,
sehingga beberapa informasi masih bisa diteruskan,
bahkan ketika argumennya negatif:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoid Function

[**Fungsi *sigmoid* mentransformasi input**]
yang memiliki nilai dalam domain $\mathbb{R}$,
(**ke output dalam interval (0, 1).**)
Karena alasan itu, sigmoid sering disebut sebagai
fungsi *squashing*:
fungsi ini "memadatkan" setiap input dalam rentang (-inf, inf)
ke suatu nilai dalam rentang (0, 1):

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

Pada jaringan saraf awal, para ilmuwan
tertarik untuk memodelkan neuron biologis
yang *menembak* atau *tidak menembak*.
Oleh karena itu, pionir di bidang ini,
sejak zaman McCulloch dan Pitts,
pencipta neuron buatan,
fokus pada unit thresholding :cite:`McCulloch.Pitts.1943`.
Aktivasi thresholding bernilai 0
ketika inputnya berada di bawah ambang tertentu
dan bernilai 1 ketika input melebihi ambang batas.

Ketika perhatian beralih ke pembelajaran berbasis gradien,
fungsi sigmoid menjadi pilihan alami
karena fungsi ini merupakan aproksimasi yang halus dan diferensiabel
terhadap unit thresholding.
Sigmoid masih banyak digunakan sebagai
fungsi aktivasi pada unit output
ketika kita ingin menafsirkan output sebagai probabilitas
untuk masalah klasifikasi biner: Anda bisa menganggap sigmoid sebagai kasus khusus dari softmax.
Namun, sigmoid sebagian besar telah digantikan
oleh ReLU yang lebih sederhana dan lebih mudah dilatih
untuk sebagian besar penggunaan di hidden layer. Banyak hal ini terkait
dengan kenyataan bahwa sigmoid menghadirkan tantangan untuk optimisasi
:cite:`LeCun.Bottou.Orr.ea.1998` karena gradiennya menghilang untuk argumen yang positif *dan* negatif besar.
Hal ini dapat menyebabkan plateau yang sulit diatasi.
Meski demikian, sigmoid tetap penting. Pada bab-bab selanjutnya (misalnya, :numref:`sec_lstm`) tentang jaringan saraf berulang,
kami akan menjelaskan arsitektur yang memanfaatkan unit sigmoid
untuk mengontrol aliran informasi dari waktu ke waktu.

Di bawah ini, kami memplot fungsi sigmoid.
Perhatikan bahwa ketika input mendekati 0,
fungsi sigmoid mendekati
transformasi linear.


```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

Turunan dari fungsi sigmoid diberikan oleh persamaan berikut:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

Turunan dari fungsi sigmoid diplot di bawah ini.
Perhatikan bahwa ketika input bernilai 0,
turunan dari fungsi sigmoid
mencapai maksimum sebesar 0.25.
Ketika input bergerak menjauh dari 0 ke arah positif atau negatif,
turunan mendekati 0.


```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out gradients sebelumnya
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### Tanh Function
:label:`subsec_tanh`

Seperti fungsi sigmoid, [**fungsi tanh (tangen hiperbolik)
juga memadatkan inputnya**],
mentransformasinya menjadi elemen dalam interval (**antara $-1$ dan $1$**):

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

Kami memplot fungsi tanh di bawah ini. Perhatikan bahwa ketika input mendekati 0, fungsi tanh mendekati transformasi linear. 
Meskipun bentuk fungsinya mirip dengan fungsi sigmoid, fungsi tanh menunjukkan simetri titik terhadap asal sistem koordinat :cite:`Kalman.Kwasny.1992`.


```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

Turunan dari fungsi tanh adalah:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

Grafik turunan ini ditampilkan di bawah.
Ketika input mendekati 0,
turunan dari fungsi tanh mendekati maksimum sebesar 1.
Dan seperti yang kita lihat pada fungsi sigmoid,
ketika input bergerak menjauh dari 0 ke arah positif atau negatif,
turunan dari fungsi tanh mendekati 0.


```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## Summary and Discussion

Sekarang kita tahu cara memasukkan non-linearitas
untuk membangun arsitektur jaringan saraf multilayer yang ekspresif.
Sebagai catatan tambahan, pengetahuan Anda sekarang
sudah memberi Anda toolkit yang mirip dengan
praktisi sekitar tahun 1990.
Dalam beberapa hal, Anda memiliki keunggulan
dibandingkan siapapun yang bekerja pada saat itu,
karena Anda dapat memanfaatkan framework
deep learning open-source yang kuat
untuk membangun model dengan cepat, hanya dengan beberapa baris kode.
Sebelumnya, melatih jaringan ini
membutuhkan para peneliti untuk mengkodekan lapisan dan turunannya
secara eksplisit dalam C, Fortran, atau bahkan Lisp (dalam kasus LeNet).

Keuntungan tambahan adalah bahwa ReLU jauh lebih mudah
untuk dioptimalkan dibandingkan dengan fungsi sigmoid atau tanh. Bisa dikatakan bahwa ini adalah salah satu inovasi kunci yang membantu kebangkitan kembali deep learning selama dekade terakhir. Namun, penelitian tentang
fungsi aktivasi belum berhenti.
Misalnya,
fungsi aktivasi GELU (Gaussian error linear unit)
$x \Phi(x)$ oleh :citet:`Hendrycks.Gimpel.2016` ($\Phi(x)$
adalah fungsi distribusi kumulatif Gaussian standar) dan
fungsi aktivasi Swish
$\sigma(x) = x \operatorname{sigmoid}(\beta x)$ seperti yang diusulkan dalam :citet:`Ramachandran.Zoph.Le.2017` dapat memberikan akurasi yang lebih baik
dalam banyak kasus.

## Exercises

1. Tunjukkan bahwa menambahkan lapisan ke jaringan *linear* yang dalam, yaitu jaringan tanpa
   non-linearitas $\sigma$ tidak akan pernah meningkatkan kekuatan ekspresif jaringan.
   Berikan contoh di mana ini secara aktif menguranginya.
2. Hitung turunan dari fungsi aktivasi pReLU.
3. Hitung turunan dari fungsi aktivasi Swish $x \operatorname{sigmoid}(\beta x)$.
4. Tunjukkan bahwa sebuah MLP yang hanya menggunakan ReLU (atau pReLU) membentuk fungsi
   linear secara piecewise yang kontinyu.
5. Sigmoid dan tanh sangat mirip.
    1. Tunjukkan bahwa $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
    2. Buktikan bahwa kelas fungsi yang diparameterisasi oleh kedua non-linearitas ini identik. Petunjuk: lapisan afine juga memiliki bias.
6. Anggap bahwa kita memiliki non-linearitas yang berlaku untuk satu minibatch dalam satu waktu, seperti batch normalization :cite:`Ioffe.Szegedy.2015`. Jenis masalah apa yang Anda perkirakan akan terjadi?
7. Berikan contoh di mana gradien menghilang untuk fungsi aktivasi sigmoid.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/226)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17984)
:end_tab:
