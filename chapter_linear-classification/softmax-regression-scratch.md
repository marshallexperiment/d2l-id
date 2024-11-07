```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implementasi Regresi Softmax dari Awal
:label:`sec_softmax_scratch`

Karena regresi softmax sangat mendasar,
kami percaya bahwa Anda sebaiknya mengetahui
cara mengimplementasikannya sendiri.
Di sini, kita membatasi diri untuk mendefinisikan
aspek-aspek khusus dari model softmax
dan menggunakan kembali komponen-komponen lain
dari bagian regresi linear kita,
termasuk loop pelatihan.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## Fungsi Softmax

Mari kita mulai dengan bagian yang paling penting:
pemetaan dari skalar ke probabilitas.
Sebagai penyegaran, ingat kembali operasi operator penjumlahan
pada dimensi tertentu dalam tensor,
seperti yang dibahas di :numref:`subsec_lin-alg-reduction`
dan :numref:`subsec_lin-alg-non-reduction`.
[**Diberikan sebuah matriks `X`, kita dapat menjumlahkan semua elemen (secara default) atau hanya
elemen-elemen pada sumbu tertentu.**]
Variabel `axis` memungkinkan kita untuk menghitung jumlah per baris atau per kolom:


```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Menghitung softmax memerlukan tiga langkah:
(i) eksponensiasi setiap elemen;
(ii) penjumlahan per baris untuk menghitung konstanta normalisasi untuk setiap contoh;
(iii) pembagian setiap baris dengan konstanta normalisasinya,
memastikan bahwa hasil akhirnya berjumlah 1:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

Penyebutnya (logaritma dari penyebut ini)
disebut sebagai (log) *fungsi partisi*.
Fungsi ini diperkenalkan dalam [fisika statistik](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
untuk menjumlahkan semua kemungkinan keadaan dalam ensemble termodinamika.
Implementasinya cukup sederhana:

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # Mekanisme Broadcasting Diterapkan di Sini
```

Untuk setiap input `X`, [**kita mengubah setiap elemen
menjadi bilangan non-negatif.
Setiap baris berjumlah 1,**]
sesuai dengan persyaratan untuk probabilitas. Perhatian: kode di atas *tidak* tahan terhadap argumen yang sangat besar atau sangat kecil. 
Meskipun ini cukup untuk menggambarkan apa yang terjadi, Anda sebaiknya *tidak* menggunakan kode ini secara langsung untuk keperluan serius. 
Framework deep learning memiliki perlindungan semacam ini yang sudah tertanam, dan kita akan menggunakan softmax bawaan framework tersebut ke depannya.


```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## Model

Sekarang kita memiliki semua yang diperlukan
untuk mengimplementasikan [**model regresi softmax.**]
Seperti pada contoh regresi linear kita,
setiap instance akan direpresentasikan
oleh vektor dengan panjang tetap.
Karena data mentah di sini terdiri dari
gambar $28 \times 28$ piksel,
[**kita meratakan setiap gambar,
menganggapnya sebagai vektor dengan panjang 784.**]
Pada bab-bab selanjutnya, kita akan memperkenalkan
jaringan saraf konvolusi,
yang memanfaatkan struktur spasial
dengan cara yang lebih memuaskan.

Pada regresi softmax,
jumlah output dari jaringan kita
harus sama dengan jumlah kelas.
(**Karena dataset kita memiliki 10 kelas,
jaringan kita memiliki dimensi output sebesar 10.**)
Akibatnya, bobot kita akan membentuk matriks berukuran $784 \times 10$
ditambah vektor baris $1 \times 10$ untuk bias.
Seperti pada regresi linear,
kita menginisialisasi bobot `W`
dengan noise Gaussian.
Bias diinisialisasi dengan nilai nol.


```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

Kode di bawah ini mendefinisikan bagaimana jaringan
memetakan setiap input ke output.
Perhatikan bahwa kita meratakan setiap gambar $28 \times 28$ piksel dalam batch
menjadi sebuah vektor menggunakan `reshape`
sebelum mengoper data melalui model kita.


```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## Loss Cross-Entropy

Selanjutnya, kita perlu mengimplementasikan fungsi loss cross-entropy
(yang diperkenalkan di :numref:`subsec_softmax-regression-loss-func`).
Ini mungkin merupakan fungsi loss yang paling umum
dalam deep learning.
Saat ini, aplikasi deep learning
yang diformulasikan sebagai masalah klasifikasi
jauh lebih banyak daripada yang lebih baik diperlakukan sebagai masalah regresi.

Ingat bahwa cross-entropy mengambil log-likelihood negatif
dari probabilitas prediksi yang diberikan pada label yang benar.
Untuk efisiensi, kita menghindari penggunaan for-loop Python dan menggunakan indexing sebagai gantinya.
Secara khusus, one-hot encoding dalam $\mathbf{y}$
memungkinkan kita untuk memilih suku yang sesuai dalam $\hat{\mathbf{y}}$.

Untuk melihat ini dalam aksi, kita [**membuat data contoh `y_hat`
dengan 2 contoh probabilitas prediksi pada 3 kelas dan label yang sesuai `y`.**]
Label yang benar adalah $0$ dan $2$ masing-masing (yaitu, kelas pertama dan ketiga).
[**Dengan menggunakan `y` sebagai indeks probabilitas dalam `y_hat`,**]
kita dapat memilih suku dengan efisien.

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
Sekarang kita dapat (**mengimplementasikan fungsi loss cross-entropy**) dengan mengambil rata-rata dari logaritma probabilitas yang dipilih.
:end_tab:

:begin_tab:`jax`
Sekarang kita dapat (**mengimplementasikan fungsi loss cross-entropy**) dengan mengambil rata-rata dari logaritma probabilitas yang dipilih.

Perhatikan bahwa untuk memanfaatkan `jax.jit` dalam mempercepat implementasi JAX, dan
untuk memastikan bahwa `loss` adalah fungsi murni, fungsi `cross_entropy` didefinisikan ulang
di dalam `loss` untuk menghindari penggunaan variabel atau fungsi global
yang mungkin membuat fungsi `loss` menjadi tidak murni.
Pembaca yang tertarik dapat merujuk ke [dokumentasi JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) tentang `jax.jit` dan fungsi murni.
:end_tab:


```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # Dictionary kosong yang dikembalikan merupakan placeholder untuk data tambahan,
    # yang akan digunakan nanti (misalnya, untuk batch normalization)
    return cross_entropy(y_hat, y), {}
```

## Pelatihan

Kita menggunakan kembali metode `fit` yang didefinisikan di :numref:`sec_linear_scratch` untuk [**melatih model dengan 10 epoch.**]
Perhatikan bahwa jumlah epoch (`max_epochs`),
ukuran minibatch (`batch_size`),
dan learning rate (`lr`)
adalah hyperparameter yang dapat diatur.
Artinya, meskipun nilai-nilai ini tidak
dipelajari selama loop pelatihan utama,
mereka tetap mempengaruhi kinerja
model kita, baik dalam hal pelatihan
maupun kinerja generalisasi.
Dalam praktiknya, Anda akan memilih nilai-nilai ini
berdasarkan pada *split validasi* dari data
dan kemudian, akhirnya, mengevaluasi model akhir Anda
pada *split tes*.
Seperti dibahas di :numref:`subsec_generalization-model-selection`,
kita akan menganggap data tes dari Fashion-MNIST
sebagai set validasi, sehingga
melaporkan loss validasi dan akurasi validasi
pada split ini.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Prediksi

Sekarang pelatihan selesai,
model kita siap untuk [**mengklasifikasikan beberapa gambar.**]


```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

Kita lebih tertarik pada gambar yang kita labeli *secara tidak benar*. Kita memvisualisasikannya dengan
membandingkan label sebenarnya
(baris pertama dari output teks)
dengan prediksi dari model
(baris kedua dari output teks).


```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Ringkasan

Sejauh ini kita mulai mendapatkan pengalaman
dalam menyelesaikan masalah regresi linear
dan klasifikasi.
Dengan ini, kita telah mencapai apa yang bisa dibilang
merupakan state of the art dalam pemodelan statistik pada tahun 1960--1970an.
Pada bagian berikutnya, kita akan menunjukkan cara memanfaatkan
framework deep learning untuk mengimplementasikan model ini
dengan jauh lebih efisien.

## Latihan

1. Pada bagian ini, kita langsung mengimplementasikan fungsi softmax berdasarkan definisi matematis dari operasi softmax. Seperti yang dibahas di :numref:`sec_softmax`, ini bisa menyebabkan ketidakstabilan numerik.
    1. Uji apakah `softmax` masih berfungsi dengan benar jika input memiliki nilai $100$.
    1. Uji apakah `softmax` masih berfungsi dengan benar jika nilai terbesar dari semua input lebih kecil dari $-100$.
    1. Implementasikan perbaikan dengan melihat nilai relatif terhadap entri terbesar dalam argumen.
2. Implementasikan fungsi `cross_entropy` yang mengikuti definisi fungsi loss cross-entropy $\sum_i y_i \log \hat{y}_i$.
    1. Coba gunakan pada contoh kode di bagian ini.
    1. Mengapa menurut Anda fungsi ini berjalan lebih lambat?
    1. Haruskah Anda menggunakannya? Kapan ini masuk akal?
    1. Apa yang perlu Anda perhatikan? Petunjuk: pertimbangkan domain dari logaritma.
3. Apakah selalu merupakan ide yang baik untuk mengembalikan label yang paling mungkin? Misalnya, apakah Anda akan melakukan ini untuk diagnosis medis? Bagaimana Anda akan mencoba mengatasi ini?
4. Asumsikan bahwa kita ingin menggunakan regresi softmax untuk memprediksi kata berikutnya berdasarkan beberapa fitur. Apa saja masalah yang mungkin muncul dari kosakata yang besar?
5. Bereksperimenlah dengan hyperparameter dari kode di bagian ini. Khususnya:
    1. Plot bagaimana perubahan loss validasi saat Anda mengubah learning rate.
    1. Apakah loss validasi dan pelatihan berubah saat Anda mengubah ukuran minibatch? Seberapa besar atau kecil Anda harus mencoba untuk melihat efeknya?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17982)
:end_tab:
