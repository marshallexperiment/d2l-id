```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implementasi Singkat dari Regresi Softmax
:label:`sec_softmax_concise`

Seperti halnya framework deep learning tingkat tinggi
yang mempermudah implementasi regresi linear
(lihat :numref:`sec_linear_concise`),
mereka juga sama nyamannya digunakan di sini.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Mendefinisikan Model

Seperti pada :numref:`sec_linear_concise`, 
kita membangun layer fully connected 
menggunakan layer bawaan. 
Metode bawaan `__call__` kemudian memanggil `forward` 
setiap kali kita perlu menerapkan jaringan pada input tertentu.

:begin_tab:`mxnet`
Meskipun input `X` adalah tensor orde keempat, 
layer `Dense` bawaan 
akan secara otomatis mengonversi `X` menjadi tensor orde kedua 
dengan mempertahankan dimensi sepanjang sumbu pertama tidak berubah.
:end_tab:

:begin_tab:`pytorch`
Kita menggunakan layer `Flatten` untuk mengonversi tensor orde keempat `X` menjadi orde kedua 
dengan mempertahankan dimensi sepanjang sumbu pertama tidak berubah.
:end_tab:

:begin_tab:`tensorflow`
Kita menggunakan layer `Flatten` untuk mengonversi tensor orde keempat `X` 
dengan mempertahankan dimensi sepanjang sumbu pertama tidak berubah.
:end_tab:

:begin_tab:`jax`
Flax memungkinkan pengguna untuk menulis kelas jaringan dengan cara yang lebih ringkas 
menggunakan dekorator `@nn.compact`. Dengan `@nn.compact`, 
pengguna cukup menulis semua logika jaringan di dalam satu metode "forward pass", 
tanpa perlu mendefinisikan metode `setup` standar dalam dataclass.
:end_tab:


```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## Peninjauan Kembali Softmax
:label:`subsec_softmax-implementation-revisited`

Pada :numref:`sec_softmax_scratch`, kita menghitung output model
dan menerapkan loss cross-entropy. Meskipun ini sepenuhnya
masuk akal secara matematis, ini berisiko secara komputasional karena
terjadinya underflow dan overflow numerik saat melakukan eksponensial.

Ingat bahwa fungsi softmax menghitung probabilitas melalui
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$.
Jika beberapa nilai $o_k$ sangat besar, yaitu sangat positif,
maka $\exp(o_k)$ mungkin menjadi lebih besar dari nilai maksimum
yang dapat disimpan dalam tipe data tertentu. Hal ini disebut *overflow*. Begitu juga,
jika setiap argumen adalah angka negatif yang sangat besar, kita akan mengalami *underflow*.
Sebagai contoh, angka floating point presisi tunggal mencakup kisaran
dari sekitar $10^{-38}$ hingga $10^{38}$. Karena itu, jika elemen terbesar dalam $\mathbf{o}$
terletak di luar interval $[-90, 90]$, hasilnya tidak akan stabil.
Cara mengatasi masalah ini adalah dengan mengurangkan $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ dari
semua elemen:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

Dengan konstruksi ini, kita tahu bahwa $o_j - \bar{o} \leq 0$ untuk semua $j$. Jadi, untuk masalah klasifikasi dengan $q$ kelas,
penyebutnya berada dalam interval $[1, q]$. Selain itu,
pembilangnya tidak pernah melebihi $1$, sehingga mencegah terjadinya overflow numerik. Underflow numerik hanya terjadi
ketika $\exp(o_j - \bar{o})$ secara numerik bernilai $0$. Meski demikian, beberapa langkah berikutnya
mungkin akan bermasalah ketika kita ingin menghitung $\log \hat{y}_j$ sebagai $\log 0$.
Khususnya, dalam proses backpropagation,
kita mungkin akan dihadapkan pada layar penuh hasil `NaN` (Not a Number).

Untungnya, kita terselamatkan oleh fakta bahwa
meskipun kita menghitung fungsi eksponensial,
kita pada akhirnya berniat untuk mengambil log-nya
(saat menghitung loss cross-entropy).
Dengan menggabungkan softmax dan cross-entropy,
kita bisa menghindari masalah stabilitas numerik sama sekali. Kita memiliki:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

Ini menghindari baik overflow maupun underflow.
Kita akan tetap mempertahankan fungsi softmax konvensional
jika suatu saat kita ingin mengevaluasi probabilitas output model kita.
Namun, alih-alih mengirimkan probabilitas softmax ke dalam fungsi loss baru kita,
kita cukup
[**mengirimkan nilai logits dan menghitung softmax beserta log-nya
secara langsung di dalam fungsi loss cross-entropy,**]
yang melakukan penanganan cerdas seperti ["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp).


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## Pelatihan

Selanjutnya, kita melatih model kita. Kita menggunakan gambar Fashion-MNIST yang telah diratakan menjadi vektor fitur berdimensi 784.


```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Seperti sebelumnya, algoritma ini berkonvergensi menuju solusi
dengan akurasi yang cukup baik,
meskipun kali ini dengan lebih sedikit baris kode dibandingkan sebelumnya.

## Ringkasan

API tingkat tinggi sangat nyaman karena mampu menyembunyikan aspek-aspek yang berpotensi berbahaya, seperti stabilitas numerik, dari penggunanya. Selain itu, mereka memungkinkan pengguna merancang model secara ringkas dengan sangat sedikit baris kode. Hal ini merupakan kelebihan sekaligus kekurangan. Manfaat yang jelas adalah membuat hal-hal menjadi sangat mudah diakses, bahkan bagi insinyur yang belum pernah mengambil satu kelas statistik pun dalam hidup mereka (sebenarnya, mereka adalah bagian dari audiens target buku ini). Namun, menyembunyikan aspek-aspek yang rumit juga datang dengan konsekuensi: kurangnya dorongan untuk menambahkan komponen baru dan berbeda secara mandiri, karena sedikit pengalaman praktis untuk melakukannya. Selain itu, hal ini membuatnya lebih sulit untuk *memperbaiki* sesuatu ketika framework gagal menutupi semua kasus.

Oleh karena itu, kami sangat mendorong Anda untuk meninjau *kedua* versi dasar dan versi elegan dari banyak implementasi yang mengikuti. Meskipun kami menekankan kemudahan pemahaman, implementasi-implementasi ini biasanya tetap cukup efisien (dengan pengecualian besar pada konvolusi). Kami berniat agar Anda dapat membangun dari implementasi ini ketika Anda menemukan sesuatu yang baru yang belum disediakan oleh framework manapun.

## Latihan

1. Deep learning menggunakan berbagai format angka yang berbeda, termasuk presisi ganda FP64 (sangat jarang digunakan),
   presisi tunggal FP32, BFLOAT16 (baik untuk representasi yang terkompresi), FP16 (sangat tidak stabil), TF32 (format baru dari NVIDIA), dan INT8. Hitung argumen terkecil dan terbesar dari fungsi eksponensial untuk hasil yang tidak menyebabkan underflow atau overflow numerik.
2. INT8 adalah format yang sangat terbatas yang hanya terdiri dari angka non-nol dari $1$ hingga $255$. Bagaimana Anda bisa memperluas jangkauan dinamisnya tanpa menggunakan lebih banyak bit? Apakah perkalian dan penjumlahan standar masih berfungsi?
3. Tingkatkan jumlah epoch untuk pelatihan. Mengapa akurasi validasi mungkin menurun setelah beberapa waktu? Bagaimana kita bisa memperbaiki ini?
4. Apa yang terjadi ketika Anda meningkatkan learning rate? Bandingkan kurva loss untuk beberapa learning rate. Mana yang bekerja lebih baik? Kapan?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17983)
:end_tab:
