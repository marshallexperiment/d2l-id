```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Self-Attention dan Positional Encoding
:label:`sec_self-attention-and-positional-encoding`

Dalam deep learning, kita sering menggunakan CNN atau RNN untuk meng-encode urutan (sequence). Sekarang dengan mekanisme attention dalam pikiran, 
bayangkan memberi makan serangkaian token ke mekanisme attention sedemikian rupa sehingga pada setiap langkah, setiap token memiliki query, key, dan value sendiri-sendiri.
Di sini, saat menghitung nilai dari representasi token pada lapisan berikutnya, token dapat memperhatikan (melalui vektor query-nya) 
token lain (berdasarkan kecocokan dengan vektor key mereka).

Dengan menggunakan seluruh set skor kompatibilitas query-key, kita dapat menghitung representasi untuk setiap token dengan membangun penjumlahan berbobot yang tepat atas 
token-token lain. Karena setiap token memperhatikan setiap token lain (berbeda dengan kasus ketika langkah-langkah decoder memperhatikan langkah-langkah encoder), arsitektur 
seperti ini biasanya disebut sebagai model *self-attention* :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`, dan di tempat lain disebut sebagai model *intra-attention* :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`. 
Pada bagian ini, kita akan membahas encoding urutan menggunakan self-attention, termasuk penggunaan informasi tambahan untuk urutan tersebut.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## [**Self-Attention**]

Diberikan sebuah urutan input token
$\mathbf{x}_1, \ldots, \mathbf{x}_n$ di mana setiap $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$),
output self-attention memiliki urutan dengan panjang yang sama
$\mathbf{y}_1, \ldots, \mathbf{y}_n$, di mana

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

sesuai dengan definisi attention pooling pada
:eqref:`eq_attention_pooling`.
Dengan menggunakan multi-head attention,
potongan kode berikut ini
menghitung self-attention dari tensor
dengan bentuk (ukuran batch, jumlah time step atau panjang urutan dalam token, $d$).
Tensor output memiliki bentuk yang sama.


```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, X, X, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## Membandingkan CNNs, RNNs, dan Self-Attention
:label:`subsec_cnn-rnn-self-attention`

Mari kita
membandingkan arsitektur untuk memetakan
sebuah urutan dengan $n$ token
ke urutan lain dengan panjang yang sama,
di mana setiap token input atau output direpresentasikan oleh
sebuah vektor berdimensi $d$.
Secara spesifik,
kita akan mempertimbangkan CNNs, RNNs, dan self-attention.
Kita akan membandingkan
kompleksitas komputasi, 
operasi sekuensial,
dan panjang jalur maksimum.
Perlu dicatat bahwa operasi sekuensial mencegah komputasi paralel,
sementara jalur yang lebih pendek antara
kombinasi posisi dalam urutan
memudahkan pembelajaran ketergantungan jangka panjang 
dalam urutan tersebut :cite:`Hochreiter.Bengio.Frasconi.ea.2001`.


![Membandingkan arsitektur CNN (token padding diabaikan), RNN, dan self-attention.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`



Mari kita anggap setiap urutan teks sebagai "gambar satu dimensi". Demikian pula, CNN satu dimensi dapat memproses fitur lokal seperti $n$-gram dalam teks.
Diberikan sebuah urutan dengan panjang $n$,
pertimbangkan lapisan konvolusional dengan ukuran kernel $k$,
dan jumlah saluran input dan output keduanya adalah $d$.
Kompleksitas komputasi dari lapisan konvolusional adalah $\mathcal{O}(knd^2)$.
Seperti yang ditunjukkan pada :numref:`fig_cnn-rnn-self-attention`,
CNN bersifat hierarkis,
sehingga terdapat $\mathcal{O}(1)$ operasi sekuensial
dan panjang jalur maksimum adalah $\mathcal{O}(n/k)$.
Sebagai contoh, $\mathbf{x}_1$ dan $\mathbf{x}_5$
berada dalam receptive field dari dua lapisan CNN
dengan ukuran kernel 3 pada :numref:`fig_cnn-rnn-self-attention`.

Saat memperbarui hidden state dari RNN,
perkalian matriks berat $d \times d$
dan hidden state berdimensi $d$ memiliki
kompleksitas komputasi $\mathcal{O}(d^2)$.
Karena panjang urutannya adalah $n$,
kompleksitas komputasi dari lapisan rekuren
adalah $\mathcal{O}(nd^2)$.
Menurut :numref:`fig_cnn-rnn-self-attention`,
terdapat $\mathcal{O}(n)$ operasi sekuensial
yang tidak dapat diparalelisasi
dan panjang jalur maksimum juga $\mathcal{O}(n)$.

Pada self-attention,
queries, keys, dan values 
semuanya adalah matriks $n \times d$.
Pertimbangkan scaled dot product attention pada
:eqref:`eq_softmax_QK_V`,
di mana sebuah matriks $n \times d$ dikalikan dengan
sebuah matriks $d \times n$,
kemudian output matriks $n \times n$ dikalikan
dengan matriks $n \times d$.
Sebagai hasilnya,
self-attention memiliki kompleksitas komputasi $\mathcal{O}(n^2d)$.
Seperti yang dapat kita lihat dari :numref:`fig_cnn-rnn-self-attention`,
setiap token terhubung langsung
dengan token lainnya melalui self-attention.
Oleh karena itu,
komputasi dapat dilakukan secara paralel dengan $\mathcal{O}(1)$ operasi sekuensial
dan panjang jalur maksimum juga $\mathcal{O}(1)$.

Secara keseluruhan,
baik CNN maupun self-attention mendukung komputasi paralel
dan self-attention memiliki panjang jalur maksimum yang paling pendek.
Namun, kompleksitas komputasi kuadratik terhadap panjang urutan
membuat self-attention sangat lambat untuk urutan yang sangat panjang.



## [**Positional Encoding**]
:label:`subsec_positional-encoding`

Tidak seperti RNN yang memproses token dalam suatu urutan secara berulang satu per satu, self-attention meninggalkan operasi sekuensial demi komputasi paralel.
Perhatikan bahwa self-attention sendiri tidak mempertahankan urutan dari urutan token.
Bagaimana jika sangat penting bagi model untuk mengetahui dalam urutan apa input diterima?

Pendekatan dominan untuk mempertahankan informasi tentang urutan token adalah dengan merepresentasikannya kepada model sebagai input tambahan yang diasosiasikan dengan setiap token.
Input-input ini disebut *positional encodings*, dan mereka dapat dipelajari atau ditetapkan *a priori*.
Sekarang kita akan menjelaskan skema sederhana untuk positional encodings tetap yang berdasarkan pada fungsi sinus dan kosinus :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Misalkan representasi input 
$\mathbf{X} \in \mathbb{R}^{n \times d}$ 
mengandung embedding berdimensi $d$ 
untuk $n$ token dari suatu urutan.
Positional encoding akan menghasilkan
$\mathbf{X} + \mathbf{P}$
dengan menggunakan matriks embedding posisi 
$\mathbf{P} \in \mathbb{R}^{n \times d}$ dengan bentuk yang sama,
yang elemennya pada baris ke-$i$ 
dan kolom ke-$(2j)$ atau $(2j + 1)$ adalah

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

Sekilas, desain fungsi trigonometri ini terlihat aneh.
Sebelum kita memberikan penjelasan tentang desain ini, mari kita implementasikan terlebih dahulu dalam kelas `PositionalEncoding` berikut.


```{.python .input}
%%tab mxnet
class PositionalEncoding(nn.Block):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
%%tab pytorch
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
%%tab tensorflow
class PositionalEncoding(tf.keras.layers.Layer):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

```{.python .input}
%%tab jax
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        # Create a long enough P
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(
            -1, 1) / jnp.power(10000, jnp.arange(
            0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X))

    @nn.compact
    def __call__(self, X, training=False):
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'P', self.P)
        X = X + self.P[:, :X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)
```

Dalam matriks embedding posisi $\mathbf{P}$,
[**baris-baris berkorespondensi dengan posisi dalam urutan,
sedangkan kolom-kolom merepresentasikan dimensi-dimensi yang berbeda dari positional encoding**].
Dalam contoh berikut,
kita dapat melihat bahwa
kolom ke-$6$ dan kolom ke-$7$ dari matriks embedding posisi 
memiliki frekuensi yang lebih tinggi dibandingkan
kolom ke-$8$ dan kolom ke-$9$.
Perbedaan ini terjadi karena adanya perbedaan fase antara
kolom ke-$6$ dan kolom ke-$7$ (sama halnya untuk kolom ke-$8$ dan kolom ke-$9$),
yang disebabkan oleh pergantian antara fungsi sinus dan kosinus.


```{.python .input}
%%tab mxnet
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

```{.python .input}
%%tab jax
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
params = pos_encoding.init(d2l.get_key(), d2l.zeros((1, num_steps, encoding_dim)))
X, inter_vars = pos_encoding.apply(params, d2l.zeros((1, num_steps, encoding_dim)),
                                   mutable='intermediates')
P = inter_vars['intermediates']['P'][0]  # retrieve intermediate value P
P = P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### Informasi Posisi Absolut

Untuk melihat bagaimana frekuensi yang menurun secara monoton 
sepanjang dimensi encoding berhubungan dengan informasi posisi absolut,
mari kita cetak [**representasi biner**] dari $0, 1, \ldots, 7$.
Seperti yang dapat kita lihat, bit terendah, bit kedua terendah, 
dan bit ketiga terendah bergantian setiap angka, 
setiap dua angka, dan setiap empat angka, secara berturut-turut.


```{.python .input}
%%tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

Dalam representasi biner, bit yang lebih tinggi 
memiliki frekuensi yang lebih rendah dibandingkan dengan bit yang lebih rendah.
Demikian pula, seperti yang ditunjukkan pada peta panas di bawah ini,
[**encoding posisi menurunkan frekuensi sepanjang dimensi encoding**]
dengan menggunakan fungsi trigonometri.
Karena outputnya adalah angka desimal,
representasi yang berkelanjutan seperti ini
lebih efisien dalam hal ruang dibandingkan dengan representasi biner.


```{.python .input}
%%tab mxnet
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab jax
P = jnp.expand_dims(jnp.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### Informasi Posisi Relatif

Selain menangkap informasi posisi absolut,
encoding posisi di atas
juga memungkinkan
model untuk dengan mudah mempelajari perhatian berdasarkan posisi relatif.
Ini karena
untuk setiap offset posisi tetap $\delta$,
encoding posisi pada posisi $i + \delta$
dapat direpresentasikan melalui proyeksi linier
dari encoding posisi pada posisi $i$.


Proyeksi ini dapat dijelaskan
secara matematis.
Misalkan
$\omega_j = 1/10000^{2j/d}$,
setiap pasangan $(p_{i, 2j}, p_{i, 2j+1})$ 
pada :eqref:`eq_positional-encoding-def`
dapat 
diproyeksikan secara linier menjadi $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$
untuk setiap offset tetap $\delta$:

$$\begin{aligned}
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

dimana matriks proyeksi berukuran $2\times 2$ tidak bergantung pada indeks posisi $i$.

## Ringkasan

Pada self-attention, query, key, dan value semuanya berasal dari tempat yang sama.
Baik CNN maupun self-attention dapat menikmati komputasi paralel
dan self-attention memiliki panjang jalur maksimum yang terpendek.
Namun, kompleksitas komputasi kuadratik
terhadap panjang urutan
membuat self-attention sangat lambat
untuk urutan yang sangat panjang.
Untuk menggunakan informasi urutan,
kita dapat menyuntikkan informasi posisi absolut atau relatif
dengan menambahkan encoding posisi ke dalam representasi input.

## Latihan

1. Misalkan kita merancang arsitektur yang dalam untuk merepresentasikan sebuah urutan dengan menumpuk lapisan self-attention dengan encoding posisi. Apa kemungkinan masalah yang akan muncul?
2. Bisakah kamu merancang metode encoding posisi yang dapat dipelajari?
3. Bisakah kita memberikan embedding yang berbeda yang dipelajari sesuai dengan offset yang berbeda antara query dan key yang dibandingkan pada self-attention? Petunjuk: kamu dapat merujuk ke embedding posisi relatif :cite:`shaw2018self,huang2018music`.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1652)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3870)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18030)
:end_tab:
