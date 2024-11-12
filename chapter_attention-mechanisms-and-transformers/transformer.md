```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Arsitektur Transformer
:label:`sec_transformer`

Kita telah membandingkan CNN, RNN, dan self-attention di
:numref:`subsec_cnn-rnn-self-attention`.
Secara khusus, self-attention menikmati keuntungan dalam hal komputasi paralel dan memiliki panjang jalur maksimum terpendek.
Oleh karena itu, menarik untuk merancang arsitektur dalam dengan menggunakan self-attention.
Berbeda dengan model self-attention sebelumnya yang masih mengandalkan RNN untuk representasi input :cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Paulus.Xiong.Socher.2017`,
model Transformer sepenuhnya didasarkan pada mekanisme perhatian (attention) tanpa lapisan konvolusional atau berulang :cite:`Vaswani.Shazeer.Parmar.ea.2017`.
Meskipun awalnya diusulkan untuk pembelajaran sequence-to-sequence pada data teks,
Transformer telah menjadi model yang pervasif di berbagai aplikasi pembelajaran mendalam modern,
seperti dalam bidang bahasa, visi, ucapan, dan pembelajaran penguatan.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import pandas as pd
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import pandas as pd
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
import pandas as pd
```

## Model

Sebagai salah satu contoh dari arsitektur encoder--decoder,
keseluruhan arsitektur Transformer
diperlihatkan di :numref:`fig_transformer`.
Seperti yang dapat kita lihat,
Transformer terdiri dari encoder dan decoder.
Berbeda dengan
Bahdanau attention
untuk pembelajaran sequence-to-sequence di :numref:`fig_s2s_attention_details`,
embedding dari input (sumber) dan output (target)
ditambahkan dengan positional encoding
sebelum dimasukkan ke dalam
encoder dan decoder
yang menyusun modul-modul berdasarkan self-attention.


![Arsitektur Transformer.](../img/transformer.svg)
:width:`320px`
:label:`fig_transformer`

Sekarang kita memberikan gambaran umum tentang
arsitektur Transformer pada :numref:`fig_transformer`.
Secara umum,
Transformer encoder adalah tumpukan dari beberapa lapisan identik,
di mana setiap lapisan
memiliki dua sublayer (keduanya disebut sebagai $\textrm{sublayer}$).
Yang pertama
adalah multi-head self-attention pooling,
dan yang kedua adalah jaringan feed-forward positionwise.
Secara khusus,
dalam encoder self-attention,
queries, keys, dan values semuanya berasal dari
keluaran dari lapisan encoder sebelumnya.
Terinspirasi oleh desain ResNet pada :numref:`sec_resnet`,
digunakan koneksi residual
di sekitar kedua sublayer.
Pada Transformer,
untuk setiap input $\mathbf{x} \in \mathbb{R}^d$ pada posisi apa pun dari urutan,
kita membutuhkan bahwa $\textrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ sehingga
koneksi residual $\mathbf{x} + \textrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ dapat dilakukan.
Penambahan dari koneksi residual ini langsung
diikuti oleh layer normalization :cite:`Ba.Kiros.Hinton.2016`.
Hasilnya, Transformer encoder mengeluarkan representasi vektor berdimensi $d$
untuk setiap posisi dari urutan input.

Transformer decoder juga merupakan tumpukan dari beberapa lapisan identik
dengan koneksi residual dan normalisasi lapisan.
Selain dua sublayer yang dijelaskan di
encoder, decoder menambahkan
sublayer ketiga yang dikenal sebagai
encoder--decoder attention,
di antara kedua sublayer tersebut.
Dalam encoder--decoder attention,
queries berasal dari
keluaran dari sublayer self-attention decoder,
dan keys dan values berasal dari
keluaran Transformer encoder.
Dalam decoder self-attention,
queries, keys, dan values semuanya berasal dari
keluaran dari lapisan decoder sebelumnya.
Namun, setiap posisi dalam decoder hanya
diizinkan untuk melakukan attend pada semua posisi di decoder
hingga posisi tersebut.
*Masked* attention ini
mempertahankan sifat autoregressive,
memastikan bahwa prediksi hanya bergantung
pada token output yang telah dihasilkan.

Kita telah mendeskripsikan dan mengimplementasikan
multi-head attention berdasarkan scaled dot products
di :numref:`sec_multihead-attention`
dan positional encoding di :numref:`subsec_positional-encoding`.
Selanjutnya, kita akan mengimplementasikan
sisa dari model Transformer.

## [**Jaringan Feed-Forward Positionwise**]
:label:`subsec_positionwise-ffn`

Jaringan feed-forward positionwise mengubah
representasi di semua posisi urutan
menggunakan MLP yang sama.
Itulah mengapa kita menyebutnya *positionwise*.
Dalam implementasi di bawah ini,
input `X` dengan bentuk
(ukuran batch, jumlah time steps atau panjang urutan dalam token,
jumlah unit tersembunyi atau dimensi fitur)
akan diubah oleh MLP dua lapis menjadi
tensorn output dengan bentuk
(ukuran batch, jumlah time steps, `ffn_num_outputs`).


```{.python .input}
%%tab mxnet
class PositionWiseFFN(nn.Block):  #@save
    """Jaringan feed-forward positionwise."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
%%tab pytorch
class PositionWiseFFN(nn.Module):  #@save
    """Jaringan feed-forward positionwise."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab tensorflow
class PositionWiseFFN(tf.keras.layers.Layer):  #@save
    """Jaringan feed-forward positionwise."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab jax
class PositionWiseFFN(nn.Module):  #@save
    """Jaringan feed-forward positionwise."""
    ffn_num_hiddens: int
    ffn_num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.ffn_num_hiddens)
        self.dense2 = nn.Dense(self.ffn_num_outputs)

    def __call__(self, X):
        return self.dense2(nn.relu(self.dense1(X)))
```

Contoh berikut menunjukkan bahwa [**dimensi terdalam dari sebuah tensor berubah**] menjadi jumlah keluaran pada jaringan feed-forward positionwise. 
Karena MLP yang sama mentransformasi di semua posisi, ketika input di semua posisi tersebut sama, maka output mereka juga identik.


```{.python .input}
%%tab mxnet
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
ffn(np.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab pytorch
ffn = PositionWiseFFN(4, 8)
ffn.eval()
ffn(d2l.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab tensorflow
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab jax
ffn = PositionWiseFFN(4, 8)
ffn.init_with_output(d2l.get_key(), jnp.ones((2, 3, 4)))[0][0]
```

## Koneksi Residual dan Normalisasi Lapisan

Sekarang mari kita fokus pada komponen "add & norm" dalam :numref:`fig_transformer`. Seperti yang telah dijelaskan di awal bagian ini, komponen ini adalah koneksi residual yang langsung diikuti dengan normalisasi lapisan. Keduanya adalah kunci arsitektur dalam yang efektif.

Di :numref:`sec_batch_norm`, kita telah menjelaskan bagaimana batch normalization merecenter dan mereskalakan berdasarkan contoh-contoh dalam minibatch. Seperti yang dijelaskan dalam :numref:`subsec_layer-normalization-in-bn`, layer normalization mirip dengan batch normalization kecuali bahwa yang pertama menormalkan berdasarkan dimensi fitur, sehingga memiliki keuntungan ketergantungan pada skala dan ukuran batch.

Meskipun batch normalization banyak digunakan dalam visi komputer, biasanya batch normalization secara empiris kurang efektif dibandingkan layer normalization pada tugas-tugas pemrosesan bahasa alami, di mana input sering kali berupa urutan dengan panjang yang bervariasi.

Kode berikut membandingkan [**normalisasi pada berbagai dimensi menggunakan layer normalization dan batch normalization**].


```{.python .input}
%%tab mxnet
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = d2l.tensor([[1, 2], [2, 3]])
# Menghitung rata-rata dan variansi dari X dalam mode pelatihan
with autograd.record():
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab pytorch
ln = nn.LayerNorm(2)
bn = nn.LazyBatchNorm1d()
X = d2l.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Menghitung rata-rata dan variansi dari X dalam mode pelatihan
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab tensorflow
ln = tf.keras.layers.LayerNormalization()
bn = tf.keras.layers.BatchNormalization()
X = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X, training=True))
```

```{.python .input}
%%tab jax
ln = nn.LayerNorm()
bn = nn.BatchNorm()
X = d2l.tensor([[1, 2], [2, 3]], dtype=d2l.float32)
# Menghitung rata-rata dan variansi dari X dalam mode pelatihan
print('layer norm:', ln.init_with_output(d2l.get_key(), X)[0],
      '\nbatch norm:', bn.init_with_output(d2l.get_key(), X,
                                           use_running_average=False)[0])
```

Sekarang kita bisa mengimplementasikan kelas `AddNorm`
[**menggunakan koneksi residual diikuti dengan normalisasi lapisan (layer normalization)**].
Dropout juga diterapkan untuk regularisasi.


```{.python .input}
%%tab mxnet
class AddNorm(nn.Block):  #@save
    """Koneksi residual diikuti dengan normalisasi lapisan (layer normalization)."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab pytorch
class AddNorm(nn.Module):  #@save
    """Koneksi residual diikuti dengan normalisasi lapisan (layer normalization)."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab tensorflow
class AddNorm(tf.keras.layers.Layer):  #@save
    """Koneksi residual diikuti dengan normalisasi lapisan (layer normalization)."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
```

```{.python .input}
%%tab jax
class AddNorm(nn.Module):  #@save
    """Koneksi residual diikuti dengan normalisasi lapisan (layer normalization)."""
    dropout: int

    @nn.compact
    def __call__(self, X, Y, training=False):
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)
```

Koneksi residual mengharuskan dua input memiliki bentuk yang sama sehingga [**tensor output juga memiliki bentuk yang sama setelah operasi penjumlahan**].


```{.python .input}
%%tab mxnet
add_norm = AddNorm(0.5)
add_norm.initialize()
shape = (2, 3, 4)
d2l.check_shape(add_norm(d2l.ones(shape), d2l.ones(shape)), shape)
```

```{.python .input}
%%tab pytorch
add_norm = AddNorm(4, 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(d2l.ones(shape), d2l.ones(shape)), shape)
```

```{.python .input}
%%tab tensorflow
# Normalized_shape is: [i for i in range(len(input.shape))][1:]
add_norm = AddNorm([1, 2], 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(tf.ones(shape), tf.ones(shape), training=False),
                shape)
```

```{.python .input}
%%tab jax
add_norm = AddNorm(0.5)
shape = (2, 3, 4)
output, _ = add_norm.init_with_output(d2l.get_key(), d2l.ones(shape),
                                      d2l.ones(shape))
d2l.check_shape(output, shape)
```

## Encoder
:label:`subsec_transformer-encoder`

Dengan semua komponen penting untuk merangkai encoder Transformer, mari kita mulai dengan mengimplementasikan [**satu lapisan dalam encoder**].
Kelas `TransformerEncoderBlock` berikut ini berisi dua sublapisan: multi-head self-attention dan jaringan feed-forward secara posisi,
di mana koneksi residual diikuti dengan normalisasi lapisan digunakan di sekitar kedua sublapisan tersebut.


```{.python .input}
%%tab mxnet
class TransformerEncoderBlock(nn.Block):  #@save
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab pytorch
class TransformerEncoderBlock(nn.Module):  #@save
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab tensorflow
class TransformerEncoderBlock(tf.keras.layers.Layer):  #@save
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs),
                          **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
```

```{.python .input}
%%tab jax
class TransformerEncoderBlock(nn.Module):  #@save
    """Transformer encoder block."""
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.addnorm1 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm2 = AddNorm(self.dropout)

    def __call__(self, X, valid_lens, training=False):
        output, attention_weights = self.attention(X, X, X, valid_lens,
                                                   training=training)
        Y = self.addnorm1(X, output, training=training)
        return self.addnorm2(Y, self.ffn(Y), training=training), attention_weights
```

Seperti yang dapat kita lihat,
[**tidak ada lapisan di dalam encoder Transformer
yang mengubah bentuk dari inputnya.**]


```{.python .input}
%%tab mxnet
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab tensorflow
X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
d2l.check_shape(encoder_blk(X, valid_lens, training=False), X.shape)
```

```{.python .input}
%%tab jax
X = jnp.ones((2, 100, 24))
valid_lens = jnp.array([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
(output, _), _ = encoder_blk.init_with_output(d2l.get_key(), X, valid_lens,
                                              training=False)
d2l.check_shape(output, X.shape)
```

Dalam implementasi [**encoder Transformer**] berikut,
kita menumpuk `num_blks` instance dari kelas `TransformerEncoderBlock` yang telah dijelaskan sebelumnya.
Karena kita menggunakan positional encoding yang tetap
yang nilainya selalu berada dalam rentang $-1$ hingga $1$,
kita mengalikan nilai embedding input yang dapat dipelajari
dengan akar kuadrat dari dimensi embedding
untuk melakukan rescaling sebelum menjumlahkan embedding input dan positional encoding.


```{.python .input}
%%tab mxnet
class TransformerEncoder(d2l.Encoder):  #@save
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
        self.initialize()

    def forward(self, X, valid_lens):
        # Karena nilai positional encoding berada dalam rentang -1 hingga 1, nilai embedding
        # dikalikan dengan akar kuadrat dari dimensi embedding
        # untuk melakukan rescaling sebelum dijumlahkan.
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab pytorch
class TransformerEncoder(d2l.Encoder):  #@save
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Karena nilai positional encoding berada dalam rentang -1 hingga 1, nilai embedding
        # dikalikan dengan akar kuadrat dari dimensi embedding
        # untuk melakukan rescaling sebelum dijumlahkan.
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab tensorflow
class TransformerEncoder(d2l.Encoder):  #@save
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerEncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_blks)]

    def call(self, X, valid_lens, **kwargs):
        # Karena nilai positional encoding berada dalam rentang -1 hingga 1, nilai embedding
        # dikalikan dengan akar kuadrat dari dimensi embedding
        # untuk melakukan rescaling sebelum dijumlahkan.
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab jax
class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    vocab_size: int
    num_hiddens:int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens, self.dropout)
        self.blks = [TransformerEncoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads,
                                             self.dropout, self.use_bias)
                     for _ in range(self.num_blks)]

    def __call__(self, X, valid_lens, training=False):
        # Karena nilai positional encoding berada dalam rentang -1 hingga 1, nilai embedding
        # dikalikan dengan akar kuadrat dari dimensi embedding
        # untuk melakukan rescaling sebelum dijumlahkan.
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X, training=training)
        attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X, attention_w = blk(X, valid_lens, training=training)
            attention_weights[i] = attention_w
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'enc_attention_weights', attention_weights)
        return X
```

Di bawah ini kita menentukan hyperparameter untuk [**membuat encoder Transformer dengan dua lapisan**]. 
Bentuk keluaran dari encoder Transformer adalah 
(batch size, jumlah langkah waktu, `num_hiddens`).

```{.python .input}
%%tab mxnet
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(np.ones((2, 100)), valid_lens), (2, 100, 24))
```

```{.python .input}
%%tab pytorch
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(d2l.ones((2, 100), dtype=torch.long), valid_lens),
                (2, 100, 24))
```

```{.python .input}
%%tab tensorflow
encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
d2l.check_shape(encoder(tf.ones((2, 100)), valid_lens, training=False),
                (2, 100, 24))
```

```{.python .input}
%%tab jax
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder.init_with_output(d2l.get_key(),
                                         jnp.ones((2, 100), dtype=jnp.int32),
                                         valid_lens)[0],
                (2, 100, 24))
```

## Decoder

Seperti yang ditunjukkan pada :numref:`fig_transformer`,
[**decoder Transformer terdiri dari beberapa lapisan yang identik**].
Setiap lapisan diimplementasikan dalam kelas 
`TransformerDecoderBlock` berikut,
yang terdiri dari tiga sublapisan:
self-attention pada decoder,
encoder--decoder attention,
dan jaringan feed-forward positionwise.
Setiap sublapisan ini menggunakan
koneksi residual di sekitar mereka
diikuti dengan normalisasi lapisan.


Seperti yang telah kita jelaskan di bagian ini,
dalam masked multi-head self-attention decoder 
(sublapisan pertama),
queries, keys, dan values
semua berasal dari keluaran lapisan decoder sebelumnya.
Saat melatih model sequence-to-sequence,
token di semua posisi (time steps) dari urutan output diketahui.
Namun,
selama prediksi,
urutan output dihasilkan token demi token;
sehingga,
pada setiap langkah waktu decoder,
hanya token yang telah dihasilkan 
yang dapat digunakan dalam self-attention decoder.
Untuk menjaga autoregression di dalam decoder,
masked self-attention di dalamnya
menentukan `dec_valid_lens` sehingga
setiap query hanya mengacu pada
semua posisi di dalam decoder
hingga posisi query.


```{.python .input}
%%tab mxnet
class TransformerDecoderBlock(nn.Block):
    # Blok ke-i dalam Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # Selama pelatihan, semua token dari setiap urutan output diproses
        # secara bersamaan, sehingga state[2][self.i] adalah None seperti yang diinisialisasi. 
        # Ketika mendekode setiap token urutan output satu per satu selama prediksi,
        # state[2][self.i] berisi representasi dari output yang telah didekode pada
        # blok ke-i hingga langkah waktu saat ini.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # Bentuk dari dec_valid_lens: (batch_size, num_steps), di mana setiap
            # baris adalah [1, 2, ..., num_steps]
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Bentuk dari enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab pytorch
class TransformerDecoderBlock(nn.Module):
    # Blok ke-i dalam Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # Selama pelatihan, semua token dari setiap urutan output diproses
        # secara bersamaan, sehingga state[2][self.i] adalah None seperti yang diinisialisasi. Ketika
        # mendekode setiap token urutan output secara bertahap selama prediksi,
        # state[2][self.i] berisi representasi dari output yang sudah didekode
        # pada blok ke-i hingga langkah waktu saat ini
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Bentuk dari dec_valid_lens: (batch_size, num_steps), di mana setiap
            # baris adalah [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Bentuk dari enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab tensorflow
class TransformerDecoderBlock(tf.keras.layers.Layer):
    # Blok ke-i dalam Transformer decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # Selama pelatihan, semua token dari setiap urutan output diproses
        # secara bersamaan, sehingga state[2][self.i] adalah None seperti yang diinisialisasi.
        # Ketika mendekode setiap token urutan output secara bertahap selama prediksi,
        # state[2][self.i] berisi representasi dari output yang sudah didekode
        # pada blok ke-i hingga langkah waktu saat ini
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Bentuk dari dec_valid_lens: (batch_size, num_steps), di mana setiap
            # baris adalah [1, 2, ..., num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1),
                           shape=(-1, num_steps)), repeats=batch_size, axis=0)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens,
                             **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Bentuk dari enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,
                             **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
```

```{.python .input}
%%tab jax
class TransformerDecoderBlock(nn.Module):
    # Blok ke-i dalam Transformer decoder
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    i: int

    def setup(self):
        self.attention1 = d2l.MultiHeadAttention(self.num_hiddens,
                                                 self.num_heads,
                                                 self.dropout)
        self.addnorm1 = AddNorm(self.dropout)
        self.attention2 = d2l.MultiHeadAttention(self.num_hiddens,
                                                 self.num_heads,
                                                 self.dropout)
        self.addnorm2 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm3 = AddNorm(self.dropout)

    def __call__(self, X, state, training=False):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # Selama pelatihan, semua token dari setiap urutan output diproses secara bersamaan,
        # sehingga state[2][self.i] diinisialisasi sebagai None.
        # Saat mendekode urutan output token demi token selama prediksi,
        # state[2][self.i] berisi representasi dari output yang telah didekode
        # pada blok ke-i hingga langkah waktu saat ini.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = jnp.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if training:
            batch_size, num_steps, _ = X.shape
            # Bentuk dari dec_valid_lens: (batch_size, num_steps), di mana setiap
            # baris adalah [1, 2, ..., num_steps]
            dec_valid_lens = jnp.tile(jnp.arange(1, num_steps + 1),
                                      (batch_size, 1))
        else:
            dec_valid_lens = None
        # Self-attention
        X2, attention_w1 = self.attention1(X, key_values, key_values,
                                           dec_valid_lens, training=training)
        Y = self.addnorm1(X, X2, training=training)
        # Encoder-decoder attention. Bentuk dari enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2, attention_w2 = self.attention2(Y, enc_outputs, enc_outputs,
                                           enc_valid_lens, training=training)
        Z = self.addnorm2(Y, Y2, training=training)
        return self.addnorm3(Z, self.ffn(Z), training=training), state, attention_w1, attention_w2
```

Untuk memfasilitasi operasi dot product berskala dalam attention encoder--decoder
dan operasi penjumlahan dalam koneksi residual,
[**dimensi fitur (`num_hiddens`) dari decoder adalah sama dengan dimensi dari encoder.**]


```{.python .input}
%%tab mxnet
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab pytorch
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab tensorflow
decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
X = tf.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state, training=False)[0], X.shape)
```

```{.python .input}
%%tab jax
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk.init_with_output(d2l.get_key(), X, valid_lens)[0][0],
         valid_lens, [None]]
d2l.check_shape(decoder_blk.init_with_output(d2l.get_key(), X, state)[0][0],
                X.shape)
```

Sekarang kita [**membangun keseluruhan Transformer decoder**]
yang terdiri dari `num_blks` instance `TransformerDecoderBlock`.
Pada akhirnya,
sebuah layer fully connected menghitung prediksi
untuk semua token keluaran yang mungkin dengan `vocab_size`.
Baik bobot self-attention pada decoder
maupun bobot attention encoder--decoder
disimpan untuk visualisasi di kemudian hari.


```{.python .input}
%%tab mxnet
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add(TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize()

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerDecoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, i)
                     for i in range(num_blks)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        # 2 attention layers in decoder
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = (
                blk.attention1.attention.attention_weights)
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = (
                blk.attention2.attention.attention_weights)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class TransformerDecoder(nn.Module):
    vocab_size: int
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens,
                                                   self.dropout)
        self.blks = [TransformerDecoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads, self.dropout, i)
                     for i in range(self.num_blks)]
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def __call__(self, X, state, training=False):
        X = self.embedding(X) * jnp.sqrt(jnp.float32(self.num_hiddens))
        X = self.pos_encoding(X, training=training)
        attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state, attention_w1, attention_w2 = blk(X, state,
                                                       training=training)
            # Decoder self-attention weights
            attention_weights[0][i] = attention_w1
            # Encoder-decoder attention weights
            attention_weights[1][i] = attention_w2
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'dec_attention_weights', attention_weights)
        return self.dense(X), state
```

## [**Pelatihan**]

Mari kita instansiasi sebuah model encoder--decoder
dengan mengikuti arsitektur Transformer.
Di sini kita menetapkan bahwa
baik Transformer encoder maupun Transformer decoder
memiliki dua lapisan yang menggunakan 4-head attention.
Seperti pada :numref:`sec_seq2seq_training`,
kita melatih model Transformer
untuk pembelajaran sequence-to-sequence pada dataset terjemahan mesin Inggris--Prancis.


```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
if tab.selected('tensorflow'):
    key_size, query_size, value_size = 256, 256, 256
    norm_shape = [2]
if tab.selected('pytorch', 'mxnet', 'jax'):
    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001, training=True)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = TransformerEncoder(
            len(data.src_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        decoder = TransformerDecoder(
            len(data.tgt_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.001)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

Setelah pelatihan,
kita menggunakan model Transformer
untuk [**menerjemahkan beberapa kalimat bahasa Inggris**] ke dalam bahasa Prancis dan menghitung skor BLEU-nya.


```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(
        trainer.state.params, data.build(engs, fras), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

Mari kita [**visualisasikan bobot perhatian Transformer**] ketika menerjemahkan kalimat bahasa Inggris terakhir menjadi bahasa Prancis.
Bentuk dari bobot perhatian self-attention pada encoder adalah (jumlah lapisan encoder, jumlah kepala perhatian, `num_steps` atau jumlah kueri, `num_steps` atau jumlah pasangan kunci-nilai).


```{.python .input}
%%tab pytorch, mxnet, tensorflow
_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
enc_attention_weights = d2l.concat(model.encoder.attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = d2l.reshape(enc_attention_weights, shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

```{.python .input}
%%tab jax
_, (dec_attention_weights, enc_attention_weights) = model.predict_step(
    trainer.state.params, data.build([engs[-1]], [fras[-1]]),
    data.num_steps, True)
enc_attention_weights = d2l.concat(enc_attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = d2l.reshape(enc_attention_weights, shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

In the encoder self-attention,
both queries and keys come from the same input sequence.
Since padding tokens do not carry meaning,
with specified valid length of the input sequence
no query attends to positions of padding tokens.
In the following,
two layers of multi-head attention weights
are presented row by row.
Each head independently attends
based on a separate representation subspace of queries, keys, and values.

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.show_heatmaps(
    enc_attention_weights, xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

```{.python .input}
%%tab pytorch
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

[**Untuk memvisualisasikan bobot self-attention decoder dan bobot encoder--decoder attention, kita membutuhkan lebih banyak manipulasi data.**]
Sebagai contoh, kita mengisi bobot perhatian yang ter-mask dengan nilai nol.
Perhatikan bahwa bobot self-attention decoder dan bobot encoder--decoder attention
keduanya memiliki kueri yang sama:
token awal urutan (beginning-of-sequence) diikuti oleh token output dan mungkin token akhir urutan (end-of-sequence).


```{.python .input}
%%tab mxnet
dec_attention_weights_2d = [d2l.tensor(head[0]).tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, (
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab pytorch
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
shape = (-1, 2, num_blks, num_heads, data.num_steps)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, shape)
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab tensorflow
dec_attention_weights_2d = [head[0] for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = tf.convert_to_tensor(
    np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(
        0.0).values).astype(np.float32))
dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(
    dec_attention_weights, perm=(1, 2, 3, 0, 4))
```

```{.python .input}
%%tab jax
dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape(
    (-1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab all
d2l.check_shape(dec_self_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
d2l.check_shape(dec_inter_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

Because of the autoregressive property of the decoder self-attention,
no query attends to key--value pairs after the query position.

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

Mirip dengan kasus pada self-attention encoder,
melalui panjang yang valid dari urutan input yang telah ditentukan,
[**tidak ada query dari urutan output yang memperhatikan token padding dari urutan input.**]


```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

Meskipun arsitektur Transformer awalnya diusulkan untuk pembelajaran sequence-to-sequence, seperti yang akan kita bahas nanti di buku ini, baik encoder Transformer maupun decoder Transformer sering kali digunakan secara individual untuk berbagai tugas pembelajaran mendalam.

## Ringkasan

Transformer adalah contoh dari arsitektur encoder-decoder, meskipun pada praktiknya baik encoder maupun decoder dapat digunakan secara individual. Dalam arsitektur Transformer, multi-head self-attention digunakan untuk merepresentasikan urutan input dan urutan output, meskipun decoder harus mempertahankan sifat autoregresif melalui versi yang dimasking. Baik koneksi residual maupun normalisasi lapisan dalam Transformer penting untuk melatih model yang sangat dalam. Network feed-forward positionwise dalam model Transformer mentransformasi representasi pada semua posisi urutan dengan menggunakan MLP yang sama.

## Latihan

1. Latih Transformer yang lebih dalam dalam eksperimen. Bagaimana ini mempengaruhi kecepatan pelatihan dan kinerja terjemahan?
2. Apakah mengganti scaled dot product attention dengan additive attention di Transformer adalah ide yang bagus? Mengapa?
3. Untuk pemodelan bahasa, apakah kita harus menggunakan encoder Transformer, decoder Transformer, atau keduanya? Bagaimana Anda akan merancang metode ini?
4. Tantangan apa yang bisa dihadapi oleh Transformer jika urutan input sangat panjang? Mengapa?
5. Bagaimana Anda akan meningkatkan efisiensi komputasi dan memori Transformer? Petunjuk: Anda dapat merujuk pada makalah survei oleh :citet:`Tay.Dehghani.Bahri.ea.2020`.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/348)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1066)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3871)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18031)
:end_tab:
