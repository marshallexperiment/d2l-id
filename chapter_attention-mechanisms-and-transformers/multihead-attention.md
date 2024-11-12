```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Multi-Head Attention
:label:`sec_multihead-attention`

Dalam praktiknya, dengan diberikan set query, key, dan value yang sama, kita mungkin ingin model kita menggabungkan pengetahuan dari
perilaku berbeda dari mekanisme attention yang sama,
seperti menangkap ketergantungan berbagai rentang
(misalnya, rentang pendek vs. rentang panjang) dalam sebuah urutan.
Dengan demikian, mungkin bermanfaat untuk membiarkan mekanisme attention kita menggunakan berbagai subruang representasi dari query, key, dan value secara bersamaan.

Untuk tujuan ini, alih-alih melakukan 
sebuah attention pooling tunggal,
query, key, dan value 
dapat ditransformasikan dengan $h$ proyeksi linear yang dipelajari secara independen.
Kemudian $h$ query, key, dan value yang diproyeksikan ini 
diberikan pada attention pooling secara paralel.
Pada akhirnya,
$h$ output dari attention pooling 
digabungkan dan ditransformasikan dengan proyeksi linear lain yang dipelajari
untuk menghasilkan output akhir.
Desain ini disebut *multi-head attention*,
dimana setiap dari $h$ output attention pooling 
disebut *head* :cite:`Vaswani.Shazeer.Parmar.ea.2017`.
Menggunakan lapisan fully connected 
untuk melakukan transformasi linear yang dapat dipelajari,
:numref:`fig_multi-head-attention`
menjelaskan multi-head attention.

![Multi-head attention, di mana beberapa head digabungkan kemudian ditransformasikan secara linear.](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`



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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## Model

Sebelum memberikan implementasi dari multi-head attention,
mari kita formalisasikan model ini secara matematis.
Diberikan sebuah query $\mathbf{q} \in \mathbb{R}^{d_q}$,
sebuah key $\mathbf{k} \in \mathbb{R}^{d_k}$,
dan sebuah value $\mathbf{v} \in \mathbb{R}^{d_v}$,
setiap head attention $\mathbf{h}_i$  ($i = 1, \ldots, h$)
dihitung sebagai

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

dimana 
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$,
$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$,
dan $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$
adalah parameter yang dapat dipelajari dan
$f$ adalah attention pooling,
seperti 
additive attention dan scaled dot product attention
di :numref:`sec_attention-scoring-functions`.
Output dari multi-head attention 
adalah transformasi linear lain melalui 
parameter yang dapat dipelajari
$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$
dari gabungan $h$ head:

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

Berdasarkan desain ini, setiap head dapat memberikan perhatian
pada bagian input yang berbeda.
Fungsi yang lebih canggih daripada sekadar rata-rata berbobot dapat diekspresikan.

## Implementasi

Dalam implementasi ini,
kita [**memilih scaled dot product attention
untuk setiap head**] dari multi-head attention.
Untuk menghindari pertumbuhan yang signifikan dalam biaya komputasi dan parameterisasi,
kita menetapkan $p_q = p_k = p_v = p_o / h$.
Perhatikan bahwa $h$ head dapat dihitung secara paralel
jika kita menetapkan jumlah output
dari transformasi linear
untuk query, key, dan value menjadi $p_q h = p_k h = p_v h = p_o$.
Dalam implementasi berikut,
$p_o$ ditentukan melalui argumen `num_hiddens`.


```{.python .input}
%%tab mxnet
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # Bentuk queries, keys, atau values:
        # (batch_size, jumlah queries atau pasangan key-value, num_hiddens)
        # Bentuk valid_lens: (batch_size,) atau (batch_size, jumlah queries)
        # Setelah ditranspos, bentuk output queries, keys, atau values:
        # (batch_size * num_heads, jumlah queries atau pasangan key-value,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # Pada axis 0, salin item pertama (skalar atau vektor) sebanyak num_heads
            # kali, lalu salin item berikutnya, dan seterusnya
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # Bentuk output: (batch_size * num_heads, jumlah queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        
        # Bentuk output_concat: (batch_size, jumlah queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab pytorch
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Bentuk dari queries, keys, atau values:
        # (batch_size, jumlah queries atau pasangan key-value, num_hiddens)
        # Bentuk dari valid_lens: (batch_size,) atau (batch_size, jumlah queries)
        # Setelah transpos, bentuk output queries, keys, atau values:
        # (batch_size * num_heads, jumlah queries atau pasangan key-value,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # Pada axis 0, salin item pertama (skalar atau vektor) sebanyak num_heads
            # kali, lalu salin item berikutnya, dan seterusnya
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Bentuk dari output: (batch_size * num_heads, jumlah queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Bentuk dari output_concat: (batch_size, jumlah queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab tensorflow
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, **kwargs):
        # Bentuk dari queries, keys, atau values:
        # (batch_size, jumlah queries atau pasangan key-value, num_hiddens)
        # Bentuk dari valid_lens: (batch_size,) atau (batch_size, jumlah queries)
        # Setelah ditranspos, bentuk dari output queries, keys, atau values:
        # (batch_size * num_heads, jumlah queries atau pasangan key-value,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            # Pada axis 0, salin item pertama (skalar atau vektor) sebanyak num_heads
            # kali, lalu salin item berikutnya, dan seterusnya
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        # Bentuk dari output: (batch_size * num_heads, jumlah queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        
        # Bentuk dari output_concat: (batch_size, jumlah queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab jax
class MultiHeadAttention(nn.Module):  #@save
    num_hiddens: int
    num_heads: int
    dropout: float
    bias: bool = False

    def setup(self):
        self.attention = d2l.DotProductAttention(self.dropout)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_k = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_v = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        # Bentuk dari queries, keys, atau values:
        # (batch_size, jumlah queries atau pasangan key-value, num_hiddens)
        # Bentuk dari valid_lens: (batch_size,) atau (batch_size, jumlah queries)
        # Setelah ditranspos, bentuk dari output queries, keys, atau values:
        # (batch_size * num_heads, jumlah queries atau pasangan key-value,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # Pada axis 0, salin item pertama (skalar atau vektor) sebanyak num_heads
            # kali, lalu salin item berikutnya, dan seterusnya
            valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

        # Bentuk dari output: (batch_size * num_heads, jumlah queries,
        # num_hiddens / num_heads)
        output, attention_weights = self.attention(
            queries, keys, values, valid_lens, training=training)
        # Bentuk dari output_concat: (batch_size, jumlah queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attention_weights
```

Untuk memungkinkan [**komputasi paralel dari beberapa kepala**],
kelas `MultiHeadAttention` di atas menggunakan dua metode transposisi seperti yang didefinisikan di bawah ini.
Secara spesifik,
metode `transpose_output` membalikkan operasi
dari metode `transpose_qkv`.


```{.python .input}
%%tab mxnet
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposisi untuk komputasi paralel dari beberapa kepala atensi."""
    # Bentuk input X: (batch_size, jumlah query atau pasangan key-value,
    # num_hiddens). Bentuk output X: (batch_size, jumlah query atau
    # pasangan key-value, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Bentuk output X: (batch_size, num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    X = X.transpose(0, 2, 1, 3)
    # Bentuk output: (batch_size * num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Membalikkan operasi dari transpose_qkv."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposisi untuk komputasi paralel dari beberapa kepala atensi."""
    # Bentuk input X: (batch_size, jumlah query atau pasangan key-value,
    # num_hiddens). Bentuk output X: (batch_size, jumlah query atau
    # pasangan key-value, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Bentuk output X: (batch_size, num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # Bentuk output: (batch_size * num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Membalikkan operasi dari transpose_qkv."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposisi untuk komputasi paralel dari beberapa kepala atensi."""
    # Bentuk input X: (batch_size, jumlah query atau pasangan key-value,
    # num_hiddens). Bentuk output X: (batch_size, jumlah query atau
    # pasangan key-value, num_heads, num_hiddens / num_heads)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
    # Bentuk output X: (batch_size, num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    # Bentuk output: (batch_size * num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Membalikkan operasi dari transpose_qkv."""
    X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transposisi untuk komputasi paralel dari beberapa kepala atensi."""
    # Bentuk input X: (batch_size, jumlah query atau pasangan key-value,
    # num_hiddens). Bentuk output X: (batch_size, jumlah query atau
    # pasangan key-value, num_heads, num_hiddens / num_heads)
    X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
    # Bentuk output X: (batch_size, num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    X = jnp.transpose(X, (0, 2, 1, 3))
    # Bentuk output: (batch_size * num_heads, jumlah query atau pasangan
    # key-value, num_hiddens / num_heads)
    return X.reshape((-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """Membalikkan operasi dari transpose_qkv."""
    X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
    X = jnp.transpose(X, (0, 2, 1, 3))
    return X.reshape((X.shape[0], X.shape[1], -1))
```

Mari kita [**menguji implementasi**] kelas `MultiHeadAttention` kita
menggunakan contoh sederhana di mana kunci (keys) dan nilai (values) adalah sama.
Sebagai hasilnya,
bentuk dari output multi-head attention adalah
(`batch_size`, `num_queries`, `num_hiddens`).


```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, Y, Y, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## Ringkasan

Multi-head attention menggabungkan pengetahuan dari pooling attention yang sama 
melalui subruang representasi yang berbeda dari query, key, dan value.
Untuk menghitung beberapa head dari multi-head attention secara paralel, 
dibutuhkan manipulasi tensor yang tepat.


## Latihan

1. Visualisasikan bobot perhatian (attention weights) dari beberapa head dalam percobaan ini.
2. Misalkan kita memiliki model yang telah dilatih berdasarkan multi-head attention dan kita ingin memangkas head perhatian yang kurang penting untuk meningkatkan kecepatan prediksi. Bagaimana kita dapat merancang eksperimen untuk mengukur pentingnya sebuah head perhatian?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/1634)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1635)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3869)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18029)
:end_tab:
