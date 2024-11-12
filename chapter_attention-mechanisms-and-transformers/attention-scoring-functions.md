```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Fungsi Skor Perhatian (_Attention Scoring Function_)
:label:`sec_attention-scoring-functions`

Dalam :numref:`sec_attention-pooling`, kita telah menggunakan sejumlah kernel berbasis jarak yang berbeda, termasuk kernel Gaussian untuk memodelkan interaksi antara *query* dan *key*. Ternyata, fungsi jarak sedikit lebih mahal untuk dihitung dibandingkan dengan produk titik (*dot product*). Oleh karena itu, dengan operasi *softmax* untuk memastikan bobot perhatian yang tidak negatif, banyak pekerjaan yang dilakukan dalam *fungsi skor perhatian* $a$ dalam :eqref:`eq_softmax_attention` dan :numref:`fig_attention_output` yang lebih sederhana untuk dihitung.

![Menghitung output dari pooling perhatian sebagai rata-rata tertimbang dari nilai-nilai, di mana bobot dihitung dengan fungsi skor perhatian $\mathit{a}$ dan operasi softmax.](../img/attention-output.svg)
:label:`fig_attention_output`


```{.python .input}
%%tab mxnet
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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
import math
```

## [**Perhatian Produk Titik/_Dot Product Attention_ **]

Mari kita ulas kembali fungsi perhatian (tanpa eksponensiasi) dari kernel Gaussian:

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.
$$

Pertama, perhatikan bahwa suku terakhir hanya bergantung pada $\mathbf{q}$. Dengan demikian, suku ini identik untuk semua pasangan $(\mathbf{q}, \mathbf{k}_i)$. Normalisasi bobot perhatian menjadi 1, seperti yang dilakukan pada :eqref:`eq_softmax_attention`, memastikan bahwa suku ini sepenuhnya menghilang. Kedua, perhatikan bahwa baik normalisasi batch maupun normalisasi layer (yang akan dibahas nanti) menghasilkan aktivasi dengan norma $\|\mathbf{k}_i\|$ yang terbatas dengan baik, dan sering kali konstan. Ini terjadi, misalnya, saat *keys* $\mathbf{k}_i$ dihasilkan oleh *layer norm*. Dengan demikian, kita bisa menghilangkannya dari definisi $a$ tanpa perubahan signifikan pada hasilnya.

Terakhir, kita perlu menjaga agar skala argumen dalam fungsi eksponensial tetap terkendali. Asumsikan bahwa semua elemen dari *query* $\mathbf{q} \in \mathbb{R}^d$ dan *key* $\mathbf{k}_i \in \mathbb{R}^d$ adalah variabel acak yang independen dan identik terdistribusi dengan rata-rata nol dan varians satu. Produk titik antara kedua vektor memiliki rata-rata nol dan varians $d$. Untuk memastikan bahwa varians produk titik tetap 1 terlepas dari panjang vektor, kita menggunakan fungsi skor perhatian *produk titik terukur* (*scaled dot product attention*). Artinya, kita menskalakan ulang produk titik dengan $1/\sqrt{d}$. Dengan demikian, kita tiba pada fungsi perhatian pertama yang sering digunakan, misalnya dalam Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

Perhatikan bahwa bobot perhatian $\alpha$ masih memerlukan normalisasi. Kita dapat menyederhanakan ini lebih lanjut melalui :eqref:`eq_softmax_attention` dengan menggunakan operasi *softmax*:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$
:eqlabel:`eq_attn-scoring-alpha`

Ternyata, semua mekanisme perhatian populer menggunakan *softmax*, sehingga kita akan membatasi diri kita pada hal ini untuk sisa bab ini.

## Fungsi-Fungsi Pendukung (_Convinience Function_)

Kita memerlukan beberapa fungsi untuk membuat mekanisme perhatian lebih efisien untuk diterapkan. Ini termasuk alat untuk menangani string dengan panjang variabel (umum dalam pemrosesan bahasa alami) dan alat untuk evaluasi yang efisien pada minibatch (perkalian matriks batch).

### [**Operasi Softmax Bertopeng**]

Salah satu aplikasi mekanisme perhatian yang paling populer adalah pada model sekuensial. Oleh karena itu, kita perlu dapat menangani sekuens dengan panjang yang berbeda-beda. Dalam beberapa kasus, sekuens-sekuens tersebut mungkin berakhir dalam minibatch yang sama, sehingga membutuhkan padding dengan token dummy untuk sekuens yang lebih pendek (lihat :numref:`sec_machine_translation` untuk contohnya). Token khusus ini tidak membawa arti. Misalnya, asumsikan bahwa kita memiliki tiga kalimat berikut:


```
Dive  into  Deep    Learning 
Learn to    code    <blank>
Hello world <blank> <blank>
```


Karena kita tidak ingin memiliki bagian kosong dalam model atensi kita, kita hanya perlu membatasi $\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ menjadi $\sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ sesuai dengan panjang aktual kalimatnya, di mana $l \leq n$. Karena masalah ini sangat umum, masalah ini memiliki nama khusus: *operasi softmax bertopeng* (masked softmax operation).

Mari kita implementasikan ini. Sebenarnya, implementasinya sedikit "menipu" dengan mengatur nilai $\mathbf{v}_i$ untuk $i > l$ menjadi nol. Selain itu, ia mengatur bobot atensi ke angka negatif yang sangat besar, seperti $-10^{6}$, agar kontribusi mereka terhadap gradien dan nilai benar-benar hilang dalam praktiknya. Ini dilakukan karena kernel aljabar linear dan operator-operatornya sangat dioptimalkan untuk GPU, sehingga lebih cepat untuk sedikit membuang-buang komputasi daripada harus memiliki kode dengan pernyataan kondisional (if then else).


```{.python .input}
%%tab mxnet
def masked_softmax(X, valid_lens):  #@save
    """Melakukan operasi softmax dengan memasking elemen pada sumbu terakhir."""
    # X: tensor 3D, valid_lens: tensor 1D atau 2D
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Pada sumbu terakhir, gantikan elemen yang dimasking dengan nilai 
        # negatif yang sangat besar, yang eksponensiasinya menghasilkan 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
%%tab pytorch
def masked_softmax(X, valid_lens):  #@save
    """Melakukan operasi softmax dengan memasking elemen pada sumbu terakhir."""
    # X: tensor 3D, valid_lens: tensor 1D atau 2D 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Pada sumbu terakhir, gantikan elemen yang dimasking dengan nilai 
        # negatif yang sangat besar, yang eksponensiasinya menghasilkan 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
%%tab tensorflow
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: tensor 3D, valid_lens: tensor 1D atau 2D 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)
    
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # Pada sumbu terakhir, gantikan elemen yang dimasking dengan nilai 
        # negatif yang sangat besar, yang eksponensiasinya menghasilkan 0
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```{.python .input}
%%tab jax
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: tensor 3D, valid_lens: tensor 1D atau 2D 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Pada sumbu terakhir, gantikan elemen yang dimasking dengan nilai 
        # negatif yang sangat besar, yang eksponensiasinya menghasilkan 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)
```

Untuk [**mengilustrasikan bagaimana fungsi ini bekerja**], pertimbangkan sebuah minibatch yang terdiri dari dua contoh dengan ukuran $2 \times 4$, di mana panjang yang valid untuk masing-masing adalah $2$ dan $3$. 
Akibat dari operasi softmax yang dimasking, nilai-nilai yang melebihi panjang valid untuk setiap pasangan vektor akan dimasking menjadi nol.


```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)), jnp.array([2, 3]))
```

Jika kita memerlukan kontrol yang lebih rinci untuk menentukan panjang valid untuk setiap dua vektor dari setiap contoh, kita cukup menggunakan tensor dua dimensi untuk panjang yang valid. Ini menghasilkan:


```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform((2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)),
               jnp.array([[1, 3], [2, 4]]))
```

### Perkalian Matriks Batch
:label:`subsec_batch_dot`

Operasi lain yang sering digunakan adalah mengalikan batch dari matriks satu dengan yang lainnya. Operasi ini sangat berguna ketika kita memiliki minibatch dari query, key, dan value. Secara lebih spesifik, misalkan:

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \\
    \mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}.
$$

Kemudian, perkalian matriks batch (*Batch Matrix Multiplication* atau BMM) menghitung hasil kali elemen secara per elemen:

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}.$$
:eqlabel:`eq_batch-matrix-mul`

Mari kita lihat bagaimana ini bekerja dalam kerangka deep learning.




```{.python .input}
%%tab mxnet
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(npx.batch_dot(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab pytorch
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab tensorflow
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(tf.matmul(Q, K).numpy(), (2, 3, 6))
```

```{.python .input}
%%tab jax
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(jax.lax.batch_matmul(Q, K), (2, 3, 6))
```

## [**Scaled Dot Product Attention**]

Mari kita kembali ke *dot product attention* yang diperkenalkan di :eqref:`eq_dot_product_attention`. 
Secara umum, ini memerlukan baik *query* maupun *key* memiliki panjang vektor yang sama, misalkan $d$, meskipun hal ini dapat dengan mudah diatasi dengan mengganti 
$\mathbf{q}^\top \mathbf{k}$ dengan $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ di mana $\mathbf{M}$ adalah matriks yang dipilih dengan tepat untuk menerjemahkan antara kedua ruang. Untuk sekarang, asumsikan bahwa dimensinya cocok.

Dalam praktiknya, kita sering memikirkan minibatch untuk efisiensi,
seperti menghitung *attention* untuk $n$ *query* dan $m$ pasangan *key-value*,
di mana *query* dan *key* memiliki panjang $d$
dan *value* memiliki panjang $v$. *Scaled dot product attention* 
dari *query* $\mathbf Q\in\mathbb R^{n\times d}$,
*key* $\mathbf K\in\mathbb R^{m\times d}$,
dan *value* $\mathbf V\in\mathbb R^{m\times v}$
dapat ditulis sebagai

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

Perhatikan bahwa saat menerapkannya pada sebuah minibatch, kita memerlukan perkalian matriks batch (*batch matrix multiplication*) yang diperkenalkan di :eqref:`eq_batch-matrix-mul`. Dalam implementasi berikut dari *scaled dot product attention*, kita menggunakan *dropout* untuk regularisasi model.




```{.python .input}
%%tab mxnet
class DotProductAttention(nn.Block):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Bentuk queries: (batch_size, jumlah queries, d)
    # Bentuk keys: (batch_size, jumlah pasangan key-value, d)
    # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi value)
    # Bentuk valid_lens: (batch_size,) atau (batch_size, jumlah queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set transpose_b=True untuk menukar dua dimensi terakhir dari keys
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Bentuk queries: (batch_size, jumlah queries, d)
    # Bentuk keys: (batch_size, jumlah pasangan key-value, d)
    # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi value)
    # Bentuk valid_lens: (batch_size,) atau (batch_size, jumlah queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Tukar dua dimensi terakhir dari keys dengan keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class DotProductAttention(tf.keras.layers.Layer):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # Bentuk queries: (batch_size, jumlah queries, d)
    # Bentuk keys: (batch_size, jumlah pasangan key-value, d)
    # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi value)
    # Bentuk valid_lens: (batch_size,) atau (batch_size, jumlah queries)
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    dropout: float

    # Bentuk queries: (batch_size, jumlah queries, d)
    # Bentuk keys: (batch_size, jumlah pasangan key-value, d)
    # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi value)
    # Bentuk valid_lens: (batch_size,) atau (batch_size, jumlah queries)
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None,
                 training=False):
        d = queries.shape[-1]
        # Tukar dua dimensi terakhir dari keys dengan keys.swapaxes(1, 2)
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights
```

Untuk [**mengilustrasikan bagaimana kelas `DotProductAttention` bekerja**], kita menggunakan `keys`, `values`, dan `valid lengths` yang sama dari contoh sebelumnya untuk additive attention. Dalam contoh ini, 
kita mengasumsikan bahwa kita memiliki ukuran minibatch sebesar $2$, total $10$ keys dan values, dan bahwa dimensi dari values adalah $4$. 
Terakhir, kita mengasumsikan bahwa panjang yang valid per observasi adalah $2$ dan $6$ secara berturut-turut. Berdasarkan hal ini, kita mengharapkan output berupa tensor berukuran $2 \times 1 \times 4$, 
yaitu satu baris per contoh dalam minibatch.


```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
keys = tf.random.normal(shape=(2, 10, 2))
values = tf.random.normal(shape=(2, 10, 4))
valid_lens = tf.constant([2, 6])

attention = DotProductAttention(dropout=0.5)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 2))
keys = jax.random.normal(d2l.get_key(), (2, 10, 2))
values = jax.random.normal(d2l.get_key(), (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

Mari kita cek apakah bobot perhatian (*attention weights*) benar-benar bernilai nol untuk semua kolom di luar kolom kedua dan keenam masing-masing (karena kita menetapkan panjang yang valid menjadi $2$ dan $6$).


```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## [**Perhatian Aditif**]
:label:`subsec_additive-attention`

Ketika kueri $\mathbf{q}$ dan kunci $\mathbf{k}$ adalah vektor dengan dimensi yang berbeda, kita dapat menggunakan matriks untuk mengatasi ketidakcocokan melalui $\mathbf{q}^\top \mathbf{M} \mathbf{k}$, atau kita dapat menggunakan *additive attention* sebagai fungsi skor. Manfaat lainnya adalah, seperti yang ditunjukkan oleh namanya, perhatian ini bersifat aditif. Hal ini dapat memberikan sedikit penghematan komputasi. 

Diberikan sebuah kueri $\mathbf{q} \in \mathbb{R}^q$ dan sebuah kunci $\mathbf{k} \in \mathbb{R}^k$, fungsi skor *additive attention* :cite:`Bahdanau.Cho.Bengio.2014` diberikan oleh:

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

dengan $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$, dan $\mathbf w_v\in\mathbb R^{h}$ adalah parameter yang dapat dipelajari. Nilai ini kemudian dimasukkan ke dalam *softmax* untuk memastikan non-negativitas dan normalisasi. 

Interpretasi yang setara dari :eqref:`eq_additive-attn` adalah bahwa kueri dan kunci digabungkan dan dimasukkan ke dalam MLP (*Multi-Layer Perceptron*) dengan satu lapisan tersembunyi. Menggunakan $\tanh$ sebagai fungsi aktivasi dan menonaktifkan bias, kita mengimplementasikan *additive attention* sebagai berikut:


```{.python .input}
%%tab mxnet
class AdditiveAttention(nn.Block):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Gunakan flatten=False agar hanya mengubah sumbu terakhir sehingga
        # bentuk untuk sumbu lainnya tetap sama
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Setelah memperluas dimensi, bentuk queries: (batch_size, jumlah
        # queries, 1, num_hiddens) dan bentuk keys: (batch_size, 1,
        # jumlah pasangan key-value, num_hiddens). Jumlahkan dengan
        # broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # Hanya ada satu output dari self.w_v, jadi kita menghapus entri
        # satu dimensi terakhir dari bentuknya. Bentuk scores:
        # (batch_size, jumlah queries, jumlah pasangan key-value)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi
        # nilai)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Setelah ekspansi dimensi, bentuk queries: (batch_size, jumlah
        # queries, 1, num_hiddens) dan bentuk keys: (batch_size, 1, jumlah
        # pasangan key-value, num_hiddens). Jumlahkan mereka dengan
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # Hanya ada satu output dari self.w_v, jadi kita menghapus entri
        # satu dimensi terakhir dari bentuknya. Bentuk scores: (batch_size,
        # jumlah queries, jumlah pasangan key-value)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi
        # nilai)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class AdditiveAttention(tf.keras.layers.Layer):  #@save
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Setelah ekspansi dimensi, bentuk queries: (batch_size, jumlah
        # queries, 1, num_hiddens) dan bentuk keys: (batch_size, 1, jumlah
        # pasangan key-value, num_hiddens). Jumlahkan mereka dengan
        # broadcasting
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # Hanya ada satu output dari self.w_v, jadi kita menghapus entri
        # satu dimensi terakhir dari bentuknya. Bentuk scores: (batch_size,
        # jumlah queries, jumlah pasangan key-value)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi
        # nilai)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class AdditiveAttention(nn.Module):  #@save
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Setelah ekspansi dimensi, bentuk queries: (batch_size, jumlah
        # queries, 1, num_hiddens) dan bentuk keys: (batch_size, 1, jumlah
        # pasangan key-value, num_hiddens). Jumlahkan dengan broadcasting
        features = jnp.expand_dims(queries, axis=2) + jnp.expand_dims(keys, axis=1)
        features = nn.tanh(features)
        # Hanya ada satu output dari self.w_v, jadi kita menghapus entri
        # satu dimensi terakhir dari bentuknya. Bentuk scores: (batch_size,
        # jumlah queries, jumlah pasangan key-value)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        # Bentuk values: (batch_size, jumlah pasangan key-value, dimensi nilai)
        return dropout_layer(attention_weights) @ values, attention_weights
```

## Melihat Cara Kerja `AdditiveAttention`

Mari kita [**lihat bagaimana `AdditiveAttention` bekerja**]. Dalam contoh sederhana ini, kita memilih queries, keys, dan values masing-masing berukuran 
$(2, 1, 20)$, $(2, 10, 2)$, dan $(2, 10, 4)$. Ini identik dengan pilihan kita untuk `DotProductAttention`, kecuali bahwa sekarang queries adalah $20$-dimensi. 
Demikian juga, kita memilih $(2, 6)$ sebagai panjang valid untuk urutan dalam minibatch.


```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 20))

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 20))
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

When reviewing the attention function we see a behavior that is qualitatively quite similar to that of `DotProductAttention`. That is, only terms within the chosen valid length $(2, 6)$ are nonzero.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## Ringkasan

Dalam bagian ini, kita memperkenalkan dua fungsi *attention scoring* utama: *dot product* dan *additive attention*. Mereka adalah alat yang efektif untuk menggabungkan informasi dari urutan dengan panjang yang bervariasi. Secara khusus, *dot product attention* adalah dasar dari arsitektur Transformer modern. Ketika *queries* dan *keys* adalah vektor dengan panjang yang berbeda, kita dapat menggunakan fungsi *additive attention* sebagai pengganti. Mengoptimalkan lapisan-lapisan ini adalah salah satu area kemajuan utama dalam beberapa tahun terakhir. Sebagai contoh, [Perpustakaan Transformer dari NVIDIA](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) dan Megatron :cite:`shoeybi2019megatron` sangat bergantung pada varian yang efisien dari mekanisme *attention*. Kita akan membahas ini lebih detail saat meninjau Transformer pada bagian-bagian selanjutnya.

## Latihan

1. Implementasikan *distance-based attention* dengan memodifikasi kode `DotProductAttention`. Perhatikan bahwa Anda hanya memerlukan norma kuadrat dari *keys* $\|\mathbf{k}_i\|^2$ untuk implementasi yang efisien.
2. Modifikasi *dot product attention* untuk memungkinkan *queries* dan *keys* dengan dimensi yang berbeda dengan menggunakan matriks untuk menyesuaikan dimensi.
3. Bagaimana skala biaya komputasi dengan dimensi dari *keys*, *queries*, *values*, dan jumlahnya? Bagaimana dengan kebutuhan lebar pita memori?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1064)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3867)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18027)
:end_tab:
