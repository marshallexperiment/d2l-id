# Deep Recurrent Neural Networks
:label:`sec_deep_rnn`

Hingga saat ini, kita fokus pada mendefinisikan jaringan yang terdiri dari masukan sekuensial, satu lapisan RNN tersembunyi, dan satu lapisan keluaran.
Meskipun hanya memiliki satu lapisan tersembunyi antara masukan pada setiap langkah waktu dan keluaran yang sesuai, ada sebuah pengertian di mana jaringan ini bisa dianggap dalam (deep).
Input dari langkah waktu pertama dapat mempengaruhi keluaran pada langkah waktu terakhir $T$ (sering kali 100 atau 1000 langkah kemudian).
Input ini melewati $T$ aplikasi dari lapisan rekuren sebelum mencapai keluaran akhir.
Namun, kita sering juga ingin mempertahankan kemampuan untuk mengekspresikan hubungan yang kompleks antara input pada langkah waktu tertentu dengan keluaran pada langkah waktu yang sama.
Untuk itu, kita sering kali membangun RNN yang dalam tidak hanya dalam arah waktu, tetapi juga dalam arah input-ke-output.
Ini adalah pengertian kedalaman yang sudah kita temui saat mengembangkan MLP dan deep CNN.

Metode standar untuk membangun jenis deep RNN ini sangat sederhana: kita menumpuk RNN secara bertumpuk.
Diberikan sekuens dengan panjang $T$, RNN pertama menghasilkan sekuens keluaran dengan panjang $T$.
Selanjutnya, keluaran tersebut menjadi masukan bagi lapisan RNN berikutnya.
Pada bagian ini, kami akan menggambarkan pola desain ini dan memberikan contoh sederhana bagaimana mengimplementasikan deep RNN.
Di bawah ini, pada :numref:`fig_deep_rnn`, kami menggambarkan deep RNN dengan $L$ lapisan tersembunyi.
Setiap status tersembunyi bekerja pada masukan sekuensial dan menghasilkan keluaran sekuensial.
Selain itu, setiap sel RNN (kotak putih di :numref:`fig_deep_rnn`) pada setiap langkah waktu bergantung pada nilai dari lapisan yang sama pada langkah waktu sebelumnya dan nilai dari lapisan sebelumnya pada langkah waktu yang sama.

![Arsitektur Deep RNN.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

Secara formal, misalkan kita memiliki masukan minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (jumlah contoh $=n$; jumlah masukan dalam setiap contoh $=d$) pada langkah waktu $t$.
Pada saat yang sama, status tersembunyi dari lapisan tersembunyi ke-$l$ ($l=1,\ldots,L$) adalah $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$ (jumlah unit tersembunyi $=h$) dan variabel lapisan keluaran adalah $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (jumlah keluaran: $q$).
Menetapkan $\mathbf{H}_t^{(0)} = \mathbf{X}_t$, status tersembunyi dari lapisan tersembunyi ke-$l$ yang menggunakan fungsi aktivasi $\phi_l$ dihitung sebagai berikut:

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

dengan bobot $\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$ dan $\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$, bersama dengan bias $\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$, yang merupakan parameter model dari lapisan tersembunyi ke-$l$.

Pada akhirnya, perhitungan dari lapisan keluaran hanya didasarkan pada status tersembunyi dari lapisan tersembunyi terakhir $L^\textrm{th}$:

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

dengan bobot $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ dan bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ yang merupakan parameter model dari lapisan keluaran.

Sama seperti MLP, jumlah lapisan tersembunyi $L$ dan jumlah unit tersembunyi $h$ adalah hyperparameter yang bisa kita atur.
Lebar lapisan RNN yang umum ($h$) berada pada rentang $(64, 2056)$, dan kedalaman ($L$) berada pada rentang $(1, 8)$.
Selain itu, kita bisa dengan mudah mendapatkan deep-gated RNN dengan menggantikan perhitungan status tersembunyi dalam :eqref:`eq_deep_rnn_H` dengan LSTM atau GRU.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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
import jax
from jax import numpy as jnp
```

## Implementasi dari Awal

Untuk mengimplementasikan multilayer RNN dari awal,
kita bisa memperlakukan setiap lapisan sebagai instance `RNNScratch`
dengan parameter yang dapat dipelajari secara terpisah.


```{.python .input}
%%tab mxnet, tensorflow
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = [d2l.RNNScratch(num_inputs if i==0 else num_hiddens,
                                    num_hiddens, sigma)
                     for i in range(num_layers)]
```

```{.python .input}
%%tab pytorch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

```{.python .input}
%%tab jax
class StackedRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    num_layers: int
    sigma: float = 0.01

    def setup(self):
        self.rnns = [d2l.RNNScratch(self.num_inputs if i==0 else self.num_hiddens,
                                    self.num_hiddens, self.sigma)
                     for i in range(self.num_layers)]
```

Perhitungan maju multilayer
hanya melakukan perhitungan maju
lapis demi lapis.


```{.python .input}
%%tab all
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    return outputs, Hs
```

Sebagai contoh, kita melatih model GRU mendalam pada
dataset *The Time Machine* (sama seperti pada :numref:`sec_rnn-scratch`).
Agar tetap sederhana, kita menetapkan jumlah lapisan menjadi 2.


```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
    model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
        model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Implementasi Ringkas

:begin_tab:`pytorch, mxnet, tensorflow`
Untungnya, banyak rincian logistik yang diperlukan
untuk mengimplementasikan beberapa lapisan RNN
telah tersedia dalam API tingkat tinggi.
Implementasi ringkas kami akan menggunakan fungsionalitas bawaan tersebut.
Kode ini menggeneralisasi kode yang kami gunakan sebelumnya di :numref:`sec_gru`,
membiarkan kita menentukan jumlah lapisan secara eksplisit 
daripada memilih default hanya satu lapisan.
:end_tab:

:begin_tab:`jax`
Flax mengambil pendekatan minimalis saat mengimplementasikan RNN.
Mendefinisikan jumlah lapisan dalam RNN atau menggabungkannya dengan dropout
tidak tersedia langsung dari kotak.
Implementasi ringkas kami akan menggunakan semua fungsionalitas bawaan dan
menambahkan fitur `num_layers` dan `dropout`.
Kode ini menggeneralisasi kode yang kami gunakan sebelumnya di :numref:`sec_gru`,
memungkinkan kita untuk menentukan jumlah lapisan secara eksplisit
daripada memilih default satu lapisan.
:end_tab:


```{.python .input}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """ multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """ multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """ multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True, time_major=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(X, state)
        return outputs, state
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):  #@save
    """ multilayer GRU model."""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                    (batch_size,), self.num_hiddens)] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # Memperkenalkan Lapisan Dropout Setelah Setiap Lapisan GRU Kecuali yang Terakhir
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU()(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # Lapisan GRU terakhir tanpa dropout
        out_state, X = GRU()(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)
```

Keputusan arsitektural seperti memilih hyperparameter sangat mirip dengan yang ada di :numref:`sec_gru`.
Kami memilih jumlah input dan output yang sama dengan jumlah token yang berbeda, yaitu `vocab_size`.
Jumlah unit tersembunyi (hidden units) masih 32.
Satu-satunya perbedaan adalah bahwa sekarang kami (**memilih jumlah lapisan tersembunyi yang tidak trivial dengan menentukan nilai `num_layers`**).


```{.python .input}
%%tab mxnet
gru = GRU(num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# Menjalankan proses membutuhkan lebih dari 1 jam (menunggu perbaikan dari MXNet)
# trainer.fit(model, data)
# model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab pytorch, tensorflow, jax
if tab.selected('tensorflow', 'jax'):
    gru = GRU(num_hiddens=32, num_layers=2)
if tab.selected('pytorch'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
if tab.selected('pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

```{.python .input}
%%tab pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

## Ringkasan

Pada deep RNN, informasi status tersembunyi diteruskan ke langkah waktu berikutnya dari layer saat ini dan juga ke langkah waktu saat ini dari layer berikutnya. Ada banyak variasi dari deep RNN, seperti LSTM, GRU, atau vanilla RNN. Untungnya, model-model ini semua tersedia sebagai bagian dari API tingkat tinggi dalam framework deep learning. Inisialisasi model membutuhkan perhatian khusus. Secara keseluruhan, deep RNN memerlukan banyak pekerjaan (seperti pengaturan learning rate dan clipping) untuk memastikan konvergensi yang tepat.

## Latihan

1. Gantilah GRU dengan LSTM dan bandingkan akurasi serta kecepatan pelatihannya.
2. Tambahkan data pelatihan untuk mencakup beberapa buku. Seberapa rendah kamu bisa mencapai dalam skala perplexity?
3. Apakah kamu ingin menggabungkan sumber dari penulis yang berbeda ketika membuat model teks? Mengapa ini ide yang bagus? Apa yang bisa salah?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1058)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3862)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18018)
:end_tab:
