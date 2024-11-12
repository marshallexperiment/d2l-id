# Gated Recurrent Units (GRU)
:label:`sec_gru`

Ketika RNN, terutama arsitektur LSTM (:numref:`sec_lstm`), semakin populer selama tahun 2010-an, beberapa peneliti mulai bereksperimen dengan arsitektur yang lebih sederhana dengan harapan untuk mempertahankan ide utama dalam menggabungkan status internal dan mekanisme penggandaan, tetapi dengan tujuan untuk mempercepat perhitungan. Gated Recurrent Unit (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` menawarkan versi streamline dari memori sel LSTM yang sering kali memberikan kinerja yang sebanding namun dengan keuntungan lebih cepat dalam perhitungan :cite:`Chung.Gulcehre.Cho.ea.2014`.


```{.python .input  n=5}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=6}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=8}
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

## Gerbang Reset (_Reset Gate_) dan Gerbang Update (_Update Gate_)

Pada GRU, tiga gerbang pada LSTM digantikan dengan dua gerbang: *gerbang reset* dan *gerbang update*. Seperti pada LSTM, kedua gerbang ini diberikan aktivasi sigmoid, memaksa nilai mereka berada pada interval $(0, 1)$. Secara intuitif, gerbang reset mengendalikan seberapa banyak status sebelumnya yang masih perlu diingat. Begitu pula, gerbang update memungkinkan kita mengendalikan seberapa banyak status baru yang merupakan salinan dari status lama. :numref:`fig_gru_1` menggambarkan input untuk gerbang reset dan gerbang update pada GRU, diberikan input dari waktu langkah saat ini dan status tersembunyi dari waktu langkah sebelumnya. Output dari gerbang-gerbang ini dihasilkan oleh dua lapisan fully connected dengan fungsi aktivasi sigmoid.

![Menghitung gerbang reset dan gerbang update pada model GRU.](../img/gru-1.svg)
:label:`fig_gru_1`

Secara matematis, pada waktu langkah $t$, misalkan input adalah sebuah minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (jumlah contoh $=n$; jumlah input $=d$) dan status tersembunyi dari waktu langkah sebelumnya adalah $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (jumlah unit tersembunyi $=h$). Maka, gerbang reset $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ dan gerbang update $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ dihitung sebagai berikut:

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z}),
\end{aligned}
$$

di mana $\mathbf{W}_{\textrm{xr}}, \mathbf{W}_{\textrm{xz}} \in \mathbb{R}^{d \times h}$ dan $\mathbf{W}_{\textrm{hr}}, \mathbf{W}_{\textrm{hz}} \in \mathbb{R}^{h \times h}$ adalah parameter bobot, dan $\mathbf{b}_\textrm{r}, \mathbf{b}_\textrm{z} \in \mathbb{R}^{1 \times h}$ adalah parameter bias.


## Status Tersembunyi Kandidat (_Candidate Hidden State_)

Selanjutnya, kita integrasikan gerbang reset $\mathbf{R}_t$ dengan mekanisme pembaruan reguler pada persamaan :eqref:`rnn_h_with_state`, sehingga menghasilkan *status tersembunyi kandidat* $\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ pada waktu langkah $t$:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$
:eqlabel:`gru_tilde_H`

di mana $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ dan $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ adalah parameter bobot,
$\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$ adalah bias,
dan simbol $\odot$ adalah operator perkalian Hadamard (perkalian elemen-per-elemen).
Di sini, kita menggunakan fungsi aktivasi tanh.

Hasilnya adalah *kandidat*, karena kita masih perlu mengintegrasikan aksi dari gerbang update.
Dibandingkan dengan persamaan :eqref:`rnn_h_with_state`,
pengaruh dari status sebelumnya sekarang dapat dikurangi dengan perkalian elemen-per-elemen antara
$\mathbf{R}_t$ dan $\mathbf{H}_{t-1}$
pada persamaan :eqref:`gru_tilde_H`.
Apabila elemen-elemen dalam gerbang reset $\mathbf{R}_t$ mendekati 1,
maka kita mendapatkan kembali RNN standar seperti yang ada pada persamaan :eqref:`rnn_h_with_state`.
Untuk semua elemen pada gerbang reset $\mathbf{R}_t$ yang mendekati 0,
status tersembunyi kandidat merupakan hasil dari MLP dengan $\mathbf{X}_t$ sebagai input.
Dengan demikian, status tersembunyi yang sudah ada sebelumnya di-*reset* ke nilai default.

:numref:`fig_gru_2` menggambarkan alur komputasi setelah menerapkan gerbang reset.

![Menghitung status tersembunyi kandidat pada model GRU.](../img/gru-2.svg)
:label:`fig_gru_2`


## Hidden State

Akhirnya, kita perlu mengintegrasikan efek dari gerbang update $\mathbf{Z}_t$.
Ini menentukan sejauh mana status tersembunyi baru $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ 
menyerupai status lama $\mathbf{H}_{t-1}$ dibandingkan dengan seberapa banyak 
status tersebut menyerupai status kandidat baru $\tilde{\mathbf{H}}_t$.
Gerbang update $\mathbf{Z}_t$ dapat digunakan untuk tujuan ini,
dengan cara mengambil kombinasi cembung elemen-per-elemen dari 
$\mathbf{H}_{t-1}$ dan $\tilde{\mathbf{H}}_t$.
Ini menghasilkan persamaan pembaruan akhir untuk GRU:

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

Kapanpun gerbang update $\mathbf{Z}_t$ mendekati 1,
kita hanya mempertahankan status lama.
Dalam kasus ini informasi dari $\mathbf{X}_t$ diabaikan,
sehingga secara efektif melewati langkah waktu $t$ dalam rantai dependensi.
Sebaliknya, kapanpun $\mathbf{Z}_t$ mendekati 0,
status laten baru $\mathbf{H}_t$ mendekati status laten kandidat $\tilde{\mathbf{H}}_t$.
:numref:`fig_gru_3` menunjukkan alur komputasi setelah gerbang update diaktifkan.

![Menghitung status tersembunyi pada model GRU.](../img/gru-3.svg)
:label:`fig_gru_3`

Sebagai rangkuman, GRU memiliki dua fitur pembeda berikut:

* Gerbang reset membantu menangkap ketergantungan jangka pendek dalam urutan.
* Gerbang update membantu menangkap ketergantungan jangka panjang dalam urutan.


## Implementasi dari Awal

Untuk mendapatkan pemahaman yang lebih baik tentang model GRU, mari kita implementasikan dari awal.

### (**Inisialisasi Parameter Model**)

Langkah pertama adalah menginisialisasi parameter model.
Kita akan menarik bobot dari distribusi Gaussian
dengan standar deviasi `sigma` dan menetapkan bias ke 0.
Hyperparameter `num_hiddens` mendefinisikan jumlah unit tersembunyi.
Kita menginstansiasi semua bobot dan bias yang terkait dengan gerbang update, 
gerbang reset, dan status tersembunyi kandidat.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        if tab.selected('mxnet'):
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))            
        if tab.selected('pytorch'):
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        if tab.selected('tensorflow'):
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))            
            
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state        
```

```{.python .input}
%%tab jax
class GRUScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        init_weight = lambda name, shape: self.param(name,
                                                     nn.initializers.normal(self.sigma),
                                                     shape)
        triple = lambda name : (
            init_weight(f'W_x{name}', (self.num_inputs, self.num_hiddens)),
            init_weight(f'W_h{name}', (self.num_hiddens, self.num_hiddens)),
            self.param(f'b_{name}', nn.initializers.zeros, (self.num_hiddens)))

        self.W_xz, self.W_hz, self.b_z = triple('z')  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple('r')  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple('h')  # Candidate hidden state
```

### Mendefinisikan Model

Sekarang kita siap untuk [**mendefinisikan perhitungan maju GRU**].
Strukturnya sama dengan sel RNN dasar, 
kecuali bahwa persamaan update-nya lebih kompleks.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GRUScratch)
```python
def forward(self, inputs, H=None):
    if H is None:
        # Status awal dengan bentuk: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
    outputs = []
    for X in inputs:
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) +
                        d2l.matmul(H, self.W_hz) + self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) + 
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) + 
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H
```

```{.python .input}
%%tab jax
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    # Menggunakan primitif lax.scan untuk menggantikan perulangan pada
    # input, karena scan menghemat waktu dalam kompilasi jit.
    def scan_fn(H, X):
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) + d2l.matmul(H, self.W_hz) +
                        self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) +
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) +
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        return H, H  # Mengembalikan carry, y

    if H is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H

    # scan menerima scan_fn, status carry awal, xs dengan leading axis yang akan di-scan
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### Pelatihan

[**Melatih**] model bahasa pada dataset *The Time Machine* dilakukan dengan cara yang sama seperti pada :numref:`sec_rnn-scratch`.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Implementasi Singkat**]

Pada API tingkat tinggi, kita dapat langsung menginstansiasi model GRU.
Ini mengenkapsulasi semua detail konfigurasi yang telah kita buat secara eksplisit di atas.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens)
        if tab.selected('tensorflow'):
            self.rnn = tf.keras.layers.GRU(num_hiddens, return_sequences=True, 
                                           return_state=True)
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None, training=False):
        if H is None:
            batch_size = inputs.shape[1]
            H = nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                                            (batch_size,), self.num_hiddens)

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        H, outputs = GRU()(H, inputs)
        return outputs, H
```

Kode ini secara signifikan lebih cepat dalam pelatihan karena menggunakan operator yang telah dikompilasi dibandingkan dengan Python.


```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('jax'):
    gru = GRU(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

Setelah selesai melatih model, kita akan mencetak nilai *perplexity* pada data pelatihan dan urutan prediksi yang dihasilkan setelah diberikan awalan (*prefix*) tertentu.


```{.python .input}
%%tab mxnet, pytorch
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

Dibandingkan dengan LSTMs, GRUs memberikan performa yang serupa tetapi cenderung lebih ringan secara komputasi. Secara umum, dibandingkan dengan RNN sederhana, RNN dengan *gated mechanism* seperti LSTM dan GRU lebih baik dalam menangkap ketergantungan pada urutan yang memiliki jarak langkah waktu yang besar. GRU mengandung RNN dasar sebagai kasus ekstremnya ketika *reset gate* diaktifkan sepenuhnya. Mereka juga dapat melewati suburutan dengan mengaktifkan *update gate*.

## Latihan

1. Asumsikan kita hanya ingin menggunakan input pada langkah waktu $t'$ untuk memprediksi keluaran pada langkah waktu $t > t'$. Apa nilai terbaik untuk *reset gate* dan *update gate* pada setiap langkah waktu?
2. Atur hiperparameter dan analisis pengaruhnya terhadap waktu eksekusi, *perplexity*, dan urutan keluaran.
3. Bandingkan waktu eksekusi, *perplexity*, dan urutan keluaran untuk implementasi `rnn.RNN` dan `rnn.GRU`.
4. Apa yang terjadi jika Anda hanya mengimplementasikan sebagian dari GRU, misalnya hanya dengan *reset gate* atau hanya dengan *update gate*?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1056)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3860)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18017)
:end_tab:
