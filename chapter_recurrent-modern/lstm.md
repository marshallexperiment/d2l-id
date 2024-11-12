# Long Short-Term Memory (LSTM)
:label:`sec_lstm`

Tak lama setelah RNN gaya Elman pertama kali dilatih menggunakan backpropagation :cite:`elman1990finding`, masalah pembelajaran ketergantungan jangka panjang (karena gradien yang menghilang dan meledak) menjadi jelas, dengan Bengio dan Hochreiter membahas masalah ini :cite:`bengio1994learning,Hochreiter.Bengio.Frasconi.ea.2001`. Hochreiter telah mengartikulasikan masalah ini sejak tahun 1991 dalam tesis masternya, meskipun hasilnya tidak begitu dikenal karena tesis tersebut ditulis dalam bahasa Jerman. 

Meskipun *gradient clipping* membantu mengatasi gradien yang meledak, menangani gradien yang menghilang tampaknya membutuhkan solusi yang lebih rumit. Salah satu teknik pertama dan paling berhasil dalam menangani gradien yang menghilang adalah model *long short-term memory* (LSTM) yang diperkenalkan oleh :citet:`Hochreiter.Schmidhuber.1997`. LSTM mirip dengan jaringan saraf berulang (RNN) standar, tetapi di sini setiap node berulang biasa digantikan oleh *memory cell*. Setiap *memory cell* memiliki *internal state*, yaitu sebuah node dengan *self-connected recurrent edge* dengan bobot tetap 1, yang memastikan bahwa gradien dapat melewati banyak langkah waktu tanpa menghilang atau meledak.

Istilah "long short-term memory" berasal dari intuisi berikut. Jaringan saraf berulang sederhana memiliki *long-term memory* dalam bentuk bobot. Bobot ini berubah secara perlahan selama pelatihan, mengkodekan pengetahuan umum tentang data. Mereka juga memiliki *short-term memory* dalam bentuk aktivasi efemer, yang melewati setiap node ke node-node berikutnya. Model LSTM memperkenalkan jenis penyimpanan antara melalui *memory cell*. Sebuah *memory cell* adalah unit komposit, dibangun dari node-node yang lebih sederhana dalam pola konektivitas tertentu, dengan penambahan node multiplikatif yang baru.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
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

## Gated Memory Cell

Setiap *memory cell* dilengkapi dengan *internal state* dan sejumlah *multiplicative gate* yang menentukan apakah (i) input tertentu harus mempengaruhi *internal state* (disebut *input gate*), (ii) *internal state* harus dihapus menjadi $0$ (disebut *forget gate*), dan (iii) *internal state* dari suatu neuron tertentu harus diperbolehkan mempengaruhi keluaran *cell* tersebut (disebut *output gate*).

### Gated Hidden State

Perbedaan utama antara RNN biasa dan LSTM adalah bahwa LSTM mendukung pengaturan *gating* terhadap *hidden state*. Ini berarti kita memiliki mekanisme khusus untuk menentukan kapan *hidden state* harus *di-update* dan kapan harus *di-reset*. Mekanisme ini dipelajari dan digunakan untuk mengatasi masalah yang disebutkan di atas. Misalnya, jika token pertama sangat penting, kita akan mempelajari cara untuk tidak memperbarui *hidden state* setelah pengamatan pertama. Begitu juga, kita akan belajar melewati pengamatan sementara yang tidak relevan. Terakhir, kita akan belajar untuk mereset *latent state* kapan pun diperlukan. Kita akan membahas ini lebih rinci di bawah.

### Input Gate, Forget Gate, dan Output Gate

Data yang masuk ke dalam *gate* LSTM adalah input pada *time step* saat ini dan *hidden state* dari *time step* sebelumnya, seperti yang diilustrasikan pada :numref:`fig_lstm_0`. Tiga lapisan koneksi penuh dengan fungsi aktivasi sigmoid digunakan untuk menghitung nilai dari *input gate*, *forget gate*, dan *output gate*. Akibat dari aktivasi sigmoid, semua nilai dari ketiga *gate* tersebut berada dalam rentang $(0, 1)$. Selain itu, kita memerlukan sebuah *input node*, yang biasanya dihitung dengan fungsi aktivasi *tanh*. Secara intuitif, *input gate* menentukan seberapa banyak nilai dari *input node* yang harus ditambahkan ke dalam *internal state* *memory cell* saat ini. *Forget gate* menentukan apakah mempertahankan nilai saat ini dari *memory* atau menghapusnya. Sedangkan *output gate* menentukan apakah *memory cell* harus mempengaruhi keluaran pada *time step* saat ini.

![Menghitung *input gate*, *forget gate*, dan *output gate* pada model LSTM.](../img/lstm-0.svg)
:label:`fig_lstm_0`

Secara matematis, misalkan ada $h$ unit tersembunyi, ukuran *batch* adalah $n$, dan jumlah input adalah $d$. Dengan demikian, input adalah $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ dan *hidden state* dari *time step* sebelumnya adalah $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$. Sesuai, *gate* pada *time step* $t$ didefinisikan sebagai berikut: *input gate* adalah $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, *forget gate* adalah $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, dan *output gate* adalah $\mathbf{O}_t \in \mathbb{R}^{n \times h}$. Mereka dihitung sebagai berikut:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$

dengan $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$ dan $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$ adalah parameter bobot, dan $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$ adalah parameter bias. Perhatikan bahwa *broadcasting* (lihat :numref:`subsec_broadcasting`) dipicu selama penjumlahan. Kita menggunakan fungsi sigmoid (seperti yang diperkenalkan di :numref:`sec_mlp`) untuk memetakan nilai input ke interval $(0, 1)$.



### Node Input

Selanjutnya kita merancang *memory cell*. Karena kita belum menetapkan tindakan dari berbagai *gate*, kita pertama-tama memperkenalkan *input node* $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$. Perhitungannya mirip dengan tiga *gate* yang dijelaskan di atas, tetapi menggunakan fungsi $\tanh$ dengan rentang nilai $(-1, 1)$ sebagai fungsi aktivasi. Hal ini mengarah pada persamaan berikut pada *time step* $t$:

$$\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}),$$

dengan $\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$ dan $\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$ adalah parameter bobot, dan $\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$ adalah parameter bias.

Ilustrasi singkat dari *input node* ditampilkan di :numref:`fig_lstm_1`.

![Menghitung *input node* dalam model LSTM.](../img/lstm-1.svg)
:label:`fig_lstm_1`


### Internal State *Memory Cell*

Pada LSTM, *input gate* $\mathbf{I}_t$ mengatur seberapa banyak data baru yang kita perhitungkan melalui $\tilde{\mathbf{C}}_t$, dan *forget gate* $\mathbf{F}_t$ menentukan seberapa banyak keadaan internal *memory cell* lama $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ yang kita pertahankan. Menggunakan operator *Hadamard* (perkalian elemen demi elemen) $\odot$, kita tiba pada persamaan pembaruan berikut:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

Jika *forget gate* selalu 1 dan *input gate* selalu 0, *internal state memory cell* $\mathbf{C}_{t-1}$ akan tetap konstan selamanya, diteruskan tanpa perubahan ke setiap *time step* berikutnya. Namun, *input gate* dan *forget gate* memberikan fleksibilitas pada model untuk mempelajari kapan harus mempertahankan nilai ini tetap tidak berubah dan kapan harus mengganggunya sebagai respons terhadap input berikutnya. Secara praktik, desain ini mengurangi masalah *vanishing gradient*, sehingga menghasilkan model yang jauh lebih mudah untuk dilatih, terutama ketika berhadapan dengan dataset dengan panjang urutan yang besar.

Kita akhirnya sampai pada diagram aliran pada :numref:`fig_lstm_2`.

![Menghitung keadaan internal *memory cell* dalam model LSTM.](../img/lstm-2.svg)

:label:`fig_lstm_2`



### Keadaan Tersembunyi (Hidden State)

Terakhir, kita perlu mendefinisikan bagaimana menghitung keluaran dari *memory cell*, yaitu keadaan tersembunyi $\mathbf{H}_t \in \mathbb{R}^{n \times h}$, seperti yang terlihat oleh lapisan lainnya. Di sinilah *output gate* berperan. Pada LSTM, kita pertama-tama menerapkan $\tanh$ pada keadaan internal *memory cell* dan kemudian menerapkan perkalian elemen demi elemen lainnya, kali ini dengan *output gate*. Ini memastikan bahwa nilai $\mathbf{H}_t$ selalu berada dalam interval $(-1, 1)$:

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$


Kapan pun *output gate* mendekati nilai 1, kita memungkinkan keadaan internal *memory cell* memengaruhi lapisan berikutnya tanpa hambatan, sedangkan untuk nilai *output gate* yang mendekati 0, kita mencegah memori saat ini memengaruhi lapisan lain dalam jaringan pada *time step* saat ini. Perhatikan bahwa sebuah *memory cell* dapat mengumpulkan informasi melintasi banyak *time step* tanpa memengaruhi sisa jaringan (selama *output gate* mengambil nilai mendekati 0), dan kemudian tiba-tiba memengaruhi jaringan pada *time step* berikutnya begitu *output gate* beralih dari nilai mendekati 0 ke nilai mendekati 1. :numref:`fig_lstm_3` memberikan ilustrasi grafis dari aliran data.

![Menghitung keadaan tersembunyi dalam model LSTM.](../img/lstm-3.svg)
:label:`fig_lstm_3`



## Implementasi dari Awal

Sekarang mari kita implementasikan LSTM dari awal.
Sama seperti eksperimen di :numref:`sec_rnn-scratch`, kita pertama-tama memuat dataset *The Time Machine*.

### [**Inisialisasi Parameter Model**]

Selanjutnya, kita perlu mendefinisikan dan menginisialisasi parameter model. 
Seperti sebelumnya, hiperparameter `num_hiddens` menentukan jumlah unit tersembunyi. 
Kita menginisialisasi bobot menggunakan distribusi Gaussian dengan deviasi standar 0.01, 
dan kita menetapkan nilai bias menjadi 0.



```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node
```

```{.python .input}
%%tab jax
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple('i')  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple('f')  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple('o')  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple('c')  # Input node
```

:begin_tab:`pytorch, mxnet, tensorflow`
[**Model aktual**] didefinisikan seperti yang dijelaskan di atas,
terdiri dari tiga gerbang dan sebuah node input.
Perlu dicatat bahwa hanya state tersembunyi yang diteruskan ke layer output.
:end_tab:

:begin_tab:`jax`
[**Model aktual**] didefinisikan seperti yang dijelaskan di atas,
terdiri dari tiga gerbang dan sebuah node input.
Perlu dicatat bahwa hanya state tersembunyi yang diteruskan ke layer output.
Loop for yang panjang dalam metode `forward` akan menghasilkan waktu JIT compilation yang sangat lama untuk run pertama. Sebagai solusi untuk ini, daripada menggunakan loop for untuk memperbarui state di setiap time step, JAX memiliki utilitas transformasi `jax.lax.scan` untuk mencapai perilaku yang sama. `scan` ini mengambil state awal yang disebut `carry` dan array `inputs` yang akan di-scan pada axis terdepannya. Transformasi `scan` ini pada akhirnya akan mengembalikan state akhir dan output yang sudah ditumpuk seperti yang diharapkan.
:end_tab:


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
            C = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    # Use lax.scan primitive instead of looping over the
    # inputs, since scan saves time in jit compilation.
    def scan_fn(carry, X):
        H, C = carry
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) + (
            d2l.matmul(H, self.W_hi)) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        return (H, C), H  # return carry, y

    if H_C is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens)), \
                jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H_C

    # scan takes the scan_fn, initial carry state, xs with leading axis to be scanned
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### [**Pelatihan**] dan Prediksi

Mari kita latih model LSTM dengan membuat instans dari kelas `RNNLMScratch` dari :numref:`sec_rnn-scratch`.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Implementasi Ringkas**]

Dengan menggunakan API tingkat tinggi,
kita dapat langsung membuat instans model LSTM.
Ini mengenkapsulasi semua detail konfigurasi
yang telah kita jelaskan sebelumnya.
Kodenya secara signifikan lebih cepat karena menggunakan
operator yang telah dikompilasi daripada Python
untuk banyak detail yang sebelumnya telah kita uraikan.


```{.python .input}
%%tab mxnet
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.LSTM(num_hiddens)

    def forward(self, inputs, H_C=None):
        if H_C is None: H_C = self.rnn.begin_state(
            inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab pytorch
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab tensorflow
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.LSTM(
                num_hiddens, return_sequences=True,
                return_state=True, time_major=True)

    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(inputs, H_C)
        return outputs, H_C
```

```{.python .input}
%%tab jax
class LSTM(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H_C=None, training=False):
        if H_C is None:
            batch_size = inputs.shape[1]
            H_C = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0),
                                                        (batch_size,),
                                                        self.num_hiddens)

        LSTM = nn.scan(nn.OptimizedLSTMCell, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})

        H_C, outputs = LSTM()(H_C, inputs)
        return outputs, H_C
```

```{.python .input}
%%tab all
if tab.selected('pytorch'):
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'tensorflow', 'jax'):
    lstm = LSTM(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

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

LSTM adalah model autoregresif dengan variabel laten yang prototipikal dengan kontrol status yang tidak sepele.
Banyak variannya telah diusulkan selama bertahun-tahun, misalnya, beberapa lapisan, koneksi residual, dan berbagai jenis regularisasi. Namun, pelatihan LSTM dan model urutan lainnya (seperti GRU) cukup mahal karena ketergantungan urutan yang panjang.
Nantinya kita akan menemukan model alternatif seperti Transformer yang dapat digunakan dalam beberapa kasus.


## Ringkasan

Meskipun LSTM diterbitkan pada tahun 1997,
model ini meraih popularitas besar
dengan beberapa kemenangan dalam kompetisi prediksi di pertengahan tahun 2000-an,
dan menjadi model dominan untuk pembelajaran urutan dari tahun 2011
hingga munculnya model Transformer, dimulai pada tahun 2017.
Bahkan Transformer memiliki beberapa ide kunci yang diilhami oleh
inovasi desain arsitektur yang diperkenalkan oleh LSTM.

LSTM memiliki tiga jenis gerbang:
gerbang input, gerbang lupa, dan gerbang output
yang mengontrol aliran informasi.
Output lapisan tersembunyi dari LSTM mencakup status tersembunyi dan status internal memori sel.
Hanya status tersembunyi yang diteruskan ke lapisan output, sedangkan
status internal memori sel tetap sepenuhnya internal.
LSTM dapat mengatasi gradien yang menghilang dan meledak.


## Latihan

1. Sesuaikan hyperparameter dan analisis pengaruhnya terhadap waktu pelatihan, perplexity, dan urutan keluaran.
2. Bagaimana Anda perlu mengubah model untuk menghasilkan kata yang tepat daripada hanya urutan karakter?
3. Bandingkan biaya komputasi untuk GRU, LSTM, dan RNN biasa untuk dimensi tersembunyi yang sama. Perhatikan secara khusus biaya pelatihan dan inferensi.
4. Karena memori sel kandidat memastikan bahwa rentang nilai berada di antara $-1$ dan $1$ dengan menggunakan fungsi $\tanh$, mengapa status tersembunyi perlu menggunakan fungsi $\tanh$ lagi untuk memastikan bahwa rentang nilai keluaran berada di antara $-1$ dan $1$?
5. Implementasikan model LSTM untuk prediksi deret waktu daripada prediksi urutan karakter.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1057)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3861)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18016)
:end_tab:
