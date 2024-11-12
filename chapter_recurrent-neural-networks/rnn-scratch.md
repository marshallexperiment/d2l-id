# Implementasi Recurrent Neural Network dari Awal
:label:`sec_rnn-scratch`

Sekarang kita siap untuk mengimplementasikan RNN dari awal.
Secara khusus, kita akan melatih RNN ini untuk berfungsi
sebagai model bahasa tingkat karakter
(lihat :numref:`sec_rnn`)
dan melatihnya pada korpus yang terdiri dari
keseluruhan teks dari *The Time Machine* karya H. G. Wells,
mengikuti langkah-langkah pemrosesan data 
yang dijelaskan di :numref:`sec_text-sequence`.
Kita mulai dengan memuat dataset.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## Model RNN

Kita mulai dengan mendefinisikan sebuah kelas 
untuk mengimplementasikan model RNN
(:numref:`subsec_rnn_w_hidden_states`).
Perhatikan bahwa jumlah unit tersembunyi `num_hiddens` 
adalah sebuah *hyperparameter* yang dapat disesuaikan.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """Model RNN yang diimplementasikan dari awal."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """Model RNN yang diimplementasikan dari awal."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**Metode `forward` di bawah ini mendefinisikan cara menghitung 
output dan *hidden state* pada setiap langkah waktu, 
dengan diberikan input saat ini dan *state* model 
pada langkah waktu sebelumnya.**]
Perhatikan bahwa model RNN ini melakukan iterasi melalui 
dimensi terluar dari `inputs`, 
memperbarui *hidden state* 
satu langkah waktu pada satu waktu.
Model di sini menggunakan fungsi aktivasi $\tanh$ (:numref:`subsec_tanh`).


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

Kita dapat memasukkan *minibatch* dari urutan input ke dalam model RNN seperti berikut.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

Mari kita periksa apakah model RNN 
menghasilkan bentuk output yang benar 
untuk memastikan bahwa dimensi 
*hidden state* tetap tidak berubah.


```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Memeriksa panjang dari sebuah list."""
    assert len(a) == n, f'panjang list {len(a)} != panjang yang diharapkan {n}'
    
def check_shape(a, shape):  #@save
    """Memeriksa bentuk dari sebuah tensor."""
    assert a.shape == shape, \
            f'bentuk tensor {a.shape} != bentuk yang diharapkan {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## Model Bahasa Berbasis RNN

Kelas `RNNLMScratch` berikut mendefinisikan 
model bahasa berbasis RNN,
di mana kita memasukkan RNN kita 
melalui argumen `rnn`
dalam metode `__init__`.
Saat melatih model bahasa, 
input dan output berasal 
dari kosakata yang sama. 
Oleh karena itu, mereka memiliki dimensi yang sama,
yaitu sama dengan ukuran kosakata.
Perhatikan bahwa kita menggunakan *perplexity* untuk mengevaluasi model. 
Seperti yang dibahas di :numref:`subsec_perplexity`, ini memastikan 
bahwa urutan dengan panjang yang berbeda dapat dibandingkan.


```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """Model bahasa berbasis RNN yang diimplementasikan dari awal."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """Model bahasa berbasis RNN yang diimplementasikan dari awal."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """Model bahasa berbasis RNN yang diimplementasikan dari awal."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**One-Hot Encoding**]

Ingat bahwa setiap token direpresentasikan 
oleh sebuah indeks numerik yang menunjukkan
posisinya dalam kosakata untuk
kata/karakter/potongan kata yang bersesuaian.
Anda mungkin tergoda untuk membangun jaringan saraf
dengan satu node input (pada setiap langkah waktu),
di mana indeks dapat dimasukkan sebagai nilai skalar.
Ini berfungsi jika kita menangani input numerik 
seperti harga atau suhu, di mana dua nilai yang
cukup berdekatan seharusnya diperlakukan mirip.
Namun, pendekatan ini tidak sepenuhnya masuk akal.
Kata ke-$45$ dan ke-$46$ 
dalam kosakata kita kebetulan adalah "their" dan "said",
yang artinya sama sekali tidak mirip.

Saat menangani data kategorikal seperti ini,
strategi yang paling umum adalah merepresentasikan
setiap item dengan *one-hot encoding*
(lihat kembali dari :numref:`subsec_classification-problem`).
*One-hot encoding* adalah vektor yang panjangnya
ditentukan oleh ukuran kosakata $N$,
di mana semua entri diatur ke $0$,
kecuali entri yang sesuai 
dengan token kita, yang diatur ke $1$.
Sebagai contoh, jika kosakata memiliki lima elemen,
maka vektor *one-hot* yang sesuai 
dengan indeks 0 dan 2 adalah sebagai berikut.


```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(***Minibatch* yang kita sampel di setiap iterasi
akan berbentuk (ukuran batch, jumlah langkah waktu).
Setelah merepresentasikan setiap input sebagai vektor *one-hot*,
kita dapat memandang setiap *minibatch* sebagai tensor tiga dimensi, 
di mana panjang sepanjang sumbu ketiga 
ditentukan oleh ukuran kosakata (`len(vocab)`).**)
Kita sering melakukan transposisi pada input sehingga kita akan mendapatkan output 
dengan bentuk (jumlah langkah waktu, ukuran batch, ukuran kosakata).
Ini akan memungkinkan kita untuk melakukan iterasi lebih mudah melalui dimensi terluar
untuk memperbarui *hidden state* dari sebuah *minibatch*,
langkah demi langkah waktu
(misalnya, dalam metode `forward` di atas).


```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### Mentransformasi Output RNN

Model bahasa menggunakan lapisan *fully connected* pada output 
untuk mentransformasi output RNN menjadi prediksi token pada setiap langkah waktu.


```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Mari kita [**periksa apakah perhitungan *forward* menghasilkan output dengan bentuk yang benar.**]


```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradient Clipping**]


Sementara Anda sudah terbiasa menganggap jaringan saraf sebagai "dalam" dalam arti bahwa banyak lapisan memisahkan input dan output bahkan dalam satu langkah waktu, panjang urutan memperkenalkan gagasan kedalaman yang baru. Selain melalui jaringan dalam arah input-ke-output, input pada langkah waktu pertama harus melewati rantai $T$ lapisan sepanjang langkah waktu agar dapat memengaruhi output model pada langkah waktu terakhir. 

Jika kita melihat dari sudut pandang kebalikannya, dalam setiap iterasi, kita melakukan backpropagation gradien melalui waktu, yang menghasilkan rantai perkalian matriks dengan panjang $\mathcal{O}(T)$. Seperti yang disebutkan di :numref:`sec_numerical_stability`, ini dapat mengakibatkan ketidakstabilan numerik, yang menyebabkan gradien meledak atau menghilang, tergantung pada sifat matriks bobot.

Mengatasi gradien yang menghilang dan meledak adalah masalah mendasar saat merancang RNN dan telah menginspirasi beberapa kemajuan terbesar dalam arsitektur jaringan saraf modern. Pada bab berikutnya, kita akan membahas arsitektur khusus yang dirancang untuk mengatasi masalah gradien yang menghilang. Namun, bahkan RNN modern sering mengalami gradien yang meledak. Salah satu solusi yang sederhana namun umum adalah dengan membatasi gradien, memaksa nilai gradien "terpotong" ini menjadi lebih kecil.

Secara umum, ketika kita mengoptimalkan suatu tujuan dengan penurunan gradien, kita secara iteratif memperbarui parameter yang diinginkan, misalnya vektor $\mathbf{x}$, dengan mendorongnya ke arah gradien negatif $\mathbf{g}$ (dalam *stochastic gradient descent*, kita menghitung gradien ini pada *minibatch* yang dipilih secara acak). Misalnya, dengan laju pembelajaran $\eta > 0$, setiap pembaruan berbentuk $\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$. 

Mari kita asumsikan bahwa fungsi tujuan $f$ cukup mulus. Secara formal, kita mengatakan bahwa tujuan ini *Lipschitz continuous* dengan konstanta $L$, yang berarti bahwa untuk setiap $\mathbf{x}$ dan $\mathbf{y}$, kita memiliki

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Seperti yang dapat Anda lihat, ketika kita memperbarui vektor parameter dengan mengurangkan $\eta \mathbf{g}$, perubahan nilai fungsi tujuan bergantung pada laju pembelajaran, norma gradien, dan $L$ sebagai berikut:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

Dengan kata lain, tujuan tidak dapat berubah lebih dari $L \eta \|\mathbf{g}\|$. Memiliki nilai kecil untuk batas atas ini dapat dianggap baik atau buruk. Di sisi negatif, kita membatasi kecepatan di mana kita dapat mengurangi nilai tujuan. Di sisi positif, ini membatasi sejauh mana kita bisa membuat kesalahan dalam satu langkah gradien.

Ketika kita mengatakan bahwa gradien meledak, kita maksudkan bahwa $\|\mathbf{g}\|$ menjadi sangat besar. Dalam kasus terburuk, kita mungkin melakukan kerusakan besar dalam satu langkah gradien sehingga kita dapat membatalkan semua kemajuan yang telah dicapai selama ribuan iterasi pelatihan. Ketika gradien menjadi sangat besar, pelatihan jaringan saraf sering kali gagal karena tidak dapat mengurangi nilai tujuan. Terkadang, pelatihan akhirnya mencapai konvergensi tetapi tidak stabil karena lonjakan besar pada *loss*.

Salah satu cara untuk membatasi ukuran $L \eta \|\mathbf{g}\|$ adalah dengan memperkecil laju pembelajaran $\eta$ hingga nilai yang sangat kecil. Ini memiliki keuntungan bahwa kita tidak mendistorsi pembaruan. Tetapi bagaimana jika kita hanya *sesekali* mendapatkan gradien besar? Pendekatan drastis ini akan memperlambat kemajuan kita di semua langkah, hanya untuk menangani peristiwa gradien meledak yang jarang terjadi. Alternatif populer adalah menggunakan heuristik *gradient clipping* yang memproyeksikan gradien $\mathbf{g}$ ke dalam bola dengan radius tertentu $\theta$ sebagai berikut:


(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

Ini memastikan bahwa norma gradien tidak pernah melebihi $\theta$ 
dan bahwa gradien yang diperbarui sepenuhnya selaras 
dengan arah asli dari $\mathbf{g}$.
Pendekatan ini juga memiliki efek samping yang diinginkan, 
yaitu membatasi pengaruh yang dapat diberikan oleh setiap *minibatch* 
(dan di dalamnya setiap sampel tertentu) 
pada vektor parameter. 
Ini memberikan tingkat ketahanan tertentu pada model. 
Untuk lebih jelasnya, ini adalah trik. 
*Gradient clipping* berarti kita tidak selalu
mengikuti gradien sejati, dan sulit 
untuk memprediksi secara analitik efek samping yang mungkin terjadi.
Namun, ini adalah trik yang sangat berguna
dan banyak digunakan dalam implementasi RNN
di sebagian besar *framework* pembelajaran mendalam.


Di bawah ini, kita mendefinisikan metode untuk melakukan *gradient clipping*,
yang dipanggil oleh metode `fit_epoch` dari
kelas `d2l.Trainer` (lihat :numref:`sec_linear_scratch`).
Perhatikan bahwa saat menghitung norma gradien,
kita menggabungkan semua parameter model,
dan memperlakukannya sebagai vektor parameter besar tunggal.


```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Pelatihan

Dengan menggunakan dataset *The Time Machine* (`data`),
kita melatih model bahasa tingkat karakter (`model`)
berdasarkan RNN (`rnn`) yang diimplementasikan dari awal.
Perhatikan bahwa kita pertama-tama menghitung gradien,
kemudian melakukan *clipping* pada gradien tersebut, dan akhirnya 
memperbarui parameter model
menggunakan gradien yang sudah dipotong.


```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Dekode

Setelah model bahasa dipelajari,
kita dapat menggunakannya tidak hanya untuk memprediksi token berikutnya
tetapi juga untuk terus memprediksi setiap token selanjutnya,
dengan memperlakukan token yang diprediksi sebelumnya seolah-olah
itu adalah token berikutnya dalam input. 
Kadang-kadang kita hanya ingin menghasilkan teks
seolah-olah kita memulai dari awal dokumen. 
Namun, sering kali berguna untuk memberikan *prefix* yang ditentukan oleh pengguna pada model bahasa.
Misalnya, jika kita mengembangkan fitur
*autocomplete* untuk mesin pencari
atau untuk membantu pengguna menulis email,
kita ingin memasukkan apa yang telah mereka tulis sejauh ini (*prefix*),
dan kemudian menghasilkan lanjutan yang mungkin.

[**Metode `predict` berikut
menghasilkan lanjutan, satu karakter pada satu waktu,
setelah menerima `prefix` yang disediakan oleh pengguna**].
Saat kita melakukan iterasi melalui karakter dalam `prefix`,
kita terus menerus meneruskan *hidden state*
ke langkah waktu berikutnya
tetapi tidak menghasilkan output.
Ini disebut sebagai periode *warm-up*.
Setelah menerima *prefix*, kita sekarang siap untuk
mulai menghasilkan karakter berikutnya,
di mana masing-masing karakter akan dimasukkan kembali ke dalam model 
sebagai input pada langkah waktu berikutnya.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Pada bagian berikut, kita menentukan *prefix* 
dan memintanya untuk menghasilkan 20 karakter tambahan.


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

Meskipun mengimplementasikan model RNN di atas dari awal sangat bermanfaat untuk pembelajaran, cara ini tidaklah praktis.
Di bagian selanjutnya, kita akan melihat cara memanfaatkan *framework* pembelajaran mendalam untuk membangun RNN
menggunakan arsitektur standar, serta mendapatkan peningkatan kinerja 
dengan mengandalkan fungsi pustaka yang sangat dioptimalkan.


## Ringkasan

Kita dapat melatih model bahasa berbasis RNN untuk menghasilkan teks yang mengikuti awalan teks yang disediakan oleh pengguna. 
Model bahasa RNN sederhana terdiri dari pengkodean input, pemodelan RNN, dan generasi output.
Selama pelatihan, *gradient clipping* dapat mengatasi masalah gradien meledak tetapi tidak mengatasi masalah gradien yang menghilang. Pada eksperimen ini, kita mengimplementasikan model bahasa RNN sederhana dan melatihnya dengan *gradient clipping* pada urutan teks yang ditokenisasi di tingkat karakter. Dengan memberikan *prefix*, kita dapat menggunakan model bahasa untuk menghasilkan lanjutan yang mungkin, yang terbukti berguna dalam banyak aplikasi, misalnya, fitur *autocomplete*.


## Latihan

1. Apakah model bahasa yang diimplementasikan memprediksi token berikutnya berdasarkan semua token sebelumnya hingga token pertama dalam *The Time Machine*?
2. Hyperparameter mana yang mengontrol panjang riwayat yang digunakan untuk prediksi?
3. Tunjukkan bahwa *one-hot encoding* setara dengan memilih *embedding* yang berbeda untuk setiap objek.
4. Sesuaikan *hyperparameter* (misalnya, jumlah epoch, jumlah unit tersembunyi, jumlah langkah waktu dalam *minibatch*, dan laju pembelajaran) untuk meningkatkan *perplexity*. Seberapa rendah yang bisa Anda capai dengan tetap mempertahankan arsitektur sederhana ini?
5. Ganti *one-hot encoding* dengan *embedding* yang dapat dipelajari. Apakah ini menghasilkan kinerja yang lebih baik?
6. Lakukan eksperimen untuk menentukan seberapa baik model bahasa ini, 
   yang dilatih pada *The Time Machine*, bekerja pada buku lain oleh H. G. Wells, 
   misalnya, *The War of the Worlds*.
7. Lakukan eksperimen lain untuk mengevaluasi *perplexity* model ini pada buku-buku yang ditulis oleh penulis lain.
8. Modifikasi metode prediksi agar menggunakan *sampling* 
   daripada memilih karakter berikutnya yang paling mungkin.
    * Apa yang terjadi?
    * Arahkan model ke output yang lebih mungkin, misalnya, 
      dengan mengambil sampel dari $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ untuk $\alpha > 1$.
9. Jalankan kode di bagian ini tanpa melakukan *clipping* pada gradien. Apa yang terjadi?
10. Ganti fungsi aktivasi yang digunakan dalam bagian ini dengan ReLU 
    dan ulangi eksperimen dalam bagian ini. Apakah kita masih memerlukan *gradient clipping*? Mengapa?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18014)
:end_tab:
