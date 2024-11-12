# Implementasi Singkat dari Recurrent Neural Networks
:label:`sec_rnn-concise`

Seperti kebanyakan implementasi kita dari awal,
:numref:`sec_rnn-scratch` dirancang 
untuk memberikan pemahaman tentang cara kerja setiap komponen.
Namun, ketika Anda menggunakan RNN setiap hari 
atau menulis kode produksi,
Anda akan ingin lebih bergantung pada pustaka
yang mengurangi waktu implementasi 
(dengan menyediakan kode pustaka untuk model dan fungsi umum)
dan waktu komputasi 
(dengan mengoptimalkan pustaka ini secara maksimal).
Bagian ini akan menunjukkan kepada Anda cara mengimplementasikan 
model bahasa yang sama dengan lebih efisien
menggunakan API tingkat tinggi yang disediakan 
oleh framework pembelajaran mendalam yang Anda gunakan.
Kita mulai, seperti sebelumnya, dengan memuat 
dataset *The Time Machine*.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
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
from jax import numpy as jnp
```

## [**Mendefinisikan Model**]

Kita mendefinisikan kelas berikut
dengan menggunakan RNN yang diimplementasikan
oleh API tingkat tinggi.

:begin_tab:`mxnet`
Secara khusus, untuk menginisialisasi *hidden state*,
kita memanggil metode anggota `begin_state`.
Metode ini mengembalikan sebuah daftar yang berisi
*hidden state* awal
untuk setiap contoh dalam *minibatch*,
dengan bentuk
(jumlah lapisan tersembunyi, ukuran batch, jumlah unit tersembunyi).
Untuk beberapa model yang akan diperkenalkan nanti
(misalnya, *long short-term memory*),
daftar ini juga akan berisi informasi lain.
:end_tab:

:begin_tab:`jax`
Flax saat ini tidak menyediakan RNNCell untuk implementasi singkat dari RNN biasa.
Namun, terdapat varian RNN yang lebih maju seperti LSTM dan GRU
yang tersedia dalam API `linen` dari Flax.
:end_tab:


```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """Model RNN yang diimplementasikan dengan High-level API."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """Model RNN yang diimplementasikan dengan High-level API."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """Model RNN yang diimplementasikan dengan High-level API."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """Model RNN yang diimplementasikan dengan High-level API."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

Mewarisi dari kelas `RNNLMScratch` di :numref:`sec_rnn-scratch`, 
kelas `RNNLM` berikut mendefinisikan sebuah model bahasa berbasis RNN yang lengkap.
Perhatikan bahwa kita perlu membuat lapisan keluaran *fully connected* yang terpisah.


```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """Model bahasa berbasis RNN yang diimplementasikan dengan High-level API."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """Model bahasa berbasis RNN yang diimplementasikan dengan High-level API."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """Model bahasa berbasis RNN yang diimplementasikan dengan High-level API."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## Pelatihan dan Prediksi

Sebelum melatih model, mari kita [**membuat prediksi 
dengan model yang diinisialisasi dengan bobot acak.**]
Karena kita belum melatih jaringan, 
model akan menghasilkan prediksi yang tidak masuk akal.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Selanjutnya, kita akan [**melatih model kita dengan memanfaatkan High-level API**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

Dibandingkan dengan :numref:`sec_rnn-scratch`,
model ini mencapai perplexity yang sebanding,
tetapi berjalan lebih cepat berkat implementasi yang dioptimalkan.
Seperti sebelumnya, kita dapat menghasilkan token yang diprediksi 
berdasarkan string awalan yang telah ditentukan.


```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Ringkasan

API tingkat tinggi dalam framework pembelajaran mendalam menyediakan implementasi dari RNN standar.
Pustaka ini membantu Anda menghindari pemborosan waktu untuk mengimplementasikan ulang model standar.
Selain itu, 
implementasi framework sering kali sangat dioptimalkan, 
sehingga memberikan peningkatan kinerja (komputasi) yang signifikan 
dibandingkan dengan implementasi dari awal.

## Latihan

1. Bisakah Anda membuat model RNN mengalami *overfit* menggunakan API tingkat tinggi?
2. Implementasikan model *autoregressive* dari :numref:`sec_sequence` menggunakan RNN.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18015)
:end_tab:
