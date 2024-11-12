```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Pembelajaran Sequence-to-Sequence untuk Terjemahan Mesin
:label:`sec_seq2seq`

Dalam masalah yang disebut sebagai sequence-to-sequence, seperti dalam terjemahan mesin
(seperti yang dibahas dalam :numref:`sec_machine_translation`),
dimana input dan output masing-masing terdiri dari urutan dengan panjang variabel yang tidak selaras,
kita umumnya mengandalkan arsitektur encoder-decoder
(:numref:`sec_encoder-decoder`).
Pada bagian ini,
kita akan mendemonstrasikan aplikasi dari arsitektur encoder-decoder,
di mana baik encoder dan decoder diimplementasikan sebagai RNN,
untuk tugas terjemahan mesin
:cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`.

Di sini, RNN encoder akan menerima urutan dengan panjang variabel sebagai input
dan mengubahnya menjadi keadaan tersembunyi (hidden state) dengan bentuk tetap.
Kemudian, pada :numref:`chap_attention-and-transformers`,
kita akan memperkenalkan mekanisme perhatian (attention),
yang memungkinkan kita untuk mengakses input yang di-encode
tanpa harus mengompresi seluruh input
menjadi representasi dengan panjang tetap.

Untuk menghasilkan urutan output,
satu token pada satu waktu,
model decoder,
yang terdiri dari RNN terpisah,
akan memprediksi setiap token target berikutnya
dengan memperhatikan baik urutan input
maupun token sebelumnya dalam output.
Selama pelatihan, decoder biasanya
akan dikondisikan pada token sebelumnya
dalam label "ground truth" resmi.
Namun, pada waktu pengujian, kita akan mengondisikan
setiap output dari decoder pada token yang sudah diprediksi.
Perlu dicatat bahwa jika kita mengabaikan encoder,
decoder dalam arsitektur sequence-to-sequence
berperilaku seperti model bahasa biasa.
:numref:`fig_seq2seq` mengilustrasikan
bagaimana menggunakan dua RNN
untuk pembelajaran sequence-to-sequence
dalam terjemahan mesin.

![Pembelajaran sequence-to-sequence dengan encoder RNN dan decoder RNN.](../img/seq2seq.svg)
:label:`fig_seq2seq`

Dalam :numref:`fig_seq2seq`,
token spesial "<eos>"
menandakan akhir dari urutan.
Model kita dapat berhenti membuat prediksi
begitu token ini dihasilkan.
Pada langkah waktu awal dari RNN decoder,
ada dua keputusan desain spesial yang perlu diperhatikan:
Pertama, kita memulai setiap input dengan token
awal urutan "<bos>".
Kedua, kita dapat memberi
keadaan tersembunyi akhir dari encoder
ke decoder
pada setiap langkah waktu decoding :cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014`.
Dalam beberapa desain lain,
seperti yang dibuat oleh :citet:`Sutskever.Vinyals.Le.2014`,
keadaan tersembunyi akhir dari encoder RNN
digunakan
untuk memulai keadaan tersembunyi decoder
hanya pada langkah decoding pertama.


```{.python .input}
%%tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import math
import optax
```

## Teacher Forcing

Sementara menjalankan encoder pada urutan input
tergolong relatif mudah,
menangani input dan output 
dari decoder memerlukan lebih banyak perhatian. 
Pendekatan yang paling umum disebut *teacher forcing*.
Di sini, urutan target asli (label token)
diberikan ke decoder sebagai input.
Secara lebih konkret,
token awal urutan spesial
dan urutan target asli,
tanpa token terakhir,
dikombinasikan sebagai input untuk decoder,
sementara output decoder (label untuk pelatihan) adalah
urutan target asli,
digeser satu token:
"<bos>", "Ils", "regardent", "." $\rightarrow$
"Ils", "regardent", ".", "<eos>" (:numref:`fig_seq2seq`).

Implementasi kita dalam
:numref:`subsec_loading-seq-fixed-len`
mempersiapkan data pelatihan untuk teacher forcing,
di mana pergeseran token untuk pembelajaran self-supervised
mirip dengan pelatihan model bahasa dalam
:numref:`sec_language-model`.
Pendekatan alternatif adalah
memberi token *prediksi*
dari langkah waktu sebelumnya
sebagai input saat ini ke decoder.

Pada bagian berikut, kami menjelaskan desain
yang digambarkan pada :numref:`fig_seq2seq`
dengan lebih detail.
Kita akan melatih model ini untuk terjemahan mesin
pada dataset bahasa Inggris-Prancis seperti yang diperkenalkan dalam
:numref:`sec_machine_translation`.

## Encoder

Ingatlah bahwa encoder mengubah urutan input dengan panjang variabel
menjadi *variabel konteks* $\mathbf{c}$ dengan bentuk tetap (lihat :numref:`fig_seq2seq`).

Pertimbangkan contoh urutan tunggal (ukuran batch 1).
Misalkan urutan input adalah $x_1, \ldots, x_T$, 
di mana $x_t$ adalah token ke-$t$.
Pada langkah waktu $t$, RNN mengubah
vektor fitur input $\mathbf{x}_t$ untuk $x_t$
dan status tersembunyi $\mathbf{h} _{t-1}$ 
dari langkah waktu sebelumnya 
menjadi status tersembunyi saat ini $\mathbf{h}_t$.
Kita dapat menggunakan fungsi $f$ untuk menyatakan 
transformasi dari lapisan rekursif RNN:

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

Secara umum, encoder mengubah 
status tersembunyi di semua langkah waktu
menjadi variabel konteks melalui fungsi khusus $q$:

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

Sebagai contoh, dalam :numref:`fig_seq2seq`,
variabel konteks adalah status tersembunyi $\mathbf{h}_T$
yang sesuai dengan representasi encoder RNN
setelah memproses token terakhir dari urutan input.

Dalam contoh ini, kita telah menggunakan RNN searah
untuk merancang encoder,
di mana status tersembunyi hanya bergantung pada suburutan input
pada dan sebelum langkah waktu dari status tersembunyi.
Kita juga dapat membuat encoder menggunakan RNN dua arah.
Dalam hal ini, status tersembunyi bergantung pada suburutan sebelum dan setelah langkah waktu
(termasuk input pada langkah waktu saat ini), 
yang mengkodekan informasi dari seluruh urutan.



Sekarang mari kita [**mengimplementasikan encoder RNN**].
Perhatikan bahwa kita menggunakan *embedding layer*
untuk mendapatkan vektor fitur untuk setiap token dalam urutan input.
Bobot dari embedding layer adalah sebuah matriks,
di mana jumlah baris sesuai dengan
ukuran dari kosakata input (`vocab_size`)
dan jumlah kolom sesuai dengan
dimensi dari vektor fitur (`embed_size`).
Untuk setiap indeks token input $i$,
embedding layer akan mengambil baris ke-$i$ 
(dari indeks 0) dari matriks bobot
untuk mengembalikan vektor fiturnya.
Di sini, kita mengimplementasikan encoder menggunakan GRU multilayer.


```{.python .input}
%%tab mxnet
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.initialize(init.Xavier())
            
    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs shape: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab pytorch
def init_seq2seq(module):  #@save
    """Inisialisasi bobot untuk pembelajaran urutan-ke-urutan (sequence-to-sequence)."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
```

```{.python .input}
%%tab pytorch
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)
            
    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab tensorflow
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
            
    def call(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs shape: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab jax
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)

    def __call__(self, X, *args, training=False):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs, training=training)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

Mari kita gunakan sebuah contoh konkret untuk [**mengilustrasikan implementasi encoder di atas**]. Di bawah ini, kita menginstansiasi sebuah encoder GRU dengan dua lapisan yang memiliki jumlah unit tersembunyi sebanyak 16. Diberikan sebuah minibatch dari input urutan `X` (ukuran batch $=4$; jumlah time step $=9$), keadaan tersembunyi dari lapisan terakhir di semua time step (`enc_outputs` yang dikembalikan oleh lapisan berulang dari encoder) adalah tensor dengan bentuk (jumlah time step, ukuran batch, jumlah hidden unit).


```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = d2l.zeros((batch_size, num_steps))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    enc_outputs, enc_state = encoder(X)
if tab.selected('jax'):
    (enc_outputs, enc_state), _ = encoder.init_with_output(d2l.get_key(), X)

d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
```

Karena kita menggunakan GRU di sini, bentuk dari keadaan tersembunyi berlapis di time step terakhir adalah (jumlah lapisan tersembunyi, ukuran batch, jumlah unit tersembunyi).


```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(enc_state, (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(enc_state, num_layers)
    d2l.check_shape(enc_state[0], (batch_size, num_hiddens))
```

## [**Decoder**]
:label:`sec_seq2seq_decoder`

Diberikan urutan output target $y_1, y_2, \ldots, y_{T'}$
untuk setiap time step $t'$
(kita menggunakan $t^\prime$ untuk membedakan dari urutan time step input),
decoder memberikan probabilitas prediksi
untuk setiap kemungkinan token yang terjadi di langkah $y_{t'+1}$
yang dikondisikan pada token-token sebelumnya dalam target
$y_1, \ldots, y_{t'}$ 
dan variabel konteks 
$\mathbf{c}$, yaitu $P(y_{t'+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$.

Untuk memprediksi token berikutnya pada $t^\prime+1$ dalam urutan target,
decoder RNN mengambil token target dari langkah sebelumnya $y_{t^\prime}$,
keadaan tersembunyi RNN dari langkah sebelumnya $\mathbf{s}_{t^\prime-1}$,
dan variabel konteks $\mathbf{c}$ sebagai inputnya,
dan mentransformasikannya menjadi keadaan tersembunyi 
$\mathbf{s}_{t^\prime}$ pada time step saat ini.
Kita dapat menggunakan fungsi $g$ untuk mengekspresikan 
transformasi dari lapisan tersembunyi decoder:

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

Setelah mendapatkan keadaan tersembunyi dari decoder,
kita dapat menggunakan lapisan output dan operasi softmax 
untuk menghitung distribusi prediktif
$p(y_{t^{\prime}+1} \mid y_1, \ldots, y_{t^\prime}, \mathbf{c})$ 
terhadap token output selanjutnya ${t^\prime+1}$.

Mengikuti :numref:`fig_seq2seq`,
ketika mengimplementasikan decoder seperti berikut ini,
kita langsung menggunakan keadaan tersembunyi pada time step terakhir
dari encoder
untuk menginisialisasi keadaan tersembunyi dari decoder.
Ini mensyaratkan bahwa encoder RNN dan decoder RNN 
memiliki jumlah lapisan dan unit tersembunyi yang sama.
Untuk lebih menggabungkan informasi urutan input yang telah dienkode,
variabel konteks dikonkatenasikan
dengan input decoder pada semua time step.
Untuk memprediksi distribusi probabilitas token output,
kita menggunakan lapisan fully connected
untuk mentransformasi keadaan tersembunyi 
pada lapisan terakhir dari decoder RNN.


```{.python .input}
%%tab mxnet
class Seq2SeqDecoder(d2l.Decoder):
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs 

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = np.tile(context, (embs.shape[0], 1, 1))
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size+num_hiddens, num_hiddens,
                           num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = tf.keras.layers.Dense(vocab_size)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def call(self, X, state):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = tf.tile(tf.expand_dims(context, 0), (embs.shape[0], 1, 1))
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.transpose(self.dense(outputs), (1, 0, 2))
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab jax
class Seq2SeqDecoder(d2l.Decoder):
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def __call__(self, X, state, training=False):
        # X shape: (batch_size, num_steps)
        # embs shape: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context shape: (batch_size, num_hiddens)
        context = enc_output[-1]
        # Broadcast context to (num_steps, batch_size, num_hiddens)
        context = jnp.tile(context, (embs.shape[0], 1, 1))
        # Concat at the feature dimension
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state,
                                         training=training)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        # hidden_state shape: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

Untuk [**mengilustrasikan decoder yang telah diimplementasikan**],
di bawah ini kita menginstansiasi decoder tersebut dengan hyperparameter yang sama seperti pada encoder yang disebutkan sebelumnya.
Seperti yang dapat kita lihat, bentuk (shape) dari output decoder menjadi (ukuran batch, jumlah time step, ukuran kosakata),
di mana dimensi terakhir dari tensor menyimpan distribusi token yang diprediksi.

```{.python .input}
%%tab all
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
if tab.selected('jax'):
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(), X)[0])
    (dec_outputs, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                       state)


d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(state[1], num_layers)
    d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

Lapisan-lapisan dalam model RNN encoder--decoder di atas 
diringkas dalam :numref:`fig_seq2seq_details`.

![Lapisan-lapisan dalam model RNN encoder--decoder.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`



## Encoder--Decoder untuk Pembelajaran Urutan-ke-Urutan (Sequence-to-Sequence)


Menggabungkan semuanya dalam kode menghasilkan yang berikut ini:


```{.python .input}
%%tab pytorch, tensorflow, mxnet
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """Encoder RNN untuk pembelajaran sequence-to-sequence."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()
        
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        # Adam optimizer digunakan disini
        if tab.selected('mxnet'):
            return gluon.Trainer(self.parameters(), 'adam',
                                 {'learning_rate': self.lr})
        if tab.selected('pytorch'):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        if tab.selected('tensorflow'):
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
```

```{.python .input}
%%tab jax
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """Encoder RNN--decoder untuk pembelajaran sequence-to-sequence."""
    encoder: nn.Module
    decoder: nn.Module
    tgt_pad: int
    lr: float

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return optax.adam(learning_rate=self.lr)
```

## Fungsi Kerugian dengan Masking

Pada setiap langkah waktu, decoder memprediksi
distribusi probabilitas untuk token output.
Seperti pada pemodelan bahasa, 
kita dapat menerapkan softmax 
untuk mendapatkan distribusi tersebut
dan menghitung cross-entropy loss untuk optimasi.
Ingat dari :numref:`sec_machine_translation`
bahwa token padding khusus 
ditambahkan di akhir urutan 
sehingga urutan dengan panjang yang berbeda
dapat dimuat secara efisien
dalam minibatch dengan bentuk yang sama.
Namun, prediksi token padding
harus dikecualikan dari perhitungan kerugian.
Untuk tujuan ini, kita dapat 
[**masking entri yang tidak relevan dengan nilai nol**]
sehingga perkalian 
setiap prediksi yang tidak relevan
dengan nol menjadi nol.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(Seq2Seq)
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=False):
    Y_hat = state.apply_fn({'params': params}, *X,
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    l = fn(Y_hat, Y)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask), {}
```

## [**Pelatihan**]
:label:`sec_seq2seq_training`

Sekarang kita dapat [**membuat dan melatih model encoder-decoder RNN**]
untuk pembelajaran sequence-to-sequence pada dataset terjemahan mesin.


```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128) 
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005)
if tab.selected('jax'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**Prediksi**]

Untuk memprediksi urutan keluaran pada setiap langkah,
token yang diprediksi dari langkah sebelumnya
diberikan ke decoder sebagai masukan.
Salah satu strategi sederhana adalah memilih token
yang diberikan probabilitas tertinggi oleh decoder
ketika memprediksi pada setiap langkah.
Seperti dalam pelatihan, pada langkah awal
token awal urutan ("&lt;bos&gt;") diberikan ke decoder.
Proses prediksi ini diilustrasikan pada :numref:`fig_seq2seq_predict`.
Ketika token akhir urutan ("&lt;eos&gt;") diprediksi,
prediksi dari urutan keluaran selesai.

![Memprediksi urutan keluaran token demi token menggunakan RNN encoder-decoder.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

Pada bagian berikutnya, kami akan memperkenalkan
strategi yang lebih canggih
berdasarkan beam search (:numref:`sec_beam-search`).


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
    if tab.selected('mxnet', 'pytorch'):
        batch = [d2l.to(a, device) for a in batch]
    src, tgt, src_valid_len, _ = batch
    if tab.selected('mxnet', 'pytorch'):
        enc_all_outputs = self.encoder(src, src_valid_len)
    if tab.selected('tensorflow'):
        enc_all_outputs = self.encoder(src, src_valid_len, training=False)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
    for _ in range(num_steps):
        if tab.selected('mxnet', 'pytorch'):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
        if tab.selected('tensorflow'):
            Y, dec_state = self.decoder(outputs[-1], dec_state, training=False)
        outputs.append(d2l.argmax(Y, 2))
        # Simpan bobot attention (akan dibahas nanti)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    return d2l.concat(outputs[1:], 1), attention_weights
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, params, batch, num_steps,
                 save_attention_weights=False):
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs, inter_enc_vars = self.encoder.apply(
        {'params': params['encoder']}, src, src_valid_len, training=False,
        mutable='intermediates')
    # Simpan bobot attention encoder jika inter_enc_vars yang mengandung
    # bobot attention encoder tidak kosong. (akan dibahas nanti)
    enc_attention_weights = []
    if bool(inter_enc_vars) and save_attention_weights:
        # Bobot Attention Encoder yang disimpan dalam koleksi intermediates
        enc_attention_weights = inter_enc_vars[
            'intermediates']['enc_attention_weights'][0]

    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:,0], 1), ], []
    for _ in range(num_steps):
        (Y, dec_state), inter_dec_vars = self.decoder.apply(
            {'params': params['decoder']}, outputs[-1], dec_state,
            training=False, mutable='intermediates')
        outputs.append(d2l.argmax(Y, 2))
        # Simpan bobot attention (akan dibahas nanti)
        if save_attention_weights:
            # Bobot Attention Decoder yang disimpan dalam koleksi intermediates
            dec_attention_weights = inter_dec_vars[
                'intermediates']['dec_attention_weights'][0]
            attention_weights.append(dec_attention_weights)
    return d2l.concat(outputs[1:], 1), (attention_weights,
                                        enc_attention_weights)
```

## Evaluasi Urutan yang Diprediksi

Kita dapat mengevaluasi urutan yang diprediksi dengan membandingkannya dengan urutan target (ground truth). Tetapi apa ukuran yang tepat untuk membandingkan kesamaan antara dua urutan?

*Bilingual Evaluation Understudy* (BLEU), yang awalnya diusulkan untuk mengevaluasi hasil terjemahan mesin :cite:`Papineni.Roukos.Ward.ea.2002`, telah digunakan secara luas untuk mengukur kualitas urutan keluaran untuk berbagai aplikasi. Pada prinsipnya, untuk setiap $n$-gram (:numref:`subsec_markov-models-and-n-grams`) dalam urutan yang diprediksi, BLEU mengevaluasi apakah $n$-gram ini muncul dalam urutan target.

Misalkan $p_n$ adalah presisi dari sebuah $n$-gram, yang didefinisikan sebagai rasio antara jumlah $n$-gram yang cocok dalam urutan yang diprediksi dan urutan target terhadap jumlah $n$-gram dalam urutan yang diprediksi. Sebagai contoh, diberikan urutan target $A$, $B$, $C$, $D$, $E$, $F$, dan urutan yang diprediksi $A$, $B$, $B$, $C$, $D$, kita memiliki $p_1 = 4/5$, $p_2 = 3/4$, $p_3 = 1/3$, dan $p_4 = 0$. Misalkan $\textrm{len}_{\textrm{label}}$ dan $\textrm{len}_{\textrm{pred}}$ adalah jumlah token dalam urutan target dan urutan yang diprediksi, masing-masing. Maka, BLEU didefinisikan sebagai

$$ \exp\left(\min\left(0, 1 - \frac{\textrm{len}_{\textrm{label}}}{\textrm{len}_{\textrm{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

dengan $k$ adalah $n$-gram terpanjang yang cocok.

Berdasarkan definisi BLEU pada :eqref:`eq_bleu`, ketika urutan yang diprediksi sama dengan urutan target, maka nilai BLEU adalah 1. Selain itu, karena mencocokkan $n$-gram yang lebih panjang lebih sulit, BLEU memberikan bobot lebih besar jika $n$-gram yang lebih panjang memiliki presisi yang tinggi. Secara khusus, ketika $p_n$ tetap, $p_n^{1/2^n}$ meningkat seiring dengan bertambahnya nilai $n$ (makalah asli menggunakan $p_n^{1/n}$). Selain itu, karena memprediksi urutan yang lebih pendek cenderung memberikan nilai $p_n$ yang lebih tinggi, koefisien sebelum faktor perkalian pada :eqref:`eq_bleu` memberikan penalti untuk urutan yang diprediksi lebih pendek. Sebagai contoh, ketika $k=2$, diberikan urutan target $A$, $B$, $C$, $D$, $E$, $F$ dan urutan yang diprediksi $A$, $B$, meskipun $p_1 = p_2 = 1$, faktor penalti $\exp(1-6/2) \approx 0.14$ akan menurunkan nilai BLEU.

Kita [**mengimplementasikan ukuran BLEU**] sebagai berikut.

```{.python .input}
%%tab all
```python
def bleu(pred_seq, label_seq, k):  #@save
    """Menghitung BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

Pada akhirnya,
kita menggunakan RNN encoder-decoder yang telah dilatih
untuk [**menerjemahkan beberapa kalimat bahasa Inggris ke dalam bahasa Prancis**]
dan menghitung skor BLEU dari hasilnya.


```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(trainer.state.params, data.build(engs, fras),
                                  data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)        
    print(f'{en} => {translation}, bleu,'
          f'{bleu(" ".join(translation), fr, k=2):.3f}')
```

## Ringkasan

Mengikuti desain arsitektur encoder-decoder, kita dapat menggunakan dua RNN untuk mendesain model pembelajaran sequence-to-sequence.
Dalam pelatihan encoder-decoder, pendekatan teacher forcing memasukkan urutan keluaran asli (berbeda dengan prediksi) ke dalam decoder.
Ketika mengimplementasikan encoder dan decoder, kita dapat menggunakan multilayer RNN.
Kita dapat menggunakan mask untuk menyaring perhitungan yang tidak relevan, seperti ketika menghitung loss.
Untuk mengevaluasi urutan keluaran,
BLEU adalah ukuran populer yang mencocokkan $n$-gram antara urutan yang diprediksi dan urutan target.


## Latihan

1. Dapatkah Anda menyesuaikan hyperparameter untuk meningkatkan hasil terjemahan?
2. Jalankan kembali eksperimen tanpa menggunakan mask dalam perhitungan loss. Hasil apa yang Anda amati? Mengapa?
3. Jika encoder dan decoder berbeda dalam jumlah lapisan atau jumlah unit tersembunyi, bagaimana kita dapat menginisialisasi status tersembunyi dari decoder?
4. Dalam pelatihan, gantikan teacher forcing dengan memasukkan prediksi pada langkah waktu sebelumnya ke dalam decoder. Bagaimana hal ini mempengaruhi performa?
5. Jalankan kembali eksperimen dengan mengganti GRU dengan LSTM.
6. Apakah ada cara lain untuk mendesain lapisan keluaran dari decoder?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1062)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3865)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18022)
:end_tab:
