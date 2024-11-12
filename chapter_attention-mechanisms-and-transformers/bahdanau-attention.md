```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Mekanisme Attention Bahdanau
:label:`sec_seq2seq_attention`

Ketika kita mempelajari terjemahan mesin pada :numref:`sec_seq2seq`,
kita mendesain arsitektur encoder-decoder untuk pembelajaran sequence-to-sequence
berdasarkan dua RNN :cite:`Sutskever.Vinyals.Le.2014`.
Secara spesifik, RNN encoder mengubah sequence dengan panjang variabel
menjadi variabel konteks *berbentuk tetap*.
Kemudian, RNN decoder menghasilkan sequence output (target) token demi token
berdasarkan token yang telah dihasilkan dan variabel konteks tersebut.

Ingat :numref:`fig_seq2seq_details` yang kita ulangi pada (:numref:`fig_s2s_attention_state`) dengan beberapa detail tambahan. Secara konvensional, pada RNN semua informasi relevan tentang sequence sumber diterjemahkan ke dalam representasi keadaan internal *berdimensi tetap* oleh encoder. Keadaan inilah yang digunakan oleh decoder sebagai sumber informasi yang lengkap dan eksklusif untuk menghasilkan sequence terjemahan. Dengan kata lain, mekanisme sequence-to-sequence menganggap bahwa keadaan intermediate adalah statistik yang cukup untuk menggambarkan string apapun yang mungkin menjadi input.

![Model sequence-to-sequence. State, yang dihasilkan oleh encoder, adalah satu-satunya bagian informasi yang dibagikan antara encoder dan decoder.](../img/seq2seq-state.svg)
:label:`fig_s2s_attention_state`

Meskipun ini cukup masuk akal untuk sequence pendek, jelas bahwa ini tidak layak untuk sequence yang panjang, seperti bab buku atau bahkan hanya kalimat yang sangat panjang. Bagaimanapun, sebelum terlalu lama tidak akan ada cukup "ruang" dalam representasi intermediate untuk menyimpan semua yang penting dalam sequence sumber. Akibatnya, decoder akan gagal menerjemahkan kalimat yang panjang dan kompleks. Salah satu yang pertama menghadapi hal ini adalah :citet:`Graves.2013` yang mencoba mendesain RNN untuk menghasilkan teks tulisan tangan. Karena teks sumber memiliki panjang yang arbitrer, mereka mendesain model attention yang dapat diturunkan (differentiable) untuk menyelaraskan karakter teks dengan jejak pena yang jauh lebih panjang, di mana penyelarasan bergerak hanya dalam satu arah. Hal ini, pada gilirannya, menggunakan algoritma decoding dalam pengenalan suara, misalnya, hidden Markov models :cite:`rabiner1993fundamentals`.

Terinspirasi oleh ide untuk belajar menyelaraskan,
:citet:`Bahdanau.Cho.Bengio.2014` mengusulkan model attention yang dapat diturunkan
*tanpa* batasan penyelarasan satu arah.
Ketika memprediksi token,
jika tidak semua token input relevan,
model menyelaraskan (atau memberikan perhatian)
hanya pada bagian dari sequence input
yang dianggap relevan untuk prediksi saat ini. Ini kemudian digunakan untuk memperbarui keadaan saat ini sebelum menghasilkan token berikutnya. Meskipun terdengar sepele dalam deskripsinya, *mekanisme attention Bahdanau* ini dapat dikatakan menjadi salah satu ide paling berpengaruh dalam dekade terakhir dalam pembelajaran mendalam, memberikan inspirasi untuk Transformers :cite:`Vaswani.Shazeer.Parmar.ea.2017` dan banyak arsitektur baru terkait lainnya.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import rnn, nn
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
from jax import numpy as jnp
import jax
```

## Model

Kita mengikuti notasi yang diperkenalkan oleh arsitektur sequence-to-sequence pada :numref:`sec_seq2seq`, khususnya :eqref:`eq_seq2seq_s_t`.
Ide utama adalah bahwa daripada menjaga state tetap,
yaitu variabel konteks $\mathbf{c}$ yang merangkum kalimat sumber, sebagai tetap, kita secara dinamis memperbaruinya sebagai fungsi dari teks asli (encoder hidden states $\mathbf{h}_{t}$) dan teks yang sudah dihasilkan (decoder hidden states $\mathbf{s}_{t'-1}$). Ini menghasilkan $\mathbf{c}_{t'}$, yang diperbarui setelah setiap langkah waktu decoding $t'$. Misalkan sequence input memiliki panjang $T$. Dalam hal ini, variabel konteks adalah output dari attention pooling:

$$\mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}.$$

Kita menggunakan $\mathbf{s}_{t' - 1}$ sebagai query, dan
$\mathbf{h}_{t}$ sebagai key dan value. Perhatikan bahwa $\mathbf{c}_{t'}$ kemudian digunakan untuk menghasilkan state $\mathbf{s}_{t'}$ dan untuk menghasilkan token baru: lihat :eqref:`eq_seq2seq_s_t`. Secara khusus, bobot attention $\alpha$ dihitung seperti dalam :eqref:`eq_attn-scoring-alpha`
menggunakan fungsi skor attention aditif yang didefinisikan oleh :eqref:`eq_additive-attn`.
Arsitektur RNN encoder--decoder
yang menggunakan mekanisme attention ini digambarkan pada :numref:`fig_s2s_attention_details`. Perhatikan bahwa model ini kemudian dimodifikasi sehingga mencakup token yang sudah dihasilkan dalam decoder sebagai konteks lebih lanjut (yaitu, penjumlahan attention tidak berhenti pada $T$, tetapi berlanjut hingga $t'-1$). Sebagai contoh, lihat :citet:`chan2015listen` untuk deskripsi strategi ini, yang diterapkan pada pengenalan suara.

![Lapisan pada model RNN encoder--decoder dengan mekanisme attention Bahdanau.](../img/seq2seq-details-attention.svg)
:label:`fig_s2s_attention_details`

## Mendefinisikan Decoder dengan Attention

Untuk mengimplementasikan RNN encoder--decoder dengan attention,
kita hanya perlu mendefinisikan ulang decoder (mengabaikan simbol yang dihasilkan dari fungsi attention untuk menyederhanakan desain). Mari kita mulai dengan [**interface dasar untuk decoder dengan attention**] dengan mendefinisikan kelas `AttentionDecoder` yang, tidak mengejutkan, dinamai demikian.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class AttentionDecoder(d2l.Decoder):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

Kita perlu [**mengimplementasikan RNN decoder**]
dalam kelas `Seq2SeqAttentionDecoder`.
State dari decoder diinisialisasi dengan
(i) hidden state dari layer terakhir encoder pada setiap langkah waktu, yang digunakan sebagai key dan value untuk mekanisme attention;
(ii) hidden state dari encoder di semua layer pada langkah waktu terakhir, yang digunakan untuk menginisialisasi hidden state dari decoder;
dan (iii) panjang yang valid dari encoder, untuk mengecualikan token padding dalam attention pooling.
Pada setiap langkah waktu decoding, hidden state dari layer terakhir decoder, yang diperoleh pada langkah waktu sebelumnya, digunakan sebagai query dalam mekanisme attention.
Baik output dari mekanisme attention maupun input embedding digabungkan untuk berfungsi sebagai input dari RNN decoder.


```{.python .input}
%%tab mxnet
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())

    def init_state(self, enc_outputs, enc_valid_lens):
        # Bentuk outputs: (num_steps, batch_size, num_hiddens).
        # Bentuk hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Bentuk enc_outputs: (batch_size, num_steps, num_hiddens).
        # Bentuk hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Bentuk output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Bentuk query: (batch_size, 1, num_hiddens)
            query = np.expand_dims(hidden_state[-1], axis=1)
            # Bentuk context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Menggabungkan pada dimensi fitur
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Mengubah bentuk x menjadi (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            hidden_state = hidden_state[0]
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # Setelah transformasi layer fully connected, bentuk outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Bentuk outputs: (num_steps, batch_size, num_hiddens).
        # Bentuk hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Bentuk enc_outputs: (batch_size, num_steps, num_hiddens).
        # Bentuk hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Bentuk output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Bentuk query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Bentuk context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Menggabungkan pada dimensi fitur
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Mengubah bentuk x menjadi (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # Setelah transformasi layer fully connected, bentuk outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Bentuk dari outputs: (batch_size, num_steps, num_hiddens).
        # Panjang dari daftar hidden_state adalah num_layers, dengan bentuk
        # elemen berupa (batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (tf.transpose(outputs, (1, 0, 2)), hidden_state,
                enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Bentuk dari output enc_outputs: (batch_size, num_steps, num_hiddens)
        # Panjang dari daftar hidden_state adalah num_layers, dengan bentuk
        # elemen berupa (batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Bentuk dari output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X)  # Input X memiliki bentuk: (batch_size, num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Bentuk dari query: (batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Bentuk dari context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Menggabungkan pada dimensi fitur
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # Setelah transformasi layer fully connected, bentuk outputs:
        # (batch_size, num_steps, vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class Seq2SeqAttentionDecoder(nn.Module):
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.attention = d2l.AdditiveAttention(self.num_hiddens, self.dropout)
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.dense = nn.Dense(self.vocab_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout=self.dropout)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Bentuk dari outputs: (num_steps, batch_size, num_hiddens).
        # Bentuk dari hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        # Attention Weights dikembalikan sebagai bagian dari state; inisialisasi dengan None
        return (outputs.transpose(1, 0, 2), hidden_state, enc_valid_lens)

    @nn.compact
    def __call__(self, X, state, training=False):
        # Bentuk dari enc_outputs: (batch_size, num_steps, num_hiddens).
        # Bentuk dari hidden_state: (num_layers, batch_size, num_hiddens)
        # Abaikan nilai Attention dalam state
        enc_outputs, hidden_state, enc_valid_lens = state
        # Bentuk dari output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).transpose(1, 0, 2)
        outputs, attention_weights = [], []
        for x in X:
            # Bentuk dari query: (batch_size, 1, num_hiddens)
            query = jnp.expand_dims(hidden_state[-1], axis=1)
            # Bentuk dari context: (batch_size, 1, num_hiddens)
            context, attention_w = self.attention(query, enc_outputs,
                                                  enc_outputs, enc_valid_lens,
                                                  training=training)
            # Menggabungkan pada dimensi fitur
            x = jnp.concatenate((context, jnp.expand_dims(x, axis=1)), axis=-1)
            # Bentuk ulang x sebagai (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.transpose(1, 0, 2), hidden_state,
                                         training=training)
            outputs.append(out)
            attention_weights.append(attention_w)

        # Flax sow API digunakan untuk menangkap variabel antara
        self.sow('intermediates', 'dec_attention_weights', attention_weights)

        # Setelah transformasi layer fully connected, bentuk dari outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(jnp.concatenate(outputs, axis=0))
        return outputs.transpose(1, 0, 2), [enc_outputs, hidden_state,
                                            enc_valid_lens]
```

Dalam contoh berikut, kita [**menguji decoder yang diimplementasikan dengan mekanisme perhatian (attention)**] menggunakan satu minibatch yang terdiri dari empat urutan, masing-masing sepanjang tujuh langkah waktu.



```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
if tab.selected('mxnet'):
    X = d2l.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('pytorch'):
    X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('tensorflow'):
    X = tf.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X, training=False), None)
    output, state = decoder(X, state, training=False)
if tab.selected('jax'):
    X = jnp.zeros((batch_size, num_steps), dtype=jnp.int32)
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(),
                                                        X, training=False)[0],
                               None)
    (output, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                  state, training=False)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

## [**Pelatihan (Training)**]

Sekarang setelah kita menentukan decoder yang baru, kita dapat melanjutkan dengan cara yang mirip dengan :numref:`sec_seq2seq_training`:
menentukan hiperparameter, membuat instansi
encoder reguler dan decoder dengan mekanisme perhatian (attention),
serta melatih model ini untuk penerjemahan mesin.


```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = d2l.Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = d2l.Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

Setelah model dilatih,
kita menggunakannya untuk [**menerjemahkan beberapa kalimat bahasa Inggris**]
ke dalam bahasa Prancis dan menghitung skor BLEU mereka.


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

Mari kita [**visualisasikan bobot perhatian**]
saat menerjemahkan kalimat terakhir dalam bahasa Inggris.
Kita melihat bahwa setiap query memberikan bobot yang tidak uniform
pada pasangan key--value.
Ini menunjukkan bahwa pada setiap langkah decoding,
bagian yang berbeda dari urutan input
secara selektif digabungkan dalam perhatian pooling.


```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
if tab.selected('jax'):
    _, (dec_attention_weights, _) = model.predict_step(
        trainer.state.params, data.build([engs[-1]], [fras[-1]]),
        data.num_steps, True)
attention_weights = d2l.concat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = d2l.reshape(attention_weights, (1, 1, -1, data.num_steps))
```

```{.python .input}
%%tab mxnet
# Tambahkan satu untuk menyertakan token akhir-sekuens
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab pytorch
# Tambahkan satu untuk menyertakan token akhir-sekuens
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab tensorflow
# Tambahkan satu untuk menyertakan token akhir-sekuens
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab jax
# Tambahkan satu untuk menyertakan token akhir-sekuens
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

## Ringkasan

Saat memprediksi token, jika tidak semua token input relevan, RNN encoder-decoder dengan mekanisme perhatian Bahdanau secara selektif mengagregasi berbagai bagian dari urutan input. Ini dicapai dengan memperlakukan state (variabel konteks) sebagai keluaran dari pooling perhatian aditif. 
Pada RNN encoder-decoder, mekanisme perhatian Bahdanau memperlakukan hidden state decoder pada langkah waktu sebelumnya sebagai query, dan hidden states encoder pada semua langkah waktu sebagai key dan value.

## Latihan

1. Gantikan GRU dengan LSTM pada eksperimen ini.
2. Modifikasi eksperimen untuk menggantikan fungsi scoring perhatian aditif dengan dot-product berskala. Bagaimana pengaruhnya terhadap efisiensi pelatihan?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1065)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3868)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18028)
:end_tab:
