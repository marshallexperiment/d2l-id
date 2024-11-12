```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Arsitektur Encoder--Decoder
:label:`sec_encoder-decoder`

Dalam masalah *sequence-to-sequence* secara umum seperti terjemahan mesin
(:numref:`sec_machine_translation`),
input dan output memiliki panjang yang bervariasi
dan tidak selaras.
Pendekatan standar untuk menangani jenis data seperti ini
adalah dengan merancang arsitektur *encoder--decoder* (:numref:`fig_encoder_decoder`)
yang terdiri dari dua komponen utama:
sebuah *encoder* yang menerima urutan dengan panjang variabel sebagai input,
dan sebuah *decoder* yang bertindak sebagai model bahasa kondisional,
mengambil input yang sudah diencode
dan konteks kiri dari urutan target
dan memprediksi token berikutnya dalam urutan target.

![Arsitektur encoder--decoder.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Mari kita ambil contoh terjemahan mesin dari bahasa Inggris ke bahasa Prancis.
Diberikan sebuah urutan input dalam bahasa Inggris:
"They", "are", "watching", ".",
arsitektur encoder--decoder ini
pertama-tama mengenkode input dengan panjang variabel ke dalam sebuah state,
kemudian mendekode state tersebut
untuk menghasilkan urutan terjemahan,
token demi token, sebagai output:
"Ils", "regardent", ".".
Karena arsitektur encoder--decoder
membentuk dasar dari berbagai model *sequence-to-sequence*
pada bagian-bagian selanjutnya,
bagian ini akan mengonversi arsitektur ini
menjadi antarmuka yang akan diimplementasikan nanti.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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
```

## (**Encoder**)

Pada antarmuka encoder, kita hanya menentukan bahwa encoder menerima urutan dengan panjang variabel sebagai input `X`. Implementasi akan disediakan oleh model apa pun yang mewarisi kelas dasar `Encoder` ini.


```{.python .input}
%%tab mxnet
class Encoder(nn.Block):  #@save
    """Antarmuka dasar encoder untuk arsitektur encoder--decoder."""
    def __init__(self):
        super().__init__()

    # Nanti bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Encoder(nn.Module):  #@save
    """Antarmuka dasar encoder untuk arsitektur encoder--decoder."""
    def __init__(self):
        super().__init__()

    # Nanti bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Encoder(tf.keras.layers.Layer):  #@save
    """Antarmuka dasar encoder untuk arsitektur encoder--decoder."""
    def __init__(self):
        super().__init__()

    # Nanti bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def call(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Encoder(nn.Module):  #@save
    """Antarmuka dasar encoder untuk arsitektur encoder--decoder."""
    def setup(self):
        raise NotImplementedError

    # Nanti bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## [**Decoder**]

Dalam antarmuka decoder berikut,
kami menambahkan metode tambahan `init_state`
untuk mengonversi keluaran dari encoder (`enc_all_outputs`)
menjadi keadaan yang sudah dienkode.
Perhatikan bahwa langkah ini
mungkin memerlukan input tambahan,
seperti panjang yang valid dari input,
yang sudah dijelaskan
di :numref:`sec_machine_translation`.
Untuk menghasilkan urutan dengan panjang variabel token demi token,
setiap kali decoder dapat memetakan sebuah input
(misalnya, token yang dihasilkan pada langkah waktu sebelumnya)
dan keadaan yang sudah dienkode
menjadi token keluaran pada langkah waktu saat ini.


```{.python .input}
%%tab mxnet
class Decoder(nn.Block):  #@save
    """Antarmuka dasar decoder untuk arsitektur encoder--decoder."""
    def __init__(self):
        super().__init__()

    # Nantinya bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Decoder(nn.Module):  #@save
    """Antarmuka dasar decoder untuk arsitektur encoder--decoder."""
    def __init__(self):
        super().__init__()

    # Nantinya bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Decoder(tf.keras.layers.Layer):  #@save
    """Antarmuka dasar decoder untuk arsitektur encoder--decoder."""
    def __init__(self):
        super().__init__()

    # Nantinya bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Decoder(nn.Module):  #@save
    """Antarmuka dasar decoder untuk arsitektur encoder--decoder."""
    def setup(self):
        raise NotImplementedError

    # Nantinya bisa ada argumen tambahan (misalnya, panjang tanpa padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## [**Menggabungkan Encoder dan Decoder**]

Dalam propagasi maju, keluaran dari encoder digunakan untuk menghasilkan state yang telah di-encode, dan state ini akan digunakan lebih lanjut oleh decoder sebagai salah satu inputnya.


```{.python .input}
%%tab mxnet, pytorch
class EncoderDecoder(d2l.Classifier):  #@save
    """dasar kelas untuk arsitektur encoder--decoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Mengembalikan output decoder saja
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
class EncoderDecoder(d2l.Classifier):  #@save
    """dasar kelas untuk arsitektur encoder--decoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Mengembalikan output decoder saja
        return self.decoder(dec_X, dec_state, training=True)[0]
```

```{.python .input}
%%tab jax
class EncoderDecoder(d2l.Classifier):  #@save
    """dasar kelas untuk arsitektur encoder--decoder."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Mengembalikan output decoder saja
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

## Ringkasan

Arsitektur encoder-decoder dapat menangani input dan output yang keduanya terdiri dari urutan dengan panjang yang bervariasi, sehingga cocok untuk masalah sequence-to-sequence seperti penerjemahan mesin. Encoder mengambil urutan dengan panjang bervariasi sebagai input dan mengubahnya menjadi suatu state dengan bentuk yang tetap. Decoder kemudian memetakan state yang telah dikodekan (encoded) dengan bentuk tetap ini menjadi urutan dengan panjang yang bervariasi.

## Latihan

1. Misalkan kita menggunakan jaringan saraf (neural network) untuk mengimplementasikan arsitektur encoder-decoder. Apakah encoder dan decoder harus menggunakan jenis jaringan saraf yang sama?
2. Selain penerjemahan mesin, apakah kamu dapat memikirkan aplikasi lain di mana arsitektur encoder-decoder dapat diterapkan?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3864)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18021)
:end_tab:

