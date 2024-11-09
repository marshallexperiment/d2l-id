```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# File I/O

Sejauh ini kita telah membahas bagaimana memproses data dan bagaimana
membangun, melatih, serta menguji model deep learning.
Namun, pada titik tertentu kita semoga akan cukup puas
dengan model yang telah dilatih dan ingin
menyimpan hasilnya untuk digunakan nanti dalam berbagai konteks
(bahkan mungkin untuk membuat prediksi dalam proses deployment).
Selain itu, saat menjalankan proses pelatihan yang panjang,
praktik terbaik adalah secara berkala menyimpan hasil antara (checkpointing)
untuk memastikan bahwa kita tidak kehilangan hasil komputasi selama beberapa hari
jika tiba-tiba daya server kita terputus.
Maka, inilah saatnya untuk mempelajari cara memuat dan menyimpan
baik vektor bobot individual maupun keseluruhan model.
Bagian ini membahas kedua hal tersebut.


```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**Memuat dan Menyimpan Tensor**)

Untuk tensor individual, kita dapat langsung
memanggil fungsi `load` dan `save`
untuk membacanya dan menulisnya masing-masing.
Kedua fungsi ini memerlukan kita memberikan sebuah nama,
dan `save` membutuhkan input berupa variabel yang akan disimpan.


```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

Kita sekarang dapat membaca data dari file yang tersimpan kembali ke dalam memori.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

Kita dapat [**menyimpan daftar tensor dan membacanya kembali ke dalam memori.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Kita bahkan dapat [**menulis dan membaca dictionary yang memetakan
dari string ke tensor.**]
Ini sangat berguna ketika kita ingin
membaca atau menulis semua bobot dalam sebuah model.


```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Memuat dan Menyimpan Parameter Model**]

Menyimpan vektor bobot individual (atau tensor lainnya) sangat berguna,
tetapi bisa menjadi sangat membosankan jika kita ingin menyimpan
(dan kemudian memuat) keseluruhan model.
Lagi pula, kita mungkin memiliki ratusan
kelompok parameter yang tersebar di seluruh model.
Untuk alasan ini, framework deep learning menyediakan fungsi bawaan
untuk memuat dan menyimpan keseluruhan jaringan.
Detail penting yang perlu diperhatikan adalah bahwa ini
menyimpan *parameter* model dan bukan keseluruhan model.
Misalnya, jika kita memiliki MLP 3-lapisan,
kita perlu menentukan arsitekturnya secara terpisah.
Alasannya adalah karena model itu sendiri dapat berisi kode sembarang,
sehingga tidak bisa diserialisasi dengan alami.
Jadi, untuk mengembalikan model, kita perlu
membuat arsitektur dalam kode
dan kemudian memuat parameter dari disk.
(**Mari kita mulai dengan MLP yang sudah kita kenal.**)


```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

Selanjutnya, kita [**menyimpan parameter model sebagai sebuah file**] dengan nama "mlp.params".

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

Untuk memulihkan model, kita membuat instans dari
model MLP asli.
Alih-alih menginisialisasi parameter model secara acak,
kita [**membaca parameter yang disimpan dalam file secara langsung**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

Since both instances have the same model parameters,
the computational result of the same input `X` should be the same.
Let's verify this.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Ringkasan

Fungsi `save` dan `load` dapat digunakan untuk melakukan operasi I/O file untuk objek tensor.
Kita dapat menyimpan dan memuat seluruh set parameter untuk jaringan melalui dictionary parameter.
Menyimpan arsitektur harus dilakukan dalam kode, bukan dalam parameter.

## Latihan

1. Meskipun tidak ada kebutuhan untuk melakukan deployment model terlatih ke perangkat lain, apa manfaat praktis dari menyimpan parameter model?
2. Misalkan kita ingin menggunakan kembali hanya sebagian dari sebuah jaringan untuk dimasukkan ke dalam jaringan dengan arsitektur yang berbeda. Bagaimana cara Anda menggunakan, misalnya, dua lapisan pertama dari jaringan sebelumnya dalam jaringan baru?
3. Bagaimana Anda akan menyimpan arsitektur jaringan dan parameternya? Batasan apa yang akan Anda terapkan pada arsitektur tersebut?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17994)
:end_tab:
