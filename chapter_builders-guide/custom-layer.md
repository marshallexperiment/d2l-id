```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lapisan Kustom

Salah satu faktor di balik keberhasilan deep learning
adalah tersedianya berbagai macam lapisan
yang dapat dikomposisikan secara kreatif
untuk merancang arsitektur yang sesuai
untuk berbagai macam tugas.
Misalnya, para peneliti telah menemukan lapisan-lapisan
khusus untuk menangani gambar, teks,
mengulang data berurutan,
dan
melakukan pemrograman dinamis.
Cepat atau lambat, Anda akan membutuhkan
lapisan yang belum ada di framework deep learning.
Dalam kasus ini, Anda harus membuat lapisan kustom.
Pada bagian ini, kami akan menunjukkan caranya.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
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
import jax
from jax import numpy as jnp
```

## (**Lapisan tanpa Parameter**)

Untuk memulai, kita akan membuat lapisan kustom
yang tidak memiliki parameter sendiri.
Ini mungkin terlihat familiar jika Anda mengingat
pengantar modul pada :numref:`sec_model_construction`.
Kelas `CenteredLayer` berikut ini
hanya mengurangkan nilai rata-rata dari input-nya.
Untuk membangunnya, kita hanya perlu mewarisi
kelas dasar layer dan mengimplementasikan fungsi propagasi maju.


```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

Mari kita verifikasi bahwa lapisan kita berfungsi sebagaimana mestinya dengan memasukkan beberapa data ke dalamnya.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Kita sekarang dapat [**menggabungkan lapisan kita sebagai komponen
dalam membangun model yang lebih kompleks.**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

Sebagai pemeriksaan tambahan, kita dapat mengirim data acak
melalui jaringan dan memeriksa apakah nilai rata-ratanya benar-benar 0.
Karena kita berurusan dengan angka floating point,
mungkin masih ada angka kecil yang tidak nol
karena proses kuantisasi.

:begin_tab:`jax`
Di sini kita menggunakan metode `init_with_output` yang mengembalikan output dari
jaringan serta parameter-parameter. Dalam kasus ini kita hanya fokus pada
output.
:end_tab:


```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**Lapisan dengan Parameter**]

Sekarang setelah kita tahu cara mendefinisikan lapisan sederhana,
mari kita lanjutkan untuk mendefinisikan lapisan dengan parameter
yang dapat disesuaikan melalui pelatihan.
Kita dapat menggunakan fungsi bawaan untuk membuat parameter, yang
menyediakan beberapa fungsi dasar manajemen parameter.
Secara khusus, fungsi ini mengatur akses, inisialisasi,
pembagian, penyimpanan, dan pemuatan parameter model.
Dengan cara ini, di antara manfaat lainnya, kita tidak perlu menulis
rutin serialisasi khusus untuk setiap lapisan kustom.

Sekarang mari kita implementasikan versi kita sendiri dari lapisan fully connected.
Ingat bahwa lapisan ini membutuhkan dua parameter,
satu untuk merepresentasikan bobot dan satu lagi untuk bias.
Dalam implementasi ini, kita memasukkan aktivasi ReLU sebagai default.
Lapisan ini membutuhkan dua argumen input: `in_units` dan `units`, yang
masing-masing menunjukkan jumlah input dan output.


```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
Selanjutnya, kita membuat instance dari kelas `MyDense`
dan mengakses parameter modelnya.
:end_tab:

:begin_tab:`pytorch`
Selanjutnya, kita membuat instance dari kelas `MyLinear`
dan mengakses parameter modelnya.
:end_tab:


```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

Kita dapat [**langsung melakukan perhitungan propagasi maju menggunakan lapisan kustom.**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

Kita juga dapat (**membangun model menggunakan lapisan kustom.**)
Setelah itu, kita dapat menggunakannya seperti lapisan fully connected bawaan.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## Ringkasan

Kita dapat merancang lapisan kustom melalui kelas dasar lapisan. Ini memungkinkan kita untuk mendefinisikan lapisan baru yang fleksibel dan berperilaku berbeda dari lapisan apa pun yang ada di dalam pustaka.
Setelah didefinisikan, lapisan kustom dapat dipanggil dalam konteks dan arsitektur apa pun.
Lapisan dapat memiliki parameter lokal, yang dapat dibuat melalui fungsi bawaan.

## Latihan

1. Rancang sebuah lapisan yang menerima input dan menghitung reduksi tensor,
   yaitu, lapisan tersebut mengembalikan $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
2. Rancang sebuah lapisan yang mengembalikan setengah koefisien Fourier terdepan dari data.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17993)
:end_tab:
