```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Inisialisasi Tertunda (Lazy Initialization)
:label:`sec_lazy_init`

Sejauh ini, mungkin tampak bahwa kita dapat lolos
dengan sedikit "ceroboh" dalam menyusun jaringan kita.
Secara spesifik, kita telah melakukan beberapa hal yang tidak intuitif,
yang mungkin tampak seolah-olah tidak seharusnya berhasil:

* Kita mendefinisikan arsitektur jaringan
  tanpa menentukan dimensi input.
* Kita menambahkan lapisan tanpa menentukan
  dimensi output dari lapisan sebelumnya.
* Kita bahkan "menginisialisasi" parameter-parameter ini
  sebelum memberikan informasi yang cukup untuk menentukan
  berapa banyak parameter yang harus dimiliki oleh model kita.

Anda mungkin terkejut bahwa kode kita dapat berjalan sama sekali.
Bagaimanapun, tidak ada cara bagi framework deep learning
untuk mengetahui dimensi input dari sebuah jaringan.
Triknya di sini adalah bahwa framework *menunda inisialisasi*,
menunggu sampai pertama kali kita melewatkan data melalui model,
untuk menyimpulkan ukuran setiap lapisan secara langsung.

Di masa mendatang, saat bekerja dengan jaringan saraf konvolusional,
teknik ini akan menjadi lebih berguna
karena dimensi input
(misalnya, resolusi gambar)
akan mempengaruhi dimensi
setiap lapisan berikutnya.
Kemampuan untuk menetapkan parameter
tanpa perlu mengetahui,
pada saat menulis kode,
nilai dimensi tersebut
dapat sangat menyederhanakan tugas dalam menentukan
dan kemudian memodifikasi model kita.
Selanjutnya, kita akan membahas lebih dalam tentang mekanisme inisialisasi.


```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

Untuk memulai, mari kita buat sebuah MLP.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

Pada titik ini, jaringan tidak mungkin mengetahui
dimensi dari bobot lapisan input
karena dimensi input masih belum diketahui.

:begin_tab:`mxnet, pytorch, tensorflow`
Akibatnya, framework belum menginisialisasi parameter apa pun.
Kita dapat mengonfirmasinya dengan mencoba mengakses parameter di bawah ini.
:end_tab:

:begin_tab:`jax`
Seperti disebutkan dalam :numref:`subsec_param-access`, parameter dan definisi jaringan dipisahkan
di Jax dan Flax, dan pengguna mengelola keduanya secara manual. Model Flax bersifat stateless
sehingga tidak ada atribut `parameters`.
:end_tab:


```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Perhatikan bahwa meskipun objek parameter sudah ada,
dimensi input untuk setiap lapisan tercantum sebagai -1.
MXNet menggunakan nilai khusus -1 untuk menunjukkan
bahwa dimensi parameter masih belum diketahui.
Pada titik ini, upaya untuk mengakses `net[0].weight.data()`
akan memicu kesalahan runtime yang menyatakan bahwa jaringan
harus diinisialisasi sebelum parameter dapat diakses.
Sekarang mari kita lihat apa yang terjadi ketika kita mencoba
menginisialisasi parameter melalui metode `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Perhatikan bahwa setiap objek lapisan sudah ada, tetapi bobotnya masih kosong.
Menggunakan `net.get_weights()` akan memunculkan error karena bobot
belum diinisialisasi.
:end_tab:


```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Seperti yang kita lihat, tidak ada yang berubah.
Ketika dimensi input tidak diketahui,
pemanggilan `initialize` tidak benar-benar menginisialisasi parameter.
Sebaliknya, pemanggilan ini mendaftarkan ke MXNet bahwa kita ingin
(dan opsional, sesuai distribusi tertentu)
menginisialisasi parameter.
:end_tab:

Selanjutnya, mari kita masukkan data ke dalam jaringan
agar framework akhirnya menginisialisasi parameter.


```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

Begitu kita mengetahui dimensi input,
yaitu 20,
framework dapat mengidentifikasi bentuk matriks bobot lapisan pertama dengan memasukkan nilai 20.
Setelah mengenali bentuk lapisan pertama, framework melanjutkan
ke lapisan kedua,
dan seterusnya melalui grafik komputasi
hingga semua bentuk diketahui.
Perhatikan bahwa dalam kasus ini,
hanya lapisan pertama yang memerlukan inisialisasi tertunda (lazy initialization),
tetapi framework tetap menginisialisasi secara berurutan.
Setelah semua bentuk parameter diketahui,
framework akhirnya dapat menginisialisasi parameter.

:begin_tab:`pytorch`
Metode berikut
memasukkan input dummy
melalui jaringan
untuk simulasi awal
untuk menyimpulkan semua bentuk parameter
dan kemudian menginisialisasi parameter-parameter tersebut.
Metode ini akan digunakan nanti saat inisialisasi acak default tidak diinginkan.
:end_tab:

:begin_tab:`jax`
Inisialisasi parameter dalam Flax selalu dilakukan secara manual dan ditangani oleh
pengguna. Metode berikut menerima input dummy dan sebuah dictionary kunci sebagai argumen.
Dictionary kunci ini memiliki rng untuk menginisialisasi parameter model
dan rng dropout untuk menghasilkan masker dropout bagi model dengan
lapisan dropout. Lebih lanjut tentang dropout akan dibahas di :numref:`sec_dropout`.
Pada akhirnya, metode ini menginisialisasi model dan mengembalikan parameter-parameter.
Kita telah menggunakannya di bagian sebelumnya secara implisit.
:end_tab:


```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## Ringkasan

Inisialisasi tertunda (lazy initialization) dapat menjadi praktis, memungkinkan framework untuk menyimpulkan bentuk parameter secara otomatis, mempermudah modifikasi arsitektur, dan menghilangkan salah satu sumber kesalahan umum.
Kita dapat memasukkan data melalui model agar framework akhirnya menginisialisasi parameter.

## Latihan

1. Apa yang terjadi jika Anda menentukan dimensi input untuk lapisan pertama tetapi tidak untuk lapisan-lapisan berikutnya? Apakah Anda mendapatkan inisialisasi langsung?
2. Apa yang terjadi jika Anda menentukan dimensi yang tidak cocok?
3. Apa yang perlu Anda lakukan jika memiliki input dengan dimensi yang bervariasi? Petunjuk: lihat pada pengikatan parameter (parameter tying).

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17992)
:end_tab:
