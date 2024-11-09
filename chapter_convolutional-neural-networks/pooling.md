```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Pooling
:label:`sec_pooling`

Dalam banyak kasus, tugas akhir kita berkaitan dengan pertanyaan global tentang gambar,
misalnya, *apakah gambar ini berisi seekor kucing?* Akibatnya, unit pada lapisan akhir 
harus peka terhadap seluruh input.
Dengan secara bertahap mengagregasi informasi, menghasilkan representasi peta yang semakin kasar,
kita mencapai tujuan untuk akhirnya mempelajari representasi global,
sambil mempertahankan semua keuntungan dari lapisan konvolusi pada lapisan pemrosesan menengah.
Semakin dalam jaringan, semakin besar receptive field (relatif terhadap input)
yang dapat dideteksi oleh setiap node tersembunyi. Mengurangi resolusi spasial 
mempercepat proses ini, 
karena kernel konvolusi mencakup area efektif yang lebih besar.

Selain itu, ketika mendeteksi fitur level rendah, seperti tepi
(seperti yang dibahas di :numref:`sec_conv_layer`),
kita sering kali menginginkan representasi kita agak invarian terhadap translasi.
Misalnya, jika kita mengambil gambar `X`
dengan batas tajam antara hitam dan putih
dan menggeser seluruh gambar satu piksel ke kanan,
yaitu, `Z[i, j] = X[i, j + 1]`,
maka output untuk gambar baru `Z` bisa sangat berbeda.
Tepi akan bergeser satu piksel.
Pada kenyataannya, objek hampir tidak pernah muncul tepat di tempat yang sama.
Bahkan dengan tripod dan objek yang tidak bergerak,
getaran kamera akibat gerakan shutter
dapat menggeser semuanya sekitar satu piksel
(kamera kelas atas dilengkapi dengan fitur khusus untuk mengatasi masalah ini).

Bagian ini memperkenalkan *lapisan pooling*,
yang berfungsi untuk dua tujuan:
mengurangi sensitivitas lapisan konvolusi terhadap lokasi
dan melakukan downsampling secara spasial pada representasi.

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
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Maximum Pooling dan Average Pooling

Seperti pada lapisan konvolusi, operator *pooling*
terdiri dari jendela berbentuk tetap yang digeser
ke seluruh wilayah pada input sesuai dengan stride-nya,
menghitung satu output untuk setiap lokasi yang dilalui
oleh jendela berbentuk tetap (kadang dikenal sebagai *pooling window*).
Namun, berbeda dengan perhitungan cross-correlation
antara input dan kernel pada lapisan konvolusi,
lapisan pooling tidak memiliki parameter (tidak ada *kernel*).
Sebaliknya, operator pooling bersifat deterministik,
biasanya menghitung nilai maksimum atau rata-rata
dari elemen-elemen dalam pooling window.
Operasi ini disebut *maximum pooling* (disingkat *max-pooling*)
dan *average pooling*, masing-masing.

*Average pooling* sudah ada sejak awal kemunculan CNN. Idenya mirip dengan 
downsampling pada gambar. Daripada hanya mengambil nilai setiap piksel kedua (atau ketiga) 
untuk gambar beresolusi lebih rendah, kita dapat menghitung rata-rata dari piksel yang berdekatan untuk mendapatkan 
gambar dengan rasio sinyal terhadap noise yang lebih baik karena kita menggabungkan informasi 
dari beberapa piksel yang berdekatan. *Max-pooling* diperkenalkan dalam 
:citet:`Riesenhuber.Poggio.1999` dalam konteks ilmu saraf kognitif untuk menggambarkan 
bagaimana agregasi informasi mungkin disusun secara hierarkis untuk tujuan 
pengenalan objek; ada versi sebelumnya dalam pengenalan ucapan :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. Dalam hampir semua kasus, max-pooling, seperti juga disebut, lebih disukai daripada average pooling.

Pada kedua kasus, seperti pada operator cross-correlation,
kita dapat menganggap pooling window
mulai dari kiri atas tensor input
dan menggesernya dari kiri ke kanan dan dari atas ke bawah.
Pada setiap lokasi yang dilalui pooling window,
operator ini menghitung nilai maksimum atau rata-rata
dari subtensor input dalam jendela,
tergantung apakah max atau average pooling digunakan.


![Max-pooling dengan bentuk pooling window $2\times 2$. Bagian yang diarsir adalah elemen output pertama serta elemen tensor input yang digunakan untuk perhitungan output: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

Tensor output pada :numref:`fig_pooling` memiliki tinggi 2 dan lebar 2.
Keempat elemen berasal dari nilai maksimum pada setiap pooling window:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Secara umum, kita dapat mendefinisikan lapisan pooling $p \times q$ dengan mengagregasi pada 
wilayah dengan ukuran tersebut. Kembali ke masalah deteksi tepi, 
kita menggunakan output dari lapisan konvolusi
sebagai input untuk max-pooling $2\times 2$.
Misalkan `X` adalah input dari lapisan konvolusi dan `Y` adalah output dari lapisan pooling. 
Terlepas dari apakah nilai `X[i, j]`, `X[i, j + 1]`, 
`X[i+1, j]`, dan `X[i+1, j + 1]` berbeda atau tidak,
lapisan pooling selalu memberikan output `Y[i, j] = 1`.
Artinya, dengan menggunakan lapisan max-pooling $2\times 2$,
kita masih dapat mendeteksi jika pola yang dikenali oleh lapisan konvolusi
bergerak tidak lebih dari satu elemen dalam tinggi atau lebar.

Pada kode di bawah ini, kita (**mengimplementasikan forward propagation
untuk lapisan pooling**) dalam fungsi `pool2d`.
Fungsi ini mirip dengan fungsi `corr2d`
pada :numref:`sec_conv_layer`.
Namun, tidak diperlukan kernel, output dihitung
sebagai nilai maksimum atau rata-rata dari setiap wilayah pada input.


```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab jax
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = jnp.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].max())
            elif mode == 'avg':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].mean())
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

Kita dapat membangun tensor input `X` seperti pada :numref:`fig_pooling` untuk [**memvalidasi output dari lapisan max-pooling dua dimensi**].

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Selain itu, kita dapat bereksperimen dengan (**lapisan average pooling**).

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**Padding dan Stride**]

Seperti pada lapisan konvolusi, lapisan pooling
mengubah bentuk output.
Dan seperti sebelumnya, kita dapat menyesuaikan operasi ini untuk mencapai bentuk output yang diinginkan
dengan menambahkan padding pada input dan mengatur stride.
Kita dapat mendemonstrasikan penggunaan padding dan stride
pada lapisan pooling melalui lapisan max-pooling dua dimensi bawaan dari framework deep learning.
Pertama, kita membuat tensor input `X` dengan bentuk empat dimensi,
di mana jumlah contoh (ukuran batch) dan jumlah kanal keduanya adalah 1.

:begin_tab:`tensorflow`
Perhatikan bahwa tidak seperti framework lain, TensorFlow
lebih menyukai dan dioptimalkan untuk input dengan format *channels-last*.
:end_tab:


```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow, jax
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Karena pooling mengumpulkan informasi dari suatu area, (**framework deep learning secara default menyamakan ukuran pooling window dan stride.**) Sebagai contoh, 
jika kita menggunakan pooling window dengan bentuk `(3, 3)`, maka secara default kita mendapatkan stride dengan bentuk `(3, 3)`.


```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# Pooling tidak memiliki parameter model, sehingga tidak memerlukan inisialisasi
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# Pooling tidak memiliki parameter model, sehingga tidak memerlukan inisialisasi
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# Pooling tidak memiliki parameter model, sehingga tidak memerlukan inisialisasi
pool2d(X)
```

```{.python .input}
%%tab jax
# Pooling tidak memiliki parameter model, sehingga tidak memerlukan inisialisasi
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

Perlu dicatat, [**stride dan padding dapat ditentukan secara manual**] untuk mengesampingkan pengaturan default framework jika diperlukan.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

Tentu saja, kita dapat menentukan pooling window berbentuk persegi panjang dengan tinggi dan lebar yang berbeda, seperti yang ditunjukkan pada contoh di bawah ini.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
%%tab jax

X_padded = jnp.pad(X, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(2, 3), strides=(2, 3), padding='VALID')
```

## Beberapa Kanal (_Multiple Channels_)

Saat memproses data input dengan beberapa kanal,
[**lapisan pooling akan melakukan pooling pada setiap kanal input secara terpisah**],
bukan menjumlahkan input di seluruh kanal
seperti pada lapisan konvolusi.
Ini berarti bahwa jumlah kanal output pada lapisan pooling
sama dengan jumlah kanal input.
Di bawah ini, kita akan menggabungkan tensor `X` dan `X + 1`
pada dimensi kanal untuk membentuk input dengan dua kanal.

:begin_tab:`tensorflow`
Perhatikan bahwa untuk TensorFlow ini akan membutuhkan
penggabungan pada dimensi terakhir karena sintaks channels-last.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# Gabungkan sepanjang `dim=3` karena sintaks channels-last
X = d2l.concat([X, X + 1], 3)
X
```

Seperti yang dapat kita lihat, jumlah kanal output tetap dua setelah pooling.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

:begin_tab:`tensorflow`
Perhatikan bahwa output untuk pooling pada TensorFlow pada awalnya tampak berbeda,
namun secara numerik menghasilkan hasil yang sama seperti MXNet dan PyTorch.
Perbedaannya terletak pada dimensi, dan membaca
output secara vertikal memberikan hasil yang sama dengan implementasi lainnya.
:end_tab:

## Ringkasan

Pooling adalah operasi yang sangat sederhana. Pooling melakukan persis seperti namanya, yaitu mengagregasi hasil dari jendela nilai. Semua semantik konvolusi, seperti stride dan padding, berlaku dengan cara yang sama seperti sebelumnya. Perlu dicatat bahwa pooling tidak bergantung pada kanal, yaitu, jumlah kanal tidak berubah dan pooling diterapkan pada setiap kanal secara terpisah. Terakhir, dari dua pilihan pooling yang populer, max-pooling lebih disukai daripada average pooling karena memberikan derajat invariansi tertentu terhadap output. Pilihan yang umum adalah menggunakan ukuran pooling window $2 \times 2$ untuk mengurangi resolusi spasial output menjadi seperempatnya.

Perlu dicatat bahwa ada banyak cara lain untuk mengurangi resolusi selain pooling. Misalnya, dalam stochastic pooling :cite:`Zeiler.Fergus.2013` dan fractional max-pooling :cite:`Graham.2014`, agregasi digabungkan dengan randomisasi. Ini dapat sedikit meningkatkan akurasi dalam beberapa kasus. Terakhir, seperti yang akan kita lihat nanti dengan mekanisme attention, ada cara yang lebih halus untuk melakukan agregasi pada output, misalnya dengan menggunakan penyelarasan antara vektor query dan representasi.


## Latihan

1. Implementasikan average pooling menggunakan konvolusi.
1. Buktikan bahwa max-pooling tidak dapat diimplementasikan hanya dengan menggunakan konvolusi.
1. Max-pooling dapat dicapai menggunakan operasi ReLU, yaitu $\textrm{ReLU}(x) = \max(0, x)$.
    1. Ekspresikan $\max (a, b)$ hanya dengan menggunakan operasi ReLU.
    1. Gunakan ini untuk mengimplementasikan max-pooling dengan konvolusi dan lapisan ReLU.
    1. Berapa banyak kanal dan lapisan yang Anda butuhkan untuk konvolusi $2 \times 2$? Berapa banyak untuk konvolusi $3 \times 3$?
1. Berapa biaya komputasi untuk lapisan pooling? Asumsikan bahwa input ke lapisan pooling berukuran $c\times h\times w$, pooling window memiliki bentuk $p_\textrm{h}\times p_\textrm{w}$ dengan padding $(p_\textrm{h}, p_\textrm{w})$ dan stride $(s_\textrm{h}, s_\textrm{w})$.
1. Mengapa Anda berharap max-pooling dan average pooling bekerja secara berbeda?
1. Apakah kita membutuhkan lapisan minimum pooling yang terpisah? Bisakah Anda menggantinya dengan operasi lain?
1. Kita bisa menggunakan operasi softmax untuk pooling. Mengapa mungkin ini tidak begitu populer?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/274)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17999)
:end_tab:
