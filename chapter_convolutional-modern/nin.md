```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Network in Network (NiN)
:label:`sec_nin`

LeNet, AlexNet, dan VGG semuanya memiliki pola desain yang sama:
mengekstrak fitur dengan memanfaatkan struktur *spasial*
melalui urutan lapisan konvolusi dan pooling,
dan memproses ulang representasi melalui lapisan fully connected.
Peningkatan dari LeNet ke AlexNet dan VGG terutama terletak
pada bagaimana jaringan yang lebih baru ini memperlebar dan memperdalam kedua modul tersebut.

Desain ini menghadirkan dua tantangan utama.
Pertama, lapisan fully connected di akhir arsitektur
mengkonsumsi sejumlah besar parameter. Sebagai contoh, bahkan model sederhana
seperti VGG-11 memerlukan matriks yang sangat besar, yang membutuhkan hampir 400MB RAM dalam presisi tunggal (FP32). Ini adalah hambatan signifikan bagi komputasi, terutama pada perangkat mobile dan embedded. Bagaimanapun, bahkan ponsel kelas atas hanya memiliki RAM tidak lebih dari 8GB. Pada saat VGG ditemukan, ini masih satu orde lebih kecil (iPhone 4S hanya memiliki 512MB). Oleh karena itu, sulit untuk membenarkan penggunaan sebagian besar memori untuk pengklasifikasi gambar.

Kedua, juga tidak mungkin untuk menambahkan lapisan fully connected
lebih awal dalam jaringan untuk meningkatkan derajat non-linearitas: melakukan hal ini akan menghancurkan
struktur spasial dan memerlukan lebih banyak memori.

Blok *network in network* (*NiN*) :cite:`Lin.Chen.Yan.2013` menawarkan alternatif,
yang mampu menyelesaikan kedua masalah tersebut dalam satu strategi sederhana.
Blok ini diajukan berdasarkan sebuah wawasan yang sangat sederhana: (i) gunakan konvolusi $1 \times 1$ untuk menambahkan non-linearitas lokal di seluruh aktivasi saluran dan (ii) gunakan global average pooling untuk mengintegrasikan seluruh lokasi pada lapisan representasi terakhir. Perhatikan bahwa global average pooling tidak akan efektif jika tidak ada non-linearitas tambahan. Mari kita selami ini lebih detail.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
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
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## (**Blok NiN**)

Ingat kembali :numref:`subsec_1x1`. Di dalamnya, kita menyebutkan bahwa input dan output dari lapisan konvolusi
terdiri dari tensor berdimensi empat dengan sumbu yang sesuai dengan contoh, saluran, tinggi, dan lebar.
Ingat juga bahwa input dan output dari lapisan fully connected
biasanya berupa tensor berdimensi dua yang sesuai dengan contoh dan fitur.
Ide di balik NiN adalah menerapkan lapisan fully connected
pada setiap lokasi piksel (untuk setiap tinggi dan lebar).
Konvolusi $1 \times 1$ yang dihasilkan dapat dianggap sebagai
lapisan fully connected yang bekerja secara independen pada setiap lokasi piksel.

:numref:`fig_nin` mengilustrasikan perbedaan struktural utama
antara VGG dan NiN, serta blok-bloknya.
Perhatikan perbedaan pada blok NiN (konvolusi awal diikuti oleh konvolusi $1 \times 1$, sementara VGG tetap menggunakan konvolusi $3 \times 3$) dan pada bagian akhir di mana kita tidak lagi membutuhkan lapisan fully connected yang sangat besar.

![Membandingkan arsitektur VGG dan NiN, serta blok-bloknya.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`


```{.python .input}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])
```

## [**Model NiN**]

NiN menggunakan ukuran konvolusi awal yang sama seperti AlexNet (diusulkan tidak lama setelahnya).
Ukuran kernel masing-masing adalah $11\times 11$, $5\times 5$, dan $3\times 3$,
dan jumlah saluran output sama dengan milik AlexNet. Setiap blok NiN diikuti oleh lapisan max-pooling
dengan stride 2 dan bentuk jendela $3\times 3$.

Perbedaan signifikan kedua antara NiN dengan AlexNet dan VGG adalah
bahwa NiN menghindari penggunaan lapisan fully connected sama sekali.
Sebagai gantinya, NiN menggunakan blok NiN dengan jumlah saluran output yang sama dengan jumlah kelas label, yang diikuti oleh lapisan *global* average pooling,
yang menghasilkan vektor logit.
Desain ini secara signifikan mengurangi jumlah parameter model yang dibutuhkan, meskipun dengan kemungkinan peningkatan waktu pelatihan.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab jax
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # global avg pooling
            lambda x: x.reshape((x.shape[0], -1))  # flatten
        ])
```

Kita membuat contoh data untuk melihat [**bentuk output dari setiap blok**].

```{.python .input}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
NiN().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
NiN(training=False).layer_summary((1, 224, 224, 1))
```

## [**Pelatihan**]

Seperti sebelumnya, kita menggunakan dataset Fashion-MNIST untuk melatih model ini dengan menggunakan optimizer yang sama dengan yang digunakan pada AlexNet dan VGG.


```{.python .input}
%%tab mxnet, pytorch, jax
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## Ringkasan

NiN memiliki jumlah parameter yang jauh lebih sedikit dibandingkan dengan AlexNet dan VGG. Hal ini terutama disebabkan oleh fakta bahwa NiN tidak memerlukan lapisan fully connected yang besar. Sebagai gantinya, NiN menggunakan global average pooling untuk mengintegrasikan seluruh lokasi gambar setelah tahap terakhir dari body jaringan. Ini menghilangkan kebutuhan akan operasi reduksi (yang dipelajari) yang mahal dan menggantikannya dengan operasi rata-rata sederhana. Yang mengejutkan para peneliti pada saat itu adalah kenyataan bahwa operasi rata-rata ini tidak merusak akurasi. Perlu dicatat bahwa rata-rata pada representasi beresolusi rendah (dengan banyak saluran) juga menambah jumlah invariansi translasi yang dapat ditangani oleh jaringan.

Memilih konvolusi yang lebih sedikit dengan kernel yang lebar dan menggantinya dengan konvolusi $1 \times 1$ lebih lanjut membantu dalam mengurangi jumlah parameter. Ini dapat menyediakan sejumlah besar non-linearitas di antara saluran pada setiap lokasi yang diberikan. Baik konvolusi $1 \times 1$ maupun global average pooling memiliki pengaruh signifikan pada desain CNN berikutnya.

## Latihan

1. Mengapa terdapat dua lapisan konvolusi $1\times 1$ per blok NiN? Tambahkan jumlahnya menjadi tiga. Kurangi jumlahnya menjadi satu. Apa yang berubah?
2. Apa yang berubah jika Anda mengganti konvolusi $1 \times 1$ dengan konvolusi $3 \times 3$?
3. Apa yang terjadi jika Anda mengganti global average pooling dengan lapisan fully connected (dalam hal kecepatan, akurasi, jumlah parameter)?
4. Hitung penggunaan sumber daya untuk NiN.
    1. Berapa jumlah parameter?
    2. Berapa jumlah komputasi yang diperlukan?
    3. Berapa jumlah memori yang diperlukan selama pelatihan?
    4. Berapa jumlah memori yang diperlukan selama prediksi?
5. Apa kemungkinan masalah jika mengurangi representasi $384 \times 5 \times 5$ menjadi representasi $10 \times 5 \times 5$ dalam satu langkah?
6. Gunakan keputusan desain struktural dalam VGG yang mengarah ke VGG-11, VGG-16, dan VGG-19 untuk merancang keluarga jaringan yang mirip NiN.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18003)
:end_tab:
