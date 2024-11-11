```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Jaringan Multi-Cabang (GoogLeNet)
:label:`sec_googlenet`

Pada tahun 2014, *GoogLeNet* memenangkan ImageNet Challenge :cite:`Szegedy.Liu.Jia.ea.2015`, dengan menggunakan struktur yang menggabungkan keunggulan NiN :cite:`Lin.Chen.Yan.2013`, blok-blok berulang :cite:`Simonyan.Zisserman.2014`, dan berbagai kernel konvolusi. Jaringan ini bisa dibilang juga merupakan jaringan pertama yang menunjukkan perbedaan yang jelas antara stem (pemrosesan awal data), body (pemrosesan data utama), dan head (prediksi) dalam CNN. Pola desain ini telah bertahan dalam perancangan jaringan dalam sejak saat itu: *stem* terdiri dari dua atau tiga konvolusi pertama yang bekerja pada gambar dan mengekstrak fitur tingkat rendah dari gambar. Ini diikuti oleh *body* yang terdiri dari blok-blok konvolusi. Akhirnya, *head* memetakan fitur yang telah diperoleh untuk menyelesaikan masalah klasifikasi, segmentasi, deteksi, atau pelacakan yang diinginkan.

Kontribusi utama GoogLeNet terletak pada desain body jaringan. GoogLeNet menyelesaikan masalah pemilihan kernel konvolusi dengan cara yang cerdas. Sementara penelitian lain mencoba mengidentifikasi konvolusi mana, mulai dari $1 \times 1$ hingga $11 \times 11$, yang paling baik, GoogLeNet cukup *menggabungkan* konvolusi multi-cabang sekaligus.
Berikut ini, kami memperkenalkan versi GoogLeNet yang sedikit disederhanakan: desain aslinya menyertakan beberapa trik untuk menstabilkan pelatihan melalui fungsi loss intermediate, yang diterapkan pada beberapa lapisan jaringan. 
Trik tersebut tidak lagi diperlukan berkat adanya algoritma pelatihan yang lebih baik.


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
from torch.nn import functional as F
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
from jax import numpy as jnp
import jax
```

## (**Blok Inception**)

Blok konvolusi dasar dalam GoogLeNet disebut *Inception block*,
terinspirasi dari meme "we need to go deeper" dari film *Inception*.

![Struktur dari blok Inception.](../img/inception.svg)
:label:`fig_inception`

Seperti yang digambarkan pada :numref:`fig_inception`,
blok inception terdiri dari empat cabang (_branch_) paralel.
Tiga cabang pertama menggunakan lapisan konvolusi
dengan ukuran jendela $1\times 1$, $3\times 3$, dan $5\times 5$
untuk mengekstrak informasi dari ukuran spasial yang berbeda.
Dua cabang di tengah juga menambahkan konvolusi $1\times 1$ dari input
untuk mengurangi jumlah saluran, sehingga mengurangi kompleksitas model.
Cabang keempat menggunakan lapisan max-pooling $3\times 3$,
diikuti oleh lapisan konvolusi $1\times 1$
untuk mengubah jumlah saluran.
Keempat cabang tersebut menggunakan padding yang sesuai sehingga tinggi dan lebar input dan output tetap sama.
Akhirnya, output dari masing-masing cabang digabungkan
dalam dimensi saluran dan membentuk output dari blok tersebut.
Hyperparameter yang sering disesuaikan pada blok Inception
adalah jumlah saluran output per lapisan, yaitu cara mengalokasikan kapasitas di antara konvolusi dengan ukuran yang berbeda.


```{.python .input}
%%tab mxnet
class Inception(nn.Block):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Branch 2
        self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Branch 3
        self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Branch 4
        self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return np.concatenate((b1, b2, b3, b4), axis=1)
```

```{.python .input}
%%tab pytorch
class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

```{.python .input}
%%tab tensorflow
class Inception(tf.keras.Model):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])
```

```{.python .input}
%%tab jax
class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each branch
    c1: int
    c2: tuple
    c3: tuple
    c4: int

    def setup(self):
        # Branch 1
        self.b1_1 = nn.Conv(self.c1, kernel_size=(1, 1))
        # Branch 2
        self.b2_1 = nn.Conv(self.c2[0], kernel_size=(1, 1))
        self.b2_2 = nn.Conv(self.c2[1], kernel_size=(3, 3), padding='same')
        # Branch 3
        self.b3_1 = nn.Conv(self.c3[0], kernel_size=(1, 1))
        self.b3_2 = nn.Conv(self.c3[1], kernel_size=(5, 5), padding='same')
        # Branch 4
        self.b4_1 = lambda x: nn.max_pool(x, window_shape=(3, 3),
                                          strides=(1, 1), padding='same')
        self.b4_2 = nn.Conv(self.c4, kernel_size=(1, 1))

    def __call__(self, x):
        b1 = nn.relu(self.b1_1(x))
        b2 = nn.relu(self.b2_2(nn.relu(self.b2_1(x))))
        b3 = nn.relu(self.b3_2(nn.relu(self.b3_1(x))))
        b4 = nn.relu(self.b4_2(self.b4_1(x)))
        return jnp.concatenate((b1, b2, b3, b4), axis=-1)
```

Untuk mendapatkan pemahaman mengapa jaringan ini bekerja dengan sangat baik,
pertimbangkan kombinasi filter yang digunakan.
Filter tersebut menjelajahi gambar dengan berbagai ukuran filter.
Ini berarti bahwa detail pada berbagai skala
dapat dikenali secara efisien oleh filter dengan ukuran yang berbeda.
Pada saat yang sama, kita dapat mengalokasikan jumlah parameter yang berbeda
untuk filter yang berbeda.



## [**Model GoogLeNet**]

Seperti yang ditunjukkan pada :numref:`fig_inception_full`, GoogLeNet menggunakan susunan dari total 9 blok inception, diatur ke dalam tiga kelompok dengan max-pooling di antaranya, dan global average pooling di bagian head untuk menghasilkan perkiraannya.
Max-pooling di antara blok inception mengurangi dimensi.
Pada bagian stem-nya, modul pertama serupa dengan AlexNet dan LeNet.

![Arsitektur GoogLeNet.](../img/inception-full-90.svg)
:label:`fig_inception_full`

Sekarang kita dapat mengimplementasikan GoogLeNet bagian demi bagian. Mari kita mulai dengan bagian stem.
Modul pertama menggunakan lapisan konvolusi $7\times 7$ dengan 64 saluran.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GoogleNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                              activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class GoogleNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10

    def setup(self):
        self.net = nn.Sequential([self.b1(), self.b2(), self.b3(), self.b4(),
                                  self.b5(), nn.Dense(self.num_classes)])

    def b1(self):
        return nn.Sequential([
                nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                      padding='same')])
```

Modul kedua menggunakan dua lapisan konvolusi:
pertama, lapisan konvolusi $1\times 1$ dengan 64 saluran,
diikuti oleh lapisan konvolusi $3\times 3$ yang menggandakan jumlah saluran menjadi tiga kali lipat. Ini sesuai dengan cabang kedua dalam blok Inception dan menyelesaikan desain body. 
Pada tahap ini, kita memiliki 192 saluran.


```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b2(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([nn.Conv(64, kernel_size=(1, 1)),
                              nn.relu,
                              nn.Conv(192, kernel_size=(3, 3), padding='same'),
                              nn.relu,
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Modul ketiga menghubungkan dua blok Inception lengkap secara seri.
Jumlah saluran output dari blok Inception pertama adalah
$64+128+32+32=256$. Ini menghasilkan rasio jumlah saluran output
di antara keempat cabang yaitu $2:4:1:1$. Untuk mencapai ini, kita terlebih dahulu mengurangi dimensi input
dengan rasio $\frac{1}{2}$ dan $\frac{1}{12}$ pada cabang kedua dan ketiga masing-masing,
sehingga kita mendapatkan 96 ($=192/2$) dan 16 ($=192/12$) saluran.

Jumlah saluran output dari blok Inception kedua
ditingkatkan menjadi $128+192+96+64=480$, menghasilkan rasio $128:192:96:64 = 4:6:3:2$. Seperti sebelumnya,
kita perlu mengurangi jumlah dimensi intermediate pada saluran kedua dan ketiga. Skala
$\frac{1}{2}$ dan $\frac{1}{8}$ masing-masing cukup, menghasilkan 128 dan 32 saluran
masing-masing. Hal ini diatur melalui argumen pada konstruktor `Inception` block berikut.


```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b3(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                             Inception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(64, (96, 128), (16, 32), 32),
                              Inception(128, (128, 192), (32, 96), 64),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Modul keempat lebih rumit.
Modul ini menghubungkan lima blok Inception secara seri,
dengan jumlah saluran output masing-masing adalah $192+208+48+64=512$, $160+224+64+64=512$,
$128+256+64+64=512$, $112+288+64+64=528$,
dan $256+320+128+128=832$.
Jumlah saluran yang dialokasikan ke cabang-cabang ini mirip
dengan modul ketiga:
cabang kedua dengan lapisan konvolusi $3\times 3$
menghasilkan jumlah saluran terbesar,
diikuti oleh cabang pertama dengan lapisan konvolusi $1\times 1$ saja,
cabang ketiga dengan lapisan konvolusi $5\times 5$,
dan cabang keempat dengan lapisan max-pooling $3\times 3$.
Cabang kedua dan ketiga akan terlebih dahulu mengurangi
jumlah saluran sesuai dengan rasio yang ditentukan.
Rasio ini sedikit berbeda di berbagai blok Inception.


```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b4(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(192, (96, 208), (16, 48), 64),
                              Inception(160, (112, 224), (24, 64), 64),
                              Inception(128, (128, 256), (24, 64), 64),
                              Inception(112, (144, 288), (32, 64), 64),
                              Inception(256, (160, 320), (32, 128), 128),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Modul kelima memiliki dua blok Inception dengan jumlah saluran output $256+320+128+128=832$
dan $384+384+128+128=1024$.
Jumlah saluran yang dialokasikan untuk setiap cabang
sama seperti pada modul ketiga dan keempat,
tetapi berbeda dalam nilai spesifiknya.
Perlu dicatat bahwa blok kelima diikuti oleh lapisan output.
Blok ini menggunakan lapisan global average pooling
untuk mengubah tinggi dan lebar setiap saluran menjadi 1, seperti pada NiN.
Akhirnya, kita mengubah output menjadi array dua dimensi
yang diikuti oleh lapisan fully connected
dengan jumlah output yang sesuai dengan jumlah kelas label.


```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b5(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D())
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten()])
    if tab.selected('jax'):
        return nn.Sequential([Inception(256, (160, 320), (32, 128), 128),
                              Inception(384, (192, 384), (48, 128), 128),
                              # Flax does not provide a GlobalAvgPool2D layer
                              lambda x: nn.avg_pool(x,
                                                    window_shape=x.shape[1:3],
                                                    strides=x.shape[1:3],
                                                    padding='valid'),
                              lambda x: x.reshape((x.shape[0], -1))])
```

Sekarang setelah kita mendefinisikan semua blok `b1` hingga `b5`, langkah selanjutnya hanyalah merangkai semua blok tersebut menjadi satu jaringan penuh.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.Sequential([
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            tf.keras.layers.Dense(num_classes)])
```

Model GoogLeNet memiliki kompleksitas komputasi yang tinggi. Perhatikan banyaknya hyperparameter yang relatif sewenang-wenang dalam hal jumlah saluran yang dipilih, jumlah blok sebelum pengurangan dimensi, pembagian kapasitas relatif di antara saluran, dan sebagainya. Sebagian besar dari ini disebabkan oleh fakta bahwa pada saat GoogLeNet diperkenalkan, alat otomatis untuk definisi jaringan atau eksplorasi desain belum tersedia. Misalnya, saat ini kita menganggap bahwa kerangka kerja deep learning yang kompeten mampu menyimpulkan dimensi tensor input secara otomatis. Pada saat itu, banyak konfigurasi seperti ini harus ditentukan secara eksplisit oleh peneliti, yang sering kali memperlambat eksperimen aktif. Selain itu, alat yang dibutuhkan untuk eksplorasi otomatis masih dalam pengembangan dan eksperimen awal sebagian besar berujung pada eksplorasi brute-force yang mahal, algoritma genetika, dan strategi serupa.

Untuk saat ini, satu-satunya modifikasi yang akan kita lakukan adalah
[**mengurangi tinggi dan lebar input dari 224 menjadi 96
untuk mendapatkan waktu pelatihan yang masuk akal pada Fashion-MNIST.**]
Ini akan menyederhanakan komputasi. Mari kita lihat
perubahan dalam bentuk output di antara berbagai modul.


```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow, jax
model = GoogleNet().layer_summary((1, 96, 96, 1))
```

## [**Pelatihan**]

Seperti sebelumnya, kita melatih model menggunakan dataset Fashion-MNIST.
Kita mengubah resolusinya menjadi $96 \times 96$ piksel
sebelum memulai prosedur pelatihan.


```{.python .input}
%%tab mxnet, pytorch, jax
model = GoogleNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = GoogleNet(lr=0.01)
    trainer.fit(model, data)
```

## Diskusi

Fitur utama GoogLeNet adalah bahwa model ini sebenarnya *lebih murah* untuk dihitung dibandingkan dengan pendahulunya, sambil tetap memberikan akurasi yang lebih baik. Ini menandai dimulainya perancangan jaringan yang lebih sengaja, yang menyeimbangkan biaya evaluasi jaringan dengan pengurangan kesalahan. Ini juga menandai awal dari eksperimen pada tingkat blok dengan hyperparameter desain jaringan, meskipun saat itu masih sepenuhnya manual. Kami akan mengulas kembali topik ini pada :numref:`sec_cnn-design` ketika membahas strategi untuk eksplorasi struktur jaringan.

Pada bagian-bagian berikutnya, kita akan menemui sejumlah pilihan desain (misalnya, batch normalization, residual connections, dan channel grouping) yang memungkinkan kita untuk meningkatkan jaringan secara signifikan. Untuk saat ini, Anda bisa bangga telah mengimplementasikan CNN modern pertama yang sesungguhnya.

## Latihan

1. GoogLeNet begitu sukses sehingga mengalami beberapa iterasi, secara bertahap meningkatkan kecepatan dan akurasi. Cobalah untuk mengimplementasikan dan menjalankan beberapa di antaranya. Ini termasuk:
    1. Menambahkan lapisan batch normalization :cite:`Ioffe.Szegedy.2015`, seperti yang dijelaskan di kemudian hari pada :numref:`sec_batch_norm`.
    1. Melakukan penyesuaian pada blok Inception (lebar, pilihan dan urutan konvolusi), seperti yang dijelaskan dalam :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    1. Menggunakan label smoothing untuk regularisasi model, seperti yang dijelaskan dalam :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    1. Melakukan penyesuaian lebih lanjut pada blok Inception dengan menambahkan residual connection :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`, seperti yang dijelaskan di kemudian hari pada :numref:`sec_resnet`.
1. Berapa ukuran gambar minimum yang dibutuhkan agar GoogLeNet dapat berfungsi?
1. Bisakah Anda merancang varian GoogLeNet yang berfungsi pada resolusi asli Fashion-MNIST sebesar $28 \times 28$ piksel? Bagaimana Anda perlu mengubah stem, body, dan head dari jaringan ini, jika perlu?
1. Bandingkan ukuran parameter model dari AlexNet, VGG, NiN, dan GoogLeNet. Bagaimana dua arsitektur jaringan terakhir secara signifikan mengurangi ukuran parameter model?
1. Bandingkan jumlah komputasi yang diperlukan pada GoogLeNet dan AlexNet. Bagaimana hal ini memengaruhi desain chip akselerator, misalnya dalam hal ukuran memori, bandwidth memori, ukuran cache, jumlah komputasi, dan manfaat operasi khusus?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/316)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18004)
:end_tab:
