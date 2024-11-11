```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Jaringan dengan Menggunakan Blok (VGG)
:label:`sec_vgg`

Meskipun AlexNet memberikan bukti empiris bahwa CNN yang dalam
dapat mencapai hasil yang baik, model ini tidak menyediakan template umum
untuk membimbing para peneliti berikutnya dalam merancang jaringan baru.
Pada bagian-bagian berikutnya, kita akan memperkenalkan beberapa konsep heuristik
yang umum digunakan untuk merancang jaringan dalam.

Perkembangan dalam bidang ini mencerminkan perkembangan pada VLSI (very large scale integration)
dalam desain chip, di mana para insinyur beralih dari menempatkan transistor
ke elemen logika dan kemudian ke blok logika :cite:`Mead.1980`.
Demikian pula, desain arsitektur jaringan saraf
semakin abstrak, dengan peneliti yang awalnya berpikir dalam hal
neuron individu beralih ke seluruh lapisan,
dan kini ke blok, pola lapisan berulang. Satu dekade kemudian, ini telah berkembang menjadi penggunaan model yang telah dilatih seluruhnya untuk berbagai tujuan,
meskipun dalam tugas yang terkait. Model besar yang telah dilatih sebelumnya ini biasanya disebut
*foundation models* :cite:`bommasani2021opportunities`.

Kembali ke desain jaringan. Ide menggunakan blok pertama kali muncul dari
Visual Geometry Group (VGG) di Universitas Oxford,
dalam jaringan bernama *VGG* sesuai nama kelompoknya :cite:`Simonyan.Zisserman.2014`.
Mengimplementasikan struktur berulang ini dalam kode
sangat mudah dengan framework deep learning modern mana pun
dengan menggunakan loop dan subrutin.


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
```

## (**Blok VGG**)
:label:`subsec_vgg-blocks`

Blok dasar dari CNN adalah urutan berikut:
(i) lapisan konvolusi dengan padding untuk menjaga resolusi,
(ii) non-linearitas seperti ReLU,
(iii) lapisan pooling seperti max-pooling untuk mengurangi resolusi. Salah satu masalah dari
pendekatan ini adalah bahwa resolusi spasial berkurang dengan cepat. Secara khusus,
ini memberikan batas keras pada jumlah lapisan konvolusi sebanyak $\log_2 d$ dalam jaringan sebelum semua
dimensi ($d$) terpakai. Misalnya, dalam kasus ImageNet, dengan pendekatan ini tidak mungkin memiliki
lebih dari 8 lapisan konvolusi.

Ide utama dari :citet:`Simonyan.Zisserman.2014` adalah menggunakan *beberapa* konvolusi di antara downsampling
melalui max-pooling dalam bentuk blok. Mereka tertarik untuk mengetahui apakah jaringan yang dalam atau
lebar yang berkinerja lebih baik. Misalnya, penerapan berturut-turut dua konvolusi $3 \times 3$
menjangkau piksel yang sama seperti yang dilakukan oleh konvolusi tunggal $5 \times 5$. Pada saat yang sama, konvolusi tunggal $5 \times 5$ menggunakan jumlah parameter yang hampir sama ($25 \cdot c^2$) dengan tiga konvolusi $3 \times 3$ ($3 \cdot 9 \cdot c^2$).
Dalam analisis yang cukup rinci, mereka menunjukkan bahwa jaringan yang dalam dan sempit jauh lebih unggul daripada jaringan yang dangkal. Ini memicu perkembangan deep learning untuk mencari jaringan yang semakin dalam dengan lebih dari 100 lapisan untuk aplikasi umum.
Penumpukan konvolusi $3 \times 3$ telah menjadi standar emas pada jaringan dalam berikutnya (sebuah keputusan desain yang baru-baru ini ditinjau kembali oleh :citet:`liu2022convnet`). Akibatnya, implementasi cepat untuk konvolusi kecil telah menjadi andalan pada GPU :cite:`lavin2016fast`.

Kembali ke VGG: sebuah blok VGG terdiri dari *urutan* konvolusi dengan kernel $3\times3$ dengan padding 1
(menjaga tinggi dan lebar) diikuti oleh lapisan max-pooling $2 \times 2$ dengan stride 2
(mengurangi tinggi dan lebar menjadi setengahnya setelah setiap blok).
Pada kode di bawah ini, kita mendefinisikan fungsi bernama `vgg_block`
untuk mengimplementasikan satu blok VGG.

Fungsi di bawah ini menerima dua argumen,
yang sesuai dengan jumlah lapisan konvolusi `num_convs`
dan jumlah saluran output `num_channels`.


```{.python .input  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input  n=4}
%%tab tensorflow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab jax
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## [**Jaringan VGG**]
:label:`subsec_vgg-network`

Seperti AlexNet dan LeNet,
Jaringan VGG dapat dibagi menjadi dua bagian:
bagian pertama terdiri terutama dari lapisan konvolusi dan pooling,
dan bagian kedua terdiri dari lapisan fully connected yang identik dengan yang ada di AlexNet.
Perbedaan utamanya adalah
bahwa lapisan konvolusi dikelompokkan dalam transformasi non-linear
yang menjaga dimensi tetap sama, diikuti dengan langkah pengurangan resolusi, seperti
yang digambarkan pada :numref:`fig_vgg`.

![Dari AlexNet ke VGG. Perbedaan utamanya adalah bahwa VGG terdiri dari blok-blok lapisan, sementara lapisan AlexNet dirancang secara individual.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

Bagian konvolusi dari jaringan menghubungkan beberapa blok VGG dari :numref:`fig_vgg` (juga didefinisikan dalam fungsi `vgg_block`)
secara berurutan. Pengelompokan konvolusi ini adalah pola yang
tetap hampir tidak berubah selama dekade terakhir, meskipun pilihan operasi tertentu telah mengalami modifikasi yang cukup besar.
Variabel `arch` terdiri dari daftar tuple (satu per blok),
di mana masing-masing berisi dua nilai: jumlah lapisan konvolusi
dan jumlah saluran output,
yang merupakan argumen yang diperlukan untuk memanggil
fungsi `vgg_block`. Dengan demikian, VGG mendefinisikan *keluarga* jaringan daripada sekadar
satu bentuk spesifik. Untuk membangun jaringan tertentu, kita cukup melakukan iterasi pada `arch` untuk menyusun blok-bloknya.


```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```{.python .input  n=5}
%%tab jax
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

Jaringan VGG asli memiliki lima blok konvolusi,
di mana dua blok pertama masing-masing memiliki satu lapisan konvolusi
dan tiga blok terakhir masing-masing berisi dua lapisan konvolusi.
Blok pertama memiliki 64 saluran output,
dan setiap blok berikutnya menggandakan jumlah saluran output,
hingga jumlah tersebut mencapai 512.
Karena jaringan ini menggunakan delapan lapisan konvolusi
dan tiga lapisan fully connected, maka sering disebut VGG-11.


```{.python .input  n=6}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

```{.python .input}
%%tab jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    training=False).layer_summary((1, 224, 224, 1))
```

Seperti yang Anda lihat, kita mengurangi setengah tinggi dan lebar di setiap blok,
hingga akhirnya mencapai tinggi dan lebar 7
sebelum melakukan flattening pada representasi
untuk diproses oleh bagian fully connected dari jaringan.
:citet:`Simonyan.Zisserman.2014` juga menjelaskan beberapa varian lain dari VGG.
Faktanya, sekarang menjadi hal yang biasa untuk mengusulkan *keluarga* jaringan dengan
perbedaan trade-off antara kecepatan dan akurasi ketika memperkenalkan arsitektur baru.

## Pelatihan

[**Karena VGG-11 membutuhkan komputasi yang lebih besar daripada AlexNet,
kami membangun jaringan dengan jumlah saluran yang lebih sedikit.**]
Ini sudah lebih dari cukup untuk pelatihan pada Fashion-MNIST.
Proses [**pelatihan model**] mirip dengan AlexNet pada :numref:`sec_alexnet`.
Perhatikan kembali kesesuaian yang erat antara loss validasi dan loss pelatihan,
yang menunjukkan hanya sedikit overfitting.


```{.python .input  n=8}
%%tab mxnet, pytorch, jax
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## Ringkasan

Orang mungkin berpendapat bahwa VGG adalah jaringan saraf konvolusional modern yang pertama. Meskipun AlexNet memperkenalkan banyak komponen yang membuat deep learning efektif pada skala besar, VGG bisa dibilang yang memperkenalkan properti kunci seperti blok konvolusi ganda dan preferensi untuk jaringan yang dalam dan sempit. Ini juga merupakan jaringan pertama yang menjadi seluruh keluarga model dengan parameterisasi serupa, memberikan praktisi berbagai pilihan trade-off antara kompleksitas dan kecepatan. Di sini pula kerangka kerja deep learning modern unggul, karena tidak lagi diperlukan file konfigurasi XML untuk menentukan jaringan, melainkan cukup dengan merakit jaringan melalui kode Python sederhana.

Baru-baru ini, ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021` menunjukkan bahwa dimungkinkan untuk mencapai performa kompetitif dengan menggunakan arsitektur yang jauh lebih dangkal melalui sejumlah besar komputasi paralel. Ini adalah perkembangan yang menarik dan ada harapan bahwa hal ini akan mempengaruhi desain arsitektur di masa depan. Namun, untuk sisa bab ini, kita akan mengikuti jalur perkembangan ilmiah selama dekade terakhir.

## Latihan

1. Dibandingkan dengan AlexNet, VGG jauh lebih lambat dalam hal komputasi, dan juga membutuhkan lebih banyak memori GPU.
    1. Bandingkan jumlah parameter yang dibutuhkan oleh AlexNet dan VGG.
    1. Bandingkan jumlah operasi floating-point yang digunakan pada lapisan konvolusi dan pada lapisan fully connected.
    1. Bagaimana Anda bisa mengurangi biaya komputasi yang diciptakan oleh lapisan fully connected?
1. Saat menampilkan dimensi yang terkait dengan berbagai lapisan dalam jaringan, kita hanya melihat informasi yang terkait dengan delapan blok (ditambah beberapa transformasi tambahan), meskipun jaringan memiliki 11 lapisan. Di mana tiga lapisan yang tersisa?
1. Gunakan Tabel 1 dalam makalah VGG :cite:`Simonyan.Zisserman.2014` untuk membangun model umum lainnya, seperti VGG-16 atau VGG-19.
1. Meningkatkan resolusi pada Fashion-MNIST sebanyak delapan kali lipat dari dimensi $28 \times 28$ ke $224 \times 224$ sangat tidak efisien. Cobalah memodifikasi arsitektur jaringan dan konversi resolusi, misalnya, menjadi 56 atau 84 dimensi untuk inputnya. Bisakah Anda melakukannya tanpa mengurangi akurasi jaringan? Konsultasikan makalah VGG :cite:`Simonyan.Zisserman.2014` untuk ide-ide dalam menambahkan lebih banyak non-linearitas sebelum downsampling.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/277)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18002)
:end_tab:
