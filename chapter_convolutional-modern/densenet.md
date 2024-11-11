```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Densely Connected Networks (DenseNet)
:label:`sec_densenet`

ResNet secara signifikan mengubah cara kita memandang parameterisasi fungsi dalam jaringan yang dalam. *DenseNet* (jaringan konvolusi padat) hingga tingkat tertentu adalah perpanjangan logis dari ini :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
DenseNet ditandai dengan pola konektivitas di mana
setiap lapisan terhubung ke semua lapisan sebelumnya
dan operasi konkatenasi (bukan operator penjumlahan seperti pada ResNet) untuk mempertahankan dan menggunakan kembali fitur
dari lapisan sebelumnya.
Untuk memahami bagaimana sampai pada konsep ini, mari kita melakukan sedikit penyimpangan ke dalam matematika.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
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

## Dari ResNet ke DenseNet

Ingat kembali ekspansi Taylor untuk fungsi. Pada titik $x = 0$, ekspansi ini dapat dituliskan sebagai

$$f(x) = f(0) + x \cdot \left[f'(0) + x \cdot \left[\frac{f''(0)}{2!}  + x \cdot \left[\frac{f'''(0)}{3!}  + \cdots \right]\right]\right].$$

Poin utamanya adalah bahwa ekspansi ini menguraikan sebuah fungsi menjadi beberapa suku dengan urutan yang semakin tinggi. Dalam cara yang serupa, ResNet menguraikan fungsi menjadi

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

Artinya, ResNet menguraikan $f$ menjadi satu suku linear sederhana dan satu suku non-linear yang lebih kompleks.
Bagaimana jika kita ingin menangkap (tidak selalu menambah) informasi di luar dua suku tersebut?
Salah satu solusinya adalah DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.

![Perbedaan utama antara ResNet (kiri) dan DenseNet (kanan) dalam koneksi antar lapisan: penggunaan penjumlahan dan penggunaan konkatenasi.](../img/densenet-block.svg)
:label:`fig_densenet_block`

Seperti yang ditunjukkan dalam :numref:`fig_densenet_block`, perbedaan utama antara ResNet dan DenseNet adalah pada kasus yang terakhir output *dikonkat* (ditandai dengan $[,]$) daripada ditambahkan.
Hasilnya, kita melakukan pemetaan dari $\mathbf{x}$ ke nilainya setelah menerapkan urutan fungsi yang semakin kompleks:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), f_3\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right)\right]\right), \ldots\right].$$

Pada akhirnya, semua fungsi ini digabungkan dalam MLP untuk mengurangi jumlah fitur kembali. Dalam hal implementasi, ini cukup sederhana: alih-alih menambahkan suku, kita mengkonkatnya. Nama DenseNet berasal dari fakta bahwa grafik ketergantungan antara variabel menjadi cukup padat. Lapisan terakhir dari rantai ini terhubung secara padat ke semua lapisan sebelumnya. Koneksi padat ini ditunjukkan dalam :numref:`fig_densenet`.

![Koneksi padat dalam DenseNet. Perhatikan bagaimana dimensi meningkat dengan kedalaman.](../img/densenet.svg)
:label:`fig_densenet`

Komponen utama yang membentuk DenseNet adalah *dense block* dan *transition layer*. Dense block mendefinisikan bagaimana input dan output dikonkat, sementara transition layer mengendalikan jumlah saluran sehingga tidak terlalu besar,
karena ekspansi $\mathbf{x} \to \left[\mathbf{x}, f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), \ldots \right]$ dapat menjadi sangat berdimensi tinggi.


## [**Dense Blocks**]

DenseNet menggunakan struktur "batch normalization, aktivasi, dan konvolusi" yang telah dimodifikasi dari ResNet (lihat latihan pada :numref:`sec_resnet`).
Pertama, kita implementasikan struktur blok konvolusi ini.


```{.python .input}
%%tab mxnet
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
%%tab pytorch
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```

```{.python .input}
%%tab tensorflow
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

```{.python .input}
%%tab jax
class ConvBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        Y = nn.relu(nn.BatchNorm(not self.training)(X))
        Y = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))(Y)
        Y = jnp.concatenate((X, Y), axis=-1)
        return Y
```

Sebuah *dense block* terdiri dari beberapa blok konvolusi, masing-masing menggunakan jumlah saluran output yang sama. Namun, dalam propagasi ke depan, kita mengkonkat input dan output dari setiap blok konvolusi pada dimensi saluran. Evaluasi malas (lazy evaluation) memungkinkan kita untuk menyesuaikan dimensi secara otomatis.

```{.python .input}
%%tab mxnet
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
%%tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
%%tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

```{.python .input}
%%tab jax
class DenseBlock(nn.Module):
    num_convs: int
    num_channels: int
    training: bool = True

    def setup(self):
        layer = []
        for i in range(self.num_convs):
            layer.append(ConvBlock(self.num_channels, self.training))
        self.net = nn.Sequential(layer)

    def __call__(self, X):
        return self.net(X)
```

Pada contoh berikut, kita [**mendefinisikan instance `DenseBlock`**] dengan dua blok konvolusi yang masing-masing memiliki 10 saluran output.
Ketika menggunakan input dengan tiga saluran, kita akan mendapatkan output dengan jumlah saluran $3 + 10 + 10 = 23$. Jumlah saluran pada blok konvolusi mengontrol pertumbuhan jumlah 
saluran output relatif terhadap jumlah saluran input. Hal ini juga disebut sebagai *growth rate*.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = DenseBlock(2, 10)
if tab.selected('mxnet'):
    X = np.random.uniform(size=(4, 3, 8, 8))
    blk.initialize()
if tab.selected('pytorch'):
    X = torch.randn(4, 3, 8, 8)
if tab.selected('tensorflow'):
    X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = DenseBlock(2, 10)
X = jnp.zeros((4, 8, 8, 3))
Y = blk.init_with_output(d2l.get_key(), X)[0]
Y.shape
```

## [**Transition Layer**]

Karena setiap dense block akan meningkatkan jumlah saluran, menambahkan terlalu banyak blok akan menghasilkan model yang terlalu kompleks. *Transition layer* digunakan untuk mengendalikan kompleksitas model. Transition layer mengurangi jumlah saluran dengan menggunakan konvolusi $1\times 1$. Selain itu, transition layer juga membagi dua tinggi dan lebar dengan menggunakan average pooling dengan stride 2.


```{.python .input}
%%tab mxnet
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
%%tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```{.python .input}
%%tab jax
class TransitionBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        X = nn.BatchNorm(not self.training)(X)
        X = nn.relu(X)
        X = nn.Conv(self.num_channels, kernel_size=(1, 1))(X)
        X = nn.avg_pool(X, window_shape=(2, 2), strides=(2, 2))
        return X
```

[**Terapkan transition layer**] dengan 10 saluran pada output dari dense block di contoh sebelumnya. Ini akan mengurangi jumlah saluran output menjadi 10, serta membagi dua tinggi dan lebar dari output tersebut.

```{.python .input}
%%tab mxnet
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
%%tab pytorch
blk = transition_block(10)
blk(Y).shape
```

```{.python .input}
%%tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

```{.python .input}
%%tab jax
blk = TransitionBlock(10)
blk.init_with_output(d2l.get_key(), Y)[0].shape
```

## [**Model DenseNet**]

Selanjutnya, kita akan membangun model DenseNet. DenseNet pertama-tama menggunakan lapisan konvolusi tunggal dan lapisan max-pooling yang sama seperti pada ResNet.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class DenseNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    64, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(
                    pool_size=3, strides=2, padding='same')])
```

```{.python .input}
%%tab jax
class DenseNet(d2l.Classifier):
    num_channels: int = 64
    growth_rate: int = 32
    arch: tuple = (4, 4, 4, 4)
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3),
                                  strides=(2, 2), padding='same')
        ])
```

Kemudian, mirip dengan empat modul yang terdiri dari residual block yang digunakan ResNet,
DenseNet menggunakan empat dense block.
Seperti halnya pada ResNet, kita bisa mengatur jumlah lapisan konvolusi yang digunakan di setiap dense block. Di sini, kita menetapkannya menjadi 4, konsisten dengan model ResNet-18 di :numref:`sec_resnet`. Selain itu, kita mengatur jumlah saluran (yaitu, growth rate) untuk lapisan konvolusi dalam dense block menjadi 32, sehingga 128 saluran akan ditambahkan ke setiap dense block.

Dalam ResNet, tinggi dan lebar berkurang di antara setiap modul oleh residual block dengan stride 2. Di sini, kita menggunakan transition layer untuk membagi dua tinggi dan lebar, serta membagi dua jumlah saluran. Mirip dengan ResNet, lapisan global pooling dan lapisan fully connected terhubung di akhir untuk menghasilkan output.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # Jumlah saluran output pada dense block sebelumnya
            num_channels += num_convs * growth_rate
            # Transition layer yang membagi dua jumlah saluran ditambahkan
            # di antara dense block
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(transition_block(num_channels))
        self.net.add(nn.BatchNorm(), nn.Activation('relu'),
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                              growth_rate))
            # Jumlah saluran output pada dense block sebelumnya
            num_channels += num_convs * growth_rate
            # Transition layer yang membagi dua jumlah saluran ditambahkan
            # di antara dense block
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # Jumlah saluran output pada dense block sebelumnya
            num_channels += num_convs * growth_rate
            # Transition layer yang membagi dua jumlah saluran ditambahkan
            # di antara dense block
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(TransitionBlock(num_channels))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(DenseNet)
def create_net(self):
    net = self.b1()
    for i, num_convs in enumerate(self.arch):
        net.layers.extend([DenseBlock(num_convs, self.growth_rate,
                                      training=self.training)])
        # Jumlah saluran output pada dense block sebelumnya
        num_channels = self.num_channels + (num_convs * self.growth_rate)
        # Transition layer yang membagi dua jumlah saluran ditambahkan
        # di antara dense block
        if i != len(self.arch) - 1:
            num_channels //= 2
            net.layers.extend([TransitionBlock(num_channels,
                                               training=self.training)])
    net.layers.extend([
        nn.BatchNorm(not self.training),
        nn.relu,
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)
    ])
    return net
```

## [**Training**]

Karena kita menggunakan jaringan yang lebih dalam di sini, pada bagian ini kita akan mengurangi tinggi dan lebar input dari 224 menjadi 96 untuk menyederhanakan komputasi.

```{.python .input}
%%tab mxnet, pytorch, jax
model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = DenseNet(lr=0.01)
    trainer.fit(model, data)
```

## Ringkasan dan Diskusi

Komponen utama yang membentuk DenseNet adalah dense block dan transition layer. Untuk transition layer, kita perlu menjaga dimensi tetap terkendali saat membangun jaringan dengan menambahkan transition layer yang mengurangi kembali jumlah saluran.
Dalam hal koneksi antar lapisan, berbeda dengan ResNet di mana input dan output dijumlahkan, DenseNet menggabungkan input dan output pada dimensi saluran.
Meskipun operasi konkatenasi ini
menggunakan kembali fitur untuk mencapai efisiensi komputasi,
sayangnya ini menyebabkan konsumsi memori GPU yang berat.
Akibatnya,
menerapkan DenseNet mungkin memerlukan implementasi yang lebih efisien dalam penggunaan memori, yang dapat meningkatkan waktu pelatihan :cite:`pleiss2017memory`.

## Latihan

1. Mengapa kita menggunakan average pooling daripada max-pooling pada transition layer?
2. Salah satu kelebihan yang disebutkan dalam paper DenseNet adalah bahwa parameter modelnya lebih kecil dibandingkan dengan ResNet. Mengapa demikian?
3. Salah satu masalah yang dikritik dari DenseNet adalah konsumsi memori yang tinggi.
    1. Apakah ini benar-benar terjadi? Coba ubah bentuk input menjadi $224\times 224$ untuk membandingkan konsumsi memori GPU secara empiris.
    2. Dapatkah Anda memikirkan cara alternatif untuk mengurangi konsumsi memori? Bagaimana Anda perlu mengubah framework?
4. Implementasikan berbagai versi DenseNet yang disajikan dalam Tabel 1 pada paper DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
5. Rancang model berbasis MLP dengan menerapkan ide DenseNet. Terapkan pada tugas prediksi harga rumah di :numref:`sec_kaggle_house`.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/331)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18008)
:end_tab:
