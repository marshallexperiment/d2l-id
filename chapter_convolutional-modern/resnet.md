```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Jaringan Residual (ResNet) dan ResNeXt
:label:`sec_resnet`

Saat kita merancang jaringan yang semakin dalam, menjadi penting untuk memahami bagaimana penambahan lapisan dapat meningkatkan kompleksitas dan kemampuan ekspresif dari jaringan tersebut.
Yang lebih penting lagi adalah kemampuan untuk merancang jaringan di mana penambahan lapisan membuat jaringan secara ketat lebih ekspresif daripada sekadar berbeda.
Untuk membuat beberapa kemajuan, kita membutuhkan sedikit matematika.


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


## Kelas Fungsi

Pertimbangkan $\mathcal{F}$, yaitu kelas fungsi yang dapat dicapai oleh arsitektur jaringan tertentu (bersama dengan learning rate dan pengaturan hyperparameter lainnya).
Artinya, untuk semua $f \in \mathcal{F}$ terdapat beberapa set parameter (misalnya, bobot dan bias) yang dapat diperoleh melalui pelatihan pada dataset yang sesuai.
Mari kita asumsikan bahwa $f^*$ adalah fungsi "kebenaran" yang benar-benar ingin kita temukan.
Jika $f^*$ berada dalam $\mathcal{F}$, kita berada dalam kondisi baik, tetapi umumnya kita tidak akan seberuntung itu.
Sebaliknya, kita akan mencoba menemukan $f^*_\mathcal{F}$ yang merupakan pilihan terbaik kita dalam $\mathcal{F}$.
Sebagai contoh,
diberikan sebuah dataset dengan fitur $\mathbf{X}$
dan label $\mathbf{y}$,
kita mungkin mencoba menemukannya dengan menyelesaikan masalah optimisasi berikut:

$$f^*_\mathcal{F} \stackrel{\textrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \textrm{ subject to } f \in \mathcal{F}.$$

Kita tahu bahwa regularisasi :cite:`tikhonov1977solutions,morozov2012methods` dapat mengontrol kompleksitas $\mathcal{F}$
dan mencapai konsistensi, sehingga ukuran data pelatihan yang lebih besar
umumnya mengarah pada $f^*_\mathcal{F}$ yang lebih baik.
Adalah wajar untuk berasumsi bahwa jika kita merancang arsitektur lain yang lebih kuat $\mathcal{F}'$, kita seharusnya mendapatkan hasil yang lebih baik. Dengan kata lain, kita mengharapkan bahwa $f^*_{\mathcal{F}'}$ lebih "baik" daripada $f^*_{\mathcal{F}}$. Namun, jika $\mathcal{F} \not\subseteq \mathcal{F}'$, tidak ada jaminan bahwa hal ini akan terjadi. Bahkan, $f^*_{\mathcal{F}'}$ mungkin justru lebih buruk.
Sebagaimana diilustrasikan pada :numref:`fig_functionclasses`,
untuk kelas fungsi yang tidak bersarang, kelas fungsi yang lebih besar tidak selalu bergerak mendekati fungsi "kebenaran" $f^*$. Sebagai contoh,
di sebelah kiri dari :numref:`fig_functionclasses`,
meskipun $\mathcal{F}_3$ lebih dekat ke $f^*$ daripada $\mathcal{F}_1$, $\mathcal{F}_6$ bergerak menjauh dan tidak ada jaminan bahwa peningkatan kompleksitas dapat mengurangi jarak dari $f^*$.
Dengan kelas fungsi bersarang
di mana $\mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_6$
di sebelah kanan :numref:`fig_functionclasses`,
kita dapat menghindari masalah yang disebutkan sebelumnya dari kelas fungsi yang tidak bersarang.


![Untuk kelas fungsi yang tidak bersarang, kelas fungsi yang lebih besar (ditunjukkan dengan area) tidak menjamin kita akan semakin dekat ke fungsi "kebenaran" ($\mathit{f}^*$). Hal ini tidak terjadi pada kelas fungsi yang bersarang.](../img/functionclasses.svg)
:label:`fig_functionclasses`

Oleh karena itu,
hanya jika kelas fungsi yang lebih besar mengandung yang lebih kecil kita dapat dijamin bahwa peningkatan tersebut secara ketat meningkatkan kekuatan ekspresif jaringan.
Untuk jaringan neural dalam,
jika kita bisa
melatih lapisan yang baru ditambahkan menjadi fungsi identitas $f(\mathbf{x}) = \mathbf{x}$, model baru akan seefektif model asli. Karena model baru mungkin mendapatkan solusi yang lebih baik untuk menyesuaikan dataset pelatihan, lapisan yang ditambahkan mungkin mempermudah pengurangan error pelatihan.

Inilah pertanyaan yang dipertimbangkan oleh :citet:`He.Zhang.Ren.ea.2016` saat mengerjakan model visi komputer yang sangat dalam.
Inti dari *jaringan residual* (*ResNet*) yang mereka usulkan adalah gagasan bahwa setiap lapisan tambahan seharusnya
lebih mudah
mengandung fungsi identitas sebagai salah satu elemennya.
Pertimbangan ini cukup mendalam tetapi mengarah pada solusi yang mengejutkan sederhana, yaitu *blok residual*.
Dengan blok ini, ResNet memenangkan ImageNet Large Scale Visual Recognition Challenge pada tahun 2015. Desain ini memiliki pengaruh mendalam pada cara
membangun jaringan neural yang dalam. Sebagai contoh, blok residual telah ditambahkan pada jaringan berulang (recurrent networks) :cite:`prakash2016neural,kim2017residual`. Demikian juga, Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017` menggunakannya untuk menumpuk banyak lapisan jaringan secara efisien. Blok ini juga digunakan dalam jaringan neural graf :cite:`Kipf.Welling.2016` dan, sebagai konsep dasar, telah digunakan secara luas dalam visi komputer :cite:`Redmon.Farhadi.2018,Ren.He.Girshick.ea.2015`.
Perlu dicatat bahwa jaringan residual didahului oleh jaringan highway :cite:`srivastava2015highway` yang berbagi beberapa motivasi yang serupa, meskipun tanpa parametrisasi yang elegan di sekitar fungsi identitas.


## (**Blok Residual**)
:label:`subsec_residual-blks`

Mari kita fokus pada bagian lokal dari jaringan neural, seperti yang digambarkan pada :numref:`fig_residual_block`. Misalkan inputnya adalah $\mathbf{x}$.
Kita mengasumsikan bahwa $f(\mathbf{x})$, pemetaan dasar yang ingin kita peroleh melalui pembelajaran, akan digunakan sebagai input ke fungsi aktivasi di bagian atas.
Di sebelah kiri,
bagian dalam kotak garis putus-putus
harus langsung mempelajari $f(\mathbf{x})$.
Di sebelah kanan,
bagian dalam kotak garis putus-putus
perlu
mempelajari *pemetaan residual* $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$,
inilah asal nama blok residual.
Jika pemetaan identitas $f(\mathbf{x}) = \mathbf{x}$ adalah pemetaan dasar yang diinginkan,
pemetaan residual menjadi $g(\mathbf{x}) = 0$, sehingga lebih mudah dipelajari:
kita hanya perlu mendorong bobot dan bias
dari
lapisan bobot atas (misalnya, fully connected layer dan lapisan konvolusi)
dalam kotak garis putus-putus
ke nol.
Gambar di sebelah kanan menggambarkan *blok residual* dari ResNet,
di mana garis solid yang membawa input lapisan
$\mathbf{x}$ ke operator penjumlahan
disebut *koneksi residual* (atau *koneksi pintas*).
Dengan blok residual, input dapat
dipropagasi ke depan lebih cepat melalui koneksi residual antar lapisan.
Faktanya,
blok residual
dapat dianggap sebagai
kasus khusus dari blok multi-cabang Inception:
blok ini memiliki dua cabang
yang salah satunya adalah pemetaan identitas.

![Dalam blok biasa (kiri), bagian dalam kotak garis putus-putus harus langsung mempelajari pemetaan $\mathit{f}(\mathbf{x})$. Dalam blok residual (kanan), bagian dalam kotak garis putus-putus perlu mempelajari pemetaan residual $\mathit{g}(\mathbf{x}) = \mathit{f}(\mathbf{x}) - \mathbf{x}$, membuat pemetaan identitas $\mathit{f}(\mathbf{x}) = \mathbf{x}$ lebih mudah dipelajari.](../img/residual-block.svg)
:label:`fig_residual_block`


ResNet menggunakan desain lapisan konvolusi penuh $3\times 3$ dari VGG. Blok residual memiliki dua lapisan konvolusi $3\times 3$ dengan jumlah channel output yang sama. Setiap lapisan konvolusi diikuti oleh lapisan batch normalization dan fungsi aktivasi ReLU. Kemudian, kita melewati dua operasi konvolusi ini dan menambahkan input secara langsung sebelum fungsi aktivasi ReLU terakhir.
Desain semacam ini mengharuskan output dari dua lapisan konvolusi memiliki bentuk yang sama dengan input, sehingga mereka dapat dijumlahkan bersama. Jika kita ingin mengubah jumlah channel, kita perlu memperkenalkan lapisan konvolusi tambahan $1\times 1$ untuk mentransformasi input ke bentuk yang diinginkan untuk operasi penjumlahan. Mari kita lihat kode di bawah ini.


```{.python .input}
%%tab mxnet
class Residual(nn.Block):  #@save
    """Blok Residual untuk model ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class Residual(nn.Module):  #@save
    """Blok Residual untuk model ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
%%tab tensorflow
class Residual(tf.keras.Model):  #@save
    """Blok Residual untuk model ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

```{.python .input}
%%tab jax
class Residual(nn.Module):  #@save
    """Blok Residual untuk model ResNet."""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        if self.use_1x1conv:
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)
```

Kode ini menghasilkan dua jenis jaringan: satu di mana kita menambahkan input ke output sebelum menerapkan nonlinearitas ReLU ketika `use_1x1conv=False`; dan satu lagi di mana kita menyesuaikan channel dan resolusi menggunakan konvolusi $1 \times 1$ sebelum melakukan penjumlahan. :numref:`fig_resnet_block` menggambarkan hal ini.

![Blok ResNet dengan dan tanpa konvolusi $1 \times 1$, yang mengubah input ke dalam bentuk yang diinginkan untuk operasi penjumlahan.](../img/resnet-block.svg)
:label:`fig_resnet_block`

Sekarang mari kita lihat [**situasi di mana input dan output memiliki bentuk yang sama**], di mana konvolusi $1 \times 1$ tidak diperlukan.


```{.python .input}
%%tab mxnet, pytorch
if tab.selected('mxnet'):
    blk = Residual(3)
    blk.initialize()
if tab.selected('pytorch'):
    blk = Residual(3)
X = d2l.randn(4, 3, 6, 6)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = Residual(3)
X = d2l.normal((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = Residual(3)
X = jax.random.normal(d2l.get_key(), (4, 6, 6, 3))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

Kita juga memiliki opsi untuk [**mengurangi setengah tinggi dan lebar output sambil meningkatkan jumlah channel output**].
Dalam kasus ini, kita menggunakan konvolusi $1 \times 1$ dengan `use_1x1conv=True`. Hal ini berguna di awal setiap blok ResNet untuk mengurangi dimensi spasial menggunakan `strides=2`.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
if tab.selected('mxnet'):
    blk.initialize()
blk(X).shape
```

```{.python .input}
%%tab jax
blk = Residual(6, use_1x1conv=True, strides=(2, 2))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## [**Model ResNet**]

Dua lapisan pertama dari ResNet sama seperti pada GoogLeNet yang telah kita bahas sebelumnya: lapisan konvolusi $7\times 7$ dengan 64 channel output dan stride 2 diikuti oleh lapisan max-pooling $3\times 3$ dengan stride 2. Perbedaannya adalah adanya lapisan batch normalization yang ditambahkan setelah setiap lapisan konvolusi pada ResNet.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet(d2l.Classifier):
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
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                       padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class ResNet(d2l.Classifier):
    arch: tuple
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                  padding='same')])
```

GoogLeNet menggunakan empat modul yang terdiri dari blok Inception.

Namun, ResNet menggunakan empat modul yang terdiri dari blok residual, di mana masing-masing modul menggunakan beberapa blok residual dengan jumlah channel output yang sama.
Jumlah channel pada modul pertama sama dengan jumlah channel input. Karena lapisan max-pooling dengan stride 2 sudah digunakan, tidak perlu lagi mengurangi tinggi dan lebar. 
Pada blok residual pertama untuk setiap modul berikutnya, jumlah channel digandakan dibandingkan dengan modul sebelumnya, dan tinggi serta lebar dikurangi setengah.


```{.python .input}
%%tab mxnet
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = tf.keras.models.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab jax
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True,
                                strides=(2, 2), training=self.training))
        else:
            blk.append(Residual(num_channels, training=self.training))
    return nn.Sequential(blk)
```

Selanjutnya, kita menambahkan semua modul ke dalam ResNet. Di sini, dua blok residual digunakan untuk setiap modul. Terakhir, seperti pada GoogLeNet, kita menambahkan lapisan global average pooling, diikuti dengan lapisan fully connected sebagai output.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
# %%tab jax
@d2l.add_to_class(ResNet)
def create_net(self):
    net = nn.Sequential([self.b1()])
    for i, b in enumerate(self.arch):
        net.layers.extend([self.block(*b, first_block=(i==0))])
    net.layers.extend([nn.Sequential([
        # Flax does not provide a GlobalAvg2D layer
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

Terdapat empat lapisan konvolusi dalam setiap modul (tidak termasuk lapisan konvolusi $1\times 1$). Bersama dengan lapisan konvolusi pertama $7\times 7$ dan lapisan fully connected terakhir, terdapat total 18 lapisan. Oleh karena itu, model ini umumnya dikenal sebagai ResNet-18.
Dengan mengonfigurasi jumlah channel dan blok residual yang berbeda dalam modul, kita dapat menciptakan model ResNet yang berbeda, seperti ResNet-152 yang lebih dalam dengan 152 lapisan. Meskipun arsitektur utama ResNet mirip dengan GoogLeNet, struktur ResNet lebih sederhana dan mudah dimodifikasi. Semua faktor ini membuat ResNet digunakan secara cepat dan luas. :numref:`fig_resnet18` menunjukkan keseluruhan arsitektur ResNet-18.

![Arsitektur ResNet-18.](../img/resnet18-90.svg)
:label:`fig_resnet18`

Sebelum melatih ResNet, mari [**amati bagaimana bentuk input berubah di berbagai modul dalam ResNet**]. Seperti pada semua arsitektur sebelumnya, resolusi menurun sementara jumlah channel meningkat hingga titik di mana lapisan global average pooling menggabungkan semua fitur.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)
```

```{.python .input}
%%tab jax
class ResNet18(ResNet):
    arch: tuple = ((2, 64), (2, 128), (2, 256), (2, 512))
    lr: float = 0.1
    num_classes: int = 10
```

```{.python .input}
%%tab pytorch, mxnet
ResNet18().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
ResNet18().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
ResNet18(training=False).layer_summary((1, 96, 96, 1))
```

## [**Pelatihan**]

Kita melatih ResNet pada dataset Fashion-MNIST, seperti sebelumnya. ResNet adalah arsitektur yang cukup kuat dan fleksibel. Grafik yang menunjukkan loss pelatihan dan validasi memperlihatkan adanya perbedaan 
signifikan antara kedua grafik tersebut, dengan loss pelatihan yang jauh lebih rendah. Untuk jaringan dengan fleksibilitas seperti ini, data pelatihan yang lebih banyak akan memberikan manfaat yang jelas dalam 
menutup celah ini dan meningkatkan akurasi.


```{.python .input}
%%tab mxnet, pytorch, jax
model = ResNet18(lr=0.01)
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
    model = ResNet18(lr=0.01)
    trainer.fit(model, data)
```

## ResNeXt
:label:`subsec_resnext`

Salah satu tantangan yang ditemui dalam desain ResNet adalah trade-off antara nonlinearitas dan dimensi dalam sebuah blok. Artinya, kita dapat menambahkan lebih banyak nonlinearitas dengan meningkatkan jumlah lapisan, atau dengan meningkatkan lebar konvolusi. Strategi alternatif adalah meningkatkan jumlah channel yang dapat membawa informasi antar blok. Sayangnya, pendekatan ini memiliki penalti kuadratik karena biaya komputasi untuk memasukkan $c_\textrm{i}$ channel dan menghasilkan $c_\textrm{o}$ channel sebanding dengan $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$ (lihat diskusi kita pada :numref:`sec_channels`).

Kita dapat mengambil inspirasi dari blok Inception pada :numref:`fig_inception` yang mengalirkan informasi melalui blok dalam grup-grup terpisah. Menerapkan ide dari beberapa grup independen ke dalam blok ResNet pada :numref:`fig_resnet_block` menghasilkan desain ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`.
Berbeda dengan beragam transformasi pada Inception,
ResNeXt mengadopsi transformasi *yang sama* pada semua cabang,
sehingga meminimalkan kebutuhan penyetelan manual untuk setiap cabang.

![Blok ResNeXt. Penggunaan grouped convolution dengan $\mathit{g}$ grup lebih cepat $\mathit{g}$ kali daripada konvolusi padat. Ini adalah blok residual bottleneck ketika jumlah channel intermediate $\mathit{b}$ lebih kecil daripada $\mathit{c}$.](../img/resnext-block.svg)
:label:`fig_resnext_block`

Memecah sebuah konvolusi dari $c_\textrm{i}$ menjadi $c_\textrm{o}$ channel menjadi $g$ grup berukuran $c_\textrm{i}/g$ yang menghasilkan $g$ keluaran berukuran $c_\textrm{o}/g$ disebut *grouped convolution*. Biaya komputasi (secara proporsional) berkurang dari $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$ menjadi $\mathcal{O}(g \cdot (c_\textrm{i}/g) \cdot (c_\textrm{o}/g)) = \mathcal{O}(c_\textrm{i} \cdot c_\textrm{o} / g)$, yaitu, lebih cepat $g$ kali. Lebih baik lagi, jumlah parameter yang dibutuhkan untuk menghasilkan keluaran juga berkurang dari matriks $c_\textrm{i} \times c_\textrm{o}$ menjadi $g$ matriks yang lebih kecil dengan ukuran $(c_\textrm{i}/g) \times (c_\textrm{o}/g)$, kembali mengurangi kebutuhan $g$ kali. Berikutnya, kita mengasumsikan bahwa $c_\textrm{i}$ dan $c_\textrm{o}$ keduanya dapat dibagi oleh $g$.

Satu-satunya tantangan dalam desain ini adalah tidak ada pertukaran informasi di antara $g$ grup. Blok ResNeXt pada
:numref:`fig_resnext_block` mengatasi hal ini dengan dua cara: grouped convolution dengan kernel $3 \times 3$ diapit di antara dua konvolusi $1 \times 1$. Konvolusi kedua berfungsi ganda untuk mengembalikan jumlah channel. Keuntungannya adalah kita hanya membayar biaya $\mathcal{O}(c \cdot b)$ untuk kernel $1 \times 1$ dan bisa bertahan dengan biaya $\mathcal{O}(b^2 / g)$ untuk kernel $3 \times 3$. Mirip dengan implementasi blok residual pada
:numref:`subsec_residual-blks`, koneksi residual digantikan (dan dengan demikian digeneralisasi) oleh konvolusi $1 \times 1$.

Gambar sebelah kanan pada :numref:`fig_resnext_block` memberikan ringkasan yang jauh lebih ringkas dari blok jaringan yang dihasilkan. Blok ini juga akan memainkan peran utama dalam desain CNN modern secara umum pada :numref:`sec_cnn-design`. Perlu dicatat bahwa ide grouped convolutions berasal dari implementasi AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`. Ketika mendistribusikan jaringan ke dua GPU dengan memori terbatas, implementasi memperlakukan masing-masing GPU sebagai channel sendiri tanpa efek samping.

Implementasi berikut dari kelas `ResNeXtBlock` mengambil argumen `groups` ($g$), dengan
`bot_channels` ($b$) sebagai channel intermediate (bottleneck). Terakhir, ketika kita perlu mengurangi tinggi dan lebar representasi, kita menambahkan stride sebesar $2$ dengan mengatur `use_1x1conv=True, strides=2`.


```{.python .input}
%%tab mxnet
class ResNeXtBlock(nn.Block):  #@save
    """Blok ResNeXt"""
    def __init__(self, num_channels, groups, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1, 
                               strides=strides, groups=bot_channels//groups)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class ResNeXtBlock(nn.Module):  #@save
    """Blok ResNeXt"""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, 
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
```

```{.python .input}
%%tab tensorflow
class ResNeXtBlock(tf.keras.Model):  #@save
    """Blok ResNeXt"""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1)
        self.conv2 = tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                            padding="same",
                                            groups=bot_channels//groups)
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                                                strides=strides)
            self.bn4 = tf.keras.layers.BatchNormalization()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return tf.keras.activations.relu(Y + X)
```

```{.python .input}
%%tab jax
class ResNeXtBlock(nn.Module):  #@save
    """Blok ResNeXt"""
    num_channels: int
    groups: int
    bot_mul: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        bot_channels = int(round(self.num_channels * self.bot_mul))
        self.conv1 = nn.Conv(bot_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.conv2 = nn.Conv(bot_channels, kernel_size=(3, 3),
                               strides=self.strides, padding='same',
                               feature_group_count=bot_channels//self.groups)
        self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)
        self.bn3 = nn.BatchNorm(not self.training)
        if self.use_1x1conv:
            self.conv4 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                       strides=self.strides)
            self.bn4 = nn.BatchNorm(not self.training)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)
```

Penggunaannya sepenuhnya mirip dengan `ResNetBlock` yang telah dibahas sebelumnya. Misalnya, ketika menggunakan (`use_1x1conv=False, strides=1`), bentuk input dan output adalah sama. 
Sebaliknya, dengan mengatur `use_1x1conv=True, strides=2`, tinggi dan lebar output berkurang setengah.

```{.python .input}
%%tab mxnet, pytorch
blk = ResNeXtBlock(32, 16, 1)
if tab.selected('mxnet'):
    blk.initialize()
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = ResNeXtBlock(32, 16, 1)
X = d2l.normal((4, 96, 96, 32))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = ResNeXtBlock(32, 16, 1)
X = jnp.zeros((4, 96, 96, 32))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## Ringkasan dan Diskusi

Kelas fungsi bersarang sangat diinginkan karena memungkinkan kita untuk memperoleh kelas fungsi yang *lebih kuat* secara ketat daripada sekadar *berbeda* saat menambah kapasitas. Salah satu cara untuk mencapainya adalah dengan memungkinkan lapisan tambahan hanya meneruskan input ke output. Koneksi residual memungkinkan hal ini. Akibatnya, ini mengubah bias induktif dari fungsi sederhana berbentuk $f(\mathbf{x}) = 0$ menjadi fungsi sederhana yang tampak seperti $f(\mathbf{x}) = \mathbf{x}$.

Pemetaan residual dapat mempelajari fungsi identitas dengan lebih mudah, seperti mendorong parameter dalam lapisan bobot ke nol. Kita dapat melatih jaringan neural *dalam* yang efektif dengan memiliki blok residual. Input dapat dipropagasi lebih cepat melalui koneksi residual antar lapisan. Akibatnya, kita dapat melatih jaringan yang jauh lebih dalam. Misalnya, makalah asli ResNet :cite:`He.Zhang.Ren.ea.2016` memungkinkan hingga 152 lapisan. Manfaat lain dari jaringan residual adalah memungkinkan kita untuk menambah lapisan, yang diinisialisasi sebagai fungsi identitas, *selama* proses pelatihan. Bagaimanapun, perilaku default suatu lapisan adalah membiarkan data melewati tanpa perubahan. Ini dapat mempercepat pelatihan jaringan yang sangat besar dalam beberapa kasus.

Sebelum koneksi residual,
jalur pintas dengan unit pengatur diperkenalkan
untuk melatih jaringan highway dengan lebih dari 100 lapisan secara efektif
:cite:`srivastava2015highway`.
Dengan menggunakan fungsi identitas sebagai jalur pintas,
ResNet berkinerja luar biasa
dalam berbagai tugas visi komputer.
Koneksi residual memiliki pengaruh besar pada desain jaringan neural dalam berikutnya, baik yang bersifat konvolusi maupun sekuensial.
Seperti yang akan kita bahas kemudian,
arsitektur Transformer :cite:`Vaswani.Shazeer.Parmar.ea.2017`
mengadopsi koneksi residual (bersama dengan pilihan desain lainnya) dan digunakan secara luas
di bidang yang beragam seperti
bahasa, visi, suara, dan pembelajaran penguatan.

ResNeXt adalah contoh bagaimana desain jaringan neural konvolusional berkembang seiring waktu: dengan lebih hemat dalam penggunaan komputasi dan menukarnya dengan ukuran aktivasi (jumlah channel), ResNeXt memungkinkan jaringan yang lebih cepat dan lebih akurat dengan biaya lebih rendah. Cara alternatif melihat grouped convolution adalah menganggapnya sebagai matriks blok-diagonal untuk bobot konvolusi. Perhatikan bahwa ada beberapa "trik" seperti ini yang menghasilkan jaringan yang lebih efisien. Misalnya, ShiftNet :cite:`wu2018shift` meniru efek konvolusi $3 \times 3$, hanya dengan menambahkan aktivasi yang digeser ke channel, memberikan peningkatan kompleksitas fungsi, kali ini tanpa biaya komputasi.

Satu fitur umum dari desain yang telah kita bahas sejauh ini adalah bahwa desain jaringan cukup manual, yang sebagian besar mengandalkan kecerdikan perancang untuk menemukan hyperparameter jaringan yang "tepat". Meskipun jelas dapat dilakukan, hal ini juga sangat mahal dalam hal waktu manusia dan tidak ada jaminan bahwa hasil akhirnya optimal dalam pengertian apa pun. Pada :numref:`sec_cnn-design` kita akan membahas sejumlah strategi untuk memperoleh jaringan berkualitas tinggi secara lebih otomatis. Secara khusus, kita akan meninjau konsep *ruang desain jaringan* yang mengarah ke model RegNetX/Y :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.

## Latihan

1. Apa perbedaan utama antara blok Inception pada :numref:`fig_inception` dan blok residual? Bagaimana perbandingannya dalam hal komputasi, akurasi, dan kelas fungsi yang dapat mereka gambarkan?
2. Rujuk ke Tabel 1 dalam makalah ResNet :cite:`He.Zhang.Ren.ea.2016` untuk mengimplementasikan berbagai varian jaringan.
3. Untuk jaringan yang lebih dalam, ResNet memperkenalkan arsitektur "bottleneck" untuk mengurangi kompleksitas model. Coba implementasikan arsitektur ini.
4. Pada versi ResNet berikutnya, penulis mengubah struktur "konvolusi, batch normalization, dan aktivasi" menjadi struktur "batch normalization, aktivasi, dan konvolusi". Lakukan peningkatan ini sendiri. Lihat Gambar 1 dalam :citet:`He.Zhang.Ren.ea.2016*1` untuk detailnya.
5. Mengapa kita tidak bisa begitu saja meningkatkan kompleksitas fungsi tanpa batas, bahkan jika kelas fungsi bersarang?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/8737)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18006)
:end_tab:
