```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Convolutional Neural Networks (LeNet)
:label:`sec_lenet`

Sekarang kita memiliki semua komponen yang diperlukan untuk merangkai
sebuah CNN yang berfungsi penuh.
Dalam pertemuan kita sebelumnya dengan data gambar, kita menerapkan
model linear dengan regresi softmax (:numref:`sec_softmax_scratch`)
dan MLP (:numref:`sec_mlp-implementation`)
untuk gambar pakaian dalam dataset Fashion-MNIST.
Untuk membuat data tersebut dapat diproses, kita pertama-tama meratakan (flatten) setiap gambar dari matriks $28\times28$
menjadi vektor berdimensi tetap $784$,
dan kemudian memprosesnya dalam lapisan fully connected.
Sekarang, dengan adanya pemahaman tentang lapisan konvolusi,
kita dapat mempertahankan struktur spasial dalam gambar kita.
Sebagai keuntungan tambahan dari menggantikan lapisan fully connected dengan lapisan konvolusi,
kita akan mendapatkan model yang lebih hemat yang memerlukan jauh lebih sedikit parameter.

Di bagian ini, kita akan memperkenalkan *LeNet*,
salah satu CNN pertama yang dipublikasikan
dan mendapat perhatian luas berkat performanya dalam tugas-tugas penglihatan komputer.
Model ini diperkenalkan oleh (dan dinamai untuk) Yann LeCun,
yang saat itu merupakan peneliti di AT&T Bell Labs,
untuk tujuan mengenali digit tulisan tangan dalam gambar :cite:`LeCun.Bottou.Bengio.ea.1998`.
Karya ini merupakan puncak dari satu dekade penelitian dalam pengembangan teknologi tersebut;
tim LeCun menerbitkan studi pertama yang berhasil
melatih CNN melalui backpropagation :cite:`LeCun.Boser.Denker.ea.1989`.

Pada masanya, LeNet mencapai hasil luar biasa
yang menyaingi kinerja support vector machines,
yang saat itu merupakan pendekatan dominan dalam pembelajaran terawasi, dengan tingkat kesalahan kurang dari 1% per digit.
LeNet akhirnya diadaptasi untuk mengenali digit
untuk memproses setoran di mesin ATM.
Hingga saat ini, beberapa ATM masih menjalankan kode
yang ditulis oleh Yann LeCun dan rekannya Leon Bottou pada tahun 1990-an!


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
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
from types import FunctionType
```

## LeNet

Secara umum, (**LeNet (LeNet-5) terdiri dari dua bagian:
(i) sebuah encoder konvolusi yang terdiri dari dua lapisan konvolusi; dan
(ii) sebuah blok dense yang terdiri dari tiga lapisan fully connected**).
Arsitektur ini dirangkum dalam :numref:`img_lenet`.

![Aliran data dalam LeNet. Inputnya adalah digit tulisan tangan, outputnya adalah probabilitas untuk 10 kemungkinan hasil.](../img/lenet.svg)
:label:`img_lenet`

Unit dasar dalam setiap blok konvolusi
adalah lapisan konvolusi, fungsi aktivasi sigmoid,
dan operasi average pooling berikutnya.
Perhatikan bahwa meskipun ReLU dan max-pooling bekerja lebih baik,
mereka belum ditemukan saat itu.
Setiap lapisan konvolusi menggunakan kernel $5\times 5$
dan fungsi aktivasi sigmoid.
Lapisan-lapisan ini memetakan input yang diatur secara spasial
ke sejumlah feature map dua dimensi, biasanya
meningkatkan jumlah channel.
Lapisan konvolusi pertama memiliki 6 output channel,
sedangkan lapisan kedua memiliki 16.
Setiap operasi pooling $2\times2$ (stride 2)
mengurangi dimensi dengan faktor $4$ melalui downsampling spasial.
Blok konvolusi menghasilkan output dengan bentuk
(ukuran batch, jumlah channel, tinggi, lebar).

Agar output dari blok konvolusi dapat diteruskan
ke blok dense,
kita harus meratakan (flatten) setiap contoh dalam minibatch.
Dengan kata lain, kita mengubah input empat dimensi ini menjadi
input dua dimensi yang diharapkan oleh lapisan fully connected:
sebagai pengingat, representasi dua dimensi yang kita inginkan menggunakan dimensi pertama untuk mengindeks contoh dalam minibatch
dan dimensi kedua untuk memberikan representasi vektor datar dari setiap contoh.
Blok dense LeNet memiliki tiga lapisan fully connected,
dengan masing-masing 120, 84, dan 10 output.
Karena kita masih melakukan klasifikasi,
lapisan output 10 dimensi ini berkorespondensi
dengan jumlah kelas output yang mungkin.

Meskipun mungkin butuh usaha untuk benar-benar memahami
apa yang terjadi di dalam LeNet,
kami harap cuplikan kode berikut akan meyakinkan Anda
bahwa mengimplementasikan model seperti itu dengan kerangka kerja deep learning modern
sangatlah sederhana.
Kita hanya perlu membuat blok `Sequential`
dan menggabungkan lapisan-lapisan yang sesuai,
menggunakan inisialisasi Xavier seperti yang
diperkenalkan di :numref:`subsec_xavier`.


```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """Menginisialisasi bobot untuk CNN."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LeNet(d2l.Classifier):  #@save
    """Model LeNet-5."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """LeNet-5 model."""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])
```

Kami telah melakukan beberapa modifikasi dalam reproduksi LeNet sejauh ini, di mana kami menggantikan lapisan aktivasi Gaussian dengan lapisan softmax. Ini sangat menyederhanakan implementasi, 
terutama karena decoder Gaussian jarang digunakan saat ini. Selain itu, jaringan ini masih sesuai dengan arsitektur asli LeNet-5.

:begin_tab:`pytorch, mxnet, tensorflow`
Mari kita lihat apa yang terjadi di dalam jaringan. Dengan melewatkan gambar satu channel (hitam-putih) berukuran
$28 \times 28$ melalui jaringan dan mencetak bentuk output di setiap lapisan, kita dapat [**memeriksa model**] untuk memastikan bahwa operasinya sesuai dengan yang kita harapkan dari :numref:`img_lenet_vert`.
:end_tab:

:begin_tab:`jax`
Mari kita lihat apa yang terjadi di dalam jaringan. Dengan melewatkan gambar satu channel (hitam-putih) berukuran
$28 \times 28$ melalui jaringan dan mencetak bentuk output di setiap lapisan, kita dapat [**memeriksa model**] untuk memastikan bahwa operasinya sesuai dengan yang kita harapkan dari :numref:`img_lenet_vert`.
Flax menyediakan `nn.tabulate`, sebuah metode berguna untuk merangkum lapisan dan parameter dalam jaringan kita. Di sini kita menggunakan metode `bind` untuk membuat model terikat.
Variabel-variabel sekarang terikat pada kelas `d2l.Module`, yaitu, model terikat ini menjadi objek stateful yang 
dapat digunakan untuk mengakses atribut objek `Sequential` berupa `net` dan `layers` di dalamnya.
Perlu dicatat bahwa metode `bind` hanya sebaiknya digunakan untuk eksperimen interaktif, dan bukan merupakan pengganti langsung untuk metode `apply`.
:end_tab:


![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape, key=d2l.get_key()):
    X = jnp.zeros(X_shape)
    params = self.init(key, X)
    bound_model = self.clone().bind(params, mutable=['batch_stats'])
    _ = bound_model(X)
    for layer in bound_model.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

Perhatikan bahwa tinggi dan lebar representasi
pada setiap lapisan di sepanjang blok konvolusi
mengalami pengurangan (dibandingkan dengan lapisan sebelumnya).
Lapisan konvolusi pertama menggunakan padding dua piksel
untuk mengompensasi pengurangan tinggi dan lebar
yang akan terjadi jika menggunakan kernel $5 \times 5$ tanpa padding.
Sebagai informasi tambahan, ukuran gambar $28 \times 28$ piksel pada dataset MNIST OCR asli merupakan hasil *pemotongan* dua baris (dan kolom) piksel dari
pemindaian asli yang berukuran $32 \times 32$ piksel. Hal ini dilakukan terutama untuk
menghemat ruang (pengurangan sekitar 30%) pada masa di mana penyimpanan dalam megabyte sangat diperhitungkan.

Sebaliknya, lapisan konvolusi kedua tidak menggunakan padding,
sehingga tinggi dan lebar keduanya berkurang sebanyak empat piksel.
Seiring dengan bertambahnya lapisan,
jumlah channel meningkat lapis demi lapis
dari 1 pada input menjadi 6 setelah lapisan konvolusi pertama
dan 16 setelah lapisan konvolusi kedua.
Namun, setiap lapisan pooling mengurangi tinggi dan lebar setengahnya.
Terakhir, setiap lapisan fully connected mengurangi dimensi,
hingga akhirnya menghasilkan output dengan dimensi
yang sesuai dengan jumlah kelas.

## Pelatihan

Sekarang setelah kita mengimplementasikan modelnya,
mari kita [**jalankan eksperimen untuk melihat bagaimana performa model LeNet-5 pada Fashion-MNIST**].

Meskipun CNN memiliki lebih sedikit parameter,
mereka bisa tetap lebih mahal secara komputasi
dibandingkan dengan MLP yang kedalamannya serupa
karena setiap parameter berpartisipasi dalam lebih banyak
perkalian.
Jika Anda memiliki akses ke GPU, ini bisa menjadi waktu yang tepat
untuk menggunakannya guna mempercepat pelatihan.
Perlu diperhatikan bahwa
kelas `d2l.Trainer` menangani semua detail yang diperlukan.
Secara default, kelas ini menginisialisasi parameter model pada perangkat yang tersedia.
Seperti halnya dengan MLP, fungsi loss kita adalah cross-entropy,
dan kita meminimalkannya menggunakan stochastic gradient descent minibatch.


```{.python .input}
%%tab pytorch, mxnet, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## Ringkasan

Kita telah membuat kemajuan yang signifikan dalam bab ini. Kita beralih dari MLP pada tahun 1980-an ke CNN pada tahun 1990-an dan awal 2000-an. Arsitektur yang diusulkan, misalnya, dalam bentuk LeNet-5, tetap relevan hingga saat ini. Penting untuk membandingkan tingkat kesalahan pada Fashion-MNIST yang dapat dicapai dengan LeNet-5, baik dengan model MLP terbaik (:numref:`sec_mlp-implementation`) maupun dengan arsitektur yang jauh lebih canggih seperti ResNet (:numref:`sec_resnet`). LeNet lebih mirip dengan arsitektur lanjutan daripada dengan MLP. Salah satu perbedaan utama, seperti yang akan kita lihat, adalah bahwa dengan peningkatan komputasi, memungkinkan adanya arsitektur yang jauh lebih kompleks.

Perbedaan kedua adalah kemudahan relatif dalam mengimplementasikan LeNet. Dahulu, ini adalah tantangan rekayasa yang memerlukan berbulan-bulan pemrograman dalam C++ dan kode assembly, mengembangkan SN, alat deep learning berbasis Lisp awal :cite:`Bottou.Le-Cun.1988`, dan akhirnya melakukan eksperimen dengan model. Kini, semua itu dapat dilakukan dalam hitungan menit. Peningkatan produktivitas yang luar biasa ini telah sangat mendemokratisasi pengembangan model deep learning. Dalam bab berikutnya, kita akan melanjutkan perjalanan ini untuk melihat ke mana arah selanjutnya.

## Latihan

1. Mari modernisasi LeNet. Implementasikan dan uji perubahan berikut:
    1. Ganti average pooling dengan max-pooling.
    1. Ganti lapisan softmax dengan ReLU.
1. Cobalah untuk mengubah ukuran jaringan gaya LeNet untuk meningkatkan akurasinya, selain menggunakan max-pooling dan ReLU.
    1. Sesuaikan ukuran jendela konvolusi.
    1. Sesuaikan jumlah output channel.
    1. Sesuaikan jumlah lapisan konvolusi.
    1. Sesuaikan jumlah lapisan fully connected.
    1. Sesuaikan laju pembelajaran (learning rate) dan detail pelatihan lainnya (misalnya, inisialisasi dan jumlah epoch).
1. Uji jaringan yang telah ditingkatkan pada dataset MNIST asli.
1. Tampilkan aktivasi dari lapisan pertama dan kedua LeNet untuk input yang berbeda (misalnya, sweater dan jaket).
1. Apa yang terjadi pada aktivasi ketika Anda memasukkan gambar yang sangat berbeda ke dalam jaringan (misalnya, kucing, mobil, atau bahkan noise acak)?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/275)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18000)
:end_tab:
