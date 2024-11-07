```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dataset Klasifikasi Gambar
:label:`sec_fashion_mnist`

(~~ Dataset MNIST adalah salah satu dataset yang banyak digunakan untuk klasifikasi gambar, namun terlalu sederhana sebagai dataset acuan. Kita akan menggunakan dataset Fashion-MNIST yang serupa, tetapi lebih kompleks ~~)

Salah satu dataset yang banyak digunakan untuk klasifikasi gambar adalah [dataset MNIST](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998` yang berisi gambar digit tulisan tangan. Saat dirilis pada 1990-an, dataset ini menimbulkan tantangan besar bagi sebagian besar algoritma machine learning, yang terdiri dari 60.000 gambar dengan resolusi $28 \times 28$ piksel (ditambah dataset uji dengan 10.000 gambar). Sebagai perspektif, pada tahun 1995, Sun SPARCStation 5 dengan RAM sebesar 64MB dan 5 MFLOPs dianggap sebagai peralatan canggih untuk machine learning di AT&T Bell Laboratories. Mencapai akurasi tinggi dalam pengenalan digit adalah komponen utama dalam otomatisasi pengurutan surat untuk USPS pada 1990-an. Jaringan deep seperti LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`, support vector machines dengan invariansi :cite:`Scholkopf.Burges.Vapnik.1996`, dan tangent distance classifiers :cite:`Simard.LeCun.Denker.ea.1998` semuanya berhasil mencapai tingkat kesalahan di bawah 1%.

Selama lebih dari satu dekade, MNIST berfungsi sebagai *acuan* untuk membandingkan algoritma machine learning. 
Meskipun telah berjalan lama sebagai dataset benchmark,
bahkan model sederhana menurut standar saat ini dapat mencapai akurasi klasifikasi di atas 95%,
menjadikannya tidak cocok untuk membedakan antara model yang kuat dan yang lebih lemah. Selain itu, dataset ini memungkinkan tingkat akurasi yang *sangat* tinggi, yang jarang terlihat dalam banyak masalah klasifikasi. Hal ini membuat perkembangan algoritmik terarah pada keluarga algoritma tertentu yang dapat memanfaatkan dataset yang bersih, seperti metode active set dan algoritma active set yang mencari batas-batas.
Saat ini, MNIST lebih berfungsi sebagai pemeriksaan sederhana daripada benchmark. ImageNet :cite:`Deng.Dong.Socher.ea.2009` menawarkan tantangan yang jauh lebih relevan. Sayangnya, ImageNet terlalu besar untuk banyak contoh dan ilustrasi dalam buku ini, karena akan memakan waktu terlalu lama untuk pelatihan sehingga contoh tidak akan interaktif. Sebagai gantinya, kita akan fokus pada pembahasan di bagian berikutnya menggunakan dataset yang serupa secara kualitatif, tetapi jauh lebih kecil yaitu dataset Fashion-MNIST :cite:`Xiao.Rasul.Vollgraf.2017` yang dirilis pada 2017. Dataset ini berisi gambar 10 kategori pakaian dengan resolusi $28 \times 28$ piksel.


```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## Memuat Dataset

Karena dataset Fashion-MNIST sangat berguna, semua framework utama menyediakan versi yang sudah diproses. 
Kita dapat [**mengunduh dan membacanya ke dalam memori menggunakan utilitas bawaan dari framework.**]


```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """Dataset Fashion-MNIST."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """dataset Fashion-MNIST."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST terdiri dari gambar dari 10 kategori, masing-masing diwakili
oleh 6000 gambar dalam dataset pelatihan dan 1000 gambar dalam dataset uji.
*Dataset uji* digunakan untuk mengevaluasi performa model (dataset ini tidak boleh digunakan untuk pelatihan).
Akibatnya, set pelatihan dan set uji
masing-masing berisi 60.000 dan 10.000 gambar.


```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

Gambar-gambar ini adalah grayscale dan di-upscale menjadi resolusi $32 \times 32$ piksel seperti di atas. 
Hal ini mirip dengan dataset MNIST asli yang terdiri dari gambar hitam putih (biner). Perlu dicatat bahwa sebagian besar data gambar modern memiliki tiga channel (merah, hijau, biru) dan bahwa gambar hiperspektral dapat memiliki lebih dari 100 channel (sensor HyMap memiliki 126 channel).
Secara konvensional, kita menyimpan gambar sebagai tensor $c \times h \times w$, di mana $c$ adalah jumlah channel warna, $h$ adalah tinggi, dan $w$ adalah lebar.


```{.python .input}
%%tab all
data.train[0][0].shape
```

[~~Two utility functions to visualize the dataset~~]

Kategori Fashion-MNIST memiliki nama yang mudah dipahami oleh manusia. 
Metode tambahan berikut ini mengonversi antara label numerik dan nama-nama mereka.



```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Membaca Minibatch

Untuk mempermudah proses membaca dari set pelatihan dan set uji,
kita menggunakan data iterator bawaan daripada membuatnya dari awal.
Ingat bahwa pada setiap iterasi, sebuah data iterator
[**membaca sebuah minibatch data dengan ukuran `batch_size`.**]
Kita juga mengacak contoh-contoh secara acak untuk data iterator pelatihan.


```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

Untuk melihat cara kerjanya, mari kita muat satu minibatch gambar dengan memanggil metode `train_dataloader`. Minibatch ini berisi 64 gambar.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Mari kita lihat waktu yang diperlukan untuk membaca gambar. Meskipun ini adalah loader bawaan, kecepatannya tidak terlalu cepat. 
Meskipun demikian, ini sudah cukup karena memproses gambar dengan jaringan deep membutuhkan waktu yang lebih lama.
Oleh karena itu, ini sudah cukup baik sehingga pelatihan jaringan tidak akan dibatasi oleh I/O.

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Visualisasi

Kita akan sering menggunakan dataset Fashion-MNIST. Sebuah fungsi tambahan `show_images` dapat digunakan untuk memvisualisasikan gambar-gambar beserta label yang terkait. 
Mengabaikan detail implementasi, kita hanya menunjukkan antarmuka di bawah ini: kita hanya perlu mengetahui cara memanggil `d2l.show_images` tanpa harus memahami cara kerjanya
untuk fungsi-fungsi utilitas semacam ini.


```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot  list dari gambar."""
    raise NotImplementedError
```

Mari kita manfaatkan fungsi ini dengan baik. Secara umum, adalah ide yang baik untuk memvisualisasikan dan memeriksa data yang Anda gunakan untuk pelatihan. 
Manusia sangat pandai dalam mendeteksi keanehan dan karena itu, visualisasi berfungsi sebagai pengaman tambahan terhadap kesalahan dan kekeliruan dalam desain eksperimen. Berikut adalah [**gambar-gambar dan label yang terkait**] (dalam bentuk teks)
untuk beberapa contoh pertama dalam dataset pelatihan.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

Sekarang kita siap bekerja dengan dataset Fashion-MNIST di bagian yang akan datang.

## Ringkasan

Kita sekarang memiliki dataset yang sedikit lebih realistis untuk digunakan dalam klasifikasi. Fashion-MNIST adalah dataset klasifikasi pakaian yang terdiri dari gambar yang mewakili 10 kategori. Kita akan menggunakan dataset ini pada bagian dan bab berikutnya untuk mengevaluasi berbagai desain jaringan, mulai dari model linear sederhana hingga jaringan residual yang lebih canggih. Seperti yang biasa kita lakukan dengan gambar, kita membacanya sebagai tensor dengan bentuk (ukuran batch, jumlah channel, tinggi, lebar). Untuk saat ini, kita hanya memiliki satu channel karena gambar adalah grayscale (visualisasi di atas menggunakan palet warna palsu untuk meningkatkan visibilitas).

Terakhir, data iterator adalah komponen kunci untuk kinerja yang efisien. Misalnya, kita mungkin menggunakan GPU untuk dekompresi gambar yang efisien, transcoding video, atau prapemrosesan lainnya. Jika memungkinkan, Anda harus mengandalkan data iterator yang diimplementasikan dengan baik yang memanfaatkan komputasi berkinerja tinggi untuk menghindari memperlambat loop pelatihan Anda.

## Latihan

1. Apakah mengurangi `batch_size` (misalnya, menjadi 1) mempengaruhi kinerja pembacaan?
2. Kinerja data iterator penting. Apakah menurut Anda implementasi saat ini cukup cepat? Jelajahi berbagai opsi untuk meningkatkannya. Gunakan profiler sistem untuk mencari tahu di mana letak bottleneck-nya.
3. Periksa dokumentasi API framework secara online. Dataset lain apa yang tersedia?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17980)
:end_tab:
