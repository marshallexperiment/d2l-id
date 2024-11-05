```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Data Regresi Sintetis
:label:`sec_synthetic-regression-data`

Pembelajaran mesin berkaitan dengan ekstraksi informasi dari data.
Jadi, Anda mungkin bertanya-tanya, apa yang mungkin bisa kita pelajari dari data sintetis?
Meskipun kita mungkin tidak secara intrinsik peduli pada pola 
yang kita sendiri masukkan ke dalam model pembangkitan data buatan,
dataset semacam itu tetap berguna untuk tujuan didaktis,
membantu kita mengevaluasi sifat algoritma pembelajaran kita
dan memastikan bahwa implementasi kita bekerja seperti yang diharapkan.
Sebagai contoh, jika kita membuat data dengan parameter yang benar diketahui *a priori*,
kita dapat memeriksa apakah model kita benar-benar dapat memulihkannya.


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## Menghasilkan Dataset

Untuk contoh ini, kita akan bekerja dalam dimensi rendah
untuk kesederhanaan.
Cuplikan kode berikut menghasilkan 1000 contoh
dengan fitur berdimensi 2 yang diambil
dari distribusi normal standar.
Matriks desain yang dihasilkan $\mathbf{X}$
berada dalam $\mathbb{R}^{1000 \times 2}$.
Kita menghasilkan setiap label dengan menerapkan
fungsi linear *ground truth*,
yang kita ganggu dengan noise aditif $\boldsymbol{\epsilon}$,
yang diambil secara independen dan identik untuk setiap contoh:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

Untuk kemudahan, kita asumsikan bahwa $\boldsymbol{\epsilon}$ diambil
dari distribusi normal dengan mean $\mu= 0$
dan simpangan baku $\sigma = 0.01$.
Perhatikan bahwa untuk desain berbasis objek,
kita menambahkan kode ke metode `__init__` dari subclass `d2l.DataModule` (diperkenalkan di :numref:`oo-design-data`).
Merupakan praktik yang baik untuk memungkinkan pengaturan hyperparameter tambahan.
Kita mencapai ini dengan `save_hyperparameters()`.
`batch_size` akan ditentukan kemudian.


```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Data sintetis untuk regresi linear."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Di bawah ini, kita menetapkan parameter sebenarnya ke $\mathbf{w} = [2, -3.4]^\top$ dan $b = 4.2$.
Nantinya, kita dapat memeriksa parameter yang kita estimasi terhadap nilai *ground truth* ini.


```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Setiap baris di `features` terdiri dari sebuah vektor di $\mathbb{R}^2$ dan setiap baris di `labels` adalah sebuah skalar.**] Mari kita lihat entri pertama.


```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Membaca Dataset

Melatih model pembelajaran mesin sering kali membutuhkan beberapa kali pemrosesan melalui dataset, 
mengambil satu minibatch contoh pada satu waktu. 
Data ini kemudian digunakan untuk memperbarui model. 
Untuk mengilustrasikan cara kerjanya, kita 
[**mengimplementasikan metode `get_dataloader`,**] 
dengan mendaftarkannya di kelas `SyntheticRegressionData` melalui `add_to_class` (diperkenalkan di :numref:`oo-design-utilities`).
Metode ini (**mengambil ukuran batch, matriks fitur,
dan vektor label, dan menghasilkan minibatch dengan ukuran `batch_size`.**)
Dengan demikian, setiap minibatch terdiri dari pasangan fitur dan label. 
Perhatikan bahwa kita perlu memperhatikan apakah kita berada dalam mode pelatihan atau validasi: 
pada yang pertama, kita ingin membaca data dalam urutan acak, 
sedangkan pada yang kedua, kemampuan untuk membaca data dalam urutan yang sudah ditentukan 
dapat penting untuk tujuan debugging.


```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

Untuk membangun intuisi, mari kita periksa minibatch pertama dari
data. Setiap minibatch fitur memberi kita ukuran serta dimensi dari fitur input.
Demikian pula, minibatch label kita akan memiliki bentuk yang sesuai dengan `batch_size`.


```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Meskipun terlihat tidak berbahaya, pemanggilan 
`iter(data.train_dataloader())`
mengilustrasikan kekuatan desain berbasis objek di Python.
Perhatikan bahwa kita menambahkan metode ke kelas `SyntheticRegressionData`
*setelah* membuat objek `data`.
Namun demikian, objek tersebut tetap dapat memanfaatkan 
penambahan fungsi secara *ex post facto* pada kelas.

Sepanjang iterasi, kita memperoleh minibatch yang berbeda
sampai seluruh dataset habis (coba lakukan ini).
Meskipun iterasi yang diterapkan di atas baik untuk tujuan didaktis,
ini tidak efisien dalam beberapa cara yang bisa menjadi masalah pada masalah nyata.
Sebagai contoh, iterasi ini membutuhkan kita untuk memuat semua data ke dalam memori
dan melakukan banyak akses memori secara acak.
Iterator bawaan yang diterapkan dalam framework pembelajaran mendalam
jauh lebih efisien dan dapat menangani
sumber seperti data yang disimpan dalam file,
data yang diterima melalui aliran (stream),
dan data yang dihasilkan atau diproses secara langsung.
Selanjutnya, mari kita coba menerapkan metode yang sama menggunakan iterator bawaan.

## Implementasi Ringkas dari Data Loader

Alih-alih menulis iterator kita sendiri,
kita bisa [**memanggil API yang sudah ada dalam framework untuk memuat data.**]
Seperti sebelumnya, kita membutuhkan dataset dengan fitur `X` dan label `y`.
Selain itu, kita menetapkan `batch_size` dalam data loader bawaan
dan membiarkannya menangani pengacakan contoh secara efisien.

:begin_tab:`jax`
JAX berfokus pada API mirip NumPy dengan percepatan perangkat dan transformasi fungsional,
sehingga setidaknya versi saat ini tidak mencakup metode pemuatan data.
Dengan pustaka lain, kita sudah memiliki data loader yang hebat di luar sana,
dan JAX menyarankan untuk menggunakan data loader tersebut.
Di sini kita akan menggunakan data loader dari TensorFlow,
dan sedikit memodifikasinya agar bekerja dengan JAX.
:end_tab:


```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Gunakan Tensorflow Datasets & Dataloader. JAX atau Flax tidak menyediakan
        # fungsi pemuatan data
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)

```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```
Data loader baru berperilaku sama seperti yang sebelumnya, kecuali bahwa ia lebih efisien dan memiliki beberapa fungsi tambahan.


```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

Sebagai contoh, data loader yang disediakan oleh API framework 
mendukung metode bawaan `__len__`, 
sehingga kita dapat mengetahui panjangnya, 
yaitu jumlah batch.


```{.python .input}
%%tab all
len(data.train_dataloader())
```


## Ringkasan

Data loader adalah cara yang nyaman untuk mengabstraksikan 
proses pemuatan dan manipulasi data.
Dengan cara ini, *algoritma* pembelajaran mesin yang sama 
dapat memproses berbagai jenis dan sumber data 
tanpa perlu modifikasi.
Salah satu hal yang menarik tentang data loader 
adalah bahwa data loader dapat digabungkan.
Misalnya, kita mungkin memuat gambar 
dan kemudian memiliki filter pascapemrosesan 
yang memotong gambar atau memodifikasinya dengan cara lain.
Dengan demikian, data loader dapat digunakan 
untuk menggambarkan seluruh pipeline pemrosesan data.

Adapun model itu sendiri, model linear dua dimensi 
adalah yang paling sederhana yang mungkin kita temui.
Model ini memungkinkan kita menguji akurasi model regresi 
tanpa khawatir tentang jumlah data yang tidak mencukupi 
atau sistem persamaan yang kurang terdefinisi.
Kita akan memanfaatkannya dengan baik di bagian selanjutnya.


## Latihan

1. Apa yang akan terjadi jika jumlah contoh tidak dapat dibagi oleh ukuran batch? Bagaimana Anda akan mengubah perilaku ini dengan menentukan argumen yang berbeda dengan menggunakan API framework?
1. Misalkan kita ingin menghasilkan dataset yang sangat besar, di mana ukuran vektor parameter `w` dan jumlah contoh `num_examples` besar.
    1. Apa yang terjadi jika kita tidak dapat menyimpan semua data dalam memori?
    1. Bagaimana Anda akan mengacak data jika disimpan di disk? Tugas Anda adalah merancang algoritma *efisien* yang tidak memerlukan terlalu banyak operasi baca atau tulis acak. Petunjuk: [generator permutasi pseudorandom](https://en.wikipedia.org/wiki/Pseudorandom_permutation) memungkinkan Anda merancang pengacakan ulang tanpa perlu menyimpan tabel permutasi secara eksplisit :cite:`Naor.Reingold.1999`.
1. Implementasikan generator data yang menghasilkan data baru secara langsung, setiap kali iterator dipanggil.
1. Bagaimana Anda akan merancang generator data acak yang menghasilkan *data yang sama* setiap kali dipanggil?


:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17975)
:end_tab:
