```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Desain Berorientasi Objek untuk Implementasi
:label:`sec_oo-design`

Dalam pengenalan kita terhadap regresi linear,
kita membahas berbagai komponen
termasuk data, model, fungsi kerugian,
dan algoritma optimasi.
Memang, regresi linear adalah
salah satu model pembelajaran mesin yang paling sederhana.
Namun, pelatihan model ini
menggunakan banyak komponen yang juga dibutuhkan oleh model lain dalam buku ini.
Oleh karena itu,
sebelum masuk ke detail implementasi,
pantas kiranya
merancang beberapa API
yang akan kita gunakan sepanjang pembahasan. 
Dengan memperlakukan komponen dalam pembelajaran mendalam
sebagai objek,
kita dapat mulai dengan
mendefinisikan kelas untuk objek-objek ini
dan interaksinya.
Desain berorientasi objek untuk implementasi ini
akan sangat menyederhanakan presentasi, dan mungkin Anda juga ingin menggunakannya dalam proyek Anda.

Terinspirasi oleh pustaka open-source seperti [PyTorch Lightning](https://www.pytorchlightning.ai/),
pada tingkat tinggi
kita ingin memiliki tiga kelas:
(i) `Module` berisi model, fungsi kerugian, dan metode optimasi;
(ii) `DataModule` menyediakan data loader untuk pelatihan dan validasi;
(iii) kedua kelas ini digabungkan menggunakan kelas `Trainer`, yang memungkinkan kita
melatih model pada berbagai platform perangkat keras.
Sebagian besar kode dalam buku ini mengadaptasi `Module` dan `DataModule`. Kita akan menyentuh kelas `Trainer` hanya ketika kita membahas GPU, CPU, pelatihan paralel, dan algoritma optimasi.


```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## Utilitas
:label:`oo-design-utilities`

Kita memerlukan beberapa utilitas untuk menyederhanakan pemrograman berorientasi objek di Jupyter notebook. Salah satu tantangannya adalah bahwa definisi kelas cenderung menjadi blok kode yang cukup panjang. Keterbacaan notebook membutuhkan fragmen kode yang pendek, diselingi dengan penjelasan, sebuah kebutuhan yang tidak sesuai dengan gaya pemrograman yang umum untuk pustaka Python. 
Fungsi utilitas pertama memungkinkan kita untuk mendaftarkan fungsi sebagai metode dalam kelas *setelah* kelas tersebut dibuat. Faktanya, kita bahkan dapat melakukannya *setelah* kita membuat instance dari kelas tersebut! Hal ini memungkinkan kita untuk membagi implementasi sebuah kelas menjadi beberapa blok kode.



```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Mendaftarkan fungsi sebagai metode dalam kelas yang telah dibuat."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Mari kita lihat cara cepat penggunaannya. Kita berencana mengimplementasikan kelas `A` dengan metode `do`. Alih-alih menulis kode untuk `A` dan `do` dalam satu blok kode yang sama, kita dapat terlebih dahulu mendeklarasikan kelas `A` dan membuat sebuah instance `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```
Selanjutnya, kita mendefinisikan metode `do` seperti biasa, tetapi tidak dalam lingkup kelas `A`. Sebagai gantinya, kita mendekorasi metode ini dengan `add_to_class` dengan kelas `A` sebagai argumennya. Dengan cara ini, metode tersebut dapat mengakses variabel anggota dari `A` seperti yang diharapkan seolah-olah metode tersebut adalah bagian dari definisi `A`. Mari kita lihat apa yang terjadi ketika kita memanggilnya untuk instance `a`.


```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

Yang kedua adalah kelas utilitas yang menyimpan semua argumen dalam metode `__init__` suatu kelas sebagai atribut kelas. 
Ini memungkinkan kita untuk memperluas tanda tangan pemanggilan konstruktor secara implisit tanpa kode tambahan.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """Kelas dasar untuk hyperparameter."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Kita menunda implementasinya hingga :numref:`sec_utils`. Untuk menggunakannya, kita mendefinisikan kelas kita yang mewarisi `HyperParameters` dan memanggil `save_hyperparameters` di dalam metode `__init__`.

```{.python .input}
%%tab all
# Memanggil kelas HyperParameters yang telah diimplementasikan sepenuhnya dan disimpan di d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Utilitas terakhir memungkinkan kita untuk memplot kemajuan eksperimen secara interaktif saat eksperimen berlangsung. Sebagai penghormatan terhadap [TensorBoard](https://www.tensorflow.org/tensorboard) yang jauh lebih kuat (dan kompleks), kita menamakannya `ProgressBoard`. Implementasinya ditunda hingga :numref:`sec_utils`. Untuk sekarang, mari kita lihat langsung penggunaannya.

Metode `draw` memplot titik `(x, y)` dalam gambar, dengan `label` yang ditentukan dalam legenda. Parameter opsional `every_n` memperhalus garis dengan hanya menampilkan $1/n$ titik dalam gambar. 
Nilai-nilai ini adalah rata-rata dari $n$ titik tetangga di gambar asli.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """Papan yang memplot titik data secara animasi."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

Dalam contoh berikut, kita memplot `sin` dan `cos` dengan tingkat kehalusan yang berbeda. Jika Anda menjalankan blok kode ini, Anda akan melihat garis-garis tersebut tumbuh secara animatif.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Model
:label:`subsec_oo-design-models`

Kelas `Module` adalah kelas dasar dari semua model yang akan kita implementasikan. Setidaknya kita membutuhkan tiga metode. Pertama, `__init__`, yang menyimpan parameter yang dapat dipelajari; metode `training_step` yang menerima satu batch data dan mengembalikan nilai kerugian; dan terakhir, `configure_optimizers` yang mengembalikan metode optimasi, atau daftar metode, yang digunakan untuk memperbarui parameter yang dapat dipelajari. Sebagai opsi, kita dapat mendefinisikan `validation_step` untuk melaporkan ukuran evaluasi.
Terkadang, kita memisahkan kode untuk menghitung output ke dalam metode `forward` agar lebih mudah digunakan kembali.

:begin_tab:`jax`
Dengan diperkenalkannya [dataclasses](https://docs.python.org/3/library/dataclasses.html)
pada Python 3.7, kelas yang dihiasi dengan `@dataclass` secara otomatis menambahkan metode
magis seperti `__init__` dan `__repr__`. Variabel anggota didefinisikan
menggunakan anotasi tipe. Semua modul Flax adalah dataclass Python 3.7.
:end_tab:


```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """Kelas dasar dari model-model."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Jaringan neural harus didefinisikan'
        return self.net(X)

    def plot(self, key, value, train):
        """Memplot sebuah titik dalam animasi."""
        assert hasattr(self, 'trainer'), 'Trainer belum diinisialisasi'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """Kelas dasar dari model-model."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # Tidak perlu menggunakan save_hyperparam saat memakai dataclass Python
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Menggunakan default_factory untuk memastikan plot baru dibuat setiap kali dijalankan
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Jaringan neural harus didefinisikan'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax tidak memiliki sintaks metode forward seperti yang lain. Flax menggunakan setup
        # dan metode magis __call__ bawaan untuk forward pass. Ditambahkan di sini
        # untuk konsistensi
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Jaringan neural harus didefinisikan'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Memplot titik dalam animasi."""
        assert hasattr(self, 'trainer'), 'Trainer belum diinisialisasi'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """Akan didefinisikan kemudian di :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

```

:begin_tab:`mxnet`
Anda mungkin memperhatikan bahwa `Module` adalah subclass dari `nn.Block`, kelas dasar jaringan neural di Gluon.
Ini menyediakan fitur yang memudahkan penanganan jaringan neural. Misalnya, jika kita mendefinisikan metode `forward`, seperti `forward(self, X)`, maka untuk instance `a` kita dapat memanggil metode ini dengan `a(X)`. Ini berfungsi karena memanggil metode `forward` dalam metode `__call__` bawaan. Anda dapat menemukan lebih banyak detail dan contoh tentang `nn.Block` di :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
Anda mungkin memperhatikan bahwa `Module` adalah subclass dari `nn.Module`, kelas dasar jaringan neural di PyTorch.
Ini menyediakan fitur yang memudahkan penanganan jaringan neural. Misalnya, jika kita mendefinisikan metode `forward`, seperti `forward(self, X)`, maka untuk instance `a` kita dapat memanggil metode ini dengan `a(X)`. Ini berfungsi karena memanggil metode `forward` dalam metode `__call__` bawaan. Anda dapat menemukan lebih banyak detail dan contoh tentang `nn.Module` di :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
Anda mungkin memperhatikan bahwa `Module` adalah subclass dari `tf.keras.Model`, kelas dasar jaringan neural di TensorFlow.
Ini menyediakan fitur yang memudahkan penanganan jaringan neural. Misalnya, ia memanggil metode `call` dalam metode `__call__` bawaan. Di sini kita mengarahkan `call` ke metode `forward`, menyimpan argumennya sebagai atribut kelas. Kami melakukan ini untuk membuat kode kami lebih mirip dengan implementasi framework lainnya.
:end_tab:

:begin_tab:`jax`
Anda mungkin memperhatikan bahwa `Module` adalah subclass dari `linen.Module`, kelas dasar jaringan neural di Flax.
Ini menyediakan fitur yang memudahkan penanganan jaringan neural. Misalnya, ini menangani parameter model, menyediakan dekorator `nn.compact` untuk menyederhanakan kode, dan memanggil metode `__call__` di antara hal-hal lainnya.
Di sini kita juga mengarahkan `__call__` ke metode `forward`. Kami melakukan ini untuk membuat kode kami lebih mirip dengan implementasi framework lainnya.
:end_tab:

## Data
:label:`oo-design-data`

Kelas `DataModule` adalah kelas dasar untuk data. Cukup sering metode `__init__` digunakan untuk menyiapkan data, termasuk mengunduh dan melakukan prapemrosesan jika diperlukan. Metode `train_dataloader` mengembalikan pemuat data untuk dataset pelatihan. Pemuat data adalah generator (Python) yang menghasilkan batch data setiap kali digunakan. Batch ini kemudian dimasukkan ke dalam metode `training_step` dari `Module` untuk menghitung loss. Ada opsi `val_dataloader` untuk mengembalikan pemuat dataset validasi. Ini berperilaku sama, kecuali menghasilkan batch data untuk metode `validation_step` dalam `Module`.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """Kelas dasar untuk data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Pelatihan
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
Kelas `Trainer` melatih parameter yang dapat dipelajari dalam kelas `Module` menggunakan data yang ditentukan di `DataModule`. Metode kunci di sini adalah `fit`, yang menerima dua argumen: `model`, sebuah instance dari `Module`, dan `data`, sebuah instance dari `DataModule`. Kemudian metode ini mengulangi seluruh dataset sebanyak `max_epochs` kali untuk melatih model. Seperti sebelumnya, kita akan menunda implementasi metode ini ke bab-bab selanjutnya.
:end_tab:

:begin_tab:`jax`
Kelas `Trainer` melatih parameter yang dapat dipelajari `params` menggunakan data yang ditentukan di `DataModule`. Metode kunci di sini adalah `fit`, yang menerima tiga argumen: `model`, sebuah instance dari `Module`, `data`, sebuah instance dari `DataModule`, dan `key`, sebuah `PRNGKeyArray` JAX. Di sini kita membuat argumen `key` menjadi opsional untuk menyederhanakan antarmuka, tetapi disarankan untuk selalu memasukkan dan menginisialisasi parameter model dengan key utama dalam JAX dan Flax. Kemudian metode ini mengulangi seluruh dataset sebanyak `max_epochs` kali untuk melatih model. Seperti sebelumnya, kita akan menunda implementasi metode ini ke bab-bab selanjutnya.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """Kelas dasar untuk melatih model dengan data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'Belum ada dukungan GPU'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Di sini batch_stats akan digunakan nanti (misalnya, untuk batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax menggunakan optax di belakang layar untuk satu objek state TrainState.
            # Lebih lanjut akan dibahas nanti di bagian dropout dan batch
            # normalization
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Ringkasan

Untuk menyoroti desain berbasis objek (object-oriented design) dalam implementasi pembelajaran mendalam yang akan kita buat ke depannya, kelas-kelas di atas menunjukkan bagaimana objek-objek tersebut menyimpan data dan berinteraksi satu sama lain. Kita akan terus memperkaya implementasi dari kelas-kelas ini, seperti menggunakan `@add_to_class`, di bagian-bagian berikutnya dalam buku ini. 

Selain itu, kelas-kelas yang telah diimplementasikan sepenuhnya disimpan di [perpustakaan D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l), yaitu *toolkit ringan* yang mempermudah pemodelan terstruktur untuk deep learning. Terutama, ini memfasilitasi penggunaan kembali banyak komponen antara berbagai proyek tanpa perlu banyak perubahan. Sebagai contoh, kita dapat mengganti hanya optimizernya, modelnya, atau dataset-nya; tingkat modularitas ini memberikan manfaat dalam hal keringkasan dan kesederhanaan sepanjang buku ini (itulah mengapa kita menambahkannya), dan ini juga dapat berguna dalam proyek-proyek Anda sendiri.

## Latihan

1. Temukan implementasi lengkap dari kelas-kelas di atas yang disimpan di [perpustakaan D2L](https://github.com/d2l-ai/d2l-en/tree/master/d2l). Kami sangat merekomendasikan Anda untuk melihat detail implementasinya setelah Anda lebih memahami pemodelan deep learning.
2. Hapus pernyataan `save_hyperparameters` dalam kelas `B`. Apakah Anda masih dapat mencetak `self.a` dan `self.b`? Opsional: jika Anda telah mempelajari implementasi lengkap dari kelas `HyperParameters`, bisakah Anda menjelaskan mengapa?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17974)
:end_tab:
