```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implementasi Singkat Regresi Linear
:label:`sec_linear_concise`

Pembelajaran mendalam (deep learning) telah menyaksikan semacam "ledakan Kambrium" dalam dekade terakhir.
Jumlah teknik, aplikasi, dan algoritma yang berkembang pesat jauh melampaui
kemajuan dari dekade-dekade sebelumnya.
Hal ini disebabkan oleh kombinasi beberapa faktor yang menguntungkan,
salah satunya adalah alat-alat gratis yang kuat
yang ditawarkan oleh beberapa framework pembelajaran mendalam open-source.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
dan Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
merupakan generasi pertama dari model-model ini
yang mendapatkan adopsi secara luas.
Sebagai perbandingan dengan karya-karya awal seperti
SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`,
yang memberikan pengalaman pemrograman mirip Lisp,
framework modern menawarkan diferensiasi otomatis
dan kenyamanan menggunakan bahasa Python.
Framework ini memungkinkan kita untuk mengotomatisasi dan memodularisasi
pekerjaan berulang dalam mengimplementasikan algoritma pembelajaran berbasis gradien.

Di :numref:`sec_linear_scratch`, kita hanya mengandalkan
(i) tensor untuk penyimpanan data dan aljabar linear;
dan (ii) diferensiasi otomatis untuk menghitung gradien.
Dalam praktiknya, karena iterator data, fungsi kerugian, optimizer,
dan lapisan neural network sangat umum,
perpustakaan modern juga mengimplementasikan komponen-komponen ini untuk kita.
Di bagian ini, (**kami akan menunjukkan cara mengimplementasikan
model regresi linear**) dari :numref:`sec_linear_scratch`
(**dengan lebih ringkas menggunakan API tingkat tinggi**) dari framework pembelajaran mendalam.


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
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Mendefinisikan Model

Ketika kita mengimplementasikan regresi linear dari awal di :numref:`sec_linear_scratch`,
kita mendefinisikan parameter model kita secara eksplisit
dan menuliskan perhitungan untuk menghasilkan output
menggunakan operasi aljabar linear dasar.
Anda *harus* tahu cara melakukan ini.
Namun, ketika model Anda menjadi lebih kompleks
dan Anda harus melakukannya hampir setiap hari,
bantuan dari framework akan sangat berguna.
Situasinya mirip dengan membangun blog sendiri dari awal.
Melakukannya sekali atau dua kali memang bermanfaat dan mendidik,
tetapi Anda akan menjadi pengembang web yang buruk
jika Anda menghabiskan sebulan untuk menciptakan ulang sesuatu yang sudah ada.

Untuk operasi standar,
kita dapat [**menggunakan lapisan yang telah didefinisikan oleh framework,**]
yang memungkinkan kita fokus pada lapisan-lapisan yang digunakan untuk membangun model
tanpa perlu khawatir tentang implementasinya.
Ingat kembali arsitektur jaringan satu lapisan
seperti dijelaskan di :numref:`fig_single_neuron`.
Lapisan ini disebut *fully connected* atau lapisan koneksi penuh,
karena setiap input terhubung
dengan setiap output
melalui perkalian matriks-vektor.

:begin_tab:`mxnet`
Di Gluon, lapisan koneksi penuh didefinisikan dalam kelas `Dense`.
Karena kita hanya ingin menghasilkan satu output skalar,
kita menetapkan angka tersebut menjadi 1.
Perlu dicatat bahwa, demi kenyamanan,
Gluon tidak mengharuskan kita untuk menentukan
bentuk input untuk setiap lapisan.
Oleh karena itu, kita tidak perlu memberitahu Gluon
berapa banyak input yang masuk ke lapisan linear ini.
Ketika kita pertama kali melewatkan data melalui model kita,
misalnya, saat kita mengeksekusi `net(X)` nanti,
Gluon akan secara otomatis menginferensi jumlah input untuk setiap lapisan dan
menginisialisasi model dengan benar.
Kami akan menjelaskan cara kerja ini lebih detail nanti.
:end_tab:

:begin_tab:`pytorch`
Di PyTorch, lapisan koneksi penuh didefinisikan dalam kelas `Linear` dan `LazyLinear` (tersedia sejak versi 1.8.0).
Kelas `LazyLinear`
memungkinkan pengguna untuk hanya menetapkan
dimensi output,
sedangkan `Linear` juga meminta
jumlah input yang masuk ke lapisan ini.
Menentukan bentuk input terkadang merepotkan dan bisa membutuhkan perhitungan yang tidak sederhana
(seperti pada lapisan konvolusi).
Jadi, untuk kemudahan, kita akan menggunakan lapisan "lazy" ini
kapan pun memungkinkan.
:end_tab:

:begin_tab:`tensorflow`
Di Keras, lapisan koneksi penuh didefinisikan dalam kelas `Dense`.
Karena kita hanya ingin menghasilkan satu output skalar,
kita menetapkan angka tersebut menjadi 1.
Perlu dicatat bahwa, demi kenyamanan,
Keras tidak mengharuskan kita untuk menentukan
bentuk input untuk setiap lapisan.
Kita tidak perlu memberitahu Keras
berapa banyak input yang masuk ke lapisan linear ini.
Ketika kita pertama kali mencoba melewatkan data melalui model kita,
misalnya, saat kita mengeksekusi `net(X)` nanti,
Keras akan secara otomatis menginferensi
jumlah input untuk setiap lapisan.
Kami akan menjelaskan cara kerjanya lebih detail nanti.
:end_tab:


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    """Model regresi linear yang diimplementasikan dengan API tingkat tinggi."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    """Model regresi linear yang diimplementasikan dengan API tingkat tinggi."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

Dalam metode `forward`, kita cukup memanggil metode `__call__` bawaan dari lapisan yang telah didefinisikan untuk menghitung output.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## Mendefinisikan Fungsi Kerugian

:begin_tab:`mxnet`
Modul `loss` mendefinisikan banyak fungsi kerugian yang berguna.
Demi kecepatan dan kenyamanan, kita tidak mengimplementasikan fungsi kerugian sendiri
dan memilih menggunakan `loss.L2Loss` bawaan.
Karena `loss` yang dikembalikan merupakan
kesalahan kuadrat untuk setiap contoh,
kita menggunakan `mean` untuk mengambil rata-rata kerugian pada minibatch.
:end_tab:

:begin_tab:`pytorch`
[**Kelas `MSELoss` menghitung rata-rata kesalahan kuadrat (tanpa faktor $1/2$ dalam :eqref:`eq_mse`).**]
Secara default, `MSELoss` mengembalikan rata-rata kerugian dari contoh-contoh yang ada.
Ini lebih cepat (dan lebih mudah digunakan) daripada mengimplementasikan sendiri.
:end_tab:

:begin_tab:`tensorflow`
Kelas `MeanSquaredError` menghitung rata-rata kesalahan kuadrat (tanpa faktor $1/2$ dalam :eqref:`eq_mse`).
Secara default, kelas ini mengembalikan rata-rata kerugian dari contoh-contoh yang ada.
:end_tab:


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## Mendefinisikan Algoritma Optimasi

:begin_tab:`mxnet`
Minibatch SGD adalah alat standar
untuk mengoptimalkan jaringan neural,
dan oleh karena itu Gluon mendukungnya beserta sejumlah
variasi dari algoritma ini melalui kelas `Trainer`.
Perhatikan bahwa kelas `Trainer` di Gluon
mewakili algoritma optimasi,
sedangkan kelas `Trainer` yang kita buat di :numref:`sec_oo-design`
berisi metode pelatihan,
yaitu, memanggil optimizer secara berulang
untuk memperbarui parameter model.
Saat kita membuat instance `Trainer`,
kita menentukan parameter yang akan dioptimalkan,
dapat diakses dari model `net` melalui `net.collect_params()`,
algoritma optimasi yang ingin kita gunakan (`sgd`),
dan sebuah dictionary hyperparameter
yang diperlukan oleh algoritma optimasi kita.
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD adalah alat standar
untuk mengoptimalkan jaringan neural,
dan oleh karena itu PyTorch mendukungnya beserta sejumlah
variasi dari algoritma ini di modul `optim`.
Ketika kita (**membuat instance `SGD`,**)
kita menentukan parameter yang akan dioptimalkan,
dapat diakses dari model kita melalui `self.parameters()`,
dan learning rate (`self.lr`)
yang dibutuhkan oleh algoritma optimasi kita.
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGD adalah alat standar
untuk mengoptimalkan jaringan neural,
dan oleh karena itu Keras mendukungnya beserta sejumlah
variasi dari algoritma ini di modul `optimizers`.
:end_tab:


```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## Pelatihan

Anda mungkin telah memperhatikan bahwa mengekspresikan model kita melalui
API tingkat tinggi dari framework pembelajaran mendalam
membutuhkan lebih sedikit baris kode.
Kita tidak perlu menetapkan parameter secara individual,
mendefinisikan fungsi kerugian kita, atau mengimplementasikan minibatch SGD.
Begitu kita mulai bekerja dengan model yang jauh lebih kompleks,
keuntungan dari API tingkat tinggi akan semakin terasa.

Sekarang setelah kita memiliki semua komponen dasar,
[**loop pelatihan itu sendiri sama
dengan yang kita implementasikan dari awal.**]
Jadi kita hanya perlu memanggil metode `fit` (diperkenalkan di :numref:`oo-design-training`),
yang bergantung pada implementasi metode `fit_epoch`
di :numref:`sec_linear_scratch`,
untuk melatih model kita.


```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Di bawah ini, kita
[**membandingkan parameter model yang dipelajari
melalui pelatihan pada data terbatas
dengan parameter asli**]
yang menghasilkan dataset kita.
Untuk mengakses parameter,
kita mengambil bobot dan bias
dari lapisan yang kita perlukan.
Seperti dalam implementasi kita dari awal,
perhatikan bahwa parameter yang kita estimasi
mendekati nilai sebenarnya.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Ringkasan

Bagian ini berisi implementasi pertama
dari jaringan dalam (di buku ini)
yang memanfaatkan kemudahan yang disediakan
oleh framework pembelajaran mendalam modern,
seperti MXNet :cite:`Chen.Li.Li.ea.2015`, 
JAX :cite:`Frostig.Johnson.Leary.2018`, 
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, 
dan TensorFlow :cite:`Abadi.Barham.Chen.ea.2016`.
Kita menggunakan pengaturan standar framework untuk memuat data, mendefinisikan lapisan,
fungsi kerugian, optimizer, dan loop pelatihan.
Kapan pun framework menyediakan semua fitur yang diperlukan,
umumnya ide yang baik untuk menggunakannya,
karena implementasi perpustakaan dari komponen-komponen ini
cenderung dioptimalkan secara menyeluruh untuk performa
dan diuji dengan baik untuk keandalan.
Namun, perlu diingat bahwa modul-modul ini *dapat* diimplementasikan secara langsung.
Hal ini terutama penting bagi calon peneliti
yang ingin berada di garis depan pengembangan model,
di mana Anda akan menciptakan komponen baru
yang mungkin belum tersedia di perpustakaan manapun saat ini.

:begin_tab:`mxnet`
Di Gluon, modul `data` menyediakan alat untuk pemrosesan data,
modul `nn` mendefinisikan banyak lapisan jaringan neural,
dan modul `loss` mendefinisikan banyak fungsi kerugian umum.
Selain itu, `initializer` memberikan berbagai pilihan
untuk inisialisasi parameter.
Dengan kemudahan ini, dimensi dan penyimpanan diinferensi secara otomatis.
Konsekuensi dari inisialisasi malas (lazy initialization) ini adalah
Anda tidak boleh mencoba mengakses parameter
sebelum parameter tersebut diinisialisasi.
:end_tab:

:begin_tab:`pytorch`
Di PyTorch, modul `data` menyediakan alat untuk pemrosesan data,
modul `nn` mendefinisikan banyak lapisan jaringan neural dan fungsi kerugian umum.
Kita dapat menginisialisasi parameter dengan mengganti nilainya
dengan metode yang diakhiri dengan `_`.
Perhatikan bahwa kita perlu menentukan dimensi input dari jaringan.
Meskipun ini sederhana untuk sekarang, hal ini dapat memiliki dampak besar
saat kita ingin merancang jaringan yang kompleks dengan banyak lapisan.
Pertimbangan yang hati-hati tentang cara mengatur jaringan ini
diperlukan agar mudah dipindahkan.
:end_tab:

:begin_tab:`tensorflow`
Di TensorFlow, modul `data` menyediakan alat untuk pemrosesan data,
modul `keras` mendefinisikan banyak lapisan jaringan neural dan fungsi kerugian umum.
Selain itu, modul `initializers` menyediakan berbagai metode untuk inisialisasi parameter model.
Dimensi dan penyimpanan jaringan diinferensi secara otomatis
(tetapi berhati-hatilah untuk tidak mencoba mengakses parameter sebelum diinisialisasi).
:end_tab:

## Latihan

1. Bagaimana Anda perlu mengubah learning rate jika Anda mengganti agregat kerugian di minibatch
   dengan rata-rata kerugian pada minibatch?
1. Tinjau dokumentasi framework untuk melihat fungsi kerugian apa saja yang disediakan. Secara khusus,
   ganti kerugian kuadrat dengan fungsi kerugian robust Huber. Yaitu, gunakan fungsi kerugian
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
1. Bagaimana cara mengakses gradien dari bobot model?
1. Apa pengaruhnya pada solusi jika Anda mengubah learning rate dan jumlah epoch? Apakah terus meningkat?
1. Bagaimana solusi berubah saat Anda memvariasikan jumlah data yang dihasilkan?
    1. Gambarkan kesalahan estimasi untuk $\hat{\mathbf{w}} - \mathbf{w}$ dan $\hat{b} - b$ sebagai fungsi dari jumlah data. Petunjuk: tingkatkan jumlah data secara logaritmik daripada linier, misalnya, 5, 10, 20, 50, ..., 10.000 daripada 1.000, 2.000, ..., 10.000.
    2. Mengapa saran dalam petunjuk tersebut tepat?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17977)
:end_tab:
