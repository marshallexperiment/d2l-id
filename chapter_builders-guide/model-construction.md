```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lapisan dan Modul
:label:`sec_model_construction`

Saat pertama kali memperkenalkan jaringan neural,
kita berfokus pada model linear dengan satu keluaran.
Di sini, keseluruhan model hanya terdiri dari satu neuron.
Perhatikan bahwa satu neuron
(i) menerima beberapa input;
(ii) menghasilkan keluaran skalar yang sesuai;
dan (iii) memiliki sekumpulan parameter terkait yang dapat diperbarui
untuk mengoptimalkan fungsi tujuan tertentu.
Kemudian, ketika kita mulai berpikir tentang jaringan dengan banyak keluaran,
kita memanfaatkan aritmatika vektorisasi
untuk mencirikan seluruh lapisan neuron.
Seperti halnya neuron individual,
lapisan (i) menerima sejumlah input,
(ii) menghasilkan keluaran yang sesuai,
dan (iii) dijelaskan oleh serangkaian parameter yang dapat disesuaikan.
Saat kita membahas regresi softmax,
sebuah lapisan tunggal menjadi model itu sendiri.
Namun, bahkan ketika kita kemudian memperkenalkan MLP,
kita masih dapat menganggap model
memiliki struktur dasar yang sama.

Menariknya, pada MLP,
baik model secara keseluruhan maupun lapisan-lapisannya
memiliki struktur ini.
Keseluruhan model menerima input mentah (fitur),
menghasilkan keluaran (prediksi),
dan memiliki parameter
(gabungan parameter dari semua lapisan penyusunnya).
Demikian juga, setiap lapisan individual menerima input
(diberikan oleh lapisan sebelumnya),
menghasilkan keluaran (input untuk lapisan berikutnya),
dan memiliki sekumpulan parameter yang dapat disesuaikan yang diperbarui
sesuai dengan sinyal yang mengalir mundur
dari lapisan berikutnya.

Meskipun mungkin Anda berpikir bahwa neuron, lapisan, dan model
memberikan cukup banyak abstraksi untuk kebutuhan kita,
ternyata sering kali lebih nyaman
untuk berbicara tentang komponen yang
lebih besar dari satu lapisan tetapi lebih kecil dari keseluruhan model.
Sebagai contoh, arsitektur ResNet-152,
yang sangat populer di bidang visi komputer,
memiliki ratusan lapisan.
Lapisan-lapisan ini terdiri dari pola berulang *kelompok lapisan*. Mengimplementasikan jaringan seperti itu satu lapisan pada satu waktu dapat menjadi sangat melelahkan.
Kekhawatiran ini bukan sekadar hipotetis—pola desain seperti ini umum digunakan dalam praktik.
Arsitektur ResNet yang disebutkan di atas
memenangkan kompetisi visi komputer ImageNet dan COCO 2015
baik untuk pengenalan maupun deteksi :cite:`He.Zhang.Ren.ea.2016`
dan tetap menjadi arsitektur pilihan untuk banyak tugas visi komputer.
Arsitektur serupa, di mana lapisan-lapisan diatur
dalam berbagai pola berulang,
sekarang tersebar luas di bidang lain,
termasuk pemrosesan bahasa alami dan pengenalan suara.

Untuk mengimplementasikan jaringan yang kompleks ini,
kita memperkenalkan konsep *modul* jaringan neural.
Sebuah modul bisa menggambarkan satu lapisan,
komponen yang terdiri dari beberapa lapisan,
atau keseluruhan model itu sendiri!
Salah satu manfaat bekerja dengan abstraksi modul
adalah modul-modul ini dapat digabungkan menjadi artefak yang lebih besar,
sering kali secara rekursif. Hal ini diilustrasikan pada :numref:`fig_blocks`. Dengan mendefinisikan kode untuk menghasilkan modul
dengan kompleksitas apa pun sesuai permintaan,
kita bisa menulis kode yang sangat ringkas
namun tetap dapat mengimplementasikan jaringan neural yang kompleks.

![Beberapa lapisan digabungkan menjadi modul, membentuk pola berulang dari model yang lebih besar.](../img/blocks.svg)
:label:`fig_blocks`

Dari sudut pandang pemrograman, modul direpresentasikan oleh *kelas*.
Setiap subclass dari modul ini harus mendefinisikan metode forward propagation
yang mengubah input menjadi output
dan harus menyimpan parameter yang diperlukan.
Perhatikan bahwa beberapa modul tidak memerlukan parameter sama sekali.
Terakhir, modul harus memiliki metode backpropagation,
untuk keperluan menghitung gradien.
Untungnya, berkat beberapa mekanisme di balik layar
yang disediakan oleh auto differentiation
(diperkenalkan pada :numref:`sec_autograd`),
saat mendefinisikan modul kita sendiri,
kita hanya perlu memikirkan parameter
dan metode forward propagation.


```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

[**Untuk memulai, kita meninjau kembali kode
yang kita gunakan untuk mengimplementasikan MLP**]
(:numref:`sec_mlp`).
Kode berikut menghasilkan jaringan
dengan satu lapisan tersembunyi fully connected
dengan 256 unit dan aktivasi ReLU,
diikuti oleh lapisan output fully connected
dengan sepuluh unit (tanpa fungsi aktivasi).


```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

# get_key adalah fungsi yang disimpan d2l yang mengembalikan jax.random.PRNGKey(random_seed)
X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

:begin_tab:`mxnet`
Dalam contoh ini, kita membangun
model dengan menginstansiasi `nn.Sequential`,
dan menetapkan objek yang dikembalikan ke variabel `net`.
Selanjutnya, kita berulang kali memanggil metode `add`,
menambahkan lapisan-lapisan dalam urutan
yang seharusnya dijalankan.
Singkatnya, `nn.Sequential` mendefinisikan jenis `Block` khusus,
kelas yang merepresentasikan *modul* dalam Gluon.
Ini mempertahankan daftar berurutan dari `Block`-`Block` penyusunnya.
Metode `add` memudahkan
penambahan setiap `Block` berturut-turut ke dalam daftar.
Perhatikan bahwa setiap lapisan adalah instance dari kelas `Dense`
yang merupakan subclass dari `Block`.
Metode forward propagation (`forward`) juga sangat sederhana:
ia merangkaikan setiap `Block` dalam daftar,
mengirimkan output dari satu sebagai input ke yang berikutnya.
Perhatikan bahwa hingga sekarang, kita telah memanggil model
melalui konstruksi `net(X)` untuk mendapatkan outputnya.
Ini sebenarnya adalah singkatan dari `net.forward(X)`,
trik Python yang rapi yang dicapai melalui
metode `__call__` pada kelas `Block`.
:end_tab:

:begin_tab:`pytorch`
Dalam contoh ini, kita membangun
model dengan menginstansiasi `nn.Sequential`, dengan lapisan-lapisan yang akan dieksekusi sesuai urutan yang diberikan sebagai argumen.
Singkatnya, (**`nn.Sequential` mendefinisikan jenis `Module` khusus**),
kelas yang merepresentasikan modul dalam PyTorch.
Ini mempertahankan daftar berurutan dari `Module`-`Module` penyusunnya.
Perhatikan bahwa kedua lapisan fully connected adalah instance dari kelas `Linear`
yang merupakan subclass dari `Module`.
Metode forward propagation (`forward`) juga sangat sederhana:
ia merangkaikan setiap modul dalam daftar,
mengirimkan output dari satu sebagai input ke yang berikutnya.
Perhatikan bahwa hingga sekarang, kita telah memanggil model
melalui konstruksi `net(X)` untuk mendapatkan outputnya.
Ini sebenarnya adalah singkatan dari `net.__call__(X)`.
:end_tab:

:begin_tab:`tensorflow`
Dalam contoh ini, kita membangun
model dengan menginstansiasi `keras.models.Sequential`, dengan lapisan-lapisan yang akan dieksekusi sesuai urutan yang diberikan sebagai argumen.
Singkatnya, `Sequential` mendefinisikan jenis `keras.Model` khusus,
kelas yang merepresentasikan modul dalam Keras.
Ini mempertahankan daftar berurutan dari `Model`-`Model` penyusunnya.
Perhatikan bahwa kedua lapisan fully connected adalah instance dari kelas `Dense`
yang merupakan subclass dari `Model`.
Metode forward propagation (`call`) juga sangat sederhana:
ia merangkaikan setiap modul dalam daftar,
mengirimkan output dari satu sebagai input ke yang berikutnya.
Perhatikan bahwa hingga sekarang, kita telah memanggil model
melalui konstruksi `net(X)` untuk mendapatkan outputnya.
Ini sebenarnya adalah singkatan dari `net.call(X)`,
trik Python yang rapi yang dicapai melalui
metode `__call__` pada kelas modul.
:end_tab:


## [**Modul Kustom**]

Mungkin cara termudah untuk mengembangkan intuisi
tentang bagaimana modul bekerja
adalah dengan mengimplementasikannya sendiri.
Sebelum kita melakukannya,
kami akan merangkum secara singkat fungsionalitas dasar
yang harus disediakan oleh setiap modul:

1. Mengambil data input sebagai argumen untuk metode forward propagation-nya.
2. Menghasilkan output dengan mengembalikan nilai dari metode forward propagation. Perhatikan bahwa output mungkin memiliki bentuk berbeda dari input. Misalnya, lapisan fully connected pertama dalam model kita di atas menerima input berdimensi sembarang tetapi mengembalikan output dengan dimensi 256.
3. Menghitung gradien dari output terhadap input, yang dapat diakses melalui metode backpropagation-nya. Biasanya ini terjadi secara otomatis.
4. Menyimpan dan menyediakan akses ke parameter-parameter yang diperlukan
   untuk menjalankan perhitungan forward propagation.
5. Menginisialisasi parameter model sesuai kebutuhan.

Dalam cuplikan berikut,
kami membuat modul dari awal
yang sesuai dengan MLP
dengan satu lapisan tersembunyi berisi 256 unit,
dan lapisan output berdimensi 10.
Perhatikan bahwa kelas `MLP` di bawah ini mewarisi kelas yang merepresentasikan modul.
Kami sangat bergantung pada metode-metode dari kelas induk,
dengan hanya menyediakan konstruktor kita sendiri (`__init__` dalam Python) dan metode forward propagation.


```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # Call the constructor of the MLP parent class nn.Block to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # Mendefinisikan forward propagation model, yaitu, cara mengembalikan
    # output model yang diinginkan berdasarkan input X
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # Memanggil konstruktor dari kelas induk nn.Module untuk melakukan
        # inisialisasi yang diperlukan
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # Mendefinisikan forward propagation dari model, yaitu, cara mengembalikan
    # output model yang diinginkan berdasarkan input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # Memanggil konstruktor dari kelas induk tf.keras.Model untuk melakukan
        # inisialisasi yang diperlukan
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # Mendefinisikan forward propagation dari model, yaitu cara mengembalikan
    # output model yang diinginkan berdasarkan input X
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        # Define the layers
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    # Mendefinisikan forward propagation dari model, yaitu cara mengembalikan
    # output model yang diinginkan berdasarkan input X
    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

Mari kita fokus terlebih dahulu pada metode forward propagation.
Perhatikan bahwa metode ini menerima `X` sebagai input,
menghitung representasi tersembunyi
dengan fungsi aktivasi yang diterapkan,
dan menghasilkan output berupa logits.
Dalam implementasi `MLP` ini,
kedua lapisan adalah variabel instance.
Untuk melihat mengapa ini masuk akal, bayangkan
menginstansiasi dua MLP, `net1` dan `net2`,
dan melatih mereka pada data yang berbeda.
Secara alami, kita berharap mereka
merepresentasikan dua model terlatih yang berbeda.

Kita [**menginstansiasi lapisan-lapisan MLP**]
di dalam konstruktor
(**dan kemudian memanggil lapisan-lapisan ini**)
setiap kali metode forward propagation dipanggil.
Perhatikan beberapa detail penting.
Pertama, metode `__init__` yang kita kustomisasi
memanggil metode `__init__` dari kelas induk
melalui `super().__init__()`
sehingga kita tidak perlu menulis ulang
kode dasar yang berlaku untuk sebagian besar modul.
Kemudian, kita menginstansiasi dua lapisan fully connected,
menetapkannya ke `self.hidden` dan `self.out`.
Perhatikan bahwa kecuali jika kita mengimplementasikan lapisan baru,
kita tidak perlu khawatir tentang metode backpropagation
atau inisialisasi parameter.
Sistem akan menghasilkan metode-metode ini secara otomatis.
Mari kita coba ini.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

```{.python .input}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

Salah satu keunggulan utama dari abstraksi modul adalah fleksibilitasnya.
Kita dapat membuat subclass dari modul untuk membuat lapisan
(seperti kelas fully connected layer),
keseluruhan model (seperti kelas `MLP` di atas),
atau berbagai komponen dengan kompleksitas menengah.
Kita akan memanfaatkan fleksibilitas ini
di sepanjang bab-bab mendatang,
seperti saat membahas
jaringan saraf konvolusional (_convolutional neural networks_).


## [**Modul Sequential**]
:label:`subsec_model-construction-sequential`

Sekarang kita dapat melihat lebih dekat
bagaimana kelas `Sequential` bekerja.
Ingat bahwa `Sequential` dirancang
untuk merangkai modul-modul lain secara berurutan.
Untuk membangun `MySequential` sederhana kita sendiri,
kita hanya perlu mendefinisikan dua metode kunci:

1. Metode untuk menambahkan modul satu per satu ke dalam daftar.
2. Metode forward propagation untuk melewatkan input melalui rantai modul, dalam urutan yang sama dengan ketika modul-modul tersebut ditambahkan.

Kelas `MySequential` berikut memberikan fungsi yang sama
seperti kelas `Sequential` bawaan.


```{.python .input}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # Di sini, block adalah instance dari subclass Block, dan kita mengasumsikan bahwa
        # ia memiliki nama yang unik. Kita menyimpannya di dalam variabel anggota _children dari
        # kelas Block, dan tipe datanya adalah OrderedDict. Ketika instance MySequential
        # memanggil metode initialize, sistem secara otomatis
        # menginisialisasi semua anggota dari _children.
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict menjamin bahwa anggota akan ditelusuri sesuai urutan
        # saat mereka ditambahkan.
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
%%tab jax
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
Metode `add` menambahkan sebuah blok tunggal
ke dalam dictionary terurut `_children`.
Anda mungkin bertanya-tanya mengapa setiap `Block` di Gluon
memiliki atribut `_children`
dan mengapa kita menggunakannya daripada hanya
mendefinisikan daftar Python sendiri.
Singkatnya, keuntungan utama dari `_children`
adalah bahwa selama inisialisasi parameter blok kita,
Gluon tahu untuk melihat di dalam dictionary `_children`
untuk menemukan sub-blok yang
parameternya juga perlu diinisialisasi.
:end_tab:

:begin_tab:`pytorch`
Dalam metode `__init__`, kita menambahkan setiap modul
dengan memanggil metode `add_modules`. Modul-modul ini dapat diakses dengan metode `children` di kemudian hari.
Dengan cara ini, sistem tahu modul-modul yang ditambahkan,
dan ini akan menginisialisasi parameter setiap modul dengan benar.
:end_tab:

Ketika metode forward propagation dari `MySequential` kita dipanggil,
setiap modul yang ditambahkan dieksekusi
dalam urutan di mana mereka ditambahkan.
Sekarang kita dapat mengimplementasikan kembali MLP
menggunakan kelas `MySequential` kita.


```{.python .input}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

Perhatikan bahwa penggunaan `MySequential` ini
identik dengan kode yang sebelumnya kita tulis
untuk kelas `Sequential`
(seperti yang dijelaskan pada :numref:`sec_mlp`).


## [**Menjalankan Kode dalam Metode Forward Propagation**]

Kelas `Sequential` memudahkan konstruksi model,
memungkinkan kita untuk menyusun arsitektur baru
tanpa harus mendefinisikan kelas kita sendiri.
Namun, tidak semua arsitektur adalah rantai sederhana.
Ketika fleksibilitas lebih dibutuhkan,
kita akan ingin mendefinisikan blok kita sendiri.
Sebagai contoh, kita mungkin ingin mengeksekusi
aliran kontrol Python dalam metode forward propagation.
Selain itu, kita mungkin ingin melakukan
operasi matematika sembarang,
tidak hanya mengandalkan lapisan jaringan neural yang sudah ditentukan.

Anda mungkin telah memperhatikan bahwa hingga sekarang,
semua operasi dalam jaringan kita
telah bekerja pada aktivasi jaringan kita
dan parameter-parameter yang dimilikinya.
Namun, terkadang kita mungkin ingin
memasukkan istilah-istilah
yang bukan hasil dari lapisan sebelumnya
atau parameter yang dapat diperbarui.
Kita menyebutnya *parameter konstan*.
Misalnya, katakanlah kita menginginkan sebuah lapisan
yang menghitung fungsi
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
di mana $\mathbf{x}$ adalah input, $\mathbf{w}$ adalah parameter kita,
dan $c$ adalah beberapa konstanta yang ditentukan
yang tidak diperbarui selama optimisasi.
Jadi kita mengimplementasikan kelas `FixedHiddenMLP` seperti berikut.


```{.python .input}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # Parameter bobot acak yang dibuat dengan metode get_constant
        # tidak diperbarui selama pelatihan (yaitu, parameter konstan)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Gunakan parameter konstan yang telah dibuat, serta fungsi relu dan dot
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Gunakan kembali lapisan fully connected. Ini setara dengan berbagi
        # parameter antara dua lapisan fully connected
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameter bobot acak yang tidak akan menghitung gradien dan
        # karena itu tetap konstan selama pelatihan
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # Gunakan kembali lapisan fully connected. Ini setara dengan berbagi
        # parameter antara dua lapisan fully connected
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Parameter bobot acak yang dibuat dengan tf.constant tidak diperbarui
        # selama pelatihan (yaitu, parameter konstan)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Gunakan parameter konstan yang telah dibuat, serta fungsi relu dan
        # matmul
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Gunakan kembali lapisan fully connected. Ini setara dengan berbagi
        # parameter antara dua lapisan fully connected.
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
%%tab jax
class FixedHiddenMLP(nn.Module):
    # Parameter bobot acak yang tidak akan menghitung gradien dan
    # karena itu tetap konstan selama pelatihan.
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        # Gunakan kembali lapisan fully connected. Ini setara dengan berbagi
        # parameter antara dua lapisan fully connected.
        X = self.dense(X)
        # Control flow
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

Dalam model ini,
kami mengimplementasikan lapisan tersembunyi yang bobotnya
(`self.rand_weight`) diinisialisasi secara acak
pada saat instansiasi dan kemudian tetap konstan.
Bobot ini bukan parameter model
dan karenanya tidak pernah diperbarui oleh backpropagation.
Jaringan kemudian melewatkan output dari lapisan "tetap" ini
melalui lapisan fully connected.

Perhatikan bahwa sebelum mengembalikan output,
model kami melakukan sesuatu yang tidak biasa.
Kami menjalankan loop while, menguji
kondisi bahwa norma $\ell_1$-nya lebih besar dari $1$,
dan membagi vektor output kami dengan $2$
hingga memenuhi kondisi tersebut.
Akhirnya, kami mengembalikan jumlah entri dalam `X`.
Sejauh yang kami ketahui, tidak ada jaringan neural standar
yang melakukan operasi ini.
Perhatikan bahwa operasi khusus ini mungkin tidak berguna
dalam tugas dunia nyata manapun.
Poin kami hanya untuk menunjukkan bagaimana mengintegrasikan
kode sembarang ke dalam alur perhitungan
neural network Anda.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

```{.python .input}
%%tab jax
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

Kita dapat [**menggabungkan berbagai cara
untuk menyusun modul-modul bersama-sama.**]
Dalam contoh berikut, kita menempatkan modul-modul
dalam cara yang kreatif.


```{.python .input}
%%tab mxnet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab jax
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))


chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
chimera.apply(params, X)
```

## Ringkasan

Lapisan-lapisan individual dapat berupa modul.
Banyak lapisan dapat membentuk sebuah modul.
Banyak modul dapat membentuk sebuah modul.

Sebuah modul dapat berisi kode.
Modul menangani banyak pekerjaan administratif, termasuk inisialisasi parameter dan backpropagation.
Konkatenasi berurutan dari lapisan dan modul ditangani oleh modul `Sequential`.


## Latihan

1. Masalah apa yang akan terjadi jika Anda mengubah `MySequential` untuk menyimpan modul-modul dalam daftar Python?
2. Implementasikan sebuah modul yang menerima dua modul sebagai argumen, misalnya `net1` dan `net2`, dan mengembalikan output gabungan dari kedua jaringan dalam forward propagation. Ini juga disebut *parallel module*.
3. Misalkan Anda ingin menggabungkan beberapa instance dari jaringan yang sama. Implementasikan fungsi pembuat (factory function) yang menghasilkan beberapa instance dari modul yang sama dan bangun jaringan yang lebih besar dari modul tersebut.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/264)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17989)
:end_tab:
