```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implementasi Multilayer Perceptrons
:label:`sec_mlp-implementation`

Multilayer perceptrons (MLPs) tidak jauh lebih rumit untuk diimplementasikan dibandingkan dengan model linear sederhana. 
Perbedaan konseptual utama adalah bahwa kita sekarang menggabungkan beberapa lapisan.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

## Implementasi dari Awal

Mari kita mulai lagi dengan mengimplementasikan jaringan seperti ini dari awal.

### Inisialisasi Parameter Model

Ingat bahwa Fashion-MNIST memiliki 10 kelas,
dan setiap gambar terdiri dari grid pixel grayscale sebesar $28 \times 28 = 784$.
Seperti sebelumnya, kita akan mengabaikan struktur spasial
di antara pixel untuk saat ini,
sehingga kita dapat menganggapnya sebagai dataset klasifikasi
dengan 784 fitur input dan 10 kelas.
Untuk memulai, kita akan [**mengimplementasikan sebuah MLP
dengan satu hidden layer dan 256 unit tersembunyi.**]
Baik jumlah lapisan maupun lebar lapisan dapat disesuaikan
(keduanya dianggap sebagai hyperparameter).
Biasanya, kita memilih lebar lapisan agar dapat dibagi oleh pangkat dua yang lebih besar.
Ini efisien secara komputasi karena cara
memori dialokasikan dan diakses pada perangkat keras.

Sekali lagi, kita akan merepresentasikan parameter kita dengan beberapa tensor.
Perhatikan bahwa *untuk setiap lapisan*, kita harus melacak
satu matriks bobot dan satu vektor bias.
Seperti biasa, kita mengalokasikan memori
untuk gradien dari loss terhadap parameter ini.

:begin_tab:`mxnet`
Dalam kode di bawah ini, kita pertama-tama mendefinisikan dan menginisialisasi parameter
dan kemudian mengaktifkan pelacakan gradien.
:end_tab:

:begin_tab:`pytorch`
Dalam kode di bawah ini, kita menggunakan `nn.Parameter`
untuk secara otomatis mendaftarkan
atribut kelas sebagai parameter yang akan dilacak oleh `autograd` (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
Dalam kode di bawah ini, kita menggunakan `tf.Variable`
untuk mendefinisikan parameter model.
:end_tab:

:begin_tab:`jax`
Dalam kode di bawah ini, kita menggunakan `flax.linen.Module.param`
untuk mendefinisikan parameter model.
:end_tab:


```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### Model

Untuk memastikan kita memahami cara kerja setiap bagian,
kita akan [**mengimplementasikan fungsi aktivasi ReLU**] sendiri
alih-alih langsung menggunakan fungsi `relu` bawaan.


```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

Karena kita mengabaikan struktur spasial,
kita akan `reshape` setiap gambar dua dimensi menjadi
vektor datar dengan panjang `num_inputs`.
Terakhir, kita (**mengimplementasikan model kita**)
dengan hanya beberapa baris kode. Karena kita menggunakan autograd bawaan framework, inilah semua yang diperlukan.

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Training

Untungnya, [**loop pelatihan untuk MLP
sama persis seperti pada softmax regression.**] Kita mendefinisikan model, data, dan pelatih (trainer), lalu akhirnya memanggil metode `fit` pada model dan data.


```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Implementasi Singkat

Seperti yang mungkin Anda harapkan, dengan mengandalkan API tingkat tinggi, kita dapat mengimplementasikan MLP dengan lebih ringkas.

### Model

Dibandingkan dengan implementasi singkat
dari softmax regression
(:numref:`sec_softmax_concise`),
satu-satunya perbedaan adalah bahwa kita menambahkan
*dua* fully connected layers di mana sebelumnya kita hanya menambahkan *satu*.
Yang pertama adalah [**hidden layer**],
sedangkan yang kedua adalah output layer.


```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

Sebelumnya, kita mendefinisikan metode `forward` untuk model untuk mentransformasi input menggunakan parameter model.
Operasi ini pada dasarnya adalah sebuah pipeline:
kita mengambil sebuah input dan
menerapkan transformasi (misalnya,
perkalian matriks dengan bobot diikuti dengan penambahan bias),
kemudian secara berulang menggunakan output dari transformasi saat ini sebagai
input untuk transformasi berikutnya.
Namun, Anda mungkin memperhatikan bahwa 
tidak ada metode `forward` yang didefinisikan di sini.
Faktanya, `MLP` mewarisi metode `forward` dari kelas `Module` (:numref:`subsec_oo-design-models`) untuk
cukup memanggil `self.net(X)` (`X` adalah input),
yang sekarang didefinisikan sebagai rangkaian transformasi
melalui kelas `Sequential`.
Kelas `Sequential` mengabstraksi proses forward
sehingga kita dapat fokus pada transformasi-transformasi tersebut.
Kita akan membahas lebih lanjut cara kerja kelas `Sequential` di :numref:`subsec_model-construction-sequential`.

### Training

[**Loop pelatihan**] persis sama
seperti saat kita mengimplementasikan softmax regression.
Modularitas ini memungkinkan kita untuk memisahkan
permasalahan terkait arsitektur model
dari pertimbangan lain yang terpisah.


```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Summary

Sekarang kita telah memiliki lebih banyak latihan dalam merancang jaringan yang dalam, langkah dari satu lapisan menjadi beberapa lapisan jaringan dalam tidak lagi menjadi tantangan yang begitu signifikan. Secara khusus, kita dapat menggunakan kembali algoritma pelatihan dan data loader. Namun, perlu dicatat bahwa mengimplementasikan MLP dari awal tetaplah rumit: memberi nama dan melacak parameter model membuatnya sulit untuk memperluas model. Misalnya, bayangkan ingin menambahkan lapisan lain di antara lapisan 42 dan 43. Lapisan ini mungkin menjadi lapisan 42b, kecuali jika kita bersedia melakukan penamaan ulang secara berurutan. Selain itu, jika kita mengimplementasikan jaringan dari awal, akan lebih sulit bagi framework untuk melakukan optimisasi kinerja yang bermakna.

Namun demikian, Anda kini telah mencapai tingkat teknologi tercanggih pada akhir 1980-an ketika fully connected deep networks adalah metode pilihan untuk pemodelan jaringan saraf. Langkah konseptual kita berikutnya adalah mempertimbangkan gambar. Sebelum kita melakukannya, kita perlu meninjau sejumlah dasar statistik dan detail tentang cara menghitung model secara efisien.


## Exercises

1. Ubah jumlah unit tersembunyi `num_hiddens` dan buat plot bagaimana jumlah ini memengaruhi akurasi model. Berapakah nilai terbaik dari hyperparameter ini?
2. Coba tambahkan satu hidden layer untuk melihat bagaimana hal ini memengaruhi hasil.
3. Mengapa ide yang buruk untuk memasukkan hidden layer dengan satu neuron? Apa yang bisa salah?
4. Bagaimana perubahan pada learning rate memengaruhi hasil Anda? Dengan semua parameter lain tetap, learning rate mana yang memberikan hasil terbaik? Bagaimana ini terkait dengan jumlah epoch?
5. Mari kita optimalkan seluruh hyperparameter secara bersama-sama, yaitu, learning rate, jumlah epoch, jumlah hidden layer, dan jumlah unit tersembunyi per lapisan.
    1. Apa hasil terbaik yang bisa Anda dapatkan dengan mengoptimalkan semua hyperparameter ini?
    2. Mengapa menangani banyak hyperparameter jauh lebih menantang?
    3. Jelaskan strategi yang efisien untuk mengoptimalkan banyak parameter secara bersama-sama.
6. Bandingkan kecepatan framework dan implementasi dari awal untuk masalah yang menantang. Bagaimana perbedaannya seiring dengan kompleksitas jaringan?
7. Ukur kecepatan perkalian tensor--matriks untuk matriks yang selaras dan tidak selaras. Misalnya, uji matriks dengan dimensi 1024, 1025, 1026, 1028, dan 1032.
    1. Bagaimana perbedaan ini antara GPU dan CPU?
    2. Tentukan lebar bus memori CPU dan GPU Anda.
8. Cobalah berbagai fungsi aktivasi. Mana yang bekerja paling baik?
9. Apakah ada perbedaan antara inisialisasi bobot jaringan? Apakah ini penting?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17985)
:end_tab:
