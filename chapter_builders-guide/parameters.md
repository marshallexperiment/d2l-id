```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Manajemen Parameter

Setelah kita memilih sebuah arsitektur
dan menetapkan hyperparameter kita,
kita lanjut ke loop pelatihan,
di mana tujuan kita adalah menemukan nilai parameter
yang meminimalkan fungsi loss kita.
Setelah pelatihan selesai, kita akan memerlukan parameter-parameter ini
untuk membuat prediksi di masa depan.
Selain itu, terkadang kita ingin
mengekstrak parameter-parameter tersebut
mungkin untuk digunakan kembali dalam konteks lain,
menyimpan model kita ke disk sehingga
dapat dijalankan di perangkat lunak lain,
atau untuk diperiksa dengan harapan
mendapatkan pemahaman ilmiah.

Sebagian besar waktu, kita dapat
mengabaikan detail-detail teknis
tentang bagaimana parameter dideklarasikan
dan dimanipulasi, dengan mengandalkan framework deep learning
untuk menangani bagian yang rumit.
Namun, ketika kita bergerak menjauh dari
arsitektur bertumpuk dengan lapisan-lapisan standar,
terkadang kita perlu mendalami detail
tentang deklarasi dan manipulasi parameter.
Dalam bagian ini, kita akan membahas hal-hal berikut:

* Mengakses parameter untuk debugging, diagnosis, dan visualisasi.
* Berbagi parameter di berbagai komponen model yang berbeda.


```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

(**Kita mulai dengan berfokus pada MLP dengan satu lapisan tersembunyi.**)


```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**Akses Parameter**]
:label:`subsec_param-access`

Mari kita mulai dengan cara mengakses parameter
dari model yang sudah Anda ketahui.

:begin_tab:`mxnet, pytorch, tensorflow`
Ketika sebuah model didefinisikan melalui kelas `Sequential`,
kita dapat mengakses setiap lapisan dengan melakukan indexing
ke dalam model seolah-olah itu adalah sebuah list.
Parameter setiap lapisan terletak dengan mudah
pada atributnya.
:end_tab:

:begin_tab:`jax`
Flax dan JAX memisahkan model dan parameter seperti yang
mungkin telah Anda amati pada model yang didefinisikan sebelumnya.
Ketika sebuah model didefinisikan melalui kelas `Sequential`,
kita pertama-tama perlu menginisialisasi jaringan untuk menghasilkan
dictionary parameter. Kita dapat mengakses
parameter dari setiap lapisan melalui kunci pada dictionary ini.
:end_tab:

Kita dapat memeriksa parameter dari lapisan fully connected kedua sebagai berikut.


```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

We can see that this fully connected layer
contains two parameters,
corresponding to that layer's
weights and biases, respectively.


### [**Parameter yang Ditargetkan**]

Perhatikan bahwa setiap parameter direpresentasikan
sebagai sebuah instance dari kelas parameter.
Untuk melakukan sesuatu yang berguna dengan parameter-parameter ini,
kita pertama-tama perlu mengakses nilai numerik dasarnya.
Ada beberapa cara untuk melakukannya.
Beberapa cara lebih sederhana sementara yang lain lebih umum.
Kode berikut mengekstrak bias
dari lapisan kedua jaringan neural, yang mengembalikan sebuah instance kelas parameter,
dan kemudian mengakses nilai dari parameter tersebut.


```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet, pytorch`
Parameter adalah objek yang kompleks,
mengandung nilai, gradien,
dan informasi tambahan lainnya.
Itulah sebabnya kita perlu meminta nilainya secara eksplisit.

Selain nilai, setiap parameter juga memungkinkan kita untuk mengakses gradien. Karena kita belum melakukan backpropagation untuk jaringan ini, gradien masih dalam keadaan awal.
:end_tab:

:begin_tab:`jax`
Berbeda dengan framework lainnya, JAX tidak melacak gradien dari
parameter jaringan neural, melainkan parameter dan jaringan dipisahkan.
JAX memungkinkan pengguna mengekspresikan perhitungan mereka sebagai
fungsi Python, dan menggunakan transformasi `grad` untuk tujuan yang sama.
:end_tab:


```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**Semua Parameter Sekaligus**]

Ketika kita perlu melakukan operasi pada semua parameter,
mengaksesnya satu per satu bisa menjadi membosankan.
Situasi ini bisa menjadi sangat merepotkan
ketika kita bekerja dengan modul yang lebih kompleks, misalnya, modul yang bersarang,
karena kita perlu menelusuri seluruh pohon
untuk mengekstrak parameter dari setiap sub-modul. Di bawah ini kami menunjukkan cara mengakses parameter dari semua lapisan.


```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**Parameter yang Terikat**]

Seringkali, kita ingin berbagi parameter di beberapa lapisan.
Mari kita lihat cara melakukannya dengan elegan.
Di bawah ini kita mengalokasikan sebuah lapisan fully connected
dan kemudian menggunakan parameter-parameter tersebut secara spesifik
untuk menetapkan parameter-parameter pada lapisan lain.
Di sini kita perlu menjalankan propagasi maju
`net(X)` sebelum mengakses parameter-parameter tersebut.


```{.python .input}
%%tab mxnet
net = nn.Sequential()
# Kita perlu memberi nama pada layer yang akan dibagikan agar kita bisa merujuk ke parameternya
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Memeriksa apakah parameter-parameter tersebut sama
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Memastikan bahwa parameter tersebut benar-benar objek yang sama, bukan hanya memiliki nilai yang sama
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# Kita perlu memberi nama pada layer yang akan dibagikan agar kita bisa merujuk ke parameternya
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Memeriksa apakah parameter-parameter tersebut sama
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Memastikan bahwa parameter tersebut benar-benar objek yang sama, bukan hanya memiliki nilai yang sama
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras berperilaku sedikit berbeda. Ini secara otomatis menghapus layer duplikat
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Memeriksa apakah parameter-parameter berbeda
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# Kita perlu memberi nama pada layer yang akan dibagikan agar kita bisa merujuk ke parameternya
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Memeriksa apakah parameter-parameter berbeda
print(len(params['params']) == 3)
```

Contoh ini menunjukkan bahwa parameter
dari lapisan kedua dan ketiga terikat.
Parameter-parameter tersebut tidak hanya sama, tetapi
diwakili oleh tensor yang sama persis.
Dengan demikian, jika kita mengubah salah satu dari parameter tersebut,
parameter yang lain juga ikut berubah.

:begin_tab:`mxnet, pytorch, tensorflow`
Anda mungkin bertanya-tanya,
ketika parameter-parameter terikat,
apa yang terjadi pada gradiennya?
Karena parameter model mengandung gradien,
gradien dari lapisan tersembunyi kedua
dan lapisan tersembunyi ketiga akan dijumlahkan
selama proses backpropagation.
:end_tab:


## Ringkasan

Kita memiliki beberapa cara untuk mengakses dan mengikat parameter model.


## Latihan

1. Gunakan model `NestMLP` yang didefinisikan dalam :numref:`sec_model_construction` dan akses parameter dari berbagai lapisan.
2. Bangun sebuah MLP yang mengandung lapisan parameter bersama dan latih model tersebut. Selama proses pelatihan, amati parameter model dan gradien dari setiap lapisan.
3. Mengapa berbagi parameter merupakan ide yang baik?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17990)
:end_tab:
