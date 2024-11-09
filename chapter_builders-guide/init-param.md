```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Inisialisasi Parameter

Sekarang setelah kita tahu cara mengakses parameter,
mari kita lihat cara menginisialisasinya dengan benar.
Kita telah membahas pentingnya inisialisasi yang tepat di :numref:`sec_numerical_stability`.
Framework deep learning menyediakan inisialisasi acak default untuk lapisannya.
Namun, sering kali kita ingin menginisialisasi bobot
sesuai dengan berbagai protokol lainnya. Framework ini menyediakan protokol-protokol yang paling umum digunakan, dan juga memungkinkan kita untuk membuat inisialisasi kustom.


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

:begin_tab:`mxnet`
Secara default, MXNet menginisialisasi parameter bobot dengan menggambar secara acak dari distribusi uniform $U(-0.07, 0.07)$,
dan mengatur parameter bias ke nol.
Modul `init` pada MXNet menyediakan berbagai metode inisialisasi yang telah disediakan.
:end_tab:

:begin_tab:`pytorch`
Secara default, PyTorch menginisialisasi matriks bobot dan bias
secara uniform dengan menggambar dari rentang yang dihitung sesuai dengan dimensi input dan output.
Modul `nn.init` pada PyTorch menyediakan berbagai metode inisialisasi yang telah disediakan.
:end_tab:

:begin_tab:`tensorflow`
Secara default, Keras menginisialisasi matriks bobot secara uniform dengan menggambar dari rentang yang dihitung sesuai dengan dimensi input dan output, dan semua parameter bias diatur ke nol.
TensorFlow menyediakan berbagai metode inisialisasi baik di modul utama maupun di modul `keras.initializers`.
:end_tab:

:begin_tab:`jax`
Secara default, Flax menginisialisasi bobot menggunakan `jax.nn.initializers.lecun_normal`,
yaitu dengan menggambar sampel dari distribusi normal terpotong yang berpusat pada 0 dengan
standar deviasi yang ditetapkan sebagai akar kuadrat dari $1 / \textrm{fan}_{\textrm{in}}$
di mana `fan_in` adalah jumlah unit input dalam tensor bobot. Parameter bias
diatur semuanya ke nol.
Modul `nn.initializers` pada Jax menyediakan berbagai metode inisialisasi yang telah disediakan.
:end_tab:


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
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
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

## [**Inisialisasi Bawaan**]

Mari kita mulai dengan memanggil inisialisasi bawaan.
Kode di bawah ini menginisialisasi semua parameter bobot
sebagai variabel acak Gaussian
dengan standar deviasi 0.01, sementara parameter bias diatur ke nol.


```{.python .input}
%%tab mxnet
# Di sini force_reinit memastikan bahwa parameter diinisialisasi ulang meskipun
# parameter tersebut sudah diinisialisasi sebelumnya
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

Kita juga dapat menginisialisasi semua parameter ke nilai konstan tertentu (misalnya, 1).

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**Kita juga dapat menerapkan inisialisasi yang berbeda untuk blok tertentu.**]  
Sebagai contoh, di bawah ini kita menginisialisasi lapisan pertama  
dengan inisialisasi Xavier  
dan menginisialisasi lapisan kedua  
dengan nilai konstan 42.


```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**Inisialisasi Kustom**]

Terkadang, metode inisialisasi yang kita butuhkan
tidak disediakan oleh framework deep learning.
Dalam contoh di bawah ini, kita mendefinisikan inisialisasi
untuk setiap parameter bobot $w$ menggunakan distribusi unik berikut:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ dengan probabilitas } \frac{1}{4} \\
            0    & \textrm{ dengan probabilitas } \frac{1}{2} \\
        U(-10, -5) & \textrm{ dengan probabilitas } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Di sini kita mendefinisikan subclass dari kelas `Initializer`.
Biasanya, kita hanya perlu mengimplementasikan fungsi `_init_weight`
yang mengambil argumen tensor (`data`)
dan menetapkan nilai-nilai yang diinginkan padanya.
:end_tab:

:begin_tab:`pytorch`
Sekali lagi, kita mengimplementasikan fungsi `my_init` untuk diterapkan pada `net`.
:end_tab:

:begin_tab:`tensorflow`
Di sini kita mendefinisikan subclass dari `Initializer` dan mengimplementasikan fungsi `__call__`
yang mengembalikan tensor yang diinginkan berdasarkan bentuk dan tipe data yang diberikan.
:end_tab:

:begin_tab:`jax`
Fungsi inisialisasi Jax menerima `PRNGKey`, `shape`, dan `dtype` sebagai argumen.
Di sini kita mengimplementasikan fungsi `my_init` yang mengembalikan tensor yang diinginkan berdasarkan bentuk dan tipe data yang diberikan.
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
Perhatikan bahwa kita selalu memiliki opsi
untuk mengatur parameter secara langsung.
:end_tab:

:begin_tab:`jax`
Saat menginisialisasi parameter di JAX dan Flax, dictionary parameter yang dikembalikan
memiliki tipe `flax.core.frozen_dict.FrozenDict`. Tidak disarankan dalam ekosistem JAX
untuk mengubah nilai array secara langsung, oleh karena itu tipe data umumnya bersifat immutable.
Anda dapat menggunakan `params.unfreeze()` untuk melakukan perubahan.
:end_tab:


```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Ringkasan

Kita dapat menginisialisasi parameter menggunakan inisialisasi bawaan dan inisialisasi kustom.

## Latihan

Cari dokumentasi online untuk lebih banyak inisialisasi bawaan.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17991)
:end_tab:
