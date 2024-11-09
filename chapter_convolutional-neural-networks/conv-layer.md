```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Konvolusi untuk Gambar
:label:`sec_conv_layer`

Sekarang setelah kita memahami cara kerja lapisan konvolusi secara teori,
kita siap untuk melihat cara kerjanya dalam praktik.
Berdasarkan motivasi kita tentang jaringan saraf konvolusi
sebagai arsitektur yang efisien untuk mengeksplorasi struktur dalam data gambar,
kita akan tetap menggunakan gambar sebagai contoh utama kita.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Operasi _Cross-Correlation_

Ingat bahwa secara ketat, lapisan konvolusi
adalah istilah yang kurang tepat, karena operasi yang diekspresikan
lebih akurat disebut sebagai korelasi silang.
Berdasarkan deskripsi kita tentang lapisan konvolusi di :numref:`sec_why-conv`,
pada lapisan seperti ini, tensor input
dan tensor kernel digabungkan
untuk menghasilkan tensor output melalui sebuah (**operasi korelasi silang.**)

Mari kita abaikan kanal untuk saat ini dan lihat cara kerjanya
dengan data dua dimensi dan representasi tersembunyi.
Pada :numref:`fig_correlation`,
input adalah tensor dua dimensi
dengan tinggi 3 dan lebar 3.
Kita menandai bentuk tensor sebagai $3 \times 3$ atau ($3$, $3$).
Tinggi dan lebar kernel masing-masing adalah 2.
Bentuk dari *jendela kernel* (atau *jendela konvolusi*)
ditentukan oleh tinggi dan lebar kernel
(di sini adalah $2 \times 2$).

![Operasi korelasi silang dua dimensi. Bagian yang diarsir adalah elemen output pertama serta elemen tensor input dan kernel yang digunakan untuk perhitungan output: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

Pada operasi korelasi silang dua dimensi,
kita memulai dengan posisi jendela konvolusi
di sudut kiri atas dari tensor input
dan menggesernya melintasi tensor input,
baik dari kiri ke kanan maupun dari atas ke bawah.
Ketika jendela konvolusi bergeser ke posisi tertentu,
subtensor input yang berada di dalam jendela tersebut
dan tensor kernel dikalikan secara elementwise,
kemudian hasil tensor tersebut dijumlahkan
untuk menghasilkan satu nilai skalar.
Hasil ini memberikan nilai tensor output
pada lokasi yang sesuai.
Di sini, tensor output memiliki tinggi 2 dan lebar 2,
dan keempat elemennya diperoleh dari
operasi korelasi silang dua dimensi:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Perhatikan bahwa sepanjang setiap sumbu, ukuran output
sedikit lebih kecil dari ukuran input.
Karena kernel memiliki lebar dan tinggi lebih besar dari $1$,
kita hanya dapat menghitung korelasi silang secara benar
untuk lokasi-lokasi di mana kernel sepenuhnya berada di dalam gambar,
ukuran output diberikan oleh ukuran input $n_\textrm{h} \times n_\textrm{w}$
dikurangi ukuran kernel konvolusi $k_\textrm{h} \times k_\textrm{w}$
melalui rumus

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$

Ini terjadi karena kita membutuhkan ruang yang cukup
untuk "menggeser" kernel konvolusi melintasi gambar.
Nantinya, kita akan melihat cara mempertahankan ukuran agar tetap sama
dengan menambahkan padding berupa nol di sekitar batas gambar
sehingga ada cukup ruang untuk menggeser kernel.
Selanjutnya, kita mengimplementasikan proses ini dalam fungsi `corr2d`,
yang menerima tensor input `X` dan tensor kernel `K`
dan mengembalikan tensor output `Y`.


```{.python .input}
%%tab mxnet
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab jax
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y = Y.at[i, j].set((X[i:i + h, j:j + w] * K).sum())
    return Y
```

```{.python .input}
%%tab tensorflow
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

Kita dapat membangun tensor input `X` dan tensor kernel `K`
dari :numref:`fig_correlation`
untuk [**memvalidasi output dari implementasi di atas**]
dalam operasi korelasi silang dua dimensi.


```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Lapisan Konvolusi

Lapisan konvolusi melakukan korelasi silang antara input dan kernel
serta menambahkan bias skalar untuk menghasilkan output.
Dua parameter dari lapisan konvolusi adalah kernel dan bias skalar.
Saat melatih model berbasis lapisan konvolusi,
biasanya kita menginisialisasi kernel secara acak,
seperti yang kita lakukan pada lapisan fully connected.

Sekarang kita siap untuk [**mengimplementasikan lapisan konvolusi dua dimensi**]
berdasarkan fungsi `corr2d` yang telah didefinisikan sebelumnya.
Dalam metode konstruktor `__init__`,
kita mendeklarasikan `weight` dan `bias` sebagai dua parameter model.
Metode propagasi maju memanggil fungsi `corr2d` dan menambahkan bias.

```{.python .input}
%%tab mxnet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
%%tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

```{.python .input}
%%tab jax
class Conv2D(nn.Module):
    kernel_size: int

    def setup(self):
        self.weight = nn.param('w', nn.initializers.uniform, self.kernel_size)
        self.bias = nn.param('b', nn.initializers.zeros, 1)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

Pada konvolusi $h \times w$
atau kernel konvolusi $h \times w$,
tinggi dan lebar kernel konvolusi masing-masing adalah $h$ dan $w$.
Kita juga menyebut lapisan konvolusi dengan kernel konvolusi $h \times w$
sebagai lapisan konvolusi $h \times w$.

## Deteksi Tepi Objek dalam Gambar

Mari kita pahami [**aplikasi sederhana dari lapisan konvolusi:
mendeteksi tepi objek dalam gambar**]
dengan menemukan lokasi perubahan piksel.
Pertama, kita membangun sebuah "gambar" dengan ukuran $6\times 8$ piksel.
Empat kolom tengahnya berwarna hitam ($0$) dan sisanya berwarna putih ($1$).


```{.python .input}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
%%tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

```{.python .input}
%%tab jax
X = jnp.ones((6, 8))
X = X.at[:, 2:6].set(0)
X
```

Selanjutnya, kita membangun kernel `K` dengan tinggi 1 dan lebar 2.
Saat kita melakukan operasi korelasi silang dengan input,
jika elemen-elemen yang berdekatan secara horizontal memiliki nilai yang sama,
outputnya adalah 0. Jika tidak, outputnya bernilai non-nol.
Perhatikan bahwa kernel ini adalah kasus khusus dari operator beda hingga. Pada lokasi $(i,j)$, kernel ini menghitung $x_{i,j} - x_{(i+1),j}$, 
yaitu menghitung perbedaan antara nilai piksel yang berdekatan secara horizontal. Ini adalah pendekatan diskret dari turunan pertama dalam arah horizontal.
Sebab, untuk fungsi $f(i,j)$, turunannya $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$. Mari kita lihat cara kerjanya dalam praktik.


```{.python .input}
%%tab all
K = d2l.tensor([[1.0, -1.0]])
```

Kita siap untuk melakukan operasi korelasi silang
dengan argumen `X` (input kita) dan `K` (kernel kita).
Seperti yang dapat Anda lihat, [**kita mendeteksi $1$ untuk tepi dari putih ke hitam
dan $-1$ untuk tepi dari hitam ke putih.**]
Semua output lainnya bernilai $0$.


```{.python .input}
%%tab all
Y = corr2d(X, K)
Y
```

Sekarang kita dapat menerapkan kernel pada gambar yang telah ditransposisi.
Seperti yang diharapkan, hasilnya hilang. [**Kernel `K` hanya mendeteksi vertikal edges.**]


```{.python .input}
%%tab all
corr2d(d2l.transpose(X), K)
```

## Mempelajari Kernel

Merancang detektor tepi menggunakan beda hingga `[1, -1]` sangat efektif
jika kita tahu persis itulah yang kita cari.
Namun, ketika kita menggunakan kernel yang lebih besar,
dan mempertimbangkan lapisan konvolusi berturut-turut,
mungkin mustahil untuk menentukan
secara manual apa yang seharusnya dilakukan setiap filter.

Sekarang mari kita lihat apakah kita bisa [**mempelajari kernel yang menghasilkan `Y` dari `X`**]
dengan hanya melihat pasangan input-output.
Pertama, kita membangun lapisan konvolusi
dan menginisialisasi kernel sebagai tensor acak.
Selanjutnya, pada setiap iterasi, kita akan menggunakan error kuadrat
untuk membandingkan `Y` dengan output dari lapisan konvolusi.
Kemudian, kita bisa menghitung gradien untuk memperbarui kernel.
Demi kesederhanaan,
dalam contoh berikut
kita menggunakan kelas bawaan
untuk lapisan konvolusi dua dimensi
dan mengabaikan bias.


```{.python .input}
%%tab mxnet
# Membangun lapisan konvolusi dua dimensi dengan 1 kanal output dan
# kernel berukuran (1, 2). Untuk kesederhanaan, kita mengabaikan bias di sini
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# Lapisan konvolusi dua dimensi menggunakan input dan output berdimensi empat
# dalam format (contoh, kanal, tinggi, lebar), di mana ukuran batch 
# (jumlah contoh dalam batch) dan jumlah kanal masing-masing adalah 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Laju pembelajaran

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Memperbarui kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
%%tab pytorch
# Membangun lapisan konvolusi dua dimensi dengan 1 kanal output dan
# kernel berukuran (1, 2). Untuk kesederhanaan, kita mengabaikan bias di sini
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# Lapisan konvolusi dua dimensi menggunakan input dan output berdimensi empat
# dalam format (contoh, kanal, tinggi, lebar), di mana ukuran batch 
# (jumlah contoh dalam batch) dan jumlah kanal masing-masing adalah 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Laju pembelajaran

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Memperbarui kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
%%tab tensorflow
# Membangun lapisan konvolusi dua dimensi dengan 1 kanal output dan
# kernel berukuran (1, 2). Untuk kesederhanaan, kita mengabaikan bias di sini
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# Lapisan konvolusi dua dimensi menggunakan input dan output berdimensi empat
# dalam format (contoh, tinggi, lebar, kanal), di mana ukuran batch 
# (jumlah contoh dalam batch) dan jumlah kanal masing-masing adalah 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Laju pembelajaran

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Memperbarui kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

```{.python .input}
%%tab jax
# Membangun lapisan konvolusi dua dimensi dengan 1 kanal output dan
# kernel berukuran (1, 2). Untuk kesederhanaan, kita mengabaikan bias di sini
conv2d = nn.Conv(1, kernel_size=(1, 2), use_bias=False, padding='VALID')

# Lapisan konvolusi dua dimensi menggunakan input dan output berdimensi empat
# dalam format (contoh, tinggi, lebar, kanal), di mana ukuran batch 
# (jumlah contoh dalam batch) dan jumlah kanal masing-masing adalah 1
X = X.reshape((1, 6, 8, 1))
Y = Y.reshape((1, 6, 7, 1))
lr = 3e-2  # Laju pembelajaran

params = conv2d.init(jax.random.PRNGKey(d2l.get_seed()), X)

def loss(params, X, Y):
    Y_hat = conv2d.apply(params, X)
    return ((Y_hat - Y) ** 2).sum()

for i in range(10):
    l, grads = jax.value_and_grad(loss)(params, X, Y)
    # Memperbarui kernel
    params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l:.3f}')
```

Perhatikan bahwa error telah turun ke nilai yang kecil setelah 10 iterasi. Sekarang kita akan [**melihat tensor kernel yang telah kita pelajari.**]

```{.python .input}
%%tab mxnet
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
%%tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

```{.python .input}
%%tab jax
params['params']['kernel'].reshape((1, 2))
```

Memang, tensor kernel yang dipelajari sangat mendekati
tensor kernel `K` yang kita definisikan sebelumnya.

## Korelasi Silang dan Konvolusi

Ingat pengamatan kita dari :numref:`sec_why-conv` mengenai hubungan
antara operasi korelasi silang dan konvolusi.
Di sini, mari kita lanjutkan dengan mempertimbangkan lapisan konvolusi dua dimensi.
Bagaimana jika lapisan-lapisan tersebut
melakukan operasi konvolusi yang ketat
seperti yang didefinisikan pada :eqref:`eq_2d-conv-discrete`
alih-alih korelasi silang?
Untuk mendapatkan output dari operasi *konvolusi* yang ketat, kita hanya perlu membalik tensor kernel dua dimensi secara horizontal dan vertikal, lalu melakukan operasi *korelasi silang* dengan tensor input.

Perlu dicatat bahwa karena kernel dipelajari dari data dalam deep learning,
output dari lapisan konvolusi tetap tidak terpengaruh
apakah lapisan tersebut
melakukan operasi konvolusi yang ketat
atau operasi korelasi silang.

Untuk mengilustrasikan ini, misalkan sebuah lapisan konvolusi melakukan *korelasi silang* dan mempelajari kernel pada :numref:`fig_correlation`, yang di sini disebut sebagai matriks $\mathbf{K}$.
Dengan asumsi bahwa kondisi lain tetap tidak berubah,
ketika lapisan ini melakukan *konvolusi* yang ketat,
kernel yang dipelajari $\mathbf{K}'$ akan sama dengan $\mathbf{K}$
setelah $\mathbf{K}'$
dibalik secara horizontal dan vertikal.
Dengan kata lain,
ketika lapisan konvolusi
melakukan *konvolusi* yang ketat
untuk input pada :numref:`fig_correlation`
dan $\mathbf{K}'$,
output yang sama seperti pada :numref:`fig_correlation`
(korelasi silang dari input dan $\mathbf{K}$)
akan diperoleh.

Sejalan dengan terminologi standar dalam literatur deep learning,
kita akan tetap merujuk operasi korelasi silang
sebagai konvolusi meskipun, secara ketat, terdapat sedikit perbedaan.
Selain itu,
kita menggunakan istilah *elemen* untuk merujuk pada
entri (atau komponen) dari tensor apa pun yang mewakili representasi lapisan atau kernel konvolusi.



## Feature Map dan Receptive Field

Seperti yang dijelaskan pada :numref:`subsec_why-conv-channels`,
output dari lapisan konvolusi pada
:numref:`fig_correlation`
kadang disebut sebagai *feature map*,
karena dapat dianggap sebagai
representasi (fitur) yang dipelajari
dalam dimensi spasial (misalnya, lebar dan tinggi)
untuk lapisan berikutnya.
Dalam CNN,
untuk setiap elemen $x$ pada suatu lapisan,
*receptive field* dari $x$ mengacu pada
semua elemen (dari semua lapisan sebelumnya)
yang dapat mempengaruhi perhitungan $x$
selama propagasi maju.
Perhatikan bahwa receptive field
mungkin lebih besar daripada ukuran input yang sebenarnya.

Mari kita gunakan lagi :numref:`fig_correlation` untuk menjelaskan receptive field.
Diberikan kernel konvolusi $2 \times 2$,
receptive field dari elemen output yang diarsir (bernilai $19$)
adalah empat elemen dalam bagian input yang diarsir.
Sekarang misalkan output $2 \times 2$
disebut $\mathbf{Y}$
dan kita mempertimbangkan CNN yang lebih dalam
dengan lapisan konvolusi $2 \times 2$ tambahan yang menerima $\mathbf{Y}$
sebagai input, dan menghasilkan
satu elemen $z$.
Dalam kasus ini,
receptive field dari $z$ pada $\mathbf{Y}$ mencakup keempat elemen dari $\mathbf{Y}$,
sedangkan
receptive field
pada input mencakup semua sembilan elemen input.
Dengan demikian,
ketika elemen mana pun dalam feature map
memerlukan receptive field yang lebih besar
untuk mendeteksi fitur input di area yang lebih luas,
kita dapat membangun jaringan yang lebih dalam.

Istilah receptive field berasal dari neurofisiologi.
Serangkaian eksperimen pada berbagai hewan menggunakan stimulus berbeda
:cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968` meneliti respons korteks visual
terhadap stimulus tersebut. Secara umum, ditemukan bahwa level yang lebih rendah merespons tepi dan bentuk terkait.
Kemudian, :citet:`Field.1987` mengilustrasikan efek ini pada gambar alami dengan, apa yang bisa disebut, kernel konvolusi.
Kita mencetak ulang gambar kunci di :numref:`field_visual` untuk menunjukkan kesamaan yang mencolok.

![Gambar dan keterangan diambil dari :citet:`Field.1987`: Contoh pengkodean dengan enam kanal berbeda. (Kiri) Contoh dari enam jenis sensor yang terkait dengan setiap kanal. (Kanan) Konvolusi dari gambar di (Tengah) dengan enam sensor yang ditunjukkan di (Kiri). Respons dari sensor individu ditentukan dengan mengambil sampel dari gambar yang difilter ini pada jarak proporsional dengan ukuran sensor (ditunjukkan dengan titik-titik). Diagram ini menunjukkan respons hanya untuk sensor simetris genap.](../img/field-visual.png)
:label:`field_visual`

Ternyata, hubungan ini bahkan berlaku untuk fitur yang dihitung oleh lapisan lebih dalam pada jaringan yang dilatih untuk tugas klasifikasi gambar, seperti yang ditunjukkan, misalnya, oleh :citet:`Kuzovkin.Vicente.Petton.ea.2018`. Singkatnya, konvolusi telah terbukti menjadi alat yang sangat kuat untuk computer vision, baik dalam biologi maupun dalam kode. Oleh karena itu, tidak mengherankan (dalam retrospeksi) bahwa konvolusi menjadi kunci kesuksesan dalam deep learning baru-baru ini.

## Ringkasan

Perhitungan inti yang diperlukan untuk lapisan konvolusi adalah operasi korelasi silang. Kita melihat bahwa hanya diperlukan nested for-loop sederhana untuk menghitung nilainya. Jika kita memiliki beberapa kanal input dan beberapa kanal output, kita melakukan operasi matriks-matriks antara kanal-kanal tersebut. Seperti yang terlihat, perhitungannya sederhana dan, yang paling penting, sangat *lokal*. Hal ini memungkinkan optimasi perangkat keras yang signifikan dan banyak hasil terbaru dalam computer vision hanya mungkin karena hal tersebut. Bagaimanapun, ini berarti bahwa desainer chip dapat berinvestasi dalam komputasi cepat daripada memori ketika mengoptimalkan untuk konvolusi. Meskipun ini mungkin tidak menghasilkan desain optimal untuk aplikasi lain, ini membuka pintu bagi computer vision yang mudah diakses dan terjangkau.

Dalam hal konvolusi itu sendiri, konvolusi dapat digunakan untuk banyak tujuan, misalnya mendeteksi tepi dan garis, mengaburkan gambar, atau mempertajamnya. Yang terpenting, tidak perlu ahli statistik (atau insinyur) yang merancang filter yang cocok. Sebaliknya, kita dapat *mempelajarinya* dari data. Ini menggantikan heuristik rekayasa fitur dengan statistik berbasis bukti. Terakhir, dan yang cukup menyenangkan, filter ini tidak hanya menguntungkan untuk membangun jaringan dalam tetapi juga sesuai dengan receptive fields dan feature maps di otak. Ini memberi kita keyakinan bahwa kita berada di jalur yang benar.

## Latihan

1. Buat gambar `X` dengan tepi diagonal.
    1. Apa yang terjadi jika Anda menerapkan kernel `K` dalam bagian ini padanya?
    1. Apa yang terjadi jika Anda mentransposisi `X`?
    1. Apa yang terjadi jika Anda mentransposisi `K`?
1. Rancang beberapa kernel secara manual.
    1. Diberikan vektor arah $\mathbf{v} = (v_1, v_2)$, turunkan kernel deteksi tepi yang mendeteksi
       tepi yang ortogonal terhadap $\mathbf{v}$, yaitu, tepi dalam arah $(v_2, -v_1)$.
    1. Turunkan operator beda hingga untuk turunan kedua. Berapa ukuran minimum
       dari kernel konvolusi yang terkait dengannya? Struktur apa dalam gambar yang merespons paling kuat terhadapnya?
    1. Bagaimana Anda akan merancang kernel blur? Mengapa Anda ingin menggunakan kernel seperti itu?
    1. Berapa ukuran minimum kernel untuk memperoleh turunan dengan urutan $d$?
1. Ketika Anda mencoba menemukan gradien secara otomatis untuk kelas `Conv2D` yang kita buat, pesan error seperti apa yang Anda lihat?
1. Bagaimana Anda merepresentasikan operasi korelasi silang sebagai perkalian matriks dengan mengubah tensor input dan kernel?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/271)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17996)
:end_tab:
