```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```



# Aljabar Linear
:label:`sec_linear-algebra`

Sejauh ini, kita sudah bisa memuat dataset ke dalam tensor
dan memanipulasi tensor tersebut
dengan operasi matematika dasar.
Untuk mulai membangun model yang lebih canggih,
kita juga memerlukan beberapa alat dari aljabar linear.
Bagian ini memberikan pengenalan ringan
tentang konsep-konsep yang paling penting,
dimulai dari aritmatika skalar
hingga ke perkalian matriks.


```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Skalar

Sebagian besar matematika sehari-hari
terdiri dari memanipulasi
angka satu per satu.
Secara formal, kita menyebut nilai-nilai ini sebagai *skalar*.
Misalnya, suhu di Palo Alto
adalah 72 derajat Fahrenheit.
Jika Anda ingin mengonversi suhu ke Celsius,
Anda akan menghitung ekspresi
$c = \frac{5}{9}(f - 32)$, dengan menetapkan $f$ ke 72.
Dalam persamaan ini, nilai
$5$, $9$, dan $32$ adalah skalar konstan.
Variabel $c$ dan $f$
umumnya merepresentasikan skalar yang tidak diketahui.

Kita menandai skalar
dengan huruf kecil biasa
(misalnya, $x$, $y$, dan $z$)
dan ruang dari semua skalar
*bernilai real* (kontinu) dengan $\mathbb{R}$.
Untuk kemudahan, kita akan melewatkan
definisi yang ketat tentang *ruang*:
ingat saja bahwa ekspresi $x \in \mathbb{R}$
adalah cara formal untuk menyatakan bahwa $x$ adalah skalar bernilai real.
Simbol $\in$ (dibaca "di dalam")
menyatakan keanggotaan dalam suatu himpunan.
Sebagai contoh, $x, y \in \{0, 1\}$
menunjukkan bahwa $x$ dan $y$ adalah variabel
yang hanya dapat mengambil nilai $0$ atau $1$.

(**Skalar diimplementasikan sebagai tensor
yang hanya berisi satu elemen.**)
Di bawah ini, kita menetapkan dua skalar
dan melakukan operasi penjumlahan, perkalian,
pembagian, dan perpangkatan yang sudah dikenal.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## Vektor

Untuk saat ini, [**Anda dapat menganggap vektor sebagai array berdimensi tetap yang terdiri dari skalar.**]
Seperti pada padanan kode mereka,
kita menyebut skalar-skalar ini sebagai *elemen* vektor
(sinonim lainnya adalah *entri* dan *komponen*).
Ketika vektor merepresentasikan contoh dari dataset dunia nyata,
nilai-nilainya memiliki arti penting dalam konteks dunia nyata.
Misalnya, jika kita melatih model untuk memprediksi
risiko gagal bayar pinjaman,
kita mungkin mengaitkan setiap pemohon dengan sebuah vektor
yang komponennya sesuai dengan nilai-nilai
seperti pendapatan mereka, lama bekerja,
atau jumlah gagal bayar sebelumnya.
Jika kita mempelajari risiko serangan jantung,
setiap vektor mungkin mewakili seorang pasien
dan komponennya mungkin sesuai dengan
tanda vital terbaru mereka, kadar kolesterol,
menit latihan per hari, dll.
Kita menandai vektor dengan huruf kecil tebal,
(misalnya, $\mathbf{x}$, $\mathbf{y}$, dan $\mathbf{z}$).

Vektor diimplementasikan sebagai tensor berdimensi pertama ($1^{\textrm{st}}$-order).
Secara umum, tensor seperti ini dapat memiliki panjang sebarang,
tergantung pada batasan memori. Catatan: dalam Python, seperti dalam sebagian besar bahasa pemrograman, indeks vektor dimulai dari $0$, yang juga dikenal sebagai *zero-based indexing*, sedangkan dalam aljabar linear subskrip dimulai dari $1$ (one-based indexing).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```


```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

Kita dapat merujuk ke elemen suatu vektor dengan menggunakan subskrip.
Misalnya, $x_2$ menunjukkan elemen kedua dari $\mathbf{x}$.
Karena $x_2$ adalah skalar, kita tidak mencetaknya dengan huruf tebal.
Secara default, kita memvisualisasikan vektor
dengan menumpuk elemennya secara vertikal:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

Di sini $x_1, \ldots, x_n$ adalah elemen-elemen dari vektor tersebut.
Nantinya, kita akan membedakan antara *vektor kolom* seperti ini
dan *vektor baris* yang elemennya ditumpuk secara horizontal.
Ingat bahwa [**kita mengakses elemen-elemen tensor melalui indexing.**]


```{.python .input}
%%tab all
x[2]
```

Untuk menunjukkan bahwa sebuah vektor berisi $n$ elemen,
kita menuliskan $\mathbf{x} \in \mathbb{R}^n$.
Secara formal, kita menyebut $n$ sebagai *dimensi* dari vektor.
[**Dalam kode, ini sesuai dengan panjang tensor**],
yang dapat diakses melalui fungsi bawaan `len` dalam Python.


```{.python .input}
%%tab all
len(x)
```

Kita juga dapat mengakses panjang melalui atribut `shape`.
`Shape` adalah sebuah tuple yang menunjukkan panjang tensor di setiap sumbu.
(**Tensor dengan hanya satu sumbu memiliki shape dengan hanya satu elemen.**)


```{.python .input}
%%tab all
x.shape
```

Seringkali, kata "dimensi" memiliki dua makna, yaitu 
jumlah sumbu (axes) dan panjang sepanjang sumbu tertentu.
Untuk menghindari kebingungan ini,
kita menggunakan istilah *order* untuk merujuk pada jumlah sumbu
dan *dimensi* secara eksklusif untuk merujuk 
pada jumlah komponen.


## Matriks

Seperti halnya skalar adalah tensor berorde ke-0 ($0^{\textrm{th}}$-order)
dan vektor adalah tensor berorde pertama ($1^{\textrm{st}}$-order),
maka matriks adalah tensor berorde kedua ($2^{\textrm{nd}}$-order).
Kita menandai matriks dengan huruf kapital tebal
(misalnya, $\mathbf{X}$, $\mathbf{Y}$, dan $\mathbf{Z}$),
dan merepresentasikannya dalam kode sebagai tensor dengan dua sumbu.
Ekspresi $\mathbf{A} \in \mathbb{R}^{m \times n}$
menunjukkan bahwa matriks $\mathbf{A}$
berisi $m \times n$ skalar bernilai real,
yang disusun dalam $m$ baris dan $n$ kolom.
Ketika $m = n$, kita mengatakan bahwa matriks tersebut adalah *persegi*.
Secara visual, kita dapat menggambarkan matriks sebagai sebuah tabel.
Untuk merujuk pada elemen individu,
kita menggunakan subskrip pada indeks baris dan kolom, misalnya,
$a_{ij}$ adalah nilai yang termasuk dalam baris ke-$i$ dan kolom ke-$j$ dari $\mathbf{A}$:

$$
\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\ 
a_{21} & a_{22} & \cdots & a_{2n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{m1} & a_{m2} & \cdots & a_{mn} 
\end{bmatrix}.
$$:eqlabel:`eq_matrix_def`

Dalam kode, kita merepresentasikan matriks $\mathbf{A} \in \mathbb{R}^{m \times n}$
dengan tensor berorde kedua ($2^{\textrm{nd}}$-order) dengan bentuk ($m$, $n$).
[**Kita dapat mengonversi tensor berukuran $m \times n$ yang sesuai
menjadi matriks $m \times n**]
dengan memberikan bentuk yang diinginkan pada `reshape`:


```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

Kadang-kadang kita ingin membalikkan sumbu.
Ketika kita menukar baris dan kolom suatu matriks,
hasilnya disebut sebagai *transpose* dari matriks tersebut.
Secara formal, kita menandakan transpose matriks $\mathbf{A}$
dengan $\mathbf{A}^\top$ dan jika $\mathbf{B} = \mathbf{A}^\top$,
maka $b_{ij} = a_{ji}$ untuk semua $i$ dan $j$.
Jadi, transpose dari matriks $m \times n$
adalah matriks $n \times m$:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Dalam kode, kita dapat mengakses (**transpose dari matriks**) sebagai berikut:


```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**Matriks simetris adalah subset dari matriks persegi
yang sama dengan transposenya sendiri:
$\mathbf{A} = \mathbf{A}^\top$.**]
Matriks berikut ini adalah simetris:


```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```
Matriks sangat berguna untuk merepresentasikan dataset.
Biasanya, baris merepresentasikan catatan individu
dan kolom merepresentasikan atribut yang berbeda.


## Tensor

Meskipun Anda bisa melangkah jauh dalam perjalanan pembelajaran mesin Anda
dengan hanya skalar, vektor, dan matriks,
akhirnya Anda mungkin perlu bekerja dengan
[**tensor**] berorde lebih tinggi.
Tensor (**memberi kita cara generik untuk menggambarkan
perluasan ke array berorde- $n^{\textrm{th}}$.**)
Kami menyebut objek perangkat lunak dari *kelas tensor* sebagai "tensor"
karena mereka juga dapat memiliki jumlah sumbu yang sebarang.
Meskipun mungkin membingungkan menggunakan kata
*tensor* baik untuk objek matematika
maupun realisasinya dalam kode,
maknanya biasanya jelas dari konteks.
Kita menandai tensor umum dengan huruf kapital
dengan gaya font khusus
(misalnya, $\mathsf{X}$, $\mathsf{Y}$, dan $\mathsf{Z}$)
dan mekanisme pengindeksannya
(misalnya, $x_{ijk}$ dan $[\mathsf{X}]_{1, 2i-1, 3}$)
secara alami mengikuti mekanisme pengindeksan matriks.

Tensor akan menjadi lebih penting
saat kita mulai bekerja dengan gambar.
Setiap gambar muncul sebagai tensor berorde-3
dengan sumbu yang sesuai dengan tinggi, lebar, dan *channel*.
Pada setiap lokasi spasial, intensitas
masing-masing warna (merah, hijau, dan biru)
ditumpuk di sepanjang channel.
Selanjutnya, kumpulan gambar direpresentasikan
dalam kode sebagai tensor berorde-4,
di mana gambar yang berbeda diindeks
di sepanjang sumbu pertama.
Tensor berorde lebih tinggi dibangun, seperti halnya vektor dan matriks,
dengan menambah jumlah komponen shape.


```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## Properti Dasar Aritmatika Tensor

Skalar, vektor, matriks,
dan tensor berorde lebih tinggi
semua memiliki beberapa properti yang berguna.
Misalnya, operasi elemen demi elemen
menghasilkan output yang memiliki
shape yang sama dengan operannya.


```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```


[**Produk elemen demi elemen dari dua matriks
disebut sebagai *produk Hadamard***] (dilambangkan dengan $\odot$).
Kita dapat menuliskan entri-entri
dari produk Hadamard dua matriks
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**Menambahkan atau mengalikan skalar dan tensor**] menghasilkan hasil
dengan shape yang sama seperti tensor aslinya.
Di sini, setiap elemen tensor ditambahkan (atau dikalikan) dengan skalar.


```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## Reduksi
:label:`subsec_lin-alg-reduction`

Seringkali, kita ingin menghitung [**jumlah dari elemen-elemen tensor.**]
Untuk mengekspresikan jumlah elemen dalam vektor $\mathbf{x}$ dengan panjang $n$,
kita menuliskannya sebagai $\sum_{i=1}^n x_i$. Ada fungsi sederhana untuk itu:


```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

Untuk mengekspresikan [**jumlah elemen dari tensor dengan shape yang sebarang**],
kita cukup menjumlahkan semua sumbunya.
Misalnya, jumlah elemen
dari matriks $m \times n$ $\mathbf{A}$
dapat ditulis sebagai $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.


```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

Secara default, memanggil fungsi `sum`
akan *mereduksi* tensor di sepanjang semua sumbunya,
dan akhirnya menghasilkan sebuah skalar.
Pustaka kita juga memungkinkan kita untuk [**menentukan sumbu
yang ingin kita reduksi pada tensor.**]
Untuk menjumlahkan semua elemen di sepanjang baris (sumbu 0),
kita menentukan `axis=0` dalam `sum`.
Karena matriks input direduksi di sepanjang sumbu 0
untuk menghasilkan vektor output,
sumbu ini tidak ada dalam shape output.


```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

Menentukan `axis=1` akan mereduksi dimensi kolom (sumbu 1) dengan menjumlahkan elemen-elemen dari semua kolom.


```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Mereduksi matriks di sepanjang baris dan kolom melalui penjumlahan
sama dengan menjumlahkan semua elemen dalam matriks.


```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**Kuantitas terkait adalah *mean*, yang juga disebut sebagai *rata-rata*.**]
Kita menghitung mean dengan membagi jumlah
dengan jumlah total elemen.
Karena menghitung mean sangat umum dilakukan,
maka ada fungsi pustaka khusus yang bekerja
secara analog dengan `sum`.


```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Demikian pula, fungsi untuk menghitung mean
juga dapat mereduksi tensor di sepanjang sumbu tertentu.


```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Penjumlahan Tanpa Reduksi
:label:`subsec_lin-alg-non-reduction`

Kadang-kadang berguna untuk [**mempertahankan jumlah sumbu tetap sama**]
saat memanggil fungsi untuk menghitung jumlah atau mean.
Ini penting saat kita ingin menggunakan mekanisme broadcasting.


```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Sebagai contoh, karena `sum_A` mempertahankan dua sumbunya setelah menjumlahkan setiap baris,
kita bisa (**membagi `A` dengan `sum_A` menggunakan broadcasting**)
untuk membuat matriks di mana setiap barisnya memiliki jumlah total $1$.


```{.python .input}
%%tab all
A / sum_A
```

Jika kita ingin menghitung [**jumlah kumulatif elemen-elemen dari `A` di sepanjang sumbu tertentu**],
misalnya `axis=0` (baris demi baris), kita bisa memanggil fungsi `cumsum`.
Secara desain, fungsi ini tidak mereduksi tensor input di sepanjang sumbu mana pun.


```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Produk Dot

Sejauh ini, kita hanya melakukan operasi elemen demi elemen, penjumlahan, dan rata-rata.
Dan jika hanya itu yang bisa kita lakukan, aljabar linear
tidak layak mendapatkan bagiannya sendiri.
Untungnya, ini adalah bagian di mana hal-hal menjadi lebih menarik.
Salah satu operasi paling mendasar adalah produk dot.
Diberikan dua vektor $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,
*produk dot* mereka $\mathbf{x}^\top \mathbf{y}$ (juga dikenal sebagai *produk dalam*, $\langle \mathbf{x}, \mathbf{y}  \rangle$)
adalah jumlah dari hasil kali elemen pada posisi yang sama:
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~*Produk dot* dari dua vektor adalah jumlah dari hasil kali elemen pada posisi yang sama~~]


```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

Sebagai alternatif, (**kita dapat menghitung produk dot dari dua vektor
dengan melakukan perkalian elemen demi elemen diikuti dengan penjumlahan:**)


```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

Produk dot berguna dalam berbagai konteks.
Misalnya, diberikan suatu set nilai,
dilambangkan dengan vektor $\mathbf{x} \in \mathbb{R}^n$,
dan sebuah set bobot, dilambangkan dengan $\mathbf{w} \in \mathbb{R}^n$,
jumlah tertimbang dari nilai-nilai dalam $\mathbf{x}$
sesuai dengan bobot $\mathbf{w}$
dapat dinyatakan sebagai produk dot $\mathbf{x}^\top \mathbf{w}$.
Ketika bobot-bobot bernilai non-negatif
dan jumlahnya sama dengan $1$, yaitu, $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
produk dot tersebut mengekspresikan *rata-rata tertimbang*.
Setelah menormalkan dua vektor agar memiliki panjang unit,
produk dot akan menunjukkan kosinus dari sudut di antara mereka.
Nanti dalam bagian ini, kita akan secara formal memperkenalkan konsep *panjang* ini.


## Produk Matriks--Vektor

Sekarang setelah kita tahu cara menghitung produk dot,
kita bisa mulai memahami *produk*
antara matriks $m \times n$ $\mathbf{A}$
dan vektor berdimensi $n$ $\mathbf{x}$.
Sebagai langkah awal, kita bisa memvisualisasikan matriks kita
dalam hal vektor-vektor barisnya

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

di mana setiap $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
adalah vektor baris yang merepresentasikan baris ke-$i$
dari matriks $\mathbf{A}$.

[**Produk matriks--vektor $\mathbf{A}\mathbf{x}$
hanyalah vektor kolom dengan panjang $m$,
di mana elemen ke-$i$-nya adalah produk dot
$\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Kita bisa menganggap perkalian dengan matriks
$\mathbf{A}\in \mathbb{R}^{m \times n}$
sebagai sebuah transformasi yang memproyeksikan vektor-vektor
dari $\mathbb{R}^{n}$ ke $\mathbb{R}^{m}$.
Transformasi ini sangat berguna.
Sebagai contoh, kita bisa merepresentasikan rotasi
sebagai perkalian dengan matriks persegi tertentu.
Produk matriks--vektor juga menggambarkan
perhitungan kunci yang terlibat dalam menghitung
output dari setiap lapisan dalam jaringan saraf
diberikan output dari lapisan sebelumnya.

:begin_tab:`mxnet`
Untuk mengekspresikan produk matriks--vektor dalam kode,
kita menggunakan fungsi `dot` yang sama.
Operasi ini disimpulkan
berdasarkan tipe argumennya.
Perlu dicatat bahwa dimensi kolom `A`
(panjang di sepanjang sumbu 1)
harus sama dengan dimensi `x` (panjangnya).
:end_tab:

:begin_tab:`pytorch`
Untuk mengekspresikan produk matriks--vektor dalam kode,
kita menggunakan fungsi `mv`.
Perlu dicatat bahwa dimensi kolom `A`
(panjang di sepanjang sumbu 1)
harus sama dengan dimensi `x` (panjangnya).
Python memiliki operator `@` yang nyaman
untuk menjalankan produk matriks--vektor
maupun matriks--matriks (tergantung pada argumennya).
Sehingga kita bisa menulis `A@x`.
:end_tab:

:begin_tab:`tensorflow`
Untuk mengekspresikan produk matriks--vektor dalam kode,
kita menggunakan fungsi `matvec`.
Perlu dicatat bahwa dimensi kolom `A`
(panjang di sepanjang sumbu 1)
harus sama dengan dimensi `x` (panjangnya).
:end_tab:


```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## Perkalian Matriks--Matriks

Setelah Anda memahami produk dot dan produk matriks--vektor,
maka *perkalian matriks--matriks* seharusnya cukup sederhana.

Misalkan kita memiliki dua matriks
$\mathbf{A} \in \mathbb{R}^{n \times k}$
dan $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Biarkan $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ mewakili
vektor baris yang mewakili baris ke- $i$
dari matriks $\mathbf{A}$
dan biarkan $\mathbf{b}_{j} \in \mathbb{R}^k$ mewakili
vektor kolom dari kolom ke- $j$
dari matriks $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


Untuk membentuk hasil perkalian matriks $\mathbf{C} \in \mathbb{R}^{n \times m}$,
kita cukup menghitung setiap elemen $c_{ij}$
sebagai produk dot antara
baris ke- $i$ dari $\mathbf{A}$
dan kolom ke- $j$ dari $\mathbf{B}$,
yaitu $\mathbf{a}^\top_i \mathbf{b}_j$:

$$
\mathbf{C} = \mathbf{AB} = \begin{bmatrix} \mathbf{a}^\top_{1} \\ \mathbf{a}^\top_{2} \\ \vdots \\ \mathbf{a}^\top_n \end{bmatrix} \begin{bmatrix} \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \end{bmatrix} = \begin{bmatrix} \mathbf{a}^\top_{1} \mathbf{b}_{1} & \mathbf{a}^\top_{1} \mathbf{b}_{2} & \cdots & \mathbf{a}^\top_{1} \mathbf{b}_{m} \\ \mathbf{a}^\top_{2} \mathbf{b}_{1} & \mathbf{a}^\top_{2} \mathbf{b}_{2} & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_{m} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{a}^\top_{n} \mathbf{b}_{1} & \mathbf{a}^\top_{n} \mathbf{b}_{2} & \cdots & \mathbf{a}^\top_{n} \mathbf{b}_{m} \end{bmatrix}.
$$

[**Kita dapat memandang perkalian matriks--matriks $\mathbf{AB}$
sebagai melakukan $m$ produk matriks--vektor
atau $m \times n$ produk dot
dan menggabungkan hasilnya
untuk membentuk matriks $n \times m$.**]
Dalam cuplikan berikut,
kita melakukan perkalian matriks pada `A` dan `B`.
Di sini, `A` adalah matriks dengan dua baris dan tiga kolom,
dan `B` adalah matriks dengan tiga baris dan empat kolom.
Setelah perkalian, kita memperoleh matriks dengan dua baris dan empat kolom.


```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

Istilah *perkalian matriks--matriks*
sering disederhanakan menjadi *perkalian matriks*,
dan seharusnya tidak disamakan dengan produk Hadamard.



## Norma
:label:`subsec_lin-algebra-norms`

Beberapa operator paling berguna dalam aljabar linear adalah *norma*.
Secara informal, norma dari sebuah vektor menunjukkan seberapa *besar* vektor tersebut.
Sebagai contoh, norma $\ell_2$ mengukur panjang (Euklidean) dari sebuah vektor.
Di sini, kita menggunakan konsep *ukuran* yang berkaitan dengan besarnya komponen vektor
(bukan dimensinya).

Norma adalah sebuah fungsi $\| \cdot \|$ yang memetakan sebuah vektor
menjadi sebuah skalar dan memenuhi tiga properti berikut:

1. Diberikan sebuah vektor $\mathbf{x}$, jika kita mengalikan (semua elemen) vektor
   dengan skalar $\alpha \in \mathbb{R}$, maka normanya akan tereskalasi sesuai:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Untuk setiap vektor $\mathbf{x}$ dan $\mathbf{y}$:
   norma memenuhi ketaksamaan segitiga:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. Norma dari sebuah vektor selalu non-negatif dan hanya bernilai nol jika vektor tersebut nol:
   $$\|\mathbf{x}\| > 0 \textrm{ untuk semua } \mathbf{x} \neq 0.$$

Banyak fungsi yang memenuhi kriteria norma dan norma yang berbeda
menggambarkan konsep ukuran yang berbeda pula.
Norma Euklidean yang kita pelajari dalam geometri sekolah dasar
saat menghitung panjang hipotenusa segitiga siku-siku
adalah akar kuadrat dari jumlah kuadrat elemen-elemen vektor.
Secara formal, ini disebut sebagai [**norma $\ell_2$**] dan dinyatakan sebagai

(**$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.
$$**)

Metode `norm` menghitung norma $\ell_2$.


```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**Norma $\ell_1$**] juga umum digunakan
dan ukuran terkaitnya disebut sebagai jarak Manhattan.
Secara definisi, norma $\ell_1$ menjumlahkan
nilai absolut dari elemen-elemen vektor:

(**$$
\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|.
$$**)

Dibandingkan dengan norma $\ell_2$, norma ini kurang sensitif terhadap nilai pencilan.
Untuk menghitung norma $\ell_1$,
kita menggabungkan operasi nilai absolut
dengan operasi penjumlahan.


```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

Baik norma $\ell_2$ maupun $\ell_1$ adalah kasus khusus
dari norma $\ell_p$ yang lebih umum:

$$
\|\mathbf{x}\|_{p} = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}.
$$


Dalam kasus matriks, masalahnya menjadi lebih rumit.
Bagaimanapun juga, matriks dapat dipandang baik sebagai kumpulan elemen individual
*maupun* sebagai objek yang beroperasi pada vektor dan mengubahnya menjadi vektor lain.
Sebagai contoh, kita dapat bertanya seberapa panjang
produk matriks-vektor $\mathbf{X} \mathbf{v}$
dibandingkan dengan $\mathbf{v}$.
Pemikiran ini mengarah pada apa yang disebut sebagai norma *spektral*.
Untuk sekarang, kita memperkenalkan [**norma *Frobenius*,
yang jauh lebih mudah untuk dihitung**] dan didefinisikan sebagai
akar kuadrat dari jumlah kuadrat elemen-elemen dalam matriks:

$$\|\mathbf{X}\|_{\textrm{F}} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}$$


Norma Frobenius berperilaku seolah-olah
sebagai norma $\ell_2$ dari vektor berbentuk matriks.
Memanggil fungsi berikut akan menghitung
norma Frobenius dari suatu matriks.


```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

Sementara kita tidak ingin terlalu jauh ke depan,
kita sudah bisa menanamkan beberapa intuisi tentang mengapa konsep-konsep ini berguna.
Dalam pembelajaran mendalam, kita sering mencoba menyelesaikan masalah optimasi:
*maksimalkan* probabilitas yang diberikan kepada data yang diamati;
*maksimalkan* pendapatan yang terkait dengan model rekomendasi;
*minimalkan* jarak antara prediksi
dan pengamatan kebenaran dasar;
*minimalkan* jarak antara representasi
foto dari orang yang sama
sementara *memaksimalkan* jarak antara representasi
foto dari orang yang berbeda.
Jarak-jarak ini, yang membentuk
tujuan algoritma pembelajaran mendalam,
sering kali diekspresikan sebagai norma.

## Diskusi

Pada bagian ini, kita telah meninjau semua aljabar linier
yang akan Anda butuhkan untuk memahami
sebagian besar dari pembelajaran mendalam modern.
Namun, masih ada banyak hal dalam aljabar linier
dan banyak di antaranya berguna untuk pembelajaran mesin.
Sebagai contoh, matriks dapat diuraikan menjadi faktor-faktor,
dan uraian ini dapat mengungkapkan
struktur dimensi rendah dalam dataset dunia nyata.
Ada sub-bidang pembelajaran mesin yang berfokus
pada penggunaan uraian matriks
dan generalisasinya ke tensor berorde tinggi
untuk menemukan struktur dalam dataset
dan menyelesaikan masalah prediksi.
Namun, buku ini berfokus pada pembelajaran mendalam.
Dan kami yakin Anda akan lebih tertarik
untuk belajar lebih banyak matematika
setelah Anda mulai mengaplikasikan pembelajaran mesin
pada dataset nyata.
Jadi, sementara kami berhak
untuk memperkenalkan lebih banyak matematika di kemudian hari,
kami mengakhiri bagian ini di sini.

Jika Anda ingin belajar lebih banyak aljabar linier,
ada banyak buku dan sumber daya online yang sangat baik.
Untuk kursus kilat yang lebih lanjut, pertimbangkan untuk melihat
:citet:`Strang.1993`, :citet:`Kolter.2008`, dan :citet:`Petersen.Pedersen.ea.2008`.

Sebagai rangkuman:

* Skalar, vektor, matriks, dan tensor adalah
  objek matematika dasar yang digunakan dalam aljabar linier
  dan masing-masing memiliki sumbu 0, satu, dua, dan jumlah sumbu yang tidak terbatas.
* Tensor dapat dipotong atau dikurangi di sepanjang sumbu yang ditentukan
  melalui pengindeksan, atau operasi seperti `sum` dan `mean`.
* Produk elemen-wise disebut produk Hadamard.
  Sebaliknya, produk dot, produk matriks-vektor, dan produk matriks-matriks
  bukan operasi elemen-wise dan umumnya menghasilkan objek
  dengan bentuk yang berbeda dari operan.
* Dibandingkan dengan produk Hadamard, produk matriks-matriks
  membutuhkan waktu komputasi yang jauh lebih lama (waktu kubik daripada kuadrat).
* Norma menangkap berbagai konsep ukuran dari vektor (atau matriks),
  dan sering kali diterapkan pada perbedaan dua vektor
  untuk mengukur jarak di antara mereka.
* Norma vektor yang umum meliputi norma $\ell_1$ dan $\ell_2$,
   dan norma matriks yang umum meliputi norma *spektral* dan *Frobenius*.


## Latihan

1. Buktikan bahwa transpose dari transpose sebuah matriks adalah matriks itu sendiri: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
2. Diberikan dua matriks $\mathbf{A}$ dan $\mathbf{B}$, tunjukkan bahwa penjumlahan dan transpose bersifat komutatif: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
3. Diberikan matriks persegi $\mathbf{A}$, apakah $\mathbf{A} + \mathbf{A}^\top$ selalu simetris? Bisakah Anda membuktikan hasilnya dengan hanya menggunakan hasil dari dua latihan sebelumnya?
4. Kami mendefinisikan tensor `X` dengan bentuk (2, 3, 4) pada bagian ini. Apa output dari `len(X)`? Tulis jawaban Anda tanpa mengimplementasikan kode, lalu periksa jawaban Anda dengan kode.
5. Untuk tensor `X` dengan bentuk yang sewenang-wenang, apakah `len(X)` selalu sesuai dengan panjang sumbu tertentu dari `X`? Sumbu yang mana?
6. Jalankan `A / A.sum(axis=1)` dan lihat apa yang terjadi. Bisakah Anda menganalisis hasilnya?
7. Ketika bepergian antara dua titik di pusat kota Manhattan, apa jarak yang perlu Anda tempuh dalam hal koordinat, yaitu dalam hal avenue dan street? Bisakah Anda bepergian secara diagonal?
8. Pertimbangkan tensor dengan bentuk (2, 3, 4). Apa bentuk output penjumlahan di sepanjang sumbu 0, 1, dan 2?
9. Berikan tensor dengan tiga atau lebih sumbu ke fungsi `linalg.norm` dan amati outputnya. Apa yang dihitung fungsi ini untuk tensor dengan bentuk yang sewenang-wenang?
10. Pertimbangkan tiga matriks besar, misalnya $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ dan $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, yang diinisialisasi dengan variabel acak Gaussian. Anda ingin menghitung produk $\mathbf{A} \mathbf{B} \mathbf{C}$. Apakah ada perbedaan dalam penggunaan memori dan kecepatan, tergantung pada apakah Anda menghitung $(\mathbf{A} \mathbf{B}) \mathbf{C}$ atau $\mathbf{A} (\mathbf{B} \mathbf{C})$? Mengapa?
11. Pertimbangkan tiga matriks besar, misalnya $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ dan $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Apakah ada perbedaan kecepatan tergantung pada apakah Anda menghitung $\mathbf{A} \mathbf{B}$ atau $\mathbf{A} \mathbf{C}^\top$? Mengapa? Apa yang berubah jika Anda menginisialisasi $\mathbf{C} = \mathbf{B}^\top$ tanpa menyalin memori? Mengapa?
12. Pertimbangkan tiga matriks, misalnya $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Konstruksikan tensor dengan tiga sumbu dengan menyusun $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. Apa dimensi dari tensor tersebut? Potong koordinat kedua dari sumbu ketiga untuk mendapatkan kembali $\mathbf{B}$. Periksa bahwa jawaban Anda benar.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17968)
:end_tab:
