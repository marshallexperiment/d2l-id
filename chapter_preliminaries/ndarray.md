```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Manipulasi Data
:label:`sec_ndarray`

Untuk melakukan apa pun, 
kita memerlukan cara untuk menyimpan dan memanipulasi data.
Secara umum, ada dua hal penting 
yang perlu kita lakukan dengan data: 
(i) memperolehnya; 
dan (ii) memprosesnya setelah berada di dalam komputer. 
Tidak ada gunanya mengumpulkan data 
tanpa cara untuk menyimpannya, 
jadi untuk memulai, mari kita bekerja langsung
dengan array berdimensi $n$, 
yang juga kita sebut *tensor*.
Jika Anda sudah mengetahui paket komputasi ilmiah NumPy, 
ini akan terasa mudah.
Untuk semua framework deep learning modern,
*kelas tensor* (`ndarray` di MXNet, 
`Tensor` di PyTorch dan TensorFlow) 
mirip dengan `ndarray` di NumPy,
dengan beberapa fitur unggulan tambahan.
Pertama, kelas tensor
mendukung diferensiasi otomatis.
Kedua, kelas ini memanfaatkan GPU
untuk mempercepat perhitungan numerik,
sedangkan NumPy hanya berjalan di CPU.
Sifat-sifat ini membuat jaringan saraf 
mudah dikodekan dan cepat dijalankan.



## Memulai

:begin_tab:`mxnet`
Untuk memulai, kita mengimpor modul `np` (`numpy`) dan
`npx` (`numpy_extension`) dari MXNet.
Di sini, modul `np` mencakup 
fungsi-fungsi yang didukung oleh NumPy,
sementara modul `npx` berisi serangkaian ekstensi
yang dikembangkan untuk mendukung deep learning 
dalam lingkungan yang mirip dengan NumPy.
Saat menggunakan tensor, kita hampir selalu 
memanggil fungsi `set_np`:
ini untuk kompatibilitas pemrosesan tensor 
oleh komponen lain dari MXNet.
:end_tab:

:begin_tab:`pytorch`
(**Untuk memulai, kita mengimpor pustaka PyTorch.
Perhatikan bahwa nama paketnya adalah `torch`.**)
:end_tab:

:begin_tab:`tensorflow`
Untuk memulai, kita mengimpor `tensorflow`. 
Untuk singkatnya, praktisi 
sering menggunakan alias `tf`.
:end_tab:



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
import jax
from jax import numpy as jnp
```
[**Sebuah tensor merepresentasikan array nilai numerik (mungkin berdimensi banyak).**]
Dalam kasus satu dimensi, yaitu ketika hanya diperlukan satu sumbu untuk data,
tensor disebut sebagai *vektor*.
Dengan dua sumbu, tensor disebut sebagai *matriks*.
Untuk $k > 2$ sumbu, kita tidak menggunakan nama khusus
dan cukup menyebut objek tersebut sebagai *tensor orde ke-*$k^\textrm{th}$.

:begin_tab:`mxnet`
MXNet menyediakan berbagai fungsi 
untuk membuat tensor baru 
yang sudah diisi dengan nilai. 
Misalnya, dengan memanggil `arange(n)`,
kita dapat membuat vektor dengan nilai-nilai yang berjarak merata,
dimulai dari 0 (termasuk) 
dan berakhir di `n` (tidak termasuk).
Secara default, ukuran interval adalah $1$.
Kecuali ditentukan lain, 
tensor baru disimpan di memori utama 
dan ditujukan untuk komputasi berbasis CPU.
:end_tab:

:begin_tab:`pytorch`
PyTorch menyediakan berbagai fungsi 
untuk membuat tensor baru 
yang sudah diisi dengan nilai. 
Misalnya, dengan memanggil `arange(n)`,
kita dapat membuat vektor dengan nilai-nilai yang berjarak merata,
dimulai dari 0 (termasuk) 
dan berakhir di `n` (tidak termasuk).
Secara default, ukuran interval adalah $1$.
Kecuali ditentukan lain, 
tensor baru disimpan di memori utama 
dan ditujukan untuk komputasi berbasis CPU.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow menyediakan berbagai fungsi 
untuk membuat tensor baru 
yang sudah diisi dengan nilai. 
Misalnya, dengan memanggil `range(n)`,
kita dapat membuat vektor dengan nilai-nilai yang berjarak merata,
dimulai dari 0 (termasuk) 
dan berakhir di `n` (tidak termasuk).
Secara default, ukuran interval adalah $1$.
Kecuali ditentukan lain, 
tensor baru disimpan di memori utama 
dan ditujukan untuk komputasi berbasis CPU.
:end_tab:


```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
Setiap nilai ini disebut
sebagai *elemen* dari tensor.
Tensor `x` mengandung 12 elemen.
Kita dapat memeriksa jumlah total elemen 
dalam sebuah tensor melalui atribut `size`.
:end_tab:

:begin_tab:`pytorch`
Setiap nilai ini disebut
sebagai *elemen* dari tensor.
Tensor `x` mengandung 12 elemen.
Kita dapat memeriksa jumlah total elemen 
dalam sebuah tensor melalui metode `numel`.
:end_tab:

:begin_tab:`tensorflow`
Setiap nilai ini disebut
sebagai *elemen* dari tensor.
Tensor `x` mengandung 12 elemen.
Kita dapat memeriksa jumlah total elemen 
dalam sebuah tensor melalui fungsi `size`.
:end_tab:


```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**Kita dapat mengakses *shape* tensor**) 
(panjang di sepanjang setiap sumbu)
dengan memeriksa atribut `shape`-nya.
Karena kita sedang berurusan dengan vektor di sini,
`shape` hanya berisi satu elemen
dan identik dengan ukuran.


```{.python .input}
%%tab all
x.shape
```

Kita dapat [**mengubah bentuk dari sebuah tensor
tanpa mengubah ukuran atau nilainya**]
dengan memanggil `reshape`.
Misalnya, kita dapat mengubah 
vektor `x` kita yang memiliki bentuk (12,) 
menjadi matriks `X` dengan bentuk (3, 4).
Tensor baru ini mempertahankan semua elemen
tetapi mengonfigurasinya kembali menjadi matriks.
Perhatikan bahwa elemen-elemen dari vektor kita
diletakkan satu baris pada satu waktu sehingga
`x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Perhatikan bahwa menentukan setiap komponen bentuk
untuk `reshape` sebenarnya tidak perlu.
Karena kita sudah mengetahui ukuran tensor kita,
kita dapat menghitung satu komponen bentuk dengan mengetahui komponen lainnya.
Misalnya, diberikan sebuah tensor dengan ukuran $n$
dan bentuk target ($h$, $w$),
kita tahu bahwa $w = n/h$.
Untuk secara otomatis menentukan satu komponen bentuk,
kita dapat memasukkan `-1` pada komponen bentuk
yang seharusnya diisi secara otomatis.
Dalam kasus kita, alih-alih memanggil `x.reshape(3, 4)`,
kita dapat memanggil `x.reshape(-1, 4)` atau `x.reshape(3, -1)`.

Praktisi sering kali perlu bekerja dengan tensor
yang diinisialisasi untuk berisi semua 0 atau 1.
[**Kita dapat membuat tensor dengan semua elemen diatur ke 0**] (~~atau satu~~)
dan dengan bentuk (2, 3, 4) melalui fungsi `zeros`.


```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

Demikian pula, kita dapat membuat tensor 
dengan semua elemen 1 dengan memanggil `ones`.


```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

Kita sering ingin 
[**mengambil sampel setiap elemen secara acak (dan independen)**] 
dari distribusi probabilitas tertentu.
Misalnya, parameter dari jaringan saraf
sering kali diinisialisasi secara acak.
Potongan kode berikut membuat sebuah tensor 
dengan elemen-elemen yang diambil dari 
distribusi Gaussian standar (normal)
dengan rata-rata 0 dan deviasi standar 1.


```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

Terakhir, kita dapat membuat tensor dengan
[**menyediakan nilai pasti untuk setiap elemen**] 
dengan memberikan (mungkin berupa daftar bertingkat) list Python 
yang berisi nilai-nilai numerik.
Di sini, kita membuat sebuah matriks dengan daftar daftar,
di mana daftar terluar sesuai dengan sumbu 0,
dan daftar dalam sesuai dengan sumbu 1.


```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indexing dan Slicing

Seperti pada list Python,
kita dapat mengakses elemen tensor 
dengan melakukan indexing (dimulai dari 0).
Untuk mengakses elemen berdasarkan posisinya
relatif terhadap akhir list,
kita dapat menggunakan indexing negatif.
Akhirnya, kita dapat mengakses rentang indeks secara keseluruhan 
melalui slicing (misalnya, `X[start:stop]`), 
di mana nilai yang dikembalikan mencakup 
indeks pertama (`start`) *tetapi tidak yang terakhir* (`stop`).
Terakhir, ketika hanya satu indeks (atau slice)
ditentukan untuk tensor orde ke-$k$,
indeks tersebut diterapkan pada sumbu 0.
Jadi, dalam kode berikut,
[**`[-1]` memilih baris terakhir dan `[1:3]`
memilih baris kedua dan ketiga**].


```{.python .input}
%%tab all
X[-1], X[1:3]
```
:begin_tab:`mxnet, pytorch`
Selain membaca elemen, (**kita juga dapat *menulis* elemen dari sebuah matriks dengan menentukan indeks.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` dalam TensorFlow bersifat immutable, dan tidak dapat diberikan nilai langsung.
`Variables` dalam TensorFlow adalah kontainer mutable yang mendukung
penugasan nilai. Ingat bahwa gradien dalam TensorFlow tidak mengalir mundur
melalui penugasan `Variable`.

Selain menetapkan nilai ke seluruh `Variable`, kita dapat menulis elemen dari
`Variable` dengan menentukan indeks.
:end_tab:


```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

Jika kita ingin [**menetapkan nilai yang sama untuk beberapa elemen,
kita menerapkan indexing pada sisi kiri 
dari operasi penugasan.**]
Misalnya, `[:2, :]` mengakses 
baris pertama dan kedua,
di mana `:` mengambil semua elemen di sepanjang sumbu 1 (kolom).
Meskipun kita membahas indexing untuk matriks,
ini juga berlaku untuk vektor
dan untuk tensor dengan lebih dari dua dimensi.


```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## Operasi

Sekarang kita tahu cara membangun tensor
dan cara membaca serta menulis elemen-elemennya,
kita bisa mulai memanipulasinya
dengan berbagai operasi matematika.
Di antara yang paling berguna 
adalah operasi *elementwise*.
Operasi ini menerapkan operasi skalar standar
pada setiap elemen dari sebuah tensor.
Untuk fungsi yang menerima dua tensor sebagai masukan,
operasi elementwise menerapkan beberapa operator biner standar
pada setiap pasangan elemen yang sesuai.
Kita dapat membuat fungsi elementwise 
dari fungsi apa pun yang memetakan 
dari skalar ke skalar.

Dalam notasi matematika, kita menyebut
operator skalar *unary* (menerima satu input)
dengan tanda 
$f: \mathbb{R} \rightarrow \mathbb{R}$.
Ini berarti fungsi tersebut memetakan
dari setiap bilangan riil ke bilangan riil lainnya.
Sebagian besar operator standar, termasuk operator unary seperti $e^x$, dapat diterapkan secara elementwise.


```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

Demikian pula, kita menyebut operator skalar *binary*,
yang memetakan pasangan bilangan riil
ke satu bilangan riil
dengan tanda 
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Diberikan dua vektor $\mathbf{u}$ 
dan $\mathbf{v}$ *dengan bentuk yang sama*,
dan operator binary $f$, kita dapat menghasilkan vektor
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
dengan menetapkan $c_i \gets f(u_i, v_i)$ untuk semua $i$,
di mana $c_i, u_i$, dan $v_i$ adalah elemen ke- $i$,
dari vektor $\mathbf{c}, \mathbf{u}$, dan $\mathbf{v}$.
Di sini, kita menghasilkan $F$ yang bernilai vektor
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
dengan *mengangkat* fungsi skalar
menjadi operasi vektor elementwise.
Operator aritmetika standar umum
untuk penjumlahan (`+`), pengurangan (`-`), 
perkalian (`*`), pembagian (`/`), 
dan perpangkatan (`**`)
semuanya telah *diangkat* menjadi operasi aritmatika
untuk tensor yang identik bentuknya dengan bentuk sembarang.


```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

Selain perhitungan elementwise,
kita juga dapat melakukan operasi aljabar linear,
seperti perkalian dot dan perkalian matriks.
Kita akan menjelaskan ini lebih lanjut 
pada :numref:`sec_linear-algebra`.

Kita juga dapat [***menggabungkan* beberapa tensor,**]
dengan menumpuknya ujung ke ujung untuk membentuk tensor yang lebih besar.
Kita hanya perlu memberikan daftar tensor
dan memberi tahu sistem di sepanjang sumbu mana untuk menggabungkan.
Contoh di bawah ini menunjukkan apa yang terjadi ketika kita menggabungkan
dua matriks sepanjang baris (sumbu 0)
bukan kolom (sumbu 1).
Kita dapat melihat bahwa panjang sumbu-0 dari output pertama ($6$)
adalah jumlah dari panjang sumbu-0 kedua tensor input ($3 + 3$);
sementara panjang sumbu-1 dari output kedua ($8$)
adalah jumlah dari panjang sumbu-1 kedua tensor input ($4 + 4$).


```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

Terkadang, kita ingin 
[**membuat tensor biner melalui *pernyataan logika*.**]
Ambil contoh `X == Y`.
Untuk setiap posisi `i, j`, jika `X[i, j]` dan `Y[i, j]` sama, 
maka entri yang sesuai dalam hasil akan bernilai `1`,
jika tidak, maka bernilai `0`.


```{.python .input}
%%tab all
X == Y
```

[**Menjumlahkan semua elemen dalam tensor**] menghasilkan tensor dengan hanya satu elemen.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Broadcasting
:label:`subsec_broadcasting`

Saat ini, Anda sudah tahu cara melakukan 
operasi biner elementwise 
pada dua tensor dengan bentuk yang sama. 
Dalam kondisi tertentu,
bahkan ketika bentuknya berbeda, 
kita masih bisa [**melakukan operasi biner elementwise
dengan menggunakan *mekanisme broadcasting*.**]
Broadcasting bekerja sesuai dengan 
prosedur dua langkah berikut:
(i) memperluas satu atau kedua array
dengan menyalin elemen di sepanjang sumbu dengan panjang 1
sehingga setelah transformasi ini,
kedua tensor memiliki bentuk yang sama;
(ii) melakukan operasi elementwise
pada array yang dihasilkan.


```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

Karena `a` dan `b` masing-masing adalah matriks $3\times1$ 
dan $1\times2$, bentuknya tidak cocok.
Broadcasting menghasilkan matriks $3\times2$ yang lebih besar 
dengan mereplikasi matriks `a` di sepanjang kolom
dan matriks `b` di sepanjang baris
sebelum menambahkannya secara elementwise.


```{.python .input}
%%tab all
a + b
```

## Menghemat Memori

[**Menjalankan operasi dapat menyebabkan memori baru dialokasikan untuk menyimpan hasil.**]
Misalnya, jika kita menulis `Y = X + Y`,
kita menghapus referensi tensor yang sebelumnya ditunjuk oleh `Y`
dan kemudian menunjuk `Y` ke memori yang baru dialokasikan.
Kita dapat mendemonstrasikan masalah ini dengan fungsi `id()` dalam Python,
yang memberikan kita alamat pasti 
dari objek yang direferensikan dalam memori.
Perhatikan bahwa setelah kita menjalankan `Y = Y + X`,
`id(Y)` menunjuk ke lokasi yang berbeda.
Ini terjadi karena Python pertama-tama mengevaluasi `Y + X`,
mengalokasikan memori baru untuk hasilnya 
dan kemudian menunjuk `Y` ke lokasi baru ini dalam memori.


```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Ini mungkin tidak diinginkan karena dua alasan.
Pertama, kita tidak ingin terus-menerus
mengalokasikan memori secara tidak perlu.
Dalam machine learning, kita sering memiliki
ratusan megabyte parameter
dan memperbarui semuanya beberapa kali per detik.
Jika memungkinkan, kita ingin melakukan pembaruan ini *di tempat*.
Kedua, kita mungkin menunjuk 
parameter yang sama dari beberapa variabel.
Jika kita tidak memperbarui di tempat, 
kita harus berhati-hati untuk memperbarui semua referensi ini,
agar kita tidak mengalami kebocoran memori 
atau secara tidak sengaja merujuk ke parameter yang sudah usang.

:begin_tab:`mxnet, pytorch`
Untungnya, (**melakukan operasi di tempat**) cukup mudah.
Kita dapat menetapkan hasil dari suatu operasi
ke array `Y` yang telah dialokasikan sebelumnya
dengan menggunakan notasi slice: `Y[:] = <expression>`.
Untuk mengilustrasikan konsep ini, 
kita menimpa nilai tensor `Z`,
setelah menginisialisasinya dengan `zeros_like`,
agar memiliki bentuk yang sama dengan `Y`.
:end_tab:

:begin_tab:`tensorflow`
`Variables` adalah kontainer mutable dari state dalam TensorFlow. Mereka menyediakan
cara untuk menyimpan parameter model Anda.
Kita dapat menetapkan hasil dari suatu operasi
ke sebuah `Variable` dengan `assign`.
Untuk mengilustrasikan konsep ini, 
kita menimpa nilai `Variable` `Z`
setelah menginisialisasinya dengan `zeros_like`,
agar memiliki bentuk yang sama dengan `Y`.
:end_tab:


```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**Jika nilai `X` tidak digunakan kembali dalam perhitungan selanjutnya,
kita juga dapat menggunakan `X[:] = X + Y` atau `X += Y`
untuk mengurangi overhead memori dari operasi tersebut.**]
:end_tab:

:begin_tab:`tensorflow`
Bahkan setelah Anda menyimpan state secara permanen dalam sebuah `Variable`, 
Anda mungkin ingin mengurangi penggunaan memori lebih lanjut dengan menghindari alokasi berlebihan
untuk tensor yang bukan parameter model Anda.
Karena `Tensors` dalam TensorFlow bersifat immutable 
dan gradien tidak mengalir melalui penugasan `Variable`, 
TensorFlow tidak menyediakan cara eksplisit untuk menjalankan
sebuah operasi secara langsung di tempat.

Namun, TensorFlow menyediakan decorator `tf.function` 
untuk membungkus perhitungan di dalam graf TensorFlow 
yang dikompilasi dan dioptimalkan sebelum dijalankan.
Ini memungkinkan TensorFlow untuk menghapus nilai yang tidak digunakan, 
dan menggunakan kembali alokasi sebelumnya yang tidak lagi dibutuhkan. 
Ini meminimalkan overhead memori dari perhitungan TensorFlow.
:end_tab:


```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # Nilai yang tidak digunakan ini akan dihapus
    A = X + Y  # Alokasi akan digunakan kembali ketika tidak lagi diperlukan
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Konversi ke Objek Python Lain

:begin_tab:`mxnet, tensorflow`
[**Mengonversi ke tensor NumPy (`ndarray`)**], atau sebaliknya, sangat mudah.
Hasil konversi tidak berbagi memori.
Ketidaknyamanan kecil ini sebenarnya cukup penting:
saat Anda melakukan operasi di CPU atau GPU,
Anda tidak ingin menghentikan perhitungan sambil menunggu
apakah paket NumPy dari Python 
mungkin ingin melakukan sesuatu dengan
bagian memori yang sama.
:end_tab:

:begin_tab:`pytorch`
[**Mengonversi ke tensor NumPy (`ndarray`)**], atau sebaliknya, sangat mudah.
Tensor torch dan array NumPy 
akan berbagi memori dasar mereka, 
dan mengubah salah satunya melalui operasi di tempat 
juga akan mengubah yang lainnya.
:end_tab:


```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

Untuk (**mengonversi tensor berukuran 1 menjadi skalar Python**),
kita dapat memanggil fungsi `item` atau menggunakan fungsi bawaan Python.


```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Ringkasan

Kelas tensor adalah antarmuka utama untuk menyimpan dan memanipulasi data dalam pustaka deep learning.
Tensor menyediakan berbagai fungsionalitas termasuk rutinitas konstruksi; indexing dan slicing; operasi matematika dasar; broadcasting; penugasan yang efisien dalam hal memori; serta konversi ke dan dari objek Python lainnya.

## Latihan

1. Jalankan kode dalam bagian ini. Ubah pernyataan kondisional `X == Y` menjadi `X < Y` atau `X > Y`, lalu lihat jenis tensor apa yang dapat Anda peroleh.
2. Ganti dua tensor yang dioperasikan secara elementwise dalam mekanisme broadcasting dengan bentuk lain, misalnya, tensor 3 dimensi. Apakah hasilnya sesuai dengan yang diharapkan?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17966)
:end_tab:

