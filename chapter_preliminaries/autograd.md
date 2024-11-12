```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Differensiasi Otomatis
:label:`sec_autograd`

Ingat kembali dari :numref:`sec_calculus` 
bahwa menghitung turunan adalah langkah penting
dalam semua algoritma optimisasi
yang akan kita gunakan untuk melatih jaringan dalam (*Deep Net*).
Meskipun perhitungannya cukup sederhana,
melakukan perhitungan secara manual bisa membosankan dan rentan kesalahan, 
dan masalah ini hanya akan bertambah
seiring meningkatnya kompleksitas model kita.

Untungnya, semua framework deep learning modern
membebaskan kita dari pekerjaan ini
dengan menawarkan *diferensiasi otomatis*
(yang sering disingkat sebagai *autograd*). 
Saat kita mengalirkan data melalui setiap fungsi berturut-turut,
framework tersebut membangun sebuah *graf komputasi* 
yang melacak bagaimana setiap nilai bergantung pada nilai lainnya.
Untuk menghitung turunan, 
diferensiasi otomatis 
bekerja mundur melalui graf ini
menerapkan aturan rantai. 
Algoritma komputasi untuk menerapkan aturan rantai
dengan cara ini disebut *backpropagation*.

Meskipun perpustakaan autograd menjadi
perhatian penting selama satu dekade terakhir,
mereka memiliki sejarah panjang. 
Faktanya, referensi paling awal mengenai autograd
berkisar lebih dari setengah abad yang lalu :cite:`Wengert.1964`.
Ide inti di balik backpropagation modern
berasal dari tesis PhD pada tahun 1980 :cite:`Speelpenning.1980`
dan dikembangkan lebih lanjut pada akhir 1980-an :cite:`Griewank.1989`.
Meskipun backpropagation telah menjadi metode default 
untuk menghitung gradien, ini bukan satu-satunya pilihan. 
Sebagai contoh, bahasa pemrograman Julia menggunakan 
propagasi maju :cite:`Revels.Lubin.Papamarkou.2016`. 
Sebelum menjelajahi metode-metode ini, 
mari kita pelajari terlebih dahulu paket autograd.


```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
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

## Fungsi Sederhana

Misalkan kita tertarik untuk (**mendiferensiasi fungsi
$y = 2\mathbf{x}^{\top}\mathbf{x}$
terhadap vektor kolom $\mathbf{x}$.**)
Untuk memulai, kita menetapkan nilai awal untuk `x`.


```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**Sebelum kita menghitung gradien
dari $y$ terhadap $\mathbf{x}$,
kita perlu menyediakan tempat untuk menyimpannya.**]
Secara umum, kita menghindari alokasi memori baru
setiap kali kita mengambil turunan 
karena deep learning membutuhkan 
perhitungan turunan berturut-turut
terhadap parameter yang sama berkali-kali,
dan ini bisa berisiko menyebabkan kehabisan memori.
Perhatikan bahwa gradien dari fungsi bernilai skalar
terhadap vektor $\mathbf{x}$
bernilai vektor dengan 
bentuk yang sama dengan $\mathbf{x}$.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# Kita menyediakan memori untuk gradien tensor dengan memanggil `attach_grad`
x.attach_grad()
# Setelah kita menghitung gradien yang diambil terhadap `x`, kita akan dapat
# mengaksesnya melalui atribut `grad`, yang nilainya diinisialisasi dengan 0
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Dapat juga membuat x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # Gradien awalnya `None` secara default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**Sekarang kita menghitung fungsi dari `x` dan menetapkan hasilnya ke `y`.**)


```{.python .input  n=10}
%%tab mxnet
# Kode kita berada dalam scope `autograd.record` untuk membangun graf komputasi
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Merekam semua perhitungan ke dalam sebuah tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**Sekarang kita dapat mengambil gradien dari `y`
terhadap `x`**] dengan memanggil 
metode `backward`-nya.
Selanjutnya, kita dapat mengakses gradien 
melalui atribut `grad` milik `x`.
:end_tab:

:begin_tab:`pytorch`
[**Sekarang kita dapat mengambil gradien dari `y`
terhadap `x`**] dengan memanggil 
metode `backward`-nya.
Selanjutnya, kita dapat mengakses gradien 
melalui atribut `grad` milik `x`.
:end_tab:

:begin_tab:`tensorflow`
[**Sekarang kita dapat menghitung gradien dari `y`
terhadap `x`**] dengan memanggil 
metode `gradient`.
:end_tab:

:begin_tab:`jax`
[**Sekarang kita dapat mengambil gradien dari `y`
terhadap `x`**] dengan meneruskan melalui 
transformasi `grad`.
:end_tab:


```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

(**Kita sudah tahu bahwa gradien dari fungsi $y = 2\mathbf{x}^{\top}\mathbf{x}$
terhadap $\mathbf{x}$ seharusnya adalah $4\mathbf{x}$.**)
Sekarang kita dapat memverifikasi bahwa perhitungan gradien otomatis
dan hasil yang diharapkan identik.


```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Sekarang mari kita hitung 
fungsi lain dari `x`
dan ambil gradiennya.**] 
Perhatikan bahwa MXNet mereset buffer gradien 
setiap kali kita merekam gradien baru.
:end_tab:

:begin_tab:`pytorch`
[**Sekarang mari kita hitung 
fungsi lain dari `x`
dan ambil gradiennya.**]
Perhatikan bahwa PyTorch tidak secara otomatis 
mereset buffer gradien 
ketika kita merekam gradien baru. 
Sebaliknya, gradien baru
ditambahkan ke gradien yang sudah disimpan.
Perilaku ini berguna
saat kita ingin mengoptimalkan jumlah 
dari beberapa fungsi objektif.
Untuk mereset buffer gradien,
kita dapat memanggil `x.grad.zero_()` seperti berikut:
:end_tab:

:begin_tab:`tensorflow`
[**Sekarang mari kita hitung 
fungsi lain dari `x`
dan ambil gradiennya.**]
Perhatikan bahwa TensorFlow mereset buffer gradien 
setiap kali kita merekam gradien baru.
:end_tab:


```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Ditimpa oleh gradien yang baru dihitung
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Me-reset gradien nya
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Ditimpa oleh gradien yang baru dihitung
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```
## Backward untuk Variabel Non-Skalar

Ketika `y` adalah sebuah vektor, 
representasi paling alami 
dari turunan `y`
terhadap vektor `x` 
adalah sebuah matriks yang disebut *Jacobian*
yang berisi turunan parsial
dari setiap komponen `y` 
terhadap setiap komponen `x`.
Demikian pula, untuk `y` dan `x` yang lebih tinggi order-nya,
hasil diferensiasi dapat berupa tensor dengan order yang lebih tinggi lagi.

Meskipun Jacobian muncul dalam beberapa
teknik machine learning tingkat lanjut,
lebih umum kita ingin menjumlahkan 
gradien dari setiap komponen `y`
terhadap vektor penuh `x`,
sehingga menghasilkan vektor dengan bentuk yang sama dengan `x`.
Misalnya, kita sering memiliki sebuah vektor 
yang merepresentasikan nilai fungsi loss kita
yang dihitung secara terpisah untuk setiap contoh dalam
sebuah *batch* dari contoh pelatihan.
Di sini, kita hanya ingin (**menjumlahkan gradien
yang dihitung secara individual untuk setiap contoh**).

:begin_tab:`mxnet`
MXNet menangani masalah ini dengan mereduksi semua tensor menjadi skalar 
dengan menjumlahkan sebelum menghitung gradien. 
Dengan kata lain, alih-alih mengembalikan Jacobian 
$\partial_{\mathbf{x}} \mathbf{y}$,
ia mengembalikan gradien dari jumlah
$\partial_{\mathbf{x}} \sum_i y_i$. 
:end_tab:

:begin_tab:`pytorch`
Karena framework deep learning berbeda-beda 
dalam cara mereka menafsirkan gradien dari
tensor non-skalar,
PyTorch mengambil beberapa langkah untuk menghindari kebingungan.
Memanggil `backward` pada tensor non-skalar akan menimbulkan error 
kecuali kita memberi tahu PyTorch cara mereduksi objek menjadi skalar. 
Secara lebih formal, kita perlu menyediakan beberapa vektor $\mathbf{v}$ 
sehingga `backward` akan menghitung 
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ 
alih-alih $\partial_{\mathbf{x}} \mathbf{y}$. 
Bagian berikut mungkin membingungkan,
tetapi untuk alasan yang akan menjadi jelas nanti, 
argumen ini (yang merepresentasikan $\mathbf{v}$) dinamai `gradient`. 
Untuk deskripsi lebih rinci, lihat 
[postingan Medium Yang Zhang](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29). 
:end_tab:

:begin_tab:`tensorflow`
Secara default, TensorFlow mengembalikan gradien dari jumlahnya.
Dengan kata lain, alih-alih mengembalikan 
Jacobian $\partial_{\mathbf{x


```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## Memisahkan Perhitungan

Kadang-kadang, kita ingin [**memindahkan beberapa perhitungan
di luar graf komputasi yang direkam.**]
Misalnya, kita menggunakan input 
untuk membuat beberapa istilah antara yang bersifat tambahan 
dan kita tidak ingin menghitung gradien untuknya. 
Dalam kasus ini, kita perlu *memisahkan* 
graf komputasi yang terkait dari hasil akhir. 
Contoh sederhana berikut akan memperjelas hal ini: 
misalkan kita memiliki `z = x * y` dan `y = x * x`, 
tetapi kita ingin fokus pada pengaruh *langsung* dari `x` terhadap `z` 
alih-alih pengaruh yang disalurkan melalui `y`. 
Dalam kasus ini, kita dapat membuat variabel baru `u`
yang memiliki nilai yang sama dengan `y` 
tetapi *asal-usulnya* (bagaimana itu dibuat)
telah dihapus.
Dengan demikian, `u` tidak memiliki nenek moyang dalam graf
dan gradien tidak mengalir melalui `u` ke `x`.
Sebagai contoh, menghitung gradien dari `z = x * u`
akan menghasilkan nilai `u`,
(bukan `3 * x * x` seperti yang mungkin Anda 
perkirakan karena `z = x * x * x`).


```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Setel persistent=True untuk mempertahankan graf komputasi. 
# Ini memungkinkan kita menjalankan t.gradient lebih dari sekali
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# Primitif jax.lax adalah pembungkus Python untuk operasi XLA
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

Perhatikan bahwa meskipun prosedur ini
memisahkan nenek moyang `y`
dari graf yang mengarah ke `z`, 
graf komputasi yang mengarah ke `y` 
tetap ada, sehingga kita masih dapat menghitung
gradien `y` terhadap `x`.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## Gradien dan Alur Kendali Python

Sejauh ini, kita telah meninjau kasus-kasus di mana jalur dari input ke output 
didefinisikan dengan baik melalui sebuah fungsi seperti `z = x * x * x`.
Pemrograman menawarkan kita lebih banyak kebebasan dalam cara kita menghitung hasil. 
Sebagai contoh, kita dapat membuatnya bergantung pada variabel tambahan 
atau memilih kondisi berdasarkan hasil antara. 
Salah satu keuntungan menggunakan diferensiasi otomatis
adalah bahwa [**bahkan jika**] membangun graf komputasi dari 
(**suatu fungsi memerlukan pengaturan melalui labirin alur kendali Python**)
(misalnya, pernyataan kondisional, loop, dan pemanggilan fungsi acak),
(**kita masih dapat menghitung gradien dari variabel yang dihasilkan.**)
Untuk menggambarkan hal ini, pertimbangkan potongan kode berikut di mana 
jumlah iterasi dari loop `while`
dan evaluasi dari pernyataan `if`
keduanya bergantung pada nilai dari input `a`.


```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Di bawah ini, kita memanggil fungsi ini dengan memberikan nilai acak sebagai input.
Karena input adalah variabel acak, 
kita tidak tahu bentuk apa 
yang akan diambil oleh graf komputasi.
Namun, setiap kali kita menjalankan `f(a)` 
dengan input tertentu, kita mewujudkan 
graf komputasi tertentu
dan dapat menjalankan `backward` setelahnya.


```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

Meskipun fungsi kita `f`, untuk tujuan demonstrasi, sedikit dibuat-buat,
ketergantungannya pada input cukup sederhana: 
fungsi ini adalah fungsi *linier* dari `a` 
dengan skala yang ditentukan secara sepotong-sepotong. 
Dengan demikian, `f(a) / a` adalah vektor dengan entri konstan 
dan, selain itu, `f(a) / a` harus sesuai 
dengan gradien dari `f(a)` terhadap `a`.


```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

Alur kendali dinamis sangat umum dalam deep learning. 
Misalnya, saat memproses teks, graf komputasi 
bergantung pada panjang input. 
Dalam kasus ini, diferensiasi otomatis 
menjadi sangat penting untuk pemodelan statistik 
karena mustahil menghitung gradien *a priori*. 

## Diskusi

Anda kini telah mendapatkan gambaran tentang kekuatan dari diferensiasi otomatis. 
Pengembangan pustaka untuk menghitung turunan 
baik secara otomatis maupun efisien 
telah menjadi pendorong produktivitas besar 
bagi para praktisi deep learning,
membebaskan mereka untuk dapat lebih fokus pada aspek-aspek yang tidak repetitif.
Selain itu, autograd memungkinkan kita merancang model besar 
yang perhitungan gradiennya dengan tangan 
akan sangat memakan waktu.
Menariknya, meskipun kita menggunakan autograd untuk *mengoptimalkan* model 
(dalam arti statistik),
*optimalisasi* pustaka autograd itu sendiri 
(dalam arti komputasional) 
merupakan topik yang kaya 
dan sangat penting bagi perancang framework.
Di sini, alat-alat dari compiler dan manipulasi graf 
dimanfaatkan untuk menghitung hasil 
dengan cara yang paling cepat dan efisien dalam hal memori. 

Untuk saat ini, cobalah mengingat konsep dasar berikut: (i) lampirkan gradien pada variabel-variabel terhadap mana kita ingin mencari turunan; (ii) rekam perhitungan dari nilai target; (iii) jalankan fungsi backpropagation; dan (iv) akses gradien yang dihasilkan.


## Latihan

1. Mengapa turunan kedua jauh lebih mahal untuk dihitung daripada turunan pertama?
2. Setelah menjalankan fungsi untuk backpropagation, segera jalankan lagi dan lihat apa yang terjadi. Investigasi hasilnya.
3. Dalam contoh alur kendali di mana kita menghitung turunan `d` terhadap `a`, apa yang akan terjadi jika kita mengganti variabel `a` dengan vektor acak atau matriks? Pada titik ini, hasil dari perhitungan `f(a)` tidak lagi menjadi skalar. Apa yang terjadi pada hasilnya? Bagaimana kita menganalisis ini?
4. Misalkan $f(x) = \sin(x)$. Plot grafik dari $f$ dan turunannya $f'$. Jangan gunakan fakta bahwa $f'(x) = \cos(x)$, tetapi gunakan diferensiasi otomatis untuk mendapatkan hasilnya. 
5. Misalkan $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Tuliskan graf dependensi yang melacak hasil dari $x$ ke $f(x)$. 
6. Gunakan aturan rantai untuk menghitung turunan $\frac{df}{dx}$ dari fungsi yang disebutkan sebelumnya, dengan menempatkan setiap istilah pada graf dependensi yang telah Anda bangun sebelumnya. 
7. Diberikan graf dan hasil turunan antara, Anda memiliki sejumlah opsi saat menghitung gradien. Evaluasi hasilnya sekali dari $x$ ke $f$ dan sekali dari $f$ menelusuri kembali ke $x$. Jalur dari $x$ ke $f$ umumnya dikenal sebagai *diferensiasi maju* (*forward differentiation*), sedangkan jalur dari $f$ ke $x$ dikenal sebagai diferensiasi mundur (*backward differentiation*).
8. Kapan Anda ingin menggunakan diferensiasi maju dan kapan menggunakan diferensiasi mundur? Petunjuk: pertimbangkan jumlah data antara yang diperlukan, kemampuan untuk melakukan paralelisasi langkah-langkah, dan ukuran matriks serta vektor yang terlibat.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17970)
:end_tab:
