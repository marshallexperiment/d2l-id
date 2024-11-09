```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Padding dan Stride
:label:`sec_padding`

Ingat kembali contoh konvolusi pada :numref:`fig_correlation`.
Input memiliki tinggi dan lebar masing-masing 3
dan kernel konvolusi memiliki tinggi dan lebar masing-masing 2,
menghasilkan representasi output dengan dimensi $2\times2$.
Dengan asumsi bahwa bentuk input adalah $n_\textrm{h}\times n_\textrm{w}$
dan bentuk kernel konvolusi adalah $k_\textrm{h}\times k_\textrm{w}$,
maka bentuk output akan menjadi $(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1)$:
kita hanya bisa menggeser kernel konvolusi sejauh tertentu hingga kehabisan
piksel untuk menerapkan konvolusi.

Selanjutnya kita akan mengeksplorasi sejumlah teknik,
termasuk padding dan konvolusi dengan stride,
yang memberikan lebih banyak kontrol atas ukuran output.
Sebagai motivasi, perhatikan bahwa karena kernel umumnya
memiliki lebar dan tinggi lebih besar dari $1$,
setelah menerapkan konvolusi secara berurutan,
kita cenderung mendapatkan output yang
jauh lebih kecil dari input kita.
Jika kita memulai dengan gambar berukuran $240 \times 240$ piksel,
sepuluh lapisan konvolusi $5 \times 5$
akan mengurangi gambar menjadi $200 \times 200$ piksel,
memotong $30 \%$ dari gambar dan menghilangkan
informasi penting di tepi gambar asli.
*Padding* adalah alat paling populer untuk menangani masalah ini.
Dalam kasus lain, kita mungkin ingin mengurangi dimensi secara drastis,
misalnya, jika resolusi input asli dianggap terlalu besar.
*Strided convolutions* adalah teknik populer yang dapat membantu dalam kasus seperti ini.


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

## Padding

Seperti yang dijelaskan di atas, salah satu masalah ketika menerapkan lapisan konvolusi adalah kecenderungan kehilangan piksel di tepi gambar kita. Pertimbangkan :numref:`img_conv_reuse` yang menggambarkan penggunaan piksel berdasarkan ukuran kernel konvolusi dan posisi dalam gambar. Piksel di sudut hampir tidak terpakai sama sekali.

![Penggunaan piksel untuk konvolusi berukuran $1 \times 1$, $2 \times 2$, dan $3 \times 3$.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

Karena kita biasanya menggunakan kernel yang kecil,
untuk setiap konvolusi kita mungkin hanya kehilangan beberapa piksel,
tetapi hal ini bisa bertambah seiring penerapan banyak lapisan konvolusi berturut-turut.
Satu solusi sederhana untuk masalah ini adalah dengan menambahkan piksel ekstra di sekitar batas gambar input kita,
sehingga meningkatkan ukuran efektif gambar.
Biasanya, nilai piksel ekstra ini diatur menjadi nol.
Dalam :numref:`img_conv_pad`, kita melakukan padding pada input berukuran $3 \times 3$,
meningkatkan ukurannya menjadi $5 \times 5$.
Output yang sesuai kemudian meningkat menjadi matriks berukuran $4 \times 4$.
Bagian yang diarsir menunjukkan elemen output pertama serta elemen tensor input dan kernel yang digunakan untuk perhitungan output: $0\times0+0\times1+0\times2+0\times3=0$.

![Korelasi silang dua dimensi dengan padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

Secara umum, jika kita menambahkan total $p_\textrm{h}$ baris padding
(kurang lebih setengah di atas dan setengah di bawah)
dan total $p_\textrm{w}$ kolom padding
(kurang lebih setengah di kiri dan setengah di kanan),
bentuk output akan menjadi

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$

Ini berarti bahwa tinggi dan lebar output akan meningkat masing-masing sebesar $p_\textrm{h}$ dan $p_\textrm{w}$.

Dalam banyak kasus, kita akan menetapkan $p_\textrm{h}=k_\textrm{h}-1$ dan $p_\textrm{w}=k_\textrm{w}-1$
untuk memberi input dan output tinggi dan lebar yang sama.
Ini akan memudahkan dalam memprediksi bentuk output dari setiap lapisan
saat membangun jaringan.
Dengan asumsi bahwa $k_\textrm{h}$ ganjil di sini,
kita akan melakukan padding $p_\textrm{h}/2$ baris di kedua sisi tinggi.
Jika $k_\textrm{h}$ genap, salah satu caranya adalah
melakukan padding $\lceil p_\textrm{h}/2\rceil$ baris di atas input
dan $\lfloor p_\textrm{h}/2\rfloor$ baris di bawahnya.
Kita akan melakukan padding pada kedua sisi lebar dengan cara yang sama.

CNN biasanya menggunakan kernel konvolusi
dengan tinggi dan lebar ganjil, seperti 1, 3, 5, atau 7.
Memilih ukuran kernel ganjil memiliki manfaat
bahwa kita dapat mempertahankan dimensi
sambil melakukan padding dengan jumlah baris yang sama di atas dan di bawah,
dan jumlah kolom yang sama di kiri dan kanan.

Selain itu, praktik menggunakan kernel ganjil
dan padding untuk mempertahankan dimensi
menawarkan manfaat administratif.
Untuk tensor dua dimensi `X`,
ketika ukuran kernel ganjil
dan jumlah baris serta kolom padding
di semua sisi sama,
sehingga menghasilkan output dengan tinggi dan lebar yang sama dengan input,
kita tahu bahwa output `Y[i, j]` dihitung
dengan korelasi silang antara input dan kernel konvolusi
dengan jendela yang terpusat pada `X[i, j]`.

Dalam contoh berikut, kita membuat lapisan konvolusi dua dimensi
dengan tinggi dan lebar 3
dan (**menerapkan padding 1 piksel di semua sisi.**)
Dengan input berukuran tinggi dan lebar 8,
kita menemukan bahwa tinggi dan lebar output juga 8.


```{.python .input}
%%tab mxnet
# Kita mendefinisikan fungsi pembantu untuk menghitung konvolusi. Fungsi ini menginisialisasi 
# bobot lapisan konvolusi dan melakukan penyesuaian dimensi yang diperlukan 
# pada input dan output.
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) menunjukkan bahwa ukuran batch dan jumlah channel keduanya adalah 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Menghapus dua dimensi pertama: contoh dan channel
    return Y.reshape(Y.shape[2:])

# Padding 1 baris dan kolom di setiap sisi, sehingga total 2 baris atau kolom ditambahkan
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# Kita mendefinisikan fungsi pembantu untuk menghitung konvolusi. Fungsi ini menginisialisasi 
# bobot lapisan konvolusi dan melakukan penyesuaian dimensi pada input dan output.
def comp_conv2d(conv2d, X):
    # (1, 1) menunjukkan bahwa ukuran batch dan jumlah channel keduanya adalah 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Menghilangkan dua dimensi pertama: contoh dan channel
    return Y.reshape(Y.shape[2:])

# Padding 1 baris dan 1 kolom di setiap sisi, sehingga total 2 baris atau kolom ditambahkan
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# Kita mendefinisikan fungsi pembantu untuk menghitung konvolusi. Fungsi ini menginisialisasi
# bobot lapisan konvolusi dan melakukan penyesuaian dimensi pada input dan output.
def comp_conv2d(conv2d, X):
    # (1, 1) menunjukkan bahwa ukuran batch dan jumlah channel keduanya adalah 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Menghilangkan dua dimensi pertama: contoh dan channel
    return tf.reshape(Y, Y.shape[1:3])

# Padding 1 baris dan 1 kolom di setiap sisi, sehingga total 2 baris atau kolom ditambahkan
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# Kita mendefinisikan fungsi pembantu untuk menghitung konvolusi. Fungsi ini menginisialisasi
# bobot lapisan konvolusi dan melakukan penyesuaian dimensi pada input dan output.
def comp_conv2d(conv2d, X):
    # (1, X.shape, 1) menunjukkan bahwa ukuran batch dan jumlah channel masing-masing adalah 1
    key = jax.random.PRNGKey(d2l.get_seed())
    X = X.reshape((1,) + X.shape + (1,))
    Y, _ = conv2d.init_with_output(key, X)
    # Menghilangkan dimensi contoh dan channel
    return Y.reshape(Y.shape[1:3])

# Padding 1 baris dan kolom di setiap sisi, sehingga total 2 baris atau kolom ditambahkan
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

Ketika tinggi dan lebar kernel konvolusi berbeda,
kita dapat membuat tinggi dan lebar output sama dengan input
dengan [**mengatur jumlah padding yang berbeda untuk tinggi dan lebar.**]


```{.python .input}
%%tab mxnet
# Kita menggunakan kernel konvolusi dengan tinggi 5 dan lebar 3. Padding pada
# masing-masing sisi tinggi dan lebar adalah 2 dan 1, secara berurutan
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# Kita menggunakan kernel konvolusi dengan tinggi 5 dan lebar 3. Padding pada
# masing-masing sisi tinggi dan lebar adalah 2 dan 1, secara berurutan
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# Kita menggunakan kernel konvolusi dengan tinggi 5 dan lebar 3. Padding pada
# masing-masing sisi tinggi dan lebar adalah 2 dan 1, secara berurutan
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# Kita menggunakan kernel konvolusi dengan tinggi 5 dan lebar 3. Padding pada
# masing-masing sisi tinggi dan lebar adalah 2 dan 1, secara berurutan
conv2d = nn.Conv(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## Stride

Saat menghitung korelasi silang,
kita memulai dengan jendela konvolusi
di sudut kiri atas dari tensor input,
kemudian menggesernya ke semua lokasi baik ke bawah maupun ke kanan.
Dalam contoh sebelumnya, kita secara default menggeser satu elemen setiap kali.
Namun, kadang-kadang, baik untuk efisiensi komputasi
atau karena kita ingin melakukan downsampling,
kita menggeser jendela lebih dari satu elemen setiap kali,
melewati lokasi-lokasi antara. Hal ini sangat berguna jika kernel konvolusi 
berukuran besar karena mampu menangkap area yang lebih luas dari gambar dasar.

Kita menyebut jumlah baris dan kolom yang dilalui per geseran sebagai *stride*.
Sejauh ini, kita menggunakan stride sebesar 1, baik untuk tinggi maupun lebar.
Kadang-kadang, kita mungkin ingin menggunakan stride yang lebih besar.
:numref:`img_conv_stride` menunjukkan operasi korelasi silang dua dimensi
dengan stride sebesar 3 secara vertikal dan 2 secara horizontal.
Bagian yang diarsir adalah elemen output serta elemen tensor input dan kernel yang digunakan untuk perhitungan output: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
Kita dapat melihat bahwa ketika elemen kedua dari kolom pertama dihasilkan,
jendela konvolusi bergeser ke bawah sebanyak tiga baris.
Jendela konvolusi bergeser dua kolom ke kanan
ketika elemen kedua dari baris pertama dihasilkan.
Ketika jendela konvolusi terus bergeser dua kolom ke kanan pada input,
tidak ada output yang dihasilkan karena elemen input tidak dapat memenuhi jendela
(kecuali kita menambahkan kolom padding tambahan).

![Korelasi silang dengan stride 3 dan 2 untuk tinggi dan lebar, secara berurutan.](../img/conv-stride.svg)
:label:`img_conv_stride`

Secara umum, ketika stride untuk tinggi adalah $s_\textrm{h}$
dan stride untuk lebar adalah $s_\textrm{w}$, bentuk output adalah

$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$

Jika kita menetapkan $p_\textrm{h}=k_\textrm{h}-1$ dan $p_\textrm{w}=k_\textrm{w}-1$,
maka bentuk output dapat disederhanakan menjadi
$\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$.
Lebih jauh lagi, jika tinggi dan lebar input
dapat dibagi dengan stride pada tinggi dan lebar,
maka bentuk output akan menjadi $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$.

Di bawah ini, kita [**mengatur stride untuk tinggi dan lebar menjadi 2**],
sehingga tinggi dan lebar input berkurang setengahnya.


```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 3), padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

Mari kita lihat (**contoh yang sedikit lebih rumit**).

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## Ringkasan dan Diskusi

Padding dapat meningkatkan tinggi dan lebar output. Ini sering digunakan untuk membuat output memiliki tinggi dan lebar yang sama dengan input, guna menghindari pengurangan ukuran output yang tidak diinginkan. Selain itu, padding memastikan bahwa semua piksel digunakan dengan frekuensi yang sama. Biasanya, kita memilih padding simetris pada kedua sisi tinggi dan lebar input. Dalam hal ini, kita merujuk pada padding $(p_\textrm{h}, p_\textrm{w})$. Umumnya, kita menetapkan $p_\textrm{h} = p_\textrm{w}$, sehingga kita cukup menyebutkan padding $p$. 

Konvensi serupa berlaku untuk stride. Ketika stride horizontal $s_\textrm{h}$ dan stride vertikal $s_\textrm{w}$ sama, kita cukup berbicara tentang stride $s$. Stride dapat mengurangi resolusi output, misalnya mengurangi tinggi dan lebar output menjadi hanya $1/n$ dari tinggi dan lebar input untuk $n > 1$. Secara default, padding adalah 0 dan stride adalah 1.

Sejauh ini, semua padding yang kita bahas hanya memperluas gambar dengan nilai nol. Ini memiliki manfaat komputasi yang signifikan karena sangat mudah dilakukan. Selain itu, operator dapat dirancang untuk memanfaatkan padding ini secara implisit tanpa perlu mengalokasikan memori tambahan. Pada saat yang sama, hal ini memungkinkan CNN untuk menyandikan informasi posisi implisit dalam gambar, cukup dengan mempelajari di mana "ruang kosong" berada. Ada banyak alternatif untuk zero-padding. :citet:`Alsallakh.Kokhlikyan.Miglani.ea.2020` menyediakan tinjauan mendalam tentang alternatif tersebut (meskipun tanpa kasus jelas tentang kapan harus menggunakan padding non-nol kecuali jika terjadi artefak).

## Latihan

1. Diberikan contoh kode terakhir dalam bagian ini dengan ukuran kernel $(3, 5)$, padding $(0, 1)$, dan stride $(3, 4)$,
   hitung bentuk output untuk memeriksa apakah konsisten dengan hasil eksperimen.
1. Untuk sinyal audio, apa arti dari stride sebesar 2?
1. Implementasikan mirror padding, yaitu padding di mana nilai pada tepi cermin digunakan untuk memperpanjang tensor.
1. Apa manfaat komputasi dari stride yang lebih besar dari 1?
1. Apa manfaat statistik dari stride yang lebih besar dari 1?
1. Bagaimana Anda akan mengimplementasikan stride sebesar $\frac{1}{2}$? Apa artinya? Kapan ini akan berguna?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/272)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17997)
:end_tab:
