```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Banyak Kanal Input dan Output
:label:`sec_channels`

Meskipun kita telah menjelaskan beberapa kanal yang membentuk setiap gambar (misalnya, gambar berwarna memiliki kanal RGB standar untuk menunjukkan jumlah merah, hijau, dan biru) dan lapisan konvolusi untuk beberapa kanal di :numref:`subsec_why-conv-channels`,
hingga sekarang, kita menyederhanakan semua contoh numerik kita
dengan hanya bekerja dengan satu kanal input dan satu kanal output.
Ini memungkinkan kita memikirkan input, kernel konvolusi,
dan output masing-masing sebagai tensor dua dimensi.

Ketika kita menambahkan kanal dalam perhitungan,
input dan representasi tersembunyi kita
keduanya menjadi tensor tiga dimensi.
Sebagai contoh, setiap gambar RGB memiliki bentuk $3\times h\times w$.
Kita merujuk pada sumbu ini, dengan ukuran 3, sebagai dimensi *kanal*. Konsep kanal sudah ada sejak pertama kali CNN diperkenalkan: misalnya, LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995` menggunakannya.
Di bagian ini, kita akan melihat lebih dalam
pada kernel konvolusi dengan beberapa kanal input dan beberapa kanal output.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Multiple Channel Input

Ketika data input memiliki beberapa kanal,
kita perlu membangun kernel konvolusi
dengan jumlah kanal input yang sama dengan data input,
sehingga dapat melakukan operasi korelasi silang dengan data input.
Dengan asumsi bahwa jumlah kanal untuk data input adalah $c_\textrm{i}$,
maka jumlah kanal input dari kernel konvolusi juga perlu $c_\textrm{i}$. Jika bentuk jendela kernel konvolusi kita adalah $k_\textrm{h}\times k_\textrm{w}$,
maka, ketika $c_\textrm{i}=1$, kita dapat menganggap kernel konvolusi kita
sebagai tensor dua dimensi dengan bentuk $k_\textrm{h}\times k_\textrm{w}$.

Namun, ketika $c_\textrm{i}>1$, kita membutuhkan kernel
yang memiliki tensor berukuran $k_\textrm{h}\times k_\textrm{w}$ untuk *setiap* kanal input. Menggabungkan $c_\textrm{i}$ tensor ini bersama-sama
menghasilkan kernel konvolusi dengan bentuk $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$.
Karena input dan kernel konvolusi masing-masing memiliki $c_\textrm{i}$ kanal,
kita dapat melakukan operasi korelasi silang
pada tensor dua dimensi dari input
dan tensor dua dimensi dari kernel konvolusi
untuk setiap kanal, menambahkan hasil $c_\textrm{i}$ bersama-sama
(menjumlahkan sepanjang kanal)
untuk menghasilkan tensor dua dimensi.
Ini adalah hasil dari korelasi silang dua dimensi
antara input multi-kanal dan
kernel konvolusi dengan multi-kanal input.

:numref:`fig_conv_multi_in` memberikan contoh
korelasi silang dua dimensi dengan dua kanal input.
Bagian yang diarsir adalah elemen output pertama
serta elemen tensor input dan kernel yang digunakan untuk perhitungan output:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Perhitungan korelasi silang dengan dua kanal input.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

Untuk memastikan kita benar-benar memahami apa yang terjadi di sini,
kita dapat (**mengimplementasikan operasi korelasi silang dengan beberapa kanal input**) sendiri.
Perhatikan bahwa yang kita lakukan hanyalah melakukan operasi korelasi silang
untuk setiap kanal dan kemudian menjumlahkan hasilnya.

```{.python .input}
%%tab mxnet, pytorch, jax
def corr2d_multi_in(X, K):
    # Iterasi melalui dimensi ke-0 (kanal) dari K terlebih dahulu, kemudian jumlahkan hasilnya
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
%%tab tensorflow
def corr2d_multi_in(X, K):
    # Iterasi melalui dimensi ke-0 (kanal) dari K terlebih dahulu, kemudian jumlahkan hasilnya
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

Kita dapat membangun tensor input `X` dan tensor kernel `K`
yang sesuai dengan nilai pada :numref:`fig_conv_multi_in`
untuk (**memvalidasi output**) dari operasi korelasi silang.


```{.python .input}
%%tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Multiple Channel Output
:label:`subsec_multi-output-channels`

Terlepas dari jumlah kanal input,
sejauh ini kita selalu berakhir dengan satu kanal output.
Namun, seperti yang kita bahas di :numref:`subsec_why-conv-channels`,
ternyata memiliki beberapa kanal di setiap lapisan adalah penting.
Pada arsitektur neural network yang paling populer,
dimensi kanal biasanya ditingkatkan seiring dengan bertambahnya kedalaman jaringan,
dengan downsampling untuk mengimbangi resolusi spasial
dengan *kedalaman kanal* yang lebih besar.
Secara intuitif, kita bisa menganggap setiap kanal
sebagai respon terhadap serangkaian fitur yang berbeda.
Kenyataannya sedikit lebih rumit dari ini. Interpretasi naif mungkin menyiratkan 
bahwa representasi dipelajari secara independen per piksel atau per kanal. 
Namun, kanal-kanal dioptimalkan untuk menjadi berguna secara bersama-sama.
Ini berarti bahwa daripada memetakan satu kanal ke detektor tepi, ini mungkin hanya berarti 
bahwa beberapa arah dalam ruang kanal berkorespondensi dengan pendeteksian tepi.

Misalkan $c_\textrm{i}$ dan $c_\textrm{o}$ adalah jumlah
kanal input dan output, berturut-turut,
dan $k_\textrm{h}$ dan $k_\textrm{w}$ adalah tinggi dan lebar kernel.
Untuk mendapatkan output dengan banyak kanal,
kita bisa membuat tensor kernel
dengan bentuk $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$
untuk *setiap* kanal output.
Kita menggabungkan semua kernel ini pada dimensi kanal output,
sehingga bentuk kernel konvolusi menjadi
$c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$.
Dalam operasi korelasi silang,
hasil pada setiap kanal output dihitung
dari kernel konvolusi yang sesuai dengan kanal output tersebut
dan mengambil input dari semua kanal dalam tensor input.

Kita mengimplementasikan fungsi korelasi silang
untuk [**menghitung output dari beberapa kanal**] seperti yang ditunjukkan di bawah ini.


```{.python .input}
%%tab all
def corr2d_multi_in_out(X, K):
    # Iterasi melalui dimensi ke-0 dari K, dan setiap kali, lakukan
    # operasi korelasi silang dengan input X. Semua hasilnya
    # ditumpuk bersama
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Kita membangun kernel konvolusi sederhana dengan tiga kanal output
dengan menggabungkan tensor kernel untuk `K` dengan `K+1` dan `K+2`.


```{.python .input}
%%tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Di bawah ini, kita melakukan operasi korelasi silang
pada tensor input `X` dengan tensor kernel `K`.
Sekarang output memiliki tiga kanal.
Hasil pada kanal pertama konsisten
dengan hasil dari tensor input `X` sebelumnya
dan kernel dengan banyak kanal input serta satu kanal output.


```{.python .input}
%%tab all
corr2d_multi_in_out(X, K)
```

## Lapisan Konvolusi $1\times 1$
:label:`subsec_1x1`

Pada awalnya, [**konvolusi $1 \times 1**], yaitu $k_\textrm{h} = k_\textrm{w} = 1$, tampak kurang masuk akal.
Bagaimanapun, konvolusi biasanya mengkorelasikan piksel yang berdekatan.
Namun, konvolusi $1 \times 1$ jelas tidak demikian.
Meski begitu, operasi ini populer dan kadang-kadang dimasukkan
dalam desain jaringan dalam deep learning yang kompleks :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`.
Mari kita lihat secara detail apa yang sebenarnya dilakukan oleh operasi ini.

Karena menggunakan jendela minimum,
konvolusi $1\times 1$ kehilangan kemampuan
lapisan konvolusi yang lebih besar
untuk mengenali pola yang terdiri dari interaksi
antara elemen-elemen berdekatan dalam dimensi tinggi dan lebar.
Satu-satunya perhitungan pada konvolusi $1\times 1$
terjadi pada dimensi kanal.

:numref:`fig_conv_1x1` menunjukkan perhitungan korelasi silang
menggunakan kernel konvolusi $1\times 1$
dengan 3 kanal input dan 2 kanal output.
Perhatikan bahwa input dan output memiliki tinggi dan lebar yang sama.
Setiap elemen pada output dihasilkan
dari kombinasi linear elemen *pada posisi yang sama*
dalam gambar input.
Anda dapat menganggap lapisan konvolusi $1\times 1$
sebagai lapisan fully connected yang diterapkan pada setiap lokasi piksel,
untuk mengubah $c_\textrm{i}$ nilai input yang sesuai menjadi $c_\textrm{o}$ nilai output.
Karena ini masih merupakan lapisan konvolusi,
bobot digunakan bersama di seluruh lokasi piksel.
Dengan demikian, lapisan konvolusi $1\times 1$ membutuhkan $c_\textrm{o}\times c_\textrm{i}$ bobot
(ditambah bias). Selain itu, lapisan konvolusi biasanya diikuti oleh non-linearitas. Ini memastikan bahwa konvolusi $1 \times 1$ tidak dapat begitu saja digabungkan dengan konvolusi lain.

![Perhitungan korelasi silang menggunakan kernel konvolusi $1\times 1$ dengan tiga kanal input dan dua kanal output. Input dan output memiliki tinggi dan lebar yang sama.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Mari kita periksa apakah ini benar-benar berfungsi dalam praktik:
kita mengimplementasikan konvolusi $1 \times 1$
menggunakan lapisan fully connected.
Satu-satunya hal yang perlu kita lakukan adalah menyesuaikan
bentuk data sebelum dan sesudah perkalian matriks.


```{.python .input}
%%tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

Ketika melakukan konvolusi $1\times 1$,
fungsi di atas setara dengan fungsi korelasi silang `corr2d_multi_in_out` yang telah diimplementasikan sebelumnya.
Mari kita periksa ini dengan beberapa data sampel.


```{.python .input}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab jax
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (3, 3, 3)) + 0 * 1
K = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 3, 1, 1)) + 0 * 1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Diskusi

Kanal memungkinkan kita menggabungkan yang terbaik dari dua dunia: MLP yang memungkinkan non-linearitas signifikan dan konvolusi yang memungkinkan analisis fitur *lokal*. Secara khusus, kanal memungkinkan CNN untuk memproses berbagai fitur secara bersamaan, seperti deteksi tepi dan bentuk. Kanal juga menawarkan keseimbangan praktis antara pengurangan parameter yang drastis akibat translasi invarian dan lokalitas, serta kebutuhan akan model yang ekspresif dan beragam dalam computer vision.

Namun, fleksibilitas ini memiliki harga. Diberikan gambar berukuran $(h \times w)$, biaya untuk menghitung konvolusi $k \times k$ adalah $\mathcal{O}(h \cdot w \cdot k^2)$. Untuk kanal input $c_\textrm{i}$ dan kanal output $c_\textrm{o}$, biaya ini meningkat menjadi $\mathcal{O}(h \cdot w \cdot k^2 \cdot c_\textrm{i} \cdot c_\textrm{o})$. Untuk gambar 256 x 256 piksel dengan kernel 5 x 5 dan masing-masing 128 kanal input dan output, ini menghasilkan lebih dari 53 miliar operasi (dengan perkalian dan penjumlahan dihitung secara terpisah). Nanti, kita akan mempelajari strategi yang efektif untuk mengurangi biaya ini, misalnya, dengan memaksa operasi antar-kanal menjadi berbentuk blok-diagonal, yang menghasilkan arsitektur seperti ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`.

## Latihan

1. Misalkan kita memiliki dua kernel konvolusi masing-masing berukuran $k_1$ dan $k_2$
   (tanpa non-linearitas di antaranya).
    1. Buktikan bahwa hasil operasi ini dapat diekspresikan dengan satu konvolusi tunggal.
    1. Berapa dimensi konvolusi tunggal yang setara?
    1. Apakah hal sebaliknya benar, yaitu, bisakah Anda selalu memecah sebuah konvolusi menjadi dua konvolusi yang lebih kecil?
1. Misalkan input berukuran $c_\textrm{i}\times h\times w$ dan kernel konvolusi berukuran 
   $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$, dengan padding $(p_\textrm{h}, p_\textrm{w})$ dan stride $(s_\textrm{h}, s_\textrm{w})$.
    1. Berapa biaya komputasi (perkalian dan penjumlahan) untuk propagasi maju?
    1. Berapa jejak memori yang dibutuhkan?
    1. Berapa jejak memori untuk komputasi mundur?
    1. Berapa biaya komputasi untuk backpropagation?
1. Dengan faktor berapa jumlah perhitungan meningkat jika kita menggandakan jumlah kanal input 
   $c_\textrm{i}$ dan jumlah kanal output $c_\textrm{o}$? Apa yang terjadi jika kita menggandakan padding?
1. Apakah variabel `Y1` dan `Y2` dalam contoh akhir bagian ini persis sama? Mengapa?
1. Ekspresikan konvolusi sebagai perkalian matriks, bahkan ketika jendela konvolusi bukan $1 \times 1$.
1. Tugas Anda adalah mengimplementasikan konvolusi cepat dengan kernel $k \times k$. Salah satu kandidat algoritma adalah memindai secara horizontal, membaca strip selebar $k$ dan menghitung strip output selebar $1$ satu nilai setiap kali. Alternatifnya adalah membaca strip selebar $k + \Delta$ dan menghitung strip output selebar $\Delta$. Mengapa alternatif kedua lebih disukai? Apakah ada batasan seberapa besar Anda harus memilih $\Delta$?
1. Misalkan kita memiliki matriks $c \times c$.
    1. Seberapa cepat perkalian dilakukan dengan matriks blok-diagonal jika matriks dibagi menjadi $b$ blok?
    1. Apa kekurangan dari memiliki $b$ blok? Bagaimana Anda dapat memperbaikinya, setidaknya sebagian?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/273)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17998)
:end_tab:
