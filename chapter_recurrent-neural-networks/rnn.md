# Recurrent Neural Networks
:label:`sec_rnn`

Pada :numref:`sec_language-model`, kita telah menjelaskan model Markov dan $n$-gram untuk pemodelan bahasa, di mana probabilitas bersyarat dari token $x_t$ pada langkah waktu $t$ hanya bergantung pada $n-1$ token sebelumnya.
Jika kita ingin memasukkan kemungkinan efek dari token yang lebih awal dari langkah waktu $t-(n-1)$ pada $x_t$,
kita perlu meningkatkan $n$.
Namun, jumlah parameter model juga akan meningkat secara eksponensial bersamanya, karena kita perlu menyimpan $|\mathcal{V}|^n$ angka untuk set kosakata $\mathcal{V}$.
Oleh karena itu, daripada memodelkan $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$, lebih disukai untuk menggunakan model variabel laten,

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

di mana $h_{t-1}$ adalah *hidden state* yang menyimpan informasi urutan hingga langkah waktu $t-1$.
Secara umum,
*hidden state* pada setiap langkah waktu $t$ dapat dihitung berdasarkan input saat ini $x_{t}$ dan *hidden state* sebelumnya $h_{t-1}$:

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

Dengan fungsi $f$ yang cukup kuat dalam :eqref:`eq_ht_xt`, model variabel laten bukanlah sebuah aproksimasi. Bagaimanapun, $h_t$ dapat menyimpan semua data yang telah diamati sejauh ini.
Namun, ini dapat membuat komputasi dan penyimpanan menjadi mahal.

Ingat bahwa kita telah membahas lapisan tersembunyi dengan unit tersembunyi pada :numref:`chap_perceptrons`.
Perlu dicatat bahwa
lapisan tersembunyi dan *hidden states* mengacu pada dua konsep yang sangat berbeda.
Lapisan tersembunyi, seperti yang dijelaskan, adalah lapisan yang tersembunyi dari pandangan pada jalur dari input ke output.
*Hidden states* secara teknis merupakan *input* untuk apa pun yang kita lakukan pada suatu langkah tertentu,
dan hanya dapat dihitung dengan melihat data pada langkah waktu sebelumnya.

*Recurrent Neural Networks* (RNNs) adalah jaringan saraf dengan *hidden states*. Sebelum memperkenalkan model RNN, pertama-tama kita meninjau kembali model MLP yang diperkenalkan pada :numref:`sec_mlp`.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

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
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

## Neural Networks tanpa Hidden States

Mari kita lihat sebuah MLP dengan satu lapisan tersembunyi.
Misalkan fungsi aktivasi pada lapisan tersembunyi adalah $\phi$.
Dengan diberikan *minibatch* dari contoh $\mathbf{X} \in \mathbb{R}^{n \times d}$ dengan ukuran batch $n$ dan $d$ masukan, keluaran lapisan tersembunyi $\mathbf{H} \in \mathbb{R}^{n \times h}$ dihitung sebagai

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_without_state`

Dalam :eqref:`rnn_h_without_state`, kita memiliki parameter bobot $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$, parameter bias $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$, dan jumlah unit tersembunyi $h$ untuk lapisan tersembunyi.
Setelah itu, kita menerapkan *broadcasting* (lihat :numref:`subsec_broadcasting`) selama penjumlahan.
Selanjutnya, keluaran lapisan tersembunyi $\mathbf{H}$ digunakan sebagai input lapisan keluaran, yang diberikan oleh

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

di mana $\mathbf{O} \in \mathbb{R}^{n \times q}$ adalah variabel output, $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ adalah parameter bobot, dan $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ adalah parameter bias dari lapisan keluaran. Jika ini adalah masalah klasifikasi, kita dapat menggunakan $\mathrm{softmax}(\mathbf{O})$ untuk menghitung distribusi probabilitas dari kategori output.

Ini sepenuhnya analog dengan masalah regresi yang kita selesaikan sebelumnya di :numref:`sec_sequence`, jadi kita menghilangkan detailnya.
Cukup dikatakan bahwa kita dapat memilih pasangan fitur-label secara acak dan mempelajari parameter jaringan kita melalui diferensiasi otomatis dan *stochastic gradient descent*.


## Recurrent Neural Networks dengan Hidden States
:label:`subsec_rnn_w_hidden_states`

Segalanya menjadi sangat berbeda ketika kita memiliki *hidden state*. Mari kita lihat strukturnya secara lebih rinci.

Misalkan kita memiliki *minibatch* dari input 
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
pada langkah waktu $t$.
Dengan kata lain,
untuk *minibatch* dari $n$ contoh urutan,
setiap baris dari $\mathbf{X}_t$ sesuai dengan satu contoh pada langkah waktu $t$ dari urutan tersebut.
Selanjutnya,
misalkan $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ adalah output lapisan tersembunyi pada langkah waktu $t$.
Berbeda dengan MLP, di sini kita menyimpan output lapisan tersembunyi $\mathbf{H}_{t-1}$ dari langkah waktu sebelumnya dan memperkenalkan parameter bobot baru $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ untuk menggambarkan cara menggunakan output lapisan tersembunyi dari langkah waktu sebelumnya pada langkah waktu saat ini. Secara khusus, perhitungan output lapisan tersembunyi pada langkah waktu saat ini ditentukan oleh input pada langkah waktu saat ini bersamaan dengan output lapisan tersembunyi dari langkah waktu sebelumnya:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_with_state`

Dibandingkan dengan :eqref:`rnn_h_without_state`, :eqref:`rnn_h_with_state` menambahkan satu suku lagi $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ dan dengan demikian mewujudkan :eqref:`eq_ht_xt`.
Dari hubungan antara output lapisan tersembunyi $\mathbf{H}_t$ dan $\mathbf{H}_{t-1}$ pada langkah waktu yang berdekatan,
kita tahu bahwa variabel-variabel ini menangkap dan menyimpan informasi historis urutan hingga langkah waktu saat ini, seperti halnya status atau memori langkah waktu saat ini dari jaringan saraf. Oleh karena itu, output lapisan tersembunyi semacam ini disebut sebagai *hidden state*.
Karena *hidden state* menggunakan definisi yang sama dari langkah waktu sebelumnya pada langkah waktu saat ini, perhitungan :eqref:`rnn_h_with_state` bersifat *recurrent*. Oleh karena itu, seperti yang dikatakan sebelumnya, jaringan saraf dengan *hidden states* yang berdasarkan perhitungan berulang dinamakan *recurrent neural networks*.
Lapisan yang melakukan perhitungan :eqref:`rnn_h_with_state` dalam RNN disebut *recurrent layers*.

Ada banyak cara berbeda untuk membangun RNN.
RNN dengan *hidden state* yang didefinisikan oleh :eqref:`rnn_h_with_state` adalah yang paling umum.
Pada langkah waktu $t$,
output dari lapisan output mirip dengan perhitungan dalam MLP:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Parameter RNN meliputi bobot $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$,
dan bias $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$
dari lapisan tersembunyi,
bersama dengan bobot $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$
dan bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
dari lapisan output.
Perlu dicatat bahwa
bahkan pada langkah waktu yang berbeda,
RNN selalu menggunakan parameter model ini.
Oleh karena itu, biaya parametrisasi dari sebuah RNN
tidak bertambah seiring dengan meningkatnya jumlah langkah waktu.

:numref:`fig_rnn` menggambarkan logika komputasi dari sebuah RNN pada tiga langkah waktu yang berdekatan.
Pada langkah waktu $t$ mana pun,
perhitungan *hidden state* dapat dianggap sebagai:
(i) menggabungkan input $\mathbf{X}_t$ pada langkah waktu saat ini $t$ dan *hidden state* $\mathbf{H}_{t-1}$ pada langkah waktu sebelumnya $t-1$;
(ii) memasukkan hasil penggabungan tersebut ke dalam lapisan *fully connected* dengan fungsi aktivasi $\phi$.
Output dari lapisan *fully connected* ini adalah *hidden state* $\mathbf{H}_t$ pada langkah waktu saat ini $t$.
Dalam hal ini,
parameter model adalah penggabungan dari $\mathbf{W}_{\textrm{xh}}$ dan $\mathbf{W}_{\textrm{hh}}$, serta bias $\mathbf{b}_\textrm{h}$, semuanya dari :eqref:`rnn_h_with_state`.
*Hidden state* dari langkah waktu saat ini $t$, $\mathbf{H}_t$, akan berpartisipasi dalam menghitung *hidden state* $\mathbf{H}_{t+1}$ pada langkah waktu berikutnya $t+1$.
Selain itu, $\mathbf{H}_t$ juga akan
dimasukkan ke dalam lapisan *fully connected* output
untuk menghitung output
$\mathbf{O}_t$ dari langkah waktu saat ini $t$.

![Sebuah RNN dengan *hidden state*.](../img/rnn.svg)
:label:`fig_rnn`

Kita baru saja menyebutkan bahwa perhitungan $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ untuk *hidden state* setara dengan
perkalian matriks dari penggabungan $\mathbf{X}_t$ dan $\mathbf{H}_{t-1}$
dengan penggabungan $\mathbf{W}_{\textrm{xh}}$ dan $\mathbf{W}_{\textrm{hh}}$.
Meskipun ini dapat dibuktikan secara matematis,
berikut ini kita hanya akan menggunakan cuplikan kode sederhana sebagai demonstrasi.
Untuk memulai,
kita mendefinisikan matriks `X`, `W_xh`, `H`, dan `W_hh`, yang masing-masing berbentuk (3, 1), (1, 4), (3, 4), dan (4, 4).
Dengan mengalikan `X` dengan `W_xh`, dan `H` dengan `W_hh`, lalu menjumlahkan kedua hasil perkalian ini,
kita akan memperoleh matriks berbentuk (3, 4).


```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab jax
X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
                                                        d2l.get_key(), (1, 4))
H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
                                                        d2l.get_key(), (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Sekarang kita menggabungkan matriks `X` dan `H`
sepanjang kolom (sumbu 1),
dan matriks
`W_xh` dan `W_hh` sepanjang baris (sumbu 0).
Kedua penggabungan ini
menghasilkan
matriks dengan bentuk (3, 5)
dan (5, 4), masing-masing.
Dengan mengalikan kedua matriks yang telah digabungkan ini,
kita akan memperoleh matriks output dengan bentuk (3, 4)
sama seperti sebelumnya.


```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## Model Bahasa Tingkat Karakter Berbasis RNN

Ingat kembali bahwa untuk pemodelan bahasa di :numref:`sec_language-model`,
tujuan kita adalah memprediksi token berikutnya berdasarkan
token saat ini dan token sebelumnya;
oleh karena itu kita menggeser urutan asli sebesar satu token
sebagai target (label).
:citet:`Bengio.Ducharme.Vincent.ea.2003` pertama kali mengusulkan
untuk menggunakan jaringan saraf dalam pemodelan bahasa.
Berikut ini kita ilustrasikan bagaimana RNN dapat digunakan untuk membangun model bahasa.
Misalkan ukuran *minibatch* adalah satu, dan urutan teksnya adalah "machine".
Untuk menyederhanakan pelatihan di bagian selanjutnya,
kita memisahkan teks menjadi karakter alih-alih kata
dan mempertimbangkan *model bahasa tingkat karakter*.
:numref:`fig_rnn_train` menunjukkan bagaimana memprediksi karakter berikutnya berdasarkan karakter saat ini dan karakter sebelumnya melalui RNN untuk pemodelan bahasa tingkat karakter.

![Model bahasa tingkat karakter berbasis RNN. Urutan input dan target masing-masing adalah "machin" dan "achine".](../img/rnn-train.svg)
:label:`fig_rnn_train`

Selama proses pelatihan,
kita menjalankan operasi *softmax* pada output dari lapisan output untuk setiap langkah waktu, lalu menggunakan *cross-entropy loss* untuk menghitung kesalahan antara output model dan target.
Karena perhitungan berulang dari *hidden state* di lapisan tersembunyi, output $\mathbf{O}_3$ pada langkah waktu ke-3 di :numref:`fig_rnn_train` ditentukan oleh urutan teks "m", "a", dan "c". Karena karakter berikutnya dalam urutan pada data pelatihan adalah "h", maka *loss* pada langkah waktu ke-3 akan bergantung pada distribusi probabilitas dari karakter berikutnya yang dihasilkan berdasarkan urutan fitur "m", "a", "c" dan target "h" pada langkah waktu ini.

Dalam praktiknya, setiap token direpresentasikan oleh vektor berdimensi $d$, dan kita menggunakan ukuran batch $n>1$. Oleh karena itu, input $\mathbf X_t$ pada langkah waktu $t$ akan menjadi matriks $n\times d$, yang identik dengan yang kita bahas di :numref:`subsec_rnn_w_hidden_states`.

Pada bagian berikut, kita akan mengimplementasikan RNN
untuk model bahasa tingkat karakter.


## Ringkasan

Jaringan saraf yang menggunakan perhitungan berulang untuk *hidden state* disebut *recurrent neural network* (RNN).
*Hidden state* dari RNN dapat menangkap informasi historis urutan hingga langkah waktu saat ini. Dengan perhitungan berulang, jumlah parameter model RNN tidak bertambah seiring bertambahnya jumlah langkah waktu. Dalam aplikasi, RNN dapat digunakan untuk membuat model bahasa tingkat karakter.


## Latihan

1. Jika kita menggunakan RNN untuk memprediksi karakter berikutnya dalam urutan teks, berapakah dimensi yang diperlukan untuk setiap output?
2. Mengapa RNN dapat mengekspresikan probabilitas bersyarat dari token pada langkah waktu tertentu berdasarkan semua token sebelumnya dalam urutan teks?
3. Apa yang terjadi pada gradien jika Anda melakukan *backpropagation* melalui urutan yang panjang?
4. Apa saja masalah yang terkait dengan model bahasa yang dijelaskan di bagian ini?


:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1051)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/180013)
:end_tab:
