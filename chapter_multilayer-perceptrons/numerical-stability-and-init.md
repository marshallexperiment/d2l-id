```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Numerical Stability and Initialization
:label:`sec_numerical_stability`

Sejauh ini, setiap model yang telah kita implementasikan
memerlukan inisialisasi parameter
sesuai dengan distribusi yang telah ditentukan sebelumnya.
Sampai saat ini, kita menerima begitu saja skema inisialisasi ini,
tanpa membahas detail mengenai cara pengambilan keputusan ini.
Anda mungkin bahkan mendapat kesan bahwa pilihan ini
tidak terlalu penting.
Sebaliknya, pemilihan skema inisialisasi
memainkan peran penting dalam pembelajaran jaringan saraf,
dan hal ini bisa sangat krusial untuk menjaga stabilitas numerik.
Lebih jauh lagi, pilihan ini dapat berkaitan erat
dengan pemilihan fungsi aktivasi non-linear.
Fungsi aktivasi yang kita pilih dan cara kita menginisialisasi parameter
dapat menentukan seberapa cepat algoritma optimisasi kita berkonvergensi.
Pilihan yang buruk di sini dapat menyebabkan kita menghadapi
gradien yang meledak atau menghilang saat pelatihan.
Pada bagian ini, kita akan mendalami topik-topik ini lebih jauh
dan membahas beberapa heuristik yang berguna
yang akan bermanfaat sepanjang karir Anda dalam deep learning.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## Vanishing dan Exploding Gradients

Pertimbangkan jaringan dalam dengan $L$ lapisan,
input $\mathbf{x}$ dan output $\mathbf{o}$.
Dengan setiap lapisan $l$ didefinisikan oleh transformasi $f_l$
yang diparameterisasi oleh bobot $\mathbf{W}^{(l)}$,
dengan output hidden layer $\mathbf{h}^{(l)}$ (dengan asumsi $\mathbf{h}^{(0)} = \mathbf{x}$),
jaringan kita dapat dinyatakan sebagai:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ dan dengan demikian } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

Jika semua output hidden layer dan input adalah vektor,
kita dapat menulis gradien dari $\mathbf{o}$ terhadap
sekumpulan parameter $\mathbf{W}^{(l)}$ sebagai berikut:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

Dengan kata lain, gradien ini adalah
produk dari $L-l$ matriks
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
dan vektor gradien $\mathbf{v}^{(l)}$.
Dengan demikian, kita rentan terhadap masalah
underflow numerik yang sering muncul
saat mengalikan terlalu banyak probabilitas bersama-sama.
Saat menangani probabilitas, trik umum adalah beralih ke log-space, yaitu, menggeser
tekanan dari mantissa ke eksponen dalam representasi numerik.
Sayangnya, masalah kita di atas lebih serius:
matriks $\mathbf{M}^{(l)}$ pada awalnya mungkin memiliki berbagai nilai eigen.
Nilai-nilai tersebut bisa kecil atau besar,
dan hasil perkaliannya bisa menjadi *sangat besar* atau *sangat kecil*.

Risiko yang disebabkan oleh gradien yang tidak stabil
melampaui masalah representasi numerik.
Gradien dengan besar yang tidak dapat diprediksi
juga mengancam stabilitas algoritma optimisasi kita.
Kita mungkin menghadapi pembaruan parameter yang
(i) sangat besar, yang dapat merusak model kita
(masalah *exploding gradient*);
atau (ii) sangat kecil
(masalah *vanishing gradient*),
yang membuat pembelajaran menjadi mustahil karena parameter
nyaris tidak bergerak pada setiap pembaruan.


### (**Vanishing Gradients**)

Salah satu penyebab umum masalah vanishing gradient
adalah pemilihan fungsi aktivasi $\sigma$
yang ditambahkan setelah setiap operasi linear dari lapisan.
Secara historis, fungsi sigmoid
$1/(1 + \exp(-x))$ (diperkenalkan di :numref:`sec_mlp`)
populer karena mirip dengan fungsi ambang batas.
Karena jaringan saraf tiruan awal terinspirasi
oleh jaringan saraf biologis,
gagasan tentang neuron yang menembak secara *penuh* atau *tidak sama sekali*
(seperti neuron biologis) tampak menarik.
Mari kita lihat lebih dekat sigmoid
untuk memahami mengapa fungsi ini bisa menyebabkan vanishing gradients.


```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Seperti yang Anda lihat, (**gradien sigmoid akan menghilang
baik ketika inputnya besar maupun kecil**).
Selain itu, ketika melakukan backpropagation melalui banyak lapisan,
kecuali kita berada dalam zona "Goldilocks", di mana
input ke banyak fungsi sigmoid mendekati nol,
gradien dari keseluruhan produk mungkin akan hilang.
Ketika jaringan kita memiliki banyak lapisan,
kecuali kita berhati-hati, gradien
kemungkinan besar akan terputus pada beberapa lapisan.
Masalah ini memang sering menjadi hambatan
dalam pelatihan jaringan yang dalam di masa lalu.
Akibatnya, ReLU, yang lebih stabil
(tetapi kurang mendekati neuron biologis),
muncul sebagai pilihan default bagi praktisi.


### [**Exploding Gradients**]

Masalah yang berlawanan, yaitu saat gradien meledak,
bisa sama menjengkelkannya.
Untuk mengilustrasikan ini dengan lebih baik,
kita menggambar 100 matriks acak Gaussian
dan mengalikan matriks-matriks tersebut dengan matriks awal.
Untuk skala yang kita pilih
(variansi $\sigma^2=1$),
produk matriks meledak.
Ketika hal ini terjadi karena inisialisasi
dari jaringan yang dalam, kita tidak memiliki peluang
untuk mendapatkan konvergensi pada optimizer gradient descent.


```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # men-generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### Breaking the Symmetry

Masalah lain dalam desain neural network
adalah simetri yang melekat dalam parameterisasi mereka.
Misalkan kita memiliki MLP sederhana
dengan satu hidden layer dan dua unit.
Dalam kasus ini, kita bisa menukar bobot $\mathbf{W}^{(1)}$
dari lapisan pertama dan menukar
bobot dari lapisan output
untuk mendapatkan fungsi yang sama.
Tidak ada yang membedakan
unit tersembunyi pertama dan kedua.
Dengan kata lain, kita memiliki simetri permutasi
di antara unit-unit tersembunyi dari setiap lapisan.

Ini lebih dari sekadar gangguan teoretis.
Pertimbangkan MLP satu-hidden-layer yang disebutkan sebelumnya
dengan dua unit tersembunyi.
Sebagai ilustrasi,
anggap bahwa lapisan output mengubah dua unit tersembunyi menjadi satu unit output.
Bayangkan apa yang akan terjadi jika kita menginisialisasi
semua parameter lapisan tersembunyi sebagai $\mathbf{W}^{(1)} = c$ untuk beberapa konstanta $c$.
Dalam kasus ini, selama forward propagation,
kedua unit tersembunyi menerima input dan parameter yang sama,
menghasilkan aktivasi yang sama
yang diteruskan ke unit output.
Selama backpropagation,
mendiferensiasi unit output terhadap parameter $\mathbf{W}^{(1)}$ memberikan gradien di mana semua elemennya memiliki nilai yang sama.
Oleh karena itu, setelah iterasi berbasis gradien (misalnya, stochastic gradient descent minibatch),
semua elemen dari $\mathbf{W}^{(1)}$ masih memiliki nilai yang sama.
Iterasi seperti ini tidak akan pernah *memecahkan simetri* dengan sendirinya
dan kita mungkin tidak akan pernah bisa mewujudkan
kekuatan ekspresif dari jaringan tersebut.
Lapisan tersembunyi akan bertindak
seolah-olah hanya memiliki satu unit.
Perhatikan bahwa sementara stochastic gradient descent minibatch tidak akan memecah simetri ini,
regularisasi dropout (yang akan diperkenalkan nanti) bisa!


## Inisialisasi Parameter 

Salah satu cara untuk mengatasi---atau setidaknya mengurangi---masalah-masalah yang disebutkan di atas adalah melalui inisialisasi yang hati-hati.
Seperti yang akan kita lihat nanti,
perhatian tambahan selama optimasi
dan regularisasi yang sesuai dapat lebih meningkatkan stabilitas.


### Insisialisasi _Default_

Di bagian sebelumnya, misalnya di :numref:`sec_linear_concise`,
kita menggunakan distribusi normal
untuk menginisialisasi nilai bobot kita.
Jika kita tidak menentukan metode inisialisasi, framework akan
menggunakan metode inisialisasi acak default, yang sering kali bekerja dengan baik dalam praktik
untuk masalah berukuran sedang.


### Inisialisasi Xavier 
:label:`subsec_xavier`

Mari kita lihat distribusi skala
output $o_{i}$ untuk beberapa fully connected layer
*tanpa nonlinearitas*.
Dengan $n_\textrm{in}$ input $x_j$
dan bobot yang terkait $w_{ij}$ untuk lapisan ini,
sebuah output diberikan oleh

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

Bobot $w_{ij}$ semuanya diambil
secara independen dari distribusi yang sama.
Selain itu, mari kita asumsikan bahwa distribusi ini
memiliki rata-rata nol dan variansi $\sigma^2$.
Perhatikan bahwa ini tidak berarti bahwa distribusi tersebut harus Gaussian,
hanya bahwa rata-rata dan variansi harus ada.
Untuk saat ini, mari kita asumsikan bahwa input ke lapisan $x_j$
juga memiliki rata-rata nol dan variansi $\gamma^2$
dan bahwa input-input ini independen dari $w_{ij}$ dan saling independen.
Dalam kasus ini, kita dapat menghitung rata-rata dari $o_i$:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

dan variansi:

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Salah satu cara untuk menjaga variansi tetap konstan
adalah dengan menetapkan $n_\textrm{in} \sigma^2 = 1$.
Sekarang pertimbangkan backpropagation.
Di sana kita menghadapi masalah serupa,
walaupun gradien disebarkan dari lapisan yang lebih dekat ke output.
Menggunakan alasan yang sama seperti pada forward propagation,
kita melihat bahwa variansi gradien bisa meledak
kecuali jika $n_\textrm{out} \sigma^2 = 1$,
di mana $n_\textrm{out}$ adalah jumlah output dari lapisan ini.
Ini membawa kita pada dilema:
kita tidak mungkin memenuhi kedua kondisi secara bersamaan.
Sebagai gantinya, kita mencoba memenuhi:

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ atau dengan kata lain }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

Ini adalah alasan mendasar dari *Xavier initialization*
yang sekarang menjadi standar dan sangat bermanfaat secara praktis,
dinamai sesuai dengan penulis utama dari pembuatnya :cite:`Glorot.Bengio.2010`.
Biasanya, inisialisasi Xavier
mengambil bobot dari distribusi Gaussian
dengan rata-rata nol dan variansi
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$.
Kita juga dapat menyesuaikan ini untuk
memilih variansi saat mengambil bobot
dari distribusi uniform.
Perhatikan bahwa distribusi uniform $U(-a, a)$ memiliki variansi $\frac{a^2}{3}$.
Dengan memasukkan $\frac{a^2}{3}$ ke dalam kondisi kita untuk $\sigma^2$,
kita memperoleh inisialisasi menurut

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Meskipun asumsi untuk non-eksistensi nonlinearitas
dalam alasan matematis di atas
dapat dengan mudah dilanggar dalam neural network,
metode Xavier initialization
ternyata bekerja dengan baik dalam praktik.



### Beyond

Penjelasan di atas baru menggores permukaan
pendekatan modern untuk inisialisasi parameter.
Framework deep learning sering kali mengimplementasikan lebih dari selusin heuristik yang berbeda.
Selain itu, inisialisasi parameter masih menjadi
area penelitian fundamental yang panas dalam deep learning.
Di antaranya ada heuristik khusus untuk
parameter yang terikat (terbagi), super-resolution,
model urutan (sequence models), dan situasi lainnya.
Misalnya,
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` menunjukkan kemungkinan melatih
neural network dengan 10.000 lapisan tanpa trik arsitektural
dengan menggunakan metode inisialisasi yang dirancang dengan hati-hati.

Jika topik ini menarik bagi Anda, kami sarankan
untuk menyelami penawaran modul ini,
membaca makalah yang mengusulkan dan menganalisis setiap heuristik,
dan kemudian menjelajahi publikasi terbaru tentang topik ini.
Mungkin Anda akan menemukan atau bahkan menemukan
ide cerdas dan memberikan kontribusi implementasi ke dalam framework deep learning.


## Ringkasan

Vanishing dan exploding gradient adalah masalah umum dalam jaringan dalam. Perhatian besar diperlukan dalam inisialisasi parameter untuk memastikan bahwa gradien dan parameter tetap terkontrol dengan baik.
Heuristik inisialisasi diperlukan untuk memastikan bahwa gradien awal tidak terlalu besar atau terlalu kecil.
Inisialisasi acak penting untuk memastikan bahwa simetri terpecah sebelum optimasi dimulai.
Xavier initialization menyarankan bahwa, untuk setiap lapisan, variansi output tidak dipengaruhi oleh jumlah input, dan variansi gradien tidak dipengaruhi oleh jumlah output.
Fungsi aktivasi ReLU membantu mengurangi masalah vanishing gradient, yang dapat mempercepat konvergensi.


## Latihan

1. Dapatkah Anda merancang kasus lain di mana neural network mungkin menunjukkan simetri yang perlu dipecahkan, selain simetri permutasi pada lapisan MLP?
2. Bisakah kita menginisialisasi semua parameter bobot dalam linear regression atau softmax regression dengan nilai yang sama?
3. Cari batasan analitik pada nilai eigen dari produk dua matriks. Apa yang dikatakan ini tentang memastikan bahwa gradien memiliki kondisi yang baik?
4. Jika kita tahu bahwa beberapa istilah mengalami divergensi, dapatkah kita memperbaikinya setelah kejadian tersebut? Lihat makalah tentang layerwise adaptive rate scaling sebagai inspirasi :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17986)
:end_tab:
