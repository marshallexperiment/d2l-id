# Optimisasi dan Deep Learning
:label:`sec_optimization-intro`

Di bagian ini, kita akan membahas hubungan antara optimisasi dan deep learning serta tantangan dalam menggunakan optimisasi di deep learning. Untuk masalah deep learning, biasanya kita akan mendefinisikan *fungsi loss* terlebih dahulu. Setelah kita memiliki fungsi loss, kita dapat menggunakan algoritma optimisasi untuk mencoba meminimalkan loss tersebut. Dalam optimisasi, fungsi loss sering disebut sebagai *fungsi objektif* dari masalah optimisasi. Berdasarkan tradisi dan konvensi, sebagian besar algoritma optimisasi berfokus pada *minimisasi*. Jika kita perlu memaksimalkan suatu objektif, ada solusi sederhana: cukup balik tanda pada objektif tersebut.

## Tujuan Optimisasi

Meskipun optimisasi menyediakan cara untuk meminimalkan fungsi loss pada deep learning, pada dasarnya tujuan dari optimisasi dan deep learning berbeda secara fundamental. Yang pertama terutama berfokus pada meminimalkan suatu objektif, sedangkan yang terakhir berfokus pada menemukan model yang sesuai, mengingat data yang terbatas. Dalam :numref:`sec_generalization_basics`, kita telah membahas perbedaan antara kedua tujuan ini secara rinci.

Sebagai contoh, error pelatihan dan error generalisasi biasanya berbeda: karena fungsi objektif dari algoritma optimisasi biasanya adalah fungsi loss berdasarkan dataset pelatihan, tujuan optimisasi adalah mengurangi error pelatihan. Namun, tujuan deep learning (atau lebih luas lagi, inferensi statistik) adalah untuk mengurangi error generalisasi. Untuk mencapai yang terakhir, kita perlu memperhatikan overfitting selain menggunakan algoritma optimisasi untuk mengurangi error pelatihan.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Untuk mengilustrasikan perbedaan tujuan yang telah disebutkan, mari kita pertimbangkan risiko empiris dan risiko. Seperti yang dijelaskan dalam :numref:`subsec_empirical-risk-and-risk`, risiko empiris adalah rata-rata loss pada dataset pelatihan, sedangkan risiko adalah loss yang diharapkan pada seluruh populasi data. Di bawah ini kita mendefinisikan dua fungsi: fungsi risiko `f` dan fungsi risiko empiris `g`. Misalkan kita hanya memiliki sejumlah data pelatihan yang terbatas. Akibatnya, `g` di sini menjadi kurang mulus dibandingkan dengan `f`.


```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Grafik di bawah ini mengilustrasikan bahwa minimum dari risiko empiris pada dataset pelatihan mungkin berada di lokasi yang berbeda dari minimum risiko (error generalisasi).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Tantangan Optimisasi dalam Deep Learning

Di bab ini, kita akan fokus secara spesifik pada kinerja algoritma optimisasi dalam meminimalkan fungsi objektif, bukan error generalisasi model. Dalam :numref:`sec_linear_regression`, kita membedakan antara solusi analitik dan solusi numerik dalam masalah optimisasi. Dalam deep learning, sebagian besar fungsi objektif adalah kompleks dan tidak memiliki solusi analitik. Sebaliknya, kita harus menggunakan algoritma optimisasi numerik. Algoritma optimisasi dalam bab ini semuanya termasuk dalam kategori ini.

Ada banyak tantangan dalam optimisasi deep learning. Beberapa tantangan yang paling sulit adalah minimum lokal, saddle points, dan vanishing gradients. Mari kita lihat satu per satu.


### Minimum Lokal

Untuk setiap fungsi objektif $f(x)$, jika nilai $f(x)$ pada titik $x$ lebih kecil dari nilai $f(x)$ di titik lain di sekitar $x$, maka $f(x)$ bisa menjadi minimum lokal. Jika nilai $f(x)$ pada $x$ adalah nilai minimum dari fungsi objektif di seluruh domain, maka $f(x)$ adalah minimum global.

Sebagai contoh, dengan fungsi

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ untuk } -1.0 \leq x \leq 2.0,$$

kita dapat memperkirakan minimum lokal dan minimum global dari fungsi ini.


```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Fungsi objektif dari model deep learning biasanya memiliki banyak optimum lokal. Ketika solusi numerik dari suatu masalah optimisasi berada di dekat optimum lokal, solusi numerik yang diperoleh pada iterasi akhir mungkin hanya meminimalkan fungsi objektif *secara lokal*, bukan *secara global*, karena gradien dari solusi fungsi objektif mendekati atau menjadi nol. Hanya beberapa derajat noise yang mungkin bisa menggeser parameter keluar dari minimum lokal. Faktanya, ini adalah salah satu sifat menguntungkan dari stochastic gradient descent minibatch, di mana variasi alami dari gradien pada minibatch mampu melepaskan parameter dari minimum lokal.


### Saddle Points

Selain minimum lokal, saddle points adalah alasan lain mengapa gradien bisa menghilang. *Saddle point* adalah setiap lokasi di mana semua gradien dari suatu fungsi menghilang, tetapi bukan merupakan minimum global maupun minimum lokal. Pertimbangkan fungsi $f(x) = x^3$. Turunan pertama dan kedua dari fungsi ini hilang untuk $x=0$. Proses optimisasi mungkin terhenti pada titik ini, meskipun titik ini bukan minimum.


```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Saddle points dalam dimensi yang lebih tinggi bahkan lebih sulit diatasi, seperti yang ditunjukkan oleh contoh di bawah ini. Pertimbangkan fungsi $f(x, y) = x^2 - y^2$. Fungsi ini memiliki saddle point pada $(0, 0)$. Titik ini merupakan maksimum terhadap $y$ dan minimum terhadap $x$. Selain itu, bentuknya *terlihat* seperti pelana, yang menjadi asal dari nama properti matematis ini.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Kita asumsikan bahwa input dari suatu fungsi adalah sebuah vektor berdimensi $k$ dan outputnya adalah sebuah skalar, sehingga matriks Hessian-nya akan memiliki $k$ nilai eigen. Solusi dari fungsi tersebut bisa berupa minimum lokal, maksimum lokal, atau saddle point pada posisi di mana gradien fungsi tersebut bernilai nol:

* Ketika nilai-nilai eigen dari matriks Hessian fungsi pada posisi gradien nol semuanya positif, maka kita memiliki minimum lokal untuk fungsi tersebut.
* Ketika nilai-nilai eigen dari matriks Hessian fungsi pada posisi gradien nol semuanya negatif, maka kita memiliki maksimum lokal untuk fungsi tersebut.
* Ketika nilai-nilai eigen dari matriks Hessian fungsi pada posisi gradien nol memiliki nilai negatif dan positif, maka kita memiliki saddle point untuk fungsi tersebut.

Untuk masalah berdimensi tinggi, kemungkinan bahwa setidaknya *beberapa* nilai eigen adalah negatif cukup tinggi. Ini membuat saddle points lebih mungkin terjadi dibandingkan minimum lokal. Kita akan membahas beberapa pengecualian terhadap situasi ini di bagian berikutnya saat memperkenalkan konsep convexity. Singkatnya, fungsi konveks adalah fungsi di mana nilai-nilai eigen dari Hessian-nya tidak pernah negatif. Sayangnya, sebagian besar masalah deep learning tidak termasuk dalam kategori ini. Meskipun demikian, konsep ini adalah alat yang berguna untuk mempelajari algoritma optimisasi.

### Gradien yang Menghilang

Masalah yang mungkin paling sulit diatasi adalah gradien yang menghilang. Ingat kembali fungsi aktivasi yang sering kita gunakan dan turunannya dalam :numref:`subsec_activation-functions`. Sebagai contoh, misalkan kita ingin meminimalkan fungsi $f(x) = \tanh(x)$ dan kebetulan memulai pada $x = 4$. Seperti yang dapat kita lihat, gradien dari $f$ mendekati nol. Lebih spesifik, $f'(x) = 1 - \tanh^2(x)$ dan dengan demikian $f'(4) = 0.0013$. Akibatnya, proses optimisasi akan terhenti untuk waktu yang lama sebelum kita dapat membuat kemajuan. Ini ternyata menjadi salah satu alasan mengapa pelatihan model deep learning cukup sulit sebelum diperkenalkannya fungsi aktivasi ReLU.


```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Seperti yang kita lihat, optimisasi untuk deep learning penuh dengan tantangan. Untungnya, ada berbagai algoritma yang kuat yang bekerja dengan baik dan mudah digunakan bahkan bagi pemula. Lebih lanjut, tidak perlu menemukan *solusi terbaik* secara mutlak. Optimum lokal atau bahkan solusi perkiraan dari optimum tersebut masih sangat berguna.

## Ringkasan

* Meminimalkan error pelatihan *tidak* menjamin bahwa kita menemukan set parameter terbaik untuk meminimalkan error generalisasi.
* Masalah optimisasi dapat memiliki banyak minimum lokal.
* Masalah ini bahkan mungkin memiliki lebih banyak saddle points, karena umumnya masalah ini tidak konveks.
* Gradien yang menghilang dapat menyebabkan optimisasi terhenti. Sering kali, mereparametrisasi masalah dapat membantu. Inisialisasi parameter yang baik juga dapat bermanfaat.


## Latihan

1. Pertimbangkan sebuah MLP sederhana dengan satu hidden layer, misalnya berdimensi $d$ pada hidden layer dan satu output. Tunjukkan bahwa untuk setiap minimum lokal terdapat setidaknya $d!$ solusi ekuivalen yang berperilaku identik.
2. Misalkan kita memiliki matriks acak simetris $\mathbf{M}$ di mana elemen-elemen
   $M_{ij} = M_{ji}$ masing-masing ditarik dari beberapa distribusi probabilitas
   $p_{ij}$. Selain itu, anggap bahwa $p_{ij}(x) = p_{ij}(-x)$, yaitu distribusinya simetris (lihat, misalnya, :citet:`Wigner.1958` untuk detailnya).
    1. Buktikan bahwa distribusi nilai eigen juga simetris. Artinya, untuk setiap eigenvektor $\mathbf{v}$, probabilitas bahwa nilai eigen yang terkait $\lambda$ memenuhi $P(\lambda > 0) = P(\lambda < 0)$.
    2. Mengapa hal di atas *tidak* menyiratkan $P(\lambda > 0) = 0.5$?
3. Tantangan lain apa yang menurut Anda terlibat dalam optimisasi deep learning?
4. Misalkan Anda ingin menyeimbangkan sebuah bola (nyata) pada sebuah pelana (nyata).
    1. Mengapa ini sulit?
    2. Dapatkah Anda memanfaatkan efek ini juga untuk algoritma optimisasi?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/489)
:end_tab:
