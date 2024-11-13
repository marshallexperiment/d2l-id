# Kekonveksan
:label:`sec_convexity`

Kekonveksan memainkan peran penting dalam desain algoritma optimasi.
Hal ini sebagian besar karena lebih mudah untuk menganalisis dan menguji algoritma dalam konteks tersebut.
Dengan kata lain,
jika algoritma berkinerja buruk bahkan dalam kondisi konveks,
umumnya kita tidak dapat berharap untuk melihat hasil yang baik pada kondisi lainnya.
Selain itu, meskipun masalah optimasi dalam deep learning umumnya nonkonveks, mereka sering menunjukkan beberapa sifat dari masalah konveks di dekat minimum lokal. Hal ini dapat mengarah pada varian optimasi baru yang menarik seperti yang diusulkan oleh :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.


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

## Definisi

Sebelum membahas analisis konveksitas, kita perlu mendefinisikan *himpunan konveks* dan *fungsi konveks*. Kedua konsep ini akan menghasilkan alat-alat matematika yang sering diterapkan pada machine learning.

### Himpunan Konveks

Himpunan adalah dasar dari konveksitas. Secara sederhana, suatu himpunan $\mathcal{X}$ dalam ruang vektor disebut *konveks* jika untuk setiap $a, b \in \mathcal{X}$, segmen garis yang menghubungkan $a$ dan $b$ juga termasuk dalam $\mathcal{X}$. Dalam bentuk matematis, hal ini berarti untuk semua $\lambda \in [0, 1]$ kita memiliki

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \textrm{ jika } a, b \in \mathcal{X}.$$

Ini terdengar agak abstrak. Pertimbangkan :numref:`fig_pacman`. Himpunan pertama tidak konveks karena terdapat segmen garis yang tidak termasuk di dalamnya. Dua himpunan lainnya tidak mengalami masalah tersebut.

![Himpunan pertama tidak konveks dan dua lainnya adalah konveks.](../img/pacman.svg)
:label:`fig_pacman`

Definisi saja tidak terlalu berguna kecuali kita bisa melakukan sesuatu dengan mereka.
Dalam hal ini, kita dapat melihat pada irisan-irisan seperti yang ditunjukkan pada :numref:`fig_convex_intersect`.
Misalkan $\mathcal{X}$ dan $\mathcal{Y}$ adalah himpunan konveks. Maka $\mathcal{X} \cap \mathcal{Y}$ juga merupakan himpunan konveks. Untuk melihat ini, anggap bahwa $a, b \in \mathcal{X} \cap \mathcal{Y}$. Karena $\mathcal{X}$ dan $\mathcal{Y}$ konveks, segmen garis yang menghubungkan $a$ dan $b$ termasuk dalam kedua $\mathcal{X}$ dan $\mathcal{Y}$. Dengan demikian, segmen tersebut juga harus termasuk dalam $\mathcal{X} \cap \mathcal{Y}$, sehingga membuktikan teorema kita.

![Irisan antara dua himpunan konveks adalah konveks.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Kita dapat memperkuat hasil ini dengan sedikit usaha: diberikan himpunan-himpunan konveks $\mathcal{X}_i$, irisan mereka $\cap_{i} \mathcal{X}_i$ juga konveks.
Untuk melihat bahwa kebalikannya tidak benar, pertimbangkan dua himpunan yang tidak bersinggungan $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Sekarang pilih $a \in \mathcal{X}$ dan $b \in \mathcal{Y}$. Segmen garis pada :numref:`fig_nonconvex` yang menghubungkan $a$ dan $b$ harus mengandung bagian yang tidak termasuk dalam $\mathcal{X}$ maupun $\mathcal{Y}$, karena kita mengasumsikan bahwa $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Oleh karena itu, segmen garis tersebut juga tidak termasuk dalam $\mathcal{X} \cup \mathcal{Y}$, sehingga membuktikan bahwa pada umumnya gabungan dari himpunan konveks tidak harus konveks.

![Gabungan dua himpunan konveks tidak selalu konveks.](../img/nonconvex.svg)
:label:`fig_nonconvex`

Biasanya masalah dalam deep learning didefinisikan pada himpunan konveks. Misalnya, $\mathbb{R}^d$,
himpunan vektor berdimensi $d$ dengan bilangan real,
adalah himpunan konveks (karena garis antara dua titik di $\mathbb{R}^d$ tetap berada di $\mathbb{R}^d$). Dalam beberapa kasus, kita bekerja dengan variabel dengan panjang terbatas, seperti bola dengan radius $r$ yang didefinisikan oleh $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ dan } \|\mathbf{x}\| \leq r\}$.

### Fungsi Konveks

Sekarang setelah kita memiliki himpunan konveks, kita dapat memperkenalkan *fungsi konveks* $f$.
Diberikan himpunan konveks $\mathcal{X}$, fungsi $f: \mathcal{X} \to \mathbb{R}$ disebut *konveks* jika untuk semua $x, x' \in \mathcal{X}$ dan untuk semua $\lambda \in [0, 1]$ kita memiliki

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Untuk mengilustrasikan ini, mari kita plot beberapa fungsi dan periksa fungsi mana yang memenuhi syarat tersebut.
Di bawah ini kita akan mendefinisikan beberapa fungsi, baik yang konveks maupun yang tidak konveks.



```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Seperti yang diharapkan, fungsi kosinus adalah *nonkonveks*, sedangkan parabola dan fungsi eksponensial adalah konveks. Perhatikan bahwa syarat bahwa $\mathcal{X}$ adalah himpunan konveks diperlukan agar kondisi ini masuk akal. Jika tidak, hasil dari $f(\lambda x + (1-\lambda) x')$ mungkin tidak didefinisikan dengan baik.


### Ketaksamaan Jensen

Diberikan sebuah fungsi konveks $f$, salah satu alat matematika yang paling berguna adalah *Ketaksamaan Jensen*. Ini merupakan generalisasi dari definisi konveksitas:

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \textrm{ dan }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

di mana $\alpha_i$ adalah bilangan real non-negatif sehingga $\sum_i \alpha_i = 1$ dan $X$ adalah variabel acak. Dengan kata lain, ekspektasi dari fungsi konveks tidak lebih kecil daripada fungsi konveks dari ekspektasi, di mana yang terakhir biasanya merupakan ekspresi yang lebih sederhana. Untuk membuktikan ketaksamaan pertama, kita berulang kali menerapkan definisi konveksitas pada satu suku dalam penjumlahan setiap saat.

Salah satu penerapan umum dari Ketaksamaan Jensen adalah untuk membatasi ekspresi yang lebih rumit dengan yang lebih sederhana. Sebagai contoh, penerapannya bisa dilakukan sehubungan dengan log-likelihood dari variabel acak yang hanya diamati sebagian. Artinya, kita menggunakan

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

karena $\int P(Y) P(X \mid Y) dY = P(X)$. Ini bisa digunakan dalam metode variasional. Di sini $Y$ biasanya merupakan variabel acak yang tidak teramati, $P(Y)$ adalah perkiraan terbaik tentang bagaimana distribusinya, dan $P(X)$ adalah distribusi dengan $Y$ yang telah diintegrasikan. Misalnya, dalam klastering, $Y$ bisa menjadi label klaster dan $P(X \mid Y)$ adalah model generatif saat menerapkan label klaster.

## Properti

Fungsi konveks memiliki banyak properti yang berguna. Berikut ini kami jelaskan beberapa yang sering digunakan.


### Minimum Lokal adalah Minimum Global

Pertama dan terutama, minimum lokal dari fungsi konveks juga merupakan minimum global. Kita bisa membuktikannya dengan kontradiksi sebagai berikut.

Pertimbangkan fungsi konveks $f$ yang didefinisikan pada himpunan konveks $\mathcal{X}$. Misalkan $x^{\ast} \in \mathcal{X}$ adalah minimum lokal: terdapat nilai positif kecil $p$ sehingga untuk $x \in \mathcal{X}$ yang memenuhi $0 < |x - x^{\ast}| \leq p$ kita memiliki $f(x^{\ast}) < f(x)$.

Asumsikan bahwa minimum lokal $x^{\ast}$ bukanlah minimum global dari $f$: terdapat $x' \in \mathcal{X}$ di mana $f(x') < f(x^{\ast})$. Terdapat juga $\lambda \in [0, 1)$ seperti $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ sehingga $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$.

Namun, menurut definisi fungsi konveks, kita memiliki

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

yang bertentangan dengan pernyataan kita bahwa $x^{\ast}$ adalah minimum lokal. Oleh karena itu, tidak ada $x' \in \mathcal{X}$ untuk mana $f(x') < f(x^{\ast})$. Minimum lokal $x^{\ast}$ juga adalah minimum global.

Sebagai contoh, fungsi konveks $f(x) = (x-1)^2$ memiliki minimum lokal di $x=1$, yang juga merupakan minimum global.



```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Fakta bahwa minimum lokal untuk fungsi konveks juga merupakan minimum global sangatlah menguntungkan. Ini berarti bahwa jika kita meminimalkan fungsi, kita tidak akan "terjebak". Namun, perlu dicatat bahwa ini tidak berarti bahwa tidak ada lebih dari satu minimum global atau bahkan bahwa minimum tersebut mungkin tidak ada. Sebagai contoh, fungsi $f(x) = \mathrm{max}(|x|-1, 0)$ mencapai nilai minimumnya di interval $[-1, 1]$. Sebaliknya, fungsi $f(x) = \exp(x)$ tidak mencapai nilai minimum di $\mathbb{R}$: untuk $x \to -\infty$, nilai fungsi tersebut mendekati $0$, tetapi tidak ada $x$ untuk mana $f(x) = 0$.

### Himpunan Bawah dari Fungsi Konveks adalah Konveks

Kita bisa dengan mudah mendefinisikan himpunan konveks melalui *himpunan bawah* dari fungsi konveks. Secara konkret, diberikan sebuah fungsi konveks $f$ yang didefinisikan pada himpunan konveks $\mathcal{X}$, setiap himpunan bawah

$$\mathcal{S}_b \stackrel{\textrm{def}}{=} \{x | x \in \mathcal{X} \textrm{ dan } f(x) \leq b\}$$

adalah konveks.

Mari kita buktikan ini dengan cepat. Ingatlah bahwa untuk setiap $x, x' \in \mathcal{S}_b$, kita perlu menunjukkan bahwa $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ selama $\lambda \in [0, 1]$. Karena $f(x) \leq b$ dan $f(x') \leq b$, menurut definisi konveksitas kita memiliki

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### Konveksitas dan Turunan Kedua

Jika turunan kedua dari sebuah fungsi $f: \mathbb{R}^n \rightarrow \mathbb{R}$ ada, sangat mudah untuk memeriksa apakah $f$ konveks. Yang perlu kita lakukan hanyalah memeriksa apakah Hessian dari $f$ adalah semi-definit positif: $\nabla^2f \succeq 0$, yaitu dengan menandai matriks Hessian $\nabla^2f$ sebagai $\mathbf{H}$, $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ untuk semua $\mathbf{x} \in \mathbb{R}^n$. Sebagai contoh, fungsi $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ adalah konveks karena $\nabla^2 f = \mathbf{1}$, yaitu Hessian-nya adalah matriks identitas.

Secara formal, sebuah fungsi satu-dimensi yang dapat diturunkan dua kali $f: \mathbb{R} \rightarrow \mathbb{R}$ adalah konveks jika dan hanya jika turunannya yang kedua $f'' \geq 0$. Untuk setiap fungsi multidimensional yang dapat diturunkan dua kali $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$, fungsi tersebut adalah konveks jika dan hanya jika Hessian-nya $\nabla^2f \succeq 0$.

Pertama, kita perlu membuktikan kasus satu-dimensi. Untuk melihat bahwa konveksitas dari $f$ menyiratkan $f'' \geq 0$, kita menggunakan fakta bahwa

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

Karena turunan kedua diberikan oleh limit dari perbedaan hingga, maka

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

Untuk melihat bahwa $f'' \geq 0$ menyiratkan $f$ adalah konveks, kita menggunakan fakta bahwa $f'' \geq 0$ menyiratkan bahwa $f'$ adalah fungsi yang tidak berkurang secara monoton. Misalkan $a < x < b$ adalah tiga titik di $\mathbb{R}$, di mana $x = (1-\lambda)a + \lambda b$ dan $\lambda \in (0, 1)$. Menurut teorema nilai rata-rata, terdapat $\alpha \in [a, x]$ dan $\beta \in [x, b]$ sehingga

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \textrm{ dan } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

Karena monotonisitas $f'(\beta) \geq f'(\alpha)$, maka

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Karena $x = (1-\lambda)a + \lambda b$, kita memiliki

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

sehingga membuktikan konveksitas.

Kedua, kita memerlukan sebuah lema sebelum membuktikan kasus multidimensional: $f: \mathbb{R}^n \rightarrow \mathbb{R}$ adalah konveks jika dan hanya jika untuk semua $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$

$$g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \textrm{ di mana } z \in [0,1]$$ 

adalah konveks.

Untuk membuktikan bahwa konveksitas dari $f$ menyiratkan bahwa $g$ adalah konveks, kita dapat menunjukkan bahwa untuk semua $a, b, \lambda \in [0, 1]$ (jadi $0 \leq \lambda a + (1-\lambda) b \leq 1$)

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

Untuk membuktikan kebalikannya, kita dapat menunjukkan bahwa untuk semua $\lambda \in [0, 1]$

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\end{aligned}$$

Akhirnya, menggunakan lema di atas dan hasil dari kasus satu-dimensi, kasus multidimensional dapat dibuktikan sebagai berikut. Fungsi multidimensional $f: \mathbb{R}^n \rightarrow \mathbb{R}$ adalah konveks jika dan hanya jika untuk semua $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, di mana $z \in [0,1]$, adalah konveks. Menurut kasus satu-dimensi, ini berlaku jika dan hanya jika $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2f$) untuk semua $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, yang setara dengan $\mathbf{H} \succeq 0$ sesuai dengan definisi matriks semi-definit positif.


## Batasan

Salah satu sifat bagus dari optimasi konveks adalah memungkinkan kita untuk menangani batasan secara efisien. Artinya, ini memungkinkan kita untuk menyelesaikan masalah *optimasi terbatas* dalam bentuk:

$$\begin{aligned} \mathop{\textrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \textrm{ subject to } & c_i(\mathbf{x}) \leq 0 \textrm{ untuk semua } i \in \{1, \ldots, n\},
\end{aligned}$$

di mana $f$ adalah tujuan dan fungsi $c_i$ adalah fungsi batasan. Untuk memahami ini, mari pertimbangkan kasus di mana $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. Dalam kasus ini, parameter $\mathbf{x}$ dibatasi oleh bola satuan. Jika batasan kedua adalah $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, maka ini berarti semua $\mathbf{x}$ berada di ruang setengah. Memenuhi kedua batasan secara simultan berarti memilih bagian dari sebuah bola.

### Lagrangian

Secara umum, menyelesaikan masalah optimasi terbatas adalah hal yang sulit. Salah satu cara mengatasinya berasal dari fisika dengan intuisi yang cukup sederhana. Bayangkan sebuah bola di dalam kotak. Bola akan menggelinding ke tempat yang paling rendah, dan gaya gravitasi akan diimbangi dengan gaya yang dikenakan oleh dinding kotak pada bola. Singkatnya, gradien dari fungsi tujuan (yaitu gravitasi) akan diimbangi oleh gradien dari fungsi batasan (bola perlu tetap berada dalam kotak karena dindingnya "mendorong kembali"). Perhatikan bahwa beberapa batasan mungkin tidak aktif: dinding yang tidak disentuh oleh bola tidak akan mampu memberikan gaya pada bola.

Dengan melewatkan penurunan *Lagrangian* $L$, penalaran di atas dapat diekspresikan melalui masalah optimasi titik sadel berikut:

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \textrm{ di mana } \alpha_i \geq 0.$$

Di sini, variabel $\alpha_i$ ($i=1,\ldots,n$) adalah yang disebut *pengganda Lagrange* yang memastikan batasan diterapkan dengan benar. Mereka dipilih cukup besar untuk memastikan bahwa $c_i(\mathbf{x}) \leq 0$ untuk semua $i$. Sebagai contoh, untuk setiap $\mathbf{x}$ di mana $c_i(\mathbf{x}) < 0$ secara alami, kita akan memilih $\alpha_i = 0$. Selain itu, ini adalah masalah optimasi titik sadel di mana kita ingin *memaksimalkan* $L$ dengan menghormati semua $\alpha_i$ dan secara bersamaan *meminimalkan* dengan menghormati $\mathbf{x}$. Ada banyak literatur yang menjelaskan cara mencapai fungsi $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$. Untuk tujuan kita, cukup diketahui bahwa titik sadel dari $L$ adalah di mana masalah optimasi terbatas asli diselesaikan secara optimal.

### Penalti

Salah satu cara untuk memenuhi masalah optimasi terbatas setidaknya *secara mendekati* adalah dengan menyesuaikan Lagrangian $L$. Alih-alih memenuhi $c_i(\mathbf{x}) \leq 0$, kita cukup menambahkan $\alpha_i c_i(\mathbf{x})$ ke fungsi tujuan $f(x)$. Ini memastikan bahwa batasan tidak akan dilanggar terlalu parah.

Faktanya, kita telah menggunakan trik ini sepanjang waktu. Pertimbangkan penurunan berat dalam :numref:`sec_weight_decay`. Di dalamnya, kita menambahkan $\frac{\lambda}{2} \|\mathbf{w}\|^2$ ke fungsi tujuan untuk memastikan bahwa $\mathbf{w}$ tidak tumbuh terlalu besar. Dari sudut pandang optimasi terbatas, kita dapat melihat bahwa ini akan memastikan bahwa $\|\mathbf{w}\|^2 - r^2 \leq 0$ untuk beberapa radius $r$. Menyesuaikan nilai $\lambda$ memungkinkan kita untuk mengubah ukuran $\mathbf{w}$.

Secara umum, menambahkan penalti adalah cara yang baik untuk memastikan pemenuhan batasan secara mendekati. Dalam praktiknya, ini ternyata jauh lebih kuat daripada pemenuhan yang ketat. Selain itu, untuk masalah nonkonveks banyak sifat yang membuat pendekatan ketat sangat menarik dalam kasus konveks (misalnya, optimalitas) tidak lagi berlaku.

### Proyeksi

Strategi alternatif untuk memenuhi batasan adalah proyeksi. Sekali lagi, kita pernah menemui ini sebelumnya, misalnya ketika menangani pemotongan gradien dalam :numref:`sec_rnn-scratch`. Di sana, kita memastikan bahwa gradien memiliki panjang yang dibatasi oleh $\theta$ melalui

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

Ternyata ini adalah *proyeksi* dari $\mathbf{g}$ ke bola dengan radius $\theta$. Secara umum, proyeksi pada himpunan konveks $\mathcal{X}$ didefinisikan sebagai

$$\textrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

yang merupakan titik terdekat di $\mathcal{X}$ ke $\mathbf{x}$.

![Proyeksi Konveks.](../img/projections.svg)
:label:`fig_projections`

Definisi matematis dari proyeksi mungkin terdengar sedikit abstrak. :numref:`fig_projections` menjelaskannya dengan lebih jelas. Di dalamnya kita memiliki dua himpunan konveks, sebuah lingkaran dan berlian.
Titik-titik di dalam kedua himpunan (kuning) tetap tidak berubah selama proyeksi.
Titik-titik di luar kedua himpunan (hitam) diproyeksikan ke
titik-titik di dalam himpunan (merah) yang paling dekat dengan titik asli (hitam).
Sementara untuk bola $\ell_2$ arah tidak berubah, hal ini tidak selalu berlaku secara umum, seperti yang dapat dilihat dalam kasus berlian.

Salah satu penggunaan proyeksi konveks adalah untuk menghitung vektor bobot yang jarang. Dalam hal ini kita memproyeksikan vektor bobot ke bola $\ell_1$, yang merupakan versi umum dari kasus berlian dalam :numref:`fig_projections`.

## Ringkasan

Dalam konteks deep learning, tujuan utama dari fungsi konveks adalah untuk memotivasi algoritma optimasi dan membantu kita memahami detailnya. Selanjutnya, kita akan melihat bagaimana gradient descent dan stochastic gradient descent dapat diturunkan sesuai dengan ini.

* Perpotongan himpunan konveks adalah konveks. Gabungan tidak.
* Harapan dari sebuah fungsi konveks tidak kurang dari fungsi konveks dari harapan (Ketidaksamaan Jensen).
* Sebuah fungsi yang dapat diturunkan dua kali adalah konveks jika dan hanya jika Hessiannya (matriks turunan kedua) adalah semi-definit positif.
* Batasan konveks dapat ditambahkan melalui Lagrangian. Dalam praktiknya, kita dapat menambahkannya dengan penalti ke fungsi tujuan.
* Proyeksi memetakan titik-titik dalam himpunan konveks yang paling dekat ke titik-titik asli.

## Latihan

1. Misalkan kita ingin memverifikasi konveksitas suatu himpunan dengan menggambar semua garis antara titik-titik dalam himpunan dan memeriksa apakah garis-garis tersebut termasuk.
    1. Buktikan bahwa cukup untuk memeriksa hanya titik-titik pada batas.
    1. Buktikan bahwa cukup untuk memeriksa hanya simpul-simpul dari himpunan.
1. Nyatakan $\mathcal{B}_p[r] \stackrel{\textrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ dan } \|\mathbf{x}\|_p \leq r\}$ sebagai bola dengan radius $r$ menggunakan norma $p$. Buktikan bahwa $\mathcal{B}_p[r]$ adalah konveks untuk semua $p \geq 1$.
1. Diberikan fungsi konveks $f$ dan $g$, tunjukkan bahwa $\mathrm{max}(f, g)$ juga konveks. Buktikan bahwa $\mathrm{min}(f, g)$ tidak konveks.
1. Buktikan bahwa normalisasi dari fungsi softmax adalah konveks. Lebih khusus lagi, buktikan konveksitas
    $f(x) = \log \sum_i \exp(x_i)$.
1. Buktikan bahwa subruang linier, yaitu $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$, adalah himpunan konveks.
1. Buktikan bahwa dalam kasus subruang linier dengan $\mathbf{b} = \mathbf{0}$, proyeksi $\textrm{Proj}_\mathcal{X}$ dapat ditulis sebagai $\mathbf{M} \mathbf{x}$ untuk beberapa matriks $\mathbf{M}$.
1. Tunjukkan bahwa untuk fungsi konveks yang dapat diturunkan dua kali $f$, kita dapat menulis $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ untuk beberapa $\xi \in [0, \epsilon]$.
1. Diberikan himpunan konveks $\mathcal{X}$ dan dua vektor $\mathbf{x}$ dan $\mathbf{y}$, buktikan bahwa proyeksi tidak pernah menambah jarak, yaitu $\|\mathbf{x} - \mathbf{y}\| \geq \|\textrm{Proj}_\mathcal{X}(\mathbf{x}) - \textrm{Proj}_\mathcal{X}(\mathbf{y})\|$.

[Diskusi](https://discuss.d2l.ai/t/350)

