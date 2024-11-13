# Stochastic Gradient Descent
:label:`sec_sgd`

Di bab-bab sebelumnya, kita terus menggunakan penurunan gradien stokastik dalam prosedur pelatihan kita, namun, tanpa menjelaskan mengapa itu bekerja.
Untuk memberikan penjelasan lebih lanjut,
kita telah menjelaskan prinsip dasar dari penurunan gradien
di :numref:`sec_gd`.
Pada bagian ini, kita akan melanjutkan dengan membahas
*penurunan gradien stokastik* secara lebih rinci.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```


## Pembaruan Stochastic Gradient

Dalam deep learning, fungsi objektif biasanya merupakan rata-rata dari fungsi loss untuk setiap contoh dalam dataset pelatihan.
Diberikan dataset pelatihan dengan $n$ contoh,
kita mengasumsikan bahwa $f_i(\mathbf{x})$ adalah fungsi loss
terhadap contoh pelatihan dengan indeks $i$,
di mana $\mathbf{x}$ adalah vektor parameter.
Kemudian kita mendapatkan fungsi objektif

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

Gradien dari fungsi objektif pada $\mathbf{x}$ dihitung sebagai

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Jika menggunakan gradient descent, biaya komputasi untuk setiap iterasi variabel independen adalah $\mathcal{O}(n)$, yang bertambah secara linear dengan $n$. Oleh karena itu, ketika dataset pelatihan lebih besar, biaya gradient descent untuk setiap iterasi akan semakin tinggi.

Stochastic gradient descent (SGD) mengurangi biaya komputasi pada setiap iterasi. Pada setiap iterasi stochastic gradient descent, kita secara acak mengambil satu indeks $i\in\{1,\ldots, n\}$ dari contoh data, lalu menghitung gradien $\nabla f_i(\mathbf{x})$ untuk memperbarui $\mathbf{x}$:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

di mana $\eta$ adalah learning rate. Kita dapat melihat bahwa biaya komputasi untuk setiap iterasi turun dari $\mathcal{O}(n)$ pada gradient descent menjadi konstan $\mathcal{O}(1)$. Selain itu, kami ingin menekankan bahwa gradien stokastik $\nabla f_i(\mathbf{x})$ adalah estimasi tak bias dari gradien penuh $\nabla f(\mathbf{x})$ karena

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Ini berarti bahwa, secara rata-rata, gradien stokastik adalah estimasi yang baik dari gradien.

Sekarang, kita akan membandingkannya dengan gradient descent dengan menambahkan noise acak dengan rata-rata 0 dan varians 1 pada gradien untuk mensimulasikan stochastic gradient descent.


```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Seperti yang bisa kita lihat, lintasan variabel dalam stochastic gradient descent jauh lebih berisik dibandingkan yang kita amati dalam gradient descent di :numref:`sec_gd`. Hal ini disebabkan oleh sifat stokastik dari gradien. Artinya, bahkan ketika kita sudah dekat dengan titik minimum, kita masih mengalami ketidakpastian yang disebabkan oleh gradien instan melalui $\eta \nabla f_i(\mathbf{x})$. Bahkan setelah 50 langkah, kualitasnya masih belum cukup baik. Lebih buruk lagi, kualitasnya tidak akan membaik setelah langkah tambahan (kami mendorong Anda untuk bereksperimen dengan jumlah langkah yang lebih banyak untuk mengonfirmasi ini). Hal ini hanya menyisakan satu alternatif: mengubah learning rate $\eta$. Namun, jika kita memilihnya terlalu kecil, kita tidak akan membuat kemajuan yang berarti pada awalnya. Di sisi lain, jika kita memilihnya terlalu besar, kita tidak akan mendapatkan solusi yang baik, seperti yang terlihat di atas. Satu-satunya cara untuk menyelesaikan tujuan yang saling bertentangan ini adalah dengan mengurangi learning rate *secara dinamis* seiring kemajuan optimasi.

Inilah alasan mengapa kita menambahkan fungsi learning rate `lr` ke dalam fungsi langkah `sgd`. Pada contoh di atas, setiap fungsi untuk penjadwalan learning rate tidak aktif karena kita menetapkan fungsi `lr` tersebut sebagai konstanta.

## Learning Rate Dinamis

Mengganti $\eta$ dengan learning rate yang bergantung pada waktu $\eta(t)$ menambah kompleksitas dalam mengendalikan konvergensi algoritma optimasi. Secara khusus, kita perlu mencari tahu seberapa cepat $\eta$ harus berkurang. Jika terlalu cepat, kita akan berhenti mengoptimasi sebelum waktunya. Jika terlalu lambat berkurangnya, kita membuang terlalu banyak waktu untuk optimasi. Berikut adalah beberapa strategi dasar yang digunakan dalam penyesuaian $\eta$ seiring waktu (kita akan membahas strategi yang lebih canggih nanti):

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ jika } t_i \leq t \leq t_{i+1}  && \textrm{konstanta potongan (piecewise constant)} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{penurunan eksponensial (exponential decay)} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{penurunan polinomial (polynomial decay)}
\end{aligned}
$$

Dalam skenario pertama yang *konstanta potongan*, kita mengurangi learning rate, misalnya, setiap kali kemajuan optimasi terhenti. Ini adalah strategi umum untuk melatih jaringan dalam. Sebagai alternatif, kita bisa menguranginya lebih agresif dengan *penurunan eksponensial*. Sayangnya, hal ini sering menyebabkan berhenti sebelum waktunya sebelum algoritma benar-benar berkumpul. Pilihan yang populer adalah *penurunan polinomial* dengan $\alpha = 0.5$. Dalam kasus optimasi konveks, terdapat beberapa bukti yang menunjukkan bahwa tingkat ini memiliki perilaku yang baik.

Mari kita lihat bagaimana penurunan eksponensial terlihat dalam praktiknya.


```{.python .input}
#@tab all
def exponential_lr():
    # Variabel global yang didefinisikan di luar fungsi ini dan diperbarui di dalamnya
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```


Seperti yang diharapkan, varians dalam parameter berkurang secara signifikan. Namun, ini mengorbankan kegagalan untuk konvergen ke solusi optimal $\mathbf{x} = (0, 0)$. Bahkan setelah 1000 langkah iterasi, kita masih sangat jauh dari solusi optimal. Memang, algoritma gagal untuk konvergen sama sekali. Di sisi lain, jika kita menggunakan *polynomial decay* di mana laju pembelajaran berkurang dengan akar kuadrat terbalik dari jumlah langkah, konvergensi menjadi lebih baik hanya setelah 50 langkah (_step_).


```{.python .input}
#@tab all
def polynomial_lr():
    # Variabel global yang didefinisikan di luar fungsi ini dan diperbarui di dalamnya
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Terdapat banyak pilihan lain mengenai bagaimana menetapkan learning rate. Misalnya, kita dapat memulai dengan learning rate yang kecil, kemudian meningkatkannya dengan cepat, dan kemudian menurunkannya lagi, meskipun secara lebih perlahan. Kita bahkan bisa mengganti-ganti antara learning rate yang kecil dan besar. Ada banyak variasi jadwal seperti ini. Untuk sekarang, mari kita fokus pada jadwal learning rate yang memungkinkan analisis teoretis yang komprehensif, yaitu pada learning rate dalam pengaturan yang konveks. Untuk masalah nonkonveks secara umum, sangat sulit untuk mendapatkan jaminan konvergensi yang bermakna, karena secara umum meminimalkan masalah nonlinear nonkonveks adalah NP-hard. Untuk survei lebih lanjut, lihat catatan kuliah [lecture notes](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf) dari Tibshirani 2015 yang sangat baik.

## Analisis Konvergensi untuk Tujuan Konveks

Analisis konvergensi berikut ini untuk stochastic gradient descent pada fungsi tujuan konveks adalah opsional dan bertujuan untuk memberikan lebih banyak intuisi tentang permasalahan ini. Kita membatasi diri pada salah satu bukti yang paling sederhana :cite:`Nesterov.Vial.2000`. Terdapat teknik bukti yang jauh lebih maju, misalnya ketika fungsi tujuan memiliki sifat yang sangat baik.

Misalkan fungsi tujuan $f(\boldsymbol{\xi}, \mathbf{x})$ adalah konveks dalam $\mathbf{x}$ untuk semua $\boldsymbol{\xi}$. Secara lebih konkret, kita mempertimbangkan pembaruan stochastic gradient descent:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

di mana $f(\boldsymbol{\xi}_t, \mathbf{x})$ adalah fungsi tujuan dengan respect pada contoh pelatihan $\boldsymbol{\xi}_t$ yang diambil dari distribusi tertentu pada langkah $t$, dan $\mathbf{x}$ adalah parameter model. Kita menandai dengan:

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

resiko yang diharapkan dan dengan $R^*$ sebagai minimumnya terkait dengan $\mathbf{x}$. Terakhir, misalkan $\mathbf{x}^*$ adalah minimizer (kita asumsikan ia ada dalam domain di mana $\mathbf{x}$ terdefinisi). Dalam kasus ini, kita bisa melacak jarak antara parameter saat ini $\mathbf{x}_t$ pada waktu $t$ dan risk minimizer $\mathbf{x}^*$ dan melihat apakah parameter tersebut membaik seiring waktu:

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Kita asumsikan bahwa norma $\ell_2$ dari stochastic gradient $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ dibatasi oleh konstanta $L$, sehingga kita memiliki:

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

Kita tertarik pada bagaimana jarak antara $\mathbf{x}_t$ dan $\mathbf{x}^*$ berubah dalam *ekspektasi*. Faktanya, untuk setiap urutan langkah tertentu, jarak mungkin saja meningkat, tergantung pada $\boldsymbol{\xi}_t$ yang kita temui. Maka dari itu, kita perlu membatasi produk dot. Karena untuk setiap fungsi konveks $f$ berlaku bahwa:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$$

untuk semua $\mathbf{x}$ dan $\mathbf{y}$, dengan konveksitas kita memiliki:

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

Memasukkan kedua ketaksamaan :eqref:`eq_sgd-L` dan :eqref:`eq_sgd-f-xi-xstar` ke dalam :eqref:`eq_sgd-xt+1-xstar`, kita memperoleh batasan pada jarak antara parameter pada waktu $t+1$ sebagai berikut:

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Ini berarti bahwa kita membuat kemajuan selama perbedaan antara loss saat ini dan loss optimal melebihi $\eta_t L^2/2$. Karena perbedaan ini cenderung konvergen ke nol, maka learning rate $\eta_t$ juga perlu *menghilang*.

Selanjutnya kita ambil ekspektasi dari :eqref:`eqref_sgd-xt-diff`. Ini menghasilkan:

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Langkah terakhir melibatkan menjumlahkan ketaksamaan untuk $t \in \{1, \ldots, T\}$. Karena jumlahnya teleskop dan dengan membuang term bawah kita memperoleh:

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

Perhatikan bahwa kita memanfaatkan bahwa $\mathbf{x}_1$ sudah diberikan dan dengan demikian ekspektasi dapat dihilangkan. Terakhir, definisikan:

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Karena:

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

dengan ketidaksetaraan Jensen (dengan $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ dalam :eqref:`eq_jensens-inequality`) dan konveksitas dari $R$, berlaku bahwa $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, sehingga:

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Memasukkan ini ke dalam ketaksamaan :eqref:`eq_sgd-x1-xstar` menghasilkan batasan:

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

di mana $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ adalah batas pada jarak antara nilai awal parameter dan hasil akhirnya. Singkatnya, kecepatan konvergensi tergantung pada seberapa besar norma dari stochastic gradient ($L$) dan seberapa jauh dari optimalitas nilai parameter awal ($r$). Perhatikan bahwa batas ini dinyatakan dalam $\bar{\mathbf{x}}$ bukan dalam $\mathbf{x}_T$. Hal ini karena $\bar{\mathbf{x}}$ adalah versi smoothed dari jalur optimasi. Apabila $r, L$, dan $T$ diketahui, kita dapat memilih learning rate $\eta = r/(L \sqrt{T})$. Ini menghasilkan batas atas $rL/\sqrt{T}$. Artinya, kita konvergen dengan laju $\mathcal{O}(1/\sqrt{T})$ menuju solusi optimal.





## Gradien Stokastik dan Sampel Terbatas

Sejauh ini, kita sedikit longgar dalam membicarakan tentang stochastic gradient descent. Kita menganggap bahwa kita menarik instance $x_i$, biasanya dengan label $y_i$ dari beberapa distribusi $p(x, y)$ dan menggunakan ini untuk memperbarui parameter model dengan cara tertentu. Secara khusus, untuk ukuran sampel terbatas, kita cukup berargumen bahwa distribusi diskrit $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ untuk beberapa fungsi $\delta_{x_i}$ dan $\delta_{y_i}$ memungkinkan kita melakukan stochastic gradient descent di atasnya.

Namun, sebenarnya ini bukan yang kita lakukan. Dalam contoh mainan di bagian ini, kita hanya menambahkan noise ke gradien yang sebenarnya non-stokastik, yaitu, kita berpura-pura memiliki pasangan $(x_i, y_i)$. Ternyata ini dibenarkan di sini (lihat latihan untuk diskusi lebih rinci). Yang lebih mengganggu adalah bahwa dalam semua diskusi sebelumnya jelas kita tidak melakukan ini. Sebaliknya, kita mengiterasi semua instance *tepat sekali*. Untuk melihat mengapa ini lebih disukai, pertimbangkan sebaliknya, yaitu kita mengambil $n$ pengamatan dari distribusi diskrit *dengan pengulangan*. Probabilitas memilih elemen $i$ secara acak adalah $1/n$. Jadi untuk memilihnya *setidaknya sekali* adalah

$$P(\textrm{terpilih~} i) = 1 - P(\textrm{dibuang~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Pemikiran serupa menunjukkan bahwa probabilitas memilih beberapa sampel (yaitu contoh pelatihan) *tepat sekali* diberikan oleh

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Pengambilan sampel dengan pengulangan mengarah pada peningkatan varians dan efisiensi data yang lebih rendah dibandingkan dengan pengambilan sampel *tanpa pengulangan*. Oleh karena itu, dalam praktiknya kita melakukan yang terakhir (dan ini adalah pilihan default di sepanjang buku ini). Terakhir, perhatikan bahwa lintasan yang berulang melalui dataset pelatihan menelusurinya dalam urutan acak yang *berbeda*.


## Ringkasan

* Untuk masalah konveks, kita dapat membuktikan bahwa untuk berbagai pilihan learning rate, stochastic gradient descent akan konvergen ke solusi optimal.
* Untuk deep learning, ini umumnya tidak terjadi. Namun, analisis masalah konveks memberikan kita wawasan berguna tentang cara pendekatan optimisasi, yaitu untuk mengurangi learning rate secara progresif, meskipun tidak terlalu cepat.
* Masalah muncul ketika learning rate terlalu kecil atau terlalu besar. Dalam praktiknya, learning rate yang sesuai sering kali ditemukan setelah beberapa percobaan.
* Ketika ada lebih banyak contoh dalam dataset pelatihan, maka biaya untuk menghitung setiap iterasi dengan gradient descent lebih tinggi, sehingga stochastic gradient descent lebih disukai dalam kasus ini.
* Jaminan optimalitas untuk stochastic gradient descent umumnya tidak tersedia dalam kasus nonkonveks karena jumlah minimum lokal yang harus diperiksa mungkin eksponensial.


## Latihan

1. Eksperimen dengan berbagai jadwal learning rate untuk stochastic gradient descent dan dengan berbagai jumlah iterasi. Khususnya, plot jarak dari solusi optimal $(0, 0)$ sebagai fungsi dari jumlah iterasi.
2. Buktikan bahwa untuk fungsi $f(x_1, x_2) = x_1^2 + 2 x_2^2$, menambahkan noise normal ke gradien setara dengan meminimalkan fungsi loss $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ di mana $\mathbf{x}$ diambil dari distribusi normal.
3. Bandingkan konvergensi stochastic gradient descent ketika kamu mengambil sampel dari $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ dengan pengulangan dan ketika kamu mengambil sampel tanpa pengulangan.
4. Bagaimana kamu akan mengubah solver stochastic gradient descent jika beberapa gradien (atau lebih tepatnya beberapa koordinat terkait dengannya) secara konsisten lebih besar dari semua gradien lainnya?
5. Misalkan $f(x) = x^2 (1 + \sin x)$. Berapa banyak minimum lokal yang dimiliki $f$? Bisakah kamu mengubah $f$ sedemikian rupa sehingga untuk meminimalkannya seseorang perlu mengevaluasi semua minimum lokal?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1067)
:end_tab:
