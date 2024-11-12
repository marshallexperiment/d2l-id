# Momentum
:label:`sec_momentum`

Pada :numref:`sec_sgd`, kita telah meninjau apa yang terjadi ketika melakukan stochastic gradient descent, yaitu saat melakukan optimisasi di mana hanya varian gradien yang berisik (noisy) yang tersedia. Secara khusus, kita melihat bahwa untuk gradien yang berisik, kita harus berhati-hati dalam memilih learning rate saat menghadapi noise. Jika kita menurunkannya terlalu cepat, konvergensi akan terhenti. Jika terlalu longgar, kita gagal mencapai solusi yang cukup baik karena noise terus menjauhkan kita dari titik optimal.

## Dasar-Dasar

Di bagian ini, kita akan mengeksplorasi algoritma optimisasi yang lebih efektif, terutama untuk jenis-jenis masalah optimisasi tertentu yang umum ditemui dalam praktik.


### Leaky Averages

Pada bagian sebelumnya, kita membahas minibatch SGD sebagai sarana untuk mempercepat komputasi. Pendekatan ini juga memiliki efek samping yang baik, yaitu rata-rata gradien mengurangi jumlah variansi. Minibatch stochastic gradient descent dapat dihitung sebagai berikut:

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Untuk kesederhanaan notasi, di sini kita menggunakan $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ sebagai stochastic gradient descent untuk sampel $i$ menggunakan bobot yang diperbarui pada waktu $t-1$. Akan lebih baik jika kita bisa mendapatkan manfaat dari efek pengurangan variansi bahkan di luar rata-rata gradien pada minibatch. Salah satu opsi untuk mencapai tugas ini adalah mengganti perhitungan gradien dengan "rata-rata bocor" (*leaky average*):

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

dengan $\beta \in (0, 1)$. Ini secara efektif menggantikan gradien instan dengan yang telah dirata-rata selama beberapa gradien *masa lalu*. $\mathbf{v}$ disebut *kecepatan* (*velocity*). Ini mengakumulasi gradien masa lalu, mirip dengan bagaimana bola berat yang bergulir menuruni lanskap fungsi objektif mengintegrasikan gaya-gaya masa lalu. Untuk melihat apa yang terjadi secara lebih rinci, mari kita kembangkan $\mathbf{v}_t$ secara rekursif menjadi

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

Nilai $\beta$ yang besar berarti rata-rata jangka panjang, sedangkan nilai $\beta$ yang kecil berarti hanya koreksi kecil dibandingkan metode gradien. Penggantian gradien yang baru ini tidak lagi mengarah ke arah turunan terbesar pada sebuah instance tertentu tetapi lebih ke arah rata-rata tertimbang dari gradien masa lalu. Ini memungkinkan kita untuk mewujudkan sebagian besar manfaat dari rata-rata batch tanpa biaya komputasi gradien pada batch tersebut. Kita akan meninjau prosedur rata-rata ini secara lebih rinci nanti.

Pemikiran di atas membentuk dasar dari apa yang sekarang dikenal sebagai metode gradien *terakselerasi*, seperti gradien dengan momentum. Metode ini memiliki manfaat tambahan, yaitu lebih efektif dalam kasus di mana masalah optimisasi kurang terkondisi dengan baik (misalnya, di mana ada beberapa arah di mana kemajuan jauh lebih lambat daripada di arah lain, mirip dengan sebuah ngarai sempit). Selain itu, metode ini memungkinkan kita untuk merata-rata gradien berturut-turut untuk mendapatkan arah turunan yang lebih stabil. Faktanya, aspek percepatan bahkan untuk masalah konveks tanpa noise adalah salah satu alasan utama mengapa momentum bekerja dan mengapa ini bekerja dengan sangat baik.

Seperti yang dapat diduga, karena keefektifannya, momentum adalah subjek yang dipelajari dengan baik dalam optimisasi untuk deep learning dan di luar itu. Lihat misalnya [artikel ekspositori](https://distill.pub/2017/momentum/) yang indah oleh :citet:`Goh.2017` untuk analisis mendalam dan animasi interaktif. Metode ini diusulkan oleh :citet:`Polyak.1964`. :citet:`Nesterov.2018` memiliki pembahasan teoretis rinci dalam konteks optimisasi konveks. Momentum dalam deep learning telah diketahui bermanfaat sejak lama. Lihat misalnya diskusi oleh :citet:`Sutskever.Martens.Dahl.ea.2013` untuk detail lebih lanjut.

### Masalah yang Kurang Terkondisi dengan Baik

Untuk mendapatkan pemahaman yang lebih baik tentang sifat geometris dari metode momentum, kita akan meninjau kembali gradient descent, tetapi dengan fungsi objektif yang jauh lebih tidak menyenangkan. Ingat bahwa pada :numref:`sec_gd` kita menggunakan $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, yaitu fungsi objektif elipsoid yang agak terdistorsi. Kita mendistorsi fungsi ini lebih jauh dengan memperpanjangnya ke arah $x_1$ melalui

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Seperti sebelumnya, $f$ memiliki minimumnya pada $(0, 0)$. Fungsi ini *sangat* datar dalam arah $x_1$. Mari kita lihat apa yang terjadi ketika kita melakukan gradient descent seperti sebelumnya pada fungsi baru ini. Kita memilih learning rate sebesar $0.4$.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Secara konstruksi, gradien dalam arah $x_2$ *jauh* lebih tinggi dan berubah jauh lebih cepat dibandingkan dengan arah horizontal $x_1$. Akibatnya, kita terjebak di antara dua pilihan yang tidak diinginkan: jika kita memilih learning rate yang kecil, kita memastikan bahwa solusi tidak divergen dalam arah $x_2$, tetapi kita harus menghadapi konvergensi yang lambat dalam arah $x_1$. Sebaliknya, dengan learning rate yang besar, kita dapat maju dengan cepat dalam arah $x_1$, tetapi solusi kita divergen dalam arah $x_2$. Contoh di bawah ini mengilustrasikan apa yang terjadi bahkan setelah sedikit peningkatan learning rate dari $0.4$ menjadi $0.6$. Konvergensi dalam arah $x_1$ membaik, tetapi kualitas keseluruhan solusi jauh lebih buruk.


```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Metode Momentum

Metode momentum memungkinkan kita untuk mengatasi masalah gradient descent yang dijelaskan di atas. Melihat jejak optimisasi di atas, kita mungkin merasa bahwa merata-rata gradien dari masa lalu akan bekerja dengan baik. Bagaimanapun, dalam arah $x_1$, ini akan menggabungkan gradien yang sejajar dengan baik, sehingga meningkatkan jarak yang kita tempuh di setiap langkah. Sebaliknya, dalam arah $x_2$ di mana gradien berosilasi, gradien agregat akan mengurangi ukuran langkah karena osilasi yang saling meniadakan.

Dengan menggunakan $\mathbf{v}_t$ alih-alih gradien $\mathbf{g}_t$, kita memperoleh persamaan pembaruan berikut:


$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Perhatikan bahwa untuk $\beta = 0$, kita mendapatkan kembali metode gradient descent biasa. Sebelum kita mendalami sifat matematisnya lebih lanjut, mari kita lihat sekilas bagaimana algoritma ini berperilaku dalam praktik.


```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Seperti yang kita lihat, bahkan dengan learning rate yang sama seperti sebelumnya, momentum masih dapat mencapai konvergensi dengan baik. Mari kita lihat apa yang terjadi ketika kita menurunkan parameter momentum. Menguranginya menjadi setengah, yaitu $\beta = 0.25$, menghasilkan lintasan yang hampir tidak konvergen sama sekali. Meskipun begitu, ini masih jauh lebih baik daripada tanpa momentum (di mana solusi divergen).


```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Perlu dicatat bahwa kita dapat menggabungkan momentum dengan stochastic gradient descent dan khususnya dengan minibatch stochastic gradient descent. Satu-satunya perubahan adalah bahwa dalam kasus tersebut kita mengganti gradien $\mathbf{g}_{t, t-1}$ dengan $\mathbf{g}_t$. Terakhir, untuk kemudahan, kita menginisialisasi $\mathbf{v}_0 = 0$ pada waktu $t=0$. Mari kita lihat apa yang sebenarnya dilakukan oleh leaky averaging pada pembaruan.

### Bobot Sampel Efektif

Ingat bahwa $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Dalam batasnya, istilah-istilah ini bertambah menjadi $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Dengan kata lain, alih-alih mengambil langkah sebesar $\eta$ dalam gradient descent atau stochastic gradient descent, kita mengambil langkah sebesar $\frac{\eta}{1-\beta}$ sambil mendapatkan arah descent yang berpotensi lebih stabil. Ini adalah dua manfaat dalam satu. Untuk mengilustrasikan bagaimana pembobotan berperilaku untuk berbagai pilihan $\beta$, perhatikan diagram di bawah ini.


```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Eksperimen Praktis

Mari kita lihat bagaimana momentum bekerja dalam praktik, yaitu saat digunakan dalam konteks optimizer yang sebenarnya. Untuk ini, kita memerlukan implementasi yang lebih skalabel.

### Implementasi dari Awal

Dibandingkan dengan stochastic gradient descent (minibatch), metode momentum perlu mempertahankan satu set variabel tambahan, yaitu kecepatan (velocity). Variabel ini memiliki bentuk yang sama seperti gradien (dan variabel dari masalah optimisasi). Dalam implementasi di bawah ini, kita menyebut variabel ini sebagai `states`.


```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Let's see how this works in practice.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Ketika kita meningkatkan hyperparameter momentum `momentum` menjadi 0.9, ini setara dengan ukuran sampel efektif yang jauh lebih besar, yaitu $\frac{1}{1 - 0.9} = 10$. Kita sedikit menurunkan learning rate menjadi $0.01$ untuk menjaga stabilitas.


```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

Menurunkan learning rate lebih lanjut dapat mengatasi masalah pada optimisasi yang tidak mulus. Dengan mengaturnya ke $0.005$, kita mendapatkan sifat konvergensi yang baik.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Implementasi Ringkas

Di Gluon, hanya sedikit yang perlu dilakukan karena solver `sgd` standar sudah memiliki momentum bawaan. Dengan mengatur parameter yang sesuai, kita mendapatkan lintasan yang sangat mirip.


```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Analisis Teoretis

Sejauh ini, contoh 2D dari $f(x) = 0.1 x_1^2 + 2 x_2^2$ tampak agak dibuat-buat. Sekarang kita akan melihat bahwa ini sebenarnya cukup representatif untuk jenis masalah yang mungkin ditemui, setidaknya dalam kasus meminimalkan fungsi objektif kuadratik konveks.

### Fungsi Kuadratik Konveks

Pertimbangkan fungsi

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Ini adalah fungsi kuadratik umum. Untuk matriks positif definit $\mathbf{Q} \succ 0$, yaitu, untuk matriks dengan nilai eigen positif, ini memiliki minimizer pada $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ dengan nilai minimum $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Dengan demikian, kita dapat menulis ulang $h$ sebagai

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradiennya diberikan oleh $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Artinya, gradiennya diberikan oleh jarak antara $\mathbf{x}$ dan minimizer, dikalikan dengan $\mathbf{Q}$. Akibatnya, kecepatan (*velocity*) juga merupakan kombinasi linear dari istilah $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

Karena $\mathbf{Q}$ adalah positif definit, maka matriks ini dapat diuraikan ke dalam sistem eigen melalui $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ untuk matriks ortogonal (rotasi) $\mathbf{O}$ dan matriks diagonal $\boldsymbol{\Lambda}$ yang berisi nilai eigen positif. Hal ini memungkinkan kita untuk melakukan perubahan variabel dari $\mathbf{x}$ ke $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ untuk memperoleh ekspresi yang lebih sederhana:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Di sini, $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Karena $\mathbf{O}$ hanya merupakan matriks ortogonal, ini tidak mengganggu gradien secara signifikan. Jika dinyatakan dalam $\mathbf{z}$, gradient descent menjadi

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Fakta penting dalam ekspresi ini adalah bahwa gradient descent *tidak mencampur* antara ruang eigen yang berbeda. Artinya, ketika dinyatakan dalam sistem eigen dari $\mathbf{Q}$, masalah optimisasi berjalan dengan cara koordinat per koordinat. Hal ini juga berlaku untuk

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Dengan melakukan ini, kita baru saja membuktikan teorema berikut: gradient descent dengan atau tanpa momentum untuk fungsi kuadratik konveks terurai menjadi optimisasi koordinat per koordinat dalam arah vektor eigen dari matriks kuadratik.

### Fungsi Skalar

Berdasarkan hasil di atas, mari kita lihat apa yang terjadi ketika kita meminimalkan fungsi $f(x) = \frac{\lambda}{2} x^2$. Untuk gradient descent, kita memiliki

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Jika $|1 - \eta \lambda| < 1$, optimisasi ini akan konvergen secara eksponensial karena setelah $t$ langkah, kita memiliki $x_t = (1 - \eta \lambda)^t x_0$. Ini menunjukkan bagaimana laju konvergensi awalnya meningkat ketika kita menambah learning rate $\eta$ hingga $\eta \lambda = 1$. Setelah itu, nilai-nilai mulai divergen, dan untuk $\eta \lambda > 2$, masalah optimisasi menjadi divergen.


```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Untuk menganalisis konvergensi dalam kasus momentum, kita mulai dengan menuliskan ulang persamaan pembaruan dalam bentuk dua skalar: satu untuk $x$ dan satu untuk kecepatan $v$. Ini menghasilkan:


$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Di sini kita menggunakan $\mathbf{R}$ untuk menunjukkan matriks $2 \times 2$ yang mengatur perilaku konvergensi. Setelah $t$ langkah, pilihan awal $[v_0, x_0]$ menjadi $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Dengan demikian, kecepatan konvergensi ditentukan oleh nilai eigen dari $\mathbf{R}$. Lihat [Distill post](https://distill.pub/2017/momentum/) oleh :citet:`Goh.2017` untuk animasi yang bagus dan :citet:`Flammarion.Bach.2015` untuk analisis mendetail. Dapat ditunjukkan bahwa untuk $0 < \eta \lambda < 2 + 2 \beta$, kecepatan konvergen. Ini adalah rentang parameter yang lebih besar dibandingkan dengan $0 < \eta \lambda < 2$ untuk gradient descent. Ini juga menunjukkan bahwa secara umum nilai $\beta$ yang besar diinginkan. Detail lebih lanjut memerlukan cukup banyak teknis, dan kami menyarankan pembaca yang tertarik untuk melihat publikasi asli.

## Ringkasan

* Momentum menggantikan gradien dengan rata-rata bocor dari gradien masa lalu, yang mempercepat konvergensi secara signifikan.
* Momentum diinginkan untuk gradient descent bebas-noise dan stochastic gradient descent yang (berisik).
* Momentum mencegah terhentinya proses optimisasi yang lebih mungkin terjadi pada stochastic gradient descent.
* Jumlah efektif gradien diberikan oleh $\frac{1}{1-\beta}$ karena penurunan eksponensial data masa lalu.
* Dalam kasus masalah kuadratik konveks, hal ini dapat dianalisis secara rinci.
* Implementasi cukup sederhana tetapi membutuhkan penyimpanan vektor status tambahan (kecepatan $\mathbf{v}$).

## Latihan

1. Gunakan kombinasi lain dari hyperparameter momentum dan learning rate dan amati serta analisis hasil eksperimen yang berbeda.
2. Coba gradient descent dan momentum untuk masalah kuadratik dengan beberapa nilai eigen, yaitu $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, misalnya, $\lambda_i = 2^{-i}$. Buat grafik bagaimana nilai $x$ menurun untuk inisialisasi $x_i = 1$.
3. Turunkan nilai minimum dan minimizer untuk $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
4. Apa yang berubah ketika kita melakukan stochastic gradient descent dengan momentum? Apa yang terjadi ketika kita menggunakan minibatch stochastic gradient descent dengan momentum? Bereksperimenlah dengan parameter.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1071)
:end_tab:
