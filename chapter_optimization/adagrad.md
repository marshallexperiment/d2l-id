# Adagrad
:label:`sec_adagrad`

Mari kita mulai dengan mempertimbangkan masalah pembelajaran dengan fitur yang jarang muncul.


## Fitur Jarang dan Learning Rates

Bayangkan kita sedang melatih sebuah model bahasa. Untuk mendapatkan akurasi yang baik, kita biasanya ingin menurunkan learning rate seiring dengan pelatihan, biasanya pada laju $\mathcal{O}(t^{-\frac{1}{2}})$ atau lebih lambat. Sekarang, pertimbangkan sebuah model yang melatih pada fitur yang jarang, yaitu fitur yang hanya muncul sesekali. Ini umum dalam bahasa alami, misalnya, jauh lebih jarang kita akan melihat kata *preconditioning* dibandingkan dengan *learning*. Namun, ini juga umum di area lain seperti periklanan komputasional dan penyaringan kolaboratif yang dipersonalisasi. Bagaimanapun, ada banyak hal yang hanya menarik bagi sejumlah kecil orang.

Parameter yang terkait dengan fitur yang jarang hanya menerima pembaruan yang bermakna setiap kali fitur tersebut muncul. Dengan learning rate yang menurun, kita mungkin berada dalam situasi di mana parameter untuk fitur yang umum dengan cepat mencapai nilai optimalnya, sedangkan untuk fitur yang jarang, kita belum cukup sering melihatnya untuk menentukan nilai optimalnya. Dengan kata lain, learning rate menurun terlalu lambat untuk fitur yang sering atau terlalu cepat untuk fitur yang jarang.

Salah satu cara untuk mengatasi masalah ini adalah menghitung berapa kali kita melihat suatu fitur tertentu dan menggunakan ini sebagai acuan untuk menyesuaikan learning rate. Artinya, daripada memilih learning rate dalam bentuk $\eta = \frac{\eta_0}{\sqrt{t + c}}$, kita dapat menggunakan $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$. Di sini, $s(i, t)$ menghitung jumlah elemen non-nol untuk fitur $i$ yang telah kita amati hingga waktu $t$. Ini sebenarnya cukup mudah untuk diterapkan tanpa overhead yang signifikan. Namun, pendekatan ini gagal ketika kita tidak memiliki kelangkaan yang jelas tetapi hanya data di mana gradien sering kali sangat kecil dan hanya sesekali besar. Lagi pula, tidak jelas di mana kita akan menarik batas antara fitur yang dianggap teramati atau tidak.

Adagrad oleh :citet:`Duchi.Hazan.Singer.2011` mengatasi masalah ini dengan mengganti penghitung kasar $s(i, t)$ dengan agregat dari kuadrat gradien yang diamati sebelumnya. Secara khusus, Adagrad menggunakan $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ sebagai cara untuk menyesuaikan learning rate. Ini memiliki dua manfaat: pertama, kita tidak lagi perlu memutuskan kapan suatu gradien cukup besar. Kedua, metode ini menyesuaikan skala otomatis dengan besarnya gradien. Koordinat yang sering terkait dengan gradien besar diperkecil secara signifikan, sedangkan yang dengan gradien kecil diperlakukan lebih ringan. Dalam praktiknya, ini menghasilkan prosedur optimisasi yang sangat efektif untuk periklanan komputasional dan masalah terkait. Namun, ini menyembunyikan beberapa manfaat tambahan yang melekat pada Adagrad, yang paling baik dipahami dalam konteks preconditioning.


## Preconditioning

Masalah optimisasi konveks adalah contoh yang baik untuk menganalisis karakteristik algoritma. Bagaimanapun, untuk sebagian besar masalah nonkonveks, sulit untuk mendapatkan jaminan teoretis yang bermakna, tetapi *intuisi* dan *wawasan* sering kali tetap berlaku. Mari kita lihat masalah meminimalkan $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

Seperti yang kita lihat di :numref:`sec_momentum`, kita bisa menulis ulang masalah ini dalam bentuk dekomposisi eigennya $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ untuk sampai pada masalah yang jauh lebih sederhana di mana setiap koordinat dapat diselesaikan secara individual:

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Di sini kita menggunakan $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$ dan akibatnya $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$. Masalah yang dimodifikasi ini memiliki peminimumnya di $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ dan nilai minimum $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Ini jauh lebih mudah dihitung karena $\boldsymbol{\Lambda}$ adalah matriks diagonal yang berisi nilai eigen dari $\mathbf{Q}$.

Jika kita mengubah $\mathbf{c}$ sedikit, kita berharap hanya menemukan perubahan kecil dalam peminimum $f$. Sayangnya, ini tidak terjadi. Sementara perubahan kecil dalam $\mathbf{c}$ menyebabkan perubahan yang sama kecilnya dalam $\bar{\mathbf{c}}$, ini tidak berlaku untuk peminimum $f$ (dan dari $\bar{f}$). Setiap kali nilai eigen $\boldsymbol{\Lambda}_i$ besar, kita hanya akan melihat perubahan kecil dalam $\bar{x}_i$ dan dalam minimum $\bar{f}$. Sebaliknya, untuk $\boldsymbol{\Lambda}_i$ yang kecil, perubahan dalam $\bar{x}_i$ bisa dramatis. Rasio antara nilai eigen terbesar dan terkecil disebut angka kondisi masalah optimisasi.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Jika angka kondisi $\kappa$ besar, sulit untuk menyelesaikan masalah optimisasi dengan akurat. Kita perlu memastikan bahwa kita berhati-hati dalam memperhitungkan rentang dinamis nilai yang besar. Analisis kita mengarah pada pertanyaan yang jelas, meskipun agak naif: tidakkah kita bisa "memperbaiki" masalah dengan mendistorsi ruang sehingga semua nilai eigen adalah $1$. Secara teori ini cukup mudah: kita hanya perlu nilai eigen dan vektor eigen dari $\mathbf{Q}$ untuk mengubah masalah dari $\mathbf{x}$ ke satu dalam $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. Dalam sistem koordinat baru $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ bisa disederhanakan menjadi $\|\mathbf{z}\|^2$. Sayangnya, ini adalah saran yang cukup tidak praktis. Menghitung nilai eigen dan vektor eigen umumnya *jauh lebih* mahal daripada menyelesaikan masalah itu sendiri.

Meskipun menghitung nilai eigen secara tepat mungkin mahal, menebaknya dan menghitungnya bahkan secara perkiraan mungkin sudah jauh lebih baik daripada tidak melakukan apa-apa. Secara khusus, kita dapat menggunakan elemen diagonal dari $\mathbf{Q}$ dan mengubah skala sesuai.

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Dalam hal ini, kita memiliki $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ dan khususnya $\tilde{\mathbf{Q}}_{ii} = 1$ untuk semua $i$. Dalam kebanyakan kasus ini menyederhanakan angka kondisi secara signifikan. Misalnya, dalam kasus yang kita bahas sebelumnya, ini sepenuhnya menghilangkan masalah yang ada karena masalah ini sejajar dengan sumbu.

Sayangnya, kita menghadapi masalah lain: dalam deep learning, kita biasanya bahkan tidak memiliki akses ke turunan kedua dari fungsi objektif: untuk $\mathbf{x} \in \mathbb{R}^d$ turunan kedua bahkan pada satu minibatch mungkin memerlukan ruang dan pekerjaan sebesar $\mathcal{O}(d^2)$ untuk dihitung, sehingga tidak dapat diterapkan secara praktis. Ide cerdas dari Adagrad adalah menggunakan proksi untuk elemen diagonal dari Hessian yang sulit didapat tetapi relatif murah dihitung dan efektif—magnitudo gradien itu sendiri.

Untuk melihat mengapa ini berhasil, mari kita lihat $\bar{f}(\bar{\mathbf{x}})$. Kita memiliki

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

di mana $\bar{\mathbf{x}}_0$ adalah peminimum dari $\bar{f}$. Oleh karena itu, magnitudo gradien bergantung pada $\boldsymbol{\Lambda}$ dan jarak dari optimalitas. Jika $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ tidak berubah, ini akan menjadi semua yang dibutuhkan. Bagaimanapun, dalam kasus ini, magnitudo dari gradien $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ sudah mencukupi. Karena Adagrad adalah algoritma stochastic gradient descent, kita akan melihat gradien dengan varians non-nol bahkan pada optimalitas. Akibatnya, kita dapat menggunakan varians gradien sebagai proksi murah untuk skala dari Hessian. Analisis mendalam berada di luar cakupan bagian ini (akan memerlukan beberapa halaman). Kami merujuk pembaca pada :cite:`Duchi.Hazan.Singer.2011` untuk detailnya.

## Algoritma

Mari kita formalkan diskusi di atas. Kita menggunakan variabel $\mathbf{s}_t$ untuk mengakumulasi varians gradien masa lalu sebagai berikut.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Di sini operasi diterapkan per koordinat. Artinya, $\mathbf{v}^2$ memiliki elemen $v_i^2$. Demikian pula $\frac{1}{\sqrt{v}}$ memiliki elemen $\frac{1}{\sqrt{v_i}}$ dan $\mathbf{u} \cdot \mathbf{v}$ memiliki elemen $u_i v_i$. Seperti sebelumnya, $\eta$ adalah learning rate dan $\epsilon$ adalah konstanta tambahan yang memastikan kita tidak membagi dengan $0$. Terakhir, kita inisialisasi $\mathbf{s}_0 = \mathbf{0}$.

Seperti pada kasus momentum, kita perlu melacak variabel tambahan, dalam hal ini untuk memungkinkan learning rate individual per koordinat. Ini tidak meningkatkan biaya Adagrad secara signifikan relatif terhadap SGD, karena biaya utama biasanya adalah menghitung $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ dan turunannya.

Perhatikan bahwa mengakumulasi kuadrat gradien dalam $\mathbf{s}_t$ berarti $\mathbf{s}_t$ tumbuh secara linear (sedikit lebih lambat secara praktik, karena gradien awalnya menurun). Ini menghasilkan learning rate $\mathcal{O}(t^{-\frac{1}{2}})$, meskipun disesuaikan per koordinat. Untuk masalah konveks, ini sudah memadai. Namun, dalam deep learning, kita mungkin ingin menurunkan learning rate lebih lambat. Hal ini melahirkan berbagai varian Adagrad yang akan kita bahas di bab berikutnya. Untuk sekarang, mari kita lihat bagaimana Adagrad bekerja pada masalah konveks kuadratik. Kita menggunakan masalah yang sama seperti sebelumnya:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Kita akan mengimplementasikan Adagrad menggunakan learning rate yang sama seperti sebelumnya, yaitu $\eta = 0.4$. Seperti yang kita lihat, lintasan iteratif dari variabel independen lebih mulus. Namun, karena efek kumulatif dari $\boldsymbol{s}_t$, learning rate terus menurun, sehingga variabel independen tidak bergerak sebanyak pada tahap-tahap iterasi yang lebih akhir.

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Ketika kita meningkatkan learning rate menjadi $2$, kita melihat perilaku yang jauh lebih baik. Hal ini sudah mengindikasikan bahwa penurunan learning rate mungkin terlalu agresif, bahkan dalam kasus tanpa noise, dan kita perlu memastikan bahwa parameter dapat berkumpul secara tepat.


```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Implementasi dari Awal

Seperti metode momentum, Adagrad perlu mempertahankan variabel status dengan bentuk yang sama seperti parameter.


```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Dibandingkan dengan eksperimen pada :numref:`sec_minibatch_sgd`, kita menggunakan learning rate yang lebih besar untuk melatih model.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Implementasi Ringkas

Dengan menggunakan instance `Trainer` dari algoritma `adagrad`, kita dapat memanggil algoritma Adagrad di Gluon.


```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Ringkasan

* Adagrad menurunkan learning rate secara dinamis berdasarkan setiap koordinat.
* Adagrad menggunakan magnitudo gradien sebagai cara untuk menyesuaikan seberapa cepat kemajuan dicapai - koordinat dengan gradien besar dikompensasi dengan learning rate yang lebih kecil.
* Menghitung turunan kedua yang tepat biasanya tidak dapat dilakukan pada masalah deep learning karena keterbatasan memori dan komputasi. Gradien dapat menjadi proksi yang berguna.
* Jika masalah optimisasi memiliki struktur yang agak tidak merata, Adagrad dapat membantu mengurangi distorsi.
* Adagrad sangat efektif untuk fitur yang jarang di mana learning rate perlu menurun lebih lambat untuk istilah yang jarang muncul.
* Pada masalah deep learning, Adagrad kadang-kadang terlalu agresif dalam mengurangi learning rate. Kita akan membahas strategi untuk mengurangi hal ini dalam konteks :numref:`sec_adam`.

## Latihan

1. Buktikan bahwa untuk matriks ortogonal $\mathbf{U}$ dan vektor $\mathbf{c}$ berlaku: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Mengapa ini berarti bahwa magnitudo gangguan tidak berubah setelah perubahan variabel ortogonal?
2. Coba Adagrad untuk $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ dan juga untuk fungsi objektif yang diputar sebesar 45 derajat, yaitu $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Apakah perilakunya berbeda?
3. Buktikan [teorema lingkaran Gerschgorin](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) yang menyatakan bahwa nilai eigen $\lambda_i$ dari matriks $\mathbf{M}$ memenuhi $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ untuk setidaknya satu pilihan $j$.
4. Apa yang teorema Gerschgorin katakan tentang nilai eigen dari matriks yang dikondisi-diagonal $\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
5. Coba Adagrad untuk jaringan deep yang sesuai, seperti :numref:`sec_lenet` yang diterapkan pada Fashion-MNIST.
6. Bagaimana Anda perlu memodifikasi Adagrad untuk mencapai pengurangan learning rate yang tidak terlalu agresif?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1073)
:end_tab:
