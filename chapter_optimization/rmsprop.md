# RMSProp
:label:`sec_rmsprop`

Salah satu masalah utama dalam :numref:`sec_adagrad` adalah bahwa laju pembelajaran (learning rate) menurun dengan jadwal yang telah ditentukan secara efektif $\mathcal{O}(t^{-\frac{1}{2}})$. Meskipun ini umumnya sesuai untuk masalah *convex*, hal ini mungkin tidak ideal untuk masalah *nonconvex*, seperti yang sering ditemui dalam *deep learning*. Namun, adaptivitas per-koordinat dari Adagrad sangat diinginkan sebagai *preconditioner*.

:citet:`Tieleman.Hinton.2012` mengusulkan algoritma RMSProp sebagai solusi sederhana untuk memisahkan penjadwalan laju dari laju pembelajaran adaptif per-koordinat. Masalahnya adalah bahwa Adagrad mengakumulasi kuadrat dari gradien $\mathbf{g}_t$ ke dalam vektor status $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. Akibatnya, $\mathbf{s}_t$ terus tumbuh tanpa batas karena kurangnya normalisasi, secara esensial linear seiring algoritma konvergen.

Salah satu cara untuk memperbaiki masalah ini adalah dengan menggunakan $\mathbf{s}_t / t$. Untuk distribusi $\mathbf{g}_t$ yang wajar, ini akan konvergen. Sayangnya, bisa memakan waktu yang sangat lama sampai perilaku limit mulai berpengaruh karena prosedur ini mengingat seluruh lintasan nilai. Alternatifnya adalah menggunakan rata-rata bocor (leaky average) dengan cara yang sama seperti yang kita gunakan pada metode momentum, yaitu $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ untuk beberapa parameter $\gamma > 0$. Menjaga bagian lain tetap tidak berubah menghasilkan RMSProp.

## Algoritma

Mari kita tuliskan persamaannya secara detail.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Konstanta $\epsilon > 0$ biasanya diatur ke $10^{-6}$ untuk memastikan bahwa kita tidak mengalami pembagian dengan nol atau ukuran langkah yang terlalu besar. Dengan ekspansi ini, kita sekarang bebas mengendalikan laju pembelajaran $\eta$ secara independen dari skala yang diterapkan pada setiap koordinat. Dalam hal rata-rata bocor (leaky average), kita dapat menerapkan pemikiran yang sama seperti yang sebelumnya diterapkan pada kasus metode momentum. Dengan mengembangkan definisi $\mathbf{s}_t$, kita mendapatkan

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots \right).
\end{aligned}
$$

Seperti sebelumnya dalam :numref:`sec_momentum`, kita menggunakan $1 + \gamma + \gamma^2 + \ldots = \frac{1}{1-\gamma}$. Oleh karena itu, jumlah bobot dinormalisasi menjadi $1$ dengan waktu paruh dari sebuah observasi sebesar $\gamma^{-1}$. Mari kita visualisasikan bobot untuk 40 langkah waktu terakhir untuk berbagai pilihan $\gamma$.


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
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Implementasi dari Awal

Seperti sebelumnya, kita menggunakan fungsi kuadrat $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ untuk mengamati lintasan dari RMSProp. Ingat bahwa pada :numref:`sec_adagrad`, ketika kita menggunakan Adagrad dengan laju pembelajaran sebesar 0.4, variabel-variabel bergerak sangat lambat pada tahap akhir algoritma karena laju pembelajaran menurun terlalu cepat. Karena $\eta$ dikendalikan secara terpisah, hal ini tidak terjadi pada RMSProp.


```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Selanjutnya, kita mengimplementasikan RMSProp untuk digunakan dalam jaringan dalam (deep network). Implementasinya juga cukup sederhana.


```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Kami menetapkan laju pembelajaran awal ke 0.01 dan parameter pembobot $\gamma$ ke 0.9. Artinya, $\mathbf{s}$ mengakumulasi rata-rata dari 1/(1-\gamma) = 10 pengamatan dari kuadrat gradien sebelumnya.



```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Implementasi Singkat

Karena RMSProp adalah algoritma yang cukup populer, maka algoritma ini juga tersedia dalam instance `Trainer`. Yang perlu kita lakukan hanyalah menginstansiasi menggunakan algoritma bernama `rmsprop`, menetapkan nilai $\gamma$ ke parameter `gamma1`.


```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Ringkasan

* RMSProp sangat mirip dengan Adagrad sejauh keduanya menggunakan kuadrat dari gradien untuk mengatur koefisien.
* RMSProp memiliki kesamaan dengan momentum dalam hal rata-rata eksponensial. Namun, RMSProp menggunakan teknik ini untuk menyesuaikan preconditioner per-koordinat.
* Learning rate perlu dijadwalkan oleh eksperimen secara praktik.
* Koefisien $\gamma$ menentukan berapa lama sejarah digunakan saat menyesuaikan skala per-koordinat.

## Latihan

1. Apa yang terjadi secara eksperimental jika kita menetapkan $\gamma = 1$? Mengapa?
2. Rotasikan masalah optimasi untuk meminimalkan $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Apa yang terjadi pada konvergensi?
3. Coba apa yang terjadi pada RMSProp pada masalah machine learning nyata, seperti pelatihan pada Fashion-MNIST. Bereksperimenlah dengan berbagai pilihan untuk menyesuaikan learning rate.
4. Apakah Anda ingin menyesuaikan $\gamma$ seiring perkembangan optimasi? Seberapa sensitif RMSProp terhadap hal ini?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
