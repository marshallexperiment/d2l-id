# Adadelta
:label:`sec_adadelta`

Adadelta adalah varian lain dari AdaGrad (:numref:`sec_adagrad`). Perbedaan utamanya adalah pada pengurangan adaptivitas learning rate terhadap koordinat. Selain itu, secara tradisional, Adadelta disebut tidak memiliki learning rate karena menggunakan jumlah perubahan itu sendiri sebagai kalibrasi untuk perubahan di masa depan. Algoritma ini diusulkan oleh :citet:`Zeiler.2012`. Berdasarkan pembahasan algoritma sebelumnya, konsep Adadelta cukup sederhana.

## Algoritma

Secara singkat, Adadelta menggunakan dua variabel status: $\mathbf{s}_t$ untuk menyimpan rata-rata bocor dari momen kedua gradien dan $\Delta\mathbf{x}_t$ untuk menyimpan rata-rata bocor dari momen kedua perubahan parameter dalam model itu sendiri. Perlu dicatat bahwa kita menggunakan notasi dan nama asli dari penulis untuk kompatibilitas dengan publikasi dan implementasi lainnya (tidak ada alasan khusus mengapa kita harus menggunakan variabel Yunani yang berbeda untuk menunjukkan parameter yang sama seperti pada momentum, Adagrad, RMSProp, dan Adadelta).

Berikut adalah rincian teknis dari Adadelta. Diberikan parameter du jour $\rho$, kita mendapatkan pembaruan bocor berikut yang mirip dengan :numref:`sec_rmsprop`:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Perbedaannya dengan :numref:`sec_rmsprop` adalah bahwa kita melakukan pembaruan dengan gradien yang telah diskalakan ulang $\mathbf{g}_t'$, yaitu

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Jadi, apa itu gradien yang diskalakan ulang $\mathbf{g}_t'$? Kita dapat menghitungnya sebagai berikut:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

di mana $\Delta \mathbf{x}_{t-1}$ adalah rata-rata bocor dari gradien yang telah diskalakan ulang $\mathbf{g}_t'$. Kita menginisialisasi $\Delta \mathbf{x}_{0}$ dengan $0$ dan memperbaruinya di setiap langkah menggunakan $\mathbf{g}_t'$, yaitu

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

dan $\epsilon$ (nilai kecil seperti $10^{-5}$) ditambahkan untuk menjaga stabilitas numerik.



## Implementasi

Adadelta perlu mempertahankan dua variabel status untuk setiap variabel, yaitu $\mathbf{s}_t$ dan $\Delta\mathbf{x}_t$. Hal ini menghasilkan implementasi berikut.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Memilih $\rho = 0.9$ setara dengan waktu paruh 10 untuk setiap pembaruan parameter. Pilihan ini cenderung bekerja dengan baik. Kita mendapatkan perilaku sebagai berikut.


```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Untuk implementasi ringkas, kita cukup menggunakan algoritma Adadelta dari API tingkat tinggi. Hal ini menghasilkan satu baris kode untuk pemanggilan yang jauh lebih sederhana.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# Adadelta tidak mengalami konvergensi pada learning rate default
# tetapi mengalami konvergensi pada lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Ringkasan

* Adadelta tidak memiliki parameter learning rate. Sebagai gantinya, ia menggunakan laju perubahan pada parameter itu sendiri untuk menyesuaikan learning rate.
* Adadelta memerlukan dua variabel status untuk menyimpan momen kedua dari gradien dan perubahan parameter.
* Adadelta menggunakan rata-rata bocor untuk mempertahankan estimasi statistik yang sesuai secara berkelanjutan.

## Latihan

1. Sesuaikan nilai $\rho$. Apa yang terjadi?
2. Tunjukkan bagaimana cara mengimplementasikan algoritma tanpa menggunakan $\mathbf{g}_t'$. Mengapa ini mungkin ide yang baik?
3. Apakah Adadelta benar-benar bebas learning rate? Bisakah Anda menemukan masalah optimisasi yang membuat Adadelta gagal?
4. Bandingkan Adadelta dengan Adagrad dan RMSProp untuk mendiskusikan perilaku konvergensinya.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1077)
:end_tab:
