# Adam
:label:`sec_adam`

Dalam pembahasan sebelumnya, kita telah menemukan beberapa teknik untuk optimisasi yang efisien. Mari kita ulas secara rinci di sini:

* Kita melihat bahwa :numref:`sec_sgd` lebih efektif daripada Gradient Descent dalam menyelesaikan masalah optimisasi, misalnya, karena ketahanannya terhadap data yang redundan.
* Kita melihat bahwa :numref:`sec_minibatch_sgd` memberikan efisiensi tambahan yang signifikan melalui vektorisasi, dengan menggunakan set pengamatan yang lebih besar dalam satu minibatch. Ini adalah kunci untuk pemrosesan paralel yang efisien, baik pada mesin multi, multi-GPU, dan skala keseluruhan.
* :numref:`sec_momentum` menambahkan mekanisme untuk menggabungkan sejarah gradien masa lalu guna mempercepat konvergensi.
* :numref:`sec_adagrad` menggunakan penskalaan per-koordinat untuk memungkinkan preconditioner yang efisien secara komputasional.
* :numref:`sec_rmsprop` memisahkan penskalaan per-koordinat dari penyesuaian learning rate.

Adam :cite:`Kingma.Ba.2014` menggabungkan semua teknik ini menjadi satu algoritma pembelajaran yang efisien. Seperti yang diharapkan, ini adalah algoritma yang cukup populer sebagai salah satu algoritma optimisasi yang lebih andal dan efektif untuk digunakan dalam deep learning. Namun, algoritma ini tidak tanpa masalah. Secara khusus, :cite:`Reddi.Kale.Kumar.2019` menunjukkan bahwa ada situasi di mana Adam dapat divergen karena pengendalian varians yang buruk. Dalam penelitian lanjutan, :citet:`Zaheer.Reddi.Sachan.ea.2018` mengusulkan perbaikan untuk Adam yang disebut Yogi, yang menangani masalah ini. Akan kita bahas lebih lanjut nanti. Untuk saat ini, mari kita tinjau algoritma Adam.

## Algoritma

Salah satu komponen utama dari Adam adalah penggunaan eksponensial weighted moving averages (juga dikenal sebagai leaky averaging) untuk mendapatkan estimasi baik dari momentum maupun momen kedua dari gradien. Artinya, algoritma ini menggunakan variabel status

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Di sini, $\beta_1$ dan $\beta_2$ adalah parameter pembobotan yang tidak negatif. Pilihan umum untuk parameter ini adalah $\beta_1 = 0.9$ dan $\beta_2 = 0.999$. Artinya, estimasi varians bergerak *jauh lebih lambat* daripada istilah momentum. Perlu dicatat bahwa jika kita menginisialisasi $\mathbf{v}_0 = \mathbf{s}_0 = 0$, kita memiliki kecenderungan bias awal menuju nilai yang lebih kecil. Hal ini dapat diatasi dengan menggunakan fakta bahwa $\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ untuk menormalkan ulang istilah-istilah tersebut. Dengan demikian, variabel status yang dinormalisasi diberikan oleh

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ dan } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Dengan estimasi yang tepat, sekarang kita dapat menuliskan persamaan pembaruan. Pertama, kita melakukan penskalaan ulang gradien mirip dengan RMSProp untuk mendapatkan

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Berbeda dengan RMSProp, pembaruan kita menggunakan momentum $\hat{\mathbf{v}}_t$ alih-alih gradien itu sendiri. Selain itu, terdapat perbedaan kecil dalam kosmetika, di mana penskalaan terjadi menggunakan $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ alih-alih $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Yang pertama umumnya sedikit lebih baik dalam praktik, sehingga berbeda dari RMSProp. Biasanya kita memilih $\epsilon = 10^{-6}$ untuk keseimbangan yang baik antara stabilitas numerik dan fidelitas.

Sekarang kita memiliki semua bagian yang diperlukan untuk menghitung pembaruan. Ini sedikit antiklimaks dan kita memiliki pembaruan sederhana dalam bentuk

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Meninjau desain dari Adam, inspirasinya menjadi jelas. Momentum dan skala terlihat jelas dalam variabel status. Definisi mereka yang cukup unik memaksa kita untuk menghilangkan bias (ini bisa diatasi dengan sedikit mengubah inisialisasi dan kondisi pembaruan). Kedua, kombinasi dari kedua istilah cukup lugas, mengingat RMSProp. Terakhir, learning rate $\eta$ eksplisit memungkinkan kita untuk mengontrol panjang langkah guna mengatasi masalah konvergensi.

## Implementasi

Mengimplementasikan Adam dari awal tidak terlalu sulit. Untuk kenyamanan, kita menyimpan penghitung langkah waktu $t$ dalam dictionary `hyperparams`. Selain itu, semuanya cukup sederhana.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Kita siap menggunakan Adam untuk melatih model. Kita akan menggunakan learning rate $\eta = 0.01$.


```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Implementasi yang lebih ringkas cukup mudah dilakukan karena `adam` adalah salah satu algoritma yang disediakan sebagai bagian dari pustaka optimasi `trainer` di Gluon. Oleh karena itu, kita hanya perlu memberikan parameter konfigurasi untuk implementasi di Gluon.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Salah satu masalah dengan Adam adalah bahwa algoritma ini dapat gagal untuk konvergen bahkan dalam kondisi konveks ketika estimasi momen kedua pada $\mathbf{s}_t$ meningkat drastis. Sebagai solusinya, :citet:`Zaheer.Reddi.Sachan.ea.2018` mengusulkan pembaruan (dan inisialisasi) yang lebih halus untuk $\mathbf{s}_t$. Untuk memahami lebih lanjut, mari kita tuliskan ulang pembaruan Adam sebagai berikut:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Setiap kali $\mathbf{g}_t^2$ memiliki varians tinggi atau ketika pembaruan jarang terjadi, $\mathbf{s}_t$ mungkin akan melupakan nilai-nilai masa lalu terlalu cepat. Salah satu solusi potensial untuk masalah ini adalah menggantikan $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ dengan $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Dengan demikian, besarnya pembaruan tidak lagi bergantung pada besar kecilnya deviasi. Hal ini menghasilkan pembaruan Yogi:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Para penulis juga menyarankan untuk menginisialisasi momentum pada batch awal yang lebih besar daripada hanya estimasi titik awal. Kami mengabaikan detail lebih lanjut karena tidak terlalu material dalam pembahasan ini dan karena, bahkan tanpa inisialisasi ini, konvergensi tetap cukup baik.


```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```


## Ringkasan

* Adam menggabungkan fitur dari banyak algoritma optimisasi menjadi sebuah aturan pembaruan yang cukup andal.
* Didasarkan pada RMSProp, Adam juga menggunakan EWMA pada gradien stokastik minibatch.
* Adam menggunakan koreksi bias untuk mengatasi startup yang lambat ketika mengestimasi momentum dan momen kedua.
* Untuk gradien dengan varians signifikan, kita mungkin menemui masalah konvergensi. Hal ini dapat diatasi dengan menggunakan minibatch yang lebih besar atau beralih ke estimasi yang lebih baik untuk $\mathbf{s}_t$. Yogi menawarkan alternatif tersebut.

## Latihan

1. Sesuaikan learning rate dan amati serta analisis hasil eksperimen.
2. Bisakah Anda menulis ulang pembaruan momentum dan momen kedua sehingga tidak memerlukan koreksi bias?
3. Mengapa Anda perlu mengurangi learning rate $\eta$ saat kita mendekati konvergensi?
4. Coba buat sebuah kasus di mana Adam mengalami divergensi dan Yogi tetap konvergen.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1079)
:end_tab:

