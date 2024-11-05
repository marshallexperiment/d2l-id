```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Implementasi Regresi Linear dari Awal
:label:`sec_linear_scratch`

Sekarang kita siap untuk bekerja melalui
implementasi penuh dari regresi linear.
Di bagian ini, 
(**kita akan mengimplementasikan seluruh metode dari awal,
termasuk (i) model; (ii) fungsi kerugian;
(iii) optimizer minibatch stochastic gradient descent;
dan (iv) fungsi pelatihan 
yang menghubungkan semua bagian ini.**)
Akhirnya, kita akan menjalankan pembangkit data sintetis kita
dari :numref:`sec_synthetic-regression-data`
dan menerapkan model kita
pada dataset yang dihasilkan.
Meskipun framework pembelajaran mendalam modern
dapat mengotomatiskan hampir semua pekerjaan ini,
mengimplementasikan semuanya dari awal adalah satu-satunya cara
untuk memastikan bahwa Anda benar-benar tahu apa yang Anda lakukan.
Selain itu, ketika saatnya tiba untuk menyesuaikan model,
mendefinisikan layer atau fungsi kerugian sendiri,
memahami cara kerja di balik layar akan sangat membantu.
Di bagian ini, kita hanya akan mengandalkan 
tensor dan diferensiasi otomatis.
Nantinya, kita akan memperkenalkan implementasi yang lebih ringkas,
memanfaatkan fitur tambahan dari framework pembelajaran mendalam 
sambil mempertahankan struktur yang akan kita kembangkan di bawah ini.


```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Mendefinisikan Model

[**Sebelum kita dapat mulai mengoptimalkan parameter model**] dengan minibatch SGD,
(**kita perlu memiliki beberapa parameter terlebih dahulu.**)
Pada langkah berikut, kita menginisialisasi bobot dengan mengambil
angka acak dari distribusi normal dengan mean 0
dan simpangan baku 0.01.
Angka 0.01 seringkali bekerja dengan baik dalam praktik,
tetapi Anda dapat menentukan nilai yang berbeda
melalui argumen `sigma`.
Selain itu, kita menetapkan bias ke 0.
Perhatikan bahwa untuk desain berbasis objek
kita menambahkan kode ke metode `__init__` dari subclass `d2l.Module` (diperkenalkan di :numref:`subsec_oo-design-models`).


```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """Model regresi linear yang diimplementasikan dari awal."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """Model regresi linear yang diimplementasikan dari awal."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

Selanjutnya kita harus [**mendefinisikan model kita,
menghubungkan input dan parameternya dengan output.**]
Dengan menggunakan notasi yang sama seperti pada :eqref:`eq_linreg-y-vec`
untuk model linear kita, kita cukup mengambil hasil perkalian matriks-vektor
dari fitur input $\mathbf{X}$
dan bobot model $\mathbf{w}$,
dan menambahkan offset $b$ ke setiap contoh.
Hasil kali $\mathbf{Xw}$ adalah sebuah vektor dan $b$ adalah skalar.
Karena mekanisme broadcasting
(lihat :numref:`subsec_broadcasting`),
ketika kita menambahkan vektor dan skalar,
skalar tersebut ditambahkan ke setiap komponen vektor.
Metode `forward` yang dihasilkan
didaftarkan dalam kelas `LinearRegressionScratch`
melalui `add_to_class` (diperkenalkan di :numref:`oo-design-utilities`).


```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Mendefinisikan Fungsi Kerugian

Karena [**memperbarui model kita memerlukan pengambilan
gradien dari fungsi kerugian,**]
kita perlu (**mendefinisikan fungsi kerugian terlebih dahulu.**)
Di sini kita menggunakan fungsi kerugian kuadrat
pada :eqref:`eq_mse`.
Dalam implementasinya, kita perlu mengubah nilai sebenarnya `y`
menjadi bentuk yang sama dengan nilai prediksi `y_hat`.
Hasil yang dikembalikan oleh metode berikut
juga akan memiliki bentuk yang sama dengan `y_hat`.
Kita juga mengembalikan nilai kerugian rata-rata
di antara semua contoh dalam minibatch.


```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Mendefinisikan Algoritma Optimasi

Seperti yang dibahas pada :numref:`sec_linear_regression`,
regresi linear memiliki solusi dalam bentuk tertutup.
Namun, tujuan kita di sini adalah untuk mengilustrasikan 
cara melatih jaringan neural yang lebih umum,
yang memerlukan kita untuk mempelajari 
cara menggunakan minibatch SGD.
Oleh karena itu, kita akan menggunakan kesempatan ini
untuk memperkenalkan contoh pertama Anda tentang SGD yang berfungsi.
Pada setiap langkah, dengan menggunakan minibatch 
yang diambil secara acak dari dataset kita,
kita memperkirakan gradien dari kerugian
terhadap parameter.
Selanjutnya, kita memperbarui parameter
ke arah yang mungkin mengurangi kerugian.

Kode berikut menerapkan pembaruan, 
diberikan satu set parameter dan learning rate `lr`.
Karena kerugian kita dihitung sebagai rata-rata pada minibatch, 
kita tidak perlu menyesuaikan learning rate terhadap ukuran batch. 
Di bab-bab berikutnya, kita akan menyelidiki 
bagaimana learning rate harus disesuaikan
untuk minibatch yang sangat besar 
dalam pembelajaran skala besar terdistribusi.
Untuk saat ini, kita dapat mengabaikan ketergantungan ini.


:begin_tab:`mxnet`
Kita mendefinisikan kelas `SGD` kita, 
sebuah subclass dari `d2l.HyperParameters` (diperkenalkan di :numref:`oo-design-utilities`),
agar memiliki API yang mirip
dengan optimizer SGD bawaan.
Kita memperbarui parameter dalam metode `step`.
Metode ini menerima argumen `batch_size` yang dapat diabaikan.
:end_tab:

:begin_tab:`pytorch`
Kita mendefinisikan kelas `SGD` kita,
sebuah subclass dari `d2l.HyperParameters` (diperkenalkan di :numref:`oo-design-utilities`),
agar memiliki API yang mirip
dengan optimizer SGD bawaan.
Kita memperbarui parameter dalam metode `step`.
Metode `zero_grad` mengatur semua gradien menjadi 0,
yang harus dijalankan sebelum langkah backpropagation.
:end_tab:

:begin_tab:`tensorflow`
Kita mendefinisikan kelas `SGD` kita,
sebuah subclass dari `d2l.HyperParameters` (diperkenalkan di :numref:`oo-design-utilities`),
agar memiliki API yang mirip
dengan optimizer SGD bawaan.
Kita memperbarui parameter dalam metode `apply_gradients`.
Metode ini menerima daftar pasangan parameter dan gradien.
:end_tab:


```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Stochastic gradient descent dengan minibatch."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Stochastic gradient descent dengan minibatch."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Stochastic gradient descent dengan minibatch."""
    # Transformasi kunci dari Optax adalah GradientTransformation
    # yang didefinisikan oleh dua metode, yaitu init dan update.
    # Init menginisialisasi state, dan update mengubah gradien.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # Ketika metode state.apply_gradients dipanggil untuk memperbarui flax's
        # objek train_state, metode ini secara internal memanggil optax.apply_updates
        # menambahkan params ke persamaan update yang didefinisikan di bawah ini.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

Selanjutnya kita mendefinisikan metode `configure_optimizers`, yang mengembalikan instance dari kelas `SGD`.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Pelatihan

Sekarang setelah kita memiliki semua komponen yang diperlukan
(parameter, fungsi kerugian, model, dan optimizer),
kita siap untuk [**mengimplementasikan loop pelatihan utama.**]
Memahami kode ini sepenuhnya adalah hal yang sangat penting
karena Anda akan menggunakan loop pelatihan serupa
untuk setiap model pembelajaran mendalam lainnya
yang dibahas dalam buku ini.
Di setiap *epoch*, kita akan melakukan iterasi
melalui seluruh dataset pelatihan,
melewati setiap contoh satu kali
(dengan asumsi jumlah contoh
dapat dibagi oleh ukuran batch).
Di setiap *iterasi*, kita mengambil satu minibatch dari contoh pelatihan,
dan menghitung kerugiannya melalui metode `training_step` dari model.
Kemudian kita menghitung gradien terhadap setiap parameter.
Terakhir, kita akan memanggil algoritma optimasi
untuk memperbarui parameter model.
Singkatnya, kita akan mengeksekusi loop berikut:

* Inisialisasi parameter $(\mathbf{w}, b)$
* Ulangi sampai selesai
    * Hitung gradien $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Perbarui parameter $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
Ingat bahwa dataset regresi sintetis
yang kita hasilkan di :numref:``sec_synthetic-regression-data``
tidak menyediakan dataset validasi.
Namun, dalam sebagian besar kasus,
kita ingin memiliki dataset validasi
untuk mengukur kualitas model kita.
Di sini kita melewati dataloader validasi
satu kali di setiap epoch untuk mengukur performa model.
Mengikuti desain berbasis objek kita,
metode `prepare_batch` dan `fit_epoch`
terdaftar dalam kelas `d2l.Trainer`
(diperkenalkan di :numref:`oo-design-training`).


```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # Didiskusikan nanti
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Dapat diabaikan untuk model tanpa Lapisan Dropout
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

Kita hampir siap untuk melatih model,
tetapi pertama-tama kita membutuhkan data pelatihan.
Di sini kita menggunakan kelas `SyntheticRegressionData`
dan memasukkan beberapa parameter ground truth.
Kemudian kita melatih model kita dengan
learning rate `lr=0.03`
dan menetapkan `max_epochs=3`.
Perlu dicatat bahwa, secara umum, baik jumlah epoch
maupun learning rate adalah hyperparameter.
Menetapkan hyperparameter bisa jadi rumit,
dan kita biasanya ingin menggunakan pembagian tiga arah,
satu set untuk pelatihan,
set kedua untuk pemilihan hyperparameter,
dan yang ketiga disimpan untuk evaluasi akhir.
Kita abaikan detail ini untuk saat ini, tetapi akan kita revisi
di bagian selanjutnya.


```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Karena kita sendiri yang mensintesis dataset,
kita tahu persis apa saja parameter sebenarnya.
Oleh karena itu, kita dapat [**mengevaluasi keberhasilan pelatihan kita
dengan membandingkan parameter sebenarnya
dengan parameter yang kita pelajari**] melalui loop pelatihan kita.
Hasilnya, parameter yang dipelajari ternyata sangat mendekati parameter sebenarnya.


```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'kesalahan dalam estimasi w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'kesalahan dalam estimasi b: {data.b - model.b}')

```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'kesalahan dalam mengestimasi w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'kesalahan dalam mengestimasi b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"kesalahan dalam mengestimasi w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"kesalahan dalam mengestimasi b: {data.b - params['b']}")
```


Kita tidak boleh menerima begitu saja kemampuan
untuk tepat mengembalikan parameter ground truth.
Secara umum, untuk model dalam, solusi unik
untuk parameter tidak selalu ada,
dan bahkan untuk model linear,
mengembalikan parameter secara tepat
hanya mungkin jika tidak ada fitur
yang secara linear bergantung pada fitur lainnya.
Namun, dalam pembelajaran mesin,
kita sering kurang peduli
dengan pemulihan parameter dasar yang sebenarnya,
tetapi lebih peduli dengan parameter
yang menghasilkan prediksi yang sangat akurat :cite:`Vapnik.1992`.
Untungnya, bahkan pada masalah optimasi yang sulit,
stochastic gradient descent sering kali dapat menemukan solusi yang sangat baik,
sebagian karena fakta bahwa, untuk jaringan dalam,
terdapat banyak konfigurasi parameter
yang menghasilkan prediksi yang sangat akurat.


## Ringkasan

Di bagian ini, kita mengambil langkah signifikan
menuju perancangan sistem pembelajaran mendalam
dengan mengimplementasikan model jaringan neural
dan loop pelatihan yang sepenuhnya fungsional.
Dalam proses ini, kita membangun pemuat data,
model, fungsi kerugian, prosedur optimasi,
dan alat visualisasi serta pemantauan.
Kita melakukannya dengan menyusun objek Python
yang berisi semua komponen relevan untuk melatih model.
Meskipun ini belum merupakan implementasi tingkat profesional,
implementasi ini sudah sangat fungsional dan kode seperti ini
dapat membantu Anda menyelesaikan masalah kecil dengan cepat.
Di bagian selanjutnya, kita akan melihat cara melakukannya
dengan *lebih ringkas* (menghindari kode berulang)
dan *lebih efisien* (memanfaatkan GPU kita secara maksimal).


## Latihan

1. Apa yang akan terjadi jika kita menginisialisasi bobot ke nol? Apakah algoritma masih akan bekerja? Bagaimana jika kita
   menginisialisasi parameter dengan varians $1000$ daripada $0.01$?
1. Asumsikan bahwa Anda adalah [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) yang mencoba mengembangkan
   model untuk resistansi yang berhubungan dengan tegangan dan arus. Bisakah Anda menggunakan diferensiasi otomatis
   untuk mempelajari parameter model Anda?
1. Bisakah Anda menggunakan [Hukum Planck](https://en.wikipedia.org/wiki/Planck%27s_law) untuk menentukan suhu suatu objek
   menggunakan kerapatan energi spektral? Sebagai referensi, kerapatan spektral $B$ dari radiasi yang dipancarkan oleh benda hitam adalah
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Di sini
   $\lambda$ adalah panjang gelombang, $T$ adalah suhu, $c$ adalah kecepatan cahaya, $h$ adalah konstanta Planck, dan $k$ adalah
   konstanta Boltzmann. Anda mengukur energi untuk berbagai panjang gelombang $\lambda$ dan Anda sekarang perlu menyesuaikan kurva kerapatan
   spektral ke hukum Planck.
1. Apa saja masalah yang mungkin Anda temui jika Anda ingin menghitung turunan kedua dari kerugian? Bagaimana cara Anda memperbaikinya?
1. Mengapa metode `reshape` diperlukan dalam fungsi `loss`?
1. Eksperimen dengan menggunakan learning rate yang berbeda untuk melihat seberapa cepat nilai fungsi kerugian menurun. Bisakah Anda mengurangi
   kesalahan dengan meningkatkan jumlah epoch pelatihan?
1. Jika jumlah contoh tidak dapat dibagi oleh ukuran batch, apa yang terjadi pada `data_iter` di akhir suatu epoch?
1. Cobalah mengimplementasikan fungsi kerugian yang berbeda, seperti kerugian nilai absolut `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Lihat apa yang terjadi untuk data biasa.
    1. Periksa apakah ada perbedaan perilaku jika Anda secara aktif mengganggu beberapa entri, seperti $y_5 = 10000$, dari $\mathbf{y}$.
    1. Bisakah Anda memikirkan solusi murah untuk menggabungkan aspek terbaik dari kerugian kuadrat dan kerugian nilai absolut?
       Petunjuk: bagaimana Anda dapat menghindari nilai gradien yang sangat besar?
1. Mengapa kita perlu mengacak ulang dataset? Bisakah Anda merancang kasus di mana dataset yang disusun secara tidak baik dapat merusak algoritma optimasi?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17976)
:end_tab:
