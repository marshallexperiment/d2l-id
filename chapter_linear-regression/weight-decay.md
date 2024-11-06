```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Regularisasi Bobot (Weight Decay)
:label:`sec_weight_decay`

Setelah kita memahami permasalahan *overfitting*,
kita bisa memperkenalkan teknik *regularisasi* pertama kita.
Ingat bahwa kita selalu dapat mengurangi *overfitting*
dengan mengumpulkan lebih banyak data pelatihan.
Namun, hal itu bisa mahal, memakan waktu,
atau bahkan di luar kendali kita,
sehingga tidak mungkin dilakukan dalam waktu singkat.
Untuk saat ini, kita bisa menganggap bahwa kita sudah memiliki
data berkualitas tinggi sebanyak mungkin yang diizinkan oleh sumber daya kita
dan fokus pada alat yang ada di tangan
ketika dataset yang ada telah ditetapkan.

Ingat kembali bahwa dalam contoh regresi polinomial kita
(:numref:`subsec_polynomial-curve-fitting`)
kita dapat membatasi kapasitas model kita
dengan menyesuaikan derajat polinomial yang dipasang.
Memang, membatasi jumlah fitur
adalah teknik populer untuk mengurangi *overfitting*.
Namun, membuang fitur begitu saja
bisa menjadi instrumen yang terlalu kasar.
Mengikuti contoh regresi polinomial,
pertimbangkan apa yang mungkin terjadi
pada masukan berdimensi tinggi.
Perluasan alami dari polinomial
ke data multivariat disebut *monomial*,
yang merupakan produk dari pangkat variabel.
Derajat monomial adalah jumlah pangkatnya.
Sebagai contoh, $x_1^2 x_2$, dan $x_3 x_5^2$
keduanya adalah monomial berderajat 3.

Perlu diperhatikan bahwa jumlah suku dengan derajat $d$
akan meningkat drastis seiring bertambahnya $d$.
Diberikan $k$ variabel, jumlah monomial
dengan derajat $d$ adalah ${k - 1 + d} \choose {k - 1}$.
Bahkan perubahan kecil dalam derajat, misalnya dari $2$ ke $3$,
dapat secara dramatis meningkatkan kompleksitas model kita.
Oleh karena itu, kita sering membutuhkan alat yang lebih halus
untuk menyesuaikan kompleksitas fungsi.


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
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
import optax
```

## Norma dan Regularisasi Bobot (Weight Decay)

(**Daripada langsung memanipulasi jumlah parameter,
*regularisasi bobot* (weight decay) bekerja dengan membatasi nilai
yang bisa diambil oleh parameter-parameter.**)
Di luar lingkup deep learning,
teknik ini lebih dikenal sebagai regularisasi $\ell_2$
dan ketika dioptimalkan menggunakan *stochastic gradient descent* minibatch,
regularisasi bobot mungkin merupakan teknik paling umum
untuk mengatur model pembelajaran mesin parametrik.
Motivasi di balik teknik ini didasari oleh intuisi dasar
bahwa di antara semua fungsi $f$,
fungsi $f = 0$ (yang memberikan nilai $0$ untuk semua input)
dalam beberapa hal adalah fungsi yang *paling sederhana*,
dan kita bisa mengukur kompleksitas
dari suatu fungsi dengan jarak parameternya dari nol.
Namun, bagaimana tepatnya kita harus mengukur
jarak antara suatu fungsi dengan nol?
Tidak ada satu jawaban yang benar.
Faktanya, cabang matematika tertentu,
seperti analisis fungsional
dan teori ruang Banach,
didedikasikan untuk membahas pertanyaan-pertanyaan seperti ini.

Salah satu interpretasi sederhana
adalah mengukur kompleksitas fungsi linear
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
dengan norma vektor bobotnya, misalnya $\| \mathbf{w} \|^2$.
Ingat bahwa kita telah memperkenalkan norma $\ell_2$ dan norma $\ell_1$,
yang merupakan kasus khusus dari norma $\ell_p$ yang lebih umum,
di :numref:`subsec_lin-algebra-norms`.
Metode paling umum untuk memastikan vektor bobot yang kecil
adalah dengan menambahkan normanya sebagai istilah penalti
ke dalam masalah minimisasi loss.
Dengan demikian, kita mengganti tujuan awal kita,
*mengurangi loss prediksi pada label pelatihan*,
dengan tujuan baru,
*mengurangi jumlah loss prediksi dan istilah penalti*.
Sekarang, jika vektor bobot kita terlalu besar,
algoritma pembelajaran kita mungkin lebih berfokus
pada meminimalkan norma bobot $\| \mathbf{w} \|^2$
daripada meminimalkan kesalahan pelatihan.
Itulah yang kita inginkan.
Untuk menggambarkan hal ini dalam kode,
kita menghidupkan kembali contoh sebelumnya
dari :numref:`sec_linear_regression` untuk regresi linear.
Di sana, loss kita diberikan oleh

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Ingat bahwa $\mathbf{x}^{(i)}$ adalah fitur,
$y^{(i)}$ adalah label untuk contoh data $i$, dan $(\mathbf{w}, b)$
masing-masing adalah parameter bobot dan bias.
Untuk memberikan penalti pada ukuran vektor bobot,
kita harus menambahkan $\| \mathbf{w} \|^2$ ke fungsi loss,
tetapi bagaimana model seharusnya menyeimbangkan
antara loss standar dengan penalti tambahan ini?
Dalam praktiknya, kita menyesuaikan keseimbangan ini
melalui *konstanta regularisasi* $\lambda$,
sebuah hyperparameter non-negatif
yang kita sesuaikan menggunakan data validasi:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$

Untuk $\lambda = 0$, kita mendapatkan kembali fungsi loss asli.
Untuk $\lambda > 0$, kita membatasi ukuran $\| \mathbf{w} \|$.
Kita membagi dengan $2$ sebagai konvensi:
ketika kita mengambil turunan dari fungsi kuadrat,
$2$ dan $1/2$ saling membatalkan, memastikan bahwa ekspresi
untuk pembaruan terlihat rapi dan sederhana.
Pembaca yang cermat mungkin bertanya mengapa kita menggunakan norma kuadrat
dan bukan norma standar (yaitu, jarak Euclidean).
Kita melakukannya demi kemudahan komputasi.
Dengan mengkuadratkan norma $\ell_2$, kita menghilangkan akar kuadrat,
sehingga tersisa jumlah kuadrat dari
masing-masing komponen vektor bobot.
Ini membuat turunan dari penalti mudah dihitung:
jumlah turunan sama dengan turunan dari jumlah.

Lebih lanjut, Anda mungkin bertanya mengapa kita menggunakan norma $\ell_2$
daripada, misalnya, norma $\ell_1$.
Faktanya, pilihan lain juga valid dan
populer di bidang statistik.
Sementara model linear yang diregularisasi $\ell_2$
mewakili algoritma *ridge regression* klasik,
regresi linear yang diregularisasi $\ell_1$
adalah metode fundamental dalam statistik,
dikenal sebagai *lasso regression*.
Salah satu alasan menggunakan norma $\ell_2$
adalah karena ia memberikan penalti besar
pada komponen besar dari vektor bobot.
Ini mendorong algoritma pembelajaran kita
untuk memilih model yang mendistribusikan bobot secara merata
di antara banyak fitur.
Dalam praktiknya, hal ini dapat membuat model lebih tahan
terhadap kesalahan pengukuran dalam satu variabel.
Sebaliknya, penalti $\ell_1$ menghasilkan model
yang memusatkan bobot pada sejumlah kecil fitur
dengan menyetel bobot lainnya ke nol.
Ini memberi kita metode efektif untuk *seleksi fitur*,
yang mungkin diinginkan karena alasan lain.
Misalnya, jika model kita hanya bergantung pada beberapa fitur,
kita mungkin tidak perlu mengumpulkan, menyimpan, atau mentransmisikan data
untuk fitur lainnya (yang dibuang).

Dengan notasi yang sama pada :eqref:`eq_linreg_batch_update`,
pembaruan *stochastic gradient descent* minibatch
untuk regresi yang diregularisasi $\ell_2$ adalah sebagai berikut:

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Seperti sebelumnya, kita memperbarui $\mathbf{w}$ berdasarkan seberapa jauh
perkiraan kita dari pengamatan.
Namun, kita juga mengurangi ukuran $\mathbf{w}$ menuju nol.
Itulah mengapa metode ini kadang disebut "weight decay":
dengan hanya mempertimbangkan istilah penalti,
algoritma optimisasi kita *menurunkan*
bobot di setiap langkah pelatihan.
Berbeda dengan seleksi fitur,
weight decay menawarkan mekanisme untuk terus menyesuaikan kompleksitas fungsi.
Nilai $\lambda$ yang lebih kecil menunjukkan
$\mathbf{w}$ yang lebih longgar,
sedangkan nilai $\lambda$ yang lebih besar
lebih membatasi $\mathbf{w}$.
Apakah kita menyertakan penalti bias $b^2$
dapat bervariasi antar implementasi,
dan mungkin berbeda antar lapisan dalam jaringan neural.
Seringkali, kita tidak melakukan regularisasi pada parameter bias.
Selain itu,
meskipun regularisasi $\ell_2$ mungkin tidak setara dengan weight decay pada algoritma optimisasi lainnya,
gagasan regularisasi melalui
pengecilan ukuran bobot
tetap berlaku.


## Regresi Linear Berdimensi Tinggi

Kita bisa menggambarkan manfaat dari regularisasi bobot
melalui contoh sintetik sederhana.

Pertama, kita [**menghasilkan beberapa data seperti sebelumnya**]:

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ di mana }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Dalam dataset sintetik ini, label kita ditentukan
oleh fungsi linear dasar dari input kita,
yang terganggu oleh noise Gaussian
dengan mean nol dan deviasi standar 0.01.
Untuk tujuan ilustratif,
kita dapat membuat efek dari overfitting lebih jelas
dengan meningkatkan dimensi masalah kita menjadi $d = 200$
dan menggunakan set pelatihan kecil dengan hanya 20 contoh data.


```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Implementasi dari Awal

Sekarang, mari kita coba mengimplementasikan regularisasi bobot dari awal.
Karena *stochastic gradient descent* minibatch adalah optimisasi kita,
kita hanya perlu menambahkan penalti kuadrat $\ell_2$
ke fungsi loss yang asli.

### (**Mendefinisikan Penalti Norma $\ell_2$**)

Mungkin cara yang paling mudah untuk mengimplementasikan penalti ini
adalah dengan mengkuadratkan semua elemen pada tempatnya dan menjumlahkannya.


```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Mendefinisikan Model

Pada model akhir,
regresi linear dan loss kuadrat tidak berubah sejak :numref:`sec_linear_scratch`,
jadi kita hanya akan mendefinisikan sebuah subclass dari `d2l.LinearRegressionScratch`.
Perubahan satu-satunya di sini adalah bahwa loss kita sekarang mencakup istilah penalti.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

Kode berikut menyesuaikan model kita pada set pelatihan dengan 20 contoh dan mengevaluasinya pada set validasi dengan 100 contoh.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**Pelatihan tanpa Regularisasi**]

Sekarang kita menjalankan kode ini dengan `lambd = 0`,
yang menonaktifkan regularisasi bobot.
Perhatikan bahwa kita mengalami overfitting yang parah,
di mana error pada pelatihan menurun tetapi tidak pada
error validasi—ini adalah contoh textbook dari overfitting.


```{.python .input}
%%tab all
train_scratch(0)
```

### [**Menggunakan Regularisasi Bobot**]

Di bawah ini, kita menjalankan pelatihan dengan regularisasi bobot yang cukup besar.
Perhatikan bahwa error pada pelatihan meningkat
tetapi error pada validasi menurun.
Inilah efek yang kita harapkan dari regularisasi.


```{.python .input}
%%tab all
train_scratch(3)
```

## [**Implementasi Singkat**]

Karena regularisasi bobot sangat umum
dalam optimisasi jaringan saraf,
kerangka kerja deep learning membuatnya sangat nyaman,
dengan mengintegrasikan regularisasi bobot langsung ke dalam algoritma optimisasi itu sendiri,
sehingga mudah digunakan dengan berbagai fungsi loss.
Selain itu, integrasi ini memiliki keuntungan komputasional,
memungkinkan penerapan trik implementasi untuk menambahkan regularisasi bobot ke algoritma
tanpa overhead komputasional tambahan.
Karena bagian pembaruan regularisasi bobot
hanya bergantung pada nilai parameter saat ini,
optimisasi harus mengakses setiap parameter sekali saja.

:begin_tab:`mxnet`
Di bawah ini, kita menentukan
hiperparameter regularisasi bobot secara langsung
melalui `wd` saat menginstansiasi `Trainer`.
Secara default, Gluon melakukan regularisasi
baik pada bobot maupun bias secara bersamaan.
Perhatikan bahwa hiperparameter `wd`
akan dikalikan dengan `wd_mult`
saat memperbarui parameter model.
Jadi, jika kita mengatur `wd_mult` ke nol,
parameter bias $b$ tidak akan mengalami regularisasi.
:end_tab:

:begin_tab:`pytorch`
Di bawah ini, kita menentukan
hiperparameter regularisasi bobot secara langsung
melalui `weight_decay` saat menginstansiasi optimisasi.
Secara default, PyTorch melakukan regularisasi
baik pada bobot maupun bias secara bersamaan, namun
kita dapat mengkonfigurasi optimisasi untuk menangani parameter berbeda
dengan kebijakan yang berbeda.
Di sini, kita hanya menetapkan `weight_decay`
untuk bobot (`net.weight`), sehingga
bias (`net.bias`) tidak akan mengalami regularisasi.
:end_tab:

:begin_tab:`tensorflow`
Di bawah ini, kita membuat regularizer $\ell_2$ dengan
hiperparameter regularisasi bobot `wd` dan menerapkannya pada bobot lapisan
melalui argumen `kernel_regularizer`.
:end_tab:


```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**Plot ini terlihat mirip dengan plot ketika
kita mengimplementasikan regularisasi bobot dari awal**].
Namun, versi ini berjalan lebih cepat
dan lebih mudah diimplementasikan,
keuntungan yang akan menjadi lebih
nyata saat Anda mengatasi masalah yang lebih besar
dan pekerjaan ini menjadi lebih rutin.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('Norma L2 dari w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('Norma L2 dari w:', float(l2_penalty(model.get_w_b()[0])))
```

Sejauh ini, kita telah membahas satu konsep tentang
apa yang dianggap sebagai fungsi linear yang sederhana.
Namun, bahkan untuk fungsi non-linear sederhana, situasinya bisa jauh lebih kompleks. Untuk melihat hal ini, konsep [ruang Hilbert kernel reproduksi (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
memungkinkan penerapan alat-alat yang diperkenalkan
untuk fungsi linear dalam konteks non-linear.
Sayangnya, algoritme berbasis RKHS
cenderung tidak efisien untuk data besar yang berdimensi tinggi.
Dalam buku ini, kita akan sering mengadopsi heuristik umum
di mana weight decay diterapkan
ke semua lapisan jaringan dalam.

## Ringkasan

Regularisasi adalah metode umum untuk menangani overfitting. Teknik regularisasi klasik menambahkan istilah penalti pada fungsi loss (saat pelatihan) untuk mengurangi kompleksitas model yang dipelajari.
Salah satu pilihan khusus untuk menjaga model tetap sederhana adalah dengan menggunakan penalti $\ell_2$. Ini menghasilkan weight decay pada langkah pembaruan dari algoritme *stochastic gradient descent* minibatch.
Dalam praktiknya, fungsi weight decay disediakan dalam optimizer dari framework pembelajaran mendalam.
Set parameter yang berbeda dapat memiliki perilaku pembaruan yang berbeda dalam loop pelatihan yang sama.


## Latihan

1. Bereksperimenlah dengan nilai $\lambda$ dalam masalah estimasi di bagian ini. Buatlah plot akurasi pelatihan dan validasi sebagai fungsi dari $\lambda$. Apa yang Anda amati?
2. Gunakan set validasi untuk menemukan nilai optimal dari $\lambda$. Apakah ini benar-benar nilai optimal? Apakah ini penting?
3. Bagaimana bentuk persamaan pembaruan jika, alih-alih $\|\mathbf{w}\|^2$, kita menggunakan $\sum_i |w_i|$ sebagai penalti yang dipilih (regularisasi $\ell_1$)?
4. Kita tahu bahwa $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Bisakah Anda menemukan persamaan serupa untuk matriks (lihat norma Frobenius di :numref:`subsec_lin-algebra-norms`)?
5. Tinjau hubungan antara error pelatihan dan error generalisasi. Selain weight decay, pelatihan yang lebih lama, dan penggunaan model dengan kompleksitas yang sesuai, metode apa lagi yang dapat membantu kita menangani overfitting?
6. Dalam statistik Bayesian, kita menggunakan produk prior dan likelihood untuk mendapatkan posterior melalui $P(w \mid x) \propto P(x \mid w) P(w)$. Bagaimana Anda dapat mengidentifikasi $P(w)$ dengan regularisasi?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17979)
:end_tab:
