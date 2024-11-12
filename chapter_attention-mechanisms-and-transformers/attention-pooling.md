# Attention Pooling dengan Similaritas

:label:`sec_attention-pooling`

Sekarang setelah kita memperkenalkan komponen utama dari mekanisme perhatian (attention mechanism), mari kita menggunakannya dalam pengaturan yang lebih klasik, yaitu regresi dan klasifikasi melalui estimasi kepadatan kernel (*kernel density estimation*) :cite:`Nadaraya.1964,Watson.1964`. Pembahasan ini hanya merupakan penjelasan tambahan: sepenuhnya opsional dan dapat dilewatkan jika diinginkan.
Pada intinya, estimator Nadaraya--Watson mengandalkan beberapa kernel similaritas $\alpha(\mathbf{q}, \mathbf{k})$ yang menghubungkan query $\mathbf{q}$ dengan kunci $\mathbf{k}$. Beberapa kernel umum adalah:

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ jika } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

Ada banyak pilihan lain yang bisa kita ambil. Lihat artikel [Wikipedia](https://en.wikipedia.org/wiki/Kernel_(statistics)) untuk ulasan yang lebih luas dan bagaimana pemilihan kernel terkait dengan estimasi kepadatan kernel, yang kadang-kadang disebut juga sebagai *Parzen Windows* :cite:`parzen1957consistent`. Semua kernel ini adalah heuristik dan dapat disesuaikan. Sebagai contoh, kita dapat menyesuaikan lebar kernel, tidak hanya secara global tetapi bahkan per koordinat. Namun demikian, semua kernel tersebut mengarah ke persamaan berikut untuk regresi dan klasifikasi:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

Dalam kasus regresi (skalar) dengan pengamatan $(\mathbf{x}_i, y_i)$ untuk fitur dan label masing-masing, $\mathbf{v}_i = y_i$ adalah skalar, $\mathbf{k}_i = \mathbf{x}_i$ adalah vektor, dan query $\mathbf{q}$ menunjukkan lokasi baru di mana $f$ harus dievaluasi. Dalam kasus klasifikasi (multikelas), kita menggunakan *one-hot encoding* dari $y_i$ untuk mendapatkan $\mathbf{v}_i$. Salah satu sifat yang nyaman dari estimator ini adalah bahwa ia tidak memerlukan pelatihan. Bahkan lebih baik lagi, jika kita mempersempit kernel dengan tepat saat jumlah data meningkat, pendekatan ini akan konsisten :cite:`mack1982weak`, yaitu akan konvergen ke beberapa solusi yang optimal secara statistik. Mari kita mulai dengan memeriksa beberapa kernel.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**Kernel dan Data**]

Semua kernel $\alpha(\mathbf{k}, \mathbf{q})$ yang didefinisikan dalam bagian ini bersifat *invarian terhadap translasi dan rotasi*; artinya, jika kita menggeser dan memutar $\mathbf{k}$ dan $\mathbf{q}$ dengan 
cara yang sama, nilai $\alpha$ tetap tidak berubah. Untuk kesederhanaan, kita memilih argumen skalar $k, q \in \mathbb{R}$ dan memilih kunci $k = 0$ sebagai titik asal. Ini menghasilkan:


```{.python .input}
%%tab all
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

Kernel yang berbeda sesuai dengan konsep rentang dan kehalusan yang berbeda. Sebagai contoh, kernel boxcar hanya memperhatikan observasi dalam jarak $1$ (atau beberapa hiperparameter yang didefinisikan lainnya) dan melakukannya secara tidak diskriminatif.

Untuk melihat estimasi Nadaraya--Watson dalam tindakan, mari kita definisikan beberapa data pelatihan. Dalam contoh berikut, kita menggunakan ketergantungan

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

di mana $\epsilon$ diambil dari distribusi normal dengan mean nol dan varians satu. Kita mengambil 40 contoh pelatihan.


```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Pooling Attention melalui Regresi Nadaraya--Watson**]

Sekarang kita memiliki data dan kernel, yang kita butuhkan hanyalah sebuah fungsi yang menghitung estimasi regresi kernel. Perhatikan bahwa kita juga ingin memperoleh bobot kernel relatif untuk melakukan beberapa diagnostik kecil. Oleh karena itu, pertama-tama kita menghitung kernel antara semua fitur pelatihan (`x_train`) dan semua fitur validasi (`x_val`). Ini menghasilkan matriks, yang kemudian kita normalisasi. Ketika dikalikan dengan label pelatihan (`y_train`), kita akan mendapatkan estimasi.

Ingat kembali pooling perhatian pada :eqref:`eq_attention_pooling`. Biarkan setiap fitur validasi menjadi query, dan setiap pasangan fitur-label pelatihan menjadi pasangan kunci--nilai. Sebagai hasilnya, bobot kernel relatif yang telah dinormalisasi (`attention_w` di bawah) adalah *bobot perhatian*.


```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Setiap kolom/baris berhubungan dengan setiap query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalisasi atas key untuk setiap query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

Mari kita lihat jenis estimasi yang dihasilkan oleh kernel yang berbeda.


```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Hal pertama yang menonjol adalah bahwa ketiga kernel non-trivial (Gaussian, Boxcar, dan Epanechikov) menghasilkan estimasi yang cukup dapat digunakan dan tidak terlalu jauh dari fungsi sebenarnya. Hanya kernel 
konstan yang menghasilkan estimasi trivial $f(x) = \frac{1}{n} \sum_i y_i$ yang memberikan hasil yang agak tidak realistis. Mari kita inspeksi bobot perhatian (attention weighting) ini dengan lebih dekat:


```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Visualisasi dengan jelas menunjukkan mengapa estimasi untuk Gaussian, Boxcar, dan Epanechikov sangat mirip: bagaimanapun, estimasi ini diperoleh dari bobot perhatian yang sangat mirip, meskipun bentuk fungsional dari kernel berbeda. Hal ini menimbulkan pertanyaan apakah ini selalu terjadi.

## [**Menyesuaikan Attention Pooling**]

Kita dapat mengganti kernel Gaussian dengan kernel yang memiliki lebar berbeda. Artinya, kita dapat menggunakan 
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$ di mana $\sigma^2$ menentukan lebar kernel. Mari kita lihat apakah hal ini mempengaruhi hasil yang diperoleh.


```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

Jelas bahwa semakin sempit kernel, semakin kurang halus estimasi yang diperoleh. Pada saat yang sama, estimasi ini menjadi lebih adaptif terhadap variasi lokal. Mari kita lihat bobot perhatian yang sesuai.


```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

Seperti yang kita duga, semakin sempit kernel, semakin sempit pula rentang bobot perhatian yang besar. Hal ini juga jelas menunjukkan bahwa menggunakan lebar yang sama mungkin tidak ideal. Faktanya, :citet:`Silverman86` mengusulkan sebuah heuristik yang bergantung pada kepadatan lokal. Banyak "trik" lain yang telah diusulkan. Sebagai contoh, :citet:`norelli2022asif` menggunakan teknik interpolasi tetangga terdekat yang serupa untuk merancang representasi silang-modal gambar dan teks.

Pembaca yang jeli mungkin bertanya-tanya mengapa kita memberikan pembahasan mendalam untuk metode yang sudah ada lebih dari setengah abad. Pertama, ini adalah salah satu pendahulu paling awal dari mekanisme perhatian modern. Kedua, metode ini sangat baik untuk visualisasi. Ketiga, dan sama pentingnya, ini menunjukkan batasan dari mekanisme perhatian yang dibuat secara manual. Strategi yang jauh lebih baik adalah dengan *mempelajari* mekanisme tersebut, dengan mempelajari representasi untuk query dan key. Inilah yang akan kita lakukan pada bagian berikutnya.


## Ringkasan

Regresi kernel Nadaraya--Watson adalah salah satu pendahulu paling awal dari mekanisme perhatian saat ini.
Metode ini dapat digunakan langsung dengan sedikit atau tanpa pelatihan atau penyesuaian, baik untuk klasifikasi maupun regresi.
Bobot perhatian diberikan berdasarkan kesamaan (atau jarak) antara query dan key, serta berdasarkan berapa banyak pengamatan yang serupa tersedia.


## Latihan

1. Estimasi densitas Parzen windows diberikan oleh $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$. Buktikan bahwa untuk klasifikasi biner, fungsi $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$, seperti yang diperoleh oleh Parzen windows, setara dengan klasifikasi Nadaraya--Watson.
2. Implementasikan stochastic gradient descent untuk mempelajari nilai yang baik untuk lebar kernel dalam regresi Nadaraya--Watson.
   1. Apa yang terjadi jika Anda hanya menggunakan estimasi di atas untuk meminimalkan $(f(\mathbf{x_i}) - y_i)^2$ secara langsung? Petunjuk: $y_i$ adalah bagian dari term yang digunakan untuk menghitung $f$.
   2. Hapus pasangan $(\mathbf{x}_i, y_i)$ dari estimasi $f(\mathbf{x}_i)$ dan optimalkan lebar kernel. Apakah Anda masih mengamati overfitting?
3. Asumsikan bahwa semua $\mathbf{x}$ berada di bola satuan, yaitu semua memenuhi $\|\mathbf{x}\| = 1$. Dapatkah Anda menyederhanakan term $\|\mathbf{x} - \mathbf{x}_i\|^2$ dalam eksponensial? Petunjuk: kita nantinya akan melihat bahwa ini sangat terkait dengan perhatian produk titik (dot product attention).
4. Ingat bahwa :citet:`mack1982weak` membuktikan bahwa estimasi Nadaraya--Watson adalah konsisten. Seberapa cepat Anda harus mengurangi skala untuk mekanisme perhatian seiring dengan bertambahnya data? Berikan beberapa intuisi untuk jawaban Anda. Apakah ini bergantung pada dimensi data? Bagaimana?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18026)
:end_tab:
