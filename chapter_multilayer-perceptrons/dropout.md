```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dropout
:label:`sec_dropout`

Mari kita pikirkan sebentar tentang apa yang kita harapkan dari sebuah model prediktif yang baik.
Kita menginginkan model tersebut bekerja dengan baik pada data yang belum pernah dilihat.
Teori generalisasi klasik menunjukkan bahwa untuk menutup kesenjangan antara
kinerja pada data pelatihan dan data uji,
kita sebaiknya menggunakan model yang sederhana.
Kesederhanaan bisa diwujudkan dalam bentuk
jumlah dimensi yang sedikit.
Kita mengeksplorasi hal ini ketika mendiskusikan
fungsi basis monomial dari model linear
di :numref:`sec_generalization_basics`.
Selain itu, seperti yang kita lihat saat mendiskusikan weight decay
($\ell_2$ regularization) di :numref:`sec_weight_decay`,
(norm invers) dari parameter juga
merepresentasikan ukuran kesederhanaan yang berguna.
Konsep lain dari kesederhanaan adalah kelancaran (smoothness),
yaitu bahwa fungsi tersebut tidak seharusnya sensitif
terhadap perubahan kecil pada input.
Misalnya, ketika kita mengklasifikasikan gambar,
kita mengharapkan bahwa menambahkan sedikit noise acak
pada piksel seharusnya tidak banyak berpengaruh.

:citet:`Bishop.1995` meresmikan
gagasan ini ketika ia membuktikan bahwa pelatihan dengan input noise
setara dengan regularisasi Tikhonov.
Pekerjaan ini membuat hubungan matematis yang jelas
antara persyaratan bahwa sebuah fungsi harus halus (dan oleh karena itu sederhana),
dan persyaratan bahwa fungsi tersebut harus tahan
terhadap gangguan pada input.

Kemudian, :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`
mengembangkan ide cerdas tentang bagaimana menerapkan gagasan Bishop
pada lapisan-lapisan internal dari sebuah jaringan.
Ide mereka, yang disebut *dropout*, melibatkan
penyuntikan noise saat menghitung
setiap lapisan internal selama forward propagation,
dan ini telah menjadi teknik standar
untuk melatih neural networks.
Metode ini disebut *dropout* karena kita benar-benar
*menghapus* beberapa neuron selama pelatihan.
Selama pelatihan, pada setiap iterasi,
dropout standar terdiri dari menghapus
beberapa fraksi node di setiap lapisan
sebelum menghitung lapisan berikutnya.

Untuk lebih jelasnya, kami menyajikan
narasi kami sendiri terkait Bishop.
Makalah asli tentang dropout
menawarkan intuisi melalui analogi yang mengejutkan
dengan reproduksi seksual.
Para penulis berpendapat bahwa overfitting pada neural networks
dikarakterisasi oleh keadaan di mana
setiap lapisan bergantung pada pola aktivasi tertentu pada lapisan sebelumnya,
menyebut kondisi ini sebagai *co-adaptation*.
Dropout, klaim mereka, memecah co-adaptation
sebagaimana reproduksi seksual diklaim memecah gen-gen yang co-adapted.
Meskipun pembenaran teori ini tentunya dapat diperdebatkan,
teknik dropout itu sendiri telah terbukti bertahan lama,
dan berbagai bentuk dropout diterapkan
di sebagian besar library deep learning.

Tantangan utamanya adalah bagaimana menyuntikkan noise ini.
Salah satu idenya adalah menyuntikkannya secara *unbiased*
sehingga nilai harapan dari setiap lapisan—saat memperbaiki yang lain—
sama dengan nilai yang akan diambilnya jika tidak ada noise.
Dalam pekerjaan Bishop, ia menambahkan Gaussian noise
pada input dari model linear.
Pada setiap iterasi pelatihan, ia menambahkan noise
yang diambil dari distribusi dengan rata-rata nol
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ ke input $\mathbf{x}$,
menghasilkan titik terganggu $\mathbf{x}' = \mathbf{x} + \epsilon$.
Dalam ekspektasi, $E[\mathbf{x}'] = \mathbf{x}$.

Pada regularisasi dropout standar,
kita meniadakan beberapa fraksi node di setiap lapisan
dan kemudian *menyesuaikan bias* pada setiap lapisan dengan melakukan normalisasi
berdasarkan fraksi node yang dipertahankan (tidak dihapus).
Dengan kata lain,
dengan *dropout probability* $p$,
setiap aktivasi antara $h$ digantikan oleh
variabel acak $h'$ sebagai berikut:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ dengan probabilitas } p \\
    \frac{h}{1-p} & \textrm{ selain itu}
\end{cases}
\end{aligned}
$$

Dengan desain ini, ekspektasinya tetap tidak berubah, yaitu $E[h'] = h$.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Dropout in Practice

Ingat kembali MLP dengan satu hidden layer dan lima unit tersembunyi dari :numref:`fig_mlp`.
Ketika kita menerapkan dropout pada hidden layer,
menghapus setiap unit tersembunyi dengan probabilitas $p$,
hasilnya dapat dilihat sebagai jaringan
yang hanya mengandung sebagian dari neuron asli.
Pada :numref:`fig_dropout2`, $h_2$ dan $h_5$ dihapus.
Akibatnya, perhitungan output
tidak lagi bergantung pada $h_2$ atau $h_5$,
dan gradiennya juga menjadi nol
saat melakukan backpropagation.
Dengan cara ini, perhitungan output layer
tidak dapat terlalu bergantung pada salah satu elemen dari $h_1, \ldots, h_5$.

![MLP sebelum dan sesudah dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

Biasanya, kita menonaktifkan dropout pada saat pengujian (test time).
Dengan model yang sudah terlatih dan contoh baru,
kita tidak menghapus node apa pun
dan oleh karena itu tidak perlu melakukan normalisasi.
Namun, ada beberapa pengecualian:
beberapa peneliti menggunakan dropout pada saat pengujian sebagai heuristik
untuk mengestimasi *ketidakpastian* prediksi neural network:
jika prediksi konsisten di berbagai hasil dropout,
maka kita dapat mengatakan bahwa jaringan lebih yakin.

## Implementasi dari Awal

Untuk mengimplementasikan fungsi dropout untuk satu lapisan,
kita harus mengambil sampel sebanyak mungkin
dari variabel acak Bernoulli (biner)
seperti jumlah dimensi pada lapisan kita,
di mana variabel acak ini bernilai $1$ (dipertahankan)
dengan probabilitas $1-p$ dan $0$ (dihapus) dengan probabilitas $p$.
Cara mudah untuk mengimplementasikannya adalah dengan terlebih dahulu mengambil sampel
dari distribusi uniform $U[0, 1]$.
Kemudian kita mempertahankan node-node di mana sampel terkait
lebih besar dari $p$, dan menghapus sisanya.

Dalam kode berikut, kita (**mengimplementasikan fungsi `dropout_layer`
yang menghapus elemen-elemen dalam tensor input `X`
dengan probabilitas `dropout`**),
dan menskalakan ulang elemen yang tersisa seperti dijelaskan di atas:
membagi elemen yang tersisa dengan `1.0-dropout`.


```{.python .input}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

Kita dapat [**menguji fungsi `dropout_layer` pada beberapa contoh**].
Pada baris kode berikut,
kita melewatkan input `X` melalui operasi dropout,
dengan probabilitas 0, 0.5, dan 1, secara berurutan.

```{.python .input}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### Mendefinisikan Model

Model di bawah ini menerapkan dropout pada output
dari setiap hidden layer (setelah fungsi aktivasi).
Kita dapat mengatur probabilitas dropout untuk setiap lapisan secara terpisah.
Pilihan umum adalah menetapkan
probabilitas dropout yang lebih rendah semakin dekat ke input layer.
Kita memastikan bahwa dropout hanya aktif selama pelatihan.


```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)

    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    def setup(self):
        self.lin1 = nn.Dense(self.num_hiddens_1)
        self.lin2 = nn.Dense(self.num_hiddens_2)
        self.lin3 = nn.Dense(self.num_outputs)
        self.relu = nn.relu

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**Training**]

Bagian berikut mirip dengan pelatihan MLP yang telah dijelaskan sebelumnya.


```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**Implementasi Singkat**]

Dengan API tingkat tinggi, yang perlu kita lakukan hanyalah menambahkan lapisan `Dropout`
setelah setiap fully connected layer,
dengan memasukkan probabilitas dropout
sebagai satu-satunya argumen ke konstruktor.
Selama pelatihan, lapisan `Dropout` secara acak
akan menghapus output dari lapisan sebelumnya
(atau setara dengan menghapus input ke lapisan berikutnya)
sesuai dengan probabilitas dropout yang ditentukan.
Saat tidak dalam mode pelatihan,
lapisan `Dropout` hanya akan meneruskan data selama pengujian.

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class DropoutMLP(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    @nn.compact
    def __call__(self, X):
        x = nn.relu(nn.Dense(self.num_hiddens_1)(X.reshape((X.shape[0], -1))))
        x = nn.Dropout(self.dropout_1, deterministic=not self.training)(x)
        x = nn.relu(nn.Dense(self.num_hiddens_2)(x))
        x = nn.Dropout(self.dropout_2, deterministic=not self.training)(x)
        return nn.Dense(self.num_outputs)(x)
```

:begin_tab:`jax`
Perhatikan bahwa kita perlu mendefinisikan ulang fungsi loss karena jaringan
dengan lapisan dropout membutuhkan PRNGKey saat menggunakan `Module.apply()`,
dan RNG seed ini harus diberi nama `dropout` secara eksplisit. Kunci ini
digunakan oleh lapisan `dropout` dalam Flax untuk menghasilkan
masker dropout acak secara internal. Penting untuk menggunakan kunci `dropout_rng` yang unik
pada setiap epoch dalam loop pelatihan, jika tidak, masker dropout yang dihasilkan
tidak akan bersifat stokastik dan akan berbeda antar setiap epoch.
Kunci `dropout_rng` ini dapat disimpan dalam objek
`TrainState` (dalam kelas `d2l.Trainer` yang didefinisikan di
:numref:`oo-design-training`) sebagai atribut dan pada setiap epoch
kunci ini diganti dengan `dropout_rng` baru. Kami telah menangani ini dengan
metode `fit_epoch` yang didefinisikan di :numref:`sec_linear_scratch`.
:end_tab:


```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False,  # To be used later (e.g., batch norm)
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

Selanjutnya, kita [**melatih model**].


```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## Summary

Selain mengontrol jumlah dimensi dan ukuran vektor bobot, dropout adalah alat lain untuk menghindari overfitting. Seringkali alat-alat ini digunakan secara bersama-sama.
Perlu diperhatikan bahwa dropout
hanya digunakan selama pelatihan:
dropout menggantikan aktivasi $h$ dengan variabel acak yang memiliki nilai harapan $h$.


## Exercises

1. Apa yang terjadi jika Anda mengubah probabilitas dropout untuk lapisan pertama dan kedua? Secara khusus, apa yang terjadi jika Anda menukar probabilitas dropout untuk kedua lapisan tersebut? Rancang eksperimen untuk menjawab pertanyaan ini, gambarkan hasil Anda secara kuantitatif, dan ringkas kesimpulan kualitatif yang diperoleh.
2. Tingkatkan jumlah epoch dan bandingkan hasil yang diperoleh saat menggunakan dropout dengan saat tidak menggunakannya.
3. Berapakah variansi aktivasi pada setiap hidden layer ketika dropout diterapkan dan tidak diterapkan? Gambarkan plot untuk menunjukkan bagaimana kuantitas ini berkembang seiring waktu untuk kedua model.
4. Mengapa dropout biasanya tidak digunakan pada saat pengujian?
5. Menggunakan model di bagian ini sebagai contoh, bandingkan efek menggunakan dropout dan weight decay. Apa yang terjadi ketika dropout dan weight decay digunakan bersamaan? Apakah hasilnya aditif? Apakah ada keuntungan yang menurun (atau lebih buruk)? Apakah keduanya saling meniadakan?
6. Apa yang terjadi jika kita menerapkan dropout pada bobot individual dari matriks bobot daripada pada aktivasi?
7. Ciptakan teknik lain untuk menyuntikkan noise acak pada setiap lapisan yang berbeda dari teknik dropout standar. Bisakah Anda mengembangkan metode yang mengungguli dropout pada dataset Fashion-MNIST (untuk arsitektur yang sama)?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/261)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17987)
:end_tab:
