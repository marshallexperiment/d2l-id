```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Model Klasifikasi Dasar
:label:`sec_classification`

Anda mungkin telah memperhatikan bahwa implementasi dari awal dan implementasi ringkas menggunakan fungsionalitas framework cukup mirip dalam kasus regresi.
Hal yang sama juga berlaku untuk klasifikasi. Karena banyak model dalam buku ini berurusan dengan klasifikasi, 
ada baiknya menambahkan fungsionalitas untuk mendukung pengaturan ini secara khusus. 
Bagian ini menyediakan kelas dasar untuk model klasifikasi guna menyederhanakan kode di masa mendatang.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Kelas `Classifier`

:begin_tab:`pytorch, mxnet, tensorflow`
Kami mendefinisikan kelas `Classifier` di bawah ini. Pada `validation_step` kami melaporkan nilai loss dan akurasi klasifikasi pada batch validasi. 
Kami mengambil pembaruan untuk setiap `num_val_batches` batch. Hal ini memiliki keuntungan menghasilkan rata-rata loss dan akurasi pada seluruh data validasi.
Angka rata-rata ini mungkin tidak sepenuhnya akurat jika batch terakhir berisi lebih sedikit contoh, 
tetapi kami mengabaikan perbedaan kecil ini demi menjaga kesederhanaan kode.
:end_tab:

:begin_tab:`jax`
Kami mendefinisikan kelas `Classifier` di bawah ini. Pada `validation_step` kami melaporkan nilai loss dan akurasi klasifikasi pada batch validasi. 
Kami mengambil pembaruan untuk setiap `num_val_batches` batch. Hal ini memiliki keuntungan menghasilkan rata-rata loss dan akurasi pada seluruh data validasi.
Angka rata-rata ini mungkin tidak sepenuhnya akurat jika batch terakhir berisi lebih sedikit contoh, tetapi kami mengabaikan perbedaan kecil ini demi menjaga kesederhanaan kode.

Kami juga mendefinisikan ulang metode `training_step` untuk JAX karena semua model yang akan
menjadi subclass `Classifier` nanti akan memiliki loss yang mengembalikan data tambahan.
Data tambahan ini dapat digunakan untuk model dengan batch normalization
(akan dijelaskan di :numref:`sec_batch_norm`), sementara dalam semua kasus lainnya,
kami juga akan membuat loss mengembalikan placeholder (kamus kosong) untuk
mewakili data tambahan.
:end_tab:


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """Kelas dasar untuk model klasifikasi."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """Kelas dasar untuk model klasifikasi."""
    def training_step(self, params, batch, state):
        # Di sini `value` adalah tuple karena model dengan lapisan BatchNorm membutuhkan
        # loss untuk mengembalikan data tambahan
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Abaikan nilai kedua yang dikembalikan. Nilai ini digunakan untuk melatih model
        # dengan lapisan BatchNorm karena loss juga mengembalikan data tambahan
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

Secara default, kita menggunakan optimizer stochastic gradient descent, yang bekerja pada minibatch, seperti yang kita lakukan pada konteks regresi linear.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Akurasi

Diberikan distribusi probabilitas prediksi `y_hat`,
kita biasanya memilih kelas dengan probabilitas prediksi tertinggi
setiap kali kita harus memberikan prediksi tegas.
Memang, banyak aplikasi mengharuskan kita untuk membuat pilihan.
Misalnya, Gmail harus mengkategorikan email ke dalam "Primary", "Social", "Updates", "Forums", atau "Spam".
Gmail mungkin memperkirakan probabilitas secara internal,
tetapi pada akhirnya ia harus memilih salah satu dari kelas-kelas tersebut.

Ketika prediksi konsisten dengan kelas label `y`, prediksi tersebut benar.
Akurasi klasifikasi adalah fraksi dari semua prediksi yang benar.
Meskipun bisa sulit untuk mengoptimalkan akurasi secara langsung (karena tidak terdiferensiasi),
ini sering kali menjadi ukuran kinerja yang paling kita perhatikan. Ini sering kali *merupakan*
kuantitas relevan dalam benchmark. Oleh karena itu, kita hampir selalu melaporkannya saat melatih classifier.

Akurasi dihitung sebagai berikut.
Pertama, jika `y_hat` adalah matriks,
kita mengasumsikan bahwa dimensi kedua menyimpan skor prediksi untuk setiap kelas.
Kita menggunakan `argmax` untuk mendapatkan kelas prediksi melalui indeks dari entri terbesar di setiap baris.
Kemudian kita [**membandingkan kelas prediksi dengan `y` yang sebenarnya secara elementwise.**]
Karena operator kesetaraan `==` sensitif terhadap jenis data,
kita mengonversi jenis data `y_hat` agar sesuai dengan `y`.
Hasilnya adalah tensor yang berisi entri 0 (salah) dan 1 (benar).
Mengambil jumlahnya akan menghasilkan jumlah prediksi yang benar.


```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Menghitung jumlah prediksi yang benar."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Menghitung jumlah prediksi yang benar."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Ringkasan

Klasifikasi adalah masalah yang cukup umum sehingga memerlukan fungsi-fungsi tambahan yang memudahkan. Hal yang sangat penting dalam klasifikasi adalah *akurasi* dari classifier. 
Perlu dicatat bahwa meskipun kita sering kali sangat peduli terhadap akurasi, 
kita melatih classifier untuk mengoptimalkan berbagai tujuan lain karena alasan statistik dan komputasi. 
Namun, terlepas dari fungsi loss mana yang diminimalkan selama pelatihan, 
akan sangat berguna memiliki metode tambahan untuk menilai akurasi classifier kita secara empiris.

## Latihan

1. Diberikan $L_\textrm{v}$ sebagai loss validasi, dan biarkan $L_\textrm{v}^\textrm{q}$ sebagai perkiraan kasar yang dihitung melalui rata-rata fungsi loss dalam bagian ini. Terakhir, berikan $l_\textrm{v}^\textrm{b}$ sebagai loss pada minibatch terakhir. Ekspresikan $L_\textrm{v}$ dalam hal $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, serta ukuran sampel dan minibatch.
2. Tunjukkan bahwa perkiraan cepat $L_\textrm{v}^\textrm{q}$ tidak bias. Artinya, tunjukkan bahwa $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$. Mengapa Anda mungkin masih ingin menggunakan $L_\textrm{v}$?
3. Diberikan sebuah loss klasifikasi multikelas, di mana $l(y,y')$ adalah penalti dalam memperkirakan $y'$ ketika kita melihat $y$ dan diberikan probabilitas $p(y \mid x)$, rumuskan aturan untuk pemilihan optimal dari $y'$. Petunjuk: ekspresikan loss ekspektasi, menggunakan $l$ dan $p(y \mid x)$.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17981)
:end_tab:

