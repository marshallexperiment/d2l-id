# Jaringan Saraf Berulang Dua Arah
:label:`sec_bi_rnn`

Sejauh ini, contoh utama kita tentang tugas pembelajaran urutan adalah pemodelan bahasa,
di mana tujuan kita adalah memprediksi token berikutnya berdasarkan semua token sebelumnya dalam sebuah urutan.
Dalam skenario ini, kita hanya ingin mempertimbangkan konteks ke arah kiri,
dan karena itu chaining satu arah dari RNN standar tampaknya sesuai.
Namun, ada banyak konteks tugas pembelajaran urutan lain
di mana sangat wajar untuk mempertimbangkan prediksi pada setiap langkah waktu
berdasarkan konteks baik di kiri maupun kanan.
Sebagai contoh, dalam deteksi part of speech.
Mengapa kita tidak mempertimbangkan konteks di kedua arah
saat menilai part of speech yang terkait dengan suatu kata?

Contoh umum lainnya—sering digunakan sebagai latihan pra-pelatihan
sebelum fine-tuning model pada tugas yang sebenarnya—adalah
menyembunyikan token acak dalam sebuah dokumen teks dan kemudian melatih
model urutan untuk memprediksi nilai dari token yang hilang.
Perhatikan bahwa tergantung pada apa yang datang setelah kekosongan,
nilai token yang hilang mungkin berubah secara dramatis:

* Saya merasa `___`.
* Saya merasa `___` lapar.
* Saya merasa `___` lapar, dan saya bisa makan setengah ekor babi.

Dalam kalimat pertama, "senang" tampaknya menjadi kandidat yang mungkin.
Kata-kata "tidak" dan "sangat" tampaknya masuk akal di kalimat kedua,
tetapi "tidak" tampaknya tidak sesuai dengan kalimat ketiga.


Untungnya, teknik sederhana mengubah RNN satu arah menjadi RNN dua arah :cite:`Schuster.Paliwal.1997`.
Kita cukup mengimplementasikan dua lapisan RNN satu arah yang
dirantai bersama dalam arah yang berlawanan
dan bekerja pada input yang sama (:numref:`fig_birnn`).
Untuk lapisan RNN pertama, input pertama adalah $\mathbf{x}_1$
dan input terakhir adalah $\mathbf{x}_T$,
tetapi untuk lapisan RNN kedua,
input pertama adalah $\mathbf{x}_T$
dan input terakhir adalah $\mathbf{x}_1$.
Untuk menghasilkan output dari lapisan RNN dua arah ini,
kita cukup menggabungkan bersama output yang sesuai
dari kedua lapisan RNN satu arah di bawahnya.


![Arsitektur dari RNN dua arah.](../img/birnn.svg)
:label:`fig_birnn`


Secara formal, untuk setiap langkah waktu $t$,
kita mempertimbangkan input minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$
(jumlah contoh $=n$; jumlah input dalam setiap contoh $=d$)
dan misalkan fungsi aktivasi lapisan tersembunyi adalah $\phi$.
Dalam arsitektur dua arah,
keadaan tersembunyi maju dan mundur untuk langkah waktu ini
adalah $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$
dan $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$, masing-masing,
di mana $h$ adalah jumlah unit tersembunyi.
Pembaruan keadaan tersembunyi maju dan mundur adalah sebagai berikut:


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}
$$

dengan bobot $\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ dan } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$, dan bias $\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$ serta $\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$ sebagai parameter model.

Selanjutnya, kita menggabungkan keadaan tersembunyi maju dan mundur
$\overrightarrow{\mathbf{H}}_t$ dan $\overleftarrow{\mathbf{H}}_t$
untuk mendapatkan keadaan tersembunyi $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ untuk dimasukkan ke dalam lapisan output.
Dalam RNN dua arah yang dalam dengan beberapa lapisan tersembunyi,
informasi ini diteruskan sebagai *input* ke lapisan dua arah berikutnya.
Terakhir, lapisan output menghitung output
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (jumlah output $=q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Di sini, matriks bobot $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$
dan bias $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
adalah parameter model dari lapisan output.
Meskipun secara teknis, kedua arah dapat memiliki jumlah unit tersembunyi yang berbeda,
pilihan desain ini jarang dilakukan dalam praktik.
Sekarang kami akan mendemonstrasikan implementasi sederhana dari RNN dua arah.


```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import npx, np
from mxnet.gluon import rnn
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
from jax import numpy as jnp
```

## Implementasi dari Awal

Jika kita ingin mengimplementasikan sebuah RNN bidirectional dari awal, kita bisa memasukkan dua instance `RNNScratch` unidirectional dengan parameter-parameter yang dapat dipelajari secara terpisah.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

```{.python .input}
%%tab jax
class BiRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

Status dari RNN forward dan backward diperbarui secara terpisah, sementara output dari kedua RNN tersebut digabungkan.


```{.python .input}
%%tab all
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)
```

## Implementasi Ringkas

:begin_tab:`pytorch, mxnet, tensorflow`
Dengan menggunakan API tingkat tinggi, kita dapat mengimplementasikan RNN bidirectional dengan lebih ringkas.
Di sini kita mengambil model GRU sebagai contoh.
:end_tab:

:begin_tab:`jax`
API Flax tidak menyediakan layer RNN sehingga tidak ada konsep `bidirectional` argument. Kita perlu membalik input secara manual seperti yang ditunjukkan dalam implementasi dari awal, jika diperlukan layer bidirectional.
:end_tab:


```{.python .input}
%%tab mxnet, pytorch
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

## Ringkasan

Dalam RNN bidirectional, hidden state untuk setiap langkah waktu ditentukan secara simultan oleh data sebelum dan sesudah langkah waktu saat ini. RNN bidirectional paling berguna untuk encoding urutan dan estimasi observasi dengan konteks bidirectional. RNN bidirectional sangat mahal untuk dilatih karena rantai gradien yang panjang.

## Latihan

1. Jika arah yang berbeda menggunakan jumlah unit tersembunyi yang berbeda, bagaimana bentuk dari $\mathbf{H}_t$ akan berubah?
2. Rancang sebuah RNN bidirectional dengan beberapa lapisan tersembunyi.
3. Polisemi adalah hal yang umum dalam bahasa alami. Misalnya, kata "bank" memiliki arti yang berbeda dalam konteks “saya pergi ke bank untuk menyetor uang tunai” dan “saya pergi ke tepi sungai untuk duduk”. Bagaimana kita dapat merancang model jaringan saraf sehingga, dengan memberikan urutan konteks dan sebuah kata, representasi vektor kata dalam konteks yang benar akan dikembalikan? Jenis arsitektur neural apa yang lebih disukai untuk menangani polisemi?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1059)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18019)
:end_tab:
