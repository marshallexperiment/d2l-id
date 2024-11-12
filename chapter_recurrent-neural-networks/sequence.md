# Bekerja dengan Urutan (_Sequence_)
:label:`sec_sequence`

Sejauh ini, kita telah berfokus pada model yang inputnya terdiri dari satu vektor fitur $\mathbf{x} \in \mathbb{R}^d$.
Perubahan perspektif utama saat mengembangkan model
yang mampu memproses urutan adalah bahwa sekarang
kita fokus pada input yang terdiri dari daftar terurut
vektor fitur $\mathbf{x}_1, \dots, \mathbf{x}_T$,
di mana setiap vektor fitur $\mathbf{x}_t$ 
diindeks oleh langkah waktu $t \in \mathbb{Z}^+$
yang berada dalam $\mathbb{R}^d$.

Beberapa dataset terdiri dari satu urutan besar.
Misalnya, data aliran sensor yang sangat panjang
yang mungkin tersedia untuk ilmuwan iklim.
Dalam kasus seperti itu, kita dapat membuat dataset pelatihan
dengan secara acak mengambil subsekuens dari panjang tertentu.
Lebih sering, data kita hadir sebagai kumpulan urutan.
Pertimbangkan contoh berikut:
(i) kumpulan dokumen,
masing-masing direpresentasikan sebagai urutan kata tersendiri,
dengan panjang masing-masing $T_i$;
(ii) representasi urutan
rawat inap pasien di rumah sakit,
di mana setiap rawat inap terdiri dari sejumlah kejadian
dan panjang urutan kurang lebih bergantung
pada lama waktu rawat inap tersebut.

Sebelumnya, saat bekerja dengan input individual,
kita mengasumsikan bahwa mereka diambil secara independen
dari distribusi dasar yang sama $P(X)$.
Meskipun kita masih mengasumsikan bahwa seluruh urutan
(misalnya, seluruh dokumen atau perjalanan pasien)
diambil secara independen,
kita tidak dapat mengasumsikan bahwa data yang datang
pada setiap langkah waktu tidak saling bergantung.
Misalnya, kata-kata yang mungkin muncul kemudian dalam dokumen
sangat bergantung pada kata-kata yang muncul sebelumnya dalam dokumen.
Obat yang kemungkinan besar akan diterima pasien
pada hari ke-10 kunjungan rumah sakit
sangat bergantung pada apa yang terjadi
dalam sembilan hari sebelumnya.

Ini seharusnya tidak mengejutkan.
Jika kita tidak percaya bahwa elemen dalam urutan berhubungan,
kita tidak akan repot-repot memodelkannya sebagai urutan sejak awal.
Pertimbangkan kegunaan fitur *auto-fill*
yang populer pada alat pencarian dan klien email modern.
Fitur ini berguna karena sering kali memungkinkan
untuk memprediksi (secara tidak sempurna, tetapi lebih baik daripada menebak acak)
kemungkinan lanjutan dari suatu urutan,
dengan memberi awalan tertentu.
Untuk sebagian besar model urutan,
kita tidak memerlukan independensi,
atau bahkan *stationarity*, dari urutan kita.
Sebaliknya, kita hanya memerlukan bahwa
urutan tersebut diambil
dari distribusi dasar tetap tertentu
atas seluruh urutan.

Pendekatan yang fleksibel ini memungkinkan fenomena seperti
(i) dokumen terlihat sangat berbeda di awal dibandingkan di akhir;
atau (ii) status pasien berkembang baik
menuju pemulihan atau kematian
selama masa rawat inap;
atau (iii) selera pelanggan yang berkembang dengan cara yang dapat diprediksi
selama interaksi berkelanjutan dengan sistem rekomendasi.



Terkadang kita ingin memprediksi target tetap $y$
dengan diberikan input yang terstruktur secara berurutan
(misalnya, klasifikasi sentimen berdasarkan ulasan film).
Di lain waktu, kita ingin memprediksi target yang terstruktur secara berurutan
($y_1, \ldots, y_T$)
dengan input tetap (misalnya, pembuatan keterangan gambar).
Pada kesempatan lain, tujuan kita adalah memprediksi target yang terstruktur secara berurutan
berdasarkan input yang juga terstruktur secara berurutan
(misalnya, terjemahan mesin atau pembuatan keterangan video).
Tugas *sequence-to-sequence* seperti ini dapat berbentuk dua macam:
(i) *aligned* (selaras): di mana input pada setiap langkah waktu
sesuai dengan target yang bersesuaian (misalnya, penandaan kelas kata);
(ii) *unaligned* (tidak selaras): di mana input dan target
tidak harus menunjukkan kesesuaian langkah demi langkah
(misalnya, terjemahan mesin).

Sebelum kita memikirkan tentang cara menangani target jenis apa pun,
kita bisa menangani masalah yang paling sederhana terlebih dahulu:
pemodelan densitas tak berlabel (juga disebut *sequence modeling*).
Di sini, dengan diberikan kumpulan urutan,
tujuan kita adalah memperkirakan fungsi distribusi probabilitas
yang memberi tahu kita seberapa besar kemungkinan kita melihat urutan tertentu,
yaitu $p(\mathbf{x}_1, \ldots, \mathbf{x}_T)$.


```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=9}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

## Model Autoregresif

Sebelum memperkenalkan jaringan saraf khusus
yang dirancang untuk menangani data yang terstruktur secara berurutan,
mari kita lihat beberapa data urutan yang sebenarnya
dan membangun beberapa intuisi dasar serta alat statistik.
Kita akan berfokus pada data harga saham
dari indeks FTSE 100 (:numref:`fig_ftse100`).
Pada setiap *time step* $t \in \mathbb{Z}^+$, kita mengamati
harga, $x_t$, dari indeks tersebut pada waktu itu.

![Indeks FTSE 100 selama sekitar 30 tahun.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`

Sekarang misalkan seorang pedagang ingin melakukan perdagangan jangka pendek,
dengan strategi masuk atau keluar dari indeks secara strategis,
tergantung pada apakah mereka percaya
bahwa harga akan naik atau turun
pada langkah waktu berikutnya.
Tanpa fitur tambahan
(berita, data laporan keuangan, dll.),
sinyal yang tersedia untuk memprediksi
nilai berikutnya hanyalah riwayat harga hingga saat ini.
Pedagang tersebut tertarik untuk mengetahui
distribusi probabilitas

$$P(x_t \mid x_{t-1}, \ldots, x_1)$$

atas harga yang mungkin diambil oleh indeks
pada langkah waktu berikutnya.
Meskipun memperkirakan seluruh distribusi
untuk variabel acak dengan nilai kontinu bisa jadi sulit,
pedagang akan senang
berfokus pada beberapa statistik utama dari distribusi tersebut,
terutama nilai harapan dan variansi.
Salah satu strategi sederhana untuk memperkirakan ekspektasi kondisional

$$\mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)],$$

adalah dengan menerapkan model regresi linier
(lihat kembali :numref:`sec_linear_regression`).
Model yang melakukan regresi nilai suatu sinyal
berdasarkan nilai-nilai sebelumnya dari sinyal yang sama
secara alami disebut *model autoregresif*.
Ada satu masalah utama: jumlah input,
$x_{t-1}, \ldots, x_1$ bervariasi, tergantung pada $t$.
Dengan kata lain, jumlah input meningkat
seiring dengan bertambahnya data yang kita temui.
Oleh karena itu, jika kita ingin memperlakukan data historis kita
sebagai set pelatihan, kita menghadapi masalah
di mana setiap contoh memiliki jumlah fitur yang berbeda.
Sebagian besar yang akan kita bahas dalam bab ini
akan berputar di sekitar teknik
untuk mengatasi tantangan ini
dalam masalah pemodelan *autoregresif* seperti ini
di mana objek yang kita perhatikan adalah
$P(x_t \mid x_{t-1}, \ldots, x_1)$
atau beberapa statistik dari distribusi ini.

Ada beberapa strategi yang sering digunakan.
Pertama-tama,
kita mungkin percaya bahwa meskipun tersedia urutan panjang
$x_{t-1}, \ldots, x_1$,
tidak selalu perlu
melihat begitu jauh ke belakang dalam riwayat
saat memprediksi masa depan dekat.
Dalam hal ini kita mungkin cukup puas
untuk mengkondisikan pada jendela dengan panjang $\tau$
dan hanya menggunakan pengamatan $x_{t-1}, \ldots, x_{t-\tau}$.
Manfaat langsungnya adalah bahwa sekarang jumlah argumen
selalu sama, setidaknya untuk $t > \tau$.
Ini memungkinkan kita untuk melatih model linier atau jaringan dalam apa pun
yang membutuhkan vektor dengan panjang tetap sebagai input.
Kedua, kita bisa mengembangkan model yang mempertahankan
ringkasan $h_t$ dari pengamatan masa lalu
(lihat :numref:`fig_sequence-model`)
dan secara bersamaan memperbarui $h_t$
selain prediksi $\hat{x}_t$.
Ini menghasilkan model yang tidak hanya memperkirakan $x_t$
dengan $\hat{x}_t = P(x_t \mid h_{t})$
tetapi juga pembaruan dalam bentuk
$h_t = g(h_{t-1}, x_{t-1})$.
Karena $h_t$ tidak pernah diamati,
model-model ini juga disebut
*model autoregresif laten*.

![Model autoregresif laten.](../img/sequence-model.svg)
:label:`fig_sequence-model`

Untuk membangun data pelatihan dari data historis, biasanya
kita membuat contoh dengan mengambil jendela secara acak.
Secara umum, kita tidak mengharapkan waktu untuk diam.
Namun, kita sering mengasumsikan bahwa meskipun
nilai spesifik dari $x_t$ mungkin berubah,
dinamika yang mengatur bagaimana setiap pengamatan selanjutnya
dihasilkan berdasarkan pengamatan sebelumnya tetap sama.
Statistikawan menyebut dinamika yang tidak berubah ini sebagai *stationary*.




## Model Urutan

Kadang-kadang, terutama saat bekerja dengan bahasa,
kita ingin memperkirakan probabilitas gabungan
dari seluruh urutan.
Ini adalah tugas umum ketika bekerja dengan urutan
yang terdiri dari *token* diskret, seperti kata-kata.
Secara umum, fungsi yang diperkirakan ini disebut *model urutan*
dan untuk data bahasa alami, disebut *model bahasa*.
Bidang pemodelan urutan telah didorong begitu kuat oleh pemrosesan bahasa alami,
sehingga kita sering menggambarkan model urutan sebagai "model bahasa",
bahkan ketika bekerja dengan data non-bahasa.
Model bahasa terbukti berguna karena berbagai alasan.
Kadang-kadang kita ingin mengevaluasi kemungkinan suatu kalimat.
Misalnya, kita mungkin ingin membandingkan
kealamian dua keluaran kandidat
yang dihasilkan oleh sistem terjemahan mesin
atau oleh sistem pengenalan ucapan.
Namun, model bahasa tidak hanya memberi kita
kemampuan untuk *mengevaluasi* kemungkinan,
tetapi juga kemampuan untuk *menghasilkan* urutan,
dan bahkan mengoptimalkan untuk urutan yang paling mungkin.

Meskipun pemodelan bahasa mungkin pada pandangan pertama
tidak terlihat seperti masalah autoregresif,
kita dapat mengurangi pemodelan bahasa menjadi prediksi autoregresif
dengan mendekomposisi densitas gabungan dari urutan $p(x_1, \ldots, x_T)$
menjadi produk dari densitas bersyarat
secara bertahap dari kiri ke kanan
dengan menerapkan aturan rantai dari probabilitas:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

Perlu dicatat bahwa jika kita bekerja dengan sinyal diskret seperti kata-kata,
maka model autoregresif harus menjadi pengklasifikasi probabilistik,
yang menghasilkan distribusi probabilitas penuh
atas kosakata untuk kata berikutnya,
dengan konteks ke kiri sebagai referensi.



### Model Markov
:label:`subsec_markov-models`

Sekarang misalkan kita ingin menerapkan strategi yang disebutkan di atas,
di mana kita hanya mengondisikan pada $\tau$ langkah waktu sebelumnya,
yaitu $x_{t-1}, \ldots, x_{t-\tau}$, daripada
seluruh riwayat urutan $x_{t-1}, \ldots, x_1$.
Setiap kali kita bisa mengabaikan riwayat
di luar $\tau$ langkah sebelumnya
tanpa kehilangan kekuatan prediksi,
kita mengatakan bahwa urutan tersebut memenuhi *kondisi Markov*,
yaitu *bahwa masa depan bersifat independen secara kondisional terhadap masa lalu,
dengan diberikan riwayat terbaru*.
Ketika $\tau = 1$, kita mengatakan bahwa data dikarakterisasi
oleh *model Markov orde pertama*,
dan ketika $\tau = k$, kita mengatakan bahwa data dikarakterisasi
oleh model Markov orde $k$.
Ketika kondisi Markov orde pertama terpenuhi ($\tau = 1$),
faktorisasi probabilitas gabungan kita menjadi produk
dari probabilitas masing-masing kata yang diberikan oleh kata *sebelumnya*:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$

Seringkali kita merasa berguna untuk bekerja dengan model yang berjalan
seolah-olah kondisi Markov terpenuhi,
meskipun kita tahu bahwa ini hanya *benar secara aproksimasi*.
Dengan dokumen teks nyata, kita terus memperoleh informasi
seiring kita menyertakan lebih banyak konteks ke kiri.
Namun, keuntungan ini berkurang dengan cepat.
Oleh karena itu, terkadang kita berkompromi dengan menghilangkan kesulitan komputasi dan statistik
dengan melatih model yang validitasnya bergantung
pada kondisi Markov orde $k$.
Bahkan model bahasa RNN dan Transformer besar saat ini
jarang mengintegrasikan lebih dari ribuan kata dalam konteksnya.

Dengan data diskret, model Markov sejati
cukup menghitung jumlah kemunculan
setiap kata dalam setiap konteks, menghasilkan
perkiraan frekuensi relatif dari $P(x_t \mid x_{t-1})$.
Kapan pun data hanya memiliki nilai diskret
(seperti dalam bahasa),
urutan kata yang paling mungkin dapat dihitung secara efisien
menggunakan pemrograman dinamis.



### Urutan Dekoding

Anda mungkin bertanya-tanya mengapa kami merepresentasikan
faktorisasi dari urutan teks $P(x_1, \ldots, x_T)$
sebagai rantai probabilitas bersyarat dari kiri ke kanan.
Mengapa tidak dari kanan ke kiri atau dalam urutan lain yang tampaknya acak?
Pada prinsipnya, tidak ada yang salah dengan mengurai
$P(x_1, \ldots, x_T)$ dalam urutan terbalik.
Hasilnya tetap merupakan faktorisasi yang valid:

$$P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

Namun, ada banyak alasan mengapa faktorisasi teks
dalam arah yang sama dengan cara kita membacanya
(dari kiri ke kanan untuk sebagian besar bahasa,
tetapi dari kanan ke kiri untuk bahasa Arab dan Ibrani)
lebih disukai untuk tugas pemodelan bahasa.
Pertama, arah ini lebih alami bagi kita untuk berpikir.
Setiap hari kita membaca teks,
dan proses ini dipandu oleh kemampuan kita
untuk mengantisipasi kata-kata dan frasa
yang kemungkinan besar akan muncul berikutnya.
Pikirkan saja berapa kali Anda menyelesaikan
kalimat orang lain.
Jadi, bahkan jika kita tidak memiliki alasan lain untuk memilih dekoding dalam urutan ini,
arah ini berguna hanya karena kita memiliki intuisi yang lebih baik
untuk apa yang mungkin terjadi ketika memprediksi dalam urutan ini.

Kedua, dengan melakukan faktorisasi dalam urutan yang sama,
kita dapat menetapkan probabilitas untuk urutan dengan panjang berapa pun
menggunakan model bahasa yang sama.
Untuk mengubah probabilitas atas langkah $1$ hingga $t$
menjadi satu yang mencakup kata $t+1$ kita cukup
mengalikan dengan probabilitas kondisional
dari token tambahan yang diberikan oleh token sebelumnya:
$P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1)$.

Ketiga, kita memiliki model prediktif yang lebih kuat
untuk memprediksi kata-kata yang berdekatan dibandingkan
kata-kata pada posisi lain secara acak.
Meskipun semua urutan faktorisasi valid,
tidak semuanya merepresentasikan masalah prediktif
yang sama mudahnya.
Hal ini berlaku tidak hanya untuk bahasa,
tetapi juga untuk jenis data lain,
misalnya ketika data memiliki struktur kausal.
Sebagai contoh, kita percaya bahwa peristiwa di masa depan tidak dapat mempengaruhi masa lalu.
Karena itu, jika kita mengubah $x_t$, kita mungkin dapat mempengaruhi
apa yang terjadi pada $x_{t+1}$ ke depan, tetapi tidak sebaliknya.
Artinya, jika kita mengubah $x_t$, distribusi peristiwa masa lalu tidak akan berubah.
Dalam beberapa konteks, ini membuatnya lebih mudah untuk memprediksi $P(x_{t+1} \mid x_t)$
daripada memprediksi $P(x_t \mid x_{t+1})$.
Misalnya, dalam beberapa kasus, kita dapat menemukan $x_{t+1} = f(x_t) + \epsilon$
dengan beberapa noise tambahan $\epsilon$,
sedangkan kebalikannya tidak berlaku :cite:`Hoyer.Janzing.Mooij.ea.2009`.
Ini adalah kabar baik, karena biasanya arah ke depan
yang ingin kita estimasi.
Buku oleh :citet:`Peters.Janzing.Scholkopf.2017` membahas lebih lanjut tentang topik ini.
Kami hanya menyentuh permukaan topik ini.


## Pelatihan

Sebelum kita fokus pada data teks,
mari coba hal ini dengan beberapa
data sintetis bernilai kontinu.

(**Di sini, data sintetis kita sebanyak 1000 sampel akan mengikuti
fungsi trigonometri `sin`,
yang diterapkan pada langkah waktu 0.01 kali.
Untuk membuat masalah ini sedikit lebih menarik,
kita merusak setiap sampel dengan tambahan noise.**)
Dari urutan ini kita ekstraksi contoh pelatihan,
masing-masing terdiri dari fitur dan label.


```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2
        if tab.selected('jax'):
            key = d2l.get_key()
            self.x = d2l.sin(0.01 * self.time) + jax.random.normal(key,
                                                                   [T]) * 0.2
```

```{.python .input}
%%tab all
data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

Untuk memulai, kita mencoba model yang bertindak seolah-olah
data memenuhi kondisi Markov orde-$\tau$,
dan dengan demikian memprediksi $x_t$ hanya menggunakan $\tau$ pengamatan sebelumnya.
[**Jadi, untuk setiap langkah waktu, kita memiliki sebuah contoh
dengan label $y = x_t$ dan fitur
$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$.**]
Pembaca yang jeli mungkin telah memperhatikan bahwa
ini menghasilkan $1000-\tau$ contoh,
karena kita tidak memiliki riwayat yang cukup untuk $y_1, \ldots, y_\tau$.
Meskipun kita bisa mengisi $\tau$ urutan pertama dengan nol,
untuk menyederhanakan, kita abaikan bagian ini untuk sekarang.
Dataset yang dihasilkan berisi $T - \tau$ contoh,
di mana setiap input untuk model memiliki panjang urutan $\tau$.
Kita (**membuat iterator data pada 600 contoh pertama**),
mencakup periode dari fungsi sin.


```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

Dalam contoh ini, model kita akan berupa regresi linear standar.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## Prediksi

[**Untuk mengevaluasi model kita, pertama-tama kita periksa
seberapa baik model ini dalam melakukan prediksi satu langkah ke depan**].


```{.python .input}
%%tab pytorch, mxnet, tensorflow
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

```{.python .input}
%%tab jax
onestep_preds = model.apply({'params': trainer.state.params}, data.features)
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

Prediksi ini terlihat bagus,
bahkan mendekati akhir pada $t=1000$.

Tetapi bagaimana jika kita hanya mengamati data urutan
hingga langkah waktu 604 (`n_train + tau`)
dan ingin membuat prediksi beberapa langkah
ke masa depan?
Sayangnya, kita tidak dapat secara langsung menghitung
prediksi satu langkah ke depan untuk langkah waktu 609,
karena kita tidak mengetahui input yang sesuai,
hanya mengamati hingga $x_{604}$.
Kita dapat mengatasi masalah ini dengan memasukkan
prediksi sebelumnya sebagai input ke model kita
untuk membuat prediksi selanjutnya,
melakukan proyeksi ke depan, satu langkah pada satu waktu,
hingga mencapai langkah waktu yang diinginkan:

$$\begin{aligned}
\hat{x}_{605} &= f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} &= f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} &= f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} &= f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} &= f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
&\vdots\end{aligned}$$

Secara umum, untuk urutan yang diamati $x_1, \ldots, x_t$,
output prediksi $\hat{x}_{t+k}$ pada langkah waktu $t+k$
disebut sebagai *prediksi $k$-langkah ke depan*.
Karena kita telah mengamati hingga $x_{604}$,
prediksi $k$-langkah ke depan adalah $\hat{x}_{604+k}$.
Dengan kata lain, kita harus terus menggunakan prediksi kita sendiri
untuk membuat prediksi beberapa langkah ke depan.
Mari kita lihat seberapa baik cara ini berjalan.


```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab jax
multistep_preds = d2l.zeros(data.T)
multistep_preds = multistep_preds.at[:].set(data.x)
for i in range(data.num_train + data.tau, data.T):
    pred = model.apply({'params': trainer.state.params},
                       d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
    multistep_preds = multistep_preds.at[i].set(pred.item())
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

Sayangnya, dalam kasus ini kita gagal dengan sangat parah.
Prediksi dengan cepat menyusut menjadi konstan
setelah beberapa langkah ke depan.
Mengapa algoritma berkinerja jauh lebih buruk
saat memprediksi lebih jauh ke masa depan?
Pada akhirnya, ini terjadi karena
kesalahan yang semakin menumpuk.
Misalnya, setelah langkah pertama kita memiliki beberapa kesalahan $\epsilon_1 = \bar\epsilon$.
Sekarang *input* untuk langkah kedua terganggu oleh $\epsilon_1$,
sehingga kita mengalami kesalahan pada urutan
$\epsilon_2 = \bar\epsilon + c \epsilon_1$
untuk beberapa konstanta $c$, dan seterusnya.
Prediksi bisa dengan cepat menyimpang
dari pengamatan yang sebenarnya.
Anda mungkin sudah akrab
dengan fenomena umum ini.
Misalnya, perkiraan cuaca untuk 24 jam ke depan
biasanya cukup akurat, tetapi setelah itu,
akurasi menurun dengan cepat.
Kita akan membahas metode untuk memperbaiki hal ini
di sepanjang bab ini dan seterusnya.

Mari kita [**melihat lebih dekat kesulitan dalam prediksi $k$-langkah ke depan**]
dengan menghitung prediksi pada seluruh urutan untuk $k = 1, 4, 16, 64$.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab jax
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model.apply({'params': trainer.state.params},
                            d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

Hal ini dengan jelas menunjukkan bagaimana kualitas prediksi berubah
saat kita mencoba memprediksi lebih jauh ke masa depan.
Sementara prediksi 4 langkah ke depan masih terlihat bagus,
apa pun di luar itu hampir tidak berguna.

## Ringkasan

Terdapat perbedaan yang cukup signifikan dalam tingkat kesulitan
antara interpolasi dan ekstrapolasi.
Oleh karena itu, jika Anda memiliki urutan data, selalu hormati
urutan temporal data saat melatih model,
yaitu, jangan pernah melatih model menggunakan data masa depan.
Dengan jenis data ini,
model urutan memerlukan alat statistik khusus untuk estimasi.
Dua pilihan populer adalah model autoregresif
dan model autoregresif variabel tersembunyi.
Untuk model kausal (misalnya, waktu yang bergerak maju),
estimasi dalam arah ke depan biasanya
jauh lebih mudah daripada arah sebaliknya.
Untuk urutan yang diamati hingga langkah waktu $t$,
output yang diprediksi pada langkah waktu $t+k$
disebut sebagai *prediksi $k$-langkah ke depan*.
Saat kita memprediksi lebih jauh dengan meningkatkan $k$,
kesalahan semakin menumpuk dan kualitas prediksi menurun,
seringkali secara drastis.

## Latihan

1. Tingkatkan model pada eksperimen di bagian ini.
    1. Apakah dapat menggabungkan lebih dari empat pengamatan terakhir? Berapa banyak yang sebenarnya dibutuhkan?
    1. Berapa banyak pengamatan sebelumnya yang Anda butuhkan jika tidak ada noise? Petunjuk: Anda dapat menulis $\sin$ dan $\cos$ sebagai persamaan diferensial.
    1. Bisakah Anda memasukkan pengamatan yang lebih lama sambil mempertahankan jumlah fitur total tetap? Apakah ini meningkatkan akurasi? Mengapa?
    1. Ubah arsitektur jaringan saraf dan evaluasi kinerjanya. Anda dapat melatih model baru dengan lebih banyak epoch. Apa yang Anda amati?
1. Seorang investor ingin menemukan sekuritas yang baik untuk dibeli.
   Mereka melihat pengembalian masa lalu untuk memutuskan mana yang kemungkinan akan memberikan hasil yang baik.
   Apa yang bisa salah dengan strategi ini?
1. Apakah kausalitas juga berlaku untuk teks? Sejauh mana?
1. Berikan contoh di mana model autoregresif laten
   mungkin diperlukan untuk menangkap dinamika data.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1048)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18010)
:end_tab:
