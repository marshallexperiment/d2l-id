```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Probabilitas dan Statistika
:label:`sec_prob`

Dalam satu atau lain cara,
machine learning selalu berkaitan dengan ketidakpastian.
Dalam pembelajaran terawasi (*supervised learning*), kita ingin memprediksi
sesuatu yang tidak diketahui (disebut *target*)
dengan menggunakan sesuatu yang diketahui (disebut *fitur*).
Bergantung pada tujuan kita,
kita mungkin mencoba memprediksi
nilai target yang paling mungkin.
Atau kita mungkin memprediksi nilai dengan jarak yang diharapkan terkecil dari target.
Dan terkadang kita tidak hanya ingin
memprediksi nilai tertentu
tetapi juga *mengukur ketidakpastian* kita.
Misalnya, diberikan beberapa fitur
yang menggambarkan seorang pasien,
kita mungkin ingin mengetahui *seberapa besar kemungkinan*
mereka mengalami serangan jantung dalam setahun ke depan.
Dalam pembelajaran tanpa pengawasan (*unsupervised learning*),
kita sering peduli terhadap ketidakpastian.
Untuk menentukan apakah serangkaian pengukuran adalah anomali,
sangat membantu jika kita mengetahui seberapa besar kemungkinan
mengamati nilai-nilai dalam populasi yang menjadi perhatian.
Selain itu, dalam reinforcement learning,
kita ingin mengembangkan agen
yang bertindak secara cerdas di berbagai lingkungan.
Ini memerlukan pemikiran tentang
bagaimana lingkungan mungkin diharapkan berubah
dan hadiah (*reward*) apa yang mungkin diharapkan sebagai respons
terhadap setiap tindakan yang tersedia.

*Probabilitas* adalah bidang matematika
yang berkaitan dengan pemikiran dalam ketidakpastian.
Diberikan sebuah model probabilistik dari suatu proses,
kita dapat berpikir tentang kemungkinan berbagai peristiwa.
Penggunaan probabilitas untuk menggambarkan
frekuensi dari kejadian yang dapat diulang
(seperti lemparan koin)
cukup tidak kontroversial.
Faktanya, ahli *frekuentis* berpegang teguh
pada interpretasi probabilitas
yang hanya berlaku pada kejadian-kejadian yang dapat diulang.
Sebaliknya, ahli *Bayesian*
menggunakan bahasa probabilitas lebih luas
untuk memformalkan pemikiran dalam ketidakpastian.
Probabilitas Bayesian ditandai
oleh dua fitur unik:
(i) memberikan derajat keyakinan
terhadap peristiwa yang tidak dapat diulang,
misalnya, berapa *probabilitas*
sebuah bendungan akan runtuh?;
dan (ii) subjektivitas. Probabilitas Bayesian
menyediakan aturan yang tidak ambigu
tentang bagaimana seseorang harus memperbarui keyakinan mereka
dalam cahaya bukti baru,
namun memungkinkan setiap individu
untuk memulai dengan keyakinan *prior* yang berbeda.
*Statistika* membantu kita berpikir mundur,
dimulai dengan pengumpulan dan pengorganisasian data
hingga kesimpulan apa yang mungkin bisa kita tarik
tentang proses yang menghasilkan data tersebut.
Setiap kali kita menganalisis sebuah dataset, mencari pola
yang kita harap dapat menggambarkan populasi yang lebih luas,
kita sedang menerapkan pemikiran statistik.
Banyak kursus, jurusan, tesis, karier, departemen,
perusahaan, dan institusi telah didedikasikan
untuk mempelajari probabilitas dan statistika.
Meskipun bagian ini hanya menggores permukaan,
kita akan memberikan dasar
yang Anda butuhkan untuk mulai membangun model.


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## Contoh Sederhana: Melempar Koin

Bayangkan bahwa kita berencana untuk melempar sebuah koin
dan ingin mengukur seberapa besar kemungkinan
kita melihat sisi kepala (dibandingkan sisi ekor).
Jika koin tersebut *adil*,
maka kedua hasilnya
(kepala dan ekor),
sama-sama mungkin terjadi.
Lebih jauh lagi, jika kita berencana melempar koin sebanyak $n$ kali,
maka fraksi dari sisi kepala
yang kita *harapkan* untuk dilihat
harus tepat sesuai
dengan fraksi *harapan* dari sisi ekor.
Salah satu cara intuitif untuk melihat hal ini
adalah melalui simetri:
untuk setiap kemungkinan hasil
dengan $n_\textrm{h}$ kepala dan $n_\textrm{t} = (n - n_\textrm{h})$ ekor,
ada kemungkinan hasil yang sama
dengan $n_\textrm{t}$ kepala dan $n_\textrm{h}$ ekor.
Perhatikan bahwa hal ini hanya mungkin terjadi
jika rata-rata kita mengharapkan
$1/2$ dari lemparan menunjukkan kepala
dan $1/2$ menunjukkan ekor.
Tentu saja, jika Anda melakukan percobaan ini
banyak kali dengan $n=1000000$ lemparan setiap kali,
Anda mungkin tidak pernah melihat percobaan
di mana $n_\textrm{h} = n_\textrm{t}$ tepat sama.

Secara formal, nilai $1/2$ ini disebut *probabilitas*
dan di sini ini mencerminkan kepastian bahwa
setiap lemparan koin memiliki kemungkinan menunjukkan sisi kepala.
Probabilitas memberikan skor antara $0$ dan $1$
pada hasil yang diinginkan, yang disebut *peristiwa*.
Di sini peristiwa yang diinginkan adalah $\textrm{kepala}$
dan kita menyebut probabilitas yang sesuai sebagai $P(\textrm{kepala})$.
Probabilitas $1$ menunjukkan kepastian mutlak
(bayangkan koin yang memiliki kedua sisinya kepala)
dan probabilitas $0$ menunjukkan kemustahilan
(misalnya, jika kedua sisinya adalah ekor).
Frekuensi $n_\textrm{h}/n$ dan $n_\textrm{t}/n$ bukanlah probabilitas
melainkan *statistik*.
Probabilitas adalah kuantitas *teoretis*
yang mendasari proses penghasil data.
Di sini, probabilitas $1/2$
adalah sifat dari koin itu sendiri.
Sebaliknya, statistik adalah kuantitas *empiris*
yang dihitung sebagai fungsi dari data yang diamati.
Ketertarikan kita pada kuantitas probabilistik dan statistik
sangat terkait erat.
Kita sering merancang statistik khusus yang disebut *penduga* (*estimators*)
yang, diberikan suatu dataset, menghasilkan *takrifan* (*estimates*)
dari parameter model seperti probabilitas.
Selain itu, ketika penduga-penduga tersebut memenuhi
sebuah sifat yang disebut *konsistensi*,
takrifan kita akan konvergen
ke probabilitas yang sesuai.
Dengan demikian, probabilitas yang disimpulkan ini
memberitahu kita tentang kemungkinan sifat statistik
dari data dari populasi yang sama
yang mungkin akan kita temui di masa mendatang.

Misalkan kita menemukan koin asli
yang mana kita tidak mengetahui
nilai sebenarnya dari $P(\textrm{kepala})$.
Untuk menyelidiki nilai ini
dengan metode statistik,
kita perlu (i) mengumpulkan beberapa data;
dan (ii) merancang sebuah penduga.
Pengumpulan data di sini cukup mudah;
kita dapat melempar koin berkali-kali
dan mencatat semua hasilnya.
Secara formal, mengambil realisasi
dari beberapa proses acak yang mendasarinya
disebut *sampling*.
Seperti yang mungkin sudah Anda duga,
salah satu penduga alami
adalah rasio
antara jumlah *kepala* yang diamati
dan total jumlah lemparan.

Sekarang, misalkan koin tersebut sebenarnya adil,
yaitu $P(\textrm{kepala}) = 0.5$.
Untuk mensimulasikan lemparan koin yang adil,
kita dapat menggunakan generator bilangan acak apa pun.
Ada beberapa cara mudah untuk mengambil sampel
dari suatu peristiwa dengan probabilitas $0.5$.
Sebagai contoh, `random.random` dari Python
menghasilkan angka dalam interval $[0,1]$
di mana probabilitas berada
dalam sub-interval $[a, b] \subset [0,1]$
sama dengan $b-a$.
Dengan demikian, kita dapat memperoleh `0` dan `1` dengan probabilitas `0.5` masing-masing
dengan menguji apakah angka float yang dihasilkan lebih besar dari `0.5`:


```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

Secara lebih umum, kita dapat mensimulasikan beberapa pengambilan sampel
dari variabel apa pun dengan sejumlah hasil yang terbatas
(seperti lemparan koin atau lemparan dadu)
dengan memanggil fungsi *multinomial*,
dengan argumen pertama
sebagai jumlah pengambilan sampel
dan argumen kedua sebagai daftar probabilitas
yang terkait dengan masing-masing hasil yang mungkin.
Untuk mensimulasikan sepuluh kali lemparan koin yang adil,
kita menetapkan vektor probabilitas `[0.5, 0.5]`,
dengan menginterpretasikan indeks 0 sebagai kepala
dan indeks 1 sebagai ekor.
Fungsi tersebut mengembalikan sebuah vektor
dengan panjang yang sama dengan jumlah
hasil yang mungkin (dalam hal ini, 2),
di mana komponen pertama memberi tahu kita
jumlah kejadian kepala
dan komponen kedua memberi tahu kita
jumlah kejadian ekor.


```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab jax
fair_probs = [0.5, 0.5]
# jax.random does not have multinomial distribution implemented
np.random.multinomial(100, fair_probs)
```

Setiap kali Anda menjalankan proses pengambilan sampel ini,
Anda akan menerima nilai acak baru
yang mungkin berbeda dari hasil sebelumnya.
Membagi dengan jumlah lemparan
memberikan kita *frekuensi*
dari setiap hasil dalam data kita.
Perhatikan bahwa frekuensi ini,
sama seperti probabilitas
yang mereka dimaksudkan untuk
ditaksir, jumlahnya adalah $1$.


```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab jax
np.random.multinomial(100, fair_probs) / 100
```
Di sini, meskipun koin yang kita simulasikan adalah adil
(kita sendiri yang menetapkan probabilitas `[0.5, 0.5]`),
jumlah kepala dan ekor mungkin tidak identik.
Hal ini karena kita hanya mengambil sampel dalam jumlah yang relatif kecil.
Jika kita tidak melakukan simulasi sendiri,
dan hanya melihat hasilnya,
bagaimana kita bisa tahu apakah koin tersebut sedikit tidak adil
atau apakah kemungkinan penyimpangan dari $1/2$ itu
hanya sebuah artefak dari ukuran sampel yang kecil?
Mari kita lihat apa yang terjadi ketika kita mensimulasikan 10.000 kali lemparan.



```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

Secara umum, untuk rata-rata dari kejadian yang berulang (seperti lemparan koin),
ketika jumlah pengulangan meningkat,
takrifan kita dijamin akan konvergen
ke probabilitas sebenarnya yang mendasarinya.
Formulasi matematis dari fenomena ini
disebut *hukum bilangan besar* (*law of large numbers*)
dan *teorema limit pusat* (*central limit theorem*)
yang mengatakan bahwa dalam banyak situasi,
ketika ukuran sampel $n$ meningkat,
kesalahan-kesalahan ini seharusnya berkurang
dengan laju $(1/\sqrt{n})$.
Mari kita dapatkan lebih banyak intuisi dengan mempelajari
bagaimana takrifan kita berkembang ketika kita meningkatkan
jumlah lemparan dari 1 hingga 10.000.


```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Setiap kurva solid sesuai dengan salah satu dari dua nilai pada koin
dan memberikan taksiran probabilitas kita bahwa koin menunjukkan nilai tersebut
setelah setiap kelompok percobaan.
Garis putus-putus hitam menunjukkan probabilitas sebenarnya yang mendasarinya.
Saat kita mendapatkan lebih banyak data dengan melakukan lebih banyak percobaan,
kurva-kurva tersebut semakin konvergen menuju probabilitas sebenarnya.
Anda mungkin sudah mulai melihat bentuk
beberapa pertanyaan yang lebih maju
yang menjadi perhatian para ahli statistika:
Seberapa cepat konvergensi ini terjadi?
Jika kita telah menguji banyak koin
yang diproduksi di pabrik yang sama,
bagaimana kita dapat menggabungkan informasi ini?

## Perlakuan yang Lebih Formal

Kita sudah cukup jauh: mengajukan
model probabilistik,
menghasilkan data sintetis,
menjalankan penduga statistik,
menilai konvergensi secara empiris,
dan melaporkan metrik kesalahan (memeriksa penyimpangan).
Namun, untuk melangkah lebih jauh,
kita perlu lebih tepat.

Ketika berurusan dengan keacakan,
kita menyebut himpunan kemungkinan hasil $\mathcal{S}$
dan menyebutnya sebagai *ruang sampel* atau *ruang hasil*.
Di sini, setiap elemen adalah *hasil* yang mungkin berbeda.
Dalam kasus melempar satu koin,
$\mathcal{S} = \{\textrm{kepala}, \textrm{ekor}\}$.
Untuk satu dadu, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
Saat melempar dua koin, kemungkinan hasilnya adalah
$\{(\textrm{kepala}, \textrm{kepala}), (\textrm{kepala}, \textrm{ekor}), (\textrm{ekor}, \textrm{kepala}),  (\textrm{ekor}, \textrm{ekor})\}$.
*Peristiwa* adalah bagian dari ruang sampel.
Sebagai contoh, peristiwa "lemparan koin pertama menunjukkan kepala"
sesuai dengan himpunan $\{(\textrm{kepala}, \textrm{kepala}), (\textrm{kepala}, \textrm{ekor})\}$.
Setiap kali hasil $z$ dari percobaan acak memenuhi
$z \in \mathcal{A}$, maka peristiwa $\mathcal{A}$ telah terjadi.
Untuk satu lemparan dadu, kita dapat mendefinisikan peristiwa
"melihat angka $5$" ($\mathcal{A} = \{5\}$)
dan "melihat angka ganjil" ($\mathcal{B} = \{1, 3, 5\}$).
Dalam kasus ini, jika hasil dadu adalah $5$,
kita akan mengatakan bahwa baik $\mathcal{A}$ maupun $\mathcal{B}$ terjadi.
Di sisi lain, jika $z = 3$,
maka $\mathcal{A}$ tidak terjadi
tetapi $\mathcal{B}$ terjadi.

Fungsi *probabilitas* memetakan peristiwa
ke nilai riil ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
Probabilitas, yang dilambangkan dengan $P(\mathcal{A})$, dari suatu peristiwa $\mathcal{A}$
dalam ruang sampel $\mathcal{S}$ yang diberikan,
memiliki sifat-sifat sebagai berikut:

* Probabilitas dari setiap peristiwa $\mathcal{A}$ adalah bilangan riil non-negatif, yaitu $P(\mathcal{A}) \geq 0$;
* Probabilitas dari seluruh ruang sampel adalah $1$, yaitu $P(\mathcal{S}) = 1$;
* Untuk setiap urutan peristiwa yang dapat dihitung $\mathcal{A}_1, \mathcal{A}_2, \ldots$ yang *saling eksklusif* (yaitu, $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ untuk semua $i \neq j$), probabilitas bahwa salah satu dari mereka terjadi sama dengan jumlah dari probabilitas individu mereka, yaitu $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Aksioma-aksioma teori probabilitas ini,
diusulkan oleh :citet:`Kolmogorov.1933`,
dapat diterapkan untuk dengan cepat mendapatkan sejumlah konsekuensi penting.
Misalnya, segera kita dapat melihat
bahwa probabilitas dari setiap peristiwa $\mathcal{A}$
*atau* komplemennya $\mathcal{A}'$ terjadi adalah 1
(karena $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Kita juga dapat membuktikan bahwa $P(\emptyset) = 0$
karena $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Akibatnya, probabilitas dari setiap peristiwa $\mathcal{A}$
*dan* komplemennya $\mathcal{A}'$ terjadi secara bersamaan
adalah $P(\mathcal{A} \cap \mathcal{A}') = 0$.
Secara informal, ini memberi tahu kita bahwa peristiwa yang tidak mungkin
memiliki probabilitas nol untuk terjadi.


## Variabel Acak

Ketika kita berbicara tentang peristiwa seperti hasil lemparan dadu
yang menunjukkan angka ganjil atau lemparan koin pertama yang menunjukkan kepala,
kita sedang mengacu pada ide *variabel acak*.
Secara formal, variabel acak adalah pemetaan
dari ruang sampel yang mendasari
ke serangkaian nilai (yang mungkin banyak).
Anda mungkin bertanya-tanya bagaimana variabel acak
berbeda dari ruang sampel,
karena keduanya adalah kumpulan hasil.
Yang penting, variabel acak bisa jauh lebih kasar
daripada ruang sampel mentah.
Kita dapat mendefinisikan variabel acak biner seperti "lebih besar dari 0,5"
bahkan ketika ruang sampel yang mendasarinya tak terbatas,
misalnya, titik-titik pada segmen garis antara $0$ dan $1$.
Selain itu, beberapa variabel acak
dapat berbagi ruang sampel yang sama.
Misalnya, "apakah alarm rumah saya berbunyi"
dan "apakah rumah saya dibobol" adalah
keduanya variabel acak biner
yang berbagi ruang sampel yang mendasarinya.
Akibatnya, mengetahui nilai yang diambil oleh satu variabel acak
dapat memberi kita informasi tentang kemungkinan nilai dari variabel acak lainnya.
Mengetahui bahwa alarm berbunyi,
kita mungkin curiga bahwa rumah kemungkinan dibobol.

Setiap nilai yang diambil oleh variabel acak
berkaitan dengan bagian dari ruang sampel yang mendasarinya.
Jadi, kejadian di mana variabel acak $X$
mengambil nilai $v$, yang dinyatakan dengan $X=v$, adalah sebuah *peristiwa*
dan $P(X=v)$ menyatakan probabilitasnya.
Kadang notasi ini bisa menjadi rumit,
dan kita bisa menyederhanakan notasi ketika konteksnya jelas.
Misalnya, kita mungkin menggunakan $P(X)$ untuk merujuk secara umum
kepada *distribusi* $X$, yaitu,
fungsi yang memberi tahu kita probabilitas
bahwa $X$ mengambil nilai tertentu.
Terkadang kita menulis ekspresi
seperti $P(X,Y) = P(X) P(Y)$,
sebagai singkatan untuk menyatakan
pernyataan yang benar untuk semua nilai
yang dapat diambil oleh variabel acak $X$ dan $Y$, yaitu,
untuk semua $i,j$, berlaku bahwa $P(X=i \textrm{ dan } Y=j) = P(X=i)P(Y=j)$.
Kadang kita juga menyederhanakan notasi dengan menulis
$P(v)$ ketika variabel acak sudah jelas dari konteksnya.
Karena peristiwa dalam teori probabilitas adalah kumpulan hasil dari ruang sampel,
kita dapat menentukan rentang nilai yang dapat diambil oleh variabel acak.
Misalnya, $P(1 \leq X \leq 3)$ menyatakan probabilitas peristiwa $\{1 \leq X \leq 3\}$.

Perhatikan bahwa terdapat perbedaan halus
antara variabel acak *diskrit*,
seperti lemparan koin atau lemparan dadu,
dan variabel acak *kontinu*,
seperti berat badan dan tinggi badan seseorang
yang diambil secara acak dari populasi.
Dalam hal ini kita jarang benar-benar peduli
tentang tinggi badan seseorang yang persis.
Selain itu, jika kita melakukan pengukuran yang cukup tepat,
kita akan menemukan bahwa tidak ada dua orang di dunia ini
yang memiliki tinggi badan yang benar-benar sama.
Faktanya, dengan pengukuran yang sangat tepat,
Anda tidak akan pernah memiliki tinggi badan yang sama
saat Anda bangun tidur dan saat Anda pergi tidur.
Tidak ada gunanya bertanya tentang
probabilitas persis seseorang
berukuran 1.801392782910287192 meter.
Sebagai gantinya, kita biasanya lebih peduli
dengan dapat mengatakan apakah tinggi badan seseorang
berada dalam suatu interval tertentu,
misalnya antara 1,79 dan 1,81 meter.
Dalam kasus ini, kita bekerja dengan *densitas* probabilitas.
Tinggi badan yang persis 1,80 meter
tidak memiliki probabilitas, tetapi memiliki densitas yang tidak nol.
Untuk mengetahui probabilitas yang diberikan pada suatu interval,
kita harus mengambil *integral* dari densitas tersebut
pada interval tersebut.


## Beberapa Variabel Acak

Anda mungkin memperhatikan bahwa kita bahkan tidak dapat
melewati bagian sebelumnya tanpa
membuat pernyataan yang melibatkan interaksi
di antara beberapa variabel acak
(ingat bahwa $P(X,Y) = P(X) P(Y)$).
Sebagian besar machine learning
berkaitan dengan hubungan semacam itu.
Di sini, ruang sampel bisa berupa
populasi yang menjadi perhatian,
misalnya pelanggan yang bertransaksi dengan bisnis,
foto-foto di Internet,
atau protein yang dikenal oleh para ahli biologi.
Setiap variabel acak akan mewakili
(nilai yang tidak diketahui) dari atribut yang berbeda.
Setiap kali kita mengambil sampel individu dari populasi,
kita mengamati realisasi dari masing-masing variabel acak.
Karena nilai-nilai yang diambil oleh variabel acak
berkaitan dengan bagian-bagian dari ruang sampel
yang bisa tumpang tindih, sebagian tumpang tindih,
atau sepenuhnya tidak tumpang tindih,
mengetahui nilai yang diambil oleh satu variabel acak
dapat menyebabkan kita memperbarui keyakinan kita
tentang nilai apa yang mungkin diambil oleh variabel acak lainnya.
Jika seorang pasien masuk ke rumah sakit
dan kita mengamati bahwa mereka
mengalami kesulitan bernapas
dan kehilangan indera penciuman,
maka kita percaya bahwa mereka lebih mungkin
mengalami COVID-19 dibandingkan jika
mereka tidak memiliki masalah pernapasan
dan memiliki indera penciuman yang normal.

Ketika bekerja dengan beberapa variabel acak,
kita dapat membentuk peristiwa yang sesuai
dengan setiap kombinasi nilai
yang dapat diambil oleh variabel-variabel tersebut secara bersama-sama.
Fungsi probabilitas yang menetapkan
probabilitas untuk setiap kombinasi ini
(misalnya $A=a$ dan $B=b$)
disebut sebagai *probabilitas gabungan* (*joint probability*)
dan hanya mengembalikan probabilitas yang diberikan
untuk perpotongan dari bagian-bagian yang sesuai
dari ruang sampel.
*Probabilitas gabungan* yang diberikan kepada peristiwa
di mana variabel acak $A$ dan $B$
masing-masing mengambil nilai $a$ dan $b$,
dinotasikan dengan $P(A = a, B = b)$,
di mana koma menunjukkan "dan".
Perhatikan bahwa untuk setiap nilai $a$ dan $b$,
berlaku bahwa

$$P(A=a, B=b) \leq P(A=a) \textrm{ dan } P(A=a, B=b) \leq P(B = b),$$

karena agar $A=a$ dan $B=b$ terjadi,
$A=a$ harus terjadi *dan* $B=b$ juga harus terjadi.
Menariknya, probabilitas gabungan
memberi tahu kita semua yang dapat kita ketahui tentang
variabel acak ini dalam pengertian probabilistik,
dan dapat digunakan untuk memperoleh banyak kuantitas lain
yang berguna, termasuk mendapatkan kembali
distribusi individual $P(A)$ dan $P(B)$.
Untuk mendapatkan kembali $P(A=a)$ kita cukup menjumlahkan
$P(A=a, B=v)$ di atas semua nilai $v$
yang dapat diambil oleh variabel acak $B$:
$P(A=a) = \sum_v P(A=a, B=v)$.

Rasio $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
ternyata sangat penting.
Rasio ini disebut *probabilitas bersyarat* (*conditional probability*),
dan dinyatakan dengan simbol "$\mid$":

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

Ini memberi tahu kita probabilitas baru
yang terkait dengan peristiwa $B=b$,
setelah kita mensyaratkan bahwa $A=a$ terjadi.
Kita dapat memandang probabilitas bersyarat ini
sebagai membatasi perhatian hanya pada bagian
ruang sampel yang terkait dengan $A=a$
dan kemudian menormalisasi kembali sehingga
semua probabilitas berjumlah 1.
Probabilitas bersyarat
pada dasarnya adalah probabilitas biasa
dan dengan demikian memenuhi semua aksioma,
selama kita mensyaratkan semua istilah
pada peristiwa yang sama dan dengan demikian
membatasi perhatian pada ruang sampel yang sama.
Misalnya, untuk peristiwa yang saling lepas
$\mathcal{B}$ dan $\mathcal{B}'$, kita memiliki bahwa
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.



Dengan menggunakan definisi probabilitas bersyarat,
kita dapat menurunkan hasil terkenal yang disebut *teorema Bayes*.
Secara konstruksi, kita memiliki $P(A, B) = P(B\mid A) P(A)$
dan $P(A, B) = P(A\mid B) P(B)$.
Menggabungkan kedua persamaan menghasilkan
$P(B\mid A) P(A) = P(A\mid B) P(B)$ dan oleh karena itu

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$

Persamaan sederhana ini memiliki implikasi yang mendalam karena
memungkinkan kita untuk membalik urutan kondisional.
Jika kita tahu cara memperkirakan $P(B\mid A)$, $P(A)$, dan $P(B)$,
maka kita dapat memperkirakan $P(A\mid B)$.
Kita sering kali lebih mudah memperkirakan satu istilah secara langsung
tetapi tidak yang lainnya dan teorema Bayes dapat membantu di sini.
Sebagai contoh, jika kita tahu prevalensi gejala untuk penyakit tertentu,
dan prevalensi keseluruhan dari penyakit dan gejala masing-masing,
kita dapat menentukan seberapa besar kemungkinan seseorang
menderita penyakit tersebut berdasarkan gejalanya.
Dalam beberapa kasus, kita mungkin tidak memiliki akses langsung ke $P(B)$,
seperti prevalensi gejala.
Dalam kasus ini, versi sederhana dari teorema Bayes dapat digunakan:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Karena kita tahu bahwa $P(A \mid B)$ harus dinormalisasi menjadi $1$, yaitu $\sum_a P(A=a \mid B) = 1$,
kita dapat menggunakannya untuk menghitung

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

Dalam statistika Bayesian, kita menganggap seorang pengamat
memiliki beberapa keyakinan (subjektif) sebelumnya
tentang kemungkinan dari hipotesis yang ada
yang dikodekan dalam *prior* $P(H)$,
dan *fungsi likelihood* yang menunjukkan seberapa besar kemungkinan
kita mengamati nilai dari bukti yang dikumpulkan
untuk setiap hipotesis dalam kelas $P(E \mid H)$.
Teorema Bayes kemudian ditafsirkan sebagai memberi tahu kita
bagaimana memperbarui *prior* awal $P(H)$
dalam cahaya bukti yang ada $E$
untuk menghasilkan keyakinan *posterior*
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$.
Secara informal, ini dapat dinyatakan sebagai
"posterior sama dengan prior kali likelihood, dibagi dengan bukti".
Karena bukti $P(E)$ sama untuk semua hipotesis,
kita cukup menormalisasi di antara hipotesis.

Perhatikan bahwa $\sum_a P(A=a \mid B) = 1$ juga memungkinkan kita untuk *marginalisasi* atas variabel acak. Artinya, kita dapat menghilangkan variabel dari distribusi gabungan seperti $P(A, B)$. Setelah semua, kita memiliki bahwa

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

Independensi adalah konsep fundamental lainnya
yang membentuk tulang punggung dari
banyak ide penting dalam statistika.
Singkatnya, dua variabel adalah *independen*
jika mensyaratkan nilai $A$ tidak menyebabkan
perubahan apa pun pada distribusi probabilitas
yang terkait dengan $B$ dan sebaliknya.
Secara formal, independensi, dilambangkan dengan $A \perp B$,
mensyaratkan bahwa $P(A \mid B) = P(A)$ dan, akibatnya,
bahwa $P(A,B) = P(A \mid B) P(B) = P(A) P(B)$.
Independensi sering kali merupakan asumsi yang tepat.
Misalnya, jika variabel acak $A$
mewakili hasil dari lemparan satu koin adil
dan variabel acak $B$
mewakili hasil dari lemparan koin lainnya,
maka mengetahui apakah $A$ menunjukkan kepala
tidak akan mempengaruhi probabilitas
$B$ menunjukkan kepala.

Independensi sangat berguna ketika terjadi pada penarikan berturut-turut
dari data kita dari beberapa distribusi yang mendasari
(memungkinkan kita membuat kesimpulan statistik yang kuat)
atau ketika terjadi di antara berbagai variabel dalam data kita,
memungkinkan kita untuk bekerja dengan model yang lebih sederhana
yang mengkodekan struktur independensi ini.
Di sisi lain, memperkirakan ketergantungan
di antara variabel acak sering kali menjadi tujuan utama dari pembelajaran.
Kita peduli untuk memperkirakan probabilitas penyakit berdasarkan gejala
terutama karena kita percaya
bahwa penyakit dan gejala *tidak* independen.



Perhatikan bahwa karena probabilitas bersyarat adalah probabilitas yang sebenarnya,
konsep independensi dan ketergantungan juga berlaku untuk mereka.
Dua variabel acak $A$ dan $B$ adalah *kondisional independen*
diberikan variabel ketiga $C$ jika dan hanya jika $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Menariknya, dua variabel dapat independen secara umum
tetapi menjadi bergantung ketika dikondisikan pada variabel ketiga.
Hal ini sering terjadi ketika dua variabel acak $A$ dan $B$
berkaitan dengan penyebab dari variabel ketiga $C$.
Misalnya, patah tulang dan kanker paru-paru mungkin independen
dalam populasi umum tetapi jika kita mensyaratkan bahwa orang tersebut berada di rumah sakit,
maka kita mungkin menemukan bahwa patah tulang berkorelasi negatif dengan kanker paru-paru.
Hal ini terjadi karena patah tulang *menjelaskan* mengapa seseorang berada di rumah sakit
dan dengan demikian menurunkan probabilitas bahwa mereka dirawat di rumah sakit karena menderita kanker paru-paru.

Sebaliknya, dua variabel acak yang bergantung
dapat menjadi independen ketika dikondisikan pada variabel ketiga.
Ini sering terjadi ketika dua kejadian yang tidak terkait satu sama lain
memiliki penyebab yang sama.
Ukuran sepatu dan tingkat membaca sangat berkorelasi
di antara siswa sekolah dasar,
tetapi korelasi ini menghilang jika kita mensyaratkan pada usia.



## Sebuah Contoh
:label:`subsec_probability_hiv_app`

Mari kita uji keterampilan kita.
Misalkan seorang dokter memberikan tes HIV kepada seorang pasien.
Tes ini cukup akurat dan hanya gagal dengan probabilitas 1%
jika pasien sehat tetapi dilaporkan sakit,
yaitu, pasien sehat diuji positif dalam 1% kasus.
Selain itu, tes ini tidak pernah gagal mendeteksi HIV jika pasien benar-benar mengidapnya.
Kita menggunakan $D_1 \in \{0, 1\}$ untuk menunjukkan diagnosis
($0$ jika negatif dan $1$ jika positif)
dan $H \in \{0, 1\}$ untuk menunjukkan status HIV.

| Probabilitas Bersyarat | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

Perhatikan bahwa jumlah dari setiap kolom adalah 1 (tetapi jumlah dari setiap baris tidak),
karena mereka adalah probabilitas bersyarat.
Mari kita hitung probabilitas pasien menderita HIV
jika tesnya kembali positif, yaitu $P(H = 1 \mid D_1 = 1)$.
Secara intuitif, ini akan bergantung pada seberapa umum penyakit tersebut,
karena ini memengaruhi jumlah alarm palsu.
Misalkan populasi cukup bebas dari penyakit ini, misalnya $P(H=1) = 0.0015$.
Untuk menerapkan teorema Bayes, kita perlu menerapkan marginalisasi
untuk menentukan

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Ini membawa kita pada

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

Dengan kata lain, hanya ada kemungkinan 13.06%
bahwa pasien benar-benar menderita HIV,
meskipun tes tersebut cukup akurat.
Seperti yang kita lihat, probabilitas dapat menjadi tidak intuitif.
Apa yang harus dilakukan seorang pasien setelah menerima kabar mengerikan ini?
Kemungkinan besar, pasien akan meminta dokter
untuk melakukan tes lain untuk mendapatkan kejelasan.
Tes kedua memiliki karakteristik yang berbeda
dan tidak sebaik tes pertama.

| Probabilitas Bersyarat | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

Sayangnya, tes kedua juga kembali positif.
Mari kita hitung probabilitas yang diperlukan untuk menerapkan teorema Bayes
dengan mengasumsikan independensi kondisional:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

Sekarang kita dapat menerapkan marginalisasi untuk mendapatkan probabilitas
bahwa kedua tes tersebut kembali positif:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Akhirnya, probabilitas pasien menderita HIV dengan asumsi bahwa kedua tes positif adalah

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$


Artinya, tes kedua memungkinkan kita untuk mendapatkan keyakinan yang jauh lebih tinggi bahwa ada sesuatu yang salah.
Meskipun tes kedua jauh kurang akurat dibandingkan tes pertama,
namun tes tersebut masih secara signifikan meningkatkan estimasi kita.
Asumsi bahwa kedua tes bersifat independen secara kondisional satu sama lain
sangat penting bagi kemampuan kita untuk menghasilkan estimasi yang lebih akurat.
Ambil contoh ekstrem di mana kita melakukan tes yang sama dua kali.
Dalam situasi ini, kita akan mengharapkan hasil yang sama pada kedua kali,
sehingga tidak ada wawasan tambahan yang diperoleh dari menjalankan tes yang sama lagi.
Pembaca yang cermat mungkin memperhatikan bahwa diagnosis berperilaku
seperti sebuah *classifier* yang terlihat jelas
di mana kemampuan kita untuk memutuskan apakah seorang pasien sehat
meningkat saat kita mendapatkan lebih banyak fitur (hasil tes).

## Ekspektasi

Seringkali, membuat keputusan memerlukan tidak hanya melihat
pada probabilitas yang diberikan untuk setiap kejadian,
tetapi juga menggabungkannya menjadi agregat yang berguna
yang dapat memberikan kita panduan.
Misalnya, ketika variabel acak mengambil nilai skalar kontinu,
kita sering peduli untuk mengetahui nilai apa yang diharapkan *rata-rata*.
Kuantitas ini secara formal disebut *ekspektasi*.
Jika kita melakukan investasi,
hal pertama yang menjadi perhatian
mungkin adalah pengembalian yang bisa kita harapkan,
dengan merata-ratakan semua hasil yang mungkin terjadi
(dan memberi bobot dengan probabilitas yang sesuai).
Misalnya, dengan probabilitas 50%,
investasi bisa gagal total,
dengan probabilitas 40% bisa memberikan pengembalian 2$\times$,
dan dengan probabilitas 10% bisa memberikan pengembalian 10$\times$.
Untuk menghitung pengembalian yang diharapkan,
kita menjumlahkan semua pengembalian, mengalikan setiap pengembalian
dengan probabilitas terjadinya.
Ini menghasilkan ekspektasi
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$.
Dengan demikian, pengembalian yang diharapkan adalah 1.8$\times$.


Secara umum, *ekspektasi* (atau rata-rata)
dari variabel acak $X$ didefinisikan sebagai

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Demikian pula, untuk densitas kita mendapatkan $E[X] = \int x \;dp(x)$.
Kadang-kadang kita tertarik pada nilai ekspektasi
dari beberapa fungsi $x$.
Kita dapat menghitung ekspektasi ini sebagai

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ dan } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

untuk probabilitas diskrit dan densitas, masing-masing.
Kembali ke contoh investasi di atas,
$f$ mungkin merupakan *utilitas* (kebahagiaan)
yang terkait dengan pengembalian.
Ahli ekonomi perilaku telah lama mencatat
bahwa orang mengaitkan ketidaknyamanan yang lebih besar
dengan kehilangan uang daripada kegunaan yang diperoleh
dari menghasilkan satu dolar relatif terhadap dasar mereka.
Selain itu, nilai uang cenderung sub-linear.
Memiliki uang 100 ribu dolar versus nol dolar
dapat membuat perbedaan antara mampu membayar sewa,
makan dengan baik, dan menikmati layanan kesehatan yang berkualitas
dibandingkan dengan mengalami tunawisma.
Di sisi lain, peningkatan karena memiliki
200 ribu dolar dibandingkan 100 ribu dolar kurang dramatis.
Pemikiran seperti ini memotivasi klise
bahwa "utilitas uang adalah logaritmik".

Jika utilitas yang terkait dengan kerugian total adalah $-1$,
dan utilitas yang terkait dengan pengembalian $1$, $2$, dan $10$
adalah $1$, $2$, dan $4$, masing-masing,
maka kebahagiaan yang diharapkan dari investasi
adalah $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
(kehilangan utilitas yang diharapkan sebesar 30%).
Jika memang ini adalah fungsi utilitas Anda,
mungkin yang terbaik adalah menyimpan uang di bank.

Untuk keputusan finansial,
kita mungkin juga ingin mengukur
seberapa *berisiko* suatu investasi.
Di sini, kita tidak hanya peduli pada nilai ekspektasi
tetapi seberapa besar nilai yang sebenarnya cenderung *bervariasi*
relatif terhadap nilai ini.
Perhatikan bahwa kita tidak bisa hanya mengambil
ekspektasi dari perbedaan antara
nilai sebenarnya dan nilai ekspektasi.
Ini karena ekspektasi dari suatu perbedaan
adalah perbedaan dari ekspektasi,
yaitu, $E[X - E[X]] = E[X] - E[E[X]] = 0$.
Namun, kita dapat melihat ekspektasi
dari fungsi apa pun yang tidak negatif dari perbedaan ini.
*Varians* dari variabel acak dihitung dengan melihat
nilai harapan dari perbedaan *kuadrat*:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Persamaan di atas dihasilkan dengan memperluas
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
dan mengambil ekspektasi untuk setiap suku.
Akar kuadrat dari varians adalah kuantitas lain
yang berguna yang disebut *simpangan baku*.
Meskipun simpangan baku dan varians
mengandung informasi yang sama (keduanya dapat dihitung satu sama lain),
simpangan baku memiliki sifat yang baik
yaitu diekspresikan dalam satuan yang sama
dengan kuantitas asli yang diwakili
oleh variabel acak.

Terakhir, varians dari fungsi
dari suatu variabel acak
didefinisikan secara analogi sebagai

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$




Kembali ke contoh investasi kita,
sekarang kita dapat menghitung varians dari investasi.
Varians diberikan oleh $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$.
Untuk semua maksud dan tujuan, ini adalah investasi yang berisiko.
Perhatikan bahwa secara konvensi matematis, rata-rata dan varians
sering dirujuk sebagai $\mu$ dan $\sigma^2$.
Hal ini terutama berlaku ketika kita menggunakannya
untuk memparametrisasi distribusi Gaussian.

Dengan cara yang sama saat kita memperkenalkan ekspektasi
dan varians untuk variabel acak *skalar*,
kita dapat melakukannya untuk variabel yang bernilai vektor.
Ekspektasi mudah dilakukan, karena kita dapat menerapkannya pada setiap elemen.
Misalnya, $\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
memiliki koordinat $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
*Covarians* lebih rumit.
Kita mendefinisikannya dengan mengambil ekspektasi dari *produk luar*
dari perbedaan antara variabel acak dan rata-ratanya:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$



Matriks ini, $\boldsymbol{\Sigma}$, dikenal sebagai matriks kovarians.
Cara mudah untuk melihat efeknya adalah dengan mempertimbangkan sebuah vektor $\mathbf{v}$
yang ukurannya sama dengan $\mathbf{x}$.
Dari situ, kita dapat menyimpulkan bahwa

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$


Dengan demikian, $\boldsymbol{\Sigma}$ memungkinkan kita untuk menghitung varians
untuk setiap fungsi linier dari $\mathbf{x}$
hanya dengan perkalian matriks sederhana.
Elemen-elemen di luar diagonal menunjukkan seberapa berkorelasi koordinat-koordinat tersebut:
nilai 0 berarti tidak ada korelasi,
sedangkan nilai positif yang lebih besar
berarti korelasi yang lebih kuat.




## Diskusi

Dalam machine learning, ada banyak hal yang bisa kita ragu-ragukan!
Kita bisa merasa tidak pasti tentang nilai label yang diberikan sebuah input.
Kita bisa tidak pasti tentang nilai estimasi dari sebuah parameter.
Bahkan, kita bisa ragu apakah data yang tiba saat penerapan
berasal dari distribusi yang sama dengan data pelatihan.

Dengan *ketidakpastian aleatorik* (*aleatoric uncertainty*), kita merujuk pada ketidakpastian
yang melekat pada masalah tersebut, yang disebabkan oleh keacakan sejati
yang tidak dapat dijelaskan oleh variabel yang diamati.
Dengan *ketidakpastian epistemik* (*epistemic uncertainty*), kita merujuk pada ketidakpastian
terhadap parameter model, jenis ketidakpastian yang dapat kita harapkan untuk berkurang
dengan mengumpulkan lebih banyak data.
Kita mungkin memiliki ketidakpastian epistemik
tentang probabilitas sebuah koin menunjukkan sisi kepala,
tetapi bahkan setelah mengetahui probabilitas ini,
kita masih memiliki ketidakpastian aleatorik
tentang hasil lemparan di masa depan.
Tidak peduli seberapa lama kita mengamati seseorang melempar koin yang adil,
kita tidak akan pernah lebih atau kurang dari 50% yakin
bahwa lemparan berikutnya akan menunjukkan sisi kepala.
Istilah-istilah ini berasal dari pemodelan mekanis
(lihat misalnya, :citet:`Der-Kiureghian.Ditlevsen.2009` untuk ulasan tentang aspek [kuantifikasi ketidakpastian](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
Namun, perlu dicatat bahwa istilah-istilah ini merupakan penyederhanaan bahasa.
Istilah *epistemik* mengacu pada segala sesuatu yang berkaitan dengan *pengetahuan*
dan, dalam arti filosofis, semua ketidakpastian adalah epistemik.

Kita melihat bahwa pengambilan sampel data dari distribusi probabilitas yang tidak diketahui
dapat memberi kita informasi yang dapat digunakan untuk mengestimasi
parameter dari distribusi pembangkit data.
Namun demikian, laju di mana hal ini dimungkinkan bisa cukup lambat.
Dalam contoh lemparan koin kita (dan banyak kasus lainnya)
kita tidak bisa lebih baik daripada merancang estimator
yang berkonvergen pada laju $1/\sqrt{n}$,
di mana $n$ adalah ukuran sampel (misalnya, jumlah lemparan).
Ini berarti bahwa dengan meningkat dari 10 menjadi 1000 observasi (biasanya tugas yang sangat mungkin tercapai)
kita melihat pengurangan ketidakpastian sepuluh kali lipat,
sementara 1000 observasi berikutnya relatif membantu sedikit,
hanya menawarkan pengurangan sebesar 1,41 kali.
Ini adalah fitur persisten dalam machine learning:
meskipun ada perolehan yang mudah di awal, membutuhkan sangat banyak data,
dan sering kali membutuhkan komputasi yang besar, untuk memperoleh perbaikan lebih lanjut.
Untuk tinjauan empiris fakta ini pada model bahasa berskala besar, lihat :citet:`Revels.Lubin.Papamarkou.2016`.

Kita juga memperjelas bahasa dan alat kita untuk pemodelan statistik.
Dalam proses itu, kita mempelajari tentang probabilitas bersyarat
dan tentang salah satu persamaan terpenting dalam statistik---teorema Bayes.
Ini adalah alat yang efektif untuk memisahkan informasi yang disampaikan oleh data
melalui sebuah istilah likelihood $P(B \mid A)$ yang menunjukkan
seberapa baik observasi $B$ sesuai dengan pilihan parameter $A$,
dan probabilitas prior $P(A)$ yang mengatur seberapa mungkin
sebuah pilihan $A$ sejak awal.
Secara khusus, kita melihat bagaimana aturan ini dapat diterapkan
untuk menetapkan probabilitas pada diagnosis,
berdasarkan efektivitas tes *dan*
prevalensi penyakit itu sendiri (yaitu, prior kita).

Terakhir, kita memperkenalkan serangkaian pertanyaan nontrivial pertama
tentang efek dari distribusi probabilitas tertentu,
yaitu ekspektasi dan varians.
Meskipun ada banyak ekspektasi lainnya selain yang linear dan kuadratik
untuk distribusi probabilitas, dua ekspektasi ini saja sudah memberikan
banyak wawasan tentang kemungkinan perilaku distribusi tersebut.
Sebagai contoh, [ketaksamaan Chebyshev](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
menyatakan bahwa $P(|X - \mu| \geq k \sigma) \leq 1/k^2$,
di mana $\mu$ adalah ekspektasi, $\sigma^2$ adalah varians dari distribusi,
dan $k > 1$ adalah parameter kepercayaan yang kita pilih.
Ini menunjukkan bahwa nilai yang diambil dari suatu distribusi berada
dengan probabilitas minimal 50%
dalam interval $[-\sqrt{2} \sigma, \sqrt{2} \sigma]$
yang berpusat pada ekspektasi.


## Latihan

1. Berikan contoh di mana mengamati lebih banyak data dapat mengurangi ketidakpastian tentang hasil hingga ke tingkat yang sangat rendah.
2. Berikan contoh di mana mengamati lebih banyak data hanya akan mengurangi jumlah ketidakpastian hingga batas tertentu dan kemudian tidak akan berkurang lebih jauh. Jelaskan mengapa hal ini terjadi dan di mana Anda memperkirakan batas ini terjadi.
3. Kita telah menunjukkan secara empiris konvergensi menuju nilai rata-rata dalam lemparan koin. Hitung varians dari estimasi probabilitas bahwa kita melihat sisi kepala setelah mengambil $n$ sampel.
    1. Bagaimana skala varians dengan jumlah pengamatan?
    2. Gunakan ketaksamaan Chebyshev untuk membatasi deviasi dari ekspektasi.
    3. Bagaimana hubungannya dengan teorema limit pusat?
4. Asumsikan bahwa kita mengambil $m$ sampel $x_i$ dari distribusi probabilitas dengan nilai rata-rata nol dan varians satu. Hitung rata-rata $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Bisakah kita menerapkan ketaksamaan Chebyshev untuk setiap $z_m$ secara independen? Mengapa tidak?
5. Diberikan dua peristiwa dengan probabilitas $P(\mathcal{A})$ dan $P(\mathcal{B})$, hitung batas atas dan batas bawah untuk $P(\mathcal{A} \cup \mathcal{B})$ dan $P(\mathcal{A} \cap \mathcal{B})$. Petunjuk: gambarkan situasi menggunakan [diagram Venn](https://en.wikipedia.org/wiki/Venn_diagram).
6. Asumsikan bahwa kita memiliki urutan variabel acak, misalnya $A$, $B$, dan $C$, di mana $B$ hanya bergantung pada $A$, dan $C$ hanya bergantung pada $B$. Bisakah Anda menyederhanakan probabilitas gabungan $P(A, B, C)$? Petunjuk: ini adalah [rantai Markov](https://en.wikipedia.org/wiki/Markov_chain).
7. Dalam :numref:`subsec_probability_hiv_app`, asumsikan bahwa hasil dari kedua tes tidak independen. Secara khusus, asumsikan bahwa masing-masing tes memiliki tingkat positif palsu sebesar 10% dan tingkat negatif palsu sebesar 1%. Artinya, asumsikan bahwa $P(D =1 \mid H=0) = 0.1$ dan bahwa $P(D = 0 \mid H=1) = 0.01$. Selain itu, asumsikan bahwa untuk $H = 1$ (terinfeksi) hasil tes bersifat independen secara kondisional, yaitu, $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$, tetapi untuk pasien sehat hasilnya terkait melalui $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Susun tabel probabilitas gabungan untuk $D_1$ dan $D_2$, mengingat $H=0$ berdasarkan informasi yang Anda miliki sejauh ini.
    2. Turunkan probabilitas bahwa pasien terinfeksi ($H=1$) setelah satu tes kembali positif. Anda dapat mengasumsikan probabilitas dasar yang sama $P(H=1) = 0.0015$ seperti sebelumnya.
    3. Turunkan probabilitas bahwa pasien terinfeksi ($H=1$) setelah kedua tes kembali positif.
8. Asumsikan bahwa Anda adalah seorang manajer aset untuk sebuah bank investasi dan Anda memiliki pilihan saham $s_i$ untuk diinvestasikan. Portofolio Anda harus berjumlah $1$ dengan bobot $\alpha_i$ untuk setiap saham. Saham-saham tersebut memiliki rata-rata pengembalian $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ dan kovarians $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Hitung pengembalian yang diharapkan untuk portofolio tertentu $\boldsymbol{\alpha}$.
    2. Jika Anda ingin memaksimalkan pengembalian portofolio, bagaimana Anda harus memilih investasi Anda?
    3. Hitung *varians* dari portofolio.
    4. Formulasikan masalah optimisasi untuk memaksimalkan pengembalian dengan mempertahankan varians di bawah batas atas. Ini adalah model [portofolio Markovitz](https://en.wikipedia.org/wiki/Markowitz_model) pemenang Hadiah Nobel :cite:`Mangram.2013`. Untuk menyelesaikannya, Anda memerlukan penyelesai pemrograman kuadratik, yang berada di luar cakupan buku ini.





:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17971)
:end_tab:
