# Model Bahasa
:label:`sec_language-model`

Di :numref:`sec_text-sequence`, kita telah melihat cara memetakan urutan teks menjadi token, di mana token-token ini dapat dilihat sebagai urutan observasi diskrit seperti kata atau karakter. Misalkan token-token dalam urutan teks dengan panjang $T$ adalah $x_1, x_2, \ldots, x_T$.
Tujuan dari *model bahasa* adalah untuk memperkirakan probabilitas gabungan dari seluruh urutan:

$$P(x_1, x_2, \ldots, x_T),$$

di mana alat-alat statistik
di :numref:`sec_sequence`
dapat diterapkan.

Model bahasa sangat bermanfaat. Misalnya, model bahasa yang ideal seharusnya dapat menghasilkan teks alami dengan sendirinya, hanya dengan mengambil satu token pada satu waktu $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Berbeda dengan monyet yang menggunakan mesin tik, semua teks yang dihasilkan oleh model ini akan tampak sebagai bahasa alami, misalnya, teks bahasa Inggris. Selain itu, model ini akan cukup untuk menghasilkan dialog yang bermakna, hanya dengan memberikan teks berdasarkan fragmen dialog sebelumnya.
Jelas kita masih jauh dari merancang sistem seperti itu, karena model ini perlu *memahami* teks, bukan hanya menghasilkan konten yang masuk akal secara tata bahasa.

Meskipun demikian, model bahasa sudah sangat berguna bahkan dalam bentuk yang terbatas.
Sebagai contoh, frasa "to recognize speech" dan "to wreck a nice beach" terdengar sangat mirip.
Hal ini dapat menyebabkan ambiguitas dalam pengenalan suara,
yang dapat dengan mudah diselesaikan dengan model bahasa yang menolak terjemahan kedua sebagai sesuatu yang tidak masuk akal.
Demikian pula, dalam algoritma untuk merangkum dokumen,
mengetahui bahwa "dog bites man" jauh lebih umum daripada "man bites dog" atau bahwa "I want to eat grandma" adalah pernyataan yang cukup mengganggu, sementara "I want to eat, grandma" jauh lebih ramah, adalah sesuatu yang sangat berharga.


```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Mempelajari Model Bahasa

Pertanyaan yang jelas adalah bagaimana kita seharusnya memodelkan sebuah dokumen, atau bahkan sebuah urutan token. 
Misalkan kita melakukan tokenisasi data teks pada tingkat kata.
Mari kita mulai dengan menerapkan aturan dasar probabilitas:


$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Sebagai contoh, 
probabilitas dari urutan teks yang mengandung empat kata dapat diberikan sebagai:

$$\begin{aligned}&P(\textrm{deep}, \textrm{learning}, \textrm{is}, \textrm{fun}) \\
=&P(\textrm{deep}) P(\textrm{learning}  \mid  \textrm{deep}) P(\textrm{is}  \mid  \textrm{deep}, \textrm{learning}) P(\textrm{fun}  \mid  \textrm{deep}, \textrm{learning}, \textrm{is}).\end{aligned}$$

### Model Markov dan $n$-gram
:label:`subsec_markov-models-and-n-grams`

Di antara analisis model urutan di :numref:`sec_sequence`,
mari kita terapkan model Markov pada pemodelan bahasa.
Sebuah distribusi pada urutan memenuhi properti Markov ordo pertama jika $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. Ordo yang lebih tinggi menunjukkan ketergantungan yang lebih panjang. Ini mengarah pada beberapa pendekatan yang dapat kita terapkan untuk memodelkan sebuah urutan:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Formula probabilitas yang melibatkan satu, dua, dan tiga variabel biasanya disebut sebagai model *unigram*, *bigram*, dan *trigram*. 
Untuk menghitung model bahasa, kita perlu menghitung
probabilitas kata dan probabilitas bersyarat suatu kata dengan
beberapa kata sebelumnya.
Perhatikan bahwa
probabilitas semacam itu merupakan
parameter model bahasa.



### Frekuensi Kata

Di sini, kita mengasumsikan bahwa dataset pelatihan adalah korpus teks yang besar, seperti semua
entri Wikipedia, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg),
dan semua teks yang diposting di
internet.
Probabilitas kata dapat dihitung dari frekuensi kata relatif dari kata yang diberikan dalam dataset pelatihan.
Sebagai contoh, estimasi $\hat{P}(\textrm{deep})$ dapat dihitung sebagai
probabilitas dari setiap kalimat yang dimulai dengan kata "deep". Pendekatan
yang sedikit kurang akurat adalah dengan menghitung semua kemunculan
kata "deep" dan membaginya dengan jumlah total kata dalam
korpus.
Ini bekerja cukup baik, terutama untuk kata-kata yang sering muncul. Selanjutnya, kita bisa mencoba mengestimasi

$$\hat{P}(\textrm{learning} \mid \textrm{deep}) = \frac{n(\textrm{deep, learning})}{n(\textrm{deep})},$$

di mana $n(x)$ dan $n(x, x')$ adalah jumlah kemunculan dari satu kata
dan pasangan kata berturut-turut, masing-masing.
Sayangnya, 
mengestimasi probabilitas pasangan kata lebih sulit, karena
kemunculan "deep learning" jauh lebih jarang. 
Secara khusus, untuk beberapa kombinasi kata yang tidak biasa, mungkin sulit untuk
menemukan cukup banyak kemunculan untuk mendapatkan estimasi yang akurat.
Seperti yang ditunjukkan oleh hasil empiris di :numref:`subsec_natural-lang-stat`,
situasinya menjadi lebih buruk untuk kombinasi tiga kata atau lebih.
Akan ada banyak kombinasi tiga kata yang masuk akal namun kemungkinan besar tidak ada dalam dataset kita.
Kecuali kita memberikan solusi untuk memberi kombinasi kata tersebut jumlah yang tidak nol, kita tidak akan dapat menggunakannya dalam model bahasa. Jika datasetnya kecil atau jika kata-katanya sangat jarang, mungkin kita tidak menemukan satu pun dari kombinasi tersebut.


### Laplace Smoothing

Salah satu strategi umum adalah melakukan beberapa bentuk *Laplace smoothing*.
Solusinya adalah menambahkan konstanta kecil pada semua jumlah kemunculan.
Misalkan $n$ adalah jumlah total kata dalam
set pelatihan
dan $m$ adalah jumlah kata unik.
Solusi ini membantu menangani kata yang muncul hanya sekali, misalnya melalui

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Di sini $\epsilon_1,\epsilon_2$, dan $\epsilon_3$ adalah *hyperparameter*.
Ambil contoh $\epsilon_1$: ketika $\epsilon_1 = 0$, tidak ada *smoothing* yang diterapkan; 
ketika $\epsilon_1$ mendekati tak terhingga positif, 
$\hat{P}(x)$ mendekati probabilitas seragam $1/m$. 
Pendekatan di atas adalah varian yang cukup sederhana dari apa yang dapat dicapai oleh teknik lainnya :cite:`Wood.Gasthaus.Archambeau.ea.2011`.

Sayangnya, model seperti ini menjadi sulit digunakan dengan cepat
karena alasan berikut.
Pertama, 
seperti dibahas di :numref:`subsec_natural-lang-stat`,
banyak $n$-gram sangat jarang muncul, 
membuat *Laplace smoothing* kurang cocok untuk pemodelan bahasa.
Kedua, kita perlu menyimpan semua jumlah kemunculan.
Ketiga, model ini sepenuhnya mengabaikan makna kata. Sebagai contoh, "cat" dan "feline" seharusnya muncul dalam konteks yang berhubungan.
Sangat sulit untuk menyesuaikan model semacam ini dengan konteks tambahan, 
sementara model bahasa berbasis pembelajaran mendalam sangat cocok untuk mempertimbangkan hal tersebut.
Terakhir, urutan kata yang panjang hampir pasti akan baru, sehingga model yang hanya menghitung frekuensi urutan kata yang telah dilihat sebelumnya cenderung berkinerja buruk di sini.
Oleh karena itu, kita akan fokus menggunakan jaringan saraf untuk pemodelan bahasa pada sisa bab ini.


## Perplexity
:label:`subsec_perplexity`

Selanjutnya, mari kita bahas bagaimana cara mengukur kualitas model bahasa, yang nantinya akan kita gunakan untuk mengevaluasi model kita di bagian selanjutnya.
Salah satu caranya adalah dengan melihat seberapa mengejutkan teks tersebut.
Model bahasa yang baik dapat memprediksi dengan akurasi tinggi token yang akan datang.
Pertimbangkan kelanjutan berikut dari frasa "It is raining", yang diusulkan oleh berbagai model bahasa:

1. "It is raining outside"
2. "It is raining banana tree"
3. "It is raining piouw;kcj pwepoiut"

Dari segi kualitas, Contoh 1 jelas yang terbaik. Kata-katanya masuk akal dan koheren secara logis.
Meskipun tidak sepenuhnya mencerminkan kata apa yang secara semantis mengikuti frasa tersebut ("in San Francisco" atau "in winter" akan menjadi kelanjutan yang masuk akal), model ini mampu menangkap jenis kata yang mungkin muncul berikutnya.
Contoh 2 jauh lebih buruk karena menghasilkan kelanjutan yang tidak masuk akal. Namun, setidaknya model telah belajar cara mengeja kata dan beberapa korelasi antara kata-kata.
Terakhir, Contoh 3 menunjukkan model yang dilatih dengan buruk dan tidak sesuai dengan data.

Kita dapat mengukur kualitas model dengan menghitung *likelihood* dari urutan tersebut.
Sayangnya, ini adalah angka yang sulit dipahami dan sulit dibandingkan.
Lagi pula, urutan yang lebih pendek jauh lebih mungkin terjadi daripada urutan yang lebih panjang,
sehingga mengevaluasi model pada karya utama Tolstoy 
*War and Peace* akan menghasilkan *likelihood* yang jauh lebih kecil daripada, misalnya, pada novel pendek Saint-Exupery *The Little Prince*. Yang kurang adalah semacam rata-rata.

Teori informasi sangat membantu di sini.
Kita telah mendefinisikan entropi, *surprisal*, dan *cross-entropy*
saat memperkenalkan regresi softmax
(:numref:`subsec_info_theory_basics`).
Jika kita ingin mengompresi teks, kita dapat menanyakan tentang
memperkirakan token berikutnya berdasarkan rangkaian token saat ini.
Model bahasa yang lebih baik memungkinkan kita memprediksi token berikutnya dengan lebih akurat.
Dengan demikian, model tersebut memungkinkan kita untuk menggunakan lebih sedikit bit dalam mengompresi urutan tersebut.
Jadi kita dapat mengukurnya melalui *cross-entropy loss* yang dirata-ratakan
untuk semua $n$ token dalam urutan:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

di mana $P$ diberikan oleh model bahasa dan $x_t$ adalah token aktual yang diamati pada langkah waktu $t$ dalam urutan.
Ini membuat kinerja pada dokumen dengan panjang yang berbeda menjadi sebanding. Untuk alasan historis, para peneliti dalam pemrosesan bahasa alami lebih suka menggunakan kuantitas yang disebut *perplexity*. Singkatnya, *perplexity* adalah eksponensial dari :eqref:`eq_avg_ce_for_lm`:

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

Perplexity dapat dipahami sebagai kebalikan dari rata-rata geometrik jumlah pilihan nyata yang kita miliki saat memutuskan token mana yang akan dipilih berikutnya. Mari kita lihat beberapa kasus:

* Dalam skenario terbaik, model selalu memperkirakan probabilitas token target sebagai 1. Dalam kasus ini, *perplexity* model adalah 1.
* Dalam skenario terburuk, model selalu memperkirakan probabilitas token target sebagai 0. Dalam situasi ini, *perplexity* adalah tak terhingga positif.
* Pada *baseline*, model memperkirakan distribusi seragam untuk semua token yang tersedia dalam kosakata. Dalam kasus ini, *perplexity* sama dengan jumlah token unik dalam kosakata. Faktanya, jika kita ingin menyimpan urutan tanpa kompresi apa pun, ini akan menjadi yang terbaik yang dapat kita lakukan untuk mengkodekannya. Oleh karena itu, ini memberikan batas atas yang tidak sepele yang harus dilampaui oleh model yang berguna.


## Memisahkan Urutan
:label:`subsec_partitioning-seqs`

Kita akan merancang model bahasa menggunakan jaringan saraf
dan menggunakan *perplexity* untuk mengevaluasi 
seberapa baik model tersebut dalam 
memprediksi token berikutnya berdasarkan kumpulan token saat ini
dalam urutan teks.
Sebelum memperkenalkan modelnya,
misalkan model ini
memproses *minibatch* dari urutan dengan panjang yang telah ditentukan
pada satu waktu.
Sekarang pertanyaannya adalah bagaimana [**membaca *minibatch* dari urutan input dan urutan target secara acak**].

Misalkan dataset berbentuk urutan $T$ indeks token dalam `corpus`.
Kita akan
membaginya
menjadi sub-urutan, di mana setiap sub-urutan memiliki $n$ token (time step).
Untuk mengiterasi 
(almost) seluruh token dari dataset 
pada setiap epoch
dan mendapatkan semua sub-urutan dengan panjang $n$ yang mungkin,
kita dapat memperkenalkan elemen acak.
Secara lebih konkret,
di awal setiap epoch,
buang token pertama sebanyak $d$,
di mana $d\in [0,n)$ diambil secara acak dengan distribusi seragam.
Sisa urutan
kemudian dibagi
menjadi $m=\lfloor (T-d)/n \rfloor$ sub-urutan.
Dinyatakan sebagai $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$ sub-urutan sepanjang $n$ yang dimulai dari token $x_t$ pada *time step* $t$. 
Hasilnya, $m$ sub-urutan yang dibagi adalah
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}.$
Setiap sub-urutan akan digunakan sebagai urutan input ke dalam model bahasa.

Untuk pemodelan bahasa,
tujuannya adalah untuk memprediksi token berikutnya berdasarkan token-token yang telah kita lihat sejauh ini; oleh karena itu, targetnya (label) adalah urutan asli, digeser satu token.
Urutan target untuk setiap urutan input $\mathbf x_t$
adalah $\mathbf x_{t+1}$ dengan panjang $n$.

![Memperoleh lima pasang urutan input dan urutan target dari sub-urutan sepanjang 5 yang telah dipartisi.](../img/lang-model-data.svg) 
:label:`fig_lang_model_data`

:numref:`fig_lang_model_data` menunjukkan contoh memperoleh lima pasang urutan input dan urutan target dengan $n=5$ dan $d=2$.


```{.python .input  n=5}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = d2l.tensor([corpus[i:i+num_steps+1] 
                        for i in range(len(corpus)-num_steps)])
    self.X, self.Y = array[:,:-1], array[:,1:]
```

Untuk melatih model bahasa,
kita akan secara acak mengambil sampel 
pasangan urutan input dan urutan target
dalam *minibatch*.
*Data loader* berikut secara acak menghasilkan *minibatch* dari dataset setiap kali dipanggil.
Argumen `batch_size` menentukan jumlah contoh sub-urutan dalam setiap *minibatch*
dan `num_steps` adalah panjang sub-urutan dalam token.


```{.python .input  n=6}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

Seperti yang dapat kita lihat di bawah ini, 
sebuah *minibatch* dari urutan target
dapat diperoleh 
dengan menggeser urutan input
sebanyak satu token.


```{.python .input  n=7}
%%tab all
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
```

## Ringkasan dan Diskusi

Model bahasa memperkirakan probabilitas gabungan dari suatu urutan teks. Untuk urutan panjang, $n$-gram menyediakan model yang mudah dengan memotong ketergantungan pada konteks yang lebih jauh. Namun, terdapat banyak struktur tetapi tidak cukup frekuensi untuk menangani kombinasi kata yang jarang dengan efektif melalui *Laplace smoothing*. Oleh karena itu, kita akan fokus pada pemodelan bahasa berbasis jaringan saraf pada bagian selanjutnya.
Untuk melatih model bahasa, kita dapat secara acak mengambil pasangan urutan input dan urutan target dalam *minibatch*. Setelah pelatihan, kita akan menggunakan *perplexity* untuk mengukur kualitas model bahasa.

Model bahasa dapat ditingkatkan skalanya dengan peningkatan ukuran data, ukuran model, dan jumlah komputasi yang digunakan saat pelatihan. Model bahasa besar dapat melakukan tugas-tugas yang diinginkan dengan memprediksi teks keluaran berdasarkan instruksi teks masukan. Seperti yang akan kita bahas nanti (misalnya, :numref:`sec_large-pretraining-transformers`),
saat ini
model bahasa besar menjadi dasar bagi sistem mutakhir dalam berbagai tugas.


## Latihan

1. Misalkan ada 100.000 kata dalam dataset pelatihan. Berapa banyak frekuensi kata dan frekuensi kata-kata berurutan yang perlu disimpan oleh model empat-gram?
2. Bagaimana Anda akan memodelkan sebuah dialog?
3. Metode lain apa yang dapat Anda pikirkan untuk membaca data urutan panjang?
4. Pertimbangkan metode kita untuk membuang sejumlah token pertama secara acak pada awal setiap epoch.
    1. Apakah ini benar-benar menghasilkan distribusi yang sempurna secara seragam pada urutan dalam dokumen?
    2. Apa yang perlu Anda lakukan untuk membuatnya lebih seragam lagi? 
5. Jika kita ingin contoh urutan berupa kalimat lengkap, masalah apa yang akan muncul dalam *minibatch sampling*? Bagaimana kita bisa memperbaikinya?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18012)
:end_tab:
