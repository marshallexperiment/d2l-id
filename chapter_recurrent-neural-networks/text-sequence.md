# Mengonversi Teks Mentah menjadi Data Urutan
:label:`sec_text-sequence`

Sepanjang buku ini,
kita akan sering bekerja dengan data teks
yang direpresentasikan sebagai urutan
kata, karakter, atau potongan kata.
Untuk memulainya, kita membutuhkan beberapa alat dasar
untuk mengonversi teks mentah
menjadi urutan dalam bentuk yang sesuai.
Pipeline *preprocessing* yang umum
melakukan langkah-langkah berikut:

1. Memuat teks sebagai string ke dalam memori.
2. Memecah string menjadi token (misalnya, kata atau karakter).
3. Membangun kamus kosakata untuk menghubungkan setiap elemen kosakata dengan indeks numerik.
4. Mengonversi teks menjadi urutan indeks numerik.


```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## Membaca Dataset

Di sini, kita akan bekerja dengan buku karya H. G. Wells berjudul [The Time Machine](http://www.gutenberg.org/ebooks/35),
sebuah buku yang berisi lebih dari 30.000 kata.
Meskipun aplikasi nyata biasanya
melibatkan dataset yang jauh lebih besar,
ini sudah cukup untuk mendemonstrasikan
*preprocessing pipeline*.
Metode `_download` berikut (**membaca teks mentah ke dalam sebuah string**).


```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """dataset Time Machine."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

Untuk menyederhanakan, kita mengabaikan tanda baca dan kapitalisasi saat melakukan *preprocessing* pada teks mentah.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenisasi

*Token* adalah unit terkecil yang tidak dapat dibagi dari teks.
Setiap langkah waktu (time step) sesuai dengan 1 token,
tetapi apa yang tepatnya merupakan sebuah token adalah keputusan desain.
Sebagai contoh, kita bisa merepresentasikan kalimat
"Baby needs a new pair of shoes"
sebagai sebuah urutan dari 7 kata,
di mana kumpulan dari semua kata tersebut membentuk
kosakata yang besar (biasanya puluhan
atau bahkan ratusan ribu kata).
Atau kita bisa merepresentasikan kalimat yang sama
sebagai urutan yang lebih panjang dari 30 karakter,
dengan kosakata yang jauh lebih kecil
(hanya ada 256 karakter ASCII yang berbeda).
Di bawah ini, kita akan melakukan tokenisasi pada teks yang telah diproses
menjadi sebuah urutan karakter.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Vocabulary

Token-token ini masih dalam bentuk string.
Namun, masukan ke model kita
pada akhirnya harus terdiri dari
input numerik.
[**Selanjutnya, kita memperkenalkan kelas
untuk membangun *kosakata*,
yaitu, objek yang mengaitkan
setiap nilai token yang unik
dengan indeks yang unik.**]
Pertama, kita menentukan kumpulan token unik dalam *corpus* pelatihan kita.
Kemudian kita memberikan indeks numerik untuk setiap token unik.
Elemen kosakata yang jarang muncul sering kali dihapus untuk kemudahan.
Setiap kali kita menemukan token pada saat pelatihan atau pengujian
yang belum pernah dilihat sebelumnya atau dihapus dari kosakata (_vocabulary_),
kita merepresentasikannya dengan token khusus "&lt;unk&gt;",
yang menandakan bahwa ini adalah nilai *unknown* atau tidak dikenal.


```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
   def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Memipihkan list 2D jika diperlukan
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Menghitung frekuensi token
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # Daftar token unik
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # Mendapatkan indeks dari token, atau indeks untuk token tidak dikenal jika token tidak ada
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        # Mengonversi indeks menjadi token
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Indeks untuk token tidak dikenal
        return self.token_to_idx['<unk>']

```

Sekarang kita akan [**membangun kosakata**] untuk dataset kita,
mengonversi urutan string
menjadi daftar indeks numerik.
Perhatikan bahwa kita tidak kehilangan informasi apa pun
dan dapat dengan mudah mengonversi dataset kita
kembali ke representasi aslinya (string).


```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Menyatukan Semua

Dengan menggunakan kelas dan metode di atas,
kita akan [**mengemas semuanya ke dalam
metode `build` dari kelas `TimeMachine` berikut**],
yang mengembalikan `corpus`, berupa daftar indeks token, dan `vocab`,
yaitu kosakata dari korpus *The Time Machine*.
Modifikasi yang kita lakukan di sini adalah:
(i) kita melakukan tokenisasi teks menjadi karakter, bukan kata,
untuk menyederhanakan pelatihan pada bagian selanjutnya;
(ii) `corpus` adalah sebuah daftar tunggal, bukan daftar dari daftar token,
karena setiap baris teks dalam dataset *The Time Machine*
tidak selalu merupakan sebuah kalimat atau paragraf.


```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## Statistik Eksplorasi Bahasa
:label:`subsec_natural-lang-stat`

Dengan menggunakan korpus nyata dan kelas `Vocab` yang didefinisikan berdasarkan kata,
kita dapat memeriksa statistik dasar terkait penggunaan kata dalam korpus kita.
Di bawah ini, kita membangun kosakata dari kata-kata yang digunakan dalam *The Time Machine*
dan mencetak sepuluh kata yang paling sering muncul.


```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Perhatikan bahwa (**sepuluh kata yang paling sering muncul**) 
tidak begitu bersifat deskriptif. 
Anda bahkan mungkin membayangkan bahwa 
daftar yang sangat mirip dapat muncul 
jika kita memilih buku secara acak. 
Kata-kata seperti "the" dan "a", 
kata ganti seperti "i" dan "my", 
dan preposisi seperti "of", "to", dan "in" 
sering muncul karena peran sintaktisnya yang umum.
Kata-kata umum namun tidak begitu deskriptif ini 
sering disebut sebagai (***stop words***) dan,
dalam generasi sebelumnya pada klasifikasi teks 
yang berbasis pada representasi *bag-of-words*, 
kata-kata ini paling sering dihilangkan.
Namun, mereka tetap memiliki makna,
dan tidak perlu dihapus saat bekerja dengan 
model neural modern berbasis RNN dan Transformer.
Jika Anda melihat lebih jauh ke bawah daftar,
Anda akan memperhatikan bahwa
frekuensi kata menurun dengan cepat.
Kata yang paling sering muncul ke-$10$ 
kurang dari $1/5$ seumum kata yang paling populer.
Frekuensi kata cenderung mengikuti distribusi hukum pangkat
(lebih spesifiknya distribusi Zipfian) saat kita bergerak ke peringkat yang lebih rendah.
Untuk mendapatkan pemahaman yang lebih baik, kita [**membuat plot dari frekuensi kata**].


```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Setelah menangani beberapa kata pertama sebagai pengecualian,
semua kata lainnya kira-kira mengikuti garis lurus pada plot log-log.
Fenomena ini dijelaskan oleh *hukum Zipf*,
yang menyatakan bahwa frekuensi $n_i$
dari kata ke-$i$ yang paling sering muncul adalah:


$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

yang setara dengan

$$\log n_i = -\alpha \log i + c,$$

di mana $\alpha$ adalah eksponen yang mencirikan
distribusi dan $c$ adalah konstanta.
Hal ini seharusnya sudah membuat kita berpikir dua kali jika kita ingin
memodelkan kata berdasarkan statistik hitungan.
Bagaimanapun, kita akan secara signifikan melebih-lebihkan frekuensi dari kata-kata yang berada di bagian akhir distribusi, atau yang dikenal sebagai kata-kata yang jarang muncul. Tetapi [**bagaimana dengan kombinasi kata lainnya, seperti dua kata berturut-turut (bigram), tiga kata berturut-turut (trigram)**], dan seterusnya?
Mari kita lihat apakah frekuensi bigram berperilaku dengan cara yang sama seperti frekuensi kata tunggal (unigram).


```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Satu hal yang perlu diperhatikan di sini. Dari sepuluh pasang kata yang paling sering muncul, sembilan di antaranya terdiri dari *stop words*, dan hanya satu yang relevan dengan isi buku sebenarnya—yaitu "the time". Selanjutnya, mari kita lihat apakah frekuensi trigram berperilaku dengan cara yang sama.


```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Sekarang, mari kita [**visualisasikan frekuensi token**] di antara ketiga model ini: unigram, bigram, dan trigram.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Gambar ini cukup menarik.
Pertama, di luar kata unigram, urutan kata
juga tampaknya mengikuti hukum Zipf,
meskipun dengan eksponen $\alpha$ yang lebih kecil
dalam :eqref:`eq_zipf_law`,
tergantung pada panjang urutannya.
Kedua, jumlah $n$-gram yang berbeda tidak terlalu besar.
Ini memberi kita harapan bahwa terdapat cukup banyak struktur dalam bahasa.
Ketiga, banyak $n$-gram yang sangat jarang muncul.
Hal ini membuat beberapa metode kurang cocok untuk pemodelan bahasa
dan memotivasi penggunaan model pembelajaran mendalam.
Kita akan membahas ini di bagian berikutnya.



## Ringkasan

Teks adalah salah satu bentuk data urutan yang paling umum ditemui dalam pembelajaran mendalam.
Pilihan umum untuk apa yang dianggap sebagai token adalah karakter, kata, dan potongan kata.
Untuk melakukan *preprocessing* teks, kita biasanya (i) memecah teks menjadi token; (ii) membangun kosakata untuk memetakan string token ke indeks numerik; dan (iii) mengonversi data teks menjadi indeks token agar dapat dimanipulasi oleh model.
Dalam praktiknya, frekuensi kata cenderung mengikuti hukum Zipf. Hal ini berlaku tidak hanya untuk kata individual (unigram), tetapi juga untuk $n$-gram.


## Latihan

1. Dalam percobaan pada bagian ini, tokenisasi teks menjadi kata-kata dan variasikan nilai argumen `min_freq` pada instance `Vocab`. Karakterisasikan secara kualitatif bagaimana perubahan `min_freq` memengaruhi ukuran kosakata yang dihasilkan.
2. Estimasikan eksponen distribusi Zipfian untuk unigram, bigram, dan trigram dalam korpus ini.
3. Temukan beberapa sumber data lainnya (unduh dataset pembelajaran mesin standar, pilih buku lain yang berada dalam domain publik, *scrape* sebuah situs web, dll). Untuk masing-masing, lakukan tokenisasi data pada tingkat kata dan karakter. Bagaimana perbandingan ukuran kosakata dengan korpus *The Time Machine* pada nilai `min_freq` yang setara? Estimasikan eksponen distribusi Zipfian yang sesuai dengan distribusi unigram dan bigram untuk korpus ini. Bagaimana perbandingannya dengan nilai yang Anda amati pada korpus *The Time Machine*?

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
[Diskusi](https://discuss.d2l.ai/t/18011)
:end_tab:
