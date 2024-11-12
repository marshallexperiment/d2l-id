```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Penerjemahan Mesin dan Dataset
:label:`sec_machine_translation`

Salah satu terobosan besar yang memicu minat yang luas terhadap RNN modern
adalah kemajuan besar dalam bidang penerapan
*penerjemahan mesin* statistik.
Di sini, model diberikan sebuah kalimat dalam satu bahasa
dan harus memprediksi kalimat yang sesuai dalam bahasa lain.
Perlu dicatat bahwa di sini kalimat-kalimat tersebut mungkin memiliki panjang yang berbeda,
dan kata-kata yang bersesuaian dalam kedua kalimat tersebut
mungkin tidak muncul dalam urutan yang sama,
karena perbedaan dalam struktur tata bahasa
dari kedua bahasa tersebut.

Banyak masalah yang memiliki sifat seperti pemetaan
antara dua urutan yang "tidak selaras" seperti ini.
Contohnya termasuk pemetaan 
dari prompt dialog menjadi balasan
atau dari pertanyaan menjadi jawaban.
Secara luas, masalah-masalah seperti ini disebut
*sequence-to-sequence* (seq2seq) 
dan ini menjadi fokus kita
baik untuk sisa dari bab ini
dan sebagian besar dari :numref:`chap_attention-and-transformers`.

Dalam bagian ini, kita akan memperkenalkan masalah penerjemahan mesin
dan contoh dataset yang akan kita gunakan dalam contoh-contoh selanjutnya.
Selama beberapa dekade, formulasi statistik untuk penerjemahan antar bahasa
telah populer :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`,
bahkan sebelum peneliti berhasil menggunakan pendekatan jaringan saraf
(metode ini sering dikelompokkan dengan istilah *penerjemahan mesin berbasis jaringan saraf*).

Pertama-tama kita akan memerlukan beberapa kode baru untuk memproses data kita.
Berbeda dengan pemodelan bahasa yang kita lihat pada :numref:`sec_language-model`,
di sini setiap contoh terdiri dari dua urutan teks terpisah,
satu dalam bahasa sumber dan yang lainnya (terjemahan) dalam bahasa target.
Cuplikan kode berikut akan menunjukkan bagaimana
memuat data yang sudah diproses menjadi minibatch untuk pelatihan.


```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**Mengunduh dan Memproses Dataset**]

Untuk memulai, kita mengunduh dataset bahasa Inggris--Perancis
yang terdiri dari [pasangan kalimat bilingual dari Proyek Tatoeba](http://www.manythings.org/anki/).
Setiap baris dalam dataset ini adalah pasangan kalimat
yang dipisahkan oleh tab, yang terdiri dari teks bahasa Inggris (sebagai *sumber*)
dan teks terjemahan bahasa Perancis (sebagai *target*).
Perlu dicatat bahwa setiap urutan teks
dapat berupa satu kalimat saja,
atau berupa paragraf dengan beberapa kalimat.


```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Setelah mengunduh dataset,
kita [**melanjutkan dengan beberapa langkah praproses**]
untuk data teks mentah.
Sebagai contoh, kita mengganti spasi tak terputus dengan spasi biasa,
mengubah huruf besar menjadi huruf kecil,
dan menambahkan spasi antara kata dan tanda baca.


```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Ganti spasi tak terputus dengan spasi biasa
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Menambahkan spasi antara kata dan tanda baca
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**Tokenisasi**]

Tidak seperti tokenisasi tingkat karakter yang dibahas di :numref:`sec_language-model`, untuk penerjemahan mesin kita lebih memilih tokenisasi tingkat kata di sini (model terbaru saat ini menggunakan teknik tokenisasi yang lebih kompleks). Metode `_tokenize` berikut ini melakukan tokenisasi pada pasangan urutan teks `max_examples` pertama, di mana setiap token adalah sebuah kata atau tanda baca. Kita menambahkan token khusus "&lt;eos&gt;" di akhir setiap urutan untuk menunjukkan akhir dari urutan tersebut.

Ketika sebuah model memprediksi dengan menghasilkan urutan token satu per satu, generasi dari token "&lt;eos&gt;" dapat mengindikasikan bahwa urutan keluaran telah selesai. Pada akhirnya, metode di bawah ini mengembalikan dua daftar dari daftar token: `src` dan `tgt`. Secara spesifik, `src[i]` adalah daftar token dari urutan teks ke-$i$ dalam bahasa sumber (dalam hal ini bahasa Inggris) dan `tgt[i]` adalah daftar token dari urutan teks yang setara dalam bahasa target (dalam hal ini bahasa Perancis).


```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

Mari kita [**plot histogram jumlah token per urutan teks.**]
Pada dataset sederhana Inggris--Perancis ini, kebanyakan urutan teks memiliki kurang dari 20 token.


```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
     """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Memuat Urutan dengan Panjang Tetap
:label:`subsec_loading-seq-fixed-len`

Ingat bahwa dalam pemodelan bahasa
[**setiap contoh urutan**], baik itu segmen dari satu kalimat
atau rentang di beberapa kalimat,
(**memiliki panjang tetap.**)
Ini ditentukan oleh argumen `num_steps`
(jumlah langkah waktu atau token) dari :numref:`sec_language-model`.
Dalam terjemahan mesin, setiap contoh adalah
sepasang urutan teks sumber dan target,
di mana kedua urutan teks tersebut dapat memiliki panjang yang berbeda.

Untuk efisiensi komputasi,
kita masih dapat memproses sebuah minibatch urutan teks
dalam satu waktu dengan cara *pemotongan* dan *pengisian*.
Misalkan setiap urutan dalam minibatch yang sama
harus memiliki panjang yang sama `num_steps`.
Jika suatu urutan teks memiliki lebih sedikit dari `num_steps` token,
kita akan terus menambahkan token khusus "&lt;pad&gt;" 
hingga panjangnya mencapai `num_steps`.
Sebaliknya, kita akan memotong urutan teks tersebut
dengan hanya mengambil `num_steps` token pertama
dan membuang sisanya.
Dengan cara ini, setiap urutan teks
akan memiliki panjang yang sama
untuk dimuat dalam minibatch dengan bentuk yang sama.
Selain itu, kita juga mencatat panjang urutan sumber tanpa menyertakan token padding.
Informasi ini akan dibutuhkan oleh beberapa model yang akan kita bahas nanti.

Karena dataset terjemahan mesin
terdiri dari pasangan bahasa,
kita dapat membangun dua kosakata
untuk kedua bahasa sumber dan
bahasa target secara terpisah.
Dengan tokenisasi pada level kata,
ukuran kosakata akan secara signifikan lebih besar
daripada jika menggunakan tokenisasi pada level karakter.
Untuk mengatasi hal ini,
di sini kita memperlakukan token yang jarang muncul
(muncul kurang dari dua kali)
sebagai token yang tidak dikenal ("&lt;unk&gt;").
Seperti yang akan kami jelaskan nanti (:numref:`fig_seq2seq`),
saat pelatihan dengan urutan target,
output decoder (token label)
bisa sama dengan input decoder (token target),
yang digeser satu token;
dan token khusus awal urutan "&lt;bos&gt;" 
akan digunakan sebagai token input pertama
untuk memprediksi urutan target (:numref:`fig_seq2seq_predict`).


```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**Membaca Dataset**]

Terakhir, kita mendefinisikan metode `get_dataloader`
untuk mengembalikan iterator data.


```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

Mari kita [**baca minibatch pertama dari dataset Bahasa Inggris--Prancis.**]


```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Kami menampilkan sepasang urutan sumber dan urutan target
yang diproses oleh metode `_build_arrays` di atas
(dalam format string).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Ringkasan

Dalam pemrosesan bahasa alami, *machine translation* (penerjemahan mesin) mengacu pada tugas untuk secara otomatis memetakan dari urutan yang mewakili teks dalam bahasa *sumber* ke urutan yang mewakili terjemahan yang masuk akal dalam bahasa *target*. Menggunakan tokenisasi pada tingkat kata, ukuran kosakata akan jauh lebih besar dibandingkan dengan menggunakan tokenisasi pada tingkat karakter, namun panjang urutan akan jauh lebih pendek. Untuk mengurangi ukuran kosakata yang besar, kita dapat memperlakukan token yang jarang muncul sebagai "token tidak dikenal". Kita dapat memotong (truncate) dan menambahkan padding pada urutan teks agar semua urutan memiliki panjang yang sama sehingga dapat dimuat dalam bentuk minibatch. Implementasi modern sering kali mengelompokkan (bucket) urutan dengan panjang yang serupa untuk menghindari pemborosan komputasi yang berlebihan pada padding.

## Latihan

1. Coba nilai yang berbeda untuk argumen `max_examples` dalam metode `_tokenize`. Bagaimana hal ini memengaruhi ukuran kosakata bahasa sumber dan bahasa target?
2. Teks dalam beberapa bahasa seperti Mandarin dan Jepang tidak memiliki penanda batas kata (misalnya, spasi). Apakah tokenisasi pada tingkat kata masih merupakan ide yang baik untuk kasus seperti itu? Mengapa atau mengapa tidak?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18020)
:end_tab:
