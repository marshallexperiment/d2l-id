```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Preprocessing Data
:label:`sec_pandas`

Sejauh ini, kita telah bekerja dengan data sintetis
yang disajikan dalam tensor siap pakai.
Namun, untuk menerapkan deep learning di dunia nyata,
kita harus mengekstrak data mentah
yang disimpan dalam format arbitrer,
dan memprosesnya sesuai kebutuhan kita.
Untungnya, pustaka *pandas* [library](https://pandas.pydata.org/) 
dapat menangani banyak dari pekerjaan ini.
Bagian ini, meskipun bukan pengganti
untuk *pandas* [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) yang lengkap,
akan memberi Anda kursus singkat
tentang beberapa rutinitas yang paling umum digunakan.

## Membaca Dataset

File comma-separated values (CSV) sangat umum digunakan 
untuk penyimpanan data tabular (seperti spreadsheet).
Di dalamnya, setiap baris sesuai dengan satu catatan
dan terdiri dari beberapa kolom (dipisahkan koma), misalnya,
"Albert Einstein,14 Maret 1879,Ulm,Sekolah politeknik federal,bidang fisika gravitasi".
Untuk mendemonstrasikan cara memuat file CSV dengan `pandas`, 
kita (**membuat file CSV di bawah**) `../data/house_tiny.csv`. 
File ini merepresentasikan dataset rumah,
di mana setiap baris mewakili rumah yang berbeda
dan kolom-kolomnya merepresentasikan jumlah kamar (`NumRooms`),
jenis atap (`RoofType`), dan harga (`Price`).


```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Sekarang mari kita impor `pandas` dan muat dataset menggunakan `read_csv`.


```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Persiapan Data

Dalam pembelajaran terawasi (supervised learning), kita melatih model
untuk memprediksi nilai *target* yang ditentukan,
dengan diberikan beberapa nilai *input*. 
Langkah pertama dalam memproses dataset
adalah memisahkan kolom yang sesuai
untuk nilai input dan nilai target. 
Kita dapat memilih kolom berdasarkan nama atau
menggunakan pengindeksan berbasis integer-location (`iloc`).

Anda mungkin telah memperhatikan bahwa `pandas` menggantikan
semua entri CSV dengan nilai `NA`
dengan nilai khusus `NaN` (*not a number*). 
Ini juga dapat terjadi setiap kali ada entri yang kosong,
misalnya, "3,,,270000".
Ini disebut sebagai *nilai yang hilang* 
dan mereka adalah "kutu busuk" dalam ilmu data,
sebuah ancaman yang terus-menerus Anda hadapi
sepanjang karir Anda. 
Bergantung pada konteksnya, 
nilai yang hilang dapat ditangani
baik melalui *imputasi* atau *penghapusan*.
Imputasi menggantikan nilai yang hilang 
dengan perkiraan nilai tersebut
sementara penghapusan hanya membuang 
baris atau kolom yang berisi nilai yang hilang. 

Berikut beberapa heuristik imputasi umum.
[**Untuk kolom input kategorikal, 
kita dapat memperlakukan `NaN` sebagai sebuah kategori.**]
Karena kolom `RoofType` memiliki nilai `Slate` dan `NaN`,
`pandas` dapat mengonversi kolom ini 
menjadi dua kolom `RoofType_Slate` dan `RoofType_nan`.
Sebuah baris dengan jenis atap `Slate` akan mengatur nilai 
`RoofType_Slate` dan `RoofType_nan` menjadi 1 dan 0, berturut-turut.
Sebaliknya berlaku untuk baris dengan nilai `RoofType` yang hilang.


```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Untuk nilai numerik yang hilang, 
salah satu heuristik umum adalah 
[**menggantikan entri `NaN` dengan 
nilai rata-rata dari kolom yang sesuai**].


```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```
## Konversi ke Format Tensor

Sekarang, [**semua entri dalam `inputs` dan `targets` adalah numerik,
kita dapat memuatnya ke dalam tensor**] (ingat :numref:`sec_ndarray`).


```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Diskusi

Anda sekarang tahu cara mempartisi kolom data, 
mengisi variabel yang hilang, 
dan memuat data `pandas` ke dalam tensor. 
Di :numref:`sec_kaggle_house`, Anda akan 
memperoleh lebih banyak keterampilan dalam pemrosesan data. 
Meskipun kursus singkat ini sederhana,
pemrosesan data bisa menjadi rumit.
Misalnya, daripada datang dalam satu file CSV,
dataset kita mungkin tersebar di beberapa file
yang diambil dari database relasional.
Misalnya, dalam aplikasi e-commerce,
alamat pelanggan mungkin ada di satu tabel
dan data pembelian di tabel lain.
Selain itu, praktisi menghadapi beragam jenis data
di luar kategori dan numerik, misalnya,
string teks, gambar,
data audio, dan point cloud. 
Sering kali, alat-alat canggih dan algoritma yang efisien 
diperlukan untuk mencegah pemrosesan data menjadi
hambatan terbesar dalam alur kerja machine learning. 
Masalah ini akan muncul saat kita sampai pada 
visi komputer dan pemrosesan bahasa alami. 
Akhirnya, kita harus memperhatikan kualitas data.
Dataset dunia nyata sering kali memiliki 
outlier, pengukuran sensor yang salah, dan kesalahan pencatatan, 
yang harus ditangani sebelum 
memasukkan data ke dalam model apa pun. 
Alat visualisasi data seperti [seaborn](https://seaborn.pydata.org/), 
[Bokeh](https://docs.bokeh.org/), atau [matplotlib](https://matplotlib.org/)
dapat membantu Anda secara manual memeriksa data 
dan mengembangkan intuisi tentang 
jenis masalah yang mungkin perlu Anda atasi.


## Latihan

1. Cobalah memuat dataset, misalnya, Abalone dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets) dan periksa propertinya. Berapa fraksi dari dataset tersebut yang memiliki nilai yang hilang? Berapa fraksi dari variabel yang bersifat numerik, kategorikal, atau teks?
2. Cobalah mengindeks dan memilih kolom data berdasarkan nama daripada nomor kolom. Dokumentasi pandas tentang [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) memiliki detail lebih lanjut tentang cara melakukannya.
3. Menurut Anda, seberapa besar dataset yang bisa Anda muat dengan cara ini? Apa keterbatasannya? Petunjuk: pertimbangkan waktu untuk membaca data, representasi, pemrosesan, dan jejak memori. Coba ini di laptop Anda. Apa yang terjadi jika Anda mencobanya di server?
4. Bagaimana Anda menangani data yang memiliki sejumlah besar kategori? Bagaimana jika label kategori semuanya unik? Haruskah Anda menyertakan yang terakhir?
5. Alternatif apa yang dapat Anda pikirkan selain pandas? Bagaimana dengan [memuat tensor NumPy dari file](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Lihat juga [Pillow](https://python-pillow.org/), Python Imaging Library.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17967)
:end_tab:

