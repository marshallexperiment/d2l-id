```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Predicting House Prices on Kaggle
:label:`sec_kaggle_house`

Sekarang setelah kita memperkenalkan beberapa alat dasar
untuk membangun dan melatih jaringan dalam
serta melakukan regularisasi dengan teknik termasuk
weight decay dan dropout,
kita siap untuk mempraktikkan semua pengetahuan ini
dengan berpartisipasi dalam kompetisi Kaggle.
Kompetisi prediksi harga rumah
adalah tempat yang tepat untuk memulai.
Data ini cukup umum dan tidak menunjukkan struktur yang eksotis
yang mungkin memerlukan model khusus (seperti pada data audio atau video).
Dataset ini, yang dikumpulkan oleh :citet:`De-Cock.2011`,
meliputi harga rumah di Ames, Iowa selama periode 2006–2010.
Dataset ini secara signifikan lebih besar daripada [dataset perumahan Boston](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) yang terkenal dari Harrison dan Rubinfeld (1978),
dengan lebih banyak contoh dan lebih banyak fitur.

Pada bagian ini, kita akan menjelaskan secara detail
tentang pra-pemrosesan data, desain model, dan pemilihan hyperparameter.
Kami berharap bahwa melalui pendekatan praktik langsung ini,
Anda akan mendapatkan intuisi yang akan membimbing Anda
dalam karir Anda sebagai data scientist.


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## Mengunduh Data

Sepanjang buku ini, kita akan melatih dan menguji model
pada berbagai dataset yang diunduh.
Di sini, kita (**mengimplementasikan dua fungsi utilitas**)
untuk mengunduh dan mengekstrak file zip atau tar.
Sekali lagi, kita akan melewatkan rincian implementasi
dari fungsi utilitas seperti ini.


```{.python .input  n=2}
%%tab all
def download(url, folder, sha1_hash=None):
    """Mengunduh file ke dalam folder dan mengembalikan path file lokal."""

def extract(filename, folder):
    """Mengekstrak file zip/tar ke dalam folder."""

```

## Kaggle

[Kaggle](https://www.kaggle.com) adalah platform populer
yang menyelenggarakan kompetisi machine learning.
Setiap kompetisi berpusat pada dataset tertentu dan banyak yang
disponsori oleh pemangku kepentingan yang menawarkan hadiah
untuk solusi pemenang.
Platform ini membantu pengguna untuk berinteraksi
melalui forum dan kode bersama,
mendorong kolaborasi sekaligus kompetisi.
Meskipun ada kecenderungan untuk mengejar peringkat secara berlebihan,
dengan peneliti berfokus secara sempit pada langkah-langkah pra-pemrosesan
tanpa mempertimbangkan pertanyaan-pertanyaan fundamental,
ada nilai luar biasa dalam objektivitas platform ini
yang memfasilitasi perbandingan kuantitatif langsung
antara pendekatan-pendekatan yang bersaing serta berbagi kode
sehingga semua orang dapat mempelajari apa yang berhasil dan apa yang tidak.
Jika Anda ingin berpartisipasi dalam kompetisi Kaggle,
Anda harus terlebih dahulu mendaftar untuk akun
(lihat :numref:`fig_kaggle`).

![Situs web Kaggle.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

Pada halaman kompetisi prediksi harga rumah, seperti yang ditunjukkan
di :numref:`fig_house_pricing`,
Anda dapat menemukan dataset (di bawah tab "Data"),
mengirimkan prediksi, dan melihat peringkat Anda.
URL kompetisi adalah sebagai berikut:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![Halaman kompetisi prediksi harga rumah.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Mengakses dan Membaca Dataset

Perhatikan bahwa data kompetisi dipisahkan
menjadi set pelatihan dan set uji.
Setiap entri mencakup nilai properti rumah
dan atribut-atribut seperti jenis jalan, tahun konstruksi,
jenis atap, kondisi basement, dan lain-lain.
Fitur-fitur ini terdiri dari berbagai tipe data.
Misalnya, tahun konstruksi
direpresentasikan dengan bilangan bulat,
jenis atap dengan kategori diskrit,
dan fitur lainnya dengan angka desimal.
Di sinilah kenyataan memperumit keadaan:
untuk beberapa contoh, beberapa data hilang
dengan nilai yang hilang ditandai hanya sebagai "na".
Harga setiap rumah hanya disertakan
untuk set pelatihan saja
(ini kompetisi, bagaimanapun juga).
Kita perlu mempartisi set pelatihan
untuk membuat set validasi,
tetapi kita hanya bisa mengevaluasi model kita pada set uji resmi
setelah mengunggah prediksi ke Kaggle.
Tab "Data" pada halaman kompetisi di
:numref:`fig_house_pricing`
memiliki tautan untuk mengunduh data.

Untuk memulai, kita akan [**membaca dan memproses data
menggunakan `pandas`**], yang telah kita perkenalkan di :numref:`sec_pandas`.
Untuk kenyamanan, kita bisa mengunduh dan menyimpan
dataset perumahan Kaggle.
Jika file yang sesuai dengan dataset ini sudah ada di direktori cache dan SHA-1-nya cocok dengan `sha1_hash`, kode kita akan menggunakan file yang ada di cache untuk menghindari pengunduhan berulang yang tidak perlu.


```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

The training dataset includes 1460 examples,
80 features, and one label, while the validation data
contains 1459 examples and 80 features.

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## Data Preprocessing

Mari kita [**lihat empat fitur pertama dan dua fitur terakhir
serta labelnya (SalePrice)**] dari empat contoh pertama.

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

Kita dapat melihat bahwa pada setiap contoh, fitur pertama adalah identifier.
Fitur ini membantu model mengenali setiap contoh pelatihan.
Meskipun ini berguna, identifier ini tidak membawa
informasi untuk tujuan prediksi.
Oleh karena itu, kita akan menghapusnya dari dataset
sebelum memasukkan data ke dalam model.
Selain itu, mengingat berbagai tipe data yang ada,
kita perlu melakukan pra-pemrosesan data sebelum memulai pemodelan.


Mari kita mulai dengan fitur numerik.
Pertama, kita menerapkan heuristik,
dengan [**mengganti semua nilai yang hilang
dengan rata-rata dari fitur yang bersangkutan.**]
Kemudian, untuk menempatkan semua fitur pada skala yang sama,
kita (***standarkan* data dengan
mengubah fitur-fitur tersebut menjadi nilai dengan rata-rata nol dan variansi satu**):

$$x \leftarrow \frac{x - \mu}{\sigma},$$

di mana $\mu$ dan $\sigma$ masing-masing menunjukkan rata-rata dan deviasi standar.
Untuk memastikan bahwa transformasi ini
membuat fitur (variabel) kita memiliki rata-rata nol dan variansi satu,
perhatikan bahwa $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$
dan bahwa $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Secara intuitif, kita melakukan standarisasi data
untuk dua alasan.
Pertama, ini mempermudah optimasi.
Kedua, karena kita tidak tahu *a priori*
fitur mana yang akan relevan,
kita tidak ingin memberi penalti berlebih pada koefisien
yang diberikan ke satu fitur lebih dari fitur lainnya.

[**Selanjutnya kita menangani nilai-nilai diskrit.**]
Nilai-nilai ini termasuk fitur seperti "MSZoning".
(**Kita menggantinya dengan one-hot encoding**)
dengan cara yang sama seperti yang kita lakukan sebelumnya
untuk mengubah label multikelas menjadi vektor (lihat :numref:`subsec_classification-problem`).
Misalnya, "MSZoning" memiliki nilai "RL" dan "RM".
Dengan menghapus fitur "MSZoning",
dua fitur indikator baru
"MSZoning_RL" dan "MSZoning_RM" dibuat dengan nilai 0 atau 1.
Menurut one-hot encoding,
jika nilai asli "MSZoning" adalah "RL",
maka "MSZoning_RL" adalah 1 dan "MSZoning_RM" adalah 0.
Paket `pandas` melakukan ini secara otomatis untuk kita.


```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Menghapus kolom ID dan label
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    
    # Standarisasi kolom numerik
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    
    # Mengganti nilai NAN pada fitur numerik dengan 0
    features[numeric_features] = features[numeric_features].fillna(0)
    
    # Mengganti fitur diskrit dengan one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    
    # Menyimpan fitur yang telah diproses
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

Anda dapat melihat bahwa konversi ini meningkatkan
jumlah fitur dari 79 menjadi 331 (tidak termasuk kolom ID dan label).


```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## Error Measure

Sebagai langkah awal, kita akan melatih model linear dengan squared loss. Tidak mengherankan, model linear kita mungkin tidak menghasilkan hasil terbaik dalam kompetisi, tetapi model ini bisa menjadi cek awal untuk memastikan apakah ada informasi yang bermakna dalam data. Jika hasil kita tidak lebih baik dari tebakan acak, kemungkinan besar ada kesalahan dalam pemrosesan data. Jika hasilnya cukup baik, model linear ini akan menjadi baseline yang memberikan intuisi tentang seberapa dekat model sederhana ini dengan model terbaik yang dilaporkan, serta memberi gambaran tentang seberapa besar peningkatan yang bisa kita harapkan dari model yang lebih kompleks.

Pada harga rumah, seperti halnya dengan harga saham,
kita lebih peduli pada besaran relatif
dibandingkan dengan besaran absolut.
Dengan demikian, [**kita lebih cenderung peduli pada
error relatif $\frac{y - \hat{y}}{y}$**]
daripada pada error absolut $y - \hat{y}$.
Sebagai contoh, jika prediksi kita meleset sebesar \$100.000
saat memperkirakan harga rumah di pedesaan Ohio,
di mana nilai rata-rata rumah adalah \$125.000,
maka kemungkinan besar hasil kita sangat buruk.
Namun, jika kita membuat kesalahan sebesar ini
di Los Altos Hills, California,
ini mungkin merupakan prediksi yang sangat akurat
(di sana, harga median rumah melebihi \$4 juta).

(**Salah satu cara untuk menangani masalah ini adalah dengan
mengukur perbedaan dalam logaritma estimasi harga.**)
Faktanya, ini juga merupakan ukuran error resmi
yang digunakan dalam kompetisi untuk mengevaluasi kualitas pengajuan.
Bagaimanapun, nilai kecil $\delta$ untuk $|\log y - \log \hat{y}| \leq \delta$
mengindikasikan bahwa $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
Hal ini mengarah pada root-mean-squared-error antara logaritma harga yang diprediksi dan logaritma harga sebenarnya:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$


```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data:
        return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float), dtype=d2l.float32)
    
    # Menghitung logaritma dari harga
    tensors = (
        get_tensor(data.drop(columns=[label])),  # X
        d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1))  # Y
    )
    return self.get_tensorloader(tensors, train)
```

## $K$-Fold Cross-Validation

Anda mungkin ingat bahwa kita telah memperkenalkan [**cross-validation**]
di :numref:`subsec_generalization-model-selection`, di mana kita membahas bagaimana menangani
pemilihan model.
Kita akan memanfaatkan metode ini untuk memilih desain model
dan menyesuaikan hyperparameter.
Kita pertama-tama membutuhkan sebuah fungsi yang mengembalikan
bagian ke-$i$ dari data
dalam prosedur $K$-fold cross-validation.
Proses ini dilakukan dengan memotong segmen ke-$i$
sebagai data validasi dan mengembalikan sisanya sebagai data pelatihan.
Perhatikan bahwa ini bukanlah cara yang paling efisien untuk menangani data
dan kita pasti akan melakukan sesuatu yang lebih cerdas
jika dataset kita jauh lebih besar.
Namun, kompleksitas tambahan ini bisa membuat kode kita tidak perlu rumit,
sehingga kita dapat mengabaikannya di sini karena masalah ini cukup sederhana.


```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

[**Rata-rata error validasi dikembalikan**]
ketika kita melatih $K$ kali dalam $K$-fold cross-validation.


```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**Pemilihan Model**]

Pada contoh ini, kita memilih sekumpulan hyperparameter yang belum disesuaikan
dan menyerahkannya kepada pembaca untuk meningkatkan model.
Menemukan pilihan yang baik bisa memakan waktu,
tergantung pada seberapa banyak variabel yang dioptimalkan.
Dengan dataset yang cukup besar,
dan jenis hyperparameter yang normal,
$K$-fold cross-validation cenderung
cukup tahan terhadap multiple testing.
Namun, jika kita mencoba terlalu banyak pilihan yang tidak masuk akal,
kita mungkin menemukan bahwa kinerja validasi kita
tidak lagi mewakili error sebenarnya.


```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

Perhatikan bahwa terkadang jumlah error pada set pelatihan
untuk sekumpulan hyperparameter bisa sangat rendah,
meskipun jumlah error pada $K$-fold cross-validation
meningkat cukup tinggi.
Hal ini menunjukkan bahwa kita mengalami overfitting.
Selama pelatihan, Anda perlu memantau kedua angka tersebut.
Overfitting yang lebih sedikit mungkin mengindikasikan bahwa data kita mendukung model yang lebih kuat.
Overfitting yang besar mungkin menyarankan bahwa kita bisa mendapatkan keuntungan
dengan memasukkan teknik regularisasi.

##  [**Mengirimkan Prediksi ke Kaggle**]

Sekarang kita mengetahui pilihan hyperparameter yang baik,
kita bisa menghitung rata-rata prediksi 
pada set uji oleh semua model $K$.
Menyimpan prediksi dalam file CSV
akan memudahkan pengunggahan hasil ke Kaggle.
Kode berikut akan menghasilkan file bernama `submission.csv`.


```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
                         d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]

# Mengambil eksponensial dari prediksi dalam skala logaritmik
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id': data.raw_val.Id,
                           'SalePrice': d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

Selanjutnya, seperti yang diperlihatkan pada :numref:`fig_kaggle_submit2`,
kita dapat mengunggah prediksi kita di Kaggle
dan melihat bagaimana perbandingannya dengan harga rumah aktual (label)
pada set uji.
Langkah-langkahnya cukup sederhana:

* Masuk ke situs web Kaggle dan kunjungi halaman kompetisi prediksi harga rumah.
* Klik tombol “Submit Predictions” atau “Late Submission”.
* Klik tombol “Upload Submission File” di kotak bergaris di bagian bawah halaman dan pilih file prediksi yang ingin Anda unggah.
* Klik tombol “Make Submission” di bagian bawah halaman untuk melihat hasil Anda.

![Mengunggah data ke Kaggle.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Rangkuman dan Diskusi

Data nyata sering kali mengandung berbagai tipe data dan perlu diproses terlebih dahulu.
Mengubah data bernilai riil ke skala dengan rata-rata nol dan variansi satu adalah pilihan default yang baik. Begitu juga dengan mengganti nilai yang hilang dengan rata-ratanya.
Selain itu, mengubah fitur kategorikal menjadi fitur indikator memungkinkan kita memperlakukan mereka seperti vektor one-hot.
Ketika kita lebih peduli pada
error relatif dibandingkan error absolut,
kita dapat mengukur perbedaan dalam logaritma prediksi.
Untuk memilih model dan menyesuaikan hyperparameter,
kita bisa menggunakan $K$-fold cross-validation.


## Latihan

1. Kirimkan prediksi Anda untuk bagian ini ke Kaggle. Seberapa bagus hasilnya?
2. Apakah selalu merupakan ide yang baik untuk mengganti nilai yang hilang dengan rata-rata? Petunjuk: bisakah Anda membuat situasi di mana nilai-nilai tersebut tidak hilang secara acak?
3. Tingkatkan skor dengan menyesuaikan hyperparameter melalui $K$-fold cross-validation.
4. Tingkatkan skor dengan memperbaiki model (misalnya, jumlah lapisan, weight decay, dan dropout).
5. Apa yang terjadi jika kita tidak melakukan standarisasi pada fitur numerik kontinu seperti yang telah kita lakukan di bagian ini?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17988)
:end_tab:

