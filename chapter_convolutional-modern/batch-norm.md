```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Batch Normalization
:label:`sec_batch_norm`

Melatih deep neural networks adalah hal yang sulit.
Mendapatkan jaringan untuk konvergen dalam waktu yang wajar bisa menjadi tantangan.
Dalam bagian ini, kita membahas *batch normalization*, sebuah teknik yang populer dan efektif
yang secara konsisten mempercepat konvergensi jaringan yang dalam :cite:`Ioffe.Szegedy.2015`.
Bersama dengan residual blocks---yang akan dibahas lebih lanjut di :numref:`sec_resnet`---batch normalization
telah memungkinkan praktisi untuk secara rutin melatih jaringan dengan lebih dari 100 lapisan.
Manfaat tambahan (serendipitous) dari batch normalization adalah sifatnya yang secara inheren memberikan regularisasi.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
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
from jax import numpy as jnp
import jax
import optax
```

## Pelatihan Jaringan Dalam (Deep networks Train)

Saat bekerja dengan data, kita sering melakukan prapemrosesan sebelum pelatihan.
Pilihan terkait prapemrosesan data sering kali memberikan perbedaan besar pada hasil akhir.
Ingat penerapan MLP pada prediksi harga rumah (:numref:`sec_kaggle_house`).
Langkah pertama kita saat bekerja dengan data nyata
adalah menstandarkan fitur input kita agar memiliki
rata-rata nol $\boldsymbol{\mu} = 0$ dan varians unit $\boldsymbol{\Sigma} = \boldsymbol{1}$ pada beberapa pengamatan :cite:`friedman1987exploratory`, sering kali menskalakan ulang agar diagonal menjadi satuan, yaitu $\Sigma_{ii} = 1$.
Strategi lain adalah mengatur panjang vektor agar menjadi satuan, dengan rata-rata nol *per pengamatan*.
Hal ini dapat bekerja dengan baik, misalnya untuk data sensor spasial. Teknik prapemrosesan ini dan banyak lainnya, bermanfaat dalam menjaga masalah estimasi tetap terkendali.
Untuk ulasan tentang pemilihan dan ekstraksi fitur, lihat artikel dari :citet:`guyon2008feature`, misalnya.
Menstandarkan vektor juga memiliki efek samping yang baik, yaitu membatasi kompleksitas fungsi yang beroperasi di atasnya. Misalnya, radius-margin bound :cite:`Vapnik95` pada support vector machine dan Perceptron Convergence Theorem :cite:`Novikoff62` bergantung pada input dengan norma terbatas.

Secara intuitif, standarisasi ini bekerja dengan baik dengan optimizer kita
karena ini menempatkan parameter *a priori* pada skala yang serupa.
Dengan demikian, wajar untuk bertanya apakah langkah normalisasi serupa *dalam* jaringan yang dalam
akan memberikan manfaat. Meskipun ini bukan alasan yang tepat yang mengarah pada penemuan batch normalization :cite:`Ioffe.Szegedy.2015`, ini adalah cara yang berguna untuk memahaminya dan sepupunya, layer normalization :cite:`Ba.Kiros.Hinton.2016`, dalam kerangka kerja yang terpadu.

Kedua, untuk MLP atau CNN tipikal, saat kita melatih,
variabel-variabel 
pada lapisan menengah (misalnya, output transformasi afine dalam MLP)
dapat mengambil nilai dengan besaran yang sangat bervariasi:
baik pada lapisan dari input hingga output, di antara unit di lapisan yang sama,
maupun dari waktu ke waktu akibat pembaruan parameter model.
Penemu batch normalization menduga secara informal
bahwa pergeseran distribusi dari variabel-variabel tersebut dapat menghambat konvergensi jaringan.
Secara intuitif, kita dapat memperkirakan bahwa jika satu
lapisan memiliki aktivasi variabel 100 kali lipat dibandingkan lapisan lain,
ini mungkin memerlukan penyesuaian kompensasi pada tingkat pembelajaran. Solver adaptif
seperti AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`, atau Distributed Shampoo :cite:`anil2020scalable` bertujuan untuk mengatasi ini dari sudut pandang optimisasi, misalnya dengan menambahkan aspek dari metode orde kedua.
Alternatifnya adalah mencegah masalah ini dengan normalisasi adaptif.

Ketiga, jaringan yang lebih dalam cenderung lebih kompleks dan rentan terhadap overfitting.
Ini berarti bahwa regularisasi menjadi lebih penting. Teknik umum untuk regularisasi adalah injeksi noise.
Ini telah dikenal sejak lama, misalnya pada injeksi noise untuk
input :cite:`Bishop.1995`. Ini juga menjadi dasar dari dropout di :numref:`sec_dropout`. Ternyata, secara kebetulan, batch normalization memberikan ketiga manfaat tersebut: prapemrosesan, stabilitas numerik, dan regularisasi.

Batch normalization diterapkan pada lapisan individual, atau bisa juga pada semua lapisan:
Pada setiap iterasi pelatihan,
kita pertama-tama menormalkan input (dari batch normalization)
dengan mengurangi rata-rata mereka
dan membaginya dengan standar deviasi,
di mana keduanya diperkirakan berdasarkan statistik dari minibatch saat ini.
Selanjutnya, kita menerapkan koefisien skala dan offset untuk mengembalikan derajat kebebasan (_Degree of Freedom_) yang hilang.
Karena *normalisasi* ini didasarkan pada statistik *batch*,
nama *batch normalization* berasal dari sini.

Perhatikan bahwa jika kita mencoba menerapkan batch normalization dengan ukuran minibatch 1,
kita tidak akan dapat mempelajari apa pun.
Ini karena setelah mengurangi rata-rata,
setiap unit tersembunyi akan bernilai 0.
Seperti yang bisa Anda duga, karena kita mencurahkan seluruh bagian ini untuk batch normalization,
dengan ukuran minibatch yang cukup besar, pendekatan ini terbukti efektif dan stabil.
Satu hal yang perlu diingat di sini adalah bahwa saat menerapkan batch normalization,
pilihan ukuran batch menjadi
lebih penting daripada tanpa batch normalization, atau setidaknya,
diperlukan kalibrasi yang sesuai jika kita mengatur ukuran batch.

Misalkan $\mathcal{B}$ adalah minibatch dan $\mathbf{x} \in \mathcal{B}$ adalah input untuk 
batch normalization ($\textrm{BN}$). Dalam kasus ini, batch normalization didefinisikan sebagai berikut:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

Pada persamaan :eqref:`eq_batchnorm`,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ adalah rata-rata sampel
dan $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ adalah standar deviasi sampel dari minibatch $\mathcal{B}$.
Setelah menerapkan standarisasi,
minibatch yang dihasilkan
memiliki rata-rata nol dan varians satuan.
Pemilihan varians satuan
(daripada beberapa nilai ajaib lainnya) bersifat sewenang-wenang. Kita mengembalikan derajat kebebasan ini
dengan memasukkan
*parameter skala* $\boldsymbol{\gamma}$ dan *parameter shift* $\boldsymbol{\beta}$
yang memiliki bentuk yang sama dengan $\mathbf{x}$. Keduanya adalah parameter yang
perlu dipelajari sebagai bagian dari pelatihan model.

Selama pelatihan, magnitudo variabel
pada lapisan menengah tidak dapat menyimpang
karena batch normalization secara aktif menggeser dan menskalakan kembali mereka
ke rata-rata dan ukuran tertentu (melalui $\hat{\boldsymbol{\mu}}_\mathcal{B}$ dan ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
Pengalaman praktis mengonfirmasi bahwa, seperti yang diisyaratkan saat membahas penskalaan fitur, batch normalization memungkinkan penggunaan tingkat pembelajaran yang lebih agresif.
Kita menghitung $\hat{\boldsymbol{\mu}}_\mathcal{B}$ dan ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ dalam persamaan :eqref:`eq_batchnorm` sebagai berikut:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ dan }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Perhatikan bahwa kita menambahkan konstanta kecil $\epsilon > 0$
pada estimasi varians
untuk memastikan bahwa kita tidak pernah mencoba membagi dengan nol,
bahkan dalam kasus di mana estimasi varians empiris mungkin sangat kecil atau hilang.
Estimasi $\hat{\boldsymbol{\mu}}_\mathcal{B}$ dan ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ mengatasi masalah penskalaan
dengan menggunakan estimasi rata-rata dan varians yang penuh noise.
Anda mungkin berpikir bahwa noise ini bisa menjadi masalah.
Sebaliknya, noise ini sebenarnya memberikan manfaat.

Ini ternyata menjadi tema yang sering muncul dalam deep learning.
Untuk alasan yang belum sepenuhnya dijelaskan secara teoritis,
berbagai sumber noise dalam optimisasi
sering kali mempercepat pelatihan dan mengurangi overfitting:
variasi ini tampaknya berfungsi sebagai bentuk regularisasi.
:citet:`Teye.Azizpour.Smith.2018` dan :citet:`Luo.Wang.Shao.ea.2018`
mengaitkan properti batch normalization dengan Bayesian priors dan penalti, masing-masing.
Secara khusus, ini memberikan pemahaman mengapa
batch normalization bekerja paling baik dengan ukuran minibatch moderat, antara 50 hingga 100.
Ukuran minibatch ini tampaknya menyuntikkan jumlah "noise" yang tepat di setiap lapisan, baik dalam hal skala melalui $\hat{\boldsymbol{\sigma}}$, dan dalam hal offset melalui $\hat{\boldsymbol{\mu}}$:
minibatch yang lebih besar memberikan regularisasi yang lebih sedikit karena estimasi yang lebih stabil, sedangkan minibatch yang terlalu kecil
menghilangkan sinyal yang berguna karena varian yang tinggi. Menggali lebih jauh ke arah ini, mempertimbangkan jenis prapemrosesan dan penyaringan alternatif mungkin akan menghasilkan bentuk regularisasi yang efektif lainnya.

Dengan model yang telah dilatih, Anda mungkin berpikir
bahwa kita akan lebih suka menggunakan seluruh dataset
untuk memperkirakan rata-rata dan varians.
Setelah pelatihan selesai, mengapa kita ingin
gambar yang sama diklasifikasikan berbeda,
tergantung pada batch tempat gambar itu berada?
Selama pelatihan, perhitungan tepat ini tidak mungkin dilakukan
karena variabel perantara
untuk semua contoh data
berubah setiap kali kita memperbarui model.
Namun, setelah model dilatih,
kita dapat menghitung rata-rata dan varians
dari variabel setiap lapisan berdasarkan seluruh dataset.
Memang, ini adalah praktik standar untuk
model yang menggunakan batch normalization;
dengan demikian, lapisan batch normalization berfungsi secara berbeda
dalam *mode pelatihan* (menormalkan berdasarkan statistik minibatch)
dan *mode prediksi* (menormalkan berdasarkan statistik dataset).
Dalam bentuk ini, mereka menyerupai perilaku regularisasi dropout di :numref:`sec_dropout`,
di mana noise hanya disuntikkan selama pelatihan.



## Lapisan Batch Normalization

Implementasi batch normalization untuk lapisan fully connected
dan lapisan konvolusi sedikit berbeda.
Satu perbedaan utama antara batch normalization dan lapisan lainnya
adalah karena batch normalization beroperasi pada satu minibatch penuh sekaligus,
kita tidak bisa begitu saja mengabaikan dimensi batch
seperti yang kita lakukan sebelumnya saat memperkenalkan lapisan lainnya.

### Lapisan Fully Connected

Saat menerapkan batch normalization pada lapisan fully connected,
:citet:`Ioffe.Szegedy.2015`, dalam makalah asli mereka memasukkan batch normalization setelah transformasi afine
dan *sebelum* fungsi aktivasi nonlinear. Aplikasi selanjutnya bereksperimen dengan
memasukkan batch normalization tepat *setelah* fungsi aktivasi.
Menyatakan input pada lapisan fully connected dengan $\mathbf{x}$,
transformasi afine
dengan $\mathbf{W}\mathbf{x} + \mathbf{b}$ (dengan parameter bobot $\mathbf{W}$ dan parameter bias $\mathbf{b}$),
dan fungsi aktivasi dengan $\phi$,
kita dapat menyatakan komputasi output lapisan fully connected dengan batch normalization $\mathbf{h}$ sebagai berikut:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Ingat bahwa rata-rata dan varians dihitung
pada *minibatch yang sama*
di mana transformasi diterapkan.

### Lapisan Konvolusi

Demikian pula, pada lapisan konvolusi,
kita dapat menerapkan batch normalization setelah konvolusi
namun sebelum fungsi aktivasi nonlinear. Perbedaan utama dari batch normalization
pada lapisan fully connected adalah kita menerapkan operasi ini pada setiap kanal secara terpisah
*di semua lokasi*. Ini sejalan dengan asumsi invariansi translasi
yang mengarah pada konvolusi: kita berasumsi bahwa lokasi spesifik dari suatu pola
dalam gambar tidaklah penting untuk tujuan pemahaman.

Asumsikan bahwa minibatch kita berisi $m$ contoh
dan bahwa untuk setiap kanal,
output dari konvolusi memiliki tinggi $p$ dan lebar $q$.
Untuk lapisan konvolusi, kita melakukan setiap batch normalization
pada $m \cdot p \cdot q$ elemen per kanal output secara bersamaan.
Dengan demikian, kita mengumpulkan nilai-nilai di semua lokasi spasial
saat menghitung rata-rata dan varians,
dan menerapkan rata-rata dan varians yang sama
dalam kanal tertentu
untuk menormalkan nilai di setiap lokasi spasial.
Setiap kanal memiliki parameter skala dan shift sendiri,
keduanya berupa skalar.

### Layer Normalization
:label:`subsec_layer-normalization-in-bn`

Perhatikan bahwa dalam konteks konvolusi, batch normalization tetap terdefinisi dengan baik bahkan untuk
minibatch dengan ukuran 1: bagaimanapun juga, kita memiliki semua lokasi dalam gambar untuk dirata-rata. Pertimbangan ini
mendorong :citet:`Ba.Kiros.Hinton.2016` untuk memperkenalkan konsep *layer normalization*. Layer normalization bekerja mirip dengan
batch normalization, hanya saja diterapkan pada satu pengamatan pada satu waktu. Akibatnya, baik faktor offset maupun skala menjadi skalar. Untuk vektor berdimensi $n$ $\mathbf{x}$, layer normalization diberikan oleh 

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

di mana skala dan offset diterapkan secara koefisien demi koefisien
dan didefinisikan oleh 

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ dan }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

Seperti sebelumnya, kita menambahkan offset kecil $\epsilon > 0$ untuk mencegah pembagian dengan nol. Salah satu manfaat utama penggunaan layer normalization adalah mencegah divergensi. Setelah semua, jika mengabaikan $\epsilon$, output dari layer normalization independen terhadap skala. Artinya, kita memiliki $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ untuk setiap pilihan $\alpha \neq 0$. Ini menjadi persamaan yang tepat untuk $|\alpha| \to \infty$ (persamaan mendekati ini terjadi karena offset $\epsilon$ pada varians). 

Keuntungan lain dari layer normalization adalah tidak tergantung pada ukuran minibatch. Layer normalization juga independen dari apakah kita sedang dalam mode pelatihan atau prediksi. Dengan kata lain, ini hanya transformasi deterministik yang menstandarkan aktivasi pada skala tertentu. Hal ini dapat sangat bermanfaat untuk mencegah divergensi dalam optimasi. Kami tidak membahas lebih lanjut dan menyarankan pembaca yang tertarik untuk merujuk pada makalah aslinya.

### Batch Normalization Selama Prediksi

Seperti yang telah disebutkan sebelumnya, batch normalization umumnya berperilaku berbeda
dalam mode pelatihan dan mode prediksi.
Pertama, noise dalam rata-rata sampel dan varians sampel
yang timbul dari estimasi pada minibatch
tidak lagi diinginkan setelah model dilatih.
Kedua, kita mungkin tidak memiliki kemewahan
untuk menghitung statistik normalisasi per batch.
Misalnya,
kita mungkin perlu menerapkan model untuk membuat satu prediksi setiap kali.

Biasanya, setelah pelatihan, kita menggunakan seluruh dataset
untuk menghitung estimasi statistik variabel yang stabil
dan kemudian menguncinya saat prediksi.
Oleh karena itu, batch normalization berperilaku berbeda selama pelatihan dan pada saat pengujian.
Ingat bahwa dropout juga menunjukkan karakteristik ini.

## (**Implementasi dari Awal**)

Untuk melihat bagaimana batch normalization bekerja dalam praktik, kita akan mengimplementasikannya dari awal di bawah ini.


```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Gunakan autograd untuk menentukan apakah dalam mode pelatihan
    if not autograd.is_training():
        # Dalam mode prediksi, gunakan mean dan varians yang diperoleh dari moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Saat menggunakan lapisan fully connected, hitung mean dan
            # varians pada dimensi fitur
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # Saat menggunakan lapisan konvolusi dua dimensi, hitung mean dan
            # varians pada dimensi channel (axis=1). Di sini kita perlu menjaga
            # bentuk X, sehingga operasi broadcasting dapat dilakukan nantinya
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # Dalam mode pelatihan, mean dan varians saat ini digunakan 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update mean dan varians menggunakan moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Skala dan geser
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Gunakan is_grad_enabled untuk menentukan apakah kita sedang dalam mode pelatihan
    if not torch.is_grad_enabled():
        # Dalam mode prediksi, gunakan mean dan varians yang diperoleh dari moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Saat menggunakan lapisan fully connected, hitung mean dan
            # varians pada dimensi fitur
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # Saat menggunakan lapisan konvolusi dua dimensi, hitung mean dan
            # varians pada dimensi channel (axis=1). Di sini kita perlu menjaga
            # bentuk X, sehingga operasi broadcasting dapat dilakukan nantinya
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # Dalam mode pelatihan, gunakan mean dan varians saat ini
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Perbarui mean dan varians menggunakan moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Skala dan geser
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Hitung kebalikan dari akar kuadrat varians bergerak secara elemenwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale dan Shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Gunakan `deterministic` untuk menentukan apakah mode saat ini adalah mode pelatihan atau mode prediksi
    if deterministic:
        # Dalam mode prediksi, gunakan mean dan varians yang diperoleh dari moving average
        # `linen.Module.variables` memiliki atribut `value` yang mengandung array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Saat menggunakan lapisan fully connected, hitung mean dan varians pada dimensi fitur
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # Saat menggunakan lapisan konvolusi dua dimensi, hitung mean dan varians pada dimensi channel (axis=1).
            # Di sini kita perlu menjaga bentuk `X`, sehingga operasi broadcasting dapat dilakukan nantinya
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # Dalam mode pelatihan, gunakan mean dan varians saat ini
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Perbarui mean dan varians menggunakan moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Skala dan geser
    return Y
```

Kita sekarang bisa [**membuat layer `BatchNorm` yang sesungguhnya**].  
Layer ini akan menyimpan parameter yang diperlukan untuk skala `gamma` dan shift `beta`, yang keduanya akan diperbarui selama pelatihan. Selain itu, layer ini juga akan menyimpan moving averages dari nilai mean dan varians yang nantinya akan digunakan selama prediksi model.

Mengesampingkan detail algoritmik, perhatikan pola desain di balik implementasi layer ini. 
Biasanya, kita mendefinisikan operasi matematis di fungsi terpisah, misalnya `batch_norm`. Kemudian kita mengintegrasikan fungsionalitas ini ke dalam layer khusus, 
yang sebagian besar kode di dalamnya menangani hal-hal administratif, seperti memindahkan data ke konteks perangkat yang sesuai, 
mengalokasikan dan menginisialisasi variabel yang diperlukan, menjaga catatan moving averages (di sini untuk nilai mean dan varians), 
dan sebagainya. Pola ini memungkinkan pemisahan yang jelas antara matematika dan kode boilerplate.

Selain itu, untuk kemudahan, kita tidak perlu khawatir tentang deteksi otomatis terhadap bentuk input di sini; 
sehingga kita perlu menentukan jumlah fitur secara eksplisit. Saat ini, semua framework deep learning modern telah menawarkan deteksi ukuran dan bentuk 
otomatis dalam API batch normalization tingkat tinggi (dalam praktiknya kita akan menggunakan API ini sebagai gantinya).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: jumlah output untuk layer fully connected
    # atau jumlah channel output untuk layer konvolusi. `num_dims`:
    # 2 untuk layer fully connected dan 4 untuk layer konvolusi
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Parameter skala dan parameter shift (parameter model) diinisialisasi ke 1 dan 0
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # Variabel yang bukan parameter model diinisialisasi ke 0 dan 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # Jika `X` tidak berada di memori utama, salin `moving_mean` dan
        # `moving_var` ke perangkat di mana `X` berada
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Simpan `moving_mean` dan `moving_var` yang diperbarui
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: jumlah output untuk layer fully connected atau
    # jumlah channel output untuk layer konvolusi. num_dims: 2 untuk
    # layer fully connected dan 4 untuk layer konvolusi
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Parameter skala dan parameter shift (parameter model) diinisialisasi
        # masing-masing menjadi 1 dan 0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # Variabel yang bukan parameter model diinisialisasi ke 0 dan 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # Jika X tidak berada di memori utama, salin moving_mean dan moving_var
        # ke perangkat di mana X berada
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Simpan moving_mean dan moving_var yang diperbarui
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # Parameter skala dan parameter shift (parameter model) diinisialisasi
        # masing-masing menjadi 1 dan 0
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # Variabel yang bukan parameter model diinisialisasi ke 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: jumlah output untuk lapisan fully connected
    # atau jumlah output channel untuk lapisan konvolusi.
    # `num_dims`: 2 untuk lapisan fully connected dan 4 untuk lapisan konvolusi
    # Gunakan `deterministic` untuk menentukan apakah mode saat ini adalah
    # mode pelatihan atau prediksi
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # Parameter skala dan parameter shift (parameter model) diinisialisasi
        # masing-masing menjadi 1 dan 0
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # Variabel yang bukan parameter model diinisialisasi ke 0 dan 1.
        # Simpan ke koleksi 'batch_stats'
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

Kami menggunakan `momentum` untuk mengatur agregasi atas estimasi rata-rata dan varians sebelumnya. Ini sebenarnya merupakan sebuah *misnomer* karena tidak ada hubungannya dengan istilah *momentum* dalam optimisasi. Namun demikian, ini adalah nama yang umum digunakan untuk istilah ini dan sesuai dengan konvensi penamaan API, kita menggunakan nama variabel yang sama dalam kode kita.

## [**LeNet dengan Batch Normalization**]

Untuk melihat cara menerapkan `BatchNorm` dalam konteksnya, berikut kami menerapkannya pada model LeNet tradisional (:numref:`sec_lenet`). Ingat bahwa batch normalization diterapkan setelah lapisan konvolusi atau lapisan fully connected tetapi sebelum fungsi aktivasi yang sesuai.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
Karena lapisan `BatchNorm` perlu menghitung statistik batch (rata-rata dan varians), Flax melacak dictionary `batch_stats`, memperbaruinya dengan setiap minibatch. 
Koleksi seperti `batch_stats` dapat disimpan dalam objek `TrainState` (di dalam kelas `d2l.Trainer` yang didefinisikan di :numref:`oo-design-training`) sebagai sebuah atribut, 
dan selama proses forward pass model, ini harus diteruskan ke argumen `mutable`, sehingga Flax mengembalikan variabel yang telah dimutasi.
:end_tab:


```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

Seperti sebelumnya, kita akan [**melatih jaringan kita pada dataset Fashion-MNIST**].
Kode ini hampir identik dengan ketika kita pertama kali melatih LeNet.


```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

Mari kita [**melihat parameter skala `gamma`
dan parameter shift `beta`**] yang dipelajari
dari lapisan normalisasi batch pertama.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**Implementasi Singkat**]

Dibandingkan dengan kelas `BatchNorm`, yang baru saja kita definisikan sendiri, kita dapat langsung menggunakan kelas `BatchNorm` yang telah disediakan dalam API tingkat tinggi dari kerangka deep learning. 
Kodenya hampir sama dengan implementasi kita di atas, hanya saja kita tidak perlu lagi memberikan argumen tambahan untuk mendapatkan dimensi yang tepat.


```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

Di bawah ini, kita [**menggunakan hyperparameter yang sama untuk melatih model kita.**]  
Perhatikan bahwa seperti biasa, varian API tingkat tinggi berjalan jauh lebih cepat karena kodenya telah dikompilasi ke C++ atau CUDA, sementara implementasi kustom kita harus diinterpretasikan oleh Python.


```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Diskusi

Secara intuitif, normalisasi batch dianggap dapat membuat lanskap optimasi lebih halus. Namun, kita harus berhati-hati untuk membedakan antara intuisi spekulatif dan penjelasan yang benar-benar menjelaskan fenomena yang kita amati saat melatih model deep learning. Ingat bahwa kita bahkan belum mengetahui mengapa jaringan neural yang lebih sederhana (seperti MLP dan CNN konvensional) mampu melakukan generalisasi dengan baik. Bahkan dengan dropout dan peluruhan bobot, model ini tetap fleksibel sehingga kemampuannya untuk melakukan generalisasi pada data yang belum pernah dilihat mungkin membutuhkan jaminan teori pembelajaran yang lebih terperinci.

Makalah asli yang mengusulkan normalisasi batch :cite:`Ioffe.Szegedy.2015`, selain memperkenalkan alat yang kuat dan berguna, menawarkan penjelasan mengapa teknik ini berhasil, yakni dengan mengurangi *internal covariate shift*. Diduga, istilah *internal covariate shift* merujuk pada intuisi di atas—gagasan bahwa distribusi nilai variabel berubah selama pelatihan. Namun, ada dua masalah dengan penjelasan ini: i) Drift ini sangat berbeda dari *covariate shift*, sehingga nama ini tidak tepat. Jika ada, ini lebih mendekati konsep drift. ii) Penjelasan ini menawarkan intuisi yang kurang rinci dan meninggalkan pertanyaan mengapa teknik ini berhasil. Sepanjang buku ini, kami berupaya menyampaikan intuisi yang digunakan praktisi untuk mengembangkan jaringan saraf dalam, namun penting untuk memisahkan intuisi panduan ini dari fakta ilmiah yang telah terbukti.

Setelah kesuksesan normalisasi batch, penjelasan *internal covariate shift* kembali muncul dalam debat literatur teknis tentang cara menyajikan penelitian machine learning. Dalam pidato yang dikenang saat menerima penghargaan Test of Time di konferensi NeurIPS 2017, Ali Rahimi menggunakan *internal covariate shift* sebagai pusat dalam argumen yang menyamakan praktik deep learning modern dengan alkimia. Contoh ini kemudian dibahas dalam makalah :cite:`Lipton.Steinhardt.2018`, dengan beberapa penulis menawarkan penjelasan alternatif atas keberhasilan normalisasi batch :cite:`Santurkar.Tsipras.Ilyas.ea.2018`, yang mengklaim bahwa keberhasilan teknik ini terjadi meski dalam beberapa hal bertentangan dengan klaim dalam makalah asli.

Kami mencatat bahwa *internal covariate shift* tidak lebih layak dikritik daripada ribuan klaim serupa lainnya yang dibuat setiap tahun dalam literatur machine learning. Kemungkinan, istilah ini menjadi pusat perdebatan karena daya tariknya bagi audiens yang dituju. Normalisasi batch telah menjadi metode yang sangat diperlukan, diterapkan dalam hampir semua pengklasifikasi gambar yang digunakan, dan makalah yang memperkenalkan teknik ini telah dikutip puluhan ribu kali. Kami memperkirakan prinsip-prinsip pemandu dari regularisasi melalui injeksi noise, percepatan melalui penskalaan ulang, dan prapemrosesan mungkin akan mengarah pada penemuan layer dan teknik lain di masa depan.

Catatan praktis tentang normalisasi batch yang perlu diingat:

* Selama pelatihan model, normalisasi batch secara terus-menerus menyesuaikan output antara layer dengan menggunakan rata-rata dan deviasi standar dari minibatch, sehingga nilai output setiap layer lebih stabil.
* Normalisasi batch sedikit berbeda untuk layer fully connected dibandingkan dengan layer konvolusi. Dalam beberapa kasus, normalisasi layer dapat digunakan sebagai alternatif untuk konvolusi.
* Seperti layer dropout, layer normalisasi batch memiliki perilaku berbeda dalam mode pelatihan dan prediksi.
* Normalisasi batch berguna untuk regularisasi dan meningkatkan konvergensi dalam optimasi. Namun, motivasi asli untuk mengurangi internal covariate shift tampaknya bukan penjelasan yang valid.
* Untuk model yang lebih kuat yang tidak terlalu sensitif terhadap gangguan input, pertimbangkan untuk menghapus normalisasi batch :cite:`wang2022removing`.

## Latihan

1. Haruskah kita menghapus parameter bias dari layer fully connected atau layer konvolusi sebelum normalisasi batch? Mengapa?
1. Bandingkan learning rate untuk LeNet dengan dan tanpa normalisasi batch.
    1. Plot peningkatan akurasi validasi.
    1. Seberapa besar Anda dapat menambah learning rate sebelum optimasi gagal pada kedua kasus?
1. Apakah kita perlu normalisasi batch di setiap layer? Coba eksperimenkan.
1. Implementasikan versi "ringan" dari normalisasi batch yang hanya menghilangkan rata-rata, atau yang hanya menghilangkan variansi. Bagaimana perilakunya?
1. Tetapkan parameter `beta` dan `gamma` sebagai tetap. Amati dan analisis hasilnya.
1. Bisakah Anda mengganti dropout dengan normalisasi batch? Bagaimana perubahannya?
1. Penelitian ide baru: pikirkan transformasi normalisasi lain yang dapat Anda terapkan:
    1. Bisakah Anda menerapkan transformasi integral probabilitas?
    1. Bisakah Anda menggunakan estimasi kovarians full-rank? Mengapa ini mungkin bukan ide bagus?
    1. Bisakah Anda menggunakan varian matriks kompak lainnya (block-diagonal, low-displacement rank, Monarch, dll.)?
    1. Apakah kompresi sparsifikasi berfungsi sebagai regularizer?
    1. Apakah ada proyeksi lain (misalnya, cone cembung, transformasi khusus grup simetri) yang dapat Anda gunakan?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18005)
:end_tab:
