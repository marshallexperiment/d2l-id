```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Merancang Arsitektur Jaringan Konvolusi
:label:`sec_cnn-design`

Bagian-bagian sebelumnya telah membawa kita dalam tur desain jaringan modern untuk computer vision. Yang umum dari semua pekerjaan yang telah kita bahas adalah bahwa pekerjaan tersebut sangat bergantung pada intuisi para ilmuwan. Banyak arsitektur dipengaruhi oleh kreativitas manusia dan hanya sedikit oleh eksplorasi sistematis dari ruang desain yang ditawarkan oleh jaringan dalam. Namun, pendekatan *rekayasa jaringan* ini telah berhasil secara luar biasa.

Sejak AlexNet (:numref:`sec_alexnet`)
mengalahkan model computer vision konvensional pada ImageNet,
membangun jaringan yang sangat dalam
dengan menumpuk blok konvolusi yang dirancang dengan pola yang sama telah menjadi populer.
Secara khusus, konvolusi $3 \times 3$ 
dipopulerkan oleh jaringan VGG (:numref:`sec_vgg`).
NiN (:numref:`sec_nin`) menunjukkan bahwa konvolusi $1 \times 1$ 
juga dapat menguntungkan dengan menambahkan non-linearitas lokal. 
Selain itu, NiN menyelesaikan masalah agregasi informasi pada bagian akhir jaringan
dengan mengumpulkan informasi di seluruh lokasi.
GoogLeNet (:numref:`sec_googlenet`) menambahkan beberapa cabang dengan lebar konvolusi yang berbeda,
menggabungkan keunggulan VGG dan NiN dalam blok Inception-nya.
ResNets (:numref:`sec_resnet`) 
mengubah bias induktif menuju pemetaan identitas (dari $f(x) = 0$). Ini memungkinkan jaringan yang sangat dalam. Hampir satu dekade kemudian, desain ResNet masih populer, yang menunjukkan keunggulan desainnya. Terakhir, ResNeXt (:numref:`subsec_resnext`) menambahkan konvolusi yang dikelompokkan, menawarkan keseimbangan yang lebih baik antara parameter dan komputasi. Sebagai pendahulu Transformers untuk vision, Squeeze-and-Excitation Networks (SENets) memungkinkan transfer informasi yang efisien antar lokasi :cite:`Hu.Shen.Sun.2018`. Hal ini dicapai dengan menghitung fungsi perhatian global per saluran.

Sampai sekarang, kita belum membahas jaringan yang diperoleh melalui *neural architecture search* (NAS) :cite:`zoph2016neural,liu2018darts`. Kami memilih untuk tidak membahasnya karena biayanya biasanya sangat besar, mengandalkan pencarian brute-force, algoritma genetika, pembelajaran penguatan, atau bentuk lain dari optimisasi hyperparameter. Dengan ruang pencarian tetap, 
NAS menggunakan strategi pencarian untuk secara otomatis memilih
arsitektur berdasarkan estimasi kinerja yang dikembalikan.
Hasil dari NAS
adalah satu instance jaringan. EfficientNets adalah salah satu hasil penting dari pencarian ini :cite:`tan2019efficientnet`.

Selanjutnya, kami membahas ide yang sangat berbeda dari pencarian *jaringan terbaik tunggal*. Ini secara komputasi relatif murah, menghasilkan wawasan ilmiah sepanjang jalan, dan cukup efektif dalam hal kualitas hasil. Mari kita tinjau strategi oleh :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` untuk *merancang ruang desain jaringan*. Strategi ini menggabungkan kekuatan desain manual dan NAS. Hal ini dicapai dengan mengoperasikan pada *distribusi jaringan* dan mengoptimalkan distribusi tersebut untuk memperoleh performa yang baik bagi seluruh keluarga jaringan. Hasil dari pendekatan ini adalah *RegNets*, khususnya RegNetX dan RegNetY, serta berbagai prinsip panduan untuk desain CNN yang berkinerja baik.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## AnyNet Design Space
:label:`subsec_the-anynet-design-space`

Deskripsi di bawah ini mengikuti penjelasan dalam :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` dengan beberapa penyesuaian agar sesuai dengan cakupan buku ini.
Untuk memulai, kita memerlukan template untuk keluarga jaringan yang akan dieksplorasi. Salah satu kesamaan desain dalam bab ini adalah bahwa jaringan terdiri dari *stem*, *body*, dan *head*. Bagian stem melakukan pemrosesan awal gambar, sering kali melalui konvolusi dengan ukuran jendela yang lebih besar. Bagian body terdiri dari beberapa blok yang melaksanakan sebagian besar transformasi yang diperlukan untuk mengubah gambar mentah menjadi representasi objek. Terakhir, bagian head mengonversi representasi ini menjadi output yang diinginkan, misalnya melalui softmax regressor untuk klasifikasi multi-kelas.
Bagian body, pada gilirannya, terdiri dari beberapa tahap yang beroperasi pada gambar dengan resolusi yang semakin rendah. Baik bagian stem maupun setiap tahap berikutnya mengurangi resolusi spasial hingga seperempatnya. Terakhir, setiap tahap terdiri dari satu atau lebih blok. Pola ini umum untuk semua jaringan, mulai dari VGG hingga ResNeXt. Untuk desain jaringan generik AnyNet, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` menggunakan blok ResNeXt seperti yang ditunjukkan pada :numref:`fig_resnext_block`.

![Ruang desain AnyNet. Angka $(\mathit{c}, \mathit{r})$ di sepanjang setiap panah menunjukkan jumlah saluran $c$ dan resolusi $\mathit{r} \times \mathit{r}$ dari gambar pada titik tersebut. Dari kiri ke kanan: struktur jaringan generik yang terdiri dari stem, body, dan head; body yang terdiri dari empat tahap; struktur detail dari suatu tahap; dua struktur alternatif untuk blok, satu tanpa downsampling dan satu lagi yang mengurangi resolusi pada setiap dimensi. Pilihan desain mencakup kedalaman $\mathit{d_i}$, jumlah saluran output $\mathit{c_i}$, jumlah grup $\mathit{g_i}$, dan rasio bottleneck $\mathit{k_i}$ untuk setiap tahap $\mathit{i}$.](../img/anynet.svg)
:label:`fig_anynet_full`

Mari kita tinjau struktur yang dijelaskan dalam :numref:`fig_anynet_full` secara detail. Seperti yang disebutkan, AnyNet terdiri dari stem, body, dan head. Stem menerima input berupa gambar RGB (3 saluran), menggunakan konvolusi $3 \times 3$ dengan stride $2$, diikuti oleh batch normalization, untuk mengurangi resolusi dari $r \times r$ menjadi $r/2 \times r/2$. Selain itu, stem menghasilkan $c_0$ saluran yang berfungsi sebagai input untuk body.

Karena jaringan dirancang untuk bekerja dengan gambar ImageNet berukuran $224 \times 224 \times 3$, body bertugas menguranginya menjadi $7 \times 7 \times c_4$ melalui 4 tahap (ingat bahwa $224 / 2^{1+4} = 7$), masing-masing dengan stride $2$. Terakhir, head menggunakan desain standar melalui global average pooling, mirip dengan NiN (:numref:`sec_nin`), diikuti oleh lapisan fully connected untuk menghasilkan vektor berdimensi $n$ untuk klasifikasi $n$-kelas.

Sebagian besar keputusan desain yang relevan ada pada bagian body dari jaringan. Bagian ini terdiri dari beberapa tahap, di mana setiap tahap terdiri dari tipe blok ResNeXt yang sama seperti yang dibahas di :numref:`subsec_resnext`. Desain ini sepenuhnya generik: kita memulai dengan blok yang mengurangi resolusi dengan menggunakan stride $2$ (paling kanan dalam :numref:`fig_anynet_full`). Untuk mencocokkannya, cabang residual dari blok ResNeXt perlu melalui konvolusi $1 \times 1$. Blok ini diikuti oleh sejumlah blok ResNeXt tambahan yang menjaga resolusi dan jumlah saluran tetap tidak berubah. Perlu dicatat bahwa praktik desain umum adalah menambahkan sedikit bottleneck dalam desain blok konvolusi.
Dengan demikian, dengan rasio bottleneck $k_i \geq 1$ kita menyediakan sejumlah saluran, $c_i/k_i$, dalam setiap blok untuk tahap $i$ (seperti yang ditunjukkan eksperimen, ini sebenarnya tidak efektif dan sebaiknya dilewati). Terakhir, karena kita menggunakan blok ResNeXt, kita juga perlu memilih jumlah grup $g_i$ untuk konvolusi yang dikelompokkan pada tahap $i$.

Ruang desain yang tampak generik ini tetap memberikan banyak parameter: kita dapat mengatur lebar blok (jumlah saluran) $c_0, \ldots c_4$, kedalaman (jumlah blok) per tahap $d_1, \ldots d_4$, rasio bottleneck $k_1, \ldots k_4$, dan lebar grup (jumlah grup) $g_1, \ldots g_4$.
Secara keseluruhan, ini menghasilkan 17 parameter, yang mengarah pada jumlah konfigurasi yang sangat besar dan perlu dieksplorasi. Kita membutuhkan alat untuk mengurangi ruang desain yang besar ini secara efektif. Inilah keindahan konseptual dari ruang desain. Sebelum melanjutkan, mari kita implementasikan desain generik ini terlebih dahulu.


```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

Setiap tahap terdiri dari `depth` blok ResNeXt,
dengan `num_channels` yang menentukan lebar blok.
Perlu dicatat bahwa blok pertama mengurangi setengah tinggi dan lebar gambar input.


```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

Dengan menggabungkan bagian stem, body, dan head dari jaringan,
kita menyelesaikan implementasi AnyNet.


```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## Distribusi dan Parameter Ruang Desain

Seperti yang telah dibahas di :numref:`subsec_the-anynet-design-space`, parameter dari ruang desain adalah hyperparameter dari jaringan dalam ruang desain tersebut.
Pertimbangkan masalah identifikasi parameter yang baik dalam ruang desain AnyNet. Kita dapat mencoba menemukan pilihan parameter *terbaik tunggal* untuk sejumlah komputasi tertentu (misalnya, FLOPs dan waktu komputasi). Jika kita hanya mengizinkan *dua* pilihan untuk setiap parameter, kita harus mengeksplorasi $2^{17} = 131072$ kombinasi untuk menemukan solusi terbaik. Ini jelas tidak mungkin dilakukan karena biayanya yang sangat tinggi. Lebih buruk lagi, kita tidak benar-benar mempelajari apa pun dari latihan ini tentang bagaimana seharusnya kita merancang jaringan. Lain kali, jika kita menambahkan tahap X, atau operasi shift, atau sejenisnya, kita harus memulai dari awal. Bahkan lebih buruk, karena adanya faktor stochastik dalam pelatihan (pembulatan, pengacakan, kesalahan bit), kemungkinan besar tidak ada dua pelatihan yang menghasilkan hasil yang persis sama. Strategi yang lebih baik adalah mencoba menentukan pedoman umum tentang bagaimana pilihan parameter harus saling berhubungan. Misalnya, rasio bottleneck, jumlah saluran, blok, grup, atau perubahan antar lapisan sebaiknya diatur oleh sekumpulan aturan sederhana. Pendekatan pada :citet:`radosavovic2019network` bergantung pada empat asumsi berikut:

1. Kita mengasumsikan bahwa prinsip desain umum benar-benar ada, sehingga banyak jaringan yang memenuhi persyaratan ini harus menawarkan kinerja yang baik. Akibatnya, mengidentifikasi *distribusi* jaringan bisa menjadi strategi yang masuk akal. Dengan kata lain, kita berasumsi bahwa ada banyak jarum yang bagus di tumpukan jerami.
2. Kita tidak perlu melatih jaringan hingga konvergen sebelum kita dapat menilai apakah sebuah jaringan baik. Sebaliknya, cukup menggunakan hasil antara sebagai panduan yang andal untuk akurasi akhir. Penggunaan proxy (pendekatan) yang (mendekati) untuk mengoptimalkan suatu tujuan ini disebut sebagai optimisasi multi-fidelity :cite:`forrester2007multi`. Dengan demikian, optimisasi desain dilakukan berdasarkan akurasi yang dicapai setelah hanya beberapa kali melalui dataset, sehingga secara signifikan mengurangi biaya.
3. Hasil yang diperoleh pada skala yang lebih kecil (untuk jaringan yang lebih kecil) digeneralisasi ke jaringan yang lebih besar. Oleh karena itu, optimisasi dilakukan untuk jaringan yang strukturnya serupa, tetapi dengan jumlah blok yang lebih sedikit, jumlah saluran yang lebih sedikit, dll. Hanya di akhir kita perlu memverifikasi bahwa jaringan yang ditemukan juga memberikan kinerja yang baik pada skala besar.
4. Aspek desain dapat difaktorkan secara mendekati sehingga memungkinkan untuk menyimpulkan efeknya terhadap kualitas hasil secara terpisah. Dengan kata lain, masalah optimisasi relatif mudah.

Asumsi-asumsi ini memungkinkan kita menguji banyak jaringan dengan biaya yang murah. Secara khusus, kita dapat *mengambil sampel* secara acak dari ruang konfigurasi dan mengevaluasi kinerjanya. Selanjutnya, kita dapat mengevaluasi kualitas pilihan parameter dengan meninjau *distribusi* error/akurasi yang dapat dicapai oleh jaringan-jaringan tersebut. Denotasi $F(e)$ adalah fungsi distribusi kumulatif (CDF) untuk kesalahan yang dibuat oleh jaringan dalam ruang desain tertentu, diambil menggunakan distribusi probabilitas $p$. Artinya,

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

Tujuan kita sekarang adalah menemukan distribusi $p$ atas *jaringan* sedemikian rupa sehingga sebagian besar jaringan memiliki tingkat kesalahan yang sangat rendah dan di mana dukungan $p$ adalah ringkas. Tentu saja, ini secara komputasi tidak mungkin dilakukan secara akurat. Kita mengandalkan sampel jaringan $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$ (dengan kesalahan $e_1, \ldots, e_n$ masing-masing) dari $p$ dan menggunakan CDF empiris $\hat{F}(e, \mathcal{Z})$ sebagai gantinya:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

Setiap kali CDF untuk satu set pilihan lebih besar (atau cocok) dengan CDF lain, ini menunjukkan bahwa pilihan parameternya lebih baik (atau setara). Sesuai dengan itu, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` bereksperimen dengan rasio bottleneck jaringan yang sama $k_i = k$ untuk semua tahap $i$ dari jaringan. Ini menghilangkan tiga dari empat parameter yang mengatur rasio bottleneck. Untuk menilai apakah hal ini (secara negatif) mempengaruhi kinerja, seseorang dapat mengambil jaringan dari distribusi yang dibatasi dan dari distribusi yang tidak terbatas dan membandingkan CDF yang sesuai. Ternyata batasan ini tidak mempengaruhi akurasi distribusi jaringan sama sekali, seperti yang dapat dilihat pada panel pertama di :numref:`fig_regnet-fig`.
Demikian pula, kita dapat memilih untuk menggunakan lebar grup yang sama $g_i = g$ pada berbagai tahap dalam jaringan. Sekali lagi, ini tidak mempengaruhi kinerja, seperti yang dapat dilihat pada panel kedua di :numref:`fig_regnet-fig`.
Kedua langkah ini bersama-sama mengurangi jumlah parameter bebas sebanyak enam.

![Membandingkan fungsi distribusi empiris error pada ruang desain. $\textrm{AnyNet}_\mathit{A}$ adalah ruang desain asli; $\textrm{AnyNet}_\mathit{B}$ mengikat rasio bottleneck, $\textrm{AnyNet}_\mathit{C}$ juga mengikat lebar grup, $\textrm{AnyNet}_\mathit{D}$ meningkatkan kedalaman jaringan di berbagai tahap. Dari kiri ke kanan: (i) mengikat rasio bottleneck tidak berpengaruh terhadap kinerja; (ii) mengikat lebar grup tidak berpengaruh terhadap kinerja; (iii) meningkatkan lebar jaringan (saluran) di berbagai tahap meningkatkan kinerja; (iv) meningkatkan kedalaman jaringan di berbagai tahap meningkatkan kinerja. Gambar berasal dari :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

Selanjutnya, kita mencari cara untuk mengurangi banyaknya pilihan potensial untuk lebar dan kedalaman tahap. Merupakan asumsi yang masuk akal bahwa, seiring dengan semakin dalamnya jaringan, jumlah saluran harus meningkat, yaitu $c_i \geq c_{i-1}$ ($w_{i+1} \geq w_i$ menurut notasi mereka dalam :numref:`fig_regnet-fig`), menghasilkan $\textrm{AnyNetX}_D$. Demikian pula, merupakan asumsi yang masuk akal bahwa semakin jauh tahap berlanjut, kedalaman seharusnya bertambah, yaitu $d_i \geq d_{i-1}$, menghasilkan $\textrm{AnyNetX}_E$. Hal ini dapat diverifikasi secara eksperimental pada panel ketiga dan keempat di :numref:`fig_regnet-fig`.


## RegNet

Ruang desain $\textrm{AnyNetX}_E$ yang dihasilkan terdiri dari jaringan sederhana
yang mengikuti prinsip desain yang mudah diinterpretasikan:

* Bagikan rasio bottleneck $k_i = k$ untuk semua tahap $i$;
* Bagikan lebar grup $g_i = g$ untuk semua tahap $i$;
* Tingkatkan lebar jaringan di seluruh tahap: $c_{i} \leq c_{i+1}$;
* Tingkatkan kedalaman jaringan di seluruh tahap: $d_{i} \leq d_{i+1}$.

Ini menyisakan kita dengan pilihan terakhir: bagaimana memilih nilai spesifik untuk parameter di atas dalam ruang desain akhir $\textrm{AnyNetX}_E$. Dengan mempelajari jaringan dengan performa terbaik dari distribusi dalam $\textrm{AnyNetX}_E$, seseorang dapat mengamati hal berikut: lebar jaringan sebaiknya meningkat secara linear dengan indeks blok di seluruh jaringan, yaitu $c_j \approx c_0 + c_a j$, di mana $j$ adalah indeks blok dan slope $c_a > 0$. Mengingat bahwa kita memilih lebar blok yang berbeda hanya untuk setiap tahap, kita sampai pada fungsi konstan secara parsial, yang direkayasa untuk mencocokkan ketergantungan ini. Lebih jauh lagi, eksperimen juga menunjukkan bahwa rasio bottleneck $k = 1$ memberikan performa terbaik, artinya kita disarankan untuk tidak menggunakan bottleneck sama sekali.

Kami merekomendasikan pembaca yang tertarik untuk meninjau lebih lanjut detail dalam desain jaringan spesifik untuk jumlah komputasi yang berbeda dengan membaca :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`. Misalnya, varian RegNetX 32-lapisan yang efektif diberikan oleh $k = 1$ (tanpa bottleneck), $g = 16$ (lebar grup adalah 16), $c_1 = 32$ dan $c_2 = 80$ saluran untuk tahap pertama dan kedua, masing-masing, dipilih menjadi $d_1=4$ dan $d_2=6$ blok dalam. Wawasan yang mengejutkan dari desain ini adalah bahwa ia masih berlaku, bahkan ketika menyelidiki jaringan dalam skala yang lebih besar. Lebih baik lagi, ini bahkan berlaku untuk desain jaringan Squeeze-and-Excitation (SE) (RegNetY) yang memiliki aktivasi saluran global :cite:`Hu.Shen.Sun.2018`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

Kita dapat melihat bahwa setiap tahap RegNetX secara progresif mengurangi resolusi dan meningkatkan jumlah saluran output.

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## Pelatihan

Pelatihan RegNetX 32-lapisan pada dataset Fashion-MNIST dilakukan seperti sebelumnya.


```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## Diskusi

Dengan bias induktif (asumsi atau preferensi) yang diinginkan seperti lokalitas dan invarian translasi (:numref:`sec_why-conv`)
untuk visi, CNN telah menjadi arsitektur dominan di area ini. Hal ini tetap terjadi dari LeNet hingga Transformer (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` mulai melampaui CNN dalam hal akurasi. Meskipun banyak dari kemajuan baru-baru ini dalam hal Vision Transformer *dapat* diterapkan kembali ke CNN :cite:`liu2022convnet`, ini hanya mungkin dilakukan dengan biaya komputasi yang lebih tinggi. Yang sama pentingnya, optimisasi perangkat keras terbaru (NVIDIA Ampere dan Hopper) hanya memperlebar kesenjangan yang mendukung Transformer.

Perlu dicatat bahwa Transformer memiliki derajat bias induktif terhadap lokalitas dan invarian translasi yang jauh lebih rendah dibandingkan dengan CNN. Struktur yang dipelajari lebih unggul karena, tidak sedikit, ketersediaan koleksi gambar besar seperti LAION-400m dan LAION-5B :cite:`schuhmann2022laion` yang memiliki hingga 5 miliar gambar. Yang cukup mengejutkan, beberapa karya yang lebih relevan dalam konteks ini bahkan mencakup MLP :cite:`tolstikhin2021mlp`.

Singkatnya, Vision Transformer (:numref:`sec_vision-transformer`) sekarang memimpin dalam hal 
kinerja terbaik pada klasifikasi gambar skala besar,
menunjukkan bahwa *skalabilitas lebih penting daripada bias induktif* :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
Ini mencakup pretraining Transformer berskala besar (:numref:`sec_large-pretraining-transformers`) dengan multi-head self-attention (:numref:`sec_multihead-attention`). Kami mengundang pembaca untuk menelaah bab-bab ini untuk diskusi yang lebih rinci.

## Latihan

1. Tingkatkan jumlah tahap menjadi empat. Bisakah Anda merancang RegNetX yang lebih dalam dan memberikan kinerja yang lebih baik?
1. Hapus elemen ResNeXt dari RegNet dengan mengganti blok ResNeXt dengan blok ResNet. Bagaimana kinerja model baru Anda?
1. Implementasikan beberapa instance dari keluarga "VioNet" dengan *melanggar* prinsip desain RegNetX. Bagaimana kinerjanya? Manakah dari ($d_i$, $c_i$, $g_i$, $b_i$) yang merupakan faktor paling penting?
1. Tujuan Anda adalah merancang MLP "sempurna". Bisakah Anda menggunakan prinsip desain yang diperkenalkan di atas untuk menemukan arsitektur yang baik? Apakah mungkin mengekstrapolasi dari jaringan kecil ke jaringan besar?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/8738)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18009)
:end_tab:
