```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Jaringan Neural Konvolusi Dalam (AlexNet)
:label:`sec_alexnet`

Meskipun CNN sudah dikenal luas di kalangan komunitas visi komputer dan pembelajaran mesin setelah diperkenalkannya LeNet :cite:`LeCun.Jackel.Bottou.ea.1995`, CNN tidak langsung mendominasi bidang ini. Meskipun LeNet mencapai hasil yang baik pada dataset kecil awal, kinerja dan kelayakan melatih CNN pada dataset yang lebih besar dan realistis belum terbukti. Faktanya, selama sebagian besar waktu antara awal 1990-an dan hasil yang sangat penting pada tahun 2012 :cite:`Krizhevsky.Sutskever.Hinton.2012`, jaringan neural sering kali dilampaui oleh metode pembelajaran mesin lainnya, seperti metode kernel :cite:`Scholkopf.Smola.2002`, metode ensemble :cite:`Freund.Schapire.ea.1996`, dan estimasi terstruktur :cite:`Taskar.Guestrin.Koller.2004`.

Untuk visi komputer, perbandingan ini mungkin tidak sepenuhnya akurat. Artinya, meskipun input ke jaringan konvolusi terdiri dari nilai piksel mentah atau yang diproses secara ringan (misalnya, dengan sentralisasi), para praktisi tidak akan pernah memberikan piksel mentah ke model tradisional. Sebaliknya, pipeline visi komputer yang umum terdiri dari pipeline ekstraksi fitur yang dirancang secara manual, seperti SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, dan bags of visual words :cite:`Sivic.Zisserman.2003`. Alih-alih *mempelajari* fitur, fitur tersebut *dibuat secara manual*. Sebagian besar kemajuan datang dari ide-ide cerdas untuk ekstraksi fitur di satu sisi dan pemahaman mendalam tentang geometri :cite:`Hartley.Zisserman.2000` di sisi lain. Algoritma pembelajaran sering dianggap sebagai pemikiran tambahan.

Meskipun beberapa akselerator jaringan neural sudah tersedia pada tahun 1990-an, akselerator ini belum cukup kuat untuk membuat CNN dengan saluran banyak, lapisan dalam, dan sejumlah besar parameter. Misalnya, NVIDIA GeForce 256 dari tahun 1999 mampu memproses paling banyak 480 juta operasi floating-point per detik (MFLOPS) tanpa adanya kerangka pemrograman yang berarti untuk operasi di luar game. Akselerator saat ini mampu melakukan lebih dari 1000 TFLOPs per perangkat. Selain itu, dataset masih relatif kecil: OCR pada 60.000 gambar resolusi rendah $28 \times 28$ piksel dianggap sebagai tugas yang sangat menantang. Ditambah lagi, trik-trik penting untuk melatih jaringan neural, seperti heuristik inisialisasi parameter :cite:`Glorot.Bengio.2010`, varian pintar dari stochastic gradient descent :cite:`Kingma.Ba.2014`, fungsi aktivasi yang tidak mengurangi nilai :cite:`Nair.Hinton.2010`, dan teknik regularisasi efektif :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` masih belum ada.

Jadi, alih-alih melatih sistem *end-to-end* (dari piksel ke klasifikasi), pipeline klasik lebih mirip seperti ini:

1. Dapatkan dataset yang menarik. Pada masa-masa awal, dataset ini membutuhkan sensor yang mahal. Misalnya, [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake) dari tahun 1994 memiliki resolusi 0,3 megapiksel (VGA), mampu menyimpan hingga 8 gambar, dengan harga sekitar $1000.
2. Pralakukan dataset dengan fitur yang dibuat secara manual berdasarkan pengetahuan tentang optik, geometri, alat analitis lain, dan terkadang juga dari penemuan kebetulan oleh mahasiswa yang beruntung.
3. Proses data melalui serangkaian ekstraktor fitur standar seperti SIFT (scale-invariant feature transform) :cite:`Lowe.2004`, SURF (speeded up robust features) :cite:`Bay.Tuytelaars.Van-Gool.2006`, atau pipeline lainnya yang disetel secara manual. OpenCV masih menyediakan ekstraktor SIFT hingga hari ini!
4. Masukkan representasi yang dihasilkan ke dalam pengklasifikasi favorit Anda, yang kemungkinan adalah model linear atau metode kernel, untuk melatih pengklasifikasi.

Jika Anda berbicara dengan peneliti pembelajaran mesin, mereka akan mengatakan bahwa pembelajaran mesin itu penting dan indah. Teori elegan membuktikan sifat dari berbagai pengklasifikasi :cite:`boucheron2005theory` dan optimisasi konveks :cite:`Boyd.Vandenberghe.2004` telah menjadi andalan untuk memperolehnya. Bidang pembelajaran mesin berkembang pesat, ketat, dan sangat berguna. Namun, jika Anda berbicara dengan peneliti visi komputer, Anda akan mendengar cerita yang sangat berbeda. Mereka akan mengatakan bahwa kebenaran pahit dari pengenalan gambar adalah bahwa fitur, geometri :cite:`Hartley.Zisserman.2000,hartley2009global`, dan rekayasa, bukan algoritma pembelajaran baru, yang mendorong kemajuan. Peneliti visi komputer secara wajar percaya bahwa dataset yang sedikit lebih besar atau lebih bersih atau pipeline ekstraksi fitur yang sedikit lebih baik jauh lebih penting untuk akurasi akhir daripada algoritma pembelajaran apa pun.


```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```
## Pembelajaran Representasi (_Representation Learning_)

Cara lain untuk melihat keadaan ini adalah bahwa
bagian terpenting dari pipeline adalah representasi.
Dan hingga tahun 2012, representasi sebagian besar dihitung secara mekanis.
Faktanya, merancang serangkaian fungsi fitur baru, meningkatkan hasil, dan menulis metode
semuanya menjadi bagian penting dalam makalah penelitian.
SIFT :cite:`Lowe.2004`,
SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`,
HOG (histograms of oriented gradient) :cite:`Dalal.Triggs.2005`,
bags of visual words :cite:`Sivic.Zisserman.2003`,
dan ekstraktor fitur serupa mendominasi.

Sekelompok peneliti lain,
termasuk Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari, dan Juergen Schmidhuber,
memiliki rencana berbeda.
Mereka percaya bahwa fitur itu sendiri seharusnya dipelajari.
Lebih jauh, mereka meyakini bahwa untuk menjadi cukup kompleks,
fitur tersebut sebaiknya disusun secara hierarkis
dengan beberapa lapisan yang dipelajari bersama, masing-masing dengan parameter yang dapat dipelajari.
Dalam kasus gambar, lapisan terendah mungkin akan mendeteksi
tepi, warna, dan tekstur, mirip dengan cara sistem visual pada hewan
memproses inputnya. Secara khusus, desain otomatis fitur visual seperti yang diperoleh melalui sparse coding :cite:`olshausen1996emergence` tetap menjadi tantangan terbuka hingga munculnya CNN modern.
Hingga penelitian :citet:`Dean.Corrado.Monga.ea.2012,le2013building`, ide untuk menghasilkan fitur dari data gambar secara otomatis mulai mendapatkan perhatian signifikan.

CNN modern pertama :cite:`Krizhevsky.Sutskever.Hinton.2012`, yang dinamai
*AlexNet* dari salah satu penemunya, Alex Krizhevsky, sebagian besar merupakan perbaikan evolusioner
atas LeNet. Model ini mencapai kinerja luar biasa dalam tantangan ImageNet tahun 2012.

![Filter gambar yang dipelajari oleh lapisan pertama AlexNet. Reproduksi dari :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Menariknya, di lapisan-lapisan terendah jaringan,
model mempelajari ekstraktor fitur yang menyerupai beberapa filter tradisional.
:numref:`fig_filters`
menunjukkan deskriptor gambar tingkat rendah.
Lapisan-lapisan yang lebih tinggi dalam jaringan mungkin membangun representasi ini
untuk merepresentasikan struktur yang lebih besar, seperti mata, hidung, bilah rumput, dan sebagainya.
Lapisan yang lebih tinggi lagi mungkin merepresentasikan objek-objek keseluruhan
seperti manusia, pesawat, anjing, atau frisbee.
Pada akhirnya, keadaan tersembunyi terakhir mempelajari representasi kompak
dari gambar yang merangkum isinya
sehingga data yang tergolong dalam kategori yang berbeda dapat dipisahkan dengan mudah.

AlexNet (2012) dan pendahulunya LeNet (1995) memiliki banyak elemen arsitektur yang serupa. Hal ini menimbulkan pertanyaan: mengapa membutuhkan waktu begitu lama?
Perbedaan utama adalah bahwa, selama dua dekade sebelumnya, jumlah data dan daya komputasi yang tersedia telah meningkat secara signifikan. Oleh karena itu, AlexNet jauh lebih besar: model ini dilatih pada lebih banyak data, dan pada GPU yang jauh lebih cepat dibandingkan dengan CPU yang tersedia pada tahun 1995.


### Bahan yang Hilang: Data

Model deep dengan banyak lapisan membutuhkan sejumlah besar data
untuk mencapai kondisi di mana model tersebut secara signifikan
melebihi kinerja metode tradisional yang berbasis optimisasi cembung 
(misalnya, metode linier dan kernel).
Namun, karena keterbatasan kapasitas penyimpanan komputer,
mahalnya sensor (seperti pencitraan), 
dan anggaran penelitian yang lebih ketat pada tahun 1990-an,
sebagian besar penelitian bergantung pada dataset kecil.
Banyak makalah mengandalkan koleksi dataset UCI,
yang sebagian besar hanya berisi ratusan atau (beberapa) ribu gambar
dengan resolusi rendah dan seringkali dengan latar belakang yang bersih secara artifisial.

Pada tahun 2009, dataset ImageNet dirilis :cite:`Deng.Dong.Socher.ea.2009`,
menantang para peneliti untuk melatih model dengan 1 juta contoh,
masing-masing 1000 dari 1000 kategori objek yang berbeda. Kategori-kategori ini
berdasarkan pada kata benda paling populer dalam WordNet :cite:`Miller.1995`.
Tim ImageNet menggunakan Google Image Search untuk menyaring set kandidat besar
untuk setiap kategori dan menggunakan
pipeline crowdsourcing Amazon Mechanical Turk
untuk memastikan setiap gambar termasuk dalam kategori yang terkait.
Skala ini belum pernah terjadi sebelumnya, melebihi yang lain hingga lebih dari satu kali lipat
(misalnya, CIFAR-100 hanya memiliki 60.000 gambar). Aspek lain adalah gambar-gambar ini memiliki
resolusi yang relatif tinggi sebesar $224 \times 224$ piksel, berbeda dengan dataset TinyImages yang berisi 80 juta gambar :cite:`Torralba.Fergus.Freeman.2008`, yang terdiri dari thumbnail berukuran $32 \times 32$ piksel.
Hal ini memungkinkan pembentukan fitur tingkat yang lebih tinggi.
Kompetisi terkait, yang dinamakan ImageNet Large Scale Visual Recognition
Challenge :cite:`russakovsky2015imagenet`,
mendorong penelitian di bidang computer vision dan machine learning,
menantang peneliti untuk mengidentifikasi model mana yang berkinerja terbaik
dalam skala yang lebih besar daripada yang sebelumnya dipertimbangkan oleh akademisi. Dataset vision terbesar, seperti LAION-5B
:cite:`schuhmann2022laion` mengandung miliaran gambar dengan metadata tambahan.


### Bahan yang Hilang: Perangkat Keras

Model deep learning merupakan konsumen komputasi yang sangat besar.
Proses pelatihan bisa memakan ratusan epoch, dan setiap iterasi
membutuhkan data yang melewati banyak lapisan operasi aljabar linear yang memerlukan daya komputasi tinggi.
Inilah salah satu alasan utama mengapa pada tahun 1990-an dan awal 2000-an,
algoritma sederhana berbasis optimisasi cembung yang lebih efisien lebih disukai.

*Graphical Processing Units* (GPU) terbukti menjadi pengubah permainan
dalam membuat deep learning menjadi mungkin.
Chip ini awalnya dikembangkan untuk mempercepat
pemrosesan grafis demi manfaat dalam game komputer.
Secara khusus, GPU dioptimalkan untuk throughput tinggi dalam produk matriks-vektor $4 \times 4$, yang diperlukan untuk banyak tugas grafis komputer.
Untungnya, matematika yang digunakan dalam pemrosesan ini sangat mirip
dengan yang dibutuhkan untuk menghitung lapisan konvolusional.
Pada saat itu, NVIDIA dan ATI mulai mengoptimalkan GPU
untuk operasi komputasi umum :cite:`Fernando.2004`,
bahkan hingga memasarkannya sebagai *general-purpose GPUs* (GPGPUs).

Untuk memberikan sedikit intuisi, pertimbangkan inti dari mikroprosesor modern
(CPU).
Masing-masing inti cukup kuat, berjalan pada frekuensi clock tinggi
dan memiliki cache besar (hingga beberapa megabyte L3).
Setiap inti cocok untuk menjalankan berbagai instruksi,
dengan prediktor cabang, pipeline dalam, unit eksekusi khusus,
eksekusi spekulatif,
dan berbagai fitur tambahan
yang memungkinkannya menjalankan berbagai program dengan aliran kontrol yang kompleks.
Namun, keunggulan ini juga menjadi kelemahan:
inti serbaguna sangat mahal untuk dibangun. Mereka unggul dalam menjalankan kode serbaguna dengan banyak aliran kontrol.
Ini membutuhkan area chip yang besar, tidak hanya untuk
ALU (arithmetic logical unit) tempat komputasi terjadi, tetapi juga untuk
semua fitur tambahan tersebut,
plus antarmuka memori, logika caching antar inti,
interkoneksi kecepatan tinggi, dan sebagainya. CPU
relatif buruk dalam melakukan satu tugas tertentu dibandingkan dengan perangkat keras yang khusus dirancang untuk tugas tersebut.
Laptop modern memiliki 4–8 inti,
dan bahkan server kelas atas jarang memiliki lebih dari 64 inti per soket,
karena hal ini tidak hemat biaya.

Sebaliknya, GPU dapat terdiri dari ribuan elemen pemrosesan kecil (chip Ampere terbaru dari NVIDIA memiliki hingga 6912 CUDA core), seringkali dikelompokkan menjadi grup yang lebih besar (NVIDIA menyebutnya sebagai warps).
Detail ini berbeda antara NVIDIA, AMD, ARM, dan vendor chip lainnya. Meskipun setiap inti relatif lemah,
dengan frekuensi clock sekitar 1GHz,
jumlah total inti yang besar membuat GPU berkali-kali lebih cepat daripada CPU.
Misalnya, GPU NVIDIA terbaru, Ampere A100, menawarkan lebih dari 300 TFLOPs per chip untuk perkalian matriks-matriks presisi 16-bit khusus (BFLOAT16) dan hingga 20 TFLOPs untuk operasi floating-point presisi umum (FP32).
Pada saat yang sama, performa floating point CPU jarang melebihi 1 TFLOPs. Misalnya, Amazon Graviton 3 mencapai puncak performa 2 TFLOPs untuk operasi presisi 16-bit, angka yang mirip dengan performa GPU pada prosesor Apple M1.

Ada banyak alasan mengapa GPU jauh lebih cepat daripada CPU dalam hal FLOPs.
Pertama, konsumsi daya cenderung meningkat *kuadrat* dengan frekuensi clock.
Oleh karena itu, untuk anggaran daya satu inti CPU yang berjalan empat kali lebih cepat (angka umum),
Anda dapat menggunakan 16 inti GPU pada kecepatan $\frac{1}{4}$,
yang menghasilkan kinerja $16 \times \frac{1}{4} = 4$ kali lebih baik.
Kedua, inti GPU jauh lebih sederhana
(sebenarnya, untuk waktu yang lama mereka bahkan *tidak mampu*
menjalankan kode serbaguna),
yang membuatnya lebih efisien dalam hal energi. Misalnya, (i) mereka cenderung tidak mendukung evaluasi spekulatif, (ii) umumnya tidak memungkinkan pemrograman setiap elemen pemrosesan secara individual, dan (iii) cache per inti cenderung lebih kecil.
Terakhir, banyak operasi dalam deep learning membutuhkan bandwidth memori yang tinggi.
GPU unggul dalam hal ini dengan bus yang setidaknya 10 kali lebih lebar dibandingkan banyak CPU.

Kembali ke tahun 2012. Sebuah terobosan besar terjadi
ketika Alex Krizhevsky dan Ilya Sutskever
mengimplementasikan CNN dalam GPU.
Mereka menyadari bahwa hambatan komputasi dalam CNN,
konvolusi dan perkalian matriks,
adalah operasi yang dapat diparalelisasi dalam perangkat keras.
Menggunakan dua NVIDIA GTX 580 dengan memori 3GB, yang masing-masing mampu 1,5 TFLOPs (masih menjadi tantangan bagi sebagian besar CPU satu dekade kemudian),
mereka mengimplementasikan konvolusi cepat.
Kode [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) mereka
cukup baik sehingga selama beberapa tahun
menjadi standar industri dan mendukung
tahun-tahun pertama ledakan deep learning.


## AlexNet

AlexNet, yang menggunakan CNN dengan 8 lapisan,
memenangkan kompetisi ImageNet Large Scale Visual Recognition Challenge 2012
dengan selisih yang besar :cite:`Russakovsky.Deng.Huang.ea.2013`.
Jaringan ini menunjukkan, untuk pertama kalinya,
bahwa fitur yang diperoleh melalui pembelajaran dapat melampaui fitur yang dirancang secara manual, mengubah paradigma sebelumnya dalam computer vision.

Arsitektur AlexNet dan LeNet memiliki kesamaan yang mencolok,
sebagaimana diilustrasikan pada :numref:`fig_alexnet`.
Perlu dicatat bahwa kami menyajikan versi AlexNet yang sedikit disederhanakan,
menghapus beberapa elemen desain yang diperlukan pada tahun 2012
agar model tersebut bisa muat di dua GPU kecil.

![Dari LeNet (kiri) ke AlexNet (kanan).](../img/alexnet.svg)
:label:`fig_alexnet`

Terdapat juga perbedaan signifikan antara AlexNet dan LeNet.
Pertama, AlexNet jauh lebih dalam dibandingkan dengan LeNet-5 yang relatif kecil.
AlexNet terdiri dari delapan lapisan: lima lapisan konvolusi,
dua lapisan tersembunyi fully connected, dan satu lapisan output fully connected.
Kedua, AlexNet menggunakan fungsi aktivasi ReLU daripada sigmoid.
Mari kita pelajari detailnya di bawah ini.

### Arsitektur

Pada lapisan pertama AlexNet, bentuk jendela konvolusi adalah $11\times11$.
Karena gambar dalam ImageNet delapan kali lebih tinggi dan lebih lebar
daripada gambar MNIST,
objek dalam data ImageNet cenderung menempati lebih banyak piksel dengan lebih banyak detail visual.
Akibatnya, jendela konvolusi yang lebih besar diperlukan untuk menangkap objek tersebut.
Bentuk jendela konvolusi pada lapisan kedua
dikurangi menjadi $5\times5$, diikuti dengan $3\times3$.
Selain itu, setelah lapisan konvolusi pertama, kedua, dan kelima,
jaringan menambahkan lapisan max-pooling
dengan bentuk jendela $3\times3$ dan stride 2.
Selain itu, AlexNet memiliki sepuluh kali lebih banyak saluran konvolusi dibandingkan LeNet.

Setelah lapisan konvolusi terakhir, terdapat dua lapisan fully connected yang sangat besar
dengan 4096 output.
Lapisan ini membutuhkan hampir 1GB parameter model.
Karena keterbatasan memori pada GPU awal,
AlexNet asli menggunakan desain aliran data ganda,
sehingga masing-masing dari dua GPU mereka hanya bertanggung jawab
untuk menyimpan dan menghitung setengah dari model tersebut.
Untungnya, memori GPU sekarang relatif melimpah,
sehingga kita jarang perlu membagi model ke beberapa GPU akhir-akhir ini
(versi AlexNet kami menyimpang
dari makalah aslinya dalam aspek ini).

### Fungsi Aktivasi

Selain itu, AlexNet mengganti fungsi aktivasi sigmoid dengan fungsi aktivasi ReLU yang lebih sederhana. Di satu sisi, perhitungan fungsi aktivasi ReLU lebih sederhana. Sebagai contoh, ReLU tidak memiliki operasi eksponensiasi seperti yang ditemukan dalam fungsi aktivasi sigmoid.
Di sisi lain, fungsi aktivasi ReLU memudahkan proses pelatihan model ketika menggunakan metode inisialisasi parameter yang berbeda. Hal ini karena, ketika output dari fungsi aktivasi sigmoid sangat mendekati 0 atau 1, gradien pada wilayah tersebut hampir 0, sehingga backpropagation tidak dapat terus memperbarui beberapa parameter model. Sebaliknya, gradien fungsi aktivasi ReLU pada interval positif selalu bernilai 1 (:numref:`subsec_activation-functions`). Oleh karena itu, jika parameter model tidak diinisialisasi dengan benar, fungsi sigmoid mungkin mendapatkan gradien yang hampir 0 pada interval positif, yang berarti bahwa model tidak dapat dilatih dengan efektif.

### Kontrol Kapasitas dan Prapemrosesan

AlexNet mengontrol kompleksitas model pada lapisan fully connected
dengan dropout (:numref:`sec_dropout`),
sementara LeNet hanya menggunakan weight decay.
Untuk memperbesar data lebih jauh, proses pelatihan AlexNet
menambahkan banyak augmentasi gambar,
seperti flipping, clipping, dan perubahan warna.
Ini membuat model menjadi lebih robust dan ukuran sampel yang lebih besar secara efektif mengurangi overfitting.
Lihat :citet:`Buslaev.Iglovikov.Khvedchenya.ea.2020` untuk tinjauan mendalam mengenai langkah-langkah prapemrosesan tersebut.


```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class AlexNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=96, kernel_size=(11, 11), strides=4, padding=1),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=256, kernel_size=(5, 5)),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3)), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=self.num_classes)
        ])
```

Kita [**membangun contoh data dengan satu saluran (single-channel)**] dengan tinggi dan lebar 224 (**untuk mengamati bentuk output dari setiap lapisan**). 
Ini sesuai dengan arsitektur AlexNet pada :numref:`fig_alexnet`.


```{.python .input  n=6}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
AlexNet(training=False).layer_summary((1, 224, 224, 1))
```

## Pelatihan

Meskipun AlexNet dilatih pada ImageNet dalam :citet:`Krizhevsky.Sutskever.Hinton.2012`,
di sini kita menggunakan Fashion-MNIST
karena melatih model ImageNet hingga konvergen bisa memakan waktu berjam-jam atau bahkan berhari-hari
bahkan pada GPU modern.
Salah satu masalah dalam menerapkan AlexNet langsung pada [**Fashion-MNIST**]
adalah bahwa (**gambar-gambar pada dataset ini memiliki resolusi lebih rendah**) ($28 \times 28$ piksel)
(**dibandingkan dengan gambar-gambar di ImageNet**).
Agar dapat berfungsi, (**kita melakukan upsampling gambar menjadi $224 \times 224$**).
Secara umum, ini bukan praktik yang cerdas, karena hanya meningkatkan kompleksitas komputasi tanpa menambahkan informasi tambahan. Namun demikian, kita melakukannya di sini untuk tetap setia pada arsitektur AlexNet.
Kita melakukan perubahan ukuran ini dengan argumen `resize` dalam konstruktor `d2l.FashionMNIST`.

Sekarang, kita bisa [**memulai pelatihan AlexNet.**]
Dibandingkan dengan LeNet pada :numref:`sec_lenet`,
perubahan utama di sini adalah penggunaan laju pembelajaran yang lebih kecil
dan pelatihan yang jauh lebih lambat karena jaringan yang lebih dalam dan lebih lebar,
resolusi gambar yang lebih tinggi, dan konvolusi yang lebih mahal.


```{.python .input  n=8}
%%tab pytorch, mxnet, jax
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## Diskusi

Struktur AlexNet memiliki kemiripan yang mencolok dengan LeNet, dengan sejumlah peningkatan penting, baik untuk akurasi (dropout) maupun kemudahan pelatihan (ReLU). Yang sama mencoloknya adalah perkembangan besar dalam alat deep learning. Apa yang dulu membutuhkan beberapa bulan kerja pada tahun 2012 kini dapat diselesaikan hanya dalam beberapa baris kode dengan menggunakan framework modern apa pun.

Dalam meninjau arsitektur, kita melihat bahwa AlexNet memiliki kelemahan utama dalam hal efisiensi: dua lapisan tersembunyi terakhir membutuhkan matriks berukuran $6400 \times 4096$ dan $4096 \times 4096$. Ini setara dengan 164 MB memori dan 81 MFLOP untuk komputasi, yang keduanya adalah pengeluaran yang signifikan, terutama pada perangkat yang lebih kecil, seperti ponsel. Inilah salah satu alasan mengapa AlexNet telah digantikan oleh arsitektur yang jauh lebih efisien yang akan kita bahas pada bagian berikut. Namun demikian, AlexNet adalah langkah kunci dari jaringan dangkal menuju jaringan dalam yang digunakan saat ini. Perlu dicatat bahwa meskipun jumlah parameter jauh melampaui jumlah data pelatihan dalam eksperimen kita (dua lapisan terakhir memiliki lebih dari 40 juta parameter, dilatih pada dataset yang berisi 60 ribu gambar), hampir tidak ada overfitting: loss pelatihan dan validasi hampir identik sepanjang pelatihan. Ini berkat regularisasi yang ditingkatkan, seperti dropout, yang melekat pada desain jaringan dalam modern.

Meskipun tampaknya hanya ada beberapa baris tambahan dalam implementasi AlexNet dibandingkan dengan LeNet, dibutuhkan waktu bertahun-tahun bagi komunitas akademis untuk menerima perubahan konseptual ini dan memanfaatkan hasil eksperimen yang sangat baik. Ini juga disebabkan oleh kurangnya alat komputasi yang efisien. Pada saat itu, baik DistBelief :cite:`Dean.Corrado.Monga.ea.2012` maupun Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014` belum ada, dan Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` masih belum memiliki banyak fitur yang membedakannya. Ketersediaan TensorFlow :cite:`Abadi.Barham.Chen.ea.2016` secara dramatis mengubah situasi.

## Latihan

1. Berdasarkan diskusi di atas, analisis sifat komputasi AlexNet.
    1. Hitung penggunaan memori untuk konvolusi dan lapisan fully connected. Manakah yang lebih dominan?
    1. Hitung biaya komputasi untuk konvolusi dan lapisan fully connected.
    1. Bagaimana memori (bandwidth baca dan tulis, latensi, ukuran) memengaruhi komputasi? Apakah ada perbedaan efeknya pada pelatihan dan inferensi?
1. Anda adalah seorang desainer chip dan perlu menyeimbangkan antara komputasi dan bandwidth memori. Misalnya, chip yang lebih cepat membutuhkan lebih banyak daya dan mungkin area chip yang lebih besar. Bandwidth memori yang lebih besar membutuhkan lebih banyak pin dan logika kontrol, sehingga juga membutuhkan area lebih. Bagaimana Anda mengoptimalkannya?
1. Mengapa para insinyur tidak lagi melaporkan tolok ukur kinerja pada AlexNet?
1. Coba tingkatkan jumlah epoch saat melatih AlexNet. Dibandingkan dengan LeNet, bagaimana perbedaannya? Mengapa demikian?
1. AlexNet mungkin terlalu kompleks untuk dataset Fashion-MNIST, terutama karena resolusi gambar awal yang rendah.
    1. Coba sederhanakan model agar pelatihan lebih cepat, dengan memastikan bahwa akurasinya tidak turun secara signifikan.
    1. Rancang model yang lebih baik yang langsung bekerja pada gambar $28 \times 28$.
1. Ubah ukuran batch, dan amati perubahan throughput (gambar/detik), akurasi, dan memori GPU.
1. Terapkan dropout dan ReLU pada LeNet-5. Apakah ini meningkatkan hasil? Dapatkah Anda meningkatkan lebih lanjut dengan prapemrosesan untuk memanfaatkan invarian yang melekat pada gambar?
1. Bisakah Anda membuat AlexNet mengalami overfitting? Fitur mana yang perlu Anda hilangkan atau ubah untuk mengganggu pelatihan?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/276)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/18001)
:end_tab:
