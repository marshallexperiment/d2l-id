# Compiler dan Interpreter
:label:`sec_hybridize`

Sejauh ini, buku ini berfokus pada pemrograman imperatif, yang menggunakan pernyataan seperti `print`, `+`, dan `if` untuk mengubah status program. Pertimbangkan contoh berikut dari program imperatif sederhana.


```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python adalah bahasa *interpreted*. Ketika mengevaluasi fungsi `fancy_func` di atas, ia akan melakukan operasi yang membentuk tubuh fungsi *secara berurutan*. Artinya, ia akan mengevaluasi `e = add(a, b)` dan menyimpan hasilnya sebagai variabel `e`, dengan demikian mengubah status program. Dua pernyataan berikutnya `f = add(c, d)` dan `g = add(e, f)` akan dieksekusi dengan cara yang serupa, melakukan penjumlahan dan menyimpan hasilnya sebagai variabel. :numref:`fig_compute_graph` mengilustrasikan aliran data.

![Aliran data dalam program imperatif.](../img/computegraph.svg)
:label:`fig_compute_graph`

Meskipun pemrograman imperatif nyaman, ini mungkin tidak efisien. Di satu sisi, meskipun fungsi `add` dipanggil berulang kali sepanjang `fancy_func`, Python akan mengeksekusi ketiga panggilan fungsi tersebut secara terpisah. Jika ini dieksekusi, misalnya, pada GPU (atau bahkan pada beberapa GPU), overhead yang muncul dari interpreter Python bisa menjadi sangat besar. Selain itu, ia perlu menyimpan nilai variabel `e` dan `f` hingga semua pernyataan di `fancy_func` dieksekusi. Hal ini karena kita tidak tahu apakah variabel `e` dan `f` akan digunakan oleh bagian lain dari program setelah pernyataan `e = add(a, b)` dan `f = add(c, d)` dieksekusi.

## Pemrograman Simbolik

Pertimbangkan alternatifnya, yaitu *pemrograman simbolik*, di mana komputasi biasanya dilakukan hanya setelah proses sepenuhnya didefinisikan. Strategi ini digunakan oleh banyak framework deep learning, termasuk Theano dan TensorFlow (yang terakhir memiliki ekstensi imperatif). Biasanya melibatkan langkah-langkah berikut:

1. Mendefinisikan operasi yang akan dieksekusi.
2. Mengompilasi operasi menjadi program yang dapat dieksekusi.
3. Memberikan input yang diperlukan dan memanggil program yang telah dikompilasi untuk dieksekusi.

Pendekatan ini memungkinkan banyak optimasi. Pertama, kita bisa melewati interpreter Python dalam banyak kasus, sehingga menghilangkan hambatan kinerja yang bisa menjadi signifikan pada beberapa GPU yang cepat yang dipasangkan dengan satu thread Python pada CPU.
Kedua, kompilator mungkin mengoptimalkan dan menulis ulang kode di atas menjadi `print((1 + 2) + (3 + 4))` atau bahkan `print(10)`. Hal ini dimungkinkan karena kompilator dapat melihat kode lengkap sebelum mengubahnya menjadi instruksi mesin. Misalnya, ia bisa melepaskan memori (atau tidak pernah mengalokasikannya) setiap kali sebuah variabel tidak lagi diperlukan. Atau ia bisa mengubah kode sepenuhnya menjadi bagian yang setara.
Untuk mendapatkan gambaran yang lebih baik, pertimbangkan simulasi pemrograman imperatif berikut (bagaimanapun juga ini adalah Python) di bawah ini.


```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

Perbedaan antara pemrograman imperatif (interpreted) dan pemrograman simbolik adalah sebagai berikut:

* Pemrograman imperatif lebih mudah. Ketika pemrograman imperatif digunakan dalam Python, sebagian besar kode cukup sederhana dan mudah ditulis. Pemrograman imperatif juga lebih mudah untuk di-debug. Hal ini karena lebih mudah untuk mendapatkan dan mencetak semua nilai variabel antara yang relevan, atau menggunakan alat debugging bawaan Python.
* Pemrograman simbolik lebih efisien dan lebih mudah dipindahkan. Pemrograman simbolik memudahkan pengoptimalan kode selama kompilasi, sekaligus memiliki kemampuan untuk memindahkan program ke dalam format yang independen dari Python. Hal ini memungkinkan program dijalankan di lingkungan non-Python, sehingga menghindari masalah kinerja potensial yang terkait dengan interpreter Python.

## Pemrograman Hibrid

Secara historis, sebagian besar framework deep learning memilih antara pendekatan imperatif atau simbolik. Misalnya, Theano, TensorFlow (terinspirasi dari Theano), Keras, dan CNTK memformulasikan model secara simbolik. Sebaliknya, Chainer dan PyTorch mengambil pendekatan imperatif. Mode imperatif ditambahkan ke TensorFlow 2.0 dan Keras pada revisi berikutnya.

:begin_tab:`mxnet`
Saat merancang Gluon, pengembang mempertimbangkan apakah mungkin untuk menggabungkan manfaat dari kedua paradigma pemrograman tersebut. Hal ini menghasilkan model hibrid yang memungkinkan pengguna mengembangkan dan melakukan debug dengan pemrograman imperatif murni, sambil memiliki kemampuan untuk mengubah sebagian besar program menjadi program simbolik yang akan dijalankan ketika diperlukan kinerja komputasi pada tingkat produksi dan implementasi.

Dalam praktiknya, ini berarti kita membangun model menggunakan kelas `HybridBlock` atau `HybridSequential`. Secara default, keduanya dieksekusi dengan cara yang sama seperti kelas `Block` atau `Sequential` dieksekusi dalam pemrograman imperatif.
Kelas `HybridSequential` adalah subclass dari `HybridBlock` (seperti `Sequential` yang merupakan subclass dari `Block`). Ketika fungsi `hybridize` dipanggil, Gluon mengompilasi model ke dalam bentuk yang digunakan dalam pemrograman simbolik. Ini memungkinkan kita untuk mengoptimalkan komponen-komponen yang intensif komputasi tanpa mengorbankan cara model diimplementasikan. Kami akan menggambarkan manfaatnya di bawah ini, dengan fokus pada model dan blok sekuensial.
:end_tab:

:begin_tab:`pytorch`
Seperti yang disebutkan sebelumnya, PyTorch didasarkan pada pemrograman imperatif dan menggunakan grafik komputasi dinamis. Dalam upaya untuk memanfaatkan portabilitas dan efisiensi pemrograman simbolik, pengembang mempertimbangkan apakah mungkin untuk menggabungkan manfaat dari kedua paradigma pemrograman tersebut. Hal ini menghasilkan torchscript yang memungkinkan pengguna mengembangkan dan melakukan debug dengan pemrograman imperatif murni, sambil memiliki kemampuan untuk mengubah sebagian besar program menjadi program simbolik yang akan dijalankan ketika diperlukan kinerja komputasi pada tingkat produksi dan implementasi.
:end_tab:

:begin_tab:`tensorflow`
Paradigma pemrograman imperatif sekarang menjadi default di TensorFlow 2, yang merupakan perubahan menyambut bagi mereka yang baru mempelajari bahasa ini. Namun, teknik pemrograman simbolik dan grafik komputasi yang terkait masih ada di TensorFlow, dan dapat diakses dengan menggunakan dekorator `tf.function` yang mudah digunakan. Hal ini membawa paradigma pemrograman imperatif ke TensorFlow, memungkinkan pengguna untuk mendefinisikan fungsi yang lebih intuitif, lalu membungkusnya dan mengompilasinya menjadi grafik komputasi secara otomatis menggunakan fitur yang disebut tim TensorFlow sebagai [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
:end_tab:

## Meng-hybridkan Kelas `Sequential`

Cara termudah untuk memahami bagaimana hybridisasi bekerja adalah dengan mempertimbangkan jaringan dalam dengan beberapa lapisan. Secara konvensional, interpreter Python perlu mengeksekusi kode untuk semua lapisan untuk menghasilkan instruksi yang kemudian dapat diteruskan ke CPU atau GPU. Untuk perangkat komputasi (cepat) tunggal, ini tidak menimbulkan masalah besar. Di sisi lain, jika kita menggunakan server 8-GPU canggih seperti instance AWS P3dn.24xlarge, Python akan kesulitan menjaga agar semua GPU tetap sibuk. Interpreter Python yang bersifat single-threaded menjadi bottleneck di sini. Mari kita lihat bagaimana kita dapat mengatasi ini untuk bagian-bagian penting dari kode dengan mengganti `Sequential` dengan `HybridSequential`. Kita mulai dengan mendefinisikan MLP sederhana.


```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory untuk networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory untuk networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory untuk networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
Dengan memanggil fungsi `hybridize`, kita dapat mengompilasi dan mengoptimalkan komputasi dalam MLP. Hasil komputasi model tetap tidak berubah.
:end_tab:

:begin_tab:`pytorch`
Dengan mengonversi model menggunakan fungsi `torch.jit.script`, kita dapat mengompilasi dan mengoptimalkan komputasi dalam MLP. Hasil komputasi model tetap tidak berubah.
:end_tab:

:begin_tab:`tensorflow`
Sebelumnya, semua fungsi yang dibangun di TensorFlow dibangun sebagai grafik komputasi, dan dengan demikian dikompilasi oleh JIT secara default. Namun, dengan dirilisnya TensorFlow 2.X dan EagerTensor, ini tidak lagi menjadi perilaku default.
Kita bisa mengaktifkan kembali fungsi ini dengan `tf.function`. `tf.function` lebih umum digunakan sebagai dekorator fungsi, tetapi juga memungkinkan untuk memanggilnya secara langsung seperti fungsi Python biasa, seperti yang ditunjukkan di bawah ini. Hasil komputasi model tetap tidak berubah.
:end_tab:


```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Ini tampak hampir terlalu bagus untuk menjadi kenyataan: cukup tentukan sebuah blok menjadi `HybridSequential`, tulis kode yang sama seperti sebelumnya, dan panggil `hybridize`. Setelah ini terjadi, jaringan dioptimalkan (kita akan mengukur kinerjanya di bawah). Sayangnya, ini tidak bekerja secara ajaib untuk setiap lapisan. Meskipun demikian, sebuah lapisan tidak akan dioptimalkan jika mewarisi dari kelas `Block` alih-alih kelas `HybridBlock`.
:end_tab:

:begin_tab:`pytorch`
Ini tampak hampir terlalu bagus untuk menjadi kenyataan: tulis kode yang sama seperti sebelumnya dan cukup konversi model menggunakan `torch.jit.script`. Setelah ini terjadi, jaringan dioptimalkan (kita akan mengukur kinerjanya di bawah).
:end_tab:

:begin_tab:`tensorflow`
Ini tampak hampir terlalu bagus untuk menjadi kenyataan: tulis kode yang sama seperti sebelumnya dan cukup konversi model menggunakan `tf.function`. Setelah ini terjadi, jaringan dibangun sebagai grafik komputasi dalam representasi menengah MLIR milik TensorFlow dan dioptimalkan secara intensif di tingkat kompilator untuk eksekusi cepat (kita akan mengukur kinerjanya di bawah).
Dengan secara eksplisit menambahkan flag `jit_compile = True` pada panggilan `tf.function()`, kita mengaktifkan fungsionalitas XLA (Accelerated Linear Algebra) di TensorFlow. XLA dapat lebih mengoptimalkan kode yang dikompilasi JIT dalam beberapa kasus. Eksekusi dalam mode grafik diaktifkan tanpa definisi eksplisit ini, namun XLA dapat membuat operasi aljabar linear yang besar (dalam konteks yang sering kita lihat dalam aplikasi deep learning) jauh lebih cepat, terutama di lingkungan GPU.
:end_tab:

### Akselerasi dengan Hybridisasi

Untuk menunjukkan peningkatan kinerja yang diperoleh melalui kompilasi, kita akan membandingkan waktu yang dibutuhkan untuk mengevaluasi `net(x)` sebelum dan setelah hybridisasi. Mari kita definisikan kelas untuk mengukur waktu ini terlebih dahulu. Kelas ini akan berguna sepanjang bab ini saat kita berusaha mengukur (dan meningkatkan) kinerja.


```{.python .input}
#@tab all
#@save
class Benchmark:
    """untuk mengukur running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```


:begin_tab:`mxnet`
Sekarang kita dapat memanggil jaringan dua kali, sekali dengan dan sekali tanpa hybridisasi.
:end_tab:

:begin_tab:`pytorch`
Sekarang kita dapat memanggil jaringan dua kali, sekali dengan dan sekali tanpa torchscript.
:end_tab:

:begin_tab:`tensorflow`
Sekarang kita dapat memanggil jaringan tiga kali, sekali dengan eksekusi eager, sekali dengan eksekusi mode grafik, dan sekali lagi menggunakan kompilasi JIT dengan XLA.
:end_tab:



```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
Seperti yang diamati dalam hasil di atas, setelah sebuah instance `HybridSequential` memanggil fungsi `hybridize`, kinerja komputasi meningkat melalui penggunaan pemrograman simbolik.
:end_tab:

:begin_tab:`pytorch`
Seperti yang diamati dalam hasil di atas, setelah sebuah instance `nn.Sequential` disusun menggunakan fungsi `torch.jit.script`, kinerja komputasi meningkat melalui penggunaan pemrograman simbolik.
:end_tab:

:begin_tab:`tensorflow`
Seperti yang diamati dalam hasil di atas, setelah sebuah instance `tf.keras.Sequential` disusun menggunakan fungsi `tf.function`, kinerja komputasi meningkat melalui penggunaan pemrograman simbolik melalui eksekusi mode grafik di tensorflow.
:end_tab:

### Serialisasi

:begin_tab:`mxnet`
Salah satu manfaat dari mengompilasi model adalah kita dapat men-serialisasi (menyimpan) model dan parameternya ke disk. Ini memungkinkan kita untuk menyimpan model dengan cara yang independen dari bahasa front-end yang dipilih. Hal ini memungkinkan kita untuk menerapkan model yang telah dilatih ke perangkat lain dan dengan mudah menggunakan bahasa pemrograman front-end lainnya. Pada saat yang sama, kode sering kali lebih cepat dibandingkan apa yang bisa dicapai dalam pemrograman imperatif. Mari kita lihat fungsi `export` dalam aksi.
:end_tab:

:begin_tab:`pytorch`
Salah satu manfaat dari mengompilasi model adalah kita dapat men-serialisasi (menyimpan) model dan parameternya ke disk. Ini memungkinkan kita untuk menyimpan model dengan cara yang independen dari bahasa front-end yang dipilih. Hal ini memungkinkan kita untuk menerapkan model yang telah dilatih ke perangkat lain dan dengan mudah menggunakan bahasa pemrograman front-end lainnya. Pada saat yang sama, kode sering kali lebih cepat dibandingkan apa yang bisa dicapai dalam pemrograman imperatif. Mari kita lihat fungsi `save` dalam aksi.
:end_tab:

:begin_tab:`tensorflow`
Salah satu manfaat dari mengompilasi model adalah kita dapat men-serialisasi (menyimpan) model dan parameternya ke disk. Ini memungkinkan kita untuk menyimpan model dengan cara yang independen dari bahasa front-end yang dipilih. Hal ini memungkinkan kita untuk menerapkan model yang telah dilatih ke perangkat lain dan dengan mudah menggunakan bahasa pemrograman front-end lainnya atau mengeksekusi model yang telah dilatih di server. Pada saat yang sama, kode sering kali lebih cepat dibandingkan apa yang bisa dicapai dalam pemrograman imperatif.
API tingkat rendah yang memungkinkan kita untuk menyimpan dalam TensorFlow adalah `tf.saved_model`.
Mari kita lihat instance `saved_model` dalam aksi.
:end_tab:


```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
Model ini dipecah menjadi file parameter (biner besar) dan deskripsi JSON dari program yang diperlukan untuk melakukan komputasi model. File-file ini dapat dibaca oleh bahasa front-end lainnya yang didukung oleh Python atau MXNet, seperti C++, R, Scala, dan Perl. Mari kita lihat beberapa baris pertama dalam deskripsi model.
:end_tab:


```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
Sebelumnya, kami menunjukkan bahwa setelah memanggil fungsi `hybridize`, model dapat mencapai kinerja komputasi dan portabilitas yang lebih unggul. Namun, perlu dicatat bahwa hybridisasi dapat mempengaruhi fleksibilitas model, khususnya dalam hal aliran kontrol.

Selain itu, berbeda dengan instance `Block`, yang perlu menggunakan fungsi `forward`, untuk instance `HybridBlock` kita perlu menggunakan fungsi `hybrid_forward`.
:end_tab:



```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
Kode di atas mengimplementasikan jaringan sederhana dengan 4 unit tersembunyi dan 2 keluaran. Fungsi `hybrid_forward` menerima argumen tambahan `F`. Hal ini diperlukan karena, tergantung pada apakah kode telah di-hybridisasi atau belum, kode tersebut akan menggunakan pustaka yang sedikit berbeda (`ndarray` atau `symbol`) untuk pemrosesan. Kedua kelas ini melakukan fungsi yang sangat mirip, dan MXNet secara otomatis menentukan argumennya. Untuk memahami apa yang sedang terjadi, kita mencetak argumen sebagai bagian dari pemanggilan fungsi.
:end_tab:


```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
Mengulangi komputasi forward akan menghasilkan keluaran yang sama (detailnya diabaikan). Sekarang mari kita lihat apa yang terjadi jika kita memanggil fungsi `hybridize`.
:end_tab:


```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Alih-alih menggunakan modul `ndarray`, kita sekarang menggunakan modul `symbol` untuk `F`. Selain itu, meskipun inputnya adalah tipe `ndarray`, data yang mengalir melalui jaringan sekarang dikonversi ke tipe `symbol` sebagai bagian dari proses kompilasi. Mengulangi pemanggilan fungsi menghasilkan hasil yang mengejutkan:
:end_tab:


```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
Ini sangat berbeda dari yang kita lihat sebelumnya. Semua pernyataan print, seperti yang didefinisikan dalam `hybrid_forward`, dihilangkan. Memang, setelah hybridisasi, eksekusi `net(x)` tidak lagi melibatkan interpreter Python. Ini berarti bahwa setiap kode Python yang tidak penting dihilangkan (seperti pernyataan print) demi eksekusi yang jauh lebih efisien dan kinerja yang lebih baik. Sebagai gantinya, MXNet langsung memanggil backend C++. Perhatikan juga bahwa beberapa fungsi tidak didukung dalam modul `symbol` (misalnya, `asnumpy`) dan operasi in-place seperti `a += b` dan `a[:] = a + b` harus ditulis ulang sebagai `a = a + b`. Namun demikian, kompilasi model sangat berharga ketika kecepatan sangat penting. Manfaatnya dapat berkisar dari peningkatan persentase kecil hingga lebih dari dua kali lipat kecepatan, tergantung pada kompleksitas model, kecepatan CPU, serta kecepatan dan jumlah GPU.
:end_tab:

## Ringkasan

* Pemrograman imperatif memudahkan untuk merancang model baru karena memungkinkan menulis kode dengan aliran kontrol dan menggunakan sebagian besar ekosistem perangkat lunak Python.
* Pemrograman simbolik mengharuskan kita menentukan program dan mengompilasinya sebelum mengeksekusinya. Manfaatnya adalah peningkatan kinerja.

:begin_tab:`mxnet` 
* MXNet mampu menggabungkan keuntungan dari kedua pendekatan sesuai kebutuhan.
* Model yang dibangun oleh kelas `HybridSequential` dan `HybridBlock` dapat mengubah program imperatif menjadi program simbolik dengan memanggil fungsi `hybridize`.
:end_tab:

## Latihan

:begin_tab:`mxnet` 
1. Tambahkan `x.asnumpy()` ke baris pertama dari fungsi `hybrid_forward` pada kelas `HybridNet` di bagian ini. Jalankan kode dan amati kesalahan yang terjadi. Mengapa hal ini terjadi?
2. Apa yang terjadi jika kita menambahkan aliran kontrol, misalnya pernyataan Python `if` dan `for` dalam fungsi `hybrid_forward`?
3. Tinjau model yang menarik bagi Anda di bab sebelumnya. Bisakah Anda meningkatkan kinerja komputasi mereka dengan mengimplementasikannya kembali?
:end_tab:

:begin_tab:`pytorch,tensorflow` 
1. Tinjau model yang menarik bagi Anda di bab sebelumnya. Bisakah Anda meningkatkan kinerja komputasi mereka dengan mengimplementasikannya kembali?
:end_tab:

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/2492)
:end_tab:
