# Minibatch Stochastic Gradient Descent
:label:`sec_minibatch_sgd`

Sejauh ini kita telah menemukan dua pendekatan ekstrim dalam pembelajaran berbasis gradien: :numref:`sec_gd` menggunakan seluruh dataset untuk menghitung gradien dan memperbarui parameter, satu lintasan dalam satu waktu. Sebaliknya, :numref:`sec_sgd` memproses satu contoh pelatihan setiap kali untuk membuat kemajuan.
Keduanya memiliki kekurangan masing-masing.
Gradient descent tidak terlalu *efisien dalam data* ketika data sangat mirip.
Stochastic gradient descent tidak terlalu *efisien secara komputasi* karena CPU dan GPU tidak dapat sepenuhnya memanfaatkan kekuatan vektorisasi.
Hal ini menunjukkan bahwa mungkin ada sesuatu di antara keduanya,
dan sebenarnya, itulah yang telah kita gunakan sejauh ini dalam contoh-contoh yang kita bahas.

## Vektorisasi dan Cache

Inti dari keputusan untuk menggunakan minibatch adalah efisiensi komputasi. Hal ini paling mudah dipahami ketika mempertimbangkan paralelisasi ke beberapa GPU dan beberapa server. Dalam kasus ini, kita perlu mengirim setidaknya satu gambar ke setiap GPU. Dengan 8 GPU per server dan 16 server, kita sudah sampai pada ukuran minibatch yang tidak lebih kecil dari 128.

Situasi menjadi sedikit lebih halus ketika berurusan dengan satu GPU atau bahkan CPU. Perangkat ini memiliki beberapa jenis memori, sering kali beberapa jenis unit komputasi, dan batasan bandwidth yang berbeda-beda di antara mereka.
Misalnya, sebuah CPU memiliki sejumlah register kecil dan kemudian cache L1, L2, dan bahkan L3 dalam beberapa kasus (yang dibagi di antara berbagai inti prosesor).
Cache-cache ini memiliki ukuran dan latensi yang meningkat (dan pada saat yang sama bandwidth yang menurun).
Singkatnya, prosesor mampu melakukan lebih banyak operasi daripada antarmuka memori utama yang mampu menyediakannya.

Pertama, CPU 2GHz dengan 16 inti dan vektorisasi AVX-512 dapat memproses hingga $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ byte per detik. Kemampuan GPU dengan mudah melebihi angka ini hingga 100 kali lipat. Di sisi lain, prosesor server kelas menengah mungkin tidak memiliki lebih dari 100 GB/s bandwidth, yaitu kurang dari sepersepuluh dari apa yang dibutuhkan untuk memberi makan prosesor. Lebih buruk lagi, tidak semua akses memori sama: antarmuka memori biasanya memiliki lebar 64 bit atau lebih (misalnya, pada GPU hingga 384 bit), sehingga membaca satu byte mengakibatkan biaya akses yang lebih lebar.

Kedua, ada overhead yang signifikan untuk akses pertama sementara akses berurutan relatif murah (ini sering disebut sebagai burst read). Ada banyak hal lagi yang perlu diingat, seperti caching saat kita memiliki beberapa soket, chiplet, dan struktur lainnya.
Lihat [artikel Wikipedia ini](https://en.wikipedia.org/wiki/Cache_hierarchy)
untuk pembahasan lebih mendalam.

Cara untuk mengurangi keterbatasan ini adalah dengan menggunakan hierarki cache CPU yang benar-benar cukup cepat untuk memasok prosesor dengan data. Inilah pendorong utama di balik batching dalam deep learning. Untuk menyederhanakan, pertimbangkan perkalian matriks-matriks, misalnya $\mathbf{A} = \mathbf{B}\mathbf{C}$. Kita memiliki beberapa opsi untuk menghitung $\mathbf{A}$. Misalnya, kita dapat mencoba hal berikut:

1. Kita dapat menghitung $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$, yaitu kita dapat menghitungnya secara elemen-wise melalui dot product.
2. Kita dapat menghitung $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$, yaitu kita dapat menghitungnya satu kolom pada satu waktu. Demikian pula, kita dapat menghitung $\mathbf{A}$ satu baris $\mathbf{A}_{i,:}$ pada satu waktu.
3. Kita dapat menghitung $\mathbf{A} = \mathbf{B} \mathbf{C}$ secara langsung.
4. Kita dapat memecah $\mathbf{B}$ dan $\mathbf{C}$ menjadi blok-blok matriks yang lebih kecil dan menghitung $\mathbf{A}$ satu blok pada satu waktu.

Jika kita mengikuti opsi pertama, kita perlu menyalin satu baris dan satu kolom vektor ke dalam CPU setiap kali kita ingin menghitung elemen $\mathbf{A}_{ij}$. Lebih buruk lagi, karena elemen matriks diatur secara berurutan, kita harus mengakses banyak lokasi yang tidak berurutan untuk salah satu dari dua vektor saat membacanya dari memori. Opsi kedua jauh lebih menguntungkan. Dalam hal ini, kita dapat menyimpan vektor kolom $\mathbf{C}_{:,j}$ dalam cache CPU sementara kita terus melakukan traversal melalui $\mathbf{B}$. Ini mengurangi setengah kebutuhan bandwidth memori dengan akses yang lebih cepat. Tentu saja, opsi 3 adalah yang paling diinginkan. Sayangnya, sebagian besar matriks mungkin tidak sepenuhnya muat di cache (inilah yang sedang kita bahas). Namun, opsi 4 menawarkan alternatif yang berguna secara praktis: kita dapat memindahkan blok-blok matriks ke dalam cache dan mengalikannya secara lokal. Pustaka yang dioptimalkan mengurus hal ini untuk kita. Mari kita lihat seberapa efisien operasi-operasi ini dalam praktik.

Selain efisiensi komputasi, overhead yang diperkenalkan oleh Python dan oleh framework deep learning itu sendiri cukup besar. Ingat bahwa setiap kali kita menjalankan perintah, interpreter Python mengirim perintah ke engine MXNet yang perlu memasukkannya ke dalam computational graph dan menangani scheduling. Overhead seperti itu dapat sangat merugikan. Singkatnya, sangat disarankan untuk menggunakan vektorisasi (dan matriks) jika memungkinkan.



```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Karena kita akan mengukur waktu eksekusi secara berkala sepanjang sisa buku ini, mari kita definisikan sebuah timer.


```{.python .input}
#@tab all
class Timer:  #@save
    """Mencatat beberapa waktu eksekusi."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Mulai timer."""
        self.tik = time.time()

    def stop(self):
        """Hentikan timer dan catat waktu dalam sebuah list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Kembalikan waktu rata-rata."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Kembalikan jumlah total waktu."""
        return sum(self.times)

    def cumsum(self):
        """Kembalikan waktu yang terakumulasi."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

Penugasan elemen-wise hanya mengiterasi semua baris dan kolom dari $\mathbf{B}$ dan $\mathbf{C}$ masing-masing untuk menetapkan nilai ke $\mathbf{A}$.

```{.python .input}
#@tab mxnet
# Hitung A = BC satu elemen pada satu waktu
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Hitung A = BC satu elemen pada satu waktu
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Hitung A = BC satu elemen pada satu waktu
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Strategi yang lebih cepat adalah melakukan penugasan secara kolom.


```{.python .input}
#@tab mxnet
# Hitung A = BC satu kolom pada satu waktu
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Hitung A = BC satu kolom pada satu waktu
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```


Terakhir, cara paling efektif adalah melakukan seluruh operasi dalam satu blok.
Perhatikan bahwa mengalikan dua matriks $\mathbf{B} \in \mathbb{R}^{m \times n}$ dan $\mathbf{C} \in \mathbb{R}^{n \times p}$ membutuhkan sekitar $2mnp$ operasi titik mengambang,
ketika perkalian skalar dan penjumlahan dihitung sebagai operasi terpisah (digabungkan dalam praktiknya).
Jadi, mengalikan dua matriks $256 \times 256$
memerlukan $0.03$ miliar operasi titik mengambang.
Mari kita lihat seberapa cepat masing-masing operasi tersebut.



```{.python .input}
#@tab mxnet
# Hitung A = BC sekaligus
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Hitung A = BC sekaligus
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatches

:label:`sec_minibatches`

Sebelumnya, kita menerima begitu saja bahwa kita akan membaca *minibatches* data daripada satu observasi untuk memperbarui parameter. Sekarang kita akan memberikan justifikasi singkat untuk hal ini. Memproses observasi tunggal mengharuskan kita melakukan banyak perkalian matriks-vektor (atau bahkan vektor-vektor), yang cukup mahal dan menimbulkan overhead signifikan pada framework deep learning yang mendasarinya. Hal ini berlaku baik untuk mengevaluasi jaringan yang diterapkan pada data (sering disebut sebagai inference) maupun saat menghitung gradien untuk memperbarui parameter. Artinya, hal ini berlaku kapanpun kita melakukan $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ dimana

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Kita dapat meningkatkan efisiensi *komputasi* dari operasi ini dengan menerapkannya pada minibatch observasi sekaligus. Artinya, kita menggantikan gradien $\mathbf{g}_t$ pada satu observasi dengan satu gradien pada batch kecil

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Mari kita lihat apa efeknya terhadap properti statistik dari $\mathbf{g}_t$: karena baik $\mathbf{x}_t$ maupun semua elemen dari minibatch $\mathcal{B}_t$ diambil secara acak dari set pelatihan, ekspektasi dari gradien tetap tidak berubah. Varians, di sisi lain, berkurang secara signifikan. Karena gradien pada minibatch terdiri dari $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ gradien independen yang sedang diambil rata-ratanya, standar deviasi berkurang dengan faktor $b^{-\frac{1}{2}}$. Ini, pada dasarnya, adalah hal yang baik, karena ini berarti bahwa pembaruan lebih selaras dengan gradien penuh secara andal.

Secara naif, hal ini menunjukkan bahwa memilih minibatch yang besar $\mathcal{B}_t$ akan selalu diinginkan. Sayangnya, setelah beberapa titik, pengurangan tambahan dalam standar deviasi menjadi minimal dibandingkan dengan peningkatan linier dalam biaya komputasi. Dalam praktiknya, kita memilih minibatch yang cukup besar untuk menawarkan efisiensi komputasi yang baik namun masih dapat muat dalam memori GPU. Untuk mengilustrasikan penghematan, mari kita lihat beberapa kode. Dalam kode tersebut, kita melakukan perkalian matriks-matriks yang sama, tetapi kali ini dipecah menjadi "minibatches" dari 64 kolom sekaligus.


```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performa di Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performa di Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performa di Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

Seperti yang kita lihat, komputasi pada minibatch pada dasarnya sama efisiennya dengan pada matriks penuh. Namun, ada hal yang perlu diperhatikan. Dalam :numref:`sec_batch_norm` kita menggunakan jenis regularisasi yang sangat bergantung pada jumlah varians dalam sebuah minibatch. Saat kita meningkatkan jumlah tersebut, varians menurun, dan seiring dengan itu manfaat dari injeksi noise akibat batch normalization juga menurun. Lihat misalnya, :citet:`Ioffe.2017` untuk detail tentang cara reskalasi dan perhitungan istilah yang sesuai.

## Membaca Dataset

Mari kita lihat bagaimana minibatch dapat dihasilkan secara efisien dari data. Dalam contoh berikut, kita menggunakan dataset yang dikembangkan oleh NASA untuk menguji [kebisingan sayap dari berbagai pesawat](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) untuk membandingkan algoritma optimasi ini. Untuk kenyamanan, kita hanya menggunakan $1,500$ contoh pertama. Data diproses terlebih dahulu dengan metode whitening, yaitu kita menghilangkan rata-rata dan menyesuaikan varians menjadi $1$ per koordinat.


```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Implementasi dari Awal

Ingat implementasi stochastic gradient descent dengan minibatch dari :numref:`sec_linear_scratch`. Di bagian berikut, kami menyediakan implementasi yang sedikit lebih umum. Untuk kenyamanan, implementasi ini memiliki format pemanggilan yang sama dengan algoritma optimasi lainnya yang diperkenalkan nanti di bab ini. Secara khusus, kami menambahkan input status `states` dan menempatkan hiperparameter dalam dictionary `hyperparams`. Selain itu, kami akan mengambil rata-rata dari loss setiap contoh dalam minibatch di fungsi pelatihan, sehingga gradien dalam algoritma optimasi tidak perlu dibagi dengan ukuran batch.



```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Selanjutnya, kita mengimplementasikan fungsi pelatihan umum untuk memfasilitasi penggunaan algoritma optimasi lain yang diperkenalkan di bab ini. Fungsi ini menginisialisasi model regresi linier dan dapat digunakan untuk melatih model dengan stochastic gradient descent menggunakan minibatch dan algoritma lain yang akan diperkenalkan kemudian.


```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Mari kita lihat bagaimana optimasi berjalan untuk batch gradient descent. Hal ini dapat dicapai dengan mengatur ukuran minibatch menjadi 1500 (yaitu, sama dengan jumlah total contoh). Akibatnya, parameter model hanya diperbarui sekali per epoch. Hasilnya menunjukkan sedikit kemajuan. Bahkan, setelah 6 langkah, kemajuan menjadi terhenti.


```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Ketika ukuran batch sama dengan 1, kita menggunakan stochastic gradient descent untuk optimasi. Untuk kesederhanaan implementasi, kita memilih nilai learning rate yang konstan (meskipun kecil). Dalam stochastic gradient descent, parameter model diperbarui setiap kali contoh diproses. Dalam kasus kita, ini berarti 1500 pembaruan per epoch. Seperti yang kita lihat, penurunan nilai dari fungsi objektif melambat setelah satu epoch. Meskipun kedua prosedur memproses 1500 contoh dalam satu epoch, stochastic gradient descent menghabiskan lebih banyak waktu daripada gradient descent dalam percobaan kita. Ini karena stochastic gradient descent memperbarui parameter lebih sering dan kurang efisien untuk memproses satu observasi setiap kali.


```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Terakhir, ketika ukuran batch sama dengan 100, kita menggunakan stochastic gradient descent dengan minibatch untuk optimasi. Waktu yang diperlukan per epoch lebih pendek dibandingkan dengan waktu yang dibutuhkan untuk stochastic gradient descent dan juga untuk batch gradient descent.


```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Dengan mengurangi ukuran batch menjadi 10, waktu untuk setiap epoch meningkat karena beban kerja untuk setiap batch menjadi kurang efisien untuk dieksekusi.



```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Sekarang kita dapat membandingkan waktu vs. loss untuk keempat percobaan sebelumnya. Seperti yang dapat dilihat, meskipun stochastic gradient descent berkonvergensi lebih cepat dibandingkan dengan GD dalam hal jumlah contoh yang diproses, ia membutuhkan lebih banyak waktu untuk mencapai nilai loss yang sama dibandingkan GD karena menghitung gradien per contoh tidak seefisien metode batch. Minibatch stochastic gradient descent mampu melakukan trade-off antara kecepatan konvergensi dan efisiensi komputasi. Ukuran minibatch 10 lebih efisien dibandingkan dengan stochastic gradient descent; ukuran minibatch 100 bahkan mengungguli GD dalam hal waktu eksekusi.




```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Implementasi Singkat

Di Gluon, kita dapat menggunakan kelas `Trainer` untuk memanggil algoritma optimasi. Ini digunakan untuk mengimplementasikan fungsi pelatihan umum. Kita akan menggunakan ini sepanjang bab ini.


```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` menghitung error kuadrat tanpa faktor 1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
               # `MeanSquaredError` menghitung error kuadrat tanpa faktor 1/2
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Menggunakan Gluon untuk mengulangi percobaan terakhir menunjukkan perilaku yang identik.


```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Ringkasan

* Vektorisasi membuat kode lebih efisien karena overhead yang berkurang akibat framework deep learning dan karena peningkatan locality memori dan caching pada CPU dan GPU.
* Ada trade-off antara efisiensi statistik yang diperoleh dari stochastic gradient descent dan efisiensi komputasi yang diperoleh dari memproses batch data yang besar sekaligus.
* Stochastic gradient descent dengan minibatch menawarkan yang terbaik dari kedua dunia: efisiensi komputasi dan statistik.
* Dalam stochastic gradient descent dengan minibatch, kita memproses batch data yang diperoleh melalui permutasi acak dari data pelatihan (yaitu, setiap observasi hanya diproses sekali per epoch, meskipun dalam urutan acak).
* Sangat disarankan untuk menurunkan learning rate selama pelatihan.
* Secara umum, stochastic gradient descent dengan minibatch lebih cepat daripada stochastic gradient descent dan gradient descent untuk konvergensi pada risiko yang lebih kecil, jika diukur dalam waktu nyata (clock time).

## Latihan

1. Modifikasi ukuran batch dan learning rate, lalu amati laju penurunan nilai fungsi objektif dan waktu yang dikonsumsi di setiap epoch.
1. Baca dokumentasi MXNet dan gunakan fungsi `set_learning_rate` dari kelas `Trainer` untuk mengurangi learning rate dari stochastic gradient descent dengan minibatch menjadi 1/10 dari nilai sebelumnya setelah setiap epoch.
1. Bandingkan stochastic gradient descent dengan minibatch dengan varian yang sebenarnya *mengambil sampel dengan pengembalian* dari set pelatihan. Apa yang terjadi?
1. Seorang jin jahat menggandakan dataset Anda tanpa memberi tahu Anda (yaitu, setiap observasi terjadi dua kali dan dataset Anda menjadi dua kali lipat ukuran aslinya, tetapi tidak ada yang memberi tahu Anda). Bagaimana perilaku stochastic gradient descent, stochastic gradient descent dengan minibatch, dan gradient descent berubah?

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1069)
:end_tab:

