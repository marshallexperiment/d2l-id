# Pelatihan pada Banyak GPU
:label:`sec_multi_gpu`

Sejauh ini kita telah membahas cara melatih model secara efisien pada CPU dan GPU. Kita bahkan menunjukkan bagaimana framework deep learning memungkinkan seseorang untuk melakukan paralelisasi komputasi dan komunikasi secara otomatis di antara keduanya pada :numref:`sec_auto_para`. Kita juga menunjukkan pada :numref:`sec_use_gpu` bagaimana cara mendaftar semua GPU yang tersedia di komputer menggunakan perintah `nvidia-smi`.
Namun, yang belum kita bahas adalah bagaimana sebenarnya melakukan paralelisasi pelatihan deep learning.
Sebaliknya, kita sempat mengisyaratkan bahwa kita dapat membagi data di beberapa perangkat dan membuatnya bekerja. Bagian ini akan menjelaskan detail tersebut dan menunjukkan cara melatih jaringan secara paralel dari awal. Detail tentang cara memanfaatkan fungsionalitas dalam API tingkat tinggi akan dibahas pada :numref:`sec_multi_gpu_concise`.
Kami mengasumsikan bahwa Anda sudah familiar dengan algoritma minibatch stochastic gradient descent seperti yang dijelaskan pada :numref:`sec_minibatch_sgd`.

## Membagi Masalah

Mari kita mulai dengan masalah visi komputer yang sederhana dan jaringan yang sedikit kuno, misalnya dengan beberapa lapisan konvolusi, pooling, dan mungkin beberapa lapisan fully connected pada akhirnya.
Artinya, mari kita mulai dengan jaringan yang terlihat cukup mirip dengan LeNet :cite:`LeCun.Bottou.Bengio.ea.1998` atau AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`.
Diberikan beberapa GPU (2 jika itu adalah server desktop, 4 pada instance AWS g4dn.12xlarge, 8 pada p3.16xlarge, atau 16 pada p2.16xlarge), kita ingin membagi pelatihan sedemikian rupa untuk mencapai peningkatan kecepatan yang baik sambil tetap mendapatkan keuntungan dari pilihan desain yang sederhana dan dapat direproduksi. Banyak GPU, bagaimanapun, meningkatkan *memori* dan kemampuan *komputasi*. Singkatnya, kita memiliki beberapa pilihan berikut, mengingat minibatch data pelatihan yang ingin kita klasifikasikan.

Pertama, kita dapat membagi jaringan di beberapa GPU. Artinya, setiap GPU menerima data yang mengalir ke lapisan tertentu, memproses data di sejumlah lapisan berikutnya, dan kemudian mengirimkan data ke GPU berikutnya.
Ini memungkinkan kita memproses data dengan jaringan yang lebih besar dibandingkan dengan apa yang dapat ditangani oleh satu GPU.
Selain itu,
jejak memori per GPU dapat dikendalikan dengan baik (yaitu merupakan bagian dari total jejak jaringan).

Namun, antarmuka antar lapisan (dan dengan demikian antar GPU) membutuhkan sinkronisasi yang ketat. Ini bisa menjadi rumit, terutama jika beban kerja komputasi tidak cocok di antara lapisan. Masalah ini diperparah jika jumlah GPU sangat besar.
Antarmuka antar lapisan juga
membutuhkan transfer data dalam jumlah besar,
seperti aktivasi dan gradien.
Ini bisa membebani bandwidth bus GPU.
Selain itu, operasi intensif-komputasi yang bersifat sekuensial tidak mudah dibagi. Lihat misalnya :citet:`Mirhoseini.Pham.Le.ea.2017` untuk upaya terbaik dalam hal ini. Ini tetap menjadi masalah yang sulit dan belum jelas apakah mungkin untuk mencapai skalabilitas yang baik (linear) pada masalah non-sepele. Kami tidak merekomendasikan pendekatan ini kecuali terdapat dukungan framework atau sistem operasi yang sangat baik untuk menghubungkan beberapa GPU.

Kedua, kita bisa membagi pekerjaan per lapisan. Misalnya, daripada menghitung 64 channel pada satu GPU, kita bisa membagi masalah tersebut ke 4 GPU, masing-masing menghasilkan data untuk 16 channel.
Demikian pula, untuk lapisan fully connected, kita bisa membagi jumlah unit output. :numref:`fig_alexnet_original` (diambil dari :citet:`Krizhevsky.Sutskever.Hinton.2012`)
mengilustrasikan desain ini, di mana strategi ini digunakan untuk menangani GPU yang memiliki footprint memori yang sangat kecil (2 GB pada saat itu).
Ini memungkinkan skalabilitas yang baik dalam hal komputasi, dengan catatan jumlah channel (atau unit) tidak terlalu kecil.
Selain itu,
beberapa GPU dapat memproses jaringan yang semakin besar karena memori yang tersedia bertambah secara linear.

![Paralelisme model dalam desain AlexNet asli karena keterbatasan memori GPU.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

Namun,
kita membutuhkan jumlah operasi sinkronisasi atau penghalang yang *sangat besar* karena setiap lapisan bergantung pada hasil dari semua lapisan lainnya.
Selain itu, jumlah data yang perlu ditransfer berpotensi lebih besar daripada ketika mendistribusikan lapisan antar GPU. Oleh karena itu, kami tidak merekomendasikan pendekatan ini karena biaya bandwidth dan kompleksitasnya.

Terakhir, kita bisa membagi data di beberapa GPU. Dengan cara ini semua GPU melakukan jenis pekerjaan yang sama, meskipun pada pengamatan yang berbeda. Gradien dikumpulkan di antara GPU setelah setiap minibatch data pelatihan.
Ini adalah pendekatan yang paling sederhana dan dapat diterapkan dalam situasi apa pun.
Kita hanya perlu melakukan sinkronisasi setelah setiap minibatch. Meski begitu, sangat diinginkan untuk mulai bertukar parameter gradien saat yang lain masih dihitung.
Selain itu, semakin banyak GPU berarti semakin besar ukuran minibatch, sehingga meningkatkan efisiensi pelatihan.
Namun, menambah jumlah GPU tidak memungkinkan kita melatih model yang lebih besar.

![Paralelisasi pada beberapa GPU. Dari kiri ke kanan: masalah asli, pembagian jaringan, pembagian lapisan, paralelisme data.](../img/splitting.svg)
:label:`fig_splitting`

Perbandingan berbagai cara paralelisasi pada beberapa GPU ditunjukkan pada :numref:`fig_splitting`.
Secara umum, paralelisme data adalah cara yang paling mudah untuk dilanjutkan, asalkan kita memiliki akses ke GPU dengan memori yang cukup besar. Lihat juga :cite:`Li.Andersen.Park.ea.2014` untuk deskripsi detail tentang pembagian untuk pelatihan terdistribusi. Memori GPU dulu menjadi masalah di masa-masa awal deep learning. Saat ini masalah ini telah diselesaikan untuk semua kecuali kasus yang paling tidak biasa. Kami akan fokus pada paralelisme data dalam bagian berikutnya.

## Paralelisme Data

Misalkan ada $k$ GPU pada sebuah mesin. Mengingat model yang akan dilatih, setiap GPU akan mempertahankan satu set lengkap parameter model secara independen meskipun nilai parameter di seluruh GPU adalah identik dan disinkronkan.
Sebagai contoh,
:numref:`fig_data_parallel` mengilustrasikan
pelatihan dengan
paralelisme data ketika $k=2$.

![Perhitungan stochastic gradient descent menggunakan paralelisme data pada dua GPU.](../img/data-parallel.svg)
:label:`fig_data_parallel`

Secara umum, pelatihan berlangsung sebagai berikut:

* Dalam iterasi pelatihan apa pun, diberikan satu minibatch acak, kita membagi contoh dalam batch tersebut menjadi $k$ bagian dan mendistribusikannya secara merata di seluruh GPU.
* Setiap GPU menghitung loss dan gradien parameter model berdasarkan subset minibatch yang ditugaskan.
* Gradien lokal dari masing-masing $k$ GPU dikumpulkan untuk mendapatkan minibatch stochastic gradient saat ini.
* Gradien yang dikumpulkan didistribusikan kembali ke setiap GPU.
* Setiap GPU menggunakan stochastic gradient ini untuk memperbarui set lengkap parameter model yang dipertahankannya.

Perhatikan bahwa dalam praktik kita *meningkatkan* ukuran minibatch sebesar $k$-lipat ketika melatih pada $k$ GPU sehingga setiap GPU memiliki jumlah pekerjaan yang sama seperti jika kita hanya melatih pada satu GPU saja. Pada server dengan 16 GPU, ini dapat meningkatkan ukuran minibatch secara signifikan dan kita mungkin harus meningkatkan laju pembelajaran sesuai.
Juga perhatikan bahwa batch normalization pada :numref:`sec_batch_norm` perlu disesuaikan, misalnya dengan mempertahankan koefisien batch normalization yang terpisah per GPU.
Selanjutnya, kita akan menggunakan jaringan sederhana untuk mengilustrasikan pelatihan multi-GPU.


```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**Jaringan Mainan**]

Kita menggunakan LeNet seperti yang diperkenalkan pada :numref:`sec_lenet` (dengan sedikit modifikasi). Kita mendefinisikannya dari awal untuk mengilustrasikan pertukaran parameter dan sinkronisasi secara rinci.


```{.python .input}
#@tab mxnet
# Insialisasi parameter 
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Insialisasi parameter
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## Sinkronisasi Data

Untuk pelatihan multi-GPU yang efisien, kita memerlukan dua operasi dasar.
Pertama, kita perlu memiliki kemampuan untuk [**mendistribusikan daftar parameter ke beberapa perangkat**] dan melampirkan gradien (`get_params`). Tanpa parameter, tidak mungkin mengevaluasi jaringan pada GPU.
Kedua, kita memerlukan kemampuan untuk menjumlahkan parameter di berbagai perangkat, yaitu, kita memerlukan fungsi `allreduce`.


```{.python .input}
#@tab mxnet
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

Mari kita coba dengan menyalin parameter model ke satu GPU.


```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

Karena kita belum melakukan komputasi apa pun, gradien terkait dengan parameter bias masih bernilai nol.
Sekarang misalkan kita memiliki sebuah vektor yang didistribusikan di beberapa GPU. Fungsi [**`allreduce` berikut menjumlahkan semua vektor dan menyiarkan hasilnya kembali ke semua GPU**]. Perhatikan bahwa agar ini dapat bekerja, kita perlu menyalin data ke perangkat yang mengakumulasi hasilnya.



```{.python .input}
#@tab mxnet
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

Mari kita uji ini dengan membuat vektor dengan nilai yang berbeda pada perangkat yang berbeda dan mengagregasikannya.



```{.python .input}
#@tab mxnet
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('Sebelum allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('Sesudah allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('Sebelum allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('Sesudah allreduce:\n', data[0], '\n', data[1])
```

## Mendistribusikan Data

Kita memerlukan fungsi utilitas sederhana untuk [**mendistribusikan minibatch secara merata di beberapa GPU**]. Misalnya, pada dua GPU, kita ingin agar setengah dari data disalin ke masing-masing GPU.
Karena lebih nyaman dan ringkas, kita menggunakan fungsi bawaan dari framework deep learning untuk mencobanya pada matriks berukuran $4 \times 5$.


```{.python .input}
#@tab mxnet
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

Untuk digunakan kembali di kemudian hari, kita mendefinisikan fungsi `split_batch` yang membagi data dan label.



```{.python .input}
#@tab mxnet
#@save
def split_batch(X, y, devices):
    """Split `X` dan `y` ke banyak devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` dan `y` ke banyak devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## Pelatihan

Sekarang kita bisa mengimplementasikan [**pelatihan multi-GPU pada satu minibatch**]. Implementasinya terutama didasarkan pada pendekatan paralelisme data yang dijelaskan di bagian ini.
Kita akan menggunakan fungsi tambahan yang baru saja kita bahas, yaitu `allreduce` dan `split_and_load`, untuk menyinkronkan data di antara beberapa GPU. 
Perhatikan bahwa kita tidak perlu menulis kode khusus untuk mencapai paralelisme.
Karena grafik komputasi tidak memiliki ketergantungan antar perangkat dalam satu minibatch, maka grafik tersebut dieksekusi secara paralel *secara otomatis*.


```{.python .input}
#@tab mxnet
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss dihitung secara terpisah pada setiap GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagasi dilakukan secara terpisah pada setiap GPU
        l.backward()
    # Jumlahkan semua gradien dari setiap GPU dan siarkan hasilnya ke semua GPU
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # Parameter model diperbarui secara terpisah pada setiap GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Di sini, kita menggunakan ukuran batch penuh
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss dihitung secara terpisah pada setiap GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagasi dilakukan secara terpisah pada setiap GPU
        l.backward()
    # Jumlahkan semua gradien dari setiap GPU dan siarkan hasilnya ke semua GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # Parameter model diperbarui secara terpisah pada setiap GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Di sini, kita menggunakan ukuran batch penuh
```

Sekarang, kita bisa mendefinisikan [**fungsi pelatihan**]. Fungsi ini sedikit berbeda dari yang digunakan di bab-bab sebelumnya: kita perlu mengalokasikan GPU dan menyalin semua parameter model ke semua perangkat.
Jelas bahwa setiap batch diproses menggunakan fungsi `train_batch` untuk menangani beberapa GPU. Untuk kemudahan (dan keringkasan kode), kita menghitung akurasi pada satu GPU saja, meskipun ini *tidak efisien* karena GPU lainnya dalam keadaan tidak aktif.


```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Menyalin parameter model ke `num_gpus` GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Melakukan pelatihan multi-GPU untuk satu minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Mengevaluasi model pada GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Menyalin parameter model ke `num_gpus` GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Melakukan pelatihan multi-GPU untuk satu minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()  # Menyinkronkan semua operasi GPU
        timer.stop()
        # Mengevaluasi model pada GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Mari kita lihat seberapa baik ini bekerja [**pada satu GPU**].
Pertama, kita menggunakan ukuran batch sebesar 256 dan laju pembelajaran sebesar 0.2.


```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

Dengan mempertahankan ukuran batch dan laju pembelajaran tetap tidak berubah dan [**meningkatkan jumlah GPU menjadi 2**], kita dapat melihat bahwa akurasi pengujian kurang lebih tetap sama dibandingkan dengan percobaan sebelumnya.
Dalam hal algoritma optimasi, mereka identik. Sayangnya, tidak ada peningkatan kecepatan yang signifikan di sini: model ini terlalu kecil; selain itu, kita hanya memiliki dataset kecil, di mana pendekatan kita yang kurang canggih dalam mengimplementasikan pelatihan multi-GPU mengalami overhead Python yang signifikan. Ke depannya, kita akan menemui model yang lebih kompleks dan cara paralelisasi yang lebih canggih.
Mari kita lihat apa yang terjadi pada Fashion-MNIST meskipun demikian.


```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## Ringkasan

* Ada beberapa cara untuk membagi pelatihan jaringan dalam di beberapa GPU. Kita bisa membaginya di antara lapisan, di seluruh lapisan, atau di seluruh data. Dua yang pertama memerlukan transfer data yang sangat terkoordinasi. Paralelisme data adalah strategi yang paling sederhana.
* Pelatihan dengan paralelisme data cukup mudah. Namun, ini meningkatkan ukuran minibatch efektif agar efisien.
* Dalam paralelisme data, data dibagi di beberapa GPU, di mana setiap GPU menjalankan operasi forward dan backward sendiri, kemudian gradien dikumpulkan dan hasilnya disiarkan kembali ke GPU.
* Kita dapat menggunakan laju pembelajaran yang sedikit lebih tinggi untuk minibatch yang lebih besar.

## Latihan

1. Saat melatih pada $k$ GPU, ubahlah ukuran minibatch dari $b$ menjadi $k \cdot b$, yaitu skalakan sesuai dengan jumlah GPU.
2. Bandingkan akurasi untuk berbagai laju pembelajaran. Bagaimana skala tersebut dengan jumlah GPU?
3. Implementasikan fungsi `allreduce` yang lebih efisien yang mengumpulkan parameter yang berbeda pada GPU yang berbeda? Mengapa ini lebih efisien?
4. Implementasikan komputasi akurasi uji multi-GPU.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1669)
:end_tab:
