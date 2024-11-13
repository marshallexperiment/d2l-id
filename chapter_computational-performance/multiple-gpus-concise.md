# Implementasi Ringkas untuk Beberapa GPU
:label:`sec_multi_gpu_concise`

Mengimplementasikan paralelisme dari awal untuk setiap model baru tidaklah menyenangkan. Selain itu, ada manfaat signifikan dalam mengoptimalkan alat-alat sinkronisasi untuk kinerja yang tinggi. Di bagian berikut ini, kita akan menunjukkan cara melakukan ini menggunakan API level tinggi dari framework deep learning.
Matematika dan algoritma yang digunakan sama seperti di :numref:`sec_multi_gpu`.
Seperti yang sudah bisa diperkirakan, Anda memerlukan setidaknya dua GPU untuk menjalankan kode di bagian ini.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Jaringan Mainan (Toy Network)**]

Mari kita gunakan jaringan yang sedikit lebih bermakna daripada LeNet dari :numref:`sec_multi_gpu`, namun tetap cukup mudah dan cepat untuk dilatih.
Kami memilih varian ResNet-18 :cite:`He.Zhang.Ren.ea.2016`. Karena gambar input yang digunakan cukup kecil, kami melakukan sedikit modifikasi pada arsitektur. Secara khusus, perbedaannya dari :numref:`sec_resnet` adalah kami menggunakan kernel konvolusi yang lebih kecil, stride yang lebih pendek, dan padding di awal jaringan.
Selain itu, kami juga menghapus layer max-pooling.


```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """Model ResNet-18 yang sedikit dimodifikasi"""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # Model ini menggunakan kernel konvolusi yang lebih kecil, stride yang lebih pendek, dan padding
    # serta menghapus layer max-pooling
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """Model ResNet-18 yang sedikit dimodifikasi"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # Model ini menggunakan kernel konvolusi yang lebih kecil, stride yang lebih pendek, dan padding
    # serta menghapus layer max-pooling
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Inisialisasi Jaringan

:begin_tab:`mxnet`
Fungsi `initialize` memungkinkan kita untuk menginisialisasi parameter pada perangkat pilihan kita.
Untuk ulasan tentang metode inisialisasi, lihat :numref:`sec_numerical_stability`. Yang sangat memudahkan adalah fungsi ini juga memungkinkan kita untuk menginisialisasi jaringan pada *beberapa* perangkat sekaligus. Mari kita coba bagaimana ini bekerja dalam praktik.
:end_tab:

:begin_tab:`pytorch`
Kita akan menginisialisasi jaringan di dalam loop pelatihan.
Untuk ulasan tentang metode inisialisasi, lihat :numref:`sec_numerical_stability`.
:end_tab:



```{.python .input}
#@tab mxnet
net = resnet18(10)
# Mendapatkan daftar GPU
devices = d2l.try_all_gpus()
# Menginisialisasi semua parameter jaringan
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Mendapatkan daftar GPU
devices = d2l.try_all_gpus()
# Kita akan menginisialisasi jaringan di dalam loop pelatihan
```

:begin_tab:`mxnet`
Menggunakan fungsi `split_and_load` yang diperkenalkan di :numref:`sec_multi_gpu`, kita dapat membagi minibatch data dan menyalin bagian-bagiannya ke dalam daftar perangkat yang disediakan oleh variabel `devices`.
Instance jaringan *secara otomatis* menggunakan GPU yang sesuai untuk menghitung nilai dari forward propagation. Di sini kita menghasilkan 4 observasi dan membaginya di antara GPU.
:end_tab:


```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Begitu data melewati jaringan, parameter yang sesuai diinisialisasi *pada perangkat tempat data melewati*. 
Ini berarti bahwa inisialisasi terjadi berdasarkan per perangkat. Karena kita memilih GPU 0 dan GPU 1 untuk inisialisasi, jaringan hanya diinisialisasi di sana, dan tidak di CPU. Bahkan, parameter tersebut tidak ada di CPU. Kita dapat memverifikasinya dengan mencetak parameter-parameter tersebut dan mengamati apakah ada kesalahan yang mungkin terjadi.
:end_tab:


```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Selanjutnya, mari kita ganti kode untuk [**mengevaluasi akurasi**] dengan satu yang bekerja (**secara paralel di beberapa perangkat**). Ini berfungsi sebagai pengganti fungsi `evaluate_accuracy_gpu` dari :numref:`sec_lenet`. Perbedaan utamanya adalah kita membagi satu minibatch sebelum memanggil jaringan. Sisanya pada dasarnya identik.
:end_tab:


```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Menghitung akurasi untuk model pada dataset menggunakan beberapa GPU."""
    # Mendapatkan daftar perangkat
    devices = list(net.collect_params().values())[0].list_ctx()
    # Jumlah prediksi yang benar, jumlah keseluruhan prediksi
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Menjalankan secara paralel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Pelatihan**]

Seperti sebelumnya, kode pelatihan harus melakukan beberapa fungsi dasar untuk paralelisme yang efisien:

* Parameter jaringan perlu diinisialisasi di semua perangkat.
* Saat iterasi atas minibatch dataset, minibatch harus dibagi di semua perangkat.
* Kita menghitung loss dan gradiennya secara paralel di seluruh perangkat.
* Gradien dikumpulkan dan parameter diperbarui sesuai.

Pada akhirnya, kita menghitung akurasi (juga secara paralel) untuk melaporkan kinerja akhir dari jaringan. Rutinitas pelatihan ini sangat mirip dengan implementasi di bab-bab sebelumnya, hanya saja kita perlu membagi dan menggabungkan data.


```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Mengatur Model pada Beberapa GPU
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Mari kita lihat bagaimana ini bekerja dalam praktik. Sebagai pemanasan, kita [**melatih jaringan pada satu GPU.**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Selanjutnya kita [**menggunakan 2 GPU untuk pelatihan**]. Dibandingkan dengan LeNet yang dievaluasi pada :numref:`sec_multi_gpu`, model untuk ResNet-18 jauh lebih kompleks. Di sinilah paralelisasi menunjukkan keunggulannya. Waktu untuk komputasi secara signifikan lebih besar dibandingkan dengan waktu untuk sinkronisasi parameter. Hal ini meningkatkan skalabilitas karena overhead untuk paralelisasi menjadi kurang relevan.


```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Ringkasan

:begin_tab:`mxnet`
* Gluon menyediakan primitif untuk inisialisasi model di beberapa perangkat dengan menyediakan daftar konteks.
:end_tab:
* Data secara otomatis dievaluasi pada perangkat di mana data tersebut ditemukan.
* Pastikan untuk menginisialisasi jaringan pada setiap perangkat sebelum mencoba mengakses parameter pada perangkat tersebut. Jika tidak, Anda akan menemui error.
* Algoritma optimasi secara otomatis mengagregasi di beberapa GPU.

## Latihan

:begin_tab:`mxnet`
1. Bagian ini menggunakan ResNet-18. Coba berbagai epoch, ukuran batch, dan tingkat pembelajaran. Gunakan lebih banyak GPU untuk komputasi. Apa yang terjadi jika Anda mencobanya dengan 16 GPU (misalnya, pada instance AWS p2.16xlarge)?
2. Kadang-kadang, perangkat yang berbeda memiliki daya komputasi yang berbeda. Kita bisa menggunakan GPU dan CPU pada waktu yang sama. Bagaimana kita harus membagi pekerjaan? Apakah usaha ini layak dilakukan? Mengapa? Mengapa tidak?
3. Apa yang terjadi jika kita menghilangkan `npx.waitall()`? Bagaimana Anda memodifikasi pelatihan sehingga Anda memiliki overlap hingga dua langkah untuk paralelisme?
:end_tab:

:begin_tab:`pytorch`
1. Bagian ini menggunakan ResNet-18. Coba berbagai epoch, ukuran batch, dan tingkat pembelajaran. Gunakan lebih banyak GPU untuk komputasi. Apa yang terjadi jika Anda mencobanya dengan 16 GPU (misalnya, pada instance AWS p2.16xlarge)?
2. Kadang-kadang, perangkat yang berbeda memiliki daya komputasi yang berbeda. Kita bisa menggunakan GPU dan CPU pada waktu yang sama. Bagaimana kita harus membagi pekerjaan? Apakah usaha ini layak dilakukan? Mengapa? Mengapa tidak?
:end_tab:

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1403)
:end_tab:
