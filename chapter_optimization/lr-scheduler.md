# Pengaturan Learning Rate
:label:`sec_scheduler`

Sejauh ini, kita terutama berfokus pada *algoritma* optimisasi untuk memperbarui vektor bobot, daripada pada *kecepatan* pembaruannya. Meskipun demikian, penyesuaian learning rate sering kali sama pentingnya dengan algoritma itu sendiri. Ada beberapa aspek yang perlu dipertimbangkan:

* Yang paling jelas adalah *besarnya* learning rate itu sendiri. Jika terlalu besar, optimisasi dapat menyimpang; jika terlalu kecil, pelatihan memerlukan waktu yang lama atau kita berakhir dengan hasil yang suboptimal. Kita telah melihat sebelumnya bahwa angka kondisi dari masalah tersebut penting (lihat misalnya :numref:`sec_momentum` untuk detail). Secara intuitif, ini adalah rasio perubahan dalam arah yang paling tidak sensitif vs. yang paling sensitif.
* Kedua, laju penurunan learning rate juga sama pentingnya. Jika learning rate tetap besar, kita mungkin hanya akan berputar-putar di sekitar minimum dan tidak mencapai titik optimal. :numref:`sec_minibatch_sgd` membahas hal ini secara rinci, dan kita menganalisis jaminan kinerjanya di :numref:`sec_sgd`. Singkatnya, kita ingin learning rate menurun, tetapi mungkin lebih lambat dari $\mathcal{O}(t^{-\frac{1}{2}})$ yang merupakan pilihan yang baik untuk masalah konveks.
* Aspek lain yang juga penting adalah *inisialisasi*. Ini berkaitan baik dengan bagaimana parameter ditetapkan awalnya (lihat kembali :numref:`sec_numerical_stability` untuk detail) dan juga bagaimana mereka berkembang pada awalnya. Ini sering disebut sebagai *warmup*, yaitu seberapa cepat kita mulai bergerak menuju solusi pada awalnya. Langkah besar di awal mungkin tidak menguntungkan, terutama karena parameter awal adalah acak. Arah pembaruan awal juga mungkin tidak terlalu berarti.
* Terakhir, ada beberapa varian optimisasi yang melakukan penyesuaian learning rate secara siklis. Ini berada di luar cakupan bab ini. Kami merekomendasikan pembaca untuk melihat lebih detail pada :citet:`Izmailov.Podoprikhin.Garipov.ea.2018`, misalnya, tentang cara mendapatkan solusi yang lebih baik dengan merata-rata di sepanjang *jalur* parameter.

Mengingat banyaknya detail yang dibutuhkan untuk mengelola learning rate, sebagian besar framework deep learning memiliki alat untuk menangani ini secara otomatis. Di bab ini, kita akan meninjau efek dari jadwal berbeda pada akurasi dan juga menunjukkan bagaimana ini dapat dikelola secara efisien melalui *learning rate scheduler*.

## Masalah Sederhana

Kita mulai dengan masalah sederhana yang cukup murah untuk dihitung dengan mudah, tetapi cukup non-trivial untuk mengilustrasikan beberapa aspek utama. Untuk itu, kita memilih versi LeNet yang sedikit dimodernisasi (`relu` alih-alih aktivasi `sigmoid`, MaxPooling daripada AveragePooling), yang diterapkan pada Fashion-MNIST. Selain itu, kita meng-hybrid-kan jaringan untuk kinerja. Karena sebagian besar kode adalah standar, kita hanya memperkenalkan dasar-dasarnya tanpa pembahasan rinci. Lihat :numref:`chap_cnn` untuk penyegaran jika diperlukan.



```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Kode ini hampir identik dengan `d2l.train_ch6` yang didefinisikan di
# bagian lenet pada bab jaringan saraf konvolusional
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Kode ini hampir identik dengan `d2l.train_ch6` yang didefinisikan di
# bagian lenet pada bab jaringan saraf konvolusional
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Kode ini hampir identik dengan `d2l.train_ch6` yang didefinisikan di
# bagian lenet pada bab jaringan saraf konvolusional
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Mari kita lihat apa yang terjadi jika kita menjalankan algoritma ini dengan pengaturan default, seperti learning rate sebesar $0.3$ dan pelatihan selama $30$ iterasi. Perhatikan bagaimana akurasi pelatihan terus meningkat sementara kemajuan dalam hal akurasi uji berhenti setelah titik tertentu. Kesenjangan antara kedua kurva menunjukkan overfitting.


```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Scheduler

Salah satu cara untuk menyesuaikan learning rate adalah dengan menetapkannya secara eksplisit pada setiap langkah. Hal ini dapat dilakukan dengan mudah menggunakan metode `set_learning_rate`. Kita dapat menyesuaikannya ke bawah setelah setiap epoch (atau bahkan setelah setiap minibatch), misalnya, secara dinamis sebagai respons terhadap perkembangan optimisasi.


```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate Sekarang {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate Sekarang {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate Sekarang ,', dummy_model.optimizer.lr.numpy())
```

Secara lebih umum, kita ingin mendefinisikan sebuah scheduler. Ketika dipanggil dengan jumlah pembaruan, scheduler ini akan mengembalikan nilai learning rate yang sesuai. Mari kita definisikan sebuah scheduler sederhana yang menetapkan learning rate sebagai $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.


```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Mari kita plot perilakunya dalam rentang beberapa nilai.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Sekarang mari kita lihat bagaimana ini bekerja saat melatih model pada Fashion-MNIST. Kita cukup memberikan scheduler sebagai argumen tambahan untuk algoritma pelatihan.


```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Hasil ini bekerja jauh lebih baik dibandingkan sebelumnya. Ada dua hal yang menonjol: kurva yang dihasilkan jauh lebih halus daripada sebelumnya. Kedua, terjadi lebih sedikit overfitting. Sayangnya, belum ada jawaban yang jelas mengenai mengapa strategi tertentu menghasilkan lebih sedikit overfitting secara *teori*. Ada argumen bahwa langkah yang lebih kecil akan menghasilkan parameter yang lebih dekat ke nol dan dengan demikian lebih sederhana. Namun, hal ini tidak sepenuhnya menjelaskan fenomena ini karena kita tidak benar-benar berhenti lebih awal, tetapi hanya mengurangi learning rate secara perlahan.

## Kebijakan

Meskipun kita tidak mungkin membahas semua variasi dari learning rate scheduler, kita mencoba memberikan gambaran singkat mengenai kebijakan yang populer di bawah ini. Pilihan umum adalah pengurangan secara polinomial dan jadwal konstan pada bagian tertentu. Selain itu, learning rate schedule berbasis kosinus terbukti bekerja dengan baik secara empiris pada beberapa masalah. Terakhir, dalam beberapa masalah ada manfaat dari melakukan warm up pada optimizer sebelum menggunakan learning rate yang besar.

### Scheduler Faktor

Salah satu alternatif dari penurunan polinomial adalah penurunan secara multiplikatif, yaitu $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ untuk $\alpha \in (0, 1)$. Untuk mencegah learning rate menurun di bawah batas bawah yang masuk akal, persamaan pembaruan sering kali dimodifikasi menjadi $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.


```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Hal ini juga dapat dicapai dengan menggunakan scheduler bawaan di MXNet melalui objek `lr_scheduler.FactorScheduler`. Objek ini memiliki beberapa parameter tambahan, seperti periode warmup, mode warmup (linear atau konstan), jumlah maksimum pembaruan yang diinginkan, dll.; Selanjutnya, kita akan menggunakan scheduler bawaan sesuai kebutuhan dan hanya menjelaskan fungsinya di sini. Seperti yang diilustrasikan, cukup mudah untuk membuat scheduler sendiri jika diperlukan.

### Multi Factor Scheduler

Strategi umum untuk melatih jaringan deep adalah dengan menjaga learning rate tetap konstan pada beberapa bagian dan menurunkannya dengan jumlah tertentu dari waktu ke waktu. Misalnya, diberikan set waktu untuk menurunkan rate, seperti $s = \{5, 10, 20\}$, maka kita menurunkan $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ setiap kali $t \in s$. Dengan asumsi bahwa nilainya dikurangi setengah pada setiap langkah, kita dapat mengimplementasikannya sebagai berikut.


```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Intuisi di balik jadwal learning rate yang konstan pada beberapa bagian ini adalah bahwa kita membiarkan optimisasi berlanjut hingga mencapai titik stasioner dalam hal distribusi vektor bobot. Kemudian (dan hanya setelah itu) kita menurunkan learning rate untuk mendapatkan proksi kualitas yang lebih tinggi menuju minimum lokal yang baik. Contoh di bawah ini menunjukkan bagaimana ini dapat menghasilkan solusi yang sedikit lebih baik setiap saat.


```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Cosine Scheduler

Sebuah heuristik yang cukup membingungkan diusulkan oleh :citet:`Loshchilov.Hutter.2016`. Pendekatan ini didasarkan pada pengamatan bahwa kita mungkin tidak ingin menurunkan learning rate terlalu drastis di awal, dan juga kita mungkin ingin "memperbaiki" solusi pada akhirnya menggunakan learning rate yang sangat kecil. Hal ini menghasilkan jadwal berbentuk kosinus dengan bentuk fungsional berikut untuk learning rate dalam rentang $t \in [0, T]$.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

Di sini $\eta_0$ adalah learning rate awal, dan $\eta_T$ adalah learning rate target pada waktu $T$. Selain itu, untuk $t > T$ kita hanya menetapkan nilai ke $\eta_T$ tanpa meningkatkannya lagi. Pada contoh berikut, kita menetapkan langkah pembaruan maksimum $T = 20$.


```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Dalam konteks computer vision, jadwal ini *dapat* menghasilkan hasil yang lebih baik. Namun, perlu dicatat bahwa perbaikan seperti itu tidak dijamin (seperti yang dapat dilihat di bawah ini).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Warmup

Dalam beberapa kasus, inisialisasi parameter saja tidak cukup untuk menjamin solusi yang baik. Ini terutama menjadi masalah untuk beberapa desain jaringan yang canggih yang dapat menyebabkan masalah optimisasi yang tidak stabil. Kita dapat mengatasi ini dengan memilih learning rate yang cukup kecil untuk mencegah divergensi di awal. Sayangnya, ini berarti kemajuan akan lambat. Sebaliknya, learning rate yang besar di awal dapat menyebabkan divergensi.

Salah satu solusi sederhana untuk dilema ini adalah dengan menggunakan periode warmup di mana learning rate *meningkat* hingga mencapai maksimum awalnya, dan kemudian menurunkan rate hingga akhir proses optimisasi. Untuk kesederhanaan, biasanya digunakan peningkatan linear untuk tujuan ini. Hal ini menghasilkan jadwal seperti yang ditunjukkan di bawah ini.


```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Perhatikan bahwa jaringan lebih baik mengalami konvergensi pada awalnya (terutama amati kinerjanya selama 5 epoch pertama).


```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Perlu dicatat bahwa warmup dapat diterapkan pada scheduler apa pun (bukan hanya cosine). Untuk pembahasan yang lebih rinci tentang learning rate schedule dan lebih banyak eksperimen, lihat juga :cite:`Gotmare.Keskar.Xiong.ea.2018`. Secara khusus, mereka menemukan bahwa fase warmup membatasi jumlah divergensi parameter dalam jaringan yang sangat dalam. Ini masuk akal secara intuitif karena kita memperkirakan akan terjadi divergensi yang signifikan akibat inisialisasi acak pada bagian-bagian jaringan yang membutuhkan waktu paling lama untuk membuat kemajuan di awal.

## Ringkasan

* Mengurangi learning rate selama pelatihan dapat meningkatkan akurasi dan (yang paling membingungkan) mengurangi overfitting pada model.
* Penurunan learning rate secara bertahap setiap kali kemajuan mulai datar terbukti efektif dalam praktik. Hal ini memastikan bahwa kita mencapai konvergensi secara efisien menuju solusi yang sesuai, dan hanya setelah itu mengurangi variansi parameter yang melekat dengan mengurangi learning rate.
* Cosine scheduler populer digunakan pada beberapa masalah computer vision. Lihat misalnya [GluonCV](http://gluon-cv.mxnet.io) untuk detail mengenai scheduler ini.
* Periode warmup sebelum optimisasi dapat mencegah divergensi.
* Optimisasi memiliki berbagai tujuan dalam deep learning. Selain meminimalkan fungsi objektif pelatihan, pilihan algoritma optimisasi dan pengaturan learning rate yang berbeda dapat menyebabkan tingkat generalisasi dan overfitting yang berbeda pada set pengujian (untuk jumlah error pelatihan yang sama).

## Latihan

1. Bereksperimenlah dengan perilaku optimisasi untuk learning rate tetap yang diberikan. Model terbaik apa yang bisa Anda peroleh dengan cara ini?
2. Bagaimana konvergensi berubah jika Anda mengubah eksponen penurunan learning rate? Gunakan `PolyScheduler` untuk kenyamanan dalam eksperimen.
3. Terapkan cosine scheduler untuk masalah computer vision yang besar, misalnya, melatih ImageNet. Bagaimana dampaknya terhadap kinerja dibandingkan dengan scheduler lainnya?
4. Berapa lama warmup seharusnya berlangsung?
5. Bisakah Anda menghubungkan optimisasi dan sampling? Mulailah dengan menggunakan hasil dari :citet:`Welling.Teh.2011` tentang Stochastic Gradient Langevin Dynamics.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/1081)
:end_tab:

