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

A faster strategy is to perform column-wise assignment.

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
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

Last, the most effective manner is to perform the entire operation in one block. 
Note that multiplying any two matrices $\mathbf{B} \in \mathbb{R}^{m \times n}$ and $\mathbf{C} \in \mathbb{R}^{n \times p}$ takes approximately $2mnp$ floating point operations,
when scalar multiplication and addition are counted as separate operations (fused in practice).
Thus, multiplying two $256 \times 256$ matrices
takes $0.03$ billion floating point operations.
Let's see what the respective speed of the operations is.

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
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
# Compute A = BC in one go
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

In the past we took it for granted that we would read *minibatches* of data rather than single observations to update parameters. We now give a brief justification for it. Processing single observations requires us to perform many single matrix-vector (or even vector-vector) multiplications, which is quite expensive and which incurs a significant overhead on behalf of the underlying deep learning framework. This applies both to evaluating a network when applied to data (often referred to as inference) and when computing gradients to update parameters. That is, this applies whenever we perform $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ where

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

We can increase the *computational* efficiency of this operation by applying it to a minibatch of observations at a time. That is, we replace the gradient $\mathbf{g}_t$ over a single observation by one over a small batch

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Let's see what this does to the statistical properties of $\mathbf{g}_t$: since both $\mathbf{x}_t$ and also all elements of the minibatch $\mathcal{B}_t$ are drawn uniformly at random from the training set, the expectation of the gradient remains unchanged. The variance, on the other hand, is reduced significantly. Since the minibatch gradient is composed of $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ independent gradients which are being averaged, its standard deviation is reduced by a factor of $b^{-\frac{1}{2}}$. This, by itself, is a good thing, since it means that the updates are more reliably aligned with the full gradient.

Naively this would indicate that choosing a large minibatch $\mathcal{B}_t$ would be universally desirable. Alas, after some point, the additional reduction in standard deviation is minimal when compared to the linear increase in computational cost. In practice we pick a minibatch that is large enough to offer good computational efficiency while still fitting into the memory of a GPU. To illustrate the savings let's have a look at some code. In it we perform the same matrix-matrix multiplication, but this time broken up into "minibatches" of 64 columns at a time.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

As we can see, the computation on the minibatch is essentially as efficient as on the full matrix. A word of caution is in order. In :numref:`sec_batch_norm` we used a type of regularization that was heavily dependent on the amount of variance in a minibatch. As we increase the latter, the variance decreases and with it the benefit of the noise-injection due to batch normalization. See e.g., :citet:`Ioffe.2017` for details on how to rescale and compute the appropriate terms.

## Reading the Dataset

Let's have a look at how minibatches are efficiently generated from data. In the following we use a dataset developed by NASA to test the wing [noise from different aircraft](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) to compare these optimization algorithms. For convenience we only use the first $1,500$ examples. The data is whitened for preprocessing, i.e., we remove the mean and rescale the variance to $1$ per coordinate.

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

## Implementation from Scratch

Recall the minibatch stochastic gradient descent implementation from :numref:`sec_linear_scratch`. In the following we provide a slightly more general implementation. For convenience it has the same call signature as the other optimization algorithms introduced later in this chapter. Specifically, we add the status
input `states` and place the hyperparameter in dictionary `hyperparams`. In
addition, we will average the loss of each minibatch example in the training
function, so the gradient in the optimization algorithm does not need to be
divided by the batch size.

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

Next, we implement a generic training function to facilitate the use of the other optimization algorithms introduced later in this chapter. It initializes a linear regression model and can be used to train the model with minibatch stochastic gradient descent and other algorithms introduced subsequently.

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

Let's see how optimization proceeds for batch gradient descent. This can be achieved by setting the minibatch size to 1500 (i.e., to the total number of examples). As a result the model parameters are updated only once per epoch. There is little progress. In fact, after 6 steps progress stalls.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

When the batch size equals 1, we use stochastic gradient descent for optimization. For simplicity of implementation we picked a constant (albeit small) learning rate. In stochastic gradient descent, the model parameters are updated whenever an example is processed. In our case this amounts to 1500 updates per epoch. As we can see, the decline in the value of the objective function slows down after one epoch. Although both the procedures processed 1500 examples within one epoch, stochastic gradient descent consumes more time than gradient descent in our experiment. This is because stochastic gradient descent updated the parameters more frequently and since it is less efficient to process single observations one at a time.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Finally, when the batch size equals 100, we use minibatch stochastic gradient descent for optimization. The time required per epoch is shorter than the time needed for stochastic gradient descent and the time for batch gradient descent.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Reducing the batch size to 10, the time for each epoch increases because the workload for each batch is less efficient to execute.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Now we can compare the time vs. loss for the previous four experiments. As can be seen, although stochastic gradient descent converges faster than GD in terms of number of examples processed, it uses more time to reach the same loss than GD because computing the gradient example by example is not as efficient. Minibatch stochastic gradient descent is able to trade-off convergence speed and computation efficiency. A minibatch size of 10 is more efficient than stochastic gradient descent; a minibatch size of 100 even outperforms GD in terms of runtime.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Concise Implementation

In Gluon, we can use the `Trainer` class to call optimization algorithms. This is used to implement a generic training function. We will use this throughout the current chapter.

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
                # `MSELoss` computes squared error without the 1/2 factor
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
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Using Gluon to repeat the last experiment shows identical behavior.

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

## Summary

* Vectorization makes code more efficient due to reduced overhead arising from the deep learning framework and due to better memory locality and caching on CPUs and GPUs.
* There is a trade-off between statistical efficiency arising from stochastic gradient descent and computational efficiency arising from processing large batches of data at a time.
* Minibatch stochastic gradient descent offers the best of both worlds: computational and statistical efficiency.
* In minibatch stochastic gradient descent we process batches of data obtained by a random permutation of the training data (i.e., each observation is processed only once per epoch, albeit in random order).
* It is advisable to decay the learning rates during training.
* In general, minibatch stochastic gradient descent is faster than stochastic gradient descent and gradient descent for convergence to a smaller risk, when measured in terms of clock time.

## Exercises

1. Modify the batch size and learning rate and observe the rate of decline for the value of the objective function and the time consumed in each epoch.
1. Read the MXNet documentation and use the `Trainer` class `set_learning_rate` function to reduce the learning rate of the minibatch stochastic gradient descent to 1/10 of its previous value after each epoch.
1. Compare minibatch stochastic gradient descent with a variant that actually *samples with replacement* from the training set. What happens?
1. An evil genie replicates your dataset without telling you (i.e., each observation occurs twice and your dataset grows to twice its original size, but nobody told you). How does the behavior of stochastic gradient descent, minibatch stochastic gradient descent and that of gradient descent change?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
