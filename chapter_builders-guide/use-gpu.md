```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU
:label:`sec_use_gpu`

Pada :numref:`tab_intro_decade`, kami mengilustrasikan pertumbuhan pesat
dalam komputasi selama dua dekade terakhir.
Secara singkat, performa GPU telah meningkat
dengan faktor 1000 setiap dekade sejak tahun 2000.
Ini menawarkan peluang besar, tetapi juga menunjukkan
bahwa ada permintaan yang signifikan untuk performa semacam itu.

Pada bagian ini, kita mulai membahas bagaimana memanfaatkan
performa komputasi ini untuk riset Anda.
Pertama dengan menggunakan satu GPU dan di lain waktu,
bagaimana menggunakan beberapa GPU dan beberapa server (dengan beberapa GPU).

Secara spesifik, kita akan membahas cara
menggunakan satu GPU NVIDIA untuk perhitungan.
Pertama, pastikan Anda memiliki setidaknya satu GPU NVIDIA yang terpasang.
Lalu, unduh [driver NVIDIA dan CUDA](https://developer.nvidia.com/cuda-downloads)
dan ikuti petunjuknya untuk mengatur path yang sesuai.
Setelah persiapan ini selesai,
perintah `nvidia-smi` dapat digunakan
untuk (**melihat informasi kartu grafis**).

:begin_tab:`mxnet`
Anda mungkin menyadari bahwa tensor MXNet
terlihat hampir identik dengan `ndarray` pada NumPy.
Namun, ada beberapa perbedaan krusial.
Salah satu fitur utama yang membedakan MXNet
dari NumPy adalah dukungannya untuk perangkat keras yang beragam.

Di MXNet, setiap array memiliki konteks.
Sejauh ini, secara default, semua variabel
dan perhitungan terkait
telah ditetapkan pada CPU.
Biasanya, konteks lain mungkin adalah berbagai GPU.
Situasinya bisa menjadi lebih rumit ketika
kita mendistribusikan pekerjaan di beberapa server.
Dengan menetapkan array ke konteks secara cerdas,
kita dapat meminimalkan waktu yang dihabiskan
untuk mentransfer data antar perangkat.
Sebagai contoh, saat melatih jaringan saraf di server dengan GPU,
biasanya kita menginginkan parameter model berada di GPU.

Selanjutnya, kita perlu mengonfirmasi bahwa
versi MXNet untuk GPU sudah terpasang.
Jika versi CPU MXNet sudah terpasang,
kita perlu menghapusnya terlebih dahulu.
Sebagai contoh, gunakan perintah `pip uninstall mxnet`,
lalu instal versi MXNet yang sesuai
dengan versi CUDA Anda.
Misalnya, jika Anda memiliki CUDA 10.0 terpasang,
Anda dapat memasang versi MXNet
yang mendukung CUDA 10.0 melalui `pip install mxnet-cu100`.
:end_tab:

:begin_tab:`pytorch`
Dalam PyTorch, setiap array memiliki perangkat; kita sering menyebutnya sebagai *konteks*.
Sejauh ini, secara default, semua variabel
dan perhitungan terkait
telah ditetapkan pada CPU.
Biasanya, konteks lain mungkin adalah berbagai GPU.
Situasinya bisa menjadi lebih rumit ketika
kita mendistribusikan pekerjaan di beberapa server.
Dengan menetapkan array ke konteks secara cerdas,
kita dapat meminimalkan waktu yang dihabiskan
untuk mentransfer data antar perangkat.
Sebagai contoh, saat melatih jaringan saraf di server dengan GPU,
biasanya kita menginginkan parameter model berada di GPU.
:end_tab:

Untuk menjalankan program dalam bagian ini,
Anda memerlukan setidaknya dua GPU.
Perlu dicatat bahwa ini mungkin berlebihan untuk sebagian besar komputer desktop,
tetapi mudah didapatkan di cloud, misalnya,
dengan menggunakan instans multi-GPU di AWS EC2.
Hampir semua bagian lainnya *tidak* memerlukan beberapa GPU, tetapi di sini kami ingin mengilustrasikan aliran data antara berbagai perangkat.


```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

## [**Perangkat Komputasi**]

Kita dapat menentukan perangkat, seperti CPU dan GPU,
untuk penyimpanan dan perhitungan.
Secara default, tensor dibuat di memori utama
dan kemudian CPU digunakan untuk perhitungan.

:begin_tab:`mxnet`
Dalam MXNet, CPU dan GPU dapat ditunjukkan dengan `cpu()` dan `gpu()`.
Perlu dicatat bahwa `cpu()`
(atau angka apa pun di dalam tanda kurung)
berarti semua CPU fisik dan memori.
Ini berarti bahwa perhitungan MXNet
akan mencoba menggunakan semua inti CPU.
Namun, `gpu()` hanya mewakili satu kartu
dan memori yang sesuai.
Jika ada beberapa GPU, kita menggunakan `gpu(i)`
untuk mewakili GPU ke-$i$ ($i$ dimulai dari 0).
Selain itu, `gpu(0)` dan `gpu()` adalah ekuivalen.
:end_tab:

:begin_tab:`pytorch`
Dalam PyTorch, CPU dan GPU dapat ditunjukkan dengan `torch.device('cpu')` dan `torch.device('cuda')`.
Perlu dicatat bahwa perangkat `cpu`
berarti semua CPU fisik dan memori.
Ini berarti bahwa perhitungan PyTorch
akan mencoba menggunakan semua inti CPU.
Namun, perangkat `gpu` hanya mewakili satu kartu
dan memori yang sesuai.
Jika ada beberapa GPU, kita menggunakan `torch.device(f'cuda:{i}')`
untuk mewakili GPU ke-$i$ ($i$ dimulai dari 0).
Selain itu, `gpu:0` dan `gpu` adalah ekuivalen.
:end_tab:

```{.python .input}
%%tab pytorch
def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def cpu():  #@save
    """Get the CPU device."""
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')
    if tab.selected('jax'):
        return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    """Get a GPU device."""
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')
    if tab.selected('jax'):
        return jax.devices('gpu')[i]

cpu(), gpu(), gpu(1)
```

Kita dapat (**memeriksa jumlah GPU yang tersedia.**)

```{.python .input}
%%tab pytorch
def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

num_gpus()
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def num_gpus():  #@save
    """Get the number of available GPUs."""
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))
    if tab.selected('jax'):
        try:
            return jax.device_count('gpu')
        except:
            return 0  # No GPU backend found

num_gpus()
```

Sekarang kita [**mendefinisikan dua fungsi yang memungkinkan kita
untuk menjalankan kode meskipun GPU yang diminta tidak tersedia.**]


```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensor dan GPU

:begin_tab:`pytorch`
Secara default, tensor dibuat di CPU.
Kita dapat [**memeriksa perangkat tempat tensor berada.**]
:end_tab:

:begin_tab:`mxnet`
Secara default, tensor dibuat di CPU.
Kita dapat [**memeriksa perangkat tempat tensor berada.**]
:end_tab:

:begin_tab:`tensorflow, jax`
Secara default, tensor dibuat di GPU/TPU jika tersedia,
dan jika tidak tersedia maka akan menggunakan CPU.
Kita dapat [**memeriksa perangkat tempat tensor berada.**]
:end_tab:

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
%%tab jax
x = jnp.array([1, 2, 3])
x.device()
```

Penting untuk dicatat bahwa setiap kali kita ingin
melakukan operasi pada beberapa term,
mereka harus berada di perangkat yang sama.
Misalnya, jika kita menjumlahkan dua tensor,
kita perlu memastikan bahwa kedua argumen
berada di perangkat yang sama—jika tidak, framework
tidak akan tahu di mana harus menyimpan hasilnya
atau bahkan bagaimana memutuskan tempat melakukan perhitungan.

### Penyimpanan di GPU

Ada beberapa cara untuk [**menyimpan tensor di GPU.**]
Sebagai contoh, kita dapat menentukan perangkat penyimpanan saat membuat tensor.
Selanjutnya, kita membuat variabel tensor `X` pada `gpu` pertama.
Tensor yang dibuat pada GPU hanya mengonsumsi memori GPU tersebut.
Kita dapat menggunakan perintah `nvidia-smi` untuk melihat penggunaan memori GPU.
Secara umum, kita perlu memastikan bahwa kita tidak membuat data yang melebihi batas memori GPU.


```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
%%tab jax
# By default JAX puts arrays to GPUs or TPUs if available
X = jax.device_put(jnp.ones((2, 3)), try_gpu())
X
```

Dengan asumsi Anda memiliki setidaknya dua GPU, kode berikut akan (**membuat tensor acak, `Y`, pada GPU kedua.**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
%%tab jax
Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                   try_gpu(1))
Y
```

### Penyalinan

[**Jika kita ingin menghitung `X + Y`,
kita perlu memutuskan di mana operasi ini akan dilakukan.**]
Sebagai contoh, seperti yang ditunjukkan pada :numref:`fig_copyto`,
kita dapat mentransfer `X` ke GPU kedua
dan melakukan operasi di sana.
*Jangan* langsung menambahkan `X` dan `Y`,
karena ini akan menghasilkan kesalahan.
Mesin runtime tidak akan tahu apa yang harus dilakukan:
ia tidak dapat menemukan data di perangkat yang sama dan gagal.
Karena `Y` berada di GPU kedua,
kita perlu memindahkan `X` ke sana sebelum kita dapat menambahkan keduanya.

![Salin data untuk melakukan operasi pada perangkat yang sama.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

```{.python .input}
%%tab jax
Z = jax.device_put(X, try_gpu(1))
print(X)
print(Z)
```

Sekarang [**data (baik `Z` maupun `Y`) berada di GPU yang sama, kita dapat menjumlahkannya.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
Bayangkan bahwa variabel `Z` Anda sudah berada di GPU kedua.
Apa yang terjadi jika kita masih memanggil `Z.copyto(gpu(1))`?
Ini akan membuat salinan dan mengalokasikan memori baru,
meskipun variabel tersebut sudah berada di perangkat yang diinginkan.
Terkadang, tergantung pada lingkungan tempat kode kita dijalankan,
dua variabel mungkin sudah berada di perangkat yang sama.
Jadi, kita hanya ingin membuat salinan jika variabel-variabel tersebut
saat ini berada di perangkat yang berbeda.
Dalam kasus ini, kita dapat memanggil `as_in_ctx`.
Jika variabel sudah berada di perangkat yang ditentukan,
maka ini adalah operasi yang tidak dilakukan.
Kecuali Anda secara khusus ingin membuat salinan,
`as_in_ctx` adalah metode yang dipilih.
:end_tab:

:begin_tab:`pytorch`
Namun, bagaimana jika variabel `Z` Anda sudah berada di GPU kedua?
Apa yang terjadi jika kita tetap memanggil `Z.cuda(1)`?
Ini akan mengembalikan `Z` alih-alih membuat salinan dan mengalokasikan memori baru.
:end_tab:

:begin_tab:`tensorflow`
Bayangkan bahwa variabel `Z` Anda sudah berada di GPU kedua.
Apa yang terjadi jika kita tetap memanggil `Z2 = Z` dalam ruang lingkup perangkat yang sama?
Ini akan mengembalikan `Z` alih-alih membuat salinan dan mengalokasikan memori baru.
:end_tab:

:begin_tab:`jax`
Bayangkan bahwa variabel `Z` Anda sudah berada di GPU kedua.
Apa yang terjadi jika kita tetap memanggil `Z2 = Z` dalam ruang lingkup perangkat yang sama?
Ini akan mengembalikan `Z` alih-alih membuat salinan dan mengalokasikan memori baru.
:end_tab:


```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

```{.python .input}
%%tab jax
Z2 = jax.device_put(Z, try_gpu(1))
Z2 is Z
```

### Catatan Tambahan

Orang menggunakan GPU untuk melakukan machine learning
karena mereka mengharapkan GPU bekerja dengan cepat.
Namun, mentransfer variabel antar perangkat sangat lambat: jauh lebih lambat daripada perhitungan.
Jadi, kami ingin Anda benar-benar yakin
bahwa Anda ingin melakukan sesuatu yang lambat sebelum kami mengizinkannya.
Jika framework deep learning langsung melakukan penyalinan secara otomatis
tanpa memberi peringatan, maka Anda mungkin tidak menyadari
bahwa Anda telah menulis kode yang lambat.

Transfer data tidak hanya lambat, tetapi juga membuat paralelisasi menjadi jauh lebih sulit,
karena kita harus menunggu data untuk dikirim (atau lebih tepatnya diterima)
sebelum kita dapat melanjutkan ke operasi berikutnya.
Inilah mengapa operasi penyalinan harus dilakukan dengan sangat hati-hati.
Sebagai aturan praktis, banyak operasi kecil
jauh lebih buruk daripada satu operasi besar.
Selain itu, beberapa operasi sekaligus
jauh lebih baik daripada banyak operasi tunggal yang tersebar dalam kode
kecuali jika Anda benar-benar tahu apa yang Anda lakukan.
Hal ini terjadi karena operasi seperti itu bisa tertunda jika satu perangkat
harus menunggu perangkat lainnya sebelum dapat melakukan operasi lain.
Ini mirip dengan memesan kopi dalam antrean
daripada memesan sebelumnya melalui telepon
dan mendapati bahwa kopi sudah siap saat Anda tiba.

Terakhir, saat kita mencetak tensor atau mengonversinya ke format NumPy,
jika data tidak ada di memori utama,
framework akan menyalinnya ke memori utama terlebih dahulu,
yang menghasilkan overhead transmisi tambahan.
Bahkan lebih buruk lagi, data ini kini tunduk pada global interpreter lock
yang membuat semua hal lain harus menunggu Python selesai.

## [**Jaringan Neural dan GPU**]

Demikian juga, model jaringan neural dapat menentukan perangkat.
Kode berikut menempatkan parameter model pada GPU.


```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(1)])

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (10,))  # Dummy input
params = net.init(key2, x)  # Initialization call
```

Kita akan melihat lebih banyak contoh
cara menjalankan model di GPU pada bab-bab berikut,
karena modelnya akan menjadi lebih intensif secara komputasi.

Sebagai contoh, ketika input berupa tensor di GPU, model akan menghitung hasilnya di GPU yang sama.


```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

Mari kita (**konfirmasi bahwa parameter model disimpan di GPU yang sama.**)

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
%%tab jax
print(jax.tree_util.tree_map(lambda x: x.device(), params))
```

Buat agar pelatih (trainer) mendukung GPU.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch
```

Singkatnya, selama semua data dan parameter berada di perangkat yang sama, kita dapat melatih model secara efisien. Pada bab-bab berikutnya, kita akan melihat beberapa contoh seperti ini.

## Ringkasan

- Kita dapat menentukan perangkat untuk penyimpanan dan perhitungan, seperti CPU atau GPU.
  Secara default, data dibuat di memori utama
  dan menggunakan CPU untuk perhitungan.
- Framework deep learning memerlukan semua data input untuk perhitungan
  berada di perangkat yang sama,
  baik itu CPU atau GPU yang sama.
- Anda bisa kehilangan performa yang signifikan dengan memindahkan data tanpa perencanaan yang matang.
  Kesalahan yang umum terjadi adalah sebagai berikut: menghitung loss
  untuk setiap minibatch pada GPU dan melaporkannya kembali
  kepada pengguna di command line (atau mencatatnya dalam `ndarray` NumPy)
  akan memicu global interpreter lock yang menghentikan semua GPU.
  Lebih baik menyediakan memori
  untuk pencatatan di dalam GPU dan hanya memindahkan log yang lebih besar.

## Latihan

1. Coba tugas komputasi yang lebih besar, seperti perkalian matriks besar,
   dan lihat perbedaan kecepatan antara CPU dan GPU.
   Bagaimana dengan tugas dengan jumlah perhitungan yang kecil?
2. Bagaimana cara membaca dan menulis parameter model di GPU?
3. Ukur waktu yang diperlukan untuk menghitung 1000
   perkalian matriks--matriks berukuran $100 \times 100$
   dan log norm Frobenius dari matriks keluaran satu per satu. Bandingkan dengan mencatat log di GPU dan hanya mentransfer hasil akhirnya.
4. Ukur berapa waktu yang diperlukan untuk melakukan dua perkalian matriks--matriks
   pada dua GPU secara bersamaan. Bandingkan dengan menghitungnya secara berurutan
   pada satu GPU. Petunjuk: Anda seharusnya melihat skala yang hampir linear.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/270)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17995)
:end_tab:
