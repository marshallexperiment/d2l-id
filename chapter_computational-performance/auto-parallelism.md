# Paralelisme Otomatis
:label:`sec_auto_para`

Framework deep learning (misalnya, MXNet dan PyTorch) secara otomatis membangun grafik komputasi di backend. Dengan menggunakan grafik komputasi, sistem dapat mengetahui semua ketergantungan, dan dapat secara selektif mengeksekusi beberapa tugas yang tidak saling bergantung secara paralel untuk meningkatkan kecepatan. Misalnya, :numref:`fig_asyncgraph` pada :numref:`sec_async` menginisialisasi dua variabel secara independen. Akibatnya, sistem dapat memilih untuk mengeksekusinya secara paralel.

Biasanya, satu operator akan menggunakan semua sumber daya komputasi pada semua CPU atau pada satu GPU. Sebagai contoh, operator `dot` akan menggunakan semua core (dan thread) pada semua CPU, bahkan jika ada beberapa prosesor CPU pada satu mesin. Hal yang sama berlaku untuk satu GPU. Oleh karena itu, paralelisasi tidak begitu berguna untuk komputer dengan satu perangkat. Namun, pada perangkat dengan banyak GPU, paralelisasi menjadi lebih penting. Meskipun paralelisasi umumnya paling relevan di antara banyak GPU, menambahkan CPU lokal juga dapat meningkatkan performa sedikit. Misalnya, lihat :citet:`Hadjis.Zhang.Mitliagkas.ea.2016` yang berfokus pada pelatihan model visi komputer dengan menggabungkan GPU dan CPU. Dengan kemudahan framework yang melakukan paralelisasi secara otomatis, kita dapat mencapai tujuan yang sama hanya dengan beberapa baris kode Python. Secara lebih luas, diskusi kita tentang komputasi paralel otomatis berfokus pada komputasi paralel menggunakan CPU dan GPU, serta paralelisasi antara komputasi dan komunikasi.

Perhatikan bahwa kita membutuhkan setidaknya dua GPU untuk menjalankan eksperimen dalam bagian ini.


```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## Komputasi Paralel pada GPU

Mari kita mulai dengan mendefinisikan beban kerja referensi untuk diuji: fungsi `run` di bawah ini melakukan 10 perkalian matriks-matriks pada perangkat pilihan kita menggunakan data yang dialokasikan ke dalam dua variabel: `x_gpu1` dan `x_gpu2`.


```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
Sekarang kita menerapkan fungsi tersebut pada data. Untuk memastikan bahwa caching tidak mempengaruhi hasil, kita memanaskan perangkat terlebih dahulu dengan melakukan satu kali operasi pada masing-masing perangkat sebelum melakukan pengukuran.
:end_tab:

:begin_tab:`pytorch`
Sekarang kita menerapkan fungsi tersebut pada data. Untuk memastikan bahwa caching tidak mempengaruhi hasil, kita memanaskan perangkat terlebih dahulu dengan melakukan satu kali operasi pada masing-masing perangkat sebelum melakukan pengukuran. `torch.cuda.synchronize()` menunggu semua kernel di semua stream pada perangkat CUDA selesai. Fungsi ini membutuhkan argumen `device`, yaitu perangkat yang perlu kita sinkronkan. Jika argumen perangkat adalah `None` (default), maka ia akan menggunakan perangkat saat ini yang diberikan oleh `current_device()`.
:end_tab:



```{.python .input}
#@tab mxnet
run(x_gpu1)  # Warm-up kedua devices
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up semua devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Jika kita menghapus pernyataan `waitall` di antara kedua tugas, sistem bebas untuk melakukan paralelisasi komputasi pada kedua perangkat secara otomatis.
:end_tab:

:begin_tab:`pytorch`
Jika kita menghapus pernyataan `synchronize` di antara kedua tugas, sistem bebas untuk melakukan paralelisasi komputasi pada kedua perangkat secara otomatis.
:end_tab:



```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

Pada kasus di atas, waktu eksekusi total lebih sedikit dibandingkan dengan jumlah waktu eksekusi dari setiap bagian, karena framework deep learning secara otomatis menjadwalkan komputasi pada kedua perangkat GPU tanpa memerlukan kode yang rumit dari pengguna.

## Komputasi Paralel dan Komunikasi

Dalam banyak kasus, kita perlu memindahkan data di antara perangkat yang berbeda, misalnya antara CPU dan GPU, atau antara GPU yang berbeda.
Sebagai contoh, hal ini terjadi ketika kita ingin melakukan optimasi terdistribusi di mana kita perlu mengagregasi gradien dari beberapa kartu akselerator. Mari kita simulasikan ini dengan melakukan komputasi pada GPU dan kemudian menyalin hasilnya kembali ke CPU.



```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Ini agak tidak efisien. Perhatikan bahwa kita sebenarnya sudah bisa mulai menyalin sebagian dari `y` ke CPU sementara sisanya dari daftar masih dihitung. Situasi ini terjadi, misalnya, ketika kita menghitung gradien pada satu minibatch. Gradien dari beberapa parameter akan tersedia lebih awal dibandingkan yang lain. Oleh karena itu, kita diuntungkan jika mulai menggunakan bandwidth bus PCI-Express sementara GPU masih berjalan. Menghapus `waitall` antara kedua bagian memungkinkan kita mensimulasikan skenario ini.
:end_tab:

:begin_tab:`pytorch`
Ini agak tidak efisien. Perhatikan bahwa kita sebenarnya sudah bisa mulai menyalin sebagian dari `y` ke CPU sementara sisanya dari daftar masih dihitung. Situasi ini terjadi, misalnya, ketika kita menghitung gradien (backprop) pada satu minibatch. Gradien dari beberapa parameter akan tersedia lebih awal dibandingkan yang lain. Oleh karena itu, kita diuntungkan jika mulai menggunakan bandwidth bus PCI-Express sementara GPU masih berjalan. Dalam PyTorch, beberapa fungsi seperti `to()` dan `copy_()` menerima argumen `non_blocking` secara eksplisit, yang memungkinkan pemanggil untuk melewati sinkronisasi saat tidak diperlukan. Mengatur `non_blocking=True` memungkinkan kita mensimulasikan skenario ini.
:end_tab:




```{.python .input}
#@tab mxnet
with d2l.Benchmark('Jalankan pada GPU1 dan salin ke CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Jalankan pada GPU1 dan salin ke CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Waktu total yang dibutuhkan untuk kedua operasi tersebut (seperti yang diharapkan) lebih sedikit dibandingkan dengan jumlah dari masing-masing bagiannya.
Perlu dicatat bahwa tugas ini berbeda dari komputasi paralel karena menggunakan sumber daya yang berbeda: bus antara CPU dan GPU. Faktanya, kita bisa melakukan komputasi pada kedua perangkat dan komunikasi, semuanya secara bersamaan. Seperti yang disebutkan di atas, terdapat ketergantungan antara komputasi dan komunikasi: `y[i]` harus dihitung sebelum dapat disalin ke CPU. Untungnya, sistem dapat menyalin `y[i-1]` sambil menghitung `y[i]` untuk mengurangi waktu eksekusi total.

Kita menyimpulkan dengan ilustrasi dari grafik komputasi dan ketergantungannya untuk MLP dua lapisan sederhana saat dilatih pada CPU dan dua GPU, seperti yang digambarkan pada :numref:`fig_twogpu`. Akan sangat merepotkan untuk menjadwalkan program paralel yang dihasilkan dari ini secara manual. Di sinilah keuntungan memiliki backend komputasi berbasis grafik untuk optimasi.

![Grafik komputasi dan ketergantungannya dari MLP dua lapisan pada CPU dan dua GPU.](../img/twogpu.svg)
:label:`fig_twogpu`



## Ringkasan

* Sistem modern memiliki berbagai perangkat, seperti beberapa GPU dan CPU. Perangkat-perangkat ini dapat digunakan secara paralel, asinkron.
* Sistem modern juga memiliki berbagai sumber daya untuk komunikasi, seperti PCI Express, penyimpanan (biasanya solid-state drive atau melalui jaringan), dan bandwidth jaringan. Sumber daya ini dapat digunakan secara paralel untuk mencapai efisiensi puncak.
* Backend dapat meningkatkan kinerja melalui komputasi dan komunikasi paralel otomatis.

## Latihan

1. Delapan operasi dilakukan dalam fungsi `run` yang didefinisikan di bagian ini. Tidak ada ketergantungan di antara mereka. Rancang sebuah eksperimen untuk melihat apakah framework deep learning secara otomatis akan mengeksekusinya secara paralel.
2. Ketika beban kerja dari sebuah operator cukup kecil, paralelisasi dapat membantu bahkan pada satu CPU atau GPU. Rancang sebuah eksperimen untuk memverifikasi ini.
3. Rancang sebuah eksperimen yang menggunakan komputasi paralel pada CPU, GPU, dan komunikasi antara kedua perangkat.
4. Gunakan debugger seperti [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) dari NVIDIA untuk memverifikasi bahwa kode Anda efisien.
5. Merancang tugas komputasi yang mencakup ketergantungan data yang lebih kompleks, dan menjalankan eksperimen untuk melihat apakah Anda dapat memperoleh hasil yang benar sambil meningkatkan kinerja.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/1681)
:end_tab:

