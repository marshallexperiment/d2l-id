# Komputasi Asinkron
:label:`sec_async`

Komputer modern saat ini adalah sistem yang sangat paralel, terdiri dari banyak inti CPU (sering kali memiliki beberapa thread per inti), beberapa elemen pemrosesan per GPU, dan sering kali beberapa GPU per perangkat. Singkatnya, kita dapat memproses banyak hal yang berbeda pada saat yang sama, sering kali pada perangkat yang berbeda. Sayangnya, Python bukanlah cara yang baik untuk menulis kode paralel dan asinkron, setidaknya tidak tanpa bantuan tambahan. Bagaimanapun, Python adalah single-threaded dan ini tidak mungkin berubah di masa depan. Framework deep learning seperti MXNet dan TensorFlow mengadopsi model *pemrograman asinkron* untuk meningkatkan kinerja, sementara PyTorch menggunakan scheduler Python sendiri yang mengarah pada kompromi kinerja yang berbeda. 

Untuk PyTorch, secara default, operasi GPU bersifat asinkron. Ketika Anda memanggil fungsi yang menggunakan GPU, operasi tersebut dimasukkan ke dalam antrian pada perangkat tertentu, tetapi tidak harus dieksekusi secara langsung. Hal ini memungkinkan kita untuk mengeksekusi lebih banyak komputasi secara paralel, termasuk operasi pada CPU atau GPU lainnya.

Oleh karena itu, memahami cara kerja pemrograman asinkron membantu kita mengembangkan program yang lebih efisien, dengan secara proaktif mengurangi kebutuhan komputasi dan ketergantungan bersama. Ini memungkinkan kita untuk mengurangi overhead memori dan meningkatkan pemanfaatan prosesor.


```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## Asinkronitas melalui Backend

:begin_tab:`mxnet`
Sebagai pemanasan, pertimbangkan masalah mainan berikut: kita ingin menghasilkan sebuah matriks acak dan mengalikannya. Mari kita lakukan itu baik di NumPy maupun di `mxnet.np` untuk melihat perbedaannya.
:end_tab:

:begin_tab:`pytorch`
Sebagai pemanasan, pertimbangkan masalah mainan berikut: kita ingin menghasilkan sebuah matriks acak dan mengalikannya. Mari kita lakukan itu baik di NumPy maupun di PyTorch tensor untuk melihat perbedaannya.
Perhatikan bahwa PyTorch `tensor` didefinisikan pada GPU.
:end_tab:


```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
Hasil benchmark melalui MXNet jauh lebih cepat. Karena keduanya dieksekusi pada prosesor yang sama, pasti ada sesuatu yang lain yang terjadi.
Memaksa MXNet untuk menyelesaikan semua komputasi backend sebelum mengembalikan kontrol menunjukkan apa yang terjadi sebelumnya: komputasi dieksekusi oleh backend sementara frontend mengembalikan kontrol ke Python.
:end_tab:

:begin_tab:`pytorch`
Hasil benchmark melalui PyTorch jauh lebih cepat.
Operasi dot product pada NumPy dieksekusi di prosesor CPU, sedangkan
perkalian matriks pada PyTorch dieksekusi di GPU, sehingga yang terakhir ini
diperkirakan jauh lebih cepat. Namun, perbedaan waktu yang sangat besar menunjukkan bahwa ada sesuatu yang lain yang terjadi.
Secara default, operasi GPU bersifat asinkron di PyTorch.
Memaksa PyTorch untuk menyelesaikan semua komputasi sebelum mengembalikan kontrol menunjukkan apa yang terjadi sebelumnya: komputasi sedang dieksekusi oleh backend sementara frontend mengembalikan kontrol ke Python.
:end_tab:


```{.python .input}
#@tab mxnet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
Secara umum, MXNet memiliki frontend untuk berinteraksi langsung dengan pengguna, misalnya melalui Python, serta backend yang digunakan oleh sistem untuk melakukan komputasi. 
Seperti yang ditunjukkan pada :numref:`fig_frontends`, pengguna dapat menulis program MXNet dalam berbagai bahasa frontend, seperti Python, R, Scala, dan C++. Terlepas dari bahasa pemrograman frontend yang digunakan, eksekusi program MXNet sebagian besar terjadi di backend yang diimplementasikan dalam C++. Operasi yang dikeluarkan oleh bahasa frontend diteruskan ke backend untuk dieksekusi. 
Backend mengelola thread-nya sendiri yang terus-menerus mengumpulkan dan mengeksekusi tugas-tugas dalam antrian. Perhatikan bahwa agar ini bisa bekerja, backend harus mampu melacak ketergantungan antara berbagai langkah dalam grafik komputasi. Oleh karena itu, tidak mungkin untuk melakukan paralelisasi operasi yang saling bergantung.
:end_tab:

:begin_tab:`pytorch`
Secara umum, PyTorch memiliki frontend untuk berinteraksi langsung dengan pengguna, misalnya melalui Python, serta backend yang digunakan oleh sistem untuk melakukan komputasi. 
Seperti yang ditunjukkan pada :numref:`fig_frontends`, pengguna dapat menulis program PyTorch dalam berbagai bahasa frontend, seperti Python dan C++. Terlepas dari bahasa pemrograman frontend yang digunakan, eksekusi program PyTorch sebagian besar terjadi di backend yang diimplementasikan dalam C++. Operasi yang dikeluarkan oleh bahasa frontend diteruskan ke backend untuk dieksekusi.
Backend mengelola thread-nya sendiri yang terus-menerus mengumpulkan dan mengeksekusi tugas-tugas dalam antrian.
Perhatikan bahwa agar ini bisa bekerja, backend harus mampu melacak ketergantungan antara berbagai langkah dalam grafik komputasi.
Oleh karena itu, tidak mungkin untuk melakukan paralelisasi operasi yang saling bergantung.
:end_tab:

![Bahasa pemrograman frontend dan backend framework deep learning.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

Mari kita lihat contoh mainan lain untuk memahami grafik ketergantungan dengan lebih baik.



```{.python .input}
#@tab mxnet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![Backend melacak ketergantungan antara berbagai langkah dalam grafik komputasi.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

Cuplikan kode di atas juga diilustrasikan pada :numref:`fig_asyncgraph`.
Setiap kali thread frontend Python mengeksekusi salah satu dari tiga pernyataan pertama, ia hanya mengembalikan tugas ke antrian backend. Ketika hasil dari pernyataan terakhir perlu *dicetak*, thread frontend Python akan menunggu thread backend C++ untuk menyelesaikan komputasi hasil dari variabel `z`. Salah satu keuntungan dari desain ini adalah bahwa thread frontend Python tidak perlu melakukan komputasi yang sebenarnya. Oleh karena itu, kinerja Python tidak berdampak besar pada kinerja keseluruhan program. :numref:`fig_threading` mengilustrasikan bagaimana frontend dan backend saling berinteraksi.

![Interaksi antara frontend dan backend.](../img/threading.svg)
:label:`fig_threading`

## Hambatan dan Pemblokir

:begin_tab:`mxnet`
Ada sejumlah operasi yang akan memaksa Python untuk menunggu hingga selesai:

* Yang paling jelas adalah `npx.waitall()` yang akan menunggu hingga semua komputasi selesai, terlepas dari kapan instruksi komputasi diberikan. Dalam praktiknya, menggunakan operator ini adalah ide buruk kecuali jika benar-benar diperlukan karena dapat menyebabkan kinerja yang buruk.
* Jika kita hanya ingin menunggu hingga variabel tertentu tersedia, kita bisa memanggil `z.wait_to_read()`. Dalam kasus ini, MXNet akan memblokir pengembalian ke Python hingga variabel `z` selesai dihitung. Komputasi lainnya bisa terus dilanjutkan setelahnya.

Mari kita lihat bagaimana ini bekerja dalam praktik.
:end_tab:


```{.python .input}
#@tab mxnet
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
Kedua operasi memerlukan waktu yang hampir sama untuk diselesaikan. Selain operasi pemblokiran yang jelas, kami menyarankan agar Anda menyadari *pemblokir implisit*. Mencetak variabel jelas memerlukan variabel tersebut untuk tersedia dan dengan demikian menjadi pemblokir. Terakhir, konversi ke NumPy melalui `z.asnumpy()` dan konversi ke skalar melalui `z.item()` adalah operasi pemblokiran, karena NumPy tidak memiliki konsep asinkroni. Ia memerlukan akses ke nilai, seperti halnya fungsi `print`.

Menyalin data dalam jumlah kecil secara sering dari lingkup MXNet ke NumPy dan sebaliknya dapat merusak kinerja kode yang sebenarnya efisien, karena setiap operasi semacam itu memerlukan grafik komputasi untuk mengevaluasi semua hasil antara yang diperlukan untuk mendapatkan istilah terkait *sebelum* hal lain dapat dilakukan.
:end_tab:




```{.python .input}
#@tab mxnet
with d2l.Benchmark('konversi numpy'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('konversi scalar'):
    b = np.dot(a, a)
    b.sum().item()
```

## Meningkatkan Komputasi

:begin_tab:`mxnet`
Pada sistem dengan banyak thread (bahkan laptop biasa memiliki 4 thread atau lebih dan pada server dengan multi-soket jumlah ini dapat melebihi 256), overhead dari penjadwalan operasi bisa menjadi signifikan. Oleh karena itu, sangat diinginkan agar komputasi dan penjadwalan terjadi secara asinkron dan paralel. Untuk mengilustrasikan manfaat dari hal ini, mari kita lihat apa yang terjadi jika kita menambah variabel sebesar 1 beberapa kali, baik secara berurutan maupun asinkron. Kita mensimulasikan eksekusi sinkron dengan menyisipkan penghalang `wait_to_read` di antara setiap penambahan.
:end_tab:


```{.python .input}
#@tab mxnet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Interaksi yang sedikit disederhanakan antara thread frontend Python dan thread backend C++ dapat diringkas sebagai berikut:
1. Frontend memerintahkan backend untuk memasukkan tugas komputasi `y = x + 1` ke dalam antrian.
2. Backend kemudian menerima tugas komputasi dari antrian dan melakukan komputasi yang sebenarnya.
3. Backend kemudian mengembalikan hasil komputasi ke frontend.
Misalkan durasi dari ketiga tahap ini adalah $t_1, t_2$, dan $t_3$, secara berturut-turut. Jika kita tidak menggunakan pemrograman asinkron, total waktu yang dibutuhkan untuk melakukan 10000 komputasi kira-kira adalah $10000 (t_1+ t_2 + t_3)$. Jika pemrograman asinkron digunakan, total waktu yang dibutuhkan untuk melakukan 10000 komputasi dapat dikurangi menjadi $t_1 + 10000 t_2 + t_3$ (dengan asumsi $10000 t_2 > 9999t_1$), karena frontend tidak harus menunggu backend mengembalikan hasil komputasi untuk setiap loop.
:end_tab:


## Ringkasan

* Framework deep learning dapat memisahkan frontend Python dari backend eksekusi. Ini memungkinkan penyisipan perintah secara asinkron yang cepat ke dalam backend dan paralelisme yang terkait.
* Asinkroni menghasilkan frontend yang cukup responsif. Namun, perlu hati-hati agar tidak mengisi antrian tugas secara berlebihan karena dapat menyebabkan konsumsi memori yang berlebihan. Disarankan untuk melakukan sinkronisasi pada setiap minibatch untuk menjaga frontend dan backend tetap sinkron.
* Vendor chip menawarkan alat analisis kinerja yang canggih untuk memperoleh wawasan yang lebih terperinci tentang efisiensi deep learning.

:begin_tab:`mxnet`
* Perlu diketahui bahwa konversi dari manajemen memori MXNet ke Python akan memaksa backend untuk menunggu hingga variabel tertentu siap. Fungsi seperti `print`, `asnumpy`, dan `item` memiliki efek ini. Hal ini bisa diinginkan, tetapi penggunaan sinkronisasi yang ceroboh dapat merusak kinerja.
:end_tab:


## Latihan

:begin_tab:`mxnet`
1. Kami menyebutkan di atas bahwa menggunakan komputasi asinkron dapat mengurangi jumlah total waktu yang dibutuhkan untuk melakukan 10000 komputasi menjadi $t_1 + 10000 t_2 + t_3$. Mengapa kita harus mengasumsikan $10000 t_2 > 9999 t_1$ di sini?
2. Ukur perbedaan antara `waitall` dan `wait_to_read`. Petunjuk: lakukan sejumlah instruksi dan sinkronkan untuk hasil antara.
:end_tab:

:begin_tab:`pytorch`
1. Pada CPU, benchmark operasi perkalian matriks yang sama seperti pada bagian ini. Apakah Anda masih dapat mengamati asinkroni melalui backend?
:end_tab:

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/2564)
:end_tab:
