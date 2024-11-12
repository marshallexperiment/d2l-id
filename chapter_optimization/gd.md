# Gradient Descent
:label:`sec_gd`

Pada bagian ini, kita akan memperkenalkan konsep dasar yang mendasari *gradient descent*. Meskipun jarang digunakan secara langsung dalam deep learning, pemahaman tentang gradient descent sangat penting untuk memahami algoritma stochastic gradient descent. Misalnya, masalah optimisasi mungkin menyimpang karena learning rate yang terlalu besar. Fenomena ini sudah dapat dilihat dalam gradient descent. Begitu juga dengan preconditioning yang merupakan teknik umum dalam gradient descent dan diterapkan pada algoritma yang lebih canggih. Mari kita mulai dengan kasus khusus yang sederhana.

## Gradient Descent Satu Dimensi

Gradient descent dalam satu dimensi adalah contoh yang sangat baik untuk menjelaskan mengapa algoritma gradient descent dapat mengurangi nilai fungsi objektif. Pertimbangkan beberapa fungsi bernilai real yang dapat didiferensialkan secara kontinu $f: \mathbb{R} \rightarrow \mathbb{R}$. Menggunakan ekspansi Taylor, kita mendapatkan

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Artinya, dalam pendekatan orde pertama, $f(x+\epsilon)$ diberikan oleh nilai fungsi $f(x)$ dan turunan pertama $f'(x)$ di $x$. Tidak berlebihan untuk berasumsi bahwa untuk $\epsilon$ yang kecil, bergerak dalam arah gradien negatif akan mengurangi nilai $f$. Untuk menyederhanakan, kita memilih ukuran langkah tetap $\eta > 0$ dan memilih $\epsilon = -\eta f'(x)$. Dengan memasukkan ini ke dalam ekspansi Taylor di atas, kita mendapatkan

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Jika turunan $f'(x) \neq 0$ tidak nol, kita membuat kemajuan karena $\eta f'^2(x)>0$. Selain itu, kita selalu dapat memilih $\eta$ cukup kecil sehingga suku-suku orde lebih tinggi menjadi tidak relevan. Maka kita sampai pada

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Ini berarti bahwa jika kita menggunakan

$$x \leftarrow x - \eta f'(x)$$

untuk mengiterasi $x$, nilai fungsi $f(x)$ dapat menurun. Oleh karena itu, dalam gradient descent kita pertama-tama memilih nilai awal $x$ dan konstanta $\eta > 0$ kemudian menggunakannya untuk terus mengiterasi $x$ sampai kondisi penghentian tercapai, misalnya, ketika besar gradien $|f'(x)|$ cukup kecil atau jumlah iterasi telah mencapai nilai tertentu.

Untuk kesederhanaan, kita memilih fungsi objektif $f(x)=x^2$ untuk mengilustrasikan cara mengimplementasikan gradient descent. Meskipun kita tahu bahwa $x=0$ adalah solusi untuk meminimalkan $f(x)$, kita masih menggunakan fungsi sederhana ini untuk mengamati bagaimana $x$ berubah.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Selanjutnya, kita menggunakan $x=10$ sebagai nilai awal dan mengasumsikan $\eta=0.2$. Dengan menggunakan gradient descent untuk mengiterasi $x$ sebanyak 10 kali, kita dapat melihat bahwa pada akhirnya nilai $x$ mendekati solusi optimal.


```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

Perkembangan optimisasi pada $x$ dapat digambarkan sebagai berikut.


```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Learning Rate
:label:`subsec_gd-learningrate`

Learning rate $\eta$ dapat ditetapkan oleh perancang algoritma. Jika kita menggunakan learning rate yang terlalu kecil, hal ini akan menyebabkan $x$ diperbarui dengan sangat lambat, sehingga memerlukan lebih banyak iterasi untuk mendapatkan solusi yang lebih baik. Untuk menunjukkan apa yang terjadi dalam kasus seperti ini, pertimbangkan perkembangan pada masalah optimisasi yang sama untuk $\eta = 0.05$. Seperti yang kita lihat, bahkan setelah 10 langkah, kita masih sangat jauh dari solusi optimal.


```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Sebaliknya, jika kita menggunakan learning rate yang terlalu tinggi, $\left|\eta f'(x)\right|$ mungkin terlalu besar untuk memenuhi formula ekspansi Taylor orde pertama. Artinya, suku $\mathcal{O}(\eta^2 f'^2(x))$ dalam :eqref:`gd-taylor-2` mungkin menjadi signifikan. Dalam kasus ini, kita tidak dapat menjamin bahwa iterasi $x$ akan dapat menurunkan nilai $f(x)$. Misalnya, ketika kita menetapkan learning rate menjadi $\eta=1.1$, nilai $x$ melampaui solusi optimal $x=0$ dan secara bertahap menyimpang.


```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Minimum Lokal

Untuk mengilustrasikan apa yang terjadi pada fungsi non-konveks, pertimbangkan kasus $f(x) = x \cdot \cos(cx)$ untuk beberapa konstanta $c$. Fungsi ini memiliki tak hingga banyaknya minimum lokal. Bergantung pada pilihan learning rate kita dan pada seberapa baik kondisi masalahnya, kita mungkin berakhir dengan salah satu dari banyak solusi. Contoh di bawah ini mengilustrasikan bagaimana learning rate yang (secara tidak realistis) tinggi akan mengarah ke minimum lokal yang buruk.


```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Fungsi Obyektif
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient dari fungsi obyektif
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Gradient Descent Multivariat

Setelah kita memiliki intuisi yang lebih baik tentang kasus univariat, sekarang kita pertimbangkan situasi di mana $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. Artinya, fungsi objektif $f: \mathbb{R}^d \to \mathbb{R}$ memetakan vektor ke skalar. Dengan demikian, gradiennya juga multivariat. Ini adalah vektor yang terdiri dari $d$ turunan parsial:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Setiap elemen turunan parsial $\partial f(\mathbf{x})/\partial x_i$ dalam gradien menunjukkan laju perubahan $f$ pada $\mathbf{x}$ terhadap input $x_i$. Seperti pada kasus univariat, kita dapat menggunakan pendekatan Taylor yang sesuai untuk fungsi multivariat untuk mendapatkan gambaran tentang apa yang harus kita lakukan. Secara khusus, kita memiliki:

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Dengan kata lain, hingga suku orde kedua dalam $\boldsymbol{\epsilon}$, arah turunan paling curam diberikan oleh gradien negatif $-\nabla f(\mathbf{x})$. Memilih learning rate yang sesuai $\eta > 0$ menghasilkan algoritma gradient descent prototipikal:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Untuk melihat bagaimana algoritma ini berperilaku dalam praktik, mari kita buat fungsi objektif $f(\mathbf{x})=x_1^2+2x_2^2$ dengan vektor dua dimensi $\mathbf{x} = [x_1, x_2]^\top$ sebagai input dan skalar sebagai output. Gradiennya diberikan oleh $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Kita akan mengamati lintasan $\mathbf{x}$ menggunakan gradient descent dari posisi awal $[-5, -2]$.

Untuk memulai, kita membutuhkan dua fungsi pembantu tambahan. Fungsi pertama menggunakan fungsi pembaruan dan menerapkannya sebanyak 20 kali pada nilai awal. Fungsi pembantu kedua memvisualisasikan lintasan $\mathbf{x}$.


```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Mengoptimalkan fungsi objektif 2D dengan pelatih khusus."""
    # `s1` dan `s2` adalah variabel status internal yang akan digunakan dalam Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
   """Menampilkan jejak (trace) variabel 2D selama optimisasi."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """Menampilkan jejak (trace) variabel 2D selama optimisasi."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """Menampilkan jejak (trace) variabel 2D selama optimisasi."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Selanjutnya, kita mengamati lintasan dari variabel optimisasi $\mathbf{x}$ untuk learning rate $\eta = 0.1$. Kita dapat melihat bahwa setelah 20 langkah, nilai $\mathbf{x}$ mendekati minimum pada $[0, 0]$. Kemajuannya cukup baik meskipun agak lambat.


```{.python .input}
#@tab all
def f_2d(x1, x2):  # Fungsi Obyektif
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient dari Fungsi Obyektif
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Metode Adaptif

Seperti yang dapat kita lihat di :numref:`subsec_gd-learningrate`, menentukan learning rate $\eta$ yang "tepat" cukup rumit. Jika kita memilihnya terlalu kecil, kemajuan akan lambat. Jika kita memilihnya terlalu besar, solusinya akan berosilasi dan dalam kasus terburuk bahkan dapat menyimpang. Bagaimana jika kita bisa menentukan $\eta$ secara otomatis atau bahkan menghilangkan keharusan memilih learning rate sama sekali?
Metode orde kedua yang tidak hanya melihat nilai dan gradien fungsi objektif, tetapi juga *kelengkungannya*, dapat membantu dalam kasus ini. Meskipun metode ini tidak dapat diterapkan langsung pada deep learning karena biaya komputasi, metode ini memberikan intuisi yang berguna tentang cara merancang algoritma optimisasi lanjutan yang meniru banyak properti yang diinginkan dari algoritma yang diuraikan di bawah ini.

### Metode Newton

Mereview ekspansi Taylor dari suatu fungsi $f: \mathbb{R}^d \rightarrow \mathbb{R}$, tidak perlu berhenti pada suku pertama. Faktanya, kita dapat menuliskannya sebagai

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Untuk menghindari notasi yang rumit, kita mendefinisikan $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$ sebagai Hessian dari $f$, yang merupakan matriks $d \times d$. Untuk $d$ yang kecil dan masalah sederhana, $\mathbf{H}$ mudah dihitung. Namun, untuk jaringan neural yang dalam, $\mathbf{H}$ mungkin terlalu besar, karena biaya penyimpanan $\mathcal{O}(d^2)$ entri. Selain itu, mungkin terlalu mahal untuk dihitung melalui backpropagation. Untuk sekarang mari kita abaikan pertimbangan ini dan lihat algoritma apa yang akan kita dapatkan.

Bagaimanapun, minimum dari $f$ memenuhi $\nabla f = 0$.
Mengikuti aturan kalkulus pada :numref:`subsec_calculus-grad`,
dengan mengambil turunan dari :eqref:`gd-hot-taylor` terhadap $\boldsymbol{\epsilon}$ dan mengabaikan suku-suku orde lebih tinggi, kita mendapatkan

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ dan karenanya }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Artinya, kita perlu membalikkan Hessian $\mathbf{H}$ sebagai bagian dari masalah optimisasi.

Sebagai contoh sederhana, untuk $f(x) = \frac{1}{2} x^2$ kita memiliki $\nabla f(x) = x$ dan $\mathbf{H} = 1$. Oleh karena itu, untuk setiap $x$ kita memperoleh $\epsilon = -x$. Dengan kata lain, *satu* langkah sudah cukup untuk mencapai konvergensi sempurna tanpa memerlukan penyesuaian apa pun! Sayangnya, kita agak beruntung di sini: ekspansi Taylor tepat karena $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$.

Mari kita lihat apa yang terjadi pada masalah lain.
Diberikan fungsi konveks kosinus hiperbolik $f(x) = \cosh(cx)$ untuk beberapa konstanta $c$, kita dapat melihat bahwa
minimum global pada $x=0$ tercapai
setelah beberapa iterasi.


```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Fungsi Obyektif
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient dari Fungsi Obyektif
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian dari Fungsi Obyektif
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Sekarang mari kita pertimbangkan sebuah fungsi *nonkonveks*, seperti $f(x) = x \cos(c x)$ untuk beberapa konstanta $c$. Ingatlah bahwa dalam metode Newton kita akan membagi dengan Hessian. Ini berarti jika turunan kedua *negatif*, kita mungkin bergerak ke arah yang *meningkatkan* nilai $f$. Ini adalah kelemahan fatal dari algoritma ini. Mari kita lihat apa yang terjadi dalam praktik.


```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Fungsi Obyektif
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient dari Fungsi Obyektif
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian dari Fungsi Obyektif
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Ini benar-benar gagal secara spektakuler. Bagaimana kita bisa memperbaikinya? Salah satu cara adalah dengan "memperbaiki" Hessian dengan mengambil nilai absolutnya. Strategi lain adalah dengan membawa kembali learning rate. Ini mungkin tampak mengalahkan tujuan, tetapi tidak sepenuhnya. Memiliki informasi orde kedua memungkinkan kita untuk berhati-hati ketika kelengkungan besar dan mengambil langkah lebih panjang ketika fungsi objektif lebih datar.
Mari kita lihat bagaimana ini bekerja dengan learning rate yang sedikit lebih kecil, misalnya $\eta = 0.5$. Seperti yang dapat kita lihat, kita memiliki algoritma yang cukup efisien.


```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Analisis Konvergensi

Kita hanya akan menganalisis laju konvergensi dari metode Newton untuk fungsi objektif konveks dan tiga kali dapat didiferensialkan $f$, di mana turunan keduanya tidak nol, yaitu $f'' > 0$. Pembuktian multivariat adalah perpanjangan langsung dari argumen satu dimensi di bawah ini dan tidak disertakan karena tidak terlalu membantu kita dalam hal intuisi.

Nyatakan $x^{(k)}$ sebagai nilai $x$ pada iterasi ke-$k$ dan biarkan $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$ sebagai jarak dari optimalitas pada iterasi ke-$k$. Berdasarkan ekspansi Taylor, kita memiliki kondisi $f'(x^*) = 0$ yang dapat dituliskan sebagai

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

yang berlaku untuk beberapa $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$. Dengan membagi ekspansi di atas dengan $f''(x^{(k)})$, kita mendapatkan

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Ingat bahwa kita memiliki pembaruan $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$.
Memasukkan persamaan pembaruan ini dan mengambil nilai absolut dari kedua sisinya, kita mendapatkan

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Akibatnya, setiap kali kita berada di wilayah $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ yang terbatas, kita memiliki error yang menurun secara kuadrat

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

Sebagai catatan, peneliti optimisasi menyebut ini sebagai konvergensi *linear*, sedangkan kondisi seperti $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ disebut sebagai laju konvergensi *konstan*.
Perlu dicatat bahwa analisis ini disertai dengan sejumlah peringatan.
Pertama, kita sebenarnya tidak memiliki banyak jaminan kapan kita akan mencapai wilayah konvergensi cepat. Sebaliknya, kita hanya tahu bahwa setelah kita mencapainya, konvergensi akan sangat cepat. Kedua, analisis ini mengharuskan $f$ berperilaku baik hingga turunan orde lebih tinggi. Hal ini bergantung pada memastikan bahwa $f$ tidak memiliki sifat yang "mengejutkan" dalam hal bagaimana ia mungkin mengubah nilainya.

### Preconditioning

Tidak mengherankan, menghitung dan menyimpan Hessian penuh sangat mahal. Oleh karena itu, diinginkan untuk mencari alternatif. Salah satu cara untuk memperbaiki masalah ini adalah *preconditioning*. Ini menghindari perhitungan Hessian secara keseluruhan dan hanya menghitung entri *diagonal*-nya. Hal ini mengarah pada algoritma pembaruan seperti berikut

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

Meskipun ini tidak sebaik metode Newton penuh, namun masih jauh lebih baik dibandingkan jika tidak menggunakannya.
Untuk melihat mengapa ini bisa menjadi ide yang bagus, pertimbangkan situasi di mana satu variabel menyatakan tinggi dalam milimeter dan yang lainnya menyatakan tinggi dalam kilometer. Dengan asumsi bahwa untuk keduanya skala alami adalah dalam meter, kita memiliki ketidakcocokan parametrisasi yang sangat buruk. Untungnya, penggunaan preconditioning menghilangkan masalah ini. Secara efektif, preconditioning dengan gradient descent berarti memilih learning rate yang berbeda untuk setiap variabel (koordinat dari vektor $\mathbf{x}$).
Seperti yang akan kita lihat nanti, preconditioning mendorong beberapa inovasi dalam algoritma optimisasi stochastic gradient descent.

### Gradient Descent dengan Line Search

Salah satu masalah utama dalam gradient descent adalah kita mungkin melampaui tujuan atau membuat kemajuan yang tidak cukup. Solusi sederhana untuk masalah ini adalah menggunakan line search bersamaan dengan gradient descent. Artinya, kita menggunakan arah yang diberikan oleh $\nabla f(\mathbf{x})$ dan kemudian melakukan pencarian biner untuk mengetahui learning rate $\eta$ yang meminimalkan $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$.

Algoritma ini mengalami konvergensi dengan cepat (untuk analisis dan pembuktian lihat misalnya :citet:`Boyd.Vandenberghe.2004`). Namun, untuk tujuan deep learning, ini tidak cukup layak, karena setiap langkah dari line search akan mengharuskan kita mengevaluasi fungsi objektif pada seluruh dataset. Hal ini terlalu mahal untuk dilakukan.

## Ringkasan

* Learning rate penting. Terlalu besar bisa menyebabkan divergensi, terlalu kecil dan kita tidak membuat kemajuan.
* Gradient descent dapat terjebak pada minimum lokal.
* Pada dimensi tinggi, menyesuaikan learning rate adalah hal yang rumit.
* Preconditioning dapat membantu dalam penyesuaian skala.
* Metode Newton jauh lebih cepat begitu ia bekerja dengan baik pada masalah konveks.
* Berhati-hatilah menggunakan metode Newton tanpa penyesuaian apa pun pada masalah nonkonveks.

## Latihan

1. Bereksperimenlah dengan learning rate dan fungsi objektif yang berbeda untuk gradient descent.
2. Implementasikan line search untuk meminimalkan fungsi konveks dalam interval $[a, b]$.
    1. Apakah Anda memerlukan turunan untuk pencarian biner, yaitu, untuk memutuskan apakah memilih $[a, (a+b)/2]$ atau $[(a+b)/2, b]$?
    2. Seberapa cepat laju konvergensi untuk algoritma ini?
    3. Implementasikan algoritma ini dan terapkan untuk meminimalkan $\log (\exp(x) + \exp(-2x -3))$.
3. Rancang fungsi objektif yang didefinisikan pada $\mathbb{R}^2$ di mana gradient descent sangat lambat. Petunjuk: skala koordinat yang berbeda dengan berbeda.
4. Implementasikan versi ringan dari metode Newton menggunakan preconditioning:
    1. Gunakan Hessian diagonal sebagai preconditioner.
    2. Gunakan nilai absolut dari itu daripada nilai sebenarnya (yang mungkin bertanda).
    3. Terapkan ini pada masalah di atas.
5. Terapkan algoritma di atas pada sejumlah fungsi objektif (baik konveks maupun tidak). Apa yang terjadi jika Anda memutar koordinat sebesar $45$ derajat?

[Diskusi](https://discuss.d2l.ai/t/351)
