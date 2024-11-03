```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Kalkulus
:label:`sec_calculus`

Selama berabad-abad, cara menghitung 
luas lingkaran tetap menjadi misteri.
Kemudian, di Yunani Kuno, matematikawan Archimedes
mendapat ide cerdik untuk
menyusun serangkaian poligon
dengan jumlah sisi yang meningkat
di dalam lingkaran
(:numref:`fig_circle_area`). 
Untuk poligon dengan $n$ titik sudut,
kita mendapatkan $n$ segitiga.
Tinggi setiap segitiga mendekati jari-jari $r$ 
seiring dengan pembagian lingkaran yang semakin halus. 
Pada saat yang sama, basisnya mendekati $2 \pi r/n$, 
karena rasio antara busur dan sekutannya mendekati 1 
untuk jumlah titik sudut yang besar. 
Dengan demikian, luas poligon mendekati
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$.

![Menemukan luas lingkaran sebagai prosedur limit.](../img/polygon-circle.svg)
:label:`fig_circle_area`

Prosedur limit ini adalah dasar dari 
*kalkulus diferensial* dan *kalkulus integral*. 
Yang pertama dapat memberi tahu kita bagaimana cara 
meningkatkan atau menurunkan nilai suatu fungsi 
dengan memanipulasi argumennya. 
Ini sangat berguna untuk *masalah optimisasi*
yang kita hadapi dalam pembelajaran mendalam,
di mana kita terus memperbarui parameter kita 
untuk mengurangi fungsi kerugian (loss function).
Optimisasi menangani bagaimana kita menyesuaikan model kita dengan data latih,
dan kalkulus adalah prasyarat utamanya.
Namun, jangan lupa bahwa tujuan utama kita
adalah tampil baik pada data *yang sebelumnya tidak terlihat*.
Masalah ini disebut *generalization*
dan akan menjadi fokus utama dari beberapa bab lainnya.


```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## Turunan dan Diferensiasi

Secara sederhana, *turunan* adalah tingkat perubahan
dalam suatu fungsi terhadap perubahan argumennya.
Turunan dapat memberi tahu kita seberapa cepat fungsi kerugian
akan meningkat atau menurun jika kita 
*meningkatkan* atau *menurunkan* setiap parameter
dengan jumlah yang sangat kecil.
Secara formal, untuk fungsi $f: \mathbb{R} \rightarrow \mathbb{R}$,
yang memetakan skalar ke skalar,
[**turunan dari $f$ di titik $x$ didefinisikan sebagai**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

Istilah di sisi kanan disebut *limit* 
dan ini memberi tahu kita apa yang terjadi 
pada nilai suatu ekspresi
saat variabel tertentu
mendekati nilai tertentu.
Limit ini memberi tahu kita apa 
rasio antara gangguan $h$
dan perubahan nilai fungsi 
$f(x + h) - f(x)$ yang akan terjadi
ketika kita mengurangi nilainya hingga nol.

Ketika $f'(x)$ ada, maka $f$ dikatakan 
*berturunan* di $x$;
dan ketika $f'(x)$ ada untuk semua $x$
pada suatu himpunan, misalnya pada interval $[a,b]$, 
kita mengatakan bahwa $f$ dapat diturunkan pada himpunan ini.
Tidak semua fungsi dapat diturunkan,
termasuk banyak fungsi yang ingin kita optimalkan,
seperti akurasi dan area di bawah
kurva karakteristik penerima (AUC).
Namun, karena menghitung turunan dari kerugian 
adalah langkah penting di hampir semua 
algoritma untuk melatih jaringan saraf mendalam,
kita sering kali mengoptimalkan *surrogat* yang dapat diturunkan sebagai gantinya.


Kita dapat menginterpretasikan turunan 
$f'(x)$
sebagai tingkat perubahan *seketika* 
dari $f(x)$ terhadap $x$.
Mari kita kembangkan sedikit intuisi dengan sebuah contoh.
(**Definisikan $u = f(x) = 3x^2-4x$.**)


```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

[**Dengan mengatur $x=1$, kita melihat bahwa $\frac{f(x+h) - f(x)}{h}$**] (**mendekati $2$
ketika $h$ mendekati $0$.**)
Meskipun percobaan ini tidak memiliki 
ketelitian seperti pembuktian matematis,
kita dapat dengan cepat melihat bahwa memang $f'(1) = 2$.


```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

Terdapat beberapa konvensi notasi yang setara untuk turunan.
Diberikan $y = f(x)$, ekspresi berikut adalah setara:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

di mana simbol $\frac{d}{dx}$ dan $D$ adalah *operator diferensiasi*.
Berikut ini adalah turunan dari beberapa fungsi umum:

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{untuk konstanta apa pun $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{untuk } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

Fungsi yang dibentuk dari komposisi fungsi-fungsi yang dapat didiferensiasikan 
seringkali juga dapat didiferensiasikan.
Aturan-aturan berikut berguna 
untuk bekerja dengan komposisi 
fungsi-fungsi yang dapat didiferensiasikan 
$f$ dan $g$, serta konstanta $C$.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{Aturan perkalian konstanta} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{Aturan penjumlahan} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{Aturan perkalian} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{Aturan pembagian} \end{aligned}$$

Dengan ini, kita dapat menerapkan aturan-aturan tersebut 
untuk menemukan turunan dari $3 x^2 - 4x$ melalui

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

Memasukkan $x = 1$ menunjukkan bahwa, benar,
turunannya bernilai $2$ pada lokasi ini. 
Perhatikan bahwa turunan memberi tahu kita 
*kemiringan* suatu fungsi 
pada lokasi tertentu.

## Utilitas Visualisasi

[**Kita dapat memvisualisasikan kemiringan fungsi menggunakan pustaka `matplotlib`**].
Kita perlu mendefinisikan beberapa fungsi.
Seperti namanya, `use_svg_display` 
memberi tahu `matplotlib` untuk menampilkan grafik 
dalam format SVG untuk gambar yang lebih tajam.
Komentar `#@save` adalah modifikasi khusus 
yang memungkinkan kita menyimpan fungsi apa pun, 
kelas, atau blok kode lainnya ke dalam paket `d2l` 
sehingga kita dapat memanggilnya nanti 
tanpa mengulangi kode, 
misalnya, melalui `d2l.use_svg_display()`.


```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

Dengan mudah, kita dapat mengatur ukuran gambar menggunakan `set_figsize`.
Karena pernyataan impor `from matplotlib import pyplot as plt`
ditandai dengan `#@save` dalam paket `d2l`, kita dapat memanggilnya melalui `d2l.plt`.


```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

Fungsi `set_axes` dapat menghubungkan sumbu
dengan properti, termasuk label, rentang,
dan skala.


```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Mengatur sumbu untuk matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Dengan ketiga fungsi ini, kita bisa mendefinisikan fungsi `plot` 
untuk menumpuk beberapa kurva. 
Sebagian besar kode di sini hanya memastikan 
bahwa ukuran dan bentuk input sesuai.


```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Sekarang kita dapat [**memplot fungsi $u = f(x)$ dan garis singgungnya $y = 2x - 3$ pada $x=1$**],
dengan koefisien $2$ sebagai kemiringan garis singgung.


```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Turunan Parsial dan Gradien
:label:`subsec_calculus-grad`

Sejauh ini, kita telah mempelajari cara menurunkan
fungsi yang hanya memiliki satu variabel.
Dalam deep learning, kita juga perlu bekerja
dengan fungsi yang memiliki *banyak* variabel.
Di sini, kita akan memperkenalkan konsep turunan
yang berlaku untuk fungsi *multivariat* semacam itu.


Misalkan $y = f(x_1, x_2, \ldots, x_n)$ adalah sebuah fungsi dengan $n$ variabel. 
*Turunan parsial* dari $y$ 
terhadap parameter ke-$i$ $x_i$ adalah

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

Untuk menghitung $\frac{\partial y}{\partial x_i}$, 
kita dapat menganggap $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ sebagai konstanta 
dan menghitung turunan dari $y$ terhadap $x_i$.
Notasi-notasi berikut untuk turunan parsial 
sering digunakan dan memiliki arti yang sama:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

Kita dapat menggabungkan turunan parsial 
dari fungsi multivariat 
terhadap semua variabelnya 
untuk mendapatkan vektor yang disebut
*gradien* dari fungsi tersebut.
Misalkan input dari fungsi 
$f: \mathbb{R}^n \rightarrow \mathbb{R}$ 
adalah sebuah vektor berdimensi-$n$ 
$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ 
dan output-nya adalah sebuah skalar. 
Gradien dari fungsi $f$ 
terhadap $\mathbf{x}$ 
adalah sebuah vektor dari $n$ turunan parsial:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

Jika tidak ada ambiguitas,
$\nabla_{\mathbf{x}} f(\mathbf{x})$ 
biasanya ditulis sebagai 
$\nabla f(\mathbf{x})$.
Berikut ini adalah beberapa aturan
yang berguna untuk menurunkan fungsi multivariat:

* Untuk semua $\mathbf{A} \in \mathbb{R}^{m \times n}$ berlaku $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ dan $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* Untuk matriks persegi $\mathbf{A} \in \mathbb{R}^{n \times n}$ berlaku $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ dan khususnya
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Demikian pula, untuk setiap matriks $\mathbf{X}$, 
berlaku $\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$.




## Aturan Rantai

Dalam deep learning, gradien yang kita hitung
sering kali sulit dihitung
karena kita bekerja dengan fungsi 
yang memiliki tingkat kedalaman yang dalam 
(dalam fungsi (dalam fungsi ...)).
Untungnya, *aturan rantai* menyelesaikan masalah ini. 
Kembali ke fungsi dengan satu variabel,
misalkan $y = f(g(x))$
dan fungsi dasarnya 
$y=f(u)$ dan $u=g(x)$ 
keduanya terdiferensiasi.
Aturan rantai menyatakan bahwa

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

Kembali ke fungsi multivariat,
misalkan $y = f(\mathbf{u})$ memiliki variabel
$u_1, u_2, \ldots, u_m$, 
di mana setiap $u_i = g_i(\mathbf{x})$ 
memiliki variabel $x_1, x_2, \ldots, x_n$,
yaitu, $\mathbf{u} = g(\mathbf{x})$.
Maka aturan rantai menyatakan bahwa

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ dan } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

di mana $\mathbf{A} \in \mathbb{R}^{n \times m}$ adalah sebuah *matriks*
yang mengandung turunan vektor $\mathbf{u}$
terhadap vektor $\mathbf{x}$.
Dengan demikian, evaluasi gradien memerlukan 
menghitung hasil kali vektor-matriks. 
Inilah salah satu alasan utama mengapa aljabar linear 
merupakan komponen penting 
dalam membangun sistem deep learning. 


## Diskusi

Walaupun kita hanya menggoreskan permukaan dari topik yang mendalam ini,
beberapa konsep utama sudah mulai terlihat:
pertama, aturan komposisi untuk diferensiasi
dapat diterapkan secara rutin, memungkinkan kita
untuk menghitung gradien secara *otomatis*.
Tugas ini tidak memerlukan kreativitas, sehingga 
kita bisa fokus pada aspek lain yang lebih penting.
Kedua, menghitung turunan fungsi yang menghasilkan vektor 
memerlukan kita untuk mengalikan matriks ketika kita melacak 
graf dependensi variabel dari output ke input. 
Graf ini dilalui dalam arah *maju* 
saat kita mengevaluasi fungsi 
dan dalam arah *mundur* 
saat kita menghitung gradien. 
Bab selanjutnya akan memperkenalkan backpropagation,
prosedur komputasi untuk menerapkan aturan rantai.

Dari sudut pandang optimasi, gradien memungkinkan kita 
menentukan cara untuk mengubah parameter model
untuk mengurangi loss,
dan setiap langkah dalam algoritma optimasi 
yang digunakan di sepanjang buku ini 
akan memerlukan perhitungan gradien.

## Latihan

1. Sejauh ini kita menganggap aturan turunan sebagai hal yang sudah ada. 
   Menggunakan definisi dan limit, buktikan sifat-sifat 
   untuk (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$ dan (iv) $f(x) = \log x$.
2. Dengan cara yang sama, buktikan aturan perkalian, penjumlahan, dan pembagian dari prinsip dasar.
3. Buktikan bahwa aturan perkalian konstanta adalah kasus khusus dari aturan perkalian.
4. Hitung turunan dari $f(x) = x^x$. 
5. Apa artinya jika $f'(x) = 0$ untuk beberapa $x$? 
   Berikan contoh fungsi $f$ 
   dan posisi $x$ di mana hal ini mungkin terjadi. 
6. Gambarkan fungsi $y = f(x) = x^3 - \frac{1}{x}$ 
   dan gambarkan garis singgungnya di $x = 1$.
7. Temukan gradien dari fungsi 
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
8. Apa gradien dari fungsi 
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$? Apa yang terjadi jika $\mathbf{x} = \mathbf{0}$?
9. Dapatkah Anda menuliskan aturan rantai untuk kasus 
   di mana $u = f(x, y, z)$ dan $x = x(a, b)$, $y = y(a, b)$, dan $z = z(a, b)$?
10. Diberikan sebuah fungsi $f(x)$ yang dapat dibalik, 
   hitung turunan dari inversinya $f^{-1}(x)$. 
   Di sini berlaku bahwa $f^{-1}(f(x)) = x$ dan sebaliknya $f(f^{-1}(y)) = y$. 
   Petunjuk: gunakan properti ini dalam perhitungan Anda.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/197)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17969)
:end_tab:
