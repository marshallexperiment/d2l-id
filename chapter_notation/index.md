# Notasi
:label:`chap_notation`

Sepanjang buku ini, kami mengikuti konvensi notasi berikut.
Perhatikan bahwa beberapa simbol ini adalah placeholder,
sedangkan yang lainnya merujuk pada objek spesifik.
Sebagai aturan umum,
artikel tak tentu "a" sering menunjukkan
bahwa simbol tersebut adalah placeholder
dan bahwa simbol dengan format serupa
dapat menunjukkan objek lain dari jenis yang sama.
Sebagai contoh, "$x$: sebuah skalar" berarti
bahwa huruf kecil umumnya
mewakili nilai skalar,
tetapi "$\mathbb{Z}$: himpunan bilangan bulat"
merujuk secara spesifik pada simbol $\mathbb{Z}$.

## Objek Numerik

* $x$: sebuah skalar
* $\mathbf{x}$: sebuah vektor
* $\mathbf{X}$: sebuah matriks
* $\mathsf{X}$: tensor umum
* $\mathbf{I}$: matriks identitas (dari dimensi tertentu), yaitu matriks persegi dengan $1$ pada semua elemen diagonal dan $0$ pada elemen di luar diagonal
* $x_i$, $[\mathbf{x}]_i$: elemen ke-$i$ dari vektor $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$, $[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: elemen dari matriks $\mathbf{X}$ pada baris $i$ dan kolom $j$.

## Teori Himpunan

* $\mathcal{X}$: sebuah himpunan
* $\mathbb{Z}$: himpunan bilangan bulat
* $\mathbb{Z}^+$: himpunan bilangan bulat positif
* $\mathbb{R}$: himpunan bilangan real
* $\mathbb{R}^n$: himpunan vektor berdimensi-$n$ dari bilangan real
* $\mathbb{R}^{a\times b}$: himpunan matriks bilangan real dengan $a$ baris dan $b$ kolom
* $|\mathcal{X}|$: kardinalitas (jumlah elemen) dari himpunan $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: gabungan himpunan $\mathcal{A}$ dan $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: irisan himpunan $\mathcal{A}$ dan $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: pengurangan himpunan $\mathcal{B}$ dari $\mathcal{A}$ (mengandung hanya elemen dari $\mathcal{A}$ yang tidak termasuk dalam $\mathcal{B}$)

## Fungsi dan Operator

* $f(\cdot)$: sebuah fungsi
* $\log(\cdot)$: logaritma natural (basis $e$)
* $\log_2(\cdot)$: logaritma dengan basis $2$
* $\exp(\cdot)$: fungsi eksponensial
* $\mathbf{1}(\cdot)$: fungsi indikator; bernilai $1$ jika argumen boolean benar, dan $0$ sebaliknya
* $\mathbf{1}_{\mathcal{X}}(z)$: fungsi indikator keanggotaan himpunan; bernilai $1$ jika elemen $z$ termasuk dalam himpunan $\mathcal{X}$ dan $0$ sebaliknya
* $\mathbf{(\cdot)}^\top$: transpose dari vektor atau matriks
* $\mathbf{X}^{-1}$: invers dari matriks $\mathbf{X}$
* $\odot$: hasil kali Hadamard (elemen-wise)
* $[\cdot, \cdot]$: konkatenasi
* $\|\cdot\|_p$: norma $\ell_p$
* $\|\cdot\|$: norma $\ell_2$
* $\langle \mathbf{x}, \mathbf{y} \rangle$: produk dalam (dot product) dari vektor $\mathbf{x}$ dan $\mathbf{y}$
* $\sum$: penjumlahan atas suatu koleksi elemen
* $\prod$: perkalian atas suatu koleksi elemen
* $\stackrel{\textrm{def}}{=}$: kesetaraan yang ditegaskan sebagai definisi dari simbol di sisi kiri

## Kalkulus

* $\frac{dy}{dx}$: turunan dari $y$ terhadap $x$
* $\frac{\partial y}{\partial x}$: turunan parsial dari $y$ terhadap $x$
* $\nabla_{\mathbf{x}} y$: gradien dari $y$ terhadap $\mathbf{x}$
* $\int_a^b f(x) \;dx$: integral tentu dari $f$ dari $a$ ke $b$ terhadap $x$
* $\int f(x) \;dx$: integral tak tentu dari $f$ terhadap $x$

## Teori Probabilitas dan Informasi

* $X$: sebuah variabel acak
* $P$: sebuah distribusi probabilitas
* $X \sim P$: variabel acak $X$ mengikuti distribusi $P$
* $P(X=x)$: probabilitas yang diberikan kepada kejadian di mana variabel acak $X$ mengambil nilai $x$
* $P(X \mid Y)$: distribusi probabilitas kondisional dari $X$ diberikan $Y$
* $p(\cdot)$: fungsi densitas probabilitas (PDF) yang terkait dengan distribusi $P$
* ${E}[X]$: ekspektasi dari variabel acak $X$
* $X \perp Y$: variabel acak $X$ dan $Y$ independen
* $X \perp Y \mid Z$: variabel acak $X$ dan $Y$ kondisional independen diberikan $Z$
* $\sigma_X$: standar deviasi dari variabel acak $X$
* $\textrm{Var}(X)$: variansi dari variabel acak $X$, sama dengan $\sigma^2_X$
* $\textrm{Cov}(X, Y)$: kovarians dari variabel acak $X$ dan $Y$
* $\rho(X, Y)$: koefisien korelasi Pearson antara $X$ dan $Y$, sama dengan $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: entropi dari variabel acak $X$
* $D_{\textrm{KL}}(P\|Q)$: KL-divergence (atau entropi relatif) dari distribusi $Q$ ke distribusi $P$

[Diskusi](https://discuss.d2l.ai/t/25)
