# Dari Lapisan Fully Connected ke Konvolusi
:label:`sec_why-conv`

Hingga saat ini,
model yang telah kita bahas sejauh ini
tetap menjadi pilihan yang sesuai
ketika kita berhadapan dengan data tabular.
Dengan data tabular, kita maksudkan bahwa data terdiri dari
baris yang sesuai dengan contoh dan kolom yang sesuai dengan fitur.
Pada data tabular, kita mungkin memperkirakan bahwa pola yang kita cari
dapat melibatkan interaksi antar fitur,
tetapi kita tidak mengasumsikan adanya struktur *a priori*
mengenai cara fitur tersebut berinteraksi.

Kadang-kadang, kita benar-benar kekurangan pengetahuan untuk dapat membimbing pembuatan arsitektur yang lebih kompleks.
Dalam kasus ini, MLP mungkin merupakan pilihan terbaik yang dapat kita lakukan.
Namun, untuk data perseptual berdimensi tinggi,
jaringan tanpa struktur ini bisa menjadi sangat besar.

Sebagai contoh, mari kita kembali ke contoh kita
tentang membedakan kucing dari anjing.
Misalkan kita melakukan pengumpulan data secara menyeluruh,
mengumpulkan dataset foto beranotasi dengan resolusi satu megapiksel.
Ini berarti bahwa setiap input ke jaringan memiliki satu juta dimensi.
Bahkan jika kita mengurangi secara agresif menjadi seribu dimensi tersembunyi,
lapisan fully connected akan membutuhkan $10^6 \times 10^3 = 10^9$ parameter.
Kecuali kita memiliki banyak GPU, bakat untuk optimasi terdistribusi,
dan kesabaran yang luar biasa, mempelajari parameter dari jaringan ini
mungkin tidak dapat dilakukan.

Pembaca yang teliti mungkin membantah argumen ini
dengan alasan bahwa resolusi satu megapiksel mungkin tidak diperlukan.
Namun, meskipun kita bisa mengurangi menjadi seratus ribu piksel,
lapisan tersembunyi kita yang berukuran 1000 terlalu kecil untuk
mempelajari representasi gambar yang baik,
sehingga sistem praktis masih akan membutuhkan miliaran parameter.
Selain itu, mempelajari classifier dengan menyesuaikan begitu banyak parameter
mungkin memerlukan pengumpulan dataset yang sangat besar.
Namun, saat ini baik manusia maupun komputer mampu
membedakan kucing dari anjing dengan sangat baik,
seolah-olah bertentangan dengan intuisi tersebut.
Hal ini karena gambar memiliki struktur kaya yang dapat dimanfaatkan
oleh manusia maupun model machine learning.
Convolutional neural networks (CNNs) adalah salah satu cara kreatif
yang diadopsi machine learning untuk memanfaatkan
struktur yang dikenal dalam gambar alami.


## Invarian

Bayangkan kita ingin mendeteksi objek dalam sebuah gambar.
Tampaknya wajar bahwa metode apa pun yang kita gunakan untuk mengenali objek
tidak perlu terlalu peduli dengan lokasi tepat objek dalam gambar.
Idealnya, sistem kita harus memanfaatkan pengetahuan ini.
Babi biasanya tidak terbang dan pesawat biasanya tidak berenang.
Namun demikian, kita harus tetap mengenali babi
jika muncul di bagian atas gambar.
Kita bisa mengambil inspirasi dari permainan anak-anak "Di mana Waldo"
(yang sendiri telah menginspirasi banyak tiruan kehidupan nyata, seperti yang digambarkan dalam :numref:`img_waldo`).
Permainan ini terdiri dari beberapa adegan kacau
penuh dengan aktivitas.
Waldo muncul di suatu tempat dalam setiap adegan,
biasanya bersembunyi di lokasi yang tidak biasa.
Tujuan pembaca adalah menemukannya.
Meskipun pakaian khasnya,
ini bisa sangat sulit,
karena ada begitu banyak gangguan.
Namun, *penampilan Waldo*
tidak bergantung pada *lokasi Waldo*.
Kita bisa menyapu gambar dengan detektor Waldo
yang dapat memberikan skor pada setiap patch,
yang menunjukkan kemungkinan bahwa patch tersebut berisi Waldo.
Faktanya, banyak algoritma deteksi dan segmentasi objek
berbasis pada pendekatan ini :cite:`Long.Shelhamer.Darrell.2015`.
CNNs mengatur ide *invarian spasial* ini secara sistematis,
memanfaatkannya untuk mempelajari representasi yang berguna
dengan lebih sedikit parameter.

![Bisakah Anda menemukan Waldo (gambar milik William Murphy (Infomatique))?](../img/waldo-football.jpg)
:width:`400px`
:label:`img_waldo`


Sekarang kita dapat membuat intuisi ini lebih konkret
dengan merinci beberapa kriteria untuk memandu desain
arsitektur jaringan saraf yang sesuai untuk computer vision:

1. Pada lapisan-lapisan awal, jaringan kita
   harus merespons secara serupa terhadap patch yang sama,
   terlepas dari di mana patch tersebut muncul dalam gambar. Prinsip ini disebut *translation invariance* (atau *translation equivariance*).
1. Lapisan-lapisan awal dari jaringan harus berfokus pada wilayah lokal,
   tanpa memperhatikan isi gambar di wilayah yang jauh. Ini adalah prinsip *locality*.
   Pada akhirnya, representasi lokal ini dapat digabungkan
   untuk membuat prediksi di level gambar keseluruhan.
1. Seiring berjalannya waktu, lapisan-lapisan yang lebih dalam harus mampu menangkap fitur yang memiliki jangkauan lebih jauh dalam gambar, dengan cara yang mirip dengan tingkat penglihatan yang lebih tinggi di alam.

Mari kita lihat bagaimana ini diterjemahkan ke dalam matematika.


## Membatasi MLP

Untuk memulai, kita dapat mempertimbangkan sebuah MLP
dengan gambar dua dimensi $\mathbf{X}$ sebagai input
dan representasi tersembunyinya $\mathbf{H}$ yang juga direpresentasikan sebagai matriks (mereka adalah tensor dua dimensi dalam kode), di mana $\mathbf{X}$ dan $\mathbf{H}$ memiliki bentuk yang sama.
Pahami ini sejenak.
Sekarang kita membayangkan bahwa tidak hanya input tetapi
juga representasi tersembunyi memiliki struktur spasial.

Misalkan $[\mathbf{X}]_{i, j}$ dan $[\mathbf{H}]_{i, j}$ masing-masing adalah piksel
di lokasi $(i,j)$
dalam gambar input dan representasi tersembunyi.
Akibatnya, agar setiap unit tersembunyi
menerima input dari setiap piksel input,
kita akan beralih dari menggunakan matriks bobot
(seperti yang kita lakukan sebelumnya pada MLP)
ke representasi parameter kita
sebagai tensor bobot ordo keempat $\mathsf{W}$.
Misalkan $\mathbf{U}$ mengandung bias,
kita dapat mengekspresikan lapisan fully connected secara formal sebagai

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

Peralihan dari $\mathsf{W}$ ke $\mathsf{V}$ adalah kosmetik saja untuk saat ini
karena ada korespondensi satu-ke-satu
antara koefisien dalam kedua tensor ordo keempat.
Kita cukup mengindeks ulang subscripts $(k, l)$
sehingga $k = i+a$ dan $l = j+b$.
Dengan kata lain, kita menetapkan $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
Indeks $a$ dan $b$ mencakup offset positif dan negatif,
meliputi seluruh gambar.
Untuk lokasi ($i$, $j$) dalam representasi tersembunyi $[\mathbf{H}]_{i, j}$,
kita menghitung nilainya dengan menjumlahkan piksel di $x$,
yang berpusat di sekitar $(i, j)$ dan ditimbang oleh $[\mathsf{V}]_{i, j, a, b}$. Sebelum kita melanjutkan, mari kita pertimbangkan total jumlah parameter yang dibutuhkan untuk *satu* lapisan dalam parametrisasi ini: gambar $1000 \times 1000$ (1 megapiksel) dipetakan ke representasi tersembunyi $1000 \times 1000$. Ini memerlukan $10^{12}$ parameter, jauh melebihi kemampuan komputer saat ini.

### Translation Invariance

Sekarang mari kita gunakan prinsip pertama
yang sudah kita tetapkan di atas: translation invariance :cite:`Zhang.ea.1988`.
Ini menyiratkan bahwa pergeseran dalam input $\mathbf{X}$
hanya akan menyebabkan pergeseran dalam representasi tersembunyi $\mathbf{H}$.
Ini hanya mungkin jika $\mathsf{V}$ dan $\mathbf{U}$ sebenarnya tidak bergantung pada $(i, j)$. Dengan demikian,
kita memiliki $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ dan $\mathbf{U}$ adalah konstanta, misalnya $u$.
Akibatnya, kita dapat menyederhanakan definisi $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


Ini adalah sebuah *konvolusi*!
Kita secara efektif memberi bobot pada piksel di $(i+a, j+b)$
di sekitar lokasi $(i, j)$ dengan koefisien $[\mathbf{V}]_{a, b}$
untuk mendapatkan nilai $[\mathbf{H}]_{i, j}$.
Perhatikan bahwa $[\mathbf{V}]_{a, b}$ membutuhkan jauh lebih sedikit koefisien daripada $[\mathsf{V}]_{i, j, a, b}$ karena bobot ini
tidak lagi bergantung pada lokasi dalam gambar. Akibatnya, jumlah parameter yang dibutuhkan tidak lagi $10^{12}$ tetapi lebih masuk akal $4 \times 10^6$: kita masih memiliki ketergantungan pada $a, b \in (-1000, 1000)$. Singkatnya, kita telah membuat kemajuan yang signifikan. Time-delay neural networks (TDNNs) adalah beberapa contoh pertama yang memanfaatkan ide ini :cite:`Waibel.Hanazawa.Hinton.ea.1989`.


### Locality

Sekarang mari kita gunakan prinsip kedua: locality.
Seperti yang dijelaskan sebelumnya, kita percaya bahwa kita tidak perlu melihat terlalu jauh dari lokasi $(i, j)$ untuk memperoleh informasi yang relevan dalam menilai apa yang terjadi di $[\mathbf{H}]_{i, j}$.
Ini berarti bahwa di luar jangkauan tertentu $|a|> \Delta$ atau $|b| > \Delta$,
kita harus menetapkan $[\mathbf{V}]_{a, b} = 0$.
Dengan demikian, kita dapat menulis ulang $[\mathbf{H}]_{i, j}$ sebagai

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Ini mengurangi jumlah parameter dari $4 \times 10^6$ menjadi $4 \Delta^2$, di mana $\Delta$ biasanya lebih kecil dari $10$. Dengan demikian, kita mengurangi jumlah parameter sebesar empat kali lipat lagi. Perhatikan bahwa :eqref:`eq_conv-layer`, adalah apa yang disebut sebagai *lapisan konvolusi*.
*Convolutional neural networks* (CNNs)
adalah keluarga khusus dari jaringan saraf yang mengandung lapisan konvolusi.
Dalam komunitas riset deep learning,
$\mathbf{V}$ disebut sebagai *kernel konvolusi*,
sebuah *filter*, atau hanya sebagai *bobot* lapisan yang merupakan parameter yang dapat dipelajari.

Sementara sebelumnya, kita mungkin memerlukan miliaran parameter
untuk mewakili hanya satu lapisan dalam jaringan pemrosesan gambar,
kita sekarang biasanya hanya memerlukan beberapa ratus parameter, tanpa
mengubah dimensi input atau representasi tersembunyi.
Harga yang dibayar untuk pengurangan drastis dalam parameter ini
adalah bahwa fitur kita sekarang menjadi translation invariant
dan lapisan kita hanya dapat memasukkan informasi lokal
ketika menentukan nilai setiap aktivasi tersembunyi.
Semua pembelajaran bergantung pada penerapan bias induktif.
Ketika bias tersebut sesuai dengan kenyataan,
kita mendapatkan model yang efisien dalam penggunaan sampel
yang dapat digeneralisasi dengan baik ke data yang belum pernah dilihat.
Namun, tentu saja, jika bias tersebut tidak sesuai dengan kenyataan,
misalnya jika gambar ternyata tidak translation invariant,
model kita mungkin kesulitan untuk menyesuaikan bahkan dengan data pelatihan.

Pengurangan dramatis dalam parameter ini membawa kita ke kriteria terakhir,
yaitu bahwa lapisan yang lebih dalam harus mewakili aspek yang lebih besar dan lebih kompleks dari sebuah gambar. Hal ini dapat dicapai dengan menggabungkan lapisan nonlinearitas dan lapisan konvolusi secara berulang kali.


## Konvolusi

Mari kita tinjau secara singkat mengapa :eqref:`eq_conv-layer` disebut sebagai konvolusi.
Dalam matematika, *konvolusi* antara dua fungsi :cite:`Rudin.1973`,
misalnya $f, g: \mathbb{R}^d \to \mathbb{R}$, didefinisikan sebagai

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Artinya, kita mengukur seberapa besar overlap antara $f$ dan $g$
ketika salah satu fungsi di "balik" dan digeser oleh $\mathbf{x}$.
Setiap kali kita memiliki objek diskret, integral tersebut berubah menjadi penjumlahan.
Sebagai contoh, untuk vektor dari
himpunan vektor berdimensi tak hingga yang dapat dijumlahkan dengan indeks yang menjalankan $\mathbb{Z}$, kita memperoleh definisi berikut:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

Untuk tensor dua dimensi, kita memiliki penjumlahan yang sesuai
dengan indeks $(a, b)$ untuk $f$ dan $(i-a, j-b)$ untuk $g$, masing-masing:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Ini tampak mirip dengan :eqref:`eq_conv-layer`, dengan satu perbedaan utama.
Alih-alih menggunakan $(i+a, j+b)$, kita menggunakan perbedaan di antara keduanya.
Namun, perlu dicatat bahwa perbedaan ini lebih bersifat kosmetik
karena kita selalu bisa menyamakan notasi antara
:eqref:`eq_conv-layer` dan :eqref:`eq_2d-conv-discrete`.
Definisi awal kita dalam :eqref:`eq_conv-layer` lebih tepat menggambarkan sebuah *cross-correlation*.
Kita akan kembali ke hal ini pada bagian berikutnya.



## Channels
:label:`subsec_why-conv-channels`

Kembali ke detektor Waldo kita, mari kita lihat bagaimana tampilannya.
Lapisan konvolusi mengambil jendela dengan ukuran tertentu
dan memberi bobot pada intensitas sesuai dengan filter $\mathsf{V}$, seperti yang ditunjukkan pada :numref:`fig_waldo_mask`.
Tujuan kita adalah melatih model sedemikian rupa sehingga
di mana pun "waldo-ness" (keberadaan Waldo) paling tinggi,
kita akan menemukan puncak pada representasi lapisan tersembunyi.

![Mendeteksi Waldo (gambar milik William Murphy (Infomatique)).](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

Namun, ada satu masalah dengan pendekatan ini.
Sejauh ini, kita dengan santai mengabaikan bahwa gambar terdiri dari
tiga kanal: merah, hijau, dan biru.
Singkatnya, gambar bukanlah objek dua dimensi
melainkan tensor ordo ketiga,
yang ditandai oleh tinggi, lebar, dan kanal,
misalnya, dengan bentuk $1024 \times 1024 \times 3$ piksel.
Sementara dua sumbu pertama berkaitan dengan hubungan spasial,
sumbu ketiga dapat dianggap sebagai representasi multidimensi untuk setiap lokasi piksel.
Dengan demikian, kita mengindeks $\mathsf{X}$ sebagai $[\mathsf{X}]_{i, j, k}$.
Filter konvolusi harus menyesuaikan dengan hal ini.
Alih-alih $[\mathbf{V}]_{a,b}$, kita sekarang memiliki $[\mathsf{V}]_{a,b,c}$.

Selain itu, seperti halnya input kita berupa tensor ordo ketiga,
ternyata ide yang bagus untuk merumuskan
representasi tersembunyi kita juga sebagai tensor ordo ketiga $\mathsf{H}$.
Dengan kata lain, daripada hanya memiliki satu representasi tersembunyi
yang sesuai dengan setiap lokasi spasial,
kita menginginkan seluruh vektor representasi tersembunyi
yang sesuai dengan setiap lokasi spasial.
Kita dapat membayangkan bahwa representasi tersembunyi ini terdiri dari
beberapa grid dua dimensi yang ditumpuk satu sama lain.
Seperti pada input, ini kadang disebut *channels*.
Mereka juga kadang disebut *feature maps*,
karena masing-masing menyediakan satu set spasialisasi
fitur yang dipelajari untuk lapisan berikutnya.
Secara intuitif, Anda mungkin membayangkan bahwa pada lapisan yang lebih rendah dekat dengan input,
beberapa kanal bisa menjadi spesialis untuk mengenali tepi sedangkan
yang lain bisa mengenali tekstur.

Untuk mendukung beberapa kanal baik pada input ($\mathsf{X}$) maupun representasi tersembunyi ($\mathsf{H}$),
kita dapat menambahkan koordinat keempat ke $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$.
Dengan menggabungkan semuanya kita memiliki:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

di mana $d$ mengindeks kanal output pada representasi tersembunyi $\mathsf{H}$. Lapisan konvolusi berikutnya akan menerima tensor ordo ketiga, $\mathsf{H}$, sebagai input.
Kita mengambil :eqref:`eq_conv-layer-channels`,
karena sifat umum yang dimilikinya, sebagai
definisi lapisan konvolusi untuk beberapa kanal, di mana $\mathsf{V}$ adalah kernel atau filter dari lapisan tersebut.

Masih ada banyak operasi yang perlu kita bahas.
Misalnya, kita perlu mencari cara untuk menggabungkan semua representasi tersembunyi
ke dalam satu output, misalnya, apakah ada Waldo *di mana pun* dalam gambar.
Kita juga perlu memutuskan cara menghitung semuanya secara efisien,
cara menggabungkan beberapa lapisan,
fungsi aktivasi yang tepat,
dan cara membuat pilihan desain yang masuk akal
untuk menghasilkan jaringan yang efektif dalam praktik.
Kita akan membahas masalah ini di sisa bab ini.


## Ringkasan dan Diskusi

Dalam bagian ini, kita menurunkan struktur convolutional neural networks (CNN) dari prinsip dasar. Meskipun tidak jelas apakah ini adalah jalur yang diambil dalam penemuan CNN, memuaskan untuk mengetahui bahwa CNN adalah pilihan yang *tepat* ketika menerapkan prinsip yang masuk akal tentang cara kerja algoritma pemrosesan gambar dan computer vision, setidaknya pada level yang lebih rendah. Secara khusus, translation invariance dalam gambar menyiratkan bahwa semua bagian gambar akan diperlakukan dengan cara yang sama. Locality berarti bahwa hanya lingkungan kecil dari piksel yang akan digunakan untuk menghitung representasi tersembunyi yang sesuai. Beberapa referensi paling awal mengenai CNN adalah dalam bentuk Neocognitron :cite:`Fukushima.1982`.

Prinsip kedua yang kita temui dalam penalaran kita adalah bagaimana mengurangi jumlah parameter dalam kelas fungsi tanpa membatasi kekuatan ekspresifnya, setidaknya, selama asumsi tertentu pada model berlaku. Kita melihat pengurangan kompleksitas yang dramatis sebagai hasil dari pembatasan ini, mengubah masalah yang tidak layak secara komputasi dan statistik menjadi model yang dapat diselesaikan.

Penambahan kanal memungkinkan kita untuk mengembalikan sebagian kompleksitas yang hilang akibat pembatasan yang diberlakukan pada kernel konvolusi oleh locality dan translation invariance. Perlu dicatat bahwa menambahkan kanal selain merah, hijau, dan biru adalah hal yang wajar. Banyak gambar satelit, terutama untuk pertanian dan meteorologi, memiliki puluhan hingga ratusan kanal, menghasilkan gambar hiperspektral. Mereka melaporkan data pada berbagai panjang gelombang. Selanjutnya, kita akan melihat cara menggunakan konvolusi secara efektif untuk memanipulasi dimensi gambar yang mereka operasikan, cara berpindah dari representasi berbasis lokasi ke representasi berbasis kanal, dan cara menangani sejumlah besar kategori secara efisien.

## Latihan

1. Asumsikan bahwa ukuran kernel konvolusi adalah $\Delta = 0$.
   Tunjukkan bahwa dalam kasus ini kernel konvolusi
   menerapkan MLP secara independen untuk setiap set kanal. Ini mengarah ke arsitektur Network in Network :cite:`Lin.Chen.Yan.2013`.
1. Data audio sering diwakili sebagai urutan satu dimensi.
    1. Kapan Anda mungkin ingin menerapkan locality dan translation invariance pada audio?
    1. Turunkan operasi konvolusi untuk audio.
    1. Bisakah Anda memperlakukan audio menggunakan alat yang sama dengan computer vision? Petunjuk: gunakan spektrogram.
1. Mengapa translation invariance mungkin bukan ide yang baik? Berikan contoh.
1. Menurut Anda, apakah lapisan konvolusi juga dapat diterapkan untuk data teks?
   Masalah apa yang mungkin Anda temui dengan bahasa?
1. Apa yang terjadi dengan konvolusi ketika sebuah objek berada di batas gambar?
1. Buktikan bahwa konvolusi bersifat simetris, yaitu, $f * g = g * f$.

[Diskusi](https://discuss.d2l.ai/t/64)
