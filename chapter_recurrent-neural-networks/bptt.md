# Backpropagation Through Time
:label:`sec_bptt`

Jika Anda telah menyelesaikan latihan di :numref:`sec_rnn-scratch`,
Anda akan melihat bahwa kliping gradien sangat penting 
untuk mencegah gradien yang sangat besar sesekali 
yang dapat mengacaukan pelatihan.
Kami menyinggung bahwa gradien yang meledak 
berasal dari proses backpropagation pada urutan yang panjang.
Sebelum memperkenalkan berbagai arsitektur RNN modern,
mari kita melihat lebih dekat bagaimana *backpropagation*
bekerja dalam model urutan secara detail matematis.
Diharapkan, pembahasan ini akan membawa kejelasan 
terhadap konsep *vanishing* dan *exploding* gradients.
Jika Anda mengingat pembahasan kita tentang forward dan backward 
propagation melalui computational graph
saat kita memperkenalkan MLP di :numref:`sec_backprop`,
maka forward propagation dalam RNN 
seharusnya relatif sederhana.
Penerapan backpropagation dalam RNN 
disebut *backpropagation through time* :cite:`Werbos.1990`.
Proses ini membutuhkan kita untuk memperluas (atau me-*unroll*) 
computational graph dari sebuah RNN
satu langkah waktu setiap kali.
RNN yang telah di-unroll pada dasarnya 
merupakan jaringan neural feedforward 
dengan properti khusus 
bahwa parameter yang sama 
diulang sepanjang jaringan yang di-unroll,
muncul di setiap langkah waktu.
Kemudian, seperti pada jaringan neural feedforward lainnya,
kita dapat menerapkan chain rule, 
melakukan backpropagation gradien melalui jaringan yang di-unroll.
Gradien terhadap setiap parameter
harus dijumlahkan di seluruh tempat 
di mana parameter tersebut muncul dalam jaringan yang di-unroll.
Penanganan pengikatan bobot seperti itu seharusnya sudah tidak asing 
dari bab kita tentang convolutional neural networks.


Masalah muncul karena urutan 
dapat sangat panjang.
Tidak jarang kita bekerja dengan urutan teks 
yang terdiri dari lebih dari seribu token. 
Perhatikan bahwa hal ini menimbulkan masalah baik dari sudut pandang 
komputasi (terlalu banyak memori)
maupun optimasi (ketidakstabilan numerik).
Input dari langkah pertama harus melalui 
lebih dari 1000 kali perkalian matriks sebelum sampai pada output, 
dan diperlukan 1000 kali perkalian matriks lagi 
untuk menghitung gradien. 
Sekarang kita akan menganalisis apa yang bisa salah dan 
bagaimana cara mengatasinya dalam praktik.



## Analisis Gradien pada RNN
:label:`subsec_bptt_analysis`

Kita mulai dengan model sederhana tentang cara kerja sebuah RNN.
Model ini mengabaikan detail spesifik 
dari status tersembunyi (*hidden state*) dan bagaimana status tersebut diperbarui.
Notasi matematis di sini
tidak secara eksplisit membedakan
skalar, vektor, dan matriks.
Kita hanya mencoba mengembangkan intuisi dasar.
Dalam model yang disederhanakan ini,
kita menyebut $h_t$ sebagai status tersembunyi (*hidden state*),
$x_t$ sebagai input, dan $o_t$ sebagai output
pada langkah waktu $t$.
Ingat pembahasan kita di
:numref:`subsec_rnn_w_hidden_states`
bahwa input dan status tersembunyi
dapat digabungkan sebelum dikalikan 
dengan satu variabel bobot pada lapisan tersembunyi.
Oleh karena itu, kita menggunakan $w_\textrm{h}$ dan $w_\textrm{o}$ untuk menunjukkan bobot 
dari lapisan tersembunyi dan lapisan output, masing-masing.
Sebagai hasilnya, status tersembunyi dan output 
pada setiap langkah waktu adalah

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

di mana $f$ dan $g$ adalah transformasi
dari lapisan tersembunyi dan lapisan output, masing-masing.
Dengan demikian, kita memiliki rantai nilai 
$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ 
yang saling bergantung melalui komputasi berulang.
Forward propagation cukup sederhana.
Yang kita butuhkan adalah melintasi triplet $(x_t, h_t, o_t)$ satu langkah waktu pada satu waktu.
Perbedaan antara output $o_t$ dan target yang diinginkan $y_t$ 
kemudian dievaluasi oleh fungsi objektif 
melalui semua $T$ langkah waktu sebagai

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



Untuk backpropagation, masalah menjadi sedikit lebih rumit, 
terutama ketika kita menghitung gradien 
terhadap parameter $w_\textrm{h}$ dari fungsi objektif $L$. 
Secara spesifik, dengan aturan rantai,

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

Faktor pertama dan kedua dari
produk di :eqref:`eq_bptt_partial_L_wh`
mudah dihitung.
Faktor ketiga $\partial h_t/\partial w_\textrm{h}$ adalah bagian yang sulit, 
karena kita perlu secara berulang menghitung efek dari parameter $w_\textrm{h}$ pada $h_t$.
Menurut perhitungan berulang
di :eqref:`eq_bptt_ht_ot`,
$h_t$ bergantung pada $h_{t-1}$ dan $w_\textrm{h}$,
di mana perhitungan dari $h_{t-1}$
juga bergantung pada $w_\textrm{h}$.
Dengan demikian, evaluasi turunan total dari $h_t$ 
terhadap $w_\textrm{h}$ menggunakan aturan rantai menghasilkan

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


Untuk menurunkan gradien di atas, anggap kita memiliki 
tiga urutan $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ 
yang memenuhi $a_{0}=0$ dan $a_{t}=b_{t}+c_{t}a_{t-1}$ untuk $t=1, 2,\ldots$.
Maka untuk $t\geq 1$, mudah ditunjukkan bahwa

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

Dengan mengganti $a_t$, $b_t$, dan $c_t$ sesuai dengan

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},\end{aligned}$$

perhitungan gradien dalam :eqref:`eq_bptt_partial_ht_wh_recur` memenuhi
$a_{t}=b_{t}+c_{t}a_{t-1}$.
Dengan demikian, menurut :eqref:`eq_bptt_at`, 
kita dapat menghilangkan perhitungan berulang 
dalam :eqref:`eq_bptt_partial_ht_wh_recur` dengan

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

Meskipun kita dapat menggunakan aturan rantai untuk menghitung $\partial h_t/\partial w_\textrm{h}$ secara rekursif, 
rantai ini bisa menjadi sangat panjang jika $t$ besar.
Mari kita bahas beberapa strategi untuk mengatasi masalah ini.


### Perhitungan Penuh ###

Salah satu ide adalah untuk menghitung jumlah penuh dalam :eqref:`eq_bptt_partial_ht_wh_gen`.
Namun, ini sangat lambat dan gradien bisa meledak,
karena perubahan kecil pada kondisi awal
dapat sangat memengaruhi hasil akhirnya.
Artinya, kita bisa melihat hal yang mirip dengan efek kupu-kupu,
di mana perubahan minimal pada kondisi awal 
menyebabkan perubahan hasil yang tidak proporsional.
Ini umumnya tidak diinginkan.
Bagaimanapun, kita mencari estimator yang tangguh dan mampu melakukan generalisasi dengan baik. 
Oleh karena itu, strategi ini hampir tidak pernah digunakan dalam praktik.

### Memotong Langkah Waktu###

Sebagai alternatif,
kita dapat memotong jumlah dalam
:eqref:`eq_bptt_partial_ht_wh_gen`
setelah $\tau$ langkah. 
Ini adalah apa yang telah kita diskusikan sejauh ini. 
Ini mengarah pada sebuah *pendekatan* terhadap gradien sebenarnya,
dengan menghentikan jumlah pada $\partial h_{t-\tau}/\partial w_\textrm{h}$. 
Dalam praktiknya, pendekatan ini bekerja cukup baik. 
Pendekatan ini biasa disebut sebagai *truncated backpropagation through time* :cite:`Jaeger.2002`.
Salah satu konsekuensi dari pendekatan ini adalah model 
fokus terutama pada pengaruh jangka pendek 
daripada konsekuensi jangka panjang. 
Ini sebenarnya *diharapkan*, karena pendekatan ini mengarahkan estimasi 
ke model yang lebih sederhana dan lebih stabil.


### Pemotongan Acak ###

Terakhir, kita dapat mengganti $\partial h_t/\partial w_\textrm{h}$
dengan variabel acak yang benar secara ekspektasi 
tetapi memotong urutan.
Ini dicapai dengan menggunakan urutan $\xi_t$
dengan nilai yang ditentukan sebelumnya $0 \leq \pi_t \leq 1$,
di mana $P(\xi_t = 0) = 1-\pi_t$ dan 
$P(\xi_t = \pi_t^{-1}) = \pi_t$, sehingga $E[\xi_t] = 1$.
Kita menggunakan ini untuk mengganti gradien
$\partial h_t/\partial w_\textrm{h}$
dalam :eqref:`eq_bptt_partial_ht_wh_recur`
dengan

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$

Dari definisi $\xi_t$, kita mendapatkan bahwa $E[z_t] = \partial h_t/\partial w_\textrm{h}$.
Setiap kali $\xi_t = 0$, perhitungan berulang 
berakhir pada langkah waktu $t$ tersebut.
Pendekatan ini menghasilkan jumlah berbobot dari urutan dengan panjang yang bervariasi,
di mana urutan panjang jarang terjadi tetapi diberi bobot yang sesuai. 
Ide ini diusulkan oleh 
:citet:`Tallec.Ollivier.2017`.


### Membandingkan Strategi

![Membandingkan strategi untuk menghitung gradien pada RNN. Dari atas ke bawah: pemotongan acak, pemotongan reguler, dan perhitungan penuh.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt` menggambarkan tiga strategi 
saat menganalisis beberapa karakter pertama dari *The Time Machine* 
menggunakan backpropagation through time untuk RNN:

* Baris pertama adalah pemotongan acak yang membagi teks menjadi segmen dengan panjang yang bervariasi.
* Baris kedua adalah pemotongan reguler yang membagi teks menjadi subsekuensi dengan panjang yang sama. Inilah yang telah kita lakukan dalam eksperimen RNN.
* Baris ketiga adalah backpropagation through time penuh yang mengarah pada ekspresi yang tidak memungkinkan secara komputasional.


Sayangnya, meskipun secara teori menarik, 
pemotongan acak tidak bekerja 
jauh lebih baik daripada pemotongan reguler, 
kemungkinan besar karena beberapa faktor.
Pertama, efek dari sebuah observasi 
setelah beberapa langkah backpropagation ke masa lalu cukup memadai 
untuk menangkap ketergantungan dalam praktik. 
Kedua, varians yang meningkat mengimbangi fakta 
bahwa gradien lebih akurat dengan lebih banyak langkah. 
Ketiga, kita sebenarnya *ingin* model yang hanya 
memiliki jangkauan interaksi yang pendek. 
Oleh karena itu, backpropagation through time dengan pemotongan reguler 
memiliki efek regularisasi ringan yang bisa jadi diinginkan.

## Backpropagation Through Time secara Detail

Setelah membahas prinsip umum,
mari kita bahas backpropagation through time secara detail.
Berbeda dengan analisis di :numref:`subsec_bptt_analysis`,
selanjutnya kita akan menunjukkan cara menghitung
gradien dari fungsi objektif
terhadap semua parameter model yang telah diuraikan.
Untuk menyederhanakan, kita akan mempertimbangkan 
RNN tanpa parameter bias,
dengan fungsi aktivasi pada lapisan tersembunyi
menggunakan pemetaan identitas ($\phi(x)=x$).
Untuk langkah waktu $t$, misalkan input contoh tunggal 
dan targetnya adalah $\mathbf{x}_t \in \mathbb{R}^d$ dan $y_t$, masing-masing. 
Status tersembunyi $\mathbf{h}_t \in \mathbb{R}^h$ 
dan output $\mathbf{o}_t \in \mathbb{R}^q$
dihitung sebagai

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$

di mana $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$, dan
$\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$
adalah parameter bobot.
Dilambangkan dengan $l(\mathbf{o}_t, y_t)$
sebagai kehilangan pada langkah waktu $t$. 
Fungsi objektif kita,
kehilangan selama $T$ langkah waktu
dari awal urutan adalah

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


Untuk memvisualisasikan ketergantungan antara
variabel dan parameter model selama perhitungan
RNN,
kita dapat menggambar sebuah computational graph untuk model tersebut,
seperti yang ditunjukkan di :numref:`fig_rnn_bptt`.
Sebagai contoh, perhitungan status tersembunyi pada langkah waktu ke-3,
$\mathbf{h}_3$, bergantung pada parameter model
$\mathbf{W}_\textrm{hx}$ dan $\mathbf{W}_\textrm{hh}$,
status tersembunyi dari langkah waktu sebelumnya $\mathbf{h}_2$,
dan input dari langkah waktu saat ini $\mathbf{x}_3$.


![Computational graph yang menunjukkan ketergantungan untuk model RNN dengan tiga langkah waktu. Kotak mewakili variabel (tidak diarsir) atau parameter (diarsir) dan lingkaran mewakili operator.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

Seperti yang baru saja disebutkan, parameter model dalam :numref:`fig_rnn_bptt` 
adalah $\mathbf{W}_\textrm{hx}$, $\mathbf{W}_\textrm{hh}$, dan $\mathbf{W}_\textrm{qh}$. 
Umumnya, pelatihan model ini membutuhkan 
perhitungan gradien terhadap parameter-parameter ini:
$\partial L/\partial \mathbf{W}_\textrm{hx}$, $\partial L/\partial \mathbf{W}_\textrm{hh}$, dan $\partial L/\partial \mathbf{W}_\textrm{qh}$.
Berdasarkan ketergantungan dalam :numref:`fig_rnn_bptt`,
kita dapat menelusuri arah yang berlawanan dengan panah
untuk menghitung dan menyimpan gradien secara bergantian.
Untuk mengekspresikan perkalian dari 
matriks, vektor, dan skalar dengan bentuk yang berbeda dalam aturan rantai,
kita tetap menggunakan operator $\textrm{prod}$ 
seperti yang dijelaskan di :numref:`sec_backprop`.


Pertama-tama, diferensiasi fungsi objektif
terhadap output model pada setiap langkah waktu $t$
cukup sederhana:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Sekarang kita dapat menghitung gradien dari objektif 
terhadap parameter $\mathbf{W}_\textrm{qh}$
di lapisan output:
$\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$. 
Berdasarkan :numref:`fig_rnn_bptt`, 
objektif $L$ bergantung pada $\mathbf{W}_\textrm{qh}$ 
melalui $\mathbf{o}_1, \ldots, \mathbf{o}_T$. 
Menggunakan aturan rantai menghasilkan

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

di mana $\partial L/\partial \mathbf{o}_t$
diberikan oleh :eqref:`eq_bptt_partial_L_ot`.

Selanjutnya, seperti yang ditunjukkan dalam :numref:`fig_rnn_bptt`,
pada langkah waktu terakhir $T$,
fungsi objektif
$L$ bergantung pada status tersembunyi $\mathbf{h}_T$ 
hanya melalui $\mathbf{o}_T$.
Oleh karena itu, kita dapat menemukan gradien 
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$
menggunakan aturan rantai:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

Menjadi lebih rumit untuk setiap langkah waktu $t < T$,
di mana fungsi objektif $L$ bergantung pada 
$\mathbf{h}_t$ melalui $\mathbf{h}_{t+1}$ dan $\mathbf{o}_t$.
Menurut aturan rantai,
gradien status tersembunyi
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$
pada setiap langkah waktu $t < T$ dapat dihitung secara rekursif sebagai:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

Untuk analisis, memperluas perhitungan rekursif
untuk setiap langkah waktu $1 \leq t \leq T$ menghasilkan

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

Kita dapat melihat dari :eqref:`eq_bptt_partial_L_ht` 
bahwa contoh linear sederhana ini sudah
memperlihatkan beberapa masalah utama pada model urutan panjang:
terdapat pangkat besar dari $\mathbf{W}_\textrm{hh}^\top$.
Dalam hal ini, nilai eigen yang lebih kecil dari 1 akan lenyap,
sedangkan nilai eigen yang lebih besar dari 1 akan menyebar.
Hal ini tidak stabil secara numerik,
yang bermanifestasi dalam bentuk gradien yang menghilang 
dan gradien yang meledak.
Salah satu cara untuk mengatasi hal ini adalah dengan memotong langkah waktu
pada ukuran yang memungkinkan secara komputasi 
seperti yang dibahas di :numref:`subsec_bptt_analysis`. 
Dalam praktiknya, pemotongan ini juga dapat dilakukan 
dengan melepaskan gradien setelah sejumlah langkah waktu tertentu.
Nantinya, kita akan melihat bagaimana model urutan yang lebih canggih 
seperti long short-term memory dapat mengurangi masalah ini lebih jauh. 

Terakhir, :numref:`fig_rnn_bptt` menunjukkan 
bahwa fungsi objektif $L$ 
bergantung pada parameter model $\mathbf{W}_\textrm{hx}$ dan $\mathbf{W}_\textrm{hh}$
di lapisan tersembunyi melalui status tersembunyi
$\mathbf{h}_1, \ldots, \mathbf{h}_T$.
Untuk menghitung gradien terhadap parameter tersebut
$\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$ dan $\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$,
kita menerapkan aturan rantai sebagai berikut

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

di mana $\partial L/\partial \mathbf{h}_t$
yang dihitung secara rekursif melalui
:eqref:`eq_bptt_partial_L_hT_final_step`
dan :eqref:`eq_bptt_partial_L_ht_recur`
adalah kuantitas kunci yang memengaruhi stabilitas numerik.



Karena backpropagation through time adalah penerapan backpropagation dalam RNN,
seperti yang telah dijelaskan di :numref:`sec_backprop`,
pelatihan RNN menggabungkan forward propagation dengan
backpropagation through time.
Selain itu, backpropagation through time
menghitung dan menyimpan gradien di atas secara bergantian.
Secara khusus, nilai-nilai antara yang disimpan
digunakan kembali untuk menghindari perhitungan yang berulang,
seperti menyimpan $\partial L/\partial \mathbf{h}_t$
untuk digunakan dalam perhitungan $\partial L / \partial \mathbf{W}_\textrm{hx}$ 
dan $\partial L / \partial \mathbf{W}_\textrm{hh}$.


## Ringkasan

*Backpropagation through time* hanyalah penerapan *backpropagation* pada model urutan dengan *hidden state*.
Pemotongan, seperti pemotongan reguler atau acak, diperlukan untuk kenyamanan komputasi dan stabilitas numerik.
Pangkat matriks yang tinggi dapat menyebabkan nilai eigen menyebar atau menghilang, yang kemudian bermanifestasi sebagai gradien yang meledak atau menghilang.
Untuk efisiensi komputasi, nilai perantara disimpan selama *backpropagation through time*.


## Latihan

1. Asumsikan kita memiliki matriks simetris $\mathbf{M} \in \mathbb{R}^{n \times n}$ dengan nilai eigen $\lambda_i$ dan vektor eigen terkait $\mathbf{v}_i$ ($i = 1, \ldots, n$). Tanpa mengurangi keumuman, anggap nilai eigen tersebut diurutkan dalam urutan $|\lambda_i| \geq |\lambda_{i+1}|$. 
   1. Tunjukkan bahwa $\mathbf{M}^k$ memiliki nilai eigen $\lambda_i^k$.
   1. Buktikan bahwa untuk vektor acak $\mathbf{x} \in \mathbb{R}^n$, dengan probabilitas tinggi, $\mathbf{M}^k \mathbf{x}$ akan sangat selaras dengan vektor eigen $\mathbf{v}_1$ dari $\mathbf{M}$. Formalisasikan pernyataan ini.
   1. Apa arti hasil di atas untuk gradien dalam RNN?
2. Selain *gradient clipping*, dapatkah Anda memikirkan metode lain untuk mengatasi ledakan gradien pada *recurrent neural networks*?

[Diskusi](https://discuss.d2l.ai/t/334)
