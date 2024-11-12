# Pencarian Beam
:label:`sec_beam-search`

Pada :numref:`sec_seq2seq`, kita telah memperkenalkan arsitektur encoder-decoder, dan teknik standar untuk melatihnya secara end-to-end. Namun, ketika sampai pada prediksi pada waktu uji (test-time), kita hanya menyebutkan strategi *greedy*, di mana pada setiap langkah waktu kita memilih token yang memiliki probabilitas tertinggi untuk muncul berikutnya, hingga, pada langkah tertentu, kita menemukan bahwa kita telah memprediksi token khusus yang menandakan akhir urutan, yaitu "&lt;eos&gt;". Pada bagian ini, kita akan mulai dengan memformalkan strategi *greedy search* ini dan mengidentifikasi beberapa masalah yang biasanya dihadapi oleh praktisi. Selanjutnya, kita akan membandingkan strategi ini dengan dua alternatif lainnya: *exhaustive search* (untuk ilustrasi tapi tidak praktis) dan *beam search* (metode standar yang digunakan dalam praktik).

Mari kita mulai dengan menetapkan notasi matematika kita, dengan meminjam konvensi dari :numref:`sec_seq2seq`. Pada setiap langkah waktu $t'$, decoder menghasilkan prediksi yang mewakili probabilitas setiap token dalam kosakata (vocabulary) untuk muncul berikutnya dalam urutan (kemungkinan nilai dari $y_{t'+1}$), yang dikondisikan pada token-token sebelumnya $y_1, \ldots, y_{t'}$ dan variabel konteks $\mathbf{c}$, yang dihasilkan oleh encoder untuk merepresentasikan urutan input. Untuk menghitung biaya komputasi, kita nyatakan $\mathcal{Y}$ sebagai kosakata keluaran (termasuk token khusus "&lt;eos&gt;"). Kita juga tentukan jumlah maksimum token dalam urutan keluaran sebagai $T'$. Tujuan kita adalah mencari keluaran ideal dari semua urutan keluaran yang mungkin berjumlah $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$. Perlu dicatat bahwa angka ini sedikit melebih-lebihkan jumlah keluaran yang berbeda karena tidak ada token yang muncul setelah token "&lt;eos&gt;". Namun, untuk tujuan kita, angka ini kurang lebih menangkap ukuran ruang pencarian.

## Pencarian Greedy

Pertimbangkan strategi *greedy search* sederhana dari :numref:`sec_seq2seq`. Di sini, pada setiap langkah waktu $t'$, kita cukup memilih token dengan probabilitas kondisional tertinggi dari $\mathcal{Y}$, yaitu

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$

Setelah model kita mengeluarkan "&lt;eos&gt;" (atau kita mencapai panjang maksimum $T'$) urutan keluaran dianggap selesai.

Strategi ini mungkin tampak masuk akal, dan faktanya tidak terlalu buruk! Mengingat betapa rendahnya biaya komputasi yang dibutuhkan, sulit untuk menemukan hasil yang lebih baik dibandingkan dengan sumber daya yang digunakan. Namun, jika kita mengesampingkan efisiensi sejenak, mungkin lebih masuk akal untuk mencari *urutan yang paling mungkin*, bukan urutan dari *token yang paling mungkin* yang dipilih secara greedy. Ternyata, kedua objek ini bisa sangat berbeda. Urutan yang paling mungkin adalah urutan yang memaksimalkan ekspresi $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$. Dalam contoh penerjemahan mesin kita, jika decoder benar-benar berhasil menemukan probabilitas dari proses generatif yang mendasari, maka ini akan memberi kita terjemahan yang paling mungkin. Sayangnya, tidak ada jaminan bahwa pencarian greedy akan memberikan kita urutan ini.

Mari kita ilustrasikan dengan sebuah contoh. Misalkan ada empat token "A", "B", "C", dan "&lt;eos&gt;" dalam kamus keluaran. Pada :numref:`fig_s2s-prob1`, keempat angka di bawah setiap langkah waktu mewakili probabilitas kondisional untuk menghasilkan "A", "B", "C", dan "&lt;eos&gt;" pada langkah waktu tersebut.

![Pada setiap langkah waktu, pencarian greedy memilih token dengan probabilitas kondisional tertinggi.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Pada setiap langkah waktu, pencarian greedy memilih token dengan probabilitas kondisional tertinggi. Oleh karena itu, urutan keluaran yang diprediksi adalah "A", "B", "C", dan "&lt;eos&gt;" (:numref:`fig_s2s-prob1`). Probabilitas kondisional dari urutan keluaran ini adalah $0.5\times0.4\times0.4\times0.6 = 0.048$.



Selanjutnya, mari kita lihat contoh lain pada :numref:`fig_s2s-prob2`. Berbeda dengan :numref:`fig_s2s-prob1`, pada langkah waktu 2 kita memilih token "C", yang memiliki probabilitas kondisional tertinggi *kedua*.

![Empat angka di bawah setiap langkah waktu mewakili probabilitas kondisional untuk menghasilkan "A", "B", "C", dan "&lt;eos&gt;" pada langkah waktu tersebut. Pada langkah waktu 2, token "C", yang memiliki probabilitas kondisional tertinggi kedua, dipilih.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

Karena subsekuens keluaran pada langkah waktu 1 dan 2, yang menjadi dasar langkah waktu 3, telah berubah dari "A" dan "B" pada :numref:`fig_s2s-prob1` menjadi "A" dan "C" pada :numref:`fig_s2s-prob2`, probabilitas kondisional dari setiap token pada langkah waktu 3 juga telah berubah pada :numref:`fig_s2s-prob2`. Misalkan kita memilih token "B" pada langkah waktu 3. Sekarang langkah waktu 4 bergantung pada subsekuens keluaran pada tiga langkah waktu pertama "A", "C", dan "B", yang telah berubah dari "A", "B", dan "C" pada :numref:`fig_s2s-prob1`. Oleh karena itu, probabilitas kondisional untuk menghasilkan setiap token pada langkah waktu 4 dalam :numref:`fig_s2s-prob2` juga berbeda dengan yang ada pada :numref:`fig_s2s-prob1`. Akibatnya, probabilitas kondisional dari urutan keluaran "A", "C", "B", dan "&lt;eos&gt;" pada :numref:`fig_s2s-prob2` adalah $0.5\times0.3\times0.6\times0.6=0.054$, yang lebih besar dari probabilitas pencarian greedy pada :numref:`fig_s2s-prob1`. Dalam contoh ini, urutan keluaran "A", "B", "C", dan "&lt;eos&gt;" yang diperoleh dari pencarian greedy tidak optimal.




## Pencarian Ekshaustif

Jika tujuannya adalah untuk mendapatkan urutan yang paling mungkin,
kita bisa mempertimbangkan menggunakan *pencarian ekshaustif*: 
yaitu dengan menelusuri semua kemungkinan urutan keluaran 
beserta probabilitas kondisionalnya,
dan kemudian menghasilkan keluaran dengan skor probabilitas tertinggi.


Meskipun ini akan memberikan hasil yang kita inginkan,
pendekatan ini akan menghasilkan biaya komputasi yang sangat besar 
sebesar $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$,
yang tumbuh secara eksponensial terhadap panjang urutan dan memiliki basis yang sangat besar, yakni ukuran kosakata.
Misalnya, ketika $|\mathcal{Y}|=10000$ dan $T'=10$, 
yang merupakan angka-angka kecil jika dibandingkan dengan yang ada dalam aplikasi nyata, kita perlu mengevaluasi $10000^{10} = 10^{40}$ urutan, yang sudah melebihi kemampuan komputer saat ini.
Di sisi lain, biaya komputasi dari pencarian greedy adalah 
$\mathcal{O}(\left|\mathcal{Y}\right|T')$: 
sangat murah, tetapi jauh dari optimal.
Misalnya, ketika $|\mathcal{Y}|=10000$ dan $T'=10$, 
kita hanya perlu mengevaluasi $10000\times10=10^5$ urutan.


## Pencarian Beam (Beam Search)

Anda dapat melihat strategi decoding urutan sebagai terletak pada spektrum,
dengan *beam search* menjadi kompromi 
antara efisiensi pencarian greedy
dan optimalitas pencarian ekshaustif.
Versi paling sederhana dari beam search 
ditandai dengan satu hyperparameter,
yaitu *ukuran beam* ($k$).
Mari kita jelaskan terminologi ini.
Pada langkah waktu 1, kita memilih $k$ token 
dengan probabilitas prediksi tertinggi.
Setiap token akan menjadi token pertama dari 
$k$ kandidat urutan keluaran, masing-masing.
Pada setiap langkah waktu berikutnya, 
berdasarkan $k$ kandidat urutan keluaran
pada langkah waktu sebelumnya,
kita terus memilih $k$ kandidat urutan keluaran 
dengan probabilitas prediksi tertinggi 
dari $k\left|\mathcal{Y}\right|$ pilihan yang mungkin.

![Proses pencarian beam (ukuran beam $=2$; panjang maksimum urutan keluaran $=3$). Kandidat urutan keluaran adalah $\mathit{A}$, $\mathit{C}$, $\mathit{AB}$, $\mathit{CE}$, $\mathit{ABD}$, dan $\mathit{CED}$.](../img/beam-search.svg)
:label:`fig_beam-search`



:numref:`fig_beam-search` menunjukkan
proses beam search melalui sebuah contoh.
Misalkan bahwa kosakata keluaran hanya berisi lima elemen:
$\mathcal{Y} = \{A, B, C, D, E\}$,
di mana salah satunya adalah "&lt;eos&gt;".
Misalkan ukuran beam adalah dua
dan panjang maksimum urutan keluaran adalah tiga.
Pada langkah waktu ke-1,
misalkan token-token dengan probabilitas kondisional tertinggi
$P(y_1 \mid \mathbf{c})$ adalah $A$ dan $C$.
Pada langkah waktu ke-2, untuk semua $y_2 \in \mathcal{Y}$,
kita menghitung

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$

dan memilih dua nilai terbesar di antara sepuluh nilai ini, misalnya
$P(A, B \mid \mathbf{c})$ dan $P(C, E \mid \mathbf{c})$.
Kemudian pada langkah waktu ke-3, untuk semua $y_3 \in \mathcal{Y}$, kita menghitung

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$

dan memilih dua nilai terbesar di antara sepuluh nilai ini, misalnya
$P(A, B, D \mid \mathbf{c})$ dan $P(C, E, D \mid \mathbf{c})$.
Akibatnya, kita mendapatkan enam kandidat urutan keluaran:
(i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; dan (vi) $C$, $E$, $D$.

Pada akhirnya, kita memperoleh set kandidat urutan keluaran akhir
berdasarkan enam urutan ini (misalnya, membuang bagian yang mencakup dan setelah "&lt;eos&gt;").
Kemudian kita memilih urutan keluaran yang memaksimalkan skor berikut:

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c});$$
:eqlabel:`eq_beam-search-score`

di sini $L$ adalah panjang urutan kandidat akhir
dan $\alpha$ biasanya diatur menjadi 0,75.
Karena urutan yang lebih panjang memiliki lebih banyak istilah logaritma
dalam penjumlahan :eqref:`eq_beam-search-score`,
istilah $L^\alpha$ di penyebutnya menghukum
urutan yang panjang.

Biaya komputasi dari beam search adalah $\mathcal{O}(k\left|\mathcal{Y}\right|T')$.
Hasil ini berada di antara pencarian greedy dan pencarian ekshaustif.
Pencarian greedy dapat dianggap sebagai kasus khusus dari beam search
yang muncul ketika ukuran beam ditetapkan menjadi 1.


## Ringkasan

Strategi pencarian urutan mencakup
pencarian greedy, pencarian ekshaustif, dan beam search.
Beam search menyediakan kompromi antara akurasi dan
biaya komputasi melalui pilihan ukuran beam yang fleksibel.


## Latihan

1. Bisakah kita menganggap pencarian ekshaustif sebagai jenis beam search khusus? Mengapa atau mengapa tidak?
2. Terapkan beam search pada masalah penerjemahan mesin di :numref:`sec_seq2seq`. Bagaimana ukuran beam memengaruhi hasil terjemahan dan kecepatan prediksi?
3. Kita menggunakan pemodelan bahasa untuk menghasilkan teks berdasarkan awalan yang disediakan pengguna di :numref:`sec_rnn-scratch`. Jenis strategi pencarian apa yang digunakan? Bisakah Anda memperbaikinya?

[Diskusi](https://discuss.d2l.ai/t/338)

