# Generalisasi
:label:`sec_generalization_basics`

Bayangkan dua mahasiswa yang dengan tekun
mempersiapkan diri untuk ujian akhir mereka.
Persiapan ini biasanya melibatkan
latihan dan pengujian kemampuan mereka
dengan mengerjakan soal-soal ujian dari tahun-tahun sebelumnya.
Namun, melakukan dengan baik pada ujian masa lalu
tidak menjamin mereka akan sukses saat ujian sesungguhnya.
Misalnya, bayangkan seorang mahasiswa, Ellie yang Luar Biasa,
yang persiapannya hanya berfokus
pada menghafal jawaban dari soal-soal ujian
tahun-tahun sebelumnya.
Meskipun Ellie memiliki ingatan yang luar biasa
dan dapat mengingat dengan sempurna jawaban
untuk semua soal yang pernah dia lihat,
dia mungkin akan terdiam ketika dihadapkan
pada soal baru (*yang belum pernah dilihat sebelumnya*).
Sebagai perbandingan, bayangkan mahasiswa lain,
Irene yang Induktif, yang memiliki kemampuan menghafal yang kurang,
tetapi memiliki bakat untuk mengenali pola.
Perhatikan bahwa jika ujian tersebut
benar-benar terdiri dari soal-soal daur ulang,
Ellie akan unggul jauh di atas Irene.
Meskipun prediksi pola yang ditemukan oleh Irene
memiliki tingkat akurasi 90%,
mereka tidak akan dapat bersaing dengan
ingatan sempurna Ellie yang mencapai 100%.
Namun, jika ujian tersebut seluruhnya terdiri dari soal baru,
Irene mungkin akan tetap menjaga rata-rata 90%-nya.

Sebagai ilmuwan machine learning,
tujuan kita adalah untuk menemukan *pola*.
Tetapi bagaimana kita bisa yakin bahwa
kita benar-benar menemukan *pola umum*
dan bukan sekadar menghafal data kita?
Sebagian besar waktu, prediksi kita hanya berguna
jika model kita benar-benar menemukan pola semacam itu.
Kita tidak ingin memprediksi harga saham kemarin, tetapi besok.
Kita tidak perlu mengenali
penyakit yang sudah terdiagnosis untuk pasien yang sudah pernah diperiksa,
tetapi justru untuk penyakit yang belum terdiagnosis
pada pasien yang belum pernah diperiksa sebelumnya.
Masalah ini—bagaimana menemukan pola yang *dapat digeneralisasi*—
adalah masalah mendasar dalam machine learning,
dan bisa dikatakan dalam seluruh statistik.
Kita bisa menganggap masalah ini sebagai satu bagian kecil
dari pertanyaan yang jauh lebih besar
yang mencakup seluruh ilmu pengetahuan:
kapan kita berhak membuat lompatan dari pengamatan khusus
ke pernyataan yang lebih umum?

Dalam kehidupan nyata, kita harus melatih model kita
menggunakan sekumpulan data yang terbatas.
Skala data yang tersedia bervariasi drastis tergantung pada domainnya.
Untuk banyak masalah medis yang penting,
kita hanya dapat mengakses beberapa ribu titik data.
Dalam mempelajari penyakit langka,
kita mungkin beruntung jika bisa mengakses ratusan titik data.
Sebaliknya, dataset publik terbesar
yang terdiri dari foto-foto berlabel,
misalnya ImageNet :cite:`Deng.Dong.Socher.ea.2009`,
mengandung jutaan gambar.
Beberapa koleksi gambar tanpa label,
seperti dataset Flickr YFC100M,
bahkan lebih besar, dengan lebih dari
100 juta gambar :cite:`thomee2016yfcc100m`.
Namun, meskipun pada skala ekstrem ini,
jumlah data yang tersedia tetap sangat kecil
dibandingkan dengan ruang semua gambar
yang mungkin ada pada resolusi megapiksel.
Kapan pun kita bekerja dengan sampel yang terbatas,
kita harus mengingat risiko
bahwa kita mungkin hanya menyesuaikan data pelatihan,
hanya untuk menyadari bahwa kita gagal
menemukan pola yang dapat digeneralisasi.


Fenomena di mana model menyesuaikan lebih baik pada data pelatihan
daripada pada distribusi yang mendasari disebut *overfitting*,
dan teknik untuk melawan overfitting
sering kali disebut sebagai metode *regularisasi*.
Meskipun ini bukan pengganti yang memadai
untuk pengenalan teori pembelajaran statistik (lihat :citet:`Vapnik98,boucheron2005theory`),
kami akan memberikan cukup banyak intuisi agar Anda bisa memulai.
Kita akan mengunjungi kembali konsep generalisasi di banyak bab
di sepanjang buku ini,
mengeksplorasi baik apa yang diketahui tentang
prinsip-prinsip dasar generalisasi
dalam berbagai model,
dan juga teknik heuristik
yang telah ditemukan (secara empiris)
dapat meningkatkan generalisasi
pada tugas-tugas yang memiliki kepentingan praktis.



## Error Pelatihan dan Error Generalisasi

Dalam pengaturan pembelajaran terawasi yang standar,
kita mengasumsikan bahwa data pelatihan dan data uji
diambil *secara independen* dari distribusi yang *identik*.
Ini biasanya disebut sebagai *asumsi IID* (Independent and Identically Distributed).
Meskipun asumsi ini kuat, penting untuk dicatat bahwa,
tanpa asumsi ini, kita akan menghadapi kesulitan besar.
Mengapa kita harus percaya bahwa data pelatihan
yang diambil dari distribusi $P(X,Y)$
dapat memberi tahu kita bagaimana membuat prediksi
pada data uji yang dihasilkan oleh
*distribusi yang berbeda* $Q(X,Y)$?
Melakukan lompatan seperti itu membutuhkan asumsi yang kuat
tentang bagaimana $P$ dan $Q$ berhubungan.
Nantinya, kita akan membahas beberapa asumsi
yang memungkinkan adanya perubahan distribusi,
tetapi pertama-tama kita perlu memahami kasus IID,
di mana $P(\cdot) = Q(\cdot)$.

Untuk memulai, kita perlu membedakan antara
*error pelatihan* $R_\textrm{emp}$,
yang merupakan *statistik* yang dihitung pada dataset pelatihan,
dan *error generalisasi* $R$,
yang merupakan *ekspektasi* yang diambil
dengan menghormati distribusi dasar.
Anda bisa menganggap error generalisasi sebagai
apa yang akan Anda lihat jika Anda menerapkan model Anda
pada aliran data tambahan yang tak terbatas
yang diambil dari distribusi data dasar yang sama.
Secara formal, error pelatihan diekspresikan sebagai *jumlah* (dengan notasi yang sama seperti pada :numref:`sec_linear_regression`):

$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$

sementara error generalisasi diekspresikan sebagai integral:

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] =
\int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Masalahnya, kita tidak pernah dapat menghitung
error generalisasi $R$ dengan tepat.
Tidak ada yang pernah memberi tahu kita bentuk pasti
dari fungsi densitas $p(\mathbf{x}, y)$.
Selain itu, kita tidak bisa mengambil sampel data dalam jumlah tak terbatas.
Oleh karena itu, dalam praktiknya, kita harus *mengestimasi* error generalisasi
dengan menerapkan model kita pada set uji independen
yang terdiri dari pilihan acak contoh
$\mathbf{X}'$ dan label $\mathbf{y}'$
yang tidak termasuk dalam set pelatihan kita.
Ini terdiri dari menerapkan formula yang sama
yang digunakan untuk menghitung error pelatihan empiris
tetapi pada set uji $\mathbf{X}', \mathbf{y}'$.

Yang penting, ketika kita mengevaluasi pengklasifikasi kita pada set uji,
kita bekerja dengan pengklasifikasi yang *tetap*
(yang tidak bergantung pada sampel set uji),
dan dengan demikian mengestimasi error-nya
hanya merupakan masalah estimasi rata-rata.
Namun, hal yang sama tidak dapat dikatakan
untuk set pelatihan.
Perhatikan bahwa model yang kita peroleh
bergantung secara eksplisit pada pilihan set pelatihan
dan, oleh karena itu, error pelatihan
umumnya akan menjadi estimasi bias dari error sebenarnya
pada populasi dasar.
Pertanyaan utama dalam generalisasi
adalah kapan kita harus mengharapkan error pelatihan kita
mendekati error populasi
(dan dengan demikian error generalisasi).

### Kompleksitas Model

Dalam teori klasik, ketika kita memiliki
model yang sederhana dan data yang melimpah,
error pelatihan dan generalisasi cenderung mendekati.
Namun, ketika kita bekerja dengan
model yang lebih kompleks dan/atau contoh yang lebih sedikit,
kita mengharapkan error pelatihan turun
tetapi kesenjangan generalisasi meningkat.
Hal ini seharusnya tidak mengejutkan.
Bayangkan kelas model yang begitu ekspresif
sehingga untuk dataset $n$ contoh mana pun,
kita dapat menemukan seperangkat parameter
yang dapat menyesuaikan label apa pun dengan sempurna,
bahkan jika diberikan secara acak.
Dalam kasus ini, bahkan jika kita menyesuaikan data pelatihan kita dengan sempurna,
bagaimana kita bisa menyimpulkan apa pun tentang error generalisasi?
Yang kita tahu, error generalisasi kita
mungkin tidak lebih baik dari tebakan acak.

Secara umum, tanpa ada batasan pada kelas model kita,
kita tidak bisa menyimpulkan, hanya berdasarkan penyesuaian data pelatihan,
bahwa model kita telah menemukan pola yang dapat digeneralisasi :cite:`vapnik1994measuring`.
Di sisi lain, jika kelas model kita
tidak mampu menyesuaikan label secara arbitrer,
maka harus ada pola yang ditemukan.
Gagasan teori pembelajaran tentang kompleksitas model
mengambil inspirasi dari gagasan
Karl Popper, seorang filsuf ilmu yang berpengaruh,
yang memformalkan kriteria *dapat dipalsukan*.
Menurut Popper, teori yang
dapat menjelaskan semua pengamatan yang ada
bukanlah teori ilmiah sama sekali!
Lagi pula, apa yang diceritakan teori tersebut tentang dunia
jika teori tersebut tidak menyingkirkan kemungkinan apa pun?
Singkatnya, yang kita inginkan adalah hipotesis
yang *tidak bisa* menjelaskan semua pengamatan
yang mungkin kita buat,
namun tetap cocok dengan pengamatan yang kita *buat*.

Sekarang, apa yang secara tepat membentuk gagasan
tentang kompleksitas model yang sesuai adalah masalah yang kompleks.
Seringkali, model dengan lebih banyak parameter
mampu menyesuaikan lebih banyak
label yang diberikan secara arbitrer.
Namun, ini tidak selalu benar.
Misalnya, metode kernel beroperasi di ruang
dengan jumlah parameter tak terbatas,
namun kompleksitasnya dikendalikan
dengan cara lain :cite:`Scholkopf.Smola.2002`.
Salah satu gagasan kompleksitas yang sering berguna
adalah rentang nilai yang dapat diambil parameter.
Di sini, model yang parameternya diizinkan
mengambil nilai sewenang-wenang
akan menjadi lebih kompleks.
Kita akan mengunjungi kembali gagasan ini di bagian berikutnya,
ketika kita memperkenalkan *penurunan bobot*,
teknik regularisasi praktis pertama Anda.
Perlu dicatat, sulit membandingkan
kompleksitas antara anggota dari kelas model yang sangat berbeda
(misalnya, pohon keputusan vs. jaringan saraf).




Pada titik ini, kita harus menekankan poin penting lainnya
yang akan kita bahas kembali ketika memperkenalkan jaringan saraf dalam.
Ketika sebuah model mampu menyesuaikan label secara arbitrer,
error pelatihan yang rendah tidak serta-merta
menunjukkan error generalisasi yang rendah.
*Namun, ini juga tidak serta-merta
menunjukkan error generalisasi yang tinggi!*
Yang bisa kita katakan dengan percaya diri adalah
bahwa error pelatihan yang rendah saja tidak cukup
untuk memastikan error generalisasi yang rendah.
Jaringan saraf dalam ternyata adalah model seperti ini:
meskipun mereka melakukan generalisasi dengan baik dalam praktiknya,
mereka terlalu kuat untuk memungkinkan kita menyimpulkan
banyak hal hanya berdasarkan error pelatihan.
Dalam kasus ini, kita harus lebih mengandalkan
data holdout untuk memastikan generalisasi
setelah fakta.
Error pada data holdout, yaitu, set validasi,
disebut *error validasi*.

## Underfitting atau Overfitting?

Ketika kita membandingkan error pelatihan dan error validasi,
kita ingin waspada terhadap dua situasi umum.
Pertama, kita ingin memperhatikan kasus
di mana error pelatihan dan error validasi kita keduanya cukup besar
tetapi hanya ada sedikit perbedaan di antara keduanya.
Jika model tidak dapat mengurangi error pelatihan,
itu bisa berarti bahwa model kita terlalu sederhana
(yaitu, tidak cukup ekspresif)
untuk menangkap pola yang ingin kita modelkan.
Selain itu, karena *gap generalisasi* ($R_\textrm{emp} - R$)
antara error pelatihan dan error generalisasi kita kecil,
kita memiliki alasan untuk percaya bahwa kita bisa menggunakan model yang lebih kompleks.
Fenomena ini dikenal sebagai *underfitting*.

Di sisi lain, seperti yang telah kita bahas di atas,
kita ingin waspada terhadap kasus
di mana error pelatihan kita jauh lebih rendah
daripada error validasi kita, yang mengindikasikan *overfitting* yang parah.
Perlu dicatat bahwa overfitting tidak selalu buruk.
Dalam pembelajaran mendalam khususnya,
model prediktif terbaik sering kali menunjukkan
kinerja jauh lebih baik pada data pelatihan dibandingkan pada data holdout.
Pada akhirnya, biasanya kita peduli tentang
menurunkan error generalisasi,
dan hanya peduli pada perbedaannya sejauh
itu menjadi hambatan untuk mencapai tujuan tersebut.
Perhatikan bahwa jika error pelatihan adalah nol,
maka gap generalisasi persis sama dengan error generalisasi
dan kita hanya bisa membuat kemajuan dengan mengurangi perbedaan ini.

### Penyesuaian Kurva Polinomial
:label:`subsec_polynomial-curve-fitting`

Untuk mengilustrasikan beberapa intuisi klasik
tentang overfitting dan kompleksitas model,
pertimbangkan hal berikut:
diberikan data pelatihan yang terdiri dari satu fitur $x$
dan label bernilai nyata $y$ yang bersesuaian,
kita mencoba menemukan polinomial derajat $d$

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

untuk memperkirakan label $y$.
Ini hanyalah masalah regresi linear
di mana fitur kita diberikan oleh pangkat $x$,
bobot model diberikan oleh $w_i$,
dan bias diberikan oleh $w_0$ karena $x^0 = 1$ untuk semua $x$.
Karena ini hanya masalah regresi linear,
kita dapat menggunakan error kuadrat sebagai fungsi kerugian kita.

Fungsi polinomial dengan derajat yang lebih tinggi lebih kompleks
daripada fungsi polinomial dengan derajat lebih rendah,
karena fungsi polinomial dengan derajat lebih tinggi memiliki lebih banyak parameter
dan rentang pemilihan fungsi model lebih luas.
Dengan data pelatihan yang tetap,
fungsi polinomial dengan derajat lebih tinggi harus selalu
mencapai error pelatihan yang lebih rendah (paling tidak sama)
dibandingkan dengan polinomial dengan derajat lebih rendah.
Bahkan, kapan pun setiap contoh data
memiliki nilai $x$ yang berbeda,
fungsi polinomial dengan derajat
sama dengan jumlah contoh data
dapat menyesuaikan set pelatihan secara sempurna.
Kita membandingkan hubungan antara derajat polinomial (kompleksitas model)
dan underfitting serta overfitting dalam :numref:`fig_capacity_vs_error`.

![Pengaruh kompleksitas model pada underfitting dan overfitting.](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`



### Ukuran Dataset

Seperti yang ditunjukkan oleh batas di atas,
pertimbangan besar lainnya yang perlu diingat adalah ukuran dataset.
Dengan model yang tetap, semakin sedikit sampel
yang kita miliki di dataset pelatihan,
semakin besar kemungkinan (dan semakin parah)
kita akan mengalami overfitting.
Seiring kita menambah jumlah data pelatihan,
error generalisasi biasanya menurun.
Selain itu, secara umum, lebih banyak data tidak pernah merugikan.
Untuk tugas dan distribusi data yang tetap,
kompleksitas model tidak boleh meningkat
lebih cepat dari jumlah data yang ada.
Dengan lebih banyak data, kita mungkin mencoba
untuk menyesuaikan model yang lebih kompleks.
Tanpa data yang cukup, model yang lebih sederhana
mungkin lebih sulit untuk dikalahkan.
Untuk banyak tugas, pembelajaran mendalam (deep learning)
hanya mengungguli model linear
ketika ribuan contoh pelatihan tersedia.
Keberhasilan pembelajaran mendalam saat ini sebagian besar
berutang pada melimpahnya dataset masif
yang berasal dari perusahaan internet, penyimpanan murah,
perangkat yang terhubung, dan digitalisasi ekonomi yang luas.

## Pemilihan Model
:label:`subsec_generalization-model-selection`

Biasanya, kita memilih model akhir kita
hanya setelah mengevaluasi beberapa model
yang berbeda dalam berbagai cara
(arsitektur yang berbeda, tujuan pelatihan,
fitur yang dipilih, pra-pemrosesan data,
laju pembelajaran, dll.).
Memilih di antara banyak model ini disebut
*pemilihan model*.

Pada prinsipnya, kita tidak boleh menyentuh set uji
sampai kita telah memilih semua hiperparameter.
Jika kita menggunakan data uji dalam proses pemilihan model,
ada risiko bahwa kita mungkin melakukan overfitting pada data uji.
Jika ini terjadi, kita akan berada dalam masalah besar.
Jika kita melakukan overfitting pada data pelatihan,
kita selalu dapat mengevaluasi model pada data uji untuk menjaga kejujuran.
Tetapi jika kita melakukan overfitting pada data uji, bagaimana kita akan tahu?
Lihat :citet:`ong2005learning` untuk contoh bagaimana
ini dapat mengarah pada hasil yang absurd bahkan untuk model di mana kompleksitasnya
dapat dikendalikan secara ketat.

Oleh karena itu, kita sebaiknya tidak mengandalkan data uji untuk pemilihan model.
Namun, kita juga tidak dapat hanya mengandalkan data pelatihan
untuk pemilihan model karena
kita tidak dapat memperkirakan error generalisasi
pada data yang sama yang kita gunakan untuk melatih model.

Dalam aplikasi praktis, gambarannya menjadi lebih keruh.
Meskipun idealnya kita hanya akan menyentuh data uji sekali,
untuk menilai model terbaik atau untuk membandingkan
sejumlah kecil model satu sama lain,
data uji di dunia nyata jarang dibuang setelah satu kali penggunaan.
Kita jarang mampu menyediakan set uji baru untuk setiap putaran eksperimen.
Faktanya, menggunakan kembali data benchmark selama beberapa dekade
dapat memiliki dampak signifikan pada
pengembangan algoritme,
misalnya untuk [klasifikasi gambar](https://paperswithcode.com/sota/image-classification-on-imagenet)
dan [pengenalan karakter optik](https://paperswithcode.com/sota/image-classification-on-mnist).

Praktik umum untuk mengatasi masalah *pelatihan pada set uji*
adalah membagi data kita menjadi tiga bagian,
menggabungkan *set validasi*
selain dataset pelatihan dan uji.
Hasilnya adalah bisnis yang tidak jelas di mana batas
antara data validasi dan data uji sangat ambigu.
Kecuali disebutkan secara eksplisit, dalam eksperimen di buku ini
kita sebenarnya bekerja dengan apa yang seharusnya disebut
data pelatihan dan data validasi, tanpa set uji yang sebenarnya.
Oleh karena itu, akurasi yang dilaporkan dalam setiap eksperimen di buku ini sebenarnya
adalah akurasi validasi dan bukan akurasi set uji yang sebenarnya.

### Cross-Validation

Ketika data pelatihan langka,
kita mungkin bahkan tidak mampu menyisihkan
data yang cukup untuk membentuk set validasi yang layak.
Salah satu solusi populer untuk masalah ini adalah menggunakan
*cross-validation K-kali lipat*.
Di sini, data pelatihan asli dibagi menjadi $K$ subset yang tidak saling tumpang tindih.
Kemudian pelatihan model dan validasi dijalankan $K$ kali,
setiap kali melatih pada $K-1$ subset dan memvalidasi
pada subset yang berbeda (yang tidak digunakan untuk pelatihan pada putaran tersebut).
Akhirnya, error pelatihan dan validasi diperkirakan
dengan menghitung rata-rata hasil dari $K$ eksperimen tersebut.


## Ringkasan

Bagian ini mengeksplorasi beberapa dasar 
dari generalisasi dalam pembelajaran mesin.
Beberapa ide ini menjadi rumit
dan kontraintuitif saat kita masuk ke model yang lebih dalam; di sini, model memiliki kemampuan untuk melakukan overfitting data secara signifikan,
dan gagasan tentang kompleksitas yang relevan
dapat menjadi implisit dan kontraintuitif
(misalnya, arsitektur yang lebih besar dengan lebih banyak parameter
justru lebih baik dalam generalisasi).
Berikut adalah beberapa aturan praktis:

1. Gunakan set validasi (atau *cross-validation K-kali lipat*) untuk pemilihan model;
2. Model yang lebih kompleks seringkali membutuhkan lebih banyak data;
3. Gagasan kompleksitas yang relevan mencakup baik jumlah parameter maupun rentang nilai yang diizinkan untuk diambil;
4. Dengan kondisi lain yang sama, lebih banyak data hampir selalu mengarah pada generalisasi yang lebih baik;
5. Semua pembicaraan tentang generalisasi ini bergantung pada asumsi IID. Jika kita melonggarkan asumsi ini, memungkinkan distribusi bergeser antara periode pelatihan dan pengujian, maka kita tidak bisa mengatakan apapun tentang generalisasi tanpa asumsi tambahan (yang mungkin lebih ringan).

## Latihan

1. Kapan Anda bisa menyelesaikan masalah regresi polinomial secara eksak?
2. Berikan setidaknya lima contoh di mana variabel acak yang bergantung membuat menganggap masalah sebagai data IID tidak disarankan.
3. Apakah Anda pernah mengharapkan error pelatihan menjadi nol? Dalam keadaan apa Anda akan melihat error generalisasi nol?
4. Mengapa $K$-fold cross-validation sangat mahal untuk dihitung?
5. Mengapa estimasi error *K*-fold cross-validation bias?
6. Dimensi VC didefinisikan sebagai jumlah maksimum titik yang dapat diklasifikasikan dengan label sembarang $\{\pm 1\}$ oleh fungsi dari suatu kelas fungsi. Mengapa ini mungkin bukan ide yang baik untuk mengukur seberapa kompleks kelas fungsi tersebut? Petunjuk: pertimbangkan besarnya fungsi.
7. Manajer Anda memberi Anda dataset yang sulit di mana algoritma saat ini tidak bekerja dengan baik. Bagaimana Anda akan meyakinkan manajer bahwa Anda membutuhkan lebih banyak data? Petunjuk: Anda tidak dapat menambah data tetapi Anda dapat menguranginya.

:begin_tab:`mxnet`
[Diskusi](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Diskusi](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Diskusi](https://discuss.d2l.ai/t/234)
:end_tab:

:begin_tab:`jax`
[Diskusi](https://discuss.d2l.ai/t/17978)
:end_tab:
