# generalisasi dari _Deep Learning_

Di :numref:`chap_regression` dan :numref:`chap_classification`,
kita menyelesaikan masalah regresi dan klasifikasi
dengan menyesuaikan model linear terhadap data pelatihan.
Dalam kedua kasus, kita menyediakan algoritma praktis
untuk menemukan parameter yang memaksimalkan
kemungkinan label pelatihan yang diamati.
Dan kemudian, di akhir setiap bab,
kita mengingatkan bahwa menyesuaikan data pelatihan
hanyalah tujuan antara.
Pencarian utama kita sejak awal adalah menemukan *pola umum*
sebagai dasar untuk membuat prediksi yang akurat
bahkan pada contoh baru yang diambil dari populasi yang sama.
Para peneliti machine learning adalah *konsumen* dari algoritma optimisasi.
Kadang-kadang, kita bahkan harus mengembangkan algoritma optimisasi baru.
Namun pada akhirnya, optimisasi hanyalah sarana untuk mencapai tujuan.
Pada intinya, machine learning adalah disiplin statistik
dan kita ingin mengoptimalkan *training loss* hanya sejauh
prinsip statistik tertentu (dikenal atau tidak diketahui)
membuat model yang dihasilkan dapat digeneralisasi melebihi data pelatihan.


Di sisi positif, ternyata jaringan neural dalam
yang dilatih dengan stochastic gradient descent dapat digeneralisasi dengan sangat baik
pada berbagai masalah prediksi, termasuk visi komputer;
pemrosesan bahasa alami; data deret waktu; sistem rekomendasi;
rekam medis elektronik; pelipatan protein;
pendekatan fungsi nilai dalam permainan video
dan permainan papan; serta banyak domain lainnya.
Di sisi negatif, jika Anda mencari penjelasan yang langsung
tentang baik cerita optimisasi
(mengapa kita bisa menyesuaikan mereka dengan data pelatihan)
atau cerita generalisasi
(mengapa model yang dihasilkan dapat digeneralisasi ke contoh yang belum pernah dilihat),
maka Anda mungkin perlu menyeduh minuman.
Sementara prosedur kita untuk mengoptimalkan model linear
dan sifat statistik dari solusinya
sudah dijelaskan dengan baik oleh teori komprehensif,
pemahaman kita tentang deep learning
masih seperti dunia yang liar di kedua aspek tersebut.

Baik teori maupun praktik deep learning
sedang berkembang pesat,
dengan para ahli teori yang mengadopsi strategi baru
untuk menjelaskan apa yang sedang terjadi,
sementara para praktisi terus
berinovasi dengan kecepatan luar biasa,
membangun sekumpulan heuristik untuk melatih jaringan dalam
dan intuisi serta pengetahuan yang berkembang
yang memberi panduan untuk memutuskan
teknik mana yang diterapkan dalam situasi tertentu.

Ringkasan dari momen saat ini adalah bahwa teori deep learning
telah menghasilkan beberapa serangan yang menjanjikan dan hasil yang menarik,
namun masih jauh dari pemahaman yang komprehensif
tentang (i) mengapa kita dapat mengoptimalkan jaringan neural
dan (ii) bagaimana model yang dipelajari oleh gradient descent
dapat digeneralisasi dengan sangat baik, bahkan pada tugas berdimensi tinggi.
Namun, dalam praktiknya, (i) jarang menjadi masalah
(kita selalu dapat menemukan parameter yang sesuai dengan semua data pelatihan kita)
dan karenanya pemahaman tentang generalisasi menjadi masalah yang jauh lebih besar.
Di sisi lain, bahkan tanpa kenyamanan teori ilmiah yang koheren,
para praktisi telah mengembangkan kumpulan teknik yang luas
yang dapat membantu Anda menghasilkan model yang dapat digeneralisasi dengan baik dalam praktik.
Meskipun tidak ada ringkasan singkat yang bisa menggambarkan
topik generalisasi dalam deep learning yang sangat luas,
dan meskipun kondisi penelitian secara keseluruhan jauh dari terselesaikan,
kami berharap, dalam bagian ini, untuk menyajikan gambaran umum
tentang kondisi penelitian dan praktik saat ini.


## Revisiting Overfitting dan Regularization

Menurut "no free lunch" theorem dari :citet:`wolpert1995no`,
algoritma pembelajaran apapun menghasilkan generalisasi yang lebih baik pada data dengan distribusi tertentu, dan lebih buruk pada distribusi lainnya.
Oleh karena itu, dengan data pelatihan yang terbatas,
sebuah model bergantung pada asumsi tertentu:
untuk mencapai performa setara manusia
mungkin berguna untuk mengidentifikasi *inductive biases*
yang mencerminkan cara manusia berpikir tentang dunia.
Bias induktif seperti itu menunjukkan preferensi
untuk solusi dengan sifat tertentu.
Misalnya,
MLP yang dalam memiliki bias induktif
menuju pembentukan fungsi yang rumit melalui komposisi fungsi yang lebih sederhana.

Dengan model machine learning yang mengkodekan bias induktif,
pendekatan kita untuk melatih model ini
biasanya terdiri dari dua fase: (i) menyesuaikan data pelatihan;
dan (ii) memperkirakan *generalization error*
(kesalahan sejati pada populasi yang mendasarinya)
dengan mengevaluasi model pada data holdout.
Perbedaan antara kecocokan kita pada data pelatihan
dan kecocokan kita pada data uji disebut *generalization gap* dan ketika ini besar,
kita mengatakan bahwa model kita *overfit* pada data pelatihan.
Dalam kasus overfitting yang ekstrem,
kita mungkin sangat menyesuaikan data pelatihan,
bahkan ketika kesalahan uji tetap signifikan.
Dan dalam pandangan klasik,
interpretasinya adalah bahwa model kita terlalu kompleks,
membutuhkan kita untuk mengurangi jumlah fitur,
jumlah parameter yang tidak nol yang dipelajari,
atau ukuran parameter sebagaimana dihitung.
Ingat grafik kompleksitas model dibandingkan dengan loss
(:numref:`fig_capacity_vs_error`)
dari :numref:`sec_generalization_basics`.


Namun deep learning memperumit gambaran ini dengan cara yang tidak intuitif.
Pertama, untuk masalah klasifikasi,
model kita biasanya cukup ekspresif
untuk benar-benar menyesuaikan setiap contoh pelatihan,
bahkan dalam dataset yang terdiri dari jutaan
:cite:`zhang2021understanding`.
Dalam pandangan klasik, kita mungkin berpikir
bahwa pengaturan ini terletak pada ekstrem kanan
dari sumbu kompleksitas model,
dan bahwa setiap peningkatan pada *generalization error*
harus dilakukan melalui regularisasi,
baik dengan mengurangi kompleksitas kelas model,
atau dengan menerapkan penalti, sangat membatasi
set nilai yang mungkin diambil oleh parameter kita.
Namun, di sinilah hal-hal mulai menjadi aneh.

Anehnya, untuk banyak tugas deep learning
(misalnya, pengenalan gambar dan klasifikasi teks)
kita biasanya memilih di antara arsitektur model,
semua di antaranya dapat mencapai *training loss* yang sangat rendah
(dan *training error* yang nol).
Karena semua model yang dipertimbangkan mencapai *training error* yang nol,
*satu-satunya jalan untuk meningkatkan lebih jauh adalah dengan mengurangi overfitting*.
Yang lebih aneh lagi, sering kali
meskipun menyesuaikan data pelatihan secara sempurna,
kita sebenarnya bisa *mengurangi generalization error*
lebih jauh dengan membuat model *lebih ekspresif*,
misalnya, menambahkan lapisan, node, atau melatih
untuk jumlah epoch yang lebih besar.
Yang lebih aneh lagi, pola yang menghubungkan generalization gap
dengan *kompleksitas* model (seperti kedalaman atau lebar jaringan)
bisa menjadi non-monoton,
dengan peningkatan kompleksitas yang awalnya merugikan
namun kemudian membantu dalam pola "double-descent" yang disebutkan
:cite:`nakkiran2021deep`.
Oleh karena itu, praktisi deep learning memiliki sekumpulan trik,
beberapa di antaranya tampaknya membatasi model dalam beberapa cara
dan yang lainnya tampaknya membuatnya lebih ekspresif,
dan semua ini, dalam beberapa hal, diterapkan untuk mengurangi overfitting.

Menambah kerumitan, sementara jaminan yang diberikan oleh teori pembelajaran klasik
bisa konservatif bahkan untuk model klasik,
mereka tampaknya tidak berdaya untuk menjelaskan mengapa
jaringan neural dalam bisa melakukan generalisasi.
Karena jaringan neural dalam mampu menyesuaikan
label yang sewenang-wenang bahkan untuk dataset besar,
dan meskipun menggunakan metode yang familier seperti $\ell_2$ regularisasi,
batas generalisasi berbasis kompleksitas tradisional,
misalnya, yang didasarkan pada dimensi VC
atau kompleksitas Rademacher dari kelas hipotesis
tidak dapat menjelaskan mengapa jaringan neural dapat melakukan generalisasi.


## Inspirasi dari _Nonparametrics_

Mendekati deep learning untuk pertama kalinya,
mungkin tergoda untuk menganggapnya sebagai model parametrik.
Bagaimanapun, model *memiliki* jutaan parameter.
Ketika kita memperbarui model, kita memperbarui parameternya.
Ketika kita menyimpan model, kita menulis parameternya ke disk.
Namun, matematika dan ilmu komputer penuh
dengan perubahan perspektif yang tidak intuitif,
dan isomorfisme yang mengejutkan antara masalah yang tampaknya berbeda.
Meskipun jaringan neural jelas *memiliki* parameter,
dalam beberapa hal bisa lebih bermanfaat
untuk menganggapnya sebagai model nonparametrik.
Jadi, apa sebenarnya yang membuat model menjadi nonparametrik?
Meskipun istilah ini mencakup berbagai pendekatan,
satu tema umum adalah bahwa metode nonparametrik
cenderung memiliki tingkat kompleksitas yang meningkat
seiring dengan bertambahnya data yang tersedia.

Contoh model nonparametrik yang paling sederhana
mungkin adalah algoritma $k$-nearest neighbor (kita akan membahas lebih banyak model nonparametrik nanti, misalnya di :numref:`sec_attention-pooling`).
Di sini, pada saat pelatihan,
pembelajar hanya menghafal dataset.
Kemudian, pada saat prediksi,
ketika dihadapkan dengan titik baru $\mathbf{x}$,
pembelajar mencari $k$ tetangga terdekat
($k$ titik $\mathbf{x}_i'$ yang meminimalkan
jarak tertentu $d(\mathbf{x}, \mathbf{x}_i')$).
Ketika $k=1$, algoritma ini disebut $1$-nearest neighbor,
dan algoritma ini akan selalu mencapai *training error* sebesar nol.
Namun, itu tidak berarti bahwa algoritma ini tidak bisa digeneralisasi.
Faktanya, ternyata di bawah beberapa kondisi ringan,
algoritma 1-nearest neighbor konsisten
(pada akhirnya akan menyatu dengan prediktor optimal).


Perhatikan bahwa $1$-nearest neighbor membutuhkan kita untuk menentukan
beberapa fungsi jarak $d$, atau setara,
kita menentukan beberapa fungsi basis yang bernilai vektor $\phi(\mathbf{x})$
untuk melakukan featurisasi data kita.
Untuk pilihan metrik jarak apapun,
kita akan mencapai *training error* nol
dan akhirnya mencapai prediktor optimal,
tetapi metrik jarak yang berbeda $d$
mengkodekan bias induktif yang berbeda
dan dengan jumlah data yang tersedia yang terbatas
akan menghasilkan prediktor yang berbeda.
Pilihan metrik jarak $d$ yang berbeda
mewakili asumsi yang berbeda tentang pola dasar
dan kinerja prediktor yang berbeda
akan bergantung pada seberapa kompatibel asumsi tersebut
dengan data yang diamati.

Dalam arti tertentu, karena jaringan neural berlebihan parametriknya,
memiliki banyak parameter lebih banyak daripada yang diperlukan untuk menyesuaikan data pelatihan,
jaringan ini cenderung *menginterpolasi* data pelatihan (mencocokkannya dengan sempurna)
dan karenanya berperilaku, dalam beberapa hal, lebih seperti model nonparametrik.
Penelitian teoretis yang lebih baru telah menetapkan
hubungan yang mendalam antara jaringan neural besar
dan metode nonparametrik, terutama metode kernel.
Secara khusus, :citet:`Jacot.Grabriel.Hongler.2018`
menunjukkan bahwa dalam batas tertentu, saat multilayer perceptron
dengan bobot yang diinisialisasi secara acak menjadi sangat lebar,
mereka menjadi setara dengan metode kernel (nonparametrik)
untuk pilihan fungsi kernel tertentu
(pada dasarnya, fungsi jarak),
yang mereka sebut neural tangent kernel.
Meskipun model neural tangent kernel saat ini mungkin belum sepenuhnya menjelaskan
perilaku jaringan dalam modern,
kesuksesan mereka sebagai alat analisis
menyoroti kegunaan pemodelan nonparametrik
untuk memahami perilaku jaringan neural yang berlebihan parametriknya.



## Early Stopping (Pemberhentian Awal)

Meskipun jaringan neural dalam mampu menyesuaikan label secara sewenang-wenang,
bahkan ketika label ditetapkan secara salah atau acak
:cite:`zhang2021understanding`,
kemampuan ini hanya muncul setelah banyak iterasi pelatihan.
Sebuah jalur penelitian baru :cite:`Rolnick.Veit.Belongie.Shavit.2017`
menunjukkan bahwa dalam pengaturan noise label,
jaringan neural cenderung menyesuaikan data dengan label yang bersih terlebih dahulu
dan hanya kemudian menginterpolasi data yang diberi label salah.
Lebih lanjut, telah ditetapkan bahwa fenomena ini
langsung diterjemahkan menjadi jaminan generalisasi:
kapanpun sebuah model telah menyesuaikan data yang berlabel bersih
tetapi tidak menyesuaikan contoh dengan label acak dalam data pelatihan,
maka model tersebut sebenarnya telah digeneralisasi :cite:`Garg.Balakrishnan.Kolter.Lipton.2021`.

Temuan ini membantu memotivasi teknik *early stopping*,
sebuah teknik klasik untuk regularisasi jaringan neural dalam.
Di sini, alih-alih secara langsung membatasi nilai bobot,
kita membatasi jumlah epoch pelatihan.
Cara yang paling umum untuk menentukan kriteria berhenti
adalah dengan memonitor kesalahan validasi selama pelatihan
(dengan mengecek setelah setiap epoch)
dan menghentikan pelatihan ketika kesalahan validasi
tidak berkurang lebih dari jumlah kecil $\epsilon$
selama beberapa epoch.
Ini kadang-kadang disebut sebagai *patience criterion*.
Selain memiliki potensi untuk menghasilkan generalisasi yang lebih baik
dalam pengaturan noise label,
keuntungan lain dari early stopping adalah waktu yang dihemat.
Setelah patience criterion terpenuhi, kita dapat menghentikan pelatihan.
Untuk model besar yang mungkin memerlukan waktu pelatihan berhari-hari
secara bersamaan di delapan atau lebih GPU,
early stopping yang disetel dengan baik dapat menghemat waktu berhari-hari
dan menghemat biaya penelitian yang sangat besar.

Perlu dicatat, ketika tidak ada noise label dan dataset *realizable*
(kelas-kelas benar-benar dapat dipisahkan, misalnya, membedakan kucing dari anjing),
early stopping biasanya tidak menghasilkan peningkatan signifikan dalam generalisasi.
Namun, ketika terdapat noise label,
atau variabilitas intrinsik pada label
(misalnya, memprediksi mortalitas pada pasien),
early stopping sangat penting.
Melatih model hingga menginterpolasi data yang noisy umumnya merupakan ide yang buruk.


## Metode Classical Regularization untuk Deep Networks

Di :numref:`chap_regression`, kita telah mendeskripsikan
beberapa teknik regularisasi klasik
untuk membatasi kompleksitas model kita.
Secara khusus, :numref:`sec_weight_decay`
memperkenalkan metode yang disebut weight decay,
yang terdiri dari penambahan istilah regularisasi pada fungsi loss
untuk menghukum nilai bobot yang besar.
Tergantung pada norm bobot yang dihukum,
teknik ini dikenal sebagai ridge regularization (untuk penalti $\ell_2$)
atau lasso regularization (untuk penalti $\ell_1$).
Dalam analisis klasik dari regularizer ini,
mereka dianggap cukup membatasi nilai
yang dapat diambil oleh bobot untuk mencegah model menyesuaikan label secara sewenang-wenang.

Dalam implementasi deep learning,
weight decay tetap menjadi alat yang populer.
Namun, para peneliti telah mencatat
bahwa kekuatan regulasi $\ell_2$ yang khas
tidak cukup untuk mencegah jaringan
menginterpolasi data :cite:`zhang2021understanding` dan karenanya manfaat jika ditafsirkan
sebagai regularisasi mungkin hanya masuk akal
jika dikombinasikan dengan kriteria early stopping.
Tanpa early stopping, dimungkinkan
bahwa seperti jumlah lapisan
atau jumlah node (dalam deep learning)
atau metrik jarak (dalam 1-nearest neighbor),
metode-metode ini dapat menghasilkan generalisasi yang lebih baik
bukan karena mereka benar-benar membatasi
kekuatan jaringan neural,
tetapi karena mereka mengkodekan bias induktif
yang lebih kompatibel dengan pola
yang ditemukan dalam dataset yang menarik.
Dengan demikian, regularizer klasik tetap populer
dalam implementasi deep learning,
meskipun dasar teoretis untuk keefektifannya mungkin sangat berbeda.

Perlu dicatat, para peneliti deep learning juga telah membangun
teknik yang pertama kali dipopulerkan
dalam konteks regularisasi klasik,
seperti menambahkan noise ke input model.
Pada bagian berikutnya kita akan memperkenalkan
teknik dropout yang terkenal
(yang ditemukan oleh :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`),
yang telah menjadi andalan dalam deep learning,
meskipun dasar teoritis untuk keefektifannya
juga masih agak misterius.


## Rangkuman

Tidak seperti model linear klasik,
yang cenderung memiliki parameter lebih sedikit daripada contoh,
jaringan dalam cenderung memiliki kelebihan parameter,
dan untuk sebagian besar tugas mampu
untuk menyesuaikan dataset pelatihan dengan sempurna.
Regime ini disebut sebagai *interpolation regime*
dan menantang banyak intuisi yang sudah lama dipegang teguh.
Secara fungsional, jaringan neural terlihat seperti model parametrik.
Namun, menganggapnya sebagai model nonparametrik
kadang-kadang bisa menjadi sumber intuisi yang lebih andal.
Karena seringkali semua jaringan dalam yang dipertimbangkan
mampu menyesuaikan semua label pelatihan,
hampir semua peningkatan harus datang dengan mengurangi overfitting
(menutup *generalization gap*).
Secara paradoks, intervensi
yang mengurangi generalization gap
kadang-kadang tampak meningkatkan kompleksitas model
dan di lain waktu tampak mengurangi kompleksitas.
Namun, metode-metode ini jarang mengurangi kompleksitas
secara cukup untuk teori klasik
untuk menjelaskan generalisasi jaringan dalam,
dan *mengapa pilihan tertentu menghasilkan generalisasi yang lebih baik*
sebagian besar tetap menjadi pertanyaan terbuka yang besar
meskipun ada upaya bersama dari banyak peneliti brilian.


## Latihan

1. Dalam hal apa ukuran kompleksitas tradisional gagal menjelaskan generalisasi jaringan neural dalam?
2. Mengapa *early stopping* bisa dianggap sebagai teknik regularisasi?
3. Bagaimana para peneliti biasanya menentukan kriteria berhenti?
4. Faktor penting apa yang tampaknya membedakan kasus ketika early stopping menghasilkan peningkatan besar dalam generalisasi?
5. Selain generalisasi, jelaskan manfaat lain dari early stopping.

[Discussions](https://discuss.d2l.ai/t/7473)
