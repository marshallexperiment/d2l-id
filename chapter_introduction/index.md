# Pendahuluan
:label:`chap_introduction`

Hingga baru-baru ini, hampir setiap program komputer
yang mungkin Anda interaksikan selama
hari biasa
dikodekan sebagai serangkaian aturan yang kaku
yang menentukan secara tepat bagaimana seharusnya mereka berperilaku.
Misalkan kita ingin menulis sebuah aplikasi
untuk mengelola platform e-commerce.
Setelah berkumpul di sekitar papan tulis
selama beberapa jam untuk merenungkan masalah tersebut,
kita mungkin menetapkan garis besar
dari solusi yang berfungsi, misalnya:
(i) pengguna berinteraksi dengan aplikasi melalui antarmuka
yang berjalan di browser web atau aplikasi seluler;
(ii) aplikasi kita berinteraksi dengan mesin database komersial
untuk melacak keadaan setiap pengguna dan memelihara catatan
dari transaksi historis;
dan (iii) di jantung aplikasi kita,
* logika bisnis * (Anda mungkin mengatakan, * otak *) dari aplikasi kita
menjabarkan serangkaian aturan yang memetakan setiap keadaan yang dapat dibayangkan
ke tindakan yang sesuai yang harus diambil program kita.

Untuk membangun otak dari aplikasi kita,
kita mungkin menghitung semua peristiwa umum
yang harus ditangani program kita.
Misalnya, setiap kali pelanggan mengklik
untuk menambahkan barang ke keranjang belanja mereka,
program kita harus menambahkan entri
ke tabel database keranjang belanja,
mengaitkan ID pengguna tersebut
dengan ID produk yang diminta.
Kemudian kita mungkin mencoba melangkah melalui
setiap kasus sudut yang mungkin,
menguji kelayakan aturan kita
dan melakukan modifikasi yang diperlukan.
Apa yang terjadi jika pengguna
memulai pembelian dengan keranjang kosong?
Sedikit pengembang yang pernah mendapatkannya
benar pada waktu pertama
(mungkin butuh beberapa percobaan untuk mengatasi masalah),
untuk sebagian besar kita dapat menulis program semacam itu
dan dengan percaya diri meluncurkannya
*sebelum* pernah melihat pelanggan nyata.
Kemampuan kita untuk secara manual merancang sistem otomatis
yang menggerakkan produk dan sistem yang berfungsi,
seringkali dalam situasi baru,
adalah prestasi kognitif yang luar biasa.
Dan ketika Anda dapat merancang solusi
yang bekerja $100\%$ dari waktu,
Anda biasanya tidak seharusnya
khawatir tentang pembelajaran mesin.


Untungnya bagi komunitas ilmuwan pembelajaran mesin yang terus berkembang,
banyak tugas yang ingin kita otomatisasi
tidak mudah tunduk pada kecerdikan manusia.
Bayangkan berkumpul di sekitar papan tulis
dengan orang-orang terpintar yang Anda kenal,
tetapi kali ini Anda sedang menghadapi
salah satu masalah berikut:

* Menulis program yang memprediksi cuaca esok hari berdasarkan informasi geografis, gambar satelit, dan jendela waktu cuaca masa lalu.
* Menulis program yang menerima pertanyaan faktual, yang diungkapkan dalam teks bebas, dan menjawabnya dengan benar.
* Menulis program yang, diberikan sebuah gambar, mengidentifikasi setiap orang yang digambarkan di dalamnya dan menggambar garis besar di sekitar masing-masing.
* Menulis program yang menyajikan kepada pengguna produk-produk yang kemungkinan mereka nikmati tetapi tidak mungkin, dalam jalur alami penjelajahan, untuk ditemukan.

Untuk masalah-masalah ini,
bahkan pemrogram elit akan kesulitan
untuk membuat solusi dari awal.
Alasannya bisa bervariasi.
Kadang-kadang program yang kita cari
mengikuti pola yang berubah dari waktu ke waktu,
jadi tidak ada jawaban yang benar yang tetap!
Dalam kasus seperti itu, setiap solusi yang berhasil
harus beradaptasi dengan anggun ke dunia yang berubah.
Di waktu lain, hubungan (misalnya antara piksel,
dan kategori abstrak) mungkin terlalu rumit,
membutuhkan ribuan atau jutaan perhitungan
dan mengikuti prinsip-prinsip yang tidak diketahui.
Dalam kasus pengenalan gambar,
langkah-langkah tepat yang diperlukan untuk melakukan tugas
berada di luar pemahaman sadar kita,
meskipun proses kognitif bawah sadar kita
menjalankan tugas itu dengan mudah.


*Pembelajaran mesin* adalah studi tentang algoritma
yang dapat belajar dari pengalaman.
Seiring algoritma pembelajaran mesin mengumpulkan lebih banyak pengalaman,
biasanya dalam bentuk data observasional
atau interaksi dengan lingkungan,
kinerjanya meningkat.
Bandingkan ini dengan platform e-commerce deterministik kita,
yang mengikuti logika bisnis yang sama,
tidak peduli berapa banyak pengalaman yang terakumulasi,
sampai para pengembang sendiri belajar dan memutuskan
bahwa sudah waktunya untuk memperbarui perangkat lunak.
Dalam buku ini, kami akan mengajarkan Anda
dasar-dasar pembelajaran mesin,
dengan fokus khusus pada *pembelajaran mendalam*,
seperangkat teknik yang kuat
yang mendorong inovasi di berbagai area seperti visi komputer,
pengolahan bahasa alami, kesehatan, dan genomik.


## Contoh yang Memotivasi

Sebelum mulai menulis, para penulis buku ini,
seperti sebagian besar tenaga kerja, harus terkafinasi.
Kami naik mobil dan mulai mengemudi.
Menggunakan iPhone, Alex memanggil "Hey Siri",
membangunkan sistem pengenalan suara ponsel.
Kemudian Mu memerintahkan "petunjuk ke toko kopi Blue Bottle".
Ponsel segera menampilkan transkripsi perintahnya.
Ponsel juga mengenali bahwa kami meminta petunjuk arah
dan meluncurkan aplikasi Peta (app)
untuk memenuhi permintaan kami.
Setelah diluncurkan, aplikasi Peta mengidentifikasi sejumlah rute.
Di samping setiap rute, ponsel menampilkan waktu perjalanan yang diprediksi.
Meskipun cerita ini dibuat untuk kemudahan pedagogis,
ini menunjukkan bahwa hanya dalam beberapa detik,
interaksi sehari-hari kita dengan ponsel pintar
dapat melibatkan beberapa model pembelajaran mesin.

Bayangkan hanya menulis program untuk merespons *kata bangun*
seperti "Alexa", "OK Google", dan "Hey Siri".
Cobalah membuatnya sendiri di sebuah ruangan
hanya dengan komputer dan editor kode,
seperti yang digambarkan di :numref:`fig_wake_word`.
Bagaimana Anda akan menulis program tersebut dari prinsip dasar?
Pikirkanlah... masalahnya sulit.
Setiap detik, mikrofon akan mengumpulkan sekitar
44.000 sampel.
Setiap sampel adalah pengukuran amplitudo gelombang suara.
Apa aturan yang bisa memetakan secara andal dari cuplikan audio mentah ke prediksi yang pasti
$\{\text{ya}, \text{tidak}\}$
tentang apakah cuplikan tersebut mengandung kata bangun?
Jika Anda kesulitan, jangan khawatir.
Kami juga tidak tahu cara menulis program seperti itu dari awal.
Itulah mengapa kami menggunakan pembelajaran mesin.

![Identifikasi kata bangun.](../img/wake-word.svg)
:label:`fig_wake_word`

Berikut triknya.
Sering kali, meskipun kita tidak tahu bagaimana memberi tahu komputer
secara eksplisit cara memetakan dari input ke output,
kita masih mampu melakukan prestasi kognitif itu sendiri.
Dengan kata lain, meskipun Anda tidak tahu
bagaimana memprogram komputer untuk mengenali kata "Alexa",
Anda sendiri mampu mengenalinya.
Dengan kemampuan ini, kita dapat mengumpulkan *dataset* besar
yang berisi contoh cuplikan audio dan label terkait,
menunjukkan cuplikan mana yang mengandung kata bangun.
Dalam pendekatan yang saat ini dominan terhadap pembelajaran mesin,
kita tidak mencoba untuk merancang sistem
*secara eksplisit* untuk mengenali kata bangun.
Sebaliknya, kita mendefinisikan program yang fleksibel
yang perilakunya ditentukan oleh sejumlah *parameter*.
Kemudian kita menggunakan dataset untuk menentukan nilai parameter terbaik,
yaitu, yang meningkatkan kinerja program kita
sehubungan dengan ukuran kinerja yang dipilih.


Anda dapat menganggap parameter sebagai kenop yang dapat kita putar,
memanipulasi perilaku program.
Setelah parameter ditetapkan, kami menyebut program itu sebagai *model*.
Kumpulan semua program yang berbeda (pemetaan input--output)
yang dapat kita hasilkan hanya dengan memanipulasi parameter
disebut sebagai *keluarga* model.
Dan "meta-program" yang menggunakan dataset kita
untuk memilih parameter disebut *algoritma pembelajaran*.

Sebelum kita dapat melanjutkan dan melibatkan algoritma pembelajaran,
kita harus mendefinisikan masalah secara tepat,
menetapkan sifat pasti dari input dan output,
dan memilih keluarga model yang tepat.
Dalam hal ini,
model kita menerima potongan audio sebagai *input*,
dan model
menghasilkan pilihan di antara
$\{\text{ya}, \text{tidak}\}$ sebagai *output*.
Jika semua berjalan sesuai rencana
tebakan model akan
biasanya benar mengenai
apakah cuplikan tersebut mengandung kata bangun.

Jika kita memilih keluarga model yang tepat,
harus ada satu pengaturan kenop
sehingga model menyala "ya" setiap kali mendengar kata "Alexa".
Karena pilihan kata bangun yang tepat bersifat sewenang-wenang,
kita mungkin memerlukan keluarga model yang cukup kaya sehingga,
melalui pengaturan kenop lain, itu bisa menyala "ya"
hanya setelah mendengar kata "Apricot".
Kita mengharapkan bahwa keluarga model yang sama harus cocok
untuk pengenalan "Alexa" dan "Apricot"
karena intuitif, mereka tampaknya tugas yang serupa.
Namun, kita mungkin memerlukan keluarga model yang berbeda sepenuhnya
jika kita ingin berurusan dengan input atau output yang fundamental berbeda,
misalnya jika kita ingin memetakan dari gambar ke keterangan,
atau dari kalimat bahasa Inggris ke kalimat bahasa Mandarin.

Seperti yang mungkin Anda duga, jika kita hanya mengatur semua kenop secara acak,
tidak mungkin model kita akan mengenali "Alexa",
"Apricot", atau kata bahasa Inggris lainnya.
Dalam pembelajaran mesin,
*pembelajaran* adalah proses
di mana kita menemukan pengaturan kenop yang tepat
untuk memaksa perilaku yang diinginkan dari model kita.
Dengan kata lain,
kita *melatih* model kita dengan data.
Seperti yang ditunjukkan dalam :numref:`fig_ml_loop`, proses pelatihan biasanya terlihat seperti berikut:

1. Mulai dengan model yang diinisialisasi secara acak yang tidak dapat melakukan apa pun yang berguna.
2. Ambil beberapa data Anda (mis., potongan audio dan label $\{\text{ya}, \text{tidak}\}$ yang sesuai).
3. Putar kenop untuk membuat model berkinerja lebih baik seperti yang dinilai pada contoh tersebut.
4. Ulangi Langkah 2 dan 3 sampai modelnya luar biasa.

![Proses pelatihan yang khas.](../img/ml-loop.svg)
:label:`fig_ml_loop`

Untuk merangkum, daripada membuat pengenal kata bangun,
kami membuat program yang dapat *belajar* untuk mengenali kata bangun,
jika disajikan dengan dataset berlabel besar.
Anda dapat menganggap tindakan menentukan perilaku program
dengan menyajikannya dengan dataset sebagai *pemrograman dengan data*.
Dengan kata lain, kita dapat "memprogram" detektor kucing
dengan memberikan sistem pembelajaran mesin kita
banyak contoh kucing dan anjing.
Dengan cara ini, detektor pada akhirnya akan belajar untuk mengeluarkan
angka positif yang sangat besar jika itu adalah kucing,
angka negatif yang sangat besar jika itu adalah anjing,
dan sesuatu yang lebih dekat ke nol jika tidak yakin.
Ini baru permulaan dari apa yang dapat dilakukan oleh pembelajaran mesin.
Pembelajaran mendalam, yang akan kami jelaskan lebih rinci nanti,
hanyalah salah satu dari banyak metode populer
untuk memecahkan masalah pembelajaran mesin.


## Komponen Utama

Dalam contoh kata bangun kami, kami menggambarkan dataset
yang terdiri dari potongan audio dan label biner,
dan kami memberikan gambaran bagaimana kami mungkin melatih
model untuk mendekati pemetaan dari potongan ke klasifikasi.
Jenis masalah ini,
di mana kita mencoba memprediksi label yang tidak diketahui yang ditunjuk
berdasarkan input yang diketahui
diberikan dataset yang terdiri dari contoh
yang labelnya diketahui,
disebut *pembelajaran terawasi*.
Ini hanya salah satu dari banyak jenis masalah pembelajaran mesin.
Sebelum kita menjelajahi varietas lain,
kami ingin memberikan lebih banyak cahaya
pada beberapa komponen inti yang akan mengikuti kita,
tidak peduli jenis masalah pembelajaran mesin apa yang kita tangani:

1. *Data* yang dapat kita pelajari.
2. Sebuah *model* bagaimana mengubah data.
3. Sebuah *fungsi objektif* yang mengukur seberapa baik (atau buruk) model tersebut.
4. Sebuah *algoritma* untuk menyesuaikan parameter model untuk mengoptimalkan fungsi objektif.

### Data

Mungkin sudah jelas bahwa Anda tidak dapat melakukan ilmu data tanpa data.
Kita bisa kehilangan ratusan halaman merenungkan apa sebenarnya data *adalah*,
tapi untuk saat ini, kami akan fokus pada properti kunci
dari dataset yang akan kami khawatirkan.
Umumnya, kami prihatin dengan kumpulan contoh.
Untuk bekerja dengan data secara berguna, kita biasanya
perlu membuat representasi numerik yang sesuai.
Setiap *contoh* (atau *titik data*, *instansi data*, *sampel*)
biasanya terdiri dari satu set atribut
yang disebut *fitur* (kadang disebut *covariates* atau *inputs*),
berdasarkan itu model harus membuat prediksinya.
Dalam masalah pembelajaran terawasi,
tujuan kita adalah memprediksi nilai atribut khusus,
yang disebut *label* (atau *target*),
yang tidak menjadi bagian dari input model.

Jika kita bekerja dengan data gambar,
setiap contoh mungkin terdiri dari sebuah
fotografi individu (fiturnya)
dan angka yang menunjukkan kategori
di mana fotografi itu termasuk (labelnya).
Fotografi akan diwakili secara numerik
sebagai tiga kisi nilai numerik yang mewakili
kecerahan cahaya merah, hijau, dan biru
di setiap lokasi piksel.
Misalnya, sebuah fotografi warna $200\times 200$ piksel
akan terdiri dari $200\times200\times3=120000$ nilai numerik.


Sebagai alternatif, kita mungkin bekerja dengan data rekam medis elektronik
dan menghadapi tugas memprediksi kemungkinan
seorang pasien akan bertahan hidup dalam 30 hari ke depan.
Di sini, fitur-fitur kita mungkin terdiri dari koleksi
atribut yang mudah tersedia
dan pengukuran yang sering dicatat,
termasuk usia, tanda vital, komorbiditas,
obat-obatan saat ini, dan prosedur terbaru.
Label yang tersedia untuk pelatihan adalah nilai biner
yang menunjukkan apakah setiap pasien dalam data historis
bertahan hidup dalam jendela 30 hari.

Dalam kasus seperti ini, ketika setiap contoh dicirikan
oleh jumlah fitur numerik yang sama,
kita mengatakan bahwa input adalah vektor dengan panjang tetap
dan kita menyebut panjang (konstan) dari vektor tersebut
sebagai *dimensi* dari data.
Seperti yang dapat Anda bayangkan, input dengan panjang tetap dapat menjadi nyaman,
memberikan kita satu komplikasi lebih sedikit untuk dikhawatirkan.
Namun, tidak semua data dapat dengan mudah
direpresentasikan sebagai vektor dengan panjang tetap.
Meskipun kita mungkin mengharapkan gambar mikroskop berasal dari peralatan standar,
kita tidak dapat mengharapkan gambar yang diambil dari internet semua memiliki resolusi atau bentuk yang sama.
Untuk gambar, kita mungkin mempertimbangkan
memotongnya menjadi ukuran standar,
tetapi strategi ini hanya bisa membantu sampai batas tertentu.
Kita berisiko kehilangan informasi pada bagian yang dipotong.
Selain itu, data teks bahkan lebih keras kepala dalam menolak representasi dengan panjang tetap.
Pertimbangkan ulasan pelanggan yang ditinggalkan
di situs e-commerce seperti Amazon, IMDb, dan TripAdvisor.
Beberapa pendek: "itu buruk!".
Yang lainnya bertele-tele berhalaman-halaman.
Salah satu keunggulan utama pembelajaran mendalam dibandingkan dengan metode tradisional
adalah kemampuan model modern untuk menangani data dengan panjang yang bervariasi dengan lebih anggun.

Secara umum, semakin banyak data yang kita miliki, semakin mudah pekerjaan kita.
Ketika kita memiliki lebih banyak data, kita dapat melatih model yang lebih kuat
dan mengandalkan asumsi yang sudah ada lebih sedikit.
Perubahan rezim dari data kecil ke data besar
merupakan kontributor utama bagi keberhasilan pembelajaran mendalam modern.
Untuk menggarisbawahi poin ini, banyak
model paling menarik dalam pembelajaran mendalam
tidak akan berfungsi tanpa dataset besar.
Beberapa mungkin berfungsi dalam rezim data kecil,
tetapi tidak lebih baik dari pendekatan tradisional.

Akhirnya, memiliki banyak data saja tidak cukup
dan memprosesnya dengan cerdas.
Kita memerlukan data yang *tepat*.
Jika data penuh dengan kesalahan,
atau jika fitur yang dipilih tidak prediktif
terhadap kuantitas target yang diminati,
pembelajaran akan gagal.
Situasi ini dengan baik dijelaskan oleh klise:
*sampah masuk, sampah keluar*.
Lebih lanjut, kinerja prediktif yang buruk
bukan satu-satunya konsekuensi potensial.
Dalam aplikasi pembelajaran mesin yang sensitif,
seperti polisi prediktif, penyaringan resume,
dan model risiko yang digunakan untuk peminjaman,
kita harus sangat waspada
terhadap konsekuensi dari data sampah.
Salah satu mode kegagalan yang sering terjadi adalah dataset
di mana beberapa kelompok orang tidak terwakili
dalam data pelatihan.
Bayangkan menerapkan sistem pengenalan kanker kulit
yang belum pernah melihat kulit hitam sebelumnya.
Kegagalan juga dapat terjadi ketika data
tidak hanya kurang mewakili beberapa kelompok
tetapi mencerminkan prasangka masyarakat.
Misalnya, jika keputusan perekrutan masa lalu
digunakan untuk melatih model prediktif
yang akan digunakan untuk menyaring resume
maka model pembelajaran mesin dapat secara tidak sengaja
menangkap dan mengotomatiskan ketidakadilan historis.
Perlu dicatat bahwa ini semua dapat terjadi tanpa ilmuwan data
secara aktif berkonspirasi, atau bahkan menyadari.


### Model

Sebagian besar pembelajaran mesin melibatkan transformasi data dalam beberapa cara.
Kita mungkin ingin membangun sistem yang mengolah foto dan memprediksi tingkat senyum.
Sebagai alternatif,
kita mungkin ingin mengolah sekumpulan pembacaan sensor
dan memprediksi seberapa normal atau anomali pembacaan tersebut.
Dengan *model*, kami menunjuk perangkat keras komputasi untuk mengonsumsi data
dari satu jenis,
dan mengeluarkan prediksi dari jenis yang mungkin berbeda.
Khususnya, kami tertarik pada *model statistik*
yang dapat diestimasi dari data.
Sementara model sederhana sangat mampu mengatasi
masalah yang sederhana secara tepat,
masalah yang kami fokuskan dalam buku ini menguji batas metode klasik.
Pembelajaran mendalam berbeda dari pendekatan klasik
terutama melalui kumpulan model kuat yang menjadi fokusnya.
Model-model ini terdiri dari banyak transformasi data berturut-turut
yang dihubungkan dari atas ke bawah, oleh karena itu dinamakan *pembelajaran mendalam*.
Dalam perjalanan kami mendiskusikan model-model mendalam,
kami juga akan membahas beberapa metode tradisional.

### Fungsi Objektif

Sebelumnya, kami memperkenalkan pembelajaran mesin sebagai pembelajaran dari pengalaman.
Dengan *pembelajaran* di sini,
kami maksudkan peningkatan dalam suatu tugas dari waktu ke waktu.
Tapi, siapa yang bisa menentukan apa yang merupakan peningkatan?
Anda mungkin membayangkan bahwa kita bisa mengusulkan pembaruan model kita,
dan beberapa orang mungkin tidak setuju apakah usulan kita
merupakan peningkatan atau tidak.

Untuk mengembangkan sistem matematis formal dari mesin pembelajaran,
kita perlu memiliki ukuran formal seberapa baik (atau buruk) model kita.
Dalam pembelajaran mesin, dan optimasi pada umumnya,
kita menyebut ini *fungsi objektif*.
Secara konvensi, kita biasanya mendefinisikan fungsi objektif
sehingga lebih rendah adalah lebih baik.
Ini hanya sebuah konvensi.
Anda dapat mengambil fungsi apa pun
yang lebih tinggi adalah lebih baik, dan mengubahnya menjadi fungsi baru
yang secara kualitatif identik tetapi yang lebih rendah adalah lebih baik
dengan membalik tanda.
Karena kita memilih lebih rendah adalah lebih baik, fungsi-fungsi ini terkadang disebut
*fungsi *loss**.

Ketika mencoba memprediksi nilai numerik,
fungsi *loss* yang paling umum adalah *kesalahan kuadrat*,
yaitu, kuadrat dari perbedaan antara
prediksi dan target kebenaran dasar.
Untuk klasifikasi, tujuan yang paling umum
adalah meminimalkan tingkat kesalahan,
yaitu, fraksi contoh di mana
prediksi kita tidak setuju dengan kebenaran dasar.
Beberapa tujuan (mis., kesalahan kuadrat) mudah dioptimalkan,
sedangkan yang lain (mis., tingkat kesalahan) sulit dioptimalkan secara langsung,
karena ketidakdiferensialan atau komplikasi lain.
Dalam kasus ini, umumnya lebih umum untuk mengoptimalkan *tujuan pengganti*.

Selama optimasi, kita menganggap *loss*
sebagai fungsi dari parameter model,
dan menganggap dataset pelatihan sebagai konstan.
Kita belajar
nilai terbaik dari parameter model kita
dengan meminimalkan *loss* yang ditimbulkan pada satu set
yang terdiri dari beberapa jumlah contoh yang dikumpulkan untuk pelatihan.
Namun, berkinerja baik pada data pelatihan
tidak menjamin bahwa kita akan berkinerja baik pada data yang belum terlihat.
Jadi, kita biasanya ingin membagi data yang tersedia menjadi dua partisi:
*dataset pelatihan* (atau *set pelatihan*), untuk belajar parameter model;
dan *dataset tes* (atau *set tes*), yang ditahan untuk evaluasi.
Pada akhir hari, kita biasanya melaporkan
bagaimana kinerja model kita pada kedua partisi.
Anda bisa menganggap kinerja pelatihan
sebagai analogi dengan skor yang dicapai seorang siswa
pada ujian latihan yang digunakan untuk mempersiapkan ujian akhir yang sebenarnya.
Meskipun hasilnya mendorong,
itu tidak menjamin kesuksesan pada ujian akhir.
Selama belajar, siswa
mungkin mulai menghafal pertanyaan latihan,
tampak menguasai topik tetapi gagal
ketika dihadapkan dengan pertanyaan yang belum pernah dilihat sebelumnya
pada ujian akhir yang sebenarnya.
Ketika model berkinerja baik pada set pelatihan
tetapi gagal untuk digeneralisasi ke data yang belum terlihat,
kita mengatakan bahwa itu *overfitting* ke data pelatihan.


### Algoritma Optimasi

Setelah kita memiliki sumber data dan representasi,
model, dan fungsi objektif yang jelas,
kita memerlukan algoritma yang mampu mencari
parameter terbaik untuk meminimalkan *fungsi *loss**.
Algoritma optimasi populer untuk pembelajaran mendalam
berbasis pada pendekatan yang disebut *turunan gradien*.
Secara singkat, pada setiap langkah, metode ini
memeriksa, untuk setiap parameter,
bagaimana *loss* set pelatihan akan berubah
jika Anda mengganggu parameter tersebut hanya dengan jumlah kecil.
Kemudian akan memperbarui parameter
dalam arah yang menurunkan *loss*.

## Jenis Masalah Pembelajaran Mesin

Masalah kata aktif dalam contoh motivasi kita
hanyalah salah satu dari banyak
yang dapat ditangani oleh pembelajaran mesin.
Untuk memotivasi pembaca lebih lanjut
dan memberikan kita beberapa bahasa umum
yang akan mengikuti kita sepanjang buku,
kami sekarang menyediakan gambaran luas tentang lanskap
masalah pembelajaran mesin.



### Supervised Learning (*Pembelajaran terawasi*)

Pembelajaran terawasi (Supervised Learning) menggambarkan tugas-tugas
di mana kita diberi dataset
yang mengandung fitur dan label
dan diminta untuk menghasilkan model yang memprediksi label ketika
diberikan fitur input.
Setiap pasangan fitur--label disebut contoh.
Kadang-kadang, ketika konteksnya jelas,
kita mungkin menggunakan istilah *contoh-contoh*
untuk merujuk pada kumpulan input,
bahkan ketika label yang sesuai tidak diketahui.
Supervisi terjadi
karena, untuk memilih parameter,
kami (para supervisor) menyediakan model
dengan dataset yang terdiri dari contoh berlabel.
Dalam istilah probabilistik, kita biasanya tertarik untuk memperkirakan
probabilitas bersyarat dari label yang diberikan fitur input.
Meskipun hanya salah satu di antara beberapa paradigma,
pembelajaran terawasi menyumbang mayoritas
aplikasi pembelajaran mesin yang sukses di industri.
Sebagian itu karena banyak tugas penting
dapat dijelaskan secara tajam sebagai memperkirakan probabilitas
sesuatu yang tidak diketahui berdasarkan satu set data yang tersedia:

* Prediksi kanker vs. tidak kanker, diberikan gambar tomografi komputer.
* Prediksi terjemahan yang benar dalam bahasa Prancis, diberikan sebuah kalimat dalam bahasa Inggris.
* Prediksi harga saham bulan depan berdasarkan data pelaporan keuangan bulan ini.

Meskipun semua masalah pembelajaran terawasi
dijelaskan oleh deskripsi sederhana
"memprediksi label yang diberikan fitur input",
pembelajaran terawasi itu sendiri dapat mengambil bentuk yang beragam
dan memerlukan banyak keputusan pemodelan,
tergantung pada (diantara pertimbangan lain)
jenis, ukuran, dan jumlah input dan output.
Misalnya, kita menggunakan model yang berbeda
untuk memproses urutan dengan panjang yang berubah-ubah
dan representasi vektor dengan panjang tetap.
Kami akan mengunjungi banyak masalah ini
secara mendalam sepanjang buku ini.

Secara informal, proses pembelajaran terlihat seperti berikut.
Pertama, ambil kumpulan besar contoh di mana fiturnya diketahui
dan pilih dari mereka subset acak,
memperoleh label kebenaran dasar untuk masing-masing.
Terkadang label ini mungkin adalah data yang sudah dikumpulkan
(mis., apakah pasien meninggal dalam tahun berikutnya?)
dan lain kali kita mungkin perlu mempekerjakan penilai manusia untuk melabeli data,
(mis., menetapkan gambar ke kategori).
Bersama-sama, input ini dan label yang sesuai membentuk set pelatihan.
Kami memberi dataset pelatihan ke algoritma pembelajaran terawasi,
fungsi yang mengambil dataset sebagai input
dan mengeluarkan fungsi lain: model yang dipelajari.
Akhirnya, kita dapat memberi input yang sebelumnya tidak terlihat ke model yang dipelajari,
menggunakan outputnya sebagai prediksi label yang sesuai.
Proses lengkap digambarkan di :numref:`fig_supervised_learning`.

![Pembelajaran terawasi.](../img/supervised-learning.svg)
:label:`fig_supervised_learning`


#### Regresi

Mungkin tugas pembelajaran terawasi yang paling sederhana
untuk dipahami adalah *regresi*.
Sebagai contoh, pertimbangkan serangkaian data yang dikumpulkan
dari basis data penjualan rumah.
Kita mungkin membuat sebuah tabel,
di mana setiap baris sesuai dengan rumah yang berbeda,
dan setiap kolom sesuai dengan beberapa atribut yang relevan,
seperti luas lantai rumah,
jumlah kamar tidur, jumlah kamar mandi,
dan jumlah menit (berjalan kaki) ke pusat kota.
Dalam dataset ini, setiap contoh akan menjadi rumah tertentu,
dan vektor fitur yang sesuai akan menjadi satu baris dalam tabel.
Jika Anda tinggal di New York atau San Francisco,
dan Anda bukan CEO Amazon, Google, Microsoft, atau Facebook,
vektor fitur (luas lantai, no. kamar tidur, no. kamar mandi, jarak berjalan kaki)
untuk rumah Anda mungkin terlihat seperti: $[600, 1, 1, 60]$.
Namun, jika Anda tinggal di Pittsburgh, itu mungkin terlihat lebih seperti $[3000, 4, 3, 10]$.
Vektor fitur dengan panjang tetap seperti ini penting
untuk sebagian besar algoritma pembelajaran mesin klasik.

Apa yang membuat suatu masalah menjadi regresi sebenarnya
adalah bentuk target.
Katakanlah Anda sedang mencari rumah baru.
Anda mungkin ingin memperkirakan nilai pasar yang wajar dari sebuah rumah,
dengan beberapa fitur seperti di atas.
Data di sini mungkin terdiri dari daftar rumah historis
dan labelnya mungkin adalah harga jual yang diamati.
Ketika label mengambil nilai numerik sembarang
(meskipun dalam beberapa interval),
kita menyebut ini sebagai masalah *regresi*.
Tujuannya adalah untuk menghasilkan model yang prediksinya
mendekati nilai label sebenarnya dengan cermat.

Banyak masalah praktis dengan mudah digambarkan sebagai masalah regresi.
Memprediksi peringkat yang akan diberikan pengguna kepada sebuah film
dapat dianggap sebagai masalah regresi
dan jika Anda merancang algoritma hebat
untuk mencapai prestasi ini pada tahun 2009,
Anda mungkin telah memenangkan [hadiah 1 juta dolar dari Netflix](https://en.wikipedia.org/wiki/Netflix_Prize).
Memprediksi lama tinggal pasien di rumah sakit
juga merupakan masalah regresi.
Aturan praktis yang baik adalah bahwa setiap masalah *berapa banyak?* atau *berapa lama?*
kemungkinan besar adalah regresi. Misalnya:

* Berapa jam operasi ini akan berlangsung?
* Berapa banyak hujan yang akan turun di kota ini dalam enam jam ke depan?

Bahkan jika Anda belum pernah bekerja dengan pembelajaran mesin sebelumnya,
Anda mungkin telah secara informal menyelesaikan masalah regresi.
Bayangkan, misalnya, Anda telah memperbaiki saluran pembuangan
dan kontraktor Anda menghabiskan 3 jam
menghilangkan kotoran dari pipa pembuangan Anda.
Kemudian mereka mengirimkan tagihan sebesar 350 dolar.
Sekarang bayangkan bahwa teman Anda mempekerjakan kontraktor yang sama selama 2 jam
dan menerima tagihan sebesar 250 dolar.
Jika seseorang kemudian bertanya kepada Anda berapa banyak yang diharapkan
pada tagihan penghapusan kotoran mendatang mereka
Anda mungkin membuat beberapa asumsi yang masuk akal,
seperti lebih banyak jam bekerja menghasilkan lebih banyak dolar.
Anda mungkin juga berasumsi bahwa ada biaya dasar
dan kemudian kontraktor mengenakan biaya per jam.
Jika asumsi ini benar, maka dengan dua contoh data ini,
Anda sudah bisa mengidentifikasi struktur harga kontraktor:
100 dolar per jam ditambah 50 dolar untuk muncul di rumah Anda.
Jika Anda mengikuti itu, maka Anda sudah memahami
ide dasar di balik regresi *linier*.

Dalam hal ini, kita bisa menghasilkan parameter
yang tepat sesuai dengan harga kontraktor.
Terkadang ini tidak mungkin,
mis., jika beberapa variasi
muncul dari faktor di luar dua fitur Anda.
Dalam kasus ini, kita akan mencoba belajar model
yang meminimalkan jarak antara prediksi kita dan nilai yang diamati.
Di sebagian besar bab kita, kita akan fokus pada
meminimalkan fungsi *loss* kesalahan kuadrat.
Seperti yang akan kita lihat nanti, *loss* ini sesuai dengan asumsi
bahwa data kita terkontaminasi oleh noise Gaussian.


#### Klasifikasi

Meskipun model regresi sangat baik
untuk menjawab pertanyaan *berapa banyak?*,
banyak masalah tidak cocok nyaman dalam template ini.
Pertimbangkan, misalnya, sebuah bank yang ingin
mengembangkan fitur pemindaian cek untuk aplikasi selulernya.
Idealnya, pelanggan hanya perlu mengambil foto cek
dan aplikasi secara otomatis akan mengenali teks dari gambar tersebut.
Dengan asumsi bahwa kita memiliki kemampuan
untuk memisahkan potongan gambar
yang sesuai dengan setiap karakter tulisan tangan,
maka tugas utama yang tersisa adalah
menentukan karakter mana di antara sekumpulan karakter yang diketahui
yang digambarkan di setiap potongan gambar tersebut.
Jenis masalah *yang mana?* ini disebut *klasifikasi*
dan memerlukan seperangkat alat yang berbeda
dari yang digunakan untuk regresi,
meskipun banyak teknik dapat dibawa.

Dalam *klasifikasi*, kita ingin model kita melihat fitur,
misalnya, nilai piksel dalam gambar,
dan kemudian memprediksi ke *kategori*
(kadang-kadang disebut *kelas*)
di antara beberapa set opsi diskrit,
contoh tersebut termasuk.
Untuk angka-angka tulisan tangan, kita mungkin memiliki sepuluh kelas,
yang sesuai dengan angka 0 hingga 9.
Bentuk klasifikasi yang paling sederhana adalah ketika hanya ada dua kelas,
masalah yang kita sebut *klasifikasi biner*.
Misalnya, dataset kita bisa terdiri dari gambar hewan
dan label kita mungkin kelas $\textrm{\{kucing, anjing\}}$.
Sementara dalam regresi kita mencari regresor untuk mengeluarkan nilai numerik,
dalam klasifikasi kita mencari pengklasifikasi,
yang keluarannya adalah penugasan kelas yang diprediksi.

Karena alasan yang akan kita bahas saat buku ini menjadi lebih teknis,
bisa sulit untuk mengoptimalkan model yang hanya dapat mengeluarkan
penugasan kategori *tegas*,
misalnya, "kucing" atau "anjing".
Dalam kasus ini, biasanya jauh lebih mudah untuk menyatakan
model kita dalam bahasa probabilitas.
Diberikan fitur dari sebuah contoh,
model kita menetapkan probabilitas
untuk setiap kelas yang mungkin.
Kembali ke contoh klasifikasi hewan kita
di mana kelasnya adalah $\textrm{\{kucing, anjing\}}$,
seorang pengklasifikasi mungkin melihat gambar dan mengeluarkan probabilitas
bahwa gambar tersebut adalah kucing sebesar 0.9.
Kita dapat menginterpretasikan angka ini dengan mengatakan bahwa pengklasifikasi
90\% yakin bahwa gambar tersebut menggambarkan kucing.
Besarnya probabilitas untuk kelas yang diprediksi
menyampaikan gagasan tentang ketidakpastian.
Ini bukan satu-satunya yang tersedia
dan kita akan membahas yang lain dalam bab yang membahas topik yang lebih lanjut.

Ketika kita memiliki lebih dari dua kelas yang mungkin,
kita menyebut masalah tersebut *klasifikasi multikelas*.
Contoh umum termasuk pengenalan karakter tulisan tangan $\textrm{\{0, 1, 2, ... 9, a, b, c, ...\}}$.
Sementara kita menyerang masalah regresi dengan mencoba
meminimalkan fungsi *loss* kesalahan kuadrat,
fungsi *loss* umum untuk masalah klasifikasi disebut *entropi silang*,
yang namanya akan dijelaskan
ketika kita memperkenalkan teori informasi di bab selanjutnya.

Perhatikan bahwa kelas yang paling mungkin bukanlah
yang akan Anda gunakan untuk keputusan Anda.
Anggap bahwa Anda menemukan jamur cantik di halaman belakang Anda
seperti yang ditunjukkan di :numref:`fig_death_cap`.

![Death cap---jangan dimakan!](../img/death-cap.jpg)
:width:`200px`
:label:`fig_death_cap`

Sekarang, anggap Anda membangun pengklasifikasi dan melatihnya
untuk memprediksi apakah jamur beracun berdasarkan foto.
Katakanlah pengklasifikasi pendeteksi racun kita mengeluarkan
bahwa kemungkinan :numref:`fig_death_cap` menunjukkan tutup kematian adalah 0.2.
Dengan kata lain, pengklasifikasi 80\% yakin
bahwa jamur kita bukan tutup kematian.
Namun, Anda harus bodoh untuk memakannya.
Itu karena manfaat makan malam yang lezat
tidak sebanding dengan risiko 20\% mati darinya.
Dengan kata lain, efek dari risiko yang tidak pasti
jauh lebih besar daripada manfaatnya.
Jadi, untuk membuat keputusan tentang apakah akan memakan jamur itu,
kita perlu menghitung kerugian yang diharapkan
yang terkait dengan setiap tindakan
yang bergantung pada hasil yang mungkin
dan manfaat atau kerugian yang terkait dengan masing-masing.
Dalam hal ini, kerugian yang diderita
dengan memakan jamur itu
mungkin adalah $0.2 \times \infty + 0.8 \times 0 = \infty$,
sedangkan *loss* dari membuangnya
adalah $0.2 \times 0 + 0.8 \times 1 = 0.8$.
Kewaspadaan kita dibenarkan:
seperti yang akan dikatakan oleh ahli mikologi kepada kita,
jamur di :numref:`fig_death_cap`
sebenarnya adalah tutup kematian.

Klasifikasi bisa jauh lebih rumit daripada hanya
klasifikasi biner atau multikelas.
Misalnya, ada beberapa varian klasifikasi
yang menangani kelas yang terstruktur secara hierarkis.
Dalam kasus seperti itu tidak semua kesalahan sama---jika
kita harus salah, kita mungkin lebih memilih untuk salah klasifikasi
ke kelas yang terkait daripada kelas yang jauh.
Biasanya, ini disebut sebagai *klasifikasi hierarkis*.
Untuk inspirasi, Anda mungkin memikirkan [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
yang mengatur fauna dalam hierarki.

Dalam kasus klasifikasi hewan,
mungkin tidak terlalu buruk untuk salah mengira
pudel sebagai schnauzer,
tapi model kita akan mendapat hukuman besar
jika salah mengira pudel dengan dinosaurus.
Hierarki mana yang relevan mungkin bergantung
pada bagaimana Anda berencana menggunakan model.
Misalnya, ular derik dan ular pita
mungkin dekat pada pohon filogenetik,
tapi salah mengira derik untuk pita bisa berakibat fatal.


#### Tagging (*Penandaan*)

Beberapa masalah klasifikasi dengan rapi masuk
ke dalam pengaturan klasifikasi biner atau multikelas.
Misalnya, kita bisa melatih pengklasifikasi biner normal
untuk membedakan kucing dan anjing.
Dengan keadaan saat ini dari visi komputer,
kita dapat melakukan ini dengan mudah, dengan alat yang tersedia.
Namun, tidak peduli seberapa akurat model kita,
kita mungkin menemukan masalah ketika pengklasifikasi
menemukan gambar dari *Town Musicians of Bremen*,
dongeng Jerman populer yang menampilkan empat hewan
(:numref:`fig_stackedanimals`).

![Seekor keledai, anjing, kucing, dan ayam jago.](../img/stackedanimals.png)
:width:`300px`
:label:`fig_stackedanimals`

Seperti yang Anda lihat, foto menampilkan kucing,
ayam jago, anjing, dan keledai,
dengan beberapa pohon di latar belakang.
Jika kita mengantisipasi menemui gambar seperti ini,
klasifikasi multikelas mungkin bukan
formulasi masalah yang tepat.
Sebagai gantinya, kita mungkin ingin memberi model opsi
untuk mengatakan gambar menggambarkan kucing, anjing, keledai,
*dan* ayam jago.

Masalah belajar untuk memprediksi kelas yang
tidak saling eksklusif disebut *klasifikasi multi-label*.
Masalah penandaan otomatis biasanya paling baik dijelaskan
dalam istilah klasifikasi multi-label.
Pikirkan tentang tag yang mungkin diterapkan orang
ke posting di blog teknis,
misalnya, "pembelajaran mesin", "teknologi", "gadget",
"bahasa pemrograman", "Linux", "komputasi awan", "AWS".
Artikel tipikal mungkin memiliki 5–10 tag yang diterapkan.
Biasanya, tag akan menunjukkan beberapa struktur korelasi.
Posting tentang "komputasi awan" kemungkinan akan menyebutkan "AWS"
dan posting tentang "pembelajaran mesin" kemungkinan akan menyebutkan "GPU".

Terkadang masalah penandaan tersebut
menggunakan set label yang sangat besar.
Perpustakaan Kedokteran Nasional
mempekerjakan banyak penanda profesional
yang mengaitkan setiap artikel yang akan diindeks di PubMed
dengan satu set tag yang diambil dari
ontologi Medical Subject Headings (MeSH),
koleksi sekitar 28,000 tag.
Penandaan artikel dengan benar penting
karena memungkinkan peneliti untuk melakukan
tinjauan menyeluruh dari literatur.
Ini adalah proses yang memakan waktu dan biasanya ada keterlambatan satu tahun antara pengarsipan dan penandaan.
Pembelajaran mesin dapat menyediakan tag sementara
sampai setiap artikel memiliki tinjauan manual yang tepat.
Memang, selama beberapa tahun, organisasi BioASQ
telah [mengadakan kompetisi](http://bioasq.org/)
untuk tugas ini.

#### Search (*Pencarian*)

Di bidang pengambilan informasi,
kita sering memberi peringkat pada kumpulan item.
Ambil contoh pencarian web.
Tujuannya kurang untuk menentukan *apakah*
sebuah halaman tertentu relevan untuk sebuah kueri,
melainkan mana, di antara sekumpulan hasil yang relevan,
seharusnya ditampilkan paling menonjol
kepada pengguna tertentu.
Salah satu cara melakukan ini mungkin
dengan pertama menetapkan skor
ke setiap elemen dalam kumpulan
dan kemudian mengambil elemen-elemen dengan peringkat teratas.
[PageRank](https://en.wikipedia.org/wiki/PageRank),
rahasia di balik mesin pencari Google pada awalnya,
adalah contoh awal dari sistem penilaian seperti ini.
Anehnya, skor yang diberikan oleh PageRank
tidak tergantung pada kueri aktual.
Sebaliknya, mereka mengandalkan filter relevansi sederhana
untuk mengidentifikasi kumpulan kandidat yang relevan
dan kemudian menggunakan PageRank untuk memprioritaskan
halaman yang lebih berwibawa.
Saat ini, mesin pencari menggunakan pembelajaran mesin dan model perilaku
untuk mendapatkan skor relevansi yang bergantung pada kueri.
Ada konferensi akademik yang sepenuhnya didedikasikan untuk topik ini.

#### Recommender Systems (*Sistem Rekomendasi*)
:label:`subsec_recommender_systems`

Sistem rekomendasi adalah pengaturan masalah lain
yang terkait dengan pencarian dan peringkat.
Masalahnya serupa karena tujuannya
adalah untuk menampilkan sekumpulan item yang relevan untuk pengguna.
Perbedaan utamanya adalah penekanan pada *personalisasi*
kepada pengguna tertentu dalam konteks sistem rekomendasi.
Misalnya, untuk rekomendasi film,
halaman hasil untuk penggemar fiksi ilmiah
dan halaman hasil
untuk penggemar komedi Peter Sellers
mungkin sangat berbeda.
Masalah serupa muncul dalam pengaturan rekomendasi lainnya,
misalnya, untuk produk ritel, musik, dan rekomendasi berita.

Dalam beberapa kasus, pelanggan memberikan umpan balik eksplisit,
menyampaikan seberapa banyak mereka menyukai produk tertentu
(mis., peringkat dan ulasan produk
di Amazon, IMDb, atau Goodreads).
Dalam kasus lain, mereka memberikan umpan balik implisit,
misalnya, dengan melewatkan judul di daftar putar,
yang mungkin menunjukkan 
ketidakpuasan atau mungkin hanya
menunjukkan
bahwa lagu tersebut tidak tepat dalam konteks.
Dalam formulasi paling sederhana,
sistem ini dilatih
untuk memperkirakan beberapa skor,
seperti peringkat bintang yang diharapkan
atau probabilitas bahwa pengguna tertentu
akan membeli item tertentu.

Diberikan model seperti itu, untuk pengguna mana pun,
kita dapat mengambil kumpulan objek dengan skor tertinggi,
yang kemudian dapat direkomendasikan kepada pengguna.
Sistem produksi jauh lebih canggih
dan mempertimbangkan aktivitas pengguna rinci dan karakteristik item
saat menghitung skor tersebut.
:numref:`fig_deeplearning_amazon` menampilkan buku-buku pembelajaran mendalam
yang direkomendasikan oleh Amazon berdasarkan algoritma personalisasi
yang disesuaikan dengan preferensi Aston.

![Buku-buku pembelajaran mendalam yang direkomendasikan oleh Amazon.](../img/deeplearning-amazon.jpg)
:label:`fig_deeplearning_amazon`

Meskipun memiliki nilai ekonomi yang sangat besar,
sistem rekomendasi
yang naif dibangun di atas model prediktif
menderita beberapa kekurangan konseptual serius.
Untuk memulai, kita hanya mengamati *umpan balik tersensor*:
pengguna lebih suka menilai film
yang mereka rasakan kuat tentangnya.
Misalnya, pada skala lima poin,
Anda mungkin memperhatikan bahwa item menerima
banyak peringkat satu dan lima bintang
tetapi ada sedikit peringkat tiga bintang yang mencolok.
Selain itu, kebiasaan pembelian saat ini sering merupakan hasil
dari algoritma rekomendasi yang saat ini ada,
tetapi algoritma pembelajaran tidak selalu memperhitungkan detail ini.
Dengan demikian, dimungkinkan untuk membentuk loop umpan balik
di mana sistem rekomendasi secara preferensial mendorong suatu item
yang kemudian dianggap lebih baik (karena pembelian yang lebih besar)
dan pada gilirannya direkomendasikan lebih sering.
Banyak dari masalah ini—tentang
bagaimana menangani sensor, insentif, dan loop umpan balik—adalah pertanyaan penelitian terbuka yang penting.


#### Sequence Learning (*Pembelajaran Urutan*)

Sejauh ini, kita telah melihat masalah di mana kita memiliki
jumlah input tetap dan menghasilkan jumlah output tetap.
Misalnya, kita mempertimbangkan prediksi harga rumah
dengan mengingat satu set fitur tetap:
luas lantai, jumlah kamar tidur,
jumlah kamar mandi, dan waktu tempuh ke pusat kota.
Kita juga membahas pemetaan dari gambar (dengan dimensi tetap)
ke probabilitas prediksi bahwa gambar tersebut termasuk
ke dalam salah satu dari sejumlah kelas tetap
dan memprediksi peringkat bintang yang terkait dengan pembelian
berdasarkan ID pengguna dan ID produk saja.
Dalam kasus ini, setelah model kita dilatih,
setelah setiap contoh uji dimasukkan ke dalam model kita,
itu segera dilupakan.
Kita mengasumsikan bahwa pengamatan berturut-turut adalah independen
dan dengan demikian tidak perlu menyimpan konteks ini.

Tapi bagaimana kita harus menangani potongan video?
Dalam kasus ini, setiap potongan mungkin terdiri dari jumlah bingkai yang berbeda.
Dan tebakan kita tentang apa yang terjadi di setiap bingkai mungkin jauh lebih kuat
jika kita memperhitungkan bingkai sebelumnya atau berikutnya.
Hal yang sama berlaku untuk bahasa.
Misalnya, salah satu masalah pembelajaran mendalam yang populer adalah penerjemahan mesin:
tugas mengonsumsi kalimat dalam beberapa bahasa sumber
dan memprediksi terjemahan mereka dalam bahasa lain.

Masalah seperti ini juga terjadi dalam kedokteran.
Kita mungkin ingin model untuk memantau pasien di unit perawatan intensif
dan mengirimkan peringatan kapan pun risiko mereka meninggal dalam 24 jam ke depan
melebihi ambang batas tertentu.
Di sini, kita tidak akan membuang semua
yang kita ketahui tentang riwayat pasien setiap jam,
karena kita mungkin tidak ingin membuat prediksi hanya
berdasarkan pengukuran terbaru.

Pertanyaan-pertanyaan seperti ini adalah di antara aplikasi pembelajaran mesin
yang paling menarik dan mereka adalah contoh *pembelajaran urutan*.
Mereka membutuhkan model untuk mengonsumsi urutan input
atau mengeluarkan urutan output (atau keduanya).
Secara khusus, *pembelajaran urutan-ke-urutan* mempertimbangkan masalah
di mana baik input maupun output terdiri dari urutan dengan panjang variabel.
Contoh termasuk penerjemahan mesin
dan transkripsi teks ke teks.
Meskipun tidak mungkin untuk mempertimbangkan
semua jenis transformasi urutan,
kasus-kasus khusus berikut patut disebutkan.

**tagging and parsing (*Penandaan* dan *Penguraian*)**.
Ini melibatkan anotasi urutan teks dengan atribut.
Di sini, input dan output adalah *sejajar*,
yaitu, mereka memiliki jumlah yang sama
dan terjadi dalam urutan yang sesuai.
Misalnya, dalam *penandaan bagian ucapan (PoS tagging)*,
kita menganotasi setiap kata dalam sebuah kalimat
dengan bagian ucapan yang sesuai,
yaitu, "kata benda" atau "obyek langsung".
Atau, kita mungkin ingin tahu
kelompok kata berurutan mana yang merujuk pada entitas bernama,
seperti *orang*, *tempat*, atau *organisasi*.
Dalam contoh sederhana berikut ini,
kita mungkin hanya ingin menunjukkan apakah kata dalam kalimat adalah bagian dari entitas bernama (ditandai sebagai "Ent").


```text
Tom makan malam di Washington dengan Sally
Ent  -    -    -     Ent      -    Ent
```


**Pengenalan Suara Otomatis**.
Dalam pengenalan suara, urutan masukan
adalah rekaman audio dari seorang pembicara (:numref:`fig_speech`),
dan keluarannya adalah transkrip dari apa yang dikatakan pembicara tersebut.
Tantangannya adalah ada lebih banyak bingkai audio
(suara biasanya diambil pada 8kHz atau 16kHz)
dibandingkan teks, yaitu, tidak ada korespondensi 1:1 antara audio dan teks,
karena ribuan sampel mungkin
berkorespondensi dengan satu kata yang diucapkan.
Ini adalah masalah pembelajaran urutan-ke-urutan,
di mana keluarannya jauh lebih pendek dari masukannya.
Meskipun manusia sangat pandai mengenali ucapan,
bahkan dari audio berkualitas rendah,
membuat komputer melakukan hal yang sama
adalah tantangan yang besar.

![`-D-e-e-p- L-ea-r-ni-ng-` dalam rekaman audio.](../img/speech.png)
:width:`700px`
:label:`fig_speech`

**Teks ke Suara**.
Ini adalah kebalikan dari pengenalan suara otomatis.
Di sini, masukannya adalah teks dan keluarannya adalah file audio.
Dalam kasus ini, keluarannya jauh lebih panjang dari masukannya.

**Penerjemahan Mesin**.
Berbeda dengan kasus pengenalan suara,
di mana masukan dan keluaran yang sesuai
terjadi dalam urutan yang sama,
dalam penerjemahan mesin,
data yang tidak sejajar menimbulkan tantangan baru.
Di sini urutan masukan dan keluaran
dapat memiliki panjang yang berbeda,
dan wilayah yang sesuai
dari urutan masing-masing
mungkin muncul dalam urutan yang berbeda.
Pertimbangkan contoh ilustratif berikut
tentang kecenderungan khas orang Jerman
untuk menempatkan kata kerja di akhir kalimat:


```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Have you already looked at this excellent textbook?
Wrong alignment:  Have you yourself already this excellent textbook looked at?
```


Banyak masalah terkait muncul dalam tugas pembelajaran lainnya.
Misalnya, menentukan urutan di mana pengguna
membaca halaman web adalah masalah analisis tata letak dua dimensi.
Masalah dialog menunjukkan berbagai komplikasi tambahan,
di mana menentukan apa yang harus dikatakan selanjutnya memerlukan mempertimbangkan
pengetahuan dunia nyata dan keadaan percakapan sebelumnya
di sepanjang jarak waktu yang panjang.
Topik-topik seperti ini adalah area penelitian yang aktif.


### Unsupervised and Self-Supervised Learning (*Pembelajaran Terawasi dan Pembelajaran mandiri*)

Contoh sebelumnya berfokus pada pembelajaran terawasi,
di mana kita memberi model dataset besar
yang mengandung fitur dan nilai label yang sesuai.
Anda bisa menganggap pembelajar terawasi memiliki
pekerjaan yang sangat khusus dan bos yang sangat diktator.
Bos berdiri di belakang pembelajar dan memberi tahu mereka persis apa yang harus dilakukan
di setiap situasi sampai mereka belajar memetakan dari situasi ke tindakan.
Bekerja untuk bos seperti itu terdengar cukup membosankan.
Di sisi lain, menyenangkan bos seperti itu cukup mudah.
Anda hanya perlu mengenali pola secepat mungkin
dan meniru tindakan bos.

Memikirkan situasi yang berlawanan,
bisa frustrasi bekerja untuk bos
yang tidak tahu apa yang mereka inginkan dari Anda.
Namun, jika Anda berencana menjadi ilmuwan data,
Anda sebaiknya terbiasa dengan itu.
Bos mungkin hanya memberikan Anda tumpukan data besar
dan menyuruh Anda untuk *melakukan ilmu data dengan itu!*
Ini terdengar samar karena memang samar.
Kami menyebut kelas masalah ini sebagai *pembelajaran tanpa pengawasan*,
dan jenis serta jumlah pertanyaan yang dapat kita ajukan
hanya dibatasi oleh kreativitas kita.
Kami akan membahas teknik pembelajaran tanpa pengawasan
pada bab-bab selanjutnya.
Untuk membangkitkan selera Anda sekarang,
kami menggambarkan beberapa pertanyaan berikut yang mungkin Anda ajukan.

* Bisakah kita menemukan sejumlah kecil prototipe
yang merangkum data dengan akurat?
Diberikan serangkaian foto, bisakah kita mengelompokkannya menjadi foto pemandangan,
gambar anjing, bayi, kucing, dan puncak gunung?
Demikian pula, diberikan kumpulan aktivitas penjelajahan pengguna,
bisakah kita mengelompokkan mereka menjadi pengguna dengan perilaku serupa?
Masalah ini biasanya dikenal sebagai *pengelompokan*.
* Bisakah kita menemukan sejumlah kecil parameter
yang secara akurat menangkap properti relevan dari data?
Lintasan bola dijelaskan dengan baik
oleh kecepatan, diameter, dan massa bola.
Penjahit telah mengembangkan sejumlah kecil parameter
yang menggambarkan bentuk tubuh manusia dengan cukup akurat
untuk tujuan penyesuaian pakaian.
Masalah ini disebut sebagai *estimasi subruang*.
Jika ketergantungannya linier, disebut *analisis komponen utama*.
* Apakah ada representasi objek (dengan struktur sembarang)
dalam ruang Euklides
sehingga properti simbolik dapat cocok dengan baik?
Ini dapat digunakan untuk menggambarkan entitas dan hubungan mereka,
seperti "Roma" $-$ "Italia" $+$ "Perancis" $=$ "Paris".
* Apakah ada deskripsi tentang penyebab utama
banyak data yang kita amati?
Misalnya, jika kita memiliki data demografis
tentang harga rumah, polusi, kejahatan, lokasi,
pendidikan, dan gaji, bisakah kita menemukan
bagaimana mereka saling terkait hanya berdasarkan data empiris?
Bidang yang berhubungan dengan *kausalitas* dan
*model grafik probabilistik* menangani pertanyaan semacam ini.
* Perkembangan penting dan menarik terbaru dalam pembelajaran tanpa pengawasan
adalah munculnya *model generatif mendalam*.
Model-model ini memperkirakan kepadatan data,
baik secara eksplisit atau *implisit*.
Setelah dilatih, kita dapat menggunakan model generatif
baik untuk menilai contoh berdasarkan seberapa mungkin mereka,
atau untuk membuat contoh sintetis dari distribusi yang dipelajari.
Terobosan pembelajaran mendalam awal dalam pemodelan generatif
datang dengan penemuan *autoencoder variational* :cite:`Kingma.Welling.2014,rezende2014stochastic`
dan dilanjutkan dengan pengembangan *jaringan adversarial generatif* :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
Perkembangan terbaru termasuk aliran normalisasi :cite:`dinh2014nice,dinh2017density` dan
model difusi :cite:`sohl2015deep,song2019generative,ho2020denoising,song2021score`.


Perkembangan lebih lanjut dalam pembelajaran tanpa pengawasan
telah menjadi munculnya *pembelajaran mandiri* (*self-supervised learning*),
teknik yang memanfaatkan beberapa aspek dari data yang tidak berlabel
untuk memberikan supervisi.
Untuk teks, kita dapat melatih model
untuk "mengisi kekosongan"
dengan memprediksi kata-kata yang ditutup secara acak
menggunakan kata-kata di sekitarnya (konteks)
dalam korpus besar tanpa usaha pelabelan :cite:`Devlin.Chang.Lee.ea.2018`!
Untuk gambar, kita mungkin melatih model
untuk memberi tahu posisi relatif
antara dua wilayah yang dipotong
dari gambar yang sama :cite:`Doersch.Gupta.Efros.2015`,
untuk memprediksi bagian gambar yang tertutup
berdasarkan sisa bagian gambar,
atau untuk memprediksi apakah dua contoh
adalah versi terganggu dari gambar yang sama.
Model mandiri sering kali mempelajari representasi
yang selanjutnya dimanfaatkan
dengan menyempurnakan model yang dihasilkan
pada beberapa tugas hilir yang dituhu (*downstream task of interest*).

### Berinteraksi dengan Lingkungan

Sejauh ini, kita belum membahas darimana data sebenarnya berasal,
atau apa yang sebenarnya terjadi ketika model pembelajaran mesin menghasilkan output.
Hal ini karena pembelajaran terawasi dan pembelajaran tanpa pengawasan
tidak menangani masalah ini secara canggih.
Dalam setiap kasus, kita mengumpulkan tumpukan data besar di awal,
kemudian menggerakkan mesin pengenalan pola kita
tanpa berinteraksi lagi dengan lingkungan.
Karena semua pembelajaran berlangsung
setelah algoritma terputus dari lingkungan,
ini terkadang disebut *pembelajaran offline*.
Misalnya, pembelajaran terawasi mengasumsikan
pola interaksi sederhana
yang digambarkan di :numref:`fig_data_collection`.


![Collecting data for supervised learning from an environment.](../img/data-collection.svg)
:label:`fig_data_collection`

Kesederhanaan pembelajaran offline memiliki daya tariknya sendiri.
Keuntungannya adalah kita dapat memfokuskan diri
pada pengenalan pola secara terisolasi,
tanpa perlu khawatir tentang komplikasi yang muncul
dari interaksi dengan lingkungan yang dinamis.
Namun, formulasi masalah ini memiliki keterbatasan.
Jika Anda tumbuh besar dengan membaca novel Robot karya Asimov,
maka Anda mungkin membayangkan agen cerdas buatan
yang tidak hanya mampu membuat prediksi,
tetapi juga mengambil tindakan di dunia nyata.
Kita ingin memikirkan tentang agen *cerdas*,
bukan hanya model prediktif.
Ini berarti kita perlu memikirkan memilih *tindakan*,
bukan hanya membuat prediksi.
Berbeda dengan sekadar prediksi,
tindakan sebenarnya memengaruhi lingkungan.
Jika kita ingin melatih agen cerdas,
kita harus memperhitungkan bagaimana tindakan mereka mungkin
mempengaruhi pengamatan masa depan agen, sehingga pembelajaran offline tidak sesuai.

Memikirkan interaksi dengan lingkungan
membuka serangkaian pertanyaan pemodelan baru.
Berikut adalah hanya beberapa contoh.

* Apakah lingkungan mengingat apa yang kita lakukan sebelumnya?
* Apakah lingkungan ingin membantu kita, misalnya, pengguna yang membacakan teks ke pengenal suara?
* Apakah lingkungan ingin mengalahkan kita, misalnya, spammer yang menyesuaikan email mereka untuk menghindari filter spam?
* Apakah lingkungan memiliki dinamika yang berubah-ubah? Sebagai contoh, apakah data masa depan akan selalu menyerupai masa lalu atau apakah pola akan berubah dari waktu ke waktu, baik secara alami atau sebagai respons terhadap alat otomatis kita?

Pertanyaan-pertanyaan ini menimbulkan masalah *pergeseran distribusi*,
di mana data pelatihan dan pengujian berbeda.
Salah satu contoh ini, yang mungkin banyak dari kita temui, adalah ketika mengikuti ujian yang disusun oleh dosen,
sedangkan pekerjaan rumah disusun oleh asisten pengajar mereka.
Selanjutnya, kita akan mendeskripsikan secara singkat tentang pembelajaran penguatan,
kerangka kerja yang kaya untuk memformulasikan masalah pembelajaran di mana
agen berinteraksi dengan lingkungan.

### Reinforcement Learning (*Pembelajaran penguatan*)

Jika Anda tertarik menggunakan pembelajaran mesin
untuk mengembangkan agen yang berinteraksi dengan lingkungan
dan melakukan tindakan, maka Anda kemungkinan akan berfokus pada *pembelajaran penguatan*.
Ini mungkin termasuk aplikasi untuk robotika,
sistem dialog,
dan bahkan pengembangan kecerdasan buatan (AI)
untuk video game.
*Pembelajaran penguatan mendalam*, yang menerapkan
pembelajaran mendalam pada masalah pembelajaran penguatan,
telah meningkat popularitasnya.
Jaringan Q mendalam yang mengalahkan manusia
dalam permainan Atari hanya menggunakan input visual :cite:`mnih2015human`,
dan program AlphaGo, yang menurunkan juara dunia
di permainan papan Go :cite:`Silver.Huang.Maddison.ea.2016`,
adalah dua contoh terkenal.

Pembelajaran penguatan memberikan pernyataan masalah yang sangat umum
di mana agen berinteraksi dengan lingkungan selama serangkaian langkah waktu.
Pada setiap langkah waktu, agen menerima beberapa *pengamatan*
dari lingkungan dan harus memilih sebuah *tindakan*
yang selanjutnya dikirim kembali ke lingkungan
melalui suatu mekanisme (kadang-kadang disebut *aktuator*), ketika, setelah setiap putaran,
agen menerima imbalan dari lingkungan.
Proses ini diilustrasikan dalam :numref:`fig_rl-environment`.
Kemudian agen menerima pengamatan selanjutnya,
dan memilih tindakan berikutnya, dan seterusnya.
Perilaku agen pembelajaran penguatan diatur oleh *kebijakan*.
Secara singkat, *kebijakan* (policy) hanyalah fungsi yang memetakan
dari pengamatan lingkungan ke tindakan.
Tujuan dari pembelajaran penguatan adalah untuk menghasilkan kebijakan yang baik.

![Interaksi antara pembelajaran penguatan dan lingkungan.](../img/rl-environment.svg)
:label:`fig_rl-environment`

Sulit untuk melebih-lebihkan keumuman
kerangka kerja pembelajaran penguatan.
Sebagai contoh, pembelajaran terawasi (*unsupervised*)
dapat diubah menjadi pembelajaran penguatan.
Katakan kita memiliki masalah klasifikasi.
Kita dapat membuat agen pembelajaran penguatan
dengan satu tindakan yang sesuai dengan setiap kelas.
Kemudian kita dapat membuat lingkungan yang memberikan imbalan
yang persis sama dengan fungsi *kerugian*
dari masalah pembelajaran terawasi asli.

Lebih lanjut, pembelajaran penguatan
juga dapat menangani banyak masalah
yang tidak dapat diatasi oleh pembelajaran terawasi.
Sebagai contoh, dalam pembelajaran terawasi,
kita selalu mengharapkan bahwa input pelatihan
datang bersama dengan label yang benar.
Tetapi dalam pembelajaran penguatan,
kita tidak mengasumsikan bahwa, untuk setiap pengamatan
lingkungan memberi tahu kita tindakan optimal.
Secara umum, kita hanya mendapatkan beberapa imbalan.
Lebih jauh lagi, lingkungan mungkin bahkan tidak memberi tahu kita
tindakan mana yang menyebabkan imbalan.

Pertimbangkan permainan catur.
Sinyal imbalan nyata hanya datang di akhir permainan
ketika kita menang, mendapatkan imbalan, katakanlah, $1$,
atau ketika kita kalah, menerima imbalan sebesar, katakanlah, $-1$.
Jadi pembelajar penguatan harus berurusan
dengan masalah *penugasan kredit*:
menentukan tindakan mana yang harus dikreditkan atau disalahkan untuk hasil akhir.
Hal yang sama berlaku untuk karyawan
yang mendapat promosi pada 11 Oktober.
Promosi tersebut kemungkinan mencerminkan sejumlah
tindakan yang dipilih dengan baik selama tahun sebelumnya.
Mendapatkan promosi di masa depan memerlukan memahami
tindakan mana di sepanjang jalan yang menyebabkan promosi sebelumnya.

Pembelajar penguatan juga mungkin harus berurusan
dengan masalah pengamatan parsial.
Yaitu, pengamatan saat ini mungkin tidak
memberitahu Anda segalanya tentang keadaan Anda saat ini.
Katakanlah robot pembersih Anda menemukan dirinya terjebak
di salah satu dari banyak lemari yang identik di rumah Anda.
Menyelamatkan robot melibatkan mengetahui
lokasi tepatnya yang mungkin memerlukan mempertimbangkan pengamatan sebelumnya sebelum masuk ke lemari.

Akhirnya, pada titik tertentu, pembelajar penguatan
mungkin tahu satu kebijakan yang baik,
tetapi mungkin ada banyak kebijakan lain yang lebih baik
yang belum pernah dicoba oleh agen.
Pembelajar penguatan harus terus memilih
apakah akan *menggunakan* strategi terbaik yang saat ini diketahui sebagai kebijakan,
atau untuk *menjelajahi* ruang strategi,
berpotensi menyerahkan beberapa imbalan jangka pendek
sebagai tukar menukar pengetahuan.

Masalah pembelajaran penguatan umum
memiliki pengaturan yang sangat umum.
Tindakan mempengaruhi pengamatan selanjutnya.
Imbalan hanya diamati ketika mereka sesuai dengan tindakan yang dipilih.
Lingkungan mungkin diamati secara penuh atau sebagian.
Mengatasi semua kompleksitas ini sekaligus mungkin terlalu berlebihan.
Selain itu, tidak setiap masalah praktis menunjukkan semua kompleksitas ini.
Akibatnya, para peneliti telah mempelajari sejumlah
kasus khusus dari masalah pembelajaran penguatan.

Ketika lingkungan sepenuhnya diamati,
kita menyebut masalah pembelajaran penguatan sebagai *proses keputusan Markov* (Markov decision process).
Ketika keadaan tidak tergantung pada tindakan sebelumnya,
kita menyebutnya sebagai *contextual bandit problem*.
Ketika tidak ada keadaan, hanya sekumpulan tindakan yang tersedia
dengan imbalan yang awalnya tidak diketahui, kita memiliki masalah klasik *multi-armed bandit problem*.

## Roots

Kita baru saja meninjau sebagian kecil dari masalah
yang dapat ditangani oleh pembelajaran mesin.
Untuk berbagai set masalah pembelajaran mesin,
pembelajaran mendalam menyediakan alat yang kuat untuk solusinya.
Meskipun banyak metode pembelajaran mendalam adalah penemuan terbaru,
ide-ide inti di balik pembelajaran dari data
telah dipelajari selama berabad-abad.
Faktanya, manusia telah memiliki keinginan untuk menganalisis data
dan memprediksi hasil di masa depan selama 
berabad-abad, dan keinginan inilah yang menjadi akar dari banyak ilmu pengetahuan alam dan matematika.
Dua contoh adalah distribusi Bernoulli, yang dinamai setelah
[Jacob Bernoulli (1655--1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli),
dan distribusi Gaussian yang ditemukan
oleh [Carl Friedrich Gauss (1777--1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss).
Gauss menemukan, misalnya, algoritma kuadrat terkecil rata-rata,
yang masih digunakan hari ini untuk banyak masalah
dari perhitungan asuransi hingga diagnostik medis.
Alat-alat seperti itu meningkatkan pendekatan eksperimental
dalam ilmu pengetahuan alam---misalnya, hukum Ohm
yang menghubungkan arus dan tegangan dalam resistor
dijelaskan dengan sempurna oleh model linier.

Bahkan di abad pertengahan, para matematikawan
memiliki intuisi tajam tentang perkiraan.
Misalnya, buku geometri [Jacob Köbel (1460--1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)
mengilustrasikan rata-rata panjang kaki 16 pria dewasa
untuk memperkirakan panjang kaki yang khas dalam populasi (:numref:`fig_koebel`).

![Memperkirakan panjang kaki.](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`



Ketika sekelompok individu keluar dari sebuah gereja,
16 pria dewasa diminta untuk berbaris dalam satu baris
dan ukuran kaki mereka diukur.
Jumlah pengukuran ini kemudian dibagi dengan 16
untuk memperoleh perkiraan untuk apa yang sekarang disebut satu kaki.
"Algoritma" ini kemudian ditingkatkan
untuk mengatasi kaki yang cacat;
Dua pria dengan kaki terpendek dan terpanjang dikirim pergi,
menghitung rata-rata hanya dari sisanya.
Ini adalah salah satu contoh terawal
dari perkiraan rata-rata yang dipangkas.

Statistik benar-benar berkembang dengan ketersediaan dan pengumpulan data.
Salah satu pelopornya, [Ronald Fisher (1890--1962)](https://en.wikipedia.org/wiki/Ronald_Fisher),
berkontribusi secara signifikan terhadap teorinya
dan juga aplikasinya dalam genetika.
Banyak algoritmanya (seperti analisis diskriminan linier)
dan konsep-konsepnya (seperti matriks informasi Fisher)
masih memiliki tempat terkemuka
dalam dasar-dasar statistik modern.
Bahkan sumber data yang dia rilis memiliki dampak yang berkelanjutan.
Dataset Iris yang Fisher rilis pada tahun 1936
masih terkadang digunakan untuk mendemonstrasikan
algoritma pembelajaran mesin.
Fisher juga merupakan pendukung eugenika,
yang seharusnya mengingatkan kita bahwa penggunaan ilmu data yang bermoral meragukan
memiliki sejarah yang panjang dan bertahan lama seperti halnya penggunaannya yang produktif
di industri dan ilmu pengetahuan alam.

Pengaruh lain untuk pembelajaran mesin
datang dari teori informasi
[Claude Shannon (1916--2001)](https://en.wikipedia.org/wiki/Claude_Shannon)
dan teori komputasi yang diajukan oleh
[Alan Turing (1912--1954)](https://en.wikipedia.org/wiki/Alan_Turing).
Turing mengajukan pertanyaan "dapatkah mesin berpikir?"
dalam makalah terkenalnya *Computing Machinery and Intelligence* :cite:`Turing.1950`.
Mendeskripsikan apa yang sekarang dikenal sebagai tes Turing, dia mengusulkan bahwa sebuah mesin
dapat dianggap *cerdas* jika sulit
bagi evaluator manusia untuk membedakan antara balasan
dari mesin dan balasan dari manusia, berdasarkan interaksi teks semata.


Pengaruh lebih lanjut datang dari neurosains dan psikologi.
Bagaimanapun, manusia jelas menunjukkan perilaku cerdas.
Banyak ilmuwan telah bertanya apakah seseorang dapat menjelaskan
dan mungkin merekayasa ulang kapasitas ini.
Salah satu algoritma terinspirasi biologi pertama
dirumuskan oleh [Donald Hebb (1904--1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).
Dalam bukunya yang revolusioner *The Organization of Behavior* :cite:`Hebb.1949`,
dia berpendapat bahwa neuron belajar melalui penguatan positif.
Ini menjadi dikenal sebagai aturan pembelajaran Hebbian.
Ide-ide ini menginspirasi pekerjaan selanjutnya, seperti
algoritma pembelajaran perceptron Rosenblatt,
dan meletakkan dasar dari banyak algoritma turunan gradien stokastik
yang mendukung pembelajaran mendalam saat ini:
memperkuat perilaku yang diinginkan dan mengurangi perilaku yang tidak diinginkan
untuk mendapatkan pengaturan parameter yang baik dalam jaringan saraf.

Inspirasi biologis adalah yang memberi *jaringan saraf* namanya.
Selama lebih dari satu abad (berawal dari model Alexander Bain, 1873,
dan James Sherrington, 1890), peneliti telah mencoba untuk merakit
sirkuit komputasi yang menyerupai jaringan neuron yang berinteraksi.
Seiring waktu, interpretasi biologi menjadi kurang harfiah,
tetapi nama itu tetap. Di intinya terdapat beberapa prinsip kunci
yang dapat ditemukan di sebagian besar jaringan saat ini:

* Pergantian unit pengolahan linier dan nonlinier, sering disebut sebagai *lapisan*.
* Penggunaan aturan rantai (juga dikenal sebagai *backpropagation*) untuk menyesuaikan parameter dalam seluruh jaringan sekaligus.

Setelah kemajuan awal yang cepat, penelitian dalam jaringan saraf
mengalami kelesuan dari sekitar tahun 1995 hingga 2005.
Ini terutama disebabkan oleh dua alasan.
Pertama, pelatihan jaringan sangat mahal secara komputasi.
Sementara memori akses acak melimpah di akhir abad lalu,
kekuatan komputasi sangat langka.
Kedua, kumpulan data relatif kecil.
Faktanya, dataset Iris Fisher dari tahun 1936
masih merupakan alat populer untuk menguji efikasi algoritme.
Dataset MNIST dengan 60.000 digit tulisan tangan dianggap besar.

Mengingat kelangkaan data dan komputasi,
alat statistik kuat seperti metode kernel,
pohon keputusan, dan model grafis
terbukti secara empiris lebih unggul dalam banyak aplikasi.
Selain itu, tidak seperti jaringan saraf,
mereka tidak memerlukan waktu berhari-hari untuk dilatih
dan memberikan hasil yang dapat diprediksi
dengan jaminan teoritis yang kuat.



## perjalanan menuju *Deep Learning* (Pembelajaran Mendalam)

Banyak perubahan terjadi dengan ketersediaan
jumlah data yang masif,
berkat World Wide Web,
munculnya perusahaan yang melayani
ratusan juta pengguna secara online,
penyebaran sensor berkualitas tinggi yang murah,
penyimpanan data yang murah (hukum Kryder),
dan komputasi yang murah (hukum Moore).
Secara khusus, lanskap komputasi dalam pembelajaran mendalam
direvolusi oleh kemajuan dalam GPU yang awalnya dirancang untuk permainan komputer.
Tiba-tiba algoritma dan model
yang tampaknya tidak layak secara komputasi
berada dalam jangkauan.
Ini diilustrasikan dengan terbaik dalam :numref:`tab_intro_decade`.

:Dataset vs. memori komputer dan daya komputasi
:label:`tab_intro_decade`


|Dekade|Dataset|Memori|Perhitungan floating point per detik|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (harga rumah di Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (pengenalan karakter optik)|10 MB|10 MF (Intel 80486)|
|2000|10 M (halaman web)|100 MB|1 GF (Intel Core)|
|2010|10 G (periklanan)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (jaringan sosial)|100 GB|1 PF (NVIDIA DGX-2)|



Perhatikan bahwa memori akses acak tidak mengikuti laju pertumbuhan data.
Pada saat yang sama, peningkatan dalam kekuatan komputasi
telah melampaui pertumbuhan dataset.
Ini berarti bahwa model statistik
perlu menjadi lebih efisien dalam penggunaan memori,
dan sehingga mereka bebas untuk menghabiskan lebih banyak siklus komputer
dalam mengoptimalkan parameter, berkat
anggaran komputasi yang meningkat.
Akibatnya, titik optimal dalam pembelajaran mesin dan statistik
berpindah dari model linear (umum) dan metode kernel
ke jaringan saraf dalam.
Ini juga salah satu alasan mengapa banyak pilar utama
pembelajaran mendalam, seperti perceptron berlapis
:cite:`McCulloch.Pitts.1943`, jaringan saraf konvolusional
:cite:`LeCun.Bottou.Bengio.ea.1998`, memori jangka panjang pendek
:cite:`Hochreiter.Schmidhuber.1997`,
dan Q-Learning :cite:`Watkins.Dayan.1992`,
pada dasarnya "ditemukan kembali" dalam dekade terakhir,
setelah terletak relatif tidak aktif untuk waktu yang cukup lama.

Kemajuan terbaru dalam model statistik, aplikasi, dan algoritma
kadang-kadang disamakan dengan ledakan Kambrium:
momen kemajuan cepat dalam evolusi spesies.
Memang, keadaan terkini bukan hanya sekedar akibat
dari sumber daya yang tersedia yang diterapkan pada algoritma berusia puluhan tahun.
Perhatikan bahwa daftar ide di bawah ini hampir tidak menyentuh permukaan
dari apa yang telah membantu peneliti mencapai kemajuan luar biasa
selama dekade terakhir.



* Metode baru untuk kontrol kapasitas, seperti *dropout*
  :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`,
  telah membantu untuk mengurangi overfitting.
  Di sini, *noise* disuntikkan :cite:`Bishop.1995`
  ke seluruh jaringan saraf selama pelatihan.
* *Mekanisme perhatian* (Attention mechanisms) menyelesaikan masalah kedua
  yang telah mengganggu statistik selama lebih dari satu abad:
  bagaimana meningkatkan memori dan kompleksitas sistem tanpa
  meningkatkan jumlah parameter yang dapat dipelajari.
  Peneliti menemukan solusi elegan
  dengan menggunakan apa yang hanya dapat dilihat sebagai
  *struktur penunjuk yang dapat dipelajari* (learnable pointer structure) :cite:`Bahdanau.Cho.Bengio.2014`.
  Daripada harus mengingat seluruh urutan teks, misalnya,
  untuk penerjemahan mesin dalam representasi berdimensi tetap,
  yang perlu disimpan hanyalah penunjuk ke keadaan antara
  proses penerjemahan. Ini memungkinkan peningkatan akurasi yang signifikan
  untuk urutan panjang, karena model
  tidak lagi perlu mengingat seluruh urutan sebelum
  memulai generasi yang baru.
* Dibangun hanya berdasarkan mekanisme perhatian,
  arsitektur *Transformer* :cite:`Vaswani.Shazeer.Parmar.ea.2017` telah menunjukkan perilaku *scaling* yang unggul: kinerjanya meningkat dengan peningkatan ukuran dataset, ukuran model, dan jumlah komputasi pelatihan :cite:`kaplan2020scaling`. Arsitektur ini telah menunjukkan keberhasilan yang menarik dalam berbagai bidang,
  seperti pemrosesan bahasa alami :cite:`Devlin.Chang.Lee.ea.2018,brown2020language`, penglihatan komputer :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`, pengenalan suara :cite:`gulati2020conformer`, pembelajaran penguatan :cite:`chen2021decision`, dan jaringan saraf grafis :cite:`dwivedi2020generalization`. Sebagai contoh, sebuah Transformer tunggal yang dilatih sebelumnya pada modalitas
  yang beragam seperti teks, gambar, torsi gabungan, dan tekanan tombol
  dapat bermain Atari, memberi keterangan pada gambar, mengobrol,
  dan mengontrol robot :cite:`reed2022generalist`.

* Pemodelan probabilitas dari urutan teks, *model bahasa* dapat memprediksi teks berdasarkan teks lain. Peningkatan data, model, dan komputasi telah membuka semakin banyak kemampuan model bahasa untuk melakukan tugas yang diinginkan melalui generasi teks yang mirip manusia berdasarkan teks masukan :cite:`brown2020language,rae2021scaling,hoffmann2022training,chowdhery2022palm,openai2023gpt4,anil2023palm,touvron2023llama,touvron2023llama2`. Sebagai contoh, menyelaraskan model bahasa dengan niat manusia :cite:`ouyang2022training`, [ChatGPT](https://chat.openai.com/) dari OpenAI memungkinkan pengguna berinteraksi dengannya secara percakapan untuk menyelesaikan masalah, seperti debugging kode dan penulisan kreatif.
* Desain multi-tahap, misalnya, melalui jaringan memori
  :cite:`Sukhbaatar.Weston.Fergus.ea.2015`
  dan pemrogram-penerjemah saraf :cite:`Reed.De-Freitas.2015`
  memungkinkan perumus model statistik untuk mendeskripsikan pendekatan iteratif untuk penalaran.
  Alat-alat ini memungkinkan keadaan internal dari jaringan saraf dalam
  untuk dimodifikasi berulang kali,
  sehingga melaksanakan langkah selanjutnya
  dalam rantai penalaran, sama seperti prosesor
  dapat memodifikasi memori untuk sebuah perhitungan.
* Pengembangan kunci dalam *pemodelan generatif mendalam* adalah penemuan
  *jaringan adversarial generatif*
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
  Secara tradisional, metode statistik untuk estimasi kepadatan
  dan model generatif berfokus pada menemukan distribusi probabilitas yang tepat
  dan algoritme (seringkali pendekatan) untuk sampling dari mereka.
  Akibatnya, algoritme ini sebagian besar terbatas oleh kurangnya
  fleksibilitas yang melekat pada model statistik.
  Inovasi penting dalam jaringan adversarial generatif adalah menggantikan sampel
  dengan algoritme sembarang dengan parameter yang dapat dibedakan.
  Parameter ini kemudian disesuaikan sedemikian rupa sehingga diskriminator
  (secara efektif tes dua-sampel) tidak dapat membedakan data palsu dari data nyata.
  Melalui kemampuan untuk menggunakan algoritme sembarang untuk menghasilkan data,
  estimasi kepadatan dibuka untuk berbagai teknik.
  Contoh zebra yang berlari :cite:`Zhu.Park.Isola.ea.2017`
  dan wajah selebriti palsu :cite:`Karras.Aila.Laine.ea.2017`
  adalah masing-masing kesaksian atas kemajuan ini.
  Bahkan penggambar amatir dapat menghasilkan
  gambar fotorealistis hanya berdasarkan sketsa yang menggambarkan tata letak sebuah adegan :cite:`Park.Liu.Wang.ea.2019`. 

* Selanjutnya, sementara proses difusi secara bertahap menambahkan kebisingan acak ke sampel data, *model difusi* :cite:`sohl2015deep,ho2020denoising` belajar proses denoising untuk secara bertahap membangun sampel data dari kebisingan acak, membalikkan proses difusi. Mereka telah mulai menggantikan jaringan adversarial generatif dalam model generatif mendalam yang lebih baru, seperti dalam DALL-E 2 :cite:`ramesh2022hierarchical` dan Imagen :cite:`saharia2022photorealistic` untuk seni kreatif dan generasi gambar berdasarkan deskripsi teks.
* Dalam banyak kasus, satu GPU tidak cukup untuk memproses jumlah data besar yang tersedia untuk pelatihan.
  Selama dekade terakhir, kemampuan untuk membangun algoritma pelatihan paralel dan
  terdistribusi telah meningkat secara signifikan.
  Salah satu tantangan kunci dalam merancang algoritma yang skalabel
  adalah bahwa kuda kerja optimisasi pembelajaran mendalam,
  stochastic gradient descent, bergantung pada
  minibatch data yang relatif kecil untuk diproses.
  Pada saat yang sama, batch kecil membatasi efisiensi GPU.
  Oleh karena itu, pelatihan pada 1.024 GPU dengan ukuran minibatch,
  katakanlah, 32 gambar per batch berjumlah agregat minibatch
  sekitar 32.000 gambar. Pekerjaan, pertama oleh :citet:`Li.2017`
  dan selanjutnya oleh :citet:`You.Gitman.Ginsburg.2017`
  dan :citet:`Jia.Song.He.ea.2018` mendorong ukuran hingga 64.000 pengamatan,
  mengurangi waktu pelatihan untuk model ResNet-50
  pada dataset ImageNet menjadi kurang dari 7 menit.
  Sebagai perbandingan, waktu pelatihan awalnya sekitar hari.
* Kemampuan untuk memparalelkan perhitungan
  juga telah berkontribusi pada kemajuan dalam *pembelajaran penguatan*.
  Ini telah mengarah pada kemajuan signifikan dalam komputer yang mencapai
  kinerja superhuman pada tugas seperti Go, permainan Atari,
  Starcraft, dan dalam simulasi fisika (misalnya, menggunakan MuJoCo)
  di mana simulator lingkungan tersedia.
  Lihat, misalnya, :citet:`Silver.Huang.Maddison.ea.2016` untuk deskripsi
  prestasi semacam itu di AlphaGo. Singkatnya,
  pembelajaran penguatan bekerja terbaik
  jika banyak tupel (state, action, reward) tersedia.
  Simulasi menyediakan jalan seperti itu.
* Kerangka kerja pembelajaran mendalam telah memainkan peran penting
  dalam menyebarluaskan ide.
  Generasi pertama kerangka kerja sumber terbuka
  untuk pemodelan jaringan saraf terdiri dari
  [Caffe](https://github.com/BVLC/caffe),
  [Torch](https://github.com/torch), dan
  [Theano](https://github.com/Theano/Theano).
  Banyak makalah penting ditulis menggunakan alat-alat ini.
  Sekarang ini telah digantikan oleh
  [TensorFlow](https://github.com/tensorflow/tensorflow) (sering digunakan melalui API tingkat tinggi [Keras](https://github.com/keras-team/keras)), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), dan [Apache MXNet](https://github.com/apache/incubator-mxnet).
  Generasi ketiga dari kerangka kerja terdiri dari
  alat *imperatif* untuk pembelajaran mendalam,
  tren yang mungkin dipicu oleh [Chainer](https://github.com/chainer/chainer),
  yang menggunakan sintaks mirip Python NumPy untuk mendeskripsikan model.
  Ide ini diadopsi oleh [PyTorch](https://github.com/pytorch/pytorch),
  [API Gluon](https://github.com/apache/incubator-mxnet) dari MXNet,
  dan [JAX](https://github.com/google/jax).



Pembagian tugas antara peneliti sistem yang membangun alat yang lebih baik
dan perumus model statistik yang membangun jaringan saraf yang lebih baik
telah sangat menyederhanakan hal-hal. Sebagai contoh,
melatih model regresi logistik linear
dulu merupakan masalah pekerjaan rumah yang tidak sepele,
layak untuk diberikan kepada mahasiswa baru doktoral pembelajaran mesin
di Universitas Carnegie Mellon pada tahun 2014.
Sekarang, tugas ini dapat diselesaikan
dengan kurang dari 10 baris kode,
menempatkannya dengan kuat dalam jangkauan setiap pemrogram.


## Kisah Sukses

Kecerdasan buatan memiliki sejarah panjang dalam memberikan hasil
yang sulit dicapai dengan cara lain.
Sebagai contoh, sistem pengurutan surat
menggunakan pengenalan karakter optik
telah dikerahkan sejak tahun 1990-an.
Ini adalah, bagaimanapun juga, sumber
dari dataset MNIST terkenal
dari digit tulisan tangan.
Hal yang sama berlaku untuk membaca cek untuk deposito bank dan penilaian
kelayakan kredit dari pelamar.
Transaksi keuangan diperiksa untuk kecurangan secara otomatis.
Ini merupakan tulang punggung dari banyak sistem pembayaran e-commerce,
seperti PayPal, Stripe, AliPay, WeChat, Apple, Visa, dan MasterCard.
Program komputer untuk catur telah kompetitif selama beberapa dekade.
Pembelajaran mesin mendukung pencarian, rekomendasi, personalisasi,
dan peringkat di Internet.
Dengan kata lain, pembelajaran mesin ada di mana-mana, meskipun sering tersembunyi dari pandangan.

Baru-baru ini kecerdasan buatan
telah menjadi sorotan, sebagian besar karena
solusi untuk masalah
yang dianggap tidak dapat diatasi sebelumnya
dan yang secara langsung terkait dengan konsumen.
Banyak kemajuan semacam itu diatribusikan pada pembelajaran mendalam.

* Asisten cerdas, seperti Siri dari Apple,
  Alexa dari Amazon, dan asisten Google,
  mampu merespons permintaan suara
  dengan tingkat akurasi yang wajar.
  Ini termasuk pekerjaan sederhana, seperti menyalakan saklar lampu,
  dan tugas yang lebih kompleks, seperti mengatur janji dengan tukang cukur
  dan menawarkan dialog dukungan telepon.
  Ini kemungkinan besar tanda yang paling nyata
  bahwa AI memengaruhi kehidupan kita.
* Salah satu bahan kunci dalam asisten digital
  adalah kemampuan mereka untuk mengenali ucapan dengan akurat.
  Akurasi sistem semacam itu telah bertambah
  hingga mencapai paritas dengan manusia
  untuk beberapa aplikasi :cite:`Xiong.Wu.Alleva.ea.2018`.
* Pengenalan objek juga telah menempuh perjalanan panjang.
  Mengidentifikasi objek dalam gambar
  merupakan tugas yang cukup menantang pada tahun 2010.
  Dalam benchmark ImageNet, para peneliti dari NEC Labs
  dan University of Illinois at Urbana-Champaign
  mencapai tingkat kesalahan lima besar sebesar 28% :cite:`Lin.Lv.Zhu.ea.2010`.
  Pada tahun 2017, tingkat kesalahan ini berkurang menjadi 2,25% :cite:`Hu.Shen.Sun.2018`.
  Hasil yang menakjubkan juga telah dicapai
  untuk mengidentifikasi suara burung dan mendiagnosis kanker kulit.
* Keahlian dalam permainan biasanya menjadi
  tolak ukur kemampuan manusia.
  Mulai dari TD-Gammon, program untuk bermain backgammon
  menggunakan pembelajaran penguatan perbedaan temporal,
  kemajuan algoritmik dan komputasi
  telah mengarah pada algoritma untuk berbagai aplikasi.
  Dibandingkan dengan backgammon, catur memiliki
  ruang keadaan dan set tindakan yang jauh lebih kompleks.
  DeepBlue mengalahkan Garry Kasparov dengan menggunakan paralelisme besar,
  perangkat keras khusus dan pencarian yang efisien
  melalui pohon permainan :cite:`Campbell.Hoane-Jr.Hsu.2002`.
  Go masih lebih sulit, karena ruang keadaannya yang sangat besar.
  AlphaGo mencapai paritas manusia pada tahun 2015,
  menggunakan pembelajaran mendalam dikombinasikan dengan sampling pohon Monte Carlo :cite:`Silver.Huang.Maddison.ea.2016`.
  Tantangan dalam Poker adalah bahwa ruang keadaannya besar
  dan hanya diamati sebagian
  (kita tidak tahu kartu lawan).
  Libratus melebihi kinerja manusia dalam Poker
  menggunakan strategi yang terstruktur dengan efisien :cite:`Brown.Sandholm.2017`.
* Indikasi lain dari kemajuan dalam AI
  adalah munculnya kendaraan otonom.
  Meskipun otonomi penuh belum tercapai,
  kemajuan yang sangat baik telah dibuat ke arah ini,
  dengan perusahaan seperti Tesla, NVIDIA,
  dan Waymo mengirimkan produk
  yang memungkinkan otonomi sebagian.
  Yang membuat otonomi penuh sangat menantang
  adalah mengemudi yang tepat membutuhkan
  kemampuan untuk mempersepsi, berpikir,
  dan menggabungkan aturan ke dalam sistem.
  Saat ini, pembelajaran mendalam terutama digunakan
  dalam aspek visual dari masalah ini.
  Sisanya sangat disesuaikan oleh insinyur.

Ini baru menggores permukaan dari aplikasi signifikan pembelajaran mesin. Misalnya, robotika, logistik, biologi komputasional, fisika partikel, dan astronomi mendapatkan beberapa kemajuan paling mengesankan mereka setidaknya sebagian dari pembelajaran mesin, yang dengan demikian menjadi alat yang lazim digunakan oleh para insinyur dan ilmuwan.

Seringkali, pertanyaan tentang datangnya kiamat AI dan kemungkinan *singularity* diangkat dalam artikel non-teknis. Ketakutannya adalah bahwa entah bagaimana sistem pembelajaran mesin akan menjadi sadar dan membuat keputusan, secara independen dari pemrogramnya, yang secara langsung memengaruhi kehidupan manusia. Hingga tingkat tertentu, AI sudah memengaruhi mata pencaharian manusia secara langsung: kelayakan kredit dinilai secara otomatis, autopilot sebagian besar mengendalikan kendaraan, keputusan tentang apakah akan memberikan jaminan menggunakan data statistik sebagai input. Lebih ringan, kita dapat meminta Alexa untuk menyalakan mesin kopi.

Untungnya, kita masih jauh dari sistem AI yang sadar yang bisa dengan sengaja memanipulasi penciptanya. Pertama, sistem AI dirancang, dilatih, dan diterapkan dengan cara yang spesifik dan berorientasi tujuan. Meskipun perilaku mereka mungkin memberikan ilusi kecerdasan umum, itu adalah kombinasi dari aturan, heuristik, dan model statistik yang mendasari desainnya. Kedua, saat ini, tidak ada alat untuk *kecerdasan umum buatan* yang mampu meningkatkan diri, merenungkan dirinya sendiri, serta memodifikasi, memperluas, dan meningkatkan arsitekturnya sendiri sambil mencoba menyelesaikan tugas-tugas umum.

Kekhawatiran yang jauh lebih mendesak adalah bagaimana AI digunakan dalam kehidupan kita sehari-hari. Kemungkinan banyak tugas rutin yang saat ini dilakukan manusia bisa dan akan diotomatisasi. Robot pertanian kemungkinan akan mengurangi biaya bagi petani organik tetapi mereka juga akan mengotomatiskan operasi pemanenan. Fase revolusi industri ini mungkin memiliki dampak yang mendalam bagi sebagian besar masyarakat, karena pekerjaan menial memberikan banyak lapangan kerja di banyak negara. Lebih lanjut, model statistik, jika diterapkan tanpa hati-hati, dapat menyebabkan bias ras, gender, atau usia dan menimbulkan kekhawatiran yang masuk akal tentang keadilan prosedural jika diotomatisasi untuk mendorong keputusan-keputusan penting. Penting untuk memastikan bahwa algoritma ini digunakan dengan hati-hati. Berdasarkan apa yang kita ketahui hari ini, ini tampaknya menjadi kekhawatiran yang jauh lebih mendesak daripada potensi kecerdasan super jahat untuk menghancurkan umat manusia.



## Esensi Deep Learning

Sejauh ini, kita telah berbicara secara umum tentang pembelajaran mesin. Pembelajaran mendalam (*deep learning*) adalah subset dari pembelajaran mesin yang berhubungan dengan model berbasis jaringan saraf berlapis-lapis. Disebut *deep* karena model ini mempelajari banyak *lapisan* transformasi. Meskipun ini terdengar sempit, pembelajaran mendalam telah melahirkan beragam model, teknik, perumusan masalah, dan aplikasi yang memusingkan. Banyak intuisi telah dikembangkan untuk menjelaskan manfaat dari kedalaman ini. Dapat dikatakan bahwa semua pembelajaran mesin memiliki banyak lapisan komputasi, yang pertama terdiri dari langkah-langkah pemrosesan fitur. Yang membedakan pembelajaran mendalam adalah bahwa operasi yang dipelajari di setiap lapisan representasi dipelajari bersama dari data.

Masalah-masalah yang telah kita bahas sejauh ini, seperti belajar dari sinyal audio mentah, nilai piksel mentah dari gambar, atau pemetaan antara kalimat dengan panjang yang sewenang-wenang dan padanannya dalam bahasa asing, adalah masalah-masalah di mana pembelajaran mendalam unggul dan metode tradisional gagal. Ternyata, model berlapis-lapis ini mampu menangani data persepsi tingkat rendah dengan cara yang tidak dapat dilakukan oleh alat sebelumnya. Mungkin kesamaan paling signifikan dalam metode pembelajaran mendalam adalah *pelatihan end-to-end*. Artinya, alih-alih merakit sistem berdasarkan komponen yang disesuaikan secara individual, kita membangun sistem tersebut dan kemudian menyetel kinerjanya secara bersamaan. Misalnya, dalam visi komputer, para ilmuwan dulu memisahkan proses *rekayasa fitur* dari proses membangun model pembelajaran mesin. Detektor tepi Canny :cite:`Canny.1987` dan ekstraktor fitur SIFT dari Lowe :cite:`Lowe.2004` memimpin selama lebih dari satu dekade sebagai algoritma untuk memetakan gambar ke dalam vektor fitur. Di masa lalu, bagian penting dari penerapan pembelajaran mesin pada masalah ini terdiri dari menemukan cara-cara rekayasa manual untuk mengubah data ke dalam bentuk yang sesuai dengan model dangkal. Sayangnya, hanya ada sedikit yang bisa dicapai manusia dengan kecerdasan mereka dibandingkan dengan evaluasi konsisten atas jutaan pilihan yang dilakukan secara otomatis oleh algoritma. Ketika pembelajaran mendalam muncul, ekstraktor fitur ini digantikan oleh filter yang disetel secara otomatis yang menghasilkan akurasi yang lebih baik.

Oleh karena itu, salah satu keunggulan utama dari pembelajaran mendalam adalah menggantikan tidak hanya model dangkal di akhir jalur pembelajaran tradisional, tetapi juga proses rekayasa fitur yang memakan waktu. Selain itu, dengan menggantikan sebagian besar pra-pemrosesan yang spesifik domain, pembelajaran mendalam telah menghilangkan banyak batasan yang sebelumnya memisahkan visi komputer, pengenalan suara, pemrosesan bahasa alami, informatika medis, dan bidang aplikasi lainnya, sehingga menawarkan satu set alat terpadu untuk menangani berbagai masalah.

Selain pelatihan end-to-end, kita juga mengalami transisi dari deskripsi statistik parametrik ke model nonparametrik sepenuhnya. Ketika data langka, kita perlu bergantung pada asumsi penyederhanaan tentang realitas untuk mendapatkan model yang berguna. Ketika data melimpah, asumsi ini dapat digantikan oleh model nonparametrik yang lebih cocok dengan data. Sampai batas tertentu, ini mencerminkan kemajuan yang dialami fisika pada pertengahan abad sebelumnya dengan tersedianya komputer. Alih-alih memecahkan dengan tangan aproksimasi parametrik tentang bagaimana elektron berperilaku, sekarang kita dapat menggunakan simulasi numerik dari persamaan diferensial parsial terkait. Ini menghasilkan model yang jauh lebih akurat, meskipun seringkali dengan mengorbankan interpretasi.

Perbedaan lain dari pekerjaan sebelumnya adalah penerimaan terhadap solusi yang tidak optimal, menangani masalah optimasi nonlinier nonkonveks, dan kesediaan untuk mencoba sesuatu sebelum membuktikannya. Empirisme baru ini dalam menangani masalah statistik, dikombinasikan dengan masuknya talenta yang cepat, telah mendorong kemajuan pesat dalam pengembangan algoritma praktis, meskipun dalam banyak kasus dengan mengorbankan modifikasi dan penemuan ulang alat yang telah ada selama beberapa dekade.

Pada akhirnya, komunitas pembelajaran mendalam bangga berbagi alat melintasi batas akademis dan perusahaan, merilis banyak pustaka (Library), model statistik, dan jaringan terlatih (*Trained Networks*) yang luar biasa sebagai sumber terbuka. Dengan semangat inilah notebook yang membentuk buku ini tersedia secara gratis untuk distribusi dan penggunaan. Kami telah bekerja keras untuk menurunkan hambatan akses bagi siapa pun yang ingin belajar tentang pembelajaran mendalam dan kami berharap pembaca kami akan mendapatkan manfaat darinya.


## Ringkasan

Pembelajaran mesin mempelajari bagaimana sistem komputer dapat memanfaatkan pengalaman (sering kali data) untuk meningkatkan kinerja pada tugas-tugas tertentu. Ini menggabungkan ide-ide dari statistik, penambangan data, dan optimisasi. Sering kali, digunakan sebagai sarana untuk menerapkan solusi AI. Sebagai kelas dari pembelajaran mesin, pembelajaran representasional berfokus pada bagaimana secara otomatis menemukan cara yang tepat untuk merepresentasikan data. Dipandang sebagai pembelajaran representasi multi-level melalui pembelajaran banyak lapisan transformasi, pembelajaran mendalam (*deep learning*) menggantikan tidak hanya model dangkal di akhir jalur pembelajaran mesin tradisional, tetapi juga proses rekayasa fitur yang memakan banyak tenaga. Sebagian besar kemajuan baru-baru ini dalam pembelajaran mendalam dipicu oleh melimpahnya data yang berasal dari sensor murah dan aplikasi berskala Internet, serta oleh kemajuan signifikan dalam komputasi, terutama melalui GPU. Selain itu, tersedianya kerangka kerja pembelajaran mendalam yang efisien telah membuat desain dan implementasi optimisasi sistem secara keseluruhan menjadi lebih mudah, dan ini merupakan komponen kunci dalam mendapatkan kinerja tinggi.

## Latihan

1. Bagian kode mana yang saat ini Anda tulis yang bisa "dipelajari", yaitu, ditingkatkan dengan mempelajari dan secara otomatis menentukan pilihan desain yang dibuat dalam kode Anda? Apakah kode Anda mencakup pilihan desain heuristik? Data apa yang mungkin Anda butuhkan untuk mempelajari perilaku yang diinginkan?
1. Masalah apa yang Anda temui yang memiliki banyak contoh untuk solusinya, namun tidak ada cara spesifik untuk mengotomasikannya? Ini mungkin merupakan kandidat utama untuk menggunakan pembelajaran mendalam.
1. Jelaskan hubungan antara algoritma, data, dan komputasi. Bagaimana karakteristik data dan sumber daya komputasi yang tersedia saat ini memengaruhi kesesuaian berbagai algoritma?
1. Sebutkan beberapa pengaturan di mana pelatihan end-to-end saat ini bukan merupakan pendekatan default, tetapi mungkin bermanfaat.

[Diskusi](https://discuss.d2l.ai/t/22)

