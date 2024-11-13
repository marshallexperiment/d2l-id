# Hardware
:label:`sec_hardware`

Membangun sistem dengan performa tinggi memerlukan pemahaman yang baik mengenai algoritma dan model untuk menangkap aspek statistik dari masalah. Pada saat yang sama, juga sangat penting untuk memiliki setidaknya sedikit pengetahuan mengenai hardware yang mendasarinya. Bagian ini bukanlah pengganti untuk kursus yang benar tentang desain hardware dan sistem. Sebaliknya, bagian ini bisa menjadi titik awal untuk memahami mengapa beberapa algoritma lebih efisien daripada yang lain dan bagaimana mencapai throughput yang baik. Desain yang baik bisa dengan mudah membuat perbedaan hingga satu tingkat besar, yang pada gilirannya bisa membuat perbedaan antara mampu melatih sebuah jaringan (misalnya dalam satu minggu) dan tidak sama sekali (dalam 3 bulan, sehingga melewati batas waktu yang ada). 
Kita akan memulai dengan melihat komputer secara umum. Lalu kita akan memperbesar untuk melihat lebih detail pada CPU dan GPU. Terakhir, kita akan memperkecil kembali untuk meninjau bagaimana beberapa komputer dihubungkan dalam pusat server atau di cloud.

![Angka Latensi yang perlu diketahui setiap programmer.](../img/latencynumbers.png)
:label:`fig_latencynumbers`

Pembaca yang tidak sabar mungkin bisa mendapatkan gambaran dari :numref:`fig_latencynumbers`. Gambar ini diambil dari [posting interaktif](https://people.eecs.berkeley.edu/%7Ercs/research/interactive_latency.html) oleh Colin Scott yang memberikan gambaran yang baik mengenai perkembangan selama dekade terakhir. Angka asli berasal dari [ceramah Stanford tahun 2010](https://static.googleusercontent.com/media/research.google.com/en//people/jeff/Stanford-DL-Nov-2010.pdf) oleh Jeff Dean.
Diskusi di bawah ini menjelaskan beberapa alasan untuk angka-angka tersebut dan bagaimana mereka dapat membimbing kita dalam merancang algoritma. Diskusi ini sangat bersifat umum dan singkat. Jelas ini *bukan pengganti* kursus yang sebenarnya melainkan hanya dimaksudkan untuk memberikan cukup informasi agar seorang pemodel statistik dapat membuat keputusan desain yang sesuai. Untuk gambaran mendalam tentang arsitektur komputer, kami merekomendasikan pembaca pada :cite:`Hennessy.Patterson.2011` atau kursus terbaru mengenai subjek ini, seperti kursus yang diberikan oleh [Arste Asanovic](http://inst.eecs.berkeley.edu/%7Ecs152/sp19/).

## Komputer

Sebagian besar peneliti dan praktisi deep learning memiliki akses ke komputer dengan kapasitas memori yang cukup besar, komputasi, beberapa bentuk akselerator seperti GPU, atau beberapa akselerator. Sebuah komputer terdiri dari komponen utama berikut:

* Sebuah prosesor (juga disebut CPU) yang mampu mengeksekusi program yang kita berikan (selain menjalankan sistem operasi dan banyak hal lainnya), biasanya terdiri dari 8 atau lebih core.
* Memori (RAM) untuk menyimpan dan mengambil hasil dari komputasi, seperti vektor bobot dan aktivasi, serta data pelatihan.
* Koneksi jaringan Ethernet (kadang beberapa) dengan kecepatan mulai dari 1 GB/s hingga 100 GB/s. Pada server kelas atas dapat ditemukan antarmuka interkoneksi yang lebih canggih.
* Bus ekspansi kecepatan tinggi (PCIe) untuk menghubungkan sistem dengan satu atau lebih GPU. Server memiliki hingga 8 akselerator, sering kali dihubungkan dalam topologi yang canggih, sementara sistem desktop memiliki 1 atau 2, tergantung anggaran pengguna dan ukuran daya.
* Penyimpanan tahan lama, seperti hard disk drive magnetik, solid state drive, sering kali terhubung menggunakan bus PCIe. Ini menyediakan transfer data pelatihan ke sistem dan penyimpanan checkpoint antara yang diperlukan.

![Konektivitas komponen dari sebuah komputer.](../img/mobo-symbol.svg)
:label:`fig_mobo-symbol`

Seperti yang ditunjukkan pada :numref:`fig_mobo-symbol`, sebagian besar komponen (jaringan, GPU, dan penyimpanan) terhubung ke CPU melalui bus PCIe. Bus ini terdiri dari beberapa jalur yang langsung terhubung ke CPU. Misalnya AMD Threadripper 3 memiliki 64 jalur PCIe 4.0, yang masing-masing mampu melakukan transfer data hingga 16 Gbit/s dalam kedua arah. Memori langsung terhubung ke CPU dengan total bandwidth hingga 100 GB/s.

Ketika kita menjalankan kode pada komputer, kita perlu memindahkan data ke prosesor (CPU atau GPU), melakukan komputasi, dan kemudian memindahkan hasil dari prosesor kembali ke RAM dan penyimpanan tahan lama. Oleh karena itu, untuk mendapatkan performa yang baik, kita harus memastikan bahwa proses ini berjalan mulus tanpa ada satu pun sistem yang menjadi hambatan utama. Misalnya, jika kita tidak dapat memuat gambar dengan cepat, prosesor tidak akan memiliki pekerjaan untuk dilakukan. Demikian pula, jika kita tidak dapat memindahkan matriks dengan cepat ke CPU (atau GPU), elemen pemrosesan akan kelaparan. Akhirnya, jika kita ingin menyinkronkan beberapa komputer melalui jaringan, hal ini tidak boleh memperlambat komputasi. Salah satu opsinya adalah menginterleaving komunikasi dan komputasi. Mari kita lihat berbagai komponen ini lebih detail.


## Memori

Pada dasarnya, memori digunakan untuk menyimpan data yang perlu diakses dengan cepat. Saat ini, RAM CPU biasanya berjenis [DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM) yang menawarkan bandwidth sebesar 20-25 GB/s per modul. Setiap modul memiliki bus selebar 64-bit. Biasanya pasangan modul memori digunakan untuk memungkinkan beberapa channel. CPU memiliki antara 2 hingga 4 channel memori, yaitu memiliki bandwidth puncak antara 40 GB/s hingga 100 GB/s. Sering kali ada dua bank per channel. Misalnya, AMD Zen 3 Threadripper memiliki 8 slot.

Meskipun angka-angka ini mengesankan, mereka hanya menceritakan sebagian dari cerita. Ketika kita ingin membaca sebagian dari memori, pertama-tama kita perlu memberi tahu modul memori di mana informasi tersebut dapat ditemukan. Artinya, kita pertama-tama perlu mengirim *alamat* ke RAM. Setelah ini dilakukan, kita bisa memilih untuk membaca hanya satu catatan 64-bit atau serangkaian catatan panjang. Yang terakhir disebut *burst read*. Singkatnya, mengirimkan alamat ke memori dan mengatur transfer membutuhkan waktu sekitar 100 ns (detail tergantung pada koefisien waktu spesifik dari chip memori yang digunakan), setiap transfer berikutnya hanya membutuhkan waktu 0.2 ns. Singkatnya, pembacaan pertama adalah 500 kali lebih mahal daripada pembacaan berikutnya! Perhatikan bahwa kita bisa melakukan hingga 10 juta pembacaan acak per detik. Ini menunjukkan bahwa kita sebaiknya menghindari akses memori acak sejauh mungkin dan menggunakan burst read (dan write) sebagai gantinya.

Materi lebih kompleks ketika kita mempertimbangkan bahwa kita memiliki beberapa *bank*. Setiap bank bisa membaca memori secara hampir independen. Ini berarti dua hal:
Di satu sisi, jumlah pembacaan acak yang efektif hingga 4 kali lebih tinggi, asalkan mereka tersebar merata di seluruh memori. Hal ini juga berarti bahwa melakukan pembacaan acak masih bukan ide yang baik karena burst read 4 kali lebih cepat juga. Di sisi lain, karena penyelarasan memori dengan batas 64-bit, ada baiknya untuk menyelaraskan struktur data dengan batas yang sama. Kompiler biasanya melakukan ini secara [otomatis](https://en.wikipedia.org/wiki/Data_structure_alignment) ketika flag yang sesuai diatur. Pembaca yang ingin tahu didorong untuk meninjau ceramah tentang DRAM seperti yang diberikan oleh [Zeshan Chishti](http://web.cecs.pdx.edu/%7Ezeshan/ece585_lec5.pdf).

Memori GPU memiliki kebutuhan bandwidth yang lebih tinggi karena mereka memiliki lebih banyak elemen pemrosesan daripada CPU. Secara garis besar ada dua opsi untuk mengatasinya. Pertama adalah membuat bus memori jauh lebih lebar. Misalnya, NVIDIA RTX 2080 Ti memiliki bus selebar 352-bit. Hal ini memungkinkan lebih banyak informasi ditransfer pada saat yang sama. Kedua, GPU menggunakan memori kinerja tinggi tertentu. Perangkat kelas konsumen, seperti seri NVIDIA RTX dan Titan, biasanya menggunakan chip [GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM) dengan bandwidth agregat lebih dari 500 GB/s. Alternatif lainnya adalah menggunakan modul HBM (high bandwidth memory). Modul ini menggunakan antarmuka yang sangat berbeda dan terhubung langsung dengan GPU pada wafer silikon khusus. Ini membuatnya sangat mahal dan penggunaannya biasanya terbatas pada chip server kelas atas, seperti akselerator seri NVIDIA Volta V100. Tidak mengherankan, memori GPU umumnya *jauh* lebih kecil daripada memori CPU karena biaya yang lebih tinggi. Untuk keperluan kita, secara umum karakteristik performa mereka serupa, hanya saja lebih cepat. Kita bisa dengan aman mengabaikan detail-detail ini untuk tujuan buku ini. Detail ini hanya penting saat men-tuning kernel GPU untuk throughput yang tinggi.



## Penyimpanan

Kita telah melihat bahwa beberapa karakteristik utama dari RAM adalah *bandwidth* dan *latency*. Hal yang sama juga berlaku untuk perangkat penyimpanan, hanya saja perbedaannya bisa menjadi lebih ekstrem.

### Hard Disk Drives

*Hard disk drives* (HDDs) telah digunakan selama lebih dari setengah abad. Secara sederhana, HDD mengandung sejumlah piringan berputar dengan kepala yang dapat diposisikan untuk membaca atau menulis pada setiap jalur. Hard disk kelas atas dapat menyimpan hingga 16 TB pada 9 piringan. Salah satu keuntungan utama dari HDD adalah harganya yang relatif murah. Salah satu dari banyak kelemahannya adalah mode kegagalan yang biasanya katastropik dan latensi baca yang relatif tinggi.

Untuk memahami yang terakhir, pertimbangkan fakta bahwa HDD berputar sekitar 7.200 RPM (putaran per menit). Jika mereka jauh lebih cepat, mereka akan hancur karena gaya sentrifugal yang diberikan pada piringan. Ini menjadi kelemahan utama ketika kita ingin mengakses sektor tertentu pada disk: kita perlu menunggu hingga piringan berputar ke posisi yang sesuai (kita bisa memindahkan kepala tetapi tidak mempercepat putaran disk itu sendiri). Oleh karena itu, dapat memakan waktu lebih dari 8 ms hingga data yang diminta tersedia. Cara umum untuk mengekspresikan ini adalah dengan mengatakan bahwa HDD dapat beroperasi pada sekitar 100 IOPs (input/output operations per second). Angka ini pada dasarnya tidak berubah selama dua dekade terakhir. Yang lebih buruk, sulit untuk meningkatkan bandwidth (berada pada kisaran 100--200 MB/s). Setiap kepala membaca jalur bit, sehingga laju bit hanya meningkat dengan akar kuadrat dari kepadatan informasi. Sebagai hasilnya, HDD dengan cepat menjadi penyimpanan arsip dan penyimpanan kelas rendah untuk dataset yang sangat besar.


### Solid State Drives

Solid State Drives (SSD) menggunakan memori flash untuk menyimpan informasi secara permanen. Ini memungkinkan akses yang *jauh lebih cepat* ke catatan yang disimpan. SSD modern dapat beroperasi pada 100.000 hingga 500.000 IOPs, yaitu hingga 3 kali lipat lebih cepat dari HDD. Selain itu, bandwidth mereka bisa mencapai 1--3GB/s, yaitu satu urutan lebih cepat daripada HDD. Peningkatan ini terdengar hampir terlalu bagus untuk menjadi kenyataan. Memang, peningkatan ini disertai dengan beberapa hal berikut, karena desain SSD.

* SSD menyimpan informasi dalam blok (256 KB atau lebih besar). Mereka hanya dapat ditulis secara keseluruhan, yang membutuhkan waktu yang cukup lama. Akibatnya, penulisan acak bit-wise pada SSD memiliki performa yang sangat buruk. Demikian pula, menulis data secara umum memerlukan waktu yang signifikan karena blok harus dibaca, dihapus, dan kemudian ditulis ulang dengan informasi baru. Saat ini, pengontrol dan firmware SSD telah mengembangkan algoritma untuk mengatasi ini. Namun demikian, penulisan dapat jauh lebih lambat, terutama untuk SSD QLC (quad level cell). Kunci untuk meningkatkan performa adalah menjaga *antrian* operasi, lebih memprioritaskan baca, dan menulis dalam blok besar jika memungkinkan.
* Sel memori dalam SSD habis lebih cepat (sering kali setelah beberapa ribu kali penulisan). Algoritma perlindungan wear-level mampu menyebarkan degradasi ke banyak sel. Namun demikian, tidak disarankan untuk menggunakan SSD untuk file swap atau untuk agregasi log-file yang besar.
* Terakhir, peningkatan bandwidth yang besar memaksa desainer komputer untuk menghubungkan SSD secara langsung ke bus PCIe. Drive yang mampu menangani ini, disebut NVMe (Non Volatile Memory enhanced), dapat menggunakan hingga 4 jalur PCIe. Ini setara dengan hingga 8GB/s pada PCIe 4.0.

### Penyimpanan Cloud

Penyimpanan cloud menyediakan rentang performa yang dapat dikonfigurasi. Artinya, penugasan penyimpanan ke mesin virtual bersifat dinamis, baik dalam hal kuantitas maupun dalam hal kecepatan, seperti yang dipilih oleh pengguna. Kami merekomendasikan agar pengguna meningkatkan jumlah IOPs yang disediakan kapan pun latensi terlalu tinggi, misalnya saat pelatihan dengan banyak catatan kecil.

## CPU

Central Processing Units (CPUs) adalah pusat dari setiap komputer. Mereka terdiri dari sejumlah komponen utama: *inti (_core_) prosesor* yang dapat menjalankan kode mesin, sebuah *bus* yang menghubungkan mereka (topologi spesifiknya sangat berbeda antara model prosesor, generasi, dan vendor), dan *cache* untuk memungkinkan akses memori dengan bandwidth yang lebih tinggi dan latensi yang lebih rendah dibandingkan dengan apa yang mungkin dilakukan oleh pembacaan dari memori utama. Terakhir, hampir semua CPU modern mengandung *unit pemrosesan vektor* untuk membantu dengan aljabar linear kinerja tinggi dan konvolusi, karena hal ini umum dalam pemrosesan media dan machine learning.

![CPU quad-core Intel Skylake kelas konsumen.](../img/skylake.svg)
:label:`fig_skylake`

:numref:`fig_skylake` menggambarkan CPU Intel Skylake kelas konsumen dengan empat inti. CPU ini memiliki GPU terintegrasi, cache, dan ringbus yang menghubungkan keempat inti. Periferal, seperti Ethernet, WiFi, Bluetooth, pengontrol SSD, dan USB, merupakan bagian dari chipset atau terhubung langsung (PCIe) ke CPU.



### Mikroarsitektur

Setiap inti prosesor terdiri dari satu set komponen yang cukup canggih. Meskipun detailnya berbeda antara generasi dan vendor, fungsi dasar umumnya cukup standar. Front-end memuat instruksi dan mencoba memprediksi jalur mana yang akan diambil (misalnya, untuk aliran kontrol). Instruksi kemudian di-dekode dari kode assembly ke mikroinstruksi. Kode assembly sering kali bukan merupakan kode level terendah yang dieksekusi oleh prosesor. Sebagai gantinya, instruksi yang kompleks dapat di-dekode menjadi satu set operasi level yang lebih rendah. Instruksi-instruksi ini kemudian diproses oleh inti eksekusi aktual. Seringkali, inti ini mampu melakukan banyak operasi secara bersamaan. Misalnya, inti ARM Cortex A77 pada :numref:`fig_cortexa77` mampu melakukan hingga 8 operasi sekaligus.

![Mikroarsitektur ARM Cortex A77.](../img/a77.svg)
:label:`fig_cortexa77`

Artinya, program yang efisien dapat melakukan lebih dari satu instruksi per siklus clock, asalkan dapat dijalankan secara independen. Tidak semua unit diciptakan sama. Beberapa unit berspesialisasi dalam instruksi integer sementara yang lain dioptimalkan untuk kinerja titik mengambang. Untuk meningkatkan throughput, prosesor juga dapat mengikuti beberapa jalur kode secara bersamaan dalam instruksi bercabang dan kemudian membuang hasil dari cabang yang tidak diambil. Inilah alasan mengapa unit prediksi cabang sangat penting (pada front-end) agar hanya jalur yang paling menjanjikan yang diikuti.

### Vektorisasi

Deep learning sangat membutuhkan banyak komputasi. Oleh karena itu, agar CPU cocok untuk machine learning, diperlukan banyak operasi yang dilakukan dalam satu siklus clock. Ini dicapai melalui unit vektor. Mereka memiliki nama yang berbeda: pada ARM mereka disebut NEON, pada x86 generasi terbaru mereka disebut unit [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions). Aspek umum dari unit-unit ini adalah bahwa mereka dapat melakukan operasi SIMD (single instruction multiple data). :numref:`fig_neon128` menunjukkan bagaimana 8 bilangan bulat pendek dapat dijumlahkan dalam satu siklus clock pada ARM.

![Vektorisasi NEON 128 bit.](../img/neon128.svg)
:label:`fig_neon128`

Tergantung pada pilihan arsitektur, register seperti ini dapat memiliki panjang hingga 512 bit, memungkinkan kombinasi hingga 64 pasang bilangan. Misalnya, kita dapat mengalikan dua bilangan dan menambahkannya ke bilangan ketiga, yang juga dikenal sebagai fused multiply-add. Intel's [OpenVino](https://01.org/openvinotoolkit) menggunakan ini untuk mencapai throughput yang baik untuk deep learning pada CPU kelas server. Namun, perlu dicatat bahwa angka ini sepenuhnya kalah jauh dibandingkan dengan apa yang mampu dicapai oleh GPU. Misalnya, NVIDIA RTX 2080 Ti memiliki 4.352 inti CUDA, yang masing-masing mampu memproses operasi semacam itu kapan saja.

### Cache

Pertimbangkan situasi berikut: kita memiliki inti CPU yang cukup sederhana dengan 4 inti seperti yang digambarkan pada :numref:`fig_skylake` di atas, berjalan pada frekuensi 2 GHz.
Selain itu, mari kita asumsikan bahwa kita memiliki IPC (instructions per clock) sebanyak 1 dan unit-unit tersebut telah mengaktifkan AVX2 dengan lebar 256-bit. Mari kita asumsikan juga bahwa setidaknya satu dari register yang digunakan untuk operasi AVX2 harus diambil dari memori. Ini berarti bahwa CPU mengonsumsi $4 \times 256 \textrm{ bit} = 128 \textrm{ bytes}$ data per siklus clock. Kecuali kita dapat mentransfer $2 \times 10^9 \times 128 = 256 \times 10^9$ byte ke prosesor per detik, elemen-elemen pemrosesan akan kekurangan data. Sayangnya, antarmuka memori dari chip semacam itu hanya mendukung transfer data 20--40 GB/s, yaitu sekitar satu orde magnitudo lebih rendah. Solusinya adalah menghindari pemuatan *data baru* dari memori sejauh mungkin dan lebih baik menyimpannya secara lokal di CPU. Inilah di mana cache menjadi sangat berguna. Berikut adalah beberapa nama atau konsep umum yang sering digunakan:

* **Register** secara teknis bukan bagian dari cache. Mereka membantu memuat instruksi. Namun, register CPU adalah lokasi memori yang dapat diakses CPU dengan kecepatan clock tanpa ada penalti keterlambatan. CPU memiliki puluhan register. Penggunaan register yang efisien tergantung pada compiler (atau programmer). Misalnya, bahasa pemrograman C memiliki kata kunci `register`.
* **Cache L1** adalah lini pertama pertahanan terhadap kebutuhan bandwidth memori yang tinggi. Cache L1 berukuran kecil (biasanya 32--64 KB) dan sering dibagi menjadi cache data dan instruksi. Ketika data ditemukan di cache L1, akses sangat cepat. Jika data tidak ditemukan di sana, pencarian dilanjutkan ke hierarki cache berikutnya.
* **Cache L2** adalah tempat pemberhentian berikutnya. Tergantung pada desain arsitektur dan ukuran prosesor, cache ini mungkin eksklusif. Cache L2 mungkin hanya dapat diakses oleh inti tertentu atau dibagi di antara beberapa inti. Cache L2 lebih besar (biasanya 256--512 KB per inti) dan lebih lambat daripada L1. Selain itu, untuk mengakses sesuatu di L2, kita pertama-tama perlu memeriksa dan menyadari bahwa data tersebut tidak ada di L1, yang menambah sedikit latensi tambahan.
* **Cache L3** dibagi di antara beberapa inti dan dapat cukup besar. CPU server AMD Epyc 3 memiliki cache sebesar 256 MB yang tersebar di beberapa chiplet. Angka yang lebih umum adalah dalam kisaran 4--8 MB.

Memprediksi elemen memori mana yang akan diperlukan selanjutnya adalah salah satu parameter optimasi utama dalam desain chip. Misalnya, disarankan untuk melakukan penelusuran memori ke arah *maju* karena sebagian besar algoritma caching akan mencoba untuk *membaca ke depan* daripada mundur. Demikian juga, menjaga pola akses memori tetap lokal adalah cara yang baik untuk meningkatkan performa.

Menambahkan cache adalah pedang bermata dua. Di satu sisi, cache memastikan bahwa inti prosesor tidak kekurangan data. Pada saat yang sama, cache meningkatkan ukuran chip, menggunakan area yang seharusnya dapat digunakan untuk meningkatkan daya pemrosesan. Selain itu, *cache miss* dapat menjadi mahal. Pertimbangkan skenario terburuk, *false sharing*, seperti yang digambarkan pada :numref:`fig_falsesharing`. Sebuah lokasi memori di-cache pada prosesor 0 saat sebuah thread di prosesor 1 meminta data tersebut. Untuk mendapatkannya, prosesor 0 harus berhenti dari apa yang sedang dilakukannya, menulis kembali informasi tersebut ke memori utama dan kemudian membiarkan prosesor 1 membaca dari memori. Selama operasi ini, kedua prosesor harus menunggu. Potensialnya, kode semacam ini dapat berjalan *lebih lambat* pada beberapa prosesor dibandingkan dengan implementasi prosesor tunggal yang efisien. Inilah alasan lainnya mengapa ukuran cache dibatasi dalam praktiknya (selain masalah fisik ukuran cache).

![False sharing (gambar milik Intel).](../img/falsesharing.svg)
:label:`fig_falsesharing`



## GPU dan Akselerator Lainnya

Tidak berlebihan untuk mengatakan bahwa deep learning tidak akan sukses tanpa adanya GPU. Pada saat yang sama, sangat wajar untuk mengatakan bahwa keberhasilan produsen GPU meningkat secara signifikan karena deep learning. Evolusi bersama antara perangkat keras dan algoritma ini telah mengarah pada situasi di mana, baik buruknya, deep learning menjadi paradigma pemodelan statistik yang lebih disukai. Oleh karena itu, sangat penting untuk memahami manfaat khusus yang dimiliki oleh GPU dan akselerator terkait seperti TPU :cite:`Jouppi.Young.Patil.ea.2017`.

Perlu dicatat adanya perbedaan yang sering kali dibuat dalam praktik: akselerator dioptimalkan baik untuk *training* atau *inference*. Untuk yang terakhir, kita hanya perlu melakukan komputasi forward propagation dalam jaringan. Tidak perlu menyimpan data antara untuk backpropagation. Selain itu, kita mungkin tidak memerlukan komputasi yang sangat presisi (FP16 atau INT8 umumnya sudah cukup). Di sisi lain, selama training, semua hasil antara perlu disimpan untuk menghitung gradien. Selain itu, mengakumulasi gradien memerlukan presisi yang lebih tinggi untuk menghindari underflow (atau overflow) numerik. Ini berarti FP16 (atau precision campuran dengan FP32) adalah persyaratan minimum. Semua ini memerlukan memori yang lebih cepat dan lebih besar (HBM2 vs. GDDR6) serta kekuatan pemrosesan yang lebih besar. Misalnya, GPU [Turing](https://devblogs.nvidia.com/nvidia-turing-architecture-in-depth/) T4 dari NVIDIA dioptimalkan untuk *inference*, sementara GPU V100 lebih disukai untuk *training*.

Ingat vektorisasi seperti yang diilustrasikan pada :numref:`fig_neon128`. Menambahkan unit vektor ke inti prosesor memungkinkan kita untuk meningkatkan throughput secara signifikan. Misalnya, dalam contoh di :numref:`fig_neon128`, kita mampu melakukan 16 operasi secara bersamaan. Pertama, bagaimana jika kita menambahkan operasi yang tidak hanya dioptimalkan untuk operasi antar vektor tetapi juga antara matriks? Strategi ini menghasilkan tensor cores (yang akan kita bahas sebentar lagi). Kedua, bagaimana jika kita menambahkan banyak inti lagi? Singkatnya, dua strategi ini merangkum keputusan desain dalam GPU. :numref:`fig_turing_processing_block` memberikan gambaran dari blok pemrosesan dasar. Blok ini berisi 16 unit integer dan 16 unit floating point. Selain itu, dua tensor cores mempercepat subset operasi yang relevan untuk deep learning. Setiap *streaming multiprocessor* terdiri dari empat blok tersebut.

![Blok pemrosesan NVIDIA Turing (gambar milik NVIDIA).](../img/turing-processing-block.png)
:width:`150px`
:label:`fig_turing_processing_block`

Selanjutnya, 12 *streaming multiprocessors* dikelompokkan ke dalam *graphics processing clusters* yang membentuk prosesor TU102 kelas atas. Saluran memori yang memadai dan cache L2 melengkapi susunan ini. :numref:`fig_turing` memiliki detail terkait. Salah satu alasan mendesain perangkat seperti ini adalah karena blok individual dapat ditambahkan atau dihilangkan sesuai kebutuhan untuk memungkinkan pembuatan chip yang lebih ringkas dan untuk menangani masalah hasil produksi (modul yang rusak mungkin tidak diaktifkan). Untungnya, pemrograman perangkat ini tersembunyi dari peneliti deep learning di balik lapisan CUDA dan kode kerangka kerja. Secara khusus, lebih dari satu program mungkin dieksekusi secara bersamaan pada GPU, asalkan ada sumber daya yang tersedia. Namun demikian, penting untuk menyadari keterbatasan perangkat ini agar tidak memilih model yang tidak muat dalam memori perangkat.

![Arsitektur NVIDIA Turing (gambar milik NVIDIA)](../img/turing.png)
:width:`350px`
:label:`fig_turing`

Aspek terakhir yang patut disebutkan lebih detail adalah *tensor cores*. Mereka adalah contoh tren terbaru dalam menambahkan sirkuit yang lebih dioptimalkan dan secara khusus efektif untuk deep learning. Misalnya, TPU menambahkan array sistolik :cite:`Kung.1988` untuk perkalian matriks yang cepat. Desain ini dimaksudkan untuk mendukung sejumlah kecil operasi besar (satu untuk generasi pertama TPU). Tensor cores berada di ujung lain. Mereka dioptimalkan untuk operasi kecil yang melibatkan matriks berukuran antara $4 \times 4$ dan $16 \times 16$, tergantung pada presisi numeriknya. :numref:`fig_tensorcore` memberikan gambaran tentang optimasi ini.

![Tensor cores NVIDIA pada Turing (gambar milik NVIDIA).](../img/tensorcore.jpg)
:width:`400px`
:label:`fig_tensorcore`

Tentu saja ketika mengoptimalkan untuk komputasi, kita akhirnya harus melakukan kompromi. Salah satu dari mereka adalah bahwa GPU tidak sangat baik dalam menangani interrupt dan data yang jarang (sparse data). Meskipun ada pengecualian yang mencolok, seperti [Gunrock](https://github.com/gunrock/gunrock) :cite:`Wang.Davidson.Pan.ea.2016`, pola akses matriks dan vektor yang jarang tidak cocok dengan operasi pembacaan burst bandwidth tinggi di mana GPU unggul. Menyeimbangkan kedua tujuan ini adalah area penelitian aktif. Lihat misalnya [DGL](http://dgl.ai), sebuah pustaka yang disetel untuk deep learning pada grafik.



## Jaringan dan Bus

Kapan pun sebuah perangkat tunggal tidak cukup untuk melakukan optimasi, kita perlu mentransfer data ke dan dari perangkat tersebut untuk menyinkronkan pemrosesan. Inilah tempat di mana jaringan dan bus berguna. Kita memiliki beberapa parameter desain: *bandwidth*, biaya, jarak, dan fleksibilitas.
Di satu sisi, kita memiliki WiFi yang memiliki jangkauan cukup baik, sangat mudah digunakan (tanpa kabel), murah tetapi menawarkan *bandwidth* dan *latency* yang cukup buruk jika dibandingkan. Tidak ada peneliti machine learning yang waras akan menggunakan ini untuk membangun klaster server. Pada bagian berikut, kita akan fokus pada interkoneksi yang cocok untuk deep learning.

* **PCIe** adalah bus khusus untuk koneksi titik-ke-titik dengan bandwidth sangat tinggi (hingga 32 GB/s pada PCIe 4.0 di slot 16-lane) per jalur. Latensinya berada di kisaran mikrodetik satu digit (5 μs). Jalur PCIe sangat berharga. Prosesor hanya memiliki jumlah terbatas: AMD EPYC 3 memiliki 128 jalur, Intel Xeon memiliki hingga 48 jalur per chip; pada CPU kelas desktop, jumlahnya adalah 20 (Ryzen 9) dan 16 (Core i9). Karena GPU umumnya memiliki 16 jalur, ini membatasi jumlah GPU yang dapat terhubung ke CPU dengan bandwidth penuh. Bagaimanapun, GPU harus berbagi jalur dengan perangkat berbandwidth tinggi lainnya seperti penyimpanan dan Ethernet. Sama seperti akses RAM, transfer dalam jumlah besar lebih disukai karena mengurangi overhead paket.
* **Ethernet** adalah cara yang paling umum digunakan untuk menghubungkan komputer. Meskipun Ethernet secara signifikan lebih lambat daripada PCIe, ia sangat murah dan tangguh untuk dipasang serta mencakup jarak yang lebih jauh. Bandwidth tipikal untuk server kelas rendah adalah 1 GBit/s. Perangkat kelas tinggi (misalnya, [C5 instances](https://aws.amazon.com/ec2/instance-types/c5/) di cloud) menawarkan antara 10 hingga 100 GBit/s bandwidth. Seperti pada semua kasus sebelumnya, transmisi data memiliki overhead yang signifikan. Perhatikan bahwa kita hampir tidak pernah menggunakan Ethernet mentah secara langsung, melainkan menggunakan protokol yang dijalankan di atas interkoneksi fisik (seperti UDP atau TCP/IP). Ini menambah overhead lebih lanjut. Seperti PCIe, Ethernet dirancang untuk menghubungkan dua perangkat, misalnya, sebuah komputer dan *switch*.
* **Switches** memungkinkan kita untuk menghubungkan beberapa perangkat dengan cara di mana pasangan perangkat mana pun dapat melakukan koneksi titik-ke-titik (biasanya dengan bandwidth penuh) secara bersamaan. Misalnya, *Ethernet switches* mungkin menghubungkan 40 server dengan bandwidth lintas-seksional yang tinggi. Perlu diperhatikan bahwa *switch* tidak terbatas pada jaringan komputer tradisional. Bahkan jalur PCIe dapat di[*switch*](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). Ini terjadi, misalnya, untuk menghubungkan sejumlah besar GPU ke prosesor utama, seperti halnya pada [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/).
* **NVLink** adalah alternatif untuk PCIe ketika datang ke interkoneksi dengan bandwidth sangat tinggi. NVLink menawarkan hingga 300 Gbit/s *data transfer rate* per tautan. GPU server (Volta V100) memiliki enam tautan, sementara GPU kelas konsumen (RTX 2080 Ti) hanya memiliki satu tautan, yang beroperasi pada laju 100 Gbit/s yang lebih rendah. Kami merekomendasikan untuk menggunakan [NCCL](https://github.com/NVIDIA/nccl) untuk mencapai transfer data yang tinggi antara GPU.




## Lebih Banyak Angka Latensi

Ringkasan dalam :numref:`table_latency_numbers` dan :numref:`table_latency_numbers_tesla` berasal dari [Eliot Eshelman](https://gist.github.com/eshelman) yang terus memperbarui versi angka ini sebagai [GitHub gist](https://gist.github.com/eshelman/343a1c46cb3fba142c1afdcdeec17646).

:Tabel Angka Latensi Umum.

| Aksi                                   | Waktu  | Catatan                                             |
| :------------------------------------- | ------:| :-------------------------------------------------- |
| Referensi/hit L1 cache                 | 1.5 ns | 4 siklus                                            |
| Penambahan/perkalian floating-point    | 1.5 ns | 4 siklus                                            |
| Referensi/hit L2 cache                 |   5 ns | 12 ~ 17 siklus                                      |
| Prediksi cabang gagal                  |   6 ns | 15 ~ 20 siklus                                      |
| Hit L3 cache (cache tidak berbagi)     |  16 ns | 42 siklus                                           |
| Hit L3 cache (berbagi di core lain)    |  25 ns | 65 siklus                                           |
| Kunci/membuka kunci mutex              |  25 ns |                                                     |
| Hit L3 cache (dimodifikasi di core lain)|  29 ns | 75 siklus                                           |
| Hit L3 cache (di socket CPU lain)      |  40 ns | 100 ~ 300 siklus (40 ~ 116 ns)                      |
| Lompatan QPI ke CPU lain (per lompatan)|  40 ns |                                                     |
| Referensi memori 64MB (CPU lokal)      |  46 ns | TinyMemBench pada Broadwell E5-2690v4               |
| Referensi memori 64MB (CPU jarak jauh) |  70 ns | TinyMemBench pada Broadwell E5-2690v4               |
| Referensi memori 256MB (CPU lokal)     |  75 ns | TinyMemBench pada Broadwell E5-2690v4               |
| Penulisan acak Intel Optane            |  94 ns | UCSD Non-Volatile Systems Lab                       |
| Referensi memori 256MB (CPU jarak jauh)| 120 ns | TinyMemBench pada Broadwell E5-2690v4               |
| Pembacaan acak Intel Optane            | 305 ns | UCSD Non-Volatile Systems Lab                       |
| Kirim 4KB melalui 100 Gbps HPC fabric  |   1 μs | MVAPICH2 melalui Intel Omni-Path                    |
| Kompres 1KB dengan Google Snappy       |   3 μs |                                                     |
| Kirim 4KB melalui Ethernet 10 Gbps     |  10 μs |                                                     |
| Tulis 4KB acak ke NVMe SSD             |  30 μs | DC P3608 NVMe SSD (QOS 99% adalah 500μs)            |
| Transfer 1MB ke/dari GPU NVLink        |  30 μs | ~33GB/s pada NVIDIA 40GB NVLink                     |
| Transfer 1MB ke/dari GPU PCI-E         |  80 μs | ~12GB/s pada PCIe 3.0 x16 link                      |
| Baca 4KB acak dari NVMe SSD            | 120 μs | DC P3608 NVMe SSD (QOS 99%)                         |
| Baca 1MB secara berurutan dari NVMe SSD| 208 μs | ~4.8GB/s DC P3608 NVMe SSD                          |
| Tulis 4KB acak ke SATA SSD             | 500 μs | DC S3510 SATA SSD (QOS 99.9%)                       |
| Baca 4KB acak dari SATA SSD            | 500 μs | DC S3510 SATA SSD (QOS 99.9%)                       |
| Round trip dalam pusat data yang sama  | 500 μs | Satu arah ping adalah ~250μs                        |
| Baca 1MB secara berurutan dari SATA SSD|   2 ms | ~550MB/s DC S3510 SATA SSD                          |
| Baca 1MB secara berurutan dari disk    |   5 ms | ~200MB/s server HDD                                 |
| Akses Disk Acak (seek + rotasi)        |  10 ms |                                                     |
| Kirim paket CA->Belanda->CA            | 150 ms |                                                     |
:label:`table_latency_numbers`

:Tabel Angka Latensi untuk GPU NVIDIA Tesla.

| Aksi                              | Waktu  | Catatan                                             |
| :-------------------------------- | ------:| :-------------------------------------------------- |
| Akses Memori Bersama GPU          |  30 ns | 30~90 siklus (konflik bank menambah latensi)        |
| Akses Memori Global GPU           | 200 ns | 200~800 siklus                                      |
| Luncurkan kernel CUDA pada GPU    |  10 μs | Host CPU menginstruksikan GPU untuk memulai kernel  |
| Transfer 1MB ke/dari GPU NVLink   |  30 μs | ~33GB/s pada NVIDIA 40GB NVLink                     |
| Transfer 1MB ke/dari GPU PCI-E    |  80 μs | ~12GB/s pada PCI-Express x16 link                   |
:label:`table_latency_numbers_tesla`

## Ringkasan

* Perangkat memiliki overhead untuk operasi. Oleh karena itu penting untuk bertujuan untuk transfer besar dengan jumlah kecil daripada banyak transfer kecil. Hal ini berlaku untuk RAM, SSD, jaringan, dan GPU.
* Vektorisasi adalah kunci untuk kinerja. Pastikan Anda menyadari kemampuan spesifik dari akselerator Anda. Misalnya, beberapa CPU Intel Xeon sangat baik untuk operasi INT8, GPU NVIDIA Volta unggul dalam operasi matriks-matriks FP16, dan NVIDIA Turing cemerlang dalam operasi FP16, INT8, dan INT4.
* Overflow numerik akibat tipe data kecil dapat menjadi masalah selama pelatihan (dan dalam skala yang lebih kecil selama inferensi).
* Aliasing dapat secara signifikan menurunkan kinerja. Misalnya, penyelarasan memori pada CPU 64-bit harus dilakukan dengan batas 64-bit. Pada GPU, adalah ide yang baik untuk menjaga ukuran konvolusi tetap selaras, misalnya dengan tensor core.
* Sesuaikan algoritma Anda dengan perangkat keras (misalnya, jejak memori dan bandwidth). Percepatan yang luar biasa (dalam skala besar) dapat dicapai ketika parameter cocok dengan cache.
* Kami merekomendasikan agar Anda merancang kinerja algoritma baru di atas kertas sebelum memverifikasi hasil eksperimental. Perbedaan dalam skala besar adalah alasan untuk waspada.
* Gunakan profiler untuk mendeteksi kemacetan kinerja.
* Perangkat keras pelatihan dan inferensi memiliki titik manis yang berbeda dalam hal harga dan kinerja.

## Latihan

1. Tulis kode C untuk menguji apakah ada perbedaan kecepatan antara mengakses memori yang sejajar atau tidak sejajar relatif terhadap antarmuka memori eksternal. Petunjuk: berhati-hatilah dengan efek cache.
2. Uji perbedaan kecepatan antara mengakses memori secara berurutan atau dengan *stride* tertentu.
3. Bagaimana Anda bisa mengukur ukuran cache pada CPU?
4. Bagaimana Anda akan meletakkan data di beberapa saluran memori untuk mencapai bandwidth maksimum? Bagaimana Anda akan meletakkannya jika Anda memiliki banyak *thread* kecil?
5. HDD kelas enterprise berputar pada 10,000 rpm. Berapa waktu minimum absolut yang dibutuhkan oleh HDD dalam kasus terburuk sebelum dapat membaca data (Anda dapat mengasumsikan bahwa kepala bergerak hampir seketika)? Mengapa HDD 2,5" menjadi populer untuk server komersial (dibandingkan dengan 3,5" dan 5,25")?
6. Asumsikan bahwa produsen HDD meningkatkan kepadatan penyimpanan dari 1 Tbit per inci persegi menjadi 5 Tbit per inci persegi. Berapa banyak informasi yang dapat Anda simpan pada satu lingkaran pada HDD 2,5"? Apakah ada perbedaan antara trek bagian dalam dan luar?
7. Peningkatan dari tipe data 8 bit ke 16 bit meningkatkan jumlah silikon kira-kira empat kali. Mengapa? Mengapa NVIDIA menambahkan operasi INT4 ke GPU Turing mereka?
8. Seberapa cepat membaca maju melalui memori dibandingkan dengan membaca mundur? Apakah angka ini berbeda antara komputer dan vendor CPU yang berbeda? Mengapa? Tulis kode C dan coba eksperimen.
9. Bisakah Anda mengukur ukuran cache disk Anda? Berapa ukurannya untuk HDD tipikal? Apakah SSD memerlukan cache?
10. Ukur overhead paket saat mengirim pesan melalui Ethernet. Cari tahu perbedaan antara koneksi UDP dan TCP/IP.
11. *Direct memory access* memungkinkan perangkat selain CPU untuk menulis (dan membaca) langsung ke (dari) memori. Mengapa ini ide yang baik?
12. Lihat angka kinerja untuk GPU Turing T4. Mengapa kinerja "hanya" berlipat ganda saat Anda beralih dari FP16 ke INT8 dan INT4?
13. Berapa waktu terpendek yang diperlukan untuk paket dalam perjalanan pulang antara San Francisco dan Amsterdam? Petunjuk: Anda dapat mengasumsikan bahwa jaraknya adalah 10,000 km.

[Diskusi](https://discuss.d2l.ai/t/363)
