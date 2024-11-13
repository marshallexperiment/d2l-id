# Parameter Server
:label:`sec_parameterserver`

Ketika kita beralih dari satu GPU ke beberapa GPU dan kemudian ke beberapa server yang masing-masing berisi banyak GPU, mungkin tersebar di beberapa rak dan switch jaringan, algoritma kita untuk pelatihan terdistribusi dan paralel perlu menjadi jauh lebih canggih. Detail sangat penting karena interkoneksi yang berbeda memiliki bandwidth yang sangat berbeda (misalnya, NVLink dapat menawarkan hingga 100 GB/s melalui 6 tautan dalam pengaturan yang tepat, PCIe 4.0 (16-lane) menawarkan 32 GB/s, sementara Ethernet 100GbE berkecepatan tinggi hanya sekitar 10 GB/s). Pada saat yang sama, tidaklah masuk akal untuk mengharapkan seorang ahli model statistik menjadi ahli dalam jaringan dan sistem.

Ide inti dari parameter server diperkenalkan dalam :citet:`Smola.Narayanamurthy.2010` dalam konteks model variabel laten terdistribusi. Deskripsi tentang semantik push dan pull kemudian diikuti dalam :citet:`Ahmed.Aly.Gonzalez.ea.2012`, dan deskripsi sistem serta pustaka sumber terbuka diberikan dalam :citet:`Li.Andersen.Park.ea.2014`. Berikut ini kita akan memotivasi komponen-komponen yang diperlukan untuk efisiensi.

## Pelatihan Paralel Data

Mari kita tinjau pendekatan pelatihan paralel data untuk pelatihan terdistribusi. Kita akan menggunakan ini secara eksklusif di bagian ini karena jauh lebih sederhana untuk diimplementasikan dalam praktik. Hampir tidak ada kasus penggunaan (kecuali deep learning pada graf) di mana strategi paralelisme lainnya lebih disukai karena GPU saat ini memiliki memori yang cukup besar. :numref:`fig_parameterserver` menggambarkan varian dari paralelisme data yang kita implementasikan pada :numref:`sec_multi_gpu`. Aspek kuncinya adalah agregasi gradien dilakukan pada satu GPU (GPU 0) sebelum parameter yang diperbarui disiarkan kembali ke semua GPU.

![Kiri: pelatihan GPU tunggal. Kanan: varian pelatihan multi-GPU: (1) kita menghitung loss dan gradien, (2) semua gradien dikumpulkan pada satu GPU, (3) pembaruan parameter dilakukan dan parameter didistribusikan kembali ke semua GPU.](../img/ps.svg)
:label:`fig_parameterserver`

Jika kita tinjau kembali, keputusan untuk mengumpulkan di GPU 0 tampak agak ad-hoc. Lagi pula, kita bisa saja mengumpulkan di CPU. Bahkan, kita bisa memutuskan untuk mengumpulkan beberapa parameter di satu GPU dan beberapa lainnya di GPU lain. Dengan asumsi algoritma optimasi mendukung hal ini, tidak ada alasan sebenarnya mengapa kita tidak bisa melakukannya. Misalnya, jika kita memiliki empat vektor parameter dengan gradien terkait $\mathbf{g}_1, \ldots, \mathbf{g}_4$, kita bisa mengumpulkan gradien di satu GPU untuk setiap $\mathbf{g}_i$ ($i = 1, \ldots, 4$).

Pemikiran ini tampak sewenang-wenang dan berlebihan. Bagaimanapun, matematikanya sama di semua tempat. Namun, kita berurusan dengan perangkat keras fisik nyata di mana bus yang berbeda memiliki bandwidth yang berbeda seperti yang dibahas di :numref:`sec_hardware`.
Pertimbangkan server GPU 4-way nyata seperti yang dijelaskan pada :numref:`fig_bw_hierarchy`. Jika server tersebut terhubung dengan sangat baik, server tersebut mungkin memiliki kartu jaringan 100 GbE. Angka yang lebih umum adalah dalam rentang 1--10 GbE dengan bandwidth efektif 100 MB/s hingga 1 GB/s.
Karena CPU memiliki terlalu sedikit jalur PCIe untuk terhubung langsung ke semua GPU (misalnya, CPU Intel kelas konsumen memiliki 24 jalur), kita memerlukan [multiplexer](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). Bandwidth dari CPU pada link Gen3 16x adalah 16 GB/s. Ini juga merupakan kecepatan di mana *setiap* GPU terhubung ke switch. Artinya, lebih efektif untuk berkomunikasi antar perangkat.

![Server GPU 4-way.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

Untuk kepentingan argumen, mari kita asumsikan bahwa gradien berukuran 160 MB. Dalam kasus ini, dibutuhkan waktu 30 ms untuk mengirim gradien dari 3 GPU yang tersisa ke GPU keempat (setiap transfer memakan waktu 10 ms = 160 MB / 16 GB/s). Menambahkan 30 ms lagi untuk mentransmisikan vektor bobot kembali, kita mendapatkan total 60 ms.
Jika kita mengirim semua data ke CPU, kita akan mengalami penalti sebesar 40 ms karena *setiap* dari empat GPU perlu mengirim data ke CPU, menghasilkan total 80 ms. Terakhir, asumsikan kita dapat membagi gradien menjadi 4 bagian masing-masing sebesar 40 MB. Sekarang kita dapat mengumpulkan setiap bagian di GPU yang berbeda *secara bersamaan* karena switch PCIe menawarkan operasi bandwidth penuh antara semua link. Alih-alih 30 ms, ini memakan waktu 7,5 ms, menghasilkan total 15 ms untuk operasi sinkronisasi. Singkatnya, tergantung pada cara kita menyinkronkan parameter, operasi yang sama dapat memakan waktu antara 15 ms hingga 80 ms. :numref:`fig_ps_distributed` menggambarkan berbagai strategi untuk pertukaran parameter.

![Strategi sinkronisasi parameter.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

Perhatikan bahwa kita memiliki alat lain yang dapat kita gunakan untuk meningkatkan kinerja: dalam jaringan yang dalam, dibutuhkan beberapa waktu untuk menghitung semua gradien dari atas ke bawah. Kita bisa mulai menyinkronkan gradien untuk beberapa kelompok parameter bahkan ketika kita masih sibuk menghitungnya untuk yang lain. Lihat misalnya :citet:`Sergeev.Del-Balso.2018` untuk detail tentang bagaimana melakukan ini di [Horovod](https://github.com/horovod/horovod).




## Sinkronisasi Cincin (Ring Synchronization)

Dalam hal sinkronisasi pada perangkat keras deep learning modern, kita sering kali menemui konektivitas jaringan yang sangat khusus. Misalnya, instance AWS p3.16xlarge dan NVIDIA DGX-2 berbagi struktur konektivitas seperti yang ditunjukkan pada :numref:`fig_nvlink`. Setiap GPU terhubung ke CPU host melalui link PCIe yang bekerja paling optimal pada 16 GB/s. Selain itu, setiap GPU juga memiliki 6 koneksi NVLink, yang masing-masing mampu mentransfer data sebesar 300 Gbit/s secara bidireksional. Ini setara dengan sekitar 18 GB/s per link per arah. Singkatnya, bandwidth NVLink gabungan secara signifikan lebih tinggi dibandingkan dengan bandwidth PCIe. Pertanyaannya adalah bagaimana menggunakannya dengan paling efisien.

![Konektivitas NVLink pada server GPU V100 8 (gambar milik NVIDIA).](../img/nvlink.svg)
:label:`fig_nvlink`

Ternyata strategi sinkronisasi optimal adalah membagi jaringan menjadi dua cincin dan menggunakannya untuk menyinkronkan data secara langsung :cite:`Wang.Li.Liberty.ea.2018`. :numref:`fig_nvlink_twoloop` mengilustrasikan bahwa jaringan dapat dibagi menjadi satu cincin (1-2-3-4-5-6-7-8-1) dengan bandwidth NVLink ganda dan satu cincin (1-4-6-3-5-8-2-7-1) dengan bandwidth reguler. Merancang protokol sinkronisasi yang efisien dalam kasus ini bukanlah hal yang sepele.

![Pembagian jaringan NVLink menjadi dua cincin.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`

Pertimbangkan eksperimen pemikiran berikut: mengingat sebuah cincin dari $n$ node komputasi (atau GPU), kita bisa mengirim gradien dari node pertama ke node kedua. Di sana, gradien ditambahkan ke gradien lokal dan kemudian dikirim ke node ketiga, dan seterusnya. Setelah $n-1$ langkah, gradien gabungan dapat ditemukan pada node terakhir yang dikunjungi. Artinya, waktu untuk mengumpulkan gradien bertambah secara linier dengan jumlah node. Namun, jika kita melakukannya, algoritma ini cukup tidak efisien. Bagaimanapun, pada saat tertentu hanya ada satu node yang melakukan komunikasi. Bagaimana jika kita membagi gradien menjadi $n$ bagian dan mulai menyinkronkan bagian $i$ dimulai dari node $i$?
Karena setiap bagian berukuran $1/n$, total waktu sekarang adalah $(n-1)/n \approx 1$. Dengan kata lain, waktu yang dibutuhkan untuk mengumpulkan gradien *tidak bertambah* ketika kita meningkatkan ukuran cincin. Ini adalah hasil yang cukup menakjubkan. :numref:`fig_ringsync` mengilustrasikan urutan langkah-langkah pada $n=4$ node.

![Sinkronisasi cincin di 4 node. Setiap node mulai mentransmisikan bagian dari gradien ke tetangga kirinya hingga gradien yang tersusun dapat ditemukan di tetangga kanannya.](../img/ringsync.svg)
:label:`fig_ringsync`

Jika kita menggunakan contoh yang sama untuk menyinkronkan 160 MB di 8 GPU V100, kita mendapatkan sekitar $2 \cdot 160 \textrm{MB} / (3 \cdot 18 \textrm{GB/s}) \approx 6 \textrm{ms}$. Ini lebih baik dibandingkan dengan menggunakan bus PCIe, meskipun kita sekarang menggunakan 8 GPU. Perlu dicatat bahwa dalam praktiknya angka-angka ini sedikit lebih buruk, karena framework deep learning sering gagal untuk menggabungkan komunikasi menjadi transfer burst besar.

Perhatikan bahwa ada kesalahpahaman umum bahwa sinkronisasi cincin sangat berbeda dari algoritma sinkronisasi lainnya. Satu-satunya perbedaan adalah bahwa jalur sinkronisasi sedikit lebih rumit jika dibandingkan dengan pohon sederhana.



## Pelatihan Multi-Mesin

Pelatihan terdistribusi pada beberapa mesin menambah tantangan lebih lanjut: kita perlu berkomunikasi dengan server yang hanya terhubung melalui jaringan yang relatif lebih rendah bandwidthnya, yang dalam beberapa kasus bisa lebih lambat hingga satu order besarnya. Sinkronisasi antar perangkat menjadi rumit. Bagaimanapun, mesin-mesin yang berbeda menjalankan kode pelatihan dengan kecepatan yang sedikit berbeda. Oleh karena itu, kita perlu *menyinkronkan* mereka jika kita ingin menggunakan optimasi terdistribusi sinkron. :numref:`fig_ps_multimachine` mengilustrasikan bagaimana pelatihan paralel terdistribusi berlangsung.

1. Sebuah batch data (berbeda) dibaca di setiap mesin, dibagi di beberapa GPU, dan dipindahkan ke memori GPU. Di sana, prediksi dan gradien dihitung untuk setiap batch GPU secara terpisah.
2. Gradien dari semua GPU lokal dikumpulkan di satu GPU (atau sebagian dikumpulkan di GPU yang berbeda).
3. Gradien dikirim ke CPU.
4. CPU mengirimkan gradien ke server parameter pusat yang mengumpulkan semua gradien.
5. Gradien gabungan kemudian digunakan untuk memperbarui parameter dan parameter yang diperbarui disiarkan kembali ke CPU individu.
6. Informasi dikirim ke satu (atau lebih) GPU.
7. Parameter yang diperbarui disebarkan ke semua GPU.

![Pelatihan paralel terdistribusi multi-mesin multi-GPU.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Setiap operasi ini tampaknya cukup sederhana. Dan memang, mereka dapat dilakukan secara efisien *di dalam* satu mesin. Namun, ketika kita melihat beberapa mesin, kita dapat melihat bahwa server parameter pusat menjadi bottleneck. Bagaimanapun, bandwidth per server terbatas, sehingga untuk $m$ pekerja, waktu yang dibutuhkan untuk mengirim semua gradien ke server adalah $\mathcal{O}(m)$. Kita bisa melewati hambatan ini dengan menambah jumlah server menjadi $n$. Pada titik ini, setiap server hanya perlu menyimpan $\mathcal{O}(1/n)$ dari parameter, sehingga waktu total untuk pembaruan dan optimasi menjadi $\mathcal{O}(m/n)$. Mencocokkan kedua angka ini menghasilkan skala konstan terlepas dari berapa banyak pekerja yang kita miliki. Dalam praktiknya kita menggunakan mesin *yang sama* baik sebagai pekerja maupun sebagai server. :numref:`fig_ps_multips` menggambarkan desainnya (lihat juga :cite:`Li.Andersen.Park.ea.2014` untuk detailnya).
Secara khusus, memastikan bahwa beberapa mesin bekerja tanpa keterlambatan yang tidak masuk akal bukanlah hal yang sepele.

![Atas: satu server parameter adalah bottleneck karena bandwidthnya terbatas. Bawah: beberapa server parameter menyimpan bagian dari parameter dengan bandwidth gabungan.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## Penyimpanan Kunci-Nilai

Mengimplementasikan langkah-langkah yang diperlukan untuk pelatihan multi-GPU terdistribusi dalam praktik bukanlah hal yang mudah.
Inilah mengapa menggunakan abstraksi umum, yaitu *penyimpanan kunci-nilai* dengan semantik pembaruan yang didefinisikan ulang sangat bermanfaat.

Pada banyak pekerja dan banyak GPU, komputasi untuk gradien $i$ dapat didefinisikan sebagai

$$\mathbf{g}_{i} = \sum_{k \in \textrm{workers}} \sum_{j \in \textrm{GPUs}} \mathbf{g}_{ijk},$$

di mana $\mathbf{g}_{ijk}$ adalah bagian dari gradien $i$ yang dibagi pada GPU $j$ dari pekerja $k$.
Aspek kunci dalam operasi ini adalah bahwa ini merupakan *reduksi komutatif*, yaitu, mengubah banyak vektor menjadi satu dan urutan penerapan operasi ini tidak penting. Ini sangat berguna untuk tujuan kita karena kita tidak (perlu) memiliki kontrol detail mengenai kapan gradien diterima. Selain itu, perhatikan bahwa operasi ini independen di antara $i$ yang berbeda.

Hal ini memungkinkan kita untuk mendefinisikan dua operasi berikut: *push*, yang mengumpulkan gradien, dan *pull*, yang mengambil gradien gabungan. Karena kita memiliki banyak set gradien (bagaimanapun, kita memiliki banyak lapisan), kita perlu mengindeks gradien dengan kunci $i$. Kemiripan ini dengan penyimpanan kunci-nilai, seperti yang diperkenalkan dalam Dynamo :cite:`DeCandia.Hastorun.Jampani.ea.2007` bukanlah kebetulan. Mereka juga memenuhi banyak karakteristik serupa, terutama ketika datang ke distribusi parameter di beberapa server.

Operasi push dan pull untuk penyimpanan kunci-nilai dijelaskan sebagai berikut:

* **push(key, value)** mengirimkan gradien tertentu (nilai) dari pekerja ke penyimpanan umum. Di sana nilai tersebut dikumpulkan, misalnya dengan menjumlahkannya.
* **pull(key, value)** mengambil nilai gabungan dari penyimpanan umum, misalnya setelah menggabungkan gradien dari semua pekerja.

Dengan menyembunyikan semua kompleksitas tentang sinkronisasi di balik operasi push dan pull sederhana, kita dapat memisahkan kekhawatiran para ahli model statistik yang ingin dapat mengekspresikan optimasi dalam istilah sederhana dan para insinyur sistem yang harus menangani kompleksitas yang melekat dalam sinkronisasi terdistribusi.





## Ringkasan

* Sinkronisasi perlu sangat adaptif terhadap infrastruktur jaringan spesifik dan konektivitas di dalam server. Hal ini dapat membuat perbedaan signifikan terhadap waktu yang dibutuhkan untuk sinkronisasi.
* Sinkronisasi cincin (ring-synchronization) bisa optimal untuk server p3 dan DGX-2. Untuk server lain mungkin tidak seoptimal itu.
* Strategi sinkronisasi hierarkis bekerja dengan baik ketika menambahkan beberapa server parameter untuk meningkatkan bandwidth.

## Latihan

1. Bisakah Anda meningkatkan sinkronisasi cincin lebih jauh lagi? Petunjuk: Anda dapat mengirim pesan ke dua arah.
2. Apakah memungkinkan untuk mengizinkan komunikasi asinkron (saat komputasi masih berlangsung)? Bagaimana hal ini memengaruhi kinerja?
3. Bagaimana jika kita kehilangan sebuah server selama komputasi yang berlangsung lama? Bagaimana kita bisa merancang mekanisme *fault tolerance* untuk menghindari memulai ulang komputasi secara penuh?

[Diskusi](https://discuss.d2l.ai/t/366)

