# Convolutional Neural Networks
:label:`chap_cnn`

Data gambar direpresentasikan sebagai grid dua dimensi dari piksel, baik gambar tersebut
monokromatik maupun berwarna. Maka dari itu, setiap piksel berkorespondensi dengan satu
atau beberapa nilai numerik. Sejauh ini, kita mengabaikan struktur kaya ini
dan memperlakukan gambar sebagai vektor angka dengan *meratakan* gambar tersebut, tanpa memedulikan hubungan spasial antar piksel. Pendekatan ini kurang memuaskan karena kita perlu mengonversi
vektor satu dimensi hasil tersebut agar dapat dimasukkan melalui MLP yang fully connected.

Karena jaringan ini tidak sensitif terhadap urutan fitur,
kita akan mendapatkan hasil serupa terlepas dari apakah kita mempertahankan urutan
yang berkorespondensi dengan struktur spasial piksel atau jika kita menyusun ulang
kolom dari matriks desain sebelum menyesuaikan parameter MLP.
Idealnya, kita ingin memanfaatkan pengetahuan awal kita bahwa piksel yang berdekatan
biasanya saling berhubungan, untuk membangun model yang efisien dalam
belajar dari data gambar.

Bab ini memperkenalkan *convolutional neural networks* (CNN)
:cite:`LeCun.Jackel.Bottou.ea.1995`, sebuah keluarga jaringan saraf yang sangat kuat
yang dirancang untuk tujuan ini.
Arsitektur berbasis CNN kini menjadi umum di bidang computer vision.
Misalnya, pada koleksi Imagenet
:cite:`Deng.Dong.Socher.ea.2009`, penggunaan convolutional neural
networks, atau yang sering disebut Convnets, adalah yang pertama kali memberikan peningkatan performa yang signifikan :cite:`Krizhevsky.Sutskever.Hinton.2012`.

CNN modern, demikian sebutannya, mendapatkan desainnya dari
inspirasi dari biologi, teori grup, dan sejumlah besar
eksperimen. Selain efisiensi sampel dalam
mencapai model yang akurat, CNN cenderung efisien secara komputasi,
karena mereka membutuhkan lebih sedikit parameter dibandingkan arsitektur fully connected dan karena konvolusi mudah diparalelkan di
inti GPU :cite:`Chetlur.Woolley.Vandermersch.ea.2014`.  Akibatnya, praktisi sering kali
menerapkan CNN kapan pun memungkinkan, dan CNN semakin muncul sebagai
pesaing yang andal bahkan pada tugas dengan struktur urutan satu dimensi, seperti audio :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, teks
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014`, dan analisis deret waktu
:cite:`LeCun.Bengio.ea.1995`, di mana jaringan saraf berulang (RNN) biasanya digunakan.
Beberapa adaptasi pintar dari CNN juga
membawanya ke data berstruktur graf :cite:`Kipf.Welling.2016` dan
sistem rekomendasi.

Pertama, kita akan menyelami lebih dalam motivasi di balik convolutional
neural networks. Hal ini akan diikuti dengan tinjauan operasi dasar
yang membentuk tulang punggung dari semua jaringan konvolusi.
Ini termasuk lapisan konvolusi itu sendiri,
rincian teknis termasuk padding dan stride,
lapisan pooling yang digunakan untuk mengumpulkan informasi
di wilayah spasial yang berdekatan,
penggunaan beberapa kanal di setiap lapisan,
dan diskusi yang cermat tentang struktur arsitektur modern.
Bab ini akan ditutup dengan contoh lengkap LeNet,
jaringan konvolusi pertama yang berhasil diterapkan,
jauh sebelum bangkitnya deep learning modern.
Pada bab berikutnya, kita akan mendalami implementasi lengkap
dari beberapa arsitektur CNN populer dan relatif baru
yang desainnya mewakili sebagian besar teknik
yang umum digunakan oleh para praktisi saat ini.


```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

