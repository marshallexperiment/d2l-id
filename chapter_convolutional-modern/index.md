# Jaringan Saraf Konvolusional (_Convolutional Neural Network_) Modern 
:label:`chap_modern_cnn`

Sekarang setelah kita memahami dasar-dasar menghubungkan CNN, mari kita berkeliling arsitektur CNN modern. Tur ini, karena banyaknya desain baru yang menarik yang terus bertambah, tentu saja tidak lengkap. Pentingnya tur ini berasal dari fakta bahwa tidak hanya arsitektur-arsitektur ini bisa langsung digunakan untuk tugas-tugas vision, tetapi juga berfungsi sebagai pembuat fitur dasar untuk tugas yang lebih lanjut seperti pelacakan objek :cite:`Zhang.Sun.Jiang.ea.2021`, segmentasi :cite:`Long.Shelhamer.Darrell.2015`, deteksi objek :cite:`Redmon.Farhadi.2018`, atau transformasi gaya :cite:`Gatys.Ecker.Bethge.2016`. Pada bab ini, sebagian besar bagian berhubungan dengan arsitektur CNN penting yang pada suatu waktu (atau saat ini) menjadi model dasar yang digunakan oleh banyak proyek penelitian dan sistem yang diterapkan. Setiap jaringan ini pernah menjadi arsitektur dominan dan banyak di antaranya menjadi pemenang atau runner-up dalam [kompetisi ImageNet](https://www.image-net.org/challenges/LSVRC/), yang telah menjadi tolok ukur kemajuan dalam supervised learning pada computer vision sejak 2010. Baru-baru ini, Transformer mulai menggantikan CNN, dimulai dengan :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` dan dilanjutkan oleh Swin Transformer :cite:`liu2021swin`. Kami akan membahas perkembangan ini nanti pada :numref:`chap_attention-and-transformers`. 

Meskipun ide dari jaringan saraf *dalam* sangat sederhana (menggabungkan banyak lapisan), kinerja bisa sangat bervariasi di antara arsitektur dan pilihan hyperparameter. Jaringan saraf yang dijelaskan dalam bab ini adalah hasil dari intuisi, beberapa wawasan matematis, dan banyak trial dan error. Kami menyajikan model-model ini secara kronologis, sebagian untuk memberikan gambaran sejarah sehingga Anda dapat membentuk intuisi sendiri tentang arah bidang ini dan mungkin mengembangkan arsitektur Anda sendiri. Sebagai contoh, batch normalization dan residual connections yang dijelaskan dalam bab ini telah menawarkan dua ide populer untuk melatih dan merancang model yang dalam, yang keduanya sejak itu juga diterapkan pada arsitektur di luar computer vision.

Kita mulai tur CNN modern kita dengan AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`, jaringan berskala besar pertama yang berhasil mengalahkan metode vision konvensional pada tantangan vision berskala besar; jaringan VGG :cite:`Simonyan.Zisserman.2014`, yang memanfaatkan sejumlah blok elemen yang berulang; network in network (NiN) yang mengkonvolusikan seluruh jaringan saraf patch-wise pada input :cite:`Lin.Chen.Yan.2013`; GoogLeNet yang menggunakan jaringan dengan konvolusi multi-cabang :cite:`Szegedy.Liu.Jia.ea.2015`; residual network (ResNet) :cite:`He.Zhang.Ren.ea.2016`, yang tetap menjadi salah satu arsitektur siap-pakai paling populer di computer vision; blok ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017` untuk koneksi yang lebih jarang; dan DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` untuk generalisasi dari arsitektur residual. Seiring waktu, banyak optimasi khusus untuk jaringan yang efisien telah dikembangkan, seperti pergeseran koordinat (ShiftNet) :cite:`wu2018shift`. Ini mencapai puncaknya pada pencarian otomatis untuk arsitektur efisien seperti MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`. Ini juga mencakup eksplorasi desain semi-otomatis oleh :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` yang menghasilkan RegNetX/Y yang akan kita bahas nanti pada bab ini. Pekerjaan ini berguna sejauh menawarkan jalur untuk menggabungkan kekuatan komputasi besar dengan kreativitas seorang eksperimentator dalam pencarian ruang desain yang efisien. Yang juga patut dicatat adalah karya dari :citet:`liu2022convnet`, yang menunjukkan bahwa teknik pelatihan (misalnya, optimizer, augmentasi data, dan regularisasi) berperan penting dalam meningkatkan akurasi. Hal ini juga menunjukkan bahwa asumsi yang sudah lama dipegang, seperti ukuran jendela konvolusi, mungkin perlu ditinjau kembali, mengingat peningkatan komputasi dan data. Kami akan membahas ini dan banyak pertanyaan lainnya dalam bab ini.


```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```

